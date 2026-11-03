//! Relocatable lifts and the multi-region image: no host pointer reaches the code, and an
//! image loaded at any slide runs like the JIT.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::abi::CallAbi;
use microlift::core::bridge::BridgeImport;
use microlift::core::symbol::{HostSymbol, HostSymbolKind};
use microlift::core::{AddressMode, LifterConfig, lift_to_code};
use microlift::function_traits::{DynamicSymbol, FunctionArgumentType};
use microlift::image::{Image, ImageSymbol};
use microlift::runtime::image::{SymbolResolver, load};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

const FAST_VA: u64 = 0x2000;
const BRIDGE_VA: u64 = 0x3000;
const SLOT_VA: u64 = 0x4000;

/// `call FAST; call BRIDGE; call SLOT; lea rax, [rip]; ret` at `DEFAULT_LIFT_BASE`.
fn three_imports_and_a_guest_address() -> Vec<u8> {
    let mut bytes = Vec::new();
    for target in [FAST_VA, BRIDGE_VA, SLOT_VA] {
        let next = DEFAULT_LIFT_BASE + bytes.len() as u64 + 5;
        bytes.push(0xE8);
        bytes.extend_from_slice(&((target as i64 - next as i64) as i32).to_le_bytes());
    }
    bytes.extend_from_slice(&[0x48, 0x8D, 0x05, 0, 0, 0, 0]);
    bytes.push(0xC3);
    bytes
}

fn fake_symbol(name: &'static str, address: usize) -> DynamicSymbol {
    DynamicSymbol {
        name,
        value: address as *mut _,
        arg_types: vec![FunctionArgumentType::Pointer, FunctionArgumentType::Pointer],
        is_ellipsis_function: false,
        has_execution_side_effects: false,
    }
}

/// A config whose every host reference sits at `host + k`, slid by `slide`.
fn config_at(host: usize, slide: u64, slot: &'static AtomicPtr<()>) -> LifterConfig {
    let mut config = LifterConfig {
        address_mode: AddressMode::Relocated,
        guest_slide: slide,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..LifterConfig::default()
    };
    config.register_fast_import_from(
        FAST_VA,
        "libfast.dylib",
        fake_symbol("_fast", host + 0x10),
        CallAbi::sysv_amd64(),
    );
    config.register_bridge_import(
        BRIDGE_VA,
        BridgeImport::direct("direct", (host + 0x20) as *const u8),
    );
    config.register_bridge_import(SLOT_VA, BridgeImport::slot("slotted", slot));
    config
}

fn image_bytes(config: LifterConfig, bytes: &[u8]) -> Vec<u8> {
    let fingerprint = config.codegen_fingerprint();
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    let mut image = ArtifactBackend::image(None, None, &[], fingerprint).expect("isa");
    image.add_region(code).expect("compile");
    image.finish().encode()
}

static SLOT_A: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SLOT_B: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

#[test]
fn an_image_holds_no_host_or_guest_run_time_address() {
    let bytes = three_imports_and_a_guest_address();
    let first = image_bytes(config_at(0x1111_0000, 0, &SLOT_A), &bytes);
    let second = image_bytes(config_at(0x2222_0000, 0x7000_0000, &SLOT_B), &bytes);
    assert_eq!(first, second);

    let image = Image::decode(&first).unwrap();
    let names: Vec<String> = image
        .symbols
        .iter()
        .map(|s| match s {
            ImageSymbol::Host(host) => format!("{}`{}", host.provider, host.name),
            ImageSymbol::GuestSlide => "slide".into(),
        })
        .collect();
    for expected in ["libfast.dylib`_fast", "`direct", "`slotted", "slide"] {
        assert!(
            names.iter().any(|n| n == expected),
            "{expected} missing from {names:?}"
        );
    }
}

#[test]
fn lifting_twice_builds_the_same_image() {
    let bytes = three_imports_and_a_guest_address();
    assert_eq!(
        image_bytes(config_at(0x1111_0000, 0, &SLOT_A), &bytes),
        image_bytes(config_at(0x1111_0000, 0, &SLOT_A), &bytes)
    );
}

unsafe extern "C" fn forty() -> u64 {
    40
}

unsafe extern "C" fn two() -> u64 {
    2
}

#[test]
fn one_name_from_two_providers_calls_two_functions() {
    // call 0x2000; mov rbx, rax; call 0x3000; add rax, rbx; ret
    let bytes = [
        0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x48, 0x89, 0xC3, 0xE8, 0xF3, 0x1F, 0x00, 0x00, 0x48, 0x01,
        0xD8, 0xC3,
    ];
    let out = LiftTest::new(&bytes)
        .no_unicorn()
        .with_config(|c| {
            c.register_fast_import_from(
                0x2000,
                "libA.dylib",
                DynamicSymbol::new("_value", forty as unsafe extern "C" fn() -> u64),
                CallAbi::sysv_amd64(),
            );
            c.register_fast_import_from(
                0x3000,
                "libB.dylib",
                DynamicSymbol::new("_value", two as unsafe extern "C" fn() -> u64),
                CallAbi::sysv_amd64(),
            );
        })
        .run();
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
}

unsafe extern "C" fn add_one(x: u64) -> u64 {
    x + 1
}

struct Resolver {
    slide: u64,
}

impl SymbolResolver for Resolver {
    fn resolve(&self, symbol: &HostSymbol) -> Option<usize> {
        match &*symbol.name {
            "_add_one" if matches!(symbol.kind, HostSymbolKind::Function(_)) => {
                Some(add_one as *const () as usize)
            }
            "record" => Some(record_callee as *const () as usize),
            _ => None,
        }
    }

    fn guest_slide(&self) -> u64 {
        self.slide
    }
}

/// Enters `entry` with `rdi` and a stack returning to the exit trampoline.
fn run(entry: *const u8, rip: u64, rdi: u64) -> RegisterContext {
    let mut stack = vec![0u64; 64];
    let rsp = stack.as_mut_ptr() as u64 + 63 * 8;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };
    let mut ctx = RegisterContext {
        rsp,
        rip,
        rdi,
        ..RegisterContext::default()
    };
    let f: unsafe extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(entry) };
    unsafe { f(&mut ctx) };
    ctx
}

#[test]
fn a_loaded_image_runs_at_its_slide() {
    // call 0x2000; lea rdx, [rip]; ret
    let bytes = [
        0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x48, 0x8D, 0x15, 0, 0, 0, 0, 0xC3,
    ];
    let mut config = LifterConfig {
        address_mode: AddressMode::Relocated,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..LifterConfig::default()
    };
    config.register_fast_import_from(
        0x2000,
        "libone.dylib",
        DynamicSymbol::new("_add_one", add_one as unsafe extern "C" fn(u64) -> u64),
        CallAbi::sysv_amd64(),
    );
    let bytes = image_bytes(config, &bytes);

    for slide in [0, 0x5_0000_0000] {
        let regions = load(&bytes, &Resolver { slide }).expect("load");
        assert_eq!(regions.len(), 1);
        let translation = &regions[0].translation;
        assert_eq!(translation.resume_points[0], DEFAULT_LIFT_BASE + slide);
        let ctx = run(translation.entry, DEFAULT_LIFT_BASE + slide, 41);
        assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
        assert_eq!(ctx.rax, 42);
        assert_eq!(ctx.rdx, DEFAULT_LIFT_BASE + 12 + slide);
    }
}

static RECORDED_CALLEE: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn record_callee(_ctx: *mut RegisterContext, callee: u64) -> u32 {
    RECORDED_CALLEE.store(callee, Ordering::Relaxed);
    0
}

#[test]
fn an_import_handed_to_the_host_is_slid() {
    // call BRIDGE_VA; call FAST_VA; ret
    let mut bytes = Vec::new();
    for target in [BRIDGE_VA, FAST_VA] {
        let next = DEFAULT_LIFT_BASE + bytes.len() as u64 + 5;
        bytes.push(0xE8);
        bytes.extend_from_slice(&((target as i64 - next as i64) as i32).to_le_bytes());
    }
    bytes.push(0xC3);
    let mut config = LifterConfig {
        address_mode: AddressMode::Relocated,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..LifterConfig::default()
    };
    config.register_bridge_import(
        BRIDGE_VA,
        BridgeImport::direct("record", record_callee as *const u8),
    );
    config.register_slow_import(FAST_VA);
    let bytes = image_bytes(config, &bytes);

    for slide in [0, 0x5_0000_0000] {
        let regions = load(&bytes, &Resolver { slide }).expect("load");
        let ctx = run(regions[0].translation.entry, DEFAULT_LIFT_BASE + slide, 0);
        assert_eq!(RECORDED_CALLEE.load(Ordering::Relaxed), BRIDGE_VA + slide);
        assert_eq!(ctx.exit_reason, ExitReason::ImportedFunctionCall as u32);
        assert_eq!(ctx.imported_function_va, FAST_VA + slide);
    }
}

#[test]
fn an_image_naming_a_missing_symbol_is_refused() {
    let bytes = image_bytes(
        config_at(0x1111_0000, 0, &SLOT_A),
        &three_imports_and_a_guest_address(),
    );
    let Err(error) = load(&bytes, &Resolver { slide: 0 }) else {
        panic!("loaded an image whose imports nothing provides");
    };
    assert!(error.to_string().contains("_fast"), "{error}");
}

#[test]
fn helpers_are_emitted_once_per_image() {
    // fld tbyte [rsi]; fld tbyte [rdx]; faddp; fstp tbyte [rcx]; ret
    let bytes = [0xDB, 0x2E, 0xDB, 0x2A, 0xDE, 0xC1, 0xDB, 0x39, 0xC3];
    let config = LifterConfig {
        address_mode: AddressMode::Relocated,
        ..LifterConfig::default()
    };
    let lift = |at| lift_to_code::<X86_64>(&bytes, at, config.clone()).expect("lift");
    let first = lift(0x1000);
    assert_eq!(first.helpers.len(), 1, "an x87 add is an outlined helper");

    let mut image = ArtifactBackend::image(None, None, &[], 0).unwrap();
    image.add_region(first).unwrap();
    let one = image.finish().text.len();

    let mut image = ArtifactBackend::image(None, None, &[], 0).unwrap();
    image.add_region(lift(0x1000)).unwrap();
    image.add_region(lift(0x5000)).unwrap();
    let two = image.finish();
    assert_eq!(two.regions.len(), 2);
    assert!(
        two.text.len() < 2 * one,
        "a second region re-emitted the helper: {one} then {}",
        two.text.len()
    );
}

#[test]
fn the_jit_resolves_the_guest_slide() {
    // lea rax, [rip]; ret
    let bytes = [0x48, 0x8D, 0x05, 0, 0, 0, 0, 0xC3];
    for slide in [0, 0x10_0000] {
        let out = LiftTest::new(&bytes)
            .no_unicorn()
            .address_mode(AddressMode::Relocated)
            .with_config(|c| c.guest_slide = slide)
            .run();
        assert_eq!(out.exit, ExitReason::Return);
        assert_eq!(out.ctx.rax, DEFAULT_LIFT_BASE + 7 + slide);
    }
}

#[test]
fn one_symbol_bound_to_two_addresses_demotes_the_second_site() {
    // call 0x2000; mov rbx, rax; call 0x3000; add rax, rbx; ret
    let bytes = [
        0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x48, 0x89, 0xC3, 0xE8, 0xF3, 0x1F, 0x00, 0x00, 0x48, 0x01,
        0xD8, 0xC3,
    ];
    let mut config = LifterConfig::default();
    for (va, f) in [
        (0x2000, forty as unsafe extern "C" fn() -> u64),
        (0x3000, two as unsafe extern "C" fn() -> u64),
    ] {
        config.register_fast_import_from(
            va,
            "libsame.dylib",
            DynamicSymbol::new("_value", f),
            CallAbi::sysv_amd64(),
        );
    }
    let code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    assert_eq!(
        code.imports.len(),
        1,
        "the second site must not reuse the first's callee"
    );
}
