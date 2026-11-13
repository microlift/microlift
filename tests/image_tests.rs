//! Relocatable lifts and the multi-region image: no host pointer reaches the code, and an
//! image loaded at any slide runs like the JIT.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use microlift::arch::x86_64::X86_64;
use std::sync::Arc;

use microlift::backend::ArtifactBackend;
use microlift::core::abi::CallAbi;
use microlift::core::bridge::BridgeImport;
use microlift::core::symbol::{HostSymbol, HostSymbolKind};
use microlift::core::{AddressMode, LifterConfig, lift_to_code};
use microlift::cranelift_codegen::isa::TargetIsa;
use microlift::cranelift_codegen::settings::{self, Configurable};
use microlift::function_traits::{DynamicSymbol, FunctionArgumentType};
use microlift::image::GotTarget;
use microlift::image::compile::PicCompiler;
use microlift::image::file::Writer;
use microlift::runtime::image::{LoadedRegion, MappedImage, SymbolResolver};
use microlift::runtime::jit::JitCompiler;
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

fn pic_isa() -> Arc<dyn TargetIsa> {
    let mut flags = settings::builder();
    flags.set("is_pic", "true").unwrap();
    ArtifactBackend::get_isa(None, Some(settings::Flags::new(flags)), &[]).expect("isa")
}

fn scratch_path() -> std::path::PathBuf {
    use std::sync::atomic::AtomicUsize;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "microlift-image-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// A scratch file, already unlinked.
fn scratch() -> std::fs::File {
    let path = scratch_path();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    file
}

/// An image file, linked until dropped: Darwin will not map an unlinked file's code
/// into a reservation.
struct ImageFile {
    file: std::fs::File,
    path: std::path::PathBuf,
}

impl std::ops::Deref for ImageFile {
    type Target = std::fs::File;
    fn deref(&self) -> &std::fs::File {
        &self.file
    }
}

impl std::ops::DerefMut for ImageFile {
    fn deref_mut(&mut self) -> &mut std::fs::File {
        &mut self.file
    }
}

impl Drop for ImageFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// An image file holding a region lifted at each of `at`, open read-only as a
/// loader opens one: Darwin maps no file executable while it is open for writing.
fn image_file(config: LifterConfig, bytes: &[u8], at: &[u64]) -> ImageFile {
    let fingerprint = config.codegen_fingerprint();
    let compiler = PicCompiler::new(pic_isa()).unwrap();
    let path = scratch_path();
    let out = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let mut writer = Writer::new(out, scratch(), 1, fingerprint).unwrap();
    for &base in at {
        let code = lift_to_code::<X86_64>(bytes, base, config.clone()).expect("lift");
        writer
            .add(&compiler.compile(code).expect("compile"))
            .unwrap();
    }
    drop(writer.finish(b"user").unwrap());
    ImageFile {
        file: std::fs::File::open(&path).unwrap(),
        path,
    }
}

fn image_bytes(config: LifterConfig, bytes: &[u8]) -> Vec<u8> {
    use std::io::{Read, Seek};
    let mut file = image_file(config, bytes, &[DEFAULT_LIFT_BASE]);
    let mut out = Vec::new();
    file.rewind().unwrap();
    file.read_to_end(&mut out).unwrap();
    out
}

fn load(file: &std::fs::File, resolver: &Resolver) -> MappedImage {
    let mut image = MappedImage::open(file).expect("map");
    image.bind(resolver).expect("bind");
    image.map_code().expect("map code");
    image
}

fn region_at(image: &MappedImage, pc: u64) -> Option<LoadedRegion> {
    image.region(image.lookup(pc)?).expect("region")
}

static SLOT_A: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SLOT_B: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

#[test]
fn an_image_holds_no_host_or_guest_run_time_address() {
    let bytes = three_imports_and_a_guest_address();
    let first = image_bytes(config_at(0x1111_0000, 0, &SLOT_A), &bytes);
    let second = image_bytes(config_at(0x2222_0000, 0x7000_0000, &SLOT_B), &bytes);
    assert_eq!(first, second);

    let compiler = PicCompiler::new(pic_isa()).unwrap();
    let code = lift_to_code::<X86_64>(
        &bytes,
        DEFAULT_LIFT_BASE,
        config_at(0x1111_0000, 0, &SLOT_A),
    )
    .expect("lift");
    let object = compiler.compile(code).unwrap();
    let names: Vec<String> = object
        .code
        .targets
        .iter()
        .filter_map(|t| match t {
            GotTarget::Host(host) => Some(format!("{}`{}", host.provider, host.name)),
            GotTarget::GuestSlide => Some("slide".into()),
            _ => None,
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
    let file = image_file(config, &bytes, &[DEFAULT_LIFT_BASE]);

    for slide in [0, 0x5_0000_0000] {
        let image = load(&file, &Resolver { slide });
        assert_eq!(image.region_count(), 1);
        assert_eq!(image.user(), b"user");
        #[cfg(target_os = "macos")]
        assert!(image.in_place(), "the signed code was copied, not mapped");
        let region = region_at(&image, DEFAULT_LIFT_BASE + slide).expect("a region");
        let translation = &region.translation;
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
    let file = image_file(config, &bytes, &[DEFAULT_LIFT_BASE]);

    for slide in [0, 0x5_0000_0000] {
        let image = load(&file, &Resolver { slide });
        let region = region_at(&image, DEFAULT_LIFT_BASE + slide).expect("a region");
        let ctx = run(region.translation.entry, DEFAULT_LIFT_BASE + slide, 0);
        assert_eq!(RECORDED_CALLEE.load(Ordering::Relaxed), BRIDGE_VA + slide);
        assert_eq!(ctx.exit_reason, ExitReason::ImportedFunctionCall as u32);
        assert_eq!(ctx.imported_function_va, FAST_VA + slide);
    }
}

#[test]
fn a_region_naming_a_missing_symbol_is_refused() {
    let file = image_file(
        config_at(0x1111_0000, 0, &SLOT_A),
        &three_imports_and_a_guest_address(),
        &[DEFAULT_LIFT_BASE],
    );
    let image = load(&file, &Resolver { slide: 0 });
    let index = image.lookup(DEFAULT_LIFT_BASE).expect("indexed");
    assert!(
        image.region(index).unwrap().is_none(),
        "published a region whose imports nothing provides"
    );
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

    let compiler = PicCompiler::new(pic_isa()).unwrap();
    let object = compiler.compile(first).unwrap();
    let helper = object.helpers[0].1.text.len() as u64;
    let region = object.code.text.len() as u64;

    let mut writer = Writer::new(scratch(), scratch(), 1, 0).unwrap();
    writer.add(&object).unwrap();
    writer
        .add(&compiler.compile(lift(0x5000)).unwrap())
        .unwrap();
    assert_eq!(writer.region_count(), 2);
    assert!(
        writer.text_len() < 2 * helper + 2 * region,
        "a second region re-emitted the helper: {} bytes",
        writer.text_len()
    );
}

#[test]
fn regions_are_found_at_every_resume_point_and_nowhere_else() {
    // lea rax, [rip]; ret
    let bytes = [0x48, 0x8D, 0x05, 0, 0, 0, 0, 0xC3];
    let config = LifterConfig {
        address_mode: AddressMode::Relocated,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..LifterConfig::default()
    };
    let file = image_file(config, &bytes, &[0x1000, 0x9000, 0x5000]);
    let image = load(&file, &Resolver { slide: 0x40_0000 });
    for (pc, index) in [(0x1000, 0), (0x9000, 1), (0x5000, 2)] {
        assert_eq!(image.lookup(pc + 0x40_0000), Some(index));
        let region = image.region(index).unwrap().unwrap();
        let ctx = run(region.translation.entry, pc + 0x40_0000, 0);
        assert_eq!(ctx.rax, pc + 7 + 0x40_0000);
    }
    assert_eq!(image.lookup(0x1000), None);
    assert_eq!(image.lookup(0x1004 + 0x40_0000), None);
}

#[test]
fn the_position_independent_jit_runs_at_any_slide() {
    // call 0x2000; lea rdx, [rip]; ret
    let bytes = [
        0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x48, 0x8D, 0x15, 0, 0, 0, 0, 0xC3,
    ];
    let jit = JitCompiler::new(pic_isa()).unwrap();
    for slide in [0, 0x5_0000_0000] {
        let mut config = LifterConfig {
            address_mode: AddressMode::Relocated,
            guest_slide: slide,
            exit_trampoline_address: Some(EXIT_TRAMPOLINE),
            ..LifterConfig::default()
        };
        config.register_fast_import_from(
            0x2000,
            "libone.dylib",
            DynamicSymbol::new("_add_one", add_one as unsafe extern "C" fn(u64) -> u64),
            CallAbi::sysv_amd64(),
        );
        let code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config).unwrap();
        let compiled = jit.compile(code).unwrap();
        let ctx = run(compiled.region.entry, DEFAULT_LIFT_BASE + slide, 41);
        assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
        assert_eq!(ctx.rax, 42);
        assert_eq!(ctx.rdx, DEFAULT_LIFT_BASE + 12 + slide);
    }
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
