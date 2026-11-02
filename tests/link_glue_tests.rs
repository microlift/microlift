//! The per-hop cost of a pinned link, read off the host machine code.
//! `pinned_link_tests.rs` covers what a chain computes; this covers what it costs.

use microlift::arch::x86_64::{X86_64, pin};
use microlift::core::link::BlockLinkConfig;
use microlift::core::pin::{PinnedAbi, PinnedReg, RegisterBudget};
use microlift::core::{AddressMode, LiftedCode, LifterConfig, lift_to_code};
use microlift::cranelift_codegen::Context;
use microlift::cranelift_codegen::control::ControlPlane;
use microlift::cranelift_codegen::ir::Function;
use microlift::cranelift_codegen::settings::{self, Configurable};
use microlift::target_lexicon::Triple;

/// Mono's shape: a frame, a helper through `movabs r11; call r11`, a virtual call, a `ret`.
#[rustfmt::skip]
const MONO_METHOD: &[u8] = &[
    0x55,                                                       // push rbp
    0x48, 0x89, 0xE5,                                           // mov rbp, rsp
    0x41, 0x56,                                                 // push r14
    0x48, 0x83, 0xEC, 0x10,                                     // sub rsp, 0x10
    0x4C, 0x8B, 0x37,                                           // mov r14, [rdi]
    0x49, 0xBB, 0x00, 0x10, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, // movabs r11, 0x700000001000
    0x41, 0xFF, 0xD3,                                           // call r11
    0x85, 0xC0,                                                 // test eax, eax
    0x74, 0x07,                                                 // je +7
    0x49, 0x8B, 0x06,                                           // mov rax, [r14]
    0xFF, 0x50, 0x40,                                           // call [rax+0x40]
    0x90,                                                       // nop
    0x48, 0x83, 0xC4, 0x10,                                     // add rsp, 0x10
    0x41, 0x5E,                                                 // pop r14
    0x5D,                                                       // pop rbp
    0xC3,                                                       // ret
];

/// Rosemary's priority order, cut to what the host's tail convention carries.
fn host_abi() -> PinnedAbi {
    const PRIORITY: [PinnedReg; 16] = [
        pin::RSP,
        pin::RDI,
        pin::RAX,
        pin::RBP,
        pin::RSI,
        pin::RDX,
        pin::RCX,
        pin::RBX,
        pin::R8,
        pin::R9,
        pin::R10,
        pin::R11,
        pin::R12,
        pin::R13,
        pin::R14,
        pin::R15,
    ];
    let budget = RegisterBudget::for_arch(Triple::host().architecture)
        .expect("the host records a tail-call budget");
    PinnedAbi::new().with_all(PRIORITY.into_iter().take(budget.gpr as usize - 1))
}

fn lift_linked() -> LiftedCode {
    let mut config = LifterConfig::default();
    config.address_mode = AddressMode::Direct;
    config.block_link = Some(BlockLinkConfig::new(65536).with_ways(4).pinned(host_abi()));
    config.use_vector_xmm = true;
    config.strict_resume = true;
    lift_to_code::<X86_64>(MONO_METHOD, 0x1_0000, config).expect("the region lifts")
}

/// The host's VCode listing, which names every prologue save as an `unwind SaveReg`.
fn host_listing(func: Function) -> String {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").unwrap();
    flags.set("preserve_frame_pointers", "true").unwrap();
    let isa = microlift::cranelift_native::builder()
        .expect("the host is a Cranelift target")
        .finish(settings::Flags::new(flags))
        .expect("the host ISA builds");
    let mut ctx = Context::for_function(func);
    ctx.set_disasm(true);
    let compiled = ctx
        .compile(&*isa, &mut ControlPlane::default())
        .expect("the function compiles");
    compiled.vcode.clone().expect("disassembly was requested")
}

fn prologue_saves(listing: &str) -> usize {
    listing.matches("unwind SaveReg").count()
}

/// Only the entry thunk returns to host code, so a linked region has no caller whose
/// registers it must keep: every hop would otherwise spill and reload them.
#[test]
fn a_linked_region_saves_no_callee_saved_registers() {
    let listing = host_listing(lift_linked().func);
    assert_eq!(prologue_saves(&listing), 0, "{listing}");
}

/// The other half: the thunk is a host function, and it calls a body that keeps nothing.
#[test]
fn the_entry_thunk_saves_what_the_host_expects_kept() {
    let thunk = lift_linked()
        .entry_thunk
        .expect("a pinned lift has a thunk");
    let listing = host_listing(thunk);
    let host_callee_saved_gprs = if cfg!(target_arch = "aarch64") { 10 } else { 5 };
    assert!(
        prologue_saves(&listing) >= host_callee_saved_gprs,
        "{listing}"
    );
}
