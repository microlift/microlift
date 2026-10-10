//! Import bridges: `extern "C" fn(*mut RegisterContext, extra: u64) -> u32`, 0 = keep going, non-zero = exit.
//! Covers inline, exiting, effect-masked, registry-resolved, tail-jump and indirect-call bridges.

use microlift::core::bridge::{
    BridgeEffects, BridgeImport, BridgeTarget, ImportKind, ImportRegistry, RegMask,
};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

const IMPORT_VA: u64 = 0x2000;

const CALL_IMPORT_RET: [u8; 6] = [0xE8, 0xFB, 0x0F, 0x00, 0x00, 0xC3];
const RESUME_IP: u64 = 0x1005;

const JMP_IMPORT: [u8; 5] = [0xE9, 0xFB, 0x0F, 0x00, 0x00];

macro_rules! add_bridge {
    ($func:ident, $calls:ident, $seen_va:ident) => {
        static $calls: AtomicU64 = AtomicU64::new(0);
        static $seen_va: AtomicU64 = AtomicU64::new(0);

        unsafe extern "C" fn $func(ctx: *mut RegisterContext, va: u64) -> u32 {
            let ctx = unsafe { &mut *ctx };
            $calls.fetch_add(1, Ordering::SeqCst);
            $seen_va.store(va, Ordering::SeqCst);
            ctx.rax = ctx.rdi.wrapping_add(ctx.rsi);
            0
        }
    };
}

add_bridge!(bridge_add, ADD_CALLS, ADD_VA);
add_bridge!(bridge_add_registry, REGISTRY_CALLS, REGISTRY_VA);
add_bridge!(bridge_add_tail, TAIL_CALLS, TAIL_VA);

#[test]
fn direct_bridge_handles_call_inline() {
    let bridge = BridgeImport::direct("add", bridge_add as *const u8);

    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .reg(|c| {
            c.rdi = 20;
            c.rsi = 22;
        })
        .with_config(|c| c.register_bridge_import(IMPORT_VA, bridge))
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(ADD_CALLS.load(Ordering::SeqCst), 1);
    // Import VA is both the second argument and parked in the context.
    assert_eq!(ADD_VA.load(Ordering::SeqCst), IMPORT_VA);
    assert_eq!(out.ctx.imported_function_va, IMPORT_VA);
}

unsafe extern "C" fn bridge_bail(ctx: *mut RegisterContext, _va: u64) -> u32 {
    let ctx = unsafe { &mut *ctx };
    ctx.exit_reason = ExitReason::Interrupt as u32;
    ctx.exit_code = 0x1234;
    ctx.rax = 0xDEAD;
    1
}

#[test]
fn nonzero_return_exits_with_the_bridges_own_exit_state() {
    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .reg(|c| c.rbx = 0x5555)
        .with_config(|c| {
            c.register_bridge_import(IMPORT_VA, BridgeImport::direct("bail", bridge_bail as _))
        })
        .run();

    // The lifted code must not overwrite the bridge's exit state.
    assert_eq!(out.exit, ExitReason::Interrupt);
    assert_eq!(out.ctx.exit_code, 0x1234);
    assert_eq!(out.ctx.rax, 0xDEAD);
    assert_eq!(out.ctx.rip, RESUME_IP);
    assert_eq!(out.ctx.imported_function_va, IMPORT_VA);
    assert_eq!(out.ctx.rbx, 0x5555);
}

static SLOT: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

unsafe extern "C" fn bridge_set_one(ctx: *mut RegisterContext, _va: u64) -> u32 {
    unsafe { &mut *ctx }.rax = 1;
    0
}

unsafe extern "C" fn bridge_set_two(ctx: *mut RegisterContext, _va: u64) -> u32 {
    unsafe { &mut *ctx }.rax = 2;
    0
}

#[test]
fn slot_bridge_is_reloaded_before_every_call() {
    SLOT.store(bridge_set_one as *mut (), Ordering::Release);
    let bridge = BridgeImport::slot("slotted", &SLOT as *const AtomicPtr<()>);

    let mut harness = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .with_config(|c| c.register_bridge_import(IMPORT_VA, bridge))
        .compile()
        .expect("lift/compile");

    assert_eq!(harness.enter(), ExitReason::Return);
    assert_eq!(harness.ctx.rax, 1);

    // Slot retarget takes effect without a re-lift.
    SLOT.store(bridge_set_two as *mut (), Ordering::Release);

    let rsp = harness.initial_rsp;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };
    harness.ctx.rsp = rsp;
    harness.ctx.rip = DEFAULT_LIFT_BASE;
    assert_eq!(harness.enter(), ExitReason::Return);
    assert_eq!(harness.ctx.rax, 2);
}

/// Writes declared `rax` and undeclared `rbx`; the guest's SSA `rbx` must win.
unsafe extern "C" fn bridge_narrow(ctx: *mut RegisterContext, _va: u64) -> u32 {
    let ctx = unsafe { &mut *ctx };
    ctx.rax = ctx.rdi.wrapping_add(1);
    ctx.rbx = 0x99;
    0
}

#[test]
fn declared_effects_narrow_the_sync_around_the_call() {
    use iced_x86::Register;
    use microlift::arch::x86_64::abi::gpr_mask;

    let code = [
        0xBB, 0x22, 0x00, 0x00, 0x00, // mov ebx, 0x22
        0xE8, 0xF6, 0x0F, 0x00, 0x00, // call 0x2000
        0xC3, // ret
    ];
    let bridge = BridgeImport::direct("narrow", bridge_narrow as _).with_effects(
        BridgeEffects::new(gpr_mask(&[Register::RDI]), gpr_mask(&[Register::RAX])),
    );

    let out = LiftTest::new(&code)
        .no_unicorn()
        .reg(|c| c.rdi = 41)
        .with_config(|c| c.register_bridge_import(IMPORT_VA, bridge))
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(out.ctx.rbx, 0x22);
}

#[test]
fn undeclared_state_survives_an_exiting_bridge() {
    use iced_x86::Register;
    use microlift::arch::x86_64::abi::gpr_mask;

    // Registers outside `writes` must still reach the context on a bail-out.
    let bridge = BridgeImport::direct("bail", bridge_bail as _).with_effects(BridgeEffects::new(
        RegMask::NONE,
        gpr_mask(&[Register::RAX]),
    ));

    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .reg(|c| {
            c.rbx = 0x5555;
            c.r12 = 0x7777;
        })
        .with_config(|c| c.register_bridge_import(IMPORT_VA, bridge))
        .run();

    assert_eq!(out.exit, ExitReason::Interrupt);
    assert_eq!(out.ctx.rax, 0xDEAD); // declared clobber: the bridge's value wins
    assert_eq!(out.ctx.rbx, 0x5555);
    assert_eq!(out.ctx.r12, 0x7777);
}

#[derive(Debug)]
struct OneShotRegistry {
    va: u64,
}

impl ImportRegistry for OneShotRegistry {
    fn lookup(&self, va: u64) -> Option<ImportKind> {
        (va == self.va).then(|| {
            ImportKind::Bridge(BridgeImport {
                name: "registered-late",
                target: BridgeTarget::Direct(bridge_add_registry as *const u8),
                effects: BridgeEffects::full(),
            })
        })
    }
}

#[test]
fn registry_is_consulted_at_lift_time() {
    // Only the registry, consulted at lift time, makes this an import.
    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .reg(|c| {
            c.rdi = 1;
            c.rsi = 2;
        })
        .with_config(|c| {
            c.set_import_registry(std::sync::Arc::new(OneShotRegistry { va: IMPORT_VA }))
        })
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 3);
    assert_eq!(REGISTRY_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(REGISTRY_VA.load(Ordering::SeqCst), IMPORT_VA);
}

#[test]
fn unregistered_direct_call_still_exits() {
    // Empty registry: ordinary out-of-region call, not bridged.
    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .with_config(|c| c.set_import_registry(std::sync::Arc::new(OneShotRegistry { va: 0xDEAD })))
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
}

#[test]
fn import_tail_jump_calls_the_bridge_then_returns() {
    let bridge = BridgeImport::direct("add", bridge_add_tail as *const u8);

    let out = LiftTest::new(&JMP_IMPORT)
        .no_unicorn()
        .reg(|c| {
            c.rdi = 30;
            c.rsi = 12;
        })
        .with_config(|c| c.register_bridge_import(IMPORT_VA, bridge))
        .run();

    // Tail jump = bridge call, then the imported function's RET through the trampoline.
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(TAIL_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}

#[test]
fn slow_import_tail_jump_keeps_the_old_exit() {
    // A slow import has no inline form; the jump falls back to the plain exit.
    let out = LiftTest::new(&JMP_IMPORT)
        .no_unicorn()
        .with_config(|c| c.register_slow_import(IMPORT_VA))
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(out.ctx.rip, IMPORT_VA);
}

static INDIRECT_TARGET: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn bridge_indirect(ctx: *mut RegisterContext, target: u64) -> u32 {
    let ctx = unsafe { &mut *ctx };
    INDIRECT_TARGET.store(target, Ordering::SeqCst);
    ctx.rbx = 0x1234;
    0
}

unsafe extern "C" fn bridge_indirect_bail(_ctx: *mut RegisterContext, target: u64) -> u32 {
    INDIRECT_TARGET.store(target, Ordering::SeqCst);
    1
}

const CALL_RAX_RET: [u8; 3] = [0xFF, 0xD0, 0xC3];

#[test]
fn indirect_call_bridge_handles_the_call_inline() {
    INDIRECT_TARGET.store(0, Ordering::SeqCst);

    let out = LiftTest::new(&CALL_RAX_RET)
        .no_unicorn()
        .reg(|c| c.rax = 0x3000)
        .with_config(|c| {
            c.set_indirect_call_bridge(BridgeImport::direct("dispatch", bridge_indirect as _))
        })
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(INDIRECT_TARGET.load(Ordering::SeqCst), 0x3000);
    assert_eq!(out.ctx.rbx, 0x1234);
}

#[test]
fn indirect_call_bridge_can_still_demand_the_exit() {
    INDIRECT_TARGET.store(0, Ordering::SeqCst);

    let out = LiftTest::new(&CALL_RAX_RET)
        .no_unicorn()
        .reg(|c| c.rax = 0x3000)
        .with_config(|c| {
            c.set_indirect_call_bridge(BridgeImport::direct("bail", bridge_indirect_bail as _))
        })
        .run();

    // Bridge wrote nothing, so the seeded `IndirectCall` exit state survives.
    assert_eq!(out.exit, ExitReason::IndirectCall);
    assert_eq!(out.ctx.imported_function_va, 0x3000);
    assert_eq!(out.ctx.rip, 0x1002);
    assert_eq!(out.ctx.rsp, out.initial_rsp, "stack pointer unchanged");
    assert_eq!(INDIRECT_TARGET.load(Ordering::SeqCst), 0x3000);
}

#[test]
fn indirect_call_without_a_bridge_exits_as_before() {
    let out = LiftTest::new(&CALL_RAX_RET)
        .no_unicorn()
        .reg(|c| c.rax = 0x3000)
        .run();

    assert_eq!(out.exit, ExitReason::IndirectCall);
    assert_eq!(out.ctx.imported_function_va, 0x3000);
}

/// Expressible by the ABI descriptor, so it takes the fast-call path.
unsafe extern "C" fn host_double(x: u64) -> u64 {
    x.wrapping_mul(2)
}

fn double_symbol() -> microlift::function_traits::DynamicSymbol {
    microlift::function_traits::DynamicSymbol::new(
        "host_double",
        host_double as unsafe extern "C" fn(u64) -> u64,
    )
}

#[derive(Debug)]
struct FastRegistry;

impl ImportRegistry for FastRegistry {
    fn lookup(&self, va: u64) -> Option<ImportKind> {
        (va == IMPORT_VA).then(|| {
            ImportKind::Fast(microlift::core::FastImport {
                symbol: double_symbol(),
                abi: microlift::core::abi::CallAbi::sysv_amd64(),
            })
        })
    }
}

#[test]
fn registry_supplied_fast_import_is_built_lazily() {
    // Fast-call data can only be built once the registry is queried at the call site.
    let out = LiftTest::new(&CALL_IMPORT_RET)
        .no_unicorn()
        .reg(|c| c.rdi = 21)
        .with_config(|c| c.set_import_registry(std::sync::Arc::new(FastRegistry)))
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
}

#[test]
fn fast_import_tail_jump_calls_inline_then_returns() {
    let out = LiftTest::new(&JMP_IMPORT)
        .no_unicorn()
        .reg(|c| c.rdi = 21)
        .with_config(|c| {
            c.register_fast_import(
                IMPORT_VA,
                double_symbol(),
                microlift::core::abi::CallAbi::sysv_amd64(),
            )
        })
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}
