//! Runtime-supplied import bridges: inline handling, exiting bridges, live
//! slots, declared effect masks, the lift-time import registry, tail jumps, and
//! the indirect-call bridge.
//!
//! Every bridge here follows the one contract:
//! `extern "C" fn(*mut RegisterContext, extra: u64) -> u32`, returning 0 for
//! "handled, keep going" and non-zero for "exit now".

use microlift::core::bridge::{
    BridgeEffects, BridgeImport, BridgeTarget, ImportKind, ImportRegistry, RegMask,
};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

const IMPORT_VA: u64 = 0x2000;

/// `call 0x2000 ; ret`, lifted at [`DEFAULT_LIFT_BASE`] (0x1000).
const CALL_IMPORT_RET: [u8; 6] = [0xE8, 0xFB, 0x0F, 0x00, 0x00, 0xC3];
/// The resume point: the `ret` right after the `call`.
const RESUME_IP: u64 = 0x1005;

/// `jmp 0x2000` - an import stub tail-jumped into.
const JMP_IMPORT: [u8; 5] = [0xE9, 0xFB, 0x0F, 0x00, 0x00];

/// Define an "adds its arguments" bridge with its own call counter, so tests
/// running in parallel never share one.
macro_rules! add_bridge {
    ($func:ident, $calls:ident, $seen_va:ident) => {
        static $calls: AtomicU64 = AtomicU64::new(0);
        static $seen_va: AtomicU64 = AtomicU64::new(0);

        /// Handles the call inline: `rax = rdi + rsi`.
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

    // No VM exit at the call: the bridge returned 0 and the guest ran on to its
    // own `ret`, which is what reached the exit trampoline.
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(ADD_CALLS.load(Ordering::SeqCst), 1);
    // The import VA is passed as the second argument *and* parked in the
    // context, so one compiled bridge can serve every site that names it.
    assert_eq!(ADD_VA.load(Ordering::SeqCst), IMPORT_VA);
    assert_eq!(out.ctx.imported_function_va, IMPORT_VA);
}

/// Refuses to handle the call: writes its own exit state and returns non-zero.
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

    // `exit_reason`/`exit_code` are exactly what the bridge wrote - the lifted
    // code does not overwrite them on the way out.
    assert_eq!(out.exit, ExitReason::Interrupt);
    assert_eq!(out.ctx.exit_code, 0x1234);
    assert_eq!(out.ctx.rax, 0xDEAD);
    // The PC published before the call is the resume point after the `CALL`.
    assert_eq!(out.ctx.rip, RESUME_IP);
    assert_eq!(out.ctx.imported_function_va, IMPORT_VA);
    // Guest state the bridge did not touch survives the exit.
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

    // Re-point the slot. The *already compiled* code must pick this up with no
    // re-lift - that is the whole point of `BridgeTarget::Slot`.
    SLOT.store(bridge_set_two as *mut (), Ordering::Release);

    let rsp = harness.initial_rsp;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };
    harness.ctx.rsp = rsp;
    harness.ctx.rip = DEFAULT_LIFT_BASE;
    assert_eq!(harness.enter(), ExitReason::Return);
    assert_eq!(harness.ctx.rax, 2);
}

/// Writes `rax` (declared) *and* `rbx` (not declared), so a mask that omits
/// `rbx` must leave the guest's own SSA value winning.
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

    // mov ebx, 0x22 ; call 0x2000 ; ret
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
    // `rdi` was declared read, so the bridge saw the guest's value...
    assert_eq!(out.ctx.rax, 42);
    // ...and `rbx` was not declared written, so the bridge's store to it is
    // discarded when the guest's live SSA value is synced at the exit.
    assert_eq!(out.ctx.rbx, 0x22);
}

#[test]
fn undeclared_state_survives_an_exiting_bridge() {
    use iced_x86::Register;
    use microlift::arch::x86_64::abi::gpr_mask;

    // Same narrow mask, but the bridge bails out. Registers outside `writes`
    // must still reach the context, or the host would see stale state.
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
    // Nothing is registered in `config.imports`; the address is only an import
    // because the registry says so when the call site is lifted.
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
    // Same program, empty registry: the call is an ordinary out-of-region
    // direct call and must still leave through the `UnknownJump`/`DirectCall`
    // path rather than being silently bridged.
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

    // The stub's `JMP` became: call the bridge, then perform the return the
    // imported function's own `RET` would have - straight out through the exit
    // trampoline that was sitting at `[rsp]`.
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 42);
    assert_eq!(TAIL_CALLS.load(Ordering::SeqCst), 1);
    // The return address was popped exactly once.
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}

#[test]
fn slow_import_tail_jump_keeps_the_old_exit() {
    // A slow import has no inline form, so the tail jump must fall back to the
    // ordinary out-of-region jump exit rather than inventing a return.
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

/// `call rax ; ret`
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

    // No `IndirectCall` exit at all: the runtime dispatched it host-side.
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

    // The bridge wrote nothing, so the context still describes the well-formed
    // `IndirectCall` exit the lifted code seeded before calling it.
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

/// A plain host function whose signature the ABI descriptor *can* express, so
/// it goes down the fast-call path rather than through a bridge.
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
    // The fast-call data for this import cannot have been prebuilt when the
    // engine was constructed - only the registry knows about the address, and
    // only once the call site is lifted.
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
