//! Tests targeting stack-pointer misalignment bugs and related stack
//! correctness oversights in the lifter.
//!
//! # Bugs covered
//!
//! ## Bug 1 - Out-of-region direct CALL does not adjust RSP (critical)
//!
//! In `control_flow.rs:135-163`, when `CALL target` has a statically-known
//! target outside the lifted region and is **not** a registered import, the
//! lifter emits a `TransferKind::DirectCall` VM exit without:
//!   - Decrementing RSP by 8, and
//!   - Writing the return address to `[new RSP]`.
//!
//! `TransferKind::DirectCall` documentation states:
//!   "A return address has already been pushed onto the guest stack"
//! - this is false. The guest RSP is 8 bytes too high when the runtime
//! dispatches the callee. If that callee executes a `RET`, it pops from
//! the wrong location, corrupting control flow.
//!
//! Compare with the in-region CALL path (lines 165-183) which correctly
//! adjusts RSP and pushes the return address.
//!
//! ## Bug 2 - PUSH imm8 sign-extension uses wrong accessor (medium)
//!
//! `handle_push` in `memory.rs:80-83` calls `immediate32to64()` for ALL
//! immediate variants, including `Immediate8to64` (which covers
//! `push imm8` in 64-bit mode). The VSA pass (`vsa.rs:460`) correctly
//! uses `immediate8to64()` for this variant.
//!
//! In iced-x86, `immediate32to64()` sign-extends a **32-bit** field; for
//! `Immediate8to64` instructions the stored 32-bit field may equal the raw
//! unsigned 8-bit value (0–255). For negative 8-bit values (e.g. `push -1`
//! = `6A FF`), this would push 255 instead of 0xFFFF_FFFF_FFFF_FFFF.
//!
//! ## IndirectCall exit reason - unregistered indirect CALL
//!
//! `emit_indirect_call_exit` emits `ExitReason::IndirectCall` (distinct from
//! `ImportedFunctionCall`) and leaves RSP unchanged. The host must decide how
//! to dispatch the callee:
//! - **Native function**: call via C and resume at `ctx.rip` - no stack
//!   adjustment needed.
//! - **Guest code**: push `ctx.rip` onto `ctx.rsp` before re-entering the
//!   guest so the callee's eventual `RET` finds the correct return address.

use microlift::context::TransferKind;
use microlift::testkit::prelude::*;

// Test helpers

/// Lift `bytes` at VA `0x1000`, compile with the JIT backend, run with `ctx`,
/// and return the updated context.
///
/// Unlike the helpers in `stack_tests.rs` this does **not** touch `ctx.rsp`
/// or `ctx.image_base`; the caller is responsible for setting them.
fn compile_and_run(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    // `RetAddr::None` + a caller-provided `rsp`/`image_base` reproduce this
    // helper's "don't touch the stack" contract (the testkit harness leaves a
    // preset non-zero `rsp`/`image_base` alone).
    LiftTest::new(bytes)
        .ctx(ctx)
        .ret_addr(RetAddr::None)
        .run()
        .ctx
}

/// Like `compile_and_run` but installs an automatic stack with an exit
/// trampoline at the bottom (top-of-guest-stack) and sets `image_base = 0x1000`.
/// Returns `(ctx_after, initial_rsp)` where `initial_rsp` is the value of
/// `ctx.rsp` at the point the lifted function is entered.
fn compile_and_run_with_stack(bytes: &[u8], ctx: RegisterContext) -> (RegisterContext, u64) {
    let out = LiftTest::new(bytes).ctx(ctx).run();
    (out.ctx, out.initial_rsp)
}

// Bug 1: out-of-region direct CALL does not push return address

/// An out-of-region `CALL` must:
///   1. Decrement RSP by 8.
///   2. Write the return address (next_ip rebased via `image_base`) to [new RSP].
///   3. Exit with `UnknownJump` / `DirectCall`.
///
/// Currently the lifter performs step 3 only (steps 1 and 2 are missing),
/// leaving RSP 8 bytes too high.
///
/// Byte sequence at `0x1000`:
///   `E8 FB 0F 00 00`  ->  `call 0x2000`
///   (rel32 = 0x2000 − next_ip(0x1005) = 0xFFB)
///
/// image_base = 0x1000, so rebased return address = 0x1000 + (0x1005 − 0x1000) = 0x1005.
#[test]
fn test_out_of_region_call_decrements_rsp() {
    // call 0x2000  (5 bytes, target is outside the 5-byte region)
    let bytes = [0xE8u8, 0xFB, 0x0F, 0x00, 0x00];

    // Provide a stack buffer large enough for the expected return-address slot.
    let mut stack = [0u64; 16];
    let stack_top = stack.as_mut_ptr() as u64 + 16 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    // Point RSP just above the buffer; no trampoline needed - the code exits
    // via DirectCall before reaching any RET.
    ctx.rsp = stack_top;
    let initial_rsp = ctx.rsp;

    let ctx = compile_and_run(&bytes, ctx);

    // Confirm exit kind.
    assert_eq!(ctx.exit_reason, ExitReason::UnknownJump as u32);
    assert_eq!(ctx.exit_code, TransferKind::DirectCall as u32);

    // Bug 1 assertion: RSP must have been decremented by 8.
    assert_eq!(
        ctx.rsp,
        initial_rsp - 8,
        "Bug 1: out-of-region CALL must decrement RSP by 8 (return address slot)"
    );
}

/// Companion to the previous test: the word written at [new RSP] must equal
/// the rebased return address (the instruction immediately after the CALL).
///
/// With `image_base = 0x1000` and `lift_base = 0x1000`, the return address is
/// `image_base + (next_ip − lift_base) = 0x1000 + (0x1005 − 0x1000) = 0x1005`.
///
/// Note: we start `ctx.rsp` one slot below the top of the buffer so that
/// even if Bug 1 is present and RSP is not decremented, reading from `ctx.rsp`
/// is still within the allocated array (avoiding UB in the failing case).
#[test]
fn test_out_of_region_call_writes_return_address() {
    let bytes = [0xE8u8, 0xFB, 0x0F, 0x00, 0x00]; // call 0x2000

    let mut stack = [0u64; 16];
    // Start one slot below the top: slot 14 is the "initial RSP" slot.
    // If Bug 1 is present RSP stays at stack[14]; if fixed it moves to stack[13].
    // Either way the read is within bounds.
    let stack_base = stack.as_mut_ptr() as u64;
    let initial_rsp = stack_base + 14 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    ctx.rsp = initial_rsp;

    let ctx = compile_and_run(&bytes, ctx);

    assert_eq!(ctx.exit_reason, ExitReason::UnknownJump as u32);
    assert_eq!(ctx.exit_code, TransferKind::DirectCall as u32);

    // RSP must have been decremented before we read the slot it points to.
    assert_eq!(
        ctx.rsp,
        initial_rsp - 8,
        "Bug 1: RSP must be decremented before checking the stored return address"
    );

    // Bug 1 assertion: return address must be present at the new RSP.
    let stored: u64 = unsafe { *(ctx.rsp as *const u64) };
    assert_eq!(
        stored,
        ctx.image_base + 5, // 0x1005 - the instruction after the CALL
        "Bug 1: return address was not written to [RSP] by out-of-region CALL"
    );
}

/// After a runtime `DirectCall` exit, the callee virtual address stored in
/// `ctx.imported_function_va` must be the rebased target, and `ctx.rip`
/// must be the rebased return address (resume point).  This validates the
/// existing (correct) parts of the exit path.
#[test]
fn test_out_of_region_call_exit_metadata_is_correct() {
    let bytes = [0xE8u8, 0xFB, 0x0F, 0x00, 0x00]; // call 0x2000

    let mut stack = [0u64; 16];
    let stack_top = stack.as_mut_ptr() as u64 + 16 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    ctx.rsp = stack_top;

    let ctx = compile_and_run(&bytes, ctx);

    // Exit kind.
    assert_eq!(ctx.exit_reason, ExitReason::UnknownJump as u32);
    assert_eq!(ctx.exit_code, TransferKind::DirectCall as u32);

    // Callee VA: image_base + (0x2000 − lift_base) = 0x1000 + 0x1000 = 0x2000.
    assert_eq!(ctx.imported_function_va, 0x2000);

    // RIP (resume / return address): image_base + (0x1005 − 0x1000) = 0x1005.
    assert_eq!(ctx.rip, 0x1005);
}

// Bug 2: PUSH imm8 (Immediate8to64) sign-extension

/// `push imm8` (opcode `6A`) sign-extends the 8-bit operand to 64 bits.
/// `push -1` (`6A FF`) must store `0xFFFF_FFFF_FFFF_FFFF` on the stack.
///
/// `handle_push` calls `immediate32to64()` for `Immediate8to64` operands.
/// The VSA pass uses the dedicated `immediate8to64()` accessor.  If the
/// iced-x86 `immediate32to64()` method does not correctly sign-extend 8->64
/// for this operand kind, negative byte values will be wrong on the stack.
#[test]
fn test_push_neg1_imm8_sign_extends() {
    // 6A FF  push -1   (Immediate8to64: 0xFF as i8 = −1 -> 0xFFFF_FFFF_FFFF_FFFF)
    // 58     pop rax
    // C3     ret
    let bytes = [0x6Au8, 0xFF, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax,
        u64::MAX,
        "Bug 2: push -1 (imm8, 6A FF) must sign-extend to 0xFFFF_FFFF_FFFF_FFFF"
    );
}

/// Boundary case: `push -128` (`6A 80`) - the most-negative 8-bit value.
/// Sign-extended to 64 bits: `0xFFFF_FFFF_FFFF_FF80`.
#[test]
fn test_push_neg128_imm8_sign_extends() {
    // 6A 80  push -128  (0x80 as i8 = −128 -> 0xFFFF_FFFF_FFFF_FF80)
    // 58     pop rax
    // C3     ret
    let bytes = [0x6Au8, 0x80, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax, 0xFFFF_FFFF_FFFF_FF80u64,
        "Bug 2: push -128 (imm8, 6A 80) must sign-extend to 0xFFFF_FFFF_FFFF_FF80"
    );
}

/// Non-boundary negative value: `push -100` (`6A 9C`).
/// Sign-extended: `0xFFFF_FFFF_FFFF_FF9C`.
#[test]
fn test_push_neg100_imm8_sign_extends() {
    // 6A 9C  push -100  (0x9C as i8 = −100 -> 0xFFFF_FFFF_FFFF_FF9C)
    // 58     pop rax
    // C3     ret
    let bytes = [0x6Au8, 0x9C, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax, 0xFFFF_FFFF_FFFF_FF9Cu64,
        "Bug 2: push -100 (imm8, 6A 9C) must sign-extend to 0xFFFF_FFFF_FFFF_FF9C"
    );
}

/// Sanity check: `push imm32` (opcode `68`, `Immediate32to64`) already uses
/// `immediate32to64()` directly - this should be correct already.
/// `push 0xFFFF_FFFF` sign-extends to `0xFFFF_FFFF_FFFF_FFFF`.
/// Expected to pass regardless of whether Bug 2 is fixed.
#[test]
fn test_push_neg1_imm32_sign_extends_sanity() {
    // 68 FF FF FF FF  push 0xFFFF_FFFF  (Immediate32to64 -> 0xFFFF_FFFF_FFFF_FFFF)
    // 58              pop rax
    // C3              ret
    let bytes = [0x68u8, 0xFF, 0xFF, 0xFF, 0xFF, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax,
        u64::MAX,
        "push 0xFFFFFFFF (imm32, Immediate32to64) must sign-extend to 0xFFFF_FFFF_FFFF_FFFF"
    );
}

/// Sanity check: small positive `push imm8` values work for both the
/// correct and incorrect immediate accessor, so they don't distinguish
/// the bug.  Verify that `push 1` (6A 01) gives rax = 1.
#[test]
fn test_push_positive_imm8_unaffected() {
    // 6A 01  push 1
    // 58     pop rax
    // C3     ret
    let bytes = [0x6Au8, 0x01, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 1u64);
}

// IndirectCall exit reason contract

/// `CALL RAX` (indirect call via register) exits with `ExitReason::IndirectCall`.
/// RSP is intentionally left unchanged - the host decides whether to dispatch
/// the callee as a native function (no stack adjustment needed) or as guest
/// code (host must push `ctx.rip` onto `ctx.rsp` before re-entering).
///
/// Byte sequence at `0x1000`:
///   `48 C7 C0 00 20 00 00`  ->  `mov rax, 0x2000`  (7 bytes)
///   `FF D0`                 ->  `call rax`          (2 bytes, next_ip = 0x1009)
#[test]
fn test_indirect_call_via_reg_exits_with_indirect_call_reason() {
    #[rustfmt::skip]
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x00, 0x20, 0x00, 0x00, // mov rax, 0x2000
        0xFF, 0xD0,                                  // call rax
    ];

    let mut stack = [0u64; 16];
    let stack_top = stack.as_mut_ptr() as u64 + 16 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    ctx.rsp = stack_top;
    let initial_rsp = ctx.rsp;

    let ctx = compile_and_run(&bytes, ctx);

    // Must use the dedicated IndirectCall exit reason, not ImportedFunctionCall.
    assert_eq!(ctx.exit_reason, ExitReason::IndirectCall as u32);
    // imported_function_va holds the raw runtime target (value of RAX).
    assert_eq!(ctx.imported_function_va, 0x2000);
    // rip holds the resume point (instruction after the CALL).
    assert_eq!(ctx.rip, ctx.image_base + 9);
    // RSP is unchanged - host is responsible for stack management.
    assert_eq!(
        ctx.rsp, initial_rsp,
        "IndirectCall must leave RSP unchanged; host manages the stack"
    );
}

/// `CALL [mem]` (indirect call via memory) produces the same `IndirectCall`
/// contract as `CALL reg`. This uses a stack slot as the pointer target.
#[test]
fn test_indirect_call_via_mem_exits_with_indirect_call_reason() {
    // Store target address 0x3000 at [RSP-16], then CALL [RSP-16].
    // mov qword [rsp-16], 0x3000   (48 C7 44 24 F0 00 30 00 00)  9 bytes
    // call [rsp-16]                (FF 54 24 F0)                  4 bytes  next_ip = 0x100D
    #[rustfmt::skip]
    let bytes = [
        0x48u8, 0xC7, 0x44, 0x24, 0xF0, 0x00, 0x30, 0x00, 0x00, // mov qword [rsp-16], 0x3000
        0xFF, 0x54, 0x24, 0xF0,                                   // call [rsp-16]
    ];

    let mut stack = [0u64; 32];
    let stack_top = stack.as_mut_ptr() as u64 + 32 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    ctx.rsp = stack_top;
    let initial_rsp = ctx.rsp;

    let ctx = compile_and_run(&bytes, ctx);

    assert_eq!(ctx.exit_reason, ExitReason::IndirectCall as u32);
    assert_eq!(ctx.imported_function_va, 0x3000);
    assert_eq!(ctx.rip, ctx.image_base + 13); // next_ip = 0x100D
    assert_eq!(
        ctx.rsp, initial_rsp,
        "IndirectCall must leave RSP unchanged; host manages the stack"
    );
}

// Baseline: in-region CALL/RET correctly maintains RSP (should always pass)

/// An in-region `CALL` must decrement RSP and push the return address, and
/// the matching `RET` must pop it.  RSP must be balanced across the pair.
///
/// This test verifies the correct baseline behaviour that out-of-region and
/// indirect CALLs should also exhibit after their bugs are fixed.
#[test]
fn test_in_region_call_ret_rsp_balance() {
    // 0x1000: call 0x1006    E8 01 00 00 00  (5 bytes)
    // 0x1005: ret            C3              (1 byte)  ← outer exit
    // 0x1006: ret            C3              (1 byte)  ← inner: returns to 0x1005
    #[rustfmt::skip]
    let bytes = [
        0xE8u8, 0x01, 0x00, 0x00, 0x00, // call 0x1006
        0xC3,                            // ret (outer - pops trampoline, exits Return)
        0xC3,                            // ret (callee - returns to 0x1005)
    ];

    let (ctx, initial_rsp) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);

    // The outer `ret` popped the trampoline from the initial stack slot, so
    // RSP = initial_rsp + 8 after the function returns.
    assert_eq!(
        ctx.rsp,
        initial_rsp + 8,
        "RSP must advance past the trampoline slot after the final RET"
    );
}

/// After a `CALL` into an in-region callee, the callee sees RSP equal to
/// (caller RSP − 8).  We capture this by saving RSP inside the callee.
///
/// Layout:
///   0x1000: call 0x1007  (E8 02 00 00 00 - 5 bytes)
///   0x1005: ret          (C3)
///   0x1006: (padding)    (90 - nop, makes 0x1007 a distinct block leader)
///   0x1007: mov rax, rsp (48 89 E0 - 3 bytes, saves RSP into RAX)
///   0x100A: ret          (C3)
#[test]
fn test_in_region_call_callee_rsp_is_decremented() {
    // call to 0x1007 - rel32 = 0x1007 − (0x1000 + 5) = 2
    #[rustfmt::skip]
    let bytes = [
        0xE8u8, 0x02, 0x00, 0x00, 0x00, // 0x1000: call 0x1007
        0xC3,                            // 0x1005: ret (outer)
        0x90,                            // 0x1006: nop (padding)
        0x48, 0x89, 0xE0,                // 0x1007: mov rax, rsp
        0xC3,                            // 0x100A: ret
    ];

    let (ctx, initial_rsp) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);

    // The CALL pushed 8 bytes; when the callee ran `mov rax, rsp`, RSP was
    // initial_rsp − 8.
    assert_eq!(
        ctx.rax,
        initial_rsp - 8,
        "Callee must observe RSP = caller_RSP − 8 (return address slot)"
    );
}

// `AND/OR/ADD/... r/m64, imm8` encode the immediate as a sign-extended byte.
// Reading it with `immediate64()` yields the raw unsigned byte (`0xf0` -> 240)
// instead of the sign-extended value (`-16`); for `and rsp, -16` that masks only
// the low byte of RSP and corrupts the stack pointer. Every SysV `_start` opens
// with `and rsp, -16`, so this must round-trip exactly.

#[test]
fn and_r64_imm8_sign_extends() {
    // movabs rax, 0x1_2345_6789 ; and rax, -16   (48 83 E0 F0)
    let code = [
        0x48, 0xB8, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00, 0x00, 0x48, 0x83, 0xE0, 0xF0,
    ];
    let ctx = compile_and_run(&code, RegisterContext::new());
    assert_eq!(
        ctx.rax, 0x1_2345_6780,
        "and r64,-16 must mask the low 4 bits of the full 64-bit value"
    );
}

#[test]
fn add_r64_neg_imm8_sign_extends() {
    // add rax, -1   (48 83 C0 FF)  with rax = 0  =>  rax = 0xFFFF_FFFF_FFFF_FFFF
    let mut ctx = RegisterContext::new();
    ctx.rax = 0;
    let code = [0x48, 0x83, 0xC0, 0xFF];
    let ctx = compile_and_run(&code, ctx);
    assert_eq!(
        ctx.rax,
        u64::MAX,
        "add r64,-1 must sign-extend imm8 0xff to -1, not add +255"
    );
}
