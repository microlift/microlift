//! Stack-pointer correctness of the lifter.
//! - Out-of-region direct `CALL` must decrement RSP and push the return address before the `DirectCall` exit.
//! - `push imm8` must sign-extend (`push -1` pushes all ones).
//! - Unregistered indirect `CALL` exits `IndirectCall` with RSP unchanged; the host pushes `ctx.rip` for guest callees.

use microlift::context::TransferKind;
use microlift::testkit::prelude::*;

// Test helpers

/// Lift at `0x1000`, JIT, run with `ctx`. Unlike `stack_tests.rs`, leaves `ctx.rsp`/`ctx.image_base` to the caller.
fn compile_and_run(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    // `RetAddr::None` with caller-provided `rsp`/`image_base` keeps the stack untouched.
    LiftTest::new(bytes)
        .ctx(ctx)
        .ret_addr(RetAddr::None)
        .run()
        .ctx
}

/// Like `compile_and_run` but with an automatic stack, exit trampoline and `image_base = 0x1000`; returns `(ctx_after, initial_rsp)`.
fn compile_and_run_with_stack(bytes: &[u8], ctx: RegisterContext) -> (RegisterContext, u64) {
    let out = LiftTest::new(bytes).ctx(ctx).run();
    (out.ctx, out.initial_rsp)
}

// Bug 1: out-of-region direct CALL does not push return address

/// An out-of-region `CALL` must decrement RSP by 8, write the rebased return address (`0x1005`) at [new RSP], and exit `UnknownJump`/`DirectCall`.
/// `E8 FB 0F 00 00` -> `call 0x2000` (rel32 = 0x2000 - 0x1005).
#[test]
fn test_out_of_region_call_decrements_rsp() {
    // call 0x2000  (5 bytes, target is outside the 5-byte region)
    let bytes = [0xE8u8, 0xFB, 0x0F, 0x00, 0x00];

    // Stack buffer large enough for the return-address slot.
    let mut stack = [0u64; 16];
    let stack_top = stack.as_mut_ptr() as u64 + 16 * 8;

    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x1000;
    // No trampoline: the code exits via DirectCall before any RET.
    ctx.rsp = stack_top;
    let initial_rsp = ctx.rsp;

    let ctx = compile_and_run(&bytes, ctx);

    // Confirm exit kind.
    assert_eq!(ctx.exit_reason, ExitReason::UnknownJump as u32);
    assert_eq!(ctx.exit_code, TransferKind::DirectCall as u32);

    // RSP must have been decremented by 8.
    assert_eq!(
        ctx.rsp,
        initial_rsp - 8,
        "Bug 1: out-of-region CALL must decrement RSP by 8 (return address slot)"
    );
}

/// The word at [new RSP] must be the rebased return address `image_base + (next_ip - lift_base) = 0x1005`.
/// RSP starts one slot below the top so reads stay in bounds even if the push is missing.
#[test]
fn test_out_of_region_call_writes_return_address() {
    let bytes = [0xE8u8, 0xFB, 0x0F, 0x00, 0x00]; // call 0x2000

    let mut stack = [0u64; 16];
    // Slot 14 is the initial RSP; after the push it is slot 13.
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

/// After a `DirectCall` exit, `imported_function_va` is the rebased target and `ctx.rip` the rebased return address.
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

    // Callee VA: 0x1000 + (0x2000 - 0x1000) = 0x2000.
    assert_eq!(ctx.imported_function_va, 0x2000);

    // RIP: 0x1000 + (0x1005 - 0x1000) = 0x1005.
    assert_eq!(ctx.rip, 0x1005);
}

// Bug 2: PUSH imm8 (Immediate8to64) sign-extension

/// `push imm8` (`6A`) sign-extends to 64 bits: `push -1` stores `0xFFFF_FFFF_FFFF_FFFF`.
#[test]
fn test_push_neg1_imm8_sign_extends() {
    // 6A FF push -1 ; 58 pop rax ; C3 ret
    let bytes = [0x6Au8, 0xFF, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax,
        u64::MAX,
        "Bug 2: push -1 (imm8, 6A FF) must sign-extend to 0xFFFF_FFFF_FFFF_FFFF"
    );
}

/// `push -128` (`6A 80`): most-negative 8-bit value -> `0xFFFF_FFFF_FFFF_FF80`.
#[test]
fn test_push_neg128_imm8_sign_extends() {
    // 6A 80 push -128 ; 58 pop rax ; C3 ret
    let bytes = [0x6Au8, 0x80, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax, 0xFFFF_FFFF_FFFF_FF80u64,
        "Bug 2: push -128 (imm8, 6A 80) must sign-extend to 0xFFFF_FFFF_FFFF_FF80"
    );
}

/// `push -100` (`6A 9C`) -> `0xFFFF_FFFF_FFFF_FF9C`.
#[test]
fn test_push_neg100_imm8_sign_extends() {
    // 6A 9C push -100 ; 58 pop rax ; C3 ret
    let bytes = [0x6Au8, 0x9C, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax, 0xFFFF_FFFF_FFFF_FF9Cu64,
        "Bug 2: push -100 (imm8, 6A 9C) must sign-extend to 0xFFFF_FFFF_FFFF_FF9C"
    );
}

/// `push imm32` (`68`) sign-extends: `push 0xFFFF_FFFF` -> all ones.
#[test]
fn test_push_neg1_imm32_sign_extends_sanity() {
    // 68 FF FF FF FF push ; 58 pop rax ; C3 ret
    let bytes = [0x68u8, 0xFF, 0xFF, 0xFF, 0xFF, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax,
        u64::MAX,
        "push 0xFFFFFFFF (imm32, Immediate32to64) must sign-extend to 0xFFFF_FFFF_FFFF_FFFF"
    );
}

/// Small positive `push 1` (6A 01) gives rax = 1 either way.
#[test]
fn test_push_positive_imm8_unaffected() {
    // 6A 01 push 1 ; 58 pop rax ; C3 ret
    let bytes = [0x6Au8, 0x01, 0x58, 0xC3];

    let (ctx, _) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 1u64);
}

// IndirectCall exit reason contract

/// `CALL RAX` exits `IndirectCall` with RSP unchanged; the host decides native vs guest dispatch (guest: push `ctx.rip`).
///
/// `48 C7 C0 00 20 00 00` mov rax, 0x2000 ; `FF D0` call rax (next_ip = 0x1009)
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

/// `CALL [mem]` gives the same `IndirectCall` contract as `CALL reg`.
#[test]
fn test_indirect_call_via_mem_exits_with_indirect_call_reason() {
    // mov qword [rsp-16], 0x3000 (48 C7 44 24 F0 00 30 00 00) ; call [rsp-16] (FF 54 24 F0), next_ip = 0x100D
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

/// Baseline: an in-region `CALL` decrements RSP and pushes the return address; the matching `RET` pops it, balanced.
#[test]
fn test_in_region_call_ret_rsp_balance() {
    // 0x1000 call 0x1006 (E8 01 00 00 00) ; 0x1005 ret (outer exit) ; 0x1006 ret (inner)
    #[rustfmt::skip]
    let bytes = [
        0xE8u8, 0x01, 0x00, 0x00, 0x00, // call 0x1006
        0xC3,                            // ret (outer - pops trampoline, exits Return)
        0xC3,                            // ret (callee - returns to 0x1005)
    ];

    let (ctx, initial_rsp) = compile_and_run_with_stack(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);

    // The outer `ret` popped the trampoline: RSP = initial_rsp + 8.
    assert_eq!(
        ctx.rsp,
        initial_rsp + 8,
        "RSP must advance past the trampoline slot after the final RET"
    );
}

/// The callee sees RSP = caller RSP - 8, captured by `mov rax, rsp`.
/// 0x1000 call 0x1007 ; 0x1005 ret ; 0x1006 nop (makes 0x1007 a leader) ; 0x1007 mov rax, rsp ; 0x100A ret
#[test]
fn test_in_region_call_callee_rsp_is_decremented() {
    // rel32 = 0x1007 - 0x1005 = 2
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

    // Callee saw initial_rsp - 8.
    assert_eq!(
        ctx.rax,
        initial_rsp - 8,
        "Callee must observe RSP = caller_RSP − 8 (return address slot)"
    );
}

// `AND/OR/ADD r/m64, imm8` sign-extend the byte: `immediate64()` would give 240 for `and rsp, -16` and corrupt RSP (every SysV `_start` does this).

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
