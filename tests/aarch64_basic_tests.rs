//! aarch64 guest frontend: decode classification + JIT round-trip tests for the
//! instruction-semantics vertical slice.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::arch::ControlFlowKind;
use microlift::arch::aarch64::Lifter64;
use microlift::arch::aarch64::context::Aarch64RegisterContext;
use microlift::arch::aarch64::decode::{control_flow, decode_region};
use microlift::backend::CraneliftBackend;
use microlift::context::ExitReason;
use microlift::os::get_host_calling_conv;

const BASE: u64 = 0x1000;

/// Lift `bytes` as an aarch64 guest region at `BASE`, JIT-compile, run against
/// `ctx`, and return the resulting context.
fn run(bytes: &[u8], mut ctx: Aarch64RegisterContext) -> Aarch64RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter64::new(builder);

    ctx.image_base = BASE;
    ctx.pc = BASE;

    lifter.lift(bytes, BASE).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut Aarch64RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

// Hand-encoded A64 instructions (little-endian on the wire).
const MOVZ_X0_42: [u8; 4] = [0x40, 0x05, 0x80, 0xD2]; // movz x0, #42
const MOVZ_X0_10: [u8; 4] = [0x40, 0x01, 0x80, 0xD2]; // movz x0, #10
const MOVZ_X1_20: [u8; 4] = [0x81, 0x02, 0x80, 0xD2]; // movz x1, #20
const MOVZ_X0_5: [u8; 4] = [0xA0, 0x00, 0x80, 0xD2]; // movz x0, #5
const MOVZ_X1_3: [u8; 4] = [0x61, 0x00, 0x80, 0xD2]; // movz x1, #3
const MOVZ_X0_7: [u8; 4] = [0xE0, 0x00, 0x80, 0xD2]; // movz x0, #7
const MOVZ_X1_7: [u8; 4] = [0xE1, 0x00, 0x80, 0xD2]; // movz x1, #7
const MOVZ_X0_0: [u8; 4] = [0x00, 0x00, 0x80, 0xD2]; // movz x0, #0
const ADD_X2_X0_X1: [u8; 4] = [0x02, 0x00, 0x01, 0x8B]; // add x2, x0, x1
const SUBS_X2_X0_X1: [u8; 4] = [0x02, 0x00, 0x01, 0xEB]; // subs x2, x0, x1
const SUBS_XZR_X0_X1: [u8; 4] = [0x1F, 0x00, 0x01, 0xEB]; // cmp x0, x1
const BEQ_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0x54]; // b.eq #+8
const RET: [u8; 4] = [0xC0, 0x03, 0x5F, 0xD6]; // ret

const STR_X0_SP: [u8; 4] = [0xE0, 0x03, 0x00, 0xF9]; // str x0, [sp]
const LDR_X1_SP: [u8; 4] = [0xE1, 0x03, 0x40, 0xF9]; // ldr x1, [sp]

const PACIASP: [u8; 4] = [0x3F, 0x23, 0x03, 0xD5]; // paciasp
const AUTIASP: [u8; 4] = [0xBF, 0x23, 0x03, 0xD5]; // autiasp
const STP_X0_X1_SP: [u8; 4] = [0xE0, 0x07, 0x00, 0xA9]; // stp x0, x1, [sp]
const BLR_X0: [u8; 4] = [0x00, 0x00, 0x3F, 0xD6]; // blr x0
const ADRP_X0_1: [u8; 4] = [0x00, 0x00, 0x00, 0xB0]; // adrp x0, #0x1000 (1 page)
const TBZ_X0_0_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0x36]; // tbz w0, #0, #+8
const TST_X0_X1: [u8; 4] = [0x1F, 0x00, 0x01, 0xEA]; // tst x0, x1 (ands xzr, x0, x1)
const CSEL_X2_X0_X1_EQ: [u8; 4] = [0x02, 0x00, 0x81, 0x9A]; // csel x2, x0, x1, eq

const CBZ_X0_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0xB4]; // cbz x0, #+8
const CBNZ_X0_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0xB5]; // cbnz x0, #+8
const MUL_X2_X0_X1: [u8; 4] = [0x02, 0x7C, 0x01, 0x9B]; // mul x2, x0, x1
const MADD_X2_X0_X1_X3: [u8; 4] = [0x02, 0x0C, 0x01, 0x9B]; // madd x2, x0, x1, x3
const MSUB_X2_X0_X1_X3: [u8; 4] = [0x02, 0x8C, 0x01, 0x9B]; // msub x2, x0, x1, x3
const UDIV_X2_X0_X1: [u8; 4] = [0x02, 0x08, 0xC1, 0x9A]; // udiv x2, x0, x1
const SDIV_X2_X0_X1: [u8; 4] = [0x02, 0x0C, 0xC1, 0x9A]; // sdiv x2, x0, x1
const LSLV_X2_X0_X1: [u8; 4] = [0x02, 0x20, 0xC1, 0x9A]; // lslv x2, x0, x1
const LSRV_X2_X0_X1: [u8; 4] = [0x02, 0x24, 0xC1, 0x9A]; // lsrv x2, x0, x1
const ASRV_X2_X0_X1: [u8; 4] = [0x02, 0x28, 0xC1, 0x9A]; // asrv x2, x0, x1
const RORV_X2_X0_X1: [u8; 4] = [0x02, 0x2C, 0xC1, 0x9A]; // rorv x2, x0, x1
const LDP_X0_X1_SP: [u8; 4] = [0xE0, 0x07, 0x40, 0xA9]; // ldp x0, x1, [sp]
const CSINC_X2_X0_X1_EQ: [u8; 4] = [0x02, 0x04, 0x81, 0x9A]; // csinc x2, x0, x1, eq
const CSINV_X2_X0_X1_EQ: [u8; 4] = [0x02, 0x00, 0x81, 0xDA]; // csinv x2, x0, x1, eq
const CSNEG_X2_X0_X1_EQ: [u8; 4] = [0x02, 0x04, 0x81, 0xDA]; // csneg x2, x0, x1, eq
const CLZ_X0_X1: [u8; 4] = [0x20, 0x10, 0xC0, 0xDA]; // clz x0, x1
const RBIT_X0_X1: [u8; 4] = [0x20, 0x00, 0xC0, 0xDA]; // rbit x0, x1
const REV_X0_X1: [u8; 4] = [0x20, 0x0C, 0xC0, 0xDA]; // rev x0, x1
const ADR_X0_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0x10]; // adr x0, #+8
const MVN_X0_X1: [u8; 4] = [0xE0, 0x03, 0x21, 0xAA]; // mvn x0, x1 (orn x0, xzr, x1)
const BIC_X2_X0_X1: [u8; 4] = [0x02, 0x00, 0x21, 0x8A]; // bic x2, x0, x1

fn program(parts: &[[u8; 4]]) -> Vec<u8> {
    parts.iter().flatten().copied().collect()
}

#[test]
fn decode_movz_and_ret() {
    let bytes = program(&[MOVZ_X0_42, RET]);
    let insts = decode_region(&bytes, BASE);
    assert_eq!(insts.len(), 2);
    assert!(insts[0].valid);
    assert_eq!(insts[0].ip, BASE);
    assert_eq!(insts[1].ip, BASE + 4);
    assert_eq!(control_flow(&insts[0]).kind, ControlFlowKind::Fallthrough);
    assert_eq!(control_flow(&insts[1]).kind, ControlFlowKind::Ret);
}

#[test]
fn decode_conditional_branch_target() {
    // subs at 0x1008, b.eq at 0x100C -> taken target 0x1014.
    let bytes = program(&[MOVZ_X0_7, MOVZ_X1_7, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let insts = decode_region(&bytes, BASE);
    let beq = &insts[3];
    let cf = control_flow(beq);
    assert_eq!(cf.kind, ControlFlowKind::CondBranch);
    assert_eq!(cf.static_targets, vec![BASE + 0x14]);
}

#[test]
fn movz_ret() {
    let ctx = run(&program(&[MOVZ_X0_42, RET]), Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 42);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn add_three_regs() {
    let bytes = program(&[MOVZ_X0_10, MOVZ_X1_20, ADD_X2_X0_X1, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 10);
    assert_eq!(ctx.x[1], 20);
    assert_eq!(ctx.x[2], 30);
}

#[test]
fn subs_sets_flags() {
    // 5 - 3 = 2; N=0, Z=0, C=1 (no borrow), V=0.
    let bytes = program(&[MOVZ_X0_5, MOVZ_X1_3, SUBS_X2_X0_X1, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[2], 2);
    assert_eq!(ctx.n, 0);
    assert_eq!(ctx.z, 0);
    assert_eq!(ctx.c, 1);
    assert_eq!(ctx.vf, 0);
}

#[test]
fn conditional_branch_taken() {
    // 7 == 7 -> Z=1 -> b.eq taken -> skips `movz x0, #0`, so x0 stays 7.
    let bytes = program(&[MOVZ_X0_7, MOVZ_X1_7, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.z, 1);
    assert_eq!(ctx.x[0], 7);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn store_then_load_roundtrip() {
    // movz x0,#42; str x0,[sp]; ldr x1,[sp]; ret  -> x1 == 42.
    let mut stack = [0u64; 64];
    let sp = stack.as_mut_ptr() as u64 + 256;
    let mut ctx = Aarch64RegisterContext::default();
    ctx.sp = sp;
    let bytes = program(&[MOVZ_X0_42, STR_X0_SP, LDR_X1_SP, RET]);
    let ctx = run(&bytes, ctx);
    assert_eq!(ctx.x[0], 42);
    assert_eq!(ctx.x[1], 42);
    assert_eq!(stack[32], 42); // sp pointed at stack[32]
}

#[test]
fn paciasp_autiasp_are_noops() {
    // The sign/authenticate pair must round-trip the link register to its
    // original value (identity, since we don't model PAC enforcement).
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[30] = 0xDEAD_BEEF;
    let ctx = run(&program(&[PACIASP, MOVZ_X0_42, AUTIASP, RET]), ctx);
    assert_eq!(ctx.x[30], 0xDEAD_BEEF);
    assert_eq!(ctx.x[0], 42);
}

#[test]
fn stp_stores_register_pair() {
    // stp x0, x1, [sp] writes x0 at [sp+0] and x1 at [sp+8].
    let mut stack = [0u64; 64];
    let sp = stack.as_mut_ptr() as u64 + 256;
    let mut ctx = Aarch64RegisterContext::default();
    ctx.sp = sp;
    let bytes = program(&[MOVZ_X0_42, MOVZ_X1_20, STP_X0_X1_SP, RET]);
    let _ = run(&bytes, ctx);
    assert_eq!(stack[32], 42); // sp pointed at stack[32]
    assert_eq!(stack[33], 20);
}

#[test]
fn adrp_forms_page_address() {
    // adrp x0, #0x1000 at PC 0x1000: (0x1000 & ~0xFFF) + 0x1000 = 0x2000.
    let ctx = run(&program(&[ADRP_X0_1, RET]), Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 0x2000);
}

#[test]
fn blr_exits_indirect_with_link() {
    // movz x0,#42; blr x0  -> IndirectCall to 42, x30/pc = resume (0x1008).
    let ctx = run(&program(&[MOVZ_X0_42, BLR_X0]), Aarch64RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::IndirectCall as u32);
    assert_eq!(ctx.imported_function_va, 42);
    assert_eq!(ctx.x[30], BASE + 8); // return address = instruction after BLR
    assert_eq!(ctx.pc, BASE + 8);
}

#[test]
fn tbz_taken_when_bit_clear() {
    // tbz w0,#0,#+8 with x0=4 (bit0 clear) -> branch to RET, skips movz x0,#0.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 4;
    let ctx = run(&program(&[TBZ_X0_0_PLUS8, MOVZ_X0_0, RET]), ctx);
    assert_eq!(ctx.x[0], 4);
}

#[test]
fn tbz_not_taken_when_bit_set() {
    // tbz w0,#0,#+8 with x0=5 (bit0 set) -> falls through to movz x0,#0.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 5;
    let ctx = run(&program(&[TBZ_X0_0_PLUS8, MOVZ_X0_0, RET]), ctx);
    assert_eq!(ctx.x[0], 0);
}

#[test]
fn tst_nonzero_clears_carry_and_overflow() {
    // tst x0,x1 (7 & 7 = 7): Z=0, N=0; C and V are always cleared.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.c = 1;
    ctx.vf = 1;
    let ctx = run(&program(&[MOVZ_X0_7, MOVZ_X1_7, TST_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[0], 7); // result discarded, x0 unchanged
    assert_eq!(ctx.z, 0);
    assert_eq!(ctx.n, 0);
    assert_eq!(ctx.c, 0);
    assert_eq!(ctx.vf, 0);
}

#[test]
fn tst_zero_sets_zero_flag() {
    // tst x0,x1 with 8 & 7 = 0 -> Z=1.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 8;
    ctx.x[1] = 7;
    let ctx = run(&program(&[TST_X0_X1, RET]), ctx);
    assert_eq!(ctx.z, 1);
    assert_eq!(ctx.n, 0);
}

#[test]
fn csel_picks_first_when_cond_true() {
    // csel x2,x0,x1,eq with Z=1 -> x2 = x0 = 10.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 1;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSEL_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2], 10);
}

#[test]
fn csel_picks_second_when_cond_false() {
    // csel x2,x0,x1,eq with Z=0 -> x2 = x1 = 20.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 0;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSEL_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2], 20);
}

#[test]
fn conditional_branch_not_taken() {
    // 5 != 3 -> Z=0 -> b.eq not taken -> falls through to `movz x0, #0`.
    let bytes = program(&[MOVZ_X0_5, MOVZ_X1_3, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.z, 0);
    assert_eq!(ctx.x[0], 0);
}

#[test]
fn decode_cbz_is_cond_branch() {
    let bytes = program(&[CBZ_X0_PLUS8, MOVZ_X0_0, RET]);
    let insts = decode_region(&bytes, BASE);
    let cf = control_flow(&insts[0]);
    assert_eq!(cf.kind, ControlFlowKind::CondBranch);
    assert_eq!(cf.static_targets, vec![BASE + 8]);
}

#[test]
fn cbz_taken_when_zero() {
    // x0 == 0 -> cbz branches over `movz x0,#42`, leaving x0 == 0.
    let bytes = program(&[CBZ_X0_PLUS8, MOVZ_X0_42, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 0);
}

#[test]
fn cbz_not_taken_when_nonzero() {
    // x0 != 0 -> cbz falls through to `movz x0,#42`.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 9;
    let ctx = run(&program(&[CBZ_X0_PLUS8, MOVZ_X0_42, RET]), ctx);
    assert_eq!(ctx.x[0], 42);
}

#[test]
fn cbnz_taken_when_nonzero() {
    // x0 != 0 -> cbnz branches over `movz x0,#42`, leaving x0 unchanged.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 7;
    let ctx = run(&program(&[CBNZ_X0_PLUS8, MOVZ_X0_42, RET]), ctx);
    assert_eq!(ctx.x[0], 7);
}

#[test]
fn mul_multiplies() {
    let bytes = program(&[MOVZ_X0_7, MOVZ_X1_3, MUL_X2_X0_X1, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[2], 21);
}

#[test]
fn madd_multiply_accumulate() {
    // x2 = x3 + x0 * x1 = 100 + 7 * 3 = 121.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 7;
    ctx.x[1] = 3;
    ctx.x[3] = 100;
    let ctx = run(&program(&[MADD_X2_X0_X1_X3, RET]), ctx);
    assert_eq!(ctx.x[2], 121);
}

#[test]
fn msub_multiply_subtract() {
    // x2 = x3 - x0 * x1 = 100 - 7 * 3 = 79.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 7;
    ctx.x[1] = 3;
    ctx.x[3] = 100;
    let ctx = run(&program(&[MSUB_X2_X0_X1_X3, RET]), ctx);
    assert_eq!(ctx.x[2], 79);
}

#[test]
fn udiv_divides() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 20;
    ctx.x[1] = 3;
    let ctx = run(&program(&[UDIV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 6);
}

#[test]
fn udiv_by_zero_yields_zero() {
    // A64 divide-by-zero produces 0 rather than trapping.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 20;
    ctx.x[1] = 0;
    let ctx = run(&program(&[UDIV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 0);
}

#[test]
fn sdiv_rounds_toward_zero() {
    // -7 / 2 = -3 (truncated toward zero).
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = (-7i64) as u64;
    ctx.x[1] = 2;
    let ctx = run(&program(&[SDIV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2] as i64, -3);
}

#[test]
fn sdiv_by_zero_yields_zero() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = (-7i64) as u64;
    ctx.x[1] = 0;
    let ctx = run(&program(&[SDIV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 0);
}

#[test]
fn sdiv_int_min_over_neg_one_no_trap() {
    // INT_MIN / -1 overflows; A64 defines the result as INT_MIN (no trap).
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = i64::MIN as u64;
    ctx.x[1] = (-1i64) as u64;
    let ctx = run(&program(&[SDIV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2] as i64, i64::MIN);
}

#[test]
fn lslv_shifts_left() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 1;
    ctx.x[1] = 4;
    let ctx = run(&program(&[LSLV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 16);
}

#[test]
fn lsrv_shifts_right() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 0x80;
    ctx.x[1] = 3;
    let ctx = run(&program(&[LSRV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 0x10);
}

#[test]
fn asrv_arithmetic_shift_preserves_sign() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = (-16i64) as u64;
    ctx.x[1] = 2;
    let ctx = run(&program(&[ASRV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2] as i64, -4);
}

#[test]
fn rorv_rotates_right() {
    // Rotate 0x1 right by 4 in 64-bit -> bit moves to position 60.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 0x1;
    ctx.x[1] = 4;
    let ctx = run(&program(&[RORV_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 0x1000_0000_0000_0000);
}

#[test]
fn ldp_loads_register_pair() {
    // Stack holds two values; ldp pulls them into x0 and x1.
    let mut stack = [0u64; 64];
    stack[32] = 0xAAAA;
    stack[33] = 0xBBBB;
    let sp = stack.as_mut_ptr() as u64 + 256;
    let mut ctx = Aarch64RegisterContext::default();
    ctx.sp = sp;
    let ctx = run(&program(&[LDP_X0_X1_SP, RET]), ctx);
    assert_eq!(ctx.x[0], 0xAAAA);
    assert_eq!(ctx.x[1], 0xBBBB);
}

#[test]
fn stp_ldp_roundtrip() {
    let mut stack = [0u64; 64];
    let sp = stack.as_mut_ptr() as u64 + 256;
    let mut ctx = Aarch64RegisterContext::default();
    ctx.sp = sp;
    ctx.x[0] = 0x1234;
    ctx.x[1] = 0x5678;
    // stp x0,x1,[sp]; clobber; ldp x0,x1,[sp] back.
    let ctx = run(
        &program(&[STP_X0_X1_SP, MOVZ_X0_0, LDP_X0_X1_SP, RET]),
        ctx,
    );
    assert_eq!(ctx.x[0], 0x1234);
    assert_eq!(ctx.x[1], 0x5678);
}

#[test]
fn csinc_else_increments() {
    // Z=0 -> cond false -> x2 = x1 + 1 = 21.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 0;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSINC_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2], 21);
}

#[test]
fn csinc_true_picks_first() {
    // Z=1 -> cond true -> x2 = x0 = 10 (no increment).
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 1;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSINC_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2], 10);
}

#[test]
fn csinv_else_inverts() {
    // Z=0 -> x2 = ~x1 = ~20.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 0;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSINV_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2], !20u64);
}

#[test]
fn csneg_else_negates() {
    // Z=0 -> x2 = -x1 = -20.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.z = 0;
    let ctx = run(&program(&[MOVZ_X0_10, MOVZ_X1_20, CSNEG_X2_X0_X1_EQ, RET]), ctx);
    assert_eq!(ctx.x[2] as i64, -20);
}

#[test]
fn clz_counts_leading_zeros() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[1] = 1;
    let ctx = run(&program(&[CLZ_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[0], 63);
}

#[test]
fn rbit_reverses_bits() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[1] = 1;
    let ctx = run(&program(&[RBIT_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[0], 0x8000_0000_0000_0000);
}

#[test]
fn rev_reverses_bytes() {
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[1] = 0x0011_2233_4455_6677;
    let ctx = run(&program(&[REV_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[0], 0x7766_5544_3322_1100);
}

#[test]
fn adr_forms_byte_address() {
    // adr x0,#+8 at PC 0x1000 -> 0x1008 (no page masking).
    let ctx = run(&program(&[ADR_X0_PLUS8, RET]), Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], BASE + 8);
}

#[test]
fn mvn_inverts() {
    // mvn x0,x1 (orn x0,xzr,x1): x0 = ~x1.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[1] = 0x00FF;
    let ctx = run(&program(&[MVN_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[0], !0x00FFu64);
}

#[test]
fn bic_and_with_inverted() {
    // bic x2,x0,x1: x2 = x0 & ~x1 = 0xFF & ~0x0F = 0xF0.
    let mut ctx = Aarch64RegisterContext::default();
    ctx.x[0] = 0xFF;
    ctx.x[1] = 0x0F;
    let ctx = run(&program(&[BIC_X2_X0_X1, RET]), ctx);
    assert_eq!(ctx.x[2], 0xF0);
}
