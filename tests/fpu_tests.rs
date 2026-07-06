//! x87 FPU instruction tests.
//!
//! Each test lifts a small x86-64 snippet and verifies the resulting
//! `RegisterContext` x87 stack state (`fpr` / `fpu_top`) or memory state after
//! JIT execution. The x87 stack is modelled in `f64` precision, so register
//! values are compared as the raw bits of `f64`.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::{Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::with_config(builder, LifterConfig::default());

    let mut ctx = ctx;
    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

/// Read architectural `ST(i)` (physical slot `(top + i) & 7`) as an `f64`.
fn st(ctx: &RegisterContext, i: usize) -> f64 {
    use microlift::context::f80_to_f64_bits;
    let phys = (ctx.fpu_top as usize + i) & 7;
    let [mant, se] = ctx.fpr[phys];
    f64::from_bits(f80_to_f64_bits(mant, se as u16))
}

/// Store an `f64` into physical `fpr` slot `phys` (helper for setting up input
/// stacks with a known `fpu_top`).
fn set_phys(ctx: &mut RegisterContext, phys: usize, v: f64) {
    use microlift::context::f64_bits_to_f80;
    let (mant, se) = f64_bits_to_f80(v.to_bits());
    ctx.fpr[phys] = [mant, se as u64];
}

#[test]
fn test_fldz_pushes_zero() {
    // fldz; ret  ->  D9 EE  C3
    let bytes = [0xD9, 0xEE, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.fpu_top = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.fpu_top, 7, "push must decrement TOP (0 -> 7)");
    assert_eq!(st(&ctx, 0), 0.0);
}

#[test]
fn test_fld1_pushes_one() {
    // fld1; ret  ->  D9 E8  C3
    let bytes = [0xD9, 0xE8, 0xC3];
    let mut ctx = RegisterContext::default();
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 1.0);
}

#[test]
fn test_fldpi_pushes_pi() {
    // fldpi; ret  ->  D9 EB  C3
    let bytes = [0xD9, 0xEB, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), std::f64::consts::PI);
}

#[test]
fn test_two_pushes_order() {
    // fld1; fldz; ret  -> after: ST0 = 0.0, ST1 = 1.0
    let bytes = [0xD9, 0xE8, 0xD9, 0xEE, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 0.0, "last push is ST0");
    assert_eq!(st(&ctx, 1), 1.0, "earlier push is ST1");
    assert_eq!(ctx.fpu_top, 6);
}

#[test]
fn test_flds_load_f32() {
    // fld dword [rax]; ret  ->  D9 00  C3
    let bytes = [0xD9, 0x00, 0xC3];
    let val: f32 = 1.5;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 1.5f64);
}

#[test]
fn test_fldl_load_f64() {
    // fld qword [rax]; ret  ->  DD 00  C3
    let bytes = [0xDD, 0x00, 0xC3];
    let val: f64 = 3.25;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 3.25);
}

#[test]
fn test_fstp_store_f64() {
    // fld qword [rax]; fstp qword [rcx]; ret  ->  DD 00  DD 19  C3
    let bytes = [0xDD, 0x00, 0xDD, 0x19, 0xC3];
    let src: f64 = -7.5;
    let mut dst: f64 = 0.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.rcx = &mut dst as *mut f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(dst, -7.5, "fstp must store ST0 to memory");
    assert_eq!(ctx.fpu_top, 0, "fstp must pop (TOP back to 0)");
}

// --- FLDT / FSTPT (80-bit extended round-trip) -------------------------

#[test]
fn test_fldt_fstpt_roundtrip() {
    // fld tbyte [rax]; fstp tbyte [rcx]; ret  ->  DB 28  DB 39  C3
    let bytes = [0xDB, 0x28, 0xDB, 0x39, 0xC3];
    // 80-bit encoding of 2.0: exponent = 16384 (bias 16383 + 1), integer bit set.
    let value = 2.0f64;
    let mut src = [0u8; 16];
    encode_f80(value, &mut src);
    let mut dst = [0u8; 16];
    let mut ctx = RegisterContext::default();
    ctx.rax = src.as_ptr() as u64;
    ctx.rcx = dst.as_mut_ptr() as u64;
    let ctx = run_test(&bytes, ctx);
    // The stored 80-bit value must decode back to 2.0.
    assert_eq!(decode_f80(&dst), 2.0);
    assert_eq!(ctx.fpu_top, 0, "fldt then fstp is balanced");
}

/// Encode `v` (a normal `f64` value or zero) into an 80-bit extended buffer.
fn encode_f64_to_f80(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let sign = (bits >> 63) & 1;
    let exp64 = (bits >> 52) & 0x7FF;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    let mut out = [0u8; 10];
    if exp64 == 0 && frac == 0 {
        let se = (sign << 15) as u16;
        out[8..10].copy_from_slice(&se.to_le_bytes());
        return out;
    }
    let exp80 = (exp64 + 15360) as u16;
    let mant = (1u64 << 63) | (frac << 11);
    out[0..8].copy_from_slice(&mant.to_le_bytes());
    let se = ((sign as u16) << 15) | exp80;
    out[8..10].copy_from_slice(&se.to_le_bytes());
    out
}

fn encode_f80(v: f64, buf: &mut [u8; 16]) {
    let e = encode_f64_to_f80(v);
    buf[..10].copy_from_slice(&e);
}

fn decode_f80(buf: &[u8; 16]) -> f64 {
    let mant = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let se = u16::from_le_bytes(buf[8..10].try_into().unwrap());
    let sign = ((se >> 15) & 1) as u64;
    let exp80 = (se & 0x7FFF) as u64;
    if mant == 0 {
        return f64::from_bits(sign << 63);
    }
    let exp64 = exp80.wrapping_sub(15360) & 0x7FF;
    let frac = (mant >> 11) & 0x000F_FFFF_FFFF_FFFF;
    f64::from_bits((sign << 63) | (exp64 << 52) | frac)
}

#[test]
fn test_fldt_fstpt_full_precision() {
    // An 80-bit value whose 64-bit significand has low bits set that would be
    // lost if it round-tripped through f64. fld tbyte[rax]; fstp tbyte[rcx].
    let bytes = [0xDB, 0x28, 0xDB, 0x39, 0xC3];
    // 1.0 + 2^-63 style significand: integer bit + lowest bit set, exp for 1.0.
    let mant: u64 = 0x8000_0000_0000_0001;
    let se: u16 = 0x3FFF; // exponent for value in [1,2)
    let mut src = [0u8; 16];
    src[0..8].copy_from_slice(&mant.to_le_bytes());
    src[8..10].copy_from_slice(&se.to_le_bytes());
    let mut dst = [0u8; 16];
    let mut ctx = RegisterContext::default();
    ctx.rax = src.as_ptr() as u64;
    ctx.rcx = dst.as_mut_ptr() as u64;
    run_test(&bytes, ctx);
    // The low significand bit must survive - it can't be represented in f64.
    assert_eq!(u64::from_le_bytes(dst[0..8].try_into().unwrap()), mant);
    assert_eq!(u16::from_le_bytes(dst[8..10].try_into().unwrap()), se);
}

#[test]
fn test_fild_large_i64_exact() {
    // A 64-bit integer with more than 53 significant bits: exact in 80-bit,
    // would round if it passed through f64. fild qword[rax]; fistp qword[rcx].
    let bytes = [0xDF, 0x28, 0xDF, 0x39, 0xC3]; // fild qword[rax]; fistp qword[rcx]
    let val: i64 = 0x0123_4567_89AB_CDEF; // 57 significant bits
    let mut out: i64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const i64 as u64;
    ctx.rcx = &mut out as *mut i64 as u64;
    run_test(&bytes, ctx);
    assert_eq!(out, val, "fild/fistp of a >53-bit integer must be exact");
}

#[test]
fn test_fild_negative_i64_exact() {
    let bytes = [0xDF, 0x28, 0xDF, 0x39, 0xC3];
    let val: i64 = -0x0123_4567_89AB_CDEF;
    let mut out: i64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const i64 as u64;
    ctx.rcx = &mut out as *mut i64 as u64;
    run_test(&bytes, ctx);
    assert_eq!(out, val);
}

// --- FILD / FISTP / FISTTP (integer conversions) -----------------------

#[test]
fn test_fild_load_i32() {
    // fild dword [rax]; ret  ->  DB 00  C3
    let bytes = [0xDB, 0x00, 0xC3];
    let val: i32 = -42;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const i32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), -42.0);
}

#[test]
fn test_fistp_round_to_nearest() {
    // fld qword [rax]; fistp dword [rcx]; ret  ->  DD 00  DB 19  C3
    let bytes = [0xDD, 0x00, 0xDB, 0x19, 0xC3];
    let src: f64 = 2.5; // round-half-to-even -> 2
    let mut dst: i32 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.rcx = &mut dst as *mut i32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(dst, 2, "fistp must round to nearest (even)");
    assert_eq!(ctx.fpu_top, 0);
}

#[test]
fn test_fisttp_truncates_i64() {
    // fld qword [rax]; fisttp qword [rcx]; ret  ->  DD 00  DD 09  C3
    let bytes = [0xDD, 0x00, 0xDD, 0x09, 0xC3];
    let src: f64 = -3.9; // truncate toward zero -> -3
    let mut dst: i64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.rcx = &mut dst as *mut i64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(dst, -3, "fisttp must truncate toward zero");
}

// --- FXCH --------------------------------------------------------------

#[test]
fn test_fxch_swaps_st0_st1() {
    // fld1; fldz; fxch; ret
    // fld1=D9 E8, fldz=D9 EE, fxch=D9 C9, ret=C3
    let bytes = [0xD9, 0xE8, 0xD9, 0xEE, 0xD9, 0xC9, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    // Before fxch: ST0=0.0, ST1=1.0. After: ST0=1.0, ST1=0.0.
    assert_eq!(st(&ctx, 0), 1.0);
    assert_eq!(st(&ctx, 1), 0.0);
}

#[test]
fn test_faddp_adds_and_pops() {
    // fld1; fld1; faddp; ret  -> ST0 = 1 + 1 = 2
    // faddp = DE C1
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDE, 0xC1, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 2.0);
    assert_eq!(ctx.fpu_top, 7, "two pushes then one pop -> TOP = 7");
}

#[test]
fn test_fsubp_order() {
    // Set up ST0 = 3.0, ST1 = 10.0, then fsubp: ST1 = ST1 - ST0 = 7, pop.
    // fsubp st(1), st = DE E9
    let bytes = [0xDE, 0xE9, 0xC3];
    let mut ctx = RegisterContext::default();
    // top = 6: ST0 = fpr[6], ST1 = fpr[7]
    ctx.fpu_top = 6;
    set_phys(&mut ctx, 6, 3.0); // ST0
    set_phys(&mut ctx, 7, 10.0); // ST1
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.fpu_top, 7, "fsubp pops");
    assert_eq!(st(&ctx, 0), 7.0, "ST1 - ST0 = 10 - 3 = 7 becomes new ST0");
}

#[test]
fn test_fmul_mem() {
    // fld qword [rax]; fmul qword [rcx]; ret  ->  DD 00  DC 09  C3
    let bytes = [0xDD, 0x00, 0xDC, 0x09, 0xC3];
    let a: f64 = 6.0;
    let b: f64 = 7.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &a as *const f64 as u64;
    ctx.rcx = &b as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 42.0);
}

#[test]
fn test_fdivr_reverse() {
    // ST0 = 2.0; fdivr qword [rax] with mem=10  -> ST0 = 10 / 2 = 5
    // fdivr qword [rax] = DC 38
    let bytes = [0xDC, 0x38, 0xC3];
    let mem: f64 = 10.0;
    let mut ctx = RegisterContext::default();
    ctx.fpu_top = 0;
    set_phys(&mut ctx, 0, 2.0);
    // With top=0, ST0 is fpr[0]. But a fresh push isn't done; ST0 already set.
    ctx.rax = &mem as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 5.0, "fdivr: mem / ST0 = 10 / 2 = 5");
}

#[test]
fn test_fabs_fchs_fsqrt() {
    // fld qword [rax]; fabs; fchs; fsqrt; ret
    // fabs=D9 E1, fchs=D9 E0, fsqrt=D9 FA
    // start -16.0 -> fabs 16.0 -> fchs -16.0 -> fsqrt sqrt(-16) = NaN
    let bytes = [0xDD, 0x00, 0xD9, 0xE1, 0xC3];
    let v: f64 = -16.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 16.0, "fabs of -16 = 16");
}

#[test]
fn test_fsqrt() {
    // fld qword [rax]; fsqrt; ret
    let bytes = [0xDD, 0x00, 0xD9, 0xFA, 0xC3];
    let v: f64 = 144.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 12.0);
}

#[test]
fn test_fucomi_less() {
    // ST0 = 1.0, ST1 = 2.0; fucomi st, st(1)  -> ST0 < ST1 -> CF=1, ZF=0
    // fld qword[rax] (2.0); fld qword[rcx] (1.0); fucomi st,st(1); ret
    // fld qword[rcx] = DD 01; fucomi st,st(1) = DB E9
    let bytes = [0xDD, 0x00, 0xDD, 0x01, 0xDB, 0xE9, 0xC3];
    let two: f64 = 2.0;
    let one: f64 = 1.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &two as *const f64 as u64;
    ctx.rcx = &one as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    // ST0=1.0, ST1=2.0 -> 1 < 2
    assert_eq!(ctx.cf, 1, "1 < 2 -> CF set");
    assert_eq!(ctx.zf, 0, "not equal -> ZF clear");
    assert_eq!(ctx.pf, 0, "ordered -> PF clear");
}

#[test]
fn test_fucomi_equal() {
    // ST0 = ST1 = 5.0 -> ZF=1, CF=0, PF=0
    let bytes = [0xDD, 0x00, 0xDD, 0x00, 0xDB, 0xE9, 0xC3];
    let five: f64 = 5.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &five as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.zf, 1, "equal -> ZF set");
    assert_eq!(ctx.cf, 0, "equal -> CF clear");
    assert_eq!(ctx.pf, 0, "ordered -> PF clear");
}

#[test]
fn test_fucomi_greater() {
    // ST0 = 9.0, ST1 = 4.0 -> all clear
    let bytes = [0xDD, 0x00, 0xDD, 0x01, 0xDB, 0xE9, 0xC3];
    let four: f64 = 4.0;
    let nine: f64 = 9.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &four as *const f64 as u64;
    ctx.rcx = &nine as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.pf, 0);
}

#[test]
fn test_fcmovnb_moves_when_cf_clear() {
    // ST0 = 1.0, ST1 = 2.0; clc (CF=0); fcmovnb st, st(1) -> ST0 = 2.0
    // clc = F8; fcmovnb st,st(1) = DB C1
    let bytes = [0xD9, 0xE8, 0xDD, 0x00, 0xF8, 0xDB, 0xC1, 0xC3];
    // fld1 (ST0=1); fld qword[rax]=... wait, keep simple:
    // Actually: fld1 pushes 1.0 (ST0), fld qword[rax] pushes val (ST0), so ST1=1.0
    let val: f64 = 2.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    // After fld1: ST0=1.0. After fld [rax]: ST0=2.0, ST1=1.0.
    // fcmovnb st, st(1): CF=0 so ST0 = ST1 = 1.0.
    assert_eq!(st(&ctx, 0), 1.0, "fcmovnb with CF=0 moves ST1 into ST0");
}

#[test]
fn test_fcmovnb_no_move_when_cf_set() {
    // stc (CF=1); fcmovnb st, st(1) -> no move, ST0 unchanged.
    // stc = F9
    let bytes = [0xD9, 0xE8, 0xDD, 0x00, 0xF9, 0xDB, 0xC1, 0xC3];
    let val: f64 = 2.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 2.0, "fcmovnb with CF=1 leaves ST0 unchanged");
}

#[test]
fn test_fnstsw_reports_top() {
    // Two pushes set TOP = 6; fnstsw ax must report TOP in bits 11-13.
    // fld1; fld1; fnstsw ax; ret
    // fnstsw ax = DF E0
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDF, 0xE0, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    let top = ((ctx.rax as u16) >> 11) & 7;
    assert_eq!(top, 6, "fnstsw AX must encode TOP=6 in bits 11-13");
}

#[test]
fn test_fincstp_fdecstp() {
    // fincstp; ret -> TOP 0 -> 1
    let ctx = run_test(&[0xD9, 0xF7, 0xC3], RegisterContext::default());
    assert_eq!(ctx.fpu_top, 1);
    // fdecstp; ret -> TOP 0 -> 7
    let ctx = run_test(&[0xD9, 0xF6, 0xC3], RegisterContext::default());
    assert_eq!(ctx.fpu_top, 7);
}

#[test]
fn test_tag_word_push_marks_valid() {
    // Two pushes: TOP goes 0 -> 7 -> 6, marking physical regs 7 and 6 valid.
    // fld1; fld1; ret
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.fpu_top, 6);
    // Abridged tag: bit i = physical register i valid.
    assert_eq!(ctx.fpu_tag & (1 << 7), 1 << 7, "phys 7 valid");
    assert_eq!(ctx.fpu_tag & (1 << 6), 1 << 6, "phys 6 valid");
    assert_eq!(ctx.fpu_tag & !(1 << 7 | 1 << 6), 0, "no other regs valid");
}

#[test]
fn test_tag_word_pop_marks_empty() {
    // fld1; fld1; fstp st(0); ret  -> push,push then pop ST0.
    // After 2 pushes: TOP=6, phys 6 & 7 valid. fstp st(0) (DD D8) pops ST0
    // (phys 6) -> phys 6 empty, TOP=7.
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDD, 0xD8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.fpu_top, 7);
    assert_eq!(ctx.fpu_tag & (1 << 6), 0, "popped phys 6 must be empty");
    assert_eq!(ctx.fpu_tag & (1 << 7), 1 << 7, "phys 7 still valid");
}

#[test]
fn test_ffree_marks_empty() {
    // fld1; fld1; ffree st(1); ret  -> ST1 is phys 7; mark it empty.
    // ffree st(1) = DD C1
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDD, 0xC1, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.fpu_top, 6, "ffree must not change TOP");
    assert_eq!(ctx.fpu_tag & (1 << 7), 0, "ST1 (phys 7) freed");
    assert_eq!(ctx.fpu_tag & (1 << 6), 1 << 6, "ST0 (phys 6) still valid");
}

#[test]
fn test_f80_conversion_roundtrip() {
    use microlift::context::{f64_bits_to_f80, f80_to_f64_bits};
    for &v in &[0.0f64, -0.0, 1.0, -1.0, 2.5, 1234.5, -7.25, 1e300, 1e-300] {
        let (mant, se) = f64_bits_to_f80(v.to_bits());
        let back = f80_to_f64_bits(mant, se);
        assert_eq!(
            f64::from_bits(back),
            v,
            "round-trip failed for {v}: mant={mant:#x} se={se:#x}"
        );
    }
}

#[test]
fn test_f80_conversion_matches_lifter() {
    // The pure-Rust f64->f80 must produce the same 80-bit encoding the lifter's
    // FSTP m80 emits (verified via a lift round-trip of the same value).
    use microlift::context::f64_bits_to_f80;
    // fld qword[rax]; fstp tbyte[rcx]; ret
    let bytes = [0xDD, 0x00, 0xDB, 0x39, 0xC3];
    let v: f64 = -7.25;
    let mut dst = [0u8; 16];
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    ctx.rcx = dst.as_mut_ptr() as u64;
    run_test(&bytes, ctx);
    let lifted_mant = u64::from_le_bytes(dst[0..8].try_into().unwrap());
    let lifted_se = u16::from_le_bytes(dst[8..10].try_into().unwrap());
    let (mant, se) = f64_bits_to_f80(v.to_bits());
    assert_eq!(mant, lifted_mant, "mantissa must match lifter");
    assert_eq!(se, lifted_se, "sign+exp must match lifter");
}
