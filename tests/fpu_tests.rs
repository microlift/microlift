//! x87 tests: lift a snippet and check `fpr`/`fpu_top` or memory after JIT execution.
//! The stack is modelled in `f64`, so registers are compared as raw `f64` bits.

use microlift::testkit::prelude::*;

fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

fn st(ctx: &RegisterContext, i: usize) -> f64 {
    use microlift::context::f80_to_f64_bits;
    let phys = (ctx.fpu_top as usize + i) & 7;
    let [mant, se] = ctx.fpr[phys];
    f64::from_bits(f80_to_f64_bits(mant, se as u16))
}

fn set_phys(ctx: &mut RegisterContext, phys: usize, v: f64) {
    use microlift::context::f64_bits_to_f80;
    let (mant, se) = f64_bits_to_f80(v.to_bits());
    ctx.fpr[phys] = [mant, se as u64];
}

#[test]
fn test_fldz_pushes_zero() {
    let bytes = [0xD9, 0xEE, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.fpu_top = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.fpu_top, 7, "push must decrement TOP (0 -> 7)");
    assert_eq!(st(&ctx, 0), 0.0);
}

#[test]
fn test_fld1_pushes_one() {
    let bytes = [0xD9, 0xE8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 1.0);
}

#[test]
fn test_fldpi_pushes_pi() {
    let bytes = [0xD9, 0xEB, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), std::f64::consts::PI);
}

#[test]
fn test_two_pushes_order() {
    let bytes = [0xD9, 0xE8, 0xD9, 0xEE, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 0.0, "last push is ST0");
    assert_eq!(st(&ctx, 1), 1.0, "earlier push is ST1");
    assert_eq!(ctx.fpu_top, 6);
}

#[test]
fn test_flds_load_f32() {
    let bytes = [0xD9, 0x00, 0xC3];
    let val: f32 = 1.5;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 1.5f64);
}

#[test]
fn test_fldl_load_f64() {
    let bytes = [0xDD, 0x00, 0xC3];
    let val: f64 = 3.25;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 3.25);
}

#[test]
fn test_fstp_store_f64() {
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


#[test]
fn test_fldt_fstpt_roundtrip() {
    let bytes = [0xDB, 0x28, 0xDB, 0x39, 0xC3];
    // 80-bit 2.0: exponent 16384, integer bit set.
    let value = 2.0f64;
    let mut src = [0u8; 16];
    encode_f80(value, &mut src);
    let mut dst = [0u8; 16];
    let mut ctx = RegisterContext::default();
    ctx.rax = src.as_ptr() as u64;
    ctx.rcx = dst.as_mut_ptr() as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(decode_f80(&dst), 2.0);
    assert_eq!(ctx.fpu_top, 0, "fldt then fstp is balanced");
}

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
    // Low significand bits that would be lost through f64.
    let bytes = [0xDB, 0x28, 0xDB, 0x39, 0xC3];
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
    assert_eq!(u64::from_le_bytes(dst[0..8].try_into().unwrap()), mant);
    assert_eq!(u16::from_le_bytes(dst[8..10].try_into().unwrap()), se);
}

#[test]
fn test_fild_large_i64_exact() {
    // More than 53 significant bits: exact in 80-bit, would round through f64.
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


#[test]
fn test_fild_load_i32() {
    let bytes = [0xDB, 0x00, 0xC3];
    let val: i32 = -42;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const i32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), -42.0);
}

#[test]
fn test_fistp_round_to_nearest() {
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
    let bytes = [0xDD, 0x00, 0xDD, 0x09, 0xC3];
    let src: f64 = -3.9; // truncate toward zero -> -3
    let mut dst: i64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.rcx = &mut dst as *mut i64 as u64;
    run_test(&bytes, ctx);
    assert_eq!(dst, -3, "fisttp must truncate toward zero");
}


#[test]
fn test_fxch_swaps_st0_st1() {
    let bytes = [0xD9, 0xE8, 0xD9, 0xEE, 0xD9, 0xC9, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 1.0);
    assert_eq!(st(&ctx, 1), 0.0);
}

#[test]
fn test_faddp_adds_and_pops() {
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDE, 0xC1, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(st(&ctx, 0), 2.0);
    assert_eq!(ctx.fpu_top, 7, "two pushes then one pop -> TOP = 7");
}

#[test]
fn test_fsubp_order() {
    let bytes = [0xDE, 0xE9, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.fpu_top = 6;
    set_phys(&mut ctx, 6, 3.0); // ST0
    set_phys(&mut ctx, 7, 10.0); // ST1
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.fpu_top, 7, "fsubp pops");
    assert_eq!(st(&ctx, 0), 7.0, "ST1 - ST0 = 10 - 3 = 7 becomes new ST0");
}

#[test]
fn test_fmul_mem() {
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
    let bytes = [0xDC, 0x38, 0xC3];
    let mem: f64 = 10.0;
    let mut ctx = RegisterContext::default();
    ctx.fpu_top = 0;
    set_phys(&mut ctx, 0, 2.0);
    ctx.rax = &mem as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 5.0, "fdivr: mem / ST0 = 10 / 2 = 5");
}

#[test]
fn test_fabs_fchs_fsqrt() {
    // sqrt(-16) is NaN.
    let bytes = [0xDD, 0x00, 0xD9, 0xE1, 0xC3];
    let v: f64 = -16.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 16.0, "fabs of -16 = 16");
}

#[test]
fn test_fsqrt() {
    let bytes = [0xDD, 0x00, 0xD9, 0xFA, 0xC3];
    let v: f64 = 144.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 12.0);
}

#[test]
fn test_fucomi_less() {
    let bytes = [0xDD, 0x00, 0xDD, 0x01, 0xDB, 0xE9, 0xC3];
    let two: f64 = 2.0;
    let one: f64 = 1.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &two as *const f64 as u64;
    ctx.rcx = &one as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 1, "1 < 2 -> CF set");
    assert_eq!(ctx.zf, 0, "not equal -> ZF clear");
    assert_eq!(u8::from(ctx.pf()), 0, "ordered -> PF clear");
}

#[test]
fn test_fucomi_equal() {
    let bytes = [0xDD, 0x00, 0xDD, 0x00, 0xDB, 0xE9, 0xC3];
    let five: f64 = 5.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &five as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.zf, 1, "equal -> ZF set");
    assert_eq!(ctx.cf, 0, "equal -> CF clear");
    assert_eq!(u8::from(ctx.pf()), 0, "ordered -> PF clear");
}

#[test]
fn test_fucomi_greater() {
    let bytes = [0xDD, 0x00, 0xDD, 0x01, 0xDB, 0xE9, 0xC3];
    let four: f64 = 4.0;
    let nine: f64 = 9.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &four as *const f64 as u64;
    ctx.rcx = &nine as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
    assert_eq!(u8::from(ctx.pf()), 0);
}

#[test]
fn test_fcmovnb_moves_when_cf_clear() {
    let bytes = [0xD9, 0xE8, 0xDD, 0x00, 0xF8, 0xDB, 0xC1, 0xC3];
    let val: f64 = 2.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 1.0, "fcmovnb with CF=0 moves ST1 into ST0");
}

#[test]
fn test_fcmovnb_no_move_when_cf_set() {
    let bytes = [0xD9, 0xE8, 0xDD, 0x00, 0xF9, 0xDB, 0xC1, 0xC3];
    let val: f64 = 2.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const f64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(st(&ctx, 0), 2.0, "fcmovnb with CF=1 leaves ST0 unchanged");
}

#[test]
fn test_fnstsw_reports_top() {
    // TOP is reported in status-word bits 11-13.
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDF, 0xE0, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    let top = ((ctx.rax as u16) >> 11) & 7;
    assert_eq!(top, 6, "fnstsw AX must encode TOP=6 in bits 11-13");
}

#[test]
fn test_fincstp_fdecstp() {
    let ctx = run_test(&[0xD9, 0xF7, 0xC3], RegisterContext::default());
    assert_eq!(ctx.fpu_top, 1);
    let ctx = run_test(&[0xD9, 0xF6, 0xC3], RegisterContext::default());
    assert_eq!(ctx.fpu_top, 7);
}

#[test]
fn test_tag_word_push_marks_valid() {
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
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDD, 0xD8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.fpu_top, 7);
    assert_eq!(ctx.fpu_tag & (1 << 6), 0, "popped phys 6 must be empty");
    assert_eq!(ctx.fpu_tag & (1 << 7), 1 << 7, "phys 7 still valid");
}

#[test]
fn test_ffree_marks_empty() {
    let bytes = [0xD9, 0xE8, 0xD9, 0xE8, 0xDD, 0xC1, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.fpu_top, 6, "ffree must not change TOP");
    assert_eq!(ctx.fpu_tag & (1 << 7), 0, "ST1 (phys 7) freed");
    assert_eq!(ctx.fpu_tag & (1 << 6), 1 << 6, "ST0 (phys 6) still valid");
}

#[test]
fn test_emms_marks_all_registers_empty() {
    let bytes = [0xD9, 0xE8, 0xD9, 0xEE, 0x0F, 0x77, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());

    assert_eq!(ctx.fpu_top, 6, "EMMS must preserve TOP");
    assert_eq!(
        ctx.fpu_tag, 0,
        "EMMS must mark every x87/MMX register empty"
    );
    assert_eq!(
        st(&ctx, 0),
        0.0,
        "EMMS must not overwrite register contents"
    );
    assert_eq!(
        st(&ctx, 1),
        1.0,
        "EMMS must not overwrite register contents"
    );
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
    // Must match the encoding FSTP m80 emits.
    use microlift::context::f64_bits_to_f80;
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

fn fcos(v: f64) -> RegisterContext {
    let bytes = [0xDD, 0x00, 0xD9, 0xFF, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    run_test(&bytes, ctx)
}

/// Libm agreement to a few `f64` ulp (kernel has a 64-bit significand, libm 53).
fn assert_close(got: f64, want: f64, what: &str) {
    let tol = 8.0 * f64::EPSILON * want.abs().max(f64::MIN_POSITIVE);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e} (tolerance {tol:e})"
    );
}

#[test]
fn test_fcos_small_arguments() {
    // Below 1/2 no reduction happens.
    for &v in &[0.0f64, -0.0, 0.25, -0.25, 0.5, -0.5, 1e-10, 1e-30, 1e-300] {
        let ctx = fcos(v);
        assert_close(st(&ctx, 0), v.cos(), &format!("cos({v:e})"));
        assert_eq!(ctx.fpu_sw & (1 << 10), 0, "C2 clear for in-range {v:e}");
    }
}

#[test]
fn test_fcos_all_quadrants() {
    // One argument per quadrant, plus sign symmetry.
    for &v in &[
        1.0f64,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        std::f64::consts::TAU,
        100.0,
        -100.0,
        1234.5678,
    ] {
        let ctx = fcos(v);
        assert_close(st(&ctx, 0), v.cos(), &format!("cos({v})"));
    }
}

#[test]
fn test_fcos_large_arguments() {
    // Naive reduction against a rounded pi loses every bit out here.
    for &v in &[1e6f64, 1e10, 1e15, 1e18, 4.611686018427388e18, -1e18] {
        let ctx = fcos(v);
        assert_close(st(&ctx, 0), v.cos(), &format!("cos({v:e})"));
        assert_eq!(ctx.fpu_sw & (1 << 10), 0, "C2 clear for in-range {v:e}");
    }
}

#[test]
fn test_fcos_near_multiple_of_half_pi() {
    // Near-total cancellation: the remainder comes from the low bits of the 2/pi window.
    for &v in &[
        std::f64::consts::FRAC_PI_2,
        3.0 * std::f64::consts::FRAC_PI_2,
        5.0 * std::f64::consts::FRAC_PI_2,
        7074237752028440.0, // ~ a multiple of pi/2 to within 1e-10
    ] {
        let ctx = fcos(v);
        assert_close(st(&ctx, 0), v.cos(), &format!("cos({v:e})"));
    }
}

#[test]
fn test_fcos_out_of_range_sets_c2() {
    // |x| >= 2^63 is outside the reduction domain: ST(0) untouched, C2 set.
    let v = 1e19f64;
    let ctx = fcos(v);
    assert_eq!(st(&ctx, 0), v, "out-of-range argument must be left alone");
    assert_ne!(ctx.fpu_sw & (1 << 10), 0, "C2 must be set out of range");

    let v = 9.223372036854775e18f64; // < 2^63
    let ctx = fcos(v);
    assert_eq!(ctx.fpu_sw & (1 << 10), 0, "C2 clear just inside 2^63");
    assert_close(st(&ctx, 0), v.cos(), "cos just below 2^63");
}

#[test]
fn test_fcos_nan_and_infinity() {
    // Checked on the raw 80-bit encoding; `st()` cannot see NaN/infinity exponents.
    for &v in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let ctx = fcos(v);
        let phys = ctx.fpu_top as usize;
        let [mant, se] = ctx.fpr[phys];
        assert_eq!(
            (mant, se as u16),
            (0xC000_0000_0000_0000, 0xFFFF),
            "cos({v}) must be the QNaN indefinite"
        );
        assert_eq!(
            ctx.fpu_sw & (1 << 10),
            0,
            "C2 clear for {v} (not a range error)"
        );
    }
}

#[test]
fn test_fcos_does_not_move_the_stack() {
    let ctx = fcos(1.0);
    assert_eq!(ctx.fpu_top, 7, "one fld, no push or pop from fcos");
}

#[test]
fn test_fcos_full_80_bit_precision() {
    // cos(x) to 90 digits over non-f64 arguments, including one on the 80-bit pi/2 (result ~3.4e-20).
    let bytes = [0xDB, 0x28, 0xD9, 0xFF, 0xC3];
    #[rustfmt::skip]
    const CASES: &[((u64, u16), (u64, u16))] = &[
    ((0x8000000000000000, 0x3fff), (0x8a51407da8345c92, 0x3ffe)),
    ((0xc90fdaa22168c235, 0x3fff), (0xece675d1fc8f8cbb, 0xbfbd)),
    ((0xc90fdaa22168c235, 0x4000), (0x8000000000000000, 0xbfff)),
    ((0xc90fdaa22168c235, 0x4001), (0x8000000000000000, 0x3fff)),
    ((0xb504f333f9de6484, 0x3ffe), (0xc29f63d166890d28, 0x3ffe)),
    ((0x9e3779b97f4a7c15, 0x4020), (0x8b76571692d1110a, 0xbffe)),
    ((0xf4a2b3c1d5e60718, 0x403d), (0xb25add342b6e5ee7, 0x3ffe)),
    ((0xffffffffffffffff, 0x403d), (0xfac035ec484929c2, 0x3ffd)),
    ((0x8000000000000000, 0x3f00), (0x8000000000000000, 0x3fff)),
    ((0xdeadbeefcafef00d, 0x400a), (0xf51617aa5b87f178, 0x3ffe)),
    ((0xc90fdaa22168c235, 0xbfff), (0xece675d1fc8f8cbb, 0xbfbd)),
    ((0x9e3779b97f4a7c15, 0xc020), (0x8b76571692d1110a, 0xbffe)),
    ((0x8000000000000000, 0x3ffe), (0xe0a94032dbea7cee, 0x3ffe)),
    ((0xffffffffffffffff, 0x3ffe), (0x8a51407da8345c93, 0x3ffe)),
    ];

    for &((am, ase), (wm, wse)) in CASES {
        let arg = [am, ase as u64];
        let mut ctx = RegisterContext::default();
        ctx.rax = arg.as_ptr() as u64;
        let ctx = run_test(&bytes, ctx);
        let [gm, gse] = ctx.fpr[ctx.fpu_top as usize];
        let gse = gse as u16;
        assert_eq!(
            gse, wse,
            "cos({am:#018x},{ase:#06x}): exponent {gse:#06x}, want {wse:#06x}"
        );
        // Faithful, not correctly rounded: one case in the table moves by a bit.
        let err = gm.abs_diff(wm);
        assert!(
            err <= 1,
            "cos({am:#018x},{ase:#06x}): significand {gm:#018x}, want {wm:#018x} ({err} ulp)"
        );
    }
}

fn trig_op(op: [u8; 2], v: f64) -> RegisterContext {
    let bytes = [0xDD, 0x00, op[0], op[1], 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = &v as *const f64 as u64;
    run_test(&bytes, ctx)
}

const FSIN: [u8; 2] = [0xD9, 0xFE];
const FSINCOS: [u8; 2] = [0xD9, 0xFB];
const FPTAN: [u8; 2] = [0xD9, 0xF2];

#[test]
fn test_fsin_matches_libm() {
    for &v in &[
        0.0f64,
        0.25,
        -0.25,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        -1.0,
        -2.0,
        -3.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        std::f64::consts::TAU,
        100.0,
        -100.0,
        1234.5678,
        1e-10,
        1e6,
        1e10,
        1e15,
        1e18,
        -1e18,
        4.611686018427388e18,
    ] {
        let ctx = trig_op(FSIN, v);
        assert_close(st(&ctx, 0), v.sin(), &format!("sin({v:e})"));
        assert_eq!(ctx.fpu_sw & (1 << 10), 0, "C2 clear for in-range {v:e}");
    }
}

#[test]
fn test_fsin_is_odd_about_zero() {
    // The sign of a zero result comes from the argument.
    let ctx = trig_op(FSIN, -0.0);
    let [mant, se] = ctx.fpr[ctx.fpu_top as usize];
    assert_eq!(mant, 0, "sin(-0) is a zero");
    assert_eq!(se as u16 >> 15, 1, "sin(-0) keeps the sign: -0");

    let ctx = trig_op(FSIN, 0.0);
    let [mant, se] = ctx.fpr[ctx.fpu_top as usize];
    assert_eq!((mant, se as u16 >> 15), (0, 0), "sin(+0) is +0");
}

#[test]
fn test_fsin_out_of_range_sets_c2() {
    let v = 1e19f64;
    let ctx = trig_op(FSIN, v);
    assert_eq!(st(&ctx, 0), v, "out-of-range argument must be left alone");
    assert_ne!(ctx.fpu_sw & (1 << 10), 0, "C2 must be set out of range");
}

#[test]
fn test_fsin_nan_and_infinity() {
    for &v in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let ctx = trig_op(FSIN, v);
        let [mant, se] = ctx.fpr[ctx.fpu_top as usize];
        assert_eq!(
            (mant, se as u16),
            (0xC000_0000_0000_0000, 0xFFFF),
            "sin({v}) must be the QNaN indefinite"
        );
    }
}

#[test]
fn test_fsincos_leaves_cos_over_sin() {
    for &v in &[0.0f64, 1.0, -1.0, 2.0, 5.0, 1234.5678, 1e18] {
        let ctx = trig_op(FSINCOS, v);
        assert_eq!(ctx.fpu_top, 6, "fld pushed once, fsincos once more");
        assert_close(st(&ctx, 0), v.cos(), &format!("fsincos cos({v:e})"));
        assert_close(st(&ctx, 1), v.sin(), &format!("fsincos sin({v:e})"));
        assert_eq!(ctx.fpu_tag & (1 << 6), 1 << 6, "pushed slot marked valid");
    }
}

#[test]
fn test_fsincos_out_of_range_does_not_push() {
    let v = 1e19f64;
    let ctx = trig_op(FSINCOS, v);
    assert_ne!(ctx.fpu_sw & (1 << 10), 0, "C2 must be set out of range");
    assert_eq!(ctx.fpu_top, 7, "no push when out of range");
    assert_eq!(st(&ctx, 0), v, "ST(0) must be untouched");
}

#[test]
fn test_fptan_leaves_one_over_tan() {
    for &v in &[0.0f64, 0.25, 1.0, -1.0, 2.0, 5.0, 1234.5678] {
        let ctx = trig_op(FPTAN, v);
        assert_eq!(ctx.fpu_top, 6, "fptan pushes 1.0");
        assert_eq!(st(&ctx, 0), 1.0, "fptan pushes exactly 1.0");
        assert_close(st(&ctx, 1), v.tan(), &format!("tan({v})"));
    }
}

#[test]
fn test_fptan_out_of_range_does_not_push() {
    let v = 1e19f64;
    let ctx = trig_op(FPTAN, v);
    assert_ne!(ctx.fpu_sw & (1 << 10), 0, "C2 must be set out of range");
    assert_eq!(ctx.fpu_top, 7, "no push when out of range");
    assert_eq!(st(&ctx, 0), v, "ST(0) must be untouched");
}

#[test]
fn test_fsin_full_80_bit_precision() {
    // sin(x) to 90 digits; 2^-255 proves the unreduced path keeps relative precision.
    let bytes = [0xDB, 0x28, 0xD9, 0xFE, 0xC3];
    #[rustfmt::skip]
    const CASES: &[((u64, u16), (u64, u16))] = &[
    ((0x8000000000000000, 0x3fff), (0xd76aa47848677021, 0x3ffe)),
    ((0xc90fdaa22168c235, 0x3fff), (0x8000000000000000, 0x3fff)),
    ((0xc90fdaa22168c235, 0x4000), (0xece675d1fc8f8cbb, 0xbfbe)),
    ((0xc90fdaa22168c235, 0x4001), (0xece675d1fc8f8cbb, 0x3fbf)),
    ((0xb504f333f9de6484, 0x3ffe), (0xa64e9b3f9f566d57, 0x3ffe)),
    ((0x9e3779b97f4a7c15, 0x4020), (0xd6ad595348042405, 0x3ffe)),
    ((0xf4a2b3c1d5e60718, 0x403d), (0xb7a52418ea05b817, 0xbffe)),
    ((0xffffffffffffffff, 0x403d), (0xdf327e112abeef8f, 0x3ffe)),
    ((0x8000000000000000, 0x3f00), (0x8000000000000000, 0x3f00)),
    ((0xdeadbeefcafef00d, 0x400a), (0x93e6c25502aca735, 0x3ffd)),
    ((0xc90fdaa22168c235, 0xbfff), (0x8000000000000000, 0xbfff)),
    ((0x9e3779b97f4a7c15, 0xc020), (0xd6ad595348042405, 0xbffe)),
    ((0x8000000000000000, 0x3ffe), (0xf57743a2582f7f44, 0x3ffd)),
    ((0xffffffffffffffff, 0x3ffe), (0xd76aa47848677020, 0x3ffe)),
    ];

    for &((am, ase), (wm, wse)) in CASES {
        let arg = [am, ase as u64];
        let mut ctx = RegisterContext::default();
        ctx.rax = arg.as_ptr() as u64;
        let ctx = run_test(&bytes, ctx);
        let [gm, gse] = ctx.fpr[ctx.fpu_top as usize];
        let gse = gse as u16;
        assert_eq!(
            gse, wse,
            "sin({am:#018x},{ase:#06x}): exponent {gse:#06x}, want {wse:#06x}"
        );
        let err = gm.abs_diff(wm);
        assert!(
            err <= 1,
            "sin({am:#018x},{ase:#06x}): significand {gm:#018x}, want {wm:#018x} ({err} ulp)"
        );
    }
}
