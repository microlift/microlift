//! Software x87 80-bit extended-precision floating point, emitted as pure
//! Cranelift IR.
//!
//! Each operation is a free `emit_*` function that builds its body into a given
//! [`FunctionBuilder`], taking and returning values in the lifter's raw 80-bit
//! representation: a `(mant: i64, se: i32)` pair, where `mant` is the 64-bit
//! significand (explicit integer bit at position 63 for normal values) and `se`
//! holds the sign at bit 15 and the 15-bit biased exponent (bias 16383) in the
//! low bits.
//!
//! Writing them as free IR-builders (rather than `Lifter` methods) means the
//! same body serves two emission strategies without changing the logic:
//! *inline* at each x87 site, or emitted once into a co-located helper function
//! that every site `call`s. [`F80Kernel`] names the outlinable kernels and
//! builds those helper bodies; the engine's outlined-helper table
//! ([`crate::core::LiftEngine::outlined_helper`]) declares and de-duplicates
//! them. Outlining is the default - these kernels are hundreds of IR
//! instructions each (the divide and square root contain 128-bit
//! shift/subtract loops), so inlining them at every `FMUL`/`FDIV` in a region
//! bloats the emitted code far past what the instruction cache tolerates.
//!
//! Scope and known approximations:
//!  - Handles normal values, zero, infinity and NaN correctly for `+ - * /` and
//!    the ordering compare, with round-to-nearest-even.
//!  - Subnormal *results* are flushed to zero (underflow), and subnormal
//!    *inputs* are treated via their stored significand without renormalisation.
//!    x87 `long double` from a compiler rarely produces these; full subnormal
//!    support is a later refinement.
//!  - NaN results use the canonical x87 QNaN indefinite rather than propagating
//!    input payloads bit-for-bit.

use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;

const EXP_MAX: i64 = 0x7FFF;
const BIAS: i64 = 16383;
const INT_BIT: i64 = 0x8000_0000_0000_0000u64 as i64;
const FRAC_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFF;
/// x87 QNaN floating-point indefinite: sign 1, exp all-ones, mantissa `1.1000…`.
const QNAN_MANT: i64 = 0xC000_0000_0000_0000u64 as i64;
const QNAN_SE: i64 = 0xFFFF;

fn sign_of(b: &mut FunctionBuilder, se: Value) -> Value {
    let s = b.ins().ushr_imm_u(se, 15);
    b.ins().band_imm_u(s, 1)
}

fn exp_of(b: &mut FunctionBuilder, se: Value) -> Value {
    b.ins().band_imm_u(se, EXP_MAX)
}

fn frac_of(b: &mut FunctionBuilder, mant: Value) -> Value {
    b.ins().band_imm_u(mant, FRAC_MASK)
}

fn is_exp_max(b: &mut FunctionBuilder, se: Value) -> Value {
    let e = exp_of(b, se);
    b.ins().icmp_imm_s(IntCC::Equal, e, EXP_MAX)
}

fn is_nan(b: &mut FunctionBuilder, mant: Value, se: Value) -> Value {
    let em = is_exp_max(b, se);
    let f = frac_of(b, mant);
    let fnz = b.ins().icmp_imm_s(IntCC::NotEqual, f, 0);
    b.ins().band(em, fnz)
}

fn is_inf(b: &mut FunctionBuilder, mant: Value, se: Value) -> Value {
    let em = is_exp_max(b, se);
    let f = frac_of(b, mant);
    let fz = b.ins().icmp_imm_s(IntCC::Equal, f, 0);
    b.ins().band(em, fz)
}

fn is_zero(b: &mut FunctionBuilder, mant: Value, se: Value) -> Value {
    let e = exp_of(b, se);
    let ez = b.ins().icmp_imm_s(IntCC::Equal, e, 0);
    let mz = b.ins().icmp_imm_s(IntCC::Equal, mant, 0);
    b.ins().band(ez, mz)
}

/// Assemble a `se` field (I32) from a sign bit (0/1) and biased exponent.
fn compose_se(b: &mut FunctionBuilder, sign: Value, exp: Value) -> Value {
    let s = b.ins().ishl_imm_u(sign, 15);
    b.ins().bor(s, exp)
}

fn qnan(b: &mut FunctionBuilder) -> (Value, Value) {
    let m = b.ins().iconst(types::I64, QNAN_MANT);
    let se = b.ins().iconst(types::I32, QNAN_SE);
    (m, se)
}

fn inf(b: &mut FunctionBuilder, sign: Value) -> (Value, Value) {
    let m = b.ins().iconst(types::I64, INT_BIT);
    let e = b.ins().iconst(types::I32, EXP_MAX);
    let se = compose_se(b, sign, e);
    (m, se)
}

fn signed_zero(b: &mut FunctionBuilder, sign: Value) -> (Value, Value) {
    let m = b.ins().iconst(types::I64, 0);
    let e = b.ins().iconst(types::I32, 0);
    let se = compose_se(b, sign, e);
    (m, se)
}

/// Round a 64-bit significand `sig` up by `inc` (0/1) with carry handling: if it
/// overflows to `2^64`, reset to `0x8000…0` and report a `+1` exponent step.
/// Returns `(sig_rounded, exp_carry_i32)`.
fn round_significand(b: &mut FunctionBuilder, sig: Value, inc: Value) -> (Value, Value) {
    let sig_r = b.ins().iadd(sig, inc);
    // Overflow only when sig was all-ones and inc==1, wrapping to 0.
    let carry = b.ins().icmp_imm_s(IntCC::Equal, sig_r, 0);
    let msb = b.ins().iconst(types::I64, INT_BIT);
    let sig_final = b.ins().select(carry, msb, sig_r);
    let exp_carry = b.ins().uextend(types::I32, carry);
    (sig_final, exp_carry)
}

/// Select the special-case result for `a op b`, given the normal-path result
/// `(nm, nse)` and the operand classes. `mode` picks multiply vs divide operand
/// combinations.
#[allow(clippy::too_many_arguments)]
fn apply_specials_mul(
    b: &mut FunctionBuilder,
    nm: Value,
    nse: Value,
    sign: Value,
    a_nan: Value,
    b_nan: Value,
    a_inf: Value,
    b_inf: Value,
    a_zero: Value,
    b_zero: Value,
) -> (Value, Value) {
    let (zm, zse) = signed_zero(b, sign);
    let (im, ise) = inf(b, sign);
    let (qm, qse) = qnan(b);

    let any_zero = b.ins().bor(a_zero, b_zero);
    let any_inf = b.ins().bor(a_inf, b_inf);
    let inf_x_zero = {
        let az_bi = b.ins().band(a_zero, b_inf);
        let ai_bz = b.ins().band(a_inf, b_zero);
        b.ins().bor(az_bi, ai_bz)
    };
    let any_nan = b.ins().bor(a_nan, b_nan);

    let mut m = nm;
    let mut se = nse;
    (m, se) = (
        b.ins().select(any_zero, zm, m),
        b.ins().select(any_zero, zse, se),
    );
    (m, se) = (
        b.ins().select(any_inf, im, m),
        b.ins().select(any_inf, ise, se),
    );
    (m, se) = (
        b.ins().select(inf_x_zero, qm, m),
        b.ins().select(inf_x_zero, qse, se),
    );
    (m, se) = (
        b.ins().select(any_nan, qm, m),
        b.ins().select(any_nan, qse, se),
    );
    (m, se)
}

/// Clamp a normal-path result to Inf on overflow / signed-zero on underflow.
fn clamp_over_under(
    b: &mut FunctionBuilder,
    sign: Value,
    exp: Value,
    nm: Value,
    nse: Value,
) -> (Value, Value) {
    let overflow = b
        .ins()
        .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, exp, EXP_MAX);
    let underflow = b.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, exp, 0);
    let (im, ise) = inf(b, sign);
    let (zm, zse) = signed_zero(b, sign);
    let m = b.ins().select(overflow, im, nm);
    let se = b.ins().select(overflow, ise, nse);
    let m = b.ins().select(underflow, zm, m);
    let se = b.ins().select(underflow, zse, se);
    (m, se)
}

/// Emit `a * b` for two raw 80-bit values.
pub fn emit_f80_mul(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    bm: Value,
    bse: Value,
) -> (Value, Value) {
    let sign_a = sign_of(b, ase);
    let sign_b = sign_of(b, bse);
    let sign = b.ins().bxor(sign_a, sign_b);
    let exp_a = exp_of(b, ase);
    let exp_b = exp_of(b, bse);

    let a_nan = is_nan(b, am, ase);
    let b_nan = is_nan(b, bm, bse);
    let a_inf = is_inf(b, am, ase);
    let b_inf = is_inf(b, bm, bse);
    let a_zero = is_zero(b, am, ase);
    let b_zero = is_zero(b, bm, bse);

    // 128-bit product of the two 64-bit significands. Both are in [2^63, 2^64)
    // for normals, so the product is in [2^126, 2^128): MSB at bit 126 or 127.
    let pa = b.ins().uextend(types::I128, am);
    let pb = b.ins().uextend(types::I128, bm);
    let prod = b.ins().imul(pa, pb);

    // top = 1 when the product reached bit 127.
    let top128 = b.ins().ushr_imm_u(prod, 127);
    let top = b.ins().ireduce(types::I32, top128);

    // Extract the 64-bit significand: shift = 63 + top so the integer bit lands
    // at bit 63 of the low word.
    let top128_ext = b.ins().uextend(types::I128, top);
    let extract_shift = b.ins().iadd_imm_s(top128_ext, 63);
    let sig128 = b.ins().ushr(prod, extract_shift);
    let sig = b.ins().ireduce(types::I64, sig128);

    // Round-to-nearest-even using the bits below `extract_shift`.
    let one64 = b.ins().iconst(types::I64, 1);
    let one128 = b.ins().uextend(types::I128, one64);
    let mask = {
        let sh = b.ins().ishl(one128, extract_shift);
        b.ins().iadd_imm_s(sh, -1)
    };
    let rem = b.ins().band(prod, mask);
    let half = {
        let hs = b.ins().iadd_imm_s(extract_shift, -1);
        b.ins().ishl(one128, hs)
    };
    let gt = b.ins().icmp(IntCC::UnsignedGreaterThan, rem, half);
    let eq = b.ins().icmp(IntCC::Equal, rem, half);
    let odd = {
        let o = b.ins().band_imm_u(sig, 1);
        b.ins().icmp_imm_s(IntCC::NotEqual, o, 0)
    };
    let tie = b.ins().band(eq, odd);
    let round_up = b.ins().bor(gt, tie);
    let inc = b.ins().uextend(types::I64, round_up);
    let (sig_r, exp_carry) = round_significand(b, sig, inc);

    // Biased result exponent: ea + eb - bias + top + carry.
    let esum = b.ins().iadd(exp_a, exp_b);
    let esum = b.ins().iadd_imm_s(esum, -BIAS);
    let esum = b.ins().iadd(esum, top);
    let exp = b.ins().iadd(esum, exp_carry);

    let nse = compose_se(b, sign, exp);
    let (nm, nse) = clamp_over_under(b, sign, exp, sig_r, nse);
    apply_specials_mul(b, nm, nse, sign, a_nan, b_nan, a_inf, b_inf, a_zero, b_zero)
}

/// Unsigned 128-bit / 128-bit division via restoring shift/subtract, returning
/// `(quotient, remainder)` as `I128`. The backend has no native `udiv.i128`,
/// but supports i128 shifts/adds/subtracts/compares, which is all this needs.
fn udivmod128(b: &mut FunctionBuilder, n: Value, d: Value) -> (Value, Value) {
    let i128 = types::I128;
    let zero = {
        let z = b.ins().iconst(types::I64, 0);
        b.ins().uextend(i128, z)
    };
    let one = {
        let o = b.ins().iconst(types::I64, 1);
        b.ins().uextend(i128, o)
    };

    let loop_b = b.create_block();
    b.append_block_param(loop_b, types::I64);
    b.append_block_param(loop_b, i128);
    b.append_block_param(loop_b, i128);
    let body = b.create_block();
    b.append_block_param(body, types::I64);
    b.append_block_param(body, i128);
    b.append_block_param(body, i128);
    let exit = b.create_block();
    b.append_block_param(exit, i128);
    b.append_block_param(exit, i128);

    let k0 = b.ins().iconst(types::I64, 0);
    b.ins().jump(loop_b, &[k0.into(), zero.into(), zero.into()]);

    b.switch_to_block(loop_b);
    let k = b.block_params(loop_b)[0];
    let q = b.block_params(loop_b)[1];
    let r = b.block_params(loop_b)[2];
    let done = b.ins().icmp_imm_s(IntCC::Equal, k, 128);
    b.ins().brif(
        done,
        exit,
        &[q.into(), r.into()],
        body,
        &[k.into(), q.into(), r.into()],
    );

    b.switch_to_block(body);
    let k = b.block_params(body)[0];
    let q = b.block_params(body)[1];
    let r = b.block_params(body)[2];
    let c127 = b.ins().iconst(types::I64, 127);
    let bit_pos = b.ins().isub(c127, k);
    let r_sh = b.ins().ishl_imm_u(r, 1);
    let n_bit = {
        let nb = b.ins().ushr(n, bit_pos);
        b.ins().band(nb, one)
    };
    let r_new = b.ins().bor(r_sh, n_bit);
    let ge = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, r_new, d);
    let r_sub = b.ins().isub(r_new, d);
    let r_out = b.ins().select(ge, r_sub, r_new);
    let q_bit = b.ins().ishl(one, bit_pos);
    let q_set = b.ins().bor(q, q_bit);
    let q_out = b.ins().select(ge, q_set, q);
    let k_next = b.ins().iadd_imm_s(k, 1);
    b.ins()
        .jump(loop_b, &[k_next.into(), q_out.into(), r_out.into()]);

    b.switch_to_block(exit);
    let quot = b.block_params(exit)[0];
    let rem = b.block_params(exit)[1];
    b.seal_block(loop_b);
    b.seal_block(body);
    b.seal_block(exit);
    (quot, rem)
}

/// Emit `a / b` for two raw 80-bit values.
pub fn emit_f80_div(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    bm: Value,
    bse: Value,
) -> (Value, Value) {
    let sign_a = sign_of(b, ase);
    let sign_b = sign_of(b, bse);
    let sign = b.ins().bxor(sign_a, sign_b);
    let exp_a = exp_of(b, ase);
    let exp_b = exp_of(b, bse);

    let a_nan = is_nan(b, am, ase);
    let b_nan = is_nan(b, bm, bse);
    let a_inf = is_inf(b, am, ase);
    let b_inf = is_inf(b, bm, bse);
    let a_zero = is_zero(b, am, ase);
    let b_zero = is_zero(b, bm, bse);

    // Q = floor(am * 2^64 / bm), R = remainder.  am,bm in [2^63,2^64) ->
    // Q in (2^63, 2^65).
    let num = {
        let a128 = b.ins().uextend(types::I128, am);
        b.ins().ishl_imm_u(a128, 64)
    };
    let den = b.ins().uextend(types::I128, bm);
    let (q, r) = udivmod128(b, num, den);

    let q_small = b.ins().ireduce(types::I64, q);
    let q_big_128 = b.ins().ushr_imm_u(q, 1);
    let q_big = b.ins().ireduce(types::I64, q_big_128);
    // is_big when Q >= 2^64 (bit 64 or higher set).
    let is_big = {
        let hi = b.ins().ushr_imm_u(q, 64);
        let hi = b.ins().ireduce(types::I64, hi);
        b.ins().icmp_imm_s(IntCC::NotEqual, hi, 0)
    };
    let sig = b.ins().select(is_big, q_big, q_small);

    // Rounding.
    //  big:   guard = Q&1, sticky = (R != 0).
    //  small: fraction R/bm -> compare 2R with bm.
    let r_nz = {
        let z128 = {
            let z = b.ins().iconst(types::I64, 0);
            b.ins().uextend(types::I128, z)
        };
        b.ins().icmp(IntCC::NotEqual, r, z128)
    };
    let guard_big = b.ins().band_imm_u(q_small, 1);
    let guard_big = b.ins().icmp_imm_s(IntCC::NotEqual, guard_big, 0);
    let sig_odd = {
        let o = b.ins().band_imm_u(sig, 1);
        b.ins().icmp_imm_s(IntCC::NotEqual, o, 0)
    };
    let round_big = {
        let sticky_or_odd = b.ins().bor(r_nz, sig_odd);
        b.ins().band(guard_big, sticky_or_odd)
    };
    let round_small = {
        let two_r = b.ins().ishl_imm_u(r, 1);
        let gt = b.ins().icmp(IntCC::UnsignedGreaterThan, two_r, den);
        let eq = b.ins().icmp(IntCC::Equal, two_r, den);
        let tie = b.ins().band(eq, sig_odd);
        b.ins().bor(gt, tie)
    };
    let round_up = b.ins().select(is_big, round_big, round_small);
    let inc = b.ins().uextend(types::I64, round_up);
    let (sig_r, exp_carry) = round_significand(b, sig, inc);

    // Biased result exponent: ea - eb + (bias - 1) + is_big + carry.
    let ediff = b.ins().isub(exp_a, exp_b);
    let ediff = b.ins().iadd_imm_s(ediff, BIAS - 1);
    let big_adj = b.ins().uextend(types::I32, is_big);
    let ediff = b.ins().iadd(ediff, big_adj);
    let exp = b.ins().iadd(ediff, exp_carry);

    let nse = compose_se(b, sign, exp);
    let (nm, nse) = clamp_over_under(b, sign, exp, sig_r, nse);

    // Special cases (division-specific operand combinations).
    let (zm, zse) = signed_zero(b, sign);
    let (im, ise) = inf(b, sign);
    let (qm, qse) = qnan(b);
    let mut m = nm;
    let mut se = nse;
    // finite / inf = 0
    (m, se) = (b.ins().select(b_inf, zm, m), b.ins().select(b_inf, zse, se));
    // 0 / finite = 0
    (m, se) = (
        b.ins().select(a_zero, zm, m),
        b.ins().select(a_zero, zse, se),
    );
    // finite / 0 = inf
    (m, se) = (
        b.ins().select(b_zero, im, m),
        b.ins().select(b_zero, ise, se),
    );
    // inf / finite = inf
    (m, se) = (b.ins().select(a_inf, im, m), b.ins().select(a_inf, ise, se));
    // inf / inf = NaN
    let inf_inf = b.ins().band(a_inf, b_inf);
    (m, se) = (
        b.ins().select(inf_inf, qm, m),
        b.ins().select(inf_inf, qse, se),
    );
    // 0 / 0 = NaN
    let zero_zero = b.ins().band(a_zero, b_zero);
    (m, se) = (
        b.ins().select(zero_zero, qm, m),
        b.ins().select(zero_zero, qse, se),
    );
    // NaN operand
    let any_nan = b.ins().bor(a_nan, b_nan);
    (m, se) = (
        b.ins().select(any_nan, qm, m),
        b.ins().select(any_nan, qse, se),
    );
    (m, se)
}

// --- add / subtract ------------------------------------------------------

/// Count leading zeros of a 128-bit value, returned as an `I64`.
fn clz128(b: &mut FunctionBuilder, x: Value) -> Value {
    let hi = {
        let h = b.ins().ushr_imm_u(x, 64);
        b.ins().ireduce(types::I64, h)
    };
    let lo = b.ins().ireduce(types::I64, x);
    let hi_nz = b.ins().icmp_imm_s(IntCC::NotEqual, hi, 0);
    let clz_hi = b.ins().clz(hi);
    let clz_lo = {
        let c = b.ins().clz(lo);
        b.ins().iadd_imm_s(c, 64)
    };
    b.ins().select(hi_nz, clz_hi, clz_lo)
}

/// Emit `a + b` for two raw 80-bit values.
pub fn emit_f80_add(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    bm: Value,
    bse: Value,
) -> (Value, Value) {
    let sign_a = sign_of(b, ase);
    let sign_b = sign_of(b, bse);
    let exp_a = exp_of(b, ase);
    let exp_b = exp_of(b, bse);

    let a_nan = is_nan(b, am, ase);
    let b_nan = is_nan(b, bm, bse);
    let a_inf = is_inf(b, am, ase);
    let b_inf = is_inf(b, bm, bse);

    // Order by exponent: `hi` gets the larger-or-equal exponent.
    let a_ge = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, exp_a, exp_b);
    let hi_m = b.ins().select(a_ge, am, bm);
    let lo_m = b.ins().select(a_ge, bm, am);
    let hi_sign = b.ins().select(a_ge, sign_a, sign_b);
    let lo_sign = b.ins().select(a_ge, sign_b, sign_a);
    let exp_hi = b.ins().select(a_ge, exp_a, exp_b);
    let exp_lo = b.ins().select(a_ge, exp_b, exp_a);
    let d = b.ins().isub(exp_hi, exp_lo); // >= 0

    // Mantissas in the high 64 bits of a 128-bit word (64 guard bits below).
    let hi128 = {
        let h = b.ins().uextend(types::I128, hi_m);
        b.ins().ishl_imm_u(h, 64)
    };
    let lo128 = {
        let l = b.ins().uextend(types::I128, lo_m);
        b.ins().ishl_imm_u(l, 64)
    };

    let one64 = b.ins().iconst(types::I64, 1);
    let one128 = b.ins().uextend(types::I128, one64);
    let bit127 = b.ins().ishl_imm_u(one128, 127);
    let mask64 = {
        let s = b.ins().ishl_imm_u(one128, 64);
        b.ins().iadd_imm_s(s, -1)
    };

    // Align `lo` right by `d` (capped at 127), accumulating a sticky bit.
    let big_shift = b.ins().icmp_imm_s(IntCC::SignedGreaterThan, d, 127);
    let d_cap = {
        let c127 = b.ins().iconst(types::I32, 127);
        b.ins().select(big_shift, c127, d)
    };
    let d_cap128 = b.ins().uextend(types::I128, d_cap);
    let lo_shifted = b.ins().ushr(lo128, d_cap128);
    let lost_mask = {
        let s = b.ins().ishl(one128, d_cap128);
        b.ins().iadd_imm_s(s, -1)
    };
    let lost = b.ins().band(lo128, lost_mask);
    let lost_nz = b.ins().icmp_imm_s(IntCC::NotEqual, lost, 0);
    let lo_all_nz = b.ins().icmp_imm_s(IntCC::NotEqual, lo128, 0);
    let sticky = b.ins().select(big_shift, lo_all_nz, lost_nz);
    let zero128 = {
        let z = b.ins().iconst(types::I64, 0);
        b.ins().uextend(types::I128, z)
    };
    let lo_aligned = b.ins().select(big_shift, zero128, lo_shifted);
    let sticky128 = b.ins().uextend(types::I128, sticky);
    let lo_aligned = b.ins().bor(lo_aligned, sticky128);

    let same_sign = b.ins().icmp(IntCC::Equal, sign_a, sign_b);

    // ---- effective addition ----
    let sum = b.ins().iadd(hi128, lo_aligned);
    let carry = b.ins().icmp(IntCC::UnsignedLessThan, sum, hi128);
    let sum_shr1 = b.ins().ushr_imm_u(sum, 1);
    let sum_lost = b.ins().band(sum, one128);
    let sum_carry = {
        let s = b.ins().bor(sum_shr1, bit127);
        b.ins().bor(s, sum_lost)
    };
    let add_res = b.ins().select(carry, sum_carry, sum);
    let add_exp = {
        let c = b.ins().uextend(types::I32, carry);
        b.ins().iadd(exp_hi, c)
    };

    // ---- effective subtraction ----
    // Select the (non-negative) difference directly rather than max/min of the
    // operands: `select(uge(a,b), a, b)` would be folded by the mid-end into
    // `umax.i128`, which the x86 backend cannot lower.
    let hi_ge = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, hi128, lo_aligned);
    let d_hilo = b.ins().isub(hi128, lo_aligned);
    let d_lohi = b.ins().isub(lo_aligned, hi128);
    let diff = b.ins().select(hi_ge, d_hilo, d_lohi);
    let sub_sign = b.ins().select(hi_ge, hi_sign, lo_sign);
    let diff_zero = b.ins().icmp(IntCC::Equal, diff, zero128);
    let cz = clz128(b, diff);
    let cz128 = b.ins().uextend(types::I128, cz);
    let sub_res = b.ins().ishl(diff, cz128);
    let sub_exp = {
        let cz32 = b.ins().ireduce(types::I32, cz);
        b.ins().isub(exp_hi, cz32)
    };

    // Pick add vs sub.
    let res128 = b.ins().select(same_sign, add_res, sub_res);
    let res_exp = b.ins().select(same_sign, add_exp, sub_exp);
    let res_sign = b.ins().select(same_sign, hi_sign, sub_sign);

    // Round at the bit-64 boundary (round-to-nearest-even).
    let keep = {
        let k = b.ins().ushr_imm_u(res128, 64);
        b.ins().ireduce(types::I64, k)
    };
    let round_bits = {
        let r = b.ins().band(res128, mask64);
        b.ins().ireduce(types::I64, r)
    };
    let half = b.ins().iconst(types::I64, INT_BIT);
    let gt = b.ins().icmp(IntCC::UnsignedGreaterThan, round_bits, half);
    let eq = b.ins().icmp(IntCC::Equal, round_bits, half);
    let odd = {
        let o = b.ins().band_imm_u(keep, 1);
        b.ins().icmp_imm_s(IntCC::NotEqual, o, 0)
    };
    let tie = b.ins().band(eq, odd);
    let round_up = b.ins().bor(gt, tie);
    let inc = b.ins().uextend(types::I64, round_up);
    let (sig_r, exp_carry) = round_significand(b, keep, inc);
    let final_exp = b.ins().iadd(res_exp, exp_carry);

    let nse = compose_se(b, res_sign, final_exp);
    let (nm, nse) = clamp_over_under(b, res_sign, final_exp, sig_r, nse);

    // Exact cancellation (effective subtraction to zero) -> +0.
    let not_same = b.ins().bxor_imm_u(same_sign, 1);
    let cancel = b.ins().band(not_same, diff_zero);
    let zero_m = b.ins().iconst(types::I64, 0);
    let zero_se = b.ins().iconst(types::I32, 0);
    let nm = b.ins().select(cancel, zero_m, nm);
    let nse = b.ins().select(cancel, zero_se, nse);

    // Specials: infinities and NaNs (zeros fall through the normal path
    // correctly - a zero operand has mantissa 0, so it aligns to nothing).
    let both_inf = b.ins().band(a_inf, b_inf);
    let signs_differ = b.ins().icmp(IntCC::NotEqual, sign_a, sign_b);
    let inf_nan = b.ins().band(both_inf, signs_differ);
    let any_inf = b.ins().bor(a_inf, b_inf);
    let inf_sign = b.ins().select(a_inf, sign_a, sign_b);
    let (im, ise) = inf(b, inf_sign);
    let (qm, qse) = qnan(b);
    let any_nan = b.ins().bor(a_nan, b_nan);

    let m = b.ins().select(any_inf, im, nm);
    let se = b.ins().select(any_inf, ise, nse);
    let m = b.ins().select(inf_nan, qm, m);
    let se = b.ins().select(inf_nan, qse, se);
    let m = b.ins().select(any_nan, qm, m);
    let se = b.ins().select(any_nan, qse, se);
    (m, se)
}

/// Emit `a - b` (implemented as `a + (-b)`).
pub fn emit_f80_sub(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    bm: Value,
    bse: Value,
) -> (Value, Value) {
    let neg_bse = b.ins().bxor_imm_u(bse, 0x8000);
    emit_f80_add(b, am, ase, bm, neg_bse)
}

/// Integer square root of a 128-bit value, returning `(floor_sqrt, remainder)`
/// as `I128` (`remainder = n - floor_sqrt^2`). Classic bit-by-bit restoring
/// algorithm; the result fits in 64 bits for any `n < 2^128`.
fn isqrt128(b: &mut FunctionBuilder, n0: Value) -> (Value, Value) {
    let i128 = types::I128;
    let zero = {
        let z = b.ins().iconst(types::I64, 0);
        b.ins().uextend(i128, z)
    };
    let one = {
        let o = b.ins().iconst(types::I64, 1);
        b.ins().uextend(i128, o)
    };
    let bit0 = b.ins().ishl_imm_u(one, 126); // 4^63, the largest power of four < 2^128

    let loop_b = b.create_block();
    b.append_block_param(loop_b, i128); // n
    b.append_block_param(loop_b, i128); // res
    b.append_block_param(loop_b, i128); // bit
    let body = b.create_block();
    b.append_block_param(body, i128);
    b.append_block_param(body, i128);
    b.append_block_param(body, i128);
    let exit = b.create_block();
    b.append_block_param(exit, i128); // remainder (n)
    b.append_block_param(exit, i128); // res

    b.ins().jump(loop_b, &[n0.into(), zero.into(), bit0.into()]);

    b.switch_to_block(loop_b);
    let n = b.block_params(loop_b)[0];
    let res = b.block_params(loop_b)[1];
    let bit = b.block_params(loop_b)[2];
    let cont = b.ins().icmp(IntCC::NotEqual, bit, zero);
    b.ins().brif(
        cont,
        body,
        &[n.into(), res.into(), bit.into()],
        exit,
        &[n.into(), res.into()],
    );

    b.switch_to_block(body);
    let n = b.block_params(body)[0];
    let res = b.block_params(body)[1];
    let bit = b.block_params(body)[2];
    let res_plus_bit = b.ins().iadd(res, bit);
    let ge = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, n, res_plus_bit);
    let n_sub = b.ins().isub(n, res_plus_bit);
    let n_new = b.ins().select(ge, n_sub, n);
    let res_shr = b.ins().ushr_imm_u(res, 1);
    let res_add = b.ins().iadd(res_shr, bit);
    let res_new = b.ins().select(ge, res_add, res_shr);
    let bit_new = b.ins().ushr_imm_u(bit, 2);
    b.ins()
        .jump(loop_b, &[n_new.into(), res_new.into(), bit_new.into()]);

    b.switch_to_block(exit);
    let rem = b.block_params(exit)[0];
    let sqrt = b.block_params(exit)[1];
    b.seal_block(loop_b);
    b.seal_block(body);
    b.seal_block(exit);
    (sqrt, rem)
}

/// Emit `sqrt(a)` for a raw 80-bit value.
pub fn emit_f80_sqrt(b: &mut FunctionBuilder, am: Value, ase: Value) -> (Value, Value) {
    let sign = sign_of(b, ase);
    let ebiased = exp_of(b, ase);
    let a_nan = is_nan(b, am, ase);
    let a_inf = is_inf(b, am, ase);
    let a_zero = is_zero(b, am, ase);
    let is_neg = b.ins().icmp_imm_s(IntCC::Equal, sign, 1);

    // Radicand scaled into 128 bits. The shift makes the effective exponent even
    // so the result exponent is exact: 64 for even biased exponent, 63 for odd.
    let even = {
        let lsb = b.ins().band_imm_u(ebiased, 1);
        b.ins().icmp_imm_s(IntCC::Equal, lsb, 0)
    };
    let k = {
        let k64 = b.ins().iconst(types::I32, 64);
        let k63 = b.ins().iconst(types::I32, 63);
        b.ins().select(even, k64, k63)
    };
    let k128 = b.ins().uextend(types::I128, k);
    let rad = {
        let a128 = b.ins().uextend(types::I128, am);
        b.ins().ishl(a128, k128)
    };
    let (q128, rem) = isqrt128(b, rad);
    let q = b.ins().ireduce(types::I64, q128);

    // Round to nearest: sqrt is never exactly halfway, so round up iff the
    // remainder exceeds the truncated root.
    let round_up = b.ins().icmp(IntCC::UnsignedGreaterThan, rem, q128);
    let inc = b.ins().uextend(types::I64, round_up);
    let (sig_r, exp_carry) = round_significand(b, q, inc);

    // Result biased exponent: (ebiased - 16384)/2 + 16383 (even) or
    // (ebiased - 16383)/2 + 16383 (odd); `base` is even in both cases.
    let base = {
        let be = b.ins().iadd_imm_s(ebiased, -16384);
        let bo = b.ins().iadd_imm_s(ebiased, -16383);
        b.ins().select(even, be, bo)
    };
    let er = {
        let h = b.ins().sshr_imm_u(base, 1);
        b.ins().iadd_imm_s(h, 16383)
    };
    let exp = b.ins().iadd(er, exp_carry);

    let nse = compose_se(b, sign, exp);
    let (nm, nse) = clamp_over_under(b, sign, exp, sig_r, nse);

    // Specials.
    let (qm, qse) = qnan(b);
    let (zm, zse) = signed_zero(b, sign);
    let pos = b.ins().iconst(types::I32, 0);
    let (im, ise) = inf(b, pos); // +inf
    let neg_nonzero = {
        let nz = b.ins().bxor_imm_u(a_zero, 1);
        b.ins().band(is_neg, nz)
    };

    let m = b.ins().select(a_zero, zm, nm);
    let se = b.ins().select(a_zero, zse, nse);
    let m = b.ins().select(a_inf, im, m);
    let se = b.ins().select(a_inf, ise, se);
    let m = b.ins().select(neg_nonzero, qm, m);
    let se = b.ins().select(neg_nonzero, qse, se);
    let m = b.ins().select(a_nan, qm, m);
    let se = b.ins().select(a_nan, qse, se);
    (m, se)
}

/// Emit an ordered compare of two raw 80-bit values, returning the three
/// predicates `(unordered, equal, less_than)` as I8 booleans (the interface the
/// `FCOM`/`FUCOMI` handlers consume). When unordered, `equal`/`less_than` are
/// unspecified; callers OR `unordered` into the flags.
pub fn emit_f80_cmp(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    bm: Value,
    bse: Value,
) -> (Value, Value, Value) {
    let a_nan = is_nan(b, am, ase);
    let b_nan = is_nan(b, bm, bse);
    let unord = b.ins().bor(a_nan, b_nan);

    let a_zero = is_zero(b, am, ase);
    let b_zero = is_zero(b, bm, bse);
    let sa = sign_of(b, ase);
    let sb = sign_of(b, bse);
    // Treat both signed zeros as +0 so -0 == +0.
    let zero_i32 = b.ins().iconst(types::I32, 0);
    let sa_eff = b.ins().select(a_zero, zero_i32, sa);
    let sb_eff = b.ins().select(b_zero, zero_i32, sb);

    // Magnitude key: (exp << 64) | mant. Higher exponent dominates; equal
    // exponents fall back to the significand. Zeros carry exp=0, mant=0.
    let exp_a = exp_of(b, ase);
    let exp_b = exp_of(b, bse);
    let mag = |b: &mut FunctionBuilder, exp: Value, mant: Value| -> Value {
        let e128 = b.ins().uextend(types::I128, exp);
        let e_hi = b.ins().ishl_imm_u(e128, 64);
        let m128 = b.ins().uextend(types::I128, mant);
        b.ins().bor(e_hi, m128)
    };
    let mag_a = mag(b, exp_a, am);
    let mag_b = mag(b, exp_b, bm);

    let signs_diff = b.ins().icmp(IntCC::NotEqual, sa_eff, sb_eff);
    let mag_lt = b.ins().icmp(IntCC::UnsignedLessThan, mag_a, mag_b);
    let mag_gt = b.ins().icmp(IntCC::UnsignedGreaterThan, mag_a, mag_b);
    let mag_eq = b.ins().icmp(IntCC::Equal, mag_a, mag_b);

    let a_is_neg = b.ins().icmp_imm_s(IntCC::Equal, sa_eff, 1);
    // Same sign: positive -> lt by magnitude; negative -> lt by reversed magnitude.
    let same_sign_lt = b.ins().select(a_is_neg, mag_gt, mag_lt);
    // Different signs: a < b iff a is the negative one.
    let lt = b.ins().select(signs_diff, a_is_neg, same_sign_lt);
    let not_signs_diff = b.ins().bxor_imm_u(signs_diff, 1);
    let eq = b.ins().band(not_signs_diff, mag_eq);

    (unord, eq, lt)
}

// --- outlinable kernels --------------------------------------------------

/// One 80-bit softfloat kernel, identified so it can be emitted **once** into a
/// shared helper function and `call`ed from every x87 site that needs it.
///
/// The helper body is exactly the corresponding `emit_f80_*` above, wrapped in
/// a function whose parameters and results are the same raw `(mant: i64,
/// se: i32)` pairs those builders already take and return - so inline and
/// outlined emission are bit-identical by construction, not by parallel
/// maintenance.
///
/// Note `Sub` is its own kernel rather than a call into `Add` with a flipped
/// sign: the sign flip is one `bxor`, so a separate `Subr` call site would pay
/// a second call for nothing. `FSUBR`/`FDIVR` reuse the `Sub`/`Div` helpers
/// with their arguments swapped at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum F80Kernel {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    /// Ordered compare, returning the `(unordered, equal, less_than)` triple.
    Cmp,
}

/// Every kernel, for callers that want to pre-emit or enumerate them.
pub const F80_KERNELS: [F80Kernel; 6] = [
    F80Kernel::Add,
    F80Kernel::Sub,
    F80Kernel::Mul,
    F80Kernel::Div,
    F80Kernel::Sqrt,
    F80Kernel::Cmp,
];

impl F80Kernel {
    /// The symbol the helper function is declared under. Kept stable and
    /// namespaced: AOT output containing x87 code carries these as local
    /// symbols, and two lifted regions linked together must share one copy.
    pub fn symbol(self) -> &'static str {
        match self {
            F80Kernel::Add => "__microlift_f80_add",
            F80Kernel::Sub => "__microlift_f80_sub",
            F80Kernel::Mul => "__microlift_f80_mul",
            F80Kernel::Div => "__microlift_f80_div",
            F80Kernel::Sqrt => "__microlift_f80_sqrt",
            F80Kernel::Cmp => "__microlift_f80_cmp",
        }
    }

    /// Number of 80-bit operands the kernel takes (each is one `(mant, se)`
    /// parameter pair).
    fn operand_count(self) -> usize {
        match self {
            F80Kernel::Sqrt => 1,
            _ => 2,
        }
    }

    /// The helper's signature: one `(I64 mant, I32 se)` pair per operand, and
    /// either an 80-bit result pair or - for [`F80Kernel::Cmp`] - the three
    /// `I8` predicates.
    ///
    /// `call_conv` should match the calling function's, so a call needs no
    /// convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        for _ in 0..self.operand_count() {
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I32));
        }
        match self {
            F80Kernel::Cmp => {
                for _ in 0..3 {
                    sig.returns.push(AbiParam::new(types::I8));
                }
            }
            _ => {
                sig.returns.push(AbiParam::new(types::I64));
                sig.returns.push(AbiParam::new(types::I32));
            }
        }
        sig
    }

    /// Emit the kernel's body into `b`, whose function must already have been
    /// given [`signature`](Self::signature). Creates the entry block, runs the
    /// matching `emit_f80_*` builder over the parameters, and returns its
    /// results.
    ///
    /// The `emit_f80_*` builders create and seal their own internal blocks (the
    /// division and square-root loops), and leave the builder positioned in the
    /// block holding the result, which is where the `return` lands.
    pub fn build_body(self, b: &mut FunctionBuilder) {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let p: Vec<Value> = b.block_params(entry).to_vec();

        match self {
            F80Kernel::Cmp => {
                let (unord, eq, lt) = emit_f80_cmp(b, p[0], p[1], p[2], p[3]);
                b.ins().return_(&[unord, eq, lt]);
            }
            F80Kernel::Sqrt => {
                let (m, se) = emit_f80_sqrt(b, p[0], p[1]);
                b.ins().return_(&[m, se]);
            }
            _ => {
                let (m, se) = match self {
                    F80Kernel::Add => emit_f80_add(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Sub => emit_f80_sub(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Mul => emit_f80_mul(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Div => emit_f80_div(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Sqrt | F80Kernel::Cmp => unreachable!("handled above"),
                };
                b.ins().return_(&[m, se]);
            }
        }
    }
}
