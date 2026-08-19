//! Software x87 80-bit extended-precision floating point, as pure Cranelift IR.
//!
//! Each operation is a free `emit_*` function building into a given
//! [`FunctionBuilder`], taking and returning the raw 80-bit representation: a
//! `(mant: i64, se: i32)` pair, where `mant` is the significand (explicit integer
//! bit at 63) and `se` holds the sign at bit 15 and the 15-bit biased exponent
//! (bias 16383) below it.
//!
//! Free builders rather than `Lifter` methods so one body serves both emission
//! strategies: inline at each site, or emitted once into a helper every site
//! `call`s ([`F80Kernel`]). Outlining is the default - the kernels run to
//! hundreds of IR instructions each.
//!
//! Approximations: subnormal *results* flush to zero and subnormal *inputs* are
//! used as stored; NaN results are the canonical x87 QNaN indefinite rather than
//! propagating input payloads; and [`emit_f80_trig`] is correctly *reduced* but
//! not correctly *rounded* (see its own accuracy note). Everything else is
//! correct for `+ - * /` and the ordering compare, round-to-nearest-even.

use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;

const EXP_MAX: i64 = 0x7FFF;
const BIAS: i64 = 16383;
const INT_BIT: i64 = 0x8000_0000_0000_0000u64 as i64;
const FRAC_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFF;
/// x87 QNaN floating-point indefinite: sign 1, exp all-ones, mantissa `1.1000`.
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
/// overflows to `2^64`, reset to `0x80000` and report a `+1` exponent step.
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

/// `floor((2/pi) * 2^256)`, least-significant limb first.
///
/// Argument reduction is Payne-Hanek: the quadrant and the reduced argument both
/// come out of the bits of `|x| * 2/pi`, which needs `2/pi` to far more than 80
/// bits. 256 is sized for the whole domain - `|x| < 2^63` puts the worst-case
/// cancellation around 2^-70, leaving well over 64 good bits.
const TWO_OVER_PI_Q256: [u64; 4] = [
    0xFE51_63AB_DEBB_C561,
    0xDB62_9599_3C43_9041,
    0xFC27_57D1_F534_DDC0,
    0xA2F9_836E_4E44_1529,
];

/// `floor((pi/2) * 2^127)`, as `(high, low)` 64-bit halves.
const PI_OVER_2_Q127: (u64, u64) = (0xC90F_DAA2_2168_C234, 0xC4C6_628B_80DC_1CD1);

/// Taylor coefficients of `cos` in `u = r^2`: `(-1)^k / (2k)!`, as Q126
/// two's-complement `(high, low)` halves.
///
/// Twelve terms put the truncation error below 2^-87 over `|r| <= pi/4`, far
/// under the Q126 accumulator's own rounding noise.
const COS_COEFFS_Q126: [(u64, u64); TRIG_POLY_TERMS] = [
    (0x4000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0xE000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x02AA_AAAA_AAAA_AAAA, 0xAAAA_AAAA_AAAA_AAAB),
    (0xFFE9_3E93_E93E_93E9, 0x3E93_E93E_93E9_3E94),
    (0x0000_6806_8068_0680, 0x6806_8068_0680_6807),
    (0xFFFF_FED8_1B04_8876, 0x0A38_D10F_E92C_159A),
    (0x0000_0002_3DDB_1DFF, 0x1B12_F6A8_9B53_0F5A),
    (0xFFFF_FFFF_FCD8_D16A, 0xE7F0_6C5B_E8A4_1D74),
    (0x0000_0000_0003_5CFE, 0x7CE6_7703_E23B_0CAD),
    (0xFFFF_FFFF_FFFF_FD2F, 0xB0F3_88D0_4D10_89FF),
    (0x0000_0000_0000_0001, 0xE542_BA40_2022_507B),
    (0xFFFF_FFFF_FFFF_FFFF, 0xFEF3_1C69_2480_7AD7),
];

/// Taylor coefficients of `sin(r)/r` in `u = r^2`: `(-1)^k / (2k+1)!`, Q126.
const SIN_COEFFS_Q126: [(u64, u64); TRIG_POLY_TERMS] = [
    (0x4000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0xF555_5555_5555_5555, 0x5555_5555_5555_5555),
    (0x0088_8888_8888_8888, 0x8888_8888_8888_8889),
    (0xFFFC_BFCB_FCBF_CBFC, 0xBFCB_FCBF_CBFC_BFCC),
    (0x0000_0B8E_F1D2_AB63, 0x99C7_D560_E447_2801),
    (0xFFFF_FFE5_19BA_980A, 0xBB1C_7018_B81B_47C8),
    (0x0000_0000_2C24_8C27, 0x50DA_12F9_4706_63A4),
    (0xFFFF_FFFF_FFCA_3018, 0x3198_8FC1_DC4F_352A),
    (0x0000_0000_0000_32A5, 0x8EE0_615A_94D6_4C0A),
    (0xFFFF_FFFF_FFFF_FFDA, 0x16C9_72FD_7D51_B66C),
    (0x0000_0000_0000_0000, 0x171B_8EF6_DCF5_718C),
    (0xFFFF_FFFF_FFFF_FFFF, 0xFFF4_4F25_F674_E3F3),
];

const TRIG_POLY_TERMS: usize = 12;

/// Biased exponent of `0.5` - at or above it an argument can exceed `pi/4` and
/// needs reducing; below it, it *is* its own reduced argument.
const REDUCE_MIN_EXP: i64 = 16382;

/// Materialise a 128-bit constant from its two 64-bit halves (Cranelift's
/// `iconst` tops out at `I64`).
fn const_i128(b: &mut FunctionBuilder, hi: u64, lo: u64) -> Value {
    let h = b.ins().iconst(types::I64, hi as i64);
    let l = b.ins().iconst(types::I64, lo as i64);
    join128(b, h, l)
}

/// Join two `I64` halves into an `I128`.
fn join128(b: &mut FunctionBuilder, hi: Value, lo: Value) -> Value {
    let h = b.ins().uextend(types::I128, hi);
    let h = b.ins().ishl_imm_u(h, 64);
    let l = b.ins().uextend(types::I128, lo);
    b.ins().bor(h, l)
}

fn lo64(b: &mut FunctionBuilder, v: Value) -> Value {
    b.ins().ireduce(types::I64, v)
}

fn hi64(b: &mut FunctionBuilder, v: Value) -> Value {
    let s = b.ins().ushr_imm_u(v, 64);
    b.ins().ireduce(types::I64, s)
}

fn zero128(b: &mut FunctionBuilder) -> Value {
    let z = b.ins().iconst(types::I64, 0);
    b.ins().uextend(types::I128, z)
}

/// Widen an `I32`/`I64` shift amount to `I128`, the width the value being
/// shifted has.
fn shift_amt(b: &mut FunctionBuilder, v: Value) -> Value {
    b.ins().uextend(types::I128, v)
}

/// High 128 bits of the unsigned 256-bit product `x * y`.
///
/// Cranelift's `imul` on `I128` keeps only the low half and there is no
/// `umulhi.i128`, so this is schoolbook over the four 64-bit limbs - each
/// partial product is an exact `I128` multiply of two zero-extended limbs.
fn umulhi128(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    let x0 = lo64(b, x);
    let x1 = hi64(b, x);
    let y0 = lo64(b, y);
    let y1 = hi64(b, y);
    let mul = |b: &mut FunctionBuilder, a: Value, c: Value| {
        let a = b.ins().uextend(types::I128, a);
        let c = b.ins().uextend(types::I128, c);
        b.ins().imul(a, c)
    };
    let p00 = mul(b, x0, y0);
    let p01 = mul(b, x0, y1);
    let p10 = mul(b, x1, y0);
    let p11 = mul(b, x1, y1);

    let mask = const_i128(b, 0, u64::MAX);
    let mid = {
        let c00 = b.ins().ushr_imm_u(p00, 64);
        let l01 = b.ins().band(p01, mask);
        let l10 = b.ins().band(p10, mask);
        let s = b.ins().iadd(c00, l01);
        b.ins().iadd(s, l10)
    };
    let h01 = b.ins().ushr_imm_u(p01, 64);
    let h10 = b.ins().ushr_imm_u(p10, 64);
    let hmid = b.ins().ushr_imm_u(mid, 64);
    let s = b.ins().iadd(p11, h01);
    let s = b.ins().iadd(s, h10);
    b.ins().iadd(s, hmid)
}

/// High 128 bits of `a * y` where `a` is **signed** and `y` unsigned - the
/// usual correction on top of the unsigned product.
fn mulhi_su(b: &mut FunctionBuilder, a: Value, y: Value) -> Value {
    let hi = umulhi128(b, a, y);
    let z = zero128(b);
    let neg = b.ins().icmp(IntCC::SignedLessThan, a, z);
    let corr = b.ins().select(neg, y, z);
    b.ins().isub(hi, corr)
}

/// Reduce `|x|` (as its raw significand and biased exponent) modulo `pi/2`.
///
/// Returns `(n, r_sig, r_exp, r_sign)`: the nearest quadrant index
/// `n = round(|x| / (pi/2))`, and the remainder `r = |x| - n*(pi/2)` with
/// `|r| <= pi/4`, as a normalised significand, biased exponent and sign bit.
///
/// The caller must have established that `x` is finite and `|x| < 2^63`, which
/// is exactly what makes the fixed 256-bit `2/pi` window enough.
fn emit_trig_reduce(b: &mut FunctionBuilder, m: Value, e: Value) -> (Value, Value, Value, Value) {
    let i128 = types::I128;
    let i32t = types::I32;

    // `|x| < 1/2` is already its own remainder, in quadrant 0. Both paths are
    // computed and selected between; the shifts below are clamped so the
    // discarded one cannot produce an out-of-range shift amount.
    let need = b
        .ins()
        .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, e, REDUCE_MIN_EXP);

    // `x = m * 2^E` with `E = e - 16446`; the reduction path has `E` in
    // [-64, -1], so `t = -E` lies in [1, 64].
    let t = {
        let base = b.ins().iconst(i32t, 16446);
        let t = b.ins().isub(base, e);
        let one = b.ins().iconst(i32t, 1);
        let too_small = b.ins().icmp_imm_s(IntCC::SignedLessThan, t, 1);
        let t = b.ins().select(too_small, one, t);
        let c64 = b.ins().iconst(i32t, 64);
        let too_big = b.ins().icmp_imm_s(IntCC::SignedGreaterThan, t, 64);
        b.ins().select(too_big, c64, t)
    };

    // D = floor((2/pi) * 2^(256+E)) = TWO_OVER_PI_Q256 >> t, as four limbs.
    let a_lo = const_i128(b, TWO_OVER_PI_Q256[1], TWO_OVER_PI_Q256[0]);
    let a_hi = const_i128(b, TWO_OVER_PI_Q256[3], TWO_OVER_PI_Q256[2]);
    let t128 = shift_amt(b, t);
    let inv_t = {
        let c128 = b.ins().iconst(i32t, 128);
        let d = b.ins().isub(c128, t);
        shift_amt(b, d)
    };
    let d_lo = {
        let down = b.ins().ushr(a_lo, t128);
        let up = b.ins().ishl(a_hi, inv_t);
        b.ins().bor(down, up)
    };
    let d_hi = b.ins().ushr(a_hi, t128);
    let d_limbs = [lo64(b, d_lo), hi64(b, d_lo), lo64(b, d_hi), hi64(b, d_hi)];

    // U = m * D: a 320-bit product whose limbs 0..3 are the fractional part of
    // `|x| / (pi/2)` (in units of 2^-256) and whose limb 4 is its integer part.
    let m128 = b.ins().uextend(i128, m);
    let mut carry = b.ins().iconst(types::I64, 0);
    let mut frac = [carry; 4];
    for (i, &d) in d_limbs.iter().enumerate() {
        let d128 = b.ins().uextend(i128, d);
        let p = b.ins().imul(m128, d128);
        let c128 = b.ins().uextend(i128, carry);
        let p = b.ins().iadd(p, c128);
        frac[i] = lo64(b, p);
        carry = hi64(b, p);
    }
    let int_part = carry;

    let frac_lo = join128(b, frac[1], frac[0]);
    let frac_hi = join128(b, frac[3], frac[2]);

    // Round the quotient to nearest: a fraction at or above 1/2 belongs to the
    // next quadrant, with a negative remainder.
    let half = {
        let s = b.ins().ushr_imm_u(frac_hi, 127);
        b.ins().ireduce(types::I64, s)
    };
    let half_b = b.ins().icmp_imm_s(IntCC::NotEqual, half, 0);
    let n = b.ins().iadd(int_part, half);

    // |frac - n| as a 256-bit fraction: either `frac` or its two's complement.
    let z = zero128(b);
    let neg_lo = b.ins().isub(z, frac_lo);
    let neg_hi = {
        let borrow = {
            let nz = b.ins().icmp(IntCC::NotEqual, frac_lo, z);
            b.ins().uextend(i128, nz)
        };
        let h = b.ins().isub(z, frac_hi);
        b.ins().isub(h, borrow)
    };
    let mag_lo = b.ins().select(half_b, neg_lo, frac_lo);
    let mag_hi = b.ins().select(half_b, neg_hi, frac_hi);

    // Normalise the magnitude to its top 128 bits, `M`, with `c` leading zeros
    // over the full 256: `|frac - n| = (M / 2^128) * 2^-c`.
    let hi_nz = b.ins().icmp(IntCC::NotEqual, mag_hi, z);
    let c = {
        let ch = clz128(b, mag_hi);
        let cl = {
            let c = clz128(b, mag_lo);
            b.ins().iadd_imm_s(c, 128)
        };
        b.ins().select(hi_nz, ch, cl)
    };
    let c32 = b.ins().ireduce(i32t, c);
    let c128 = shift_amt(b, c);
    let m_from_hi = {
        let up = b.ins().ishl(mag_hi, c128);
        let rest = {
            let c128v = b.ins().iconst(i32t, 128);
            let d = b.ins().isub(c128v, c32);
            let d = shift_amt(b, d);
            let r = b.ins().ushr(mag_lo, d);
            // `c == 0` would make that a 128-bit shift, which Cranelift masks
            // back to zero; there is nothing to carry in then anyway.
            let c_zero = b.ins().icmp_imm_s(IntCC::Equal, c, 0);
            b.ins().select(c_zero, z, r)
        };
        b.ins().bor(up, rest)
    };
    let m_from_lo = {
        let d = b.ins().iadd_imm_s(c32, -128);
        let d = shift_amt(b, d);
        b.ins().ishl(mag_lo, d)
    };
    let mnorm = b.ins().select(hi_nz, m_from_hi, m_from_lo);

    // |r| = (M / 2^128) * 2^-c * (pi/2); the product lands in [0.785, 1.571)
    // as a Q127 value.
    let pi2 = const_i128(b, PI_OVER_2_Q127.0, PI_OVER_2_Q127.1);
    let w = umulhi128(b, mnorm, pi2);
    let top_set = {
        let s = b.ins().ushr_imm_u(w, 127);
        let s = b.ins().ireduce(types::I64, s);
        b.ins().icmp_imm_s(IntCC::NotEqual, s, 0)
    };
    let k = {
        let k64 = b.ins().iconst(i32t, 64);
        let k63 = b.ins().iconst(i32t, 63);
        b.ins().select(top_set, k64, k63)
    };
    let sig = {
        let sh = shift_amt(b, k);
        let s = b.ins().ushr(w, sh);
        b.ins().ireduce(types::I64, s)
    };
    let round_bit = {
        let km1 = b.ins().iadd_imm_s(k, -1);
        let sh = shift_amt(b, km1);
        let s = b.ins().ushr(w, sh);
        let s = b.ins().ireduce(types::I64, s);
        b.ins().band_imm_u(s, 1)
    };
    let (sig_r, exp_carry) = round_significand(b, sig, round_bit);
    // `|r| = sig * 2^(k - 127 - c)`, i.e. biased exponent 16383 + k - 64 - c.
    let r_exp = {
        let x = b.ins().iadd_imm_s(k, 16383 - 64);
        let x = b.ins().isub(x, c32);
        b.ins().iadd(x, exp_carry)
    };

    // An argument that is an exact multiple of pi/2 reduces to zero, which has
    // no leading bit to normalise against.
    let mag_zero = {
        let l = b.ins().icmp(IntCC::Equal, mag_lo, z);
        let h = b.ins().icmp(IntCC::Equal, mag_hi, z);
        b.ins().band(l, h)
    };
    let zero64 = b.ins().iconst(types::I64, 0);
    let zero32 = b.ins().iconst(i32t, 0);
    let sig_r = b.ins().select(mag_zero, zero64, sig_r);
    let r_exp = b.ins().select(mag_zero, zero32, r_exp);

    let n = b.ins().select(need, n, zero64);
    let r_sig = b.ins().select(need, sig_r, m);
    let r_exp = b.ins().select(need, r_exp, e);
    let r_sign = {
        let no = b.ins().iconst(types::I8, 0);
        b.ins().select(need, half_b, no)
    };
    (n, r_sig, r_exp, r_sign)
}

/// [`emit_f80_trig`]'s `phase` for `cos`: the quadrant is used as reduced.
pub const TRIG_PHASE_COS: i64 = 0;
/// [`emit_f80_trig`]'s `phase` for `sin`: `sin(x) == cos(x - pi/2)`, one
/// quadrant back, plus the odd-function sign of the argument.
pub const TRIG_PHASE_SIN: i64 = 1;

/// Emit `cos(a)` or `sin(a)` for a raw 80-bit value, selected by `phase`
/// ([`TRIG_PHASE_COS`]/[`TRIG_PHASE_SIN`], an `I32`).
///
/// `phase` is a *runtime* value on purpose: the quadrant selection it feeds is
/// already a chain of `select`s, so one outlined helper serves `FSIN`, `FCOS`,
/// `FSINCOS` and `FPTAN`.
///
/// The caller owns the x87 out-of-range protocol; this assumes `|a| < 2^63` and
/// returns the QNaN indefinite for a NaN or infinite input.
///
/// Shape: Payne-Hanek reduction against a 256-bit `2/pi` gives the quadrant and
/// a remainder `|r| <= pi/4`; a twelve-term Horner evaluation in `u = r^2` gives
/// `cos(r)` or `sin(r)/r`, selected per coefficient by the quadrant so both
/// series share one evaluation. The polynomial runs in 128-bit fixed point
/// rather than through the 80-bit kernels: the series value is near 1, so
/// absolute precision there *is* relative precision, at a fraction of the IR.
/// The `sin` case multiplies by the *normalised* `r`, which is what preserves
/// relative accuracy for an argument landing almost exactly on a multiple of
/// `pi/2`.
///
/// Accuracy: faithful to the 64-bit significand. Against the 90-digit reference
/// tables in `fpu_tests::test_fcos_full_80_bit_precision` and its `fsin`
/// sibling, every case is correctly rounded but one, which is 1 ulp out - so the
/// tests bound the error at 1 ulp rather than asserting correct rounding, which
/// nothing here proves. Real x87 hardware reduces against a 66-bit `pi` and is
/// far *less* accurate for large arguments, so bit-exact agreement with it is
/// neither possible nor attempted.
pub fn emit_f80_trig(
    b: &mut FunctionBuilder,
    am: Value,
    ase: Value,
    phase: Value,
) -> (Value, Value) {
    let i128 = types::I128;
    let i32t = types::I32;

    let e = exp_of(b, ase);
    let a_nan = is_nan(b, am, ase);
    let a_inf = is_inf(b, am, ase);
    let a_sign = sign_of(b, ase);

    // Both functions are reduced against the same |x|: cos is even, and sin's
    // oddness is re-applied to the result at the end.
    let (n, r_sig, r_exp, r_sign) = emit_trig_reduce(b, am, e);

    // cos(n*pi/2 + r) is cos(r), -sin(r), -cos(r), sin(r) for n mod 4 = 0..3;
    // sin is the same table one quadrant back. `n - phase` is >= -1, so the
    // `n + 1` below stays non-negative and the logical shift is safe.
    let nq = {
        let p = b.ins().uextend(types::I64, phase);
        b.ins().isub(n, p)
    };
    let use_sin = {
        let o = b.ins().band_imm_u(nq, 1);
        b.ins().icmp_imm_s(IntCC::NotEqual, o, 0)
    };
    let base_sign = {
        let p = b.ins().iadd_imm_s(nq, 1);
        let s = b.ins().ushr_imm_u(p, 1);
        let s = b.ins().band_imm_u(s, 1);
        b.ins().ireduce(i32t, s)
    };

    // u = r^2 as Q127. `r = r_sig * 2^(r_exp - 16446)`, so the square needs a
    // right shift of `32765 - 2*r_exp`, which is at least 1 for any |r| < 1 and
    // saturates to "u is zero" for arguments too small to matter.
    let rsq = {
        let rs = b.ins().uextend(i128, r_sig);
        b.ins().imul(rs, rs)
    };
    let u = {
        let base = b.ins().iconst(i32t, 32765);
        let twice = b.ins().imul_imm_s(r_exp, 2);
        let sh = b.ins().isub(base, twice);
        let vanishes = b.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, sh, 128);
        let sh = shift_amt(b, sh);
        let u = b.ins().ushr(rsq, sh);
        let z = zero128(b);
        b.ins().select(vanishes, z, u)
    };

    // Horner over the Taylor series. Both coefficient sets have the same shape,
    // so selecting the coefficient per term costs twelve `select`s instead of a
    // second full evaluation.
    let coeff = |b: &mut FunctionBuilder, k: usize, use_sin: Value| {
        let c = const_i128(b, COS_COEFFS_Q126[k].0, COS_COEFFS_Q126[k].1);
        let s = const_i128(b, SIN_COEFFS_Q126[k].0, SIN_COEFFS_Q126[k].1);
        b.ins().select(use_sin, s, c)
    };
    let mut p = coeff(b, TRIG_POLY_TERMS - 1, use_sin);
    for k in (0..TRIG_POLY_TERMS - 1).rev() {
        // Q126 * Q127 keeps Q125 in the high half; one shift restores Q126.
        let t = mulhi_su(b, p, u);
        let t = b.ins().ishl_imm_u(t, 1);
        let c = coeff(b, k, use_sin);
        p = b.ins().iadd(t, c);
    }

    // The series value is positive and in [0.7, 1] either way; normalise it into
    // an 80-bit significand and exponent.
    let pc = clz128(b, p);
    let pn = {
        let sh = shift_amt(b, pc);
        b.ins().ishl(p, sh)
    };
    let p_sig = {
        let s = b.ins().ushr_imm_u(pn, 64);
        b.ins().ireduce(types::I64, s)
    };
    let p_round = {
        let s = b.ins().ushr_imm_u(pn, 63);
        let s = b.ins().ireduce(types::I64, s);
        b.ins().band_imm_u(s, 1)
    };
    let (p_sig, p_carry) = round_significand(b, p_sig, p_round);
    let p_exp = {
        let pc32 = b.ins().ireduce(i32t, pc);
        let base = b.ins().iconst(i32t, 16384);
        let x = b.ins().isub(base, pc32);
        b.ins().iadd(x, p_carry)
    };

    // sin(r) = r * (sin(r)/r); cos(r) is the series value itself.
    let (sm, sse) = emit_f80_mul(b, r_sig, r_exp, p_sig, p_exp);
    let vm = b.ins().select(use_sin, sm, p_sig);
    let vse = b.ins().select(use_sin, sse, p_exp);

    let sign = {
        let zero32 = b.ins().iconst(i32t, 0);
        let rs = b.ins().uextend(i32t, r_sign);
        let s = b.ins().select(use_sin, rs, zero32);
        let sign = b.ins().bxor(base_sign, s);
        // sin is odd, so it carries the argument's sign; cos is even and does
        // not. This is also what preserves `sin(-0) == -0`, which falls out of
        // the zero remainder rather than needing a special case.
        let is_sin_fn = b.ins().icmp_imm_s(IntCC::NotEqual, phase, 0);
        let arg = b.ins().select(is_sin_fn, a_sign, zero32);
        b.ins().bxor(sign, arg)
    };
    let se = {
        let mag = b.ins().band_imm_u(vse, EXP_MAX);
        compose_se(b, sign, mag)
    };

    // An infinite argument is an invalid operation, and a NaN one propagates;
    // both deliver the QNaN indefinite.
    let (qm, qse) = qnan(b);
    let bad = b.ins().bor(a_nan, a_inf);
    let m = b.ins().select(bad, qm, vm);
    let se = b.ins().select(bad, qse, se);
    (m, se)
}

// --- outlinable kernels --------------------------------------------------

/// One 80-bit softfloat kernel, identified so it can be emitted **once** into a
/// shared helper function and `call`ed from every x87 site that needs it.
///
/// The helper body is exactly the corresponding `emit_f80_*`, wrapped in a
/// function whose parameters and results are the same raw `(mant, se)` pairs -
/// so inline and outlined emission are bit-identical by construction.
///
/// `Sub` is its own kernel rather than a call into `Add` with a flipped sign:
/// the flip is one `bxor`, so routing through `Add` would pay a second call for
/// nothing. `FSUBR`/`FDIVR` reuse `Sub`/`Div` with swapped arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum F80Kernel {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    /// Sine or cosine, selected by a third `phase` argument, on the reduced
    /// domain `|x| < 2^63` (see [`emit_f80_trig`]).
    Trig,
    /// Ordered compare, returning the `(unordered, equal, less_than)` triple.
    Cmp,
}

/// Every kernel, for callers that want to pre-emit or enumerate them.
pub const F80_KERNELS: [F80Kernel; 7] = [
    F80Kernel::Add,
    F80Kernel::Sub,
    F80Kernel::Mul,
    F80Kernel::Div,
    F80Kernel::Sqrt,
    F80Kernel::Trig,
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
            F80Kernel::Trig => "__microlift_f80_trig",
            F80Kernel::Cmp => "__microlift_f80_cmp",
        }
    }

    /// Number of 80-bit operands the kernel takes (each is one `(mant, se)`
    /// parameter pair).
    fn operand_count(self) -> usize {
        match self {
            F80Kernel::Sqrt | F80Kernel::Trig => 1,
            _ => 2,
        }
    }

    /// The helper's signature: one `(I64 mant, I32 se)` pair per operand, and
    /// either an 80-bit result pair or - for [`F80Kernel::Cmp`] - three `I8`
    /// predicates. `call_conv` should match the calling function's.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        for _ in 0..self.operand_count() {
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I32));
        }
        if self == F80Kernel::Trig {
            // The quadrant phase selecting sin vs cos.
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

    /// Emit the kernel's body into `b`, whose function must already have been given
    /// [`signature`](Self::signature).
    ///
    /// The `emit_f80_*` builders create and seal their own internal blocks and leave
    /// the builder positioned in the block holding the result, which is where the
    /// `return` lands.
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
            F80Kernel::Trig => {
                let (m, se) = emit_f80_trig(b, p[0], p[1], p[2]);
                b.ins().return_(&[m, se]);
            }
            _ => {
                let (m, se) = match self {
                    F80Kernel::Add => emit_f80_add(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Sub => emit_f80_sub(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Mul => emit_f80_mul(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Div => emit_f80_div(b, p[0], p[1], p[2], p[3]),
                    F80Kernel::Sqrt | F80Kernel::Trig | F80Kernel::Cmp => {
                        unreachable!("handled above")
                    }
                };
                b.ins().return_(&[m, se]);
            }
        }
    }
}
