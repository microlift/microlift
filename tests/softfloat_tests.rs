//! Bit-exact verification of the software x87 80-bit kernels against LLVM's
//! APFloat (`rustc_apfloat::ieee::X87DoubleExtended`).
//!
//! Each test JIT-compiles a tiny wrapper that calls the kernel emitter and
//! compares its raw 80-bit output to APFloat operating on the same operands.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use microlift::arch::x86_64::semantics::softfloat;
use microlift::backend::CraneliftBackend;
use microlift::os::get_host_calling_conv;
use rustc_apfloat::ieee::X87DoubleExtended;
use rustc_apfloat::{Float, Round};

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// An 80-bit value as `(mantissa, sign_exp)`.
fn parts(bits: u128) -> (u64, u16) {
    (bits as u64, (bits >> 64) as u16)
}
fn bits(mant: u64, se: u16) -> u128 {
    (mant as u128) | ((se as u128) << 64)
}

/// JIT-compile `op` over the two operands and return the raw 80-bit result.
fn jit_binop(op: Op, a: u128, b: u128) -> u128 {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    // (a_mant, a_se, b_mant, b_se, out_ptr), all i64.
    for _ in 0..5 {
        func.signature.params.push(AbiParam::new(types::I64));
    }

    let mut fctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fctx);
    let blk = builder.create_block();
    builder.append_block_params_for_function_params(blk);
    builder.switch_to_block(blk);
    builder.seal_block(blk);
    let p = builder.block_params(blk).to_vec();
    let (am, a_se64, bm, b_se64, out) = (p[0], p[1], p[2], p[3], p[4]);
    let ase = builder.ins().ireduce(types::I32, a_se64);
    let bse = builder.ins().ireduce(types::I32, b_se64);
    let (rm, rse) = match op {
        Op::Add => softfloat::emit_f80_add(&mut builder, am, ase, bm, bse),
        Op::Sub => softfloat::emit_f80_sub(&mut builder, am, ase, bm, bse),
        Op::Mul => softfloat::emit_f80_mul(&mut builder, am, ase, bm, bse),
        Op::Div => softfloat::emit_f80_div(&mut builder, am, ase, bm, bse),
    };
    let rse64 = builder.ins().uextend(types::I64, rse);
    builder.ins().store(MemFlagsData::new(), rm, out, 0);
    builder.ins().store(MemFlagsData::new(), rse64, out, 8);
    builder.ins().return_(&[]);
    builder.finalize();

    let mut backend = CraneliftBackend::new().expect("backend");
    let code = backend.compile(func).expect("compile");
    let f: extern "C" fn(u64, u64, u64, u64, *mut u64) = unsafe { std::mem::transmute(code) };

    let (am_v, ase_v) = parts(a);
    let (bm_v, bse_v) = parts(b);
    let mut outbuf = [0u64; 2];
    f(am_v, ase_v as u64, bm_v, bse_v as u64, outbuf.as_mut_ptr());
    bits(outbuf[0], outbuf[1] as u16)
}

fn jit_cmp(a: u128, b: u128) -> (bool, bool, bool) {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    for _ in 0..5 {
        func.signature.params.push(AbiParam::new(types::I64));
    }
    let mut fctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fctx);
    let blk = builder.create_block();
    builder.append_block_params_for_function_params(blk);
    builder.switch_to_block(blk);
    builder.seal_block(blk);
    let p = builder.block_params(blk).to_vec();
    let (am, a_se64, bm, b_se64, out) = (p[0], p[1], p[2], p[3], p[4]);
    let ase = builder.ins().ireduce(types::I32, a_se64);
    let bse = builder.ins().ireduce(types::I32, b_se64);
    let (unord, eq, lt) = softfloat::emit_f80_cmp(&mut builder, am, ase, bm, bse);
    // Pack: bit2=unord, bit1=lt, bit0=eq.
    let unord = builder.ins().uextend(types::I64, unord);
    let lt = builder.ins().uextend(types::I64, lt);
    let eq = builder.ins().uextend(types::I64, eq);
    let unord = builder.ins().ishl_imm(unord, 2);
    let lt = builder.ins().ishl_imm(lt, 1);
    let code = builder.ins().bor(unord, lt);
    let code = builder.ins().bor(code, eq);
    builder.ins().store(MemFlagsData::new(), code, out, 0);
    builder.ins().return_(&[]);
    builder.finalize();

    let mut backend = CraneliftBackend::new().expect("backend");
    let cptr = backend.compile(func).expect("compile");
    let f: extern "C" fn(u64, u64, u64, u64, *mut u64) = unsafe { std::mem::transmute(cptr) };
    let (am_v, ase_v) = parts(a);
    let (bm_v, bse_v) = parts(b);
    let mut outbuf = [0u64; 1];
    f(am_v, ase_v as u64, bm_v, bse_v as u64, outbuf.as_mut_ptr());
    let c = outbuf[0];
    ((c & 4) != 0, (c & 1) != 0, (c & 2) != 0)
}

fn af(bits: u128) -> X87DoubleExtended {
    X87DoubleExtended::from_bits(bits)
}

/// JIT-compile `sqrt` over one operand and return the raw 80-bit result.
fn jit_sqrt(a: u128) -> u128 {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    for _ in 0..3 {
        func.signature.params.push(AbiParam::new(types::I64));
    }
    let mut fctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fctx);
    let blk = builder.create_block();
    builder.append_block_params_for_function_params(blk);
    builder.switch_to_block(blk);
    builder.seal_block(blk);
    let p = builder.block_params(blk).to_vec();
    let (am, a_se64, out) = (p[0], p[1], p[2]);
    let ase = builder.ins().ireduce(types::I32, a_se64);
    let (rm, rse) = softfloat::emit_f80_sqrt(&mut builder, am, ase);
    let rse64 = builder.ins().uextend(types::I64, rse);
    builder.ins().store(MemFlagsData::new(), rm, out, 0);
    builder.ins().store(MemFlagsData::new(), rse64, out, 8);
    builder.ins().return_(&[]);
    builder.finalize();
    let mut backend = CraneliftBackend::new().expect("backend");
    let cptr = backend.compile(func).expect("compile");
    let f: extern "C" fn(u64, u64, *mut u64) = unsafe { std::mem::transmute(cptr) };
    let (am_v, ase_v) = parts(a);
    let mut outbuf = [0u64; 2];
    f(am_v, ase_v as u64, outbuf.as_mut_ptr());
    bits(outbuf[0], outbuf[1] as u16)
}

/// Simple deterministic xorshift PRNG so tests are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// A random *normal* 80-bit value with a full 64-bit significand (integer bit
/// set) and a mid-range exponent, so results stay normal (no under/overflow,
/// which the kernels currently flush). The low significand bits exercise
/// precision beyond f64.
fn rand_normal(rng: &mut Rng) -> u128 {
    let mant = rng.next() | (1u64 << 63);
    let sign = (rng.next() & 1) as u16;
    // Exponent in [16200, 16560): comfortably normal, and products/quotients of
    // two such values stay within the normal range.
    let exp = 16200 + (rng.next() % 360) as u16;
    bits(mant, (sign << 15) | exp)
}

/// Regression: every kernel must lower under the optimizing mid-end. The
/// egraph pass (enabled at `opt_level=speed`, as the object/AOT backend uses)
/// once folded a 128-bit `select(uge(a,b), a, b)` into `umax.i128`, which the
/// x86 backend cannot lower. `opt_level=none` (the plain JIT tests) skips the
/// egraph and so did not catch it.
#[test]
fn test_kernels_lower_under_optimization() {
    use cranelift_codegen::settings::{self, Configurable};

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    for _ in 0..5 {
        func.signature.params.push(AbiParam::new(types::I64));
    }
    let mut fctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fctx);
    let blk = builder.create_block();
    builder.append_block_params_for_function_params(blk);
    builder.switch_to_block(blk);
    builder.seal_block(blk);
    let p = builder.block_params(blk).to_vec();
    let (am, a_se64, bm, b_se64, out) = (p[0], p[1], p[2], p[3], p[4]);
    let ase = builder.ins().ireduce(types::I32, a_se64);
    let bse = builder.ins().ireduce(types::I32, b_se64);

    // Exercise every kernel and combine outputs so none are dead-code eliminated.
    let mut acc_m = builder.ins().iconst(types::I64, 0);
    let mut acc_s = builder.ins().iconst(types::I32, 0);
    for (m, s) in [
        softfloat::emit_f80_add(&mut builder, am, ase, bm, bse),
        softfloat::emit_f80_sub(&mut builder, am, ase, bm, bse),
        softfloat::emit_f80_mul(&mut builder, am, ase, bm, bse),
        softfloat::emit_f80_div(&mut builder, am, ase, bm, bse),
        softfloat::emit_f80_sqrt(&mut builder, am, ase),
    ] {
        acc_m = builder.ins().bxor(acc_m, m);
        acc_s = builder.ins().bxor(acc_s, s);
    }
    let (unord, eq, lt) = softfloat::emit_f80_cmp(&mut builder, am, ase, bm, bse);
    let cmp = builder.ins().bor(unord, eq);
    let cmp = builder.ins().bor(cmp, lt);
    let cmp = builder.ins().uextend(types::I32, cmp);
    acc_s = builder.ins().bxor(acc_s, cmp);

    let acc_s64 = builder.ins().uextend(types::I64, acc_s);
    builder.ins().store(MemFlagsData::new(), acc_m, out, 0);
    builder.ins().store(MemFlagsData::new(), acc_s64, out, 8);
    builder.ins().return_(&[]);
    builder.finalize();

    let mut fb = settings::builder();
    fb.set("opt_level", "speed").unwrap();
    let flags = settings::Flags::new(fb);
    let mut backend = CraneliftBackend::with_flags(flags, &[]).expect("backend");
    backend
        .compile(func)
        .expect("all kernels must lower at opt_level=speed");
}

#[test]
fn test_add_matches_apfloat() {
    let mut rng = Rng(0xa11c_e50f_7734_2201);
    for _ in 0..3000 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        let got = jit_binop(Op::Add, a, b);
        let want = af(a).add_r(af(b), Round::NearestTiesToEven).value.to_bits()
            & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
        // Skip cases whose exact result underflows/overflows the normal range
        // (the kernel flushes; apfloat may produce subnormal/inf).
        if is_nan_bits(want) {
            continue;
        }
        assert_eq!(
            got, want,
            "add {a:#022x} + {b:#022x}: got {got:#022x} want {want:#022x}"
        );
    }
}

#[test]
fn test_sub_matches_apfloat() {
    let mut rng = Rng(0x5b7a_c700_bad0_f00d);
    for _ in 0..3000 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        let got = jit_binop(Op::Sub, a, b);
        let want = af(a).sub_r(af(b), Round::NearestTiesToEven).value.to_bits()
            & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
        if is_nan_bits(want) {
            continue;
        }
        assert_eq!(
            got, want,
            "sub {a:#022x} - {b:#022x}: got {got:#022x} want {want:#022x}"
        );
    }
}

#[test]
fn test_mul_matches_apfloat() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..2000 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        let got = jit_binop(Op::Mul, a, b);
        let want = af(a).mul_r(af(b), Round::NearestTiesToEven).value.to_bits()
            & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
        assert_eq!(
            got, want,
            "mul {a:#022x} * {b:#022x}: got {got:#022x} want {want:#022x}"
        );
    }
}

#[test]
fn test_div_matches_apfloat() {
    let mut rng = Rng(0xdead_beef_cafe_babe);
    for _ in 0..2000 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        let got = jit_binop(Op::Div, a, b);
        let want = af(a).div_r(af(b), Round::NearestTiesToEven).value.to_bits()
            & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
        assert_eq!(
            got, want,
            "div {a:#022x} / {b:#022x}: got {got:#022x} want {want:#022x}"
        );
    }
}

/// Reference 80-bit square root via the host x87 FPU (default control word:
/// round-to-nearest, 64-bit precision). APFloat does not implement sqrt, so the
/// hardware is the oracle. Only meaningful on x86-64 test hosts.
#[cfg(target_arch = "x86_64")]
fn host_sqrt(a: u128) -> u128 {
    let (mant, se) = parts(a);
    let mut inb = [0u8; 10];
    inb[0..8].copy_from_slice(&mant.to_le_bytes());
    inb[8..10].copy_from_slice(&se.to_le_bytes());
    let mut outb = [0u8; 10];
    unsafe {
        std::arch::asm!(
            "fld tbyte ptr [{i}]",
            "fsqrt",
            "fstp tbyte ptr [{o}]",
            i = in(reg) inb.as_ptr(),
            o = in(reg) outb.as_mut_ptr(),
            options(nostack),
        );
    }
    let mant_o = u64::from_le_bytes(outb[0..8].try_into().unwrap());
    let se_o = u16::from_le_bytes(outb[8..10].try_into().unwrap());
    bits(mant_o, se_o)
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_sqrt_matches_hardware() {
    let mut rng = Rng(0x5927_17a0_b0c0_d0e0);
    for _ in 0..3000 {
        // sqrt is defined for non-negatives; force sign 0.
        let a = rand_normal(&mut rng) & !(1u128 << 79);
        let got = jit_sqrt(a);
        let want = host_sqrt(a);
        assert_eq!(
            got, want,
            "sqrt {a:#022x}: got {got:#022x} want {want:#022x}"
        );
    }
}

#[test]
fn test_sqrt_special() {
    // sqrt(4) = 2
    assert_eq!(jit_sqrt(TWO_SQ()), TWO);
    // sqrt(+0) = +0, sqrt(-0) = -0
    assert_eq!(jit_sqrt(POS_ZERO), POS_ZERO);
    assert_eq!(jit_sqrt(NEG_ZERO), NEG_ZERO);
    // sqrt(+inf) = +inf
    assert_eq!(jit_sqrt(POS_INF), POS_INF);
    // sqrt(negative) = NaN
    let neg_two = bits_const(0x8000_0000_0000_0000, 0xC000);
    assert!(is_nan_bits(jit_sqrt(neg_two)));
    // sqrt(-inf) = NaN
    assert!(is_nan_bits(jit_sqrt(NEG_INF)));
}

#[allow(non_snake_case)]
fn TWO_SQ() -> u128 {
    // 4.0 = significand 1.0 at biased exponent 0x4001.
    bits_const(0x8000_0000_0000_0000, 0x4001)
}

#[test]
fn test_cmp_matches_apfloat() {
    let mut rng = Rng(0x0f0f_0f0f_1234_5678);
    for _ in 0..2000 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        let (unord, eq, lt) = jit_cmp(a, b);
        let ord = af(a).partial_cmp(&af(b));
        assert!(!unord, "unexpected unordered for normals");
        assert_eq!(eq, ord == Some(std::cmp::Ordering::Equal), "eq mismatch");
        assert_eq!(lt, ord == Some(std::cmp::Ordering::Less), "lt mismatch");
    }
}

const POS_INF: u128 = bits_const(0x8000_0000_0000_0000, 0x7FFF);
const NEG_INF: u128 = bits_const(0x8000_0000_0000_0000, 0xFFFF);
const POS_ZERO: u128 = 0;
const NEG_ZERO: u128 = bits_const(0, 0x8000);
const ONE: u128 = bits_const(0x8000_0000_0000_0000, 0x3FFF);
const TWO: u128 = bits_const(0x8000_0000_0000_0000, 0x4000);

const fn bits_const(mant: u64, se: u16) -> u128 {
    (mant as u128) | ((se as u128) << 64)
}

fn is_nan_bits(x: u128) -> bool {
    let se = (x >> 64) as u16;
    let mant = x as u64;
    (se & 0x7FFF) == 0x7FFF && (mant & 0x7FFF_FFFF_FFFF_FFFF) != 0
}

#[test]
fn test_mul_special() {
    // 2 * 2 = 4
    assert_eq!(
        jit_binop(Op::Mul, TWO, TWO),
        bits_const(0x8000_0000_0000_0000, 0x4001)
    );
    // inf * 2 = inf
    assert_eq!(jit_binop(Op::Mul, POS_INF, TWO), POS_INF);
    // -inf * 2 = -inf
    assert_eq!(jit_binop(Op::Mul, NEG_INF, TWO), NEG_INF);
    // inf * 0 = NaN
    assert!(is_nan_bits(jit_binop(Op::Mul, POS_INF, POS_ZERO)));
    // 0 * 2 = 0
    assert_eq!(jit_binop(Op::Mul, POS_ZERO, TWO), POS_ZERO);
    // -1 * 1 = -1  (sign)
    let neg_one = bits_const(0x8000_0000_0000_0000, 0xBFFF);
    assert_eq!(jit_binop(Op::Mul, neg_one, ONE), neg_one);
}

#[test]
fn test_div_special() {
    // 2 / 2 = 1
    assert_eq!(jit_binop(Op::Div, TWO, TWO), ONE);
    // 1 / 0 = +inf
    assert_eq!(jit_binop(Op::Div, ONE, POS_ZERO), POS_INF);
    // -1 / 0 = -inf
    let neg_one = bits_const(0x8000_0000_0000_0000, 0xBFFF);
    assert_eq!(jit_binop(Op::Div, neg_one, POS_ZERO), NEG_INF);
    // 0 / 0 = NaN
    assert!(is_nan_bits(jit_binop(Op::Div, POS_ZERO, POS_ZERO)));
    // inf / inf = NaN
    assert!(is_nan_bits(jit_binop(Op::Div, POS_INF, POS_INF)));
    // 2 / inf = 0
    assert_eq!(jit_binop(Op::Div, TWO, POS_INF), POS_ZERO);
    // inf / 2 = inf
    assert_eq!(jit_binop(Op::Div, POS_INF, TWO), POS_INF);
}

#[test]
fn test_cmp_special() {
    // +0 == -0
    let (u, e, l) = jit_cmp(POS_ZERO, NEG_ZERO);
    assert!(!u && e && !l, "+0 == -0");
    // 1 < 2
    let (u, e, l) = jit_cmp(ONE, TWO);
    assert!(!u && !e && l, "1 < 2");
    // -inf < 1
    let (u, _e, l) = jit_cmp(NEG_INF, ONE);
    assert!(!u && l, "-inf < 1");
    // NaN vs 1 -> unordered
    let nan = bits_const(0xC000_0000_0000_0000, 0x7FFF);
    let (u, _e, _l) = jit_cmp(nan, ONE);
    assert!(u, "NaN unordered");
}
