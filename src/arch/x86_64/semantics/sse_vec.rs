//! Native-vector fast paths for the SSE handlers.
//!
//! Every handler in [`sse`](super::sse) and [`sse_int`](super::sse_int) has a
//! portable scalar expansion that works on the `(lo, hi)` 64-bit half pair.
//! That expansion is required when XMM registers are backed by two `I64` SSA
//! variables (the default, and the only option for AOT targets without SIMD),
//! but it is *also* what ran under `use_vector_xmm` for every handler that had
//! no explicit fast path - paying two `extractlane`s, a scalar lane expansion
//! that can run to hundreds of IR instructions, and two `insertlane`s, all to
//! express something the host does in one SIMD instruction.
//!
//! This module holds the shared plumbing those fast paths are built from. The
//! handlers call [`Lifter::vec_mode`] and, when it holds, emit a native
//! `I64X2`-typed sequence instead.
//!
//! The single most valuable primitive here is [`Lifter::vec_shuffle`]: a large
//! part of the SSE surface (`PSHUFD`, `SHUFPS`, `SHUFPD`, the whole `PUNPCK*` /
//! `UNPCKP*` family, `PSHUFLW` / `PSHUFHW`, `PSLLDQ` / `PSRLDQ`, `PALIGNR`,
//! `MOVSLDUP` / `MOVSHDUP`, `MOVDDUP`, `MOVHLPS` / `MOVLHPS` and the
//! register-to-register `MOVSS` / `MOVSD` merges) is nothing but a *lift-time
//! constant* permutation of the 32 bytes of two 128-bit inputs, which is
//! exactly Cranelift's `shuffle`. One mask builder per instruction replaces
//! each hand-rolled mask/shift/OR chain.

use super::sse::{F32_INDEFINITE, F64_INDEFINITE};
use crate::core::GeneratedNan;
use crate::lifter::Lifter;
use cranelift::codegen::ir::ConstantData;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

/// Byte index of the `i`-th byte of the *second* `vec_shuffle` operand.
pub(super) const B: u8 = 16;

/// A packed floating-point operation, in the form the native vector path emits.
///
/// This mirrors the closures the scalar lane plumbing in [`sse`](super::sse)
/// takes; the unary forms read the *source* operand and ignore the destination,
/// so every variant can share the one `(dst, src)` operand shape.
#[derive(Copy, Clone)]
pub(super) enum PackedFpOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `MINPS`/`MINPD` - see [`Lifter::fp_min_max`](super::sse) for why this is
    /// a strict compare and select rather than an IEEE `fmin`.
    Min,
    /// `MAXPS`/`MAXPD`, the mirror of [`PackedFpOp::Min`].
    Max,
    /// `SQRTPS`/`SQRTPD` - square root of the source lane.
    Sqrt,
    /// `CMPPS`/`CMPPD` - an all-ones or all-zeros lane mask.
    Cmp(FloatCC),
    /// `RCPPS`/`RSQRTPS` - approximate reciprocal (of the square root) of the
    /// source lane.
    Rcp {
        sqrt: bool,
    },
}

impl<'a> Lifter<'a> {
    /// Get if XMM registers are backed by native `I64X2` SSA variables, so
    /// emitting vector IR costs no lane round-trip.
    pub(super) fn vec_mode(&self) -> bool {
        self.config.use_vector_xmm
    }

    /// [`Optimization::UnsafeImpreciseNaN`](crate::core::optimize::Optimization::UnsafeImpreciseNaN)
    pub(super) fn generated_nan(&self) -> GeneratedNan {
        if self
            .config
            .optimizations
            .allows(crate::core::optimize::Optimization::UnsafeImpreciseNaN)
        {
            return GeneratedNan::HostDefault;
        }
        self.config.generated_nan
    }

    /// A 128-bit all-zeros vector.
    pub(super) fn vec_zero(&mut self) -> Value {
        let zero = self.builder.ins().iconst(types::I64, 0);
        self.builder.ins().splat(types::I64X2, zero)
    }

    /// Permute the 32 bytes of `a` (indices 0-15) and `b` (indices 16-31) into
    /// a new 128-bit value, `mask[i]` giving the source of result byte `i`.
    ///
    /// Indices outside 0-31 are invalid; use a zero operand and point at one of
    /// its bytes to synthesize a zero-filled lane.
    pub(super) fn vec_shuffle(&mut self, a: Value, b: Value, mask: [u8; 16]) -> Value {
        debug_assert!(mask.iter().all(|&i| i < 32), "shuffle index out of range");
        let a = self.vec_bitcast(a, types::I8X16);
        let b = self.vec_bitcast(b, types::I8X16);
        let imm = self
            .builder
            .func
            .dfg
            .immediates
            .push(ConstantData::from(&mask[..]));
        let r = self.builder.ins().shuffle(a, b, imm);
        self.vec_bitcast(r, types::I64X2)
    }

    /// The 128-bit Cranelift integer vector type with `bits`-wide lanes.
    pub(super) fn vec_int_type(bits: u32) -> Type {
        match bits {
            8 => types::I8X16,
            16 => types::I16X8,
            32 => types::I32X4,
            _ => types::I64X2,
        }
    }

    /// Multiply the eight 16-bit lanes of `a` and `b` into two `I32X4` vectors
    /// of exact 32-bit products - the low four lanes, then the high four.
    ///
    /// This is the portable substrate the 16-bit multiply-high forms are built
    /// on. Cranelift *does* have single ops for some of them (`smulhi`,
    /// `umulhi`, `x86_pmulhrsw`), but none of those lower on aarch64, and
    /// microlift targets whatever host or AOT triple it is pointed at - so the
    /// expansion has to use ops every backend implements.
    fn vec_mul_word_widen(&mut self, a: Value, b: Value, signed: bool) -> (Value, Value) {
        let a = self.vec_bitcast(a, types::I16X8);
        let b = self.vec_bitcast(b, types::I16X8);
        let (al, ah, bl, bh) = if signed {
            (
                self.builder.ins().swiden_low(a),
                self.builder.ins().swiden_high(a),
                self.builder.ins().swiden_low(b),
                self.builder.ins().swiden_high(b),
            )
        } else {
            (
                self.builder.ins().uwiden_low(a),
                self.builder.ins().uwiden_high(a),
                self.builder.ins().uwiden_low(b),
                self.builder.ins().uwiden_high(b),
            )
        };
        // A 16x16 product is exact in 32 bits, so nothing can overflow here.
        (
            self.builder.ins().imul(al, bl),
            self.builder.ins().imul(ah, bh),
        )
    }

    /// Narrow two `I32X4` vectors to one `I16X8` by keeping the 16 bits of each
    /// dword lane starting at byte `byte_off` (0 for the low half, 2 for the
    /// high), discarding the rest.
    ///
    /// Unlike `snarrow`/`unarrow` this truncates rather than saturates, which is
    /// what the multiply forms want.
    fn vec_narrow_dwords_to_words(&mut self, lo: Value, hi: Value, byte_off: u8) -> Value {
        let mut mask = [0u8; 16];
        for j in 0..8u8 {
            let (base, lane) = if j < 4 { (0, j) } else { (B, j - 4) };
            mask[(2 * j) as usize] = base + 4 * lane + byte_off;
            mask[(2 * j + 1) as usize] = base + 4 * lane + byte_off + 1;
        }
        self.vec_shuffle(lo, hi, mask)
    }

    /// `PMULHW` / `PMULHUW` - the high half of the per-lane 16-bit product.
    pub(super) fn vec_mulhi_word(&mut self, a: Value, b: Value, signed: bool) -> Value {
        let (p_lo, p_hi) = self.vec_mul_word_widen(a, b, signed);
        self.vec_narrow_dwords_to_words(p_lo, p_hi, 2)
    }

    /// `PMULHRSW` - signed word multiply keeping the high 17 bits, rounded:
    /// `(((a * b) >> 14) + 1) >> 1`, truncated back to 16 bits.
    pub(super) fn vec_mulhrsw(&mut self, a: Value, b: Value) -> Value {
        let (p_lo, p_hi) = self.vec_mul_word_widen(a, b, true);
        // `iadd_imm_s` would fold to a *scalar* constant operand, so the
        // rounding term has to be splatted across the lanes explicitly.
        let one = self.builder.ins().iconst(types::I32, 1);
        let one = self.builder.ins().splat(types::I32X4, one);
        let round = |s: &mut Self, p: Value| {
            let shifted = s.builder.ins().sshr_imm_u(p, 14);
            let rounded = s.builder.ins().iadd(shifted, one);
            s.builder.ins().sshr_imm_u(rounded, 1)
        };
        let r_lo = round(self, p_lo);
        let r_hi = round(self, p_hi);
        self.vec_narrow_dwords_to_words(r_lo, r_hi, 0)
    }

    /// Lane-wise floating-point compare yielding an all-ones/all-zeros mask.
    ///
    /// aarch64 has no vector-float lowering for the
    /// `UnorderedOrGreaterThan{,OrEqual}` conditions (x86's `NLE` and `NLT`
    /// predicates), so those are emitted as the bitwise complement of the
    /// ordered compare each one negates. `NLT` *is* "not `<`" and `NLE` "not
    /// `<=`", unordered operands included, so the rewrite is exact; it costs one
    /// `bnot` on backends that could have taken the condition directly.
    pub(super) fn vec_fcmp(&mut self, cc: FloatCC, a: Value, b: Value) -> Value {
        let (cc, invert) = match cc {
            FloatCC::UnorderedOrGreaterThanOrEqual => (FloatCC::LessThan, true),
            FloatCC::UnorderedOrGreaterThan => (FloatCC::LessThanOrEqual, true),
            cc => (cc, false),
        };
        let c = self.builder.ins().fcmp(cc, a, b);
        if invert {
            self.builder.ins().bnot(c)
        } else {
            c
        }
    }

    /// Convert a floating-point vector to signed integer lanes with x86's
    /// out-of-range behaviour.
    ///
    /// The vector twin of [`Lifter::fp_to_sint_x86`](super::sse): saturation
    /// already gives `INT_MIN` on the too-negative side, so only the "NaN or too
    /// positive" side needs patching back from `fcvt_to_sint_sat`'s
    /// `0`/`INT_MAX` to the integer indefinite value.
    pub(super) fn vec_fp_to_sint_x86(&mut self, x: Value, ity: Type, truncate: bool) -> Value {
        let fty = self.builder.func.dfg.value_type(x);
        let x = if truncate {
            x
        } else {
            self.builder.ins().nearest(x)
        };
        let sat = self.builder.ins().fcvt_to_sint_sat(ity, x);

        let limit = if fty.lane_type() == types::F32 {
            self.builder.ins().f32const(2147483648.0f32)
        } else {
            self.builder.ins().f64const(9223372036854775808.0f64)
        };
        let limit = self.builder.ins().splat(fty, limit);
        let invalid = self.vec_fcmp(FloatCC::UnorderedOrGreaterThanOrEqual, x, limit);

        let indefinite = self
            .builder
            .ins()
            .iconst(ity.lane_type(), i64::from(i32::MIN));
        let indefinite = self.builder.ins().splat(ity, indefinite);
        self.builder.ins().bitselect(invalid, indefinite, sat)
    }

    /// Lane-wise square root with x86's NaN behaviour.
    ///
    /// The vector twin of [`fp_sqrt_x86`](super::sse::fp_sqrt_x86): a negative
    /// lane must produce x86's floating indefinite, which is sign-set, rather
    /// than whatever default NaN the host FPU picks. As with the MIN/MAX forms
    /// the per-lane choice is a `bitselect` on the compare's all-ones mask, and
    /// as there, [`GeneratedNan::HostDefault`] skips the whole thing.
    fn vec_fp_sqrt_x86(&mut self, x: Value) -> Value {
        let fty = self.builder.func.dfg.value_type(x);
        let r = self.builder.ins().sqrt(x);
        if self.generated_nan() == GeneratedNan::HostDefault {
            return r;
        }

        let (zero, indefinite) = if fty.lane_type() == types::F32 {
            let z = self.builder.ins().f32const(0.0f32);
            (z, self.builder.ins().f32const(F32_INDEFINITE))
        } else {
            let z = self.builder.ins().f64const(0.0f64);
            (z, self.builder.ins().f64const(F64_INDEFINITE))
        };
        let zero = self.builder.ins().splat(fty, zero);
        let indefinite = self.builder.ins().splat(fty, indefinite);

        let negative = self.vec_fcmp(FloatCC::LessThan, x, zero);
        let negative = self.vec_bitcast(negative, fty);
        self.builder.ins().bitselect(negative, indefinite, r)
    }

    /// Lane-wise arithmetic with x86's generated-NaN bits.
    ///
    /// The vector twin of [`fp_arith_x86`](super::sse::fp_arith_x86): a lane
    /// whose result is a NaN neither operand supplied performed an invalid
    /// operation (`inf + -inf`, `inf - inf`, `0 * inf`, `0 / 0`, `inf / inf`)
    /// and must yield x86's sign-set floating indefinite rather than the host's
    /// default NaN - unless [`GeneratedNan::HostDefault`] says the host's is
    /// acceptable, in which case nothing is emitted. Lanes with a NaN operand
    /// fall through to the host's propagation, which already matches x86.
    pub(super) fn vec_fp_arith_x86(
        &mut self,
        a: Value,
        b: Value,
        op: impl FnOnce(&mut Self, Value, Value) -> Value,
    ) -> Value {
        let fty = self.builder.func.dfg.value_type(a);
        let r = op(self, a, b);
        if self.generated_nan() == GeneratedNan::HostDefault {
            return r;
        }

        let indefinite = if fty.lane_type() == types::F32 {
            self.builder.ins().f32const(F32_INDEFINITE)
        } else {
            self.builder.ins().f64const(F64_INDEFINITE)
        };
        let indefinite = self.builder.ins().splat(fty, indefinite);

        let generated = self.vec_fcmp(FloatCC::Unordered, r, r);
        let operands_ok = self.vec_fcmp(FloatCC::Ordered, a, b);
        let invalid = self.builder.ins().band(generated, operands_ok);
        let invalid = self.vec_bitcast(invalid, fty);
        self.builder.ins().bitselect(invalid, indefinite, r)
    }

    /// Emit a packed floating-point op over whole 128-bit vectors, when XMM
    /// registers are vector-backed.
    ///
    /// Returns `false` without emitting anything in scalar mode, leaving the
    /// caller to fall through to its portable lane expansion.
    ///
    /// `single` selects `F32X4` lanes over `F64X2`. Operand 0 is the
    /// destination XMM register and operand 1 the `xmm/m128` source, the shape
    /// every packed form here shares.
    pub(super) fn try_packed_fp_vec(
        &mut self,
        instruction: &Instruction,
        who: &str,
        single: bool,
        op: PackedFpOp,
    ) -> crate::Result<bool> {
        if !self.vec_mode() {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let fty = if single { types::F32X4 } else { types::F64X2 };

        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, fty);
        let b = self.vec_bitcast(src, fty);

        let r = match op {
            PackedFpOp::Add => self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fadd(a, b)),
            PackedFpOp::Sub => self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fsub(a, b)),
            PackedFpOp::Mul => self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fmul(a, b)),
            PackedFpOp::Div => self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fdiv(a, b)),
            PackedFpOp::Sqrt => self.vec_fp_sqrt_x86(b),
            PackedFpOp::Min | PackedFpOp::Max => {
                let cc = if matches!(op, PackedFpOp::Min) {
                    FloatCC::LessThan
                } else {
                    FloatCC::GreaterThan
                };
                // A vector `fcmp` yields a per-lane all-ones/all-zeros mask, so
                // the lane-wise choice is a `bitselect` rather than the scalar
                // path's `select` on a one-bit condition.
                let c = self.vec_fcmp(cc, a, b);
                let c = self.vec_bitcast(c, fty);
                self.builder.ins().bitselect(c, a, b)
            }
            PackedFpOp::Cmp(cc) => {
                let c = self.vec_fcmp(cc, a, b);
                let r = self.vec_bitcast(c, types::I64X2);
                self.set_xmm_vec(dst_idx, r);
                return Ok(true);
            }
            PackedFpOp::Rcp { sqrt } => {
                // Through `vec_fp_sqrt_x86`, not a bare `sqrt`: a negative lane
                // generates a NaN, which must be x86's sign-set indefinite.
                let y = if sqrt { self.vec_fp_sqrt_x86(b) } else { b };
                let one = self.builder.ins().f32const(1.0);
                let one = self.builder.ins().splat(fty, one);
                self.builder.ins().fdiv(one, y)
            }
        };

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// Emit a *scalar* floating-point op on lane 0, when XMM registers are
    /// vector-backed.
    ///
    /// Returns `false` without emitting anything in scalar mode, leaving the
    /// caller to fall through to its portable lane expansion.
    ///
    /// The portable expansion reaches lane 0 through
    /// [`get_xmm_lo`](Lifter::get_xmm_lo), which in vector mode is an
    /// `extractlane` to an *integer* - so `MULSS` arrives at its `fmul` having
    /// moved the lane into a GPR, masked off the half it does not want, and
    /// moved it back, and leaves the same way. On aarch64 that is four to six
    /// instructions around a one-instruction multiply, two of them FP-to-GPR
    /// transfers, which are among the more expensive moves on the machine.
    ///
    /// Nothing about the arithmetic changes here: `op` is the same closure the
    /// portable path uses, NaN handling included. The only difference is that
    /// the value never leaves the vector register file - lane 0 of an `F32X4`
    /// *is* `sN`, so the `extractlane` costs nothing and the `insertlane` that
    /// puts the result back is the single `mov vD.s[0], vN.s[0]` that the
    /// instruction's "bits[127:32] are preserved" semantics call for.
    ///
    /// `single` selects `F32X4` lanes over `F64X2`. Operand 0 is the
    /// destination XMM register and operand 1 the `xmm/m32` (or `m64`) source.
    pub(super) fn try_scalar_fp_vec(
        &mut self,
        instruction: &Instruction,
        who: &str,
        single: bool,
        op: &impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<bool> {
        if !self.vec_mode() {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let fty = if single { types::F32X4 } else { types::F64X2 };

        let dst = self.get_xmm_vec(dst_idx);
        let dst = self.vec_bitcast(dst, fty);
        let a = self.builder.ins().extractlane(dst, 0);
        let b = self.read_scalar_fp_vec_src(instruction, 1, fty, who)?;

        let r = op(&mut self.builder, a, b);

        // Lanes 1.. are untouched, which is exactly the destination merge every
        // scalar form specifies: bits[127:32] for single, bits[127:64] for
        // double.
        let merged = self.builder.ins().insertlane(dst, r, 0);
        let merged = self.vec_bitcast(merged, types::I64X2);
        self.set_xmm_vec(dst_idx, merged);
        Ok(true)
    }

    /// Read the scalar source of a [`Lifter::try_scalar_fp_vec`] operation as a
    /// bare `F32`/`F64`, without routing it through an integer register.
    ///
    /// A memory operand is loaded at the *scalar* float type rather than as
    /// bits, so it lands in the FP register file directly (`ldr s`/`ldr d`) -
    /// loading it as an integer would reintroduce on the source side exactly
    /// the transfer this path exists to remove.
    fn read_scalar_fp_vec_src(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        fty: Type,
        who: &str,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let idx = self.xmm_operand_index(instruction, op_index, who)?;
                let v = self.get_xmm_vec(idx);
                let v = self.vec_bitcast(v, fty);
                Ok(self.builder.ins().extractlane(v, 0))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                Ok(self
                    .builder
                    .ins()
                    .load(fty.lane_type(), MemFlagsData::new(), ptr, 0))
            }
            other => Err(crate::MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-scalar-fp-src-op{}-{:?}", who, op_index, other),
            }),
        }
    }

    /// `ADDSUBPS` / `ADDSUBPD` - subtract in the even lanes, add in the odd
    /// ones - over whole 128-bit vectors, when XMM registers are vector-backed.
    ///
    /// Returns `false` without emitting anything in scalar mode, leaving the
    /// caller to fall through to its portable lane expansion.
    ///
    /// Both results are computed and then blended by a lift-time constant
    /// shuffle. The cheaper-looking alternative - flip the sign of the even
    /// source lanes and issue a single `fadd` - is *not* equivalent: an even
    /// lane whose source is a NaN would propagate that NaN with its sign bit
    /// flipped, where the real subtract propagates it unchanged.
    pub(super) fn try_addsub_vec(
        &mut self,
        instruction: &Instruction,
        who: &str,
        single: bool,
    ) -> crate::Result<bool> {
        if !self.vec_mode() {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let fty = if single { types::F32X4 } else { types::F64X2 };

        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, fty);
        let b = self.vec_bitcast(src, fty);

        let sub = self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fsub(a, b));
        let add = self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fadd(a, b));
        let sub = self.vec_bitcast(sub, types::I64X2);
        let add = self.vec_bitcast(add, types::I64X2);

        let bits = if single { 32 } else { 64 };
        let lanes = 128 / bits;
        // Operand 0 is the subtract result, operand 1 the add: a lane index at
        // or past `lanes` addresses the second.
        let mask = Self::lane_shuffle_mask(bits, |j| if j % 2 == 0 { j } else { lanes + j });
        let r = self.vec_shuffle(sub, add, mask);

        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// Build a 16-byte shuffle mask from a lane-level selector.
    ///
    /// `bits` is the lane width; `pick(j)` returns the source lane of result
    /// lane `j`, numbered across both operands (`0..128/bits` in the first,
    /// then the same range again in the second). This keeps every caller
    /// working in the lane units its mnemonic is defined in rather than bytes.
    pub(super) fn lane_shuffle_mask(bits: u32, mut pick: impl FnMut(u32) -> u32) -> [u8; 16] {
        let stride = bits / 8;
        let lanes = 128 / bits;
        let mut mask = [0u8; 16];
        for out in 0..lanes {
            let src = pick(out);
            // A lane index past the first operand's lane count addresses the
            // second operand, whose bytes start at 16.
            let base = if src >= lanes {
                B as u32 + (src - lanes) * stride
            } else {
                src * stride
            };
            for k in 0..stride {
                mask[(out * stride + k) as usize] = (base + k) as u8;
            }
        }
        mask
    }
}
