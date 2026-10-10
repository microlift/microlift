//! Native-vector fast paths for the SSE handlers.
//!
//! Each [`sse`](super::sse)/[`sse_int`](super::sse_int) handler has a scalar `(lo, hi)` expansion; under `use_vector_xmm`
//! handlers call [`Lifter::vec_mode`] and emit a native `I64X2` sequence instead.
//! [`Lifter::vec_shuffle`] is the key primitive: most shuffles/unpacks are a lift-time constant permutation of two inputs.

use super::sse::{F32_INDEFINITE, F64_INDEFINITE};
use crate::core::GeneratedNan;
use crate::lifter::Lifter;
use cranelift::codegen::ir::ConstantData;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

/// Byte index of the `i`-th byte of the *second* `vec_shuffle` operand.
pub(super) const B: u8 = 16;

/// A packed floating-point operation for the native vector path; unary forms read the *source* and ignore the destination.
#[derive(Copy, Clone)]
pub(super) enum PackedFpOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `MINPS`/`MINPD` - a strict compare and select, not IEEE `fmin` (see [`Lifter::fp_min_max`](super::sse)).
    Min,
    /// `MAXPS`/`MAXPD`, the mirror of [`PackedFpOp::Min`].
    Max,
    /// `SQRTPS`/`SQRTPD` - square root of the source lane.
    Sqrt,
    /// `CMPPS`/`CMPPD` - an all-ones or all-zeros lane mask.
    Cmp(FloatCC),
    /// `RCPPS`/`RSQRTPS` - approximate reciprocal (of the square root).
    Rcp {
        sqrt: bool,
    },
}

impl<'a> Lifter<'a> {
    /// Whether XMM registers are native `I64X2` SSA variables (no lane round-trip).
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

    pub(super) fn vec_zero(&mut self) -> Value {
        let zero = self.builder.ins().iconst(types::I64, 0);
        self.builder.ins().splat(types::I64X2, zero)
    }

    /// Permute the 32 bytes of `a` (0-15) and `b` (16-31); `mask[i]` is the source of result byte `i`. Zero-fill by pointing at a zero operand.
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

    pub(super) fn vec_int_type(bits: u32) -> Type {
        match bits {
            8 => types::I8X16,
            16 => types::I16X8,
            32 => types::I32X4,
            _ => types::I64X2,
        }
    }

    /// Multiply the eight 16-bit lanes of `a` and `b` into two `I32X4` vectors of exact products (low four, then high four);
    /// `smulhi`/`umulhi` do not lower on aarch64.
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
        (
            self.builder.ins().imul(al, bl),
            self.builder.ins().imul(ah, bh),
        )
    }

    /// Narrow two `I32X4` to one `I16X8` keeping the 16 bits of each dword at `byte_off` (0 low, 2 high); truncates, unlike `snarrow`/`unarrow`.
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
        // `iadd_imm_s` would fold to a scalar; splat the rounding term explicitly.
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
    /// aarch64 lacks `UnorderedOrGreaterThan{,OrEqual}` (x86's `NLE`/`NLT`), so they are the `bnot` of the ordered compare they negate; exact.
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

    /// Convert a float vector to signed integer lanes with x86's out-of-range behaviour; twin of [`Lifter::fp_to_sint_x86`](super::sse).
    /// Saturation already gives `INT_MIN`; only NaN or too-positive lanes are patched to the indefinite.
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

    /// Lane-wise square root with x86's NaN behaviour (twin of [`fp_sqrt_x86`](super::sse::fp_sqrt_x86)):
    /// a negative lane yields the sign-set indefinite via `bitselect`; [`GeneratedNan::HostDefault`] skips it.
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

    /// Lane-wise arithmetic with x86's generated-NaN bits (twin of [`fp_arith_x86`](super::sse::fp_arith_x86)):
    /// a lane whose NaN neither operand supplied yields the sign-set indefinite unless [`GeneratedNan::HostDefault`];
    /// NaN operands use host propagation, which matches.
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

    /// Emit a packed FP op over whole vectors when XMM is vector-backed; returns `false` without emitting in scalar mode.
    /// `single` selects `F32X4` over `F64X2`; operand 0 is the destination, operand 1 the `xmm/m128` source.
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
                // A vector `fcmp` yields a lane mask, so the choice is a `bitselect`, not a `select`.
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
                // Through `vec_fp_sqrt_x86`: a negative lane must give x86's sign-set indefinite.
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

    /// Emit a *scalar* FP op on lane 0 when XMM is vector-backed; returns `false` without emitting in scalar mode.
    /// The portable path moves lane 0 through a GPR (4-6 instructions around a one-instruction multiply on aarch64);
    /// here the value stays in the vector file and the `insertlane` is the single `mov` the merge semantics call for.
    /// `single` selects `F32X4` over `F64X2`; `op` is the same closure, NaN handling included.
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

        // Lanes 1.. are untouched: the destination merge scalar forms specify.
        let merged = self.builder.ins().insertlane(dst, r, 0);
        let merged = self.vec_bitcast(merged, types::I64X2);
        self.set_xmm_vec(dst_idx, merged);
        Ok(true)
    }

    /// Read the scalar source of a [`Lifter::try_scalar_fp_vec`] op as a bare `F32`/`F64`; memory is loaded at the float type to avoid a GPR transfer.
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
                    .load(fty.lane_type(), self.guest_flags, ptr, 0))
            }
            other => Err(crate::MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-scalar-fp-src-op{}-{:?}", who, op_index, other),
            }),
        }
    }

    /// `ADDSUBPS`/`ADDSUBPD` over whole vectors when XMM is vector-backed; returns `false` in scalar mode.
    /// Both results are blended by a constant shuffle; flipping the even lanes' sign and one `fadd` is not equivalent (a NaN even lane would flip sign).
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
        // Operand 0 is the subtract result, operand 1 the add; an index at or past `lanes` addresses the second.
        let mask = Self::lane_shuffle_mask(bits, |j| if j % 2 == 0 { j } else { lanes + j });
        let r = self.vec_shuffle(sub, add, mask);

        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// Build a 16-byte shuffle mask from a lane-level selector: `pick(j)` is the source lane (across both operands) of result lane `j`, `bits` the lane width.
    pub(super) fn lane_shuffle_mask(bits: u32, mut pick: impl FnMut(u32) -> u32) -> [u8; 16] {
        let stride = bits / 8;
        let lanes = 128 / bits;
        let mut mask = [0u8; 16];
        for out in 0..lanes {
            let src = pick(out);
            // An index past the first operand's lanes addresses the second, whose bytes start at 16.
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
