use super::sse::{PackedWidth, RoundMode, apply_round};
use super::sse_int::HorizOp;
use crate::core::specialize::{Spec, SpecTarget};
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// Whether `spec` may apply here: the target implements it, the caller enabled it, and vector specs need vector-backed XMM.
    pub(super) fn specialized(&self, spec: Spec) -> bool {
        (self.vec_mode() || !spec.requires_vector_xmm()) && self.config.specialization.allows(spec)
    }

    /// Gather even and odd lanes of `(a, b)` into two vectors; for `HADDPS` the pair is `([d0,d2,s0,s2], [d1,d3,s1,s3])`.
    /// The masks are the byte patterns aarch64 lowers to `UZP1`/`UZP2`.
    fn horizontal_pair(&mut self, a: Value, b: Value, bits: u32) -> (Value, Value) {
        let even = Self::lane_shuffle_mask(bits, |j| 2 * j);
        let odd = Self::lane_shuffle_mask(bits, |j| 2 * j + 1);
        (self.vec_shuffle(a, b, even), self.vec_shuffle(a, b, odd))
    }

    /// `ROUNDPS` / `ROUNDPD` as one packed rounding instruction (`FRINT{N,M,P,Z}` on aarch64).
    /// No NaN fixup: rounding never creates a NaN.
    ///
    /// [`Spec::PackedFpRound`]
    pub(super) fn try_packed_round_spec(
        &mut self,
        instruction: &Instruction,
        who: &str,
        single: bool,
        mode: RoundMode,
    ) -> crate::Result<bool> {
        if !self.specialized(Spec::PackedFpRound) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let fty = if single { types::F32X4 } else { types::F64X2 };

        let src = self.read_xmm_src_vec(instruction, 1)?;
        let x = self.vec_bitcast(src, fty);
        let r = apply_round(&mut self.builder, mode, x);

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `HADDPS` / `HSUBPS` / `HADDPD` / `HSUBPD` - horizontal float add/subtract: two constant shuffles and the packed op.
    /// The op goes through [`vec_fp_arith_x86`](Lifter::vec_fp_arith_x86) for x86's NaN indefinite.
    ///
    /// [`Spec::PackedFpHorizontal`]
    pub(super) fn try_horizontal_fp_spec(
        &mut self,
        instruction: &Instruction,
        who: &str,
        single: bool,
        subtract: bool,
    ) -> crate::Result<bool> {
        if !self.specialized(Spec::PackedFpHorizontal) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let fty = if single { types::F32X4 } else { types::F64X2 };
        let bits = if single { 32 } else { 64 };

        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let (even, odd) = self.horizontal_pair(dst, src, bits);
        let a = self.vec_bitcast(even, fty);
        let b = self.vec_bitcast(odd, fty);

        let r = if subtract {
            self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fsub(a, b))
        } else {
            self.vec_fp_arith_x86(a, b, |s, a, b| s.builder.ins().fadd(a, b))
        };

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PHADDW` / `PHADDD` / `PHSUBW` / `PHSUBD` / `PHADDSW` / `PHSUBSW`.
    /// Plain adds are `iadd_pairwise`; the other three reuse the even/odd gather (three instructions).
    pub(super) fn try_horizontal_int_spec(
        &mut self,
        instruction: &Instruction,
        op: HorizOp,
        width: PackedWidth,
    ) -> crate::Result<bool> {
        if !self.specialized(Spec::PackedIntHorizontal) {
            return Ok(false);
        }
        let ty = width.vector_type();
        let bits = width.bits();

        let gathers = op != HorizOp::Add;
        if gathers && bits < 32 && self.config.specialization.target() == SpecTarget::X86_64 {
            return Ok(false);
        }

        let dst_idx = self.xmm_operand_index(instruction, 0, "phadd")?;

        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;

        let r = if !gathers {
            let a = self.vec_bitcast(dst, ty);
            let b = self.vec_bitcast(src, ty);
            self.builder.ins().iadd_pairwise(a, b)
        } else {
            let (even, odd) = self.horizontal_pair(dst, src, bits);
            let a = self.vec_bitcast(even, ty);
            let b = self.vec_bitcast(odd, ty);
            match op {
                HorizOp::Add => unreachable!("handled above"),
                HorizOp::Sub => self.builder.ins().isub(a, b),
                HorizOp::AddSat => self.builder.ins().sadd_sat(a, b),
                HorizOp::SubSat => self.builder.ins().ssub_sat(a, b),
            }
        };

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PMADDWD` - signed word multiply, adjacent products summed into dwords.
    /// Widening multiply of each half then `iadd_pairwise`, which x86-64 folds back to `pmaddwd`.
    /// `-32768 * -32768` twice wraps, as on hardware (`imul` on `I32X4` wraps).
    ///
    /// [`Spec::PairwiseMultiplyAdd`]
    pub(super) fn try_pmaddwd_spec(&mut self, instruction: &Instruction) -> crate::Result<bool> {
        if !self.specialized(Spec::PairwiseMultiplyAdd) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmaddwd")?;
        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, types::I16X8);
        let b = self.vec_bitcast(src, types::I16X8);

        let lo = {
            let al = self.builder.ins().swiden_low(a);
            let bl = self.builder.ins().swiden_low(b);
            self.builder.ins().imul(al, bl)
        };
        let hi = {
            let ah = self.builder.ins().swiden_high(a);
            let bh = self.builder.ins().swiden_high(b);
            self.builder.ins().imul(ah, bh)
        };
        let r = self.builder.ins().iadd_pairwise(lo, hi);

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PMADDUBSW` - unsigned destination bytes times signed source bytes, adjacent products summed into words with signed saturation.
    /// Cranelift's `x86_pmaddubsw` needs `has_ssse3` (off in the baseline), so this sequence lowers everywhere:
    /// 1. Byte products in `I16X8`, exact - a `u8 * i8` spans `-32640..=32385`.
    /// 2. Each product vector widened and pairwise-added in `I32X4`, in the
    ///    `iadd_pairwise(swiden_low v, swiden_high v)` shape aarch64 lowers to one
    ///    `SADDLP`. At 32 bits precisely so saturation happens once, at the end.
    /// 3. `snarrow` saturates the two halves back into one `I16X8` - `SQXTN`/`SQXTN2`.
    ///
    /// [`Spec::MultiplyAddUnsignedSigned`]
    pub(super) fn try_pmaddubsw_spec(&mut self, instruction: &Instruction) -> crate::Result<bool> {
        if !self.specialized(Spec::MultiplyAddUnsignedSigned) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmaddubsw")?;
        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, types::I8X16);
        let b = self.vec_bitcast(src, types::I8X16);

        // Products of the low eight and high eight byte pairs, exact in i16.
        let p_lo = {
            let al = self.builder.ins().uwiden_low(a);
            let bl = self.builder.ins().swiden_low(b);
            self.builder.ins().imul(al, bl)
        };
        let p_hi = {
            let ah = self.builder.ins().uwiden_high(a);
            let bh = self.builder.ins().swiden_high(b);
            self.builder.ins().imul(ah, bh)
        };

        // Widening pairwise add at 32 bits so nothing wraps before the saturating narrow.
        let widen_pairs = |s: &mut Self, p: Value| {
            let lo = s.builder.ins().swiden_low(p);
            let hi = s.builder.ins().swiden_high(p);
            s.builder.ins().iadd_pairwise(lo, hi)
        };
        let s_lo = widen_pairs(self, p_lo);
        let s_hi = widen_pairs(self, p_hi);

        let r = self.builder.ins().snarrow(s_lo, s_hi);
        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PSADBW` - sum of absolute byte differences, one per 64-bit half, in that half's low word, rest zeroed.
    /// `|a - b|` of unsigned bytes is `usub_sat(a, b) | usub_sat(b, a)`; the reduction is widening pairwise adds.
    /// The last stage stays in `I32X4` (no 64-bit `iadd_pairwise`); `uwiden_low` selects the live lanes and zeroes the halves.
    ///
    /// [`Spec::SumOfAbsoluteDifferences`]
    pub(super) fn try_psadbw_spec(&mut self, instruction: &Instruction) -> crate::Result<bool> {
        if !self.specialized(Spec::SumOfAbsoluteDifferences) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "psadbw")?;
        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, types::I8X16);
        let b = self.vec_bitcast(src, types::I8X16);

        let ab = self.builder.ins().usub_sat(a, b);
        let ba = self.builder.ins().usub_sat(b, a);
        let diff = self.builder.ins().bor(ab, ba);

        // Each stage doubles the lane width, so no partial sum overflows.
        let pairwise = |s: &mut Self, v: Value| {
            let lo = s.builder.ins().uwiden_low(v);
            let hi = s.builder.ins().uwiden_high(v);
            s.builder.ins().iadd_pairwise(lo, hi)
        };
        let words = pairwise(self, diff);
        let dwords = pairwise(self, words);

        // [q0, q1, q0, q1] where q0 sums bytes 0..7 and q1 bytes 8..15; the
        // widening of the low half is the I64X2 result, zero-extended.
        let halves = self.builder.ins().iadd_pairwise(dwords, dwords);
        let r = self.builder.ins().uwiden_low(halves);

        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PMULUDQ` / `PMULDQ` - widening multiply of the even dword lanes.
    /// The portable `I64X2` `imul` expands to seven instructions on aarch64; gathering even dwords with a `UZP1`-shaped
    /// shuffle and widening the low halves on the same side gives `UMULL`/`SMULL` (three instructions).
    ///
    /// [`Spec::WideningEvenMultiply`]
    pub(super) fn try_pmuldq_spec(
        &mut self,
        instruction: &Instruction,
        signed: bool,
    ) -> crate::Result<bool> {
        if !self.specialized(Spec::WideningEvenMultiply) {
            return Ok(false);
        }
        let who = if signed { "pmuldq" } else { "pmuludq" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;

        // A one-operand permute; only the low two result lanes are read.
        let evens = Self::lane_shuffle_mask(32, |j| 2 * j);
        let a = self.vec_shuffle(dst, dst, evens);
        let b = self.vec_shuffle(src, src, evens);
        let a = self.vec_bitcast(a, types::I32X4);
        let b = self.vec_bitcast(b, types::I32X4);

        let (a, b) = if signed {
            (
                self.builder.ins().swiden_low(a),
                self.builder.ins().swiden_low(b),
            )
        } else {
            (
                self.builder.ins().uwiden_low(a),
                self.builder.ins().uwiden_low(b),
            )
        };
        let r = self.builder.ins().imul(a, b);

        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }

    /// `PMULHRSW` - signed word multiply keeping the rounded high 17 bits, `(((a * b) >> 14) + 1) >> 1`.
    /// aarch64 only: the portable expansion is already three instructions on x86-64.
    /// `SQRDMULH` saturates where x86 truncates; only `-32768 * -32768` differs (`0x8000` vs `0x7FFF`), fixed up with two compares, an AND and a select.
    ///
    /// [`Spec::RoundedMultiplyHigh`]
    pub(super) fn try_pmulhrsw_spec(&mut self, instruction: &Instruction) -> crate::Result<bool> {
        if !self.specialized(Spec::RoundedMultiplyHigh) {
            return Ok(false);
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmulhrsw")?;
        let dst = self.get_xmm_vec(dst_idx);
        let src = self.read_xmm_src_vec(instruction, 1)?;
        let a = self.vec_bitcast(dst, types::I16X8);
        let b = self.vec_bitcast(src, types::I16X8);

        let sat = self.builder.ins().sqmul_round_sat(a, b);

        // The single input pair the saturation differs on.
        let min = self.builder.ins().iconst(types::I16, i64::from(i16::MIN));
        let min = self.builder.ins().splat(types::I16X8, min);
        let a_min = self.builder.ins().icmp(IntCC::Equal, a, min);
        let b_min = self.builder.ins().icmp(IntCC::Equal, b, min);
        let both = self.builder.ins().band(a_min, b_min);
        let r = self.builder.ins().bitselect(both, min, sat);

        let r = self.vec_bitcast(r, types::I64X2);
        self.set_xmm_vec(dst_idx, r);
        Ok(true)
    }
}
