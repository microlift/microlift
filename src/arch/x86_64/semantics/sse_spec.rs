use super::sse::{PackedWidth, RoundMode, apply_round};
use super::sse_int::HorizOp;
use crate::core::specialize::{Spec, SpecTarget};
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    pub(super) fn specialized(&self, spec: Spec) -> bool {
        self.vec_mode() && self.config.specialization.allows(spec)
    }

    /// Gather the even and odd lanes of `(a, b)` into two vectors, in the order
    /// x86's horizontal instructions want them.
    ///
    /// Result lane `j` of the even vector is source lane `2j` counted across
    /// both operands, so for `HADDPS` the pair is
    /// `([d0,d2,s0,s2], [d1,d3,s1,s3])` and applying any binary op lane-wise
    /// gives `[d0+d1, d2+d3, s0+s1, s2+s3]` - the instruction's definition,
    /// including its "destination's results low, source's high" layout.
    ///
    /// Both masks are the byte patterns aarch64 lowers to `UZP1` and `UZP2`.
    fn horizontal_pair(&mut self, a: Value, b: Value, bits: u32) -> (Value, Value) {
        let even = Self::lane_shuffle_mask(bits, |j| 2 * j);
        let odd = Self::lane_shuffle_mask(bits, |j| 2 * j + 1);
        (self.vec_shuffle(a, b, even), self.vec_shuffle(a, b, odd))
    }

    /// `ROUNDPS` / `ROUNDPD` as one packed rounding instruction.
    ///
    /// `FRINTN`/`FRINTM`/`FRINTP`/`FRINTZ` on aarch64, `ROUNDPS`/`ROUNDPD` on
    /// x86-64, against a portable path that extracts each lane from its 64-bit
    /// half, bitcasts it to a float, rounds it and packs it back.
    ///
    /// No x86 NaN fixup is needed and none is emitted: rounding never creates a
    /// NaN from a non-NaN operand, and a NaN input is quieted and propagated
    /// identically by every backend here - which is exactly why the portable
    /// path calls the same [`apply_round`] with no fixup either.
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

    /// `HADDPS` / `HSUBPS` / `HADDPD` / `HSUBPD` - horizontal float add and
    /// subtract.
    ///
    /// Two constant shuffles and the ordinary packed op. The op goes through
    /// [`vec_fp_arith_x86`](Lifter::vec_fp_arith_x86) for the same reason the
    /// non-horizontal forms do: a lane whose result is a NaN that neither
    /// operand supplied performed an invalid operation and owes x86's sign-set
    /// indefinite, not the host's default NaN.
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
    ///
    /// The plain adds are literally `iadd_pairwise`, whose definition -
    /// first operand's pairwise sums in the low half, second's in the high -
    /// is `PHADD*`'s definition, so they become a single `ADDP` (aarch64) or
    /// `PHADDW`/`PHADDD` (x86-64). The other three have no pairwise opcode and
    /// reuse the even/odd gather, which costs one extra shuffle and still
    /// lowers to three instructions against a sixteen-lane scalar expansion.
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
    ///
    /// Written in the exact shape both backends pattern-match: a widening
    /// multiply of each half, then `iadd_pairwise`. x86-64 folds the entire
    /// tree back into one `pmaddwd`; aarch64 gives `SMULL`, `SMULL2`, `ADDP`.
    ///
    /// The one input pair that overflows a dword - `-32768 * -32768` twice -
    /// wraps, because `imul` on `I32X4` wraps. That is what hardware does, and
    /// it is why this cannot use a saturating narrow.
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

    /// `PMADDUBSW` - unsigned destination bytes times signed source bytes,
    /// adjacent products summed into words with signed saturation.
    ///
    /// Cranelift has an exact opcode for this - `x86_pmaddubsw` - and it is
    /// deliberately not used: its x86-64 rule is gated on `has_ssse3`, which is
    /// **off** in the baseline `x86_64` feature set, so an AOT caller that did
    /// not pass `isa_settings` would get a hard lowering failure rather than a
    /// slow path. The sequence below lowers on every backend:
    ///
    /// 1. The byte products are computed in `I16X8`, which is exact - a
    ///    `u8 * i8` spans `-32640..=32385` and cannot overflow a word.
    /// 2. Each product vector is widened and pairwise-added in `I32X4`, in the
    ///    `iadd_pairwise(swiden_low v, swiden_high v)` shape aarch64 lowers to
    ///    a single `SADDLP`. The sum is computed at 32 bits precisely so that
    ///    saturation happens once, at the end, rather than wrapping first.
    /// 3. `snarrow` saturates the two `I32X4` halves back into one `I16X8`,
    ///    destination results low - which is the `SQXTN`/`SQXTN2` pair.
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

        // Widening pairwise add, at 32 bits so nothing wraps before the
        // saturating narrow below decides what to clamp.
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

    /// `PSADBW` - sum of absolute byte differences, one sum per 64-bit half,
    /// deposited in that half's low word with the rest of the half zeroed.
    ///
    /// The absolute difference of two *unsigned* bytes is
    /// `usub_sat(a, b) | usub_sat(b, a)`: exactly one of the two saturating
    /// subtracts is non-zero (both are zero when the bytes are equal), so the
    /// OR is a select with no compare. Three instructions, everywhere.
    ///
    /// The reduction is then a ladder of widening pairwise adds - byte pairs to
    /// words (`UADDLP`), word pairs to dwords (`UADDLP`), dword pairs to the
    /// two half-sums (`ADDP`). The final stage stays in `I32X4` rather than
    /// widening to `I64X2` first, because `iadd_pairwise` is not defined on
    /// 64-bit lanes and because the sum cannot exceed `16 * 255`, so 32 bits is
    /// ample; `uwiden_low` then both selects the two live lanes and supplies
    /// the zero upper halves the instruction requires.
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

        // Each stage halves the lane count and doubles the width, so no partial
        // sum can overflow the lane holding it.
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
    ///
    /// The portable path masks or sign-extends each 64-bit lane down to its
    /// even dword and then does a full `I64X2` `imul` - an operation aarch64
    /// has no instruction for at all, and which Cranelift expands into a
    /// seven-instruction `rev64`/`mul`/`xtn`/`addp`/`shll`/`umlal` sequence.
    ///
    /// Instead, gather each operand's even dwords into its low half with a
    /// `UZP1`-shaped shuffle, and multiply the widened low halves. Written that
    /// way - both operands widened on the *same* side - the backend recognizes
    /// a widening multiply and emits `UMULL`/`SMULL`, which is precisely this
    /// instruction. Three instructions in total.
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

        // Shuffling a value against itself keeps this a one-operand permute;
        // only the low two result lanes are read afterwards.
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

    /// `PMULHRSW` - signed word multiply keeping the rounded high 17 bits,
    /// `(((a * b) >> 14) + 1) >> 1`.
    ///
    /// aarch64 only. Cranelift has an exact `x86_pmulhrsw` opcode, but - like
    /// `x86_pmaddubsw` - its x86-64 rule is gated on `has_ssse3`, which the
    /// baseline `x86_64` feature set leaves off; and the portable expansion is
    /// already three instructions there, so there is nothing to win.
    ///
    /// On aarch64 this is `SQRDMULH` plus a fixup. Cranelift's
    /// `sqmul_round_sat` computes `signed_saturate((x*y + 0x4000) >> 15)`,
    /// which is the same arithmetic - but x86 *truncates* the result to 16 bits
    /// where `SQRDMULH` *saturates*, and there is exactly one input pair where
    /// that shows: `-32768 * -32768` gives `0x8000` (`-32768`) on x86 and
    /// `0x7FFF` after saturation. Both operand lanes must be the minimum for it
    /// to arise, so the fixup is two compares, an AND and a select - and the
    /// whole sequence is still shorter than the portable widen/shift/round/narrow
    /// expansion it replaces.
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
