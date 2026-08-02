//! SSSE3 / SSE4.1 packed-integer semantics.
//!
//! These are the byte/word/dword lane operations a vectorizing compiler emits
//! around `PSHUFB` and `PMULUDQ`: cross-lane shuffles and rotates, horizontal
//! and widening arithmetic, the pack/unpack narrowing pairs, lane insert and
//! extract, sign/zero-extending widening, blends and `PTEST`.
//!
//! The portable expansion here is expressed on the `(lo, hi)` 64-bit half pair,
//! which [`Lifter::get_xmm_lo`](crate::lifter::Lifter::get_xmm_lo) and friends
//! provide under both XMM backing representations. The lanes are widened into
//! full `I64`s ([`Lifter::xmm_lanes`]), operated on there - where nothing can
//! overflow at the widths x86 defines these ops - and repacked
//! ([`Lifter::pack_xmm_lanes`]). Cranelift folds the resulting mask/shift/OR
//! chains, so the emitted IR is far larger than the machine code it lowers to.
//!
//! Most handlers additionally take a native fast path when XMM registers are
//! vector-backed, since these expansions run to hundreds of IR instructions for
//! something the host does in one. `PMADDWD`, `PMADDUBSW`, `PSADBW` and the
//! horizontal `PHADD*` / `PHSUB*` forms deliberately have none: their
//! cross-lane accumulation has no portable Cranelift expansion that beats the
//! scalar one. See [`sse_vec`](super::sse_vec).

use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::arch::x86_64::semantics::sse::PackedWidth;
use crate::arch::x86_64::semantics::sse_vec::B;
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

/// Which horizontal (adjacent-pair) operation a `PHADD*` / `PHSUB*` form
/// performs.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum HorizOp {
    /// `PHADDW` / `PHADDD` - wrapping add of adjacent lanes.
    Add,
    /// `PHSUBW` / `PHSUBD` - wrapping subtract of adjacent lanes.
    Sub,
    /// `PHADDSW` - add of adjacent word lanes, saturated to `i16`.
    AddSat,
    /// `PHSUBSW` - subtract of adjacent word lanes, saturated to `i16`.
    SubSat,
}

impl<'a> Lifter<'a> {
    //  Shared lane plumbing.

    /// Split a 128-bit value (as a `(lo, hi)` pair) into its `bits`-wide lanes,
    /// each widened into a full `I64`, lowest lane first.
    pub(super) fn xmm_lanes(
        &mut self,
        lo: Value,
        hi: Value,
        bits: u32,
        signed: bool,
    ) -> Vec<Value> {
        let per_half = 64 / bits;
        let mut out = Vec::with_capacity((per_half * 2) as usize);
        for v in [lo, hi] {
            for lane in 0..per_half {
                out.push(self.extract_lane(v, i64::from(lane * bits), bits, signed));
            }
        }
        out
    }

    /// Repack `128 / bits` widened lanes into a `(lo, hi)` pair, keeping the low
    /// `bits` of each.
    pub(super) fn pack_xmm_lanes(&mut self, lanes: &[Value], bits: u32) -> (Value, Value) {
        debug_assert_eq!(lanes.len(), (128 / bits) as usize);
        let per_half = (64 / bits) as usize;
        let mask = lane_mask(bits);
        let zero = self.builder.ins().iconst(types::I64, 0);
        let mut halves = [zero, zero];
        for (i, &lane) in lanes.iter().enumerate() {
            let shift = ((i % per_half) as u32 * bits) as i64;
            let masked = self.builder.ins().band_imm_u(lane, mask);
            let placed = if shift == 0 {
                masked
            } else {
                self.builder.ins().ishl_imm_u(masked, shift)
            };
            halves[i / per_half] = self.builder.ins().bor(halves[i / per_half], placed);
        }
        (halves[0], halves[1])
    }

    /// Clamp a widened lane to the signed range of a `bits`-wide lane.
    fn saturate_signed(&mut self, v: Value, bits: u32) -> Value {
        let lo = self.builder.ins().iconst(types::I64, -(1i64 << (bits - 1)));
        let hi = self
            .builder
            .ins()
            .iconst(types::I64, (1i64 << (bits - 1)) - 1);
        let clamped = self.builder.ins().smax(v, lo);
        self.builder.ins().smin(clamped, hi)
    }

    /// Clamp a widened (signed) lane to the unsigned range of a `bits`-wide
    /// lane - the saturation the `PACKUS*` forms use.
    fn saturate_unsigned(&mut self, v: Value, bits: u32) -> Value {
        let zero = self.builder.ins().iconst(types::I64, 0);
        let hi = self.builder.ins().iconst(types::I64, lane_mask(bits));
        let clamped = self.builder.ins().smax(v, zero);
        self.builder.ins().smin(clamped, hi)
    }

    /// Read the low `bits` of a non-XMM source operand (a GPR or a sized memory
    /// location) as an `I64`. Used by the lane insert forms and by the widening
    /// `PMOVZX*` / `PMOVSX*`, whose memory operands are *narrower* than 128 bits
    /// and so must not be read with a full `m128` load.
    fn read_scalar_src(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        bits: u32,
        who: &str,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                let val = self.get_reg_val(reg)?;
                let val = if self.builder.func.dfg.value_type(val) == types::I64 {
                    val
                } else {
                    self.builder.ins().uextend(types::I64, val)
                };
                Ok(self.builder.ins().band_imm_u(val, lane_mask(bits)))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                let flags = MemFlagsData::new();
                Ok(match bits {
                    8 => self.builder.ins().uload8(types::I64, flags, ptr, 0),
                    16 => self.builder.ins().uload16(types::I64, flags, ptr, 0),
                    32 => self.builder.ins().uload32(flags, ptr, 0),
                    _ => self.builder.ins().load(types::I64, flags, ptr, 0),
                })
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-src-{:?}", who, other),
            }),
        }
    }

    //  Shuffles and rotates.

    /// PSHUFB - shuffle the destination's bytes by a per-byte index vector.
    ///
    /// Output byte `i` is `dst[ctl[i] & 0x0F]`, or zero when `ctl[i]`'s high bit
    /// is set. Note the indices select from the *destination*: this is a full
    /// cross-lane permutation, not a lane-wise op.
    ///
    /// Each output byte is produced by shifting the selected half down by
    /// `(index & 7) * 8` and selecting between the two halves on bit 3, so the
    /// expansion is a handful of ops per byte rather than a 16-way select tree.
    pub(super) fn handle_pshufb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pshufb")?;

        if self.vec_mode() {
            let dst = self.get_xmm_vec(dst_idx);
            let ctl = self.read_xmm_src_vec(instruction, 1)?;
            let dst = self.vec_bitcast(dst, types::I8X16);
            let ctl = self.vec_bitcast(ctl, types::I8X16);
            // Cranelift's `swizzle` zeroes any index outside 0..15, whereas x86
            // only zeroes on the index's high bit and otherwise selects with
            // `index & 0x0F`. Masking with 0x8F reconciles the two: it folds the
            // don't-care bits 4..6 away and leaves the zeroing indices as
            // 0x80..0x8F, which are out of `swizzle`'s range.
            let keep = self.builder.ins().iconst(types::I8, 0x8F);
            let keep = self.builder.ins().splat(types::I8X16, keep);
            let ctl = self.builder.ins().band(ctl, keep);
            let r = self.builder.ins().swizzle(dst, ctl);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (ctl_lo, ctl_hi) = self.read_xmm_src(instruction, 1)?;

        let zero = self.builder.ins().iconst(types::I64, 0);
        let mut bytes = Vec::with_capacity(16);
        for i in 0..16u32 {
            let ctl = if i < 8 { ctl_lo } else { ctl_hi };
            let index = self.extract_lane(ctl, i64::from((i % 8) * 8), 8, false);

            let byte_shift = self.builder.ins().band_imm_u(index, 7);
            let bit_shift = self.builder.ins().ishl_imm_u(byte_shift, 3);
            let from_lo = self.builder.ins().ushr(dst_lo, bit_shift);
            let from_lo = self.builder.ins().band_imm_u(from_lo, 0xFF);
            let from_hi = self.builder.ins().ushr(dst_hi, bit_shift);
            let from_hi = self.builder.ins().band_imm_u(from_hi, 0xFF);

            // Bit 3 of the index picks the half; bit 7 forces a zero byte.
            let pick_hi_bit = self.builder.ins().band_imm_u(index, 8);
            let pick_hi = self.builder.ins().icmp_imm_s(IntCC::NotEqual, pick_hi_bit, 0);
            let byte = self.builder.ins().select(pick_hi, from_hi, from_lo);
            let clear_bit = self.builder.ins().band_imm_u(index, 0x80);
            let clear = self.builder.ins().icmp_imm_s(IntCC::NotEqual, clear_bit, 0);
            bytes.push(self.builder.ins().select(clear, zero, byte));
        }

        let (res_lo, res_hi) = self.pack_xmm_lanes(&bytes, 8);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PALIGNR - byte-granular funnel shift of `dst:src` (a 256-bit temporary
    /// with the destination in the high half) right by `imm8` bytes.
    ///
    /// The count is a lift-time constant, so this is a fixed pair of 64-bit
    /// funnel extractions: counts of 16..31 shift the destination down into the
    /// result, and 32 or more clear it.
    pub(super) fn handle_palignr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "palignr")?;
        let count = u32::from(instruction.immediate8());

        if self.vec_mode() {
            // The 256-bit temporary is `src` in bytes 0-15 and `dst` in bytes
            // 16-31, so for a count below 16 the window spans both operands and
            // the mask is simply `i + count`. From 16 up it has left `src`
            // entirely, so the operands become `dst` and a zero vector.
            let r = if count >= 32 {
                self.vec_zero()
            } else if count < 16 {
                let src = self.read_xmm_src_vec(instruction, 1)?;
                let dst = self.get_xmm_vec(dst_idx);
                let mut mask = [0u8; 16];
                for (i, m) in mask.iter_mut().enumerate() {
                    *m = (i as u32 + count) as u8;
                }
                self.vec_shuffle(src, dst, mask)
            } else {
                let dst = self.get_xmm_vec(dst_idx);
                let zero = self.vec_zero();
                let mut mask = [B; 16];
                for (i, m) in mask.iter_mut().enumerate() {
                    let from = i as u32 + count - 16;
                    if from < 16 {
                        *m = from as u8;
                    }
                }
                self.vec_shuffle(dst, zero, mask)
            };
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (res_lo, res_hi) = if count >= 32 {
            let zero = self.builder.ins().iconst(types::I64, 0);
            (zero, zero)
        } else {
            // The 256-bit stream, low word first.
            let words = [src_lo, src_hi, dst_lo, dst_hi];
            let bit = count * 8;
            (
                self.funnel_word(&words, bit),
                self.funnel_word(&words, bit + 64),
            )
        };

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Extract the 64 bits starting at bit offset `bit` of the little-endian
    /// word stream `words`, zero-filling past its end.
    fn funnel_word(&mut self, words: &[Value; 4], bit: u32) -> Value {
        let index = (bit / 64) as usize;
        let shift = i64::from(bit % 64);
        let low = match words.get(index) {
            Some(&w) if shift == 0 => return w,
            Some(&w) => self.builder.ins().ushr_imm_u(w, shift),
            None => return self.builder.ins().iconst(types::I64, 0),
        };
        match words.get(index + 1) {
            Some(&w) => {
                let high = self.builder.ins().ishl_imm_u(w, 64 - shift);
                self.builder.ins().bor(low, high)
            }
            None => low,
        }
    }

    //  Unary and sign-directed lane ops.

    /// PABSB / PABSW / PABSD - per-lane absolute value of the source.
    ///
    /// `abs` of the most negative lane value wraps back to itself, which is what
    /// hardware does (and what the mask-to-lane-width repack gives for free).
    pub(super) fn handle_pabs(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pabs")?;

        if self.vec_mode() {
            let ty = width.vector_type();
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let a = self.vec_bitcast(src, ty);
            // `iabs` wraps the most negative lane back to itself, which is what
            // hardware does.
            let r = self.builder.ins().iabs(a);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let bits = width.bits();
        let lanes = self.xmm_lanes(src_lo, src_hi, bits, true);

        let out: Vec<Value> = lanes
            .into_iter()
            .map(|x| {
                let neg = self.builder.ins().ineg(x);
                let is_neg = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, x, 0);
                self.builder.ins().select(is_neg, neg, x)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PSIGNB / PSIGNW / PSIGND - apply the sign of each source lane to the
    /// corresponding destination lane: negate it when the source lane is
    /// negative, zero it when the source lane is zero, leave it otherwise.
    pub(super) fn handle_psign(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "psign")?;

        if self.vec_mode() {
            let ty = width.vector_type();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let a = self.vec_bitcast(dst, ty);
            let s = self.vec_bitcast(src, ty);
            // Negate where the source lane is negative, then zero where it is
            // zero. Vector `icmp` yields an all-ones/all-zeros lane mask, so the
            // two choices are `bitselect`s.
            let neg = self.builder.ins().ineg(a);
            let zero = self.builder.ins().iconst(ty.lane_type(), 0);
            let zero = self.builder.ins().splat(ty, zero);
            let is_neg = self.builder.ins().icmp(IntCC::SignedLessThan, s, zero);
            let signed = self.builder.ins().bitselect(is_neg, neg, a);
            let is_zero = self.builder.ins().icmp(IntCC::Equal, s, zero);
            let r = self.builder.ins().bitselect(is_zero, zero, signed);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let bits = width.bits();
        let a = self.xmm_lanes(dst_lo, dst_hi, bits, true);
        let b = self.xmm_lanes(src_lo, src_hi, bits, true);

        let zero = self.builder.ins().iconst(types::I64, 0);
        let out: Vec<Value> = a
            .into_iter()
            .zip(b)
            .map(|(x, s)| {
                let neg = self.builder.ins().ineg(x);
                let is_neg = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, s, 0);
                let signed = self.builder.ins().select(is_neg, neg, x);
                let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, s, 0);
                self.builder.ins().select(is_zero, zero, signed)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    //  Horizontal and widening arithmetic.

    /// PHADDW / PHADDD / PHSUBW / PHSUBD and the saturating PHADDSW / PHSUBSW -
    /// combine *adjacent* lane pairs, the destination's pairs filling the low
    /// half of the result and the source's the high half.
    pub(super) fn handle_horizontal(
        &mut self,
        instruction: &Instruction,
        op: HorizOp,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "phadd")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let bits = width.bits();
        let a = self.xmm_lanes(dst_lo, dst_hi, bits, true);
        let b = self.xmm_lanes(src_lo, src_hi, bits, true);

        let mut out = Vec::with_capacity(a.len());
        for lanes in [&a, &b] {
            for pair in lanes.chunks_exact(2) {
                let (x, y) = (pair[0], pair[1]);
                let r = match op {
                    HorizOp::Add | HorizOp::AddSat => self.builder.ins().iadd(x, y),
                    HorizOp::Sub | HorizOp::SubSat => self.builder.ins().isub(x, y),
                };
                out.push(match op {
                    HorizOp::AddSat | HorizOp::SubSat => self.saturate_signed(r, bits),
                    _ => r,
                });
            }
        }

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PMULUDQ / PMULDQ - multiply the *even* dword lanes into full 64-bit
    /// products (`PMULUDQ` unsigned, `PMULDQ` signed); the odd dwords of both
    /// operands are ignored.
    pub(super) fn handle_pmuldq(
        &mut self,
        instruction: &Instruction,
        signed: bool,
    ) -> crate::Result<()> {
        let who = if signed { "pmuldq" } else { "pmuludq" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.vec_mode() {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // Reduce each 64-bit lane to its even (low) dword, extended to the
            // full lane, then a plain 64-bit lane multiply is the product.
            let ext = |s: &mut Self, v: Value| {
                let v = s.vec_bitcast(v, types::I64X2);
                if signed {
                    let up = s.builder.ins().ishl_imm_u(v, 32);
                    s.builder.ins().sshr_imm_u(up, 32)
                } else {
                    let m = s.builder.ins().iconst(types::I64, 0xFFFF_FFFF);
                    let m = s.builder.ins().splat(types::I64X2, m);
                    s.builder.ins().band(v, m)
                }
            };
            let a = ext(self, dst);
            let b = ext(self, src);
            let r = self.builder.ins().imul(a, b);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut products = Vec::with_capacity(2);
        for (a, b) in [(dst_lo, src_lo), (dst_hi, src_hi)] {
            let x = self.extract_lane(a, 0, 32, signed);
            let y = self.extract_lane(b, 0, 32, signed);
            products.push(self.builder.ins().imul(x, y));
        }

        self.set_xmm_lo(dst_idx, products[0]);
        self.set_xmm_hi(dst_idx, products[1]);
        Ok(())
    }

    /// PMADDWD - multiply signed word lanes pairwise and add adjacent products
    /// into signed dword lanes (the vector dot-product primitive).
    ///
    /// The one input pair that overflows a dword, `-32768 * -32768` twice, wraps
    /// - matching hardware, which does not saturate here.
    pub(super) fn handle_pmaddwd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmaddwd")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let a = self.xmm_lanes(dst_lo, dst_hi, 16, true);
        let b = self.xmm_lanes(src_lo, src_hi, 16, true);

        let out: Vec<Value> = (0..4)
            .map(|j| {
                let p0 = self.builder.ins().imul(a[2 * j], b[2 * j]);
                let p1 = self.builder.ins().imul(a[2 * j + 1], b[2 * j + 1]);
                self.builder.ins().iadd(p0, p1)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, 32);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PMADDUBSW - multiply *unsigned* destination bytes by *signed* source
    /// bytes and add adjacent products into signed word lanes, saturating.
    pub(super) fn handle_pmaddubsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        // No vector fast path: the pairwise multiply-accumulate has no portable
        // Cranelift expansion (`x86_pmaddubsw` exists but lowers on x86 only),
        // and open-coding the cross-lane adds is no cheaper than the scalar
        // lane expansion below.
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmaddubsw")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let a = self.xmm_lanes(dst_lo, dst_hi, 8, false);
        let b = self.xmm_lanes(src_lo, src_hi, 8, true);

        let out: Vec<Value> = (0..8)
            .map(|j| {
                let p0 = self.builder.ins().imul(a[2 * j], b[2 * j]);
                let p1 = self.builder.ins().imul(a[2 * j + 1], b[2 * j + 1]);
                let sum = self.builder.ins().iadd(p0, p1);
                self.saturate_signed(sum, 16)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, 16);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PMULHRSW - signed word multiply keeping the high 17 bits, rounded:
    /// `(((a * b) >> 14) + 1) >> 1`.
    pub(super) fn handle_pmulhrsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pmulhrsw")?;

        if self.vec_mode() {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let r = self.vec_mulhrsw(dst, src);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let a = self.xmm_lanes(dst_lo, dst_hi, 16, true);
        let b = self.xmm_lanes(src_lo, src_hi, 16, true);

        let out: Vec<Value> = a
            .into_iter()
            .zip(b)
            .map(|(x, y)| {
                let p = self.builder.ins().imul(x, y);
                let shifted = self.builder.ins().sshr_imm_u(p, 14);
                let rounded = self.builder.ins().iadd_imm_s(shifted, 1);
                self.builder.ins().sshr_imm_u(rounded, 1)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, 16);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PSADBW - sum of absolute byte differences, one sum per 64-bit half,
    /// deposited in that half's low word with the rest of the half zeroed.
    pub(super) fn handle_psadbw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "psadbw")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut sums = Vec::with_capacity(2);
        for (a, b) in [(dst_lo, src_lo), (dst_hi, src_hi)] {
            let mut acc = self.builder.ins().iconst(types::I64, 0);
            for lane in 0..8 {
                let shift = i64::from(lane * 8);
                let x = self.extract_lane(a, shift, 8, false);
                let y = self.extract_lane(b, shift, 8, false);
                let diff = self.builder.ins().isub(x, y);
                let neg = self.builder.ins().ineg(diff);
                let is_neg = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, diff, 0);
                let abs = self.builder.ins().select(is_neg, neg, diff);
                acc = self.builder.ins().iadd(acc, abs);
            }
            sums.push(acc);
        }

        self.set_xmm_lo(dst_idx, sums[0]);
        self.set_xmm_hi(dst_idx, sums[1]);
        Ok(())
    }

    //  Narrowing and widening.

    /// PACKSSWB / PACKSSDW / PACKUSWB / PACKUSDW - narrow the destination's
    /// lanes into the low half of the result and the source's into the high
    /// half, saturating each to the signed (`PACKSS*`) or unsigned (`PACKUS*`)
    /// range of the half-width lane. Source lanes are always read as signed.
    pub(super) fn handle_pack(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
        signed: bool,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pack")?;

        if self.vec_mode() {
            let ty = width.vector_type();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let a = self.vec_bitcast(dst, ty);
            let b = self.vec_bitcast(src, ty);
            // `snarrow`/`unarrow` concatenate with the first operand's narrowed
            // lanes low, which is where the destination's belong. Both read
            // their inputs as signed, matching the x86 `PACKUS*` definition.
            let r = if signed {
                self.builder.ins().snarrow(a, b)
            } else {
                self.builder.ins().unarrow(a, b)
            };
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let bits = width.bits();
        let a = self.xmm_lanes(dst_lo, dst_hi, bits, true);
        let b = self.xmm_lanes(src_lo, src_hi, bits, true);

        let out_bits = bits / 2;
        let mut out = Vec::with_capacity(a.len() * 2);
        for lane in a.into_iter().chain(b) {
            out.push(if signed {
                self.saturate_signed(lane, out_bits)
            } else {
                self.saturate_unsigned(lane, out_bits)
            });
        }

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, out_bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PMOVZX* / PMOVSX* - widen the low `128 * src_bits / dst_bits` bits of the
    /// source into full-width lanes, zero- or sign-extending.
    ///
    /// The memory forms read only the bytes they consume (`m16`/`m32`/`m64`),
    /// never a full `m128`.
    pub(super) fn handle_pmovx(
        &mut self,
        instruction: &Instruction,
        src_bits: u32,
        dst_bits: u32,
        signed: bool,
    ) -> crate::Result<()> {
        let who = if signed { "pmovsx" } else { "pmovzx" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let lanes = 128 / dst_bits;

        if self.vec_mode() {
            // Both widening ops read only the low half of their input, so the
            // consumed bytes just have to land there - which is where a register
            // source already has them, and where `scalar_to_vector` puts a
            // narrow memory load.
            let v = match self.operand_kind(instruction, 1) {
                OpKind::Register => {
                    let src_idx = self.xmm_operand_index(instruction, 1, who)?;
                    self.get_xmm_vec(src_idx)
                }
                _ => {
                    let packed = self.read_scalar_src(instruction, 1, lanes * src_bits, who)?;
                    self.builder.ins().scalar_to_vector(types::I64X2, packed)
                }
            };

            // A 4x or 8x widening is a chain of 2x steps.
            let mut v = self.vec_bitcast(v, Self::vec_int_type(src_bits));
            let mut bits = src_bits;
            while bits < dst_bits {
                v = if signed {
                    self.builder.ins().swiden_low(v)
                } else {
                    self.builder.ins().uwiden_low(v)
                };
                bits *= 2;
            }
            let r = self.vec_bitcast(v, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let packed = match self.operand_kind(instruction, 1) {
            // A register source keeps every consumed byte in its low half.
            OpKind::Register => {
                let src_idx = self.xmm_operand_index(instruction, 1, who)?;
                self.get_xmm_lo(src_idx)
            }
            _ => self.read_scalar_src(instruction, 1, lanes * src_bits, who)?,
        };

        let out: Vec<Value> = (0..lanes)
            .map(|j| self.extract_lane(packed, i64::from(j * src_bits), src_bits, signed))
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, dst_bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    //  Lane insert / extract.

    /// PINSRB / PINSRW / PINSRD / PINSRQ - insert a scalar into one lane of an
    /// XMM register, leaving every other lane untouched.
    ///
    /// `imm8`'s low `log2(lane count)` bits select the lane; a GPR source
    /// contributes only its low `bits`, and a memory source is a sized load.
    pub(super) fn handle_pinsr(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pinsr")?;
        let bits = width.bits();
        let value = self.read_scalar_src(instruction, 1, bits, "pinsr")?;

        let lanes = 128 / bits;
        let sel = u32::from(instruction.immediate8()) & (lanes - 1);
        let high_half = sel * bits >= 64;
        let shift = i64::from((sel * bits) % 64);

        let half = if high_half {
            self.get_xmm_hi(dst_idx)
        } else {
            self.get_xmm_lo(dst_idx)
        };
        let cleared = self
            .builder
            .ins()
            .band_imm_u(half, !(lane_mask(bits) << shift));
        let placed = if shift == 0 {
            value
        } else {
            self.builder.ins().ishl_imm_u(value, shift)
        };
        let merged = self.builder.ins().bor(cleared, placed);

        if high_half {
            self.set_xmm_hi(dst_idx, merged);
        } else {
            self.set_xmm_lo(dst_idx, merged);
        }
        Ok(())
    }

    /// PEXTRB / PEXTRW / PEXTRD / PEXTRQ - extract one lane of an XMM register
    /// into a GPR (zero-extended) or a sized memory location.
    pub(super) fn handle_pextr(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let src_idx = self.xmm_operand_index(instruction, 1, "pextr")?;
        let bits = width.bits();

        let lanes = 128 / bits;
        let sel = u32::from(instruction.immediate8()) & (lanes - 1);
        let half = if sel * bits >= 64 {
            self.get_xmm_hi(src_idx)
        } else {
            self.get_xmm_lo(src_idx)
        };
        let value = self.extract_lane(half, i64::from((sel * bits) % 64), bits, false);

        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                // The narrower forms always zero-extend into the full GPR, which
                // is what writing the 32-bit register already does.
                let ty = size_to_type(dst_reg.size());
                let narrowed = self.narrow_to(ty, value);
                self.set_reg_val(dst_reg, narrowed)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let flags = MemFlagsData::new();
                match bits {
                    8 => self.builder.ins().istore8(flags, value, ptr, 0),
                    16 => self.builder.ins().istore16(flags, value, ptr, 0),
                    32 => self.builder.ins().istore32(flags, value, ptr, 0),
                    _ => self.builder.ins().store(flags, value, ptr, 0),
                };
                Ok(())
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("pextr-dst-{:?}", other),
            }),
        }
    }

    //  Blends and PTEST.

    /// PBLENDW / BLENDPS / BLENDPD - select each lane from the source or the
    /// destination according to a bit of `imm8`.
    ///
    /// The selector is a lift-time constant, so this collapses to a mask, an
    /// and-not and an or.
    pub(super) fn handle_blend_imm(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "blend")?;

        if self.vec_mode() {
            // A lift-time constant lane selection between two operands is just
            // a constant shuffle.
            let imm = u32::from(instruction.immediate8());
            let lanes = 128 / width.bits();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let mask = Self::lane_shuffle_mask(width.bits(), |j| {
                if imm & (1 << j) != 0 { lanes + j } else { j }
            });
            let r = self.vec_shuffle(dst, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let bits = width.bits();
        let lanes = 128 / bits;
        let imm = u32::from(instruction.immediate8());
        let per_half = 64 / bits;
        let mut masks = [0u64; 2];
        for lane in 0..lanes {
            if imm & (1 << lane) != 0 {
                let bit = (lane % per_half) * bits;
                masks[(lane / per_half) as usize] |= (lane_mask(bits) as u64) << bit;
            }
        }

        let mut out = Vec::with_capacity(2);
        for (half, (d, s)) in [(dst_lo, src_lo), (dst_hi, src_hi)].into_iter().enumerate() {
            let mask = masks[half] as i64;
            let kept = self.builder.ins().band_imm_u(d, !mask);
            let taken = self.builder.ins().band_imm_u(s, mask);
            out.push(self.builder.ins().bor(kept, taken));
        }

        self.set_xmm_lo(dst_idx, out[0]);
        self.set_xmm_hi(dst_idx, out[1]);
        Ok(())
    }

    /// PBLENDVB / BLENDVPS / BLENDVPD - select each lane from the source or the
    /// destination according to the *sign bit* of the corresponding lane of the
    /// implicit `XMM0` mask.
    pub(super) fn handle_blendv(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "blendv")?;

        if self.vec_mode() {
            let ty = width.vector_type();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let mask = self.get_xmm_vec(0);
            let dst = self.vec_bitcast(dst, ty);
            let src = self.vec_bitcast(src, ty);
            let mask = self.vec_bitcast(mask, ty);
            // `bitselect` is bit-granular but x86 selects per lane on the mask
            // lane's *sign bit*, so broadcast that bit across its lane first.
            let bits = i64::from(width.bits()) - 1;
            let mask = self.builder.ins().sshr_imm_u(mask, bits);
            let r = self.builder.ins().bitselect(mask, src, dst);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        // The mask operand is always XMM0, implicit in the legacy encoding.
        let mask_lo = self.get_xmm_lo(0);
        let mask_hi = self.get_xmm_hi(0);

        let bits = width.bits();
        let d = self.xmm_lanes(dst_lo, dst_hi, bits, false);
        let s = self.xmm_lanes(src_lo, src_hi, bits, false);
        let m = self.xmm_lanes(mask_lo, mask_hi, bits, true);

        let out: Vec<Value> = d
            .into_iter()
            .zip(s)
            .zip(m)
            .map(|((x, y), mask)| {
                let take_src = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, mask, 0);
                self.builder.ins().select(take_src, y, x)
            })
            .collect();

        let (res_lo, res_hi) = self.pack_xmm_lanes(&out, bits);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PTEST - the 128-bit bitwise test: `ZF = (src & dst) == 0`,
    /// `CF = (src & ~dst) == 0`, and AF/OF/PF/SF cleared. Neither operand is
    /// written.
    pub(super) fn handle_ptest(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "ptest")?;

        if self.vec_mode() {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // `vany_true` is a whole-register "is any bit set" reduction, which
            // is exactly the test both flags need - negated.
            let and = self.builder.ins().band(src, dst);
            let any = self.builder.ins().vany_true(and);
            let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, any, 0);
            // band_not(x, y) = x & ~y.
            let andn = self.builder.ins().band_not(src, dst);
            let any_n = self.builder.ins().vany_true(andn);
            let cf = self.builder.ins().icmp_imm_s(IntCC::Equal, any_n, 0);
            return self.write_ptest_flags(zf, cf);
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let and_lo = self.builder.ins().band(src_lo, dst_lo);
        let and_hi = self.builder.ins().band(src_hi, dst_hi);
        let and_all = self.builder.ins().bor(and_lo, and_hi);
        let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, and_all, 0);

        // band_not(x, y) = x & ~y.
        let andn_lo = self.builder.ins().band_not(src_lo, dst_lo);
        let andn_hi = self.builder.ins().band_not(src_hi, dst_hi);
        let andn_all = self.builder.ins().bor(andn_lo, andn_hi);
        let cf = self.builder.ins().icmp_imm_s(IntCC::Equal, andn_all, 0);
        self.write_ptest_flags(zf, cf)
    }

    /// Publish `PTEST`'s flag result: ZF and CF as computed, AF/OF/PF/SF cleared.
    fn write_ptest_flags(&mut self, zf: Value, cf: Value) -> crate::Result<()> {
        self.set_flag(X86Flag::Zero, zf)?;
        self.set_flag(X86Flag::Carry, cf)?;
        let zero = self.builder.ins().iconst(types::I8, 0);
        for flag in [
            X86Flag::Auxiliary,
            X86Flag::Overflow,
            X86Flag::Parity,
            X86Flag::Sign,
        ] {
            self.set_flag(flag, zero)?;
        }
        Ok(())
    }
}

/// A mask with the low `bits` set (`bits <= 64`).
fn lane_mask(bits: u32) -> i64 {
    match bits {
        64 => -1,
        bits => ((1u64 << bits) - 1) as i64,
    }
}
