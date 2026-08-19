//! The eager flag updates, and the guards that stop them being eager.
//!
//! Every `update_flags_*` assigns a fixed set of flags, and that set is what
//! [`crate::arch::x86_64::flag_effects`] classifies each mnemonic as writing -
//! the two must agree, and the table is written against *this file*.
//!
//! Each flag's computation is wrapped in [`Lifter::flag_needed`], a constant
//! `true` unless [`LifterConfig::eliminate_dead_flags`] is on. Parity is what
//! motivated it: a `popcnt`, an `ireduce`, a `band` and a `bxor` on every
//! arithmetic instruction in the program, for a flag almost nothing reads.
//!
//! [`LifterConfig::eliminate_dead_flags`]: crate::core::LifterConfig::eliminate_dead_flags
//! [`Spec::IntegerOverflowFlags`]: crate::core::specialize::Spec::IntegerOverflowFlags

use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// PF: set when the low byte of `res` has an even population count.
    ///
    /// Four IR instructions, emitted once per arithmetic instruction by five
    /// different callers, which is why it is worth both factoring out and
    /// skipping.
    fn parity_of(&mut self, res: Value) -> Value {
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm_u(pop, 1);
        self.builder.ins().bxor_imm_u(pop_lsb, 1)
    }

    /// AF: the carry out of bit 3, which is `((lhs ^ rhs ^ res) >> 4) & 1` for
    /// every add- and subtract-shaped operation, carry-in included - the carry
    /// only reaches bit 0, and the three-way xor still captures bit 3's carry
    /// into bit 4.
    fn auxiliary_of(&mut self, lhs: Value, rhs: Value, res: Value) -> Value {
        let lhs_xor_rhs = self.builder.ins().bxor(lhs, rhs);
        let af_xor = self.builder.ins().bxor(lhs_xor_rhs, res);
        let af_shifted = self.builder.ins().ushr_imm_u(af_xor, 4);
        self.builder.ins().band_imm_u(af_shifted, 1)
    }

    /// ZF, SF, AF and PF - the four flags every add- and subtract-shaped operation
    /// derives the same way, from the operands and the result and from nothing
    /// about how the result was produced.
    ///
    /// Factored out because that is precisely what [`Spec::IntegerOverflowFlags`]
    /// does *not* change: CF and OF have a machine instruction behind them, these
    /// four do not.
    ///
    /// [`Spec::IntegerOverflowFlags`]: crate::core::specialize::Spec::IntegerOverflowFlags
    fn update_flags_zsap(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        if self.flag_needed(X86Flag::Zero) {
            let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            self.set_flag(X86Flag::Zero, zf)?;
        }

        // Sign Flag (SF): MSB of result
        if self.flag_needed(X86Flag::Sign) {
            let sf = self.builder.ins().ushr_imm_u(res, bit_width - 1);
            self.set_flag(X86Flag::Sign, sf)?;
        }

        // Auxiliary Carry (AF): carry from bit 3 -> ((lhs ^ rhs ^ res) >> 4) & 1.
        // For ADC/SBB this matches hardware because the carry-in only affects
        // bit 0, and bit-3's carry into bit 4 is still captured by the xor of
        // the three operands.
        if self.flag_needed(X86Flag::Auxiliary) {
            let af = self.auxiliary_of(lhs, rhs, res);
            self.set_flag(X86Flag::Auxiliary, af)?;
        }

        // Parity Flag (PF): 1 if low 8 bits of result have even popcount
        if self.flag_needed(X86Flag::Parity) {
            let pf = self.parity_of(res);
            self.set_flag(X86Flag::Parity, pf)?;
        }

        Ok(())
    }

    /// OF for an add-shaped operation: `((lhs ^ res) & (rhs ^ res)) >> (w-1)`,
    /// i.e. both operands' signs differ from the result's.
    fn update_overflow_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        if !self.flag_needed(X86Flag::Overflow) {
            return Ok(());
        }
        let xor1 = self.builder.ins().bxor(lhs, res);
        let xor2 = self.builder.ins().bxor(rhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm_u(and, (size * 8 - 1) as i64);
        self.set_flag(X86Flag::Overflow, of)
    }

    /// OF for a subtract-shaped operation: `((lhs ^ rhs) & (lhs ^ res)) >> (w-1)`,
    /// i.e. the operands' signs differ and the result took the subtrahend's.
    fn update_overflow_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        if !self.flag_needed(X86Flag::Overflow) {
            return Ok(());
        }
        let xor1 = self.builder.ins().bxor(lhs, rhs);
        let xor2 = self.builder.ins().bxor(lhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm_u(and, (size * 8 - 1) as i64);
        self.set_flag(X86Flag::Overflow, of)
    }

    /// STC - set carry flag.
    pub(super) fn handle_stc(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let one = self.builder.ins().iconst(types::I8, 1);
        self.set_flag(X86Flag::Carry, one)
    }

    /// CLC - clear carry flag.
    pub(super) fn handle_clc(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Carry, zero)
    }

    /// CMC - complement (toggle) carry flag.
    pub(super) fn handle_cmc(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let one = self.builder.ins().iconst(types::I8, 1);
        let toggled = self.builder.ins().bxor(cf, one);
        self.set_flag(X86Flag::Carry, toggled)
    }

    /// STD - set direction flag (string ops decrement RSI/RDI).
    pub(super) fn handle_std(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let one = self.builder.ins().iconst(types::I8, 1);
        self.set_flag(X86Flag::Direction, one)
    }

    /// CLD - clear direction flag (string ops increment RSI/RDI).
    pub(super) fn handle_cld(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Direction, zero)
    }

    /// The flags an `ADD` writes, given a sum somebody else computed.
    ///
    /// The portable definition. Handlers should call
    /// [`Lifter::add_with_flags`] instead, which is free to get CF and OF from
    /// the machine.
    pub fn update_flags_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        // Carry Flag (CF): unsigned overflow - result < lhs
        if self.flag_needed(X86Flag::Carry) {
            let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, res, lhs);
            self.set_flag(X86Flag::Carry, cf)?;
        }
        self.update_overflow_add(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags a `SUB` writes, given a difference somebody else computed.
    ///
    /// The portable definition; see [`Lifter::sub_with_flags`].
    pub fn update_flags_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        // Carry Flag (CF) - borrow: lhs < rhs (unsigned)
        if self.flag_needed(X86Flag::Carry) {
            let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
            self.set_flag(X86Flag::Carry, cf)?;
        }
        self.update_overflow_sub(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// `lhs + rhs`, and every flag an `ADD` writes.
    ///
    /// The entry point for `ADD`, `XADD`, `LOCK ADD` and `LOCK XADD` - anything
    /// whose flag half is an add. Returns the wrapped sum, which is the same
    /// value an `iadd` would have produced whichever path ran.
    pub fn add_with_flags(&mut self, lhs: Value, rhs: Value, size: usize) -> crate::Result<Value> {
        match self.try_native_add_flags(lhs, rhs)? {
            Some(res) => {
                self.update_flags_zsap(lhs, rhs, res, size)?;
                Ok(res)
            }
            None => {
                let res = self.builder.ins().iadd(lhs, rhs);
                self.update_flags_add(lhs, rhs, res, size)?;
                Ok(res)
            }
        }
    }

    /// `lhs - rhs`, and every flag a `SUB` writes.
    ///
    /// The entry point for `SUB`, `CMP` (which discards the difference), `NEG`
    /// (`0 - x`), `CMPXCHG`'s comparison, and the locked forms of all of them.
    pub fn sub_with_flags(&mut self, lhs: Value, rhs: Value, size: usize) -> crate::Result<Value> {
        match self.try_native_sub_flags(lhs, rhs)? {
            Some(res) => {
                self.update_flags_zsap(lhs, rhs, res, size)?;
                Ok(res)
            }
            None => {
                let res = self.builder.ins().isub(lhs, rhs);
                self.update_flags_sub(lhs, rhs, res, size)?;
                Ok(res)
            }
        }
    }

    /// `lhs + 1` or `lhs - 1`, and every flag an `INC`/`DEC` writes - which is
    /// all of them except CF, left exactly as it was.
    ///
    /// `rhs` is the constant one, passed in rather than materialized here so
    /// the caller's operand type is the authority on its type.
    pub fn inc_dec_with_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        is_add: bool,
        size: usize,
    ) -> crate::Result<Value> {
        match self.try_native_inc_dec_flags(lhs, rhs, is_add)? {
            Some(res) => {
                self.update_flags_zsap(lhs, rhs, res, size)?;
                Ok(res)
            }
            None => {
                let res = if is_add {
                    self.builder.ins().iadd(lhs, rhs)
                } else {
                    self.builder.ins().isub(lhs, rhs)
                };
                self.update_flags_inc_dec(lhs, rhs, res, is_add, size)?;
                Ok(res)
            }
        }
    }

    /// `lhs + rhs + cf_in` (`is_adc`) or `lhs - rhs - cf_in`, and every flag an
    /// `ADC`/`SBB` writes.
    ///
    /// `cf_in` is the incoming carry already widened to the operand type. The carry
    /// *out* is the expensive half and is built only when something reads CF; the
    /// two-step result is needed either way.
    ///
    /// Cranelift has exact opcodes for this shape (`uadd_overflow_cin`,
    /// `usub_overflow_bin`) but no backend lowers them, so the carry chain is spelled
    /// out as two ordinary overflow ops.
    pub fn adc_sbb_with_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        cf_in: Value,
        is_adc: bool,
        size: usize,
    ) -> crate::Result<Value> {
        let want_carry = self.flag_needed(X86Flag::Carry);
        let (res, cf_out) = match self.try_native_adc_sbb_flags(lhs, rhs, cf_in, is_adc)? {
            Some((res, cf_out)) => (res, Some(cf_out)),
            None => self.adc_sbb_portable(lhs, rhs, cf_in, is_adc, want_carry),
        };
        if let Some(cf_out) = cf_out {
            self.set_flag(X86Flag::Carry, cf_out)?;
        }
        if is_adc {
            self.update_overflow_add(lhs, rhs, res, size)?;
        } else {
            self.update_overflow_sub(lhs, rhs, res, size)?;
        }
        self.update_flags_zsap(lhs, rhs, res, size)?;
        Ok(res)
    }

    /// `lhs * rhs` truncated to the operand width, and the flags the two- and
    /// three-operand `IMUL` forms write.
    ///
    /// Those are CF and OF alone, both meaning "the full signed product did not
    /// fit"; ZF/SF/AF/PF are undefined and left as they were. The portable answer
    /// needs a second, widening multiply to see the dropped bits, so a dead CF
    /// *and* OF saves the whole `smulhi`.
    pub fn imul_with_flags(&mut self, lhs: Value, rhs: Value, size: usize) -> crate::Result<Value> {
        if let Some(res) = self.try_native_imul_flags(lhs, rhs)? {
            return Ok(res);
        }
        let res = self.builder.ins().imul(lhs, rhs);
        if self.flag_needed(X86Flag::Carry) || self.flag_needed(X86Flag::Overflow) {
            // Overflowed exactly when the high half is not the sign extension
            // of the low half.
            let hi = self.builder.ins().smulhi(lhs, rhs);
            let sign_ext = self.builder.ins().sshr_imm_u(res, (size * 8 - 1) as i64);
            let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
            self.set_flag(X86Flag::Carry, of_cf)?;
            self.set_flag(X86Flag::Overflow, of_cf)?;
        }
        Ok(res)
    }

    /// `lhs + rhs + cf_in` / `lhs - rhs - cf_in` and nothing else - no flags read,
    /// none written.
    ///
    /// The locked `ADC`/`SBB` retry loop needs the result on every iteration but
    /// the flags only for the iteration that won.
    pub(super) fn adc_sbb_value(
        &mut self,
        lhs: Value,
        rhs: Value,
        cf_in: Value,
        is_adc: bool,
    ) -> Value {
        if is_adc {
            let tmp = self.builder.ins().iadd(lhs, rhs);
            self.builder.ins().iadd(tmp, cf_in)
        } else {
            let tmp = self.builder.ins().isub(lhs, rhs);
            self.builder.ins().isub(tmp, cf_in)
        }
    }

    /// The two-step `ADC`/`SBB`, with the carry out assembled from the carry out
    /// of each step - the portable spelling, and the fallback for
    /// [`Lifter::adc_sbb_with_flags`].
    fn adc_sbb_portable(
        &mut self,
        lhs: Value,
        rhs: Value,
        cf_in: Value,
        is_adc: bool,
        want_carry: bool,
    ) -> (Value, Option<Value>) {
        if is_adc {
            let tmp = self.builder.ins().iadd(lhs, rhs);
            let res = self.builder.ins().iadd(tmp, cf_in);
            if !want_carry {
                return (res, None);
            }
            let c1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, lhs);
            let c2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, res, tmp);
            (res, Some(self.builder.ins().bor(c1, c2)))
        } else {
            let tmp = self.builder.ins().isub(lhs, rhs);
            let res = self.builder.ins().isub(tmp, cf_in);
            if !want_carry {
                return (res, None);
            }
            let b1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
            let b2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, cf_in);
            (res, Some(self.builder.ins().bor(b1, b2)))
        }
    }

    pub fn update_flags_logical(&mut self, res: Value, size: usize) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        if self.flag_needed(X86Flag::Zero) {
            let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            self.set_flag(X86Flag::Zero, zf)?;
        }

        // Sign Flag (SF)
        if self.flag_needed(X86Flag::Sign) {
            let sf = self.builder.ins().ushr_imm_u(res, bit_width - 1);
            self.set_flag(X86Flag::Sign, sf)?;
        }

        // Carry and Overflow are always cleared for logical ops
        if self.flag_needed(X86Flag::Carry) || self.flag_needed(X86Flag::Overflow) {
            let zero = self.builder.ins().iconst(types::I8, 0);
            self.set_flag(X86Flag::Carry, zero)?;
            self.set_flag(X86Flag::Overflow, zero)?;
        }

        // Parity Flag (PF): 1 if low 8 bits of result have even popcount
        if self.flag_needed(X86Flag::Parity) {
            let pf = self.parity_of(res);
            self.set_flag(X86Flag::Parity, pf)?;
        }

        // AF is undefined for logical ops per Intel SDM; leave unchanged. Which
        // is why `flag_effects` does not list AF among this op's writes: a
        // reader downstream sees whatever set it last, so that write has to stay
        // live across an `AND`.

        Ok(())
    }

    /// The flags an `ADC` writes, given a result and a carry-out somebody else
    /// computed. The portable definition; see [`Lifter::adc_sbb_with_flags`].
    pub fn update_flags_adc(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        self.set_flag(X86Flag::Carry, cf_out)?;
        // OF: signed overflow in lhs + rhs + cf_in.
        self.update_overflow_add(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags an `SBB` writes, given a result and a borrow-out somebody else
    /// computed. The portable definition; see [`Lifter::adc_sbb_with_flags`].
    pub fn update_flags_sbb(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        self.set_flag(X86Flag::Carry, cf_out)?;
        // OF: signed overflow in lhs - rhs - cf_in.
        self.update_overflow_sub(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags an `INC`/`DEC` writes - everything but CF, which is
    /// architecturally untouched and is left holding whatever set it last.
    ///
    /// The portable definition; see [`Lifter::inc_dec_with_flags`].
    pub fn update_flags_inc_dec(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        is_add: bool,
        size: usize,
    ) -> crate::Result<()> {
        if is_add {
            self.update_overflow_add(lhs, rhs, res, size)?;
        } else {
            self.update_overflow_sub(lhs, rhs, res, size)?;
        }
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// Update flags for a shift operation (SHL/SHR/SAR/SHLD/SHRD).
    ///
    /// `orig` is the pre-shift destination, `res` the result (both in the operand
    /// type), `cf_bit` the last bit shifted out, `count` the masked count, `size`
    /// the operand size in bytes.
    ///
    /// If the masked count is 0, **no flags are affected** - so the prior values
    /// must be preserved rather than clobbered with the garbage the shift formula
    /// produces at count 0.
    ///
    /// OF is architecturally defined only for count == 1, where for every flavour
    /// it is "the sign changed": SHL's `MSB(res) ^ CF` and SHRD's definition both
    /// reduce to `MSB(orig) ^ MSB(res)`, SHR's `MSB(orig)` matches because
    /// `MSB(res)` is 0, and SAR's 0 matches because the sign is preserved. For
    /// counts > 1 it is undefined and cleared here.
    pub fn update_flags_shift(
        &mut self,
        orig: Value,
        res: Value,
        cf_bit: Value,
        count: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Every flag here is a read-modify-write, so a dead one skips its old value,
        // its new value *and* the select between them. `flag_effects` classifies a
        // shift as reading exactly this set, so the `use_var`s below are covered.
        let (zf, sf, cf, of, pf) = (
            self.flag_needed(X86Flag::Zero),
            self.flag_needed(X86Flag::Sign),
            self.flag_needed(X86Flag::Carry),
            self.flag_needed(X86Flag::Overflow),
            self.flag_needed(X86Flag::Parity),
        );
        if !(zf || sf || cf || of || pf) {
            return Ok(());
        }

        // is_zero is I8 (icmp_imm_s result); select uses any nonzero as true.
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0);

        if zf {
            let old_zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
            let new_zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            let final_zf = self.builder.ins().select(is_zero, old_zf, new_zf);
            self.set_flag(X86Flag::Zero, final_zf)?;
        }

        if sf {
            let old_sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
            // Newly-computed flag values (in operand width), reduced to I8 to
            // match the flag SSA variable type that `select` requires.
            let new_sf_w = self.builder.ins().ushr_imm_u(res, bit_width - 1);
            let new_sf = if size == 1 {
                new_sf_w
            } else {
                self.builder.ins().ireduce(types::I8, new_sf_w)
            };
            let final_sf = self.builder.ins().select(is_zero, old_sf, new_sf);
            self.set_flag(X86Flag::Sign, final_sf)?;
        }

        if cf {
            let old_cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
            let new_cf = if self.builder.func.dfg.value_type(cf_bit) == types::I8 {
                cf_bit
            } else {
                self.builder.ins().ireduce(types::I8, cf_bit)
            };
            let final_cf = self.builder.ins().select(is_zero, old_cf, new_cf);
            self.set_flag(X86Flag::Carry, final_cf)?;
        }

        if of {
            // OF for count == 1: sign change between operand and result.
            let old_of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
            let sign_xor = self.builder.ins().bxor(orig, res);
            let sign_change_w = self.builder.ins().ushr_imm_u(sign_xor, bit_width - 1);
            let sign_change = if self.builder.func.dfg.value_type(sign_change_w) == types::I8 {
                sign_change_w
            } else {
                self.builder.ins().ireduce(types::I8, sign_change_w)
            };
            let zero_i8 = self.builder.ins().iconst(types::I8, 0);
            let is_one = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 1);
            let new_of = self.builder.ins().select(is_one, sign_change, zero_i8);
            let final_of = self.builder.ins().select(is_zero, old_of, new_of);
            self.set_flag(X86Flag::Overflow, final_of)?;
        }

        if pf {
            // PF: even popcount of the low 8 bits of the result.
            let old_pf = self.builder.use_var(self.state.flags[&X86Flag::Parity]);
            let new_pf = self.parity_of(res);
            let final_pf = self.builder.ins().select(is_zero, old_pf, new_pf);
            self.set_flag(X86Flag::Parity, final_pf)?;
        }

        Ok(())
    }
}
