//! Eager condition-flag updates; the finished SSA pass removes unused computations.
//!
//! PF and AF are stored as source bytes and folded into booleans at their readers.

use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// The byte PF is the parity of: the low byte of `res`, unchanged; the fold happens at the reader ([`Lifter::get_flag`]).
    fn parity_source(&mut self, res: Value) -> Value {
        if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        }
    }

    /// The byte AF is bit 4 of `lhs ^ rhs ^ res`; the shift and mask are the reader's.
    fn auxiliary_source(&mut self, lhs: Value, rhs: Value, res: Value) -> Value {
        let lhs_xor_rhs = self.builder.ins().bxor(lhs, rhs);
        let af_xor = self.builder.ins().bxor(lhs_xor_rhs, res);
        if self.builder.func.dfg.value_type(af_xor) == types::I8 {
            af_xor
        } else {
            self.builder.ins().ireduce(types::I8, af_xor)
        }
    }

    /// ZF, SF, AF and PF: derived from operands and result alone, unlike CF/OF which [`Spec::IntegerOverflowFlags`] takes from the machine.
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

        if self.flag_needed(X86Flag::Zero) {
            let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            self.set_flag(X86Flag::Zero, zf)?;
        }

        if self.flag_needed(X86Flag::Sign) {
            let sf = self.builder.ins().ushr_imm_u(res, bit_width - 1);
            self.set_flag(X86Flag::Sign, sf)?;
        }

        // AF is the carry out of bit 3; right for ADC/SBB since the carry-in only reaches bit 0.
        if self.flag_needed(X86Flag::Auxiliary) {
            let af = self.auxiliary_source(lhs, rhs, res);
            self.set_flag_source(X86Flag::Auxiliary, af)?;
        }

        if self.flag_needed(X86Flag::Parity) {
            let pf = self.parity_source(res);
            self.set_flag_source(X86Flag::Parity, pf)?;
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

    /// The flags an `ADD` writes given a computed sum; the portable definition behind [`Lifter::add_with_flags`].
    pub fn update_flags_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        if self.flag_needed(X86Flag::Carry) {
            let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, res, lhs);
            self.set_flag(X86Flag::Carry, cf)?;
        }
        self.update_overflow_add(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags a `SUB` writes given a computed difference; see [`Lifter::sub_with_flags`].
    pub fn update_flags_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        if self.flag_needed(X86Flag::Carry) {
            let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
            self.set_flag(X86Flag::Carry, cf)?;
        }
        self.update_overflow_sub(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// `lhs + rhs` and every flag an `ADD` writes (`ADD`, `XADD` and their locked forms).
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

    /// `lhs - rhs` and every flag a `SUB` writes (`SUB`, `CMP`, `NEG`, `CMPXCHG`'s compare, locked forms).
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

    /// `lhs ± 1` and every flag an `INC`/`DEC` writes (all but CF); `rhs` is the constant one in the operand's type.
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

    /// `lhs + rhs + cf_in` (`is_adc`) or `lhs - rhs - cf_in`, and every flag an `ADC`/`SBB` writes.
    /// `cf_in` is pre-widened; the carry out is built only when CF is read.
    /// No backend lowers `uadd_overflow_cin`/`usub_overflow_bin`, so the chain is two ordinary overflow ops.
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

    /// `lhs * rhs` truncated, with the flags of the two- and three-operand `IMUL`.
    /// CF and OF both mean "the full signed product did not fit"; ZF/SF/AF/PF are left as they were.
    pub fn imul_with_flags(&mut self, lhs: Value, rhs: Value, size: usize) -> crate::Result<Value> {
        if let Some(res) = self.try_native_imul_flags(lhs, rhs)? {
            return Ok(res);
        }
        let res = self.builder.ins().imul(lhs, rhs);
        if self.flag_needed(X86Flag::Carry) || self.flag_needed(X86Flag::Overflow) {
            // Overflowed when the high half is not the sign extension of the low half.
            let hi = self.builder.ins().smulhi(lhs, rhs);
            let sign_ext = self.builder.ins().sshr_imm_u(res, (size * 8 - 1) as i64);
            let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
            self.set_flag(X86Flag::Carry, of_cf)?;
            self.set_flag(X86Flag::Overflow, of_cf)?;
        }
        Ok(res)
    }

    /// `lhs + rhs + cf_in` / `lhs - rhs - cf_in` with no flags read or written.
    /// The locked retry loop needs flags only for the winning iteration.
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

    /// The portable two-step `ADC`/`SBB`; fallback for [`Lifter::adc_sbb_with_flags`].
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

        if self.flag_needed(X86Flag::Zero) {
            let zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            self.set_flag(X86Flag::Zero, zf)?;
        }

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

        if self.flag_needed(X86Flag::Parity) {
            let pf = self.parity_source(res);
            self.set_flag_source(X86Flag::Parity, pf)?;
        }

        // AF is undefined for logical ops; its previous definition stays live across an `AND`.

        Ok(())
    }

    /// The flags an `ADC` writes given a result and carry-out; portable, see [`Lifter::adc_sbb_with_flags`].
    pub fn update_flags_adc(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        self.set_flag(X86Flag::Carry, cf_out)?;
        self.update_overflow_add(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags an `SBB` writes given a result and borrow-out; portable, see [`Lifter::adc_sbb_with_flags`].
    pub fn update_flags_sbb(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        self.set_flag(X86Flag::Carry, cf_out)?;
        self.update_overflow_sub(lhs, rhs, res, size)?;
        self.update_flags_zsap(lhs, rhs, res, size)
    }

    /// The flags an `INC`/`DEC` writes (all but CF); see [`Lifter::inc_dec_with_flags`].
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

    /// Flags for SHL/SHR/SAR/SHLD/SHRD: `orig` pre-shift destination, `res` result, `cf_bit` last bit shifted out, `count` masked count, `size` bytes.
    /// A zero masked count affects no flags, so prior values are preserved.
    /// OF is defined only for count == 1 (the sign changed: `MSB(orig) ^ MSB(res)`); cleared for larger counts.
    pub fn update_flags_shift(
        &mut self,
        orig: Value,
        res: Value,
        cf_bit: Value,
        count: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Old flag sources feed the selects: a zero count preserves them.
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

        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0);

        if zf {
            let old_zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
            let new_zf = self.builder.ins().icmp_imm_s(IntCC::Equal, res, 0);
            let final_zf = self.builder.ins().select(is_zero, old_zf, new_zf);
            self.set_flag(X86Flag::Zero, final_zf)?;
        }

        if sf {
            let old_sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
            // Reduced to `I8`, the flag variable type `select` needs.
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
            // Select before the fold so a zero count leaves the stored PF source unchanged.
            let old_pf = self.builder.use_var(self.state.flags[&X86Flag::Parity]);
            let new_pf = self.parity_source(res);
            let final_pf = self.builder.ins().select(is_zero, old_pf, new_pf);
            self.set_flag_source(X86Flag::Parity, final_pf)?;
        }

        Ok(())
    }
}
