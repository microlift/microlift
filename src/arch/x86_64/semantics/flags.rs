use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
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


    pub fn update_flags_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        // Sign Flag (SF): MSB of result
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        // Carry Flag (CF): unsigned overflow - result < lhs
        let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, res, lhs);
        self.set_flag(X86Flag::Carry, cf)?;

        // Overflow Flag (OF): (lhs ^ res) & (rhs ^ res), then check MSB
        let xor1 = self.builder.ins().bxor(lhs, res);
        let xor2 = self.builder.ins().bxor(rhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(X86Flag::Overflow, of)?;

        // Auxiliary Carry (AF): carry from bit 3 -> ((lhs ^ rhs ^ res) >> 4) & 1
        let lhs_xor_rhs = self.builder.ins().bxor(lhs, rhs);
        let af_xor = self.builder.ins().bxor(lhs_xor_rhs, res);
        let af_shifted = self.builder.ins().ushr_imm(af_xor, 4);
        let af = self.builder.ins().band_imm(af_shifted, 1);
        self.set_flag(X86Flag::Auxiliary, af)?;

        // Parity Flag (PF): 1 if low 8 bits of result have even popcount
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        Ok(())
    }

    pub fn update_flags_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        // Carry Flag (CF) - borrow: lhs < rhs (unsigned)
        let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
        self.set_flag(X86Flag::Carry, cf)?;

        // Overflow Flag (OF): (lhs ^ rhs) & (lhs ^ res), then MSB
        let xor1 = self.builder.ins().bxor(lhs, rhs);
        let xor2 = self.builder.ins().bxor(lhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(X86Flag::Overflow, of)?;

        // Auxiliary Carry (AF): borrow from bit 3 -> ((lhs ^ rhs ^ res) >> 4) & 1
        let af_xor = self.builder.ins().bxor(xor1, res);
        let af_shifted = self.builder.ins().ushr_imm(af_xor, 4);
        let af = self.builder.ins().band_imm(af_shifted, 1);
        self.set_flag(X86Flag::Auxiliary, af)?;

        // Parity Flag (PF): 1 if low 8 bits of result have even popcount
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        Ok(())
    }

    pub fn update_flags_logical(&mut self, res: Value, size: usize) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        // Carry and Overflow are always cleared for logical ops
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Carry, zero)?;
        self.set_flag(X86Flag::Overflow, zero)?;

        // Parity Flag (PF): 1 if low 8 bits of result have even popcount
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        // AF is undefined for logical ops per Intel SDM; leave unchanged.

        Ok(())
    }

    /// Update flags for ADC (add with carry). `cf_out` is the precomputed carry-out (I8, 0/1).
    pub fn update_flags_adc(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        self.set_flag(X86Flag::Carry, cf_out)?;

        // OF: signed overflow in lhs + rhs + cf_in
        // Formula: ((lhs ^ res) & (rhs ^ res)) >> (bit_width - 1)
        let xor1 = self.builder.ins().bxor(lhs, res);
        let xor2 = self.builder.ins().bxor(rhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(X86Flag::Overflow, of)?;

        // AF: bit-3 carry/borrow chain - ((lhs ^ rhs ^ res) >> 4) & 1.
        // For ADC/SBB this matches hardware because the cf_in only affects bit 0,
        // and bit-3 carry into bit 4 is captured by the xor of the three operands.
        let lhs_xor_rhs = self.builder.ins().bxor(lhs, rhs);
        let af_xor = self.builder.ins().bxor(lhs_xor_rhs, res);
        let af_shifted = self.builder.ins().ushr_imm(af_xor, 4);
        let af = self.builder.ins().band_imm(af_shifted, 1);
        self.set_flag(X86Flag::Auxiliary, af)?;

        // PF: even popcount of low 8 bits of result.
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        Ok(())
    }

    /// Update flags for SBB (subtract with borrow). `cf_out` is the precomputed borrow-out (I8, 0/1).
    pub fn update_flags_sbb(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        cf_out: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        self.set_flag(X86Flag::Carry, cf_out)?;

        // OF: signed overflow in lhs - rhs - cf_in
        // Formula: ((lhs ^ rhs) & (lhs ^ res)) >> (bit_width - 1)
        let xor1 = self.builder.ins().bxor(lhs, rhs);
        let xor2 = self.builder.ins().bxor(lhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(X86Flag::Overflow, of)?;

        // AF: ((lhs ^ rhs ^ res) >> 4) & 1 - reuses xor1 = lhs ^ rhs.
        let af_xor = self.builder.ins().bxor(xor1, res);
        let af_shifted = self.builder.ins().ushr_imm(af_xor, 4);
        let af = self.builder.ins().band_imm(af_shifted, 1);
        self.set_flag(X86Flag::Auxiliary, af)?;

        // PF: even popcount of low 8 bits of result.
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        Ok(())
    }

    pub fn update_flags_inc_dec(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        is_add: bool,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        // Overflow Flag (OF)
        let of = if is_add {
            // INC: (lhs ^ res) & (rhs ^ res) then MSB
            let xor1 = self.builder.ins().bxor(lhs, res);
            let xor2 = self.builder.ins().bxor(rhs, res);
            let and = self.builder.ins().band(xor1, xor2);
            self.builder.ins().ushr_imm(and, bit_width - 1)
        } else {
            // DEC: (lhs ^ rhs) & (lhs ^ res) then MSB
            let xor1 = self.builder.ins().bxor(lhs, rhs);
            let xor2 = self.builder.ins().bxor(lhs, res);
            let and = self.builder.ins().band(xor1, xor2);
            self.builder.ins().ushr_imm(and, bit_width - 1)
        };
        self.set_flag(X86Flag::Overflow, of)?;

        // AF: ((lhs ^ rhs ^ res) >> 4) & 1
        let lhs_xor_rhs = self.builder.ins().bxor(lhs, rhs);
        let af_xor = self.builder.ins().bxor(lhs_xor_rhs, res);
        let af_shifted = self.builder.ins().ushr_imm(af_xor, 4);
        let af = self.builder.ins().band_imm(af_shifted, 1);
        self.set_flag(X86Flag::Auxiliary, af)?;

        // PF: even popcount of low 8 bits of result.
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let pf = self.builder.ins().bxor_imm(pop_lsb, 1);
        self.set_flag(X86Flag::Parity, pf)?;

        // Carry Flag (CF) is NOT affected by INC/DEC
        Ok(())
    }

    /// Update flags for a shift operation (SHL/SHR/SAR/SHLD/SHRD).
    /// `orig` is the pre-shift destination value, `res` the result (both in the
    /// operand type), `cf_bit` the last bit shifted out (for CF), `count` the
    /// masked shift count, `size` the operand size in bytes.
    ///
    /// Per x86 semantics: if the masked shift count is 0, **no flags are
    /// affected** - including CF/ZF/SF/OF/PF. We must preserve the prior flag
    /// values in that case rather than clobbering them with the (garbage)
    /// values that fall out of the shift formula for count=0.
    ///
    /// OF is architecturally defined only for count == 1, where for every shift
    /// flavour it equals "the sign changed": SHL's `MSB(res) ^ CF` and SHRD's
    /// documented definition both reduce to `MSB(orig) ^ MSB(res)`, SHR's
    /// `MSB(orig)` matches because `MSB(res)` is 0, and SAR's 0 matches because
    /// the sign is preserved. For counts > 1 OF is undefined; we clear it.
    pub fn update_flags_shift(
        &mut self,
        orig: Value,
        res: Value,
        cf_bit: Value,
        count: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Load prior values of each flag so we can keep them when count == 0.
        let old_zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let old_sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let old_cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let old_of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let old_pf = self.builder.use_var(self.state.flags[&X86Flag::Parity]);

        // Newly-computed flag values (in operand width), reduced to I8 to
        // match the flag SSA variable type that `select` requires.
        let new_zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        let new_sf_w = self.builder.ins().ushr_imm(res, bit_width - 1);
        let new_sf = if size == 1 {
            new_sf_w
        } else {
            self.builder.ins().ireduce(types::I8, new_sf_w)
        };
        let new_cf = if self.builder.func.dfg.value_type(cf_bit) == types::I8 {
            cf_bit
        } else {
            self.builder.ins().ireduce(types::I8, cf_bit)
        };

        // OF for count == 1: sign change between operand and result.
        let sign_xor = self.builder.ins().bxor(orig, res);
        let sign_change_w = self.builder.ins().ushr_imm(sign_xor, bit_width - 1);
        let sign_change = if self.builder.func.dfg.value_type(sign_change_w) == types::I8 {
            sign_change_w
        } else {
            self.builder.ins().ireduce(types::I8, sign_change_w)
        };
        let zero_i8 = self.builder.ins().iconst(types::I8, 0);
        let is_one = self.builder.ins().icmp_imm(IntCC::Equal, count, 1);
        let new_of = self.builder.ins().select(is_one, sign_change, zero_i8);

        // PF: even popcount of the low 8 bits of the result.
        let res_lo = if self.builder.func.dfg.value_type(res) == types::I8 {
            res
        } else {
            self.builder.ins().ireduce(types::I8, res)
        };
        let pop = self.builder.ins().popcnt(res_lo);
        let pop_lsb = self.builder.ins().band_imm(pop, 1);
        let new_pf = self.builder.ins().bxor_imm(pop_lsb, 1);

        // is_zero is I8 (icmp_imm result); select uses any nonzero as true.
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, count, 0);
        let final_zf = self.builder.ins().select(is_zero, old_zf, new_zf);
        let final_sf = self.builder.ins().select(is_zero, old_sf, new_sf);
        let final_cf = self.builder.ins().select(is_zero, old_cf, new_cf);
        let final_of = self.builder.ins().select(is_zero, old_of, new_of);
        let final_pf = self.builder.ins().select(is_zero, old_pf, new_pf);

        self.set_flag(X86Flag::Zero, final_zf)?;
        self.set_flag(X86Flag::Sign, final_sf)?;
        self.set_flag(X86Flag::Carry, final_cf)?;
        self.set_flag(X86Flag::Overflow, final_of)?;
        self.set_flag(X86Flag::Parity, final_pf)?;

        Ok(())
    }
}
