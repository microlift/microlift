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

        // Carry Flag (CF) is NOT affected by INC/DEC
        Ok(())
    }

    /// Update flags for a shift operation (SHL/SHR/SAR).
    /// `res` is the result, `cf_bit` is the last bit shifted out (for CF),
    /// `count` is the masked shift count (in operand type), `size` is the operand size in bytes.
    ///
    /// Per x86 semantics: if the masked shift count is 0, **no flags are
    /// affected** — including CF/ZF/SF/OF. We must preserve the prior flag
    /// values in that case rather than clobbering them with the (garbage)
    /// values that fall out of the shift formula for count=0.
    pub fn update_flags_shift(
        &mut self,
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

        // Newly-computed flag values (in operand width), reduced to I8 to
        // match the flag SSA variable type that `select` requires.
        let new_zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        let new_sf_w = self.builder.ins().ushr_imm(res, bit_width - 1);
        let new_sf = if size == 1 {
            new_sf_w
        } else {
            self.builder.ins().ireduce(types::I8, new_sf_w)
        };
        let new_cf = if size == 1 {
            cf_bit
        } else {
            self.builder.ins().ireduce(types::I8, cf_bit)
        };
        // OF is undefined for shifts other than by 1; conservatively clear it.
        let new_of = self.builder.ins().iconst(types::I8, 0);

        // is_zero is I8 (icmp_imm result); select uses any nonzero as true.
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, count, 0);
        let final_zf = self.builder.ins().select(is_zero, old_zf, new_zf);
        let final_sf = self.builder.ins().select(is_zero, old_sf, new_sf);
        let final_cf = self.builder.ins().select(is_zero, old_cf, new_cf);
        let final_of = self.builder.ins().select(is_zero, old_of, new_of);

        self.set_flag(X86Flag::Zero, final_zf)?;
        self.set_flag(X86Flag::Sign, final_sf)?;
        self.set_flag(X86Flag::Carry, final_cf)?;
        self.set_flag(X86Flag::Overflow, final_of)?;

        Ok(())
    }
}
