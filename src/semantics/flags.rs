use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;

impl<'a> Lifter<'a> {
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
    /// `res` is the result, `shifted_out_bit` is the last bit shifted out (for CF),
    /// `size` is the operand size in bytes.
    pub fn update_flags_shift(
        &mut self,
        res: Value,
        cf_bit: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // ZF
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(X86Flag::Zero, zf)?;

        // SF
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(X86Flag::Sign, sf)?;

        // CF = last bit shifted out
        self.set_flag(X86Flag::Carry, cf_bit)?;

        // OF is defined only for shift-by-1, but we set it to 0 conservatively
        // (full OF semantics for arbitrary shifts is "undefined" per x86 manual)
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Overflow, zero)?;

        Ok(())
    }
}
