use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// Generic SETcc: write 1 or 0 to the byte destination based on `cond` (I8, 0 or 1).
    fn handle_setcc_cond(&mut self, instruction: &Instruction, cond: Value) -> crate::Result<()> {
        self.write_operand(instruction, 0, cond)
    }

    /// Invert a boolean I8 flag (0→1, 1→0) using an equality compare.
    fn bool_not(&mut self, val: Value) -> Value {
        self.builder.ins().icmp_imm_s(IntCC::Equal, val, 0)
    }

    // SETE / SETZ: ZF=1
    pub(super) fn handle_sete(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        self.handle_setcc_cond(instruction, zf)
    }

    // SETNE / SETNZ: ZF=0
    pub(super) fn handle_setne(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let cond = self.bool_not(zf);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETB / SETC / SETNAE: CF=1
    pub(super) fn handle_setb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        self.handle_setcc_cond(instruction, cf)
    }

    // SETAE / SETNB / SETNC: CF=0
    pub(super) fn handle_setae(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let cond = self.bool_not(cf);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETBE / SETNA: CF=1 OR ZF=1
    pub(super) fn handle_setbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETA / SETNBE: CF=0 AND ZF=0
    pub(super) fn handle_seta(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        let cond = self.bool_not(cf_or_zf);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETL / SETNGE: SF != OF
    pub(super) fn handle_setl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let cond = self.builder.ins().bxor(sf, of);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETGE / SETNL: SF=OF
    pub(super) fn handle_setge(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        let cond = self.bool_not(sf_xor_of);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETLE / SETNG: ZF=1 OR SF!=OF
    pub(super) fn handle_setle(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bor(zf, sf_xor_of);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETG / SETNLE: ZF=0 AND SF=OF
    pub(super) fn handle_setg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.builder.use_var(self.state.flags[&X86Flag::Zero]);
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        let not_g = self.builder.ins().bor(zf, sf_xor_of);
        let cond = self.bool_not(not_g);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETS: SF=1
    pub(super) fn handle_sets(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        self.handle_setcc_cond(instruction, sf)
    }

    // SETNS: SF=0
    pub(super) fn handle_setns(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self.builder.use_var(self.state.flags[&X86Flag::Sign]);
        let cond = self.bool_not(sf);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETO: OF=1
    pub(super) fn handle_seto(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        self.handle_setcc_cond(instruction, of)
    }

    // SETNO: OF=0
    pub(super) fn handle_setno(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self.builder.use_var(self.state.flags[&X86Flag::Overflow]);
        let cond = self.bool_not(of);
        self.handle_setcc_cond(instruction, cond)
    }

    // SETP / SETPE: PF=1
    pub(super) fn handle_setp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self.builder.use_var(self.state.flags[&X86Flag::Parity]);
        self.handle_setcc_cond(instruction, pf)
    }

    // SETNP / SETPO: PF=0
    pub(super) fn handle_setnp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self.builder.use_var(self.state.flags[&X86Flag::Parity]);
        let cond = self.bool_not(pf);
        self.handle_setcc_cond(instruction, cond)
    }
}
