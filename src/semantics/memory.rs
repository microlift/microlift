use crate::MicroliftError;
use crate::lifter::Lifter;
use crate::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

impl<'a> Lifter<'a> {
    pub(super) fn handle_nop(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    pub(super) fn handle_mov(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let src_val = self.read_operand(instruction, 1, size)?;
        self.write_operand(instruction, 0, src_val)?;
        Ok(())
    }

    pub(super) fn handle_movzx(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = instruction.op0_register();
        let dst_ty = size_to_type(dst_reg.size());
        let src_size = self.operand_size(instruction, 1)?;
        let src_val = self.read_operand(instruction, 1, src_size)?;

        let extended = if self.builder.func.dfg.value_type(src_val) == dst_ty {
            src_val
        } else {
            self.builder.ins().uextend(dst_ty, src_val)
        };

        self.set_reg_val(dst_reg, extended)?;
        Ok(())
    }

    pub(super) fn handle_movsx(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = instruction.op0_register();
        let dst_ty = size_to_type(dst_reg.size());
        let src_size = self.operand_size(instruction, 1)?;
        let src_val = self.read_operand(instruction, 1, src_size)?;

        let extended = if self.builder.func.dfg.value_type(src_val) == dst_ty {
            src_val
        } else {
            self.builder.ins().sextend(dst_ty, src_val)
        };

        self.set_reg_val(dst_reg, extended)?;
        Ok(())
    }

    /// MOVSXD - sign-extend a 32-bit register/memory value to 64 bits.
    pub(super) fn handle_movsxd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = instruction.op0_register();
        let src_size = self.operand_size(instruction, 1)?;
        let src_val = self.read_operand(instruction, 1, src_size)?;
        let extended = self.builder.ins().sextend(types::I64, src_val);
        self.set_reg_val(dst_reg, extended)?;
        Ok(())
    }

    pub(super) fn handle_lea(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let reg = instruction.op0_register();
        let ptr = self.get_mem_ptr(instruction, 1)?;
        self.set_reg_val(reg, ptr)?;
        Ok(())
    }

    pub(super) fn handle_push(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let push_size = 8; // x86-64 PUSH is always 8 bytes

        let val = match op0 {
            OpKind::Register => self.get_reg_val(instruction.op0_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                let imm = instruction.immediate32to64();
                self.builder.ins().iconst(types::I64, imm)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("push-{:?}", op0),
                });
            }
        };

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm(rsp, -(push_size as i64));
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder.ins().store(MemFlags::new(), val, new_rsp, 0);
        Ok(())
    }

    pub(super) fn handle_pop(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let pop_size = 8;

        let rsp = self.get_reg_val(Register::RSP)?;
        let val = self.builder.ins().load(types::I64, MemFlags::new(), rsp, 0);

        let new_rsp = self.builder.ins().iadd_imm(rsp, pop_size as i64);
        self.set_reg_val(Register::RSP, new_rsp)?;

        match op0 {
            OpKind::Register => {
                self.set_reg_val(instruction.op0_register(), val)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlags::new(), val, ptr, 0);
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("pop-{:?}", op0),
                });
            }
        }
        Ok(())
    }

    /// LEAVE - high-level function epilogue: `RSP = RBP; POP RBP`.
    pub(super) fn handle_leave(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rbp = self.get_reg_val(Register::RBP)?;
        // RSP <- RBP
        self.set_reg_val(Register::RSP, rbp)?;
        // POP RBP: RBP <- [RSP]; RSP += 8
        let old_rbp = self.builder.ins().load(types::I64, MemFlags::new(), rbp, 0);
        self.set_reg_val(Register::RBP, old_rbp)?;
        let new_rsp = self.builder.ins().iadd_imm(rbp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        Ok(())
    }

    pub(super) fn handle_xchg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let a = self.read_operand(instruction, 0, size)?;
        let b = self.read_operand(instruction, 1, size)?;
        self.write_operand(instruction, 0, b)?;
        self.write_operand(instruction, 1, a)?;
        Ok(())
    }

    /// MOVSB - move byte from [RSI] to [RDI], advance both by ±1.
    pub(super) fn handle_movsb(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_movs(1)
    }

    /// MOVSW - move word (2 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsw(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_movs(2)
    }

    /// MOVSD - move dword (4 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsd_mem(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_movs(4)
    }

    /// MOVSQ - move qword (8 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_movs(8)
    }

    fn emit_movs(&mut self, size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let rsi = self.get_reg_val(Register::RSI)?;
        let rdi = self.get_reg_val(Register::RDI)?;
        let val = self.builder.ins().load(ty, MemFlags::new(), rsi, 0);
        self.builder.ins().store(MemFlags::new(), val, rdi, 0);
        // DF=0: increment; DF=1: decrement. We don't track DF, assume DF=0.
        let new_rsi = self.builder.ins().iadd_imm(rsi, size as i64);
        let new_rdi = self.builder.ins().iadd_imm(rdi, size as i64);
        self.set_reg_val(Register::RSI, new_rsi)?;
        self.set_reg_val(Register::RDI, new_rdi)?;
        Ok(())
    }

    /// STOSB - store AL to [RDI], advance RDI by ±1.
    pub(super) fn handle_stosb(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_stos(Register::AL, 1)
    }

    /// STOSW - store AX to [RDI], advance RDI by ±2.
    pub(super) fn handle_stosw(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_stos(Register::AX, 2)
    }

    /// STOSD - store EAX to [RDI], advance RDI by ±4.
    pub(super) fn handle_stosd(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_stos(Register::EAX, 4)
    }

    /// STOSQ - store RAX to [RDI], advance RDI by ±8.
    pub(super) fn handle_stosq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_stos(Register::RAX, 8)
    }

    fn emit_stos(&mut self, src_reg: Register, size: usize) -> crate::Result<()> {
        let rdi = self.get_reg_val(Register::RDI)?;
        let val = self.get_reg_val(src_reg)?;
        self.builder.ins().store(MemFlags::new(), val, rdi, 0);
        let new_rdi = self.builder.ins().iadd_imm(rdi, size as i64);
        self.set_reg_val(Register::RDI, new_rdi)?;
        Ok(())
    }

    /// LODSB/W/D/Q - load from [RSI] into AL/AX/EAX/RAX, advance RSI.
    pub(super) fn handle_lodsb(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_lods(Register::AL, 1)
    }
    pub(super) fn handle_lodsw(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_lods(Register::AX, 2)
    }
    pub(super) fn handle_lodsd(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_lods(Register::EAX, 4)
    }
    pub(super) fn handle_lodsq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.emit_lods(Register::RAX, 8)
    }

    fn emit_lods(&mut self, dst_reg: Register, size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let rsi = self.get_reg_val(Register::RSI)?;
        let val = self.builder.ins().load(ty, MemFlags::new(), rsi, 0);
        self.set_reg_val(dst_reg, val)?;
        let new_rsi = self.builder.ins().iadd_imm(rsi, size as i64);
        self.set_reg_val(Register::RSI, new_rsi)?;
        Ok(())
    }

    // FS/GS base (FSGSBASE) instructions

    /// RDFSBASE r64 - read FS segment base into a GPR.
    pub(super) fn handle_rdfsbase(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst = instruction.op0_register();
        let val = self.get_fs_base();
        self.set_reg_val(dst, val)
    }

    /// WRFSBASE r64 - write a GPR into the FS segment base.
    pub(super) fn handle_wrfsbase(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let src = instruction.op0_register();
        let val = self.get_reg_val(src)?;
        self.set_fs_base(val);
        Ok(())
    }

    /// RDGSBASE r64 - read GS segment base into a GPR.
    pub(super) fn handle_rdgsbase(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst = instruction.op0_register();
        let val = self.get_gs_base();
        self.set_reg_val(dst, val)
    }

    /// WRGSBASE r64 - write a GPR into the GS segment base.
    pub(super) fn handle_wrgsbase(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let src = instruction.op0_register();
        let val = self.get_reg_val(src)?;
        self.set_gs_base(val);
        Ok(())
    }
}
