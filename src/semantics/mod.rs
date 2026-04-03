use crate::MicroliftError;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::{Instruction, Mnemonic, OpKind, Register};

impl<'a> Lifter<'a> {
    pub fn handle_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.mnemonic() {
            Mnemonic::Mov => self.handle_mov(instruction),
            Mnemonic::Ret => self.handle_ret(instruction),
            Mnemonic::Add => self.handle_add(instruction),
            Mnemonic::Sub => self.handle_sub(instruction),
            Mnemonic::Xor => self.handle_xor(instruction),
            Mnemonic::Jmp => self.handle_jmp(instruction),
            Mnemonic::Je => self.handle_jz(instruction),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{:?}", instruction.mnemonic()),
            }),
        }
    }

    fn handle_mov(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let src_val = match op1 {
            OpKind::Register => {
                let reg = instruction.op1_register();
                self.get_reg_val(reg)?
            }
            OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                let imm = instruction.immediate64();
                self.builder.ins().iconst(types::I64, imm as i64)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "mov-src-unknown".into(),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                self.set_reg_val(reg, src_val)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlags::new(), src_val, ptr, 0);
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "mov-dst-unknown".into(),
                });
            }
        }

        Ok(())
    }

    fn handle_ret(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        // Simple return for now, assume rax is the return value
        let rax = self.get_reg_val(Register::RAX)?;
        self.builder.ins().return_(&[rax]);
        Ok(())
    }

    fn handle_xor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate32 | OpKind::Immediate64 => self
                .builder
                .ins()
                .iconst(types::I64, instruction.immediate64() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "xor-src".into(),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().bxor(dst_val, src_val);
                self.set_reg_val(reg, result)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "xor-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_add(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate32 => self
                .builder
                .ins()
                .iconst(types::I64, instruction.immediate32() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "add-src".into(),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().iadd(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_add(dst_val, src_val, result)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "add-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate32 => self
                .builder
                .ins()
                .iconst(types::I64, instruction.immediate32() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "sub-src".into(),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().isub(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_sub(dst_val, src_val, result)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "sub-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn update_flags_add(&mut self, lhs: Value, rhs: Value, res: Value) -> crate::Result<()> {
        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sign_bit = self.builder.ins().ushr_imm(res, 63);
        let sf = self.builder.ins().ireduce(types::I8, sign_bit);
        self.set_flag(crate::lifter::X86Flag::Sign, sf)?;

        // Carry Flag (CF)
        // result < lhs (unsigned)
        let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, res, lhs);
        self.set_flag(crate::lifter::X86Flag::Carry, cf)?;

        // Overflow Flag (OF)
        // (lhs ^ res) & (rhs ^ res) & MSB
        let xor1 = self.builder.ins().bxor(lhs, res);
        let xor2 = self.builder.ins().bxor(rhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of_bit = self.builder.ins().ushr_imm(and, 63);
        let of = self.builder.ins().ireduce(types::I8, of_bit);
        self.set_flag(crate::lifter::X86Flag::Overflow, of)?;

        Ok(())
    }

    fn update_flags_sub(&mut self, lhs: Value, rhs: Value, res: Value) -> crate::Result<()> {
        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sign_bit = self.builder.ins().ushr_imm(res, 63);
        let sf = self.builder.ins().ireduce(types::I8, sign_bit);
        self.set_flag(crate::lifter::X86Flag::Sign, sf)?;

        // Carry Flag (CF) - Borrow
        let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
        self.set_flag(crate::lifter::X86Flag::Carry, cf)?;

        // Overflow Flag (OF)
        let xor1 = self.builder.ins().bxor(lhs, rhs);
        let xor2 = self.builder.ins().bxor(lhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of_bit = self.builder.ins().ushr_imm(and, 63);
        let of = self.builder.ins().ireduce(types::I8, of_bit);
        self.set_flag(crate::lifter::X86Flag::Overflow, of)?;

        Ok(())
    }

    fn set_flag(&mut self, flag: crate::lifter::X86Flag, val: Value) -> crate::Result<()> {
        if let Some(&var) = self.state.flags.get(&flag) {
            // Ensure flag is I8
            let val_i8 = if self.builder.func.dfg.value_type(val) == types::I8 {
                val
            } else {
                self.builder.ins().ireduce(types::I8, val)
            };
            self.builder.def_var(var, val_i8);
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("flag-{:?}", flag),
            })
        }
    }

    fn handle_jmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let target_block = self.get_block(target);
        self.builder.ins().jump(target_block, &[]);
        Ok(())
    }

    fn handle_jz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block(target);
        let next_block = self.get_block(next_ip);

        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        // brif takes a i8/i32/i64 boolean-like value
        self.builder
            .ins()
            .brif(zf, target_block, &[], next_block, &[]);

        Ok(())
    }

    fn get_mem_ptr(&mut self, instruction: &Instruction, _op_index: u32) -> crate::Result<Value> {
        let base_reg = instruction.memory_base();
        let index_reg = instruction.memory_index();
        let scale = instruction.memory_index_scale();
        let disp = instruction.memory_displacement64();

        let mut ptr = if base_reg != Register::None {
            self.get_reg_val(base_reg)?
        } else {
            self.builder.ins().iconst(types::I64, 0)
        };

        if index_reg != Register::None {
            let index_val = self.get_reg_val(index_reg)?;
            let scaled_index = if scale > 1 {
                self.builder.ins().imul_imm(index_val, scale as i64)
            } else {
                index_val
            };
            ptr = self.builder.ins().iadd(ptr, scaled_index);
        }

        if disp != 0 {
            ptr = self.builder.ins().iadd_imm(ptr, disp as i64);
        }

        Ok(ptr)
    }

    fn get_reg_val(&mut self, reg: Register) -> crate::Result<Value> {
        let base_reg = get_base_reg_64(reg);
        if let Some(&var) = self.state.gprs.get(&base_reg) {
            Ok(self.builder.use_var(var))
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("reg-{:?}", reg),
            })
        }
    }

    fn set_reg_val(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        let base_reg = get_base_reg_64(reg);
        if let Some(&var) = self.state.gprs.get(&base_reg) {
            self.builder.def_var(var, val);
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("set-reg-{:?}", reg),
            })
        }
    }
}

fn get_base_reg_64(reg: Register) -> Register {
    match reg {
        Register::RAX | Register::EAX | Register::AX | Register::AL | Register::AH => Register::RAX,
        Register::RCX | Register::ECX | Register::CX | Register::CL | Register::CH => Register::RCX,
        Register::RDX | Register::EDX | Register::DX | Register::DL | Register::DH => Register::RDX,
        Register::RBX | Register::EBX | Register::BX | Register::BL | Register::BH => Register::RBX,
        Register::RSP | Register::ESP | Register::SP | Register::SPL => Register::RSP,
        Register::RBP | Register::EBP | Register::BP | Register::BPL => Register::RBP,
        Register::RSI | Register::ESI | Register::SI | Register::SIL => Register::RSI,
        Register::RDI | Register::EDI | Register::DI | Register::DIL => Register::RDI,
        Register::R8 | Register::R8D | Register::R8W | Register::R8L => Register::R8,
        Register::R9 | Register::R9D | Register::R9W | Register::R9L => Register::R9,
        Register::R10 | Register::R10D | Register::R10W | Register::R10L => Register::R10,
        Register::R11 | Register::R11D | Register::R11W | Register::R11L => Register::R11,
        Register::R12 | Register::R12D | Register::R12W | Register::R12L => Register::R12,
        Register::R13 | Register::R13D | Register::R13W | Register::R13L => Register::R13,
        Register::R14 | Register::R14D | Register::R14W | Register::R14L => Register::R14,
        Register::R15 | Register::R15D | Register::R15W | Register::R15L => Register::R15,
        _ => reg, // Default to self, might be error case
    }
}
