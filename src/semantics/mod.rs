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
            Mnemonic::And => self.handle_and(instruction),
            Mnemonic::Or => self.handle_or(instruction),
            Mnemonic::Test => self.handle_test(instruction),
            Mnemonic::Cmp => self.handle_cmp(instruction),
            Mnemonic::Inc => self.handle_inc(instruction),
            Mnemonic::Dec => self.handle_dec(instruction),
            Mnemonic::Lea => self.handle_lea(instruction),
            Mnemonic::Jmp => self.handle_jmp(instruction),
            Mnemonic::Je => self.handle_jz(instruction),
            Mnemonic::Jne => self.handle_jnz(instruction),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{:?}", instruction.mnemonic()),
            }),
        }
    }

    fn handle_mov(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                let imm = instruction.immediate64();
                self.builder.ins().iconst(ty, imm as i64)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("mov-src-{:?}", op1),
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
        // Sync modified state back to context before returning
        self.sync_to_context()?;
        self.builder.ins().return_(&[]);
        Ok(())
    }

    fn handle_xor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("xor-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().bxor(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_logical(result, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "xor-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_and(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("and-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().band(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_logical(result, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "and-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_or(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("or-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().bor(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_logical(result, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "or-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_test(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("test-src-{:?}", op1),
                });
            }
        };

        let dst_val = match op0 {
            OpKind::Register => self.get_reg_val(instruction.op0_register())?,
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("test-dst-{:?}", op0),
                });
            }
        };

        let result = self.builder.ins().band(dst_val, src_val);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_add(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("add-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().iadd(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_add(dst_val, src_val, result, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "add-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_inc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let one = self.builder.ins().iconst(ty, 1);
                let result = self.builder.ins().iadd(dst_val, one);
                self.set_reg_val(reg, result)?;
                self.update_flags_inc_dec(dst_val, one, result, true, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "inc-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_dec(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let one = self.builder.ins().iconst(ty, 1);
                let result = self.builder.ins().isub(dst_val, one);
                self.set_reg_val(reg, result)?;
                self.update_flags_inc_dec(dst_val, one, result, false, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "dec-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("sub-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                let result = self.builder.ins().isub(dst_val, src_val);
                self.set_reg_val(reg, result)?;
                self.update_flags_sub(dst_val, src_val, result, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "sub-dst".into(),
                });
            }
        }
        Ok(())
    }

    fn handle_cmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("cmp-src-{:?}", op1),
                });
            }
        };

        let dst_val = match op0 {
            OpKind::Register => self.get_reg_val(instruction.op0_register())?,
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("cmp-dst-{:?}", op0),
                });
            }
        };

        let result = self.builder.ins().isub(dst_val, src_val);
        self.update_flags_sub(dst_val, src_val, result, size)?;
        Ok(())
    }

    fn handle_lea(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let reg = instruction.op0_register();
        let ptr = self.get_mem_ptr(instruction, 1)?;
        self.set_reg_val(reg, ptr)?;
        Ok(())
    }

    fn update_flags_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
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
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(crate::lifter::X86Flag::Overflow, of)?;

        Ok(())
    }

    fn update_flags_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        res: Value,
        size: usize,
    ) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(crate::lifter::X86Flag::Sign, sf)?;

        // Carry Flag (CF) - Borrow
        let cf = self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
        self.set_flag(crate::lifter::X86Flag::Carry, cf)?;

        // Overflow Flag (OF)
        let xor1 = self.builder.ins().bxor(lhs, rhs);
        let xor2 = self.builder.ins().bxor(lhs, res);
        let and = self.builder.ins().band(xor1, xor2);
        let of = self.builder.ins().ushr_imm(and, bit_width - 1);
        self.set_flag(crate::lifter::X86Flag::Overflow, of)?;

        Ok(())
    }

    fn update_flags_logical(&mut self, res: Value, size: usize) -> crate::Result<()> {
        let bit_width = (size * 8) as i64;

        // Zero Flag (ZF)
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, res, 0);
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(crate::lifter::X86Flag::Sign, sf)?;

        // Carry and Overflow are cleared for logical ops
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(crate::lifter::X86Flag::Carry, zero)?;
        self.set_flag(crate::lifter::X86Flag::Overflow, zero)?;

        Ok(())
    }

    fn update_flags_inc_dec(
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
        self.set_flag(crate::lifter::X86Flag::Zero, zf)?;

        // Sign Flag (SF)
        let sf = self.builder.ins().ushr_imm(res, bit_width - 1);
        self.set_flag(crate::lifter::X86Flag::Sign, sf)?;

        // Overflow Flag (OF)
        let of = if is_add {
            // (lhs ^ res) & (rhs ^ res) & MSB. But for INC, rhs is 1.
            let xor1 = self.builder.ins().bxor(lhs, res);
            let xor2 = self.builder.ins().bxor(rhs, res);
            let and = self.builder.ins().band(xor1, xor2);
            self.builder.ins().ushr_imm(and, bit_width - 1)
        } else {
            // (lhs ^ rhs) & (lhs ^ res) & MSB.
            let xor1 = self.builder.ins().bxor(lhs, rhs);
            let xor2 = self.builder.ins().bxor(lhs, res);
            let and = self.builder.ins().band(xor1, xor2);
            self.builder.ins().ushr_imm(and, bit_width - 1)
        };
        self.set_flag(crate::lifter::X86Flag::Overflow, of)?;

        // Carry Flag (CF) is NOT affected by INC/DEC
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

        // Switch to the next block for fallthrough
        self.builder.switch_to_block(next_block);

        Ok(())
    }

    fn handle_jnz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block(target);
        let next_block = self.get_block(next_ip);

        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        // Jump if ZF is 0
        self.builder
            .ins()
            .brif(zf, next_block, &[], target_block, &[]);

        // Switch to the target block for fallthrough (which is target_block? No, it should be the NEXT instruction's block)
        // Wait, standard brif(c, t1, t2) -> t1 if c.
        // My handle_jnz: brif(zf, next_block, target_block) -> if zf is 1 (zero), skip to next_block. If zf is 0 (not zero), jump to target_block.
        // So target_block is WHERE we jump if NOT zero.
        // The fallthrough after the branch instruction is next_ip.
        // So next_block is ALWAYS where we continue if the branch isn't taken?
        // Let's re-think JNZ (Jump Not Zero).
        // if zero (ZF=1) -> fallthrough (next_ip)
        // if not zero (ZF=0) -> jump (target)
        // My brif(zf, next_block, target_block) does exactly that!

        // After handle_jnz, the current block in lifter.rs will be finished.
        // We should switch to the block that logically follows it in terms of instruction order.
        self.builder.switch_to_block(next_block);

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
        let size = reg.size();
        let is_high_8 = is_high_8_reg(reg);

        if let Some(&var) = self.state.gprs.get(&base_reg) {
            let full_val = self.builder.use_var(var);
            if size == 8 {
                Ok(full_val)
            } else if is_high_8 {
                let shifted = self.builder.ins().ushr_imm(full_val, 8);
                Ok(self.builder.ins().ireduce(types::I8, shifted))
            } else {
                let ty = size_to_type(size);
                Ok(self.builder.ins().ireduce(ty, full_val))
            }
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("reg-{:?}", reg),
            })
        }
    }

    fn set_reg_val(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        let base_reg = get_base_reg_64(reg);
        let size = reg.size();
        let is_high_8 = is_high_8_reg(reg);

        if let Some(&var) = self.state.gprs.get(&base_reg) {
            let old_val = self.builder.use_var(var);
            let new_full_val = if size == 8 {
                val
            } else if size == 4 {
                // 32-bit zero-extends to 64-bit
                self.builder.ins().uextend(types::I64, val)
            } else if is_high_8 {
                // AH, BH, CH, DH: bits 8-15
                let val_64 = self.builder.ins().uextend(types::I64, val);
                let shifted_val = self.builder.ins().ishl_imm(val_64, 8);
                let mask = self.builder.ins().iconst(types::I64, !0xFF00u64 as i64);
                let cleared = self.builder.ins().band(old_val, mask);
                self.builder.ins().bor(cleared, shifted_val)
            } else {
                // 8 or 16 bit: preserve upper bits
                let mask_val = match size {
                    1 => !0xFFu64,
                    2 => !0xFFFFu64,
                    _ => unreachable!(),
                };
                let mask = self.builder.ins().iconst(types::I64, mask_val as i64);
                let cleared = self.builder.ins().band(old_val, mask);
                let val_64 = self.builder.ins().uextend(types::I64, val);
                self.builder.ins().bor(cleared, val_64)
            };
            self.builder.def_var(var, new_full_val);
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("set-reg-{:?}", reg),
            })
        }
    }
}

fn size_to_type(size: usize) -> Type {
    match size {
        1 => types::I8,
        2 => types::I16,
        4 => types::I32,
        8 => types::I64,
        _ => panic!("Unsupported operand size: {}", size),
    }
}

fn is_high_8_reg(reg: Register) -> bool {
    matches!(
        reg,
        Register::AH | Register::BH | Register::CH | Register::DH
    )
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
