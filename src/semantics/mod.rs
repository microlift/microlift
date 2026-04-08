use crate::MicroliftError;
use crate::lifter::{ExecutionMode, Lifter};
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
            Mnemonic::Push => self.handle_push(instruction),
            Mnemonic::Pop => self.handle_pop(instruction),
            Mnemonic::Call => self.handle_call(instruction),
            Mnemonic::Syscall => self.handle_syscall(instruction),
            Mnemonic::Nop => self.handle_nop(instruction),
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

    fn handle_nop(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    fn handle_ret(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let imm = instruction.immediate16(); // for RET n

        let rsp = self.get_reg_val(Register::RSP)?;
        let target_ip = self.builder.ins().load(types::I64, MemFlags::new(), rsp, 0);

        let mut new_rsp = self.builder.ins().iadd_imm(rsp, 8);
        if imm > 0 {
            new_rsp = self.builder.ins().iadd_imm(new_rsp, imm as i64);
        }
        self.set_reg_val(Register::RSP, new_rsp)?;

        // Internal Dispatch via Switch
        let mut switch = cranelift_frontend::Switch::new();
        let image_base = self.builder.use_var(self.state.image_base);

        // target_offset = target_ip - image_base + lift_base
        let sub_base = self.builder.ins().isub(target_ip, image_base);
        let target_offset = self.builder.ins().iadd_imm(sub_base, self.lift_base as i64);

        for (&offset, &block) in &self.blocks {
            switch.set_entry(offset as u128, block);
        }

        let external_check_block = self.builder.create_block();
        switch.emit(&mut self.builder, target_offset, external_check_block);

        // Handle Return via Trampoline or Unknown
        self.builder.switch_to_block(external_check_block);
        self.builder.seal_block(external_check_block);

        let trampoline = self
            .builder
            .ins()
            .iconst(types::I64, self.exit_trampoline_address as i64);
        let is_trampoline = self.builder.ins().icmp(IntCC::Equal, target_ip, trampoline);

        let exit_block = self.builder.create_block();
        let unknown_block = self.builder.create_block();

        self.builder
            .ins()
            .brif(is_trampoline, exit_block, &[], unknown_block, &[]);

        // Handle Return via Trampoline
        self.builder.switch_to_block(exit_block);
        self.sync_to_context(Some(target_ip), crate::context::ExitReason::Return)?;
        self.builder.ins().return_(&[]);

        // Handle Unknown Jump/Return
        self.builder.switch_to_block(unknown_block);
        self.sync_to_context(Some(target_ip), crate::context::ExitReason::UnknownJump)?;
        self.builder.ins().return_(&[]);

        self.block_terminated = true;
        Ok(())
    }

    fn handle_push(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let push_size = 8; // Default to 8 bytes for x86-64 PUSH

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
                let imm = instruction.immediate64();
                self.builder.ins().iconst(types::I64, imm as i64)
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

    fn handle_pop(&mut self, instruction: &Instruction) -> crate::Result<()> {
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

    fn handle_call(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        // Ensure return address is a known block
        self.get_block(next_ip);

        // Push return address (current_ip) reconciled with image_base
        let image_base = self.builder.use_var(self.state.image_base);
        let offset = self.current_ip.wrapping_sub(self.lift_base);
        let ret_addr = self.builder.ins().iadd_imm(image_base, offset as i64);

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm(rsp, -8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder
            .ins()
            .store(MemFlags::new(), ret_addr, new_rsp, 0);

        // Jump to target
        let target_block = self.get_block(target);
        self.builder.ins().jump(target_block, &[]);
        self.block_terminated = true;
        Ok(())
    }

    fn handle_syscall(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.config.execution_mode {
            ExecutionMode::Jit => {
                let next_ip = instruction.next_ip();
                let offset = next_ip.wrapping_sub(self.lift_base);
                let image_base = self.builder.use_var(self.state.image_base);
                let rip_after = self.builder.ins().iadd_imm(image_base, offset as i64);

                self.sync_to_context(Some(rip_after), crate::context::ExitReason::Syscall)?;
                self.builder.ins().return_(&[]);
                self.block_terminated = true;
            }
            ExecutionMode::RuntimeAssisted => {
                let ctx_ptr = self.builder.use_var(self.state.context_ptr);
                let handler_offset =
                    std::mem::offset_of!(crate::context::RegisterContext, syscall_handler) as i32;
                let handler_ptr = self.builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    ctx_ptr,
                    handler_offset,
                );

                let has_handler = self.builder.ins().icmp_imm(IntCC::NotEqual, handler_ptr, 0);
                let assist_block = self.builder.create_block();
                let exit_block = self.builder.create_block();
                let resume_block = self.builder.create_block();

                self.builder
                    .ins()
                    .brif(has_handler, assist_block, &[], exit_block, &[]);

                self.builder.switch_to_block(assist_block);
                self.sync_to_context(None, crate::context::ExitReason::None)?;
                let sig = self
                    .syscall_handler_sig
                    .expect("runtime-assisted syscall mode requires a handler signature");
                self.builder
                    .ins()
                    .call_indirect(sig, handler_ptr, &[ctx_ptr]);
                self.load_state_from_context()?;
                self.builder.ins().jump(resume_block, &[]);

                self.builder.switch_to_block(exit_block);
                let next_ip = instruction.next_ip();
                let offset = next_ip.wrapping_sub(self.lift_base);
                let image_base = self.builder.use_var(self.state.image_base);
                let rip_after = self.builder.ins().iadd_imm(image_base, offset as i64);
                self.sync_to_context(Some(rip_after), crate::context::ExitReason::Syscall)?;
                self.builder.ins().return_(&[]);

                self.builder.switch_to_block(resume_block);
            }
        }

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
        self.block_terminated = true;
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
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

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

        // After handle_jnz, the current block in lifter.rs will be finished.
        // We should switch to the block that logically follows it in terms of instruction order.
        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

        Ok(())
    }

    fn get_mem_ptr(&mut self, instruction: &Instruction, _op_index: u32) -> crate::Result<Value> {
        let base_reg = instruction.memory_base();
        let index_reg = instruction.memory_index();
        let scale = instruction.memory_index_scale();
        let disp = instruction.memory_displacement64();

        let mut ptr = if base_reg == Register::RIP {
            // RIP-relative: iced-x86 sets 'disp' to the absolute target address (as decoded).
            // We reconcile this target address to our dynamic 'image_base'.
            let target_rip = disp;
            let offset = target_rip.wrapping_sub(self.lift_base);
            let image_base = self.builder.use_var(self.state.image_base);
            self.builder.ins().iadd_imm(image_base, offset as i64)
        } else if base_reg != Register::None {
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

        // Only add displacement if it was NOT RIP-relative.
        // For RIP-relative, the target address is already our 'ptr'.
        if base_reg != Register::RIP && disp != 0 {
            ptr = self.builder.ins().iadd_imm(ptr, disp as i64);
        }

        Ok(ptr)
    }

    fn get_reg_val(&mut self, reg: Register) -> crate::Result<Value> {
        // RIP is handled specially: reconcile it dynamically using the
        // image_base (loaded from context) + offset formula.
        if reg == Register::RIP {
            let base = self.builder.use_var(self.state.image_base);
            let offset = self.current_ip.wrapping_sub(self.lift_base);
            return Ok(self.builder.ins().iadd_imm(base, offset as i64));
        }

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
        // Writing RIP directly via a data instruction is illegal in x86-64.
        // Branches/jumps set RIP implicitly through control-flow IR, not here.
        if reg == Register::RIP {
            return Err(MicroliftError::UnsupportedInstruction {
                mnemonic: "set-RIP (direct write to RIP is illegal in x86-64)".into(),
            });
        }

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
