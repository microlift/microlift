use crate::MicroliftError;
use crate::lifter::{Lifter, X86Flag};
use crate::arch::x86_64::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

impl<'a> Lifter<'a> {
    pub(super) fn handle_add(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
            OpKind::Immediate32to64 => {
                self.builder.ins().iconst(ty, instruction.immediate32to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, Self::read_imm64(instruction)),
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
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlags::new(), ptr, 0);
                let result = self.builder.ins().iadd(dst_val, src_val);
                self.builder.ins().store(MemFlags::new(), result, ptr, 0);
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

    pub(super) fn handle_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
            OpKind::Immediate32to64 => {
                self.builder.ins().iconst(ty, instruction.immediate32to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, Self::read_imm64(instruction)),
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
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlags::new(), ptr, 0);
                let result = self.builder.ins().isub(dst_val, src_val);
                self.builder.ins().store(MemFlags::new(), result, ptr, 0);
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

    pub(super) fn handle_adc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        // Read CF (I8, 0 or 1), extend to operand type for arithmetic.
        let cf_i8 = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let cf_ext = if ty == types::I8 {
            cf_i8
        } else {
            self.builder.ins().uextend(ty, cf_i8)
        };

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate32to64 => {
                self.builder.ins().iconst(ty, instruction.immediate32to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, Self::read_imm64(instruction)),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("adc-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                // Two-step add to properly propagate carry.
                let tmp = self.builder.ins().iadd(dst_val, src_val);
                let cf1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, dst_val);
                let result = self.builder.ins().iadd(tmp, cf_ext);
                let cf2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, result, tmp);
                let cf_out = self.builder.ins().bor(cf1, cf2);
                self.set_reg_val(reg, result)?;
                self.update_flags_adc(dst_val, src_val, result, cf_out, size)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlags::new(), ptr, 0);
                let tmp = self.builder.ins().iadd(dst_val, src_val);
                let cf1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, dst_val);
                let result = self.builder.ins().iadd(tmp, cf_ext);
                let cf2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, result, tmp);
                let cf_out = self.builder.ins().bor(cf1, cf2);
                self.builder.ins().store(MemFlags::new(), result, ptr, 0);
                self.update_flags_adc(dst_val, src_val, result, cf_out, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "adc-dst".into(),
                });
            }
        }
        Ok(())
    }

    pub(super) fn handle_sbb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let op1 = instruction.op1_kind();

        let size = if op0 == OpKind::Register {
            instruction.op0_register().size()
        } else {
            instruction.memory_size().size()
        };
        let ty = size_to_type(size);

        // Read CF (borrow-in), extend to operand type.
        let cf_i8 = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let cf_ext = if ty == types::I8 {
            cf_i8
        } else {
            self.builder.ins().uextend(ty, cf_i8)
        };

        let src_val = match op1 {
            OpKind::Register => self.get_reg_val(instruction.op1_register())?,
            OpKind::Immediate32to64 => {
                self.builder.ins().iconst(ty, instruction.immediate32to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, Self::read_imm64(instruction)),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("sbb-src-{:?}", op1),
                });
            }
        };

        match op0 {
            OpKind::Register => {
                let reg = instruction.op0_register();
                let dst_val = self.get_reg_val(reg)?;
                // Two-step sub to properly propagate borrow.
                let tmp = self.builder.ins().isub(dst_val, src_val);
                let borrow1 =
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThan, dst_val, src_val);
                let result = self.builder.ins().isub(tmp, cf_ext);
                let borrow2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, cf_ext);
                let cf_out = self.builder.ins().bor(borrow1, borrow2);
                self.set_reg_val(reg, result)?;
                self.update_flags_sbb(dst_val, src_val, result, cf_out, size)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlags::new(), ptr, 0);
                let tmp = self.builder.ins().isub(dst_val, src_val);
                let borrow1 =
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThan, dst_val, src_val);
                let result = self.builder.ins().isub(tmp, cf_ext);
                let borrow2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, cf_ext);
                let cf_out = self.builder.ins().bor(borrow1, borrow2);
                self.builder.ins().store(MemFlags::new(), result, ptr, 0);
                self.update_flags_sbb(dst_val, src_val, result, cf_out, size)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "sbb-dst".into(),
                });
            }
        }
        Ok(())
    }

    pub(super) fn handle_xadd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let dst_val = self.read_operand(instruction, 0, size)?;
        let src_reg = instruction.op1_register();
        let src_val = self.get_reg_val(src_reg)?;

        let result = self.builder.ins().iadd(dst_val, src_val);
        self.write_operand(instruction, 0, result)?;
        self.set_reg_val(src_reg, dst_val)?;
        self.update_flags_add(dst_val, src_val, result, size)?;
        Ok(())
    }

    pub(super) fn handle_cmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
            OpKind::Immediate32to64 => {
                self.builder.ins().iconst(ty, instruction.immediate32to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, Self::read_imm64(instruction)),
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

    pub(super) fn handle_inc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let dst_val = self.read_operand(instruction, 0, size)?;
        let one = self.builder.ins().iconst(ty, 1);
        let result = self.builder.ins().iadd(dst_val, one);
        self.write_operand(instruction, 0, result)?;
        self.update_flags_inc_dec(dst_val, one, result, true, size)?;
        Ok(())
    }

    pub(super) fn handle_dec(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let dst_val = self.read_operand(instruction, 0, size)?;
        let one = self.builder.ins().iconst(ty, 1);
        let result = self.builder.ins().isub(dst_val, one);
        self.write_operand(instruction, 0, result)?;
        self.update_flags_inc_dec(dst_val, one, result, false, size)?;
        Ok(())
    }

    pub(super) fn handle_neg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let val = self.read_operand(instruction, 0, size)?;
        let zero = self.builder.ins().iconst(ty, 0);
        let result = self.builder.ins().isub(zero, val);
        self.write_operand(instruction, 0, result)?;
        // NEG sets flags as if doing 0 - val
        self.update_flags_sub(zero, val, result, size)?;
        Ok(())
    }

    /// IMUL - handles 1, 2, and 3-operand forms.
    ///
    /// - 1-operand: `imul r/m` -> RAX * src -> RDX:RAX (signed 128-bit)
    /// - 2-operand: `imul dst, src` -> dst = dst * src (truncated, flags set)
    /// - 3-operand: `imul dst, src, imm` -> dst = src * imm (truncated, flags set)
    pub(super) fn handle_imul(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op_count() {
            1 => {
                // 1-operand form: RDX:RAX = RAX * r/m
                let size = self.operand_size(instruction, 0)?;
                let ty = size_to_type(size);
                let rax = self.get_reg_val(Register::RAX)?;
                let src = self.read_operand(instruction, 0, size)?;

                let (rax_narrow, src_narrow) = if size == 8 {
                    (rax, src)
                } else {
                    (
                        self.builder.ins().ireduce(ty, rax),
                        self.builder.ins().ireduce(ty, src),
                    )
                };

                // Low half
                let lo = self.builder.ins().imul(rax_narrow, src_narrow);
                // High half (signed)
                let hi = self.builder.ins().smulhi(rax_narrow, src_narrow);

                // Write results based on size:
                // 8-bit -> AX=AH:AL, 16-bit -> DX:AX, 32-bit -> EDX:EAX, 64-bit -> RDX:RAX
                let (lo_reg, hi_reg) = match size {
                    1 => {
                        // Result in AX; AH=hi8, AL=lo8
                        let lo_64 = self.builder.ins().uextend(types::I64, lo);
                        let hi_64 = self.builder.ins().uextend(types::I64, hi);
                        let hi_shifted = self.builder.ins().ishl_imm(hi_64, 8);
                        let ax_val = self.builder.ins().bor(lo_64, hi_shifted);
                        let ax16 = self.builder.ins().ireduce(types::I16, ax_val);
                        self.set_reg_val(Register::AX, ax16)?;
                        // CF/OF set if AH != sign extension of AL
                        let sign_ext = self.builder.ins().sshr_imm(lo, 7);
                        let sign_ext8 = self.builder.ins().ireduce(types::I8, sign_ext);
                        let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext8, hi);
                        self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                        self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
                        return Ok(());
                    }
                    2 => (Register::AX, Register::DX),
                    4 => (Register::EAX, Register::EDX),
                    8 => (Register::RAX, Register::RDX),
                    _ => unreachable!(),
                };
                self.set_reg_val(lo_reg, lo)?;
                self.set_reg_val(hi_reg, hi)?;

                // CF/OF: set if hi != sign extension of lo
                let sign_ext = self.builder.ins().sshr_imm(lo, (size * 8 - 1) as i64);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            2 => {
                // 2-operand: dst = dst * src
                let size = self.operand_size(instruction, 0)?;
                let dst = self.read_operand(instruction, 0, size)?;
                let src = self.read_operand(instruction, 1, size)?;
                let result = self.builder.ins().imul(dst, src);
                self.write_operand(instruction, 0, result)?;
                // OF/CF: set if result != sign-extended truncation
                let hi = self.builder.ins().smulhi(dst, src);
                let sign_ext = self.builder.ins().sshr_imm(result, (size * 8 - 1) as i64);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            3 => {
                // 3-operand: dst = src1 * imm
                let size = self.operand_size(instruction, 0)?;
                let src = self.read_operand(instruction, 1, size)?;
                let imm = self.read_operand(instruction, 2, size)?;
                let result = self.builder.ins().imul(src, imm);
                self.write_operand(instruction, 0, result)?;
                // OF/CF
                let hi = self.builder.ins().smulhi(src, imm);
                let sign_ext = self.builder.ins().sshr_imm(result, (size * 8 - 1) as i64);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "imul (unexpected op count)".into(),
                });
            }
        }
        Ok(())
    }

    /// MUL - unsigned multiply, 1-operand form.
    /// 8-bit: AX = AL * r/m8
    /// 16-bit: DX:AX = AX * r/m16
    /// 32-bit: EDX:EAX = EAX * r/m32
    /// 64-bit: RDX:RAX = RAX * r/m64
    pub(super) fn handle_mul(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let src = self.read_operand(instruction, 0, size)?;

        match size {
            1 => {
                let rax_val = self.get_reg_val(Register::RAX)?;
                let al = self.builder.ins().ireduce(types::I8, rax_val);
                let lo = self.builder.ins().imul(al, src);
                let hi = self.builder.ins().umulhi(al, src);
                let lo_64 = self.builder.ins().uextend(types::I64, lo);
                let hi_64 = self.builder.ins().uextend(types::I64, hi);
                let hi_shifted = self.builder.ins().ishl_imm(hi_64, 8);
                let ax_val = self.builder.ins().bor(lo_64, hi_shifted);
                let ax16 = self.builder.ins().ireduce(types::I16, ax_val);
                self.set_reg_val(Register::AX, ax16)?;
                // CF/OF set if AH != 0
                let of_cf = self.builder.ins().icmp_imm(IntCC::NotEqual, hi, 0);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            2 | 4 | 8 => {
                let (acc_reg, lo_reg, hi_reg) = match size {
                    2 => (Register::AX, Register::AX, Register::DX),
                    4 => (Register::EAX, Register::EAX, Register::EDX),
                    8 => (Register::RAX, Register::RAX, Register::RDX),
                    _ => unreachable!(),
                };
                let acc = self.get_reg_val(acc_reg)?;
                let acc_narrow = if size == 8 {
                    acc
                } else {
                    self.builder.ins().ireduce(ty, acc)
                };
                let lo = self.builder.ins().imul(acc_narrow, src);
                let hi = self.builder.ins().umulhi(acc_narrow, src);
                self.set_reg_val(lo_reg, lo)?;
                self.set_reg_val(hi_reg, hi)?;
                // CF/OF set if hi != 0
                let zero = self.builder.ins().iconst(ty, 0);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, hi, zero);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("mul-size-{}", size),
                });
            }
        }
        Ok(())
    }

    /// IDIV - signed divide.
    /// 8-bit:  AL = AX / r/m8, AH = AX % r/m8
    /// 16-bit: AX = DX:AX / r/m16, DX = DX:AX % r/m16
    /// 32/64:  similar for EAX/EDX and RAX/RDX
    /// Note: we emit `sdiv`/`srem` on the low half only (no 128-bit dividend).
    /// Full 128-bit divide would require a runtime helper; this is a pragmatic approximation.
    pub(super) fn handle_idiv(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let divisor = self.read_operand(instruction, 0, size)?;

        // Special case: 8-bit IDIV uses AX (signed 16-bit) as the dividend,
        // not AL. We must widen the divisor to I16 and divide in I16, then
        // truncate the quotient/remainder back into AL/AH.
        if size == 1 {
            let dividend = self.get_reg_val(Register::AX)?; // I16
            let div16 = self.builder.ins().sextend(types::I16, divisor);
            let quot = self.builder.ins().sdiv(dividend, div16);
            let rem = self.builder.ins().srem(dividend, div16);
            let quot8 = self.builder.ins().ireduce(types::I8, quot);
            let rem8 = self.builder.ins().ireduce(types::I8, rem);
            self.set_reg_val(Register::AL, quot8)?;
            self.set_reg_val(Register::AH, rem8)?;
            return Ok(());
        }

        let (quot_reg, rem_reg, dividend) = match size {
            1 => unreachable!("handled above"),
            2 => {
                let ax = self.get_reg_val(Register::AX)?;
                let ax16 = self.builder.ins().ireduce(types::I16, ax);
                (Register::AX, Register::DX, ax16)
            }
            4 => {
                let eax = self.get_reg_val(Register::EAX)?;
                (Register::EAX, Register::EDX, eax)
            }
            8 => {
                let rax = self.get_reg_val(Register::RAX)?;
                (Register::RAX, Register::RDX, rax)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("idiv-size-{}", size),
                });
            }
        };

        let quot = self.builder.ins().sdiv(dividend, divisor);
        let rem = self.builder.ins().srem(dividend, divisor);
        self.set_reg_val(quot_reg, quot)?;
        self.set_reg_val(rem_reg, rem)?;
        // IDIV leaves flags undefined per x86 manual - we don't update them.
        Ok(())
    }

    /// DIV - unsigned divide (same layout as IDIV but unsigned).
    pub(super) fn handle_div(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let divisor = self.read_operand(instruction, 0, size)?;

        // Special case: 8-bit DIV uses AX (unsigned 16-bit) as the dividend,
        // not AL. Widen divisor to I16, divide in I16, truncate back.
        if size == 1 {
            let dividend = self.get_reg_val(Register::AX)?; // I16
            let div16 = self.builder.ins().uextend(types::I16, divisor);
            let quot = self.builder.ins().udiv(dividend, div16);
            let rem = self.builder.ins().urem(dividend, div16);
            let quot8 = self.builder.ins().ireduce(types::I8, quot);
            let rem8 = self.builder.ins().ireduce(types::I8, rem);
            self.set_reg_val(Register::AL, quot8)?;
            self.set_reg_val(Register::AH, rem8)?;
            return Ok(());
        }

        let (quot_reg, rem_reg, dividend) = match size {
            1 => unreachable!("handled above"),
            2 => {
                let ax = self.get_reg_val(Register::AX)?;
                let ax16 = self.builder.ins().ireduce(types::I16, ax);
                (Register::AX, Register::DX, ax16)
            }
            4 => {
                let eax = self.get_reg_val(Register::EAX)?;
                (Register::EAX, Register::EDX, eax)
            }
            8 => {
                let rax = self.get_reg_val(Register::RAX)?;
                (Register::RAX, Register::RDX, rax)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("div-size-{}", size),
                });
            }
        };

        let quot = self.builder.ins().udiv(dividend, divisor);
        let rem = self.builder.ins().urem(dividend, divisor);
        self.set_reg_val(quot_reg, quot)?;
        self.set_reg_val(rem_reg, rem)?;
        // DIV leaves flags undefined per x86 manual.
        Ok(())
    }

    // Sign-extension conversions (implicit operand form)

    /// CBW - AX = sign_extend(AL)
    pub(super) fn handle_cbw(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let al = self.get_reg_val(Register::AL)?;
        let ax = self.builder.ins().sextend(types::I16, al);
        self.set_reg_val(Register::AX, ax)
    }

    /// CWDE - EAX = sign_extend(AX)  (upper 32 bits of RAX zeroed per x86-64)
    pub(super) fn handle_cwde(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ax = self.get_reg_val(Register::AX)?;
        let eax = self.builder.ins().sextend(types::I32, ax);
        self.set_reg_val(Register::EAX, eax)
    }

    /// CDQE - RAX = sign_extend(EAX)
    pub(super) fn handle_cdqe(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let eax = self.get_reg_val(Register::EAX)?;
        let rax = self.builder.ins().sextend(types::I64, eax);
        self.set_reg_val(Register::RAX, rax)
    }

    /// CWD - DX:AX = sign_extend(AX)  (DX = 0 or 0xFFFF)
    pub(super) fn handle_cwd(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ax = self.get_reg_val(Register::AX)?;
        let dx = self.builder.ins().sshr_imm(ax, 15);
        self.set_reg_val(Register::DX, dx)
    }

    /// CDQ - EDX:EAX = sign_extend(EAX)  (EDX = 0 or 0xFFFFFFFF)
    pub(super) fn handle_cdq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let eax = self.get_reg_val(Register::EAX)?;
        let edx = self.builder.ins().sshr_imm(eax, 31);
        self.set_reg_val(Register::EDX, edx)
    }

    /// CQO - RDX:RAX = sign_extend(RAX)  (RDX = 0 or -1)
    pub(super) fn handle_cqo(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rax = self.get_reg_val(Register::RAX)?;
        let rdx = self.builder.ins().sshr_imm(rax, 63);
        self.set_reg_val(Register::RDX, rdx)
    }
}
