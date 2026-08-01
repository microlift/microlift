use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::ExitReason;
use crate::lifter::{Lifter, X86Flag};
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
            OpKind::Immediate32to64 => self.builder.ins().iconst(ty, instruction.immediate32to64()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                self.builder.ins().iconst(ty, Self::read_imm64(instruction))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let result = self.builder.ins().iadd(dst_val, src_val);
                self.builder
                    .ins()
                    .store(MemFlagsData::new(), result, ptr, 0);
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
            OpKind::Immediate32to64 => self.builder.ins().iconst(ty, instruction.immediate32to64()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                self.builder.ins().iconst(ty, Self::read_imm64(instruction))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let result = self.builder.ins().isub(dst_val, src_val);
                self.builder
                    .ins()
                    .store(MemFlagsData::new(), result, ptr, 0);
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
            OpKind::Immediate32to64 => self.builder.ins().iconst(ty, instruction.immediate32to64()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                self.builder.ins().iconst(ty, Self::read_imm64(instruction))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
                let cf1 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, tmp, dst_val);
                let result = self.builder.ins().iadd(tmp, cf_ext);
                let cf2 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, result, tmp);
                let cf_out = self.builder.ins().bor(cf1, cf2);
                self.set_reg_val(reg, result)?;
                self.update_flags_adc(dst_val, src_val, result, cf_out, size)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let tmp = self.builder.ins().iadd(dst_val, src_val);
                let cf1 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, tmp, dst_val);
                let result = self.builder.ins().iadd(tmp, cf_ext);
                let cf2 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, result, tmp);
                let cf_out = self.builder.ins().bor(cf1, cf2);
                self.builder
                    .ins()
                    .store(MemFlagsData::new(), result, ptr, 0);
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
            OpKind::Immediate32to64 => self.builder.ins().iconst(ty, instruction.immediate32to64()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                self.builder.ins().iconst(ty, Self::read_imm64(instruction))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
                let borrow1 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, dst_val, src_val);
                let result = self.builder.ins().isub(tmp, cf_ext);
                let borrow2 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, tmp, cf_ext);
                let cf_out = self.builder.ins().bor(borrow1, borrow2);
                self.set_reg_val(reg, result)?;
                self.update_flags_sbb(dst_val, src_val, result, cf_out, size)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let tmp = self.builder.ins().isub(dst_val, src_val);
                let borrow1 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, dst_val, src_val);
                let result = self.builder.ins().isub(tmp, cf_ext);
                let borrow2 = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, tmp, cf_ext);
                let cf_out = self.builder.ins().bor(borrow1, borrow2);
                self.builder
                    .ins()
                    .store(MemFlagsData::new(), result, ptr, 0);
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
        // Hardware order: TEMP <- SRC + DEST; SRC <- DEST; DEST <- TEMP. The
        // destination write must come last so `xadd r, r` (same register both
        // ways) ends up holding the sum, not the old value. For a memory
        // destination the store goes first instead: its effective address must
        // be computed from the pre-instruction value of the source register
        // (which may be a base/index of the address).
        if instruction.op0_kind() == OpKind::Register {
            self.set_reg_val(src_reg, dst_val)?;
            self.write_operand(instruction, 0, result)?;
        } else {
            self.write_operand(instruction, 0, result)?;
            self.set_reg_val(src_reg, dst_val)?;
        }
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
            OpKind::Immediate32to64 => self.builder.ins().iconst(ty, instruction.immediate32to64()),
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                self.builder.ins().iconst(ty, Self::read_imm64(instruction))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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

                // `rax` is the whole 64-bit register and needs narrowing;
                // `src` was read at the operand's own width and usually does
                // not. See [`Lifter::narrow_to`].
                let rax_narrow = self.narrow_to(ty, rax);
                let src_narrow = self.narrow_to(ty, src);

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
                // `acc_reg` is AX/EAX/RAX, so this is already the operand's
                // width - but say so rather than assume it.
                let acc_narrow = self.narrow_to(ty, acc);
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

    /// Assemble the full double-width dividend `high:low` for DIV/IDIV in
    /// `work_ty` (twice the operand width). The concatenation is purely
    /// bitwise, so the same routine serves both the signed and unsigned forms -
    /// `sdiv`/`srem` reinterpret the assembled bits as a signed 2N-bit value.
    fn assemble_dividend(
        &mut self,
        low: Register,
        high: Register,
        work_ty: Type,
    ) -> crate::Result<Value> {
        let shift = (low.size() * 8) as i64;
        let lo = self.get_reg_val(low)?;
        let hi = self.get_reg_val(high)?;
        let lo_w = self.builder.ins().uextend(work_ty, lo);
        let hi_w = self.builder.ins().uextend(work_ty, hi);
        let hi_shifted = self.builder.ins().ishl_imm(hi_w, shift);
        Ok(self.builder.ins().bor(hi_shifted, lo_w))
    }

    /// The `(low_acc, high_acc, working_type)` triple for a DIV/IDIV of the
    /// given operand `size`: the dividend is `high:low` (e.g. `DX:AX`) and the
    /// division is carried out in `working_type`, twice the operand width.
    fn div_regs(size: usize) -> Option<(Register, Register, Type)> {
        match size {
            1 => Some((Register::AL, Register::AH, types::I16)),
            2 => Some((Register::AX, Register::DX, types::I32)),
            4 => Some((Register::EAX, Register::EDX, types::I64)),
            8 => Some((Register::RAX, Register::RDX, types::I128)),
            _ => None,
        }
    }

    /// A zero constant of width `ty` (`iconst` only builds up to `I64`, so `I128`
    /// is materialised by zero-extending an `I64` zero).
    fn wide_zero(&mut self, ty: Type) -> Value {
        if ty == types::I128 {
            let z = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().uextend(types::I128, z)
        } else {
            self.builder.ins().iconst(ty, 0)
        }
    }

    /// Unsigned `dividend / divisor` in `work_ty`, returning `(quot, rem)`.
    /// Uses native `udiv`/`urem` where the backend supports it, falling back to
    /// the portable long-division loop for the unsupported `I128` case. The
    /// caller must have already excluded a zero divisor.
    fn emit_udivmod(&mut self, dividend: Value, divisor: Value, work_ty: Type) -> (Value, Value) {
        if work_ty == types::I128 {
            self.emit_udivmod128(dividend, divisor)
        } else {
            let q = self.builder.ins().udiv(dividend, divisor);
            let r = self.builder.ins().urem(dividend, divisor);
            (q, r)
        }
    }

    /// IDIV - signed divide, full `high:low` dividend.
    /// 8-bit:  AL = AX / r/m8, AH = AX % r/m8
    /// 16-bit: AX = DX:AX / r/m16, DX = DX:AX % r/m16
    /// 32/64:  EDX:EAX / r/m32 and RDX:RAX / r/m64 respectively.
    pub(super) fn handle_idiv(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_division(instruction, true)
    }

    /// DIV - unsigned divide (same `high:low` dividend layout as IDIV).
    pub(super) fn handle_div(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_division(instruction, false)
    }

    /// Shared DIV/IDIV lowering with architectural divide-error (#DE) checks.
    ///
    /// x86 raises #DE when the divisor is zero or the quotient does not fit in
    /// the destination width. Both are checked *before* any result is committed
    /// (a fault leaves all registers unchanged), and either one branches to a
    /// [`ExitReason::DivideError`] VM exit whose PC points at the faulting
    /// instruction. Because the guards run first, the division itself never sees
    /// a zero divisor, so the emitted `udiv`/`urem` (or the I128 loop) can't trap.
    fn emit_division(&mut self, instruction: &Instruction, signed: bool) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let (low, high, work_ty) =
            Self::div_regs(size).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("div-size-{}", size),
            })?;
        let ty = size_to_type(size);
        let n = (size * 8) as i64;

        let divisor_n = self.read_operand(instruction, 0, size)?;
        let dividend = self.assemble_dividend(low, high, work_ty)?;

        // Fault path, shared by the divide-by-zero and overflow guards.
        let de_block = self.builder.create_block();

        // Guard 1: divide by zero.
        let div0 = self.builder.ins().icmp_imm(IntCC::Equal, divisor_n, 0);
        let after_div0 = self.builder.create_block();
        self.builder
            .ins()
            .brif(div0, de_block, &[], after_div0, &[]);
        self.builder.switch_to_block(after_div0);
        self.builder.seal_block(after_div0);

        // Divisor is now known non-zero; compute the full-width quotient/remainder.
        let (quot, rem) = if signed {
            let divisor_w = self.builder.ins().sextend(work_ty, divisor_n);
            let zero = self.wide_zero(work_ty);
            let n_neg = self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, dividend, zero);
            let d_neg = self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, divisor_w, zero);
            // Divide magnitudes, then reapply signs (x86 truncates toward zero;
            // the remainder takes the dividend's sign). Working on magnitudes
            // also keeps the division away from the `INT_MIN / -1` trap case.
            let neg_n = self.builder.ins().isub(zero, dividend);
            let abs_n = self.builder.ins().select(n_neg, neg_n, dividend);
            let neg_d = self.builder.ins().isub(zero, divisor_w);
            let abs_d = self.builder.ins().select(d_neg, neg_d, divisor_w);

            let (uq, ur) = self.emit_udivmod(abs_n, abs_d, work_ty);

            let q_neg = self.builder.ins().bxor(n_neg, d_neg);
            let neg_uq = self.builder.ins().isub(zero, uq);
            let quot = self.builder.ins().select(q_neg, neg_uq, uq);
            let neg_ur = self.builder.ins().isub(zero, ur);
            let rem = self.builder.ins().select(n_neg, neg_ur, ur);
            (quot, rem)
        } else {
            let divisor_w = self.builder.ins().uextend(work_ty, divisor_n);
            self.emit_udivmod(dividend, divisor_w, work_ty)
        };

        // Guard 2: quotient overflow.
        //   signed:   the low N bits, sign-extended, must reproduce the quotient.
        //   unsigned: the bits above N must all be zero.
        let overflow = if signed {
            let q_trunc = self.builder.ins().ireduce(ty, quot);
            let q_back = self.builder.ins().sextend(work_ty, q_trunc);
            self.builder.ins().icmp(IntCC::NotEqual, q_back, quot)
        } else {
            let hi = self.builder.ins().ushr_imm(quot, n);
            self.builder.ins().icmp_imm(IntCC::NotEqual, hi, 0)
        };
        let after_ovf = self.builder.create_block();
        self.builder
            .ins()
            .brif(overflow, de_block, &[], after_ovf, &[]);

        // Fault path first: emit the #DE exit so `de_block` is terminated before
        // we switch away to the continuation (the frontend forbids leaving an
        // unfilled block).
        self.builder.switch_to_block(de_block);
        self.builder.seal_block(de_block);
        self.emit_divide_error(instruction)?;

        // Non-faulting path: commit quotient and remainder, then leave the
        // builder here so the engine keeps lifting from the continuation.
        self.builder.switch_to_block(after_ovf);
        self.builder.seal_block(after_ovf);
        let quot_n = self.builder.ins().ireduce(ty, quot);
        let rem_n = self.builder.ins().ireduce(ty, rem);
        self.set_reg_val(low, quot_n)?;
        self.set_reg_val(high, rem_n)?;
        // DIV/IDIV leave the arithmetic flags undefined per the x86 manual.
        self.block_terminated = false;
        Ok(())
    }

    /// Emit a divide-error (#DE) VM exit: sync guest state with the PC pointing
    /// *at* the faulting `DIV`/`IDIV` (faults re-execute) and return to the host.
    fn emit_divide_error(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let offset = instruction.ip().wrapping_sub(self.lift_base);
        let image_base = self.builder.use_var(self.state.image_base);
        let rip_at = self.builder.ins().iadd_imm(image_base, offset as i64);
        self.sync_to_context(Some(rip_at), ExitReason::DivideError)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// Unsigned 128-bit / 128-bit division via restoring shift/subtract long
    /// division, returning `(quotient, remainder)` as `I128`.
    ///
    /// The backend lacks native `udiv.i128`/`urem.i128`, but it does support
    /// i128 shifts, adds, subtracts and comparisons, which is all this needs.
    /// It emits a 128-iteration runtime loop; 64-bit division is not on the hot
    /// path for typical lifted code, so the simplicity is worth the cost.
    ///
    /// Blocks use explicit parameters (not `Variable`s) to carry the loop state,
    /// so they need no manual sealing beyond the engine's final `seal_all_blocks`.
    fn emit_udivmod128(&mut self, n: Value, d: Value) -> (Value, Value) {
        let i128 = types::I128;
        let zero = {
            let z = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().uextend(i128, z)
        };
        let one = {
            let o = self.builder.ins().iconst(types::I64, 1);
            self.builder.ins().uextend(i128, o)
        };

        let loop_b = self.builder.create_block();
        self.builder.append_block_param(loop_b, types::I64); // k: bit counter 0..=128
        self.builder.append_block_param(loop_b, i128); // q: quotient so far
        self.builder.append_block_param(loop_b, i128); // r: remainder so far
        let body = self.builder.create_block();
        self.builder.append_block_param(body, types::I64);
        self.builder.append_block_param(body, i128);
        self.builder.append_block_param(body, i128);
        let exit = self.builder.create_block();
        self.builder.append_block_param(exit, i128); // final q
        self.builder.append_block_param(exit, i128); // final r

        let k0 = self.builder.ins().iconst(types::I64, 0);
        self.builder
            .ins()
            .jump(loop_b, &[k0.into(), zero.into(), zero.into()]);

        // loop_b(k, q, r): stop after 128 bits, else fall through to `body`.
        self.builder.switch_to_block(loop_b);
        let k = self.builder.block_params(loop_b)[0];
        let q = self.builder.block_params(loop_b)[1];
        let r = self.builder.block_params(loop_b)[2];
        let done = self.builder.ins().icmp_imm(IntCC::Equal, k, 128);
        self.builder.ins().brif(
            done,
            exit,
            &[q.into(), r.into()],
            body,
            &[k.into(), q.into(), r.into()],
        );

        // body(k, q, r): process bit (127 - k), MSB first.
        self.builder.switch_to_block(body);
        let k = self.builder.block_params(body)[0];
        let q = self.builder.block_params(body)[1];
        let r = self.builder.block_params(body)[2];
        let onetwentyseven = self.builder.ins().iconst(types::I64, 127);
        let bit_pos = self.builder.ins().isub(onetwentyseven, k); // 127 - k
        // r = (r << 1) | ((n >> bit_pos) & 1)
        let r_shifted = self.builder.ins().ishl_imm(r, 1);
        let n_bit_raw = self.builder.ins().ushr(n, bit_pos);
        let n_bit = self.builder.ins().band(n_bit_raw, one);
        let r_new = self.builder.ins().bor(r_shifted, n_bit);
        // if r >= d: r -= d; set quotient bit.
        let ge = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, r_new, d);
        let r_sub = self.builder.ins().isub(r_new, d);
        let r_out = self.builder.ins().select(ge, r_sub, r_new);
        let q_bit = self.builder.ins().ishl(one, bit_pos);
        let q_set = self.builder.ins().bor(q, q_bit);
        let q_out = self.builder.ins().select(ge, q_set, q);
        let k_next = self.builder.ins().iadd_imm(k, 1);
        self.builder
            .ins()
            .jump(loop_b, &[k_next.into(), q_out.into(), r_out.into()]);

        // exit(q, r): division complete.
        self.builder.switch_to_block(exit);
        let quot = self.builder.block_params(exit)[0];
        let rem = self.builder.block_params(exit)[1];
        (quot, rem)
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
