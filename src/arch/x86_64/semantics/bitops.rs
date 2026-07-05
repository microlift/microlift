use crate::MicroliftError;
use crate::lifter::{Lifter, X86Flag};
use crate::arch::x86_64::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

/// Which flavour of single-count shift `emit_shift` should emit.
enum ShiftKind {
    Left,
    RightLogical,
    RightArithmetic,
}

impl<'a> Lifter<'a> {
    pub(super) fn handle_xor(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
            }
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
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let result = self.builder.ins().bxor(dst_val, src_val);
                self.builder.ins().store(MemFlagsData::new(), result, ptr, 0);
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

    pub(super) fn handle_and(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
            }
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
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let result = self.builder.ins().band(dst_val, src_val);
                self.builder.ins().store(MemFlagsData::new(), result, ptr, 0);
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

    pub(super) fn handle_or(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
            }
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
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let dst_val = self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0);
                let result = self.builder.ins().bor(dst_val, src_val);
                self.builder.ins().store(MemFlagsData::new(), result, ptr, 0);
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

    pub(super) fn handle_not(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let dst_val = self.read_operand(instruction, 0, size)?;
        let all_ones = self.builder.ins().iconst(ty, -1);
        let result = self.builder.ins().bxor(dst_val, all_ones);
        self.write_operand(instruction, 0, result)?;
        // NOT does not affect flags
        Ok(())
    }

    pub(super) fn handle_test(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("test-src-{:?}", op1),
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
                    mnemonic: format!("test-dst-{:?}", op0),
                });
            }
        };

        let result = self.builder.ins().band(dst_val, src_val);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    // Shift instructions

    /// Resolve the shift count to a value of the given type.
    fn shift_count(&mut self, instruction: &Instruction, ty: Type) -> crate::Result<Value> {
        let op1 = instruction.op1_kind();
        let count = match op1 {
            OpKind::Register => {
                // get_reg_val(CL) already returns I8; just extend to the target type.
                let cl_val = self.get_reg_val(iced_x86::Register::CL)?;
                if ty == types::I8 {
                    cl_val
                } else {
                    self.builder.ins().uextend(ty, cl_val)
                }
            }
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate8() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("shift-count-{:?}", op1),
                });
            }
        };
        // x86 masks shift count to 5 bits (6 for 64-bit)
        let mask = match ty {
            types::I8 | types::I16 | types::I32 => self.builder.ins().iconst(ty, 0x1F),
            types::I64 => self.builder.ins().iconst(ty, 0x3F),
            _ => self.builder.ins().iconst(ty, 0x3F),
        };
        Ok(self.builder.ins().band(count, mask))
    }

    pub(super) fn handle_shl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_shift(instruction, ShiftKind::Left)
    }

    pub(super) fn handle_shr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_shift(instruction, ShiftKind::RightLogical)
    }

    pub(super) fn handle_sar(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_shift(instruction, ShiftKind::RightArithmetic)
    }

    /// Shared SHL/SHR/SAR lowering.
    ///
    /// x86 masks the shift count to 5 bits (6 for 64-bit operands) and *then*
    /// shifts, so a masked count `>=` the operand width is legal and yields 0
    /// (SHL/SHR) or the sign fill (SAR). Cranelift's shift ops instead re-mask
    /// the amount modulo the operand width, which would corrupt 8/16-bit shifts
    /// whose masked count reaches 8..31. To match hardware we perform the shift
    /// in a working type at least 32 bits wide, where the 5-bit masked count is
    /// always `<` the width, then truncate back. (For 32/64-bit operands the
    /// mask already guarantees `count < width`, so the working type equals the
    /// operand type and this is a no-op widening.)
    fn emit_shift(&mut self, instruction: &Instruction, kind: ShiftKind) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let bit_width = (size * 8) as i64;
        let val = self.read_operand(instruction, 0, size)?;
        let count = self.shift_count(instruction, ty)?;

        // Working type: widen sub-32-bit operands so the masked count fits.
        let work_ty = if size < 4 { types::I32 } else { ty };
        let widen = work_ty != ty;

        // Extend the value into the working type. SAR needs sign extension so
        // both the arithmetic shift and the CF (sign) bit are correct.
        let val_w = if !widen {
            val
        } else if matches!(kind, ShiftKind::RightArithmetic) {
            self.builder.ins().sextend(work_ty, val)
        } else {
            self.builder.ins().uextend(work_ty, val)
        };
        let count_w = if !widen {
            count
        } else {
            self.builder.ins().uextend(work_ty, count)
        };
        let one_w = self.builder.ins().iconst(work_ty, 1);

        let (shifted_w, cf_bit) = match kind {
            ShiftKind::Left => {
                let shifted = self.builder.ins().ishl(val_w, count_w);
                // CF = last bit shifted out of the MSB = bit (width - count) of
                // the original value. Computed in the working type: for a
                // masked count > width, `width - count` wraps and the shift's
                // modulo-width masking makes the read land on a zero bit, which
                // is the correct CF (0) for an over-wide shift.
                let bw = self.builder.ins().iconst(work_ty, bit_width);
                let cf_shift = self.builder.ins().isub(bw, count_w);
                let cf_raw = self.builder.ins().ushr(val_w, cf_shift);
                let cf = self.builder.ins().band(cf_raw, one_w);
                (shifted, cf)
            }
            ShiftKind::RightLogical => {
                let shifted = self.builder.ins().ushr(val_w, count_w);
                // CF = bit (count-1) of the original value.
                let cf_shift = self.builder.ins().isub(count_w, one_w);
                let cf_raw = self.builder.ins().ushr(val_w, cf_shift);
                let cf = self.builder.ins().band(cf_raw, one_w);
                (shifted, cf)
            }
            ShiftKind::RightArithmetic => {
                let shifted = self.builder.ins().sshr(val_w, count_w);
                // CF = bit (count-1) of the sign-extended value (so counts past
                // the width read back the sign bit, matching hardware).
                let cf_shift = self.builder.ins().isub(count_w, one_w);
                let cf_raw = self.builder.ins().ushr(val_w, cf_shift);
                let cf = self.builder.ins().band(cf_raw, one_w);
                (shifted, cf)
            }
        };

        let result = if widen {
            self.builder.ins().ireduce(ty, shifted_w)
        } else {
            shifted_w
        };
        self.write_operand(instruction, 0, result)?;
        self.update_flags_shift(result, cf_bit, count_w, size)?;
        Ok(())
    }

    pub(super) fn handle_rol(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let val = self.read_operand(instruction, 0, size)?;
        let count = self.shift_count(instruction, ty)?;
        let bits = self.builder.ins().iconst(ty, (size * 8) as i64);
        let right_count = self.builder.ins().isub(bits, count);
        let left = self.builder.ins().ishl(val, count);
        let right = self.builder.ins().ushr(val, right_count);
        let result = self.builder.ins().bor(left, right);
        self.write_operand(instruction, 0, result)?;
        // CF = low bit of result (the bit that rotated out of MSB).
        let one = self.builder.ins().iconst(ty, 1);
        let cf_bit = self.builder.ins().band(result, one);
        self.set_rotate_cf(cf_bit, count)?;
        Ok(())
    }

    pub(super) fn handle_ror(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let val = self.read_operand(instruction, 0, size)?;
        let count = self.shift_count(instruction, ty)?;
        let bits = self.builder.ins().iconst(ty, (size * 8) as i64);
        let left_count = self.builder.ins().isub(bits, count);
        let right = self.builder.ins().ushr(val, count);
        let left = self.builder.ins().ishl(val, left_count);
        let result = self.builder.ins().bor(right, left);
        self.write_operand(instruction, 0, result)?;
        // CF = MSB of result (the bit that rotated out of bit 0).
        let cf_bit = self.builder.ins().ushr_imm(result, (size * 8 - 1) as i64);
        self.set_rotate_cf(cf_bit, count)?;
        Ok(())
    }

    /// Set CF from a rotate's carried-out bit, but leave CF untouched when the
    /// masked rotate `count` is 0 (per x86: a zero masked count affects no
    /// flags). `cf_bit` is in the operand type; `count` is the masked count.
    fn set_rotate_cf(&mut self, cf_bit: Value, count: Value) -> crate::Result<()> {
        let old_cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let new_cf = if self.builder.func.dfg.value_type(cf_bit) == types::I8 {
            cf_bit
        } else {
            self.builder.ins().ireduce(types::I8, cf_bit)
        };
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, count, 0);
        let final_cf = self.builder.ins().select(is_zero, old_cf, new_cf);
        self.set_flag(X86Flag::Carry, final_cf)
    }

    /// Build a constant of width `ty` from a (non-negative, <= 64-bit) value.
    /// `iconst` only supports integer types up to I64, so I128 constants are
    /// materialised by zero-extending an I64 constant.
    fn wide_const(&mut self, ty: Type, value: i64) -> Value {
        if ty == types::I128 {
            let lo = self.builder.ins().iconst(types::I64, value);
            self.builder.ins().uextend(types::I128, lo)
        } else {
            self.builder.ins().iconst(ty, value)
        }
    }

    /// RCR - Rotate through Carry Right.
    ///
    /// Treats `CF:dst` as a single `(N+1)`-bit value (CF is the most-significant
    /// bit) and rotates it right by the masked count. To avoid Cranelift's
    /// shift-amount-modulo-width semantics ever clobbering a rotate that crosses
    /// the operand boundary, the whole computation is performed in a working
    /// type at least two bits wider than the operand:
    ///
    /// ```text
    ///   x          = (cf << N) | dst                (the N+1-bit value)
    ///   rotated     = (x >> c) | (cf << (N-c)) | (x << (N+1-c))
    ///   dst'        = rotated[N-1:0]
    ///   CF'         = rotated[N]
    /// ```
    ///
    /// The count is masked to 5 bits (6 for 64-bit operands) and then reduced
    /// modulo `N+1`, exactly as the hardware does. OF is only architecturally
    /// defined for single-bit rotates; like the other rotates here it is left
    /// unmodelled.
    pub(super) fn handle_rcr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let n = (size * 8) as i64;

        // Working type, at least N+2 bits wide so no shift reaches its width.
        let work_ty = match size {
            1 => types::I16,
            2 => types::I32,
            4 => types::I64,
            8 => types::I128,
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("rcr-size-{}", size),
                });
            }
        };

        let val = self.read_operand(instruction, 0, size)?;
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        // Masked count (5/6 bits) in the operand type.
        let count = self.shift_count(instruction, ty)?;

        // Promote everything into the wider working type.
        let xval = self.builder.ins().uextend(work_ty, val);
        let cfw = self.builder.ins().uextend(work_ty, cf);
        let count_w = self.builder.ins().uextend(work_ty, count);

        // c = count mod (N+1)  (in [0, N]). For 32/64-bit operands the 5/6-bit
        // mask already guarantees count < N+1, so the modulo is a no-op there -
        // which is fortunate, since the backend has no `urem.i128`.
        let np1 = self.wide_const(work_ty, n + 1);
        let c = if size <= 2 {
            self.builder.ins().urem(count_w, np1)
        } else {
            count_w
        };

        // rotated = (x >> c) | (cf << (N - c)) | (x << (N + 1 - c))
        let n_w = self.wide_const(work_ty, n);
        let n_minus_c = self.builder.ins().isub(n_w, c);
        let np1_minus_c = self.builder.ins().isub(np1, c);

        let t_right = self.builder.ins().ushr(xval, c);
        let t_cf = self.builder.ins().ishl(cfw, n_minus_c);
        let t_left = self.builder.ins().ishl(xval, np1_minus_c);
        let rotated = {
            let lo = self.builder.ins().bor(t_right, t_cf);
            self.builder.ins().bor(lo, t_left)
        };

        // dst' = low N bits of the rotated value.
        let result = self.builder.ins().ireduce(ty, rotated);
        self.write_operand(instruction, 0, result)?;

        // CF' = bit N of the rotated value. This is also correct for c == 0
        // (a masked count of 0), where bit N still holds the original CF.
        let cf_shifted = self.builder.ins().ushr(rotated, n_w);
        let one_w = self.wide_const(work_ty, 1);
        let cf_bit = self.builder.ins().band(cf_shifted, one_w);
        self.set_flag(X86Flag::Carry, cf_bit)?;
        Ok(())
    }

    /// Resolve the shift count for a double-shift instruction (SHLD/SHRD).
    ///
    /// The count is op2: either an `imm8` or the `CL` register, masked to
    /// 5 bits for 8/16/32-bit operands or 6 bits for 64-bit operands.
    fn double_shift_count(&mut self, instruction: &Instruction, ty: Type) -> crate::Result<Value> {
        let count_raw = match instruction.op2_kind() {
            OpKind::Register => {
                // get_reg_val(CL) already returns I8; just extend to the target type.
                let cl = self.get_reg_val(Register::CL)?;
                self.builder.ins().uextend(ty, cl)
            }
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate8() as i64),
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("double-shift-count-{:?}", other),
                });
            }
        };
        let mask = match ty {
            types::I64 => self.builder.ins().iconst(ty, 0x3F),
            _ => self.builder.ins().iconst(ty, 0x1F),
        };
        Ok(self.builder.ins().band(count_raw, mask))
    }

    /// SHLD - Shift Left Double.
    ///
    /// `SHLD dst, src, count`
    /// result = (dst << count) | (src >> (N - count))
    /// CF = last bit shifted out of dst's MSB side.
    pub(super) fn handle_shld(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let bits = self.builder.ins().iconst(ty, (size * 8) as i64);

        let dst = self.read_operand(instruction, 0, size)?;
        let src = self.get_reg_val(instruction.op1_register())?;
        // get_reg_val returns the natural width of the register, which matches
        // the destination size for well-formed SHLD.
        let count = self.double_shift_count(instruction, ty)?;

        // result = (dst << count) | (src >> (N - count))
        let right_count = self.builder.ins().isub(bits, count);
        let shifted_dst = self.builder.ins().ishl(dst, count);
        let shifted_src = self.builder.ins().ushr(src, right_count);
        let result = self.builder.ins().bor(shifted_dst, shifted_src);

        self.write_operand(instruction, 0, result)?;

        // CF = (dst >> (N - count)) & 1  (last bit shifted out of dst's MSB)
        let cf_raw = self.builder.ins().ushr(dst, right_count);
        let one = self.builder.ins().iconst(ty, 1);
        let cf_bit = self.builder.ins().band(cf_raw, one);
        self.update_flags_shift(result, cf_bit, count, size)?;
        Ok(())
    }

    /// SHRD - Shift Right Double.
    ///
    /// `SHRD dst, src, count`
    /// result = (src << (N - count)) | (dst >> count)
    /// CF = last bit shifted out of dst's LSB side.
    pub(super) fn handle_shrd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let bits = self.builder.ins().iconst(ty, (size * 8) as i64);

        let dst = self.read_operand(instruction, 0, size)?;
        let src = self.get_reg_val(instruction.op1_register())?;
        let count = self.double_shift_count(instruction, ty)?;

        // result = (src << (N - count)) | (dst >> count)
        let left_count = self.builder.ins().isub(bits, count);
        let shifted_src = self.builder.ins().ishl(src, left_count);
        let shifted_dst = self.builder.ins().ushr(dst, count);
        let result = self.builder.ins().bor(shifted_src, shifted_dst);

        self.write_operand(instruction, 0, result)?;

        // CF = (dst >> (count - 1)) & 1  (last bit shifted out of dst's LSB)
        let one = self.builder.ins().iconst(ty, 1);
        let cf_shift = self.builder.ins().isub(count, one);
        let cf_raw = self.builder.ins().ushr(dst, cf_shift);
        let cf_bit = self.builder.ins().band(cf_raw, one);
        self.update_flags_shift(result, cf_bit, count, size)?;
        Ok(())
    }

    // Bit test instructions

    fn bt_common(&mut self, instruction: &Instruction) -> crate::Result<(Value, Value, usize)> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let base = self.read_operand(instruction, 0, size)?;
        let bit_idx = match instruction.op1_kind() {
            OpKind::Register => {
                // get_reg_val returns the natural type of the register, which for BT
                // always matches the operand size - no reduction needed.
                self.get_reg_val(instruction.op1_register())?
            }
            OpKind::Immediate8 => self
                .builder
                .ins()
                .iconst(ty, instruction.immediate8() as i64),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "bt-bit-idx".into(),
                });
            }
        };
        // Mask bit index to operand size
        let mask = self.builder.ins().iconst(ty, (size * 8 - 1) as i64);
        let masked = self.builder.ins().band(bit_idx, mask);
        // CF = (base >> bit_idx) & 1
        let shifted = self.builder.ins().ushr(base, masked);
        let one = self.builder.ins().iconst(ty, 1);
        let cf_bit = self.builder.ins().band(shifted, one);
        self.set_flag(X86Flag::Carry, cf_bit)?;
        Ok((base, masked, size))
    }

    pub(super) fn handle_bt(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.bt_common(instruction)?;
        Ok(())
    }

    pub(super) fn handle_bts(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (base, bit_idx, size) = self.bt_common(instruction)?;
        let ty = size_to_type(size);
        let one = self.builder.ins().iconst(ty, 1);
        let bit_mask = self.builder.ins().ishl(one, bit_idx);
        let result = self.builder.ins().bor(base, bit_mask);
        self.write_operand(instruction, 0, result)?;
        Ok(())
    }

    pub(super) fn handle_btr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (base, bit_idx, size) = self.bt_common(instruction)?;
        let ty = size_to_type(size);
        let one = self.builder.ins().iconst(ty, 1);
        let bit_mask = self.builder.ins().ishl(one, bit_idx);
        let inv_mask = self.builder.ins().iconst(ty, -1i64);
        let cleared_mask = self.builder.ins().bxor(bit_mask, inv_mask);
        let result = self.builder.ins().band(base, cleared_mask);
        self.write_operand(instruction, 0, result)?;
        Ok(())
    }

    pub(super) fn handle_btc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (base, bit_idx, size) = self.bt_common(instruction)?;
        let ty = size_to_type(size);
        let one = self.builder.ins().iconst(ty, 1);
        let bit_mask = self.builder.ins().ishl(one, bit_idx);
        let result = self.builder.ins().bxor(base, bit_mask);
        self.write_operand(instruction, 0, result)?;
        Ok(())
    }

    pub(super) fn handle_bsf(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 1)?;
        let src = self.read_operand(instruction, 1, size)?;
        // ZF = 1 if src == 0
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, src, 0);
        self.set_flag(X86Flag::Zero, zf)?;
        // Result: count trailing zeros (undefined if src==0, but we emit it anyway)
        let result = self.builder.ins().ctz(src);
        self.write_operand(instruction, 0, result)?;
        Ok(())
    }

    /// BSWAP - reverse byte order of a 32- or 64-bit register.
    ///
    /// For 16-bit operands the result is architecturally undefined; we follow
    /// the common convention of zeroing the destination.
    pub(super) fn handle_bswap(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let reg = instruction.op0_register();
        let size = reg.size();
        match size {
            4 | 8 => {
                let val = self.get_reg_val(reg)?;
                let result = self.builder.ins().bswap(val);
                self.set_reg_val(reg, result)?;
            }
            2 => {
                // Undefined per Intel SDM; zero the register.
                let zero = self.builder.ins().iconst(size_to_type(size), 0);
                self.set_reg_val(reg, zero)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("bswap-size-{}", size),
                });
            }
        }
        Ok(())
    }

    pub(super) fn handle_bsr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 1)?;
        let src = self.read_operand(instruction, 1, size)?;
        // ZF = 1 if src == 0
        let zf = self.builder.ins().icmp_imm(IntCC::Equal, src, 0);
        self.set_flag(X86Flag::Zero, zf)?;
        // BSR: index of highest set bit = (size*8-1) - clz(src)
        let clz = self.builder.ins().clz(src);
        let bit_width = self
            .builder
            .ins()
            .iconst(size_to_type(size), (size * 8 - 1) as i64);
        let result = self.builder.ins().isub(bit_width, clz);
        self.write_operand(instruction, 0, result)?;
        Ok(())
    }
}
