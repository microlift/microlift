use crate::MicroliftError;
use crate::arch::x86_64::regs::{FLAG_ORDER, RegisterState, gpr_index, is_segment_register};
use crate::arch::x86_64::sync::{FS_BASE_SLOT, GS_BASE_SLOT};
use crate::core::regsync::RegSlot;
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};


/// A resolved operand: a guest register, or a memory operand's already-computed effective address (so RMW forms it once).
#[derive(Debug, Clone, Copy)]
pub enum Place {
    Reg(Register),
    Mem(Value),
}

pub fn size_to_type(size: usize) -> Type {
    match size {
        1 => types::I8,
        2 => types::I16,
        4 => types::I32,
        8 => types::I64,
        _ => panic!("Unsupported operand size: {}", size),
    }
}

/// The XMM register index (0-15), or `None` if not an XMM register.
pub fn xmm_register_index(reg: Register) -> Option<u8> {
    match reg {
        Register::XMM0 => Some(0),
        Register::XMM1 => Some(1),
        Register::XMM2 => Some(2),
        Register::XMM3 => Some(3),
        Register::XMM4 => Some(4),
        Register::XMM5 => Some(5),
        Register::XMM6 => Some(6),
        Register::XMM7 => Some(7),
        Register::XMM8 => Some(8),
        Register::XMM9 => Some(9),
        Register::XMM10 => Some(10),
        Register::XMM11 => Some(11),
        Register::XMM12 => Some(12),
        Register::XMM13 => Some(13),
        Register::XMM14 => Some(14),
        Register::XMM15 => Some(15),
        _ => None,
    }
}

pub fn is_high_8_reg(reg: Register) -> bool {
    matches!(
        reg,
        Register::AH | Register::BH | Register::CH | Register::DH
    )
}

pub fn get_base_reg_64(reg: Register) -> Register {
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
        _ => reg, // Default to self
    }
}


impl<'a> Lifter<'a> {
    /// Narrow `val` to `ty`, or return it unchanged; `ireduce` requires a strictly wider source.
    pub fn narrow_to(&mut self, ty: Type, val: Value) -> Value {
        if self.builder.func.dfg.value_type(val) == ty {
            val
        } else {
            self.builder.ins().ireduce(ty, val)
        }
    }

    pub fn get_reg_val(&mut self, reg: Register) -> crate::Result<Value> {
        // RIP is the next instruction's address, per the configured `AddressMode`.
        if reg == Register::RIP {
            return Ok(self.runtime_current_ip());
        }

        if is_segment_register(reg) {
            return self.get_segment_selector(reg);
        }

        let base_reg = get_base_reg_64(reg);
        let size = reg.size();
        let is_high_8 = is_high_8_reg(reg);

        if let Some(&var) = self.state.gprs.get(&base_reg) {
            let full_val = self.builder.use_var(var);
            if size == 8 {
                Ok(full_val)
            } else if is_high_8 {
                let shifted = self.builder.ins().ushr_imm_u(full_val, 8);
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

    pub fn set_reg_val(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        // Writing RIP directly via a data instruction is illegal in x86-64.
        if reg == Register::RIP {
            return Err(MicroliftError::UnsupportedInstruction {
                mnemonic: "set-RIP (direct write to RIP is illegal in x86-64)".into(),
            });
        }

        if is_segment_register(reg) {
            return self.set_segment_selector(reg, val);
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
                let shifted_val = self.builder.ins().ishl_imm_u(val_64, 8);
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
            // The write set the lazy-sync prune reads; see `core::regsync`.
            let idx = gpr_index(base_reg).expect("a GPR variable has a GPR_ORDER index");
            self.mark_written(RegSlot::Gpr(idx));
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("set-reg-{:?}", reg),
            })
        }
    }

    /// Read a segment selector as an `I16` straight from the `RegisterContext` (also in analysis mode).
    fn get_segment_selector(&mut self, reg: Register) -> crate::Result<Value> {
        let offset = RegisterState::get_segment_offset(reg)?;
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        Ok(self.ctx_load(types::I16, ctx_ptr, offset))
    }

    /// Write a segment selector into the `RegisterContext`; the `FS`/`GS` bases are separate fields and are left alone.
    fn set_segment_selector(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        if reg == Register::CS {
            // `MOV CS, r/m16` does not encode; reaching here is a decode bug.
            return Err(MicroliftError::UnsupportedInstruction {
                mnemonic: "set-CS (CS is not writable by MOV in x86-64)".into(),
            });
        }
        let offset = RegisterState::get_segment_offset(reg)?;
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let val_i16 = self.narrow_to(types::I16, val);
        self.ctx_store(val_i16, ctx_ptr, offset);
        Ok(())
    }

    /// Flag math is always emitted; the finished SSA pass removes unused computations.
    pub fn flag_needed(&self, flag: X86Flag) -> bool {
        self.flag_live(flag as u8)
    }

    /// Whether `flag` is stored as a *source* to derive it from (see [`Self::get_flag`]): only the two flags whose computation outweighs their storage.
    pub(super) fn deferred(flag: X86Flag) -> bool {
        matches!(flag, X86Flag::Parity | X86Flag::Auxiliary)
    }

    /// Read `flag` as the logical 0/1 `I8` it is defined to be.
    pub fn get_flag(&mut self, flag: X86Flag) -> Value {
        let raw = self.builder.use_var(self.state.flags[&flag]);
        match flag {
            // PF is the even parity of the low byte: xor-fold then invert (`popcnt` is slow on aarch64's vector unit).
            X86Flag::Parity => {
                let mut folded = raw;
                for shift in [4, 2, 1] {
                    let shifted = self.builder.ins().ushr_imm_u(folded, shift);
                    folded = self.builder.ins().bxor(folded, shifted);
                }
                let odd = self.builder.ins().band_imm_u(folded, 1);
                self.builder.ins().bxor_imm_u(odd, 1)
            }
            X86Flag::Auxiliary => {
                let shifted = self.builder.ins().ushr_imm_u(raw, 4);
                self.builder.ins().band_imm_u(shifted, 1)
            }
            _ => raw,
        }
    }

    /// Assign a [deferred](Self::deferred) flag's *source* directly: the result's low byte for `Parity`, `lhs ^ rhs ^ res` for `Auxiliary`.
    ///
    /// # Panics
    ///
    /// If `flag` is not deferred: a boolean stored under a source encoding is a silent wrong answer.
    pub fn set_flag_source(&mut self, flag: X86Flag, src: Value) -> crate::Result<()> {
        assert!(
            Self::deferred(flag),
            "{flag:?} is stored as a boolean; use set_flag"
        );
        self.store_flag(flag, src)
    }

    /// Assign `flag` from its logical 0/1 value, encoding the two source-stored flags: `val ^ 1` for PF, `val << 4` for AF.
    pub fn set_flag(&mut self, flag: X86Flag, val: Value) -> crate::Result<()> {
        let encoded = match flag {
            X86Flag::Parity | X86Flag::Auxiliary if self.state.flags.contains_key(&flag) => {
                let val_i8 = if self.builder.func.dfg.value_type(val) == types::I8 {
                    val
                } else {
                    self.builder.ins().ireduce(types::I8, val)
                };
                // Mask first: callers pass any-nonzero-is-true values.
                let bit = self.builder.ins().band_imm_u(val_i8, 1);
                match flag {
                    X86Flag::Parity => self.builder.ins().bxor_imm_u(bit, 1),
                    _ => self.builder.ins().ishl_imm_u(bit, 4),
                }
            }
            _ => val,
        };
        self.store_flag(flag, encoded)
    }

    /// Write `val` into `flag`'s variable verbatim; the tail of [`Self::set_flag`] and [`Self::set_flag_source`].
    fn store_flag(&mut self, flag: X86Flag, val: Value) -> crate::Result<()> {
        if let Some(&var) = self.state.flags.get(&flag) {
            let val_i8 = if self.builder.func.dfg.value_type(val) == types::I8 {
                val
            } else {
                self.builder.ins().ireduce(types::I8, val)
            };
            self.record_flag_value(val_i8);
            self.builder.def_var(var, val_i8);
            let idx = FLAG_ORDER
                .iter()
                .position(|&f| f == flag)
                .expect("a flag variable has a FLAG_ORDER index");
            self.mark_written(RegSlot::Flag(idx as u8));
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("flag-{:?}", flag),
            })
        }
    }


    pub fn operand_kind(&self, instruction: &Instruction, op_index: u32) -> OpKind {
        match op_index {
            0 => instruction.op0_kind(),
            1 => instruction.op1_kind(),
            2 => instruction.op2_kind(),
            _ => unreachable!("unsupported operand index {op_index}"),
        }
    }

    pub fn operand_register(&self, instruction: &Instruction, op_index: u32) -> Register {
        match op_index {
            0 => instruction.op0_register(),
            1 => instruction.op1_register(),
            2 => instruction.op2_register(),
            _ => unreachable!("unsupported operand index {op_index}"),
        }
    }

    pub fn operand_size(&self, instruction: &Instruction, op_index: u32) -> crate::Result<usize> {
        let op_kind = self.operand_kind(instruction, op_index);
        match op_kind {
            OpKind::Register => Ok(self.operand_register(instruction, op_index).size()),
            OpKind::Memory => Ok(instruction.memory_size().size()),
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => Ok(1),
            OpKind::Immediate16 => Ok(2),
            OpKind::Immediate32 | OpKind::Immediate32to64 => Ok(4),
            OpKind::Immediate64 => Ok(8),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("operand-size-{}-{:?}", op_index, op_kind),
            }),
        }
    }

    pub fn read_operand(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        size: usize,
    ) -> crate::Result<Value> {
        let op_kind = self.operand_kind(instruction, op_index);
        let ty = size_to_type(size);
        match op_kind {
            OpKind::Register => self.get_reg_val(self.operand_register(instruction, op_index)),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                Ok(self.builder.ins().load(ty, self.guest_flags, ptr, 0))
            }
            OpKind::Immediate32to64 => {
                Ok(self.builder.ins().iconst(ty, instruction.immediate32to64()))
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64 => {
                Ok(self.builder.ins().iconst(ty, Self::read_imm64(instruction)))
            }
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("read-op-{}-{:?}", op_index, op_kind),
            }),
        }
    }

    /// Read both operands of a binary `r/m, r/m/imm` instruction at the destination's width, returning `(dst, src)`.
    /// Reads **src first**, the order its side effects must appear in; `size` from operand 0 also widens a sign-extended immediate.
    /// For read-modify-*write* use [`read_binary_rmw`](Self::read_binary_rmw).
    pub fn read_binary_operands(
        &mut self,
        instruction: &Instruction,
        size: usize,
    ) -> crate::Result<(Value, Value)> {
        let src = self.read_operand(instruction, 1, size)?;
        let dst = self.read_operand(instruction, 0, size)?;
        Ok((dst, src))
    }

    /// [`read_binary_operands`](Self::read_binary_operands) for a write-back instruction, returning `(place, dst, src)`.
    pub fn read_binary_rmw(
        &mut self,
        instruction: &Instruction,
        size: usize,
    ) -> crate::Result<(Place, Value, Value)> {
        let src = self.read_operand(instruction, 1, size)?;
        let place = self.resolve_place(instruction, 0)?;
        let dst = self.read_place(place, size)?;
        Ok((place, dst, src))
    }

    /// Read-modify-write of a unary instruction (`INC`, `NEG`, `NOT`, shifts, rotates): `(size, place, value)` for operand 0.
    pub fn read_unary_rmw(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<(usize, Place, Value)> {
        let size = self.operand_size(instruction, 0)?;
        let place = self.resolve_place(instruction, 0)?;
        let val = self.read_place(place, size)?;
        Ok((size, place, val))
    }

    /// Resolve an operand to its location, computing a memory effective address once.
    pub fn resolve_place(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<Place> {
        let op_kind = self.operand_kind(instruction, op_index);
        match op_kind {
            OpKind::Register => Ok(Place::Reg(self.operand_register(instruction, op_index))),
            OpKind::Memory => Ok(Place::Mem(self.get_mem_ptr(instruction, op_index)?)),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("place-{}-{:?}", op_index, op_kind),
            }),
        }
    }

    pub fn read_place(&mut self, place: Place, size: usize) -> crate::Result<Value> {
        match place {
            Place::Reg(reg) => self.get_reg_val(reg),
            Place::Mem(ptr) => {
                let ty = size_to_type(size);
                Ok(self.builder.ins().load(ty, self.guest_flags, ptr, 0))
            }
        }
    }

    pub fn write_place(&mut self, place: Place, val: Value) -> crate::Result<()> {
        match place {
            Place::Reg(reg) => self.set_reg_val(reg, val),
            Place::Mem(ptr) => {
                self.builder.ins().store(self.guest_flags, val, ptr, 0);
                Ok(())
            }
        }
    }

    /// Read `CF` as the `ADC`/`SBB` carry input, widened to `size` bytes.
    pub fn carry_in(&mut self, size: usize) -> Value {
        let ty = size_to_type(size);
        let cf = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        if ty == types::I8 {
            cf
        } else {
            self.builder.ins().uextend(ty, cf)
        }
    }

    pub fn write_operand(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        val: Value,
    ) -> crate::Result<()> {
        let op_kind = self.operand_kind(instruction, op_index);
        match op_kind {
            OpKind::Register => self.set_reg_val(self.operand_register(instruction, op_index), val),
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                self.builder.ins().store(self.guest_flags, val, ptr, 0);
                Ok(())
            }
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("write-op-{}-{:?}", op_index, op_kind),
            }),
        }
    }

    pub fn get_mem_ptr(
        &mut self,
        instruction: &Instruction,
        _op_index: u32,
    ) -> crate::Result<Value> {
        let base_reg = instruction.memory_base();
        let index_reg = instruction.memory_index();
        let scale = instruction.memory_index_scale();
        let disp = instruction.memory_displacement64();
        let segment = instruction.memory_segment();

        // A `0x67` prefix gives 32-bit base/index: compute in I64 and truncate the sum to 32 bits.
        let addr32 =
            (base_reg != Register::None && base_reg != Register::RIP && base_reg.size() == 4)
                || (index_reg != Register::None && index_reg.size() == 4);
        let widen64 = |lifter: &mut Self, v: Value| {
            if lifter.builder.func.dfg.value_type(v) == types::I64 {
                v
            } else {
                lifter.builder.ins().uextend(types::I64, v)
            }
        };

        let mut ptr = if base_reg == Register::RIP {
            // RIP-relative: iced's `disp` is the absolute lift-time VA; materialize per `AddressMode`.
            self.runtime_addr(disp)
        } else if base_reg != Register::None {
            let v = self.get_reg_val(base_reg)?;
            widen64(self, v)
        } else {
            self.builder.ins().iconst(types::I64, 0)
        };

        if index_reg != Register::None {
            let index_val = self.get_reg_val(index_reg)?;
            let index_val = widen64(self, index_val);
            let scaled_index = if scale > 1 {
                self.builder.ins().imul_imm_s(index_val, scale as i64)
            } else {
                index_val
            };
            ptr = self.builder.ins().iadd(ptr, scaled_index);
        }

        // Add the displacement only if not RIP-relative (already in `ptr`).
        if base_reg != Register::RIP && disp != 0 {
            ptr = self.builder.ins().iadd_imm_s(ptr, disp as i64);
        }

        // 32-bit addressing truncates the effective address before any segment base.
        if addr32 {
            ptr = self.builder.ins().band_imm_u(ptr, 0xFFFF_FFFF);
        }

        // Only FS and GS carry non-zero bases in long mode.
        match segment {
            Register::FS => {
                let fs_base = self.get_fs_base();
                ptr = self.builder.ins().iadd(fs_base, ptr);
            }
            Register::GS => {
                let gs_base = self.get_gs_base();
                ptr = self.builder.ins().iadd(gs_base, ptr);
            }
            _ => {}
        }

        Ok(ptr)
    }

    /// Read an immediate as an `i64`, sign-extending `imm8`/`imm32` as x86 does.
    /// [`Instruction::immediate64`] returns raw bits, so `and rax, -16` would otherwise read as `+240`.
    pub fn read_imm64(instruction: &Instruction) -> i64 {
        for i in 0..instruction.op_count() {
            match instruction.op_kind(i) {
                OpKind::Immediate8 => return instruction.immediate8() as i64,
                OpKind::Immediate8to16 => return instruction.immediate8to16() as i64,
                OpKind::Immediate8to32 => return instruction.immediate8to32() as i64,
                OpKind::Immediate8to64 => return instruction.immediate8to64(),
                OpKind::Immediate16 => return instruction.immediate16() as i64,
                OpKind::Immediate32 => return instruction.immediate32() as i64,
                OpKind::Immediate32to64 => return instruction.immediate32to64(),
                OpKind::Immediate64 => return instruction.immediate64() as i64,
                _ => {}
            }
        }
        instruction.immediate64() as i64
    }

    // XMM accessors: lo/hi work in both representations; use `get_xmm_vec`/`set_xmm_vec` to move a whole 128-bit value.

    /// Return the low 64 bits of XMM register `idx` (0-15).
    pub fn get_xmm_lo(&mut self, idx: u8) -> Value {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            let vec = self.builder.use_var(var);
            self.builder.ins().extractlane(vec, 0)
        } else {
            let var = *self.state.xmms_lo.get(&idx).unwrap();
            self.builder.use_var(var)
        }
    }

    /// Return the high 64 bits of XMM register `idx` (0-15).
    pub fn get_xmm_hi(&mut self, idx: u8) -> Value {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            let vec = self.builder.use_var(var);
            self.builder.ins().extractlane(vec, 1)
        } else {
            let var = *self.state.xmms_hi.get(&idx).unwrap();
            self.builder.use_var(var)
        }
    }

    /// Write the low 64 bits of XMM register `idx` (0-15).
    pub fn set_xmm_lo(&mut self, idx: u8, val: Value) {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            let vec = self.builder.use_var(var);
            let new_vec = self.builder.ins().insertlane(vec, val, 0);
            self.builder.def_var(var, new_vec);
        } else {
            let var = *self.state.xmms_lo.get(&idx).unwrap();
            self.builder.def_var(var, val);
        }
        // In vector mode both halves are one variable.
        self.mark_written(RegSlot::Vec(idx));
        if self.state.use_vector_xmm {
            self.mark_written(RegSlot::VecHi(idx));
        }
    }

    /// Write the high 64 bits of XMM register `idx` (0-15).
    pub fn set_xmm_hi(&mut self, idx: u8, val: Value) {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            let vec = self.builder.use_var(var);
            let new_vec = self.builder.ins().insertlane(vec, val, 1);
            self.builder.def_var(var, new_vec);
        } else {
            let var = *self.state.xmms_hi.get(&idx).unwrap();
            self.builder.def_var(var, val);
        }
        self.mark_written(RegSlot::VecHi(idx));
        if self.state.use_vector_xmm {
            self.mark_written(RegSlot::Vec(idx));
        }
    }

    /// Return the full 128-bit value of XMM `idx` as `I64X2`.
    pub fn get_xmm_vec(&mut self, idx: u8) -> Value {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            self.builder.use_var(var)
        } else {
            let lo_var = *self.state.xmms_lo.get(&idx).unwrap();
            let hi_var = *self.state.xmms_hi.get(&idx).unwrap();
            let lo = self.builder.use_var(lo_var);
            let hi = self.builder.use_var(hi_var);
            let v = self.builder.ins().scalar_to_vector(types::I64X2, lo);
            self.builder.ins().insertlane(v, hi, 1)
        }
    }

    /// Write a full 128-bit `I64X2` value to XMM `idx` and mark dirty.
    pub fn set_xmm_vec(&mut self, idx: u8, val: Value) {
        if self.state.use_vector_xmm {
            let var = *self.state.xmms_vec.get(&idx).unwrap();
            self.builder.def_var(var, val);
        } else {
            let lo = self.builder.ins().extractlane(val, 0);
            let hi = self.builder.ins().extractlane(val, 1);
            let lo_var = *self.state.xmms_lo.get(&idx).unwrap();
            let hi_var = *self.state.xmms_hi.get(&idx).unwrap();
            self.builder.def_var(lo_var, lo);
            self.builder.def_var(hi_var, hi);
        }
        self.mark_written(RegSlot::Vec(idx));
        self.mark_written(RegSlot::VecHi(idx));
    }

    /// Read the current FS segment base from the SSA variable.
    pub fn get_fs_base(&mut self) -> Value {
        self.builder.use_var(self.state.fs_base)
    }

    /// Read the current GS segment base from the SSA variable.
    pub fn get_gs_base(&mut self) -> Value {
        self.builder.use_var(self.state.gs_base)
    }

    /// Write a new FS segment base.
    pub fn set_fs_base(&mut self, val: Value) {
        self.builder.def_var(self.state.fs_base, val);
        self.mark_written(FS_BASE_SLOT);
    }

    /// Write a new GS segment base.
    pub fn set_gs_base(&mut self, val: Value) {
        self.builder.def_var(self.state.gs_base, val);
        self.mark_written(GS_BASE_SLOT);
    }
}
