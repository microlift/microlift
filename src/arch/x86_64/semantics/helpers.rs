use crate::MicroliftError;
use crate::arch::x86_64::regs::{FLAG_ORDER, RegisterState, gpr_index, is_segment_register};
use crate::arch::x86_64::sync::{FS_BASE_SLOT, GS_BASE_SLOT};
use crate::core::regsync::RegSlot;
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

// Type helpers

/// A resolved operand location: a guest register, or the already-computed
/// effective address of a memory operand.
///
/// Exists so a read-modify-write instruction forms its address once. Going
/// through `read_operand`/`write_operand` calls `get_mem_ptr` twice, emitting
/// the base/index/displacement arithmetic twice.
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

/// Returns the XMM register index (0-15) for an XMM register, or `None` if
/// the register is not an XMM register.
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

// Register and flag accessors on Lifter

impl<'a> Lifter<'a> {
    /// Narrow `val` to `ty`, or return it unchanged if it is already that wide.
    ///
    /// `ireduce` requires a *strictly* wider source, so an unconditional narrowing
    /// is malformed IR when the value already has the target type - and the
    /// verifier rejects the whole function. Every caller here reads a value whose
    /// width depends on an operand size, so "already narrow" is the ordinary case.
    pub fn narrow_to(&mut self, ty: Type, val: Value) -> Value {
        if self.builder.func.dfg.value_type(val) == ty {
            val
        } else {
            self.builder.ins().ireduce(ty, val)
        }
    }

    pub fn get_reg_val(&mut self, reg: Register) -> crate::Result<Value> {
        // RIP is handled specially: it names the address of the next
        // instruction, materialized per the configured `AddressMode`.
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

    /// Read a segment selector (`CS`/`SS`/`DS`/`ES`/`FS`/`GS`) as an `I16`.
    ///
    /// The selectors live in the `RegisterContext` and are read in place rather
    /// than through an SSA variable. The load is emitted in analysis mode too:
    /// there is no SSA value for it to be pure over, and the point of the
    /// instruction is that it reads state the host supplied.
    fn get_segment_selector(&mut self, reg: Register) -> crate::Result<Value> {
        let offset = RegisterState::get_segment_offset(reg)?;
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        Ok(self.ctx_load(types::I16, ctx_ptr, offset))
    }

    /// Write a segment selector, storing it straight into the `RegisterContext`.
    ///
    /// Selector storage only: the `FS`/`GS` *bases* the lifter forms addresses with
    /// are separate fields and are deliberately left alone, where hardware would
    /// reload each from the loaded descriptor.
    fn set_segment_selector(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        if reg == Register::CS {
            // `MOV CS, r/m16` does not encode - the only architectural writes
            // to CS are far transfers, which are not lifted. Reaching here
            // means a decode produced something impossible, so say so rather
            // than silently storing.
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

    /// Whether the instruction being lifted has to compute `flag`.
    ///
    /// The x86-64 spelling of [`LiftEngine::flag_live`], and the only place the
    /// `X86Flag` discriminant is used as a flag *index* - the numbering
    /// [`crate::arch::x86_64::flag_effects`] classifies in. Always true unless
    /// [`LifterConfig::eliminate_dead_flags`] is on.
    ///
    /// [`LiftEngine::flag_live`]: crate::core::LiftEngine::flag_live
    /// [`LifterConfig::eliminate_dead_flags`]: crate::core::LifterConfig::eliminate_dead_flags
    pub fn flag_needed(&self, flag: X86Flag) -> bool {
        self.flag_live(flag as u8)
    }

    /// Assign `flag`, unless nothing goes on to read it.
    ///
    /// The backstop for the guards in `semantics/flags.rs`: those skip the
    /// *computation*, where the cost is, and this skips the assignment whether or
    /// not a handler remembered to guard. A handler that computes a dead flag
    /// anyway is still correct and merely wasteful.
    pub fn set_flag(&mut self, flag: X86Flag, val: Value) -> crate::Result<()> {
        if !self.flag_needed(flag) {
            // Deliberately not an error even for a flag this guest has no
            // variable for: the check below is what reports that, and it still
            // runs on every path a live flag takes.
            return Ok(());
        }
        if let Some(&var) = self.state.flags.get(&flag) {
            // Ensure flag is I8
            let val_i8 = if self.builder.func.dfg.value_type(val) == types::I8 {
                val
            } else {
                self.builder.ins().ireduce(types::I8, val)
            };
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

    // Operand accessors

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

    /// Read both operands of a binary `r/m, r/m/imm` instruction, at the width of
    /// the destination.
    ///
    /// Returns `(dst, src)` but reads **src first**, which is the order the
    /// operands' side effects must appear in. `size` comes from operand 0 and is
    /// applied to operand 1 too, so a sign-extended immediate materialises at full
    /// width rather than at its encoded width.
    ///
    /// For a read-modify-*write* instruction use
    /// [`read_binary_rmw`](Self::read_binary_rmw) instead.
    pub fn read_binary_operands(
        &mut self,
        instruction: &Instruction,
        size: usize,
    ) -> crate::Result<(Value, Value)> {
        let src = self.read_operand(instruction, 1, size)?;
        let dst = self.read_operand(instruction, 0, size)?;
        Ok((dst, src))
    }

    /// [`read_binary_operands`](Self::read_binary_operands) for an instruction that
    /// writes its destination back, returning `(place, dst, src)`.
    ///
    /// The destination is resolved to a [`Place`] once and read through it, so
    /// [`write_place`](Self::write_place) stores to the address the load computed.
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

    /// read-modify-write of a unary instruction (`INC`, `NEG`,
    /// `NOT`, the shifts and rotates): returns `(size, place, value)` for
    /// operand 0, resolved once so the write-back reuses the load's address.
    pub fn read_unary_rmw(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<(usize, Place, Value)> {
        let size = self.operand_size(instruction, 0)?;
        let place = self.resolve_place(instruction, 0)?;
        let val = self.read_place(place, size)?;
        Ok((size, place, val))
    }

    /// Resolve an operand to the location it names, computing a memory
    /// operand's effective address exactly once.
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

    /// Read `CF` as the carry/borrow input of an `ADC`/`SBB`, widened to an
    /// operand of `size` bytes.
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

        // A 0x67 prefix selects 32-bit addressing: iced reports 32-bit base and
        // index registers. Compute in I64 (widening each component) and
        // truncate the final sum to 32 bits, matching hardware's wrap-around,
        // so the load/store below always sees a 64-bit address.
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
            // RIP-relative: iced-x86 sets 'disp' to the absolute target address
            // (as decoded), i.e. a lift-time VA. Materialize it per the
            // configured `AddressMode`.
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

        // Only add displacement if it was NOT RIP-relative.
        // For RIP-relative, the target address is already our 'ptr'.
        if base_reg != Register::RIP && disp != 0 {
            ptr = self.builder.ins().iadd_imm_s(ptr, disp as i64);
        }

        // 32-bit addressing truncates the effective address to 32 bits (then
        // zero-extends) before any segment base is applied.
        if addr32 {
            ptr = self.builder.ins().band_imm_u(ptr, 0xFFFF_FFFF);
        }

        // Add FS or GS segment base. In x86-64 long mode DS/ES/SS/CS have
        // base 0 and are ignored; only FS and GS carry non-zero bases.
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

    /// Read an instruction's immediate operand as an `i64`, honouring x86
    /// sign-extension of `imm8`/`imm32` immediates into wider operands.
    ///
    /// iced-x86's [`Instruction::immediate64`] returns the *raw, unsigned* bits, so
    /// the sign-extended operand kinds must go through their matching
    /// sign-extending accessors - otherwise `and rax, -16` (`imm8 = 0xf0`) would be
    /// read as `+240` and mask only the low byte.
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

    // XMM accessors.
    //
    // The lo/hi accessors work under both backing representations; in vector mode
    // they synthesize lane access via `extractlane`/`insertlane`. Use the native
    // `get_xmm_vec`/`set_xmm_vec` when a whole 128-bit value is moved, so vector
    // mode avoids the lane-shuffle round trip.

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
        // In vector mode both halves are the one variable, so writing either
        // defines the whole register.
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

    /// Return the full 128-bit value of XMM register `idx` (0-15) as a
    /// native `I64X2` value.
    ///
    /// In vector mode this is a direct `use_var`. In scalar mode it
    /// assembles the two halves with `scalar_to_vector` + `insertlane`.
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

    /// Write a full 128-bit `I64X2` value to XMM register `idx` (0-15)
    /// and mark dirty.
    ///
    /// In vector mode this is a direct `def_var`. In scalar mode it
    /// splits the lanes back into the two scalar halves.
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
