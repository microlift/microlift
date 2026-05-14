use crate::MicroliftError;
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

// Type helpers

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
    pub fn get_reg_val(&mut self, reg: Register) -> crate::Result<Value> {
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

    pub fn set_reg_val(&mut self, reg: Register, val: Value) -> crate::Result<()> {
        // Writing RIP directly via a data instruction is illegal in x86-64.
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
            // Mark this register dirty for selective sync.
            self.dirty_gprs.insert(base_reg);
            Ok(())
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("set-reg-{:?}", reg),
            })
        }
    }

    pub fn set_flag(&mut self, flag: X86Flag, val: Value) -> crate::Result<()> {
        if let Some(&var) = self.state.flags.get(&flag) {
            // Ensure flag is I8
            let val_i8 = if self.builder.func.dfg.value_type(val) == types::I8 {
                val
            } else {
                self.builder.ins().ireduce(types::I8, val)
            };
            self.builder.def_var(var, val_i8);
            // Mark this flag dirty for selective sync.
            self.dirty_flags.insert(flag);
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
                Ok(self.builder.ins().load(ty, MemFlags::new(), ptr, 0))
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => Ok(self
                .builder
                .ins()
                .iconst(ty, instruction.immediate64() as i64)),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("read-op-{}-{:?}", op_index, op_kind),
            }),
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
                self.builder.ins().store(MemFlags::new(), val, ptr, 0);
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

    /// Read the immediate from an instruction, sign-extended to i64.
    pub fn read_imm64(&self, instruction: &Instruction) -> i64 {
        instruction.immediate64() as i64
    }

    // XMM accessors (scalar)

    /// Return the low 64 bits of XMM register `idx` (0-15).
    pub fn get_xmm_lo(&mut self, idx: u8) -> Value {
        let var = *self.state.xmms_lo.get(&idx).unwrap();
        self.builder.use_var(var)
    }

    /// Return the high 64 bits of XMM register `idx` (0-15).
    pub fn get_xmm_hi(&mut self, idx: u8) -> Value {
        let var = *self.state.xmms_hi.get(&idx).unwrap();
        self.builder.use_var(var)
    }

    /// Write the low 64 bits of XMM register `idx` (0-15) and mark dirty.
    pub fn set_xmm_lo(&mut self, idx: u8, val: Value) {
        let var = *self.state.xmms_lo.get(&idx).unwrap();
        self.builder.def_var(var, val);
        self.dirty_xmms.insert(idx);
    }

    /// Write the high 64 bits of XMM register `idx` (0-15) and mark dirty.
    pub fn set_xmm_hi(&mut self, idx: u8, val: Value) {
        let var = *self.state.xmms_hi.get(&idx).unwrap();
        self.builder.def_var(var, val);
        self.dirty_xmms.insert(idx);
    }

    /// Read the current FS segment base from the SSA variable.
    pub fn get_fs_base(&mut self) -> Value {
        self.builder.use_var(self.state.fs_base)
    }

    /// Read the current GS segment base from the SSA variable.
    pub fn get_gs_base(&mut self) -> Value {
        self.builder.use_var(self.state.gs_base)
    }

    /// Write a new FS segment base and mark it dirty.
    pub fn set_fs_base(&mut self, val: Value) {
        self.builder.def_var(self.state.fs_base, val);
        self.dirty_fs_base = true;
    }

    /// Write a new GS segment base and mark it dirty.
    pub fn set_gs_base(&mut self, val: Value) {
        self.builder.def_var(self.state.gs_base, val);
        self.dirty_gs_base = true;
    }
}
