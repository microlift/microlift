use crate::MicroliftError;
use crate::context::ExitReason;
use crate::lifter::{Lifter, X86Flag};
use crate::arch::x86_64::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

/// Tracked flags and their RFLAGS bit positions, shared by PUSHFQ/POPFQ.
/// Covers the six arithmetic status flags plus DF; other RFLAGS bits are not
/// modelled.
const RFLAGS_FLAG_BITS: [(X86Flag, u8); 7] = [
    (X86Flag::Carry, 0),
    (X86Flag::Parity, 2),
    (X86Flag::Auxiliary, 4),
    (X86Flag::Zero, 6),
    (X86Flag::Sign, 7),
    (X86Flag::Direction, 10),
    (X86Flag::Overflow, 11),
];

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
        // get_mem_ptr always returns I64. set_reg_val expects the value to match
        // the register's natural width (I32 for 32-bit regs, etc.) so that its
        // internal uextend/merge logic works. Reduce before passing.
        let val = if reg.size() == 8 {
            ptr
        } else {
            let ty = size_to_type(reg.size());
            self.builder.ins().ireduce(ty, ptr)
        };
        self.set_reg_val(reg, val)?;
        Ok(())
    }

    pub(super) fn handle_push(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();
        let push_size = 8; // x86-64 PUSH is always 8 bytes

        let val = match op0 {
            OpKind::Register => self.get_reg_val(instruction.op0_register())?,
            OpKind::Immediate8to64 => {
                // Must use immediate8to64() - immediate32to64() does not
                // sign-extend from 8 bits and returns the raw unsigned value.
                self.builder
                    .ins()
                    .iconst(types::I64, instruction.immediate8to64())
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
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

    /// Signed pointer delta applied by a string op for one element of `size`
    /// bytes: `+size` when `DF=0`, `-size` when `DF=1`.
    fn string_delta(&mut self, size: usize) -> Value {
        let df = self.builder.use_var(self.state.flags[&X86Flag::Direction]);
        let pos = self.builder.ins().iconst(types::I64, size as i64);
        let neg = self.builder.ins().iconst(types::I64, -(size as i64));
        let df_set = self.builder.ins().icmp_imm(IntCC::NotEqual, df, 0);
        self.builder.ins().select(df_set, neg, pos)
    }

    fn emit_movs(&mut self, size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let rsi = self.get_reg_val(Register::RSI)?;
        let rdi = self.get_reg_val(Register::RDI)?;
        let val = self.builder.ins().load(ty, MemFlags::new(), rsi, 0);
        self.builder.ins().store(MemFlags::new(), val, rdi, 0);
        // DF=0: increment both pointers; DF=1: decrement.
        let delta = self.string_delta(size);
        let new_rsi = self.builder.ins().iadd(rsi, delta);
        let new_rdi = self.builder.ins().iadd(rdi, delta);
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
        let delta = self.string_delta(size);
        let new_rdi = self.builder.ins().iadd(rdi, delta);
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
        let delta = self.string_delta(size);
        let new_rsi = self.builder.ins().iadd(rsi, delta);
        self.set_reg_val(Register::RSI, new_rsi)?;
        Ok(())
    }

    /// INSB/INSW/INSD - input from the I/O port in `DX` into `ES:[RDI]`.
    pub(super) fn handle_insb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_ins(instruction, 1)
    }
    pub(super) fn handle_insw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_ins(instruction, 2)
    }
    pub(super) fn handle_insd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_ins(instruction, 4)
    }

    /// `INS` reads from an I/O port, which cannot be done from lifted userspace
    /// code. Emit a [`ExitReason::PortInput`] VM exit so the host can emulate the
    /// access: RIP points *at* the instruction and `exit_code` carries the
    /// element size. The port (`DX`), destination (`ES:[RDI]`), `DF`, and `RCX`
    /// (for `REP INS`) are all available in the synced register context.
    fn emit_ins(&mut self, instruction: &Instruction, size: usize) -> crate::Result<()> {
        self.emit_io_string_exit(instruction, size, ExitReason::PortInput)
    }

    /// OUTSB/OUTSW/OUTSD - output from `DS:[RSI]` to the I/O port in `DX`.
    pub(super) fn handle_outsb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_outs(instruction, 1)
    }
    pub(super) fn handle_outsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_outs(instruction, 2)
    }
    pub(super) fn handle_outsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_outs(instruction, 4)
    }

    /// `OUTS` writes to an I/O port - the mirror of `INS`. Emit a
    /// [`ExitReason::PortOutput`] VM exit so the host can emulate the access. The
    /// port (`DX`), source (`DS:[RSI]`), `DF`, and `RCX` (for `REP OUTS`) are all
    /// available in the synced register context.
    fn emit_outs(&mut self, instruction: &Instruction, size: usize) -> crate::Result<()> {
        self.emit_io_string_exit(instruction, size, ExitReason::PortOutput)
    }

    /// Shared exit emitter for the port I/O string instructions (`INS`/`OUTS`):
    /// RIP points *at* the instruction and `exit_code` carries the element size
    /// so the host can emulate the access and resume past it.
    fn emit_io_string_exit(
        &mut self,
        instruction: &Instruction,
        size: usize,
        reason: ExitReason,
    ) -> crate::Result<()> {
        // RIP at the instruction itself, computed relative to the image base so
        // it stays correct under relocation.
        let offset = instruction.ip().wrapping_sub(self.lift_base);
        let image_base = self.builder.use_var(self.state.image_base);
        let rip_at = self.builder.ins().iadd_imm(image_base, offset as i64);

        self.sync_to_context(Some(rip_at), reason)?;
        self.write_exit_code(size as u32);
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// PUSHFQ - assemble RFLAGS from the tracked status flags and push the
    /// 64-bit value onto the stack.
    ///
    /// The tracked flags are CF/PF/AF/ZF/SF/OF and DF; any other RFLAGS bit
    /// reads back as 0 except bit 1, which is architecturally always set.
    pub(super) fn handle_pushfq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        // Bit 1 of RFLAGS is reserved and always reads as 1.
        let mut rflags = self.builder.ins().iconst(types::I64, 0x2);
        for (flag, bit) in RFLAGS_FLAG_BITS {
            let raw = self.builder.use_var(self.state.flags[&flag]);
            let wide = self.builder.ins().uextend(types::I64, raw);
            // Flags are logically 1-bit; mask defensively before positioning.
            let one_bit = self.builder.ins().band_imm(wide, 1);
            let positioned = self.builder.ins().ishl_imm(one_bit, bit as i64);
            rflags = self.builder.ins().bor(rflags, positioned);
        }

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm(rsp, -8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder.ins().store(MemFlags::new(), rflags, new_rsp, 0);
        Ok(())
    }

    /// POPFQ - pop a 64-bit value off the stack and distribute its bits into the
    /// tracked status flags. Bits for untracked flags (DF, IF, etc.) and the
    /// reserved/system bits are simply discarded.
    pub(super) fn handle_popfq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rsp = self.get_reg_val(Register::RSP)?;
        let rflags = self.builder.ins().load(types::I64, MemFlags::new(), rsp, 0);
        let new_rsp = self.builder.ins().iadd_imm(rsp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;

        for (flag, bit) in RFLAGS_FLAG_BITS {
            // bit value = (rflags >> bit) & 1, narrowed to the I8 flag storage.
            let shifted = self.builder.ins().ushr_imm(rflags, bit as i64);
            let one_bit = self.builder.ins().band_imm(shifted, 1);
            let val = self.builder.ins().ireduce(types::I8, one_bit);
            self.set_flag(flag, val)?;
        }
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
