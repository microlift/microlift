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

        // In x86-64 the PUSH operand size is 8 by default, or 2 under a 0x66
        // operand-size prefix (a 16-bit push). 32-bit pushes are not encodable
        // in long mode. The slot decrement must match the value width.
        let push_size: usize = match op0 {
            OpKind::Register => instruction.op0_register().size(),
            OpKind::Memory => instruction.memory_size().size(),
            OpKind::Immediate16 => 2,
            _ => 8,
        };
        let ty = size_to_type(push_size);

        let val = match op0 {
            OpKind::Register => self.get_reg_val(instruction.op0_register())?,
            OpKind::Immediate8to64 => {
                // Must use immediate8to64() - immediate32to64() does not
                // sign-extend from 8 bits and returns the raw unsigned value.
                self.builder
                    .ins()
                    .iconst(types::I64, instruction.immediate8to64())
            }
            OpKind::Immediate16 => self
                .builder
                .ins()
                .iconst(types::I16, instruction.immediate16() as i64),
            OpKind::Immediate8
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
                self.builder.ins().load(ty, MemFlags::new(), ptr, 0)
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

        // Match the operand size: 8 by default, 2 for a 0x66-prefixed pop.
        let pop_size: usize = match op0 {
            OpKind::Register => instruction.op0_register().size(),
            OpKind::Memory => instruction.memory_size().size(),
            _ => 8,
        };
        let ty = size_to_type(pop_size);

        let rsp = self.get_reg_val(Register::RSP)?;
        let val = self.builder.ins().load(ty, MemFlags::new(), rsp, 0);

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
        // XCHG with a memory operand asserts LOCK implicitly on x86 (no prefix
        // byte required), so it must be atomic. Route it through the same
        // atomic-swap path used for `LOCK XCHG`.
        if instruction.op0_kind() == OpKind::Memory || instruction.op1_kind() == OpKind::Memory {
            return self.handle_lock_xchg(instruction);
        }

        let size = self.operand_size(instruction, 0)?;
        let a = self.read_operand(instruction, 0, size)?;
        let b = self.read_operand(instruction, 1, size)?;
        self.write_operand(instruction, 0, b)?;
        self.write_operand(instruction, 1, a)?;
        Ok(())
    }

    /// Is this string instruction carrying a `REP`/`REPE`/`REPNE` prefix?
    ///
    /// For `MOVS`/`STOS`/`LODS` only the plain `REP` (F3) form is
    /// architecturally meaningful; iced reports it through any of these
    /// accessors depending on how it classifies the F2/F3 byte, so accept all.
    fn has_rep_prefix(instruction: &Instruction) -> bool {
        instruction.has_rep_prefix()
            || instruction.has_repe_prefix()
            || instruction.has_repne_prefix()
    }

    /// Wrap a single string-operation body in an `RCX`-driven `REP` loop when
    /// the instruction has a repeat prefix; otherwise emit the body once.
    ///
    /// The emitted shape for a repeated op is:
    /// ```text
    ///   header:  if RCX == 0 { goto done }        (RCX is unsigned)
    ///   body:    <one element>; RCX -= 1
    ///            if RCX != 0 { goto body } else { goto done }
    ///   done:    ...
    /// ```
    /// which matches `REP`'s "execute RCX times, testing the count *before*
    /// each iteration" semantics (RCX == 0 performs zero iterations). The
    /// address size is assumed 64-bit (RCX/RSI/RDI), consistent with the rest
    /// of the string-op lowering.
    fn emit_string_op(
        &mut self,
        instruction: &Instruction,
        size: usize,
        body: fn(&mut Self, usize) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if !Self::has_rep_prefix(instruction) {
            return body(self, size);
        }

        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // header: skip the whole loop when the count is already zero.
        let count = self.get_reg_val(Register::RCX)?;
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, count, 0);
        self.builder
            .ins()
            .brif(is_zero, done_block, &[], body_block, &[]);

        // body: one element, decrement RCX, loop while it stays nonzero.
        self.builder.switch_to_block(body_block);
        body(self, size)?;
        let cur = self.get_reg_val(Register::RCX)?;
        let dec = self.builder.ins().iadd_imm(cur, -1);
        self.set_reg_val(Register::RCX, dec)?;
        let nonzero = self.builder.ins().icmp_imm(IntCC::NotEqual, dec, 0);
        self.builder
            .ins()
            .brif(nonzero, body_block, &[], done_block, &[]);
        self.builder.seal_block(body_block);

        self.builder.switch_to_block(done_block);
        self.builder.seal_block(done_block);
        self.block_terminated = false;
        Ok(())
    }

    /// MOVSB - move byte from [RSI] to [RDI], advance both by ±1.
    pub(super) fn handle_movsb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 1, Self::emit_movs)
    }

    /// MOVSW - move word (2 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 2, Self::emit_movs)
    }

    /// MOVSD - move dword (4 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsd_mem(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 4, Self::emit_movs)
    }

    /// MOVSQ - move qword (8 bytes) from [RSI] to [RDI].
    pub(super) fn handle_movsq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 8, Self::emit_movs)
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

    /// The accumulator register (AL/AX/EAX/RAX) implied by a string element
    /// `size`, used by STOS (source) and LODS (destination).
    fn string_acc_reg(size: usize) -> Register {
        match size {
            1 => Register::AL,
            2 => Register::AX,
            4 => Register::EAX,
            _ => Register::RAX,
        }
    }

    /// STOSB - store AL to [RDI], advance RDI by ±1.
    pub(super) fn handle_stosb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 1, Self::emit_stos)
    }

    /// STOSW - store AX to [RDI], advance RDI by ±2.
    pub(super) fn handle_stosw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 2, Self::emit_stos)
    }

    /// STOSD - store EAX to [RDI], advance RDI by ±4.
    pub(super) fn handle_stosd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 4, Self::emit_stos)
    }

    /// STOSQ - store RAX to [RDI], advance RDI by ±8.
    pub(super) fn handle_stosq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 8, Self::emit_stos)
    }

    fn emit_stos(&mut self, size: usize) -> crate::Result<()> {
        let rdi = self.get_reg_val(Register::RDI)?;
        let val = self.get_reg_val(Self::string_acc_reg(size))?;
        self.builder.ins().store(MemFlags::new(), val, rdi, 0);
        let delta = self.string_delta(size);
        let new_rdi = self.builder.ins().iadd(rdi, delta);
        self.set_reg_val(Register::RDI, new_rdi)?;
        Ok(())
    }

    /// LODSB/W/D/Q - load from [RSI] into AL/AX/EAX/RAX, advance RSI.
    pub(super) fn handle_lodsb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 1, Self::emit_lods)
    }
    pub(super) fn handle_lodsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 2, Self::emit_lods)
    }
    pub(super) fn handle_lodsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 4, Self::emit_lods)
    }
    pub(super) fn handle_lodsq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_string_op(instruction, 8, Self::emit_lods)
    }

    fn emit_lods(&mut self, size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let rsi = self.get_reg_val(Register::RSI)?;
        let val = self.builder.ins().load(ty, MemFlags::new(), rsi, 0);
        self.set_reg_val(Self::string_acc_reg(size), val)?;
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
