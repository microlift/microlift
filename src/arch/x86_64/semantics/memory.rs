use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::ExitReason;
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind, Register};

/// Tracked flags and RFLAGS bit positions for PUSHFQ/POPFQ: six status flags plus DF.
const RFLAGS_FLAG_BITS: [(X86Flag, u8); 7] = [
    (X86Flag::Carry, 0),
    (X86Flag::Parity, 2),
    (X86Flag::Auxiliary, 4),
    (X86Flag::Zero, 6),
    (X86Flag::Sign, 7),
    (X86Flag::Direction, 10),
    (X86Flag::Overflow, 11),
];

/// The five status flags `LAHF`/`SAHF` carry, by bit in the low RFLAGS byte (OF and DF sit outside it).
const LOW_BYTE_FLAG_BITS: [(X86Flag, u8); 5] = [
    (X86Flag::Carry, 0),
    (X86Flag::Parity, 2),
    (X86Flag::Auxiliary, 4),
    (X86Flag::Zero, 6),
    (X86Flag::Sign, 7),
];

/// The stack slot `PUSH`/`POP` of a segment register uses, or `None`.
/// `PUSH FS` moves `RSP` by 8 and zero-extends the selector though `Register::size()` is 2; only `0x66` makes it 2 bytes.
fn segment_stack_slot_size(instruction: &Instruction) -> Option<usize> {
    use iced_x86::Code;
    if !crate::arch::x86_64::regs::is_segment_register(instruction.op0_register()) {
        return None;
    }
    match instruction.code() {
        Code::Pushw_FS | Code::Pushw_GS | Code::Popw_FS | Code::Popw_GS => Some(2),
        _ => Some(8),
    }
}

impl<'a> Lifter<'a> {
    pub(super) fn handle_nop(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    pub(super) fn handle_mov(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let src_val = self.read_operand(instruction, 1, size)?;
        // `MOV r/m16, Sreg`: zero-extend the selector for a 32/64-bit register destination.
        // A memory destination stays 16 bits.
        let src_val = if crate::arch::x86_64::regs::is_segment_register(
            self.operand_register(instruction, 1),
        ) && size > 2
        {
            self.builder.ins().uextend(size_to_type(size), src_val)
        } else {
            src_val
        };
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

    /// MOVSXD - sign-extend to the destination register's width.
    /// Also encodable without `REX.W` (32-bit, plain `mov`) and with `0x66` (16-bit); extending to `I64` there fails the verifier.
    pub(super) fn handle_movsxd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = instruction.op0_register();
        let src_size = self.operand_size(instruction, 1)?;
        let src_val = self.read_operand(instruction, 1, src_size)?;
        let dst_ty = size_to_type(dst_reg.size());
        // Cranelift extends reject an equal-width conversion.
        let val = match dst_reg.size().cmp(&src_size) {
            std::cmp::Ordering::Greater => self.builder.ins().sextend(dst_ty, src_val),
            std::cmp::Ordering::Equal => src_val,
            std::cmp::Ordering::Less => self.builder.ins().ireduce(dst_ty, src_val),
        };
        self.set_reg_val(dst_reg, val)?;
        Ok(())
    }

    pub(super) fn handle_lea(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let reg = instruction.op0_register();
        let ptr = self.get_mem_ptr(instruction, 1)?;
        // `get_mem_ptr` is `I64`; `set_reg_val` wants the register's natural width.
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

        // Operand size is 8, or 2 under `0x66`; 32-bit pushes are not encodable.
        let push_size: usize = match op0 {
            OpKind::Register => segment_stack_slot_size(instruction)
                .unwrap_or_else(|| instruction.op0_register().size()),
            OpKind::Memory => instruction.memory_size().size(),
            // 16-bit pushes: `push imm16` and 0x66-prefixed `push imm8`.
            OpKind::Immediate16 | OpKind::Immediate8to16 => 2,
            _ => 8,
        };
        let ty = size_to_type(push_size);

        let val = match op0 {
            OpKind::Register => {
                let v = self.get_reg_val(instruction.op0_register())?;
                // `PUSH FS`/`GS` write a full 8-byte slot, selector zero-extended.
                if self.builder.func.dfg.value_type(v) == ty {
                    v
                } else {
                    self.builder.ins().uextend(ty, v)
                }
            }
            OpKind::Immediate8to64 => {
                // `immediate32to64` does not sign-extend from 8 bits.
                self.builder
                    .ins()
                    .iconst(types::I64, instruction.immediate8to64())
            }
            OpKind::Immediate16 => self
                .builder
                .ins()
                .iconst(types::I16, instruction.immediate16() as i64),
            // 0x66-prefixed `push imm8`: sign-extend to 16 bits, 2-byte slot.
            OpKind::Immediate8to16 => self
                .builder
                .ins()
                .iconst(types::I16, instruction.immediate8to16() as i64),
            OpKind::Immediate8
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to32
            | OpKind::Immediate32to64 => {
                let imm = instruction.immediate32to64();
                self.builder.ins().iconst(types::I64, imm)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().load(ty, self.guest_flags, ptr, 0)
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("push-{:?}", op0),
                });
            }
        };

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, -(push_size as i64));
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder.ins().store(self.guest_flags, val, new_rsp, 0);
        Ok(())
    }

    pub(super) fn handle_pop(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let op0 = instruction.op0_kind();

        let pop_size: usize = match op0 {
            OpKind::Register => segment_stack_slot_size(instruction)
                .unwrap_or_else(|| instruction.op0_register().size()),
            OpKind::Memory => instruction.memory_size().size(),
            _ => 8,
        };
        let ty = size_to_type(pop_size);

        let rsp = self.get_reg_val(Register::RSP)?;
        let val = self.builder.ins().load(ty, self.guest_flags, rsp, 0);

        let new_rsp = self.builder.ins().iadd_imm_s(rsp, pop_size as i64);
        self.set_reg_val(Register::RSP, new_rsp)?;

        match op0 {
            OpKind::Register => {
                self.set_reg_val(instruction.op0_register(), val)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(self.guest_flags, val, ptr, 0);
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
        self.set_reg_val(Register::RSP, rbp)?;
        let old_rbp = self
            .builder
            .ins()
            .load(types::I64, self.guest_flags, rbp, 0);
        self.set_reg_val(Register::RBP, old_rbp)?;
        let new_rsp = self.builder.ins().iadd_imm_s(rbp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        Ok(())
    }

    pub(super) fn handle_xchg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        // Memory-operand `XCHG` is implicitly locked; use the atomic-swap path.
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

    /// Whether a `REP`/`REPE`/`REPNE` prefix is present; iced reports F3 through any of the accessors.
    fn has_rep_prefix(instruction: &Instruction) -> bool {
        instruction.has_rep_prefix()
            || instruction.has_repe_prefix()
            || instruction.has_repne_prefix()
    }

    /// Address size in bytes (8/4/2) of a string instruction, from the SI/DI operand kind (`0x67` narrows it).
    fn string_addr_size(instruction: &Instruction) -> usize {
        for i in 0..instruction.op_count() {
            match instruction.op_kind(i) {
                OpKind::MemorySegRSI | OpKind::MemoryESRDI => return 8,
                OpKind::MemorySegESI | OpKind::MemoryESEDI => return 4,
                OpKind::MemorySegSI | OpKind::MemoryESDI => return 2,
                _ => {}
            }
        }
        8
    }

    /// The `(counter, source-index, dest-index)` registers for an address size: `(RCX, RSI, RDI)` / `(ECX, ESI, EDI)` / `(CX, SI, DI)`.
    fn string_addr_regs(addr_size: usize) -> (Register, Register, Register) {
        match addr_size {
            2 => (Register::CX, Register::SI, Register::DI),
            4 => (Register::ECX, Register::ESI, Register::EDI),
            _ => (Register::RCX, Register::RSI, Register::RDI),
        }
    }

    /// Zero-extended 64-bit address held in a string-op index register.
    fn string_ptr(&mut self, reg: Register) -> crate::Result<Value> {
        let v = self.get_reg_val(reg)?;
        Ok(if self.builder.func.dfg.value_type(v) == types::I64 {
            v
        } else {
            self.builder.ins().uextend(types::I64, v)
        })
    }

    /// Advance a string index register by ±`element_size` (per `DF`) in its own width, so it wraps within the address size.
    fn string_advance(&mut self, reg: Register, element_size: usize) -> crate::Result<()> {
        let cur = self.get_reg_val(reg)?;
        let ty = self.builder.func.dfg.value_type(cur);
        let df = self.builder.use_var(self.state.flags[&X86Flag::Direction]);
        let pos = self.builder.ins().iconst(ty, element_size as i64);
        let neg = self.builder.ins().iconst(ty, -(element_size as i64));
        let df_set = self.builder.ins().icmp_imm_s(IntCC::NotEqual, df, 0);
        let delta = self.builder.ins().select(df_set, neg, pos);
        let new = self.builder.ins().iadd(cur, delta);
        self.set_reg_val(reg, new)
    }

    /// Wrap one string-op body in a counter-driven `REP` loop if prefixed, else emit it once.
    ///
    /// Shape of a repeated op:
    /// ```text
    ///   header:  if count == 0 { goto done }      (count is unsigned)
    ///   body:    <one element>; count -= 1
    ///            if count != 0 { goto body } else { goto done }
    ///   done:    ...
    /// ```
    /// The count is tested before each iteration; the body receives `(element_size, addr_size)`.
    fn emit_string_op(
        &mut self,
        instruction: &Instruction,
        size: usize,
        body: fn(&mut Self, usize, usize) -> crate::Result<()>,
    ) -> crate::Result<()> {
        let addr_size = Self::string_addr_size(instruction);
        if !Self::has_rep_prefix(instruction) {
            return body(self, size, addr_size);
        }
        let (count_reg, _, _) = Self::string_addr_regs(addr_size);

        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        let count = self.get_reg_val(count_reg)?;
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0);
        self.builder
            .ins()
            .brif(is_zero, done_block, &[], body_block, &[]);

        self.builder.switch_to_block(body_block);
        body(self, size, addr_size)?;
        let cur = self.get_reg_val(count_reg)?;
        let dec = self.builder.ins().iadd_imm_s(cur, -1);
        self.set_reg_val(count_reg, dec)?;
        let nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, dec, 0);
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

    fn emit_movs(&mut self, size: usize, addr_size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let (_, si, di) = Self::string_addr_regs(addr_size);
        let sptr = self.string_ptr(si)?;
        let val = self.builder.ins().load(ty, self.guest_flags, sptr, 0);
        let dptr = self.string_ptr(di)?;
        self.builder.ins().store(self.guest_flags, val, dptr, 0);
        self.string_advance(si, size)?;
        self.string_advance(di, size)?;
        Ok(())
    }

    /// The accumulator (AL/AX/EAX/RAX) for element `size`, used by STOS and LODS.
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

    fn emit_stos(&mut self, size: usize, addr_size: usize) -> crate::Result<()> {
        let (_, _, di) = Self::string_addr_regs(addr_size);
        let dptr = self.string_ptr(di)?;
        let val = self.get_reg_val(Self::string_acc_reg(size))?;
        self.builder.ins().store(self.guest_flags, val, dptr, 0);
        self.string_advance(di, size)?;
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

    fn emit_lods(&mut self, size: usize, addr_size: usize) -> crate::Result<()> {
        let ty = size_to_type(size);
        let (_, si, _) = Self::string_addr_regs(addr_size);
        let sptr = self.string_ptr(si)?;
        let val = self.builder.ins().load(ty, self.guest_flags, sptr, 0);
        self.set_reg_val(Self::string_acc_reg(size), val)?;
        self.string_advance(si, size)?;
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

    /// `PortInput` VM exit: RIP at the instruction, `exit_code` = element size; port, destination, `DF` and `RCX` are in the synced context.
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

    /// `PortOutput` VM exit, mirror of `INS`; the source is `DS:[RSI]`.
    fn emit_outs(&mut self, instruction: &Instruction, size: usize) -> crate::Result<()> {
        self.emit_io_string_exit(instruction, size, ExitReason::PortOutput)
    }

    /// Shared exit for `INS`/`OUTS`: RIP at the instruction, `exit_code` = element size; the host resumes past it.
    fn emit_io_string_exit(
        &mut self,
        instruction: &Instruction,
        size: usize,
        reason: ExitReason,
    ) -> crate::Result<()> {
        // RIP at the instruction itself (I/O string ops re-execute).
        let rip_at = self.runtime_addr(instruction.ip());

        self.sync_to_context(Some(rip_at), reason)?;
        // The host advances past the instruction once the access is served.
        self.resumes_at(instruction.next_ip());
        self.write_exit_code(size as u32);
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// PUSHFQ - push RFLAGS built from the tracked flags; untracked bits read 0 except bit 1 (always set).
    pub(super) fn handle_pushfq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        // Bit 1 of RFLAGS is reserved and always reads as 1.
        let mut rflags = self.builder.ins().iconst(types::I64, 0x2);
        for (flag, bit) in RFLAGS_FLAG_BITS {
            let raw = self.get_flag(flag);
            let wide = self.builder.ins().uextend(types::I64, raw);
            let one_bit = self.builder.ins().band_imm_u(wide, 1);
            let positioned = self.builder.ins().ishl_imm_u(one_bit, bit as i64);
            rflags = self.builder.ins().bor(rflags, positioned);
        }

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, -8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder
            .ins()
            .store(self.guest_flags, rflags, new_rsp, 0);
        Ok(())
    }

    /// POPFQ - pop into the tracked flags (six status flags plus DF); other bits are discarded.
    pub(super) fn handle_popfq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rsp = self.get_reg_val(Register::RSP)?;
        let rflags = self
            .builder
            .ins()
            .load(types::I64, self.guest_flags, rsp, 0);
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;

        for (flag, bit) in RFLAGS_FLAG_BITS {
            let shifted = self.builder.ins().ushr_imm_u(rflags, bit as i64);
            let one_bit = self.builder.ins().band_imm_u(shifted, 1);
            let val = self.builder.ins().ireduce(types::I8, one_bit);
            self.set_flag(flag, val)?;
        }
        Ok(())
    }

    /// LAHF - `AH = SF:ZF:0:AF:0:PF:1:CF`; five flags, since OF (bit 11) is outside the byte.
    pub(super) fn handle_lahf(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let mut ah = self.builder.ins().iconst(types::I8, 0x2);
        for (flag, bit) in LOW_BYTE_FLAG_BITS {
            let raw = self.get_flag(flag);
            let one_bit = self.builder.ins().band_imm_u(raw, 1);
            let positioned = self.builder.ins().ishl_imm_u(one_bit, bit as i64);
            ah = self.builder.ins().bor(ah, positioned);
        }
        self.set_reg_val(Register::AH, ah)
    }

    /// SAHF - store AH into the low byte of RFLAGS; OF and DF are untouched.
    pub(super) fn handle_sahf(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ah = self.get_reg_val(Register::AH)?;
        for (flag, bit) in LOW_BYTE_FLAG_BITS {
            let shifted = self.builder.ins().ushr_imm_u(ah, bit as i64);
            let one_bit = self.builder.ins().band_imm_u(shifted, 1);
            self.set_flag(flag, one_bit)?;
        }
        Ok(())
    }


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
