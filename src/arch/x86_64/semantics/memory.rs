use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::ExitReason;
use crate::lifter::{Lifter, X86Flag};
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

/// The five status flags `LAHF` and `SAHF` carry, and their bit positions
/// within the low byte of RFLAGS. Overflow and Direction are deliberately
/// absent: they sit at bits 11 and 10, outside the byte.
const LOW_BYTE_FLAG_BITS: [(X86Flag, u8); 5] = [
    (X86Flag::Carry, 0),
    (X86Flag::Parity, 2),
    (X86Flag::Auxiliary, 4),
    (X86Flag::Zero, 6),
    (X86Flag::Sign, 7),
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

    /// MOVSXD - sign-extend to the destination register's width.
    ///
    /// The familiar form is `movsxd r64, r/m32` under `REX.W`, and extending to
    /// `I64` unconditionally is right for it and wrong for the other two the
    /// encoding permits. Without `REX.W` the destination is a 32-bit register
    /// and the instruction is a plain `mov`; with a `0x66` prefix it is 16-bit.
    /// Both are discouraged by the manual and neither is unreachable: a JIT
    /// emitting `movsxd ecx, [rdi-0x30]` is what found this, in Mono's generated
    /// code under rosetta3. Extending to `I64` there produced a value one type
    /// wider than the register write expected, and the whole region failed
    /// Cranelift's verifier - an error about IR rather than about the guest.
    pub(super) fn handle_movsxd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = instruction.op0_register();
        let src_size = self.operand_size(instruction, 1)?;
        let src_val = self.read_operand(instruction, 1, src_size)?;
        let dst_ty = size_to_type(dst_reg.size());
        // `sextend`, `ireduce` and `uextend` all reject a conversion to the type
        // a value already has, so the equal case has to be a no-op rather than
        // an identity conversion.
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
            // 16-bit pushes: `push imm16` and 0x66-prefixed `push imm8`.
            OpKind::Immediate16 | OpKind::Immediate8to16 => 2,
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
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
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
        self.builder
            .ins()
            .store(MemFlagsData::new(), val, new_rsp, 0);
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
        let val = self.builder.ins().load(ty, MemFlagsData::new(), rsp, 0);

        let new_rsp = self.builder.ins().iadd_imm_s(rsp, pop_size as i64);
        self.set_reg_val(Register::RSP, new_rsp)?;

        match op0 {
            OpKind::Register => {
                self.set_reg_val(instruction.op0_register(), val)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), val, ptr, 0);
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
        let old_rbp = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::new(), rbp, 0);
        self.set_reg_val(Register::RBP, old_rbp)?;
        let new_rsp = self.builder.ins().iadd_imm_s(rbp, 8);
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

    /// Address size (in bytes: 8/4/2) selected for a string instruction, read
    /// from the SI/DI operand kind iced assigns. A `0x67` address-size prefix
    /// switches the pointer/counter registers from RSI/RDI/RCX to their 32- or
    /// 16-bit sub-registers; the default in long mode is 64-bit.
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

    /// The `(counter, source-index, dest-index)` register triple for a string
    /// op of the given address size: `(RCX, RSI, RDI)` / `(ECX, ESI, EDI)` /
    /// `(CX, SI, DI)`.
    fn string_addr_regs(addr_size: usize) -> (Register, Register, Register) {
        match addr_size {
            2 => (Register::CX, Register::SI, Register::DI),
            4 => (Register::ECX, Register::ESI, Register::EDI),
            _ => (Register::RCX, Register::RSI, Register::RDI),
        }
    }

    /// Effective 64-bit linear address held in a string-op index register
    /// (`RSI`/`ESI`/`SI`, etc.), zero-extended from a narrower address size.
    fn string_ptr(&mut self, reg: Register) -> crate::Result<Value> {
        let v = self.get_reg_val(reg)?;
        Ok(if self.builder.func.dfg.value_type(v) == types::I64 {
            v
        } else {
            self.builder.ins().uextend(types::I64, v)
        })
    }

    /// Advance a string-op index register by ±`element_size` (sign per `DF`),
    /// computed in the register's own width so it wraps within the active
    /// address size (`set_reg_val` then zero-extends a 32-bit result).
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

    /// Wrap a single string-operation body in a counter-driven `REP` loop when
    /// the instruction has a repeat prefix; otherwise emit the body once.
    ///
    /// The emitted shape for a repeated op is:
    /// ```text
    ///   header:  if count == 0 { goto done }      (count is unsigned)
    ///   body:    <one element>; count -= 1
    ///            if count != 0 { goto body } else { goto done }
    ///   done:    ...
    /// ```
    /// which matches `REP`'s "execute count times, testing the count *before*
    /// each iteration" semantics (count == 0 performs zero iterations). The
    /// counter register (`RCX`/`ECX`/`CX`) tracks the instruction's address size.
    /// The body receives `(element_size, addr_size)`.
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

        // header: skip the whole loop when the count is already zero.
        let count = self.get_reg_val(count_reg)?;
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0);
        self.builder
            .ins()
            .brif(is_zero, done_block, &[], body_block, &[]);

        // body: one element, decrement the counter, loop while it stays nonzero.
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
        let val = self.builder.ins().load(ty, MemFlagsData::new(), sptr, 0);
        let dptr = self.string_ptr(di)?;
        self.builder.ins().store(MemFlagsData::new(), val, dptr, 0);
        // DF=0: increment both pointers; DF=1: decrement.
        self.string_advance(si, size)?;
        self.string_advance(di, size)?;
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

    fn emit_stos(&mut self, size: usize, addr_size: usize) -> crate::Result<()> {
        let (_, _, di) = Self::string_addr_regs(addr_size);
        let dptr = self.string_ptr(di)?;
        let val = self.get_reg_val(Self::string_acc_reg(size))?;
        self.builder.ins().store(MemFlagsData::new(), val, dptr, 0);
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
        let val = self.builder.ins().load(ty, MemFlagsData::new(), sptr, 0);
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
        // RIP at the instruction itself (I/O string ops re-execute).
        let rip_at = self.runtime_addr(instruction.ip());

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
            let one_bit = self.builder.ins().band_imm_u(wide, 1);
            let positioned = self.builder.ins().ishl_imm_u(one_bit, bit as i64);
            rflags = self.builder.ins().bor(rflags, positioned);
        }

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, -8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder
            .ins()
            .store(MemFlagsData::new(), rflags, new_rsp, 0);
        Ok(())
    }

    /// POPFQ - pop a 64-bit value off the stack and distribute its bits into
    /// the tracked flags (the six status flags plus DF). Bits for untracked
    /// flags (IF, TF, etc.) and the reserved/system bits are simply discarded.
    pub(super) fn handle_popfq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rsp = self.get_reg_val(Register::RSP)?;
        let rflags = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::new(), rsp, 0);
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;

        for (flag, bit) in RFLAGS_FLAG_BITS {
            // bit value = (rflags >> bit) & 1, narrowed to the I8 flag storage.
            let shifted = self.builder.ins().ushr_imm_u(rflags, bit as i64);
            let one_bit = self.builder.ins().band_imm_u(shifted, 1);
            let val = self.builder.ins().ireduce(types::I8, one_bit);
            self.set_flag(flag, val)?;
        }
        Ok(())
    }

    /// LAHF - load the low byte of RFLAGS into AH.
    ///
    /// `AH = SF:ZF:0:AF:0:PF:1:CF`. The two zero bits and the one bit are
    /// architectural: bits 1, 3 and 5 of RFLAGS are reserved and read as 1, 0
    /// and 0. Overflow is *not* among them - it lives at bit 11, outside the
    /// byte - which is the difference from `PUSHFQ` and the reason this is a
    /// list of five flags rather than a loop over [`RFLAGS_FLAG_BITS`].
    pub(super) fn handle_lahf(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let mut ah = self.builder.ins().iconst(types::I8, 0x2);
        for (flag, bit) in LOW_BYTE_FLAG_BITS {
            let raw = self.builder.use_var(self.state.flags[&flag]);
            // Flags are logically 1-bit; mask defensively before positioning.
            let one_bit = self.builder.ins().band_imm_u(raw, 1);
            let positioned = self.builder.ins().ishl_imm_u(one_bit, bit as i64);
            ah = self.builder.ins().bor(ah, positioned);
        }
        self.set_reg_val(Register::AH, ah)
    }

    /// SAHF - store AH into the low byte of RFLAGS.
    ///
    /// The inverse of [`handle_lahf`](Self::handle_lahf), over the same five
    /// flags. Overflow is untouched, and so is Direction: neither is in the
    /// byte.
    pub(super) fn handle_sahf(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ah = self.get_reg_val(Register::AH)?;
        for (flag, bit) in LOW_BYTE_FLAG_BITS {
            let shifted = self.builder.ins().ushr_imm_u(ah, bit as i64);
            let one_bit = self.builder.ins().band_imm_u(shifted, 1);
            self.set_flag(flag, one_bit)?;
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
