//! Atomic instruction semantics.
//!
//! Covers:
//! - Memory fence instructions: MFENCE, SFENCE, LFENCE, SERIALIZE
//! - PAUSE (spin-wait hint)
//! - CMPXCHG / CMPXCHG8B / CMPXCHG16B
//! - LOCK-prefixed arithmetic and bit operations
//!
//! Everything up to 64 bits maps onto Cranelift's `atomic_cas` / `atomic_rmw`,
//! so the LOCK prefix costs nothing extra and the handler is shared with the
//! unlocked form. `CMPXCHG16B` is the exception: only the x64 backend can
//! express a 128-bit CAS, so its lowering is a [`Atomic128Mode`] choice.

use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::lifter::{Atomic128Mode, Lifter, X86Flag};
use cranelift::prelude::*;
use cranelift_codegen::ir::AtomicRmwOp;
use iced_x86::{Instruction, Mnemonic, OpKind, Register};

enum BitOp {
    Set,
    Reset,
    Complement,
}

impl<'a> Lifter<'a> {
    /// MFENCE / SFENCE / LFENCE - emit a full sequential-consistency fence.
    pub(super) fn handle_mfence(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        self.builder.ins().fence();
        Ok(())
    }

    /// PAUSE - spin-wait hint; treated as a NOP.
    /// TODO: Properly implement this
    pub(super) fn handle_pause(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    /// CMPXCHG dst, src
    ///
    /// Compares AL/AX/EAX/RAX with `dst`. If equal: ZF=1, dst <- src.
    /// Otherwise: ZF=0, accumulator <- dst. All comparison flags (CF/OF/SF/ZF)
    /// are updated as if CMP acc, dst had been executed.
    ///
    /// When `dst` is a memory operand, Cranelift's `atomic_cas` is used so
    /// that the compare-and-swap is truly atomic (matching x86 LOCK CMPXCHG
    /// semantics). The LOCK prefix is therefore handled transparently here.
    pub(super) fn handle_cmpxchg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);

        let acc_reg = match size {
            1 => Register::AL,
            2 => Register::AX,
            4 => Register::EAX,
            8 => Register::RAX,
            _ => unreachable!(),
        };
        let acc_val = self.get_reg_val(acc_reg)?;
        // acc_val may be wider than ty (e.g. RAX used for an 8-bit cmpxchg is
        // impossible, but for safety reduce when types don't match).
        let acc_narrow = if self.builder.func.dfg.value_type(acc_val) != ty {
            self.builder.ins().ireduce(ty, acc_val)
        } else {
            acc_val
        };

        let src_val = self.read_operand(instruction, 1, size)?;

        match instruction.op0_kind() {
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                // atomic_cas: if *ptr == acc_narrow { *ptr = src_val }; returns old *ptr
                let old =
                    self.builder
                        .ins()
                        .atomic_cas(MemFlagsData::new(), ptr, acc_narrow, src_val);

                // Update flags as CMP acc, old  (acc - old)
                let cmp_result = self.builder.ins().isub(acc_narrow, old);
                self.update_flags_sub(acc_narrow, old, cmp_result, size)?;

                // If CAS failed (old != acc), write old into accumulator.
                let succeeded = self.builder.ins().icmp(IntCC::Equal, old, acc_narrow);
                let new_acc = self.builder.ins().select(succeeded, acc_narrow, old);
                self.set_reg_val(acc_reg, new_acc)?;
            }
            OpKind::Register => {
                let dst_val = self.read_operand(instruction, 0, size)?;

                // Update flags as CMP acc, dst  (acc - dst)
                let cmp_result = self.builder.ins().isub(acc_narrow, dst_val);
                self.update_flags_sub(acc_narrow, dst_val, cmp_result, size)?;

                let equal = self.builder.ins().icmp(IntCC::Equal, acc_narrow, dst_val);

                // dst <- src if equal, else unchanged
                let new_dst = self.builder.ins().select(equal, src_val, dst_val);
                self.write_operand(instruction, 0, new_dst)?;

                // acc <- dst (original) if not equal, else unchanged
                let new_acc = self.builder.ins().select(equal, acc_narrow, dst_val);
                self.set_reg_val(acc_reg, new_acc)?;
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("cmpxchg-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// CMPXCHG8B [mem]
    ///
    /// Atomically compares EDX:EAX with the 64-bit value at [mem].
    /// If equal: ZF=1, [mem] <- ECX:EBX.
    /// Otherwise: ZF=0, EDX:EAX <- [mem].
    pub(super) fn handle_cmpxchg8b(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let ptr = self.get_mem_ptr(instruction, 0)?;

        // Build expected (EDX:EAX) as a single I64.
        let eax = self.get_reg_val(Register::EAX)?; // I32
        let edx = self.get_reg_val(Register::EDX)?; // I32
        let eax64 = self.builder.ins().uextend(types::I64, eax);
        let edx64 = self.builder.ins().uextend(types::I64, edx);
        let edx_hi = self.builder.ins().ishl_imm_u(edx64, 32);
        let expected = self.builder.ins().bor(edx_hi, eax64);

        // Build replacement (ECX:EBX) as a single I64.
        let ebx = self.get_reg_val(Register::EBX)?; // I32
        let ecx = self.get_reg_val(Register::ECX)?; // I32
        let ebx64 = self.builder.ins().uextend(types::I64, ebx);
        let ecx64 = self.builder.ins().uextend(types::I64, ecx);
        let ecx_hi = self.builder.ins().ishl_imm_u(ecx64, 32);
        let replacement = self.builder.ins().bor(ecx_hi, ebx64);

        let old = self
            .builder
            .ins()
            .atomic_cas(MemFlagsData::new(), ptr, expected, replacement);

        let succeeded = self.builder.ins().icmp(IntCC::Equal, old, expected);
        self.set_flag(X86Flag::Zero, succeeded)?;

        // On failure, load old value into EDX:EAX.
        let old_eax32 = self.builder.ins().ireduce(types::I32, old);
        let old_edx_shifted = self.builder.ins().ushr_imm_u(old, 32);
        let old_edx32 = self.builder.ins().ireduce(types::I32, old_edx_shifted);
        let new_eax = self.builder.ins().select(succeeded, eax, old_eax32);
        let new_edx = self.builder.ins().select(succeeded, edx, old_edx32);
        self.set_reg_val(Register::EAX, new_eax)?;
        self.set_reg_val(Register::EDX, new_edx)?;

        Ok(())
    }

    /// CMPXCHG16B [mem]
    ///
    /// Atomically compares RDX:RAX with the 128-bit value at [mem].
    /// If equal: ZF=1, [mem] <- RCX:RBX.
    /// Otherwise: ZF=0, RDX:RAX <- [mem].
    ///
    /// Only ZF is affected - CF/PF/AF/SF/OF are untouched, unlike `CMPXCHG`,
    /// which updates all of them as if a `CMP` had run.
    ///
    /// The guest's memory operand must be 16-byte aligned or the real CPU
    /// raises `#GP`; this lifter does not model alignment faults, so a
    /// misaligned operand is simply performed.
    ///
    /// How the 128-bit RMW is realized is a [`Atomic128Mode`] decision, because
    /// only the x64 Cranelift backend can express one. See that enum for the
    /// trade-offs; the default is a portable but non-atomic load/compare/store.
    pub(super) fn handle_cmpxchg16b(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if instruction.op0_kind() != OpKind::Memory {
            return Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("cmpxchg16b-dst-{:?}", instruction.op0_kind()),
            });
        }

        if self.config.atomic128 == Atomic128Mode::Exit {
            // The host performs the whole operation and resumes; the PC the
            // exit records is already past this instruction, per the ordinary
            // unimplemented-instruction contract.
            return self.handle_unimplemented_instruction(instruction);
        }

        let ptr = self.get_mem_ptr(instruction, 0)?;
        let rax = self.get_reg_val(Register::RAX)?;
        let rdx = self.get_reg_val(Register::RDX)?;
        let rbx = self.get_reg_val(Register::RBX)?;
        let rcx = self.get_reg_val(Register::RCX)?;

        let (old_lo, old_hi, succeeded) = match self.config.atomic128 {
            Atomic128Mode::Native => {
                // `iconcat` takes (low, high); the x64 backend lowers an I128
                // `atomic_cas` straight to `lock cmpxchg16b`.
                let expected = self.builder.ins().iconcat(rax, rdx);
                let replacement = self.builder.ins().iconcat(rbx, rcx);
                let old = self
                    .builder
                    .ins()
                    .atomic_cas(MemFlagsData::new(), ptr, expected, replacement);
                let succeeded = self.builder.ins().icmp(IntCC::Equal, old, expected);
                let (lo, hi) = self.builder.ins().isplit(old);
                (lo, hi, succeeded)
            }
            // Emulated: a plain 128-bit read-modify-write as two 64-bit halves.
            // The store is unconditional, which matches the architectural
            // behaviour - `CMPXCHG16B` writes the destination even when the
            // comparison fails (it writes the value back unchanged).
            _ => {
                let flags = MemFlagsData::new();
                let lo = self.builder.ins().load(types::I64, flags, ptr, 0);
                let hi = self.builder.ins().load(types::I64, flags, ptr, 8);
                let lo_eq = self.builder.ins().icmp(IntCC::Equal, lo, rax);
                let hi_eq = self.builder.ins().icmp(IntCC::Equal, hi, rdx);
                let succeeded = self.builder.ins().band(lo_eq, hi_eq);
                let new_lo = self.builder.ins().select(succeeded, rbx, lo);
                let new_hi = self.builder.ins().select(succeeded, rcx, hi);
                self.builder.ins().store(flags, new_lo, ptr, 0);
                self.builder.ins().store(flags, new_hi, ptr, 8);
                (lo, hi, succeeded)
            }
        };

        self.set_flag(X86Flag::Zero, succeeded)?;
        // On failure the observed value lands in RDX:RAX; on success they keep
        // the expected value they already held.
        let new_rax = self.builder.ins().select(succeeded, rax, old_lo);
        let new_rdx = self.builder.ins().select(succeeded, rdx, old_hi);
        self.set_reg_val(Register::RAX, new_rax)?;
        self.set_reg_val(Register::RDX, new_rdx)?;

        Ok(())
    }

    /// Dispatcher for instructions carrying a LOCK prefix.
    ///
    /// LOCK is only architecturally valid when the destination is a memory
    /// operand, so every handler below assumes op0 is Memory.
    pub(super) fn handle_locked_instruction(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<()> {
        match instruction.mnemonic() {
            Mnemonic::Add => self.handle_lock_add(instruction),
            Mnemonic::Sub => self.handle_lock_sub(instruction),
            Mnemonic::Adc => self.handle_lock_adc_sbb(instruction, true),
            Mnemonic::Sbb => self.handle_lock_adc_sbb(instruction, false),
            Mnemonic::And => self.handle_lock_and(instruction),
            Mnemonic::Or => self.handle_lock_or(instruction),
            Mnemonic::Xor => self.handle_lock_xor(instruction),
            Mnemonic::Inc => self.handle_lock_inc(instruction),
            Mnemonic::Dec => self.handle_lock_dec(instruction),
            Mnemonic::Xadd => self.handle_lock_xadd(instruction),
            Mnemonic::Xchg => self.handle_lock_xchg(instruction),
            // LOCK CMPXCHG reuses handle_cmpxchg which already uses atomic_cas.
            Mnemonic::Cmpxchg => self.handle_cmpxchg(instruction),
            Mnemonic::Cmpxchg8b => self.handle_cmpxchg8b(instruction),
            Mnemonic::Cmpxchg16b => self.handle_cmpxchg16b(instruction),
            Mnemonic::Neg => self.handle_lock_neg(instruction),
            Mnemonic::Not => self.handle_lock_not(instruction),
            Mnemonic::Bts => self.handle_lock_bts(instruction),
            Mnemonic::Btr => self.handle_lock_btr(instruction),
            Mnemonic::Btc => self.handle_lock_btc(instruction),
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("lock-{:?}", instruction.mnemonic()),
            }),
        }
    }

    fn handle_lock_add(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Add, ptr, src);
        let result = self.builder.ins().iadd(old, src);
        self.update_flags_add(old, src, result, size)?;
        Ok(())
    }

    fn handle_lock_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Sub, ptr, src);
        let result = self.builder.ins().isub(old, src);
        self.update_flags_sub(old, src, result, size)?;
        Ok(())
    }

    /// LOCK ADC / LOCK SBB [mem], src
    ///
    /// Neither has an atomic-RMW opcode - the operation depends on the incoming
    /// CF as well as on the memory value - so both lower to the same
    /// compare-and-swap retry loop `LOCK NEG` uses: observe the value, compute
    /// `dst ± src ± CF`, `atomic_cas` it back, and retry with the fresh value if
    /// another thread got there first.
    ///
    /// The incoming CF is loop-invariant and read once in the entry block, which
    /// dominates the loop, so a retry re-uses it rather than the CF a previous
    /// iteration would have computed - matching the single-execution semantics.
    fn handle_lock_adc_sbb(
        &mut self,
        instruction: &Instruction,
        is_adc: bool,
    ) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;

        let cf_i8 = self.builder.use_var(self.state.flags[&X86Flag::Carry]);
        let cf = if ty == types::I8 {
            cf_i8
        } else {
            self.builder.ins().uextend(ty, cf_i8)
        };

        // As in `handle_lock_neg`, the loop state is carried in block params so
        // no `Variable` sealing beyond the engine's final pass is needed.
        let loop_b = self.builder.create_block();
        self.builder.append_block_param(loop_b, ty); // observed current value
        let done_b = self.builder.create_block();
        self.builder.append_block_param(done_b, ty); // value that was swapped out

        let init = self.builder.ins().atomic_load(ty, MemFlagsData::new(), ptr);
        self.builder.ins().jump(loop_b, &[init.into()]);

        self.builder.switch_to_block(loop_b);
        let cur = self.builder.block_params(loop_b)[0];
        let (result, _) = self.emit_adc_sbb(cur, src, cf, is_adc);
        let observed = self
            .builder
            .ins()
            .atomic_cas(MemFlagsData::new(), ptr, cur, result);
        let swapped = self.builder.ins().icmp(IntCC::Equal, observed, cur);
        self.builder
            .ins()
            .brif(swapped, done_b, &[cur.into()], loop_b, &[observed.into()]);
        self.builder.seal_block(loop_b);

        self.builder.switch_to_block(done_b);
        self.builder.seal_block(done_b);
        let old = self.builder.block_params(done_b)[0];
        let (result, cf_out) = self.emit_adc_sbb(old, src, cf, is_adc);
        if is_adc {
            self.update_flags_adc(old, src, result, cf_out, size)
        } else {
            self.update_flags_sbb(old, src, result, cf_out, size)
        }
    }

    /// `dst + src + cf` (ADC) or `dst - src - cf` (SBB), returning the result
    /// and the carry/borrow out. Both are computed in two steps so the carry out
    /// of each half can be OR-ed together, matching the non-locked handlers.
    fn emit_adc_sbb(&mut self, dst: Value, src: Value, cf: Value, is_adc: bool) -> (Value, Value) {
        if is_adc {
            let tmp = self.builder.ins().iadd(dst, src);
            let c1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, dst);
            let result = self.builder.ins().iadd(tmp, cf);
            let c2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, result, tmp);
            (result, self.builder.ins().bor(c1, c2))
        } else {
            let tmp = self.builder.ins().isub(dst, src);
            let b1 = self.builder.ins().icmp(IntCC::UnsignedLessThan, dst, src);
            let result = self.builder.ins().isub(tmp, cf);
            let b2 = self.builder.ins().icmp(IntCC::UnsignedLessThan, tmp, cf);
            (result, self.builder.ins().bor(b1, b2))
        }
    }

    fn handle_lock_and(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::And, ptr, src);
        let result = self.builder.ins().band(old, src);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_lock_or(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Or, ptr, src);
        let result = self.builder.ins().bor(old, src);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_lock_xor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Xor, ptr, src);
        let result = self.builder.ins().bxor(old, src);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_lock_inc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let one = self.builder.ins().iconst(ty, 1);
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Add, ptr, one);
        let result = self.builder.ins().iadd(old, one);
        self.update_flags_inc_dec(old, one, result, true, size)?;
        Ok(())
    }

    fn handle_lock_dec(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let one = self.builder.ins().iconst(ty, 1);
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Sub, ptr, one);
        let result = self.builder.ins().isub(old, one);
        self.update_flags_inc_dec(old, one, result, false, size)?;
        Ok(())
    }

    fn handle_lock_xadd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src_reg = instruction.op1_register();
        let src_val = self.get_reg_val(src_reg)?;
        // Narrow to the memory operand size if the register is wider.
        let src = if self.builder.func.dfg.value_type(src_val) != ty {
            self.builder.ins().ireduce(ty, src_val)
        } else {
            src_val
        };
        // atomic fetch-add: returns old *ptr, then *ptr += src
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Add, ptr, src);
        // src register gets the pre-add memory value
        self.set_reg_val(src_reg, old)?;
        let result = self.builder.ins().iadd(old, src);
        self.update_flags_add(old, src, result, size)?;
        Ok(())
    }

    pub(super) fn handle_lock_xchg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);

        // Determine which operand is memory (LOCK XCHG can encode either way).
        let (mem_op, reg_op) = if instruction.op0_kind() == OpKind::Memory {
            (0u32, 1u32)
        } else {
            (1u32, 0u32)
        };

        let ptr = self.get_mem_ptr(instruction, mem_op)?;
        let reg = self.operand_register(instruction, reg_op);
        let reg_val = self.get_reg_val(reg)?;
        let reg_narrow = if self.builder.func.dfg.value_type(reg_val) != ty {
            self.builder.ins().ireduce(ty, reg_val)
        } else {
            reg_val
        };
        // atomic swap: returns old *ptr, writes reg_narrow to *ptr
        let old = self.builder.ins().atomic_rmw(
            ty,
            MemFlagsData::new(),
            AtomicRmwOp::Xchg,
            ptr,
            reg_narrow,
        );
        self.set_reg_val(reg, old)?;
        // XCHG does not modify flags.
        Ok(())
    }

    /// LOCK NEG [mem]: atomically negate memory.
    ///
    /// `NEG` has no direct atomic-RMW opcode, so it is lowered as a
    /// compare-and-swap retry loop: load the current value, compute `0 - value`,
    /// and `atomic_cas` it back; on contention (the observed value changed) the
    /// loop retries with the fresh value. The flag update uses the value that
    /// was actually swapped, matching `NEG`'s "flags as if `0 - dst`" semantics.
    fn handle_lock_neg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;

        // Blocks carry the loop state as explicit parameters (no `Variable`s), so
        // they need no manual sealing beyond the engine's final `seal_all_blocks`.
        let loop_b = self.builder.create_block();
        self.builder.append_block_param(loop_b, ty); // observed current value
        let done_b = self.builder.create_block();
        self.builder.append_block_param(done_b, ty); // value that was swapped out

        let init = self.builder.ins().atomic_load(ty, MemFlagsData::new(), ptr);
        self.builder.ins().jump(loop_b, &[init.into()]);

        self.builder.switch_to_block(loop_b);
        let cur = self.builder.block_params(loop_b)[0];
        let zero = self.builder.ins().iconst(ty, 0);
        let result = self.builder.ins().isub(zero, cur);
        let observed = self
            .builder
            .ins()
            .atomic_cas(MemFlagsData::new(), ptr, cur, result);
        let swapped = self.builder.ins().icmp(IntCC::Equal, observed, cur);
        self.builder
            .ins()
            .brif(swapped, done_b, &[cur.into()], loop_b, &[observed.into()]);
        self.builder.seal_block(loop_b);

        self.builder.switch_to_block(done_b);
        self.builder.seal_block(done_b);
        let old = self.builder.block_params(done_b)[0];
        let zero = self.builder.ins().iconst(ty, 0);
        let result = self.builder.ins().isub(zero, old);
        self.update_flags_sub(zero, old, result, size)?;
        Ok(())
    }

    /// LOCK NOT [mem]: atomically bitwise-NOT memory.
    ///
    /// `~x == x ^ -1`, so this fuses into a single atomic XOR-with-all-ones.
    /// `NOT` affects no flags, so the returned old value is discarded.
    fn handle_lock_not(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let all_ones = self.builder.ins().iconst(ty, -1);
        self.builder
            .ins()
            .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Xor, ptr, all_ones);
        Ok(())
    }

    /// LOCK BTS [mem], bit - atomically set a bit, CF = original bit.
    fn handle_lock_bts(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_lock_bit_op(instruction, BitOp::Set)
    }

    /// LOCK BTR [mem], bit - atomically clear a bit, CF = original bit.
    fn handle_lock_btr(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_lock_bit_op(instruction, BitOp::Reset)
    }

    /// LOCK BTC [mem], bit - atomically complement a bit, CF = original bit.
    fn handle_lock_btc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_lock_bit_op(instruction, BitOp::Complement)
    }

    fn emit_lock_bit_op(&mut self, instruction: &Instruction, op: BitOp) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let bits = (size * 8) as i64;
        let mut ptr = self.get_mem_ptr(instruction, 0)?;

        // A register bit index selects a bit *string* position: the word
        // actually accessed is `EA + (signed_index div width) * size`, and the
        // in-word bit is `index mod width` (matching the non-locked BT family).
        // An immediate index is simply masked to the operand width.
        let shift = match instruction.op1_kind() {
            OpKind::Register => {
                let idx = self.get_reg_val(instruction.op1_register())?;
                let idx64 = if size == 8 {
                    idx
                } else {
                    self.builder.ins().sextend(types::I64, idx)
                };
                let word = self
                    .builder
                    .ins()
                    .sshr_imm_u(idx64, bits.trailing_zeros() as i64);
                let byte_off = self.builder.ins().imul_imm_s(word, size as i64);
                ptr = self.builder.ins().iadd(ptr, byte_off);
                self.builder.ins().band_imm_u(idx, bits - 1)
            }
            OpKind::Immediate8 => self
                .builder
                .ins()
                .iconst(ty, (instruction.immediate8() as i64) & (bits - 1)),
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("lock-bt-idx-{:?}", other),
                });
            }
        };

        // mask = 1 << (bit_idx % bit_width)
        let one = self.builder.ins().iconst(ty, 1);
        let mask = self.builder.ins().ishl(one, shift);

        let old = match op {
            BitOp::Set => {
                self.builder
                    .ins()
                    .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Or, ptr, mask)
            }
            BitOp::Reset => {
                let all_ones = self.builder.ins().iconst(ty, -1);
                let inv_mask = self.builder.ins().bxor(mask, all_ones);
                self.builder.ins().atomic_rmw(
                    ty,
                    MemFlagsData::new(),
                    AtomicRmwOp::And,
                    ptr,
                    inv_mask,
                )
            }
            BitOp::Complement => {
                self.builder
                    .ins()
                    .atomic_rmw(ty, MemFlagsData::new(), AtomicRmwOp::Xor, ptr, mask)
            }
        };

        // CF = original bit value at position `shift`
        let bit_val = self.builder.ins().band(old, mask);
        let cf_wide = self.builder.ins().ushr(bit_val, shift);
        let cf = self.builder.ins().ireduce(types::I8, cf_wide);
        self.set_flag(X86Flag::Carry, cf)?;
        Ok(())
    }
}
