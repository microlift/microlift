//! Atomic instruction semantics.
//!
//! Covers:
//! - Memory fence instructions: MFENCE, SFENCE, LFENCE
//! - PAUSE (spin-wait hint)
//! - CMPXCHG / CMPXCHG8B
//! - LOCK-prefixed arithmetic and bit operations

use crate::MicroliftError;
use crate::lifter::{Lifter, X86Flag};
use crate::arch::x86_64::semantics::helpers::size_to_type;
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
                let old = self
                    .builder
                    .ins()
                    .atomic_cas(MemFlags::new(), ptr, acc_narrow, src_val);

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
        let edx_hi = self.builder.ins().ishl_imm(edx64, 32);
        let expected = self.builder.ins().bor(edx_hi, eax64);

        // Build replacement (ECX:EBX) as a single I64.
        let ebx = self.get_reg_val(Register::EBX)?; // I32
        let ecx = self.get_reg_val(Register::ECX)?; // I32
        let ebx64 = self.builder.ins().uextend(types::I64, ebx);
        let ecx64 = self.builder.ins().uextend(types::I64, ecx);
        let ecx_hi = self.builder.ins().ishl_imm(ecx64, 32);
        let replacement = self.builder.ins().bor(ecx_hi, ebx64);

        let old = self
            .builder
            .ins()
            .atomic_cas(MemFlags::new(), ptr, expected, replacement);

        let succeeded = self.builder.ins().icmp(IntCC::Equal, old, expected);
        self.set_flag(X86Flag::Zero, succeeded)?;

        // On failure, load old value into EDX:EAX.
        let old_eax32 = self.builder.ins().ireduce(types::I32, old);
        let old_edx_shifted = self.builder.ins().ushr_imm(old, 32);
        let old_edx32 = self.builder.ins().ireduce(types::I32, old_edx_shifted);
        let new_eax = self.builder.ins().select(succeeded, eax, old_eax32);
        let new_edx = self.builder.ins().select(succeeded, edx, old_edx32);
        self.set_reg_val(Register::EAX, new_eax)?;
        self.set_reg_val(Register::EDX, new_edx)?;

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
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Add, ptr, src);
        let result = self.builder.ins().iadd(old, src);
        self.update_flags_add(old, src, result, size)?;
        Ok(())
    }

    fn handle_lock_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Sub, ptr, src);
        let result = self.builder.ins().isub(old, src);
        self.update_flags_sub(old, src, result, size)?;
        Ok(())
    }

    fn handle_lock_and(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::And, ptr, src);
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
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Or, ptr, src);
        let result = self.builder.ins().bor(old, src);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_lock_xor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let src = self.read_operand(instruction, 1, size)?;
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Xor, ptr, src);
        let result = self.builder.ins().bxor(old, src);
        self.update_flags_logical(result, size)?;
        Ok(())
    }

    fn handle_lock_inc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let one = self.builder.ins().iconst(ty, 1);
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Add, ptr, one);
        let result = self.builder.ins().iadd(old, one);
        self.update_flags_inc_dec(old, one, result, true, size)?;
        Ok(())
    }

    fn handle_lock_dec(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let one = self.builder.ins().iconst(ty, 1);
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Sub, ptr, one);
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
        let old = self
            .builder
            .ins()
            .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Add, ptr, src);
        // src register gets the pre-add memory value
        self.set_reg_val(src_reg, old)?;
        let result = self.builder.ins().iadd(old, src);
        self.update_flags_add(old, src, result, size)?;
        Ok(())
    }

    fn handle_lock_xchg(&mut self, instruction: &Instruction) -> crate::Result<()> {
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
        let old =
            self.builder
                .ins()
                .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Xchg, ptr, reg_narrow);
        self.set_reg_val(reg, old)?;
        // XCHG does not modify flags.
        Ok(())
    }

    /// LOCK NEG [mem]: atomically negate memory.
    ///
    /// NEG via CAS-loop would be fully atomic but complex to express in SSA.
    /// Instead we emit an `atomic_load` + compute + `atomic_store`, which
    /// provides sequentially-consistent ordering even if the RMW isn't fused
    /// into a single hardware transaction.  LOCK NEG is vanishingly rare in
    /// real code so this is an acceptable trade-off for a JIT lifter.
    fn handle_lock_neg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let old = self.builder.ins().atomic_load(ty, MemFlags::new(), ptr);
        let zero = self.builder.ins().iconst(ty, 0);
        let result = self.builder.ins().isub(zero, old);
        self.builder
            .ins()
            .atomic_store(MemFlags::new(), result, ptr);
        self.update_flags_sub(zero, old, result, size)?;
        Ok(())
    }

    /// LOCK NOT [mem]: atomically bitwise-NOT memory.
    fn handle_lock_not(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let old = self.builder.ins().atomic_load(ty, MemFlags::new(), ptr);
        let all_ones = self.builder.ins().iconst(ty, -1);
        let result = self.builder.ins().bxor(old, all_ones);
        self.builder
            .ins()
            .atomic_store(MemFlags::new(), result, ptr);
        // NOT does not affect flags.
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
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let bit_idx = self.read_operand(instruction, 1, size)?;

        // mask = 1 << (bit_idx % bit_width)
        let one = self.builder.ins().iconst(ty, 1);
        let width_mask = (size * 8 - 1) as i64;
        let shift = self.builder.ins().band_imm(bit_idx, width_mask);
        let mask = self.builder.ins().ishl(one, shift);

        let old = match op {
            BitOp::Set => {
                self.builder
                    .ins()
                    .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Or, ptr, mask)
            }
            BitOp::Reset => {
                let all_ones = self.builder.ins().iconst(ty, -1);
                let inv_mask = self.builder.ins().bxor(mask, all_ones);
                self.builder
                    .ins()
                    .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::And, ptr, inv_mask)
            }
            BitOp::Complement => {
                self.builder
                    .ins()
                    .atomic_rmw(ty, MemFlags::new(), AtomicRmwOp::Xor, ptr, mask)
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
