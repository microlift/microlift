//! Semantics layer - maps x86-64 instructions to Cranelift IR.
//!
//! This module is the dispatch hub. The actual implementations live in the
//! sub-modules below, each focused on a specific instruction category.

pub mod arithmetic;
pub mod atomics;
pub mod bitops;
pub mod control_flow;
pub mod flags;
pub mod helpers;
pub mod memory;
pub mod setcc;
pub mod sse;
pub mod syscall;
pub mod unimplemented;

use crate::MicroliftError;
use crate::lifter::Lifter;
use iced_x86::{Instruction, Mnemonic, OpKind};

impl<'a> Lifter<'a> {
    pub fn handle_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        // LOCK-prefixed instructions require atomic semantics; dispatch them
        // before the main match so that ADD, SUB, etc. only see non-locked ops.
        if instruction.has_lock_prefix() {
            return self.handle_locked_instruction(instruction);
        }

        match instruction.mnemonic() {
            // Memory / Data Movement
            Mnemonic::Mov => self.handle_mov(instruction),
            Mnemonic::Movzx => self.handle_movzx(instruction),
            Mnemonic::Movsx => self.handle_movsx(instruction),
            Mnemonic::Movsxd => self.handle_movsxd(instruction),
            Mnemonic::Lea => self.handle_lea(instruction),
            Mnemonic::Push => self.handle_push(instruction),
            Mnemonic::Pop => self.handle_pop(instruction),
            Mnemonic::Leave => self.handle_leave(instruction),
            Mnemonic::Xchg => self.handle_xchg(instruction),
            Mnemonic::Nop | Mnemonic::Reservednop => self.handle_nop(instruction),

            // Atomics / synchronisation
            Mnemonic::Cmpxchg => self.handle_cmpxchg(instruction),
            Mnemonic::Cmpxchg8b => self.handle_cmpxchg8b(instruction),
            Mnemonic::Mfence | Mnemonic::Sfence | Mnemonic::Lfence => {
                self.handle_mfence(instruction)
            }
            Mnemonic::Pause => self.handle_pause(instruction),

            // CET
            Mnemonic::Endbr64 | Mnemonic::Endbr32 => self.handle_endbr64(instruction),

            // SSE / SSE2 - XMM data movement
            Mnemonic::Movaps
            | Mnemonic::Movapd
            | Mnemonic::Movups
            | Mnemonic::Movupd
            | Mnemonic::Movdqa
            | Mnemonic::Movdqu => self.handle_movaps(instruction),
            Mnemonic::Movq => self.handle_movq_sse(instruction),
            Mnemonic::Movss => self.handle_movss(instruction),

            // SSE / SSE2 - XMM logic
            Mnemonic::Xorps | Mnemonic::Xorpd => self.handle_xorps(instruction),
            Mnemonic::Pxor => self.handle_pxor(instruction),
            Mnemonic::Andps | Mnemonic::Andpd | Mnemonic::Pand => self.handle_andps(instruction),
            Mnemonic::Orps | Mnemonic::Orpd | Mnemonic::Por => self.handle_orps(instruction),
            Mnemonic::Andnps | Mnemonic::Andnpd | Mnemonic::Pandn => {
                self.handle_andnps(instruction)
            }

            // String Operations
            Mnemonic::Movsb => self.handle_movsb(instruction),
            Mnemonic::Movsw => self.handle_movsw(instruction),
            // MOVSD is shared between the string op (no register operands) and
            // SSE2 scalar-double (XMM register operands). Disambiguate by
            // checking whether op0 is a memory-string kind.
            Mnemonic::Movsd => {
                if matches!(instruction.op0_kind(), OpKind::MemoryESDI | OpKind::MemoryESRDI) {
                    self.handle_movsd_mem(instruction)
                } else {
                    self.handle_movsd_sse(instruction)
                }
            }
            Mnemonic::Movsq => self.handle_movsq(instruction),
            Mnemonic::Stosb => self.handle_stosb(instruction),
            Mnemonic::Stosw => self.handle_stosw(instruction),
            Mnemonic::Stosd => self.handle_stosd(instruction),
            Mnemonic::Stosq => self.handle_stosq(instruction),
            Mnemonic::Lodsb => self.handle_lodsb(instruction),
            Mnemonic::Lodsw => self.handle_lodsw(instruction),
            Mnemonic::Lodsd => self.handle_lodsd(instruction),
            Mnemonic::Lodsq => self.handle_lodsq(instruction),

            // Arithmetic
            Mnemonic::Add => self.handle_add(instruction),
            Mnemonic::Adc => self.handle_adc(instruction),
            Mnemonic::Sub => self.handle_sub(instruction),
            Mnemonic::Sbb => self.handle_sbb(instruction),
            // Sign-extension conversions (implicit operand form)
            Mnemonic::Cbw => self.handle_cbw(instruction),
            Mnemonic::Cwde => self.handle_cwde(instruction),
            Mnemonic::Cdqe => self.handle_cdqe(instruction),
            Mnemonic::Cwd => self.handle_cwd(instruction),
            Mnemonic::Cdq => self.handle_cdq(instruction),
            Mnemonic::Cqo => self.handle_cqo(instruction),
            Mnemonic::Xadd => self.handle_xadd(instruction),
            Mnemonic::Cmp => self.handle_cmp(instruction),
            Mnemonic::Inc => self.handle_inc(instruction),
            Mnemonic::Dec => self.handle_dec(instruction),
            Mnemonic::Neg => self.handle_neg(instruction),
            Mnemonic::Imul => self.handle_imul(instruction),
            Mnemonic::Mul => self.handle_mul(instruction),
            Mnemonic::Idiv => self.handle_idiv(instruction),
            Mnemonic::Div => self.handle_div(instruction),

            // Bitwise / Logical
            Mnemonic::And => self.handle_and(instruction),
            Mnemonic::Or => self.handle_or(instruction),
            Mnemonic::Xor => self.handle_xor(instruction),
            Mnemonic::Not => self.handle_not(instruction),
            Mnemonic::Test => self.handle_test(instruction),

            // Shifts / Rotates
            Mnemonic::Shl => self.handle_shl(instruction),
            Mnemonic::Shr => self.handle_shr(instruction),
            Mnemonic::Sar => self.handle_sar(instruction),
            Mnemonic::Rol => self.handle_rol(instruction),
            Mnemonic::Ror => self.handle_ror(instruction),
            Mnemonic::Shld => self.handle_shld(instruction),
            Mnemonic::Shrd => self.handle_shrd(instruction),

            // Bit Manipulation
            Mnemonic::Bt => self.handle_bt(instruction),
            Mnemonic::Bts => self.handle_bts(instruction),
            Mnemonic::Btr => self.handle_btr(instruction),
            Mnemonic::Btc => self.handle_btc(instruction),
            Mnemonic::Bsf => self.handle_bsf(instruction),
            Mnemonic::Bsr => self.handle_bsr(instruction),
            Mnemonic::Bswap => self.handle_bswap(instruction),

            // Unconditional Control Flow
            Mnemonic::Jmp => self.handle_jmp(instruction),
            Mnemonic::Call => self.handle_call(instruction),
            Mnemonic::Ret => self.handle_ret(instruction),
            Mnemonic::Loop => self.handle_loop(instruction),

            // Conditional Jumps (unsigned)
            Mnemonic::Jb => self.handle_jb(instruction), // JB / JNAE / JC
            Mnemonic::Jbe => self.handle_jbe(instruction), // JBE / JNA
            Mnemonic::Ja => self.handle_ja(instruction), // JA / JNBE
            Mnemonic::Jae => self.handle_jae(instruction), // JAE / JNB / JNC

            // Conditional Jumps (equality)
            Mnemonic::Je => self.handle_jz(instruction), // JE / JZ
            Mnemonic::Jne => self.handle_jnz(instruction), // JNE / JNZ

            // Conditional Jumps (signed)
            Mnemonic::Jg => self.handle_jg(instruction), // JG / JNLE
            Mnemonic::Jge => self.handle_jge(instruction), // JGE / JNL
            Mnemonic::Jl => self.handle_jl(instruction), // JL / JNGE
            Mnemonic::Jle => self.handle_jle(instruction), // JLE / JNG

            // Conditional Jumps (flag-based)
            Mnemonic::Js => self.handle_js(instruction),
            Mnemonic::Jns => self.handle_jns(instruction),
            Mnemonic::Jo => self.handle_jo(instruction),
            Mnemonic::Jno => self.handle_jno(instruction),
            Mnemonic::Jp => self.handle_jp(instruction),
            Mnemonic::Jnp => self.handle_jnp(instruction),
            Mnemonic::Jrcxz => self.handle_jrcxz(instruction),

            // SETcc - set byte on condition
            Mnemonic::Sete => self.handle_sete(instruction),
            Mnemonic::Setne => self.handle_setne(instruction),
            Mnemonic::Setb => self.handle_setb(instruction),
            Mnemonic::Setae => self.handle_setae(instruction),
            Mnemonic::Setbe => self.handle_setbe(instruction),
            Mnemonic::Seta => self.handle_seta(instruction),
            Mnemonic::Setl => self.handle_setl(instruction),
            Mnemonic::Setge => self.handle_setge(instruction),
            Mnemonic::Setle => self.handle_setle(instruction),
            Mnemonic::Setg => self.handle_setg(instruction),
            Mnemonic::Sets => self.handle_sets(instruction),
            Mnemonic::Setns => self.handle_setns(instruction),
            Mnemonic::Seto => self.handle_seto(instruction),
            Mnemonic::Setno => self.handle_setno(instruction),
            Mnemonic::Setp => self.handle_setp(instruction),
            Mnemonic::Setnp => self.handle_setnp(instruction),

            // Conditional Move (CMOVcc)
            Mnemonic::Cmove => self.handle_cmove(instruction),
            Mnemonic::Cmovne => self.handle_cmovne(instruction),
            Mnemonic::Cmovb => self.handle_cmovb(instruction),
            Mnemonic::Cmovae => self.handle_cmovae(instruction),
            Mnemonic::Cmovbe => self.handle_cmovbe(instruction),
            Mnemonic::Cmova => self.handle_cmova(instruction),
            Mnemonic::Cmovl => self.handle_cmovl(instruction),
            Mnemonic::Cmovge => self.handle_cmovge(instruction),
            Mnemonic::Cmovle => self.handle_cmovle(instruction),
            Mnemonic::Cmovg => self.handle_cmovg(instruction),
            Mnemonic::Cmovs => self.handle_cmovs(instruction),
            Mnemonic::Cmovns => self.handle_cmovns(instruction),
            Mnemonic::Cmovo => self.handle_cmovo(instruction),
            Mnemonic::Cmovno => self.handle_cmovno(instruction),
            Mnemonic::Cmovp => self.handle_cmovp(instruction),
            Mnemonic::Cmovnp => self.handle_cmovnp(instruction),

            // Traps / faults
            Mnemonic::Ud2 => self.handle_ud2(instruction),
            Mnemonic::Int3 => self.handle_int3(instruction),

            // System
            Mnemonic::Syscall => self.handle_syscall(instruction),

            // FS/GS base (FSGSBASE extension)
            Mnemonic::Rdfsbase => self.handle_rdfsbase(instruction),
            Mnemonic::Wrfsbase => self.handle_wrfsbase(instruction),
            Mnemonic::Rdgsbase => self.handle_rdgsbase(instruction),
            Mnemonic::Wrgsbase => self.handle_wrgsbase(instruction),

            Mnemonic::Stc => self.handle_stc(instruction),
            Mnemonic::Clc => self.handle_clc(instruction),
            Mnemonic::Cmc => self.handle_cmc(instruction),

            _ => {
                if self.config.interpreter_fallback {
                    self.handle_unimplemented_instruction(instruction)
                } else {
                    Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: format!("{:?}", instruction.mnemonic()),
                    })
                }
            }
        }
    }
}
