//! Semantics layer - maps x86-64 instructions to Cranelift IR.
//!
//! This module is the dispatch hub. The actual implementations live in the
//! sub-modules below, each focused on a specific instruction category.

pub mod arithmetic;
pub mod bitops;
pub mod control_flow;
pub mod flags;
pub mod helpers;
pub mod memory;
pub mod sse;
pub mod syscall;
pub mod unimplemented;

use crate::MicroliftError;
use crate::lifter::Lifter;
use iced_x86::{Instruction, Mnemonic, OpKind};

impl<'a> Lifter<'a> {
    pub fn handle_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.mnemonic() {
            // Memory / Data Movement
            Mnemonic::Mov => self.handle_mov(instruction),
            Mnemonic::Movzx => self.handle_movzx(instruction),
            Mnemonic::Movsx => self.handle_movsx(instruction),
            Mnemonic::Movsxd => self.handle_movsxd(instruction),
            Mnemonic::Lea => self.handle_lea(instruction),
            Mnemonic::Push => self.handle_push(instruction),
            Mnemonic::Pop => self.handle_pop(instruction),
            Mnemonic::Xchg => self.handle_xchg(instruction),
            Mnemonic::Nop => self.handle_nop(instruction),

            // CET
            Mnemonic::Endbr64 | Mnemonic::Endbr32 => self.handle_endbr64(instruction),

            // SSE / SSE2 — XMM data movement
            Mnemonic::Movaps | Mnemonic::Movapd | Mnemonic::Movups | Mnemonic::Movupd
            | Mnemonic::Movdqa | Mnemonic::Movdqu => self.handle_movaps(instruction),
            Mnemonic::Movq => self.handle_movq_sse(instruction),
            Mnemonic::Movss => self.handle_movss(instruction),

            // SSE / SSE2 — XMM logic
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
                if instruction.op0_kind() == OpKind::MemoryESDI {
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
            Mnemonic::Sub => self.handle_sub(instruction),
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

            // System
            Mnemonic::Syscall => self.handle_syscall(instruction),

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
