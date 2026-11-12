//! Semantics layer: maps x86-64 instructions to Cranelift IR.
//!
//! Dispatch hub; implementations live in the category sub-modules.

pub mod aes;
pub mod arith_spec;
pub mod arithmetic;
pub mod atomics;
pub mod avx;
pub mod bitops;
pub mod bridge;
pub mod clmul;
pub mod cond;
pub mod control_flow;
pub mod crc32;
pub mod flags;
pub mod fpu;
pub mod helpers;
pub mod memory;
pub mod softfloat;
pub mod sse;
pub mod sse_int;
pub mod sse_spec;
pub mod sse_str;
pub mod sse_vec;
pub mod syscall;
pub mod unimplemented;

use crate::MicroliftError;
use crate::arch::x86_64::semantics::aes::AesKernel;
use crate::arch::x86_64::semantics::cond::Cond;
use crate::arch::x86_64::semantics::fpu::FpuOp;
use crate::arch::x86_64::semantics::sse::{PackedLaneOp, PackedShiftOp, PackedWidth};
use crate::arch::x86_64::semantics::sse_int::HorizOp;
use crate::arch::x86_64::semantics::sse_str::PcmpstrForm;
use crate::arch::x86_64::vex::{VexLowering, legacy_form, vex_lowering};
use crate::lifter::{Atomic128Mode, Lifter};
use iced_x86::{Instruction, Mnemonic, OpKind, Register};

/// Lifts one instruction.
pub type Semantics<'a> = fn(&mut Lifter<'a>, &Instruction) -> crate::Result<()>;

/// How an instruction is realized, decided without emitting IR.
pub enum Lowering<'a> {
    Lift(Semantics<'a>),
    /// Run by the host's interpreter callback ([`crate::core::callback`]).
    Interpret,
    Unsupported,
}

impl<'a> Lifter<'a> {
    pub fn handle_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.lowering(instruction) {
            Lowering::Lift(lift) => lift(self, instruction),
            Lowering::Interpret => self.handle_unimplemented_instruction(instruction),
            Lowering::Unsupported => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{:?}", instruction.mnemonic()),
            }),
        }
    }

    pub fn lowering(&self, instruction: &Instruction) -> Lowering<'a> {
        if self.interprets_cmpxchg16b(instruction) {
            return Lowering::Interpret;
        }

        // LOCK-prefixed instructions dispatch first so ADD, SUB, etc. only see unlocked forms.
        if instruction.has_lock_prefix() {
            return Lowering::Lift(Self::handle_locked_instruction);
        }

        // MMX encodings alias the unmodelled x87 stack.
        if uses_mmx_register(instruction) {
            return Lowering::Interpret;
        }

        if let Some(vex) = vex_lowering(instruction) {
            return self.vex_instruction_lowering(instruction, vex);
        }

        // A VEX spelling this frontend does not decode: interpret rather than lift wrongly.
        if is_pcmpstr(instruction.mnemonic()) && PcmpstrForm::of(instruction).is_none() {
            return Lowering::Interpret;
        }

        match semantics(instruction) {
            Some(lift) => Lowering::Lift(lift),
            None if self.config.interpreter_fallback => Lowering::Interpret,
            None => Lowering::Unsupported,
        }
    }

    /// A VEX instruction is interpreted whole when its legacy form would be.
    fn vex_instruction_lowering(
        &self,
        instruction: &Instruction,
        vex: VexLowering,
    ) -> Lowering<'a> {
        let lift = Lowering::Lift(|lifter, instruction| {
            let vex = vex_lowering(instruction).expect("a VEX lowering");
            lifter.handle_vex(instruction, vex)
        });
        let Some(legacy) = legacy_form(instruction, vex) else {
            return lift;
        };
        match self.lowering(&legacy) {
            Lowering::Lift(_) => lift,
            other => other,
        }
    }

    fn interprets_cmpxchg16b(&self, instruction: &Instruction) -> bool {
        instruction.mnemonic() == Mnemonic::Cmpxchg16b
            && instruction.op0_kind() == OpKind::Memory
            && self.config.atomic128 == Atomic128Mode::Interpret
    }
}

fn is_pcmpstr(mnemonic: Mnemonic) -> bool {
    matches!(
        mnemonic,
        Mnemonic::Pcmpistri
            | Mnemonic::Pcmpistrm
            | Mnemonic::Pcmpestri
            | Mnemonic::Pcmpestrm
            | Mnemonic::Pcmpestri64
            | Mnemonic::Pcmpestrm64
    )
}

/// The lift for an unprefixed, non-VEX `instruction`, `None` when it has none.
fn semantics<'a>(instruction: &Instruction) -> Option<Semantics<'a>> {
    let semantics: Semantics<'a> = match instruction.mnemonic() {
        // Memory / Data Movement
        Mnemonic::Mov => |this, i| this.handle_mov(i),
        Mnemonic::Movzx => |this, i| this.handle_movzx(i),
        Mnemonic::Movsx => |this, i| this.handle_movsx(i),
        Mnemonic::Movsxd => |this, i| this.handle_movsxd(i),
        Mnemonic::Lea => |this, i| this.handle_lea(i),
        Mnemonic::Push => |this, i| this.handle_push(i),
        Mnemonic::Pop => |this, i| this.handle_pop(i),
        Mnemonic::Pushfq => |this, i| this.handle_pushfq(i),
        Mnemonic::Popfq => |this, i| this.handle_popfq(i),
        Mnemonic::Lahf => |this, i| this.handle_lahf(i),
        Mnemonic::Sahf => |this, i| this.handle_sahf(i),
        Mnemonic::Leave => |this, i| this.handle_leave(i),
        Mnemonic::Xchg => |this, i| this.handle_xchg(i),
        Mnemonic::Nop | Mnemonic::Reservednop => |this, i| this.handle_nop(i),

        // Atomics / synchronisation
        Mnemonic::Cmpxchg => |this, i| this.handle_cmpxchg(i),
        Mnemonic::Cmpxchg8b => |this, i| this.handle_cmpxchg8b(i),
        Mnemonic::Cmpxchg16b => |this, i| this.handle_cmpxchg16b(i),
        Mnemonic::Mfence | Mnemonic::Sfence | Mnemonic::Lfence | Mnemonic::Serialize => |this, i| {
            this.handle_mfence(i)
        },
        Mnemonic::Pause => |this, i| this.handle_pause(i),
        // Cache flushes: nothing to do under a coherent view; ordering comes from fences.
        Mnemonic::Clflush | Mnemonic::Clflushopt | Mnemonic::Clwb => |this, i| {
            this.handle_nop(i)
        },
        // Prefetch hints: no architectural effect.
        Mnemonic::Prefetch
        | Mnemonic::Prefetcht0
        | Mnemonic::Prefetcht1
        | Mnemonic::Prefetcht2
        | Mnemonic::Prefetchnta
        | Mnemonic::Prefetchw
        | Mnemonic::Prefetchwt1
        | Mnemonic::Prefetchit0
        | Mnemonic::Prefetchit1 => |this, i| this.handle_nop(i),

        // CET
        Mnemonic::Endbr64 | Mnemonic::Endbr32 => |this, i| this.handle_endbr64(i),

        // SSE / SSE2 - XMM data movement
        Mnemonic::Movaps
        | Mnemonic::Movapd
        | Mnemonic::Movups
        | Mnemonic::Movupd
        | Mnemonic::Movdqa
        | Mnemonic::Movdqu
        // Non-temporal stores differ only in cache hint, which is not modelled.
        | Mnemonic::Movntps
        | Mnemonic::Movntpd
        | Mnemonic::Movntdq
        // LDDQU and MOVNTDQA differ from a 128-bit load only in cache hints.
        | Mnemonic::Lddqu
        | Mnemonic::Movntdqa => |this, i| this.handle_movaps(i),
        Mnemonic::Movq => |this, i| this.handle_movq_sse(i),
        Mnemonic::Movss => |this, i| this.handle_movss(i),
        Mnemonic::Movd => |this, i| this.handle_movd(i),
        Mnemonic::Movhps | Mnemonic::Movhpd => |this, i| this.handle_movhps(i),
        Mnemonic::Movlps | Mnemonic::Movlpd => |this, i| this.handle_movlps(i),
        Mnemonic::Movhlps => |this, i| this.handle_movhlps(i),
        Mnemonic::Movlhps => |this, i| this.handle_movlhps(i),
        Mnemonic::Movddup => |this, i| this.handle_movddup(i),
        // SSE2 - packed per-lane shifts
        Mnemonic::Psrlw => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Srl, PackedWidth::Word)
        },
        Mnemonic::Psrld => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Srl, PackedWidth::Dword)
        },
        Mnemonic::Psrlq => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Srl, PackedWidth::Qword)
        },
        Mnemonic::Psllw => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Sll, PackedWidth::Word)
        },
        Mnemonic::Pslld => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Sll, PackedWidth::Dword)
        },
        Mnemonic::Psllq => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Sll, PackedWidth::Qword)
        },
        Mnemonic::Psraw => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Sra, PackedWidth::Word)
        },
        Mnemonic::Psrad => |this, i| {
            this.handle_packed_shift(i, PackedShiftOp::Sra, PackedWidth::Dword)
        },
        Mnemonic::Pshufd => |this, i| this.handle_pshufd(i),
        // SSE2 - whole-register byte shifts
        Mnemonic::Pslldq => |this, i| this.handle_byte_shift_dq(i, true),
        Mnemonic::Psrldq => |this, i| this.handle_byte_shift_dq(i, false),

        // SSE / SSE2 - unpack / interleave; the `pd`/`ps` forms alias the integer ones.
        Mnemonic::Punpcklbw => |this, i| {
            this.handle_punpck(i, "punpcklbw", PackedWidth::Byte, false)
        },
        Mnemonic::Punpcklwd => |this, i| {
            this.handle_punpck(i, "punpcklwd", PackedWidth::Word, false)
        },
        Mnemonic::Punpckldq | Mnemonic::Unpcklps => |this, i| {
            this.handle_punpck(i, "punpckldq", PackedWidth::Dword, false)
        },
        Mnemonic::Punpcklqdq | Mnemonic::Unpcklpd => |this, i| {
            this.handle_punpck(i, "punpcklqdq", PackedWidth::Qword, false)
        },
        Mnemonic::Punpckhbw => |this, i| {
            this.handle_punpck(i, "punpckhbw", PackedWidth::Byte, true)
        },
        Mnemonic::Punpckhwd => |this, i| {
            this.handle_punpck(i, "punpckhwd", PackedWidth::Word, true)
        },
        Mnemonic::Punpckhdq | Mnemonic::Unpckhps => |this, i| {
            this.handle_punpck(i, "punpckhdq", PackedWidth::Dword, true)
        },
        Mnemonic::Punpckhqdq | Mnemonic::Unpckhpd => |this, i| {
            this.handle_punpck(i, "punpckhqdq", PackedWidth::Qword, true)
        },

        // SSE3 - horizontal arithmetic
        Mnemonic::Haddpd => |this, i| this.handle_haddpd(i),
        Mnemonic::Hsubpd => |this, i| this.handle_hsubpd(i),
        Mnemonic::Haddps => |this, i| this.handle_haddps(i),
        Mnemonic::Hsubps => |this, i| this.handle_hsubps(i),
        Mnemonic::Addsubps => |this, i| this.handle_addsubps(i),
        Mnemonic::Addsubpd => |this, i| this.handle_addsubpd(i),

        // SSE2 - scalar / packed double-precision arithmetic
        Mnemonic::Addpd => |this, i| this.handle_addpd(i),
        Mnemonic::Subpd => |this, i| this.handle_subpd(i),
        Mnemonic::Mulpd => |this, i| this.handle_mulpd(i),
        Mnemonic::Divpd => |this, i| this.handle_divpd(i),
        Mnemonic::Addsd => |this, i| this.handle_addsd(i),
        Mnemonic::Subsd => |this, i| this.handle_subsd(i),
        Mnemonic::Mulsd => |this, i| this.handle_mulsd(i),
        Mnemonic::Divsd => |this, i| this.handle_divsd(i),
        Mnemonic::Sqrtsd => |this, i| this.handle_sqrtsd(i),
        Mnemonic::Sqrtpd => |this, i| this.handle_sqrtpd(i),
        Mnemonic::Roundsd => |this, i| this.handle_roundsd(i),
        Mnemonic::Roundss => |this, i| this.handle_roundss(i),
        Mnemonic::Roundpd => |this, i| this.handle_roundpd(i),
        Mnemonic::Roundps => |this, i| this.handle_roundps(i),
        Mnemonic::Minsd => |this, i| this.handle_minmaxsd(i, true),
        Mnemonic::Maxsd => |this, i| this.handle_minmaxsd(i, false),
        Mnemonic::Minpd => |this, i| this.handle_minmaxpd(i, true),
        Mnemonic::Maxpd => |this, i| this.handle_minmaxpd(i, false),
        Mnemonic::Ucomisd | Mnemonic::Comisd => |this, i| this.handle_ucomisd(i),

        // SSE - scalar / packed single-precision arithmetic
        Mnemonic::Addss => |this, i| this.handle_addss(i),
        Mnemonic::Subss => |this, i| this.handle_subss(i),
        Mnemonic::Mulss => |this, i| this.handle_mulss(i),
        Mnemonic::Divss => |this, i| this.handle_divss(i),
        Mnemonic::Sqrtss => |this, i| this.handle_sqrtss(i),
        Mnemonic::Addps => |this, i| this.handle_addps(i),
        Mnemonic::Subps => |this, i| this.handle_subps(i),
        Mnemonic::Mulps => |this, i| this.handle_mulps(i),
        Mnemonic::Divps => |this, i| this.handle_divps(i),
        Mnemonic::Sqrtps => |this, i| this.handle_sqrtps(i),
        Mnemonic::Minss => |this, i| this.handle_minmaxss(i, true),
        Mnemonic::Maxss => |this, i| this.handle_minmaxss(i, false),
        Mnemonic::Minps => |this, i| this.handle_minmaxps(i, true),
        Mnemonic::Maxps => |this, i| this.handle_minmaxps(i, false),
        Mnemonic::Ucomiss | Mnemonic::Comiss => |this, i| this.handle_ucomiss(i),

        // SSE / SSE2 - integer <-> floating-point conversion
        Mnemonic::Cvtsi2ss => |this, i| this.handle_cvtsi2ss(i),
        Mnemonic::Cvttss2si => |this, i| this.handle_cvtss2si(i, true),
        Mnemonic::Cvtss2si => |this, i| this.handle_cvtss2si(i, false),
        Mnemonic::Cvtsi2sd => |this, i| this.handle_cvtsi2sd(i),
        Mnemonic::Cvttsd2si => |this, i| this.handle_cvtsd2si(i, true),
        Mnemonic::Cvtsd2si => |this, i| this.handle_cvtsd2si(i, false),
        Mnemonic::Cvtss2sd => |this, i| this.handle_cvtss2sd(i),
        Mnemonic::Cvtsd2ss => |this, i| this.handle_cvtsd2ss(i),
        Mnemonic::Cvtpd2ps => |this, i| this.handle_cvtpd2ps(i),
        Mnemonic::Cvtps2pd => |this, i| this.handle_cvtps2pd(i),
        Mnemonic::Cvtdq2ps => |this, i| this.handle_cvtdq2ps(i),
        Mnemonic::Cvtps2dq => |this, i| this.handle_cvtps2dq(i, false),
        Mnemonic::Cvttps2dq => |this, i| this.handle_cvtps2dq(i, true),
        Mnemonic::Cvtdq2pd => |this, i| this.handle_cvtdq2pd(i),
        Mnemonic::Cvtpd2dq => |this, i| this.handle_cvtpd2dq(i, false),
        Mnemonic::Cvttpd2dq => |this, i| this.handle_cvtpd2dq(i, true),

        // SSE / SSE2 - packed compares, shuffles and approximate reciprocals
        Mnemonic::Cmpps => |this, i| this.handle_cmpps(i),
        Mnemonic::Cmppd => |this, i| this.handle_cmppd(i),
        Mnemonic::Cmpss => |this, i| this.handle_cmpss(i),
        // CMPSD is shared between string compare and SSE2 scalar compare, like MOVSD.
        Mnemonic::Cmpsd if matches!(instruction.op0_kind(), OpKind::Register) => |this, i| {
            this.handle_cmpsd_sse(i)
        },
        Mnemonic::Shufps => |this, i| this.handle_shufps(i),
        Mnemonic::Shufpd => |this, i| this.handle_shufpd(i),
        Mnemonic::Pshuflw => |this, i| this.handle_pshuflhw(i, false),
        Mnemonic::Pshufhw => |this, i| this.handle_pshuflhw(i, true),
        Mnemonic::Rcpps => |this, i| this.handle_rcpps(i, false),
        Mnemonic::Rsqrtps => |this, i| this.handle_rcpps(i, true),
        Mnemonic::Rcpss => |this, i| this.handle_rcpss(i, false),
        Mnemonic::Rsqrtss => |this, i| this.handle_rcpss(i, true),
        Mnemonic::Movsldup => |this, i| this.handle_movsldup(i, false),
        Mnemonic::Movshdup => |this, i| this.handle_movsldup(i, true),

        // SSE / SSE2 - XMM logic
        Mnemonic::Xorps | Mnemonic::Xorpd => |this, i| this.handle_xorps(i),
        Mnemonic::Pxor => |this, i| this.handle_pxor(i),
        Mnemonic::Andps | Mnemonic::Andpd | Mnemonic::Pand => |this, i| this.handle_andps(i),
        Mnemonic::Orps | Mnemonic::Orpd | Mnemonic::Por => |this, i| this.handle_orps(i),
        Mnemonic::Andnps | Mnemonic::Andnpd | Mnemonic::Pandn => |this, i| {
            this.handle_andnps(i)
        },

        // SSE2 - packed integer add / subtract
        Mnemonic::Paddb => |this, i| this.handle_padd(i, PackedWidth::Byte),
        Mnemonic::Paddw => |this, i| this.handle_padd(i, PackedWidth::Word),
        Mnemonic::Paddd => |this, i| this.handle_padd(i, PackedWidth::Dword),
        Mnemonic::Paddq => |this, i| this.handle_padd(i, PackedWidth::Qword),
        Mnemonic::Psubb => |this, i| this.handle_psub(i, PackedWidth::Byte),
        Mnemonic::Psubw => |this, i| this.handle_psub(i, PackedWidth::Word),
        Mnemonic::Psubd => |this, i| this.handle_psub(i, PackedWidth::Dword),
        Mnemonic::Psubq => |this, i| this.handle_psub(i, PackedWidth::Qword),
        Mnemonic::Paddsb => |this, i| this.handle_padds(i, PackedWidth::Byte),
        Mnemonic::Paddsw => |this, i| this.handle_padds(i, PackedWidth::Word),
        Mnemonic::Paddusb => |this, i| this.handle_paddus(i, PackedWidth::Byte),
        Mnemonic::Paddusw => |this, i| this.handle_paddus(i, PackedWidth::Word),
        Mnemonic::Psubsb => |this, i| this.handle_psubs(i, PackedWidth::Byte),
        Mnemonic::Psubsw => |this, i| this.handle_psubs(i, PackedWidth::Word),
        Mnemonic::Psubusb => |this, i| this.handle_psubus(i, PackedWidth::Byte),
        Mnemonic::Psubusw => |this, i| this.handle_psubus(i, PackedWidth::Word),

        // SSE2 - packed integer compare (all-ones / all-zeros lane masks)
        Mnemonic::Pcmpeqb => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpeqb",
            PackedLaneOp::CmpEq,
            PackedWidth::Byte,
        ),
        Mnemonic::Pcmpeqw => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpeqw",
            PackedLaneOp::CmpEq,
            PackedWidth::Word,
        ),
        Mnemonic::Pcmpeqd => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpeqd",
            PackedLaneOp::CmpEq,
            PackedWidth::Dword,
        ),
        // PCMPEQQ / PCMPGTQ: SSE4.1 / SSE4.2, lane-wise one width up.
        Mnemonic::Pcmpeqq => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpeqq",
            PackedLaneOp::CmpEq,
            PackedWidth::Qword,
        ),
        Mnemonic::Pcmpgtb => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpgtb",
            PackedLaneOp::CmpGt,
            PackedWidth::Byte,
        ),
        Mnemonic::Pcmpgtw => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpgtw",
            PackedLaneOp::CmpGt,
            PackedWidth::Word,
        ),
        Mnemonic::Pcmpgtd => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpgtd",
            PackedLaneOp::CmpGt,
            PackedWidth::Dword,
        ),
        Mnemonic::Pcmpgtq => |this, i| this.handle_packed_lane_op(
            i,
            "pcmpgtq",
            PackedLaneOp::CmpGt,
            PackedWidth::Qword,
        ),

        // AES-NI: one kernel family.
        Mnemonic::Aesenc => |this, i| this.handle_aes_round(i, AesKernel::Enc),
        Mnemonic::Aesenclast => |this, i| this.handle_aes_round(i, AesKernel::EncLast),
        Mnemonic::Aesdec => |this, i| this.handle_aes_round(i, AesKernel::Dec),
        Mnemonic::Aesdeclast => |this, i| this.handle_aes_round(i, AesKernel::DecLast),
        Mnemonic::Aesimc => |this, i| this.handle_aesimc(i),
        Mnemonic::Aeskeygenassist => |this, i| this.handle_aeskeygenassist(i),
        Mnemonic::Pclmulqdq => |this, i| this.handle_pclmulqdq(i),

        // SSE4.2 packed string compares share one handler; the `REX.W` spellings are separate mnemonics.
        Mnemonic::Pcmpistri
        | Mnemonic::Pcmpistrm
        | Mnemonic::Pcmpestri
        | Mnemonic::Pcmpestrm
        | Mnemonic::Pcmpestri64
        | Mnemonic::Pcmpestrm64 => |this, i| this.handle_pcmpstr(i),

        // SSE / SSE2 - packed integer min / max / average / multiply
        Mnemonic::Pminub => |this, i| this.handle_packed_lane_op(
            i,
            "pminub",
            PackedLaneOp::MinU,
            PackedWidth::Byte,
        ),
        Mnemonic::Pmaxub => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxub",
            PackedLaneOp::MaxU,
            PackedWidth::Byte,
        ),
        Mnemonic::Pminsw => |this, i| this.handle_packed_lane_op(
            i,
            "pminsw",
            PackedLaneOp::MinS,
            PackedWidth::Word,
        ),
        Mnemonic::Pmaxsw => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxsw",
            PackedLaneOp::MaxS,
            PackedWidth::Word,
        ),
        Mnemonic::Pavgb => |this, i| this.handle_packed_lane_op(
            i,
            "pavgb",
            PackedLaneOp::Avg,
            PackedWidth::Byte,
        ),
        Mnemonic::Pavgw => |this, i| this.handle_packed_lane_op(
            i,
            "pavgw",
            PackedLaneOp::Avg,
            PackedWidth::Word,
        ),
        Mnemonic::Pmullw => |this, i| this.handle_packed_lane_op(
            i,
            "pmullw",
            PackedLaneOp::MulLow,
            PackedWidth::Word,
        ),
        Mnemonic::Pmulhw => |this, i| this.handle_packed_lane_op(
            i,
            "pmulhw",
            PackedLaneOp::MulHighS,
            PackedWidth::Word,
        ),
        Mnemonic::Pmulhuw => |this, i| this.handle_packed_lane_op(
            i,
            "pmulhuw",
            PackedLaneOp::MulHighU,
            PackedWidth::Word,
        ),

        // SSE4.1 - the min / max forms the earlier generations lacked.
        Mnemonic::Pminsb => |this, i| this.handle_packed_lane_op(
            i,
            "pminsb",
            PackedLaneOp::MinS,
            PackedWidth::Byte,
        ),
        Mnemonic::Pmaxsb => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxsb",
            PackedLaneOp::MaxS,
            PackedWidth::Byte,
        ),
        Mnemonic::Pminuw => |this, i| this.handle_packed_lane_op(
            i,
            "pminuw",
            PackedLaneOp::MinU,
            PackedWidth::Word,
        ),
        Mnemonic::Pmaxuw => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxuw",
            PackedLaneOp::MaxU,
            PackedWidth::Word,
        ),
        Mnemonic::Pminsd => |this, i| this.handle_packed_lane_op(
            i,
            "pminsd",
            PackedLaneOp::MinS,
            PackedWidth::Dword,
        ),
        Mnemonic::Pmaxsd => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxsd",
            PackedLaneOp::MaxS,
            PackedWidth::Dword,
        ),
        Mnemonic::Pminud => |this, i| this.handle_packed_lane_op(
            i,
            "pminud",
            PackedLaneOp::MinU,
            PackedWidth::Dword,
        ),
        Mnemonic::Pmaxud => |this, i| this.handle_packed_lane_op(
            i,
            "pmaxud",
            PackedLaneOp::MaxU,
            PackedWidth::Dword,
        ),
        Mnemonic::Pmulld => |this, i| this.handle_packed_lane_op(
            i,
            "pmulld",
            PackedLaneOp::MulLow,
            PackedWidth::Dword,
        ),

        // SSSE3 - byte shuffle, concat-rotate, sign ops and horizontal arithmetic.
        Mnemonic::Pshufb => |this, i| this.handle_pshufb(i),
        Mnemonic::Palignr => |this, i| this.handle_palignr(i),
        Mnemonic::Pabsb => |this, i| this.handle_pabs(i, PackedWidth::Byte),
        Mnemonic::Pabsw => |this, i| this.handle_pabs(i, PackedWidth::Word),
        Mnemonic::Pabsd => |this, i| this.handle_pabs(i, PackedWidth::Dword),
        Mnemonic::Psignb => |this, i| this.handle_psign(i, PackedWidth::Byte),
        Mnemonic::Psignw => |this, i| this.handle_psign(i, PackedWidth::Word),
        Mnemonic::Psignd => |this, i| this.handle_psign(i, PackedWidth::Dword),
        Mnemonic::Phaddw => |this, i| {
            this.handle_horizontal(i, HorizOp::Add, PackedWidth::Word)
        },
        Mnemonic::Phaddd => |this, i| {
            this.handle_horizontal(i, HorizOp::Add, PackedWidth::Dword)
        },
        Mnemonic::Phaddsw => |this, i| {
            this.handle_horizontal(i, HorizOp::AddSat, PackedWidth::Word)
        },
        Mnemonic::Phsubw => |this, i| {
            this.handle_horizontal(i, HorizOp::Sub, PackedWidth::Word)
        },
        Mnemonic::Phsubd => |this, i| {
            this.handle_horizontal(i, HorizOp::Sub, PackedWidth::Dword)
        },
        Mnemonic::Phsubsw => |this, i| {
            this.handle_horizontal(i, HorizOp::SubSat, PackedWidth::Word)
        },
        Mnemonic::Pmaddubsw => |this, i| this.handle_pmaddubsw(i),
        Mnemonic::Pmulhrsw => |this, i| this.handle_pmulhrsw(i),

        // SSE2 / SSE4.1 - widening multiply, dot product and SAD.
        Mnemonic::Pmuludq => |this, i| this.handle_pmuldq(i, false),
        Mnemonic::Pmuldq => |this, i| this.handle_pmuldq(i, true),
        Mnemonic::Pmaddwd => |this, i| this.handle_pmaddwd(i),
        Mnemonic::Psadbw => |this, i| this.handle_psadbw(i),

        // SSE2 / SSE4.1 - saturating narrowing packs.
        Mnemonic::Packsswb => |this, i| this.handle_pack(i, PackedWidth::Word, true),
        Mnemonic::Packssdw => |this, i| this.handle_pack(i, PackedWidth::Dword, true),
        Mnemonic::Packuswb => |this, i| this.handle_pack(i, PackedWidth::Word, false),
        Mnemonic::Packusdw => |this, i| this.handle_pack(i, PackedWidth::Dword, false),

        // SSE4.1 - sign / zero extending widening.
        Mnemonic::Pmovzxbw => |this, i| this.handle_pmovx(i, 8, 16, false),
        Mnemonic::Pmovzxbd => |this, i| this.handle_pmovx(i, 8, 32, false),
        Mnemonic::Pmovzxbq => |this, i| this.handle_pmovx(i, 8, 64, false),
        Mnemonic::Pmovzxwd => |this, i| this.handle_pmovx(i, 16, 32, false),
        Mnemonic::Pmovzxwq => |this, i| this.handle_pmovx(i, 16, 64, false),
        Mnemonic::Pmovzxdq => |this, i| this.handle_pmovx(i, 32, 64, false),
        Mnemonic::Pmovsxbw => |this, i| this.handle_pmovx(i, 8, 16, true),
        Mnemonic::Pmovsxbd => |this, i| this.handle_pmovx(i, 8, 32, true),
        Mnemonic::Pmovsxbq => |this, i| this.handle_pmovx(i, 8, 64, true),
        Mnemonic::Pmovsxwd => |this, i| this.handle_pmovx(i, 16, 32, true),
        Mnemonic::Pmovsxwq => |this, i| this.handle_pmovx(i, 16, 64, true),
        Mnemonic::Pmovsxdq => |this, i| this.handle_pmovx(i, 32, 64, true),

        // SSE2 / SSE4.1 - lane insert and extract.
        Mnemonic::Pinsrb => |this, i| this.handle_pinsr(i, PackedWidth::Byte),
        Mnemonic::Pinsrw => |this, i| this.handle_pinsr(i, PackedWidth::Word),
        Mnemonic::Pinsrd => |this, i| this.handle_pinsr(i, PackedWidth::Dword),
        Mnemonic::Pinsrq => |this, i| this.handle_pinsr(i, PackedWidth::Qword),
        Mnemonic::Insertps => |this, i| this.handle_insertps(i),
        Mnemonic::Pextrb => |this, i| this.handle_pextr(i, PackedWidth::Byte),
        Mnemonic::Pextrw => |this, i| this.handle_pextr(i, PackedWidth::Word),
        Mnemonic::Pextrd => |this, i| this.handle_pextr(i, PackedWidth::Dword),
        Mnemonic::Pextrq => |this, i| this.handle_pextr(i, PackedWidth::Qword),
        Mnemonic::Extractps => |this, i| this.handle_extractps(i),

        // SSE4.1 - dot products and the two cross-lane reductions.
        Mnemonic::Dpps => |this, i| this.handle_dpps(i),
        Mnemonic::Dppd => |this, i| this.handle_dppd(i),
        Mnemonic::Phminposuw => |this, i| this.handle_phminposuw(i),
        Mnemonic::Mpsadbw => |this, i| this.handle_mpsadbw(i),

        // SSE4.1 - blends and the 128-bit bitwise test.
        Mnemonic::Pblendw => |this, i| this.handle_blend_imm(i, PackedWidth::Word),
        Mnemonic::Blendps => |this, i| this.handle_blend_imm(i, PackedWidth::Dword),
        Mnemonic::Blendpd => |this, i| this.handle_blend_imm(i, PackedWidth::Qword),
        Mnemonic::Pblendvb => |this, i| this.handle_blendv(i, PackedWidth::Byte),
        Mnemonic::Blendvps => |this, i| this.handle_blendv(i, PackedWidth::Dword),
        Mnemonic::Blendvpd => |this, i| this.handle_blendv(i, PackedWidth::Qword),
        Mnemonic::Ptest => |this, i| this.handle_ptest(i),

        // SSE / SSE2 - sign-bit extraction into a GPR
        Mnemonic::Pmovmskb => |this, i| this.handle_movmsk(i, "pmovmskb", PackedWidth::Byte),
        Mnemonic::Movmskps => |this, i| this.handle_movmsk(i, "movmskps", PackedWidth::Dword),
        Mnemonic::Movmskpd => |this, i| this.handle_movmsk(i, "movmskpd", PackedWidth::Qword),

        // String Operations
        Mnemonic::Movsb => |this, i| this.handle_movsb(i),
        Mnemonic::Movsw => |this, i| this.handle_movsw(i),
        // MOVSD is shared with SSE2 scalar-double; disambiguate on a memory-string op0.
        Mnemonic::Movsd => |this, i| {
            if matches!(
                i.op0_kind(),
                OpKind::MemoryESDI | OpKind::MemoryESRDI
            ) {
                this.handle_movsd_mem(i)
            } else {
                this.handle_movsd_sse(i)
            }
        },
        Mnemonic::Movsq => |this, i| this.handle_movsq(i),
        Mnemonic::Stosb => |this, i| this.handle_stosb(i),
        Mnemonic::Stosw => |this, i| this.handle_stosw(i),
        Mnemonic::Stosd => |this, i| this.handle_stosd(i),
        Mnemonic::Stosq => |this, i| this.handle_stosq(i),
        Mnemonic::Lodsb => |this, i| this.handle_lodsb(i),
        Mnemonic::Lodsw => |this, i| this.handle_lodsw(i),
        Mnemonic::Lodsd => |this, i| this.handle_lodsd(i),
        Mnemonic::Lodsq => |this, i| this.handle_lodsq(i),
        Mnemonic::Insb => |this, i| this.handle_insb(i),
        Mnemonic::Insw => |this, i| this.handle_insw(i),
        Mnemonic::Insd => |this, i| this.handle_insd(i),
        Mnemonic::Outsb => |this, i| this.handle_outsb(i),
        Mnemonic::Outsw => |this, i| this.handle_outsw(i),
        Mnemonic::Outsd => |this, i| this.handle_outsd(i),

        // Arithmetic
        Mnemonic::Add => |this, i| this.handle_add(i),
        Mnemonic::Adc => |this, i| this.handle_adc(i),
        Mnemonic::Sub => |this, i| this.handle_sub(i),
        Mnemonic::Sbb => |this, i| this.handle_sbb(i),
        // Sign-extension conversions (implicit operand form)
        Mnemonic::Cbw => |this, i| this.handle_cbw(i),
        Mnemonic::Cwde => |this, i| this.handle_cwde(i),
        Mnemonic::Cdqe => |this, i| this.handle_cdqe(i),
        Mnemonic::Cwd => |this, i| this.handle_cwd(i),
        Mnemonic::Cdq => |this, i| this.handle_cdq(i),
        Mnemonic::Cqo => |this, i| this.handle_cqo(i),
        Mnemonic::Xadd => |this, i| this.handle_xadd(i),
        Mnemonic::Cmp => |this, i| this.handle_cmp(i),
        Mnemonic::Inc => |this, i| this.handle_inc(i),
        Mnemonic::Dec => |this, i| this.handle_dec(i),
        Mnemonic::Neg => |this, i| this.handle_neg(i),
        Mnemonic::Imul => |this, i| this.handle_imul(i),
        Mnemonic::Mul => |this, i| this.handle_mul(i),
        Mnemonic::Idiv => |this, i| this.handle_idiv(i),
        Mnemonic::Div => |this, i| this.handle_div(i),

        // Bitwise / Logical
        Mnemonic::And => |this, i| this.handle_and(i),
        Mnemonic::Or => |this, i| this.handle_or(i),
        Mnemonic::Xor => |this, i| this.handle_xor(i),
        Mnemonic::Not => |this, i| this.handle_not(i),
        Mnemonic::Test => |this, i| this.handle_test(i),

        // Shifts / Rotates
        Mnemonic::Shl => |this, i| this.handle_shl(i),
        Mnemonic::Shr => |this, i| this.handle_shr(i),
        Mnemonic::Sar => |this, i| this.handle_sar(i),
        Mnemonic::Rol => |this, i| this.handle_rol(i),
        Mnemonic::Ror => |this, i| this.handle_ror(i),
        Mnemonic::Rcl => |this, i| this.handle_rcl(i),
        Mnemonic::Rcr => |this, i| this.handle_rcr(i),
        Mnemonic::Shld => |this, i| this.handle_shld(i),
        Mnemonic::Shrd => |this, i| this.handle_shrd(i),

        // Bit Manipulation
        Mnemonic::Bt => |this, i| this.handle_bt(i),
        Mnemonic::Bts => |this, i| this.handle_bts(i),
        Mnemonic::Btr => |this, i| this.handle_btr(i),
        Mnemonic::Btc => |this, i| this.handle_btc(i),
        Mnemonic::Bsf => |this, i| this.handle_bsf(i),
        Mnemonic::Bsr => |this, i| this.handle_bsr(i),
        Mnemonic::Tzcnt => |this, i| this.handle_tzcnt(i),
        Mnemonic::Lzcnt => |this, i| this.handle_lzcnt(i),
        Mnemonic::Popcnt => |this, i| this.handle_popcnt(i),
        // SSE4.2 - CRC-32C accumulation.
        Mnemonic::Crc32 => |this, i| this.handle_crc32(i),
        Mnemonic::Bswap => |this, i| this.handle_bswap(i),

        // Unconditional Control Flow
        Mnemonic::Jmp => |this, i| this.handle_jmp(i),
        Mnemonic::Call => |this, i| this.handle_call(i),
        Mnemonic::Ret => |this, i| this.handle_ret(i),
        Mnemonic::Retf => |this, i| this.handle_retf(i),
        Mnemonic::Loop => |this, i| this.handle_loop(i),
        Mnemonic::Loope => |this, i| this.handle_loope(i),
        Mnemonic::Loopne => |this, i| this.handle_loopne(i),

        // Conditional jumps via `Cond`; JRCXZ and LOOP test a register and stay separate.
        Mnemonic::Jb => |this, i| this.handle_jcc(i, Cond::B),
        Mnemonic::Jbe => |this, i| this.handle_jcc(i, Cond::Be),
        Mnemonic::Ja => |this, i| this.handle_jcc(i, Cond::A),
        Mnemonic::Jae => |this, i| this.handle_jcc(i, Cond::Ae),
        Mnemonic::Je => |this, i| this.handle_jcc(i, Cond::E),
        Mnemonic::Jne => |this, i| this.handle_jcc(i, Cond::Ne),
        Mnemonic::Jg => |this, i| this.handle_jcc(i, Cond::G),
        Mnemonic::Jge => |this, i| this.handle_jcc(i, Cond::Ge),
        Mnemonic::Jl => |this, i| this.handle_jcc(i, Cond::L),
        Mnemonic::Jle => |this, i| this.handle_jcc(i, Cond::Le),
        Mnemonic::Js => |this, i| this.handle_jcc(i, Cond::S),
        Mnemonic::Jns => |this, i| this.handle_jcc(i, Cond::Ns),
        Mnemonic::Jo => |this, i| this.handle_jcc(i, Cond::O),
        Mnemonic::Jno => |this, i| this.handle_jcc(i, Cond::No),
        Mnemonic::Jp => |this, i| this.handle_jcc(i, Cond::P),
        Mnemonic::Jnp => |this, i| this.handle_jcc(i, Cond::Np),
        Mnemonic::Jrcxz => |this, i| this.handle_jrcxz(i),

        // SETcc - set byte on condition
        Mnemonic::Sete => |this, i| this.handle_setcc(i, Cond::E),
        Mnemonic::Setne => |this, i| this.handle_setcc(i, Cond::Ne),
        Mnemonic::Setb => |this, i| this.handle_setcc(i, Cond::B),
        Mnemonic::Setae => |this, i| this.handle_setcc(i, Cond::Ae),
        Mnemonic::Setbe => |this, i| this.handle_setcc(i, Cond::Be),
        Mnemonic::Seta => |this, i| this.handle_setcc(i, Cond::A),
        Mnemonic::Setl => |this, i| this.handle_setcc(i, Cond::L),
        Mnemonic::Setge => |this, i| this.handle_setcc(i, Cond::Ge),
        Mnemonic::Setle => |this, i| this.handle_setcc(i, Cond::Le),
        Mnemonic::Setg => |this, i| this.handle_setcc(i, Cond::G),
        Mnemonic::Sets => |this, i| this.handle_setcc(i, Cond::S),
        Mnemonic::Setns => |this, i| this.handle_setcc(i, Cond::Ns),
        Mnemonic::Seto => |this, i| this.handle_setcc(i, Cond::O),
        Mnemonic::Setno => |this, i| this.handle_setcc(i, Cond::No),
        Mnemonic::Setp => |this, i| this.handle_setcc(i, Cond::P),
        Mnemonic::Setnp => |this, i| this.handle_setcc(i, Cond::Np),

        // Conditional Move (CMOVcc)
        Mnemonic::Cmove => |this, i| this.handle_cmovcc(i, Cond::E),
        Mnemonic::Cmovne => |this, i| this.handle_cmovcc(i, Cond::Ne),
        Mnemonic::Cmovb => |this, i| this.handle_cmovcc(i, Cond::B),
        Mnemonic::Cmovae => |this, i| this.handle_cmovcc(i, Cond::Ae),
        Mnemonic::Cmovbe => |this, i| this.handle_cmovcc(i, Cond::Be),
        Mnemonic::Cmova => |this, i| this.handle_cmovcc(i, Cond::A),
        Mnemonic::Cmovl => |this, i| this.handle_cmovcc(i, Cond::L),
        Mnemonic::Cmovge => |this, i| this.handle_cmovcc(i, Cond::Ge),
        Mnemonic::Cmovle => |this, i| this.handle_cmovcc(i, Cond::Le),
        Mnemonic::Cmovg => |this, i| this.handle_cmovcc(i, Cond::G),
        Mnemonic::Cmovs => |this, i| this.handle_cmovcc(i, Cond::S),
        Mnemonic::Cmovns => |this, i| this.handle_cmovcc(i, Cond::Ns),
        Mnemonic::Cmovo => |this, i| this.handle_cmovcc(i, Cond::O),
        Mnemonic::Cmovno => |this, i| this.handle_cmovcc(i, Cond::No),
        Mnemonic::Cmovp => |this, i| this.handle_cmovcc(i, Cond::P),
        Mnemonic::Cmovnp => |this, i| this.handle_cmovcc(i, Cond::Np),

        // Traps / faults
        Mnemonic::Ud2 => |this, i| this.handle_ud2(i),
        Mnemonic::Int3 => |this, i| this.handle_int3(i),
        Mnemonic::Int => |this, i| this.handle_int(i),
        Mnemonic::Int1 => |this, i| this.handle_int1(i),

        // System
        Mnemonic::Syscall => |this, i| this.handle_syscall(i),
        Mnemonic::Cpuid => |this, i| this.handle_cpuid(i),
        Mnemonic::Xgetbv => |this, i| this.handle_xgetbv(i),
        Mnemonic::Rdtsc => |this, i| this.handle_rdtsc(i),

        // FS/GS base (FSGSBASE extension)
        Mnemonic::Rdfsbase => |this, i| this.handle_rdfsbase(i),
        Mnemonic::Wrfsbase => |this, i| this.handle_wrfsbase(i),
        Mnemonic::Rdgsbase => |this, i| this.handle_rdgsbase(i),
        Mnemonic::Wrgsbase => |this, i| this.handle_wrgsbase(i),

        // x87 FPU - loads / stores
        Mnemonic::Fld => |this, i| this.handle_fld(i),
        Mnemonic::Fild => |this, i| this.handle_fild(i),
        Mnemonic::Fldz => |this, _| this.handle_fld_const(0.0),
        Mnemonic::Fld1 => |this, _| this.handle_fld_const(1.0),
        Mnemonic::Fldpi => |this, _| this.handle_fld_const(std::f64::consts::PI),
        Mnemonic::Fst => |this, i| this.handle_fst(i, false),
        Mnemonic::Fstp => |this, i| this.handle_fst(i, true),
        Mnemonic::Fist => |this, i| this.handle_fist(i, false, false),
        Mnemonic::Fistp => |this, i| this.handle_fist(i, true, false),
        Mnemonic::Fisttp => |this, i| this.handle_fist(i, true, true),
        Mnemonic::Fxch => |this, i| this.handle_fxch(i),

        // x87 FPU - arithmetic
        Mnemonic::Fadd => |this, i| this.handle_farith(i, FpuOp::Add, false),
        Mnemonic::Faddp => |this, i| this.handle_farith(i, FpuOp::Add, true),
        Mnemonic::Fsub => |this, i| this.handle_farith(i, FpuOp::Sub, false),
        Mnemonic::Fsubp => |this, i| this.handle_farith(i, FpuOp::Sub, true),
        Mnemonic::Fsubr => |this, i| this.handle_farith(i, FpuOp::Subr, false),
        Mnemonic::Fsubrp => |this, i| this.handle_farith(i, FpuOp::Subr, true),
        Mnemonic::Fmul => |this, i| this.handle_farith(i, FpuOp::Mul, false),
        Mnemonic::Fmulp => |this, i| this.handle_farith(i, FpuOp::Mul, true),
        Mnemonic::Fdiv => |this, i| this.handle_farith(i, FpuOp::Div, false),
        Mnemonic::Fdivp => |this, i| this.handle_farith(i, FpuOp::Div, true),
        Mnemonic::Fdivr => |this, i| this.handle_farith(i, FpuOp::Divr, false),
        Mnemonic::Fdivrp => |this, i| this.handle_farith(i, FpuOp::Divr, true),
        Mnemonic::Fabs => |this, i| this.handle_fabs(i),
        Mnemonic::Fchs => |this, i| this.handle_fchs(i),
        Mnemonic::Fsqrt => |this, i| this.handle_fsqrt(i),
        Mnemonic::Fcos => |this, i| this.handle_fcos(i),
        Mnemonic::Fsin => |this, i| this.handle_fsin(i),
        Mnemonic::Fsincos => |this, i| this.handle_fsincos(i),
        Mnemonic::Fptan => |this, i| this.handle_fptan(i),

        // x87 FPU - comparisons
        Mnemonic::Fcom => |this, i| this.handle_fcom(i, false),
        Mnemonic::Fcomp => |this, i| this.handle_fcom(i, true),
        Mnemonic::Fucom => |this, i| this.handle_fcom(i, false),
        Mnemonic::Fucomp => |this, i| this.handle_fcom(i, true),
        Mnemonic::Ftst => |this, i| this.handle_ftst(i),
        Mnemonic::Fcomi => |this, i| this.handle_fcomi(i, false),
        Mnemonic::Fcomip => |this, i| this.handle_fcomi(i, true),
        Mnemonic::Fucomi => |this, i| this.handle_fcomi(i, false),
        Mnemonic::Fucomip => |this, i| this.handle_fcomi(i, true),

        // x87 FPU - conditional moves
        Mnemonic::Fcmovb => |this, i| this.handle_fcmovb(i),
        Mnemonic::Fcmovnb => |this, i| this.handle_fcmovnb(i),
        Mnemonic::Fcmove => |this, i| this.handle_fcmove(i),
        Mnemonic::Fcmovne => |this, i| this.handle_fcmovne(i),
        Mnemonic::Fcmovbe => |this, i| this.handle_fcmovbe(i),
        Mnemonic::Fcmovnbe => |this, i| this.handle_fcmovnbe(i),
        Mnemonic::Fcmovu => |this, i| this.handle_fcmovu(i),
        Mnemonic::Fcmovnu => |this, i| this.handle_fcmovnu(i),

        // x87 FPU - control / misc
        Mnemonic::Fnstsw => |this, i| this.handle_fnstsw(i),
        Mnemonic::Fnstcw => |this, i| this.handle_fnstcw(i),
        Mnemonic::Fldcw => |this, i| this.handle_fldcw(i),
        Mnemonic::Fincstp => |this, i| this.handle_fincstp(i),
        Mnemonic::Fdecstp => |this, i| this.handle_fdecstp(i),
        Mnemonic::Ffree => |this, i| this.handle_ffree(i),
        Mnemonic::Emms => |this, i| this.handle_emms(i),
        // `FNOP`/`WAIT` do nothing (`9B` is the prefix iced splits off `FSTENV`/`FCLEX`/`FINIT`).
        Mnemonic::Fnop | Mnemonic::Wait => |this, i| this.handle_fpu_nop(i),
        Mnemonic::Fnstenv => |this, i| this.handle_fnstenv(i),
        Mnemonic::Fldenv => |this, i| this.handle_fldenv(i),
        Mnemonic::Fnclex => |this, i| this.handle_fnclex(i),
        Mnemonic::Fninit => |this, i| this.handle_fninit(i),

        // SSE control/status; the AVX spellings are identical.
        Mnemonic::Stmxcsr | Mnemonic::Vstmxcsr => |this, i| this.handle_stmxcsr(i),
        Mnemonic::Ldmxcsr | Mnemonic::Vldmxcsr => |this, i| this.handle_ldmxcsr(i),

        Mnemonic::Stc => |this, i| this.handle_stc(i),
        Mnemonic::Clc => |this, i| this.handle_clc(i),
        Mnemonic::Cmc => |this, i| this.handle_cmc(i),
        Mnemonic::Std => |this, i| this.handle_std(i),
        Mnemonic::Cld => |this, i| this.handle_cld(i),

        _ => return None,
    };
    Some(semantics)
}

/// True when any operand names an MMX register (`MM0..MM7`).
fn uses_mmx_register(instruction: &Instruction) -> bool {
    (0..instruction.op_count()).any(|i| {
        matches!(
            instruction.op_register(i),
            Register::MM0
                | Register::MM1
                | Register::MM2
                | Register::MM3
                | Register::MM4
                | Register::MM5
                | Register::MM6
                | Register::MM7
        )
    })
}
