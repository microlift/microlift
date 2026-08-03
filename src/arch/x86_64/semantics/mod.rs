//! Semantics layer - maps x86-64 instructions to Cranelift IR.
//!
//! This module is the dispatch hub. The actual implementations live in the
//! sub-modules below, each focused on a specific instruction category.

pub mod arithmetic;
pub mod atomics;
pub mod bitops;
pub mod bridge;
pub mod control_flow;
pub mod flags;
pub mod fpu;
pub mod helpers;
pub mod memory;
pub mod setcc;
pub mod softfloat;
pub mod sse;
pub mod sse_int;
pub mod sse_vec;
pub mod syscall;
pub mod unimplemented;

use crate::MicroliftError;
use crate::arch::x86_64::semantics::fpu::FpuOp;
use crate::arch::x86_64::semantics::sse::{PackedLaneOp, PackedShiftOp, PackedWidth};
use crate::arch::x86_64::semantics::sse_int::HorizOp;
use crate::lifter::Lifter;
use iced_x86::{Instruction, Mnemonic, OpKind, Register};

impl<'a> Lifter<'a> {
    pub fn handle_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        // LOCK-prefixed instructions require atomic semantics; dispatch them
        // before the main match so that ADD, SUB, etc. only see non-locked ops.
        if instruction.has_lock_prefix() {
            return self.handle_locked_instruction(instruction);
        }

        // Most SSE integer opcodes have an MMX encoding (the same mnemonic
        // without the 0x66 prefix) that operates on MM0..MM7 - an aliased view
        // of the x87 stack this frontend does not model. Exit to the host
        // rather than failing the whole lift on an operand the handlers cannot
        // resolve to an XMM register.
        if uses_mmx_register(instruction) {
            return self.handle_unimplemented_instruction(instruction);
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
            Mnemonic::Pushfq => self.handle_pushfq(instruction),
            Mnemonic::Popfq => self.handle_popfq(instruction),
            Mnemonic::Leave => self.handle_leave(instruction),
            Mnemonic::Xchg => self.handle_xchg(instruction),
            Mnemonic::Nop | Mnemonic::Reservednop => self.handle_nop(instruction),

            // Atomics / synchronisation
            Mnemonic::Cmpxchg => self.handle_cmpxchg(instruction),
            Mnemonic::Cmpxchg8b => self.handle_cmpxchg8b(instruction),
            Mnemonic::Cmpxchg16b => self.handle_cmpxchg16b(instruction),
            Mnemonic::Mfence | Mnemonic::Sfence | Mnemonic::Lfence | Mnemonic::Serialize => {
                self.handle_mfence(instruction)
            }
            Mnemonic::Pause => self.handle_pause(instruction),
            // Cache maintenance. The guest shares the host's coherent view of
            // memory (guest VA == host VA, no emulated cache), so a flush or a
            // writeback has nothing to do; ordering still comes from the fences
            // that surround them.
            Mnemonic::Clflush | Mnemonic::Clflushopt | Mnemonic::Clwb => {
                self.handle_nop(instruction)
            }
            // Prefetch hints, pure performance advice with no architectural
            // effect. Common in the lock-free code that surrounds a CAS loop.
            Mnemonic::Prefetch
            | Mnemonic::Prefetcht0
            | Mnemonic::Prefetcht1
            | Mnemonic::Prefetcht2
            | Mnemonic::Prefetchnta
            | Mnemonic::Prefetchw
            | Mnemonic::Prefetchwt1
            | Mnemonic::Prefetchit0
            | Mnemonic::Prefetchit1 => self.handle_nop(instruction),

            // CET
            Mnemonic::Endbr64 | Mnemonic::Endbr32 => self.handle_endbr64(instruction),

            // SSE / SSE2 - XMM data movement
            Mnemonic::Movaps
            | Mnemonic::Movapd
            | Mnemonic::Movups
            | Mnemonic::Movupd
            | Mnemonic::Movdqa
            | Mnemonic::Movdqu
            // The non-temporal stores differ only in cache hint, which this
            // lifter does not model.
            | Mnemonic::Movntps
            | Mnemonic::Movntpd
            | Mnemonic::Movntdq
            // LDDQU and the non-temporal load MOVNTDQA differ from an ordinary
            // 128-bit load only in cache/split-load hints.
            | Mnemonic::Lddqu
            | Mnemonic::Movntdqa => self.handle_movaps(instruction),
            Mnemonic::Movq => self.handle_movq_sse(instruction),
            Mnemonic::Movss => self.handle_movss(instruction),
            Mnemonic::Movd => self.handle_movd(instruction),
            Mnemonic::Movhps | Mnemonic::Movhpd => self.handle_movhps(instruction),
            Mnemonic::Movlps | Mnemonic::Movlpd => self.handle_movlps(instruction),
            Mnemonic::Movhlps => self.handle_movhlps(instruction),
            Mnemonic::Movlhps => self.handle_movlhps(instruction),
            Mnemonic::Movddup => self.handle_movddup(instruction),
            // SSE2 - packed per-lane shifts
            Mnemonic::Psrlw => {
                self.handle_packed_shift(instruction, PackedShiftOp::Srl, PackedWidth::Word)
            }
            Mnemonic::Psrld => {
                self.handle_packed_shift(instruction, PackedShiftOp::Srl, PackedWidth::Dword)
            }
            Mnemonic::Psrlq => {
                self.handle_packed_shift(instruction, PackedShiftOp::Srl, PackedWidth::Qword)
            }
            Mnemonic::Psllw => {
                self.handle_packed_shift(instruction, PackedShiftOp::Sll, PackedWidth::Word)
            }
            Mnemonic::Pslld => {
                self.handle_packed_shift(instruction, PackedShiftOp::Sll, PackedWidth::Dword)
            }
            Mnemonic::Psllq => {
                self.handle_packed_shift(instruction, PackedShiftOp::Sll, PackedWidth::Qword)
            }
            Mnemonic::Psraw => {
                self.handle_packed_shift(instruction, PackedShiftOp::Sra, PackedWidth::Word)
            }
            Mnemonic::Psrad => {
                self.handle_packed_shift(instruction, PackedShiftOp::Sra, PackedWidth::Dword)
            }
            Mnemonic::Pshufd => self.handle_pshufd(instruction),
            // SSE2 - whole-register byte shifts
            Mnemonic::Pslldq => self.handle_byte_shift_dq(instruction, true),
            Mnemonic::Psrldq => self.handle_byte_shift_dq(instruction, false),

            // SSE / SSE2 - unpack / interleave. The `pd`/`ps` forms are exact
            // aliases of the integer ones at the same lane width.
            Mnemonic::Punpcklbw => {
                self.handle_punpck(instruction, "punpcklbw", PackedWidth::Byte, false)
            }
            Mnemonic::Punpcklwd => {
                self.handle_punpck(instruction, "punpcklwd", PackedWidth::Word, false)
            }
            Mnemonic::Punpckldq | Mnemonic::Unpcklps => {
                self.handle_punpck(instruction, "punpckldq", PackedWidth::Dword, false)
            }
            Mnemonic::Punpcklqdq | Mnemonic::Unpcklpd => {
                self.handle_punpck(instruction, "punpcklqdq", PackedWidth::Qword, false)
            }
            Mnemonic::Punpckhbw => {
                self.handle_punpck(instruction, "punpckhbw", PackedWidth::Byte, true)
            }
            Mnemonic::Punpckhwd => {
                self.handle_punpck(instruction, "punpckhwd", PackedWidth::Word, true)
            }
            Mnemonic::Punpckhdq | Mnemonic::Unpckhps => {
                self.handle_punpck(instruction, "punpckhdq", PackedWidth::Dword, true)
            }
            Mnemonic::Punpckhqdq | Mnemonic::Unpckhpd => {
                self.handle_punpck(instruction, "punpckhqdq", PackedWidth::Qword, true)
            }

            // SSE2 - scalar / packed double-precision arithmetic
            Mnemonic::Addpd => self.handle_addpd(instruction),
            Mnemonic::Subpd => self.handle_subpd(instruction),
            Mnemonic::Mulpd => self.handle_mulpd(instruction),
            Mnemonic::Divpd => self.handle_divpd(instruction),
            Mnemonic::Addsd => self.handle_addsd(instruction),
            Mnemonic::Subsd => self.handle_subsd(instruction),
            Mnemonic::Mulsd => self.handle_mulsd(instruction),
            Mnemonic::Divsd => self.handle_divsd(instruction),
            Mnemonic::Sqrtsd => self.handle_sqrtsd(instruction),
            Mnemonic::Sqrtpd => self.handle_sqrtpd(instruction),
            Mnemonic::Minsd => self.handle_minmaxsd(instruction, true),
            Mnemonic::Maxsd => self.handle_minmaxsd(instruction, false),
            Mnemonic::Minpd => self.handle_minmaxpd(instruction, true),
            Mnemonic::Maxpd => self.handle_minmaxpd(instruction, false),
            Mnemonic::Ucomisd | Mnemonic::Comisd => self.handle_ucomisd(instruction),

            // SSE - scalar / packed single-precision arithmetic
            Mnemonic::Addss => self.handle_addss(instruction),
            Mnemonic::Subss => self.handle_subss(instruction),
            Mnemonic::Mulss => self.handle_mulss(instruction),
            Mnemonic::Divss => self.handle_divss(instruction),
            Mnemonic::Sqrtss => self.handle_sqrtss(instruction),
            Mnemonic::Addps => self.handle_addps(instruction),
            Mnemonic::Subps => self.handle_subps(instruction),
            Mnemonic::Mulps => self.handle_mulps(instruction),
            Mnemonic::Divps => self.handle_divps(instruction),
            Mnemonic::Sqrtps => self.handle_sqrtps(instruction),
            Mnemonic::Minss => self.handle_minmaxss(instruction, true),
            Mnemonic::Maxss => self.handle_minmaxss(instruction, false),
            Mnemonic::Minps => self.handle_minmaxps(instruction, true),
            Mnemonic::Maxps => self.handle_minmaxps(instruction, false),
            Mnemonic::Ucomiss | Mnemonic::Comiss => self.handle_ucomiss(instruction),

            // SSE / SSE2 - integer <-> floating-point conversion
            Mnemonic::Cvtsi2ss => self.handle_cvtsi2ss(instruction),
            Mnemonic::Cvttss2si => self.handle_cvtss2si(instruction, true),
            Mnemonic::Cvtss2si => self.handle_cvtss2si(instruction, false),
            Mnemonic::Cvtsi2sd => self.handle_cvtsi2sd(instruction),
            Mnemonic::Cvttsd2si => self.handle_cvtsd2si(instruction, true),
            Mnemonic::Cvtsd2si => self.handle_cvtsd2si(instruction, false),
            Mnemonic::Cvtss2sd => self.handle_cvtss2sd(instruction),
            Mnemonic::Cvtsd2ss => self.handle_cvtsd2ss(instruction),
            Mnemonic::Cvtpd2ps => self.handle_cvtpd2ps(instruction),
            Mnemonic::Cvtps2pd => self.handle_cvtps2pd(instruction),
            Mnemonic::Cvtdq2ps => self.handle_cvtdq2ps(instruction),
            Mnemonic::Cvtps2dq => self.handle_cvtps2dq(instruction, false),
            Mnemonic::Cvttps2dq => self.handle_cvtps2dq(instruction, true),
            Mnemonic::Cvtdq2pd => self.handle_cvtdq2pd(instruction),
            Mnemonic::Cvtpd2dq => self.handle_cvtpd2dq(instruction, false),
            Mnemonic::Cvttpd2dq => self.handle_cvtpd2dq(instruction, true),

            // SSE / SSE2 - packed compares, shuffles and approximate reciprocals
            Mnemonic::Cmpps => self.handle_cmpps(instruction),
            Mnemonic::Cmppd => self.handle_cmppd(instruction),
            Mnemonic::Cmpss => self.handle_cmpss(instruction),
            // CMPSD is shared between the string compare (memory-string
            // operands) and SSE2 scalar-double compare (XMM destination), the
            // same collision MOVSD has below.
            Mnemonic::Cmpsd if matches!(instruction.op0_kind(), OpKind::Register) => {
                self.handle_cmpsd_sse(instruction)
            }
            Mnemonic::Shufps => self.handle_shufps(instruction),
            Mnemonic::Shufpd => self.handle_shufpd(instruction),
            Mnemonic::Pshuflw => self.handle_pshuflhw(instruction, false),
            Mnemonic::Pshufhw => self.handle_pshuflhw(instruction, true),
            Mnemonic::Rcpps => self.handle_rcpps(instruction, false),
            Mnemonic::Rsqrtps => self.handle_rcpps(instruction, true),
            Mnemonic::Rcpss => self.handle_rcpss(instruction, false),
            Mnemonic::Rsqrtss => self.handle_rcpss(instruction, true),
            Mnemonic::Movsldup => self.handle_movsldup(instruction, false),
            Mnemonic::Movshdup => self.handle_movsldup(instruction, true),

            // SSE / SSE2 - XMM logic
            Mnemonic::Xorps | Mnemonic::Xorpd => self.handle_xorps(instruction),
            Mnemonic::Pxor => self.handle_pxor(instruction),
            Mnemonic::Andps | Mnemonic::Andpd | Mnemonic::Pand => self.handle_andps(instruction),
            Mnemonic::Orps | Mnemonic::Orpd | Mnemonic::Por => self.handle_orps(instruction),
            Mnemonic::Andnps | Mnemonic::Andnpd | Mnemonic::Pandn => {
                self.handle_andnps(instruction)
            }

            // SSE2 - packed integer add / subtract
            Mnemonic::Paddb => self.handle_padd(instruction, PackedWidth::Byte),
            Mnemonic::Paddw => self.handle_padd(instruction, PackedWidth::Word),
            Mnemonic::Paddd => self.handle_padd(instruction, PackedWidth::Dword),
            Mnemonic::Paddq => self.handle_padd(instruction, PackedWidth::Qword),
            Mnemonic::Psubb => self.handle_psub(instruction, PackedWidth::Byte),
            Mnemonic::Psubw => self.handle_psub(instruction, PackedWidth::Word),
            Mnemonic::Psubd => self.handle_psub(instruction, PackedWidth::Dword),
            Mnemonic::Psubq => self.handle_psub(instruction, PackedWidth::Qword),
            Mnemonic::Paddsb => self.handle_padds(instruction, PackedWidth::Byte),
            Mnemonic::Paddsw => self.handle_padds(instruction, PackedWidth::Word),
            Mnemonic::Paddusb => self.handle_paddus(instruction, PackedWidth::Byte),
            Mnemonic::Paddusw => self.handle_paddus(instruction, PackedWidth::Word),
            Mnemonic::Psubsb => self.handle_psubs(instruction, PackedWidth::Byte),
            Mnemonic::Psubsw => self.handle_psubs(instruction, PackedWidth::Word),
            Mnemonic::Psubusb => self.handle_psubus(instruction, PackedWidth::Byte),
            Mnemonic::Psubusw => self.handle_psubus(instruction, PackedWidth::Word),

            // SSE2 - packed integer compare (all-ones / all-zeros lane masks)
            Mnemonic::Pcmpeqb => self.handle_packed_lane_op(
                instruction,
                "pcmpeqb",
                PackedLaneOp::CmpEq,
                PackedWidth::Byte,
            ),
            Mnemonic::Pcmpeqw => self.handle_packed_lane_op(
                instruction,
                "pcmpeqw",
                PackedLaneOp::CmpEq,
                PackedWidth::Word,
            ),
            Mnemonic::Pcmpeqd => self.handle_packed_lane_op(
                instruction,
                "pcmpeqd",
                PackedLaneOp::CmpEq,
                PackedWidth::Dword,
            ),
            // PCMPEQQ / PCMPGTQ are SSE4.1 / SSE4.2, but are the same lane-wise
            // op one width up.
            Mnemonic::Pcmpeqq => self.handle_packed_lane_op(
                instruction,
                "pcmpeqq",
                PackedLaneOp::CmpEq,
                PackedWidth::Qword,
            ),
            Mnemonic::Pcmpgtb => self.handle_packed_lane_op(
                instruction,
                "pcmpgtb",
                PackedLaneOp::CmpGt,
                PackedWidth::Byte,
            ),
            Mnemonic::Pcmpgtw => self.handle_packed_lane_op(
                instruction,
                "pcmpgtw",
                PackedLaneOp::CmpGt,
                PackedWidth::Word,
            ),
            Mnemonic::Pcmpgtd => self.handle_packed_lane_op(
                instruction,
                "pcmpgtd",
                PackedLaneOp::CmpGt,
                PackedWidth::Dword,
            ),
            Mnemonic::Pcmpgtq => self.handle_packed_lane_op(
                instruction,
                "pcmpgtq",
                PackedLaneOp::CmpGt,
                PackedWidth::Qword,
            ),

            // SSE / SSE2 - packed integer min / max / average / multiply
            Mnemonic::Pminub => self.handle_packed_lane_op(
                instruction,
                "pminub",
                PackedLaneOp::MinU,
                PackedWidth::Byte,
            ),
            Mnemonic::Pmaxub => self.handle_packed_lane_op(
                instruction,
                "pmaxub",
                PackedLaneOp::MaxU,
                PackedWidth::Byte,
            ),
            Mnemonic::Pminsw => self.handle_packed_lane_op(
                instruction,
                "pminsw",
                PackedLaneOp::MinS,
                PackedWidth::Word,
            ),
            Mnemonic::Pmaxsw => self.handle_packed_lane_op(
                instruction,
                "pmaxsw",
                PackedLaneOp::MaxS,
                PackedWidth::Word,
            ),
            Mnemonic::Pavgb => self.handle_packed_lane_op(
                instruction,
                "pavgb",
                PackedLaneOp::Avg,
                PackedWidth::Byte,
            ),
            Mnemonic::Pavgw => self.handle_packed_lane_op(
                instruction,
                "pavgw",
                PackedLaneOp::Avg,
                PackedWidth::Word,
            ),
            Mnemonic::Pmullw => self.handle_packed_lane_op(
                instruction,
                "pmullw",
                PackedLaneOp::MulLow,
                PackedWidth::Word,
            ),
            Mnemonic::Pmulhw => self.handle_packed_lane_op(
                instruction,
                "pmulhw",
                PackedLaneOp::MulHighS,
                PackedWidth::Word,
            ),
            Mnemonic::Pmulhuw => self.handle_packed_lane_op(
                instruction,
                "pmulhuw",
                PackedLaneOp::MulHighU,
                PackedWidth::Word,
            ),

            // SSE4.1 - the min / max forms the earlier generations lacked.
            Mnemonic::Pminsb => self.handle_packed_lane_op(
                instruction,
                "pminsb",
                PackedLaneOp::MinS,
                PackedWidth::Byte,
            ),
            Mnemonic::Pmaxsb => self.handle_packed_lane_op(
                instruction,
                "pmaxsb",
                PackedLaneOp::MaxS,
                PackedWidth::Byte,
            ),
            Mnemonic::Pminuw => self.handle_packed_lane_op(
                instruction,
                "pminuw",
                PackedLaneOp::MinU,
                PackedWidth::Word,
            ),
            Mnemonic::Pmaxuw => self.handle_packed_lane_op(
                instruction,
                "pmaxuw",
                PackedLaneOp::MaxU,
                PackedWidth::Word,
            ),
            Mnemonic::Pminsd => self.handle_packed_lane_op(
                instruction,
                "pminsd",
                PackedLaneOp::MinS,
                PackedWidth::Dword,
            ),
            Mnemonic::Pmaxsd => self.handle_packed_lane_op(
                instruction,
                "pmaxsd",
                PackedLaneOp::MaxS,
                PackedWidth::Dword,
            ),
            Mnemonic::Pminud => self.handle_packed_lane_op(
                instruction,
                "pminud",
                PackedLaneOp::MinU,
                PackedWidth::Dword,
            ),
            Mnemonic::Pmaxud => self.handle_packed_lane_op(
                instruction,
                "pmaxud",
                PackedLaneOp::MaxU,
                PackedWidth::Dword,
            ),
            Mnemonic::Pmulld => self.handle_packed_lane_op(
                instruction,
                "pmulld",
                PackedLaneOp::MulLow,
                PackedWidth::Dword,
            ),

            // SSSE3 - byte shuffle, concat-rotate, sign ops and horizontal
            // arithmetic.
            Mnemonic::Pshufb => self.handle_pshufb(instruction),
            Mnemonic::Palignr => self.handle_palignr(instruction),
            Mnemonic::Pabsb => self.handle_pabs(instruction, PackedWidth::Byte),
            Mnemonic::Pabsw => self.handle_pabs(instruction, PackedWidth::Word),
            Mnemonic::Pabsd => self.handle_pabs(instruction, PackedWidth::Dword),
            Mnemonic::Psignb => self.handle_psign(instruction, PackedWidth::Byte),
            Mnemonic::Psignw => self.handle_psign(instruction, PackedWidth::Word),
            Mnemonic::Psignd => self.handle_psign(instruction, PackedWidth::Dword),
            Mnemonic::Phaddw => {
                self.handle_horizontal(instruction, HorizOp::Add, PackedWidth::Word)
            }
            Mnemonic::Phaddd => {
                self.handle_horizontal(instruction, HorizOp::Add, PackedWidth::Dword)
            }
            Mnemonic::Phaddsw => {
                self.handle_horizontal(instruction, HorizOp::AddSat, PackedWidth::Word)
            }
            Mnemonic::Phsubw => {
                self.handle_horizontal(instruction, HorizOp::Sub, PackedWidth::Word)
            }
            Mnemonic::Phsubd => {
                self.handle_horizontal(instruction, HorizOp::Sub, PackedWidth::Dword)
            }
            Mnemonic::Phsubsw => {
                self.handle_horizontal(instruction, HorizOp::SubSat, PackedWidth::Word)
            }
            Mnemonic::Pmaddubsw => self.handle_pmaddubsw(instruction),
            Mnemonic::Pmulhrsw => self.handle_pmulhrsw(instruction),

            // SSE2 / SSE4.1 - widening multiply, dot product and SAD.
            Mnemonic::Pmuludq => self.handle_pmuldq(instruction, false),
            Mnemonic::Pmuldq => self.handle_pmuldq(instruction, true),
            Mnemonic::Pmaddwd => self.handle_pmaddwd(instruction),
            Mnemonic::Psadbw => self.handle_psadbw(instruction),

            // SSE2 / SSE4.1 - saturating narrowing packs.
            Mnemonic::Packsswb => self.handle_pack(instruction, PackedWidth::Word, true),
            Mnemonic::Packssdw => self.handle_pack(instruction, PackedWidth::Dword, true),
            Mnemonic::Packuswb => self.handle_pack(instruction, PackedWidth::Word, false),
            Mnemonic::Packusdw => self.handle_pack(instruction, PackedWidth::Dword, false),

            // SSE4.1 - sign / zero extending widening.
            Mnemonic::Pmovzxbw => self.handle_pmovx(instruction, 8, 16, false),
            Mnemonic::Pmovzxbd => self.handle_pmovx(instruction, 8, 32, false),
            Mnemonic::Pmovzxbq => self.handle_pmovx(instruction, 8, 64, false),
            Mnemonic::Pmovzxwd => self.handle_pmovx(instruction, 16, 32, false),
            Mnemonic::Pmovzxwq => self.handle_pmovx(instruction, 16, 64, false),
            Mnemonic::Pmovzxdq => self.handle_pmovx(instruction, 32, 64, false),
            Mnemonic::Pmovsxbw => self.handle_pmovx(instruction, 8, 16, true),
            Mnemonic::Pmovsxbd => self.handle_pmovx(instruction, 8, 32, true),
            Mnemonic::Pmovsxbq => self.handle_pmovx(instruction, 8, 64, true),
            Mnemonic::Pmovsxwd => self.handle_pmovx(instruction, 16, 32, true),
            Mnemonic::Pmovsxwq => self.handle_pmovx(instruction, 16, 64, true),
            Mnemonic::Pmovsxdq => self.handle_pmovx(instruction, 32, 64, true),

            // SSE2 / SSE4.1 - lane insert and extract.
            Mnemonic::Pinsrb => self.handle_pinsr(instruction, PackedWidth::Byte),
            Mnemonic::Pinsrw => self.handle_pinsr(instruction, PackedWidth::Word),
            Mnemonic::Pinsrd => self.handle_pinsr(instruction, PackedWidth::Dword),
            Mnemonic::Pinsrq => self.handle_pinsr(instruction, PackedWidth::Qword),
            Mnemonic::Insertps => self.handle_insertps(instruction),
            Mnemonic::Pextrb => self.handle_pextr(instruction, PackedWidth::Byte),
            Mnemonic::Pextrw => self.handle_pextr(instruction, PackedWidth::Word),
            Mnemonic::Pextrd => self.handle_pextr(instruction, PackedWidth::Dword),
            Mnemonic::Pextrq => self.handle_pextr(instruction, PackedWidth::Qword),

            // SSE4.1 - blends and the 128-bit bitwise test.
            Mnemonic::Pblendw => self.handle_blend_imm(instruction, PackedWidth::Word),
            Mnemonic::Blendps => self.handle_blend_imm(instruction, PackedWidth::Dword),
            Mnemonic::Blendpd => self.handle_blend_imm(instruction, PackedWidth::Qword),
            Mnemonic::Pblendvb => self.handle_blendv(instruction, PackedWidth::Byte),
            Mnemonic::Blendvps => self.handle_blendv(instruction, PackedWidth::Dword),
            Mnemonic::Blendvpd => self.handle_blendv(instruction, PackedWidth::Qword),
            Mnemonic::Ptest => self.handle_ptest(instruction),

            // SSE / SSE2 - sign-bit extraction into a GPR
            Mnemonic::Pmovmskb => self.handle_movmsk(instruction, "pmovmskb", PackedWidth::Byte),
            Mnemonic::Movmskps => self.handle_movmsk(instruction, "movmskps", PackedWidth::Dword),
            Mnemonic::Movmskpd => self.handle_movmsk(instruction, "movmskpd", PackedWidth::Qword),

            // String Operations
            Mnemonic::Movsb => self.handle_movsb(instruction),
            Mnemonic::Movsw => self.handle_movsw(instruction),
            // MOVSD is shared between the string op (no register operands) and
            // SSE2 scalar-double (XMM register operands). Disambiguate by
            // checking whether op0 is a memory-string kind.
            Mnemonic::Movsd => {
                if matches!(
                    instruction.op0_kind(),
                    OpKind::MemoryESDI | OpKind::MemoryESRDI
                ) {
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
            Mnemonic::Insb => self.handle_insb(instruction),
            Mnemonic::Insw => self.handle_insw(instruction),
            Mnemonic::Insd => self.handle_insd(instruction),
            Mnemonic::Outsb => self.handle_outsb(instruction),
            Mnemonic::Outsw => self.handle_outsw(instruction),
            Mnemonic::Outsd => self.handle_outsd(instruction),

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
            Mnemonic::Rcl => self.handle_rcl(instruction),
            Mnemonic::Rcr => self.handle_rcr(instruction),
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
            Mnemonic::Retf => self.handle_retf(instruction),
            Mnemonic::Loop => self.handle_loop(instruction),
            Mnemonic::Loope => self.handle_loope(instruction),
            Mnemonic::Loopne => self.handle_loopne(instruction),

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
            Mnemonic::Int => self.handle_int(instruction),
            Mnemonic::Int1 => self.handle_int1(instruction),

            // System
            Mnemonic::Syscall => self.handle_syscall(instruction),
            Mnemonic::Cpuid => self.handle_cpuid(instruction),
            Mnemonic::Xgetbv => self.handle_xgetbv(instruction),
            Mnemonic::Rdtsc => self.handle_rdtsc(instruction),

            // FS/GS base (FSGSBASE extension)
            Mnemonic::Rdfsbase => self.handle_rdfsbase(instruction),
            Mnemonic::Wrfsbase => self.handle_wrfsbase(instruction),
            Mnemonic::Rdgsbase => self.handle_rdgsbase(instruction),
            Mnemonic::Wrgsbase => self.handle_wrgsbase(instruction),

            // x87 FPU - loads / stores
            Mnemonic::Fld => self.handle_fld(instruction),
            Mnemonic::Fild => self.handle_fild(instruction),
            Mnemonic::Fldz => self.handle_fld_const(0.0),
            Mnemonic::Fld1 => self.handle_fld_const(1.0),
            Mnemonic::Fldpi => self.handle_fld_const(std::f64::consts::PI),
            Mnemonic::Fst => self.handle_fst(instruction, false),
            Mnemonic::Fstp => self.handle_fst(instruction, true),
            Mnemonic::Fist => self.handle_fist(instruction, false, false),
            Mnemonic::Fistp => self.handle_fist(instruction, true, false),
            Mnemonic::Fisttp => self.handle_fist(instruction, true, true),
            Mnemonic::Fxch => self.handle_fxch(instruction),

            // x87 FPU - arithmetic
            Mnemonic::Fadd => self.handle_farith(instruction, FpuOp::Add, false),
            Mnemonic::Faddp => self.handle_farith(instruction, FpuOp::Add, true),
            Mnemonic::Fsub => self.handle_farith(instruction, FpuOp::Sub, false),
            Mnemonic::Fsubp => self.handle_farith(instruction, FpuOp::Sub, true),
            Mnemonic::Fsubr => self.handle_farith(instruction, FpuOp::Subr, false),
            Mnemonic::Fsubrp => self.handle_farith(instruction, FpuOp::Subr, true),
            Mnemonic::Fmul => self.handle_farith(instruction, FpuOp::Mul, false),
            Mnemonic::Fmulp => self.handle_farith(instruction, FpuOp::Mul, true),
            Mnemonic::Fdiv => self.handle_farith(instruction, FpuOp::Div, false),
            Mnemonic::Fdivp => self.handle_farith(instruction, FpuOp::Div, true),
            Mnemonic::Fdivr => self.handle_farith(instruction, FpuOp::Divr, false),
            Mnemonic::Fdivrp => self.handle_farith(instruction, FpuOp::Divr, true),
            Mnemonic::Fabs => self.handle_fabs(instruction),
            Mnemonic::Fchs => self.handle_fchs(instruction),
            Mnemonic::Fsqrt => self.handle_fsqrt(instruction),

            // x87 FPU - comparisons
            Mnemonic::Fcom => self.handle_fcom(instruction, false),
            Mnemonic::Fcomp => self.handle_fcom(instruction, true),
            Mnemonic::Fucom => self.handle_fcom(instruction, false),
            Mnemonic::Fucomp => self.handle_fcom(instruction, true),
            Mnemonic::Ftst => self.handle_ftst(instruction),
            Mnemonic::Fcomi => self.handle_fcomi(instruction, false),
            Mnemonic::Fcomip => self.handle_fcomi(instruction, true),
            Mnemonic::Fucomi => self.handle_fcomi(instruction, false),
            Mnemonic::Fucomip => self.handle_fcomi(instruction, true),

            // x87 FPU - conditional moves
            Mnemonic::Fcmovb => self.handle_fcmovb(instruction),
            Mnemonic::Fcmovnb => self.handle_fcmovnb(instruction),
            Mnemonic::Fcmove => self.handle_fcmove(instruction),
            Mnemonic::Fcmovne => self.handle_fcmovne(instruction),
            Mnemonic::Fcmovbe => self.handle_fcmovbe(instruction),
            Mnemonic::Fcmovnbe => self.handle_fcmovnbe(instruction),
            Mnemonic::Fcmovu => self.handle_fcmovu(instruction),
            Mnemonic::Fcmovnu => self.handle_fcmovnu(instruction),

            // x87 FPU - control / misc
            Mnemonic::Fnstsw => self.handle_fnstsw(instruction),
            Mnemonic::Fnstcw => self.handle_fnstcw(instruction),
            Mnemonic::Fldcw => self.handle_fldcw(instruction),
            Mnemonic::Fincstp => self.handle_fincstp(instruction),
            Mnemonic::Fdecstp => self.handle_fdecstp(instruction),
            Mnemonic::Ffree => self.handle_ffree(instruction),
            Mnemonic::Fnop => self.handle_fpu_nop(instruction),

            Mnemonic::Stc => self.handle_stc(instruction),
            Mnemonic::Clc => self.handle_clc(instruction),
            Mnemonic::Cmc => self.handle_cmc(instruction),
            Mnemonic::Std => self.handle_std(instruction),
            Mnemonic::Cld => self.handle_cld(instruction),

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
