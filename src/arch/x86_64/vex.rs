//! VEX-128 lowering onto the legacy SSE semantics.
//!
//! A VEX-128 instruction is its legacy SSE form with a non-destructive first source:
//! `vop dst, src1, rest..` is `dst = src1; op dst, rest..`. The zeroing of `YMM[255:128]`
//! has nothing to act on, since no upper YMM state is modelled. VEX-256 has no lowering.

use iced_x86::{Code, Instruction};

/// How a VEX instruction is carried out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VexLowering {
    /// Operands line up with the legacy form one-for-one.
    Same(Code),
    /// Copy operand 1 into operand 0, then run the legacy form on the remaining operands.
    Merge(Code),
    ZeroUpper,
    ZeroAll,
}

pub fn vex_lowering(instruction: &Instruction) -> Option<VexLowering> {
    match instruction.code() {
        Code::VEX_Vzeroupper => Some(VexLowering::ZeroUpper),
        Code::VEX_Vzeroall => Some(VexLowering::ZeroAll),
        code => {
            let legacy = legacy_code(code)?;
            let (vex_ops, legacy_ops) = (code.op_code().op_count(), legacy.op_code().op_count());
            let merges = vex_ops == legacy_ops + 1;
            debug_assert!(merges || vex_ops == legacy_ops);
            Some(if merges {
                VexLowering::Merge(legacy)
            } else {
                VexLowering::Same(legacy)
            })
        }
    }
}

/// The legacy instruction a VEX one lowers to, operands already shifted into place.
pub fn legacy_form(instruction: &Instruction, lowering: VexLowering) -> Option<Instruction> {
    let mut legacy = *instruction;
    match lowering {
        VexLowering::Same(code) => legacy.set_code(code),
        VexLowering::Merge(code) => {
            legacy.set_code(code);
            for op in 1..legacy.op_count() {
                legacy.set_op_kind(op, instruction.op_kind(op + 1));
                legacy.set_op_register(op, instruction.op_register(op + 1));
            }
        }
        VexLowering::ZeroUpper | VexLowering::ZeroAll => return None,
    }
    Some(legacy)
}

fn legacy_code(code: Code) -> Option<Code> {
    use Code::*;
    Some(match code {
        VEX_Vmovdqa_xmm_xmmm128 => Movdqa_xmm_xmmm128,
        VEX_Vmovdqa_xmmm128_xmm => Movdqa_xmmm128_xmm,
        VEX_Vmovdqu_xmm_xmmm128 => Movdqu_xmm_xmmm128,
        VEX_Vmovdqu_xmmm128_xmm => Movdqu_xmmm128_xmm,
        VEX_Vmovaps_xmm_xmmm128 => Movaps_xmm_xmmm128,
        VEX_Vmovaps_xmmm128_xmm => Movaps_xmmm128_xmm,
        VEX_Vmovups_xmm_xmmm128 => Movups_xmm_xmmm128,
        VEX_Vmovups_xmmm128_xmm => Movups_xmmm128_xmm,
        VEX_Vmovd_xmm_rm32 => Movd_xmm_rm32,
        VEX_Vmovd_rm32_xmm => Movd_rm32_xmm,
        VEX_Vmovq_xmm_rm64 => Movq_xmm_rm64,
        VEX_Vmovq_rm64_xmm => Movq_rm64_xmm,
        VEX_Vmovq_xmm_xmmm64 => Movq_xmm_xmmm64,
        VEX_Vmovq_xmmm64_xmm => Movq_xmmm64_xmm,

        VEX_Vpxor_xmm_xmm_xmmm128 => Pxor_xmm_xmmm128,
        VEX_Vpor_xmm_xmm_xmmm128 => Por_xmm_xmmm128,
        VEX_Vpand_xmm_xmm_xmmm128 => Pand_xmm_xmmm128,
        VEX_Vpandn_xmm_xmm_xmmm128 => Pandn_xmm_xmmm128,
        VEX_Vxorps_xmm_xmm_xmmm128 => Xorps_xmm_xmmm128,

        VEX_Vpaddb_xmm_xmm_xmmm128 => Paddb_xmm_xmmm128,
        VEX_Vpaddw_xmm_xmm_xmmm128 => Paddw_xmm_xmmm128,
        VEX_Vpaddd_xmm_xmm_xmmm128 => Paddd_xmm_xmmm128,
        VEX_Vpaddq_xmm_xmm_xmmm128 => Paddq_xmm_xmmm128,
        VEX_Vpsubb_xmm_xmm_xmmm128 => Psubb_xmm_xmmm128,
        VEX_Vpsubw_xmm_xmm_xmmm128 => Psubw_xmm_xmmm128,
        VEX_Vpsubd_xmm_xmm_xmmm128 => Psubd_xmm_xmmm128,
        VEX_Vpsubq_xmm_xmm_xmmm128 => Psubq_xmm_xmmm128,
        VEX_Vpmuludq_xmm_xmm_xmmm128 => Pmuludq_xmm_xmmm128,
        VEX_Vpclmulqdq_xmm_xmm_xmmm128_imm8 => Pclmulqdq_xmm_xmmm128_imm8,

        VEX_Vpsrlw_xmm_xmm_xmmm128 => Psrlw_xmm_xmmm128,
        VEX_Vpsrld_xmm_xmm_xmmm128 => Psrld_xmm_xmmm128,
        VEX_Vpsrlq_xmm_xmm_xmmm128 => Psrlq_xmm_xmmm128,
        VEX_Vpsllw_xmm_xmm_xmmm128 => Psllw_xmm_xmmm128,
        VEX_Vpslld_xmm_xmm_xmmm128 => Pslld_xmm_xmmm128,
        VEX_Vpsllq_xmm_xmm_xmmm128 => Psllq_xmm_xmmm128,
        VEX_Vpsraw_xmm_xmm_xmmm128 => Psraw_xmm_xmmm128,
        VEX_Vpsrad_xmm_xmm_xmmm128 => Psrad_xmm_xmmm128,
        VEX_Vpsrlw_xmm_xmm_imm8 => Psrlw_xmm_imm8,
        VEX_Vpsrld_xmm_xmm_imm8 => Psrld_xmm_imm8,
        VEX_Vpsrlq_xmm_xmm_imm8 => Psrlq_xmm_imm8,
        VEX_Vpsllw_xmm_xmm_imm8 => Psllw_xmm_imm8,
        VEX_Vpslld_xmm_xmm_imm8 => Pslld_xmm_imm8,
        VEX_Vpsllq_xmm_xmm_imm8 => Psllq_xmm_imm8,
        VEX_Vpsraw_xmm_xmm_imm8 => Psraw_xmm_imm8,
        VEX_Vpsrad_xmm_xmm_imm8 => Psrad_xmm_imm8,
        VEX_Vpsrldq_xmm_xmm_imm8 => Psrldq_xmm_imm8,
        VEX_Vpslldq_xmm_xmm_imm8 => Pslldq_xmm_imm8,

        VEX_Vpshufb_xmm_xmm_xmmm128 => Pshufb_xmm_xmmm128,
        VEX_Vpshufd_xmm_xmmm128_imm8 => Pshufd_xmm_xmmm128_imm8,
        VEX_Vpalignr_xmm_xmm_xmmm128_imm8 => Palignr_xmm_xmmm128_imm8,
        VEX_Vpunpcklbw_xmm_xmm_xmmm128 => Punpcklbw_xmm_xmmm128,
        VEX_Vpunpcklwd_xmm_xmm_xmmm128 => Punpcklwd_xmm_xmmm128,
        VEX_Vpunpckldq_xmm_xmm_xmmm128 => Punpckldq_xmm_xmmm128,
        VEX_Vpunpcklqdq_xmm_xmm_xmmm128 => Punpcklqdq_xmm_xmmm128,
        VEX_Vpunpckhbw_xmm_xmm_xmmm128 => Punpckhbw_xmm_xmmm128,
        VEX_Vpunpckhwd_xmm_xmm_xmmm128 => Punpckhwd_xmm_xmmm128,
        VEX_Vpunpckhdq_xmm_xmm_xmmm128 => Punpckhdq_xmm_xmmm128,
        VEX_Vpunpckhqdq_xmm_xmm_xmmm128 => Punpckhqdq_xmm_xmmm128,

        VEX_Vpcmpeqb_xmm_xmm_xmmm128 => Pcmpeqb_xmm_xmmm128,
        VEX_Vpcmpeqw_xmm_xmm_xmmm128 => Pcmpeqw_xmm_xmmm128,
        VEX_Vpcmpeqd_xmm_xmm_xmmm128 => Pcmpeqd_xmm_xmmm128,
        VEX_Vpcmpeqq_xmm_xmm_xmmm128 => Pcmpeqq_xmm_xmmm128,
        VEX_Vpcmpgtb_xmm_xmm_xmmm128 => Pcmpgtb_xmm_xmmm128,
        VEX_Vpcmpgtw_xmm_xmm_xmmm128 => Pcmpgtw_xmm_xmmm128,
        VEX_Vpcmpgtd_xmm_xmm_xmmm128 => Pcmpgtd_xmm_xmmm128,
        VEX_Vpcmpgtq_xmm_xmm_xmmm128 => Pcmpgtq_xmm_xmmm128,

        VEX_Vpinsrb_xmm_xmm_r32m8_imm8 => Pinsrb_xmm_r32m8_imm8,
        VEX_Vpinsrw_xmm_xmm_r32m16_imm8 => Pinsrw_xmm_r32m16_imm8,
        VEX_Vpinsrd_xmm_xmm_rm32_imm8 => Pinsrd_xmm_rm32_imm8,
        VEX_Vpinsrq_xmm_xmm_rm64_imm8 => Pinsrq_xmm_rm64_imm8,
        VEX_Vpextrb_r32m8_xmm_imm8 => Pextrb_r32m8_xmm_imm8,
        VEX_Vpextrw_r32_xmm_imm8 => Pextrw_r32_xmm_imm8,
        VEX_Vpextrw_r32m16_xmm_imm8 => Pextrw_r32m16_xmm_imm8,
        VEX_Vpextrd_rm32_xmm_imm8 => Pextrd_rm32_xmm_imm8,
        VEX_Vpextrq_rm64_xmm_imm8 => Pextrq_rm64_xmm_imm8,

        VEX_Vaesenc_xmm_xmm_xmmm128 => Aesenc_xmm_xmmm128,
        VEX_Vaesenclast_xmm_xmm_xmmm128 => Aesenclast_xmm_xmmm128,
        VEX_Vaesdec_xmm_xmm_xmmm128 => Aesdec_xmm_xmmm128,
        VEX_Vaesdeclast_xmm_xmm_xmmm128 => Aesdeclast_xmm_xmmm128,
        VEX_Vaesimc_xmm_xmmm128 => Aesimc_xmm_xmmm128,
        VEX_Vaeskeygenassist_xmm_xmmm128_imm8 => Aeskeygenassist_xmm_xmmm128_imm8,
        _ => return None,
    })
}
