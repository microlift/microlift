//! Legacy x86 flag-effect metadata for callers inspecting instructions.
//!
//! Dead flag elimination does not consult this table; it follows SSA dependencies after lifting.
//!
use iced_x86::{Instruction, Mnemonic, Register, RflagsBits};

use crate::arch::x86_64::regs::X86Flag;
use crate::core::flags::{FlagEffects, FlagSet};

/// One flag, as a set; the bit index is the `X86Flag` discriminant, which the semantics also pass to
/// [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live).
const fn f(flag: X86Flag) -> FlagSet {
    FlagSet::bit(flag as u8)
}

const CF: FlagSet = f(X86Flag::Carry);
const PF: FlagSet = f(X86Flag::Parity);
const AF: FlagSet = f(X86Flag::Auxiliary);
const ZF: FlagSet = f(X86Flag::Zero);
const SF: FlagSet = f(X86Flag::Sign);
const OF: FlagSet = f(X86Flag::Overflow);
const DF: FlagSet = f(X86Flag::Direction);

const ARITH: FlagSet = CF.union(PF).union(AF).union(ZF).union(SF).union(OF);
const LOGICAL: FlagSet = CF.union(PF).union(ZF).union(SF).union(OF);
const INC_DEC: FlagSet = PF.union(AF).union(ZF).union(SF).union(OF);
const SHIFT: FlagSet = CF.union(PF).union(ZF).union(SF).union(OF);

/// The flag effects of lifting `i`, or `None` where unaudited ([`FlagEffects::CONSERVATIVE`] to callers).
pub fn flag_effects(i: &Instruction) -> Option<FlagEffects> {
    // Routes to a handler this table was not read against.
    if i.has_lock_prefix()
        || i.has_rep_prefix()
        || i.has_repne_prefix()
        || uses_mmx_register(i)
        || i.is_invalid()
    {
        return None;
    }

    let effects = match i.mnemonic() {
        // `CMP` is a `SUB` discarding its result; `NEG` is `0 - x`; `XADD`'s flag half is an `ADD`.
        Mnemonic::Add | Mnemonic::Xadd | Mnemonic::Sub | Mnemonic::Cmp | Mnemonic::Neg => {
            FlagEffects::writing(ARITH)
        }
        // Carry-in is folded into the result before the flag update.
        Mnemonic::Adc | Mnemonic::Sbb => FlagEffects {
            reads: CF,
            writes: ARITH,
        },
        // CF is untouched.
        Mnemonic::Inc | Mnemonic::Dec => FlagEffects::writing(INC_DEC),

        // AF is left as it was, so not a write.
        Mnemonic::And | Mnemonic::Or | Mnemonic::Xor | Mnemonic::Test => {
            FlagEffects::writing(LOGICAL)
        }

        // A zero count keeps the old flags, so they are read too.
        Mnemonic::Shl
        | Mnemonic::Sal
        | Mnemonic::Shr
        | Mnemonic::Sar
        | Mnemonic::Shld
        | Mnemonic::Shrd => FlagEffects {
            reads: SHIFT,
            writes: SHIFT,
        },

        Mnemonic::Imul | Mnemonic::Mul => FlagEffects::writing(CF.union(OF)),

        // `bt_common` sets CF only; BSF/BSR set ZF only.
        Mnemonic::Bt | Mnemonic::Bts | Mnemonic::Btr | Mnemonic::Btc => FlagEffects::writing(CF),
        Mnemonic::Bsf | Mnemonic::Bsr => FlagEffects::writing(ZF),
        Mnemonic::Tzcnt | Mnemonic::Lzcnt => FlagEffects::writing(CF.union(ZF)),
        // POPCNT sets ZF and clears the other five, which is a write of each.
        Mnemonic::Popcnt => FlagEffects::writing(ARITH),
        // CF/ZF/SF/OF from the comparison, AF/PF cleared; all six assigned.
        Mnemonic::Pcmpistri
        | Mnemonic::Pcmpistrm
        | Mnemonic::Pcmpestri
        | Mnemonic::Pcmpestrm
        | Mnemonic::Pcmpestri64
        | Mnemonic::Pcmpestrm64 => FlagEffects::writing(ARITH),

        Mnemonic::Stc | Mnemonic::Clc => FlagEffects::writing(CF),
        Mnemonic::Cmc => FlagEffects {
            reads: CF,
            writes: CF,
        },
        Mnemonic::Std | Mnemonic::Cld => FlagEffects::writing(DF),

        // Reads exactly the flags of its condition (iced answers per-condition); `cmp; je` leaves
        // five of CMP's six flags with no reader. Safe: reads are a superset requirement.
        m if is_condition_consumer(m) => FlagEffects::reading(from_rflags(i.rflags_read())),

        // Last, so hand-written entries win an overlap. Transparency lets liveness see through `mov`.
        m if is_flag_free(m) => FlagEffects::TRANSPARENT,

        // Everything else, including VM exits and unimplemented instructions.
        _ => return None,
    };
    Some(effects)
}

/// Mnemonics whose lift provably touches no flag and never syncs to the host.
///
/// **Generated** by `tools/derive-flag-free.py`; rerun after touching `semantics/`.
/// A handler is tainted if its body mentions `set_flag`, `state.flags`, `flag_needed`,
/// `sync_to_context`, `write_exit_reason`, `write_exit_code`, `resumes_at` or `handle_unimplemented`,
/// propagated through **any mention** of another function's name (`MOVSB` reaches DF via a
/// `Self::emit_movs` value; a call-shaped search would miscompile it).
///
/// A handler that gains a `set_flag` while staying listed is the miscompile direction.
fn is_flag_free(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Addpd
            | Mnemonic::Addps
            | Mnemonic::Addsd
            | Mnemonic::Addss
            | Mnemonic::Addsubpd
            | Mnemonic::Addsubps
            | Mnemonic::Aesdec
            | Mnemonic::Aesdeclast
            | Mnemonic::Aesenc
            | Mnemonic::Aesenclast
            | Mnemonic::Aesimc
            | Mnemonic::Aeskeygenassist
            | Mnemonic::Andnpd
            | Mnemonic::Andnps
            | Mnemonic::Andpd
            | Mnemonic::Andps
            | Mnemonic::Blendpd
            | Mnemonic::Blendps
            | Mnemonic::Blendvpd
            | Mnemonic::Blendvps
            | Mnemonic::Bswap
            | Mnemonic::Cbw
            | Mnemonic::Cdq
            | Mnemonic::Cdqe
            | Mnemonic::Clflush
            | Mnemonic::Clflushopt
            | Mnemonic::Clwb
            | Mnemonic::Cmppd
            | Mnemonic::Cmpps
            | Mnemonic::Cmpss
            | Mnemonic::Cqo
            | Mnemonic::Crc32
            | Mnemonic::Cvtdq2pd
            | Mnemonic::Cvtdq2ps
            | Mnemonic::Cvtpd2dq
            | Mnemonic::Cvtpd2ps
            | Mnemonic::Cvtps2dq
            | Mnemonic::Cvtps2pd
            | Mnemonic::Cvtsd2si
            | Mnemonic::Cvtsd2ss
            | Mnemonic::Cvtsi2sd
            | Mnemonic::Cvtsi2ss
            | Mnemonic::Cvtss2sd
            | Mnemonic::Cvtss2si
            | Mnemonic::Cvttpd2dq
            | Mnemonic::Cvttps2dq
            | Mnemonic::Cvttsd2si
            | Mnemonic::Cvttss2si
            | Mnemonic::Cwd
            | Mnemonic::Cwde
            | Mnemonic::Divpd
            | Mnemonic::Divps
            | Mnemonic::Divsd
            | Mnemonic::Divss
            | Mnemonic::Dppd
            | Mnemonic::Dpps
            | Mnemonic::Endbr32
            | Mnemonic::Endbr64
            | Mnemonic::Extractps
            | Mnemonic::Fabs
            | Mnemonic::Fadd
            | Mnemonic::Faddp
            | Mnemonic::Fchs
            | Mnemonic::Fcom
            | Mnemonic::Fcomp
            | Mnemonic::Fcos
            | Mnemonic::Fdecstp
            | Mnemonic::Fdiv
            | Mnemonic::Fdivp
            | Mnemonic::Fdivr
            | Mnemonic::Fdivrp
            | Mnemonic::Ffree
            | Mnemonic::Fild
            | Mnemonic::Fincstp
            | Mnemonic::Fist
            | Mnemonic::Fistp
            | Mnemonic::Fisttp
            | Mnemonic::Fld
            | Mnemonic::Fld1
            | Mnemonic::Fldcw
            | Mnemonic::Fldenv
            | Mnemonic::Fldpi
            | Mnemonic::Fldz
            | Mnemonic::Fmul
            | Mnemonic::Fmulp
            | Mnemonic::Fnclex
            | Mnemonic::Fninit
            | Mnemonic::Fnop
            | Mnemonic::Fnstcw
            | Mnemonic::Fnstenv
            | Mnemonic::Fnstsw
            | Mnemonic::Fptan
            | Mnemonic::Fsin
            | Mnemonic::Fsincos
            | Mnemonic::Fsqrt
            | Mnemonic::Fst
            | Mnemonic::Fstp
            | Mnemonic::Fsub
            | Mnemonic::Fsubp
            | Mnemonic::Fsubr
            | Mnemonic::Fsubrp
            | Mnemonic::Ftst
            | Mnemonic::Fucom
            | Mnemonic::Fucomp
            | Mnemonic::Fxch
            | Mnemonic::Haddpd
            | Mnemonic::Haddps
            | Mnemonic::Hsubpd
            | Mnemonic::Hsubps
            | Mnemonic::Insertps
            | Mnemonic::Jrcxz
            | Mnemonic::Lddqu
            | Mnemonic::Ldmxcsr
            | Mnemonic::Lea
            | Mnemonic::Leave
            | Mnemonic::Lfence
            | Mnemonic::Loop
            | Mnemonic::Maxpd
            | Mnemonic::Maxps
            | Mnemonic::Maxsd
            | Mnemonic::Maxss
            | Mnemonic::Mfence
            | Mnemonic::Minpd
            | Mnemonic::Minps
            | Mnemonic::Minsd
            | Mnemonic::Minss
            | Mnemonic::Mov
            | Mnemonic::Movd
            | Mnemonic::Movddup
            | Mnemonic::Movhlps
            | Mnemonic::Movhpd
            | Mnemonic::Movhps
            | Mnemonic::Movlhps
            | Mnemonic::Movlpd
            | Mnemonic::Movlps
            | Mnemonic::Movmskpd
            | Mnemonic::Movmskps
            | Mnemonic::Movntdqa
            | Mnemonic::Movq
            | Mnemonic::Movshdup
            | Mnemonic::Movsldup
            | Mnemonic::Movss
            | Mnemonic::Movsx
            | Mnemonic::Movsxd
            | Mnemonic::Movzx
            | Mnemonic::Mpsadbw
            | Mnemonic::Mulpd
            | Mnemonic::Mulps
            | Mnemonic::Mulsd
            | Mnemonic::Mulss
            | Mnemonic::Nop
            | Mnemonic::Not
            | Mnemonic::Orpd
            | Mnemonic::Orps
            | Mnemonic::Pabsb
            | Mnemonic::Pabsd
            | Mnemonic::Pabsw
            | Mnemonic::Packssdw
            | Mnemonic::Packsswb
            | Mnemonic::Packusdw
            | Mnemonic::Packuswb
            | Mnemonic::Paddb
            | Mnemonic::Paddd
            | Mnemonic::Paddq
            | Mnemonic::Paddsb
            | Mnemonic::Paddsw
            | Mnemonic::Paddusb
            | Mnemonic::Paddusw
            | Mnemonic::Paddw
            | Mnemonic::Palignr
            | Mnemonic::Pand
            | Mnemonic::Pandn
            | Mnemonic::Pause
            | Mnemonic::Pavgb
            | Mnemonic::Pavgw
            | Mnemonic::Pblendvb
            | Mnemonic::Pblendw
            | Mnemonic::Pcmpeqb
            | Mnemonic::Pcmpeqd
            | Mnemonic::Pcmpeqq
            | Mnemonic::Pcmpeqw
            | Mnemonic::Pcmpgtb
            | Mnemonic::Pcmpgtd
            | Mnemonic::Pcmpgtq
            | Mnemonic::Pcmpgtw
            | Mnemonic::Pextrb
            | Mnemonic::Pextrd
            | Mnemonic::Pextrq
            | Mnemonic::Pextrw
            | Mnemonic::Phaddd
            | Mnemonic::Phaddsw
            | Mnemonic::Phaddw
            | Mnemonic::Phminposuw
            | Mnemonic::Phsubd
            | Mnemonic::Phsubsw
            | Mnemonic::Phsubw
            | Mnemonic::Pinsrb
            | Mnemonic::Pinsrd
            | Mnemonic::Pinsrq
            | Mnemonic::Pinsrw
            | Mnemonic::Pmaddubsw
            | Mnemonic::Pmaddwd
            | Mnemonic::Pmaxsb
            | Mnemonic::Pmaxsd
            | Mnemonic::Pmaxsw
            | Mnemonic::Pmaxub
            | Mnemonic::Pmaxud
            | Mnemonic::Pmaxuw
            | Mnemonic::Pminsb
            | Mnemonic::Pminsd
            | Mnemonic::Pminsw
            | Mnemonic::Pminub
            | Mnemonic::Pminud
            | Mnemonic::Pminuw
            | Mnemonic::Pmovmskb
            | Mnemonic::Pmovsxbd
            | Mnemonic::Pmovsxbq
            | Mnemonic::Pmovsxbw
            | Mnemonic::Pmovsxdq
            | Mnemonic::Pmovsxwd
            | Mnemonic::Pmovsxwq
            | Mnemonic::Pmovzxbd
            | Mnemonic::Pmovzxbq
            | Mnemonic::Pmovzxbw
            | Mnemonic::Pmovzxdq
            | Mnemonic::Pmovzxwd
            | Mnemonic::Pmovzxwq
            | Mnemonic::Pmuldq
            | Mnemonic::Pmulhrsw
            | Mnemonic::Pmulhuw
            | Mnemonic::Pmulhw
            | Mnemonic::Pmulld
            | Mnemonic::Pmullw
            | Mnemonic::Pmuludq
            | Mnemonic::Pop
            | Mnemonic::Por
            | Mnemonic::Prefetch
            | Mnemonic::Prefetchit0
            | Mnemonic::Prefetchit1
            | Mnemonic::Prefetchnta
            | Mnemonic::Prefetcht0
            | Mnemonic::Prefetcht1
            | Mnemonic::Prefetcht2
            | Mnemonic::Prefetchw
            | Mnemonic::Prefetchwt1
            | Mnemonic::Psadbw
            | Mnemonic::Pshufb
            | Mnemonic::Pshufd
            | Mnemonic::Pshufhw
            | Mnemonic::Pshuflw
            | Mnemonic::Psignb
            | Mnemonic::Psignd
            | Mnemonic::Psignw
            | Mnemonic::Pslld
            | Mnemonic::Pslldq
            | Mnemonic::Psllq
            | Mnemonic::Psllw
            | Mnemonic::Psrad
            | Mnemonic::Psraw
            | Mnemonic::Psrld
            | Mnemonic::Psrldq
            | Mnemonic::Psrlq
            | Mnemonic::Psrlw
            | Mnemonic::Psubb
            | Mnemonic::Psubd
            | Mnemonic::Psubq
            | Mnemonic::Psubsb
            | Mnemonic::Psubsw
            | Mnemonic::Psubusb
            | Mnemonic::Psubusw
            | Mnemonic::Psubw
            | Mnemonic::Punpckhbw
            | Mnemonic::Punpckhdq
            | Mnemonic::Punpckhqdq
            | Mnemonic::Punpckhwd
            | Mnemonic::Punpcklbw
            | Mnemonic::Punpckldq
            | Mnemonic::Punpcklqdq
            | Mnemonic::Punpcklwd
            | Mnemonic::Push
            | Mnemonic::Pxor
            | Mnemonic::Rcpps
            | Mnemonic::Rcpss
            | Mnemonic::Rdfsbase
            | Mnemonic::Rdgsbase
            | Mnemonic::Reservednop
            | Mnemonic::Roundpd
            | Mnemonic::Roundsd
            | Mnemonic::Roundss
            | Mnemonic::Rsqrtps
            | Mnemonic::Rsqrtss
            | Mnemonic::Serialize
            | Mnemonic::Sfence
            | Mnemonic::Shufpd
            | Mnemonic::Shufps
            | Mnemonic::Sqrtpd
            | Mnemonic::Sqrtps
            | Mnemonic::Sqrtsd
            | Mnemonic::Sqrtss
            | Mnemonic::Stmxcsr
            | Mnemonic::Subpd
            | Mnemonic::Subps
            | Mnemonic::Subsd
            | Mnemonic::Subss
            | Mnemonic::Unpckhpd
            | Mnemonic::Unpckhps
            | Mnemonic::Unpcklpd
            | Mnemonic::Unpcklps
            | Mnemonic::Vldmxcsr
            | Mnemonic::Vstmxcsr
            | Mnemonic::Wait
            | Mnemonic::Wrfsbase
            | Mnemonic::Wrgsbase
            | Mnemonic::Xchg
            | Mnemonic::Xorpd
            | Mnemonic::Xorps
    )
}

/// A `Jcc`, `SETcc` or `CMOVcc`: its only flag interaction is evaluating a condition.
/// `JRCXZ`/`LOOP*` are absent: their handlers also decrement `RCX`, so they stay conservative.
fn is_condition_consumer(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Ja | Jae
            | Jb
            | Jbe
            | Je
            | Jg
            | Jge
            | Jl
            | Jle
            | Jne
            | Jno
            | Jnp
            | Jns
            | Jo
            | Jp
            | Js
            | Seta
            | Setae
            | Setb
            | Setbe
            | Sete
            | Setg
            | Setge
            | Setl
            | Setle
            | Setne
            | Setno
            | Setnp
            | Setns
            | Seto
            | Setp
            | Sets
            | Cmova
            | Cmovae
            | Cmovb
            | Cmovbe
            | Cmove
            | Cmovg
            | Cmovge
            | Cmovl
            | Cmovle
            | Cmovne
            | Cmovno
            | Cmovnp
            | Cmovns
            | Cmovo
            | Cmovp
            | Cmovs
    )
}

/// Translates an iced `RflagsBits` mask to [`FlagSet`]; IF and system flags have no SSA variable and are dropped.
fn from_rflags(bits: u32) -> FlagSet {
    let mut set = FlagSet::NONE;
    for (bit, flag) in [
        (RflagsBits::CF, CF),
        (RflagsBits::PF, PF),
        (RflagsBits::AF, AF),
        (RflagsBits::ZF, ZF),
        (RflagsBits::SF, SF),
        (RflagsBits::OF, OF),
        (RflagsBits::DF, DF),
    ] {
        if bits & bit != 0 {
            set = set.union(flag);
        }
    }
    set
}

/// MMX operands exit as unimplemented before semantics dispatch; mirrors `semantics::uses_mmx_register`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions};

    fn decode(bytes: &[u8]) -> Instruction {
        let mut decoder = Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE);
        decoder.decode()
    }

    /// Pins entries that differ from the architecture; taking iced's answer would miscompile each.
    #[test]
    fn the_lift_and_the_manual_disagree_in_three_places() {
        // AF is undefined here and `update_flags_logical` leaves it, so not a write.
        let and = flag_effects(&decode(&[0x21, 0xC0])).expect("AND is classified");
        assert!(!and.writes.contains(X86Flag::Auxiliary as u8));
        assert!(and.writes.contains(X86Flag::Zero as u8));

        // Handler assigns CF and OF only.
        let imul = flag_effects(&decode(&[0x0F, 0xAF, 0xC0])).expect("IMUL is classified");
        assert_eq!(imul.writes, CF.union(OF));

        // Reads the flags it may have to keep.
        let shl = flag_effects(&decode(&[0xD3, 0xE0])).expect("SHL is classified");
        assert_eq!(shl.reads, SHIFT);
        assert_eq!(shl.writes, SHIFT);
    }

    /// A `Jcc` reads only its condition's flags.
    #[test]
    fn a_conditional_branch_reads_only_its_condition() {
        let je = flag_effects(&decode(&[0x74, 0x00])).expect("JE is classified");
        assert_eq!(je.reads, ZF);
        assert_eq!(je.writes, FlagSet::NONE);

        // CF and ZF.
        let jbe = flag_effects(&decode(&[0x76, 0x00])).expect("JBE is classified");
        assert_eq!(jbe.reads, CF.union(ZF));

        // ZF, SF, OF.
        let jg = flag_effects(&decode(&[0x7F, 0x00])).expect("JG is classified");
        assert_eq!(jg.reads, ZF.union(SF).union(OF));
    }

    /// Whole-instruction short-circuits that route to unaudited handlers.
    #[test]
    fn a_different_handler_is_not_this_table() {
        assert!(flag_effects(&decode(&[0xF0, 0x01, 0x08])).is_none());
        // String op, not the unlisted `STOS`.
        assert!(flag_effects(&decode(&[0xF3, 0xAA])).is_none());
        // Exits before semantics dispatch.
        assert!(flag_effects(&decode(&[0x0F, 0xFC, 0xC1])).is_none());
    }

    /// Host-handoff instructions must stay unclassified: their sync publishes every flag.
    #[test]
    fn everything_that_exits_is_unclassified() {
        for bytes in [
            &[0x0F, 0xA2][..],
            &[0x0F, 0x05][..],
            &[0x0F, 0x31][..],
            &[0xCC][..],
            &[0x0F, 0x0B][..],
            &[0xF4][..],
            &[0xEC][..],
        ] {
            let inst = decode(bytes);
            assert!(
                flag_effects(&inst).is_none(),
                "{:?} is classified but exits to the host",
                inst.mnemonic()
            );
        }
    }

    /// A flag-free `MOV` must be transparent or liveness stops at the first one.
    #[test]
    fn data_movement_is_transparent() {
        let mov = flag_effects(&decode(&[0x48, 0x89, 0xD8])).expect("MOV is classified");
        assert_eq!(mov, FlagEffects::TRANSPARENT);
        // FP arithmetic: `addsd xmm0, xmm1`, `cvtsi2sd xmm0, rax`.
        for bytes in [
            &[0xF2, 0x0F, 0x58, 0xC1][..],
            &[0xF2, 0x48, 0x0F, 0x2A, 0xC0][..],
        ] {
            let inst = decode(bytes);
            assert_eq!(
                flag_effects(&inst),
                Some(FlagEffects::TRANSPARENT),
                "{:?} is not transparent",
                inst.mnemonic()
            );
        }
    }

    /// Generated and hand-written halves must not overlap.
    #[test]
    fn the_two_halves_of_the_table_are_disjoint() {
        for m in [
            Mnemonic::Add,
            Mnemonic::Sub,
            Mnemonic::Cmp,
            Mnemonic::Adc,
            Mnemonic::Sbb,
            Mnemonic::Inc,
            Mnemonic::Dec,
            Mnemonic::And,
            Mnemonic::Or,
            Mnemonic::Xor,
            Mnemonic::Test,
            Mnemonic::Shl,
            Mnemonic::Shr,
            Mnemonic::Sar,
            Mnemonic::Shld,
            Mnemonic::Shrd,
            Mnemonic::Imul,
            Mnemonic::Mul,
            Mnemonic::Bt,
            Mnemonic::Bts,
            Mnemonic::Btr,
            Mnemonic::Btc,
            Mnemonic::Bsf,
            Mnemonic::Bsr,
            Mnemonic::Tzcnt,
            Mnemonic::Lzcnt,
            Mnemonic::Popcnt,
            Mnemonic::Pcmpistri,
            Mnemonic::Pcmpistrm,
            Mnemonic::Pcmpestri,
            Mnemonic::Pcmpestrm,
            Mnemonic::Pcmpestri64,
            Mnemonic::Pcmpestrm64,
            Mnemonic::Stc,
            Mnemonic::Clc,
            Mnemonic::Cmc,
            Mnemonic::Std,
            Mnemonic::Cld,
            Mnemonic::Neg,
            Mnemonic::Xadd,
            Mnemonic::Je,
            Mnemonic::Sete,
            Mnemonic::Cmove,
        ] {
            assert!(
                !is_flag_free(m),
                "{m:?} is classified by hand *and* derived as flag-free"
            );
        }
    }

    /// String ops read DF through a function pointer, which a call-shaped taint search misses; they must stay unclassified.
    #[test]
    fn the_string_operations_are_not_flag_free() {
        for m in [
            Mnemonic::Movsb,
            Mnemonic::Movsw,
            Mnemonic::Stosb,
            Mnemonic::Stosd,
            Mnemonic::Lodsb,
            Mnemonic::Scasb,
            Mnemonic::Cmpsb,
        ] {
            assert!(
                !is_flag_free(m),
                "{m:?} reads DF but is derived as flag-free"
            );
        }
    }
}
