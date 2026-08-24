//! What each x86-64 instruction's *lift* does to the condition flags.
//!
//! The classification behind [`LifterConfig::eliminate_dead_flags`]. Read
//! [`crate::core::flags`] for what the analysis does with this, and
//! [`FlagEffects`] for the contract - the two directions are asymmetric.
//!
//! **This table describes `semantics/`, not the Intel manual**, which is why it
//! is hand-written rather than a call to iced's `rflags_written()`: AF is left
//! *unchanged* after a logical op where the manual says undefined, the `IMUL`
//! handlers assign only CF and OF, and `update_flags_shift` *reads* all five of
//! CF/PF/ZF/SF/OF where the manual says a shift reads nothing.
//!
//! **Absence is the safe answer.** An unnamed mnemonic classifies as
//! [`FlagEffects::CONSERVATIVE`], which is what makes the table safe without
//! knowing which instructions microlift implements. **A mnemonic belongs here
//! only once someone has read its handler.** A `LOCK` prefix, an MMX operand and
//! a `REP`/`REPNE` prefix short-circuit to conservative before the mnemonic is
//! consulted, each routing to a different handler than this table was audited
//! against.
//!
//! [`LifterConfig::eliminate_dead_flags`]: crate::core::LifterConfig::eliminate_dead_flags

use iced_x86::{Instruction, Mnemonic, Register, RflagsBits};

use crate::arch::x86_64::regs::X86Flag;
use crate::core::flags::{FlagEffects, FlagSet};

/// One flag, as a set.
///
/// The bit index is the `X86Flag` discriminant, which is also what the
/// semantics pass to [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live)
/// - one numbering, defined by the enum, used on both sides.
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

/// Everything `update_flags_add` / `_sub` / `_adc` / `_sbb` assign.
const ARITH: FlagSet = CF.union(PF).union(AF).union(ZF).union(SF).union(OF);
/// `update_flags_logical`: as `ARITH` but AF is left alone.
const LOGICAL: FlagSet = CF.union(PF).union(ZF).union(SF).union(OF);
/// `update_flags_inc_dec`: as `ARITH` but CF is left alone.
const INC_DEC: FlagSet = PF.union(AF).union(ZF).union(SF).union(OF);
/// `update_flags_shift`, which both reads and writes this set.
const SHIFT: FlagSet = CF.union(PF).union(ZF).union(SF).union(OF);

/// The flag effects of lifting `i`, or `None` where nobody has audited it.
///
/// `None` and [`FlagEffects::CONSERVATIVE`] mean the same thing to the caller;
/// the distinction is only that `None` says "not classified" rather than
/// "classified as touching everything".
pub fn flag_effects(i: &Instruction) -> Option<FlagEffects> {
    // Routes to a handler this table was not read against. See the module note.
    if i.has_lock_prefix()
        || i.has_rep_prefix()
        || i.has_repne_prefix()
        || uses_mmx_register(i)
        || i.is_invalid()
    {
        return None;
    }

    let effects = match i.mnemonic() {
        // `update_flags_add` and `update_flags_sub`. `CMP` is a `SUB` that
        // discards its result; `NEG` is `0 - x`; `XADD`'s flag half is an `ADD`.
        Mnemonic::Add | Mnemonic::Xadd | Mnemonic::Sub | Mnemonic::Cmp | Mnemonic::Neg => {
            FlagEffects::writing(ARITH)
        }
        // `update_flags_adc` / `_sbb`, plus the carry-in the handler folds into
        // the result before it ever reaches the flag update.
        Mnemonic::Adc | Mnemonic::Sbb => FlagEffects {
            reads: CF,
            writes: ARITH,
        },
        // `update_flags_inc_dec`: CF is architecturally untouched, and the
        // handler leaves it alone.
        Mnemonic::Inc | Mnemonic::Dec => FlagEffects::writing(INC_DEC),

        // AF is left as it was, so it is not a write
        Mnemonic::And | Mnemonic::Or | Mnemonic::Xor | Mnemonic::Test => {
            FlagEffects::writing(LOGICAL)
        }

        // a zero count keeps the old flags and the handler selects between them.
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

        // `bt_common` sets CF for all four of BT/BTS/BTR/BTC and touches no
        // other flag; `handle_bsf`/`handle_bsr` set ZF only.
        Mnemonic::Bt | Mnemonic::Bts | Mnemonic::Btr | Mnemonic::Btc => FlagEffects::writing(CF),
        Mnemonic::Bsf | Mnemonic::Bsr => FlagEffects::writing(ZF),
        Mnemonic::Tzcnt | Mnemonic::Lzcnt => FlagEffects::writing(CF.union(ZF)),
        // `handle_popcnt` sets ZF and *clears* the other five, which is a write
        // of each of them - unlike the bit scans above, none is left alone.
        Mnemonic::Popcnt => FlagEffects::writing(ARITH),
        // The SSE4.2 string compares define CF/ZF/SF/OF from the comparison and
        // clear AF/PF; `pcmpstr_flags` assigns all six.
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

        // A `Jcc`, `SETcc` or `CMOVcc` reads exactly the flags of its condition and
        // writes none, and *which* flags is per-condition - so iced answers it rather
        // than sixty match arms. Safe here in a way it is not for a producer: this is
        // the `reads` direction, where the requirement is a superset.
        //
        // These are also why the whole optimisation pays: `cmp; je` wants ZF, leaving
        // the other five of the `CMP`'s flags with no reader at all.
        m if is_condition_consumer(m) => FlagEffects::reading(from_rflags(i.rflags_read())),

        // Checked last so any mnemonic named above wins. The two sets are disjoint by
        // construction (`the_two_halves_of_the_table_are_disjoint` asserts it), and the
        // ordering resolves a future overlap in favour of the hand-written entry.
        //
        // Transparency is what lets liveness see *through* a stretch of code; without
        // it, it stops at the first `MOV` and `cmp; mov; mov; jne` keeps every flag.
        m if is_flag_free(m) => FlagEffects::TRANSPARENT,

        // Everything else, including every instruction that lifts to a VM exit
        // and every one microlift does not implement. See the module note on
        // why this is where safety comes from.
        _ => return None,
    };
    Some(effects)
}

/// Mnemonics whose lift provably touches no flag and never syncs to the host.
///
/// **Generated, not written.** `tools/derive-flag-free.py` re-derives it from
/// `semantics/`:
///
/// 1. Take every mnemonic `handle_instruction` dispatches and its handler.
/// 2. Mark a function *tainted* if its body mentions `set_flag`, `state.flags`,
///    `flag_needed`, `sync_to_context`, `write_exit_reason`, `write_exit_code`,
///    `resumes_at`, or `handle_unimplemented`.
/// 3. Propagate to a fixpoint through **any mention** of another function's
///    name, not just `self.f(` calls. That over-approximation is the point:
///    `MOVSB` reaches DF through `Self::emit_movs` passed as a value, and a
///    call-shaped search alone would declare it flag-free - a miscompile.
/// 4. Keep the mnemonics whose handler is untainted.
///
/// Both halves of the taint matter: a handler that touches a flag obviously
/// cannot be transparent, and one that **syncs to the host** cannot either,
/// since the sync publishes every flag and an elided write would publish a
/// stale one. That is what keeps `CPUID`, `SYSCALL`, `DIV` and the rest out.
///
/// Re-run the script after touching `semantics/`. It is an allowlist, so the
/// dangerous direction is a handler that gains a `set_flag` and stays listed; a
/// mnemonic that merely stops being listed costs speed alone.
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

/// Whether `m` is a `Jcc`, `SETcc` or `CMOVcc` - an instruction whose only flag
/// interaction is evaluating a condition.
///
/// `JRCXZ`, `LOOP`, `LOOPE` and `LOOPNE` are deliberately absent even though
/// they branch: their handlers also decrement `RCX`, so they classify
/// conservatively until someone has read them.
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

/// Translate an iced `RflagsBits` mask into this frontend's [`FlagSet`].
///
/// Only the six status flags and DF are represented; IF, and the bits iced
/// reports for the system flags, have no SSA variable here and are dropped.
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

/// An MMX operand routes to the unimplemented exit before the semantics
/// dispatch ever sees the mnemonic; mirrors `semantics::uses_mmx_register`.
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

    /// The classification is about the handler, so the entries that differ from
    /// the architecture are the ones worth pinning: each of these three would be
    /// a miscompile if iced's answer had been taken instead.
    #[test]
    fn the_lift_and_the_manual_disagree_in_three_places() {
        // `and eax, eax` - AF is architecturally undefined here, and
        // `update_flags_logical` leaves it unchanged, so it is not a write.
        let and = flag_effects(&decode(&[0x21, 0xC0])).expect("AND is classified");
        assert!(!and.writes.contains(X86Flag::Auxiliary as u8));
        assert!(and.writes.contains(X86Flag::Zero as u8));

        // `imul eax, eax` - the handler assigns CF and OF only.
        let imul = flag_effects(&decode(&[0x0F, 0xAF, 0xC0])).expect("IMUL is classified");
        assert_eq!(imul.writes, CF.union(OF));

        // `shl eax, cl` - reads the flags it may have to keep.
        let shl = flag_effects(&decode(&[0xD3, 0xE0])).expect("SHL is classified");
        assert_eq!(shl.reads, SHIFT);
        assert_eq!(shl.writes, SHIFT);
    }

    /// A `Jcc` reads its own condition and no more, which is the entire source
    /// of the optimisation: five of a `CMP`'s six flags have no reader.
    #[test]
    fn a_conditional_branch_reads_only_its_condition() {
        // `je +0`
        let je = flag_effects(&decode(&[0x74, 0x00])).expect("JE is classified");
        assert_eq!(je.reads, ZF);
        assert_eq!(je.writes, FlagSet::NONE);

        // `jbe +0` reads CF and ZF.
        let jbe = flag_effects(&decode(&[0x76, 0x00])).expect("JBE is classified");
        assert_eq!(jbe.reads, CF.union(ZF));

        // `jg +0` reads ZF, SF and OF.
        let jg = flag_effects(&decode(&[0x7F, 0x00])).expect("JG is classified");
        assert_eq!(jg.reads, ZF.union(SF).union(OF));
    }

    /// The three whole-instruction short-circuits, each of which routes to a
    /// handler the table was not audited against.
    #[test]
    fn a_different_handler_is_not_this_table() {
        // `lock add [rax], ecx` goes to `handle_locked_instruction`.
        assert!(flag_effects(&decode(&[0xF0, 0x01, 0x08])).is_none());
        // `rep stosb` is a string op, not the `STOS` this table never listed.
        assert!(flag_effects(&decode(&[0xF3, 0xAA])).is_none());
        // `paddb mm0, mm1` exits before the semantics dispatch.
        assert!(flag_effects(&decode(&[0x0F, 0xFC, 0xC1])).is_none());
    }

    /// The instructions that exit to the host must stay unclassified, because
    /// their sync publishes every flag and the analysis only knows that by
    /// their absence.
    #[test]
    fn everything_that_exits_is_unclassified() {
        // cpuid, syscall, rdtsc, int3, ud2, hlt, in al, dx
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

    /// A flag-free `MOV` has to be *transparent* rather than unclassified, or
    /// liveness stops at the first one and almost nothing is ever elided.
    #[test]
    fn data_movement_is_transparent() {
        // `mov rax, rbx`
        let mov = flag_effects(&decode(&[0x48, 0x89, 0xD8])).expect("MOV is classified");
        assert_eq!(mov, FlagEffects::TRANSPARENT);
        // and so is the floating-point arithmetic a compiled program is full
        // of: `addsd xmm0, xmm1`, `cvtsi2sd xmm0, rax`.
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

    /// The generated flag-free list and the hand-written entries must not
    /// overlap: an overlap means one of them is a lie, and since the generated
    /// half is derived from the handlers, the lie would be the hand-written one.
    ///
    /// The `match` resolves it in favour of the hand-written arm regardless -
    /// this asserts the situation does not arise rather than relying on that.
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

    /// The string operations read DF through a function pointer, which is the
    /// one thing a call-shaped taint search misses - and the reason
    /// `tools/derive-flag-free.py` propagates through any mention of a name.
    /// Their being unclassified is load-bearing, not incidental.
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
