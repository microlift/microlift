use std::fmt;

use target_lexicon::{Architecture, Triple};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SpecTarget {
    /// Unknown or portable. No specialization applies, and the lifter emits
    /// exactly what it emitted before this module existed.
    #[default]
    Generic,
    /// x86-64 with Cranelift's baseline `x86_64` feature set (SSE4.2).
    X86_64,
    /// aarch64 with baseline NEON.
    Aarch64,
}

impl SpecTarget {
    pub fn from_triple(triple: &Triple) -> Self {
        match triple.architecture {
            Architecture::X86_64 | Architecture::X86_64h => SpecTarget::X86_64,
            Architecture::Aarch64(_) => SpecTarget::Aarch64,
            _ => SpecTarget::Generic,
        }
    }

    pub fn host() -> Self {
        SpecTarget::from_triple(&Triple::host())
    }

    pub fn name(self) -> &'static str {
        match self {
            SpecTarget::Generic => "generic",
            SpecTarget::X86_64 => "x86_64",
            SpecTarget::Aarch64 => "aarch64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Spec {
    /// `ROUNDPS` / `ROUNDPD` - packed floating-point rounding to an imm8-selected
    /// mode. One instruction on both targets (`FRINT{N,M,P,Z}` / `ROUNDPS`),
    /// against a per-lane extract/bitcast/round/repack chain - the worst ratio in
    /// this table.
    PackedFpRound,
    /// `HADDPS` / `HSUBPS` / `HADDPD` / `HSUBPD` - horizontal FP add and subtract
    /// of adjacent lanes. Gathers even and odd lanes with one constant shuffle
    /// each (`UZP1`/`UZP2` on aarch64) and applies the ordinary packed op.
    PackedFpHorizontal,
    /// `PHADDW`/`PHADDD`/`PHSUBW`/`PHSUBD`/`PHADDSW`/`PHSUBSW` - integer
    /// horizontal forms. The non-saturating adds are Cranelift's `iadd_pairwise`;
    /// the subtracts and saturating pair reuse the even/odd gather from
    /// [`PackedFpHorizontal`](Spec::PackedFpHorizontal).
    PackedIntHorizontal,
    /// `PMADDWD` - signed word multiply with adjacent-pair accumulation into
    /// dwords. Emitted as a widening multiply of each half plus `iadd_pairwise`,
    /// which x86-64 folds back into one `PMADDWD` and aarch64 lowers to
    /// `SMULL`/`SMULL2`/`ADDP`.
    PairwiseMultiplyAdd,
    /// `PMADDUBSW` - unsigned-by-signed byte multiply with saturating
    /// adjacent-pair accumulation into words.
    ///
    /// x86-64 has an exact opcode (`x86_pmaddubsw`). aarch64 has none, so the path
    /// is a widening multiply, `SADDLP` and `SQXTN` - still around a tenth of the
    /// portable expansion.
    MultiplyAddUnsignedSigned,
    /// `PSADBW` - sum of absolute byte differences, one sum per 64-bit half.
    ///
    /// The absolute difference is `usub_sat(a, b) | usub_sat(b, a)`, exact for
    /// unsigned bytes; the reduction is `UADDLP`/`UADDLP`/`ADDP` rather than
    /// sixteen scalar lane extractions.
    SumOfAbsoluteDifferences,
    /// `PMULUDQ` / `PMULDQ` - widening multiply of the *even* dword lanes.
    ///
    /// Gathers the even dwords with a `UZP1`-shaped shuffle and multiplies the
    /// widened low halves, which both backends recognize as `UMULL`/`SMULL`. This
    /// replaces a full `I64X2` `imul`, which aarch64 expands to seven instructions.
    WideningEvenMultiply,
    /// `PMULHRSW` - signed word multiply keeping the rounded high 17 bits.
    ///
    /// x86-64: `x86_pmulhrsw`, which models it exactly. aarch64:
    /// `sqmul_round_sat` (`SQRDMULH`) computes the same `(x*y + 0x4000) >> 15` but
    /// *saturates*, differing on the single pair `-32768 * -32768` (x86 wraps to
    /// `-32768`, `SQRDMULH` gives `32767`). The path restores that lane with a
    /// compare-and-select and is still shorter than the portable expansion.
    RoundedMultiplyHigh,
    /// CF and OF from the machine's own overflow-producing add, subtract and
    /// multiply, rather than from arithmetic on the result.
    ///
    /// The portable definitions are the expensive half of every
    /// `ADD`/`SUB`/`CMP`/`INC`/`DEC`/`NEG`/`ADC`/`SBB`/`XADD` and their locked
    /// forms: CF is a second compare of values the machine already compared, OF a
    /// four-instruction sign-bit dance. Both are a single condition bit the
    /// hardware produced as a side effect - which is what Cranelift's
    /// `uadd_overflow`/`sadd_overflow`/`usub_overflow`/`ssub_overflow` (and
    /// `smul_overflow` for the truncating `IMUL`s) model. aarch64 gets
    /// `ADDS`/`SUBS` + `CSET`; x86-64 gets `add`/`sub` + `setb`/`seto`.
    ///
    /// Not claimed elsewhere because coverage is partial: riscv64 lowers only
    /// `uadd_overflow`, and only at `I32`/`I64`. The `IMUL` forms are aarch64-only
    /// because x64 lowers `smul_overflow` through the register-pinning one-operand
    /// `imul`, which measured as a loss.
    ///
    /// Requires [`eliminate_dead_flags`](crate::core::LifterConfig::eliminate_dead_flags)
    /// and is inert without it. Unlike the vector specializations this one is not
    /// an improvement wherever it applies: the portable flag arithmetic is dataflow
    /// over the sum and dies to DCE when nothing reads it, while an overflow op's
    /// condition bit is welded to the instruction producing the result and
    /// survives. Without liveness every site looks live and it is a measured
    /// regression. See `arch/x86_64/semantics/arith_spec.rs`.
    IntegerOverflowFlags,
    /// Emit `PAUSE` instructions as the host's equivalent mechanism via
    /// the `spin_hint` CLIF opcode.
    SpinLoopHint,
}

impl Spec {
    pub const ALL: &'static [Spec] = &[
        Spec::PackedFpRound,
        Spec::PackedFpHorizontal,
        Spec::PackedIntHorizontal,
        Spec::PairwiseMultiplyAdd,
        Spec::MultiplyAddUnsignedSigned,
        Spec::SumOfAbsoluteDifferences,
        Spec::WideningEvenMultiply,
        Spec::RoundedMultiplyHigh,
        Spec::IntegerOverflowFlags,
        Spec::SpinLoopHint,
    ];

    fn bit(self) -> u32 {
        1 << (Spec::ALL.iter().position(|&s| s == self).expect("in ALL") as u32)
    }

    pub fn name(self) -> &'static str {
        match self {
            Spec::PackedFpRound => "packed-fp-round",
            Spec::PackedFpHorizontal => "packed-fp-horizontal",
            Spec::PackedIntHorizontal => "packed-int-horizontal",
            Spec::PairwiseMultiplyAdd => "pairwise-multiply-add",
            Spec::MultiplyAddUnsignedSigned => "multiply-add-unsigned-signed",
            Spec::SumOfAbsoluteDifferences => "sum-of-absolute-differences",
            Spec::WideningEvenMultiply => "widening-even-multiply",
            Spec::RoundedMultiplyHigh => "rounded-multiply-high",
            Spec::IntegerOverflowFlags => "integer-overflow-flags",
            Spec::SpinLoopHint => "spin-loop-hint",
        }
    }

    pub fn supported_on(self, target: SpecTarget) -> bool {
        match target {
            SpecTarget::Generic => false,
            SpecTarget::Aarch64 => true,
            SpecTarget::X86_64 => matches!(
                self,
                Spec::PackedFpRound
                    | Spec::PackedFpHorizontal
                    | Spec::PackedIntHorizontal
                    | Spec::PairwiseMultiplyAdd
                    | Spec::MultiplyAddUnsignedSigned
                    | Spec::SumOfAbsoluteDifferences
                    | Spec::IntegerOverflowFlags
                    | Spec::SpinLoopHint
            ),
        }
    }

    /// Whether this specialization is inert without
    /// [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm).
    ///
    /// True for every whole-register vector sequence, i.e. all but
    /// [`IntegerOverflowFlags`](Spec::IntegerOverflowFlags) (scalar integer
    /// arithmetic) and [`SpinLoopHint`](Spec::SpinLoopHint) (operand-free).
    pub fn requires_vector_xmm(self) -> bool {
        !matches!(self, Spec::IntegerOverflowFlags | Spec::SpinLoopHint)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Specialization {
    target: SpecTarget,
    enabled: u32,
}

impl Specialization {
    pub const fn generic() -> Self {
        Specialization {
            target: SpecTarget::Generic,
            enabled: 0,
        }
    }

    pub fn for_target(target: SpecTarget) -> Self {
        Specialization {
            target,
            enabled: u32::MAX,
        }
    }

    pub fn for_triple(triple: &Triple) -> Self {
        Specialization::for_target(SpecTarget::from_triple(triple))
    }

    pub fn host() -> Self {
        Specialization::for_target(SpecTarget::host())
    }

    pub fn none(mut self) -> Self {
        self.enabled = 0;
        self
    }

    pub fn with(mut self, spec: Spec) -> Self {
        self.enabled |= spec.bit();
        self
    }

    pub fn without(mut self, spec: Spec) -> Self {
        self.enabled &= !spec.bit();
        self
    }

    pub fn only(mut self, specs: &[Spec]) -> Self {
        self.enabled = specs.iter().fold(0, |acc, s| acc | s.bit());
        self
    }

    pub fn target(self) -> SpecTarget {
        self.target
    }

    pub fn allows(self, spec: Spec) -> bool {
        self.enabled & spec.bit() != 0 && spec.supported_on(self.target)
    }

    pub fn active(self) -> impl Iterator<Item = Spec> {
        Spec::ALL.iter().copied().filter(move |&s| self.allows(s))
    }
}

impl fmt::Debug for Specialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Specialization({}", self.target.name())?;
        for spec in self.active() {
            write!(f, " +{}", spec.name())?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_allows_nothing() {
        let s = Specialization::generic();
        assert_eq!(s.target(), SpecTarget::Generic);
        assert_eq!(s.active().count(), 0);
        for &spec in Spec::ALL {
            assert!(!s.allows(spec), "{} allowed under generic", spec.name());
        }
    }

    #[test]
    fn enabling_on_generic_stays_inert() {
        let s = Specialization::generic().with(Spec::PackedFpRound);
        assert!(!s.allows(Spec::PackedFpRound));
    }

    #[test]
    fn target_constructors_enable_their_whole_set() {
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            let s = Specialization::for_target(target);
            let expected = Spec::ALL.iter().filter(|s| s.supported_on(target)).count();
            assert_eq!(s.active().count(), expected, "{}", target.name());
        }
    }

    #[test]
    fn builders_carve_the_set_down() {
        let s = Specialization::for_target(SpecTarget::Aarch64).without(Spec::PackedFpRound);
        assert!(!s.allows(Spec::PackedFpRound));
        assert!(s.allows(Spec::PairwiseMultiplyAdd));

        let s = s.only(&[Spec::PackedFpRound, Spec::SumOfAbsoluteDifferences]);
        assert!(s.allows(Spec::PackedFpRound));
        assert!(s.allows(Spec::SumOfAbsoluteDifferences));
        assert!(!s.allows(Spec::PairwiseMultiplyAdd));

        assert_eq!(s.none().active().count(), 0);
    }

    #[test]
    fn bits_are_distinct() {
        let all = Spec::ALL.iter().fold(0u32, |acc, s| acc | s.bit());
        assert_eq!(all.count_ones() as usize, Spec::ALL.len());
    }

    #[test]
    fn triples_map_to_families() {
        let cases = [
            ("x86_64-unknown-linux-gnu", SpecTarget::X86_64),
            ("aarch64-apple-darwin", SpecTarget::Aarch64),
            ("aarch64-unknown-linux-gnu", SpecTarget::Aarch64),
            ("riscv64gc-unknown-linux-gnu", SpecTarget::Generic),
        ];
        for (triple, want) in cases {
            let t: Triple = triple.parse().expect("triple");
            assert_eq!(SpecTarget::from_triple(&t), want, "{triple}");
        }
    }

    #[test]
    fn debug_names_the_effective_set() {
        let s = Specialization::for_target(SpecTarget::Aarch64)
            .only(&[Spec::PackedFpRound, Spec::RoundedMultiplyHigh]);
        assert_eq!(
            format!("{s:?}"),
            "Specialization(aarch64 +packed-fp-round +rounded-multiply-high)"
        );
    }
}
