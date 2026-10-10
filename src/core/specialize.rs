use std::fmt;

use target_lexicon::{Architecture, Triple};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SpecTarget {
    /// No specialization; portable output.
    #[default]
    Generic,
    X86_64,
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
    /// `ROUNDPS`/`ROUNDPD`: one instruction vs a per-lane chain (the worst ratio here).
    PackedFpRound,
    /// `HADDPS`/`HSUBPS`/`HADDPD`/`HSUBPD`: even/odd gather shuffles plus the packed op.
    PackedFpHorizontal,
    /// `PHADDW`/`PHADDD`/`PHSUBW`/`PHSUBD`/`PHADDSW`/`PHSUBSW`; shares the gather of [`PackedFpHorizontal`](Spec::PackedFpHorizontal).
    PackedIntHorizontal,
    /// `PMADDWD`: widening multiply plus `iadd_pairwise`.
    PairwiseMultiplyAdd,
    /// `PMADDUBSW`: exact opcode on x86-64; aarch64 widens, `SADDLP`, `SQXTN`.
    MultiplyAddUnsignedSigned,
    /// `PSADBW`: `usub_sat` both ways, then `UADDLP`/`ADDP` reduction.
    SumOfAbsoluteDifferences,
    /// `PMULUDQ`/`PMULDQ`: widening multiply of even dword lanes (avoids a 7-instruction aarch64 `I64X2` `imul`).
    WideningEvenMultiply,
    /// `PMULHRSW`: aarch64 `SQRDMULH` saturates on `-32768 * -32768` where x86 wraps, so that lane is
    /// fixed up with a select.
    RoundedMultiplyHigh,
    /// CF/OF from the machine's overflow-producing add/sub/mul instead of arithmetic on the result.
    ///
    /// Requires [`eliminate_dead_flags`](crate::core::LifterConfig::eliminate_dead_flags), inert without
    /// it; only pays when the dead-flag pass rewrites unused condition bits. riscv64 and x64 `IMUL`
    /// forms are not claimed (partial coverage, measured loss). See `arch/x86_64/semantics/arith_spec.rs`.
    IntegerOverflowFlags,
    /// `PAUSE` via the `spin_hint` CLIF opcode.
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

    /// Whether this is inert without [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm).
    ///
    /// True for every whole-register vector spec.
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
