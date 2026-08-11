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
    /// mode.
    ///
    /// aarch64: `FRINTN`/`FRINTM`/`FRINTP`/`FRINTZ` on `4S`/`2D`.
    /// x86-64: `ROUNDPS`/`ROUNDPD`.
    ///
    /// Both are one instruction; the portable path is a four- (or two-) lane
    /// extract/bitcast/round/repack chain, and is by some distance the worst
    /// ratio in this table.
    PackedFpRound,
    /// `HADDPS` / `HSUBPS` / `HADDPD` / `HSUBPD` - horizontal floating-point
    /// add and subtract of adjacent lanes.
    ///
    /// Gathers the even and odd lanes of the two operands with one constant
    /// shuffle each - `UZP1`/`UZP2` on aarch64 - and applies the ordinary
    /// packed op to the pair.
    PackedFpHorizontal,
    /// `PHADDW` / `PHADDD` / `PHSUBW` / `PHSUBD` / `PHADDSW` / `PHSUBSW` - the
    /// integer horizontal forms.
    ///
    /// The non-saturating adds are literally Cranelift's `iadd_pairwise`
    /// (`ADDP` on aarch64, `PHADDW`/`PHADDD` on x86-64); the subtracts and the
    /// saturating pair reuse the same even/odd gather as
    /// [`PackedFpHorizontal`](Spec::PackedFpHorizontal).
    PackedIntHorizontal,
    /// `PMADDWD` - signed word multiply with adjacent-pair accumulation into
    /// dwords.
    ///
    /// Emitted in the shape both backends pattern-match: a widening multiply of
    /// each half followed by `iadd_pairwise`. x86-64 folds the whole tree back
    /// into a single `PMADDWD`; aarch64 gives `SMULL`/`SMULL2`/`ADDP`.
    PairwiseMultiplyAdd,
    /// `PMADDUBSW` - unsigned-by-signed byte multiply with saturating
    /// adjacent-pair accumulation into words.
    ///
    /// x86-64 has an exact CLIF opcode for it (`x86_pmaddubsw`, one
    /// instruction). aarch64 has no equivalent, so the specialized path is a
    /// widening multiply, a widening pairwise add (`SADDLP`) and a saturating
    /// narrow (`SQXTN`) - still around a tenth of the portable expansion.
    MultiplyAddUnsignedSigned,
    /// `PSADBW` - sum of absolute byte differences, one sum per 64-bit half.
    ///
    /// The absolute difference is `usub_sat(a, b) | usub_sat(b, a)`, which is
    /// exact for unsigned bytes and lowers to three instructions everywhere;
    /// the reduction is a ladder of widening pairwise adds (`UADDLP`, `UADDLP`,
    /// `ADDP`) rather than sixteen scalar lane extractions.
    SumOfAbsoluteDifferences,
    /// `PMULUDQ` / `PMULDQ` - widening multiply of the *even* dword lanes.
    ///
    /// Gathers the even dwords with a `UZP1`-shaped shuffle and then multiplies
    /// the widened low halves, which both backends recognize as a widening
    /// multiply: `UMULL`/`SMULL` on aarch64. This replaces a full `I64X2`
    /// `imul`, which aarch64 has no instruction for and expands to a
    /// seven-instruction `rev64`/`mul`/`xtn`/`addp`/`shll`/`umlal` sequence.
    WideningEvenMultiply,
    /// `PMULHRSW` - signed word multiply keeping the rounded high 17 bits.
    ///
    /// x86-64: `x86_pmulhrsw`, which models the instruction exactly.
    ///
    /// aarch64: `sqmul_round_sat` (`SQRDMULH`) computes the same
    /// `(x*y + 0x4000) >> 15` but *saturates*, so it differs from x86 on the
    /// single input pair `-32768 * -32768` - where x86 wraps to `-32768` and
    /// `SQRDMULH` gives `32767`. The specialized path restores that one lane
    /// with a compare-and-select and is still shorter than the portable
    /// widening expansion.
    RoundedMultiplyHigh,
    /// CF and OF from the machine's own overflow-producing add, subtract and
    /// multiply, rather than from arithmetic on the result.
    ///
    /// The portable definitions of those two flags are the expensive half of
    /// every `ADD`/`SUB`/`CMP`/`INC`/`DEC`/`NEG`/`ADC`/`SBB`/`XADD` and their
    /// locked forms: CF is a second compare of values the machine already
    /// compared, and OF is a four-instruction `(lhs ^ res) & (rhs ^ res)`
    /// sign-bit dance. Both are a *single condition bit the hardware produced
    /// as a side effect of the arithmetic* - which is exactly what Cranelift's
    /// `uadd_overflow` / `sadd_overflow` / `usub_overflow` / `ssub_overflow`
    /// (and `smul_overflow` for the truncating `IMUL` forms) model.
    ///
    /// aarch64: `ADDS`/`SUBS` and a `CSET` on `HS`/`VS`.
    /// x86-64: `add`/`sub` and a `setb`/`seto` - the guest's own instruction,
    /// reading the guest's own flag, after a round trip through the IR.
    ///
    /// Not claimed elsewhere because the coverage is genuinely partial:
    /// riscv64 lowers `uadd_overflow` alone, and only at `I32`/`I64`, so a
    /// `SUB` at any width would be a hard lowering failure there. `IMUL`'s
    /// truncating forms are aarch64-only for a second reason - x64 lowers
    /// `smul_overflow` through the register-pinning one-operand `imul`, which
    /// measured as a loss.
    /// 
    /// It requires [`eliminate_dead_flags`](crate::core::LifterConfig::eliminate_dead_flags),
    /// and is entirely inert without it. Unlike a vector specialization,
    /// this one is not an improvement wherever it applies: the portable flag
    /// arithmetic is dataflow over the sum and dies to ordinary DCE when
    /// nothing reads it, while an overflow op's condition bit is welded to
    /// the instruction producing the result and survives. Without liveness
    /// every site looks live and the feature is a measured regression; with
    /// it, the dead sites emit a plain `iadd` and the live ones get the short
    /// form. See `arch/x86_64/semantics/arith_spec.rs`.
    IntegerOverflowFlags,
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
            ),
        }
    }

    /// Whether this specialization is inert without
    /// [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm).
    ///
    /// True for every whole-register vector sequence, which is all of them bar
    /// [`IntegerOverflowFlags`](Spec::IntegerOverflowFlags): that one rewrites
    /// scalar integer arithmetic and has nothing to do with how XMM registers
    /// are represented.
    pub fn requires_vector_xmm(self) -> bool {
        !matches!(self, Spec::IntegerOverflowFlags)
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
