use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Optimization {
    /// Skip the fixup that gives an SSE instruction's *generated* NaN (as opposed
    /// to one it propagates from an operand) a guest-accurate bit pattern.
    ///
    /// Safe only for a guest that never inspects the sign or payload bits of a NaN
    /// it just generated - which is most guest code, since every operation on a NaN
    /// agrees regardless of its bit pattern.
    UnsafeImpreciseNaN,
    /// Skip the divide-error (#DE) guards `DIV`/`IDIV` emit - the divisor-is-zero
    /// test, the quotient-overflow test, and the fault block with its
    /// `sync_to_context` - so a division is just the division.
    ///
    /// Safe only for a guest that never causes an invalid division. This is not a
    /// loss of precision: a division that would have raised #DE no longer exits
    /// with `ExitReason::DivideError` at all. An overflowing quotient is written
    /// back truncated; a zero divisor either traps on the host (the native
    /// `udiv`/`urem`) or produces a garbage quotient and remainder (the 128-bit
    /// software loop), and neither is a guest-visible fault.
    UnsafeImpreciseDivisionExceptions,
}

impl Optimization {
    pub const ALL: &'static [Optimization] = &[
        Optimization::UnsafeImpreciseNaN,
        Optimization::UnsafeImpreciseDivisionExceptions,
    ];

    fn bit(self) -> u32 {
        1 << (Optimization::ALL
            .iter()
            .position(|&o| o == self)
            .expect("in ALL") as u32)
    }

    pub fn name(self) -> &'static str {
        match self {
            Optimization::UnsafeImpreciseNaN => "unsafe-imprecise-nan",
            Optimization::UnsafeImpreciseDivisionExceptions => {
                "unsafe-imprecise-division-exceptions"
            }
        }
    }
}

/// A set of enabled [`Optimization`]s. Defaults to
/// [`Optimizations::none`] - none enabled, output identical to a lifter
/// without this knob.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Optimizations {
    enabled: u32,
}

impl Optimizations {
    pub const fn none() -> Self {
        Optimizations { enabled: 0 }
    }

    pub fn all() -> Self {
        Optimizations { enabled: u32::MAX }
    }

    pub fn with(mut self, opt: Optimization) -> Self {
        self.enabled |= opt.bit();
        self
    }

    pub fn without(mut self, opt: Optimization) -> Self {
        self.enabled &= !opt.bit();
        self
    }

    pub fn only(mut self, opts: &[Optimization]) -> Self {
        self.enabled = opts.iter().fold(0, |acc, o| acc | o.bit());
        self
    }

    pub fn allows(self, opt: Optimization) -> bool {
        self.enabled & opt.bit() != 0
    }

    pub fn active(self) -> impl Iterator<Item = Optimization> {
        Optimization::ALL
            .iter()
            .copied()
            .filter(move |&o| self.allows(o))
    }
}

impl fmt::Debug for Optimizations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Optimizations(")?;
        let mut first = true;
        for opt in self.active() {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "+{}", opt.name())?;
            first = false;
        }
        write!(f, ")")
    }
}
