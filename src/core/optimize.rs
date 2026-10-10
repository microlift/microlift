use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Optimization {
    /// Skip the fixup giving a *generated* NaN the guest's bit pattern.
    ///
    /// Safe only if the guest never inspects sign or payload bits of a generated NaN.
    UnsafeImpreciseNaN,
    /// Skip the #DE guards `DIV`/`IDIV` emit.
    ///
    /// Safe only for a guest that never divides invalidly: an overflowing quotient is truncated,
    /// and a zero divisor traps on the host or yields garbage; no `DivideError` exit.
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

/// A set of enabled [`Optimization`]s; defaults to none.
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
