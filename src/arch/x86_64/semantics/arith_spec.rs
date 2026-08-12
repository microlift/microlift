//! CF and OF from the machine's own arithmetic.
//!
//! [`Spec::IntegerOverflowFlags`] in the target-specialization table, and the
//! scalar-integer counterpart to [`super::sse_spec`]. Same shape: every
//! function here is a guarded early return that either emits the specialized
//! form and hands back a result, or returns `None` having emitted nothing so
//! the caller falls through to its portable path.
//!
//! Nothing here works without
//! [`eliminate_dead_flags`](crate::core::LifterConfig::eliminate_dead_flags).
//! That is not a stylistic pairing, it is what the measurement says, and the
//! reason is worth stating because it is the opposite of the intuition.
//!
//! [`Spec::IntegerOverflowFlags`]: crate::core::specialize::Spec::IntegerOverflowFlags
//! [`Lifter::flag_needed`]: crate::lifter::Lifter::flag_needed

use crate::core::specialize::{Spec, SpecTarget};
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;

/// Which of the two hardware-backed flags this site has to produce.
///
/// Both dead is the interesting case: it is the whole reason these functions
/// can decline, and the reason the answer depends on
/// [`FlagLiveness`](crate::core::flags::FlagLiveness) rather than on the
/// mnemonic alone. Note it is *this* instruction's needs that matter, not the
/// mnemonic's architectural effects - `ADD` writes both, and at most sites
/// neither is read before something else overwrites it.
#[derive(Clone, Copy)]
struct Wanted {
    carry: bool,
    overflow: bool,
}

impl Wanted {
    fn none(self) -> bool {
        !self.carry && !self.overflow
    }
}

impl<'a> Lifter<'a> {
    /// The flags this site needs from the hardware, or `None` if the
    /// specialization is not available here at all.
    fn overflow_flags_wanted(&self) -> Option<Wanted> {
        if !self.specialized(Spec::IntegerOverflowFlags) || !self.config.eliminate_dead_flags {
            return None;
        }
        Some(Wanted {
            carry: self.flag_needed(X86Flag::Carry),
            overflow: self.flag_needed(X86Flag::Overflow),
        })
    }

    /// `lhs + rhs` with CF and/or OF taken from the add itself.
    ///
    /// Returns the sum, or `None` when the caller should emit an ordinary
    /// `iadd` and compute the flags portably (or not at all).
    ///
    /// [`Spec::IntegerOverflowFlags`](crate::core::specialize::Spec::IntegerOverflowFlags)
    pub(super) fn try_native_add_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted() else {
            return Ok(None);
        };
        if want.none() {
            return Ok(None);
        }

        // When both are wanted the unsigned form's sum is the one kept and the
        // signed form's is dropped; the two are equal, so which one survives is
        // arbitrary and the choice costs nothing either way.
        let mut res = None;
        if want.carry {
            let (sum, cf) = self.builder.ins().uadd_overflow(lhs, rhs);
            self.set_flag(X86Flag::Carry, cf)?;
            res = Some(sum);
        }
        if want.overflow {
            let (sum, of) = self.builder.ins().sadd_overflow(lhs, rhs);
            self.set_flag(X86Flag::Overflow, of)?;
            res = res.or(Some(sum));
        }
        Ok(res)
    }

    /// `lhs - rhs` with CF (the borrow) and/or OF taken from the subtract
    /// itself. Returns the difference, or `None` for the portable path.
    pub(super) fn try_native_sub_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted() else {
            return Ok(None);
        };
        if want.none() {
            return Ok(None);
        }

        let mut res = None;
        if want.carry {
            let (diff, cf) = self.builder.ins().usub_overflow(lhs, rhs);
            self.set_flag(X86Flag::Carry, cf)?;
            res = Some(diff);
        }
        if want.overflow {
            let (diff, of) = self.builder.ins().ssub_overflow(lhs, rhs);
            self.set_flag(X86Flag::Overflow, of)?;
            res = res.or(Some(diff));
        }
        Ok(res)
    }

    /// `lhs ± 1` with OF taken from the operation.
    ///
    /// `INC`/`DEC` leave CF alone, so only the signed form is ever emitted and
    /// a dead OF is the only thing that declines.
    pub(super) fn try_native_inc_dec_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        is_add: bool,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted() else {
            return Ok(None);
        };
        if !want.overflow {
            return Ok(None);
        }

        let (res, of) = if is_add {
            self.builder.ins().sadd_overflow(lhs, rhs)
        } else {
            self.builder.ins().ssub_overflow(lhs, rhs)
        };
        self.set_flag(X86Flag::Overflow, of)?;
        Ok(Some(res))
    }

    /// The two-step `ADC`/`SBB` with each step's carry taken from the machine,
    /// returning the result and the combined carry out.
    ///
    /// Declines unless CF is live: the result alone is two ordinary adds, and
    /// OF here is the sign dance on the final result either way - Cranelift's
    /// `sadd_overflow_cin` would say it in one op, but no backend lowers it.
    pub(super) fn try_native_adc_sbb_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        cf_in: Value,
        is_adc: bool,
    ) -> crate::Result<Option<(Value, Value)>> {
        let Some(want) = self.overflow_flags_wanted() else {
            return Ok(None);
        };
        if !want.carry {
            return Ok(None);
        }

        let (tmp, c1) = if is_adc {
            self.builder.ins().uadd_overflow(lhs, rhs)
        } else {
            self.builder.ins().usub_overflow(lhs, rhs)
        };
        let (res, c2) = if is_adc {
            self.builder.ins().uadd_overflow(tmp, cf_in)
        } else {
            self.builder.ins().usub_overflow(tmp, cf_in)
        };
        // A carry out of either step is a carry out of the whole operation, and
        // both cannot happen at once - the same `or` the portable path builds
        // out of two compares.
        let cf_out = self.builder.ins().bor(c1, c2);
        Ok(Some((res, cf_out)))
    }

    /// The truncating `IMUL` forms - `imul r, r/m` and `imul r, r/m, imm` -
    /// with CF and OF taken from the multiply.
    ///
    /// x86 sets both to "the full product did not fit in the destination",
    /// which is `smul_overflow`'s bit exactly. The portable path answers the
    /// same question with a second, widening multiply (`smulhi`) and a compare
    /// against the sign extension of the low half.
    ///
    /// **aarch64 only**, and measured rather than assumed
    /// (`specialize_tests::specializations_shrink_the_compiled_code`): x64
    /// lowers `smul_overflow` through the *one-operand* `imul`, which writes
    /// RDX:RAX and so pins both, and the moves that costs are worse than the
    /// second multiply it saves. aarch64's `MUL`/`SMULH` pair has no such
    /// constraint. This is the same shape of hole as
    /// [`Spec::WideningEvenMultiply`](crate::core::specialize::Spec::WideningEvenMultiply)'s:
    /// a form where the guest architecture's own portable expansion is already
    /// close to the metal.
    ///
    /// Returns the truncated product, or `None` for the portable path.
    pub(super) fn try_native_imul_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted() else {
            return Ok(None);
        };
        if want.none() || self.config.specialization.target() != SpecTarget::Aarch64 {
            return Ok(None);
        }

        let (res, of) = self.builder.ins().smul_overflow(lhs, rhs);
        self.set_flag(X86Flag::Carry, of)?;
        self.set_flag(X86Flag::Overflow, of)?;
        Ok(Some(res))
    }
}
