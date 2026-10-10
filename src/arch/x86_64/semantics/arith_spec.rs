//! CF and OF from the machine's own arithmetic ([`Spec::IntegerOverflowFlags`]).
//!
//! Each function emits the specialized form and returns a result, or returns `None` having emitted nothing.
//! Only pays with [`eliminate_dead_flags`](crate::core::LifterConfig::eliminate_dead_flags).
//!
//! [`Spec::IntegerOverflowFlags`]: crate::core::specialize::Spec::IntegerOverflowFlags

use crate::core::specialize::{Spec, SpecTarget};
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;

/// Hardware-backed flag outputs to emit; unused ones are removed by the SSA pass.
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
    /// The flags this site needs from hardware, or `None` if unavailable (aarch64 declines below 32 bits).
    fn overflow_flags_wanted(&self, lhs: Value) -> Option<Wanted> {
        if !self.specialized(Spec::IntegerOverflowFlags) || !self.config.eliminate_dead_flags {
            return None;
        }
        if !self.native_overflow_width_ok(lhs) {
            return None;
        }
        Some(Wanted {
            carry: self.flag_needed(X86Flag::Carry),
            overflow: self.flag_needed(X86Flag::Overflow),
        })
    }

    /// Whether the target has flag-setting arithmetic at this width.
    /// aarch64 `ADDS`/`SUBS` exist only at 32/64 bits; elsewhere synthesized (+60 bytes per 32-instruction snippet).
    fn native_overflow_width_ok(&self, lhs: Value) -> bool {
        if self.config.specialization.target() != SpecTarget::Aarch64 {
            return true;
        }
        self.builder.func.dfg.value_type(lhs).bits() >= 32
    }

    /// `lhs + rhs` with CF/OF from the add itself, or `None` for the portable path.
    pub(super) fn try_native_add_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted(lhs) else {
            return Ok(None);
        };
        if want.none() {
            return Ok(None);
        }

        // Both wanted: the unsigned sum is kept; the two are equal.
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

    /// `lhs - rhs` with CF (borrow) and/or OF from the subtract, or `None`.
    pub(super) fn try_native_sub_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted(lhs) else {
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

    /// `lhs ± 1` with OF from the operation (`INC`/`DEC` leave CF alone).
    pub(super) fn try_native_inc_dec_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        is_add: bool,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted(lhs) else {
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

    /// Two-step `ADC`/`SBB` with each carry from the machine; declines unless CF is live.
    pub(super) fn try_native_adc_sbb_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
        cf_in: Value,
        is_adc: bool,
    ) -> crate::Result<Option<(Value, Value)>> {
        let Some(want) = self.overflow_flags_wanted(lhs) else {
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
        // Both steps cannot carry at once, so the `or` is exact.
        let cf_out = self.builder.ins().bor(c1, c2);
        Ok(Some((res, cf_out)))
    }

    /// Truncating `IMUL` forms (`imul r, r/m` and `imul r, r/m, imm`) with CF/OF from the multiply, or `None`.
    /// aarch64 only: x64's `smul_overflow` pins RDX:RAX and the moves cost more than they save.
    pub(super) fn try_native_imul_flags(
        &mut self,
        lhs: Value,
        rhs: Value,
    ) -> crate::Result<Option<Value>> {
        let Some(want) = self.overflow_flags_wanted(lhs) else {
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
