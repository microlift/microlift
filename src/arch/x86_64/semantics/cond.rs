//! x86 condition code handling.
//!
//! `Jcc`, `SETcc` and `CMOVcc` differ only in what they do with a condition.
//! the condition itself is one of sixteen flag expressions built out of eight
//! base predicates and their negations.

use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::Instruction;

/// One x86 condition code, named after the `Jcc`/`SETcc`/`CMOVcc` suffix.
///
/// The aliases each mnemonic carries (`JZ` = `JE`, `JNAE` = `JB`, ...) are
/// resolved by the caller; iced reports one canonical mnemonic per encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Cond {
    /// OF=1
    O,
    /// OF=0
    No,
    /// CF=1 - below, unsigned
    B,
    /// CF=0 - above or equal, unsigned
    Ae,
    /// ZF=1
    E,
    /// ZF=0
    Ne,
    /// CF=1 or ZF=1 - below or equal, unsigned
    Be,
    /// CF=0 and ZF=0 - above, unsigned
    A,
    /// SF=1
    S,
    /// SF=0
    Ns,
    /// PF=1
    P,
    /// PF=0
    Np,
    /// SF≠OF - less, signed
    L,
    /// SF=OF - greater or equal, signed
    Ge,
    /// ZF=1 or SF≠OF - less or equal, signed
    Le,
    /// ZF=0 and SF=OF - greater, signed
    G,
}

impl Cond {
    /// The condition this one is the negation of, paired with whether that
    /// negation is in effect. `(Self::Le, true)` for [`Cond::G`], and so on.
    ///
    /// The positive member of each pair is the one whose expression is cheaper
    /// to build, which for the four flag-pair conditions means the one that
    /// needs no final complement.
    fn base(self) -> (Base, bool) {
        match self {
            Cond::O => (Base::Of, false),
            Cond::No => (Base::Of, true),
            Cond::B => (Base::Cf, false),
            Cond::Ae => (Base::Cf, true),
            Cond::E => (Base::Zf, false),
            Cond::Ne => (Base::Zf, true),
            Cond::Be => (Base::CfOrZf, false),
            Cond::A => (Base::CfOrZf, true),
            Cond::S => (Base::Sf, false),
            Cond::Ns => (Base::Sf, true),
            Cond::P => (Base::Pf, false),
            Cond::Np => (Base::Pf, true),
            Cond::L => (Base::SfXorOf, false),
            Cond::Ge => (Base::SfXorOf, true),
            Cond::Le => (Base::ZfOrSfXorOf, false),
            Cond::G => (Base::ZfOrSfXorOf, true),
        }
    }
}

/// The eight flag expressions the sixteen conditions are built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Base {
    Cf,
    Zf,
    Sf,
    Of,
    Pf,
    /// `CF | ZF`
    CfOrZf,
    /// `SF ^ OF`
    SfXorOf,
    /// `ZF | (SF ^ OF)`
    ZfOrSfXorOf,
}

impl<'a> Lifter<'a> {
    /// Read one flag as its `I8` boolean (0 or 1).
    fn flag(&mut self, flag: X86Flag) -> Value {
        self.builder.use_var(self.state.flags[&flag])
    }

    /// Invert an `I8` boolean (0 -> 1, 1 -> 0) with an equality compare.
    fn bool_not(&mut self, val: Value) -> Value {
        self.builder.ins().icmp_imm_s(IntCC::Equal, val, 0)
    }

    /// Build `cond`'s flag expression, as `(value, invert)`: the condition holds
    /// when `value` is nonzero, unless `invert`, in which case it holds when
    /// `value` is zero.
    ///
    /// Callers that need a plain boolean should use
    /// [`eval_cond_value`](Self::eval_cond_value); the split form exists for
    /// `Jcc`, which spends the inversion on its branch destinations.
    fn eval_cond(&mut self, cond: Cond) -> Value {
        // Flags are read in the order the hand-written handlers read them, so
        // the resulting IR is value-for-value what they emitted.
        match cond.base().0 {
            Base::Cf => self.flag(X86Flag::Carry),
            Base::Zf => self.flag(X86Flag::Zero),
            Base::Sf => self.flag(X86Flag::Sign),
            Base::Of => self.flag(X86Flag::Overflow),
            Base::Pf => self.flag(X86Flag::Parity),
            Base::CfOrZf => {
                let cf = self.flag(X86Flag::Carry);
                let zf = self.flag(X86Flag::Zero);
                self.builder.ins().bor(cf, zf)
            }
            Base::SfXorOf => {
                let sf = self.flag(X86Flag::Sign);
                let of = self.flag(X86Flag::Overflow);
                self.builder.ins().bxor(sf, of)
            }
            Base::ZfOrSfXorOf => {
                let zf = self.flag(X86Flag::Zero);
                let sf = self.flag(X86Flag::Sign);
                let of = self.flag(X86Flag::Overflow);
                let sf_xor_of = self.builder.ins().bxor(sf, of);
                self.builder.ins().bor(zf, sf_xor_of)
            }
        }
    }

    /// `cond` as a plain `I8` boolean, complementing the base expression when
    /// the condition is the negated member of its pair.
    fn eval_cond_value(&mut self, cond: Cond) -> Value {
        let val = self.eval_cond(cond);
        if cond.base().1 {
            self.bool_not(val)
        } else {
            val
        }
    }

    /// `SETcc` - write the condition to the byte destination.
    pub(super) fn handle_setcc(
        &mut self,
        instruction: &Instruction,
        cond: Cond,
    ) -> crate::Result<()> {
        let val = self.eval_cond_value(cond);
        self.write_operand(instruction, 0, val)
    }

    /// `CMOVcc` - move the source to the destination when the condition holds.
    ///
    /// Both operands are read unconditionally; the choice is a `select`, not a
    /// branch, which is also what hardware does.
    pub(super) fn handle_cmovcc(
        &mut self,
        instruction: &Instruction,
        cond: Cond,
    ) -> crate::Result<()> {
        let val = self.eval_cond_value(cond);
        let size = self.operand_size(instruction, 0)?;
        let dst = self.read_operand(instruction, 0, size)?;
        let src = self.read_operand(instruction, 1, size)?;
        let result = self.builder.ins().select(val, src, dst);
        self.write_operand(instruction, 0, result)
    }

    /// `Jcc` - branch to the near target when the condition holds, otherwise
    /// fall through.
    ///
    /// The inversion is spent on the `brif` destinations rather than on an
    /// extra compare, so `JAE` costs exactly the `CF` read.
    pub(super) fn handle_jcc(
        &mut self,
        instruction: &Instruction,
        cond: Cond,
    ) -> crate::Result<()> {
        let val = self.eval_cond(cond);
        if cond.base().1 {
            self.handle_conditional_branch_inverted(instruction, val)
        } else {
            self.handle_conditional_branch(instruction, val)
        }
    }
}
