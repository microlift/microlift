//! Condition codes shared by `Jcc`, `SETcc` and `CMOVcc`.
//!
//! Sixteen conditions built from eight base predicates and their negations.

use crate::arch::x86_64::semantics::flags::FlagOrigin;
use crate::lifter::{Lifter, X86Flag};
use cranelift::codegen::ir::condcodes::CondCode;
use cranelift::prelude::*;
use iced_x86::Instruction;

/// One x86 condition code; mnemonic aliases (`JZ` = `JE`, ...) are resolved by the caller.
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
    /// The base condition this one negates, and whether the negation applies: `(Le, true)` for `G`.
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

/// A condition settled from the last flag writer's own operands.
enum Fused {
    Compare(IntCC, Value, Value),
    Constant(bool),
}

impl Fused {
    fn complement(self) -> Self {
        match self {
            Fused::Compare(cc, a, b) => Fused::Compare(cc.complement(), a, b),
            Fused::Constant(holds) => Fused::Constant(!holds),
        }
    }
}

impl Base {
    /// `self` as one compare on what `origin` computed the flags from, where the table has one.
    fn fused(self, origin: FlagOrigin, zero: Value) -> Option<Fused> {
        use Fused::{Compare, Constant};
        match (origin, self) {
            (FlagOrigin::Sub { lhs, rhs, .. }, Base::Cf) => {
                Some(Compare(IntCC::UnsignedLessThan, lhs, rhs))
            }
            (FlagOrigin::Sub { lhs, rhs, .. }, Base::Zf) => Some(Compare(IntCC::Equal, lhs, rhs)),
            (FlagOrigin::Sub { lhs, rhs, .. }, Base::CfOrZf) => {
                Some(Compare(IntCC::UnsignedLessThanOrEqual, lhs, rhs))
            }
            (FlagOrigin::Sub { lhs, rhs, .. }, Base::SfXorOf) => {
                Some(Compare(IntCC::SignedLessThan, lhs, rhs))
            }
            (FlagOrigin::Sub { lhs, rhs, .. }, Base::ZfOrSfXorOf) => {
                Some(Compare(IntCC::SignedLessThanOrEqual, lhs, rhs))
            }
            (FlagOrigin::Logical { .. }, Base::Cf | Base::Of) => Some(Constant(false)),
            (FlagOrigin::Logical { res }, Base::Zf | Base::CfOrZf) => {
                Some(Compare(IntCC::Equal, res, zero))
            }
            (FlagOrigin::Logical { res }, Base::SfXorOf) => {
                Some(Compare(IntCC::SignedLessThan, res, zero))
            }
            (FlagOrigin::Logical { res }, Base::ZfOrSfXorOf) => {
                Some(Compare(IntCC::SignedLessThanOrEqual, res, zero))
            }
            (origin, Base::Zf) => Some(Compare(IntCC::Equal, origin.result(), zero)),
            (origin, Base::Sf) => Some(Compare(IntCC::SignedLessThan, origin.result(), zero)),
            _ => None,
        }
    }
}

impl<'a> Lifter<'a> {
    fn flag(&mut self, flag: X86Flag) -> Value {
        // PF/AF are stored as source values; `get_flag` derives them.
        self.get_flag(flag)
    }

    fn bool_not(&mut self, val: Value) -> Value {
        self.builder.ins().icmp_imm_s(IntCC::Equal, val, 0)
    }

    /// `cond` as one compare on the last flag writer's operands, when that writer is known here.
    fn fused_cond(&mut self, cond: Cond) -> Option<Value> {
        let origin = self.flag_origin()?;
        let ty = self.builder.func.dfg.value_type(origin.result());
        let zero = self.builder.ins().iconst(ty, 0);
        let (base, negated) = cond.base();
        let fused = base.fused(origin, zero)?;
        Some(match if negated { fused.complement() } else { fused } {
            Fused::Compare(cc, a, b) => self.builder.ins().icmp(cc, a, b),
            Fused::Constant(holds) => self.builder.ins().iconst(types::I8, holds as i64),
        })
    }

    /// The base of `cond` from the flag variables: `cond` holds when it is nonzero, unless `cond` negates it.
    fn eval_cond(&mut self, cond: Cond) -> Value {
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

    /// `cond` as a plain `I8` boolean.
    fn eval_cond_value(&mut self, cond: Cond) -> Value {
        if let Some(fused) = self.fused_cond(cond) {
            return fused;
        }
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

    /// `CMOVcc` - move source to destination when the condition holds (a `select`; both operands read).
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

    /// `Jcc` - branch to the near target when the condition holds; inversion lands on the `brif` destinations.
    pub(super) fn handle_jcc(
        &mut self,
        instruction: &Instruction,
        cond: Cond,
    ) -> crate::Result<()> {
        if let Some(fused) = self.fused_cond(cond) {
            return self.handle_conditional_branch(instruction, fused);
        }
        let val = self.eval_cond(cond);
        if cond.base().1 {
            self.handle_conditional_branch_inverted(instruction, val)
        } else {
            self.handle_conditional_branch(instruction, val)
        }
    }
}
