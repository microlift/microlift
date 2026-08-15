//! x86-64's pinnable registers, by name.
//!
//! [`PinnedReg`] is a class and an architectural number, which is what keeps
//! [`crate::core::pin`] free of any guest's register names - and unreadable at a
//! call site. These constants are the readable spelling, and the numbering they
//! encode is [`GPR_ORDER`](crate::arch::x86_64::regs::GPR_ORDER) for the
//! integer registers and the XMM number for the vector ones.
//!
//! There is deliberately no recommended set here. Which registers are worth
//! a slot is a property of the guest program - a C++ engine spends its
//! time in `RSP`/`RBP`/`RDI` and a codec in `XMM0`-`XMM3` - and microlift has no
//! way to know which it is looking at. See
//! [`PinnedAbi::check_budget`](crate::core::pin::PinnedAbi::check_budget) for
//! how many slots there are to spend.

use crate::core::pin::PinnedReg;

/// `RAX`.
pub const RAX: PinnedReg = PinnedReg::gpr(0);
/// `RCX`.
pub const RCX: PinnedReg = PinnedReg::gpr(1);
/// `RDX`.
pub const RDX: PinnedReg = PinnedReg::gpr(2);
/// `RBX`.
pub const RBX: PinnedReg = PinnedReg::gpr(3);
/// `RSP`.
pub const RSP: PinnedReg = PinnedReg::gpr(4);
/// `RBP`.
pub const RBP: PinnedReg = PinnedReg::gpr(5);
/// `RSI`.
pub const RSI: PinnedReg = PinnedReg::gpr(6);
/// `RDI`.
pub const RDI: PinnedReg = PinnedReg::gpr(7);
/// `R8`.
pub const R8: PinnedReg = PinnedReg::gpr(8);
/// `R9`.
pub const R9: PinnedReg = PinnedReg::gpr(9);
/// `R10`.
pub const R10: PinnedReg = PinnedReg::gpr(10);
/// `R11`.
pub const R11: PinnedReg = PinnedReg::gpr(11);
/// `R12`.
pub const R12: PinnedReg = PinnedReg::gpr(12);
/// `R13`.
pub const R13: PinnedReg = PinnedReg::gpr(13);
/// `R14`.
pub const R14: PinnedReg = PinnedReg::gpr(14);
/// `R15`.
pub const R15: PinnedReg = PinnedReg::gpr(15);

/// `XMM<n>`.
///
/// Pinnable only with [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm):
/// in the scalar representation an XMM register is two independent variables,
/// and a slot is one value. Asking for one without it disables pinning for the
/// whole lift rather than pinning half a register.
pub const fn xmm(n: u16) -> PinnedReg {
    PinnedReg::vector(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::regs::GPR_ORDER;

    #[test]
    fn the_names_are_the_order_the_rest_of_the_frontend_uses() {
        // The mapping these constants assert is the one `pinned_slot_var`
        // performs, and `GPR_ORDER` is its single definition.
        for (i, name) in [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI].iter().enumerate() {
            assert_eq!(name.index as usize, i);
        }
        assert_eq!(GPR_ORDER[RSP.index as usize], iced_x86::Register::RSP);
        assert_eq!(GPR_ORDER[R15.index as usize], iced_x86::Register::R15);
    }
}
