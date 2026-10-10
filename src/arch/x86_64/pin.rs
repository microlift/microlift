//! Names for x86-64's pinnable registers ([`PinnedReg`] class + number).
//!
//! Numbering is [`GPR_ORDER`](crate::arch::x86_64::regs::GPR_ORDER) for integers, the XMM number
//! for vectors. No recommended set: see
//! [`PinnedAbi::check_budget`](crate::core::pin::PinnedAbi::check_budget).

use crate::core::pin::PinnedReg;

pub const RAX: PinnedReg = PinnedReg::gpr(0);
pub const RCX: PinnedReg = PinnedReg::gpr(1);
pub const RDX: PinnedReg = PinnedReg::gpr(2);
pub const RBX: PinnedReg = PinnedReg::gpr(3);
pub const RSP: PinnedReg = PinnedReg::gpr(4);
pub const RBP: PinnedReg = PinnedReg::gpr(5);
pub const RSI: PinnedReg = PinnedReg::gpr(6);
pub const RDI: PinnedReg = PinnedReg::gpr(7);
pub const R8: PinnedReg = PinnedReg::gpr(8);
pub const R9: PinnedReg = PinnedReg::gpr(9);
pub const R10: PinnedReg = PinnedReg::gpr(10);
pub const R11: PinnedReg = PinnedReg::gpr(11);
pub const R12: PinnedReg = PinnedReg::gpr(12);
pub const R13: PinnedReg = PinnedReg::gpr(13);
pub const R14: PinnedReg = PinnedReg::gpr(14);
pub const R15: PinnedReg = PinnedReg::gpr(15);

/// `XMM<n>`; pinnable only with
/// [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm), else pinning is disabled for the whole lift.
pub const fn xmm(n: u16) -> PinnedReg {
    PinnedReg::vector(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::regs::GPR_ORDER;

    #[test]
    fn the_names_are_the_order_the_rest_of_the_frontend_uses() {
        // `GPR_ORDER` is the single definition of the numbering.
        for (i, name) in [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI].iter().enumerate() {
            assert_eq!(name.index as usize, i);
        }
        assert_eq!(GPR_ORDER[RSP.index as usize], iced_x86::Register::RSP);
        assert_eq!(GPR_ORDER[R15.index as usize], iced_x86::Register::R15);
    }
}
