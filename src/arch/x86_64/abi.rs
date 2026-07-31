//! x86-64 calling-convention descriptors for fast-import calls.
//!
//! Provides the concrete [`CallAbi`] presets a caller registers imports with,
//! plus the translation between the arch-neutral [`GuestReg`] identifiers used
//! by [`CallAbi`] and `iced_x86::Register`.

use crate::MicroliftError;
use crate::core::abi::{CallAbi, GuestReg, StackArgs};
use iced_x86::Register;

impl From<Register> for GuestReg {
    fn from(reg: Register) -> Self {
        GuestReg(reg as u32)
    }
}

/// Resolve a [`GuestReg`] back to the x86-64 register it names.
pub fn x86_reg(reg: GuestReg) -> crate::Result<Register> {
    Register::try_from(reg.0 as usize).map_err(|_| {
        MicroliftError::UnsupportedRegister(format!("invalid guest register id {}", reg.0))
    })
}

/// Both x86-64 ABIs pass spilled arguments in 8-byte stack slots.
const X86_64_SLOT_BYTES: u32 = 8;

impl CallAbi {
    /// System V AMD64 (Linux, macOS, the BSDs).
    ///
    /// Integer/pointer arguments in RDI, RSI, RDX, RCX, R8, R9; floating-point
    /// arguments in XMM0–XMM7, counted independently of the integer arguments;
    /// anything further on the stack. Integer returns in RAX, FP returns in
    /// XMM0.
    pub fn sysv_amd64() -> Self {
        Self {
            name: "sysv_amd64",
            int_arg_regs: vec![
                Register::RDI.into(),
                Register::RSI.into(),
                Register::RDX.into(),
                Register::RCX.into(),
                Register::R8.into(),
                Register::R9.into(),
            ],
            fp_arg_regs: (0..8).collect(),
            unified_arg_slots: false,
            stack_args: Some(StackArgs {
                reserved_bytes: 0,
                slot_bytes: X86_64_SLOT_BYTES,
            }),
            int_ret_reg: Register::RAX.into(),
            fp_ret_reg: 0,
            host_call_conv: None,
        }
    }

    /// Microsoft x64 (the single convention 64-bit Windows uses - what `stdcall`,
    /// `fastcall` and `cdecl` all collapse to when compiled for x86-64).
    ///
    /// Four positional argument slots: RCX/RDX/R8/R9 for integers, XMM0–XMM3 for
    /// floats, and an FP argument consumes the integer slot at its position (and
    /// vice versa). Arguments five and up go on the stack above the caller's
    /// 32-byte shadow space. Integer returns in RAX, FP returns in XMM0.
    pub fn win64() -> Self {
        Self {
            name: "win64",
            int_arg_regs: vec![
                Register::RCX.into(),
                Register::RDX.into(),
                Register::R8.into(),
                Register::R9.into(),
            ],
            fp_arg_regs: (0..4).collect(),
            unified_arg_slots: true,
            stack_args: Some(StackArgs {
                reserved_bytes: 32,
                slot_bytes: X86_64_SLOT_BYTES,
            }),
            int_ret_reg: Register::RAX.into(),
            fp_ret_reg: 0,
            host_call_conv: None,
        }
    }

    /// The convention x86-64 guest code built for *this* host would use:
    /// [`win64`](CallAbi::win64) on Windows, [`sysv_amd64`](CallAbi::sysv_amd64)
    /// everywhere else.
    ///
    /// Convenient when lifting code that came from the same platform Microlift
    /// runs on; anything else should name its ABI explicitly.
    pub fn x86_64_host() -> Self {
        #[cfg(target_os = "windows")]
        {
            Self::win64()
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self::sysv_amd64()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::abi::ArgLocation;
    use crate::function_traits::FunctionArgumentType as F;

    #[test]
    fn guest_reg_roundtrips() {
        for reg in [Register::RDI, Register::RAX, Register::R9, Register::RCX] {
            assert_eq!(x86_reg(GuestReg::from(reg)).unwrap(), reg);
        }
    }

    #[test]
    fn sysv_counts_int_and_fp_independently() {
        let abi = CallAbi::sysv_amd64();
        let placed = abi
            .place_args(&[F::None, F::Pointer, F::DoublePrecisionFp, F::Pointer])
            .unwrap();
        assert_eq!(placed[0].1, ArgLocation::IntReg(Register::RDI.into()));
        assert_eq!(placed[1].1, ArgLocation::FpReg(0));
        assert_eq!(placed[2].1, ArgLocation::IntReg(Register::RSI.into()));
    }

    #[test]
    fn win64_shares_positional_slots() {
        let abi = CallAbi::win64();
        let placed = abi
            .place_args(&[F::None, F::Pointer, F::DoublePrecisionFp, F::Pointer])
            .unwrap();
        assert_eq!(placed[0].1, ArgLocation::IntReg(Register::RCX.into()));
        // Second argument -> XMM1, not XMM0: the FP register index follows the
        // argument position, not the count of FP arguments seen so far.
        assert_eq!(placed[1].1, ArgLocation::FpReg(1));
        assert_eq!(placed[2].1, ArgLocation::IntReg(Register::R8.into()));
    }

    #[test]
    fn spilled_args_land_on_the_stack() {
        let sysv = CallAbi::sysv_amd64();
        let seven = vec![F::None; 1]
            .into_iter()
            .chain(std::iter::repeat(F::Pointer).take(7))
            .collect::<Vec<_>>();
        let placed = sysv.place_args(&seven).unwrap();
        assert_eq!(placed[6].1, ArgLocation::Stack(0));

        // Microsoft x64 reserves 32 bytes of shadow space first.
        let five = vec![F::None; 1]
            .into_iter()
            .chain(std::iter::repeat(F::Pointer).take(5))
            .collect::<Vec<_>>();
        let placed = CallAbi::win64().place_args(&five).unwrap();
        assert_eq!(placed[4].1, ArgLocation::Stack(32));
    }

    #[test]
    fn abi_without_stack_rule_rejects_spilled_args() {
        let abi = CallAbi::sysv_amd64();
        let no_stack = CallAbi {
            stack_args: None,
            ..abi
        };
        let seven = vec![F::None; 1]
            .into_iter()
            .chain(std::iter::repeat(F::Pointer).take(7))
            .collect::<Vec<_>>();
        assert!(!no_stack.can_fast_call(&seven));
    }
}
