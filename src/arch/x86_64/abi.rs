//! x86-64 calling-convention presets for fast imports, plus [`GuestReg`] <-> `iced_x86::Register` translation.

use crate::MicroliftError;
use crate::arch::x86_64::regs::gpr_index;
use crate::core::abi::{ArgLocation, CallAbi, GuestReg, StackArgs};
use crate::core::bridge::{BridgeEffects, RegMask};
use crate::function_traits::FunctionArgumentType;
use iced_x86::Register;

impl From<Register> for GuestReg {
    fn from(reg: Register) -> Self {
        GuestReg(reg as u32)
    }
}

pub fn x86_reg(reg: GuestReg) -> crate::Result<Register> {
    Register::try_from(reg.0 as usize).map_err(|_| {
        MicroliftError::UnsupportedRegister(format!("invalid guest register id {}", reg.0))
    })
}

/// [`RegMask`] over x86-64 GPRs; sub-registers select their containing 64-bit register
/// (`EDI` and `DIL` mean `RDI`). Non-GPRs are ignored; see [`xmm_mask`].
/// ```
/// use iced_x86::Register;
/// use microlift::arch::x86_64::abi::gpr_mask;
///
/// // `int foo(int, char *)` under SysV reads EDI and RSI.
/// let reads = gpr_mask(&[Register::EDI, Register::RSI]);
/// assert!(reads.has_gpr(7) && reads.has_gpr(6));
/// ```
pub fn gpr_mask(regs: &[Register]) -> RegMask {
    let mut mask = RegMask::NONE;
    for &reg in regs {
        if let Some(idx) = gpr_index(reg) {
            mask = mask.with_gpr(idx);
        }
    }
    mask
}

/// [`RegMask`] over XMM indices (`0` = `XMM0`).
pub fn xmm_mask(indices: &[u8]) -> RegMask {
    let mut mask = RegMask::NONE;
    for &idx in indices {
        mask = mask.with_vec(idx);
    }
    mask
}

/// [`BridgeEffects`] for a bridge following `abi`: reads the registers `abi` would place `arg_types` in,
/// clobbers the return register and flags. `None` when `abi` cannot place every argument.
///
/// Not a safety net: it cannot know about guest-memory reads, varargs or other clobbers; narrow
/// [`BridgeEffects::full`] by hand if unsure.
pub fn abi_bridge_effects(
    abi: &CallAbi,
    arg_types: &[FunctionArgumentType],
) -> Option<BridgeEffects> {
    let mut reads = RegMask::NONE;
    for (_, loc) in abi.place_args(arg_types)? {
        match loc {
            ArgLocation::IntReg(r) => {
                if let Some(idx) = gpr_index(x86_reg(r).ok()?) {
                    reads = reads.with_gpr(idx);
                }
            }
            ArgLocation::FpReg(i) => reads = reads.with_vec(i),
            // Stack arguments go through SP, which sync always stores.
            ArgLocation::Stack(_) => {}
        }
    }
    let mut writes = RegMask::NONE.with_flags();
    if let Some(ret) = arg_types
        .first()
        .copied()
        .and_then(|t| abi.return_location(t))
    {
        match ret {
            ArgLocation::IntReg(r) => {
                if let Some(idx) = gpr_index(x86_reg(r).ok()?) {
                    writes = writes.with_gpr(idx);
                }
            }
            ArgLocation::FpReg(i) => writes = writes.with_vec(i),
            ArgLocation::Stack(_) => return None,
        }
    }
    Some(BridgeEffects::new(reads, writes))
}

/// Spilled arguments use 8-byte stack slots in both ABIs.
const X86_64_SLOT_BYTES: u32 = 8;

impl CallAbi {
    /// System V AMD64: integer args in RDI, RSI, RDX, RCX, R8, R9; FP in XMM0-7 (counted
    /// independently); rest on the stack. Returns in RAX / XMM0.
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

    /// Microsoft x64: four positional slots (RCX/RDX/R8/R9 or XMM0-3, an FP arg consumes the
    /// integer slot at its position); rest on the stack above 32 bytes of shadow space. Returns in RAX / XMM0.
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

    /// The convention for this host: [`win64`](CallAbi::win64) on Windows, else [`sysv_amd64`](CallAbi::sysv_amd64).
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
        // FP register index follows argument position, not FP-argument count.
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

        // 32 bytes of shadow space come first.
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
