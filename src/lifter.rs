//! Compatibility re-exports of [`crate::core`] (engine) and [`crate::arch::x86_64`] (frontend).

pub use crate::arch::x86_64::Lifter;
pub use crate::arch::x86_64::regs::{RegisterState, X86Flag};
pub use crate::core::abi::{ArgLocation, CallAbi, GuestReg, StackArgs};
pub use crate::core::{
    AddressMode, Atomic128Mode, ExecutionMode, FastCall, FastCallTarget, FastImport, GuestArch,
    LiftEngine, LifterConfig, VmResumptionPoints,
};
