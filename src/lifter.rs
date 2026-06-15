//! Compatibility re-exports.
//!
//! The guest-agnostic lift engine now lives in [`crate::core`] and the x86-64
//! frontend in [`crate::arch::x86_64`]. `Lifter<'a>` is the x86-64
//! monomorphization of the generic `LiftEngine`. These re-exports keep existing
//! `crate::lifter::…` and `microlift::lifter::…` paths working.

pub use crate::arch::x86_64::Lifter;
pub use crate::arch::x86_64::regs::{RegisterState, X86Flag};
pub use crate::core::{ExecutionMode, GuestArch, LiftEngine, LifterConfig};
