//! Compatibility re-exports: exit types from [`crate::core::exit`], x86-64
//! `RegisterContext` from [`crate::arch::x86_64::context`].

pub use crate::arch::x86_64::context::{
    MXCSR_MASK, MXCSR_RESET, RegisterContext, X87_CW_RESET, f64_bits_to_f80, f80_to_f64_bits,
};
pub use crate::core::exit::{ControlHeader, ExitReason, MemorySegment, TransferKind};
