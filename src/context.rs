//! Compatibility re-exports for guest register context and exit/control types.
//!
//! The architecture-neutral exit types live in [`crate::core::exit`]; the
//! x86-64 `RegisterContext` lives in [`crate::arch::x86_64::context`]. These
//! re-exports keep existing `crate::context::…` / `microlift::context::…` paths
//! working.

pub use crate::arch::x86_64::context::RegisterContext;
pub use crate::core::exit::{ControlHeader, ExitReason, MemorySegment, TransferKind};
