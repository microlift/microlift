//! Host callbacks lifted code calls **inline**, instead of exiting to the host.
//!
//! Two guest events are serviced this way: a system call, and an instruction
//! this frontend does not lift. Both used to be VM exits; both are now an
//! indirect call through a function pointer the host installs in the guest's
//! `RegisterContext`, after which the region keeps running. That removes an
//! exit, a re-entry and a resume point from the two paths a real guest hits
//! most often.
//!
//! The contract on the host side:
//!
//! - The callback runs with the whole guest state synced into the context and
//!   the PC pointing at the resume address (past a `syscall`, *at* the
//!   instruction the interpreter is asked to run). It may write any field.
//! - Returning normally resumes the guest at whatever PC the context then
//!   holds - unchanged for the ordinary case, redirected for a callback that
//!   moves control (signal delivery, `sigreturn`, a thread exiting). A moved PC
//!   is dispatched like an indirect jump, so it may leave the region.
//! - There is no "unset" state: a context always carries callbacks, whose
//!   defaults abort rather than return into a guest that would then run with
//!   an unserviced event behind it.

/// Where a guest's `RegisterContext` keeps its host callbacks. Returned by
/// [`GuestArch::callback_offsets`](crate::core::GuestArch::callback_offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackOffsets {
    /// `unsafe extern "C" fn(ctx: *mut Context)`, servicing one system call.
    pub syscall: i32,
    /// `unsafe extern "C" fn(ctx: *mut Context, instructions: u64)`, running
    /// that many guest instructions from the context's PC.
    pub interpreter_fallback: i32,
}
