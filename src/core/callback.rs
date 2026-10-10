//! Host callbacks (syscall, unlifted instruction) called inline through context fn pointers.
//!
//! - Runs with full state synced; PC is past a `syscall`, *at* an interpreted instruction.
//! - On return the guest resumes at the context's PC; a moved PC dispatches as an indirect jump.
//! - Contexts always carry callbacks; defaults abort.

/// Context offsets of the host callbacks ([`GuestArch::callback_offsets`](crate::core::GuestArch::callback_offsets)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackOffsets {
    /// `unsafe extern "C" fn(ctx: *mut Context)`.
    pub syscall: i32,
    /// `unsafe extern "C" fn(ctx: *mut Context, instructions: u64)`, runs that many instructions from PC.
    pub interpreter_fallback: i32,
}
