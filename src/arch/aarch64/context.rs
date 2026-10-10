//! aarch64 guest register context: the flat `#[repr(C)]` struct passed to lifted code.
//!
//! Embeds [`ControlHeader`] (reached via `Deref`), like the x86-64 context.

use crate::core::exit::ControlHeader;

/// Host routines lifted code calls inline; twin of
/// [`X86_64Callbacks`](crate::arch::x86_64::context::X86_64Callbacks). Contract: [`crate::core::callback`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aarch64Callbacks {
    /// Services one `SVC`; `pc` is the return address.
    pub handle_syscall: unsafe extern "C" fn(ctx: *mut Aarch64RegisterContext),
    /// Interprets `num_instructions` unlifted instructions from `pc`.
    pub interpreter_fallback:
        unsafe extern "C" fn(ctx: *mut Aarch64RegisterContext, num_instructions: u64),
}

/// Aborts: running on past an unserviced syscall would corrupt the guest.
unsafe extern "C" fn default_syscall_handler(_ctx: *mut Aarch64RegisterContext) {
    panic!("no syscall callback registered");
}

/// Aborts, like [`default_syscall_handler`].
unsafe extern "C" fn default_interpreter_fallback(
    _ctx: *mut Aarch64RegisterContext,
    _num_instructions: u64,
) {
    panic!("no interpreter-fallback callback registered");
}

impl Default for Aarch64Callbacks {
    fn default() -> Self {
        Self {
            handle_syscall: default_syscall_handler,
            interpreter_fallback: default_interpreter_fallback,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aarch64RegisterContext {
    /// X0-X30; X31 (XZR) is not stored. X30 is LR.
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub callbacks: Aarch64Callbacks,
    /// Placeholder, unused.
    pub resume_function: Option<unsafe extern "C" fn(ctx: *mut Aarch64RegisterContext)>,
    /// NZCV, one byte (0 or 1) each.
    pub n: u8,
    pub z: u8,
    pub c: u8,
    pub vf: u8,
    pub control: ControlHeader,
    pub v: [[u64; 2]; 32],
    pub tpidr_el0: u64,
}

impl Default for Aarch64RegisterContext {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            callbacks: Aarch64Callbacks::default(),
            resume_function: None,
            n: 0,
            z: 0,
            c: 0,
            vf: 0,
            control: ControlHeader::ZERO,
            v: [[0; 2]; 32],
            tpidr_el0: 0,
        }
    }
}

impl Aarch64RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::ops::Deref for Aarch64RegisterContext {
    type Target = ControlHeader;

    fn deref(&self) -> &ControlHeader {
        &self.control
    }
}

impl std::ops::DerefMut for Aarch64RegisterContext {
    fn deref_mut(&mut self) -> &mut ControlHeader {
        &mut self.control
    }
}
