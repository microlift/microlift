//! aarch64 guest register context (the FFI struct passed to lifted code).
//!
//! Flat `#[repr(C)]` struct, mirroring the x86-64 design - including the
//! embedded [`ControlHeader`], which is where the architecture-neutral control
//! fields live for every guest. `Deref`/`DerefMut` forward to it, so
//! `ctx.exit_reason` resolves as if the field were declared here.

use crate::core::exit::ControlHeader;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aarch64RegisterContext {
    /// General-purpose registers X0–X30. X31 is the zero register (XZR/WZR) and
    /// is not stored. X30 (index 30) is the link register (LR).
    pub x: [u64; 31],
    /// Stack pointer (SP). Encoded separately from XZR per the A64 register model.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// A syscall handler intended for the RuntimeAssisted execution mode.
    pub syscall_handler: Option<unsafe extern "C" fn(ctx: *mut Aarch64RegisterContext)>,
    /// Resume function pointer for AOT block dispatch (placeholder, unused).
    pub resume_function: Option<unsafe extern "C" fn(ctx: *mut Aarch64RegisterContext)>,
    // NZCV condition flags, each stored as an individual byte (0 or 1) for
    // simpler IR load/store, mirroring the x86 flag-byte design.
    /// Negative flag.
    pub n: u8,
    /// Zero flag.
    pub z: u8,
    /// Carry flag.
    pub c: u8,
    /// Overflow flag.
    pub vf: u8,
    /// The architecture-neutral control fields, shared verbatim with the
    /// x86-64 context. Reachable unqualified through this struct's `Deref`.
    pub control: ControlHeader,
    /// SIMD/FP registers V0–V31, each stored as [lo_64, hi_64].
    pub v: [[u64; 2]; 32],
    /// Thread-pointer base (`TPIDR_EL0`), used for thread-local storage.
    pub tpidr_el0: u64,
}

impl Default for Aarch64RegisterContext {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            syscall_handler: None,
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
