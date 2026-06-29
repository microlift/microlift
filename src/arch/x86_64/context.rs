//! x86-64 guest register context (the FFI struct passed to lifted code).
//!
//! Kept as a flat `#[repr(C)]` struct so existing host/test code that reads
//! fields by name (`ctx.rax`, `ctx.rip`, `ctx.exit_reason`, …) is unaffected.
//! The architecture-neutral control fields (`exit_reason`, `exit_code`,
//! `imported_function_va`, `image_base`) mirror [`crate::core::exit::ControlHeader`]
//! conceptually but stay inline to preserve the established layout.

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct RegisterContext {
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rbx: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub image_base: u64,
    /// A syscall handler intended for the RuntimeAssisted execution mode.
    pub syscall_handler: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    /// FIXME:
    /// # **DO NOT USE!!!**
    /// Resume function pointer for AOT block dispatch (**not implemented yet** - this is a placeholder)
    pub resume_function: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    // Flags as individual bytes for simpler IR load/store
    pub cf: u8,
    pub pf: u8,
    pub af: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    /// Direction flag (DF, RFLAGS bit 10). Controls whether string operations
    /// advance their pointers upward (`DF=0`) or downward (`DF=1`).
    pub df: u8,
    pub exit_reason: u32,
    pub exit_code: u32,
    /// On `ImportedFunctionCall` exits, the virtual address of the imported
    /// function that was called. `rip` points to the instruction *after* the
    /// `CALL` - the resume point once the host has handled the call.
    pub imported_function_va: u64,
    /// XMM0–XMM15, each stored as [lo_64, hi_64].
    pub xmm: [[u64; 2]; 16],
    /// FS segment base address. In x86-64 user mode this is set via
    /// `arch_prctl(ARCH_SET_FS)` or `wrfsbase` and used for thread-local
    /// storage. Reads/writes through `fs:` prefixed addresses add this base.
    pub fs_base: u64,
    /// GS segment base address. On Windows this points to the TEB; on Linux
    /// it is often used for kernel->user communication via `swapgs`.
    /// Reads/writes through `gs:` prefixed addresses add this base.
    pub gs_base: u64,
}

impl RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}
