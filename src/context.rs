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
    pub exit_reason: u32,
    pub exit_code: u32,
    /// On `ImportedFunctionCall` exits, contains the virtual address of the
    /// imported function that was called (i.e. the value of RIP at the `CALL`
    /// target, before execution was intercepted). For indirect calls this is
    /// the runtime-resolved target address. The host uses this to look up the
    /// function and dispatch it. `rip` points to the instruction *after* the
    /// `CALL` — the resume point once the host has handled the call.
    pub imported_function_va: u64,
    /// XMM0–XMM15, each stored as [lo_64, hi_64].
    pub xmm: [[u64; 2]; 16],
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    None = 0,
    /// A terminating block was reached. The code has executed successfully up to its VM return trampoline.
    Return = 1,
    /// A system call was issued by the guest.
    Syscall = 2,
    /// The code would have branched out to an unresolvable address in memory.
    /// It is up to the runtime to decide what to do with this.
    UnknownJump = 3,
    /// The instruction that was about to be executed is not implemented in Microlift.
    /// This becomes a resumable VM exit in RuntimeAssisted mode when ran with `interpreter_fallback`.
    UnimplementedInstruction = 4,
    /// The code that was about to be executed did not decode as
    /// valid x64.
    InvalidInstruction = 5,
    /// A `CALL` to a registered import address was intercepted (and the import
    /// either had no fast-call metadata or has a variadic/unsupported ABI), or
    /// an indirect `CALL` (via register or memory) was encountered.
    ///
    /// `RegisterContext::imported_function_va` holds the callee address.
    /// `RegisterContext::rip` holds the address of the instruction after the
    /// `CALL` - i.e. where execution should resume after the host handles it.
    ImportedFunctionCall = 6,
}

/// Describes what kind of x86 control transfer caused an `UnknownJump` VM exit.
///
/// Stored in `RegisterContext::exit_code` whenever `exit_reason == UnknownJump`.
/// All other exit reasons leave `exit_code` with its own semantics (e.g.
/// `UnimplementedInstruction` stores the raw `iced_x86::Code` there).
///
/// The runtime can branch on this to decide how to handle the exit:
/// - `DirectJump` / `IndirectJump` - no return address was pushed; the guest
///   simply transferred to a new location. The runtime must find and execute
///   the target (at `ctx.rip`) on behalf of the guest.
/// - `ConditionalJump` - same as `DirectJump` semantically, but the source
///   was a `Jcc` whose taken edge landed outside the lifted region.
/// - `Return` - a `RET` popped a return address that didn't match the exit
///   trampoline or any in-region block. The guest is returning to a caller
///   that lives outside the lifted region.
/// - `DirectCall` - an unconditional `CALL` whose static target is outside the
///   lifted region and was not registered as an import. A return address has
///   already been pushed onto the guest stack; `ctx.rip` is the return address
///   and `ctx.exit_code` identifies the callee address via a secondary field
///   (see note below). Treat like a function call that must resume.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// Origin not tracked - should not occur in practice; treat conservatively.
    Unknown = 0,
    /// Unconditional `JMP` to a static (near-branch) target outside the region.
    DirectJump = 1,
    /// `JMP` via register or memory operand; target was only known at runtime.
    IndirectJump = 2,
    /// A `Jcc` conditional branch whose taken edge fell outside the lifted region.
    ConditionalJump = 3,
    /// A `RET` that popped a return address outside the lifted region (and not the
    /// exit trampoline).
    Return = 4,
    /// An unconditional `CALL` to a static out-of-region address that was not
    /// registered as an import. A return address was pushed; the runtime must
    /// call the target and then resume at that address.
    DirectCall = 5,
}

impl RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct MemorySegment {
    pub address: u64,
    pub data: Vec<u8>,
    pub writable: bool,
}

impl MemorySegment {
    pub fn new(address: u64, data: Vec<u8>, writable: bool) -> Self {
        Self {
            address,
            data,
            writable,
        }
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.address && addr < self.address + self.data.len() as u64
    }

    pub fn read(&self, addr: u64, size: usize) -> Option<u64> {
        if !self.contains(addr) || !self.contains(addr + size as u64 - 1) {
            return None;
        }

        let offset = (addr - self.address) as usize;
        let bytes = &self.data[offset..offset + size];

        let mut val = 0u64;
        for (i, &b) in bytes.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        Some(val)
    }
}
