//! Shared exit/control state exchanged between lifted guest code and the host.
//!
//! Every guest architecture embeds a [`ControlHeader`] at a known offset in its
//! `RegisterContext` so the host's exit/dispatch loop can read why a lifted
//! region exited without knowing which guest ISA produced it. The per-arch
//! register banks (and the architectural PC) live outside this header in the
//! arch-specific context struct.

/// Architecture-neutral control fields shared by every guest's register
/// context. Embedded `#[repr(C)]` at a fixed offset so the host reads exit
/// state uniformly across guests.
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlHeader {
    /// Why the lifted region exited. Holds an [`ExitReason`] discriminant.
    pub exit_reason: u32,
    /// Secondary exit detail. For `UnknownJump` this is a [`TransferKind`]; for
    /// `UnimplementedInstruction` it is the raw guest opcode, etc.
    pub exit_code: u32,
    /// On import/indirect-call exits, the resolved callee virtual address.
    pub imported_function_va: u64,
    /// Runtime image base, used to reconcile PC against the lift-time base under
    /// ASLR (`pc = image_base + (current_ip - lift_base)`).
    pub image_base: u64,
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
    /// The code that was about to be executed did not decode as a
    /// valid guest instruction.
    InvalidInstruction = 5,
    /// A `CALL` to a registered import address was intercepted (and the import
    /// either had no fast-call metadata or has a variadic/unsupported ABI).
    ///
    /// `ControlHeader::imported_function_va` holds the callee address. The PC
    /// field holds the address of the instruction after the `CALL` — the resume
    /// point once the host has handled the call.
    ///
    /// **The stack pointer is unchanged.** The host is expected to call the
    /// target as a native C function and then resume the guest at the PC.
    ImportedFunctionCall = 6,

    /// An indirect `CALL` (via register or memory) to an address that was not
    /// registered as an import.
    ///
    /// `ControlHeader::imported_function_va` holds the runtime-resolved callee
    /// address. The PC field holds the resume point (the instruction after the
    /// `CALL`).
    ///
    /// **The stack pointer is unchanged.** The host must decide how to dispatch
    /// the callee:
    /// - If the target is a **native function**, call it via C and resume at the
    ///   PC — no stack adjustment is needed.
    /// - If the target is **guest code**, push the return address before
    ///   re-entering the guest so the callee's eventual `RET` finds it.
    IndirectCall = 7,

    /// A `UD2`/equivalent invalid-opcode trap was executed intentionally (e.g.
    /// assertions, panic handlers, sanitizer traps). The PC points *at* the
    /// trapping instruction itself.
    Ud2 = 8,

    /// An `INT3`/breakpoint trap was executed. The PC points *at* the breakpoint
    /// instruction itself. The host debugger or runtime may resume execution by
    /// advancing the PC past the instruction before re-entering the guest.
    Breakpoint = 9,
}

impl TryFrom<u32> for ExitReason {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == Self::None as u32 {
            Ok(Self::None)
        } else if value == Self::Return as u32 {
            Ok(Self::Return)
        } else if value == Self::Syscall as u32 {
            Ok(Self::Syscall)
        } else if value == Self::UnknownJump as u32 {
            Ok(Self::UnknownJump)
        } else if value == Self::UnimplementedInstruction as u32 {
            Ok(Self::UnimplementedInstruction)
        } else if value == Self::InvalidInstruction as u32 {
            Ok(Self::InvalidInstruction)
        } else if value == Self::ImportedFunctionCall as u32 {
            Ok(Self::ImportedFunctionCall)
        } else if value == Self::IndirectCall as u32 {
            Ok(Self::IndirectCall)
        } else if value == Self::Ud2 as u32 {
            Ok(Self::Ud2)
        } else if value == Self::Breakpoint as u32 {
            Ok(Self::Breakpoint)
        } else {
            Err(())
        }
    }
}

/// Describes what kind of guest control transfer caused an `UnknownJump` VM exit.
///
/// Stored in `ControlHeader::exit_code` whenever `exit_reason == UnknownJump`.
/// Other exit reasons give `exit_code` their own meaning (e.g.
/// `UnimplementedInstruction` stores the raw guest opcode there).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// Origin not tracked - should not occur in practice; treat conservatively.
    Unknown = 0,
    /// Unconditional `JMP`/`B` to a static target outside the region.
    DirectJump = 1,
    /// `JMP`/`BR` via register or memory operand; target was only known at runtime.
    IndirectJump = 2,
    /// A conditional branch whose taken edge fell outside the lifted region.
    ConditionalJump = 3,
    /// A `RET` that popped a return address outside the lifted region (and not the
    /// exit trampoline).
    Return = 4,
    /// An unconditional `CALL`/`BL` to a static out-of-region address that was not
    /// registered as an import. A return address was pushed; the runtime must
    /// call the target and then resume at that address.
    DirectCall = 5,
}

impl TryFrom<u32> for TransferKind {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == Self::Unknown as u32 {
            Ok(Self::Unknown)
        } else if value == Self::DirectJump as u32 {
            Ok(Self::DirectJump)
        } else if value == Self::IndirectJump as u32 {
            Ok(Self::IndirectJump)
        } else if value == Self::ConditionalJump as u32 {
            Ok(Self::ConditionalJump)
        } else if value == Self::Return as u32 {
            Ok(Self::Return)
        } else if value == Self::DirectCall as u32 {
            Ok(Self::DirectCall)
        } else {
            Err(())
        }
    }
}

/// A contiguous region of guest memory made available to the lifter for static
/// reads (e.g. constant pools, RIP-relative data).
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
