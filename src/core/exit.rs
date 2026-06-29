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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// field holds the address of the instruction after the `CALL` - the resume
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
    ///   PC - no stack adjustment is needed.
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

    /// A `CPUID` instruction was executed. This is a resumable VM exit: the PC
    /// points to the instruction *after* `CPUID` (the resume point). `rax` and
    /// `rcx` hold the leaf and sub-leaf inputs. The runtime should populate
    /// `rax`, `rbx`, `rcx`, and `rdx` with the desired CPUID response and
    /// re-enter the guest at the PC.
    Cpuid = 10,

    /// An `XGETBV` instruction was executed. This is a resumable VM exit: the
    /// PC points to the instruction *after* `XGETBV` (the resume point). `rcx`
    /// holds the XCR index. The runtime should populate `rax` (bits 31:0) and
    /// `rdx` (bits 63:32) with the value of XCR[rcx] and re-enter at the PC.
    Xgetbv = 11,

    /// A software interrupt was raised by `INT imm8`, `INT1`/`ICEBP`, or `INTO`
    /// (the latter only when `OF=1`). The interrupt vector is stored in
    /// `ControlHeader::exit_code` (e.g. `0x80` for `INT 0x80`, `1` for `INT1`,
    /// `4` for a taken `INTO`). The PC points to the instruction *after* the
    /// `INT`, matching the return address a real CPU would push: the host
    /// services the vector and re-enters the guest at the PC to resume.
    ///
    /// Note that `INT3` (the dedicated `0xCC` breakpoint opcode) reports
    /// [`ExitReason::Breakpoint`] instead, so debuggers can distinguish it from
    /// a general software interrupt.
    Interrupt = 12,

    /// An I/O port *input* string instruction (`INSB`/`INSW`/`INSD`) was
    /// executed. Port I/O cannot be performed from the lifted code, so this is
    /// a VM exit and the host must emulate the access in full. The PC points
    /// *at* the instruction; `exit_code` holds the element size in bytes (1, 2,
    /// or 4). The source port number is in `DX`, the destination is `ES:[RDI]`,
    /// and the host must honour `DF` (and `RCX` when the instruction carried a
    /// `REP` prefix) when advancing `RDI`/`RCX`. After servicing the access the
    /// host advances the PC past the instruction and re-enters the guest.
    PortInput = 13,

    /// An I/O port *output* string instruction (`OUTSB`/`OUTSW`/`OUTSD`) was
    /// executed. As with [`ExitReason::PortInput`], port I/O cannot be performed
    /// from the lifted code, so the host must emulate the access in full. The PC
    /// points *at* the instruction; `exit_code` holds the element size in bytes
    /// (1, 2, or 4). The destination port number is in `DX`, the source is
    /// `DS:[RSI]`, and the host must honour `DF` (and `RCX` when the instruction
    /// carried a `REP` prefix) when advancing `RSI`/`RCX`. After servicing the
    /// access the host advances the PC past the instruction and re-enters.
    PortOutput = 14,
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
        } else if value == Self::Cpuid as u32 {
            Ok(Self::Cpuid)
        } else if value == Self::Xgetbv as u32 {
            Ok(Self::Xgetbv)
        } else if value == Self::Interrupt as u32 {
            Ok(Self::Interrupt)
        } else if value == Self::PortInput as u32 {
            Ok(Self::PortInput)
        } else if value == Self::PortOutput as u32 {
            Ok(Self::PortOutput)
        } else {
            Err(())
        }
    }
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ExitReason::None => "none",
            ExitReason::Return => "return",
            ExitReason::Syscall => "syscall",
            ExitReason::UnknownJump => "unknown-jump",
            ExitReason::UnimplementedInstruction => "unimplemented-instruction",
            ExitReason::InvalidInstruction => "invalid-instruction",
            ExitReason::ImportedFunctionCall => "imported-function-call",
            ExitReason::IndirectCall => "indirect-call",
            ExitReason::Ud2 => "ud2",
            ExitReason::Breakpoint => "breakpoint",
            ExitReason::Cpuid => "cpuid",
            ExitReason::Xgetbv => "xgetbv",
            ExitReason::Interrupt => "interrupt",
            ExitReason::PortInput => "port-input",
            ExitReason::PortOutput => "port-output",
        })
    }
}

/// Describes what kind of guest control transfer caused an `UnknownJump` VM exit.
///
/// Stored in `ControlHeader::exit_code` whenever `exit_reason == UnknownJump`.
/// Other exit reasons give `exit_code` their own meaning (e.g.
/// `UnimplementedInstruction` stores the raw guest opcode there).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl std::fmt::Display for TransferKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TransferKind::Unknown => "unknown",
            TransferKind::DirectJump => "direct-jump",
            TransferKind::IndirectJump => "indirect-jump",
            TransferKind::ConditionalJump => "conditional-jump",
            TransferKind::Return => "return",
            TransferKind::DirectCall => "direct-call",
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_reason_display_and_roundtrip() {
        assert_eq!(ExitReason::ImportedFunctionCall.to_string(), "imported-function-call");
        // try_from is the inverse of the discriminant.
        for r in [ExitReason::None, ExitReason::Return, ExitReason::IndirectCall, ExitReason::Xgetbv, ExitReason::Interrupt, ExitReason::PortInput, ExitReason::PortOutput] {
            assert_eq!(ExitReason::try_from(r as u32), Ok(r));
            assert!(!r.to_string().is_empty());
        }
    }

    #[test]
    fn transfer_kind_display() {
        assert_eq!(TransferKind::DirectCall.to_string(), "direct-call");
        assert_eq!(TransferKind::try_from(TransferKind::IndirectJump as u32), Ok(TransferKind::IndirectJump));
    }
}
