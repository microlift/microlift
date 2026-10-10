//! Exit/control state shared between lifted code and the host.
//!
//! [`ControlHeader`] is embedded as `control` in every guest context; the engine reaches
//! it only through `GuestArch::*_offset()`.

/// Arch-neutral control fields embedded `#[repr(C)]` in every guest context.
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlHeader {
    /// An [`ExitReason`] discriminant.
    pub exit_reason: u32,
    /// Per reason: [`TransferKind`] for `UnknownJump`, the vector for `Interrupt`, element size for port I/O.
    pub exit_code: u32,
    /// Resolved callee VA on import/indirect-call exits.
    pub imported_function_va: u64,
    /// `pc = image_base + (current_ip - lift_base)`.
    pub image_base: u64,
}

impl ControlHeader {
    /// A `const` so guest contexts can build `Default` in const context.
    pub const ZERO: Self = Self {
        exit_reason: 0,
        exit_code: 0,
        imported_function_va: 0,
        image_base: 0,
    };
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Discriminants 2 and 4 are retired (`Syscall`, `UnimplementedInstruction`; now
/// [`crate::core::callback`]). Numbers are FFI-visible: never reuse.
pub enum ExitReason {
    None = 0,
    /// A terminating block was reached.
    Return = 1,
    /// Branch to an address outside the region; `exit_code` is a [`TransferKind`].
    UnknownJump = 3,
    /// The next instruction did not decode.
    InvalidInstruction = 5,
    /// A `CALL` to a slow import. `imported_function_va` is the callee, PC the instruction after the `CALL`.
    ///
    /// **The stack pointer is unchanged**; the host calls the target natively and resumes at PC.
    ImportedFunctionCall = 6,

    /// An indirect `CALL` to an unregistered address. `imported_function_va` is the callee, PC the resume point.
    ///
    /// **The stack pointer is unchanged.** For a native target call it and resume at PC; for
    /// guest code push the return address before re-entering.
    ///
    /// A linking region reports this only when nothing linked; a linked hit exits as
    /// `UnknownJump` / [`TransferKind::IndirectCall`] with the push done.
    IndirectCall = 7,

    /// `UD2`; PC points *at* the instruction.
    Ud2 = 8,

    /// `INT3`; PC points *at* the instruction (the host may advance past it to resume).
    Breakpoint = 9,

    /// `CPUID`; PC is *after* it, `rax`/`rcx` hold leaf/sub-leaf. The host fills `rax`/`rbx`/`rcx`/`rdx`.
    Cpuid = 10,

    /// `XGETBV`; PC is *after* it, `rcx` is the XCR index. The host fills `rax`/`rdx` (low/high).
    Xgetbv = 11,

    /// `INT imm8`, `INT1`, or `INTO` (when `OF=1`); vector in `exit_code`, PC *after* the instruction.
    ///
    /// `INT3` reports [`ExitReason::Breakpoint`].
    Interrupt = 12,

    /// `INSB`/`INSW`/`INSD`, emulated in full by the host.
    ///
    /// PC is *at* the instruction; `exit_code` is the element size, port in `DX`, destination
    /// `ES:[RDI]`. The host honours `DF` and `REP`/`RCX`, then advances PC.
    PortInput = 13,

    /// `OUTSB`/`OUTSW`/`OUTSD`; as [`ExitReason::PortInput`] with source `DS:[RSI]`.
    PortOutput = 14,

    /// `RDTSC`; PC is *after* it. The host fills `rax`/`rdx` (low/high).
    Rdtsc = 15,

    /// `DIV`/`IDIV` #DE fault: PC is *at* the instruction, no registers modified.
    DivideError = 16,
}

impl TryFrom<u32> for ExitReason {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == Self::None as u32 {
            Ok(Self::None)
        } else if value == Self::Return as u32 {
            Ok(Self::Return)
        } else if value == Self::UnknownJump as u32 {
            Ok(Self::UnknownJump)
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
        } else if value == Self::Rdtsc as u32 {
            Ok(Self::Rdtsc)
        } else if value == Self::DivideError as u32 {
            Ok(Self::DivideError)
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
            ExitReason::UnknownJump => "unknown-jump",
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
            ExitReason::Rdtsc => "rdtsc",
            ExitReason::DivideError => "divide-error",
        })
    }
}

/// Why an `UnknownJump` exited; stored in `ControlHeader::exit_code`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransferKind {
    /// Origin not tracked; treat conservatively.
    Unknown = 0,
    DirectJump = 1,
    IndirectJump = 2,
    ConditionalJump = 3,
    /// A `RET` to outside the region (and not the exit trampoline).
    Return = 4,
    /// Static `CALL`/`BL` out of region, not an import. The return address is pushed.
    DirectCall = 5,
    /// An instruction's bytes run past the region end ([`LifterConfig::region_tail_exit`]).
    ///
    /// Nothing has happened: PC is that instruction, state and stack untouched. The host lifts a
    /// new region starting at PC.
    ///
    /// [`LifterConfig::region_tail_exit`]: crate::core::LifterConfig::region_tail_exit
    RegionTail = 6,
    /// Entered at a PC that is not a resume point ([`LifterConfig::strict_resume`]).
    ///
    /// No state has changed; the host translates a region at that PC instead.
    ///
    /// [`LifterConfig::strict_resume`]: crate::core::LifterConfig::strict_resume
    NoResumePoint = 7,
    /// A linked indirect `CALL` that hit: the callee is known guest code.
    ///
    /// State as [`DirectCall`](Self::DirectCall): return address pushed, `imported_function_va`
    /// and PC are the callee, `link_hit` names the row. A miss exits as [`ExitReason::IndirectCall`].
    IndirectCall = 8,
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
        } else if value == Self::RegionTail as u32 {
            Ok(Self::RegionTail)
        } else if value == Self::NoResumePoint as u32 {
            Ok(Self::NoResumePoint)
        } else if value == Self::IndirectCall as u32 {
            Ok(Self::IndirectCall)
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
            TransferKind::RegionTail => "region-tail",
            TransferKind::NoResumePoint => "no-resume-point",
            TransferKind::IndirectCall => "indirect-call",
        })
    }
}

/// Guest memory the lifter may read statically (constant pools, RIP-relative data).
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
        assert_eq!(
            ExitReason::ImportedFunctionCall.to_string(),
            "imported-function-call"
        );
        for r in [
            ExitReason::None,
            ExitReason::Return,
            ExitReason::IndirectCall,
            ExitReason::Xgetbv,
            ExitReason::Interrupt,
            ExitReason::PortInput,
            ExitReason::PortOutput,
            ExitReason::Rdtsc,
            ExitReason::DivideError,
        ] {
            assert_eq!(ExitReason::try_from(r as u32), Ok(r));
            assert!(!r.to_string().is_empty());
        }
    }

    #[test]
    fn transfer_kind_display() {
        assert_eq!(TransferKind::DirectCall.to_string(), "direct-call");
        assert_eq!(TransferKind::IndirectCall.to_string(), "indirect-call");
        assert_eq!(
            TransferKind::try_from(TransferKind::IndirectCall as u32),
            Ok(TransferKind::IndirectCall)
        );
        assert_eq!(
            TransferKind::try_from(TransferKind::IndirectJump as u32),
            Ok(TransferKind::IndirectJump)
        );
    }

    use crate::arch::aarch64::Aarch64;
    use crate::arch::aarch64::context::Aarch64RegisterContext;
    use crate::arch::x86_64::X86_64;
    use crate::arch::x86_64::context::RegisterContext;
    use crate::core::GuestArch;
    use std::mem::offset_of;

    #[test]
    fn both_guests_locate_the_same_header() {
        macro_rules! check {
            ($ctx:ty, $arch:ty) => {{
                let base = offset_of!($ctx, control) as i32;
                assert_eq!(
                    <$arch as GuestArch>::exit_reason_offset(),
                    base + offset_of!(ControlHeader, exit_reason) as i32,
                );
                assert_eq!(
                    <$arch as GuestArch>::exit_code_offset(),
                    base + offset_of!(ControlHeader, exit_code) as i32,
                );
                assert_eq!(
                    <$arch as GuestArch>::imported_function_va_offset(),
                    base + offset_of!(ControlHeader, imported_function_va) as i32,
                );
                assert_eq!(
                    <$arch as GuestArch>::image_base_offset(),
                    base + offset_of!(ControlHeader, image_base) as i32,
                );
            }};
        }
        check!(RegisterContext, X86_64);
        check!(Aarch64RegisterContext, Aarch64);
    }

    /// Read by non-Rust hosts: layout is ABI.
    #[test]
    fn the_header_layout_is_the_published_one() {
        assert_eq!(offset_of!(ControlHeader, exit_reason), 0);
        assert_eq!(offset_of!(ControlHeader, exit_code), 4);
        assert_eq!(offset_of!(ControlHeader, imported_function_va), 8);
        assert_eq!(offset_of!(ControlHeader, image_base), 16);
        assert_eq!(size_of::<ControlHeader>(), 24);
        assert_eq!(align_of::<ControlHeader>(), 8);
    }
}
