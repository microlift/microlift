//! Guest-architecture abstraction.
//!
//! The lift engine itself is guest-agnostic and lives in [`crate::core`]. Each
//! guest plugs into it by providing a decoder, static control-flow facts per
//! instruction, an SSA register file, and per-mnemonic semantics.
//!
//! The trait tying those together is `GuestArch`, defined alongside the generic
//! `Lifter` in [`crate::core`] because its methods take `&mut Lifter<'_, Self>`.
//! This module hosts the arch-neutral value types they exchange.

pub mod aarch64;
pub mod x86_64;

/// What kind of control transfer (if any) an instruction performs. Used during
/// phase-1 scanning to discover basic-block leaders without the engine needing
/// to know anything about the guest ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ControlFlowKind {
    /// Falls through to the next instruction; no transfer.
    Fallthrough,
    /// Unconditional branch to a statically-known target (`static_targets[0]`).
    DirectBranch,
    /// Conditional branch: taken edge in `static_targets[0]`, plus fallthrough.
    CondBranch,
    /// Branch through a register/memory operand; target known only at runtime.
    IndirectBranch,
    /// A call. `static_target` is `Some` for direct calls, `None` for indirect.
    Call { static_target: Option<u64> },
    /// A return.
    Ret,
    /// The bytes did not decode as a valid instruction.
    Invalid,
}

impl std::fmt::Display for ControlFlowKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ControlFlowKind::Fallthrough => "fallthrough",
            ControlFlowKind::DirectBranch => "branch",
            ControlFlowKind::CondBranch => "cond-branch",
            ControlFlowKind::IndirectBranch => "indirect-branch",
            ControlFlowKind::Call { .. } => "call",
            ControlFlowKind::Ret => "ret",
            ControlFlowKind::Invalid => "invalid",
        })
    }
}

/// Register and memory read/write effects of a single instruction.
///
/// Register names are the guest's own lower-case mnemonics (e.g. `"rax"`,
/// `"x0"`). The lists are de-duplicated. Intended for static analysis (def-use,
/// liveness, taint) rather than precise emulation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InstructionEffects {
    /// Registers read by the instruction.
    pub reads: Vec<String>,
    /// Registers written (fully or conditionally) by the instruction.
    pub writes: Vec<String>,
    /// The instruction reads from memory.
    pub reads_memory: bool,
    /// The instruction writes to memory.
    pub writes_memory: bool,
}

/// Static control-flow description of a single decoded instruction, consumed by
/// the engine's leader-discovery pass.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ControlFlow {
    pub kind: ControlFlowKind,
    /// Statically-known branch targets (e.g. the taken edge of a `Jcc`/`B.cond`,
    /// or the destination of a direct `JMP`/`B`). Empty for indirect transfers.
    pub static_targets: Vec<u64>,
}

impl ControlFlow {
    pub fn fallthrough() -> Self {
        Self {
            kind: ControlFlowKind::Fallthrough,
            static_targets: Vec::new(),
        }
    }

    pub fn direct_branch(target: u64) -> Self {
        Self {
            kind: ControlFlowKind::DirectBranch,
            static_targets: vec![target],
        }
    }

    pub fn cond_branch(target: u64) -> Self {
        Self {
            kind: ControlFlowKind::CondBranch,
            static_targets: vec![target],
        }
    }

    pub fn indirect_branch() -> Self {
        Self {
            kind: ControlFlowKind::IndirectBranch,
            static_targets: Vec::new(),
        }
    }

    pub fn call(static_target: Option<u64>) -> Self {
        Self {
            kind: ControlFlowKind::Call { static_target },
            static_targets: Vec::new(),
        }
    }

    pub fn ret() -> Self {
        Self {
            kind: ControlFlowKind::Ret,
            static_targets: Vec::new(),
        }
    }

    pub fn invalid() -> Self {
        Self {
            kind: ControlFlowKind::Invalid,
            static_targets: Vec::new(),
        }
    }
}
