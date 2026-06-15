//! Guest-architecture abstraction.
//!
//! Microlift lifts a *guest* instruction set into Cranelift IR. The lift engine
//! itself (block discovery, resumable dispatch, import fast-calls, context
//! sync, exit handling) is guest-agnostic and lives in [`crate::core`]. Each
//! supported guest architecture plugs into that engine by providing:
//!
//! - a decoder that turns bytes into a list of instructions,
//! - static control-flow facts per instruction (for block-leader discovery),
//! - a register file (SSA variables mapped to guest registers/flags), and
//! - instruction semantics (per-mnemonic IR emission).
//!
//! The central trait tying these together is `GuestArch`. It is defined
//! alongside the generic `Lifter` in [`crate::core`] because its methods take
//! `&mut Lifter<'_, Self>`. This module hosts the arch-neutral value types the
//! trait and engine exchange.

pub mod x86_64;

/// What kind of control transfer (if any) an instruction performs. Used during
/// phase-1 scanning to discover basic-block leaders without the engine needing
/// to know anything about the guest ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Static control-flow description of a single decoded instruction, consumed by
/// the engine's leader-discovery pass.
#[derive(Debug, Clone)]
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
