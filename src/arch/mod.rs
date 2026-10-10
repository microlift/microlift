//! Guest-architecture abstraction: the arch-neutral value types exchanged with the engine.
//!
//! `GuestArch` itself lives in [`crate::core`].

pub mod aarch64;
pub mod x86_64;

/// Control-transfer kind of an instruction, used by phase-1 leader discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ControlFlowKind {
    Fallthrough,
    /// Target in `static_targets[0]`.
    DirectBranch,
    /// Taken edge in `static_targets[0]`, plus fallthrough.
    CondBranch,
    /// Target known only at runtime.
    IndirectBranch,
    /// `static_target` is `None` for indirect calls.
    Call { static_target: Option<u64> },
    Ret,
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

/// Register and memory effects of one instruction, for static analysis (def-use, liveness, taint).
///
/// Names are the guest's lower-case mnemonics (`"rax"`, `"x0"`), de-duplicated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InstructionEffects {
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub reads_memory: bool,
    pub writes_memory: bool,
}

/// Static control-flow facts of one decoded instruction, for leader discovery.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ControlFlow {
    pub kind: ControlFlowKind,
    /// Statically-known targets (taken edge, direct `JMP`/`B`); empty when indirect.
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
