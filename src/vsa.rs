//! Value Set Analysis (VSA) - a lightweight abstract interpretation pass for
//! resolving indirect control flow at lift time.
//!
//! # Overview
//!
//! VSA determines the set of possible values a register can hold at each program
//! point. This enables converting `JMP RAX` from a full runtime dispatch switch
//! into a targeted dispatch (or even a direct jump) when the value set is small.
//!
//! # Lattice
//!
//! ```text
//!              Top   (completely unknown)
//!             / | \
//!   Interval  Multiple  ...
//!           \     /
//!          Constant
//!              |
//!            Empty   (unreachable)
//! ```

use crate::cfg::ResolvedTargets;
use crate::context::MemorySegment;
use iced_x86::{
    Decoder, DecoderOptions, Instruction, InstructionInfoFactory, MemorySize, Mnemonic, OpAccess,
    OpKind, Register,
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Default caller-saved (volatile) general-purpose registers assumed clobbered
/// across a `CALL`: the union of the System V and Windows x64 volatile sets, so
/// it is sound under either standard C ABI. Registers not listed here (RBX,
/// RBP, RSP, R12-R15) are assumed preserved by the callee - which holds only if
/// the analysed code follows one of those ABIs. See [`ValueSetAnalyzer`] for how
/// to override this (e.g. [`ValueSetAnalyzer::all_gpr_clobbers`] for the
/// ABI-agnostic conservative choice).
const DEFAULT_CALL_CLOBBERS: &[Register] = &[
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RSI,
    Register::RDI,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
];

/// Registers assumed clobbered by a `SYSCALL`. RCX (<- RIP) and R11 (<- RFLAGS)
/// are written by the instruction itself on all x86-64; RAX is the syscall
/// return register under every mainstream OS (Linux/*BSD/Windows). Every other
/// GPR is preserved across a syscall under those conventions.
const DEFAULT_SYSCALL_CLOBBERS: &[Register] =
    &[Register::RAX, Register::RCX, Register::R11];

/// All 16 general-purpose registers - the ABI-agnostic conservative clobber set
/// for a `CALL` whose callee convention is unknown or untrusted.
const ALL_GPRS: &[Register] = &[
    Register::RAX,
    Register::RBX,
    Register::RCX,
    Register::RDX,
    Register::RSI,
    Register::RDI,
    Register::RBP,
    Register::RSP,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

// Value Set lattice

/// A conservative abstract value for a register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSet {
    /// Unreachable / not yet set.
    Empty,
    /// Exactly one known value.
    Constant(i64),
    /// A small finite set of known values.
    Multiple(BTreeSet<i64>),
    /// Any value in the closed range [min, max].
    Interval { min: i64, max: i64 },
    /// Completely unknown.
    Top,
}

impl ValueSet {
    /// Maximum cardinality before we widen `Multiple` to `Top`.
    const MAX_SET_SIZE: usize = 64;

    /// Lattice join (least upper bound).
    pub fn join(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, x) | (x, ValueSet::Empty) => x.clone(),
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) if a == b => ValueSet::Constant(*a),
            (ValueSet::Constant(a), ValueSet::Constant(b)) => {
                let mut s = BTreeSet::new();
                s.insert(*a);
                s.insert(*b);
                ValueSet::Multiple(s)
            }
            (ValueSet::Multiple(s), ValueSet::Constant(c))
            | (ValueSet::Constant(c), ValueSet::Multiple(s)) => {
                let mut new_s = s.clone();
                new_s.insert(*c);
                if new_s.len() > Self::MAX_SET_SIZE {
                    ValueSet::Top
                } else {
                    ValueSet::Multiple(new_s)
                }
            }
            (ValueSet::Multiple(a), ValueSet::Multiple(b)) => {
                let merged: BTreeSet<i64> = a.union(b).cloned().collect();
                if merged.len() > Self::MAX_SET_SIZE {
                    ValueSet::Top
                } else {
                    ValueSet::Multiple(merged)
                }
            }
            (
                ValueSet::Interval {
                    min: a_min,
                    max: a_max,
                },
                ValueSet::Interval {
                    min: b_min,
                    max: b_max,
                },
            ) => ValueSet::Interval {
                min: (*a_min).min(*b_min),
                max: (*a_max).max(*b_max),
            },
            // Widening: Mixed Interval + other -> Top (conservative)
            _ => ValueSet::Top,
        }
    }

    /// Abstract addition.
    pub fn abstract_add(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => {
                ValueSet::Constant(a.wrapping_add(*b))
            }
            (
                ValueSet::Interval {
                    min: a_min,
                    max: a_max,
                },
                ValueSet::Interval {
                    min: b_min,
                    max: b_max,
                },
            ) => ValueSet::Interval {
                min: a_min.wrapping_add(*b_min),
                max: a_max.wrapping_add(*b_max),
            },
            (ValueSet::Constant(c), ValueSet::Interval { min, max })
            | (ValueSet::Interval { min, max }, ValueSet::Constant(c)) => ValueSet::Interval {
                min: min.wrapping_add(*c),
                max: max.wrapping_add(*c),
            },
            (ValueSet::Multiple(s), ValueSet::Constant(c))
            | (ValueSet::Constant(c), ValueSet::Multiple(s)) => {
                let new_s: BTreeSet<i64> = s.iter().map(|v| v.wrapping_add(*c)).collect();
                ValueSet::Multiple(new_s)
            }
            _ => ValueSet::Top,
        }
    }

    /// Abstract subtraction.
    pub fn abstract_sub(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => {
                ValueSet::Constant(a.wrapping_sub(*b))
            }
            (ValueSet::Constant(a), ValueSet::Interval { min, max }) => ValueSet::Interval {
                min: a.wrapping_sub(*max),
                max: a.wrapping_sub(*min),
            },
            _ => ValueSet::Top,
        }
    }

    /// Abstract AND (useful for masking patterns).
    pub fn abstract_and(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            // x & 0 = 0 for all x (including Top).
            (ValueSet::Constant(0), _) | (_, ValueSet::Constant(0)) => ValueSet::Constant(0),
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => ValueSet::Constant(a & b),
            _ => ValueSet::Top,
        }
    }

    /// Abstract XOR (e.g., `xor rax, rax` -> 0).
    pub fn abstract_xor(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => ValueSet::Constant(a ^ b),
            _ => ValueSet::Top,
        }
    }

    /// Keep only the low `bits` bits of every concrete value, zero-extending the
    /// result. Models a narrow (sub-register) read/zero-extending write: e.g. the
    /// low 32 bits of a value for `mov ecx, eax` or `mov eax, imm32`.
    ///
    /// Only `Constant`/`Multiple` can be truncated exactly; an `Interval`'s low
    /// bits are not an interval in general, so we conservatively fall to `Top`.
    pub fn low_bits(&self, bits: u32) -> ValueSet {
        if bits >= 64 {
            return self.clone();
        }
        let mask = (1u64 << bits) - 1;
        match self {
            ValueSet::Empty => ValueSet::Empty,
            ValueSet::Constant(c) => ValueSet::Constant((*c as u64 & mask) as i64),
            ValueSet::Multiple(s) => {
                let m: BTreeSet<i64> = s.iter().map(|v| (*v as u64 & mask) as i64).collect();
                // Masking can collapse the set to a single value.
                if m.len() == 1 {
                    ValueSet::Constant(*m.iter().next().unwrap())
                } else {
                    ValueSet::Multiple(m)
                }
            }
            _ => ValueSet::Top,
        }
    }

    /// Widening operator used at control-flow merges to guarantee the worklist
    /// reaches a fixpoint in finite time. It behaves like [`join`](Self::join)
    /// except that an `Interval` whose bounds would grow jumps straight to `Top`,
    /// bounding the otherwise-unbounded ascending chain of intervals across a
    /// loop back-edge. `self` is the existing (accumulated) value; `other` is the
    /// newly arriving one.
    pub fn widen(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (
                ValueSet::Interval {
                    min: a_min,
                    max: a_max,
                },
                ValueSet::Interval {
                    min: b_min,
                    max: b_max,
                },
            ) => {
                if b_min < a_min || b_max > a_max {
                    ValueSet::Top
                } else {
                    ValueSet::Interval {
                        min: *a_min,
                        max: *a_max,
                    }
                }
            }
            _ => self.join(other),
        }
    }

    /// True if this value set has a bounded set of concrete targets.
    pub fn is_resolvable(&self) -> bool {
        matches!(self, ValueSet::Constant(_) | ValueSet::Multiple(_))
    }

    /// Collect concrete target addresses from this value set.
    pub fn targets(&self) -> Vec<u64> {
        match self {
            ValueSet::Constant(c) => vec![*c as u64],
            ValueSet::Multiple(s) => s.iter().map(|&v| v as u64).collect(),
            _ => vec![],
        }
    }
}

// Register value state at a program point

#[derive(Clone, Debug, Default)]
pub struct RegisterValueSet {
    pub values: HashMap<Register, ValueSet>,
}

impl RegisterValueSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, reg: Register) -> ValueSet {
        self.values.get(&reg).cloned().unwrap_or(ValueSet::Top)
    }

    pub fn set(&mut self, reg: Register, vs: ValueSet) {
        self.values.insert(reg, vs);
    }

    /// Join two register states (per-register lattice join).
    pub fn join(&self, other: &RegisterValueSet) -> RegisterValueSet {
        let mut result = self.clone();
        for (reg, vs) in &other.values {
            let joined = result.get(*reg).join(vs);
            result.set(*reg, joined);
        }
        result
    }

    /// Per-register [widening](ValueSet::widen) of two register states; used at
    /// merges to keep the fixpoint iteration terminating.
    pub fn widen(&self, other: &RegisterValueSet) -> RegisterValueSet {
        let mut result = self.clone();
        for (reg, vs) in &other.values {
            let widened = result.get(*reg).widen(vs);
            result.set(*reg, widened);
        }
        result
    }
}

/// Read a (possibly sub-width) register, canonicalizing to its 64-bit container
/// and masking to the operand's width. Sub-register state is tracked on the full
/// register, so `mov ecx, eax` must read the *low 32 bits* of RAX - not the whole
/// 64-bit value.
fn read_reg(state: &RegisterValueSet, reg: Register) -> ValueSet {
    // High-8 registers (AH/BH/CH/DH) alias bits 8..15 of their container, not
    // the low byte, so `low_bits(8)` would read the wrong bits (AL for AH).
    // Sub-register-of-high-byte tracking isn't modelled, so read them
    // conservatively as `Top` rather than returning a wrong concrete value.
    if matches!(
        reg,
        Register::AH | Register::BH | Register::CH | Register::DH
    ) {
        return ValueSet::Top;
    }
    let vs = state.get(reg.full_register());
    match reg.size() {
        0 | 8 => vs,
        n => vs.low_bits(n as u32 * 8),
    }
}

/// Write a (possibly sub-width) register, updating its 64-bit container so a
/// later full-width read observes the write. x86-64 semantics:
/// * a 64-bit write replaces the register outright;
/// * a 32-bit write zero-extends into the full register (upper 32 bits cleared);
/// * an 8/16-bit write preserves the untracked upper bits, so the combined
///   64-bit value is unknown (`Top`).
fn write_reg(out: &mut RegisterValueSet, reg: Register, vs: ValueSet) {
    let full = reg.full_register();
    match reg.size() {
        8 => out.set(full, vs),
        4 => out.set(full, vs.low_bits(32)),
        _ => out.set(full, ValueSet::Top),
    }
}

// Analyzer

/// Lightweight dataflow-based Value Set Analyzer.
///
/// Usage:
/// ```ignore
/// let mut vsa = ValueSetAnalyzer::new();
/// vsa.analyze(&instructions);
/// let targets = vsa.indirect_jump_targets(&jmp_instruction);
/// ```
pub struct ValueSetAnalyzer {
    /// Map from instruction IP -> inferred register value sets *before* that instruction.
    pub value_sets: HashMap<u64, RegisterValueSet>,
    pub segments: Vec<MemorySegment>,
    /// Memory addresses that should always be treated as `Top` when read,
    /// regardless of what the segment data contains.
    ///
    /// Use this to mark GOT slots / PLT relocations that hold 0 before the
    /// dynamic linker runs. Without this, VSA would infer that a
    /// `JMP [GOT+off]` or `CALL [GOT+off]` targets address 0.
    pub relocation_addresses: HashSet<u64>,
    /// When `true` (the default), VSA makes no ABI assumptions whatsoever
    /// about which registers survive a `CALL` or `SYSCALL`: every GPR is
    /// clobbered across those boundaries, so no value set is ever carried across
    /// a call, syscall, or (VM-exit-backed) indirect/imported call. This is the
    /// only choice that preserves runtime semantic equivalence regardless of the
    /// callee's calling convention or what the host does while servicing a VM
    /// exit - a resolved indirect target can never depend on a value some callee
    /// or the host might have changed.
    ///
    /// Set to `false` (see [`with_standard_abi`](Self::with_standard_abi)) to opt
    /// into ABI-aware precision, in which case `call_clobbers` / `syscall_clobbers`
    /// take effect. Only do this for code you know honours the configured ABI.
    pub conservative: bool,
    /// Registers assumed clobbered across a `CALL` when `conservative == false`
    /// (the callee's caller-saved / volatile set). Registers **not** in this set
    /// are assumed callee-preserved, which is sound *only if the analysed code
    /// follows this ABI*. Ignored while `conservative` is `true`. Defaults to
    /// [`DEFAULT_CALL_CLOBBERS`] (System V union Windows x64).
    pub call_clobbers: Vec<Register>,
    /// Registers assumed clobbered by a `SYSCALL` when `conservative == false`.
    /// Ignored while `conservative` is `true`. Defaults to
    /// [`DEFAULT_SYSCALL_CLOBBERS`] (RAX/RCX/R11).
    pub syscall_clobbers: Vec<Register>,
    worklist: VecDeque<usize>,
}

impl ValueSetAnalyzer {
    pub fn new() -> Self {
        Self::with_segments(Vec::new())
    }

    pub fn with_segments(segments: Vec<MemorySegment>) -> Self {
        Self {
            value_sets: HashMap::new(),
            segments,
            relocation_addresses: HashSet::new(),
            // Safe by default: no ABI preservation assumptions across calls or
            // syscalls. Opt into precision explicitly via `with_standard_abi`.
            conservative: true,
            call_clobbers: DEFAULT_CALL_CLOBBERS.to_vec(),
            syscall_clobbers: DEFAULT_SYSCALL_CLOBBERS.to_vec(),
            worklist: VecDeque::new(),
        }
    }

    /// Mark a set of memory addresses as relocation slots.
    ///
    /// Any memory read from these addresses will be conservatively treated as
    /// `Top` (completely unknown), preventing VSA from propagating stale
    /// pre-relocation values (typically 0) through them.
    pub fn with_relocation_addresses(mut self, addrs: HashSet<u64>) -> Self {
        self.relocation_addresses = addrs;
        self
    }

    /// Set the [`conservative`](Self::conservative) flag. `true` (the default)
    /// clobbers every GPR across calls/syscalls; `false` enables ABI-aware
    /// precision using `call_clobbers` / `syscall_clobbers`.
    pub fn with_conservative(mut self, conservative: bool) -> Self {
        self.conservative = conservative;
        self
    }

    /// Opt into ABI-aware precision (`conservative = false`), keeping the current
    /// `call_clobbers` / `syscall_clobbers` (the standard x86-64 C ABI by
    /// default). Only sound for code that honours the configured ABI.
    pub fn with_standard_abi(mut self) -> Self {
        self.conservative = false;
        self
    }

    /// Override the registers assumed clobbered across a `CALL`, and switch to
    /// ABI-aware mode (`conservative = false`). Registers omitted here are
    /// assumed callee-preserved.
    pub fn with_call_clobbers(mut self, regs: Vec<Register>) -> Self {
        self.conservative = false;
        self.call_clobbers = regs;
        self
    }

    /// Override the registers assumed clobbered by a `SYSCALL`, and switch to
    /// ABI-aware mode (`conservative = false`).
    pub fn with_syscall_clobbers(mut self, regs: Vec<Register>) -> Self {
        self.conservative = false;
        self.syscall_clobbers = regs;
        self
    }

    /// The default `CALL` clobber set (System V union Windows x64 volatile GPRs).
    pub fn default_call_clobbers() -> Vec<Register> {
        DEFAULT_CALL_CLOBBERS.to_vec()
    }

    /// Every GPR: the ABI-agnostic conservative `CALL` clobber set. Using this
    /// makes no assumption about which registers a callee preserves, so VSA
    /// never resolves an indirect transfer using a value carried across a call.
    pub fn all_gpr_clobbers() -> Vec<Register> {
        ALL_GPRS.to_vec()
    }

    /// Run the analysis over a decoded instruction stream.
    ///
    /// The instructions should already be topologically sorted (i.e. the normal
    /// decode order). This is an iterative worklist analysis that runs to a
    /// fixpoint: a block is re-processed whenever a predecessor changes its entry
    /// state, so values are correctly widened across control-flow merges. The
    /// widening operator ([`RegisterValueSet::widen`]) bounds the ascending chain
    /// (intervals that grow jump to `Top`), guaranteeing termination.
    pub fn analyze(&mut self, instructions: &[Instruction]) {
        if instructions.is_empty() {
            return;
        }

        // Seed the entry state (all registers are Top - unknown on entry)
        let entry_ip = instructions[0].ip();
        self.value_sets
            .entry(entry_ip)
            .or_insert_with(RegisterValueSet::new);

        self.worklist.push_back(0);

        while let Some(idx) = self.worklist.pop_front() {
            if idx >= instructions.len() {
                continue;
            }
            let instruction = &instructions[idx];
            let ip = instruction.ip();

            // Get pre-instruction state
            let pre_state = self
                .value_sets
                .get(&ip)
                .cloned()
                .unwrap_or_else(RegisterValueSet::new);

            let post_state = self.transfer(&pre_state, instruction);

            // Propagate to successors
            let next_ip = instruction.next_ip();

            match instruction.mnemonic() {
                Mnemonic::Jmp => {
                    if instruction.op0_kind() != OpKind::Register
                        && instruction.op0_kind() != OpKind::Memory
                    {
                        // Direct jump - propagate to target
                        let target = instruction.near_branch_target();
                        self.propagate(target, &post_state, instructions);
                    }
                    // Indirect: targets resolved by caller via `indirect_jump_targets()`
                }
                Mnemonic::Je
                | Mnemonic::Jne
                | Mnemonic::Jb
                | Mnemonic::Jbe
                | Mnemonic::Ja
                | Mnemonic::Jae
                | Mnemonic::Jg
                | Mnemonic::Jge
                | Mnemonic::Jl
                | Mnemonic::Jle
                | Mnemonic::Js
                | Mnemonic::Jns
                | Mnemonic::Jo
                | Mnemonic::Jno
                | Mnemonic::Jp
                | Mnemonic::Jnp
                | Mnemonic::Loop
                | Mnemonic::Loope
                | Mnemonic::Loopne
                | Mnemonic::Jrcxz => {
                    // Conditional transfers: both the taken target and the
                    // fallthrough are reachable. (Must match the cond-branch set
                    // in `decode::control_flow`, or a back-edge is missed and the
                    // fixpoint under-approximates the value sets - unsound.)
                    let target = instruction.near_branch_target();
                    self.propagate(target, &post_state, instructions);
                    self.propagate(next_ip, &post_state, instructions);
                }
                Mnemonic::Ret => {
                    // Terminator; no fallthrough.
                }
                _ => {
                    // Fallthrough to next instruction
                    if idx + 1 < instructions.len() {
                        let next_idx = idx + 1;
                        let next_inst_ip = instructions[next_idx].ip();
                        if self.merge(next_inst_ip, &post_state) {
                            self.worklist.push_back(next_idx);
                        }
                    }
                }
            }
        }
    }

    /// Abstract transfer function - apply one instruction to a pre-state to
    /// produce a post-state.
    pub fn transfer(&self, state: &RegisterValueSet, inst: &Instruction) -> RegisterValueSet {
        let mut out = state.clone();

        match inst.mnemonic() {
            Mnemonic::Mov => {
                if inst.op0_kind() == OpKind::Register {
                    let dst = inst.op0_register();
                    let src_vs = self.eval_operand(state, inst, 1);
                    write_reg(&mut out, dst, src_vs);
                }
            }
            Mnemonic::Xor => {
                if inst.op0_kind() == OpKind::Register
                    && inst.op1_kind() == OpKind::Register
                    && inst.op0_register() == inst.op1_register()
                {
                    // `xor rax, rax` -> 0
                    write_reg(&mut out, inst.op0_register(), ValueSet::Constant(0));
                } else if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    write_reg(&mut out, inst.op0_register(), a.abstract_xor(&b));
                }
            }
            Mnemonic::Add => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    write_reg(&mut out, inst.op0_register(), a.abstract_add(&b));
                }
            }
            Mnemonic::Sub => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    write_reg(&mut out, inst.op0_register(), a.abstract_sub(&b));
                }
            }
            Mnemonic::And => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    write_reg(&mut out, inst.op0_register(), a.abstract_and(&b));
                }
            }
            Mnemonic::Or => {
                if inst.op0_kind() == OpKind::Register {
                    // Conservative - don't try to track OR precisely
                    write_reg(&mut out, inst.op0_register(), ValueSet::Top);
                }
            }
            Mnemonic::Lea => {
                if inst.op0_kind() == OpKind::Register {
                    // LEA with a constant displacement can propagate
                    let base_reg = inst.memory_base();
                    let disp = inst.memory_displacement64() as i64;
                    let addr_vs = if inst.memory_index() == Register::None
                        && base_reg != Register::None
                    {
                        let base_vs = read_reg(state, base_reg);
                        let disp_vs = ValueSet::Constant(disp);
                        base_vs.abstract_add(&disp_vs)
                    } else {
                        ValueSet::Top
                    };
                    write_reg(&mut out, inst.op0_register(), addr_vs);
                }
            }
            Mnemonic::Call => {
                // A call (direct, indirect, or an imported call that becomes a
                // VM exit) clobbers registers. In conservative mode every GPR is
                // invalidated; no assumption about callee/host preservation.
                // Otherwise only the configured caller-saved set is invalidated;
                // registers left out are *assumed* callee-preserved, sound only
                // if the callee follows that ABI.
                let clobbers: &[Register] = if self.conservative {
                    ALL_GPRS
                } else {
                    &self.call_clobbers
                };
                for &reg in clobbers {
                    out.set(reg, ValueSet::Top);
                }
            }
            Mnemonic::Syscall => {
                // SYSCALL returns to the following instruction (it falls through
                // in the analysis). iced does not report RAX as written, so the
                // clobber is applied explicitly. Conservative mode invalidates
                // every GPR (making no assumption about what the host's syscall
                // service routine touches); otherwise only `syscall_clobbers`.
                let clobbers: &[Register] = if self.conservative {
                    ALL_GPRS
                } else {
                    &self.syscall_clobbers
                };
                for &reg in clobbers {
                    out.set(reg, ValueSet::Top);
                }
            }
            Mnemonic::Int | Mnemonic::Int1 | Mnemonic::Int3 if self.conservative => {
                // A software interrupt transfers to a handler VSA does not
                // analyse (and which becomes a host-serviced VM exit in the
                // lifter). iced reports no GP-register writes for `INT`, but a
                // handler may set any register (e.g. `int 0x80` returns in EAX),
                // so in conservative mode invalidate every GPR. In ABI-aware mode
                // this falls to the catch-all (iced's architectural write set).
                for &reg in ALL_GPRS {
                    out.set(reg, ValueSet::Top);
                }
            }
            // For all other instructions, kill *every* register the instruction
            // writes - not just the first operand. Instructions like `mul`/`div`/
            // `cpuid`/`rdtsc` write RAX/RDX implicitly, and `xchg` writes both
            // operands; consulting iced's instruction-info database ensures none
            // of those destinations keep a stale pre-instruction value that could
            // later resolve an indirect transfer to the wrong target.
            _ => {
                let mut factory = InstructionInfoFactory::new();
                let info = factory.info(inst);
                for used in info.used_registers() {
                    let reg = used.register();
                    if reg == Register::None {
                        continue;
                    }
                    if matches!(
                        used.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    ) {
                        out.set(reg.full_register(), ValueSet::Top);
                    }
                }
            }
        }

        out
    }

    /// Evaluate the abstract value of operand `op_index`.
    fn eval_operand(
        &self,
        state: &RegisterValueSet,
        inst: &Instruction,
        op_index: u32,
    ) -> ValueSet {
        let kind = match op_index {
            0 => inst.op0_kind(),
            1 => inst.op1_kind(),
            2 => inst.op2_kind(),
            _ => return ValueSet::Top,
        };
        match kind {
            OpKind::Register => {
                let reg = match op_index {
                    0 => inst.op0_register(),
                    1 => inst.op1_register(),
                    2 => inst.op2_register(),
                    _ => return ValueSet::Top,
                };
                read_reg(state, reg)
            }
            OpKind::Immediate8 => ValueSet::Constant(inst.immediate8() as i64),
            OpKind::Immediate16 => ValueSet::Constant(inst.immediate16() as i64),
            OpKind::Immediate32 => ValueSet::Constant(inst.immediate32() as i64),
            OpKind::Immediate64 => ValueSet::Constant(inst.immediate64() as i64),
            OpKind::Immediate8to16 => ValueSet::Constant(inst.immediate8to16() as i64),
            OpKind::Immediate8to32 => ValueSet::Constant(inst.immediate8to32() as i64),
            OpKind::Immediate8to64 => ValueSet::Constant(inst.immediate8to64()),
            OpKind::Immediate32to64 => ValueSet::Constant(inst.immediate32to64()),

            // Memory operands: try to resolve if they point into a rodata segment
            OpKind::Memory => {
                let base_reg = inst.memory_base();
                let index_reg = inst.memory_index();
                let disp = inst.memory_displacement64();
                let scale = inst.memory_index_scale();

                let mut addr = disp;

                if base_reg != Register::None {
                    let base_vs = read_reg(state, base_reg);
                    if let ValueSet::Constant(c) = base_vs {
                        addr = addr.wrapping_add(c as u64);
                    } else if base_reg == Register::RIP {
                        // RIP-relative addressing: iced-x86 already resolves the
                        // displacement to an absolute address (disp = RIP + offset).
                        // So 'addr' is already correct.
                    } else {
                        return ValueSet::Top;
                    }
                }

                if index_reg != Register::None {
                    let index_vs = read_reg(state, index_reg);
                    if let ValueSet::Constant(c) = index_vs {
                        addr = addr.wrapping_add((c as u64).wrapping_mul(scale as u64));
                    } else {
                        return ValueSet::Top;
                    }
                }

                // If this address is a known relocation slot, treat it as Top
                // so VSA never propagates the pre-relocation value (often 0).
                if self.relocation_addresses.contains(&addr) {
                    return ValueSet::Top;
                }

                // Address resolved to a constant. Check segments.
                for segment in &self.segments {
                    if segment.contains(addr) {
                        if segment.writable {
                            return ValueSet::Top; // Writable is unpredictable
                        }

                        let size = match inst.memory_size() {
                            MemorySize::UInt8 | MemorySize::Int8 => 1,
                            MemorySize::UInt16 | MemorySize::Int16 => 2,
                            MemorySize::UInt32 | MemorySize::Int32 | MemorySize::DwordOffset => 4,
                            MemorySize::UInt64 | MemorySize::Int64 | MemorySize::QwordOffset => 8,
                            _ => return ValueSet::Top,
                        };

                        if let Some(val) = segment.read(addr, size) {
                            // Sign extend if necessary
                            let val = match inst.memory_size() {
                                MemorySize::Int8 => val as i8 as i64,
                                MemorySize::Int16 => val as i16 as i64,
                                MemorySize::Int32 => val as i32 as i64,
                                _ => val as i64,
                            };
                            return ValueSet::Constant(val);
                        }
                    }
                }

                ValueSet::Top
            }

            // Other operands are unknown
            _ => ValueSet::Top,
        }
    }

    /// Propagate `state` to the instruction at `target_ip`.
    fn propagate(
        &mut self,
        target_ip: u64,
        state: &RegisterValueSet,
        instructions: &[Instruction],
    ) {
        let changed = self.merge(target_ip, state);
        if changed {
            if let Some(idx) = instructions.iter().position(|i| i.ip() == target_ip) {
                self.worklist.push_back(idx);
            }
        }
    }

    /// Merge `state` into the stored state at `ip`. Returns `true` if anything changed.
    fn merge(&mut self, ip: u64, state: &RegisterValueSet) -> bool {
        if let Some(existing) = self.value_sets.get_mut(&ip) {
            // Widen (not just join) so the iteration terminates even when a loop
            // back-edge keeps nudging an interval outward.
            let widened = existing.widen(state);
            if widened.values != existing.values {
                *existing = widened;
                return true;
            }
            false
        } else {
            self.value_sets.insert(ip, state.clone());
            true
        }
    }

    /// Given an indirect-jump instruction (`JMP reg`), return the set of possible
    /// target addresses if it can be resolved, or `None` if not.
    ///
    /// Retained for backward compatibility; [`Self::indirect_targets`] is the
    /// general form (it also resolves `JMP [mem]` and indirect `CALL`).
    pub fn indirect_jump_targets(&self, instruction: &Instruction) -> Option<Vec<u64>> {
        if instruction.mnemonic() != Mnemonic::Jmp {
            return None;
        }
        if instruction.op0_kind() != OpKind::Register {
            return None; // Memory indirect - use `indirect_targets`.
        }
        self.indirect_targets(instruction)
    }

    /// Resolve the possible targets of an indirect `JMP`/`CALL` whose operand is
    /// a register (`JMP rax`, `CALL rax`) or a memory reference (`JMP [mem]`,
    /// `CALL [rip+off]`). Returns `None` for direct transfers, non-transfer
    /// instructions, or when the target cannot be narrowed to a finite set.
    ///
    /// For a memory operand the resolved value is the *contents* of the pointed-to
    /// location (read from a non-writable [`MemorySegment`]); for a register
    /// operand it is the register's value set at the transfer.
    pub fn indirect_targets(&self, instruction: &Instruction) -> Option<Vec<u64>> {
        match instruction.mnemonic() {
            Mnemonic::Jmp | Mnemonic::Call => {}
            _ => return None,
        }
        let pre_state = self.value_sets.get(&instruction.ip())?;
        let vs = match instruction.op0_kind() {
            OpKind::Register => pre_state.get(instruction.op0_register()),
            OpKind::Memory => self.eval_operand(pre_state, instruction, 0),
            // Anything else (a near-branch immediate) is a *direct* transfer.
            _ => return None,
        };
        vs.is_resolvable().then(|| vs.targets())
    }
}

/// Whether `inst` transfers control through a register or memory operand (an
/// *indirect* `JMP`/`CALL`), as opposed to a direct near-branch immediate.
fn is_indirect_transfer(inst: &Instruction) -> bool {
    matches!(inst.op0_kind(), OpKind::Register | OpKind::Memory)
}

/// Run VSA over a region of raw x86-64 code (mapped at `base`) and resolve the
/// targets of every indirect `JMP`/`CALL` it contains.
///
/// This is the region-level convenience behind
/// [`crate::cfg::resolved_indirect_targets`]: it linearly decodes `bytes`,
/// exposes the region itself as a read-only segment (so in-region jump tables and
/// RIP-relative constant loads resolve), and additionally consults any
/// caller-supplied `segments` (e.g. a binary's `.rodata`/import sections) for
/// reads that leave the region.
pub fn resolved_indirect_targets(
    bytes: &[u8],
    base: u64,
    segments: &[MemorySegment],
) -> ResolvedTargets {
    let mut decoder = Decoder::with_ip(64, bytes, base, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        instructions.push(decoder.decode());
    }

    // The region is itself readable, non-writable backing store.
    let mut all_segments = Vec::with_capacity(segments.len() + 1);
    all_segments.push(MemorySegment::new(base, bytes.to_vec(), false));
    all_segments.extend(segments.iter().cloned());

    let mut vsa = ValueSetAnalyzer::with_segments(all_segments);
    vsa.analyze(&instructions);

    let mut out = ResolvedTargets::default();
    for inst in &instructions {
        if !is_indirect_transfer(inst) {
            continue;
        }
        let Some(targets) = vsa.indirect_targets(inst) else {
            continue;
        };
        if targets.is_empty() {
            continue;
        }
        match inst.mnemonic() {
            Mnemonic::Jmp => {
                out.jumps.insert(inst.ip(), targets);
            }
            Mnemonic::Call => {
                out.calls.insert(inst.ip(), targets);
            }
            _ => {}
        }
    }
    out
}

impl Default for ValueSetAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod read_reg_tests {
    use super::*;

    #[test]
    fn low_byte_read_is_exact_but_high_byte_read_is_conservative() {
        let mut state = RegisterValueSet::new();
        state.set(Register::RAX, ValueSet::Constant(0x1234));

        // AL aliases bits 0..7 -> exact low byte.
        assert_eq!(read_reg(&state, Register::AL), ValueSet::Constant(0x34));
        // EAX aliases bits 0..31 -> exact.
        assert_eq!(read_reg(&state, Register::EAX), ValueSet::Constant(0x1234));
        // AH aliases bits 8..15, which this analysis does not track: it must NOT
        // report the low byte (0x34) as if it were AH. Conservative `Top`.
        assert_eq!(read_reg(&state, Register::AH), ValueSet::Top);
        assert_eq!(read_reg(&state, Register::BH), ValueSet::Top);
    }
}
