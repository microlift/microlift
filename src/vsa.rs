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

use crate::context::MemorySegment;
use iced_x86::{Instruction, MemorySize, Mnemonic, OpKind, Register};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

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
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => ValueSet::Constant(a & b),
            (ValueSet::Constant(0), _) | (_, ValueSet::Constant(0)) => ValueSet::Constant(0),
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
    worklist: VecDeque<usize>,
    visited: HashSet<usize>,
}

impl ValueSetAnalyzer {
    pub fn new() -> Self {
        Self::with_segments(Vec::new())
    }

    pub fn with_segments(segments: Vec<MemorySegment>) -> Self {
        Self {
            value_sets: HashMap::new(),
            segments,
            worklist: VecDeque::new(),
            visited: HashSet::new(),
        }
    }

    /// Run the analysis over a decoded instruction stream.
    ///
    /// The instructions should already be topologically sorted (i.e. the normal
    /// decode order). This is a single-pass forward analysis; for loops / back-
    /// edges it is conservative (widening to `Top` on re-visit).
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
                | Mnemonic::Loop => {
                    let target = instruction.near_branch_target();
                    self.propagate(target, &post_state, instructions);
                    self.propagate(next_ip, &post_state, instructions);
                }
                Mnemonic::Ret | Mnemonic::Syscall => {
                    // Terminators - no fallthrough
                }
                _ => {
                    // Fallthrough to next instruction
                    if idx + 1 < instructions.len() {
                        let next_idx = idx + 1;
                        let next_inst_ip = instructions[next_idx].ip();
                        let changed = self.merge(next_inst_ip, &post_state);
                        if changed && !self.visited.contains(&next_idx) {
                            self.worklist.push_back(next_idx);
                        }
                    }
                }
            }

            self.visited.insert(idx);
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
                    out.set(dst, src_vs);
                }
            }
            Mnemonic::Xor => {
                if inst.op0_kind() == OpKind::Register
                    && inst.op1_kind() == OpKind::Register
                    && inst.op0_register() == inst.op1_register()
                {
                    // `xor rax, rax` -> 0
                    out.set(inst.op0_register(), ValueSet::Constant(0));
                } else if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    out.set(inst.op0_register(), a.abstract_xor(&b));
                }
            }
            Mnemonic::Add => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    out.set(inst.op0_register(), a.abstract_add(&b));
                }
            }
            Mnemonic::Sub => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    out.set(inst.op0_register(), a.abstract_sub(&b));
                }
            }
            Mnemonic::And => {
                if inst.op0_kind() == OpKind::Register {
                    let a = self.eval_operand(state, inst, 0);
                    let b = self.eval_operand(state, inst, 1);
                    out.set(inst.op0_register(), a.abstract_and(&b));
                }
            }
            Mnemonic::Or => {
                if inst.op0_kind() == OpKind::Register {
                    // Conservative - don't try to track OR precisely
                    out.set(inst.op0_register(), ValueSet::Top);
                }
            }
            Mnemonic::Lea => {
                if inst.op0_kind() == OpKind::Register {
                    // LEA with a constant displacement can propagate
                    let base_reg = inst.memory_base();
                    let disp = inst.memory_displacement64() as i64;
                    if inst.memory_index() == Register::None && base_reg != Register::None {
                        let base_vs = state.get(base_reg);
                        let disp_vs = ValueSet::Constant(disp);
                        out.set(inst.op0_register(), base_vs.abstract_add(&disp_vs));
                    } else {
                        out.set(inst.op0_register(), ValueSet::Top);
                    }
                }
            }
            // For all other instructions, kill any written destination
            _ => {
                if inst.op0_kind() == OpKind::Register {
                    out.set(inst.op0_register(), ValueSet::Top);
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
                state.get(reg)
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
                    let base_vs = state.get(base_reg);
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
                    let index_vs = state.get(index_reg);
                    if let ValueSet::Constant(c) = index_vs {
                        addr = addr.wrapping_add((c as u64).wrapping_mul(scale as u64));
                    } else {
                        return ValueSet::Top;
                    }
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
                            MemorySize::UInt32 | MemorySize::Int32 => 4,
                            MemorySize::UInt64 | MemorySize::Int64 => 8,
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
                if !self.visited.contains(&idx) {
                    self.worklist.push_back(idx);
                }
            }
        }
    }

    /// Merge `state` into the stored state at `ip`. Returns `true` if anything changed.
    fn merge(&mut self, ip: u64, state: &RegisterValueSet) -> bool {
        if let Some(existing) = self.value_sets.get_mut(&ip) {
            let joined = existing.join(state);
            if joined.values != existing.values {
                *existing = joined;
                return true;
            }
            false
        } else {
            self.value_sets.insert(ip, state.clone());
            true
        }
    }

    /// Given an indirect-jump instruction (`JMP reg` or `JMP [mem]`), return the
    /// set of possible target addresses if it can be resolved, or `None` if not.
    pub fn indirect_jump_targets(&self, instruction: &Instruction) -> Option<Vec<u64>> {
        if instruction.mnemonic() != Mnemonic::Jmp {
            return None;
        }
        if instruction.op0_kind() != OpKind::Register {
            return None; // Memory indirect - not handled yet
        }
        let reg = instruction.op0_register();
        let pre_state = self.value_sets.get(&instruction.ip())?;
        let vs = pre_state.get(reg);
        if vs.is_resolvable() {
            Some(vs.targets())
        } else {
            None
        }
    }
}

impl Default for ValueSetAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
