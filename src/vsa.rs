//! Value Set Analysis: abstract interpretation that narrows indirect `JMP`/`CALL` targets at lift time.
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

use crate::arch::ControlFlowKind;
use crate::cfg::ResolvedTargets;
use crate::context::MemorySegment;
use iced_x86::{
    Decoder, DecoderOptions, Instruction, InstructionInfoFactory, MemorySize, Mnemonic, OpAccess,
    OpKind, Register,
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Registers assumed clobbered across a `CALL` in ABI-aware mode: the System V and Windows x64
/// volatile union. Others are assumed preserved; see [`ValueSetAnalyzer::all_gpr_clobbers`].
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

/// Registers a `SYSCALL` clobbers: RCX, R11 (written by the instruction) and RAX (return value).
const DEFAULT_SYSCALL_CLOBBERS: &[Register] = &[Register::RAX, Register::RCX, Register::R11];

/// All 16 GPRs: the conservative `CALL` clobber set.
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


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueSet {
    /// Unreachable.
    Empty,
    Constant(i64),
    Multiple(BTreeSet<i64>),
    Interval { min: i64, max: i64 },
    Top,
}

impl ValueSet {
    /// Cardinality beyond which `Multiple` widens to `Top`.
    const MAX_SET_SIZE: usize = 64;

    /// Lattice join.
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
            // Mixed interval + other: widen to Top.
            _ => ValueSet::Top,
        }
    }

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

    pub fn abstract_and(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            // x & 0 = 0 for all x.
            (ValueSet::Constant(0), _) | (_, ValueSet::Constant(0)) => ValueSet::Constant(0),
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => ValueSet::Constant(a & b),
            _ => ValueSet::Top,
        }
    }

    pub fn abstract_xor(&self, other: &ValueSet) -> ValueSet {
        match (self, other) {
            (ValueSet::Empty, _) | (_, ValueSet::Empty) => ValueSet::Empty,
            (ValueSet::Top, _) | (_, ValueSet::Top) => ValueSet::Top,
            (ValueSet::Constant(a), ValueSet::Constant(b)) => ValueSet::Constant(a ^ b),
            _ => ValueSet::Top,
        }
    }

    /// Keeps the low `bits` bits, zero-extending (narrow read / zero-extending write).
    /// Only `Constant`/`Multiple` are exact; an `Interval` falls to `Top`.
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
                // Masking can collapse the set.
                if m.len() == 1 {
                    ValueSet::Constant(*m.iter().next().unwrap())
                } else {
                    ValueSet::Multiple(m)
                }
            }
            _ => ValueSet::Top,
        }
    }

    /// Join that sends a growing `Interval` straight to `Top`, so loop back-edges reach a fixpoint.
    /// `self` is the accumulated value, `other` the new one.
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

    pub fn is_resolvable(&self) -> bool {
        matches!(self, ValueSet::Constant(_) | ValueSet::Multiple(_))
    }

    pub fn targets(&self) -> Vec<u64> {
        match self {
            ValueSet::Constant(c) => vec![*c as u64],
            ValueSet::Multiple(s) => s.iter().map(|&v| v as u64).collect(),
            _ => vec![],
        }
    }
}


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

    pub fn join(&self, other: &RegisterValueSet) -> RegisterValueSet {
        let mut result = self.clone();
        for (reg, vs) in &other.values {
            let joined = result.get(*reg).join(vs);
            result.set(*reg, joined);
        }
        result
    }

    /// Per-register [widening](ValueSet::widen); keeps merge iteration terminating.
    pub fn widen(&self, other: &RegisterValueSet) -> RegisterValueSet {
        let mut result = self.clone();
        for (reg, vs) in &other.values {
            let widened = result.get(*reg).widen(vs);
            result.set(*reg, widened);
        }
        result
    }
}

/// Reads a possibly sub-width register from its 64-bit container, masked to the operand width.
fn read_reg(state: &RegisterValueSet, reg: Register) -> ValueSet {
    // AH/BH/CH/DH alias bits 8..15, not the low byte; read them as `Top`.
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

/// Writes a possibly sub-width register into its container: 64-bit replaces, 32-bit zero-extends,
/// 8/16-bit leaves untracked upper bits so the result is `Top`.
fn write_reg(out: &mut RegisterValueSet, reg: Register, vs: ValueSet) {
    let full = reg.full_register();
    match reg.size() {
        8 => out.set(full, vs),
        4 => out.set(full, vs.low_bits(32)),
        _ => out.set(full, ValueSet::Top),
    }
}


/// Dataflow-based Value Set Analyzer.
///
/// Usage:
/// ```ignore
/// let mut vsa = ValueSetAnalyzer::new();
/// vsa.analyze(&instructions);
/// let targets = vsa.indirect_jump_targets(&jmp_instruction);
/// ```
pub struct ValueSetAnalyzer {
    /// IP -> register value sets before that instruction.
    pub value_sets: HashMap<u64, RegisterValueSet>,
    pub segments: Vec<MemorySegment>,
    /// Addresses always read as `Top`, e.g. GOT slots holding 0 before the dynamic linker runs.
    pub relocation_addresses: HashSet<u64>,
    /// `true` (default): every GPR is clobbered across `CALL`/`SYSCALL`, preserving runtime
    /// equivalence whatever the callee does. `false` (see [`with_standard_abi`](Self::with_standard_abi))
    /// uses `call_clobbers`/`syscall_clobbers`; only for code honouring the configured ABI.
    pub conservative: bool,
    /// Caller-saved set clobbered across `CALL` when `conservative == false`; others are assumed
    /// preserved. Defaults to [`DEFAULT_CALL_CLOBBERS`].
    pub call_clobbers: Vec<Register>,
    /// Registers clobbered by `SYSCALL` when `conservative == false`. Defaults to [`DEFAULT_SYSCALL_CLOBBERS`].
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
            // Safe by default; opt into precision via `with_standard_abi`.
            conservative: true,
            call_clobbers: DEFAULT_CALL_CLOBBERS.to_vec(),
            syscall_clobbers: DEFAULT_SYSCALL_CLOBBERS.to_vec(),
            worklist: VecDeque::new(),
        }
    }

    /// Marks addresses as relocation slots, read as `Top` to avoid stale pre-relocation values.
    pub fn with_relocation_addresses(mut self, addrs: HashSet<u64>) -> Self {
        self.relocation_addresses = addrs;
        self
    }

    /// Sets [`conservative`](Self::conservative).
    pub fn with_conservative(mut self, conservative: bool) -> Self {
        self.conservative = conservative;
        self
    }

    /// Opts into ABI-aware precision (`conservative = false`); sound only for code honouring the ABI.
    pub fn with_standard_abi(mut self) -> Self {
        self.conservative = false;
        self
    }

    /// Overrides the `CALL` clobber set and switches to ABI-aware mode.
    pub fn with_call_clobbers(mut self, regs: Vec<Register>) -> Self {
        self.conservative = false;
        self.call_clobbers = regs;
        self
    }

    /// Overrides the `SYSCALL` clobber set and switches to ABI-aware mode.
    pub fn with_syscall_clobbers(mut self, regs: Vec<Register>) -> Self {
        self.conservative = false;
        self.syscall_clobbers = regs;
        self
    }

    pub fn default_call_clobbers() -> Vec<Register> {
        DEFAULT_CALL_CLOBBERS.to_vec()
    }

    /// Every GPR: assumes nothing about callee preservation.
    pub fn all_gpr_clobbers() -> Vec<Register> {
        ALL_GPRS.to_vec()
    }

    /// Runs a worklist analysis over decoded instructions to a fixpoint; merges widen
    /// ([`RegisterValueSet::widen`]) to guarantee termination.
    pub fn analyze(&mut self, instructions: &[Instruction]) {
        if instructions.is_empty() {
            return;
        }

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

            let pre_state = self
                .value_sets
                .get(&ip)
                .cloned()
                .unwrap_or_else(RegisterValueSet::new);

            let post_state = self.transfer(&pre_state, instruction);

            let next_ip = instruction.next_ip();

            // Successor edges come from the lifter's classification: a missed back-edge would
            // under-approximate value sets and resolve indirect jumps to too few targets.
            let cf = crate::arch::x86_64::decode::control_flow(instruction);
            match cf.kind {
                ControlFlowKind::DirectBranch => {
                    for &target in &cf.static_targets {
                        self.propagate(target, &post_state, instructions);
                    }
                }
                ControlFlowKind::CondBranch => {
                    for &target in &cf.static_targets {
                        self.propagate(target, &post_state, instructions);
                    }
                    self.propagate(next_ip, &post_state, instructions);
                }
                ControlFlowKind::IndirectBranch => {
                }
                ControlFlowKind::Ret => {
                }
                // A `CALL` returns and an invalid encoding is left to the lifter; both keep the fallthrough.
                ControlFlowKind::Fallthrough
                | ControlFlowKind::Call { .. }
                | ControlFlowKind::Invalid => {
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

    /// Transfer function: pre-state to post-state for one instruction.
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
                    // Not tracked precisely.
                    write_reg(&mut out, inst.op0_register(), ValueSet::Top);
                }
            }
            Mnemonic::Lea => {
                if inst.op0_kind() == OpKind::Register {
                    // LEA with a constant displacement propagates.
                    let base_reg = inst.memory_base();
                    let disp = inst.memory_displacement64() as i64;
                    let addr_vs =
                        if inst.memory_index() == Register::None && base_reg != Register::None {
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
                // A call clobbers every GPR in conservative mode, else only the configured
                // caller-saved set (others assumed callee-preserved).
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
                // Falls through; iced does not report RAX as written, so the clobber is explicit.
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
                // Software interrupt to an unanalysed handler (`int 0x80` returns in EAX):
                // conservative mode clobbers every GPR; ABI-aware falls to the catch-all.
                for &reg in ALL_GPRS {
                    out.set(reg, ValueSet::Top);
                }
            }
            // Kill every register the instruction writes (implicit `mul`/`div`/`cpuid` operands,
            // both `xchg` operands) via iced's info database, so none keeps a stale value.
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

    /// Abstract value of operand `op_index`.
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

            // Memory: resolve if it points into a read-only segment.
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
                        // iced resolves RIP-relative displacements to absolute addresses.
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

                // Relocation slot: Top, never the pre-relocation value.
                if self.relocation_addresses.contains(&addr) {
                    return ValueSet::Top;
                }

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

            _ => ValueSet::Top,
        }
    }

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

    /// Merges `state` at `ip`; returns `true` if anything changed.
    fn merge(&mut self, ip: u64, state: &RegisterValueSet) -> bool {
        if let Some(existing) = self.value_sets.get_mut(&ip) {
            // Widen, not join, so loop back-edges terminate.
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

    /// Targets of an indirect `JMP reg`, or `None`; superseded by [`Self::indirect_targets`].
    pub fn indirect_jump_targets(&self, instruction: &Instruction) -> Option<Vec<u64>> {
        if instruction.mnemonic() != Mnemonic::Jmp {
            return None;
        }
        if instruction.op0_kind() != OpKind::Register {
            return None; // Memory indirect - use `indirect_targets`.
        }
        self.indirect_targets(instruction)
    }

    /// Targets of an indirect `JMP`/`CALL` through a register (its value set) or memory operand
    /// (the pointed-to contents, read from a non-writable [`MemorySegment`]); `None` for direct
    /// transfers or when not narrowable to a finite set.
    pub fn indirect_targets(&self, instruction: &Instruction) -> Option<Vec<u64>> {
        match instruction.mnemonic() {
            Mnemonic::Jmp | Mnemonic::Call => {}
            _ => return None,
        }
        let pre_state = self.value_sets.get(&instruction.ip())?;
        let vs = match instruction.op0_kind() {
            OpKind::Register => pre_state.get(instruction.op0_register()),
            OpKind::Memory => self.eval_operand(pre_state, instruction, 0),
            // Near-branch immediate: a direct transfer.
            _ => return None,
        };
        vs.is_resolvable().then(|| vs.targets())
    }
}

/// Whether `inst` transfers control through a register or memory operand.
fn is_indirect_transfer(inst: &Instruction) -> bool {
    matches!(inst.op0_kind(), OpKind::Register | OpKind::Memory)
}

/// Runs VSA over raw x86-64 code mapped at `base` and resolves every indirect `JMP`/`CALL`
/// (behind [`crate::cfg::resolved_indirect_targets`]). The region counts as a read-only segment;
/// `segments` serve reads that leave it.
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

        assert_eq!(read_reg(&state, Register::AL), ValueSet::Constant(0x34));
        assert_eq!(read_reg(&state, Register::EAX), ValueSet::Constant(0x1234));
        // AH aliases untracked bits 8..15: must be `Top`, not the low byte.
        assert_eq!(read_reg(&state, Register::AH), ValueSet::Top);
        assert_eq!(read_reg(&state, Register::BH), ValueSet::Top);
    }
}
