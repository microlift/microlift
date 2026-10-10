//! **Experimental** recovery of Clang-style jump tables for indirect `JMP`s
//! ([`ControlFlowKind::IndirectBranch`] otherwise contributes no leaders).
//!
//! Tables must lie in read-only memory described via [`LifterConfig::segments`]. Matches two shapes:
//!
//! ```text
//! ; PIC: table holds i32 displacements from the table itself
//!     lea    rcx, [rip + .LJTI]
//!     movsxd rdx, dword ptr [rcx + rax*4]
//!     add    rdx, rcx
//!     jmp    rdx
//!
//! ; non-PIC: table holds absolute addresses
//!     jmp    qword ptr [.LJTI + rax*8]
//! ```
//!
//! Safe by construction: targets are only added as leaders and dispatch stays on the runtime
//! address, so a bogus entry lifts an unreached block. Writable segments are never trusted.
//!
//! [`ControlFlowKind::IndirectBranch`]: crate::arch::ControlFlowKind::IndirectBranch
//! [`LifterConfig::segments`]: crate::core::LifterConfig::segments

use crate::context::MemorySegment;
use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};
use std::collections::HashMap;

/// How far back from an indirect `JMP` to look for its target computation.
const MAX_LOOKBACK: usize = 32;
const MAX_ENTRIES: u64 = 4096;
/// Cap on entries walked when no range check was proven.
const MAX_UNBOUNDED_ENTRIES: u64 = 512;
const MAX_CONST_DEPTH: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpTableKind {
    /// The entry is the target address, zero-extended.
    Absolute,
    /// Signed displacement from `anchor` (the RIP-relative `lea` address, usually the table itself).
    Relative { anchor: u64 },
}

#[derive(Debug, Clone)]
pub struct JumpTable {
    pub jump_ip: u64,
    pub table: u64,
    pub entry_size: u8,
    pub kind: JumpTableKind,
    pub index: Register,
    /// `true` when a `cmp idx, N` / `ja` range check fixed the count; `false` for a heuristic walk.
    pub bounded: bool,
    pub targets: Vec<u64>,
    pub entries: u64,
}

impl JumpTable {
    /// Guest bytes the table occupies; a translation depends on them as on code
    /// ([`LiftedCode::data_dependencies`](crate::core::LiftedCode::data_dependencies)).
    pub fn extent(&self) -> std::ops::Range<u64> {
        self.table..self.table + self.entries * self.entry_size as u64
    }
}

/// Runs the pass over code mapped at `base`; one entry per indirect `JMP` with a recoverable table,
/// keyed by its address. The region itself counts as a readable segment.
pub fn analyze_jump_tables(
    bytes: &[u8],
    base: u64,
    segments: &[MemorySegment],
) -> HashMap<u64, JumpTable> {
    let mut decoder = Decoder::with_ip(64, bytes, base, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        instructions.push(decoder.decode());
    }

    let mut all_segments = Vec::with_capacity(segments.len() + 1);
    all_segments.push(MemorySegment::new(base, bytes.to_vec(), false));
    all_segments.extend(segments.iter().cloned());

    let region = base..base + bytes.len() as u64;

    let mut out = HashMap::new();
    for idx in 0..instructions.len() {
        let inst = &instructions[idx];
        if inst.mnemonic() != Mnemonic::Jmp {
            continue;
        }
        if !matches!(inst.op0_kind(), OpKind::Register | OpKind::Memory) {
            continue;
        }
        if let Some(table) = match_jump_table(&instructions, idx, &all_segments, &region) {
            out.insert(inst.ip(), table);
        }
    }
    out
}

struct TableAccess {
    table: u64,
    entry_size: u8,
    index: Register,
}

fn match_jump_table(
    instrs: &[Instruction],
    jmp_idx: usize,
    segments: &[MemorySegment],
    region: &std::ops::Range<u64>,
) -> Option<JumpTable> {
    let jmp = &instrs[jmp_idx];

    let (access, kind) = match jmp.op0_kind() {
        // `jmp qword ptr [table + idx*8]`: the load is the dispatch.
        OpKind::Memory => (table_access(instrs, jmp_idx, jmp)?, JumpTableKind::Absolute),
        // `jmp reg`: walk back to the definition of `reg`.
        OpKind::Register => {
            let reg = jmp.op0_register();
            let def_idx = find_def(instrs, jmp_idx, reg)?;
            match_target_expr(instrs, def_idx, reg)?
        }
        _ => return None,
    };

    if !matches!(access.entry_size, 4 | 8) {
        return None;
    }

    let bound = find_bound(instrs, jmp_idx, access.index);
    let (targets, entries) = read_targets(&access, kind, bound, segments, region)?;

    Some(JumpTable {
        jump_ip: jmp.ip(),
        table: access.table,
        entry_size: access.entry_size,
        kind,
        index: access.index,
        bounded: bound.is_some(),
        targets,
        entries,
    })
}

/// Interprets the definition of `reg` as the PIC `add entry, anchor` or a plain absolute load.
fn match_target_expr(
    instrs: &[Instruction],
    def_idx: usize,
    reg: Register,
) -> Option<(TableAccess, JumpTableKind)> {
    let def = &instrs[def_idx];

    if def.mnemonic() == Mnemonic::Add
        && def.op0_kind() == OpKind::Register
        && def.op1_kind() == OpKind::Register
    {
        // One operand is the entry, the other the RIP-relative anchor; either order.
        let (dst, src) = (def.op0_register(), def.op1_register());

        if let Some(anchor) = resolve_const(instrs, def_idx, src, 0) {
            let load_idx = find_def(instrs, def_idx, dst)?;
            let access = table_load(instrs, load_idx)?;
            return Some((access, JumpTableKind::Relative { anchor }));
        }
        if let Some(anchor) = resolve_const(instrs, def_idx, dst, 0) {
            let load_idx = find_def(instrs, def_idx, src)?;
            let access = table_load(instrs, load_idx)?;
            return Some((access, JumpTableKind::Relative { anchor }));
        }
        return None;
    }

    let _ = reg;
    let access = table_load(instrs, def_idx)?;
    Some((access, JumpTableKind::Absolute))
}

fn table_load(instrs: &[Instruction], idx: usize) -> Option<TableAccess> {
    let inst = &instrs[idx];
    match inst.mnemonic() {
        Mnemonic::Mov | Mnemonic::Movsxd | Mnemonic::Movsx | Mnemonic::Movzx => {}
        _ => return None,
    }
    if inst.op0_kind() != OpKind::Register || inst.op1_kind() != OpKind::Memory {
        return None;
    }
    table_access(instrs, idx, inst)
}

/// Extracts base, entry width and index register from `inst`'s memory operand.
fn table_access(instrs: &[Instruction], idx: usize, inst: &Instruction) -> Option<TableAccess> {
    let index = inst.memory_index();
    // No index register: a single indirection (GOT slot), already resolved by `crate::vsa`.
    if index == Register::None {
        return None;
    }
    let scale = inst.memory_index_scale();
    if !matches!(scale, 4 | 8) {
        return None;
    }

    let disp = inst.memory_displacement64();
    let table = match inst.memory_base() {
        Register::None => disp,
        // RIP-relative operands cannot carry an index.
        Register::RIP | Register::EIP => return None,
        base => resolve_const(instrs, idx, base, 0)?.wrapping_add(disp),
    };

    let entry_size = inst.memory_size().size() as u8;
    Some(TableAccess {
        table,
        entry_size,
        index,
    })
}

/// Resolves `reg` to a constant via a RIP-relative `lea`, `mov reg, imm`, or a copy that resolves.
fn resolve_const(instrs: &[Instruction], before: usize, reg: Register, depth: u32) -> Option<u64> {
    if depth > MAX_CONST_DEPTH {
        return None;
    }
    let def_idx = find_def(instrs, before, reg)?;
    let def = &instrs[def_idx];
    match def.mnemonic() {
        Mnemonic::Lea if def.memory_base() == Register::RIP => Some(def.memory_displacement64()),
        Mnemonic::Lea if def.memory_base() == Register::None => Some(def.memory_displacement64()),
        Mnemonic::Mov if is_immediate(def.op1_kind()) => Some(def.immediate(1)),
        Mnemonic::Mov if def.op1_kind() == OpKind::Register => {
            resolve_const(instrs, def_idx, def.op1_register(), depth + 1)
        }
        _ => None,
    }
}

/// Finds the entry count from the `cmp idx, N` / `ja default` check, following register copies.
fn find_bound(instrs: &[Instruction], jmp_idx: usize, index: Register) -> Option<u64> {
    let mut tracked = index.full_register();
    let stop = jmp_idx.saturating_sub(MAX_LOOKBACK);
    let mut i = jmp_idx;
    let mut info = InstructionInfoFactory::new();

    while i > stop {
        i -= 1;
        let inst = &instrs[i];

        if inst.mnemonic() == Mnemonic::Cmp
            && inst.op0_kind() == OpKind::Register
            && inst.op0_register().full_register() == tracked
            && is_immediate(inst.op1_kind())
        {
            // The guarding branch must immediately follow the compare.
            let imm = inst.immediate(1);
            return match instrs.get(i + 1)?.mnemonic() {
                Mnemonic::Ja | Mnemonic::Jg => imm.checked_add(1),
                Mnemonic::Jae | Mnemonic::Jge => Some(imm),
                _ => None,
            };
        }

        if writes_reg(&mut info, inst, tracked) {
            // A copy renames the index; anything else redefines it.
            if is_reg_copy(inst) {
                tracked = inst.op1_register().full_register();
                continue;
            }
            return None;
        }
    }
    None
}

/// Reads `count` entries (unbounded: until one looks implausible) as targets; `None` if unreadable
/// or a proven bound runs off the segments.
fn read_targets(
    access: &TableAccess,
    kind: JumpTableKind,
    bound: Option<u64>,
    segments: &[MemorySegment],
    region: &std::ops::Range<u64>,
) -> Option<(Vec<u64>, u64)> {
    let count = match bound {
        Some(0) => return None,
        Some(n) if n > MAX_ENTRIES => return None,
        Some(n) => n,
        None => MAX_UNBOUNDED_ENTRIES,
    };

    let mut targets: Vec<u64> = Vec::new();
    // Separate from `targets`, which de-duplicates, and early exit makes `count` unreliable.
    let mut entries = 0u64;
    for i in 0..count {
        let addr = access
            .table
            .checked_add(i.checked_mul(access.entry_size as u64)?)?;
        let raw = match read_ro(segments, addr, access.entry_size) {
            Some(raw) => raw,
            // Bound runs off the segments: the match was wrong; do not guess.
            None if bound.is_some() => return None,
            None => break,
        };
        let target = match kind {
            JumpTableKind::Absolute => raw,
            JumpTableKind::Relative { anchor } => {
                anchor.wrapping_add(sign_extend(raw, access.entry_size) as u64)
            }
        };
        // Unbounded: stop at the first entry that does not point back into the region.
        if bound.is_none() && !region.contains(&target) {
            break;
        }
        entries = i + 1;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    (!targets.is_empty()).then_some((targets, entries))
}

/// Reads `size` bytes from non-writable segments only (same rule as [`crate::vsa`]).
fn read_ro(segments: &[MemorySegment], addr: u64, size: u8) -> Option<u64> {
    segments
        .iter()
        .filter(|seg| !seg.writable)
        .find_map(|seg| seg.read(addr, size as usize))
}

fn sign_extend(raw: u64, size: u8) -> i64 {
    match size {
        1 => raw as u8 as i8 as i64,
        2 => raw as u16 as i16 as i64,
        4 => raw as u32 as i32 as i64,
        _ => raw as i64,
    }
}

/// Walks back for the last writer of `reg`, past conditional branches (the range check's `ja`).
fn find_def(instrs: &[Instruction], from: usize, reg: Register) -> Option<usize> {
    let full = reg.full_register();
    let stop = from.saturating_sub(MAX_LOOKBACK);
    let mut info = InstructionInfoFactory::new();
    let mut i = from;

    while i > stop {
        i -= 1;
        let inst = &instrs[i];
        match inst.flow_control() {
            FlowControl::Next | FlowControl::ConditionalBranch => {}
            _ => return None,
        }
        if writes_reg(&mut info, inst, full) {
            return Some(i);
        }
    }
    None
}

fn writes_reg(info: &mut InstructionInfoFactory, inst: &Instruction, full: Register) -> bool {
    info.info(inst).used_registers().iter().any(|used| {
        matches!(
            used.access(),
            OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
        ) && used.register().full_register() == full
    })
}

fn is_reg_copy(inst: &Instruction) -> bool {
    matches!(
        inst.mnemonic(),
        Mnemonic::Mov | Mnemonic::Movsxd | Mnemonic::Movsx | Mnemonic::Movzx
    ) && inst.op0_kind() == OpKind::Register
        && inst.op1_kind() == OpKind::Register
}

fn is_immediate(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Immediate8
            | OpKind::Immediate8_2nd
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}
