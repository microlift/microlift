//! **Experimental** recovery of Clang-style jump tables.
//!
//! An indirect `JMP` normally ends a region: the engine cannot name its
//! successors, so [`ControlFlowKind::IndirectBranch`] contributes no block
//! leaders and the lifted code dispatches to a `UnknownJump` VM exit for any
//! target that does not happen to already be a block. A C `switch` compiled by
//! Clang is the common case where the successors *are* statically knowable -
//! they sit in a table in read-only memory, and the caller has already told us
//! what read-only memory looks like via [`LifterConfig::segments`].
//!
//! This pass pattern-matches the two shapes Clang emits on x86-64 and reads the
//! table out of those segments, so the targets become ordinary block leaders and
//! the exit disappears from the hot path.
//!
//! ```text
//! ; PIC / -fPIC: table holds i32 displacements from the table itself
//!     cmp    edi, 3
//!     ja     .default
//!     mov    eax, edi
//!     lea    rcx, [rip + .LJTI]
//!     movsxd rdx, dword ptr [rcx + rax*4]
//!     add    rdx, rcx
//!     jmp    rdx
//!
//! ; non-PIC: table holds absolute addresses
//!     cmp    edi, 3
//!     ja     .default
//!     mov    eax, edi
//!     jmp    qword ptr [.LJTI + rax*8]
//! ```
//!
//! **Being wrong here is cheap, and that is the design.** A recovered target is
//! only ever *added* to the leader set: dispatch still happens on the address
//! the guest actually computed at run time, so a bogus entry lifts a block that
//! nothing reaches (wasted code, never wrong code), and a missed entry just
//! leaves the pre-existing `UnknownJump` exit in place. That is why the matcher
//! is allowed a heuristic unbounded mode when no `cmp`/`ja` range check is
//! found. It is *not* a licence to be sloppy about where entries are read
//! from - writable segments are never trusted, exactly as in [`crate::vsa`].
//!
//! [`ControlFlowKind::IndirectBranch`]: crate::arch::ControlFlowKind::IndirectBranch
//! [`LifterConfig::segments`]: crate::core::LifterConfig::segments

use crate::context::MemorySegment;
use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};
use std::collections::HashMap;

/// How far back from an indirect `JMP` to look for the instructions that
/// computed its target. The dispatch sequences above are 3-6 instructions; the
/// slack covers scheduling that has moved unrelated work in between.
const MAX_LOOKBACK: usize = 32;
/// Hard cap on the number of entries read from one table.
const MAX_ENTRIES: u64 = 4096;
/// Cap used when no range check was proven and entries are walked until one
/// stops looking like an in-region target.
const MAX_UNBOUNDED_ENTRIES: u64 = 512;
/// Depth limit when chasing register copies back to a constant.
const MAX_CONST_DEPTH: u32 = 4;

/// How the bytes in a recovered table encode a target address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpTableKind {
    /// Each entry is the target address itself, zero-extended from
    /// [`JumpTable::entry_size`].
    Absolute,
    /// Each entry is a *signed* displacement from `anchor` - the address Clang
    /// materialized with a RIP-relative `lea` and then added to the loaded
    /// entry. Almost always the table's own address.
    Relative { anchor: u64 },
}

/// One recovered jump table.
#[derive(Debug, Clone)]
pub struct JumpTable {
    /// Virtual address of the indirect `JMP` this table dispatches.
    pub jump_ip: u64,
    /// Virtual address of the first table entry.
    pub table: u64,
    /// Width of one entry in bytes (4 or 8).
    pub entry_size: u8,
    /// How to turn an entry into an address.
    pub kind: JumpTableKind,
    /// The register holding the switch index at the table load.
    pub index: Register,
    /// `true` when a `cmp idx, N` / `ja` range check fixed the entry count, so
    /// the table was read exactly. `false` when the count is a heuristic walk.
    pub bounded: bool,
    /// The recovered targets, in table order, de-duplicated.
    pub targets: Vec<u64>,
}

/// Run the pass over a region of x86-64 code mapped at `base`.
///
/// Returns one entry per indirect `JMP` whose table could be recovered, keyed by
/// the address of that `JMP`. `segments` supplies the read-only memory the table
/// is read from; the region itself is added as a read-only segment, so a table
/// emitted inside the lifted code (common for a `switch` at the end of a
/// function) resolves without the caller describing it.
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

/// Shape of the table access, before the entries have been read.
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
        // `jmp qword ptr [table + idx*8]` - the load *is* the dispatch, and what
        // it loads is the target.
        OpKind::Memory => (table_access(instrs, jmp_idx, jmp)?, JumpTableKind::Absolute),
        // `jmp reg` - walk back to whatever defined `reg`.
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
    let targets = read_targets(&access, kind, bound, segments, region)?;

    Some(JumpTable {
        jump_ip: jmp.ip(),
        table: access.table,
        entry_size: access.entry_size,
        kind,
        index: access.index,
        bounded: bound.is_some(),
        targets,
    })
}

/// Interpret the instruction at `def_idx`, which defines the `JMP`'s target
/// register `reg`, as either the `add entry, anchor` of the PIC form or a plain
/// load of an absolute entry.
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
        // One operand is the loaded entry, the other the RIP-relative anchor.
        // Clang emits both orderings depending on register allocation.
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

    // No relocation step: the loaded entry is the address.
    let _ = reg;
    let access = table_load(instrs, def_idx)?;
    Some((access, JumpTableKind::Absolute))
}

/// Match `mov`/`movsxd`/`movzx reg, [table + idx*scale]`.
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

/// Extract the table base / entry width / index register from the memory
/// operand of `inst` (which lives at `idx`).
fn table_access(instrs: &[Instruction], idx: usize, inst: &Instruction) -> Option<TableAccess> {
    let index = inst.memory_index();
    // No index register means a single indirection (a GOT slot, say), not a
    // table. `crate::vsa` already resolves those to a constant.
    if index == Register::None {
        return None;
    }
    let scale = inst.memory_index_scale();
    if !matches!(scale, 4 | 8) {
        return None;
    }

    let disp = inst.memory_displacement64();
    let table = match inst.memory_base() {
        // Absolute displacement: `[table + idx*8]`.
        Register::None => disp,
        // RIP-relative operands cannot carry an index register, so reaching
        // here would mean the decode disagrees with the ISA.
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

/// Resolve `reg` to a constant by walking back to its definition: a
/// RIP-relative `lea` (the usual way Clang materializes a table address), a
/// `mov reg, imm`, or a copy from another register that itself resolves.
fn resolve_const(instrs: &[Instruction], before: usize, reg: Register, depth: u32) -> Option<u64> {
    if depth > MAX_CONST_DEPTH {
        return None;
    }
    let def_idx = find_def(instrs, before, reg)?;
    let def = &instrs[def_idx];
    match def.mnemonic() {
        // iced resolves a RIP-relative displacement to its absolute target.
        Mnemonic::Lea if def.memory_base() == Register::RIP => Some(def.memory_displacement64()),
        Mnemonic::Lea if def.memory_base() == Register::None => Some(def.memory_displacement64()),
        Mnemonic::Mov if is_immediate(def.op1_kind()) => Some(def.immediate(1)),
        Mnemonic::Mov if def.op1_kind() == OpKind::Register => {
            resolve_const(instrs, def_idx, def.op1_register(), depth + 1)
        }
        _ => None,
    }
}

/// Find the entry count from the `cmp idx, N` / `ja default` range check Clang
/// emits ahead of the dispatch. Follows register copies backwards, since the
/// index is usually zero-extended (`mov eax, edi`) between the check and the
/// table load.
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
            // The branch guarding the dispatch must immediately follow the
            // compare; anything else and we are looking at an unrelated check.
            let imm = inst.immediate(1);
            return match instrs.get(i + 1)?.mnemonic() {
                // Above / greater: index N is in range, so N+1 entries.
                Mnemonic::Ja | Mnemonic::Jg => imm.checked_add(1),
                // Above-or-equal / greater-or-equal: N entries.
                Mnemonic::Jae | Mnemonic::Jge => Some(imm),
                _ => None,
            };
        }

        if writes_reg(&mut info, inst, tracked) {
            // A copy just renames the index; anything else redefines it and the
            // check (if any) was about a different value.
            if is_reg_copy(inst) {
                tracked = inst.op1_register().full_register();
                continue;
            }
            return None;
        }
    }
    None
}

/// Read `count` (or, unbounded, up to the first implausible) entries and turn
/// them into target addresses. Returns `None` when the table cannot be read at
/// all, or when a proven bound points at memory we do not have.
fn read_targets(
    access: &TableAccess,
    kind: JumpTableKind,
    bound: Option<u64>,
    segments: &[MemorySegment],
    region: &std::ops::Range<u64>,
) -> Option<Vec<u64>> {
    let count = match bound {
        Some(0) => return None,
        Some(n) if n > MAX_ENTRIES => return None,
        Some(n) => n,
        None => MAX_UNBOUNDED_ENTRIES,
    };

    let mut targets: Vec<u64> = Vec::new();
    for i in 0..count {
        let addr = access
            .table
            .checked_add(i.checked_mul(access.entry_size as u64)?)?;
        let raw = match read_ro(segments, addr, access.entry_size) {
            Some(raw) => raw,
            // A proven bound that runs off the end of the segments means the
            // match was wrong somewhere; do not guess at a partial table.
            None if bound.is_some() => return None,
            None => break,
        };
        let target = match kind {
            JumpTableKind::Absolute => raw,
            JumpTableKind::Relative { anchor } => {
                anchor.wrapping_add(sign_extend(raw, access.entry_size) as u64)
            }
        };
        // Unbounded: the walk has no idea where the table ends, so an entry that
        // does not point back into the region being lifted is where it stops.
        if bound.is_none() && !region.contains(&target) {
            break;
        }
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    (!targets.is_empty()).then_some(targets)
}

/// Read `size` bytes at `addr` from the *non-writable* segments only. Writable
/// memory can change between lift and run, so a value read from it proves
/// nothing - the same rule [`crate::vsa`] applies.
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

/// Walk back from `from` for the instruction that last wrote `reg`, stopping at
/// anything that ends straight-line flow. A conditional branch is walked past on
/// purpose: the `ja default` of the range check sits inside the window.
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
