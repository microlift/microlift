//! Guest-agnostic control-flow graph discovery.
//!
//! Phase 1 of [`crate::core::LiftEngine::lift`] already discovers basic-block
//! leaders from static control-flow facts, but that work is private to the lift
//! and tied to IR generation. Static-analysis consumers frequently want the CFG
//! *without* lifting to IR - for rendering, reachability, loop detection, and so
//! on.
//!
//! [`discover_cfg`] exposes that structure directly: it decodes a byte region
//! and partitions it into [`BasicBlock`]s connected by [`Edge`]s, reusing the
//! same [`GuestArch`] decode + [`ControlFlow`](crate::arch::ControlFlow)
//! classification the lifter relies on, so the two stay consistent.
//!
//! This is a *linear-sweep* CFG: every 1-/4-byte position in the region is
//! decoded from `base`. Bytes that are really embedded data will be decoded as
//! instructions; a recursive-descent variant that only follows reachable code
//! is future work.

use crate::arch::{ControlFlow, ControlFlowKind};
use crate::core::GuestArch;
use std::collections::{BTreeMap, BTreeSet};

/// How control reaches a successor block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Execution falls through to the next instruction (no taken transfer, or
    /// the not-taken edge of a conditional branch).
    Fallthrough,
    /// Unconditional direct branch.
    Branch,
    /// Taken edge of a conditional branch.
    CondBranch,
    /// A call to the target (the block itself continues past the call).
    Call,
}

/// A directed edge between two block start addresses.
///
/// `to` may point outside the analyzed region (e.g. a branch or call leaving the
/// region); such targets have no corresponding [`BasicBlock`]. Use
/// [`Cfg::block_at`] to check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub kind: EdgeKind,
}

/// A maximal straight-line run of instructions with a single entry.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Virtual address of the first instruction.
    pub start: u64,
    /// One past the last byte of the block (`== last instruction's next ip`).
    pub end: u64,
    /// Number of instructions in the block.
    pub instruction_count: usize,
    /// Control-flow kind of the block's terminating instruction.
    pub terminator: ControlFlowKind,
    /// Start addresses of statically-known successor blocks (may include
    /// out-of-region targets).
    pub successors: Vec<u64>,
}

impl BasicBlock {
    /// Byte length of the block.
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    /// Whether the block contains no instructions (never true for blocks
    /// produced by [`discover_cfg`]).
    pub fn is_empty(&self) -> bool {
        self.instruction_count == 0
    }
}

/// A control-flow graph over a region of guest code.
#[derive(Debug, Clone)]
pub struct Cfg {
    /// Virtual address the region was decoded at.
    pub base: u64,
    /// One past the last byte of the region.
    pub end: u64,
    /// Basic blocks, sorted by `start`.
    pub blocks: Vec<BasicBlock>,
    /// Edges between blocks, sorted by `(from, to)`.
    pub edges: Vec<Edge>,
}

impl Cfg {
    /// The entry block of the region (the block starting at `base`), if any.
    pub fn entry(&self) -> Option<&BasicBlock> {
        self.block_at(self.base)
    }

    /// The block starting exactly at `addr`, if one exists.
    pub fn block_at(&self, addr: u64) -> Option<&BasicBlock> {
        self.blocks
            .binary_search_by_key(&addr, |b| b.start)
            .ok()
            .map(|i| &self.blocks[i])
    }

    /// Edges leaving the block at `addr`.
    pub fn successors(&self, addr: u64) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.from == addr)
    }

    /// Whether `addr` refers to code inside the analyzed region.
    pub fn in_region(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.end
    }
}

/// Discover the control-flow graph of a region of raw machine code.
///
/// `base` is the virtual address the first byte maps to. The result partitions
/// the decoded instructions into basic blocks and records the static edges
/// between them.
pub fn discover_cfg<A: GuestArch>(bytes: &[u8], base: u64) -> Cfg {
    let end = base + bytes.len() as u64;
    let instructions = A::decode_region(bytes, base);

    // Index instructions by address and precompute their control-flow facts.
    let mut order: Vec<u64> = Vec::with_capacity(instructions.len());
    let mut cf_by_ip: BTreeMap<u64, (u64, ControlFlow)> = BTreeMap::new();
    for instr in &instructions {
        let ip = A::instr_ip(instr);
        let next = A::instr_next_ip(instr);
        order.push(ip);
        cf_by_ip.insert(ip, (next, A::control_flow(instr)));
    }

    // Phase 1: leader discovery. A leader begins a basic block.
    let mut leaders: BTreeSet<u64> = BTreeSet::new();
    if !order.is_empty() {
        leaders.insert(base);
    }
    for &ip in &order {
        let (next, cf) = &cf_by_ip[&ip];
        match cf.kind {
            ControlFlowKind::DirectBranch | ControlFlowKind::CondBranch => {
                for &t in &cf.static_targets {
                    if t >= base && t < end {
                        leaders.insert(t);
                    }
                }
                // The instruction after any branch starts a new block.
                leaders.insert(*next);
            }
            ControlFlowKind::Call { static_target } => {
                // A call does not end a block (control returns), but an
                // in-region callee is itself a block leader.
                if let Some(t) = static_target {
                    if t >= base && t < end {
                        leaders.insert(t);
                    }
                }
            }
            ControlFlowKind::Ret
            | ControlFlowKind::IndirectBranch
            | ControlFlowKind::Invalid => {
                leaders.insert(*next);
            }
            ControlFlowKind::Fallthrough => {}
        }
    }

    // Phase 2: carve blocks and collect edges by walking instructions in order.
    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    let mut i = 0usize;
    while i < order.len() {
        let start = order[i];
        let mut count = 0usize;
        // Both are assigned on every iteration of the loop below (which always
        // runs at least once) before any `break`.
        let mut block_end;
        let mut terminator;
        let mut successors: Vec<u64> = Vec::new();

        loop {
            let ip = order[i];
            let (next, cf) = &cf_by_ip[&ip];
            count += 1;
            block_end = *next;

            let ends_block = match cf.kind {
                ControlFlowKind::DirectBranch => {
                    for &t in &cf.static_targets {
                        successors.push(t);
                        edges.push(Edge { from: start, to: t, kind: EdgeKind::Branch });
                    }
                    true
                }
                ControlFlowKind::CondBranch => {
                    for &t in &cf.static_targets {
                        successors.push(t);
                        edges.push(Edge { from: start, to: t, kind: EdgeKind::CondBranch });
                    }
                    successors.push(*next);
                    edges.push(Edge { from: start, to: *next, kind: EdgeKind::Fallthrough });
                    true
                }
                ControlFlowKind::Ret | ControlFlowKind::IndirectBranch => true,
                ControlFlowKind::Invalid => true,
                ControlFlowKind::Call { static_target } => {
                    if let Some(t) = static_target {
                        edges.push(Edge { from: start, to: t, kind: EdgeKind::Call });
                    }
                    // The block continues unless the *next* instruction is a
                    // leader (e.g. it is the call's in-region callee).
                    false
                }
                ControlFlowKind::Fallthrough => false,
            };

            terminator = cf.kind;

            // Stop if this instruction ends the block, it is the last one, or
            // the following instruction begins a new block.
            let next_is_leader = leaders.contains(next);
            i += 1;
            if ends_block || i >= order.len() || next_is_leader {
                // If the block fell through into a new leader, add a
                // fallthrough edge (unless the terminator already transferred
                // control unconditionally).
                let unconditional = matches!(
                    terminator,
                    ControlFlowKind::DirectBranch
                        | ControlFlowKind::Ret
                        | ControlFlowKind::IndirectBranch
                        | ControlFlowKind::Invalid
                );
                if !ends_block && next_is_leader && !unconditional {
                    successors.push(*next);
                    edges.push(Edge { from: start, to: *next, kind: EdgeKind::Fallthrough });
                }
                break;
            }
        }

        blocks.push(BasicBlock {
            start,
            end: block_end,
            instruction_count: count,
            terminator,
            successors,
        });
    }

    blocks.sort_by_key(|b| b.start);
    edges.sort_by_key(|e| (e.from, e.to, e.kind as u8));
    edges.dedup();

    Cfg { base, end, blocks, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::X86_64;

    #[test]
    fn single_block_mov_ret() {
        // mov rax, 42 ; ret
        let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
        let cfg = discover_cfg::<X86_64>(&bytes, 0x1000);
        assert_eq!(cfg.blocks.len(), 1);
        let b = &cfg.blocks[0];
        assert_eq!(b.start, 0x1000);
        assert_eq!(b.end, 0x1008);
        assert_eq!(b.instruction_count, 2);
        assert!(matches!(b.terminator, ControlFlowKind::Ret));
        assert!(cfg.edges.is_empty());
    }

    #[test]
    fn conditional_branch_splits_blocks() {
        // xor eax,eax; cmp eax,0; je +6; mov eax,0; ret
        // 31 C0            @1000
        // 83 F8 00         @1002
        // 74 06            @1005  -> taken 0x100D (out of region: region ends 0x100D)
        // B8 00 00 00 00   @1007
        // C3               @100C
        let bytes = [
            0x31, 0xC0, 0x83, 0xF8, 0x00, 0x74, 0x06, 0xB8, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        let cfg = discover_cfg::<X86_64>(&bytes, 0x1000);

        // Entry block ends at the conditional branch (0x1007).
        let entry = cfg.entry().unwrap();
        assert_eq!(entry.start, 0x1000);
        assert_eq!(entry.end, 0x1007);
        assert!(matches!(entry.terminator, ControlFlowKind::CondBranch));

        // The cond branch has a taken edge to 0x100D and a fallthrough to 0x1007.
        let kinds: Vec<EdgeKind> = cfg.successors(0x1000).map(|e| e.kind).collect();
        assert!(kinds.contains(&EdgeKind::CondBranch));
        assert!(kinds.contains(&EdgeKind::Fallthrough));

        // Fallthrough block 0x1007..0x100D ends in ret.
        let ft = cfg.block_at(0x1007).unwrap();
        assert!(matches!(ft.terminator, ControlFlowKind::Ret));
    }
}
