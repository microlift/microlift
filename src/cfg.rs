//! Guest-agnostic CFG discovery over raw code, using the same decode and classification as the lifter.
//!
//! Linear sweep: embedded data decodes as instructions; [`decode_reachable`] avoids that.

use crate::arch::{ControlFlow, ControlFlowKind};
use crate::context::MemorySegment;
use crate::core::GuestArch;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeKind {
    /// Fallthrough, or the not-taken edge of a conditional branch.
    Fallthrough,
    Branch,
    CondBranch,
    /// Call to the target; the block continues past it.
    Call,
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EdgeKind::Fallthrough => "fallthrough",
            EdgeKind::Branch => "branch",
            EdgeKind::CondBranch => "cond-branch",
            EdgeKind::Call => "call",
        })
    }
}

/// Edge between block start addresses; `to` may be outside the region (no [`BasicBlock`], see [`Cfg::block_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BasicBlock {
    pub start: u64,
    /// One past the last byte.
    pub end: u64,
    pub instruction_count: usize,
    /// Kind of the terminating instruction.
    pub terminator: ControlFlowKind,
    /// Static successor starts, possibly out of region.
    pub successors: Vec<u64>,
}

impl BasicBlock {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    /// Never true for blocks from [`discover_cfg`].
    pub fn is_empty(&self) -> bool {
        self.instruction_count == 0
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cfg {
    pub base: u64,
    pub end: u64,
    pub blocks: Vec<BasicBlock>,
    pub edges: Vec<Edge>,
}

impl Cfg {
    pub fn entry(&self) -> Option<&BasicBlock> {
        self.block_at(self.base)
    }

    pub fn block_at(&self, addr: u64) -> Option<&BasicBlock> {
        self.blocks
            .binary_search_by_key(&addr, |b| b.start)
            .ok()
            .map(|i| &self.blocks[i])
    }

    pub fn successors(&self, addr: u64) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.from == addr)
    }

    pub fn in_region(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.end
    }
}

/// Discovers the CFG of code whose first byte is at `base`.
pub fn discover_cfg<A: GuestArch>(bytes: &[u8], base: u64) -> Cfg {
    let end = base + bytes.len() as u64;
    let instructions = A::decode_region(bytes, base);

    let mut order: Vec<u64> = Vec::with_capacity(instructions.len());
    let mut cf_by_ip: BTreeMap<u64, (u64, ControlFlow)> = BTreeMap::new();
    for instr in &instructions {
        let ip = A::instr_ip(instr);
        let next = A::instr_next_ip(instr);
        order.push(ip);
        cf_by_ip.insert(ip, (next, A::control_flow(instr)));
    }

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
                leaders.insert(*next);
            }
            ControlFlowKind::Call { static_target } => {
                // A call does not end a block, but an in-region callee is a leader.
                if let Some(t) = static_target {
                    if t >= base && t < end {
                        leaders.insert(t);
                    }
                }
            }
            ControlFlowKind::Ret | ControlFlowKind::IndirectBranch | ControlFlowKind::Invalid => {
                leaders.insert(*next);
            }
            ControlFlowKind::Fallthrough => {}
        }
    }

    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    let mut i = 0usize;
    while i < order.len() {
        let start = order[i];
        let mut count = 0usize;
        // Assigned on every iteration before any `break`.
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
                        edges.push(Edge {
                            from: start,
                            to: t,
                            kind: EdgeKind::Branch,
                        });
                    }
                    true
                }
                ControlFlowKind::CondBranch => {
                    for &t in &cf.static_targets {
                        successors.push(t);
                        edges.push(Edge {
                            from: start,
                            to: t,
                            kind: EdgeKind::CondBranch,
                        });
                    }
                    successors.push(*next);
                    edges.push(Edge {
                        from: start,
                        to: *next,
                        kind: EdgeKind::Fallthrough,
                    });
                    true
                }
                ControlFlowKind::Ret | ControlFlowKind::IndirectBranch => true,
                ControlFlowKind::Invalid => true,
                ControlFlowKind::Call { static_target } => {
                    if let Some(t) = static_target {
                        edges.push(Edge {
                            from: start,
                            to: t,
                            kind: EdgeKind::Call,
                        });
                    }
                    // Continues unless the next instruction is a leader.
                    false
                }
                ControlFlowKind::Fallthrough => false,
            };

            terminator = cf.kind;

            // Stop at a block end, the last instruction, or a following leader.
            let next_is_leader = leaders.contains(next);
            i += 1;
            if ends_block || i >= order.len() || next_is_leader {
                // Fell into a new leader: add a fallthrough edge unless already transferred unconditionally.
                let unconditional = matches!(
                    terminator,
                    ControlFlowKind::DirectBranch
                        | ControlFlowKind::Ret
                        | ControlFlowKind::IndirectBranch
                        | ControlFlowKind::Invalid
                );
                if !ends_block && next_is_leader && !unconditional {
                    successors.push(*next);
                    edges.push(Edge {
                        from: start,
                        to: *next,
                        kind: EdgeKind::Fallthrough,
                    });
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

    Cfg {
        base,
        end,
        blocks,
        edges,
    }
}

/// Recursive-descent decode from `entries` (`&[base]` for the region entry): follows only static
/// control flow, so embedded data is never decoded, but code reached only indirectly is missed.
/// Each address is decoded once; invalid encodings are kept but not followed.
pub fn decode_reachable<A: GuestArch>(bytes: &[u8], base: u64, entries: &[u64]) -> Vec<A::Instr> {
    let end = base + bytes.len() as u64;
    let mut decoded: BTreeMap<u64, A::Instr> = BTreeMap::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    let mut work: Vec<u64> = entries.to_vec();

    while let Some(addr) = work.pop() {
        if addr < base || addr >= end {
            continue;
        }
        if !seen.insert(addr) {
            continue;
        }
        let off = (addr - base) as usize;
        let Some(inst) = A::decode_one(&bytes[off..], addr) else {
            continue;
        };

        if A::is_invalid(&inst) {
            decoded.insert(addr, inst);
            continue;
        }

        let next = A::instr_next_ip(&inst);
        let cf = A::control_flow(&inst);
        decoded.insert(addr, inst);

        match cf.kind {
            ControlFlowKind::Fallthrough => work.push(next),
            ControlFlowKind::CondBranch => {
                work.extend(cf.static_targets.iter().copied());
                work.push(next);
            }
            ControlFlowKind::DirectBranch => {
                work.extend(cf.static_targets.iter().copied());
            }
            ControlFlowKind::Call { static_target } => {
                if let Some(t) = static_target {
                    work.push(t);
                }
                work.push(next);
            }
            ControlFlowKind::Ret | ControlFlowKind::IndirectBranch | ControlFlowKind::Invalid => {}
        }
    }

    decoded.into_values().collect()
}

/// Statically-resolved call and branch destinations; may point outside the scanned region.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Targets {
    /// Direct `CALL` destinations (likely function starts), sorted and de-duplicated.
    pub calls: BTreeSet<u64>,
    /// Direct branch destinations, sorted and de-duplicated.
    pub branches: BTreeSet<u64>,
}

/// Collects direct call and branch destinations by linear sweep; indirect ones need [`crate::vsa`].
pub fn static_targets<A: GuestArch>(bytes: &[u8], base: u64) -> Targets {
    let mut targets = Targets::default();
    for instr in A::decode_region(bytes, base) {
        let cf = A::control_flow(&instr);
        match cf.kind {
            ControlFlowKind::Call {
                static_target: Some(t),
            } => {
                targets.calls.insert(t);
            }
            ControlFlowKind::DirectBranch | ControlFlowKind::CondBranch => {
                targets.branches.extend(cf.static_targets.iter().copied());
            }
            _ => {}
        }
    }
    targets
}

/// Indirect targets recovered by value-set analysis ([`crate::vsa`]), keyed by the `JMP`/`CALL` address.
/// Empty for architectures without a value-set analyzer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedTargets {
    pub jumps: BTreeMap<u64, Vec<u64>>,
    /// Resolved indirect-`CALL` destinations per call IP.
    pub calls: BTreeMap<u64, Vec<u64>>,
}

impl ResolvedTargets {
    pub fn is_empty(&self) -> bool {
        self.jumps.is_empty() && self.calls.is_empty()
    }

    /// Flattens into the [`Targets`] shape of [`static_targets`].
    pub fn flatten(&self) -> Targets {
        Targets {
            calls: self.calls.values().flatten().copied().collect(),
            branches: self.jumps.values().flatten().copied().collect(),
        }
    }
}

/// Resolves indirect `JMP`/`CALL` targets via value-set analysis. `segments` adds read-only memory
/// (e.g. `.rodata`) beyond the region itself; pass `&[]` for the region alone.
pub fn resolved_indirect_targets<A: GuestArch>(
    bytes: &[u8],
    base: u64,
    segments: &[MemorySegment],
) -> ResolvedTargets {
    A::resolved_indirect_targets(bytes, base, segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::X86_64;

    #[test]
    fn single_block_mov_ret() {
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
    fn recursive_descent_skips_embedded_data() {
        // Jumps over 2 data bytes at 0x1002.
        let bytes = [0xEB, 0x02, 0xFF, 0xFF, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];

        let reachable = decode_reachable::<X86_64>(&bytes, 0x1000, &[0x1000]);
        let addrs: Vec<u64> = reachable.iter().map(X86_64::instr_ip).collect();
        assert_eq!(addrs, vec![0x1000, 0x1004, 0x1009]);

        let linear = X86_64::decode_region(&bytes, 0x1000);
        assert!(linear.len() > reachable.len());
    }

    #[test]
    fn conditional_branch_splits_blocks() {
        // xor eax,eax; cmp eax,0; je +6 (out of region, 0x100D); mov eax,0; ret
        let bytes = [
            0x31, 0xC0, 0x83, 0xF8, 0x00, 0x74, 0x06, 0xB8, 0x00, 0x00, 0x00, 0x00, 0xC3,
        ];
        let cfg = discover_cfg::<X86_64>(&bytes, 0x1000);

        let entry = cfg.entry().unwrap();
        assert_eq!(entry.start, 0x1000);
        assert_eq!(entry.end, 0x1007);
        assert!(matches!(entry.terminator, ControlFlowKind::CondBranch));

        let kinds: Vec<EdgeKind> = cfg.successors(0x1000).map(|e| e.kind).collect();
        assert!(kinds.contains(&EdgeKind::CondBranch));
        assert!(kinds.contains(&EdgeKind::Fallthrough));

        let ft = cfg.block_at(0x1007).unwrap();
        assert!(matches!(ft.terminator, ControlFlowKind::Ret));
    }

    #[test]
    fn static_targets_harvests_call_and_branch_destinations() {
        // call 0x100A; jmp 0x1009; int3 x2; ret; ret
        let bytes = [
            0xE8, 0x05, 0x00, 0x00, 0x00, 0xEB, 0x02, 0xCC, 0xCC, 0xC3, 0xC3,
        ];
        let t = static_targets::<X86_64>(&bytes, 0x1000);
        assert!(t.calls.contains(&0x100A));
        assert!(t.branches.contains(&0x1009));
        assert_eq!(t.calls.len(), 1);
    }

    #[test]
    fn static_targets_ignores_indirect_calls() {
        let bytes = [0xFF, 0xD0, 0xC3];
        let t = static_targets::<X86_64>(&bytes, 0x1000);
        assert!(t.calls.is_empty());
        assert!(t.branches.is_empty());
    }
}
