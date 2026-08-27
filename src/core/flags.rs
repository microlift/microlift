//! Which condition-flag writes a region actually needs.
//!
//! Guest flags are written eagerly and the dead ones deleted by the compiler,
//! which is correct but expensive: the IR is emitted, optimised and
//! register-allocated before anything decides it was never needed. This answers
//! the same question one stage earlier, on the guest instruction stream. Ported
//! from FEX's `RedundantFlagCalculationElimination`.
//!
//! Standard backwards liveness; a flag not in `live_out(a)` may be left
//! uncomputed at `a`:
//!
//! ```text
//! live_in(a)  = reads(a) ∪ (live_out(a) − writes(a))
//! live_out(a) = ⋃ live_in(s) for s in successors(a)
//! ```
//!
//! Being a *may* analysis, a wrong "dead" is a wrong answer with no diagnostic.
//! Three rules answer [`FlagSet::ALL`]: successors not all known (a call, return,
//! indirect branch, truncated encoding, or edge leaving the region - each of
//! which lifts to a `sync_to_context` that publishes every flag); an instruction
//! the frontend does not classify ("reads everything, writes nothing", which is
//! what makes the interpreter-fallback callback safe without the analysis
//! knowing which instructions take it); and an address never analysed.

use std::collections::{HashMap, HashSet};

use crate::arch::{ControlFlow, ControlFlowKind};
use crate::core::GuestArch;

/// A set of guest condition flags, as a bitset in the guest's own numbering.
///
/// The engine never learns what the bits mean: a frontend assigns each flag an
/// index and uses it consistently in [`FlagEffects`] and when querying
/// [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live). Eight bits
/// covers every guest here (x86-64 uses seven, aarch64 four).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FlagSet(u8);

impl FlagSet {
    /// No flags.
    pub const NONE: FlagSet = FlagSet(0);
    /// Every flag, including bits no guest has assigned a meaning to.
    ///
    /// The unassigned bits are harmless, and "all bits" rather than "all bits this
    /// guest uses" is what lets the engine stay ignorant of the numbering.
    pub const ALL: FlagSet = FlagSet(u8::MAX);

    /// The set containing just `bit`.
    pub const fn bit(bit: u8) -> FlagSet {
        FlagSet(1 << bit)
    }

    /// Whether `bit` is in the set.
    pub const fn contains(self, bit: u8) -> bool {
        self.0 & (1 << bit) != 0
    }

    /// Set union.
    pub const fn union(self, other: FlagSet) -> FlagSet {
        FlagSet(self.0 | other.0)
    }

    /// Everything in `self` that is not in `other`.
    pub const fn without(self, other: FlagSet) -> FlagSet {
        FlagSet(self.0 & !other.0)
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// What one guest instruction does to the flags, **as its lift implements it**.
///
/// The contract is about the Rust handler, not the architecture, and the two
/// directions are asymmetric:
///
/// - [`writes`](Self::writes) must be a **subset** of what the handler assigns
///   on every path. Claiming a write it does not perform lets the analysis kill
///   an earlier write something still reads. x86 leaves AF *undefined* after a
///   logical op where microlift leaves it *unchanged*, so `AND` must not claim
///   to write AF.
/// - [`reads`](Self::reads) must be a **superset** of what the handler
///   consumes. x86 says `SHL` reads no flags, but a shift by a masked count of
///   zero must leave them alone, so the handler loads all five to select on.
///
/// The default is the safe pair - reads everything, writes nothing - so an
/// unaudited instruction costs optimisation and not correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagEffects {
    /// Flags the lift of this instruction consumes.
    pub reads: FlagSet,
    /// Flags the lift of this instruction definitely assigns.
    pub writes: FlagSet,
}

impl FlagEffects {
    /// Reads everything, writes nothing: the answer that is never wrong.
    pub const CONSERVATIVE: FlagEffects = FlagEffects {
        reads: FlagSet::ALL,
        writes: FlagSet::NONE,
    };

    /// Touches no flags at all - a `MOV`, a `LEA`, a `PUSH`.
    pub const TRANSPARENT: FlagEffects = FlagEffects {
        reads: FlagSet::NONE,
        writes: FlagSet::NONE,
    };

    /// Writes `writes` and reads nothing.
    pub const fn writing(writes: FlagSet) -> FlagEffects {
        FlagEffects {
            reads: FlagSet::NONE,
            writes,
        }
    }

    /// Reads `reads` and writes nothing - a `Jcc`, a `SETcc`, a `CMOVcc`.
    pub const fn reading(reads: FlagSet) -> FlagEffects {
        FlagEffects {
            reads,
            writes: FlagSet::NONE,
        }
    }
}

impl Default for FlagEffects {
    fn default() -> Self {
        FlagEffects::CONSERVATIVE
    }
}

/// Per-address flag liveness for one lifted region, built by [`analyze`]
/// between decode and IR emission and consulted through
/// [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live).
#[derive(Debug, Clone, Default)]
pub struct FlagLiveness {
    live_out: HashMap<u64, FlagSet>,
}

impl FlagLiveness {
    /// The flags live *after* the instruction at `va` - the ones its lift has to
    /// compute. [`FlagSet::ALL`] for an address with no entry, which is the answer
    /// that makes a gap cost nothing but speed.
    pub fn live_out(&self, va: u64) -> FlagSet {
        self.live_out.get(&va).copied().unwrap_or(FlagSet::ALL)
    }

    /// How many addresses have an entry. Diagnostics and tests.
    pub fn len(&self) -> usize {
        self.live_out.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live_out.is_empty()
    }
}

/// Where control can go from one instruction, when that is knowable.
///
/// `None` is the conservative answer, covering a call, a return, an unresolved
/// indirect branch, an invalid encoding, and any edge leaving the region.
fn successors<A: GuestArch>(
    addr: u64,
    inst: &A::Instr,
    cf: &ControlFlow,
    truncated: bool,
    in_region: &impl Fn(u64) -> bool,
    decoded: &HashMap<u64, A::Instr>,
) -> Option<Vec<u64>> {
    if truncated {
        return None;
    }
    let next = A::instr_next_ip(inst);
    let targets: Vec<u64> = match cf.kind {
        ControlFlowKind::Fallthrough => vec![next],
        ControlFlowKind::DirectBranch => cf.static_targets.clone(),
        ControlFlowKind::CondBranch => {
            let mut v = cf.static_targets.clone();
            v.push(next);
            v
        }
        // Every one of these is serviced through a context sync - an inlined fast
        // import, a bridge, a VM exit, or an in-region callee returning through
        // the dispatch.
        ControlFlowKind::Call { .. }
        | ControlFlowKind::Ret
        | ControlFlowKind::IndirectBranch
        | ControlFlowKind::Invalid => return None,
    };
    // An edge out of the region is an exit, and an edge to an address phase 1
    // never decoded is one this analysis cannot follow. Both mean the flags at
    // that point belong to somebody else.
    let _ = addr;
    if targets
        .iter()
        .any(|t| !in_region(*t) || !decoded.contains_key(t))
    {
        return None;
    }
    Some(targets)
}

/// Compute per-address flag liveness for a decoded region.
///
/// `truncated` is the engine's own `runs_off_region`, passed in rather than
/// recomputed so the two cannot disagree about which instructions are only
/// partly present.
pub(crate) fn analyze<A: GuestArch>(
    decoded: &HashMap<u64, A::Instr>,
    in_region: &impl Fn(u64) -> bool,
    truncated: impl Fn(u64, &A::Instr) -> bool,
) -> FlagLiveness {
    // Successors and effects, resolved once. Both are pure functions of the
    // decode, and the fixpoint below revisits an address once per predecessor
    // update.
    let mut succs: HashMap<u64, Option<Vec<u64>>> = HashMap::with_capacity(decoded.len());
    let mut effects: HashMap<u64, FlagEffects> = HashMap::with_capacity(decoded.len());
    let mut preds: HashMap<u64, Vec<u64>> = HashMap::new();

    for (&addr, inst) in decoded {
        if !in_region(addr) {
            continue;
        }
        let cf = A::control_flow(inst);
        let s = successors::<A>(addr, inst, &cf, truncated(addr, inst), in_region, decoded);
        if let Some(targets) = &s {
            for &t in targets {
                preds.entry(t).or_default().push(addr);
            }
        }
        succs.insert(addr, s);
        effects.insert(
            addr,
            A::flag_effects(inst).unwrap_or(FlagEffects::CONSERVATIVE),
        );
    }

    // `live_in` grows monotonically from the empty set, so the worklist
    // converges on the least fixpoint - the standard formulation, and the one
    // that reports a flag live only when some path really reads it.
    let mut live_in: HashMap<u64, FlagSet> = HashMap::with_capacity(succs.len());
    let mut work: Vec<u64> = succs.keys().copied().collect();
    let mut queued: HashSet<u64> = work.iter().copied().collect();

    let out_of =
        |addr: u64, live_in: &HashMap<u64, FlagSet>, succs: &HashMap<u64, Option<Vec<u64>>>| {
            match succs.get(&addr) {
                Some(Some(targets)) => targets.iter().fold(FlagSet::NONE, |acc, t| {
                    acc.union(live_in.get(t).copied().unwrap_or(FlagSet::NONE))
                }),
                // Unknown successors, or an address that is not ours.
                _ => FlagSet::ALL,
            }
        };

    while let Some(addr) = work.pop() {
        queued.remove(&addr);
        let out = out_of(addr, &live_in, &succs);
        let eff = effects[&addr];
        let new_in = eff.reads.union(out.without(eff.writes));
        let old_in = live_in.get(&addr).copied().unwrap_or(FlagSet::NONE);
        if new_in != old_in {
            live_in.insert(addr, new_in);
            for &p in preds.get(&addr).into_iter().flatten() {
                if queued.insert(p) {
                    work.push(p);
                }
            }
        }
    }

    let live_out = succs
        .keys()
        .map(|&addr| (addr, out_of(addr, &live_in, &succs)))
        .collect();
    FlagLiveness { live_out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_is_a_set() {
        let cf = FlagSet::bit(0);
        let zf = FlagSet::bit(3);
        assert!(cf.union(zf).contains(0));
        assert!(cf.union(zf).contains(3));
        assert!(!cf.union(zf).contains(1));
        assert_eq!(cf.union(zf).without(cf), zf);
        assert!(FlagSet::NONE.is_empty());
        assert!(!FlagSet::ALL.is_empty());
        for bit in 0..8 {
            assert!(FlagSet::ALL.contains(bit));
            assert!(!FlagSet::NONE.contains(bit));
        }
    }

    /// The two defaults are the two halves of "wrong costs speed, not answers".
    #[test]
    fn the_defaults_are_the_conservative_ones() {
        assert_eq!(FlagEffects::default(), FlagEffects::CONSERVATIVE);
        assert_eq!(FlagEffects::CONSERVATIVE.reads, FlagSet::ALL);
        assert_eq!(FlagEffects::CONSERVATIVE.writes, FlagSet::NONE);
        assert_eq!(FlagLiveness::default().live_out(0x1000), FlagSet::ALL);
    }
}
