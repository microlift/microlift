//! Which condition-flag writes a region actually needs.
//!
//! ## Why this exists
//!
//! Guest flags are SSA `Variable`s written eagerly: an `ADD` emits the IR for
//! all six of CF/PF/AF/ZF/SF/OF whether or not anything reads them, and the
//! parity flag alone is a `popcnt`, an `ireduce`, a `band` and a `bxor`. Being
//! SSA, the dead ones are deleted by the compiler - that is the design note in
//! `core/mod.rs`, and it is true. It is also the *expensive* way to be right:
//! the IR is emitted, walked, optimised and register-allocated before anything
//! decides it was never needed, and a backend's register allocator is
//! superlinear in the live ranges it is handed.
//!
//! So this answers the same question one stage earlier, on the guest's own
//! instruction stream, where a flag write costs one bit instead of a dozen IR
//! instructions. It is a port of FEX's `RedundantFlagCalculationElimination`
//! (`FEXCore/Source/Interface/IR/Passes/`), with one structural difference
//! forced by where microlift keeps its flags - see [the classification
//! contract](FlagEffects).
//!
//! ## What it computes
//!
//! Standard backwards liveness over guest instructions. For each instruction
//! address, `live_out` is the set of flags some later instruction reads before
//! anything overwrites it:
//!
//! ```text
//! live_in(a)  = reads(a) ∪ (live_out(a) − writes(a))
//! live_out(a) = ⋃ live_in(s) for s in successors(a)
//! ```
//!
//! A flag not in `live_out(a)` is one the lift may decline to compute at `a`.
//!
//! ## Where it refuses to answer
//!
//! Everything here is a *may* analysis, so being wrong in the "dead" direction
//! produces a program that computes the wrong answer with no diagnostic. Three
//! rules keep that from happening, and each of them says [`FlagSet::ALL`]:
//!
//! - **An instruction whose successors are not all known.** A call, a return,
//!   an indirect branch, a truncated encoding, or any edge leaving the lifted
//!   region. This is the rule that covers the host: every one of those lifts to
//!   a `sync_to_context`, which publishes all flags, so all flags are live
//!   there by construction rather than by anyone remembering to say so.
//! - **An instruction the guest frontend does not classify.** The default is
//!   "reads everything, writes nothing", which keeps every earlier write live.
//!   This is what makes an instruction microlift lifts to an
//!   `UnimplementedInstruction` exit safe without the analysis knowing which
//!   instructions those are: an unclassified mnemonic is conservative, and only
//!   a mnemonic somebody audited is not.
//! - **An address that was never analysed** - one phase 2 decoded on demand
//!   that phase 1 never reached. [`FlagLiveness::live_out`] answers `ALL` for
//!   anything it has no entry for, so the failure mode of a gap is lost
//!   optimisation rather than lost correctness.
//!
//! Note the second rule is where this differs from FEX. FEX classifies its own
//! IR opcodes and an unrecognised one is *trivial* - no flag effects at all -
//! which is safe there because flags in FEX IR are explicit values that only
//! flag-bearing opcodes can touch. Here the unit of classification is the
//! **guest instruction**, whose flag effects are whatever its Rust handler
//! does, so an unrecognised one has to mean the opposite.

use std::collections::{HashMap, HashSet};

use crate::arch::{ControlFlow, ControlFlowKind};
use crate::core::GuestArch;

/// A set of guest condition flags, as a bitset in the guest's own numbering.
///
/// The engine never learns what the bits mean: a frontend assigns each of its
/// flags an index, uses that index consistently in [`FlagEffects`] and when
/// querying [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live), and
/// the dataflow here is just set union and difference. Eight bits is enough for
/// every guest microlift has (x86-64 uses seven, aarch64 four).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FlagSet(u8);

impl FlagSet {
    /// No flags.
    pub const NONE: FlagSet = FlagSet(0);
    /// Every flag, including bits no guest has assigned a meaning to.
    ///
    /// The unassigned bits are harmless - nothing ever queries them - and
    /// having `ALL` be "all bits" rather than "all bits this guest uses" is
    /// what lets the engine stay ignorant of the numbering.
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
/// This is the contract, and it is about the Rust handler rather than about the
/// architecture, because the handler is what actually runs. Two directions, and
/// they are asymmetric:
///
/// - [`writes`](Self::writes) must be a **subset** of the flags the handler
///   assigns on every path. Claiming a write the handler does not perform lets
///   the analysis kill an earlier write that something still reads. The
///   architecture is not the authority here: x86 leaves AF *undefined* after a
///   logical op and microlift leaves it *unchanged*, so a later reader observes
///   the older value and `AND` must not claim to write AF.
/// - [`reads`](Self::reads) must be a **superset** of the flags the handler
///   consumes. The shift handlers are the reason this is spelled out: x86 says
///   `SHL` reads no flags, but a shift by a masked count of zero must leave the
///   flags alone, so the handler loads all five of them to select against.
///
/// The default is the safe pair - reads everything, writes nothing - so an
/// instruction nobody has audited costs optimisation and not correctness.
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

/// Per-address flag liveness for one lifted region.
///
/// Built by [`analyze`] between decode and IR emission, and consulted by the
/// semantics through
/// [`LiftEngine::flag_live`](crate::core::LiftEngine::flag_live).
#[derive(Debug, Clone, Default)]
pub struct FlagLiveness {
    live_out: HashMap<u64, FlagSet>,
}

impl FlagLiveness {
    /// The flags live *after* the instruction at `va` - the ones its lift has
    /// to compute.
    ///
    /// [`FlagSet::ALL`] for an address the analysis has no entry for, which is
    /// the answer that makes a gap cost nothing but speed.
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
/// `None` is the conservative answer and covers everything whose successors the
/// analysis cannot enumerate: a call (which may run anything and comes back
/// through a context sync), a return, an unresolved indirect branch, an invalid
/// encoding, and any edge leaving the region.
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
        // A call is serviced through a context sync whatever kind it is -
        // inlined fast import, bridge, VM exit, or an in-region callee that
        // returns through the dispatch. A return and an indirect branch leave
        // through one too.
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
/// `truncated` reports whether an instruction's encoding runs past the end of
/// the region - the engine's own `runs_off_region`, passed in rather than
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
