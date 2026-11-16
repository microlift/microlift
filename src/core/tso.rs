//! x86 total store order on a weakly ordered host, by fencing guest memory accesses.
//!
//! x86 lets a store be passed by a later load and nothing else. AArch64 reorders every
//! pair, so lock-free guest code that publishes with a plain `MOV` (Mono's write barrier
//! is a compiler barrier, Unity's job queues hand nodes over with plain stores) sees
//! another thread's writes out of order. A fence goes before a guest access whenever an
//! earlier one since the last fence forms a pair x86 keeps ordered: load-load,
//! load-store or store-store. Store-load is the one TSO does not promise, so a load
//! after only stores gets none.
//!
//! Only accesses in the guest-memory alias region count; the register context and the
//! host's own tables are this thread's alone. A block starts with whatever its
//! predecessors leave pending, so a fence is not repeated across a branch. The entry
//! and every call are treated as having both kinds pending, since what ran before them
//! is not visible here.

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{AliasRegion, Block, Function, Inst, InstBuilder, Opcode};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Pending {
    load: bool,
    store: bool,
}

impl Pending {
    const UNKNOWN: Self = Self {
        load: true,
        store: true,
    };

    fn join(self, other: Self) -> Self {
        Self {
            load: self.load || other.load,
            store: self.store || other.store,
        }
    }
}

/// What one instruction means for ordering.
enum Effect {
    Unrelated,
    Ordered,
    Unknown,
    Access { load: bool, store: bool },
}

fn effect(func: &Function, inst: Inst, guest: AliasRegion) -> Effect {
    let data = &func.dfg.insts[inst];
    let opcode = data.opcode();
    if opcode.is_call() {
        return Effect::Unknown;
    }
    if opcode == Opcode::Fence {
        return Effect::Ordered;
    }
    let (load, store) = (opcode.can_load(), opcode.can_store());
    if !(load || store) || data.alias_region(&func.dfg) != Some(guest) {
        return Effect::Unrelated;
    }
    // Atomics are already sequentially consistent on every host Cranelift has.
    if load && store {
        return Effect::Ordered;
    }
    Effect::Access { load, store }
}

/// Advance `pending` past one instruction, saying whether a fence must go before it.
fn step(pending: &mut Pending, effect: Effect) -> bool {
    match effect {
        Effect::Unrelated => false,
        Effect::Ordered => {
            *pending = Pending::default();
            false
        }
        Effect::Unknown => {
            *pending = Pending::UNKNOWN;
            false
        }
        Effect::Access { load, store } => {
            let needs = match store {
                true => pending.load || pending.store,
                false => pending.load,
            };
            if needs {
                *pending = Pending::default();
            }
            pending.load |= load;
            pending.store |= store;
            needs
        }
    }
}

fn exit_state(func: &Function, block: Block, entry: Pending, guest: AliasRegion) -> Pending {
    let mut pending = entry;
    for inst in func.layout.block_insts(block) {
        step(&mut pending, effect(func, inst, guest));
    }
    pending
}

/// What is pending on entry to each block, joined over its predecessors to a fixed point.
fn entry_states(
    func: &Function,
    cfg: &ControlFlowGraph,
    guest: AliasRegion,
) -> SecondaryMap<Block, Pending> {
    let blocks: Vec<Block> = func.layout.blocks().collect();
    let mut entry = SecondaryMap::with_default(Pending::default());
    for &block in &blocks {
        if Some(block) == func.layout.entry_block() || cfg.pred_iter(block).next().is_none() {
            entry[block] = Pending::UNKNOWN;
        }
    }
    let mut exit = SecondaryMap::with_default(Pending::default());
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &blocks {
            let joined = cfg
                .pred_iter(block)
                .fold(entry[block], |acc, pred| acc.join(exit[pred.block]));
            entry[block] = joined;
            let out = exit_state(func, block, joined, guest);
            if out != exit[block] {
                exit[block] = out;
                changed = true;
            }
        }
    }
    entry
}

/// Insert the fences TSO needs around every access to `guest` in `func`.
pub fn insert_fences(func: &mut Function, guest: Option<AliasRegion>) {
    let Some(guest) = guest else { return };
    let cfg = ControlFlowGraph::with_function(func);
    let entry = entry_states(func, &cfg, guest);
    let mut cursor = FuncCursor::new(func);
    while let Some(block) = cursor.next_block() {
        let mut pending = entry[block];
        while let Some(inst) = cursor.next_inst() {
            if step(&mut pending, effect(cursor.func, inst, guest)) {
                cursor.ins().fence();
            }
        }
    }
}
