//! Dead block parameters and the branch arguments that feed them.
//!
//! Cranelift does not remove unread block parameters; the SSA builder makes many.

use cranelift_codegen::entity::{EntitySet, SecondaryMap};
use cranelift_codegen::ir::{Block, Function, Inst, Value};

/// Removes unread block parameters and their branch arguments; returns whether anything changed.
///
/// The entry block is skipped (its parameters are the signature).
pub fn remove_dead_block_params(func: &mut Function) -> bool {
    let live = live_values(func);

    let mut doomed: Vec<(Block, Vec<usize>)> = Vec::new();
    let entry = func.layout.entry_block();
    for block in func.layout.blocks() {
        if Some(block) == entry {
            continue;
        }
        let dead: Vec<usize> = func
            .dfg
            .block_params(block)
            .iter()
            .enumerate()
            .filter(|&(_, &param)| !live.contains(func.dfg.resolve_aliases(param)))
            .map(|(i, _)| i)
            .collect();
        if !dead.is_empty() {
            doomed.push((block, dead));
        }
    }
    if doomed.is_empty() {
        return false;
    }

    let preds = predecessors(func);
    for (block, dead) in doomed {
        let params: Vec<Value> = func.dfg.block_params(block).to_vec();
        for &index in dead.iter().rev() {
            for &inst in preds[block].as_slice() {
                remove_argument(func, inst, block, index);
            }
            func.dfg.remove_block_param(params[index]);
        }
    }
    true
}

/// Values something reads; a branch argument counts only if its parameter is live.
fn live_values(func: &Function) -> EntitySet<Value> {
    let mut live: EntitySet<Value> = EntitySet::with_capacity(func.dfg.num_values());

    // (parameter, argument), both resolved.
    let mut edges: Vec<(Value, Value)> = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for &arg in func.dfg.inst_args(inst) {
                live.insert(func.dfg.resolve_aliases(arg));
            }
            let calls = func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables);
            for call in calls {
                let target = call.block(&func.dfg.value_lists);
                let params = func.dfg.block_params(target);
                for (i, arg) in call.args(&func.dfg.value_lists).enumerate() {
                    let Some(&param) = params.get(i) else {
                        continue;
                    };
                    let Some(value) = arg.as_value() else {
                        continue;
                    };
                    edges.push((
                        func.dfg.resolve_aliases(param),
                        func.dfg.resolve_aliases(value),
                    ));
                }
            }
        }
    }

    loop {
        let mut grew = false;
        for &(param, value) in &edges {
            if live.contains(param) {
                grew |= live.insert(value);
            }
        }
        if !grew {
            return live;
        }
    }
}

/// Branching instructions per block, de-duplicated (a `brif` may name a block twice).
fn predecessors(func: &Function) -> SecondaryMap<Block, Vec<Inst>> {
    let mut preds: SecondaryMap<Block, Vec<Inst>> = SecondaryMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let calls = func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables);
            for call in calls {
                let target = call.block(&func.dfg.value_lists);
                let entry = &mut preds[target];
                if !entry.contains(&inst) {
                    entry.push(inst);
                }
            }
        }
    }
    preds
}

/// Drops argument `index` from every call `inst` makes to `block` (a `brif` may have two).
fn remove_argument(func: &mut Function, inst: Inst, block: Block, index: usize) {
    let dfg = &mut func.dfg;
    let pool = &mut dfg.value_lists;
    for call in dfg.insts[inst].branch_destination_mut(
        &mut dfg.jump_tables,
        &mut dfg.exception_tables,
    ) {
        if call.block(pool) == block && index < call.len(pool) {
            call.remove(index, pool);
        }
    }
}
