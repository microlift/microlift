//! Dead block parameters, and the branch arguments that feed them.
//!
//! Cranelift's own optimizer does not remove a block parameter nothing reads,
//! and the SSA builder produces a great many of them here: a lifted region's
//! `RET` dispatch has an in-region arm back to the region's earlier blocks, so
//! every guest register is a parameter of those blocks. That is harmless
//! while something consumes them, and the register file's exit sync did.

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::{Block, Function, Inst, Value};

/// Remove every block parameter nothing reads, and the branch arguments that
/// were feeding them. Returns whether anything changed.
///
/// The entry block is never touched: its parameters are the function's
/// signature.
pub fn remove_dead_block_params(func: &mut Function) -> bool {
    let live = live_values(func);

    // Descending within a block, so removing one parameter does not shift the
    // index of the next one to go.
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
            .filter(|&(_, &param)| !live.contains(&func.dfg.resolve_aliases(param)))
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
            for &inst in preds.get(&block).map_or(&[][..], Vec::as_slice) {
                remove_argument(func, inst, block, index);
            }
            func.dfg.remove_block_param(params[index]);
        }
    }
    true
}

/// Every value something actually reads.
///
/// Seeded with the *direct* operands of every instruction - a `brif`'s condition
/// counts, its block arguments do not - and then grown by the rule that an
/// argument feeding a live parameter is itself live. See the module doc for why
/// it is built in that direction.
fn live_values(func: &Function) -> HashSet<Value> {
    let mut live: HashSet<Value> = HashSet::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for &arg in func.dfg.inst_args(inst) {
                live.insert(func.dfg.resolve_aliases(arg));
            }
        }
    }

    loop {
        let mut grew = false;
        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                let calls = func.dfg.insts[inst]
                    .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables);
                for call in calls {
                    let target = call.block(&func.dfg.value_lists);
                    let params = func.dfg.block_params(target);
                    for (i, arg) in call.args(&func.dfg.value_lists).enumerate() {
                        let Some(&param) = params.get(i) else {
                            continue;
                        };
                        if !live.contains(&func.dfg.resolve_aliases(param)) {
                            continue;
                        }
                        if let Some(value) = arg.as_value() {
                            grew |= live.insert(func.dfg.resolve_aliases(value));
                        }
                    }
                }
            }
        }
        if !grew {
            return live;
        }
    }
}

/// Which instructions branch to each block, de-duplicated.
///
/// De-duplicated because one instruction may name a block twice - a `brif` whose
/// arms both go to the same place is common in this lifter's output - and
/// [`remove_argument`] already visits every call of an instruction.
fn predecessors(func: &Function) -> HashMap<Block, Vec<Inst>> {
    let mut preds: HashMap<Block, Vec<Inst>> = HashMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let calls = func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables);
            for call in calls {
                let target = call.block(&func.dfg.value_lists);
                let entry = preds.entry(target).or_default();
                if !entry.contains(&inst) {
                    entry.push(inst);
                }
            }
        }
    }
    preds
}

/// Drop argument `index` from every call `inst` makes to `block`.
///
/// Every call, not the first: a `brif` with both arms on one block has two, and
/// leaving either behind is an arity mismatch the verifier would catch.
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
