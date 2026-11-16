//! Dead block parameters and the branch arguments that feed them.
//!
//! Cranelift does not remove unread block parameters; the SSA builder makes many.

use cranelift_codegen::entity::{EntitySet, SecondaryMap};
use cranelift_codegen::ir::{Block, BlockArg, Function, Inst, Value};

/// Removes unread block parameters and their branch arguments; returns whether anything changed.
///
/// The entry block is skipped (its parameters are the signature).
pub fn remove_dead_block_params(func: &mut Function) -> bool {
    let live = live_values(func);

    let mut doomed: Vec<(Block, Vec<bool>)> = Vec::new();
    let entry = func.layout.entry_block();
    for block in func.layout.blocks() {
        if Some(block) == entry {
            continue;
        }
        let keep: Vec<bool> = func
            .dfg
            .block_params(block)
            .iter()
            .map(|&param| live.contains(func.dfg.resolve_aliases(param)))
            .collect();
        if keep.contains(&false) {
            doomed.push((block, keep));
        }
    }
    if doomed.is_empty() {
        return false;
    }

    // Rebuilt in one pass each, since removing one parameter or argument at a time
    // renumbers everything after it.
    let preds = predecessors(func);
    for (block, keep) in doomed {
        let params: Vec<Value> = func.dfg.block_params(block).to_vec();
        func.dfg.detach_block_params(block);
        for (&param, _) in params.iter().zip(&keep).filter(|&(_, &kept)| kept) {
            func.dfg.attach_block_param(block, param);
        }
        for &inst in preds[block].as_slice() {
            keep_arguments(func, inst, block, &keep);
        }
    }
    true
}

/// Values something reads; a branch argument counts only if its parameter is live.
fn live_values(func: &Function) -> EntitySet<Value> {
    let mut live: EntitySet<Value> = EntitySet::with_capacity(func.dfg.num_values());
    // Each parameter's incoming arguments, both resolved.
    let mut feeds: SecondaryMap<Value, Vec<Value>> = SecondaryMap::new();
    let mut work: Vec<Value> = Vec::new();

    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for &arg in func.dfg.inst_args(inst) {
                let arg = func.dfg.resolve_aliases(arg);
                if live.insert(arg) {
                    work.push(arg);
                }
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
                    feeds[func.dfg.resolve_aliases(param)].push(func.dfg.resolve_aliases(value));
                }
            }
        }
    }

    while let Some(value) = work.pop() {
        for &fed in &feeds[value] {
            if live.insert(fed) {
                work.push(fed);
            }
        }
    }
    live
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

/// Keeps only the arguments `keep` marks in every call `inst` makes to `block` (a `brif`
/// may have two). Arguments past the parameters, which the verifier rejects anyway, stay.
fn keep_arguments(func: &mut Function, inst: Inst, block: Block, keep: &[bool]) {
    let dfg = &mut func.dfg;
    let pool = &mut dfg.value_lists;
    for call in dfg.insts[inst].branch_destination_mut(
        &mut dfg.jump_tables,
        &mut dfg.exception_tables,
    ) {
        if call.block(pool) != block {
            continue;
        }
        let args: Vec<BlockArg> = call.args(pool).collect();
        call.clear(pool);
        call.extend(
            args.into_iter()
                .zip(keep.iter().chain(std::iter::repeat(&true)))
                .filter(|&(_, &kept)| kept)
                .map(|(arg, _)| arg),
            pool,
        );
    }
}
