//! Edge blocks for branch arguments.
//!
//! A value passed as an argument on one edge of a `brif` is used by the branch itself, so
//! Cranelift computes it before branching, on every path. Moved into a block of its own on
//! that edge, the argument is used only there and the egraph elaborates its computation
//! there: flags a loop computes only for its exit sync then cost nothing per iteration.

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{BlockArg, Function, InstBuilder, Opcode};

/// Gives every argument-carrying edge of a multi-way branch a block of its own.
pub fn split_argument_edges(func: &mut Function) {
    let branches: Vec<_> = func
        .layout
        .blocks()
        .filter_map(|block| func.layout.last_inst(block))
        .filter(|&inst| {
            matches!(
                func.dfg.insts[inst].opcode(),
                Opcode::Brif | Opcode::BrTable
            )
        })
        .collect();

    for branch in branches {
        let srcloc = func.srcloc(branch);
        let edge_count = func.dfg.insts[branch]
            .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
            .len();
        for edge in 0..edge_count {
            let edge_block = func.dfg.make_block();
            let dfg = &mut func.dfg;
            let call = &mut dfg.insts[branch]
                .branch_destination_mut(&mut dfg.jump_tables, &mut dfg.exception_tables)[edge];
            let args: Vec<BlockArg> = call.args(&dfg.value_lists).collect();
            if args.is_empty() {
                continue;
            }
            let target = call.block(&dfg.value_lists);
            call.set_block(edge_block, &mut dfg.value_lists);
            call.clear(&mut dfg.value_lists);

            func.layout.append_block(edge_block);
            let mut pos = FuncCursor::new(func).at_bottom(edge_block);
            pos.set_srcloc(srcloc);
            pos.ins().jump(target, &args);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{AbiParam, Signature, UserFuncName, types};
    use cranelift_codegen::isa::CallConv;
    use cranelift_codegen::verifier::verify_function;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    #[test]
    fn moves_branch_arguments_onto_their_own_edge() {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
        let mut ctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut func, &mut ctx);
        let (entry, head, exit) = (b.create_block(), b.create_block(), b.create_block());
        b.append_block_params_for_function_params(entry);
        b.append_block_param(head, types::I64);
        b.append_block_param(exit, types::I64);
        b.switch_to_block(entry);
        let x = b.block_params(entry)[0];
        b.ins().jump(head, &[x.into()]);
        b.switch_to_block(head);
        let p = b.block_params(head)[0];
        let doubled = b.ins().iadd(p, p);
        b.ins().brif(p, head, &[p.into()], exit, &[doubled.into()]);
        b.switch_to_block(exit);
        let param = b.block_params(exit)[0];
        b.ins().return_(&[param]);
        b.seal_all_blocks();
        b.finalize(cranelift_codegen::isa::TargetFrontendConfig {
            default_call_conv: CallConv::SystemV,
            pointer_width: target_lexicon::PointerWidth::U64,
            page_size_align_log2: 12,
        });

        split_argument_edges(&mut func);

        verify_function(
            &func,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .expect("valid after splitting");
        let branch = func.layout.last_inst(head).unwrap();
        let calls = func.dfg.insts[branch]
            .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables);
        assert!(
            calls
                .iter()
                .all(|c| c.args(&func.dfg.value_lists).count() == 0)
        );
        assert!(calls.iter().all(|c| {
            let edge = c.block(&func.dfg.value_lists);
            edge != head && edge != exit
        }));
    }
}
