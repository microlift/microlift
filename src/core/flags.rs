//! Which condition-flag writes a region actually needs.
//!
//! Every write is lifted eagerly. Only after SSA is sealed do we walk backward
//! from observable uses and discard unused, side-effect-free flag computations.
//! The candidate graph starts at values actually assigned to flags, rather than
//! an instruction table. Context stores and callbacks are ordinary live uses;
//! synthetic exits, overlapping decodes and internal handler control flow need
//! no separate model. Loads, traps, calls, stores and unknown effects stay.

use std::collections::HashMap;

use cranelift_codegen::entity::{EntitySet, SecondaryMap};
use cranelift_codegen::ir::{Function, Inst, InstructionData, Opcode, Value, ValueDef};

use crate::core::dce;

/// Remove unused pure computations feeding the recorded flag values.
///
/// Must run after all blocks are sealed and context synchronization is complete.
/// Shared data computations survive whenever any actual consumer needs them.
pub(crate) fn eliminate(func: &mut Function, flags: &[Value]) {
    if flags.is_empty() {
        return;
    }

    // A block parameter depends on every incoming argument. Build these edges
    // once, including duplicate destinations and jump-table arms.
    let mut incoming: SecondaryMap<Value, Vec<Value>> = SecondaryMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for call in func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
            {
                let params = func.dfg.block_params(call.block(&func.dfg.value_lists));
                for (&param, arg) in params.iter().zip(call.args(&func.dfg.value_lists)) {
                    if let Some(value) = arg.as_value() {
                        incoming[func.dfg.resolve_aliases(param)]
                            .push(func.dfg.resolve_aliases(value));
                    }
                }
            }
        }
    }

    let mut candidates = EntitySet::<Inst>::new();
    let mut seen = EntitySet::<Value>::new();
    let mut work = flags.to_vec();
    while let Some(value) = work.pop() {
        let value = func.dfg.resolve_aliases(value);
        if !seen.insert(value) {
            continue;
        }
        match func.dfg.value_def(value) {
            ValueDef::Result(inst, _) if removable(func, inst) => {
                if candidates.insert(inst) {
                    work.extend_from_slice(func.dfg.inst_args(inst));
                }
            }
            ValueDef::Param(_, _) => work.extend_from_slice(&incoming[value]),
            _ => {}
        }
    }

    // Everything outside the candidate graph stays, even if otherwise dead.
    // Branch arguments are conditional uses: they matter only when their
    // destination parameter does. This permits removing dead loop-carried flags.
    let mut live = EntitySet::<Value>::new();
    let mut live_insts = EntitySet::<Inst>::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if !candidates.contains(inst) {
                work.extend_from_slice(func.dfg.inst_args(inst));
            }
        }
    }
    while let Some(value) = work.pop() {
        let value = func.dfg.resolve_aliases(value);
        if !live.insert(value) {
            continue;
        }
        match func.dfg.value_def(value) {
            ValueDef::Result(inst, _) => {
                if live_insts.insert(inst) {
                    work.extend_from_slice(func.dfg.inst_args(inst));
                }
            }
            ValueDef::Param(_, _) => work.extend_from_slice(&incoming[value]),
            _ => {}
        }
    }

    let mut dead = Vec::new();
    let mut plain = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if !candidates.contains(inst) {
                continue;
            }
            if !live_insts.contains(inst) {
                dead.push(inst);
                continue;
            }
            // A native overflow op also computes an ordinary arithmetic result.
            // When only that result survives, shed the unused condition output.
            let opcode = match func.dfg.insts[inst].opcode() {
                Opcode::UaddOverflow | Opcode::SaddOverflow => Opcode::Iadd,
                Opcode::UsubOverflow | Opcode::SsubOverflow => Opcode::Isub,
                Opcode::SmulOverflow => Opcode::Imul,
                _ => continue,
            };
            let results = func.dfg.inst_results(inst);
            if !live.contains(func.dfg.resolve_aliases(results[1])) {
                plain.push((inst, opcode, results[0]));
            }
        }
    }
    for inst in dead {
        func.layout.remove_inst(inst);
        func.dfg.clear_results(inst);
    }
    for (inst, opcode, result) in plain {
        let args = func.dfg.inst_args(inst);
        func.dfg.insts[inst] = InstructionData::Binary {
            opcode,
            args: [args[0], args[1]],
        };
        let ty = func.dfg.value_type(result);
        func.dfg
            .make_inst_results_reusing(inst, ty, [Some(result)].into_iter());
    }
    dce::remove_dead_block_params(func);
}

/// Fail closed for memory, control flow, traps and other side effects. In
/// particular, guest loads stay even when their result only fed dead flags:
/// changing which guest instruction faults would be an observable change.
fn removable(func: &Function, inst: Inst) -> bool {
    let op = func.dfg.insts[inst].opcode();
    !func.dfg.inst_results(inst).is_empty()
        && !op.is_call()
        && !op.is_branch()
        && !op.is_terminator()
        && !op.is_return()
        && !op.can_trap()
        && !op.can_load()
        && !op.can_store()
        && !op.other_side_effects()
}

/// A legacy set of guest condition flags, in the frontend's own numbering.
/// Retained for instruction metadata; SSA elimination does not use these bits.
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

/// Legacy instruction flag metadata, retained for frontend compatibility.
/// Dead flag elimination uses actual SSA dependencies instead of these sets.
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

/// Legacy address-based liveness container. The engine no longer uses it;
/// flag computations are removed only after their actual SSA uses are known.
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

#[cfg(test)]
mod tests {
    use super::*;

    use cranelift::prelude::*;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    fn function(build: impl FnOnce(&mut FunctionBuilder) -> Vec<Value>) -> (Function, Vec<Value>) {
        let mut func = Function::new();
        func.signature.params.push(AbiParam::new(types::I64));
        func.signature.returns.push(AbiParam::new(types::I64));
        let mut context = FunctionBuilderContext::new();
        let flags;
        {
            let mut b = FunctionBuilder::new(&mut func, &mut context);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            flags = build(&mut b);
            b.seal_all_blocks();
            b.finalize(cranelift_codegen::isa::TargetFrontendConfig {
                default_call_conv: cranelift_codegen::isa::CallConv::SystemV,
                pointer_width: target_lexicon::PointerWidth::U64,
                page_size_align_log2: 12,
            });
        }
        (func, flags)
    }

    fn check(func: &Function) {
        let settings =
            cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        cranelift_codegen::verify_function(func, &settings).expect("valid SSA after elimination");
    }

    fn count(func: &Function, opcode: Opcode) -> usize {
        func.layout
            .blocks()
            .flat_map(|b| func.layout.block_insts(b))
            .filter(|&i| func.dfg.insts[i].opcode() == opcode)
            .count()
    }

    #[test]
    fn only_unused_flag_dependencies_are_removed() {
        let (mut func, flags) = function(|b| {
            let x = b.block_params(b.current_block().unwrap())[0];
            let dead_flag = b.ins().bxor_imm_u(x, 16);
            let dead_neighbor = b.ins().iadd_imm_s(x, 4);
            let _ = dead_neighbor; // Outside the flag candidate graph.
            b.ins().return_(&[x]);
            vec![dead_flag]
        });
        eliminate(&mut func, &flags);
        check(&func);
        assert_eq!(count(&func, Opcode::Bxor), 0);
        assert_eq!(count(&func, Opcode::Iadd), 1);
    }

    #[test]
    fn loads_traps_and_calls_survive_even_when_their_flag_result_is_unused() {
        let (mut func, flags) = function(|b| {
            let ptr = b.block_params(b.current_block().unwrap())[0];
            let load = b.ins().load(types::I64, MemFlagsData::new(), ptr, 0);
            let zero = b.ins().iconst(types::I64, 0);
            let quotient = b.ins().udiv(load, zero);
            let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
            sig.returns.push(AbiParam::new(types::I64));
            let sig = b.import_signature(sig);
            let call = b.ins().call_indirect(sig, ptr, &[]);
            let result = b.inst_results(call)[0];
            b.ins().return_(&[ptr]);
            vec![load, quotient, result]
        });
        eliminate(&mut func, &flags);
        check(&func);
        assert_eq!(count(&func, Opcode::Load), 1);
        assert_eq!(count(&func, Opcode::Udiv), 1);
        assert_eq!(count(&func, Opcode::CallIndirect), 1);
    }

    #[test]
    fn actual_reads_inside_the_same_handler_keep_the_flag_write() {
        let (mut func, flags) = function(|b| {
            let ptr = b.block_params(b.current_block().unwrap())[0];
            let flag = b.ins().icmp_imm_s(IntCC::Equal, ptr, 0);
            let observed = b.ins().uextend(types::I64, flag);
            b.ins().store(MemFlagsData::new(), observed, ptr, 0);
            b.ins().return_(&[ptr]);
            vec![flag]
        });
        eliminate(&mut func, &flags);
        check(&func);
        assert_eq!(count(&func, Opcode::Icmp), 1);
    }

    #[test]
    fn both_incoming_values_of_a_live_flag_parameter_survive() {
        let (mut func, flags) = function(|b| {
            let x = b.block_params(b.current_block().unwrap())[0];
            let merge = b.create_block();
            let param = b.append_block_param(merge, types::I64);
            let first = b.ins().bxor_imm_u(x, 16);
            let second = b.ins().bxor_imm_u(x, 32);
            // Both destinations are the same block with different arguments.
            b.ins()
                .brif(x, merge, &[first.into()], merge, &[second.into()]);
            b.switch_to_block(merge);
            b.ins().return_(&[param]);
            vec![first, second]
        });
        eliminate(&mut func, &flags);
        check(&func);
        assert_eq!(count(&func, Opcode::Bxor), 2);
    }

    #[test]
    fn dead_loop_carried_flag_cycles_are_removed() {
        let (mut func, flags) = function(|b| {
            let x = b.block_params(b.current_block().unwrap())[0];
            let body = b.create_block();
            let counter = b.append_block_param(body, types::I64);
            let old_flag = b.append_block_param(body, types::I64);
            let done = b.create_block();
            let answer = b.append_block_param(done, types::I64);
            b.ins().jump(body, &[x.into(), x.into()]);
            b.switch_to_block(body);
            let flag = b.ins().bxor_imm_u(old_flag, 16);
            let next = b.ins().iadd_imm_s(counter, -1);
            b.ins().brif(
                next,
                body,
                &[next.into(), flag.into()],
                done,
                &[next.into()],
            );
            b.switch_to_block(done);
            b.ins().return_(&[answer]);
            vec![flag]
        });
        eliminate(&mut func, &flags);
        check(&func);
        assert_eq!(count(&func, Opcode::Bxor), 0);
        assert!(
            func.layout
                .blocks()
                .skip(1)
                .all(|b| func.dfg.block_params(b).len() == 1)
        );
    }

    #[test]
    fn unused_native_overflow_outputs_become_plain_arithmetic() {
        for opcode in [
            Opcode::UaddOverflow,
            Opcode::SaddOverflow,
            Opcode::UsubOverflow,
            Opcode::SsubOverflow,
            Opcode::SmulOverflow,
        ] {
            let (mut func, flags) = function(|b| {
                let x = b.block_params(b.current_block().unwrap())[0];
                let (res, flag) = match opcode {
                    Opcode::UaddOverflow => b.ins().uadd_overflow(x, x),
                    Opcode::SaddOverflow => b.ins().sadd_overflow(x, x),
                    Opcode::UsubOverflow => b.ins().usub_overflow(x, x),
                    Opcode::SsubOverflow => b.ins().ssub_overflow(x, x),
                    _ => b.ins().smul_overflow(x, x),
                };
                b.ins().return_(&[res]);
                vec![flag]
            });
            eliminate(&mut func, &flags);
            check(&func);
            assert_eq!(count(&func, opcode), 0);
        }
    }

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
