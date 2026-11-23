//! IR shapes the x86-64 lifter promises, so an optimization cannot silently regress.

use cranelift::codegen::ir::condcodes::IntCC;
use cranelift::codegen::ir::{Function, InstructionData, Opcode, Value, ValueDef};
use microlift::arch::x86_64::X86_64;
use microlift::arch::x86_64::context::RegisterContext;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;
use std::mem::offset_of;

fn lift(bytes: &[u8]) -> Function {
    lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, LifterConfig::default())
        .expect("lift")
        .func
}

fn insts(func: &Function) -> impl Iterator<Item = cranelift::codegen::ir::Inst> + '_ {
    func.layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
}

/// The compare a value is, if it is an `icmp`.
fn compare_of(func: &Function, value: Value) -> Option<IntCC> {
    let ValueDef::Result(inst, _) = func.dfg.value_def(func.dfg.resolve_aliases(value)) else {
        return None;
    };
    match func.dfg.insts[inst] {
        InstructionData::IntCompare { cond, .. } => Some(cond),
        _ => None,
    }
}

/// Conditions of every `brif`.
fn branch_compares(func: &Function) -> Vec<Option<IntCC>> {
    insts(func)
        .filter(|&i| func.dfg.insts[i].opcode() == Opcode::Brif)
        .map(|i| compare_of(func, func.dfg.inst_args(i)[0]))
        .collect()
}

/// Offsets of the loads the region makes from its `RegisterContext`.
fn loaded_offsets(func: &Function) -> Vec<i32> {
    let entry = func.layout.entry_block().expect("entry block");
    let ctx = func.dfg.block_params(entry)[0];
    insts(func)
        .filter_map(|i| match func.dfg.insts[i] {
            InstructionData::Load {
                opcode: Opcode::Load,
                arg,
                offset,
                ..
            } if func.dfg.resolve_aliases(arg) == ctx => Some(offset.into()),
            _ => None,
        })
        .collect()
}

const FLAG_OFFSETS: [usize; 6] = [
    offset_of!(RegisterContext, cf),
    offset_of!(RegisterContext, pf_src),
    offset_of!(RegisterContext, af_src),
    offset_of!(RegisterContext, zf),
    offset_of!(RegisterContext, sf),
    offset_of!(RegisterContext, of),
];

#[test]
fn signed_compare_branches_on_the_compare_itself() {
    // cmp rcx, rsi ; jl +1 ; nop ; ret
    let func = lift(&[0x48, 0x39, 0xF1, 0x7C, 0x01, 0x90, 0xC3]);
    assert!(
        branch_compares(&func).contains(&Some(IntCC::SignedLessThan)),
        "{}",
        func.display()
    );
}

#[test]
fn test_then_jle_branches_on_a_signed_compare_with_zero() {
    // test rsi, rsi ; jle +1 ; nop ; ret
    let func = lift(&[0x48, 0x85, 0xF6, 0x7E, 0x01, 0x90, 0xC3]);
    let compares = branch_compares(&func);
    assert!(
        compares.contains(&Some(IntCC::SignedLessThanOrEqual))
            || compares.contains(&Some(IntCC::SignedGreaterThan)),
        "{}",
        func.display()
    );
}

#[test]
fn xor_zeroing_reads_no_register() {
    // xor eax, eax ; ret
    let func = lift(&[0x31, 0xC0, 0xC3]);
    let rax = offset_of!(RegisterContext, rax) as i32;
    assert!(!loaded_offsets(&func).contains(&rax), "{}", func.display());
}

#[test]
fn pxor_zeroing_reads_no_register() {
    // pxor xmm0, xmm0 ; ret
    let mut config = LifterConfig::default();
    config.use_vector_xmm = true;
    let func = lift_to_code::<X86_64>(&[0x66, 0x0F, 0xEF, 0xC0, 0xC3], DEFAULT_LIFT_BASE, config)
        .expect("lift")
        .func;
    let xmm0 = offset_of!(RegisterContext, xmm) as i32;
    assert!(!loaded_offsets(&func).contains(&xmm0), "{}", func.display());
}

#[test]
fn immediate_shift_reads_no_old_flag() {
    // shl eax, 3 ; ret
    let func = lift(&[0xC1, 0xE0, 0x03, 0xC3]);
    let loaded = loaded_offsets(&func);
    for offset in FLAG_OFFSETS {
        assert!(!loaded.contains(&(offset as i32)), "{}", func.display());
    }
}

#[test]
fn branches_carry_no_arguments() {
    // xor eax,eax ; xor ecx,ecx ; test rsi,rsi ; jle ; add eax,[rdi+rcx*4] ; inc rcx ; cmp rcx,rsi ; jl ; ret
    let func = lift(&[
        0x31, 0xC0, 0x31, 0xC9, 0x48, 0x85, 0xF6, 0x7E, 0x0B, 0x03, 0x04, 0x8F, 0x48, 0xFF, 0xC1,
        0x48, 0x39, 0xF1, 0x7C, 0xF5, 0xC3,
    ]);
    for inst in insts(&func) {
        if matches!(
            func.dfg.insts[inst].opcode(),
            Opcode::Brif | Opcode::BrTable
        ) {
            for call in func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
            {
                assert_eq!(
                    call.args(&func.dfg.value_lists).count(),
                    0,
                    "{}",
                    func.display()
                );
            }
        }
    }
}

/// A fused condition must read the flags of the instruction that last wrote them.
#[test]
fn a_flag_write_between_compare_and_branch_is_honoured() {
    // mov eax, 1 ; cmp eax, 2 ; stc ; cmc ; jb +5 ; mov ebx, 7 ; ret
    let bytes = [
        0xB8, 0x01, 0x00, 0x00, 0x00, 0x83, 0xF8, 0x02, 0xF9, 0xF5, 0x72, 0x05, 0xBB, 0x07, 0x00,
        0x00, 0x00, 0xC3,
    ];
    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.ctx.rbx, 7);
}

#[test]
fn byte_compares_are_signed_at_their_width() {
    // mov al, 0x80 ; cmp al, 1 ; setl bl ; setb cl ; ret
    let bytes = [
        0xB0, 0x80, 0x3C, 0x01, 0x0F, 0x9C, 0xC3, 0x0F, 0x92, 0xC1, 0xC3,
    ];
    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.ctx.rbx & 0xFF, 1);
    assert_eq!(out.ctx.rcx & 0xFF, 0);
}

#[test]
fn zero_idioms_set_the_flags_of_zero() {
    // mov eax, -1 ; stc ; sub eax, eax ; ret
    let bytes = [0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xF9, 0x29, 0xC0, 0xC3];
    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.ctx.rax, 0);
    assert_eq!(
        (out.ctx.cf, out.ctx.zf, out.ctx.sf, out.ctx.of),
        (0, 1, 0, 0)
    );
}

#[test]
fn a_zero_immediate_count_leaves_the_flags() {
    // mov eax, 1 ; cmp eax, 2 ; shl eax, 0x20 ; ret  (masked count 0)
    let bytes = [
        0xB8, 0x01, 0x00, 0x00, 0x00, 0x83, 0xF8, 0x02, 0xC1, 0xE0, 0x20, 0xC3,
    ];
    let out = LiftTest::new(&bytes).run();
    assert_eq!((out.ctx.cf, out.ctx.zf, out.ctx.sf), (1, 0, 1));
}
