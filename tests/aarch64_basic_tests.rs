//! aarch64 guest frontend: decode classification + JIT round-trip tests for the
//! instruction-semantics vertical slice.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::arch::ControlFlowKind;
use microlift::arch::aarch64::Lifter64;
use microlift::arch::aarch64::context::Aarch64RegisterContext;
use microlift::arch::aarch64::decode::{control_flow, decode_region};
use microlift::backend::CraneliftBackend;
use microlift::context::ExitReason;
use microlift::os::get_host_calling_conv;

const BASE: u64 = 0x1000;

/// Lift `bytes` as an aarch64 guest region at `BASE`, JIT-compile, run against
/// `ctx`, and return the resulting context.
fn run(bytes: &[u8], mut ctx: Aarch64RegisterContext) -> Aarch64RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter64::new(builder);

    ctx.image_base = BASE;
    ctx.pc = BASE;

    lifter.lift(bytes, BASE).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut Aarch64RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

// Hand-encoded A64 instructions (little-endian on the wire).
const MOVZ_X0_42: [u8; 4] = [0x40, 0x05, 0x80, 0xD2]; // movz x0, #42
const MOVZ_X0_10: [u8; 4] = [0x40, 0x01, 0x80, 0xD2]; // movz x0, #10
const MOVZ_X1_20: [u8; 4] = [0x81, 0x02, 0x80, 0xD2]; // movz x1, #20
const MOVZ_X0_5: [u8; 4] = [0xA0, 0x00, 0x80, 0xD2]; // movz x0, #5
const MOVZ_X1_3: [u8; 4] = [0x61, 0x00, 0x80, 0xD2]; // movz x1, #3
const MOVZ_X0_7: [u8; 4] = [0xE0, 0x00, 0x80, 0xD2]; // movz x0, #7
const MOVZ_X1_7: [u8; 4] = [0xE1, 0x00, 0x80, 0xD2]; // movz x1, #7
const MOVZ_X0_0: [u8; 4] = [0x00, 0x00, 0x80, 0xD2]; // movz x0, #0
const ADD_X2_X0_X1: [u8; 4] = [0x02, 0x00, 0x01, 0x8B]; // add x2, x0, x1
const SUBS_X2_X0_X1: [u8; 4] = [0x02, 0x00, 0x01, 0xEB]; // subs x2, x0, x1
const SUBS_XZR_X0_X1: [u8; 4] = [0x1F, 0x00, 0x01, 0xEB]; // cmp x0, x1
const BEQ_PLUS8: [u8; 4] = [0x40, 0x00, 0x00, 0x54]; // b.eq #+8
const RET: [u8; 4] = [0xC0, 0x03, 0x5F, 0xD6]; // ret

const STR_X0_SP: [u8; 4] = [0xE0, 0x03, 0x00, 0xF9]; // str x0, [sp]
const LDR_X1_SP: [u8; 4] = [0xE1, 0x03, 0x40, 0xF9]; // ldr x1, [sp]

fn program(parts: &[[u8; 4]]) -> Vec<u8> {
    parts.iter().flatten().copied().collect()
}

// --- decode classification ----------------------------------------------------

#[test]
fn decode_movz_and_ret() {
    let bytes = program(&[MOVZ_X0_42, RET]);
    let insts = decode_region(&bytes, BASE);
    assert_eq!(insts.len(), 2);
    assert!(insts[0].valid);
    assert_eq!(insts[0].ip, BASE);
    assert_eq!(insts[1].ip, BASE + 4);
    assert_eq!(control_flow(&insts[0]).kind, ControlFlowKind::Fallthrough);
    assert_eq!(control_flow(&insts[1]).kind, ControlFlowKind::Ret);
}

#[test]
fn decode_conditional_branch_target() {
    // subs at 0x1008, b.eq at 0x100C -> taken target 0x1014.
    let bytes = program(&[MOVZ_X0_7, MOVZ_X1_7, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let insts = decode_region(&bytes, BASE);
    let beq = &insts[3];
    let cf = control_flow(beq);
    assert_eq!(cf.kind, ControlFlowKind::CondBranch);
    assert_eq!(cf.static_targets, vec![BASE + 0x14]);
}

// --- JIT round-trips ----------------------------------------------------------

#[test]
fn movz_ret() {
    let ctx = run(&program(&[MOVZ_X0_42, RET]), Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 42);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn add_three_regs() {
    let bytes = program(&[MOVZ_X0_10, MOVZ_X1_20, ADD_X2_X0_X1, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[0], 10);
    assert_eq!(ctx.x[1], 20);
    assert_eq!(ctx.x[2], 30);
}

#[test]
fn subs_sets_flags() {
    // 5 - 3 = 2; N=0, Z=0, C=1 (no borrow), V=0.
    let bytes = program(&[MOVZ_X0_5, MOVZ_X1_3, SUBS_X2_X0_X1, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.x[2], 2);
    assert_eq!(ctx.n, 0);
    assert_eq!(ctx.z, 0);
    assert_eq!(ctx.c, 1);
    assert_eq!(ctx.vf, 0);
}

#[test]
fn conditional_branch_taken() {
    // 7 == 7 -> Z=1 -> b.eq taken -> skips `movz x0, #0`, so x0 stays 7.
    let bytes = program(&[MOVZ_X0_7, MOVZ_X1_7, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.z, 1);
    assert_eq!(ctx.x[0], 7);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn store_then_load_roundtrip() {
    // movz x0,#42; str x0,[sp]; ldr x1,[sp]; ret  -> x1 == 42.
    let mut stack = [0u64; 64];
    let sp = stack.as_mut_ptr() as u64 + 256;
    let mut ctx = Aarch64RegisterContext::default();
    ctx.sp = sp;
    let bytes = program(&[MOVZ_X0_42, STR_X0_SP, LDR_X1_SP, RET]);
    let ctx = run(&bytes, ctx);
    assert_eq!(ctx.x[0], 42);
    assert_eq!(ctx.x[1], 42);
    assert_eq!(stack[32], 42); // sp pointed at stack[32]
}

#[test]
fn conditional_branch_not_taken() {
    // 5 != 3 -> Z=0 -> b.eq not taken -> falls through to `movz x0, #0`.
    let bytes = program(&[MOVZ_X0_5, MOVZ_X1_3, SUBS_XZR_X0_X1, BEQ_PLUS8, MOVZ_X0_0, RET]);
    let ctx = run(&bytes, Aarch64RegisterContext::default());
    assert_eq!(ctx.z, 0);
    assert_eq!(ctx.x[0], 0);
}
