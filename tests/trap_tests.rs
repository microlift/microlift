//! Tests for `UD2` and `INT3` VM exits.
//!
//! Both instructions are block terminators that sync guest state and return
//! to the host with a dedicated exit reason. RIP is set to point *at* the
//! faulting instruction, matching real hardware behaviour.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::lifter::{Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

fn compile_and_run(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::new(builder);

    let mut stack = [0u64; 256];
    let stack_top = stack.as_mut_ptr() as u64 + 256 * 8;
    ctx.image_base = 0x1000;
    ctx.rsp = stack_top - 8;
    unsafe {
        *(ctx.rsp as *mut u64) = lifter.exit_trampoline_address;
    }

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::with_flags(settings::Flags::new(flag_builder))
        .expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

/// A bare `ud2` must exit with `ExitReason::Ud2`.
#[test]
fn test_ud2_exit_reason() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x0B, // ud2
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Ud2 as u32);
}

/// RIP must point *at* the `ud2` instruction (VA 0x1000), not past it.
#[test]
fn test_ud2_rip_points_at_instruction() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x0B, // ud2  @ 0x1000
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.rip, 0x1000, "rip should point at the ud2 opcode");
}

/// Code before `ud2` must be executed; side-effects must be visible on exit.
#[test]
fn test_ud2_state_synced_on_exit() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42  @ 0x1000
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7   @ 0x1007
        0x0F, 0x0B,                                // ud2           @ 0x100E
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Ud2 as u32);
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rcx, 7);
    assert_eq!(ctx.rip, 0x100E, "rip should point at the ud2 opcode");
}

/// `ud2` after a conditional branch (branch not taken) must still fire.
#[test]
fn test_ud2_after_conditional_branch_not_taken() {
    // jnz +2  (cond not taken because ZF=0 by default) -> falls through to ud2
    #[rustfmt::skip]
    let bytes = [
        0x75, 0x02,  // jnz +2  (ZF=0 -> taken, skips ud2)
        0x0F, 0x0B,  // ud2
        0xC3,        // ret  (jump target - reached when branch is taken)
    ];

    // ZF=0 -> jnz taken -> branch skips ud2 -> hits ret -> Return exit
    let ctx_taken = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(
        ctx_taken.exit_reason,
        ExitReason::Return as u32,
        "jnz taken should reach ret, not ud2"
    );

    // ZF=1 -> jnz NOT taken -> falls through to ud2
    let mut ctx_in = RegisterContext::default();
    ctx_in.zf = 1;
    let ctx_fall = compile_and_run(&bytes, ctx_in);
    assert_eq!(
        ctx_fall.exit_reason,
        ExitReason::Ud2 as u32,
        "jnz not taken should fall through to ud2"
    );
}

/// A bare `int3` must exit with `ExitReason::Breakpoint`.
#[test]
fn test_int3_exit_reason() {
    #[rustfmt::skip]
    let bytes = [
        0xCC, // int3
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Breakpoint as u32);
}

/// RIP must point *at* the `int3` byte (VA 0x1000), not past it.
#[test]
fn test_int3_rip_points_at_instruction() {
    #[rustfmt::skip]
    let bytes = [
        0xCC, // int3  @ 0x1000
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.rip, 0x1000, "rip should point at the int3 byte");
}

/// Code before `int3` must have its side-effects visible on exit.
#[test]
fn test_int3_state_synced_on_exit() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0x00, 0x00, 0x00, // mov rax, 255  @ 0x1000
        0x48, 0xC7, 0xC2, 0x01, 0x00, 0x00, 0x00, // mov rdx, 1    @ 0x1007
        0xCC,                                      // int3           @ 0x100E
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Breakpoint as u32);
    assert_eq!(ctx.rax, 255);
    assert_eq!(ctx.rdx, 1);
    assert_eq!(ctx.rip, 0x100E, "rip should point at the int3 byte");
}

/// After the host advances RIP past `int3` (+1 byte), execution can continue.
/// Simulate this by placing `int3; mov rax, 99; ret` and manually skipping the
/// breakpoint: run once (gets Breakpoint exit), bump rip by 1, re-lift and re-run.
#[test]
fn test_int3_resume_by_advancing_rip() {
    // int3 @ 0x1000 (1 byte); mov rax, 99 @ 0x1001 (7 bytes); ret @ 0x1008
    #[rustfmt::skip]
    let bytes = [
        0xCC,                                       // int3       @ 0x1000
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,  // mov rax,99 @ 0x1001
        0xC3,                                       // ret        @ 0x1008
    ];

    // First run: hits int3 immediately.
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Breakpoint as u32);
    assert_eq!(ctx.rip, 0x1000);

    // Simulate host advancing RIP by 1 and re-entering (re-lift the same region).
    let mut ctx2 = ctx;
    ctx2.rip = 0x1001; // skip the int3

    // Re-lift with emit_resume_blocks so every instruction boundary is a block
    // leader and the dispatch switch has a case for 0x1001.
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
    let config = LifterConfig {
        emit_resume_blocks: true,
        ..Default::default()
    };
    let mut lifter = Lifter::with_config(builder, config);

    let mut stack = [0u64; 256];
    let stack_top = stack.as_mut_ptr() as u64 + 256 * 8;
    ctx2.rsp = stack_top - 8;
    ctx2.image_base = 0x1000;
    unsafe {
        *(ctx2.rsp as *mut u64) = lifter.exit_trampoline_address;
    }

    lifter.lift(&bytes, 0x1000).expect("re-lift failed");

    let mut backend = CraneliftBackend::with_flags(settings::Flags::new(flag_builder))
        .expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx2);

    assert_eq!(ctx2.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx2.rax, 99);
}

/// `ud2` and `int3` must produce different exit reasons so the host can
/// tell them apart.
#[test]
fn test_ud2_and_int3_exit_reasons_are_distinct() {
    assert_ne!(ExitReason::Ud2 as u32, ExitReason::Breakpoint as u32);
}

/// `INT imm8` must exit with `ExitReason::Interrupt`, report the vector in
/// `exit_code`, and point RIP *past* the instruction (the resume point).
#[test]
fn test_int_imm8_exit_reason_and_vector() {
    #[rustfmt::skip]
    let bytes = [
        0xCD, 0x80, // int 0x80  @ 0x1000 (2 bytes)
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Interrupt as u32);
    assert_eq!(ctx.exit_code, 0x80, "exit_code should hold the interrupt vector");
    assert_eq!(ctx.rip, 0x1002, "rip should point past the int instruction");
}

/// Side-effects before `INT imm8` must be visible on exit.
#[test]
fn test_int_imm8_state_synced_on_exit() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42  @ 0x1000
        0xCD, 0x2A,                                // int 0x2A      @ 0x1007
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Interrupt as u32);
    assert_eq!(ctx.exit_code, 0x2A);
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rip, 0x1009, "rip should point past the int instruction");
}

/// `INT1`/`ICEBP` (`0xF1`) must raise vector 1 via `ExitReason::Interrupt`.
#[test]
fn test_int1_exit_reason_and_vector() {
    #[rustfmt::skip]
    let bytes = [
        0xF1, // int1 / icebp  @ 0x1000 (1 byte)
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Interrupt as u32);
    assert_eq!(ctx.exit_code, 1, "int1 raises the debug-exception vector");
    assert_eq!(ctx.rip, 0x1001, "rip should point past the int1 byte");
}

/// `INTO` (`0xCE`) does not exist in 64-bit mode - the decoder rejects it as an
/// invalid instruction rather than treating it as a software interrupt.
#[test]
fn test_into_is_invalid_in_64bit() {
    #[rustfmt::skip]
    let bytes = [
        0xCE, // into (invalid in long mode)  @ 0x1000
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::InvalidInstruction as u32);
}

/// `INT3`, `INT imm8`, and `INT1` must all be distinguishable: the dedicated
/// breakpoint opcode reports `Breakpoint`, the rest report `Interrupt`.
#[test]
fn test_int_family_exit_reasons_are_distinct() {
    assert_ne!(ExitReason::Breakpoint as u32, ExitReason::Interrupt as u32);
}
