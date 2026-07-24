//! Tests for `UD2` and `INT3` VM exits.
//!
//! Both instructions are block terminators that sync guest state and return
//! to the host with a dedicated exit reason. RIP is set to point *at* the
//! faulting instruction, matching real hardware behaviour.

use microlift::testkit::prelude::*;

fn compile_and_run(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
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
    //
    // Bind the whole `RunOutcome` rather than just its `ctx`: the returned
    // `rsp` points into the guest stack *owned by the outcome's harness*.
    // Dropping the outcome here (as `compile_and_run` does) would free that
    // stack and leave the `rsp` carried into the second run dangling - the
    // re-entry below would then write its exit trampoline into freed memory.
    let first = LiftTest::new(&bytes).run();
    let ctx = first.ctx;
    assert_eq!(ctx.exit_reason, ExitReason::Breakpoint as u32);
    assert_eq!(ctx.rip, 0x1000);

    // Simulate host advancing RIP by 1 and re-entering (re-lift the same region).
    let mut ctx2 = ctx;
    ctx2.rip = 0x1001; // skip the int3

    // Re-lift with emit_resume_blocks so every instruction boundary is a block
    // leader and the dispatch switch has a case for 0x1001. The preset rip
    // routes the entry dispatch straight to the `mov rax, 99` block.
    let out = LiftTest::new(&bytes)
        .with_config(|c| c.emit_resume_blocks = true)
        .ctx(ctx2)
        .run();

    assert_eq!(out.ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(out.ctx.rax, 99);

    // Keep the first harness (and thus the shared guest stack) alive until both
    // runs are done.
    drop(first);
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

/// `INSD` (`0x6D`) must exit with `ExitReason::PortInput`, report the element
/// size (4 bytes) in `exit_code`, and point RIP *at* the instruction so the
/// host can emulate the port access.
#[test]
fn test_insd_exit_reason_and_size() {
    #[rustfmt::skip]
    let bytes = [
        0x6D, // insd  @ 0x1000 (1 byte)
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::PortInput as u32);
    assert_eq!(ctx.exit_code, 4, "exit_code should hold the element size in bytes");
    assert_eq!(ctx.rip, 0x1000, "rip should point at the insd instruction");
}

/// `INSB` (`0x6C`) reports an element size of 1; side-effects before it remain
/// visible on exit (DX/RDI are synced for the host to emulate the access).
#[test]
fn test_insb_size_and_state_synced() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC2, 0x60, 0x00, 0x00, 0x00, // mov rdx, 0x60  @ 0x1000 (port)
        0x6C,                                      // insb            @ 0x1007
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::PortInput as u32);
    assert_eq!(ctx.exit_code, 1);
    assert_eq!(ctx.rdx, 0x60, "port number in DX must be synced for the host");
    assert_eq!(ctx.rip, 0x1007, "rip should point at the insb instruction");
}

/// `OUTSD` (`0x6F`) must exit with `ExitReason::PortOutput`, report the element
/// size (4), and point RIP *at* the instruction.
#[test]
fn test_outsd_exit_reason_and_size() {
    #[rustfmt::skip]
    let bytes = [
        0x6F, // outsd  @ 0x1000 (1 byte)
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::PortOutput as u32);
    assert_eq!(ctx.exit_code, 4, "exit_code should hold the element size in bytes");
    assert_eq!(ctx.rip, 0x1000, "rip should point at the outsd instruction");
}

/// `OUTSB` (`0x6E`) reports size 1; the source pointer (RSI) and port (DX) are
/// synced so the host can emulate the access.
#[test]
fn test_outsb_size_and_state_synced() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC2, 0x70, 0x00, 0x00, 0x00, // mov rdx, 0x70  @ 0x1000 (port)
        0x6E,                                      // outsb           @ 0x1007
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::PortOutput as u32);
    assert_eq!(ctx.exit_code, 1);
    assert_eq!(ctx.rdx, 0x70, "port number in DX must be synced for the host");
    assert_eq!(ctx.rip, 0x1007, "rip should point at the outsb instruction");
}

/// Port input and output must be distinguishable by the host.
#[test]
fn test_port_io_exit_reasons_are_distinct() {
    assert_ne!(ExitReason::PortInput as u32, ExitReason::PortOutput as u32);
}

/// `RDTSC` (`0F 31`) must exit with `ExitReason::Rdtsc` and point RIP *past* the
/// instruction (the resume point) so the host can supply the counter in RAX/RDX.
#[test]
fn test_rdtsc_exit_reason() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x31, // rdtsc  @ 0x1000 (2 bytes)
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Rdtsc as u32);
    assert_eq!(ctx.rip, 0x1002, "rip should point past rdtsc (resume point)");
}

/// State preceding `RDTSC` must be synced on exit so the host sees current
/// registers before populating the timestamp.
#[test]
fn test_rdtsc_state_synced() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC3, 0x63, 0x00, 0x00, 0x00, // mov rbx, 99  @ 0x1000
        0x0F, 0x31,                                // rdtsc         @ 0x1007
    ];
    let ctx = compile_and_run(&bytes, RegisterContext::default());
    assert_eq!(ctx.exit_reason, ExitReason::Rdtsc as u32);
    assert_eq!(ctx.rbx, 99);
    assert_eq!(ctx.rip, 0x1009);
}
