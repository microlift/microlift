use microlift::arch::x86_64::context::X86_64Callbacks;
use microlift::testkit::prelude::*;

fn run_stack_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

unsafe extern "C" fn test_syscall_handler(ctx: *mut RegisterContext) {
    unsafe {
        let ctx = &mut *ctx;
        ctx.rax = ctx.rax.wrapping_add(5);
    }
}

/// A syscall callback that moves the guest somewhere else entirely, the way a
/// `sigreturn` or a thread exit would.
unsafe extern "C" fn diverting_syscall_handler(ctx: *mut RegisterContext) {
    unsafe {
        let ctx = &mut *ctx;
        ctx.rax = 0xD1;
        ctx.rip = 0x1234_0000;
    }
}

#[test]
fn test_push_pop_jit() {
    // mov rax, 0x123456789ABCDEF0; push rax; mov rax, 0; pop rbx; ret
    let bytes = [
        0x48, 0xB8, 0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12, // mov rax, imm64
        0x50, // push rax
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x5B, // pop rbx
        0xC3, // ret
    ];
    let ctx = run_stack_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rbx, 0x123456789ABCDEF0);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn test_call_ret_nested_jit() {
    // call sub; ret; sub: mov rax, 42; ret
    let bytes = [
        0xE8, 0x01, 0x00, 0x00, 0x00, // 0x1000: call 0x1006 (relative +1 from 0x1005)
        0xC3, // 0x1005: ret (final)
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // 0x1006: mov rax, 42
        0xC3, // 0x100d: ret (returns to 0x1005)
    ];
    let ctx = run_stack_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

#[test]
fn test_syscall_calls_back_and_runs_on() {
    // mov rax, 7; syscall; add rax, 3; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x0F, 0x05, // syscall
        0x48, 0x83, 0xC0, 0x03, // add rax, 3
        0xC3, // ret
    ];

    let ctx = run_stack_test(
        &bytes,
        RegisterContext {
            callbacks: X86_64Callbacks {
                handle_syscall: test_syscall_handler,
                ..X86_64Callbacks::default()
            },
            ..RegisterContext::default()
        },
    );

    assert_eq!(ctx.rax, 15, "7, +5 in the callback, +3 after it returned");
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
}

/// The PC the callback leaves behind is what the guest resumes at, so a
/// callback that redirects control leaves the region instead of falling
/// through to the instruction after the `syscall`.
#[test]
fn test_syscall_callback_can_redirect_the_guest() {
    // mov rax, 0x60; syscall; add rax, 1; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x60, 0x00, 0x00, 0x00, // mov rax, 0x60
        0x0F, 0x05, // syscall
        0x48, 0x83, 0xC0, 0x01, // add rax, 1
        0xC3, // ret
    ];

    let ctx = run_stack_test(
        &bytes,
        RegisterContext {
            callbacks: X86_64Callbacks {
                handle_syscall: diverting_syscall_handler,
                ..X86_64Callbacks::default()
            },
            ..RegisterContext::default()
        },
    );

    assert_eq!(ctx.exit_reason, ExitReason::UnknownJump as u32);
    assert_eq!(ctx.rip, 0x1234_0000);
    assert_eq!(ctx.rax, 0xD1, "the `add` after the syscall did not run");
}

#[test]
fn test_stack_data_integrity() {
    // Test that we can actually write more data to the stack and read it back
    // push 1; push 2; push 3; pop rax; pop rbx; pop rcx; ret
    let bytes = [
        0x6A, 0x01, // push 1
        0x6A, 0x02, // push 2
        0x6A, 0x03, // push 3
        0x58, // pop rax
        0x5B, // pop rbx
        0x59, // pop rcx
        0xC3, // ret
    ];
    let ctx = run_stack_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 3);
    assert_eq!(ctx.rbx, 2);
    assert_eq!(ctx.rcx, 1);
}
