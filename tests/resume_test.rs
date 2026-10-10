//! An instruction the lifter does not implement goes to the host's callback inline and the region carries on.

use microlift::arch::x86_64::context::X86_64Callbacks;
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// `WBINVD` (`0F 09`) is ring-0 only, so it stands in for a never-implemented instruction.
const WBINVD: [u8; 2] = [0x0F, 0x09];

/// Offset of the `WBINVD`: 7 bytes of `mov rax, 42`.
const WBINVD_OFFSET: u64 = 7;

static CALLS: AtomicU64 = AtomicU64::new(0);
static SEEN_RIP: AtomicU64 = AtomicU64::new(0);
static SEEN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Treats the `WBINVD` as a no-op and advances the PC past it.
unsafe extern "C" fn interpret(ctx: *mut RegisterContext, num_instructions: u64) {
    let ctx = unsafe { &mut *ctx };
    CALLS.fetch_add(1, Ordering::SeqCst);
    SEEN_RIP.store(ctx.rip, Ordering::SeqCst);
    SEEN_COUNT.store(num_instructions, Ordering::SeqCst);
    ctx.rip += WBINVD.len() as u64;
}

#[test]
fn test_unimplemented_instruction_runs_through_the_callback() {
    let mut bytes = vec![0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]; // mov rax, 42
    bytes.extend_from_slice(&WBINVD); // unimplemented
    bytes.extend_from_slice(&[0x48, 0xFF, 0xC0]); // inc rax
    bytes.push(0xC3); // ret

    let out = LiftTest::new(&bytes)
        .no_unicorn()
        .with_config(|c| c.interpreter_fallback = true)
        .reg(|c| {
            c.callbacks = X86_64Callbacks {
                interpreter_fallback: interpret,
                ..X86_64Callbacks::default()
            }
        })
        .run();

    assert_eq!(out.exit, ExitReason::Return, "the region ran to its RET");
    assert_eq!(
        out.ctx.rax, 43,
        "state before the callback reached it, and the instruction after it ran"
    );
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        SEEN_RIP.load(Ordering::SeqCst),
        DEFAULT_LIFT_BASE + WBINVD_OFFSET,
        "the PC points *at* the instruction the interpreter is asked to run"
    );
    assert_eq!(
        SEEN_COUNT.load(Ordering::SeqCst),
        1,
        "one instruction - the rest of the region is already compiled"
    );
}
