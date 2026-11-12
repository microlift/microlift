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

thread_local! {
    /// `(offset of the PC, instruction count)` per interpreter call on this thread.
    static RUNS: std::cell::RefCell<Vec<(u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Treats every instruction it is given as a `WBINVD` no-op.
unsafe extern "C" fn interpret_wbinvd_run(ctx: *mut RegisterContext, num_instructions: u64) {
    let ctx = unsafe { &mut *ctx };
    RUNS.with(|runs| {
        runs.borrow_mut()
            .push((ctx.rip - DEFAULT_LIFT_BASE, num_instructions))
    });
    ctx.rip += num_instructions * WBINVD.len() as u64;
}

/// Runs `bytes` with `rax` preset and returns the interpreter calls it made.
fn interpreted_runs(bytes: &[u8], rax: u64) -> Vec<(u64, u64)> {
    RUNS.with(|runs| runs.borrow_mut().clear());
    let out = LiftTest::new(bytes)
        .no_unicorn()
        .with_config(|c| c.interpreter_fallback = true)
        .reg(|c| {
            c.rax = rax;
            c.callbacks = X86_64Callbacks {
                interpreter_fallback: interpret_wbinvd_run,
                ..X86_64Callbacks::default()
            }
        })
        .run();
    assert_eq!(out.exit, ExitReason::Return, "the region ran to its RET");
    RUNS.with(|runs| runs.borrow().clone())
}

#[test]
fn consecutive_unimplemented_instructions_share_one_callback() {
    let mut bytes = WBINVD.repeat(3);
    bytes.push(0xC3); // ret

    assert_eq!(interpreted_runs(&bytes, 0), vec![(0, 3)]);
}

#[test]
fn a_branch_target_splits_an_interpreted_run() {
    let mut bytes = vec![0x48, 0x85, 0xC0]; // test rax, rax
    bytes.extend_from_slice(&[0x74, 0x02]); // jz over the first WBINVD
    bytes.extend_from_slice(&WBINVD);
    bytes.extend_from_slice(&WBINVD); // the jump target
    bytes.push(0xC3); // ret

    assert_eq!(
        interpreted_runs(&bytes, 1),
        vec![(5, 1), (7, 1)],
        "falling through enters the target as its own run"
    );
    assert_eq!(interpreted_runs(&bytes, 0), vec![(7, 1)]);
}

#[test]
fn a_lifted_instruction_ends_an_interpreted_run() {
    let mut bytes = WBINVD.repeat(2);
    bytes.extend_from_slice(&[0x48, 0xFF, 0xC0]); // inc rax
    bytes.extend_from_slice(&WBINVD);
    bytes.push(0xC3); // ret

    assert_eq!(interpreted_runs(&bytes, 0), vec![(0, 2), (7, 1)]);
}
