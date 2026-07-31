//! Resumable VM exits: an instruction the lifter does not implement exits to
//! the host with `UnimplementedInstruction`, the host shims it, and re-entering
//! the same compiled region resumes at the following instruction.

use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;

/// `WBINVD` (`0F 09`) stands in for "an instruction the lifter will never
/// implement": it is ring-0 only, so it cannot appear in the usermode guest code
/// microlift targets, and it will keep falling through to the
/// interpreter-fallback path.
const WBINVD: [u8; 2] = [0x0F, 0x09];

/// Offset of `inc rax` in the test program below: 7 bytes of `mov rax, 42` plus
/// the 2-byte `WBINVD`. The fallback exit reports this as the resume point.
const RESUME_OFFSET: u64 = 9;

#[test]
fn test_resumable_unimplemented_fallback() {
    let mut bytes = vec![0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]; // mov rax, 42
    bytes.extend_from_slice(&WBINVD); // unimplemented
    bytes.extend_from_slice(&[0x48, 0xFF, 0xC0]); // inc rax
    bytes.push(0xC3); // ret

    // First entry: runs `mov`, then exits at the unimplemented instruction with
    // the guest state synced out.
    let out = LiftTest::new(&bytes).resumable().run();
    assert_eq!(
        out.exit,
        ExitReason::UnimplementedInstruction,
        "wbinvd must produce a resumable fallback exit"
    );
    assert_eq!(out.ctx.rax, 42, "state before the exit must be synced out");
    assert_eq!(
        out.ctx.rip,
        DEFAULT_LIFT_BASE + RESUME_OFFSET,
        "rip must be the resume point: the instruction after the wbinvd"
    );
    assert_ne!(
        out.ctx.exit_code, 0,
        "exit_code must carry the instruction code so the host can shim it"
    );

    // The host shims the instruction (a cache flush is a no-op here) and
    // re-enters: execution must resume at `inc rax` and run through to the
    // `RET`, which returns to the exit trampoline.
    let out = out.resume();
    assert!(
        matches!(out.exit, ExitReason::Return | ExitReason::None),
        "resumed region should run to the RET, got {:?}",
        out.exit
    );
    assert_eq!(
        out.ctx.rax, 43,
        "the instruction after the exit must execute"
    );
}
