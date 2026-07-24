//! (c) VM-exit interaction fuzzing - drive resumable exits and assert the
//! resume invariants documented in `src/core/exit.rs`: the exit reason, where
//! `rip` points relative to the faulting instruction, that `rsp` is preserved
//! across a resumable exit, and that host-supplied register values survive the
//! resume back to the final `ret`.
//!
//! Run: `cargo test --features fuzzing --test fuzz_vmexit`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use microlift::testkit::prelude::*;
use proptest::prelude::*;

const LIFT_BASE: u64 = 0x1000;

proptest! {
    /// `cpuid` is a resumable exit: `rip` advances past it, the host fills the
    /// result registers, and those survive the resume to the final `ret`. The
    /// only stack movement across the whole cycle is the 8-byte `ret` pop -
    /// proof that `cpuid` itself left `rsp` untouched.
    #[test]
    fn cpuid_resume_cycle(
        leaf in any::<u32>(),
        fa in any::<u64>(), fb in any::<u64>(), fc in any::<u64>(), fd in any::<u64>(),
    ) {
        // cpuid ; ret
        let bytes = [0x0F, 0xA2, 0xC3];
        let first = LiftTest::new(&bytes)
            .resumable()
            .reg(|c| c.rax = leaf as u64)
            .run();

        prop_assert_eq!(first.exit, ExitReason::Cpuid);
        prop_assert_eq!(first.ctx.rip, LIFT_BASE + 2, "rip must point past cpuid");
        let rsp_at_exit = first.ctx.rsp;

        // Host services the leaf and re-enters.
        let done = first.resume_with(|c| {
            c.rax = fa; c.rbx = fb; c.rcx = fc; c.rdx = fd;
        });

        prop_assert!(
            matches!(done.exit, ExitReason::Return | ExitReason::None),
            "resumed run should reach the ret; got {:?}", done.exit
        );
        prop_assert_eq!(
            done.ctx.rsp, rsp_at_exit + 8,
            "only the ret should move rsp (by +8); cpuid must not touch it"
        );
        // The host-provided values must be intact after the resume.
        prop_assert_eq!(done.ctx.rax, fa);
        prop_assert_eq!(done.ctx.rbx, fb);
        prop_assert_eq!(done.ctx.rcx, fc);
        prop_assert_eq!(done.ctx.rdx, fd);
    }

    /// `rdtsc` is likewise resumable with `rip` past the instruction.
    #[test]
    fn rdtsc_resume_cycle(lo in any::<u32>(), hi in any::<u32>()) {
        // rdtsc ; ret
        let bytes = [0x0F, 0x31, 0xC3];
        let first = LiftTest::new(&bytes).resumable().run();
        prop_assert_eq!(first.exit, ExitReason::Rdtsc);
        prop_assert_eq!(first.ctx.rip, LIFT_BASE + 2);

        let done = first.resume_with(|c| { c.rax = lo as u64; c.rdx = hi as u64; });
        prop_assert!(matches!(done.exit, ExitReason::Return | ExitReason::None));
        prop_assert_eq!(done.ctx.rax, lo as u64);
        prop_assert_eq!(done.ctx.rdx, hi as u64);
    }

    /// `ud2` faults *at* the instruction: `rip` points to it, regardless of how
    /// much straight-line code precedes it. Fuzz the amount of leading padding.
    #[test]
    fn ud2_reports_faulting_ip(pad in 0usize..12) {
        let mut bytes = vec![0x90u8; pad]; // `pad` nops
        bytes.extend_from_slice(&[0x0F, 0x0B]); // ud2
        let out = LiftTest::new(&bytes).run();
        prop_assert_eq!(out.exit, ExitReason::Ud2);
        prop_assert_eq!(out.ctx.rip, LIFT_BASE + pad as u64, "rip must point at the ud2");
    }

    /// `int3` reports `Breakpoint` with `rip` at the trap.
    #[test]
    fn int3_reports_breakpoint(pad in 0usize..12) {
        let mut bytes = vec![0x90u8; pad];
        bytes.push(0xCC); // int3
        let out = LiftTest::new(&bytes).run();
        prop_assert_eq!(out.exit, ExitReason::Breakpoint);
        prop_assert_eq!(out.ctx.rip, LIFT_BASE + pad as u64);
    }
}
