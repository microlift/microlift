use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::EXIT_TRAMPOLINE;
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// `inc r15` - the progress marker. Every exit instruction under test clobbers
/// some of the low registers (`CPUID` alone writes four), and `r15` is touched
/// by none of them, so a resumed region can be caught having run it.
const INC_R15: [u8; 3] = [0x49, 0xFF, 0xC7];

/// Where the exit reports RIP: past the instruction, or at it.
///
/// Not a detail of this file's plumbing - it is the exit's contract with the
/// host. `CPUID` has already happened when it exits, so RIP is past it; `INT3`
/// and the port-I/O string ops report their own address because the host is
/// expected to inspect or emulate the instruction and advance RIP itself. Both
/// kinds *resume* at the following instruction, which is the address that has to
/// be in the dispatch either way.
#[derive(Clone, Copy)]
enum ExitRip {
    Past,
    AtInstruction,
}

/// Lift `<exit instruction> ; inc r15 ; ret` under the default resumption mode
/// and check both halves of the obligation.
fn assert_exit_resumes(exit_insn: &[u8], expect: ExitReason, rip: ExitRip) {
    let mut bytes = exit_insn.to_vec();
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xC3);
    let resume = BASE + exit_insn.len() as u64;

    // The trampoline sentinel is the harness default, but a whole config passed
    // to `.config()` replaces it, and the closing `RET` has to exit with
    // `Return` for the resumed-region check to mean anything.
    let config = LifterConfig {
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..Default::default()
    };

    // Static half: the address the host will come back to is offered.
    let lifted = lift_to_code::<X86_64>(&bytes, BASE, config.clone()).unwrap();
    assert!(
        lifted.resume_points.contains(&resume),
        "{expect:?}: {:#x?} does not offer {resume:#x}, the address it resumes at",
        lifted.resume_points
    );

    // Dynamic half: re-entering there runs the instruction after the exit.
    let out = LiftTest::new(&bytes)
        .at(BASE)
        .image_base(BASE)
        .config(config)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| {
            c.rip = BASE;
            c.r15 = 0;
        })
        .run();
    assert_eq!(out.exit, expect, "first entry should exit with {expect:?}");
    match rip {
        ExitRip::Past => assert_eq!(out.ctx.rip, resume, "{expect:?} reports RIP past itself"),
        ExitRip::AtInstruction => {
            assert_eq!(out.ctx.rip, BASE, "{expect:?} reports its own address")
        }
    }

    // The host services the exit and re-enters where it resumes.
    let out = out.resume_with(|c| c.rip = resume);
    assert_ne!(
        out.exit,
        ExitReason::UnknownJump,
        "{expect:?}: {resume:#x} had no dispatch arm, so the region refused re-entry"
    );
    assert_eq!(
        out.exit,
        ExitReason::Return,
        "{expect:?}: the resumed region should run to the RET"
    );
    assert_eq!(
        out.ctx.r15, 1,
        "{expect:?}: the instruction after the exit did not run"
    );
}

#[test]
fn cpuid_resumes() {
    assert_exit_resumes(&[0x0F, 0xA2], ExitReason::Cpuid, ExitRip::Past);
}

#[test]
fn rdtsc_resumes() {
    assert_exit_resumes(&[0x0F, 0x31], ExitReason::Rdtsc, ExitRip::Past);
}

#[test]
fn xgetbv_resumes() {
    assert_exit_resumes(&[0x0F, 0x01, 0xD0], ExitReason::Xgetbv, ExitRip::Past);
}

#[test]
fn int_imm8_resumes() {
    // int 0x80
    assert_exit_resumes(&[0xCD, 0x80], ExitReason::Interrupt, ExitRip::Past);
}

#[test]
fn int3_resumes() {
    assert_exit_resumes(&[0xCC], ExitReason::Breakpoint, ExitRip::AtInstruction);
}

#[test]
fn port_input_resumes() {
    // insb - RIP at the instruction, since the host emulates the access itself.
    assert_exit_resumes(&[0x6C], ExitReason::PortInput, ExitRip::AtInstruction);
}

#[test]
fn port_output_resumes() {
    // outsb
    assert_exit_resumes(&[0x6E], ExitReason::PortOutput, ExitRip::AtInstruction);
}

/// A call that leaves the region is the other kind of departure the host cannot
/// name for itself, and it is offered for the same reason.
#[test]
fn a_call_out_of_region_resumes_at_its_return_address() {
    // call rel32 (far out of region) ; inc r15 ; ret
    let mut bytes = vec![0xE8, 0x00, 0x10, 0x00, 0x00];
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xC3);
    let resume = BASE + 5;

    let lifted = lift_to_code::<X86_64>(&bytes, BASE, LifterConfig::default()).unwrap();
    assert!(
        lifted.resume_points.contains(&resume),
        "{:#x?} lost the address the call returns to",
        lifted.resume_points
    );

    let out = LiftTest::new(&bytes)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| {
            c.rip = BASE;
            c.r15 = 0;
        })
        .run();
    assert_eq!(
        out.exit,
        ExitReason::UnknownJump,
        "the call leaves the region"
    );

    // The host runs the callee, whose own `RET` pops the return address the
    // `CALL` pushed, and re-enters at that address.
    let out = out.resume_with(|c| {
        c.rip = resume;
        c.rsp += 8;
    });
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.r15, 1);
}

/// Several exits in one region, resumed in sequence - the property a real
/// emulator actually depends on, since a translation is entered far more often
/// than it is built.
#[test]
fn a_region_resumes_through_every_exit_in_turn() {
    // rdtsc ; inc r15 ; cpuid ; inc r15 ; int3 ; inc r15 ; ret
    let mut bytes = vec![0x0F, 0x31];
    bytes.extend_from_slice(&INC_R15);
    bytes.extend_from_slice(&[0x0F, 0xA2]);
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xCC);
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xC3);

    let lifted = lift_to_code::<X86_64>(&bytes, BASE, LifterConfig::default()).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE, BASE + 2, BASE + 7, BASE + 11],
        "the entry, and where each of the three exits comes back to - nothing else"
    );

    let out = LiftTest::new(&bytes)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| {
            c.rip = BASE;
            c.r15 = 0;
        })
        .run();
    assert_eq!(out.exit, ExitReason::Rdtsc);
    assert_eq!(out.ctx.rip, BASE + 2);

    let out = out.resume();
    assert_eq!(
        out.exit,
        ExitReason::Cpuid,
        "the code between the exits is in this translation"
    );
    assert_eq!(out.ctx.r15, 1);
    assert_eq!(out.ctx.rip, BASE + 7);

    let out = out.resume();
    assert_eq!(out.exit, ExitReason::Breakpoint);
    assert_eq!(out.ctx.r15, 2);
    assert_eq!(out.ctx.rip, BASE + 10, "INT3 reports its own address");

    // The host steps past the breakpoint itself.
    let out = out.resume_with(|c| c.rip = BASE + 11);
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.r15, 3);
}

/// The other half of the mode's claim: an address control never *leaves* the
/// region at is not offered, however much of a block leader it is. This is the
/// saving the mode exists for, so it has to be checked as directly as the
/// coverage is.
#[test]
fn an_ordinary_branch_target_is_not_offered() {
    // cpuid ; jmp +0 ; inc r15 ; ret - the `jmp` target is a leader, the `cpuid`
    // resume point is the instruction after it, and only one of them is a
    // departure the host can come back from.
    let mut bytes = vec![0x0F, 0xA2, 0xEB, 0x00];
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xC3);

    let lifted = lift_to_code::<X86_64>(&bytes, BASE, LifterConfig::default()).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE, BASE + 2],
        "the entry and the cpuid's resume point, not the jmp target at {:#x}",
        BASE + 4
    );

    // And an unoffered address is refused rather than guessed at.
    let out = LiftTest::new(&bytes)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| {
            c.rip = BASE + 4;
            c.r15 = 7;
        })
        .run();
    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(out.ctx.exit_code),
        Ok(TransferKind::NoResumePoint)
    );
    assert_eq!(out.ctx.r15, 7, "no guest state touched");
}
