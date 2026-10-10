use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::EXIT_TRAMPOLINE;
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// Progress marker: no exit instruction under test clobbers r15.
const INC_R15: [u8; 3] = [0x49, 0xFF, 0xC7];

/// Where the exit reports RIP: CPUID is past the instruction; INT3 and port string ops are at it. Both resume at the next instruction.
#[derive(Clone, Copy)]
enum ExitRip {
    Past,
    AtInstruction,
}

/// Lift `<exit insn> ; inc r15 ; ret`; the resume point must be offered and re-entering it must run the `inc`.
fn assert_exit_resumes(exit_insn: &[u8], expect: ExitReason, rip: ExitRip) {
    let mut bytes = exit_insn.to_vec();
    bytes.extend_from_slice(&INC_R15);
    bytes.push(0xC3);
    let resume = BASE + exit_insn.len() as u64;

    // A passed config replaces the harness sentinel; the closing RET must still exit `Return`.
    let config = LifterConfig {
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..Default::default()
    };

    let lifted = lift_to_code::<X86_64>(&bytes, BASE, config.clone()).unwrap();
    assert!(
        lifted.resume_points.contains(&resume),
        "{expect:?}: {:#x?} does not offer {resume:#x}, the address it resumes at",
        lifted.resume_points
    );

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
    assert_exit_resumes(&[0xCD, 0x80], ExitReason::Interrupt, ExitRip::Past);
}

#[test]
fn int3_resumes() {
    assert_exit_resumes(&[0xCC], ExitReason::Breakpoint, ExitRip::AtInstruction);
}

#[test]
fn port_input_resumes() {
    // insb: RIP at the instruction, host emulates the access.
    assert_exit_resumes(&[0x6C], ExitReason::PortInput, ExitRip::AtInstruction);
}

#[test]
fn port_output_resumes() {
    assert_exit_resumes(&[0x6E], ExitReason::PortOutput, ExitRip::AtInstruction);
}

#[test]
fn a_call_out_of_region_resumes_at_its_return_address() {
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

    // The host re-enters at the pushed return address.
    let out = out.resume_with(|c| {
        c.rip = resume;
        c.rsp += 8;
    });
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.r15, 1);
}

/// Several exits in one region resumed in sequence.
#[test]
fn a_region_resumes_through_every_exit_in_turn() {
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

    let out = out.resume_with(|c| c.rip = BASE + 11);
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.r15, 3);
}

/// A leader that control never leaves the region at is not offered.
#[test]
fn an_ordinary_branch_target_is_not_offered() {
    // The `jmp` target is a leader; only the cpuid resume point is a departure.
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

    // Unoffered addresses are refused.
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
