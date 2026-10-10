//! Regions cut on a grid rather than instruction boundaries.
//! Covers [`LifterConfig::page_bounded_extent`], [`LifterConfig::region_tail_exit`], and [`LifterConfig::resumption`] / [`LifterConfig::resume_points`] / [`LifterConfig::strict_resume`].

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// `nop`, then a five-byte `mov eax, 1` with only three bytes in the region; decodes as invalid.
const TRUNCATED_MOV: [u8; 4] = [0x90, 0xB8, 0x01, 0x00];

#[test]
fn without_the_opt_in_a_cut_instruction_is_a_fault() {
    // Without the flag, a straddling `mov` kills the guest.
    let out = LiftTest::new(&TRUNCATED_MOV)
        .at(BASE)
        .image_base(BASE)
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.exit, ExitReason::InvalidInstruction);
    assert_eq!(out.ctx.rip, BASE + 1);
}

#[test]
fn a_cut_instruction_becomes_a_resumable_exit() {
    let out = LiftTest::new(&TRUNCATED_MOV)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.region_tail_exit = true)
        .reg(|c| c.rip = BASE)
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(out.ctx.exit_code),
        Ok(TransferKind::RegionTail)
    );
    assert_eq!(
        out.ctx.rip,
        BASE + 1,
        "the PC names the instruction that was not lifted, so the host can \
         lift a region starting at it"
    );
    assert_eq!(out.ctx.rax, 0, "the `mov` must not have half-executed");
}

/// Exemption that ends the loop: a region starting at the tail address must not raise the same exit at its first instruction.
#[test]
fn the_first_instruction_of_a_region_never_tail_exits() {
    // The truncated `mov` alone is the entry.
    let out = LiftTest::new(&TRUNCATED_MOV[1..])
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.region_tail_exit = true)
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(
        out.exit,
        ExitReason::InvalidInstruction,
        "a genuinely undecodable entry still has to report itself"
    );
}

/// The region must not also claim it can be entered at the tail address, or the host spins.
#[test]
fn a_cut_instruction_is_not_a_resume_point() {
    let config = LifterConfig {
        region_tail_exit: true,
        // Declaring it explicitly must not override the rule.
        resume_points: vec![BASE + 1],
        ..Default::default()
    };
    let lifted = lift_to_code::<X86_64>(&TRUNCATED_MOV, BASE, config).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE],
        "the region may be entered at its own start and nowhere else"
    );
}

/// An instruction decoding fully from in-region bytes is not a tail case.
#[test]
fn an_instruction_that_ends_exactly_at_the_cut_is_lifted() {
    // nop ; mov eax, 1 - all five bytes present.
    let code = [0x90, 0xB8, 0x01, 0x00, 0x00, 0x00];
    let out = LiftTest::new(&code)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.region_tail_exit = true)
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.ctx.rax, 1);
}

#[test]
fn a_page_bounded_extent_ends_at_the_page() {
    let mut config = LifterConfig {
        page_size: 0x4000,
        ..Default::default()
    };
    let available = 0x1_0000_0000..0x1_0010_0000;

    // From the PC to the end of its page.
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x1_0000_1234, available.clone()),
        0x1_0000_1234..0x1_0000_4000
    );
    // A smaller page size cuts sooner.
    config.page_size = 0x1000;
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x1_0000_1234, available.clone()),
        0x1_0000_1234..0x1_0000_2000
    );
}

/// An extent always holds one maximal instruction, so entering always makes progress.
#[test]
fn a_pc_at_the_end_of_a_page_takes_the_next_one_with_it() {
    let config = LifterConfig {
        page_size: 0x1000,
        ..Default::default()
    };
    let available = 0..0x1_0000;

    for pc in 0xFF0u64..0x1000 {
        let extent = config.page_bounded_extent::<X86_64>(pc, available.clone());
        assert!(
            extent.end - extent.start >= 15,
            "{pc:#x} got {extent:#x?}, too short to hold one instruction"
        );
    }
    // Well inside the page, the page is the answer.
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0xF00, available.clone()),
        0xF00..0x1000
    );
}

#[test]
fn an_extent_never_leaves_the_memory_it_was_given() {
    let config = LifterConfig {
        page_size: 0x4000,
        ..Default::default()
    };
    // Mapping ends before its page does.
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x100, 0..0x200),
        0x100..0x200
    );
    // PC outside the mapping has no extent.
    assert!(
        config
            .page_bounded_extent::<X86_64>(0x400, 0..0x200)
            .is_empty()
    );
}

/// `xor eax,eax ; jmp +0 ; inc eax ; ret` - `BASE + 4` is a leader but control never leaves there, so `ExitOnly` does not offer it.
const TWO_BLOCKS: [u8; 6] = [0x31, 0xC0, 0xEB, 0x00, 0xFF, 0xC0];

#[test]
fn declared_resume_points_are_the_whole_dispatch() {
    let lifted = lift_to_code::<X86_64>(&TWO_BLOCKS, BASE, LifterConfig::default()).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE],
        "declaring nothing still keeps the region's own entry"
    );
}

/// Legacy mode: every instruction boundary becomes an arm.
#[test]
fn the_all_mode_offers_every_instruction_boundary() {
    let config = LifterConfig {
        resumption: VmResumptionPoints::All,
        ..Default::default()
    };
    let lifted = lift_to_code::<X86_64>(&TWO_BLOCKS, BASE, config).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE, BASE + 2, BASE + 4],
        "every boundary, including the ones nothing branches to"
    );
}

/// With no dispatch the single entry is still reported; host caches consult it in every mode.
#[test]
fn the_none_mode_offers_only_the_entry() {
    let config = LifterConfig {
        resumption: VmResumptionPoints::None,
        // Declared points have no dispatch to go to.
        resume_points: vec![BASE + 4],
        ..Default::default()
    };
    let lifted = lift_to_code::<X86_64>(&TWO_BLOCKS, BASE, config).unwrap();
    assert_eq!(lifted.resume_points, vec![BASE]);
}

/// Without dispatch the PC is never read; execution starts at the top and `strict_resume` cannot change that.
#[test]
fn the_none_mode_ignores_the_entry_pc() {
    let out = LiftTest::new(&TWO_BLOCKS)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| {
            c.resumption = VmResumptionPoints::None;
            c.strict_resume = true;
        })
        .reg(|c| {
            c.rip = BASE + 4;
            c.rax = 7;
        })
        .run();
    assert_eq!(
        out.ctx.rax, 1,
        "ran from the `xor` at the top, not the `inc`"
    );
}

#[test]
fn a_declared_resume_point_can_be_entered_at() {
    let out = LiftTest::new(&TWO_BLOCKS)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| {
            c.resume_points = vec![BASE + 4];
            c.strict_resume = true;
        })
        .reg(|c| {
            c.rip = BASE + 4;
            c.rax = 7;
        })
        .run();
    assert_eq!(
        out.ctx.rax, 8,
        "entering at the `inc` must not have run the `xor` above it"
    );
}

/// Without `strict_resume` an unoffered PC silently runs from the top.
#[test]
fn an_unoffered_pc_exits_rather_than_starting_from_the_top() {
    let out = LiftTest::new(&TWO_BLOCKS)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| {
            c.rip = BASE + 4;
            c.rax = 7;
        })
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(out.ctx.exit_code),
        Ok(TransferKind::NoResumePoint)
    );
    assert_eq!(
        out.ctx.rip,
        BASE + 4,
        "the PC the host asked for, untouched"
    );
    assert_eq!(out.ctx.rax, 7, "and no guest state touched either");
}

/// `xor eax,eax ; cpuid ; inc eax ; rdtsc ; inc eax ; ret` - two resumable VM exits with code between.
const TWO_EXITS: [u8; 9] = [
    0x31, 0xC0, // 0: xor eax, eax
    0x0F, 0xA2, // 2: cpuid   -> host resumes at 4
    0xFF, 0xC0, // 4: inc eax
    0x0F, 0x31, // 6: rdtsc   -> host resumes at 8
    0xC3, // 8: ret
];

/// A `CPUID` exit resumes at the next instruction, which the engine must register as a leader itself; otherwise only `All` made it enterable.
#[test]
fn a_resumable_exit_offers_the_address_it_resumes_at() {
    let lifted = lift_to_code::<X86_64>(&TWO_EXITS, BASE, LifterConfig::default()).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE, BASE + 4, BASE + 8],
        "the entry, and where each exit comes back to - nothing else"
    );
}

/// Full round trip: two exits, two re-entries, one translation.
#[test]
fn a_region_resumes_through_its_exits() {
    let out = LiftTest::new(&TWO_EXITS)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.strict_resume = true)
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.exit, ExitReason::Cpuid);
    assert_eq!(out.ctx.rip, BASE + 4);

    // Host answers `CPUID` and re-enters where the exit said.
    let out = out.resume_with(|c| c.rax = 0);
    assert_eq!(
        out.exit,
        ExitReason::Rdtsc,
        "the `inc` and `rdtsc` after the \
         `cpuid` have to be in this translation at all"
    );
    assert_eq!(out.ctx.rip, BASE + 8);
    assert_eq!(out.ctx.rax, 1, "the `inc eax` between the two exits ran");

    let out = out.resume_with(|c| c.rax = 0);
    assert_eq!(out.exit, ExitReason::Return);
}

/// A call leaving the region must be able to return, so the engine keeps those arms whatever the host declared.
#[test]
fn call_resume_points_survive_a_narrow_declaration() {
    // call rel32 (out of region) ; nop ; ret
    let code = [0xE8, 0x00, 0x10, 0x00, 0x00, 0x90, 0xC3];
    let lifted = lift_to_code::<X86_64>(&code, BASE, LifterConfig::default()).unwrap();
    assert!(
        lifted.resume_points.contains(&(BASE + 5)),
        "{:#x?} lost the address the call returns to",
        lifted.resume_points
    );
}

/// An address the region does not hold must not be in the reported set (its exit stub would loop, not step).
#[test]
fn out_of_region_targets_are_not_reported_as_resume_points() {
    // jmp +0x40, past the six bytes of the region.
    let code = [0xE9, 0x40, 0x00, 0x00, 0x00, 0xC3];
    // `All`, so the out-of-region target gets an arm to filter.
    let config = LifterConfig {
        resumption: VmResumptionPoints::All,
        ..Default::default()
    };
    let lifted = lift_to_code::<X86_64>(&code, BASE, config).unwrap();
    let end = BASE + code.len() as u64;
    assert!(
        lifted.resume_points.iter().all(|&p| p >= BASE && p < end),
        "{:#x?} claims an address outside {BASE:#x}..{end:#x}",
        lifted.resume_points
    );
}
