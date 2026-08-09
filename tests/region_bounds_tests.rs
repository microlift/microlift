//! Regions cut on a grid rather than on instruction boundaries.
//!
//! A host that picks extents by walking the guest's control flow always ends a
//! region where an instruction ends, and can enter one wherever it likes because
//! every boundary is a resume point. Both of those are conveniences it pays for:
//! the walk is a prediction about bytes that may not be written yet, and a
//! resume point per instruction is what makes a large region unaffordable to
//! compile.
//!
//! This file covers the three mechanisms that let a host give both up:
//! [`LifterConfig::page_bounded_extent`] to choose an extent without predicting
//! anything, [`LifterConfig::region_tail_exit`] for the instruction the cut
//! lands inside, and [`LifterConfig::resumption`] /
//! [`LifterConfig::resume_points`] / [`LifterConfig::strict_resume`] for
//! entering a region at an address that was not offered.

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// `nop`, then a five-byte `mov eax, 1` - of which only three bytes are in the
/// region. The decoder cannot see the immediate, so it reports the `mov` as
/// invalid, indistinguishably from a byte that is not an instruction at all.
const TRUNCATED_MOV: [u8; 4] = [0x90, 0xB8, 0x01, 0x00];

#[test]
fn without_the_opt_in_a_cut_instruction_is_a_fault() {
    // The behaviour the flag exists to replace: an ordinary `mov` that happens
    // to straddle the end of the region kills the guest.
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

/// The exemption that makes the exit terminate instead of repeating: the host's
/// answer to a region-tail exit is a region starting at that address, and if
/// *that* region could raise the same exit at its own first instruction the two
/// would trade the same PC forever.
#[test]
fn the_first_instruction_of_a_region_never_tail_exits() {
    // The truncated `mov` alone, so it is the region's entry.
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

/// The other half of the tail exit's termination argument, and the one that is
/// easy to miss. The exit tells the host to lift a region *starting* at the
/// address - so this region must not also claim it can be entered there, or the
/// host re-enters, tail-exits again with the PC untouched, and spins.
#[test]
fn a_cut_instruction_is_not_a_resume_point() {
    let config = LifterConfig {
        region_tail_exit: true,
        // Ask for it explicitly: the address is a plausible thing for a host to
        // declare, and declaring it must not be able to override the rule.
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

/// An instruction whose decode *succeeds* using only in-region bytes is not a
/// tail case, however close to the end it starts.
#[test]
fn an_instruction_that_ends_exactly_at_the_cut_is_lifted() {
    // nop ; mov eax, 1 - all five bytes of the `mov` present.
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

    // Ordinary case: from the PC to the end of its page.
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x1_0000_1234, available.clone()),
        0x1_0000_1234..0x1_0000_4000
    );
    // The page size is the knob, and a smaller one cuts sooner.
    config.page_size = 0x1000;
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x1_0000_1234, available.clone()),
        0x1_0000_1234..0x1_0000_2000
    );
}

/// The property the whole policy rests on: an extent always holds at least one
/// maximal instruction, so entering at it always makes progress.
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
    // Well inside the page, the page is the answer and nothing is borrowed.
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
    // A mapping that ends before its page does.
    assert_eq!(
        config.page_bounded_extent::<X86_64>(0x100, 0..0x200),
        0x100..0x200
    );
    // A PC outside it has no extent at all.
    assert!(
        config
            .page_bounded_extent::<X86_64>(0x400, 0..0x200)
            .is_empty()
    );
}

/// `xor eax,eax ; jmp +0 ; inc eax ; ret` - the `jmp` makes `BASE + 4` a block
/// leader, but control never *leaves* the region there, so the default
/// [`VmResumptionPoints::ExitOnly`] does not offer it.
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

/// The legacy mode, and the reason it costs what it does: every instruction
/// boundary in the region becomes an arm, branch target or not.
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

/// With no dispatch there is exactly one way in, and the reported set says so
/// rather than staying empty - a host cache consults it in every mode.
#[test]
fn the_none_mode_offers_only_the_entry() {
    let config = LifterConfig {
        resumption: VmResumptionPoints::None,
        // Declared points have nowhere to go without a dispatch.
        resume_points: vec![BASE + 4],
        ..Default::default()
    };
    let lifted = lift_to_code::<X86_64>(&TWO_BLOCKS, BASE, config).unwrap();
    assert_eq!(lifted.resume_points, vec![BASE]);
}

/// And it really is a jump straight into the region: a PC the host set is not
/// read at all, so execution starts at the top whatever it says. That is the
/// mode's whole bargain, and `strict_resume` cannot soften it - there is no
/// default arm to replace.
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

/// The failure this replaces is the dangerous one: without `strict_resume` an
/// unoffered PC runs the region from the top, with no fault and no diagnostic.
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

/// `xor eax,eax ; cpuid ; inc eax ; rdtsc ; inc eax ; ret`
///
/// Two resumable VM exits with ordinary code between and after them.
const TWO_EXITS: [u8; 9] = [
    0x31, 0xC0, // 0: xor eax, eax
    0x0F, 0xA2, // 2: cpuid   -> host resumes at 4
    0xFF, 0xC0, // 4: inc eax
    0x0F, 0x31, // 6: rdtsc   -> host resumes at 8
    0xC3, // 8: ret
];

/// The gap this file's third mechanism was really about.
///
/// A `CPUID` exit says "resume at the next instruction", and until the engine
/// registered that address itself, nothing made it a block leader: the exit
/// terminated its block, phase 2 stopped there, and everything after the
/// `CPUID` was neither lifted nor enterable. [`VmResumptionPoints::All`] covered
/// it by making *every* instruction a resume point, which is why that mode
/// looked mandatory — and is exactly the cost this file exists to avoid paying.
#[test]
fn a_resumable_exit_offers_the_address_it_resumes_at() {
    let lifted = lift_to_code::<X86_64>(&TWO_EXITS, BASE, LifterConfig::default()).unwrap();
    assert_eq!(
        lifted.resume_points,
        vec![BASE, BASE + 4, BASE + 8],
        "the entry, and where each exit comes back to — nothing else"
    );
}

/// And the whole round trip actually runs: two exits, two re-entries, with the
/// code between them lifted into the same translation.
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

    // The host answers the `CPUID` and re-enters where the exit said to.
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

/// A call that leaves the region has to be able to come back, and the host
/// cannot name its resume point without decoding the region itself. So the
/// engine keeps those arms whatever the host declared.
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

/// The reported set is what a host's code cache keys on, so an address the
/// region does not hold must not be in it. Such an address *does* get a switch
/// arm - pointing at the exit stub `finish` builds for it - and entering there
/// executes nothing and leaves the PC alone, which is a loop rather than a step.
#[test]
fn out_of_region_targets_are_not_reported_as_resume_points() {
    // jmp +0x40, well past the six bytes of the region.
    let code = [0xE9, 0x40, 0x00, 0x00, 0x00, 0xC3];
    // `All`, so that the out-of-region target gets a switch arm at all - the
    // filter this is about only has something to filter under that mode.
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
