//! Regions with holes in them: [`LifterConfig::block_ranges`].
//!
//! A host that discovers a region by following the guest's control flow finds a
//! *set of basic blocks*, and between them lie bytes no block reaches — a jump
//! table in the middle of a function, alignment padding, the inline argument a
//! trampoline reads through its own return address. Naming the blocks is what
//! keeps those bytes out of the lift; without it the interval spanning the
//! blocks is the lift, and the holes are decoded as whatever they happen to
//! mean.
//!
//! What the ranges promise is narrow and worth stating: an address outside them
//! is treated exactly as an address past the end of the slice would be. Not
//! lifted, not a leader, and an edge into it is an ordinary out-of-region exit
//! for the host to answer with a region of its own.

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, VmResumptionPoints, lift_to_code};
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// The instruction addresses a lift covered.
///
/// [`VmResumptionPoints::All`] makes every decoded instruction a block leader
/// and reports the in-region ones, so the returned resume points are exactly
/// the addresses this lift decided were code. That is a heavier mode than any
/// of these tests would otherwise ask for, and it is used precisely because it
/// is the one mode whose output enumerates the answer rather than implying it.
fn lifted_addresses(bytes: &[u8], ip: u64, mut config: LifterConfig) -> Vec<u64> {
    config.resumption = VmResumptionPoints::All;
    let lifted = lift_to_code::<X86_64>(bytes, ip, config).expect("lift failed");
    let mut addrs = lifted.resume_points;
    addrs.sort_unstable();
    addrs.dedup();
    addrs
}

/// `jmp +4` over four bytes of table, then `nop ; mov eax,1 ; ret`.
///
/// The four table bytes are `add [rax],al` twice to a linear decoder, which is
/// exactly the shape that makes an embedded jump table look like code. The
/// `nop` is there so the `mov` is not the second block's *first* instruction;
/// see [`an_instruction_straddling_a_hole_tail_exits`].
const OVER_A_TABLE_NOP: [u8; 13] = [
    0xEB, 0x04, // jmp +4  -> BASE+6
    0x00, 0x00, 0x00, 0x00, // the table
    0x90, // nop
    0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
    0xC3, // ret
];

/// `jmp +4` over the same table, then `mov eax,1 ; ret`.
const OVER_A_TABLE_FULL: [u8; 12] = [
    0xEB, 0x04, // jmp +4  -> BASE+6
    0x00, 0x00, 0x00, 0x00, // the table
    0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
    0xC3, // ret
];

#[test]
fn without_ranges_the_hole_between_two_blocks_is_lifted() {
    // The behaviour the field exists to replace. Nothing here is *wrong* — the
    // engine was told these bytes are code, so it lifted them — but the four
    // table bytes cost IR, and a table whose bytes happened to decode into a
    // branch would have cost a block as well.
    let addrs = lifted_addresses(&OVER_A_TABLE_FULL, BASE, LifterConfig::default());
    assert!(
        addrs.contains(&(BASE + 2)),
        "{addrs:#x?} should include the table, since nothing said otherwise"
    );
}

#[test]
fn a_named_hole_is_not_lifted() {
    let config = LifterConfig {
        block_ranges: vec![BASE..BASE + 2, BASE + 6..BASE + 12],
        ..Default::default()
    };
    let addrs = lifted_addresses(&OVER_A_TABLE_FULL, BASE, config);
    assert_eq!(
        addrs,
        vec![BASE, BASE + 6, BASE + 11],
        "only the two blocks, and every instruction in them"
    );
}

/// The property that makes holes safe rather than merely cheap: control that
/// reaches one leaves, exactly as it would at the end of the slice.
#[test]
fn a_branch_into_a_hole_exits_rather_than_running_it() {
    // jmp +2 lands in the middle of the named hole.
    // BASE:   jmp +2      -> BASE+4
    // BASE+2: (hole)
    // BASE+4: (hole)
    // BASE+6: mov eax,1 ; ret
    let code = [
        0xEB, 0x02, // jmp -> BASE+4, which is a hole
        0x90, 0x90, 0x90, 0x90, // padding, of which BASE+2..BASE+6 is the hole
        0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3,
    ];
    let out = LiftTest::new(&code)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.block_ranges = vec![BASE..BASE + 2, BASE + 6..BASE + 12])
        .reg(|c| c.rip = BASE)
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        out.ctx.rip,
        BASE + 4,
        "the PC names the hole, so the host can decide what is there"
    );
    assert_eq!(out.ctx.rax, 0, "nothing in the hole may have executed");
}

/// A hole ends a block the same way the end of the slice does, so a block that
/// falls off its own end leaves rather than running into the next one.
#[test]
fn falling_out_of_a_block_exits_at_the_hole() {
    // Two `nop`s with no terminator, then a hole, then more code. Without the
    // ranges the `nop`s fall straight into what follows.
    let code = [0x90, 0x90, 0x90, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let out = LiftTest::new(&code)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| c.block_ranges = vec![BASE..BASE + 2, BASE + 4..BASE + 10])
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.ctx.rip, BASE + 2, "control stops where the block does");
    assert_eq!(out.ctx.rax, 0);
}

/// Two ranges that abut are one run of code, not two, so an instruction may
/// span the join. Merging them on the way in is what makes that true — leaving
/// them separate would cut the `mov` at `BASE + 2` in half.
#[test]
fn touching_ranges_are_one_range() {
    let code = [0x90, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let config = LifterConfig {
        block_ranges: vec![BASE..BASE + 4, BASE + 4..BASE + 8],
        ..Default::default()
    };
    let addrs = lifted_addresses(&code, BASE, config);
    assert_eq!(addrs, vec![BASE, BASE + 1, BASE + 2, BASE + 7]);
}

/// Unsorted, overlapping and out-of-bounds input is normalised rather than
/// rejected, because the host builds this list block by block and sorting it is
/// not the host's problem.
#[test]
fn ranges_are_normalised_not_trusted() {
    let code = [0x90, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let config = LifterConfig {
        block_ranges: vec![
            BASE + 4..BASE + 64,    // runs off the end of the slice
            BASE..BASE + 5,         // out of order, and overlaps the first
            BASE + 900..BASE + 999, // entirely outside
        ],
        ..Default::default()
    };
    let addrs = lifted_addresses(&code, BASE, config);
    assert_eq!(
        addrs,
        vec![BASE, BASE + 1, BASE + 2, BASE + 7],
        "the whole slice, arrived at from three malformed ranges"
    );
}

/// An empty list is the old behaviour, which is what every existing caller
/// relies on.
#[test]
fn no_ranges_means_the_whole_slice() {
    let code = [0x90, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    assert_eq!(
        lifted_addresses(&code, BASE, LifterConfig::default()),
        lifted_addresses(
            &code,
            BASE,
            LifterConfig {
                block_ranges: vec![BASE..BASE + 8],
                ..Default::default()
            }
        )
    );
}

/// A resume point in a hole is not a resume point, for the same reason one
/// outside the slice is not: there is no block behind it to enter.
#[test]
fn a_declared_resume_point_inside_a_hole_is_ignored() {
    let code = [0x90, 0x90, 0x90, 0x90, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let lifted = lift_to_code::<X86_64>(
        &code,
        BASE,
        LifterConfig {
            block_ranges: vec![BASE..BASE + 2, BASE + 4..BASE + 10],
            resume_points: vec![BASE + 3, BASE + 4],
            ..Default::default()
        },
    )
    .expect("lift failed");
    assert!(!lifted.resume_points.contains(&(BASE + 3)));
    assert!(lifted.resume_points.contains(&(BASE + 4)));
}

/// The truncation rule follows the block, not the slice: an instruction that
/// begins in a block and ends in a hole has the same problem as one that runs
/// off the end of the buffer, and gets the same resumable exit.
#[test]
fn an_instruction_straddling_a_hole_tail_exits() {
    let out = LiftTest::new(&OVER_A_TABLE_NOP)
        .at(BASE)
        .image_base(BASE)
        // The block is BASE+6..BASE+11: the `nop`, and four of the `mov`'s five
        // bytes.
        .with_config(|c| {
            c.block_ranges = vec![BASE..BASE + 2, BASE + 6..BASE + 11];
            c.region_tail_exit = true;
        })
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(out.ctx.exit_code),
        Ok(TransferKind::RegionTail)
    );
    assert_eq!(
        out.ctx.rip,
        BASE + 7,
        "the PC names the instruction the block could not hold"
    );
    assert_eq!(out.ctx.rax, 0, "the `mov` must not have half-executed");
}

/// And the exemption that makes that exit terminate follows the block too: the
/// host's answer to a region-tail exit is a region starting at the address, so
/// a block's own first instruction must never raise one — there would be
/// nowhere shorter to go.
#[test]
fn the_first_instruction_of_a_block_never_tail_exits() {
    // `jmp +5` lands on the `mov`, which is now the second block's first
    // instruction and still four of its five bytes long.
    let mut code = OVER_A_TABLE_NOP;
    code[1] = 0x05;
    let out = LiftTest::new(&code)
        .at(BASE)
        .image_base(BASE)
        .with_config(|c| {
            c.block_ranges = vec![BASE..BASE + 2, BASE + 7..BASE + 11];
            c.region_tail_exit = true;
        })
        .reg(|c| c.rip = BASE)
        .run();
    assert_eq!(out.exit, ExitReason::InvalidInstruction);
    assert_eq!(out.ctx.rip, BASE + 7);
}
