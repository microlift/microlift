//! [`LifterConfig::block_ranges`]: bytes outside the ranges are treated like bytes past the end of the slice.
//! Not lifted, not a leader; an edge into them is an ordinary out-of-region exit.

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, VmResumptionPoints, lift_to_code};
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// Instruction addresses a lift covered, enumerated via `VmResumptionPoints::All`.
fn lifted_addresses(bytes: &[u8], ip: u64, mut config: LifterConfig) -> Vec<u64> {
    config.resumption = VmResumptionPoints::All;
    let lifted = lift_to_code::<X86_64>(bytes, ip, config).expect("lift failed");
    let mut addrs = lifted.resume_points;
    addrs.sort_unstable();
    addrs.dedup();
    addrs
}

/// `jmp +4` over a 4-byte table, then `nop ; mov eax,1 ; ret`; the `nop` keeps the `mov` off block start.
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

/// Control reaching a hole leaves, as at the end of the slice.
#[test]
fn a_branch_into_a_hole_exits_rather_than_running_it() {
    // jmp +2 lands inside the hole.
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

/// A hole ends a block like the end of the slice does.
#[test]
fn falling_out_of_a_block_exits_at_the_hole() {
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

/// Abutting ranges merge, so an instruction may span the join.
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

/// Unsorted, overlapping and out-of-bounds ranges are normalised, not rejected.
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

/// A resume point in a hole is dropped.
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

/// An instruction truncated by a hole gets the same resumable exit as one truncated by the slice.
#[test]
fn an_instruction_straddling_a_hole_tail_exits() {
    let out = LiftTest::new(&OVER_A_TABLE_NOP)
        .at(BASE)
        .image_base(BASE)
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

/// A block's first instruction is exempt from the region-tail exit, or it could never terminate.
#[test]
fn the_first_instruction_of_a_block_never_tail_exits() {
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
