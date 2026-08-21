//! Every block-linking mechanism is inert unless the guest opted in by layout.
//!
//! There are now four of them - the table probe, link cells, the call-return
//! ring, and pinned dispatch - and each is gated below
//! [`GuestArch::block_link_offsets`], which aarch64 does not override. Rather
//! than four scattered "is this inert" arguments, this is the one mechanical
//! check: lift the same aarch64 program with everything on and everything off,
//! and compare the emitted IR **byte for byte**.
//!
//! Doing it on the guest that opted *out* is the point. An x86-64 lift with the
//! switches off proves the switches are off; an aarch64 lift with them all on
//! proves the gate holds, which is the property a frontend author needs and the
//! one that silently rots.

use microlift::arch::aarch64::Aarch64;
use microlift::arch::x86_64::X86_64;
use microlift::core::callret::CallRetConfig;
use microlift::core::link::BlockLinkConfig;
use microlift::core::pin::{PinnedAbi, PinnedReg};
use microlift::core::{LifterConfig, lift_to_code};

const BASE: u64 = 0x1000;

/// `mov x0, #42 ; ret`
const AARCH64_PROGRAM: &[u8] = &[0x40, 0x05, 0x80, 0xD2, 0xC0, 0x03, 0x5F, 0xD6];

/// Everything this port added, turned all the way up.
fn everything_on() -> LifterConfig {
    let abi = PinnedAbi::new().with(PinnedReg::gpr(0)).with(PinnedReg::gpr(1));
    LifterConfig {
        block_link: Some(BlockLinkConfig::new(256).pinned(abi)),
        link_cells: true,
        call_return_stack: Some(CallRetConfig::new(16)),
        ..LifterConfig::default()
    }
}

#[test]
fn a_guest_that_did_not_opt_in_emits_identical_ir_either_way() {
    let off = lift_to_code::<Aarch64>(AARCH64_PROGRAM, BASE, LifterConfig::default()).unwrap();
    let on = lift_to_code::<Aarch64>(AARCH64_PROGRAM, BASE, everything_on()).unwrap();

    assert_eq!(
        on.func.display().to_string(),
        off.func.display().to_string(),
        "aarch64 supplies no block-link offsets, so all four mechanisms must be \
         inert - not merely harmless"
    );
    assert!(on.link_sites.is_empty());
    assert!(
        on.entry_thunk.is_none(),
        "and pinning is off too, since it is gated the same way"
    );
}

#[test]
fn the_gate_is_the_guest_layout_and_not_the_config() {
    // The control: the *same* config on a guest that did opt in must change
    // something. Without this, a lifter that ignored all four switches
    // everywhere would pass the test above.
    let program: &[u8] = &[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    let off = lift_to_code::<X86_64>(program, BASE, LifterConfig::default()).unwrap();
    let on = lift_to_code::<X86_64>(program, BASE, everything_on()).unwrap();

    assert_ne!(
        on.func.display().to_string(),
        off.func.display().to_string(),
        "x86-64 does supply the offsets, so the same config must do something"
    );
    assert!(on.entry_thunk.is_some(), "and it pins");
}

#[test]
fn each_switch_is_inert_on_its_own_without_block_linking() {
    // Cells and the ring are part of block linking, not alternatives to it, so
    // each must be inert on its own - checked per switch rather than only in
    // combination, since a gate that consults the wrong one would still pass a
    // both-on test.
    let program: &[u8] = &[
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0xE9, 0x00, 0x10, 0x00, 0x00, // jmp out of region
    ];
    let baseline = lift_to_code::<X86_64>(program, BASE, LifterConfig::default())
        .unwrap()
        .func
        .display()
        .to_string();

    for (name, config) in [
        (
            "cells without a table",
            LifterConfig {
                link_cells: true,
                ..LifterConfig::default()
            },
        ),
        (
            "a ring without cells",
            LifterConfig {
                block_link: Some(BlockLinkConfig::new(256)),
                call_return_stack: Some(CallRetConfig::new(16)),
                ..LifterConfig::default()
            },
        ),
    ] {
        let code = lift_to_code::<X86_64>(program, BASE, config).unwrap();
        assert!(
            code.link_sites.is_empty(),
            "{name} should emit no cells, got {:?}",
            code.link_sites
        );
        if name == "cells without a table" {
            assert_eq!(
                code.func.display().to_string(),
                baseline,
                "{name} should change nothing at all"
            );
        }
    }
}

#[test]
fn analysis_mode_emits_none_of_it() {
    // Analysis IR is inspection output that is never executed, and every
    // mechanism here writes to the context. `block_link` already excludes it;
    // this is what keeps a future mechanism from forgetting.
    let program: &[u8] = &[
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xE9, 0x00, 0x10, 0x00, 0x00,
    ];
    let config = LifterConfig {
        analysis_mode: true,
        ..everything_on()
    };
    let code = lift_to_code::<X86_64>(program, BASE, config).unwrap();

    assert!(code.link_sites.is_empty());
    assert!(code.entry_thunk.is_none());
    let ir = code.func.display().to_string();
    assert!(!ir.contains("symbol_value"), "no cell array is referenced");
}

/// Two lifts of the same bytes under the same config must produce the same IR
/// **and** the same cell array.
///
/// `two_lifts_of_the_same_bytes_compile_to_the_same_bytes` already guards the
/// compiled machine code. This guards a step earlier, and it is a distinct
/// property: block creation walked a `HashSet` of leaders and exit-stub emission
/// walked a `HashMap`, both of which are seeded per instance, so the IR was
/// reordered on every lift while the machine code came out the same anyway.
///
/// That stopped being cosmetic when cells arrived. `link_cell_for` hands out
/// indices in emission order, so a permuted emission order permutes the cell
/// array - and an AOT artifact whose cell indices move between builds cannot be
/// cached.
#[test]
fn two_lifts_of_the_same_bytes_produce_the_same_ir_and_the_same_cells() {
    // Eight distinct out-of-region targets, so there is an order to get wrong.
    let mut program = Vec::new();
    for i in 0..8i32 {
        program.extend_from_slice(&[0x0F, 0x84]);
        program.extend_from_slice(&(0x10000 + i * 0x100).to_le_bytes());
    }
    program.push(0xC3);

    let config = || LifterConfig {
        block_link: Some(BlockLinkConfig::new(256)),
        link_cells: true,
        ..LifterConfig::default()
    };
    let first = lift_to_code::<X86_64>(&program, BASE, config()).unwrap();
    let want_ir = first.func.display().to_string();
    let want_cells: Vec<u64> = first.link_sites.iter().map(|s| s.target).collect();
    assert_eq!(want_cells.len(), 8, "the fixture must exercise cell ordering");

    // Repeated, not doubled: the seed differs per `HashMap` instance, so two
    // adjacent lifts can agree by luck. Twenty-five cannot.
    for i in 0..25 {
        let again = lift_to_code::<X86_64>(&program, BASE, config()).unwrap();
        assert_eq!(again.func.display().to_string(), want_ir, "lift {i}");
        assert_eq!(
            again.link_sites.iter().map(|s| s.target).collect::<Vec<_>>(),
            want_cells,
            "lift {i} permuted the cell array"
        );
    }
}
