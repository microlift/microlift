//! Block-linking mechanisms (probe, cells, call-return ring, pinned dispatch) are inert on guests that do not override `block_link_offsets`.
//! Lifts aarch64 with everything on vs off and compares IR byte for byte.

use microlift::arch::aarch64::Aarch64;
use microlift::arch::x86_64::X86_64;
use microlift::core::callret::CallRetConfig;
use microlift::core::hostframe::HostFrameConfig;
use microlift::core::link::BlockLinkConfig;
use microlift::core::pin::{PinnedAbi, PinnedReg};
use microlift::core::{LifterConfig, lift_to_code};

const BASE: u64 = 0x1000;

/// `mov x0, #42 ; ret`
const AARCH64_PROGRAM: &[u8] = &[0x40, 0x05, 0x80, 0xD2, 0xC0, 0x03, 0x5F, 0xD6];

/// All mechanisms turned on.
fn everything_on() -> LifterConfig {
    let abi = PinnedAbi::new().with(PinnedReg::gpr(0)).with(PinnedReg::gpr(1));
    LifterConfig {
        block_link: Some(BlockLinkConfig::new(256).pinned(abi)),
        link_cells: true,
        call_return_stack: Some(CallRetConfig::new(16)),
        host_call_frames: Some(HostFrameConfig::new(64)),
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
    // Control: the same config on an opted-in guest must change the IR.
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
    // Each switch is checked alone; a gate consulting the wrong one would pass a both-on test.
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
        (
            "host frames without cells",
            LifterConfig {
                block_link: Some(BlockLinkConfig::new(256)),
                host_call_frames: Some(HostFrameConfig::new(64)),
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
    // Analysis IR is never executed; no mechanism may write the context.
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
    assert!(!ir.contains("call_indirect"), "and nothing nests");
}

/// Two lifts of the same bytes yield the same IR and cell array.
/// Cell indices follow emission order, so a permuted order would break AOT caching.
#[test]
fn two_lifts_of_the_same_bytes_produce_the_same_ir_and_the_same_cells() {
    // Eight out-of-region targets give an order to get wrong.
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

    // Repeated: HashMap seeds differ per instance, so adjacent lifts can agree by luck.
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
