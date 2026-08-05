//! Guest source locations: every lifted IR instruction is tagged with the guest
//! instruction it came from, and that tag survives into the compiled code.

use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::srcloc::{GuestCodeMap, SourceLocMap};
use microlift::core::{AddressMode, LifterConfig, lift_to_code};
use microlift::testkit::{DEFAULT_LIFT_BASE, prelude::*};
use std::collections::BTreeSet;

/// `xor eax, eax ; add eax, 7 ; imul eax, eax, 3 ; ret`, and the VA of each
/// instruction when lifted at `base`.
const PROGRAM: &[u8] = &[
    0x31, 0xC0, // 0: xor eax, eax
    0x83, 0xC0, 0x07, // 2: add eax, 7
    0x6B, 0xC0, 0x03, // 5: imul eax, eax, 3
    0xC3, // 8: ret
];
const PROGRAM_OFFSETS: &[u64] = &[0, 2, 5, 8];

fn program_vas(base: u64) -> BTreeSet<u64> {
    PROGRAM_OFFSETS.iter().map(|off| base + off).collect()
}

/// The guest VA tagged onto every IR instruction in `func`, decoded with `map`.
/// `None` covers the untagged scaffolding (prologue, PC dispatch).
fn ir_source_locs(
    func: &cranelift_codegen::ir::Function,
    map: SourceLocMap,
) -> Vec<(cranelift_codegen::ir::Inst, Option<u64>)> {
    func.layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
        .map(|inst| (inst, map.decode(func.srcloc(inst))))
        .collect()
}

#[test]
fn ir_instructions_carry_their_guest_address() {
    let base = DEFAULT_LIFT_BASE;
    let code = lift_to_code::<X86_64>(PROGRAM, base, LifterConfig::default()).unwrap();

    let locs = ir_source_locs(&code.func, code.source_map);
    let tagged: BTreeSet<u64> = locs.iter().filter_map(|&(_, va)| va).collect();

    // Every guest instruction contributed IR, and nothing was attributed to an
    // address the decoder never saw.
    assert_eq!(tagged, program_vas(base), "tagged guest addresses");

    // The overwhelming majority of the IR belongs to a guest instruction; only
    // the entry scaffolding is untagged.
    let untagged = locs.iter().filter(|(_, va)| va.is_none()).count();
    assert!(
        untagged * 2 < locs.len(),
        "expected most IR to carry a source location, {untagged} of {} did not",
        locs.len()
    );
}

#[test]
fn source_locations_track_the_lift_base() {
    // The encoding is relative to the lift base, so a region lifted somewhere
    // else must report the addresses *it* was lifted at, not offsets.
    let base = 0x7F00_1234_0000;
    let code = lift_to_code::<X86_64>(PROGRAM, base, LifterConfig::default()).unwrap();
    assert_eq!(code.source_map.lift_base, base);

    let tagged: BTreeSet<u64> = ir_source_locs(&code.func, code.source_map)
        .into_iter()
        .filter_map(|(_, va)| va)
        .collect();
    assert_eq!(tagged, program_vas(base));
}

#[test]
fn disabling_source_locs_leaves_the_ir_untagged() {
    let mut config = LifterConfig::default();
    config.emit_source_locs = false;
    let code = lift_to_code::<X86_64>(PROGRAM, DEFAULT_LIFT_BASE, config).unwrap();

    let locs = ir_source_locs(&code.func, code.source_map);
    assert!(!locs.is_empty());
    assert!(
        locs.iter().all(|(_, va)| va.is_none()),
        "no IR instruction should carry a location when the knob is off"
    );
}

#[test]
fn source_locations_do_not_change_behavior() {
    // A `SourceLoc` is opaque metadata to Cranelift; turning it off must not
    // change a single guest-visible bit.
    let with = LiftTest::new(PROGRAM).run();
    let without = LiftTest::new(PROGRAM)
        .with_config(|c| c.emit_source_locs = false)
        .run();

    assert_eq!(with.ctx.rax, 21);
    assert_eq!(with.ctx.rax, without.ctx.rax);
    assert_eq!(
        (with.ctx.cf, with.ctx.zf, with.ctx.sf, with.ctx.of),
        (
            without.ctx.cf,
            without.ctx.zf,
            without.ctx.sf,
            without.ctx.of
        )
    );
    assert_eq!(with.exit, without.exit);
}

#[test]
fn compiled_code_maps_back_to_guest_instructions() {
    let out = LiftTest::new(PROGRAM).run();
    let map = out.code_map();

    assert!(!map.is_empty(), "compiled region should carry a code map");

    // Ranges are sorted, non-empty and disjoint - `lookup`'s binary search
    // depends on all three.
    let mut prev_end = 0;
    for range in map.ranges() {
        assert!(range.start >= prev_end, "ranges overlap or are unsorted");
        assert!(range.start < range.end, "empty range in map");
        prev_end = range.end;
    }

    // Every surviving guest address is one the decoder saw. It is a subset, not
    // an equality: an instruction whose whole result is dead can be optimized
    // away entirely, and `xor eax, eax` folding into the constant that follows
    // it is exactly the kind of thing that happens here.
    let guests: BTreeSet<u64> = map.guest_addresses().into_iter().collect();
    assert!(!guests.is_empty());
    assert!(
        guests.is_subset(&program_vas(DEFAULT_LIFT_BASE)),
        "map named an address the decoder never saw: {guests:x?}"
    );

    // Every byte of every range resolves to that range's guest instruction, and
    // the same answer comes back through an absolute host address.
    for range in map.ranges() {
        for offset in range.start..range.end {
            assert_eq!(map.lookup(offset), Some(range.guest_va));
            let host = out.code_ptr() as u64 + offset as u64;
            assert_eq!(
                map.lookup_host_addr(out.code_ptr(), host),
                Some(range.guest_va)
            );
        }
        assert!(map.code_size_for(range.guest_va) >= range.len());
    }

    // An address before the function, and one past its end, are not in any
    // range rather than being clamped to the nearest one.
    assert_eq!(
        map.lookup_host_addr(out.code_ptr(), out.code_ptr() as u64 - 1),
        None
    );
    let past_end = map.ranges().last().unwrap().end;
    assert_eq!(map.lookup(past_end + 4096), None);
}

#[test]
fn code_map_is_empty_when_source_locs_are_off() {
    let out = LiftTest::new(PROGRAM)
        .with_config(|c| c.emit_source_locs = false)
        .run();
    assert!(out.code_map().is_empty());
}

#[test]
fn out_of_region_exit_stubs_name_their_target() {
    // `jmp +0x40` leaves the region: the stub `finish()` emits for it is
    // attributed to the address being jumped to, which is what a debugger
    // stopped in that stub needs to be told.
    let base = DEFAULT_LIFT_BASE;
    let bytes = [0xEB, 0x40]; // jmp .+0x40
    let target = base + 0x42;

    let code = lift_to_code::<X86_64>(&bytes, base, LifterConfig::default()).unwrap();
    let tagged: BTreeSet<u64> = ir_source_locs(&code.func, code.source_map)
        .into_iter()
        .filter_map(|(_, va)| va)
        .collect();

    assert!(tagged.contains(&base), "the jmp itself");
    assert!(
        tagged.contains(&target),
        "the out-of-region stub for {target:#x}, got {tagged:x?}"
    );
}

#[test]
fn address_mode_does_not_affect_the_mapping() {
    // Source locations are lift-time VAs, so they are the same under both
    // address modes even though the emitted address arithmetic differs.
    let for_mode = |mode: AddressMode| -> BTreeSet<u64> {
        let mut config = LifterConfig::default();
        config.address_mode = mode;
        let code = lift_to_code::<X86_64>(PROGRAM, DEFAULT_LIFT_BASE, config).unwrap();
        ir_source_locs(&code.func, code.source_map)
            .into_iter()
            .filter_map(|(_, va)| va)
            .collect()
    };

    assert_eq!(
        for_mode(AddressMode::ImageBaseRelative),
        for_mode(AddressMode::Direct)
    );
}

#[test]
fn aot_output_carries_a_code_map() {
    let code = lift_to_code::<X86_64>(PROGRAM, DEFAULT_LIFT_BASE, LifterConfig::default()).unwrap();
    let (bytes, map) = ArtifactBackend::compile_lifted_to_raw_with_map(code, None, None).unwrap();

    assert!(!bytes.is_empty());
    assert!(!map.is_empty());
    // `lifted_func` is emitted first, so map offsets index the raw text.
    assert!(
        (map.ranges().last().unwrap().end as usize) <= bytes.len(),
        "map runs past the extracted .text"
    );
    let guests: BTreeSet<u64> = map.guest_addresses().into_iter().collect();
    assert!(guests.is_subset(&program_vas(DEFAULT_LIFT_BASE)));
}

/// Only present under the `disasm` feature, which brings in capstone.
#[cfg(feature = "disasm")]
#[test]
fn annotated_disassembly_labels_each_guest_instruction() {
    let code = lift_to_code::<X86_64>(PROGRAM, DEFAULT_LIFT_BASE, LifterConfig::default()).unwrap();
    let (bytes, map) = ArtifactBackend::compile_lifted_to_raw_with_map(code, None, None).unwrap();
    let text = ArtifactBackend::disassemble_annotated(&bytes, None, &map).unwrap();

    // Prologue first, then a header for each guest instruction that survived.
    assert!(text.contains("; [no source]"), "{text}");
    for va in map.guest_addresses() {
        assert!(text.contains(&format!("; guest {va:#x}\n")), "{text}");
    }
}

#[test]
fn maps_survive_a_round_trip_through_ranges() {
    let out = LiftTest::new(PROGRAM).run();
    let rebuilt = GuestCodeMap::from_ranges(out.code_map().ranges().to_vec());
    assert_eq!(&rebuilt, out.code_map());
}
