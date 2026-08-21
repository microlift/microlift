//! What a translation depends on: [`LiftedCode::covered`] and
//! [`LiftedCode::data_dependencies`].

use microlift::arch::x86_64::X86_64;
use microlift::context::MemorySegment;
use microlift::core::{LifterConfig, lift_to_code};

const BASE: u64 = 0x1000;
/// Lift-time VA of the read-only jump table, outside the code.
const RODATA: u64 = 0x2000;

/// The PIC switch from `jump_table_tests`, whose `jmp rdx` at 0x1015 dispatches
/// through a four-entry `i32` table at [`RODATA`].
#[rustfmt::skip]
const PIC_SWITCH: &[u8] = &[
    0x83, 0xFF, 0x03,                          // cmp edi, 3
    0x77, 0x2A,                                // ja  0x102f
    0x89, 0xF8,                                // mov eax, edi
    0x48, 0x8D, 0x0D, 0xF2, 0x0F, 0x00, 0x00,  // lea rcx, [rip+0xff2]
    0x48, 0x63, 0x14, 0x81,                    // movsxd rdx, [rcx+rax*4]
    0x48, 0x01, 0xCA,                          // add rdx, rcx
    0xFF, 0xE2,                                // jmp rdx
    0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,        // case0
    0xB8, 0x14, 0x00, 0x00, 0x00, 0xC3,        // case1
    0xB8, 0x1E, 0x00, 0x00, 0x00, 0xC3,        // case2
    0xB8, 0x28, 0x00, 0x00, 0x00, 0xC3,        // case3
    0xB8, 0x63, 0x00, 0x00, 0x00, 0xC3,        // dflt
];

const PIC_CASES: [u64; 4] = [0x1017, 0x101D, 0x1023, 0x1029];

fn pic_entries() -> Vec<u8> {
    let mut data = Vec::new();
    for target in PIC_CASES {
        data.extend_from_slice(&((target as i64 - RODATA as i64) as i32).to_le_bytes());
    }
    data
}

/// `mov rax, 42 ; ret` - a region with no table and nothing to depend on.
const PLAIN: &[u8] = &[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

#[test]
fn covered_is_the_region_when_the_host_named_no_sub_ranges() {
    let code = lift_to_code::<X86_64>(PLAIN, BASE, LifterConfig::default()).unwrap();

    assert_eq!(code.covered, vec![BASE..BASE + PLAIN.len() as u64]);
}

#[test]
fn covered_reports_the_holes_a_host_declared() {
    // A host that knows only part of the slice is code says so, and the
    // translation then depends on no byte inside the hole - which is the whole
    // point of reporting `covered` rather than `start..end`.
    let config = LifterConfig {
        block_ranges: vec![BASE..BASE + 8, BASE + 16..BASE + 24],
        ..LifterConfig::default()
    };

    let code = lift_to_code::<X86_64>(&[0x90; 32], BASE, config).unwrap();

    assert_eq!(code.covered, vec![BASE..BASE + 8, BASE + 16..BASE + 24]);
    let hole = BASE + 12;
    assert!(
        !code.covered.iter().any(|r| r.contains(&hole)),
        "a write inside the declared hole must not invalidate this translation"
    );
}

#[test]
fn a_lift_with_no_jump_table_depends_on_no_data() {
    // The negative control for the whole field. Without this, an implementation
    // that reported the entire address space as a data dependency would pass
    // every other case in this file.
    let code = lift_to_code::<X86_64>(PLAIN, BASE, LifterConfig::default()).unwrap();

    assert!(code.data_dependencies.is_empty());
}

#[test]
fn the_pass_being_off_is_itself_a_control() {
    // Same bytes, same table in the segments, pass off: the lifter never read
    // the table, so the translation does not depend on it. This is what makes
    // the positive case below evidence that the *pass* is what created the
    // dependency, rather than the segment list.
    let config = LifterConfig {
        segments: vec![MemorySegment::new(RODATA, pic_entries(), false)].into(),
        ..LifterConfig::default()
    };

    let code = lift_to_code::<X86_64>(PIC_SWITCH, BASE, config).unwrap();

    assert!(code.data_dependencies.is_empty());
}

#[test]
fn a_recovered_jump_table_becomes_a_data_dependency() {
    let config = LifterConfig {
        analyze_jump_tables: true,
        segments: vec![MemorySegment::new(RODATA, pic_entries(), false)].into(),
        ..LifterConfig::default()
    };

    let code = lift_to_code::<X86_64>(PIC_SWITCH, BASE, config).unwrap();

    // Four `i32` entries, and *exactly* four: the extent comes from the count
    // the pass read, not from the de-duplicated target list and not from the
    // segment's length.
    assert_eq!(code.data_dependencies, vec![RODATA..RODATA + 16]);

    // And it is disjoint from the code, which is the reason the two fields are
    // separate rather than one merged range list.
    assert_eq!(code.covered, vec![BASE..BASE + PIC_SWITCH.len() as u64]);
}

#[test]
fn the_extent_is_not_the_deduplicated_target_count() {
    // A table whose four slots name only two distinct blocks. `targets` has two
    // entries; the table is still sixteen bytes, and a write to the last four of
    // them still changes which block slot 3 dispatches to.
    let mut entries = Vec::new();
    for target in [PIC_CASES[0], PIC_CASES[0], PIC_CASES[1], PIC_CASES[1]] {
        entries.extend_from_slice(&((target as i64 - RODATA as i64) as i32).to_le_bytes());
    }

    let config = LifterConfig {
        analyze_jump_tables: true,
        segments: vec![MemorySegment::new(RODATA, entries, false)].into(),
        ..LifterConfig::default()
    };

    let code = lift_to_code::<X86_64>(PIC_SWITCH, BASE, config).unwrap();

    assert_eq!(
        code.data_dependencies,
        vec![RODATA..RODATA + 16],
        "two distinct targets, four slots, sixteen bytes"
    );
}
