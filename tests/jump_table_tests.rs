//! Clang jump-table recovery (`analyze_jump_tables`): the recovered table, and dispatch through it instead of `UnknownJump`.
//! Executed tables must exist in host memory too; [`host_image`] lays it out and returns the `image_base`.

use microlift::arch::x86_64::jumptable::{JumpTableKind, analyze_jump_tables};
use microlift::context::MemorySegment;
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;
const RODATA: u64 = 0x2000;

/// Lays out `code` at [`BASE`] and each blob at its lift-time VA; returns the `image_base`. Leaked on purpose: lifted code holds raw addresses.
fn host_image(code: &[u8], blobs: &[(u64, Vec<u8>)]) -> u64 {
    let end = blobs
        .iter()
        .map(|(va, data)| va - BASE + data.len() as u64)
        .chain(std::iter::once(code.len() as u64))
        .max()
        .unwrap();

    let mut image = vec![0u8; end as usize];
    image[..code.len()].copy_from_slice(code);
    for (va, data) in blobs {
        let off = (va - BASE) as usize;
        image[off..off + data.len()].copy_from_slice(data);
    }

    Box::leak(image.into_boxed_slice()).as_ptr() as u64
}

/// PIC form, table in a supplied read-only segment:
///
/// ```text
/// 0x1000  cmp    edi, 3
/// 0x1003  ja     0x102f
/// 0x1005  mov    eax, edi
/// 0x1007  lea    rcx, [rip + 0xff2]         ; 0x2000
/// 0x100e  movsxd rdx, dword ptr [rcx+rax*4]
/// 0x1012  add    rdx, rcx
/// 0x1015  jmp    rdx
/// 0x1017  case0..case3: mov eax, 10/20/30/40 ; ret
/// 0x102f  dflt:  mov eax, 99 ; ret
/// ```
#[rustfmt::skip]
const PIC_SWITCH: &[u8] = &[
    0x83, 0xFF, 0x03,                          // cmp edi, 3
    0x77, 0x2A,                                // ja  0x102f
    0x89, 0xF8,                                // mov eax, edi
    0x48, 0x8D, 0x0D, 0xF2, 0x0F, 0x00, 0x00,  // lea rcx, [rip+0xff2]
    0x48, 0x63, 0x14, 0x81,                    // movsxd rdx, [rcx+rax*4]
    0x48, 0x01, 0xCA,                          // add rdx, rcx
    0xFF, 0xE2,                                // jmp rdx
    0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,        // case0: mov eax, 10 ; ret
    0xB8, 0x14, 0x00, 0x00, 0x00, 0xC3,        // case1: mov eax, 20 ; ret
    0xB8, 0x1E, 0x00, 0x00, 0x00, 0xC3,        // case2: mov eax, 30 ; ret
    0xB8, 0x28, 0x00, 0x00, 0x00, 0xC3,        // case3: mov eax, 40 ; ret
    0xB8, 0x63, 0x00, 0x00, 0x00, 0xC3,        // dflt:  mov eax, 99 ; ret
];

const PIC_CASES: [u64; 4] = [0x1017, 0x101D, 0x1023, 0x1029];

/// Displacements from the table's own address, so one copy serves as lift-time hint and runtime memory.
fn pic_entries() -> Vec<u8> {
    let mut data = Vec::new();
    for target in PIC_CASES {
        data.extend_from_slice(&((target as i64 - RODATA as i64) as i32).to_le_bytes());
    }
    data
}

fn pic_table(writable: bool) -> MemorySegment {
    MemorySegment::new(RODATA, pic_entries(), writable)
}

fn pic_test() -> LiftTest<'static> {
    let image = host_image(PIC_SWITCH, &[(RODATA, pic_entries())]);
    LiftTest::new(PIC_SWITCH)
        .at(BASE)
        .image_base(image)
        .segments(vec![pic_table(false)])
}

#[test]
fn pic_table_is_recovered() {
    let tables = analyze_jump_tables(PIC_SWITCH, BASE, &[pic_table(false)]);
    let table = tables
        .get(&0x1015)
        .expect("no jump table recovered for `jmp rdx` at 0x1015");

    assert_eq!(table.table, RODATA);
    assert_eq!(table.entry_size, 4);
    assert_eq!(table.kind, JumpTableKind::Relative { anchor: RODATA });
    assert!(table.bounded, "cmp/ja range check should bound the table");
    assert_eq!(table.targets, PIC_CASES);
}

#[test]
fn pic_switch_dispatches_without_exiting() {
    for (index, expected) in [(0u64, 10u64), (1, 20), (2, 30), (3, 40)] {
        let out = pic_test()
            .analyze_jump_tables()
            .reg(|c| c.rdi = index)
            .run();

        assert_eq!(out.exit, ExitReason::Return, "index {index}");
        assert_eq!(out.ctx.rax, expected, "index {index}");
    }
}

#[test]
fn out_of_range_index_still_takes_the_default_arm() {
    // Recovery must not change out-of-range behaviour.
    let out = pic_test().analyze_jump_tables().reg(|c| c.rdi = 7).run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 99);
}

#[test]
fn without_the_pass_the_switch_exits() {
    // Baseline: case blocks are not leaders, so the jump leaves the region.
    let image = host_image(PIC_SWITCH, &[(RODATA, pic_entries())]);
    let out = LiftTest::new(PIC_SWITCH)
        .at(BASE)
        .image_base(image)
        .segments(vec![pic_table(false)])
        .reg(|c| c.rdi = 2)
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        out.ctx.rip,
        image + (0x1023 - BASE),
        "exit PC should be the unlifted case"
    );
}

#[test]
fn writable_segments_are_not_trusted() {
    // Same table in memory the guest could have rewritten.
    let tables = analyze_jump_tables(PIC_SWITCH, BASE, &[pic_table(true)]);
    assert!(tables.is_empty(), "recovered a table from writable memory");
}

#[test]
fn missing_segment_recovers_nothing() {
    let tables = analyze_jump_tables(PIC_SWITCH, BASE, &[]);
    assert!(tables.is_empty());
}

/// Non-PIC form: `jmp qword ptr [rcx + rax*8]` loads an absolute address, table at 0x2000, cases from 0x1011, dflt at 0x1029.
/// The table is reached via a register because only RIP-relative operands are relocated, not bare absolute displacements.
#[rustfmt::skip]
const ABS_SWITCH: &[u8] = &[
    0x83, 0xFF, 0x03,                          // cmp edi, 3
    0x77, 0x24,                                // ja  0x1029
    0x89, 0xF8,                                // mov eax, edi
    0x48, 0x8D, 0x0D, 0xF2, 0x0F, 0x00, 0x00,  // lea rcx, [rip+0xff2]
    0xFF, 0x24, 0xC1,                          // jmp [rcx+rax*8]
    0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,        // case0: mov eax, 10 ; ret
    0xB8, 0x14, 0x00, 0x00, 0x00, 0xC3,        // case1: mov eax, 20 ; ret
    0xB8, 0x1E, 0x00, 0x00, 0x00, 0xC3,        // case2: mov eax, 30 ; ret
    0xB8, 0x28, 0x00, 0x00, 0x00, 0xC3,        // case3: mov eax, 40 ; ret
    0xB8, 0x63, 0x00, 0x00, 0x00, 0xC3,        // dflt:  mov eax, 99 ; ret
];

const ABS_CASES: [u64; 4] = [0x1011, 0x1017, 0x101D, 0x1023];

/// Absolute entries differ per copy: VAs for the analyzer, relocated to `image_base` in host memory.
fn abs_entries(image_base: u64) -> Vec<u8> {
    let mut data = Vec::new();
    for target in ABS_CASES {
        data.extend_from_slice(&(target - BASE + image_base).to_le_bytes());
    }
    data
}

#[test]
fn absolute_table_is_recovered() {
    let segment = MemorySegment::new(RODATA, abs_entries(BASE), false);
    let tables = analyze_jump_tables(ABS_SWITCH, BASE, &[segment]);
    let table = tables.get(&0x100E).expect("no jump table recovered");

    assert_eq!(table.table, RODATA);
    assert_eq!(table.entry_size, 8);
    assert_eq!(table.kind, JumpTableKind::Absolute);
    assert!(table.bounded);
    assert_eq!(table.targets, ABS_CASES);
}

#[test]
fn absolute_switch_dispatches_without_exiting() {
    for (index, expected) in [(0u64, 10u64), (1, 20), (2, 30), (3, 40)] {
        // The image must exist before entries can be relocated into it.
        let image = host_image(ABS_SWITCH, &[(RODATA, abs_entries(BASE))]);
        let relocated = abs_entries(image);
        // SAFETY: `image` is a leaked, uniquely-owned buffer large enough for the relocated table.
        unsafe {
            std::ptr::copy_nonoverlapping(
                relocated.as_ptr(),
                (image + (RODATA - BASE)) as *mut u8,
                relocated.len(),
            );
        }

        let out = LiftTest::new(ABS_SWITCH)
            .at(BASE)
            .image_base(image)
            .segments(vec![MemorySegment::new(RODATA, abs_entries(BASE), false)])
            .analyze_jump_tables()
            .reg(|c| c.rdi = index)
            .run();

        assert_eq!(out.exit, ExitReason::Return, "index {index}");
        assert_eq!(out.ctx.rax, expected, "index {index}");
    }
}

/// PIC form with the table inside the region past the `ret`s and no segment supplied (Clang's layout for a hot `switch`).
/// `lea rcx,[rip+0x21]` at 0x1007 -> table at 0x102f; cases from 0x1017, dflt at 0x1029.
#[rustfmt::skip]
const INLINE_TABLE_SWITCH: &[u8] = &[
    0x83, 0xFF, 0x02,                          // cmp edi, 2
    0x77, 0x24,                                // ja  0x1029
    0x89, 0xF8,                                // mov eax, edi
    0x48, 0x8D, 0x0D, 0x21, 0x00, 0x00, 0x00,  // lea rcx, [rip+0x21]
    0x48, 0x63, 0x14, 0x81,                    // movsxd rdx, [rcx+rax*4]
    0x48, 0x01, 0xCA,                          // add rdx, rcx
    0xFF, 0xE2,                                // jmp rdx
    0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,        // case0: mov eax, 10 ; ret
    0xB8, 0x14, 0x00, 0x00, 0x00, 0xC3,        // case1: mov eax, 20 ; ret
    0xB8, 0x1E, 0x00, 0x00, 0x00, 0xC3,        // case2: mov eax, 30 ; ret
    0xB8, 0x63, 0x00, 0x00, 0x00, 0xC3,        // dflt:  mov eax, 99 ; ret
    0xE8, 0xFF, 0xFF, 0xFF,                    // .long 0x1017 - 0x102f
    0xEE, 0xFF, 0xFF, 0xFF,                    // .long 0x101d - 0x102f
    0xF4, 0xFF, 0xFF, 0xFF,                    // .long 0x1023 - 0x102f
];

#[test]
fn in_region_table_needs_no_segment() {
    let tables = analyze_jump_tables(INLINE_TABLE_SWITCH, BASE, &[]);
    let table = tables.get(&0x1015).expect("no jump table recovered");

    assert_eq!(table.table, 0x102F);
    assert_eq!(table.kind, JumpTableKind::Relative { anchor: 0x102F });
    assert_eq!(table.targets, [0x1017, 0x101D, 0x1023]);
}

#[test]
fn in_region_table_switch_dispatches_without_exiting() {
    for (index, expected) in [(0u64, 10u64), (1, 20), (2, 30)] {
        let out = LiftTest::new(INLINE_TABLE_SWITCH)
            .at(BASE)
            .image_base(host_image(INLINE_TABLE_SWITCH, &[]))
            .analyze_jump_tables()
            .reg(|c| c.rdi = index)
            .run();

        assert_eq!(out.exit, ExitReason::Return, "index {index}");
        assert_eq!(out.ctx.rax, expected, "index {index}");
    }
}

/// PIC dispatch with no range check: the walk must find the table end itself, by running off the region at entry 3.
/// Table at 0x1024, cases from 0x1012.
#[rustfmt::skip]
const UNBOUNDED_SWITCH: &[u8] = &[
    0x89, 0xF8,                                // mov eax, edi
    0x48, 0x8D, 0x0D, 0x1B, 0x00, 0x00, 0x00,  // lea rcx, [rip+0x1b]
    0x48, 0x63, 0x14, 0x81,                    // movsxd rdx, [rcx+rax*4]
    0x48, 0x01, 0xCA,                          // add rdx, rcx
    0xFF, 0xE2,                                // jmp rdx
    0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,        // case0: mov eax, 10 ; ret
    0xB8, 0x14, 0x00, 0x00, 0x00, 0xC3,        // case1: mov eax, 20 ; ret
    0xB8, 0x1E, 0x00, 0x00, 0x00, 0xC3,        // case2: mov eax, 30 ; ret
    0xEE, 0xFF, 0xFF, 0xFF,                    // .long 0x1012 - 0x1024
    0xF4, 0xFF, 0xFF, 0xFF,                    // .long 0x1018 - 0x1024
    0xFA, 0xFF, 0xFF, 0xFF,                    // .long 0x101e - 0x1024
];

#[test]
fn unbounded_table_walks_until_the_entries_stop_making_sense() {
    let tables = analyze_jump_tables(UNBOUNDED_SWITCH, BASE, &[]);
    let table = tables.get(&0x1010).expect("no jump table recovered");

    assert!(!table.bounded, "no range check to bound this one");
    assert_eq!(table.targets, [0x1012, 0x1018, 0x101E]);
}

#[test]
fn unbounded_table_switch_dispatches_without_exiting() {
    for (index, expected) in [(0u64, 10u64), (1, 20), (2, 30)] {
        let out = LiftTest::new(UNBOUNDED_SWITCH)
            .at(BASE)
            .image_base(host_image(UNBOUNDED_SWITCH, &[]))
            .analyze_jump_tables()
            .reg(|c| c.rdi = index)
            .run();

        assert_eq!(out.exit, ExitReason::Return, "index {index}");
        assert_eq!(out.ctx.rax, expected, "index {index}");
    }
}

#[test]
fn plain_indirect_jump_is_left_alone() {
    // Nothing to recover from a bare `jmp rax`.
    let out = LiftTest::new(&[0xFF, 0xE0])
        .at(BASE)
        .analyze_jump_tables()
        .run();
    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert!(analyze_jump_tables(&[0xFF, 0xE0], BASE, &[]).is_empty());
}
