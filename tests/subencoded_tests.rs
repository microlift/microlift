use microlift::testkit::prelude::*;

#[test]
fn test_overlapping_jmp_into_self() {
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,       // 0x1000: xor eax, eax
        0xEB, 0xFF,       // 0x1002: jmp 0x1003  (into the 0xFF byte below)
        0xC0,             // 0x1004: (this byte is 'C0'; with 0x1003's 'FF' -> inc eax)
        0xC3,             // 0x1005: ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1, "overlapping `FF C0` should decode as inc eax");
}

/// Sub-encoded instruction hidden inside a `mov` immediate.
///
/// A linear disassembler decodes a 5-byte `mov eax, imm32` at 0x1004, but control
/// flow jumps one byte into it, so the immediate bytes decode as `add eax, imm32`.
#[test]
fn test_subencoded_in_mov_immediate() {
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,                   // 0x1000: xor eax, eax
        0xEB, 0x01,                   // 0x1002: jmp 0x1005 (into the mov immediate)
        0xB8,                         // 0x1004: mov-eax opcode (skipped byte)
        0x05, 0x2A, 0x00, 0x00, 0x00, // 0x1005: add eax, 0x2A (overlaps the mov)
        0xC3,                         // 0x100A: ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 0x2A,
        "bytes entered mid-mov should decode as add eax, 0x2A"
    );
}
