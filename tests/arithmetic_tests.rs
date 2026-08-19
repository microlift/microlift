/// Tests for arithmetic instructions lacking coverage:
/// neg, imul (3 forms), mul, idiv, div,
/// cbw/cwde/cdqe/cwd/cdq/cqo, sar, rol, shld/shrd, bt.
use microlift::testkit::prelude::*;

// NEG

#[test]
fn test_neg() {
    // neg rax when rax=5 -> rax=-5, CF=1 (src!=0), ZF=0, SF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xF7, 0xD8,                          // neg rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-5i64) as u64);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.cf, 1); // CF=1 when src != 0

    // neg rax when rax=0 -> rax=0, CF=0, ZF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x31, 0xC0, // xor rax, rax -> rax=0
        0x48, 0xF7, 0xD8, // neg rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 0); // CF=0 when src == 0
    assert_eq!(ctx.zf, 1);
}

// IMUL

#[test]
fn test_imul_two_operand() {
    // imul rax, rcx  (2-operand: dst = dst * src)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0x0F, 0xAF, 0xC1,                    // imul rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    // OF=CF=0 since result fits (42 == sign-extend of 42)
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_imul_two_operand_negative() {
    // imul rax, rcx with rax=-3, rcx=4 -> rax=-12
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFD, 0xFF, 0xFF, 0xFF, // mov rax, -3 (sign-extended)
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0x0F, 0xAF, 0xC1,                    // imul rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-12i64) as u64);
}

#[test]
fn test_imul_three_operand() {
    // imul rax, rcx, 6  (3-operand: dst = src * imm)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0x6B, 0xC1, 0x06,                    // imul rax, rcx, 6
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_imul_one_operand() {
    // imul rcx  (1-operand: rdx:rax = rax * rcx, signed)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xE9,                          // imul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42); // low half in rax
    assert_eq!(ctx.rdx, 0); // high half = 0 (result fits in 64 bits)
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_imul_one_operand_overflow() {
    // i64::MAX * 2 overflows: rdx should be non-zero, OF/CF should be set
    // rax = 0x7FFFFFFFFFFFFFFF; imul rcx (rcx=2) -> rdx:rax = 0x0000...FFFE
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // mov rax, i64::MAX
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,                   // mov rcx, 2
        0x48, 0xF7, 0xE9,                                            // imul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    // rdx:rax = i64::MAX * 2 = 0x0000000000000000_FFFFFFFFFFFFFFFE
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFEu64);
    assert_eq!(ctx.rdx, 0); // high half is 0 for this particular product
    // Actually i64::MAX * 2 = 2^63 - 2, which fits in 64 unsigned bits so rdx=0
    // but the signed high half: smulhi(i64::MAX, 2) = 0
    // OF/CF should be 1 since the result doesn't equal sign-extend of the low half
    // (low half = 0xFFFFFFFFFFFFFFFE = -2 as signed, but sign-extend of that = 0xFFFF... != 0)
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}

// MUL

#[test]
fn test_mul() {
    // mul rcx  (1-operand: rdx:rax = rax * rcx, unsigned)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xE1,                          // mul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rdx, 0); // no overflow
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

/// The narrower `mul` forms, which every 64-bit test above misses: the
/// accumulator and the operand are read at the operand's own width, so
/// narrowing them again is an `ireduce` from a type to itself - malformed IR
/// that the verifier rejects for the whole function. libpng's `png_zalloc`
/// does `mul esi` on its way to every allocation, which is how this was found.
#[test]
fn test_mul_32_and_16_bit() {
    // mul esi (32-bit: edx:eax = eax * esi), with a product that overflows 32
    // bits so the high half is checked too.
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x10,             // mov eax, 0x10000000
        0xBE, 0x20, 0x00, 0x00, 0x00,             // mov esi, 0x20
        0xF7, 0xE6,                               // mul esi
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000_0000, "low half of 0x10000000 * 0x20");
    assert_eq!(ctx.rdx, 0x0000_0002, "high half");
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);

    // mul cx (16-bit: dx:ax = ax * cx)
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x10,                   // mov ax, 0x1000
        0x66, 0xB9, 0x20, 0x00,                   // mov cx, 0x20
        0x66, 0xF7, 0xE1,                         // mul cx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x0000, "low half of 0x1000 * 0x20");
    assert_eq!(ctx.rdx & 0xFFFF, 0x0002, "high half");
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}

/// The same double-narrowing in `imul`'s one-operand form, which reads its
/// source operand at the operand width and then narrowed it again.
#[test]
fn test_imul_one_operand_32_bit() {
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0xFF, 0xFF, 0xFF, 0x7F,             // mov eax, i32::MAX
        0xB9, 0x02, 0x00, 0x00, 0x00,             // mov ecx, 2
        0xF7, 0xE9,                               // imul ecx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFF_FFFE, "low half of i32::MAX * 2");
    assert_eq!(ctx.rdx, 0, "smulhi(i32::MAX, 2) is zero");
}

#[test]
fn test_mul_overflow() {
    // u64::MAX * 2 = 2^65 - 2 = rdx=1, rax=0xFFFFFFFFFFFFFFFE
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, u64::MAX
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,                   // mov rcx, 2
        0x48, 0xF7, 0xE1,                                            // mul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFEu64);
    assert_eq!(ctx.rdx, 1); // high half = 1
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}

// IDIV / DIV

#[test]
fn test_idiv() {
    // idiv rcx: rdx:rax / rcx (signed)
    // 43 / 7 = 6 remainder 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2B, 0x00, 0x00, 0x00, // mov rax, 43
        0x48, 0x31, 0xD2,                          // xor rdx, rdx  (rdx=0, sign-extend)
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF9,                          // idiv rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6); // quotient
    assert_eq!(ctx.rdx, 1); // remainder
}

#[test]
fn test_idiv_negative() {
    // -43 / 7 = -6 remainder -1  (truncation toward zero)
    // mov rax, -43: 48 C7 C0 D5 FF FF FF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xD5, 0xFF, 0xFF, 0xFF, // mov rax, -43
        0x48, 0xC7, 0xC2, 0xFF, 0xFF, 0xFF, 0xFF, // mov rdx, -1  (sign extension of -43)
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF9,                          // idiv rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-6i64) as u64); // quotient = -6
    assert_eq!(ctx.rdx, (-1i64) as u64); // remainder = -1
}

#[test]
fn test_div() {
    // div rcx: rdx:rax / rcx (unsigned)
    // 43 / 7 = 6 remainder 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2B, 0x00, 0x00, 0x00, // mov rax, 43
        0x48, 0x31, 0xD2,                          // xor rdx, rdx
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF1,                          // div rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6);
    assert_eq!(ctx.rdx, 1);
}

// Sign-extension conversions

#[test]
fn test_cbw() {
    // CBW: AX = sign_extend(AL)
    // AL = 0xFF (-1 as i8) -> AX = 0xFFFF
    #[rustfmt::skip]
    let bytes = [
        0xB0, 0xFF, // mov al, 0xFF
        0x66, 0x98, // cbw
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0xFFFF);

    // AL = 0x7F (127) -> AX = 0x007F (positive, no sign extension)
    #[rustfmt::skip]
    let bytes = [
        0xB0, 0x7F, // mov al, 0x7F
        0x66, 0x98, // cbw
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x007F);
}

#[test]
fn test_cwde() {
    // CWDE: EAX = sign_extend(AX)
    // AX = 0x8000 (-32768) -> EAX = 0xFFFF8000 -> RAX = 0x00000000FFFF8000
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x98,                   // cwde
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x00000000FFFF8000u64);

    // AX = 0x0080 (positive) -> EAX = 0x00000080 -> RAX = 0x0000000000000080
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x80, 0x00, // mov ax, 0x0080
        0x98,                   // cwde
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000000000080u64);
}

#[test]
fn test_cdqe() {
    // CDQE: RAX = sign_extend(EAX)
    // EAX = 0x80000000 -> RAX = 0xFFFFFFFF80000000
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x48, 0x98,                    // cdqe
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFF80000000u64);

    // EAX = 0x7FFFFFFF -> RAX = 0x000000007FFFFFFF (positive, no change)
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0xFF, 0xFF, 0xFF, 0x7F, // mov eax, 0x7FFFFFFF
        0x48, 0x98,                    // cdqe
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x000000007FFFFFFFu64);
}

#[test]
fn test_cwd() {
    // CWD: DX:AX = sign_extend(AX)
    // AX = 0x8000 (-32768) -> DX = 0xFFFF
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x66, 0x99,             // cwd
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x8000); // AX unchanged
    assert_eq!(ctx.rdx & 0xFFFF, 0xFFFF); // DX = -1

    // AX = 0x0001 (positive) -> DX = 0x0000
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x01, 0x00, // mov ax, 0x0001
        0x66, 0x99,             // cwd
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx & 0xFFFF, 0x0000);
}

#[test]
fn test_cdq() {
    // CDQ: EDX:EAX = sign_extend(EAX)
    // EAX = 0x80000000 -> EDX = 0xFFFFFFFF -> RDX = 0x00000000FFFFFFFF
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x99,                          // cdq
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000080000000u64);
    assert_eq!(ctx.rdx, 0x00000000FFFFFFFFu64);

    // EAX = 0x00000001 -> EDX = 0x00000000
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0x99,                          // cdq
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, 0);
}

#[test]
fn test_cqo() {
    // CQO: RDX:RAX = sign_extend(RAX)
    // RAX = 0x8000000000000000 -> RDX = 0xFFFFFFFFFFFFFFFF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 0x8000...
        0x48, 0x99,                                                    // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, u64::MAX); // 0xFFFFFFFFFFFFFFFF

    // RAX = 1 -> RDX = 0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x99,                                // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, 0);
}

// SAR (arithmetic right shift)

#[test]
fn test_sar_positive() {
    // sar rax, 4 when rax=128: 128 >> 4 = 8 (preserves sign=0)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x80, 0x00, 0x00, 0x00, // mov rax, 128
        0x48, 0xC1, 0xF8, 0x04,                    // sar rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
    assert_eq!(ctx.sf, 0);
}

#[test]
fn test_sar_negative() {
    // sar rax, 4 when rax=-128 (0xFFFFFFFFFFFFFF80):
    // -128 >> 4 (arithmetic) = -8 (0xFFFFFFFFFFFFFFF8)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x80, 0xFF, 0xFF, 0xFF, // mov rax, -128 (sign-extended)
        0x48, 0xC1, 0xF8, 0x04,                    // sar rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-8i64) as u64);
    assert_eq!(ctx.sf, 1); // result is negative
}

#[test]
fn test_sar_by_cl() {
    // sar rax, cl when rax=64, cl=3: 64 >> 3 = 8
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x40, 0x00, 0x00, 0x00, // mov rax, 64
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0xD3, 0xF8,                          // sar rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
}

// ROL (rotate left)

#[test]
fn test_rol_msb_rotates_to_lsb() {
    // rol rax, 1 when rax=0x8000000000000000:
    // result = 1, CF = 1 (the MSB that was rotated out)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 0x8000...
        0x48, 0xD1, 0xC0,                                             // rol rax, 1
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_rol_preserves_value_at_full_rotation() {
    // rol rax, 4 when rax=0x0F0F0F0F0F0F0F0F
    // result = 0xF0F0F0F0F0F0F0F0  (nibbles shifted left)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, // mov rax, 0x0F0F...
        0x48, 0xC1, 0xC0, 0x04,                                       // rol rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xF0F0F0F0F0F0F0F0u64);
}

// SHLD / SHRD (double-precision shift)

#[test]
fn test_shld() {
    // shld rax, rcx, 4: result = (rax << 4) | (rcx >> 60)
    // rax = 0x00000000000000FF, rcx = 0xF000000000000000
    // rax << 4 = 0x0000000000000FF0
    // rcx >> 60 = 0xF
    // result = 0x0000000000000FFF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0x00, 0x00, 0x00, // mov rax, 0xFF
        0x48, 0xB9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, // mov rcx, 0xF000000000000000
        0x48, 0x0F, 0xA4, 0xC8, 0x04,             // shld rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000000000FFFu64);
}

#[test]
fn test_shld_simple() {
    // shld rax, rcx, 4 when rcx=0: result = rax << 4 = 0x10
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x31, 0xC9,                          // xor rcx, rcx
        0x48, 0x0F, 0xA4, 0xC8, 0x04,             // shld rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x10);
}

#[test]
fn test_shrd() {
    // shrd rax, rcx, 4: result = (rcx << 60) | (rax >> 4)
    // rax = 0x00000000000000F0, rcx = 0x000000000000000F
    // rcx << 60 = 0xF000000000000000
    // rax >> 4  = 0x000000000000000F
    // result = 0xF00000000000000F
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xF0, 0x00, 0x00, 0x00, // mov rax, 0xF0
        0x48, 0xC7, 0xC1, 0x0F, 0x00, 0x00, 0x00, // mov rcx, 0x0F
        0x48, 0x0F, 0xAC, 0xC8, 0x04,             // shrd rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xF00000000000000Fu64);
}

#[test]
fn test_shrd_simple() {
    // shrd rax, rcx, 4 when rcx=0: result = rax >> 4 = 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0x31, 0xC9,                          // xor rcx, rcx
        0x48, 0x0F, 0xAC, 0xC8, 0x04,             // shrd rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
}

// BT (bit test, read-only)

#[test]
fn test_bt_immediate() {
    // bt rax, 3 when rax=8 (bit 3 is set): CF=1, rax unchanged
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xE0, 0x03,             // bt rax, 3
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1); // bit 3 was set
    assert_eq!(ctx.rax, 8); // rax unchanged

    // bt rax, 2 when rax=8 (bit 2 is NOT set): CF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xE0, 0x02,             // bt rax, 2
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0); // bit 2 was clear
    assert_eq!(ctx.rax, 8); // rax unchanged
}

#[test]
fn test_bt_register() {
    // bt rax, rcx when rax=0x10, rcx=4: bit 4 is set -> CF=1
    // bt rax, rcx: 48 0F A3 C8
    // ModRM: mod=11, reg=rcx(001), rm=rax(000) -> 0b11_001_000 = 0xC8
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10 (bit 4 set)
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0x0F, 0xA3, 0xC8,                    // bt rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1);

    // bt rax, rcx when rax=0x10, rcx=3: bit 3 is NOT set -> CF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0x0F, 0xA3, 0xC8,                    // bt rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_adc_no_carry_in() {
    // adc rax, 7 when CF=0: rax = 0 + 7 + 0 = 7
    // xor rax, rax clears rax and sets CF=0
    // adc rax, 7: 48 83 D0 07
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x31, 0xC0,       // xor rax, rax  (rax=0, CF=0)
        0x48, 0x83, 0xD0, 0x07, // adc rax, 7
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 7);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_adc_with_carry_in() {
    // Set CF=1 via overflowing add, then adc rax, 5
    // mov rax, -1 / add rax, 1 -> rax=0, CF=1
    // adc rax, 5 -> rax = 0 + 5 + 1 = 6, CF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x83, 0xC0, 0x01,                    // add rax, 1  (rax=0, CF=1)
        0x48, 0x83, 0xD0, 0x05,                    // adc rax, 5
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_adc_produces_carry_out() {
    // rax=0 (from overflowing add that set CF=1), rcx=-1
    // adc rax, rcx -> 0 + 0xFFFFFFFFFFFFFFFF + 1 = 0, CF=1, ZF=1
    // adc rax, rcx: opcode 0x11 (ADC r/m64, r64)
    //   ModRM: mod=11, reg=rcx(001), rm=rax(000) -> 0xC8
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x83, 0xC0, 0x01,                    // add rax, 1  (rax=0, CF=1)
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF, // mov rcx, -1
        0x48, 0x11, 0xC8,                          // adc rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 1);
}

#[test]
fn test_adc_overflow_flag() {
    // Signed overflow: 0x7FFFFFFFFFFFFFFF + 0 + 1 (CF=1) = 0x8000000000000000
    // This is i64::MAX + 1, which overflows into negative -> OF=1
    // mov rax, i64::MAX (0x7FFFFFFFFFFFFFFF): 48 B8 FF FF FF FF FF FF FF 7F
    // Set CF=1 using same trick as before but with rcx
    // Strategy: use rcx to generate CF=1, then adc rax, 0
    // mov rcx, -1 / add rcx, 1 -> rcx=0, CF=1
    // adc rax, 0: 48 83 D0 00
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // mov rax, i64::MAX
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF,                    // mov rcx, -1
        0x48, 0x83, 0xC1, 0x01,                                       // add rcx, 1 (rcx=0, CF=1)
        0x48, 0x83, 0xD0, 0x00,                                       // adc rax, 0
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, i64::MIN as u64); // 0x8000000000000000
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_sbb_no_borrow_in() {
    // sbb rax, 3 when CF=0: rax = 10 - 3 - 0 = 7
    // xor ensures CF=0, add rax, 10 adds without overflow
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x31, 0xC0,       // xor rax, rax  (rax=0, CF=0)
        0x48, 0x83, 0xC0, 0x0A, // add rax, 10
        0x48, 0x83, 0xD8, 0x03, // sbb rax, 3
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 7);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_sbb_with_borrow_in() {
    // sbb rax, rax when CF=1: rax = 0 - 0 - 1 = -1, CF=1
    // Set CF=1 via overflow, then sbb rax, rax
    // sbb rax, rax: 48 19 C0
    //   ModRM: mod=11, reg=rax(000), rm=rax(000) -> 0xC0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x83, 0xC0, 0x01,                    // add rax, 1  (rax=0, CF=1)
        0x48, 0x19, 0xC0,                          // sbb rax, rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX); // -1
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_sbb_produces_borrow_out() {
    // sbb rax, 1 when CF=0 and rax=0: 0 - 1 - 0 = -1, CF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x31, 0xC0,       // xor rax, rax  (rax=0, CF=0)
        0x48, 0x83, 0xD8, 0x01, // sbb rax, 1
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX); // 0 - 1 wraps
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_sbb_result_zero() {
    // sbb rax, rax when CF=0: rax = x - x - 0 = 0, CF=0, ZF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0x48, 0x31, 0xC9,                          // xor rcx, rcx  (clears CF)
        0x48, 0x19, 0xC0,                          // sbb rax, rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 1);
}

// Regression: shift by masked-zero count must NOT touch flags.
// Per x86: when the masked shift count is 0, CF/ZF/SF/OF are all unchanged.
// Earlier the lifter computed CF as `(val >> bit_width) & 1` for SHL and
// `(val >> -1) & 1` for SHR/SAR - both Cranelift-masked-to-0/63 - silently
// corrupting CF for any subsequent JC/JNC.

#[test]
fn test_shl_by_zero_preserves_flags() {
    // Set up RAX=1, CL=0, CF=1, ZF=1. After `shl rax, cl` with cl=0,
    // RAX must be unchanged and CF=1, ZF=1 must survive.
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x31, 0xC9,                          // xor rcx, rcx  -> cl=0, ZF=1
        0xF9,                                       // stc           -> CF=1 (ZF still 1)
        0x48, 0xD3, 0xE0,                          // shl rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 1, "SHL by 0 must preserve CF");
    assert_eq!(ctx.zf, 1, "SHL by 0 must preserve ZF");
}

#[test]
fn test_shr_by_zero_preserves_flags() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x31, 0xC9,                          // xor rcx, rcx
        0xF9,                                       // stc -> CF=1, ZF=1
        0x48, 0xD3, 0xE8,                          // shr rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 1, "SHR by 0 must preserve CF");
    assert_eq!(ctx.zf, 1, "SHR by 0 must preserve ZF");
}

#[test]
fn test_sar_by_zero_preserves_flags() {
    // Use RAX=-1 so the buggy SAR-by-0 would push MSB (=1) into CF and SF.
    // We confirm the prior (CF=1, SF=0) is preserved.
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x31, 0xC9,                          // xor rcx, rcx -> ZF=1, SF=0, CF=0
        0xF9,                                       // stc -> CF=1
        0x48, 0xD3, 0xF8,                          // sar rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX);
    assert_eq!(
        ctx.cf, 1,
        "SAR by 0 must preserve CF (not pick up MSB of val)"
    );
    assert_eq!(ctx.zf, 1, "SAR by 0 must preserve ZF");
    assert_eq!(
        ctx.sf, 0,
        "SAR by 0 must preserve SF (not pick up MSB of val)"
    );
}

// Direct STC/CLC/CMC sanity tests.

#[test]
fn test_stc_sets_cf() {
    let bytes = [0xF9, 0xC3]; // stc; ret
    let mut ctx = RegisterContext::default();
    ctx.cf = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_clc_clears_cf() {
    let bytes = [0xF8, 0xC3]; // clc; ret
    let mut ctx = RegisterContext::default();
    ctx.cf = 1;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_cmc_toggles_cf() {
    let bytes = [0xF5, 0xC3]; // cmc; ret
    let mut ctx = RegisterContext::default();
    ctx.cf = 1;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 0);

    let mut ctx = RegisterContext::default();
    ctx.cf = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_shr_by_zero_does_not_pick_msb_into_cf() {
    // Specifically targets the prior bug: SHR/SAR computed
    // cf_shift = count - 1 = -1, which Cranelift masked to bit_width-1 = 63,
    // putting the MSB of val into CF. With RAX's MSB = 1, the buggy code
    // would set CF=1 from the shift even though caller cleared CF.
    #[rustfmt::skip]
    let bytes = [
        // RAX = 0x8000000000000000 (MSB set)
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
        0x48, 0x31, 0xC9,                          // xor rcx, rcx -> CF=0, cl=0
        0x48, 0xD3, 0xE8,                          // shr rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0, "SHR by 0 must not move val's MSB into CF");
}

#[test]
fn test_shl_by_zero_does_not_pick_lsb_into_cf() {
    // Mirror of the above for SHL: buggy code computed cf_shift = bit_width-0
    // = 64, Cranelift-masked to 0, putting LSB of val into CF.
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1 (LSB=1)
        0x48, 0x31, 0xC9,                          // xor rcx, rcx -> CF=0, cl=0
        0x48, 0xD3, 0xE0,                          // shl rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0, "SHL by 0 must not move val's LSB into CF");
}

// Regression: 8-bit DIV/IDIV must use AX (not AL) as the dividend.
// x86 8-bit divide: AL = AX / r/m8, AH = AX % r/m8 - AH (high byte of AX)
// must contribute to the dividend.

#[test]
fn test_div_8bit_uses_ax_as_dividend() {
    // AX = 0x0100 (256), BL = 0x10 (16). 256 / 16 = 16 (AL=0x10), 256 % 16 = 0.
    // Buggy code took only AL (=0) as dividend and computed 0/16=0.
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x01, // mov ax, 0x0100
        0xB3, 0x10,             // mov bl, 0x10
        0xF6, 0xF3,             // div bl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0x10, "AL (quotient) should be 16");
    assert_eq!((ctx.rax >> 8) & 0xFF, 0x00, "AH (remainder) should be 0");
}

#[test]
fn test_idiv_8bit_uses_ax_as_dividend() {
    // AX = -100 (0xFF9C), BL = 7. -100 / 7 = -14 (AL = 0xF2),
    // -100 % 7 = -2 (AH = 0xFE).
    // Buggy code took AL=0x9C as the dividend, giving wrong values.
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x9C, 0xFF, // mov ax, 0xFF9C  (-100)
        0xB3, 0x07,             // mov bl, 7
        0xF6, 0xFB,             // idiv bl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!((ctx.rax & 0xFF) as i8, -14, "AL (quotient) should be -14");
    assert_eq!(
        ((ctx.rax >> 8) & 0xFF) as i8,
        -2,
        "AH (remainder) should be -2"
    );
}

// XADD operand aliasing
//
// XADD writes both of its operands, so the two writes have to be ordered
// against each other. `handle_xadd` resolves the destination to a `Place`
// before either write, which freezes a memory operand's effective address;
// these are the cases that ordering exists for.

#[test]
fn test_xadd_distinct_registers() {
    // xadd rbx, rax  ->  rbx = rbx + rax, rax = old rbx
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, // mov rbx, 7
        0x48, 0x0F, 0xC1, 0xC3,                   // xadd rbx, rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rbx, 12, "dst holds the sum");
    assert_eq!(ctx.rax, 7, "src holds the old dst");
}

#[test]
fn test_xadd_same_register() {
    // xadd rax, rax. Hardware order is TEMP <- SRC + DEST; SRC <- DEST;
    // DEST <- TEMP, so with one register named twice the sum wins.
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x09, 0x00, 0x00, 0x00, // mov rax, 9
        0x48, 0x0F, 0xC1, 0xC0,                   // xadd rax, rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 18, "the destination write lands last");
}

#[test]
fn test_xadd_memory_base_is_source() {
    // xadd [rbx], rbx - the source register is also the address base, so the
    // store must go to the address computed *before* the source was
    // overwritten with the old memory value.
    let mut cell: u64 = 100;
    let addr = &mut cell as *mut u64 as u64;

    let mut ctx = RegisterContext::default();
    ctx.rbx = addr;

    // 48 0F C1 1B: XADD [RBX], RBX
    let bytes = [0x48, 0x0F, 0xC1, 0x1B, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(cell, addr.wrapping_add(100), "[rbx] = old mem + old rbx");
    assert_eq!(ctx.rbx, 100, "rbx holds the old memory value");
}

#[test]
fn test_xadd_memory_index_is_source() {
    // xadd [rax+rcx*8], rcx - the source is the *index* this time.
    let mut cells: [u64; 4] = [0, 0, 55, 0];
    let base = cells.as_mut_ptr() as u64;

    let mut ctx = RegisterContext::default();
    ctx.rax = base;
    ctx.rcx = 2;

    // 48 0F C1 0C C8: XADD [RAX+RCX*8], RCX
    let bytes = [0x48, 0x0F, 0xC1, 0x0C, 0xC8, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(cells[2], 57, "slot 2 = old mem + old rcx");
    assert_eq!(ctx.rcx, 55, "rcx holds the old memory value");
    assert_eq!(cells[0], 0, "no other slot is touched");
}
