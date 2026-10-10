/// neg, imul, mul, idiv/div, sign-extension, sar, rol, shld/shrd, bt, adc/sbb, xadd.
use microlift::testkit::prelude::*;


#[test]
fn test_neg() {
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


#[test]
fn test_imul_two_operand() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0x0F, 0xAF, 0xC1,                    // imul rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_imul_two_operand_negative() {
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
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // mov rax, i64::MAX
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,                   // mov rcx, 2
        0x48, 0xF7, 0xE9,                                            // imul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFEu64);
    assert_eq!(ctx.rdx, 0); // high half is 0 for this particular product
    // Low half -2 does not sign-extend to the full product, so OF=CF=1.
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}


#[test]
fn test_mul() {
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

/// Narrow `mul` forms read operands at their own width; re-narrowing once produced malformed IR.
#[test]
fn test_mul_32_and_16_bit() {
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

/// Same double-narrowing in one-operand `imul`.
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


#[test]
fn test_idiv() {
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


#[test]
fn test_cbw() {
    #[rustfmt::skip]
    let bytes = [
        0xB0, 0xFF, // mov al, 0xFF
        0x66, 0x98, // cbw
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0xFFFF);

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
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x98,                   // cwde
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x00000000FFFF8000u64);

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
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x48, 0x98,                    // cdqe
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFF80000000u64);

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
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x66, 0x99,             // cwd
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x8000); // AX unchanged
    assert_eq!(ctx.rdx & 0xFFFF, 0xFFFF); // DX = -1

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
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x99,                          // cdq
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000080000000u64);
    assert_eq!(ctx.rdx, 0x00000000FFFFFFFFu64);

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
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 0x8000...
        0x48, 0x99,                                                    // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, u64::MAX); // 0xFFFFFFFFFFFFFFFF

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x99,                                // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, 0);
}


#[test]
fn test_sar_positive() {
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


#[test]
fn test_rol_msb_rotates_to_lsb() {
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
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, // mov rax, 0x0F0F...
        0x48, 0xC1, 0xC0, 0x04,                                       // rol rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xF0F0F0F0F0F0F0F0u64);
}


#[test]
fn test_shld() {
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


#[test]
fn test_bt_immediate() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xE0, 0x03,             // bt rax, 3
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1); // bit 3 was set
    assert_eq!(ctx.rax, 8); // rax unchanged

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
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10 (bit 4 set)
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0x0F, 0xA3, 0xC8,                    // bt rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1);

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

/// Regression: a masked-zero shift count must leave CF/ZF/SF/OF untouched.

#[test]
fn test_shl_by_zero_preserves_flags() {
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
    // SHR/SAR must not leak the MSB into CF via a masked count of -1.
    #[rustfmt::skip]
    let bytes = [
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
    // SHL must not leak the LSB into CF via a masked count of 64.
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

/// Regression: 8-bit DIV/IDIV divides AX, not AL.

#[test]
fn test_div_8bit_uses_ax_as_dividend() {
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

// XADD writes both operands; a memory destination's address must be frozen before either write.

#[test]
fn test_xadd_distinct_registers() {
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
    // With one register named twice the sum wins.
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
    // The store goes to the address computed before the source was overwritten.
    let mut cell: u64 = 100;
    let addr = &mut cell as *mut u64 as u64;

    let mut ctx = RegisterContext::default();
    ctx.rbx = addr;

    let bytes = [0x48, 0x0F, 0xC1, 0x1B, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(cell, addr.wrapping_add(100), "[rbx] = old mem + old rbx");
    assert_eq!(ctx.rbx, 100, "rbx holds the old memory value");
}

#[test]
fn test_xadd_memory_index_is_source() {
    // Source is the index register.
    let mut cells: [u64; 4] = [0, 0, 55, 0];
    let base = cells.as_mut_ptr() as u64;

    let mut ctx = RegisterContext::default();
    ctx.rax = base;
    ctx.rcx = 2;

    let bytes = [0x48, 0x0F, 0xC1, 0x0C, 0xC8, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(cells[2], 57, "slot 2 = old mem + old rcx");
    assert_eq!(ctx.rcx, 55, "rcx holds the old memory value");
    assert_eq!(cells[0], 0, "no other slot is touched");
}
