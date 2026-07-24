use microlift::testkit::prelude::*;

#[test]
fn test_mov_32_zero_extend() {
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xB8, 0x78, 0x56, 0x34, 0x12, // mov eax, 0x12345678
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x12345678);
}

#[test]
fn test_mov_16_partial() {
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, ...
        0x66, 0xB8, 0xAA, 0xAA, // mov ax, 0xAAAA
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566AAAA);
}

#[test]
fn test_mov_8_partial() {
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xB0,
        0xBB, // mov al, 0xBB
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x11223344556677BB);
}

#[test]
fn test_mov_8_high_partial() {
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xB4,
        0xCC, // mov ah, 0xCC
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566CC88);
}

#[test]
fn test_add_32_flags() {
    let bytes = [
        0xB8, 0xFF, 0xFF, 0xFF, 0x7F, // mov eax, 0x7FFFFFFF
        0x83, 0xC0, 0x01, // add eax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x80000000);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_sub_8_carry() {
    let bytes = [
        0xB0, 0x00, // mov al, 0
        0x2C, 0x01, // sub al, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0xFF);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.sf, 1);
}

#[test]
fn test_inc_8_overflow() {
    let bytes = [
        0xB0, 0x7F, // mov al, 0x7F
        0xFE, 0xC0, // inc al
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0x80);
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.sf, 1);
}

#[test]
fn test_not_preserves_flags() {
    let bytes = [
        0xB0, 0x00, // mov al, 0
        0x2C, 0x01, // sub al, 1
        0xF6, 0xD0, // not al
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0x00);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.sf, 1);
}

#[test]
fn test_movzx_reg8_to_reg32() {
    let bytes = [
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF, // mov rcx, -1
        0xB3, 0x80, // mov bl, 0x80
        0x0F, 0xB6, 0xCB, // movzx ecx, bl
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rcx, 0x80);
}
