use microlift::testkit::prelude::*;

/// AL from the context (low byte of RAX).
fn al(ctx: &RegisterContext) -> u8 {
    (ctx.rax & 0xFF) as u8
}

// `MOV RAX, imm32s` zero-extends 0xFFFFFFFF; load -1 via xor eax,eax ; dec rax (ZF=0 SF=1 OF=0, CF unchanged).

#[test]
fn test_sete_true() {
    // cmp rax, 0  (rax=0 from default) -> ZF=1 -> sete=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x00, // cmp rax, 0   -> ZF=1
        0x0F, 0x94, 0xC0,       // sete al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_sete_false() {
    // mov rax,10; cmp rax,5 -> ZF=0 -> sete=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, // mov rax, 10
        0x48, 0x83, 0xF8, 0x05,                   // cmp rax, 5  -> ZF=0
        0x0F, 0x94, 0xC0,                         // sete al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setne_true() {
    // mov rax,10; cmp rax,5 -> ZF=0 -> setne=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x05,                   // cmp rax, 5  -> ZF=0
        0x0F, 0x95, 0xC0,                         // setne al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setne_false() {
    // cmp rax,0 (rax=0) -> ZF=1 -> setne=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x00, // cmp rax, 0  -> ZF=1
        0x0F, 0x95, 0xC0,       // setne al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setb_true() {
    // rax=0, cmp rax,1 -> 0<1 unsigned -> CF=1 -> setb=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1  -> CF=1
        0x0F, 0x92, 0xC0,       // setb al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setb_false() {
    // mov rax,5; cmp rax,1 -> 5>=1 unsigned -> CF=0 -> setb=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,                   // cmp rax, 1  -> CF=0
        0x0F, 0x92, 0xC0,                         // setb al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setae_true() {
    // mov rax,5; cmp rax,1 -> CF=0 -> setae=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x93, 0xC0,                         // setae al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setae_false() {
    // rax=0; cmp rax,1 -> CF=1 -> setae=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x93, 0xC0,                         // setae al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setbe_true_equal() {
    // mov rax,5; cmp rax,5 -> ZF=1 -> setbe=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x05,
        0x0F, 0x96, 0xC0,                         // setbe al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setbe_true_below() {
    // rax=0; cmp rax,1 -> CF=1 -> setbe=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x96, 0xC0,                         // setbe al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setbe_false() {
    // mov rax,5; cmp rax,1 -> CF=0, ZF=0 -> setbe=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x96, 0xC0,                         // setbe al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_seta_true() {
    // mov rax,5; cmp rax,1 -> CF=0, ZF=0 -> seta=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x97, 0xC0,                         // seta al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_seta_false_equal() {
    // mov rax,5; cmp rax,5 -> ZF=1 -> seta=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x05,
        0x0F, 0x97, 0xC0,                         // seta al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_seta_false_below() {
    // rax=0; cmp rax,1 -> CF=1 -> seta=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x97, 0xC0,                         // seta al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setl_true() {
    // rax=-1 (31 C0 ; 48 FF C8); cmp rax,1 (48 83 F8 01) -> -2, SF=1, OF=0 -> setl=1
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,             // xor eax, eax  -> rax=0
        0x48, 0xFF, 0xC8,       // dec rax        -> rax=-1, SF=1, OF=0
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1    -> SF=1, OF=0
        0x0F, 0x9C, 0xC0,       // setl al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setl_false() {
    // mov rax,5; cmp rax,1 -> SF=0, OF=0 -> SF=OF -> setl=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x9C, 0xC0,                         // setl al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setge_true() {
    // mov rax,5; cmp rax,1 -> SF=0, OF=0 -> SF=OF -> setge=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x9D, 0xC0,                         // setge al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setge_false() {
    // rax=-1 (xor+dec); cmp rax,1 -> SF=1, OF=0 -> SF!=OF -> setge=0
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,       // rax=-1
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1    -> SF=1, OF=0
        0x0F, 0x9D, 0xC0,       // setge al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setle_true_equal() {
    // mov rax,5; cmp rax,5 -> ZF=1 -> setle=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x05,
        0x0F, 0x9E, 0xC0,                         // setle al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setle_true_less() {
    // rax=-1 (xor+dec); cmp rax,1 -> SF=1, OF=0 -> SF!=OF -> setle=1
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,       // rax=-1
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1    -> SF=1, OF=0
        0x0F, 0x9E, 0xC0,       // setle al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setle_false() {
    // mov rax,5; cmp rax,1 -> ZF=0, SF=0, OF=0 -> setle=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x9E, 0xC0,                         // setle al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setg_true() {
    // mov rax,5; cmp rax,1 -> ZF=0, SF=0, OF=0 -> setg=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x01,
        0x0F, 0x9F, 0xC0,                         // setg al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setg_false_equal() {
    // mov rax,5; cmp rax,5 -> ZF=1 -> setg=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x05,
        0x0F, 0x9F, 0xC0,                         // setg al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setg_false_less() {
    // rax=-1 (xor+dec); cmp rax,1 -> SF=1, OF=0 -> SF!=OF -> setg=0
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,       // rax=-1
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1    -> SF=1, OF=0
        0x0F, 0x9F, 0xC0,       // setg al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_sets_true() {
    // rax=-1 (xor+dec); cmp rax,0 -> result=-1, SF=1 -> sets=1
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,             // xor eax, eax  -> rax=0
        0x48, 0xFF, 0xC8,       // dec rax        -> rax=-1, SF=1
        0x48, 0x83, 0xF8, 0x00, // cmp rax, 0    -> result=-1, SF=1
        0x0F, 0x98, 0xC0,       // sets al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_sets_false() {
    // mov rax,5; cmp rax,0 -> SF=0 -> sets=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x00,
        0x0F, 0x98, 0xC0,                         // sets al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setns_true() {
    // mov rax,5; cmp rax,0 -> SF=0 -> setns=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xF8, 0x00,
        0x0F, 0x99, 0xC0,                         // setns al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setns_false() {
    // rax=-1 (xor+dec); cmp rax,0 -> result=-1, SF=1 -> setns=0
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,       // rax=-1
        0x48, 0x83, 0xF8, 0x00, // cmp rax, 0    -> SF=1
        0x0F, 0x99, 0xC0,       // setns al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_seto_true() {
    // rax = 0x7FFF... (xor ; dec ; shr 1); add 1 -> OF=1
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,             // xor eax, eax  -> rax=0
        0x48, 0xFF, 0xC8,       // dec rax        -> rax=0xFFFFFFFFFFFFFFFF
        0x48, 0xD1, 0xE8,       // shr rax, 1     -> rax=0x7FFFFFFFFFFFFFFF
        0x48, 0x83, 0xC0, 0x01, // add rax, 1     -> OF=1
        0x0F, 0x90, 0xC0,       // seto al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_seto_false() {
    // mov rax,5; add rax,1 -> no overflow -> OF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1   -> OF=0
        0x0F, 0x90, 0xC0,                         // seto al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setno_true() {
    // mov rax,5; add rax,1 -> OF=0 -> setno=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xC0, 0x01,
        0x0F, 0x91, 0xC0,                         // setno al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setno_false() {
    // OF=1 (same overflow sequence as test_seto_true) -> setno=0
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,       // rax=-1
        0x48, 0xD1, 0xE8,       // shr rax, 1
        0x48, 0x83, 0xC0, 0x01, // add rax, 1     -> OF=1
        0x0F, 0x91, 0xC0,       // setno al
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(al(&ctx), 0);
}

// PF is not computed by the flag routines, so pre-seed it in the context.

#[test]
fn test_setp_true() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x9A, 0xC0, // setp al
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.set_pf(1 != 0);
    let ctx = run_test(&bytes, ctx);
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setp_false() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x9A, 0xC0, // setp al
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.set_pf(0 != 0);
    let ctx = run_test(&bytes, ctx);
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setnp_true() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x9B, 0xC0, // setnp al
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.set_pf(0 != 0);
    let ctx = run_test(&bytes, ctx);
    assert_eq!(al(&ctx), 1);
}

#[test]
fn test_setnp_false() {
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x9B, 0xC0, // setnp al
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.set_pf(1 != 0);
    let ctx = run_test(&bytes, ctx);
    assert_eq!(al(&ctx), 0);
}

#[test]
fn test_setne_to_memory() {
    // mov rax,10; cmp rax,5; setne [rsp-8] (0F 95 44 24 F8); movzx rax,byte [rsp-8] (48 0F B6 44 24 F8); ret
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, // mov rax, 10
        0x48, 0x83, 0xF8, 0x05,                   // cmp rax, 5  -> ZF=0
        0x0F, 0x95, 0x44, 0x24, 0xF8,             // setne [rsp-8]
        0x48, 0x0F, 0xB6, 0x44, 0x24, 0xF8,       // movzx rax, byte [rsp-8]
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
}

#[test]
fn test_sete_to_memory() {
    // cmp rax,0 (ZF=1); sete [rsp-8]; movzx rax,byte [rsp-8]; ret
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xF8, 0x00,                   // cmp rax, 0  -> ZF=1
        0x0F, 0x94, 0x44, 0x24, 0xF8,             // sete [rsp-8]
        0x48, 0x0F, 0xB6, 0x44, 0x24, 0xF8,       // movzx rax, byte [rsp-8]
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
}
