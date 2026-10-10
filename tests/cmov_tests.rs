/// All CMOVcc variants, with flags pre-seeded in the context.
/// Each test moves rcx=42 into rax=0 when the condition holds (`48 0F <op> C1`).
use microlift::testkit::prelude::*;



#[test]
fn test_cmove_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x44, 0xC1, // cmove rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmove_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x44, 0xC1, // cmove rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovne_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x45, 0xC1, // cmovne rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovne_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x45, 0xC1, // cmovne rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovb_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.cf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x42, 0xC1, // cmovb rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovb_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x42, 0xC1, // cmovb rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovae_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x43, 0xC1, // cmovae rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovae_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.cf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x43, 0xC1, // cmovae rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovbe_true_via_cf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.cf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x46, 0xC1, // cmovbe rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovbe_true_via_zf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x46, 0xC1, // cmovbe rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovbe_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x46, 0xC1, // cmovbe rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmova_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x47, 0xC1, // cmova rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmova_false_via_cf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.cf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x47, 0xC1, // cmova rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_cmova_false_via_zf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x47, 0xC1, // cmova rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovl_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1; // SF=1, OF=0 -> SF != OF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4C, 0xC1, // cmovl rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovl_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4C, 0xC1, // cmovl rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovge_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4D, 0xC1, // cmovge rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovge_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1; // SF != OF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4D, 0xC1, // cmovge rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovle_true_via_zf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4E, 0xC1, // cmovle rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovle_true_via_sf_ne_of() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1; // SF=1, OF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4E, 0xC1, // cmovle rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovle_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4E, 0xC1, // cmovle rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovg_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4F, 0xC1, // cmovg rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovg_false_via_zf() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.zf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4F, 0xC1, // cmovg rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_cmovg_false_via_sf_ne_of() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1; // SF != OF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4F, 0xC1, // cmovg rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovs_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x48, 0xC1, // cmovs rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovs_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x48, 0xC1, // cmovs rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovns_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x49, 0xC1, // cmovns rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovns_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.sf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x49, 0xC1, // cmovns rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovo_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.of = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x40, 0xC1, // cmovo rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovo_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x40, 0xC1, // cmovo rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovno_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x41, 0xC1, // cmovno rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovno_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.of = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x41, 0xC1, // cmovno rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovp_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.set_pf(1 != 0);
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4A, 0xC1, // cmovp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovp_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4A, 0xC1, // cmovp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmovnp_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4B, 0xC1, // cmovnp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovnp_false() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.set_pf(1 != 0);
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4B, 0xC1, // cmovnp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}


#[test]
fn test_cmove_via_cmp_equal() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x39, 0xC0,       // cmp rax, rax  -> ZF=1
        0x48, 0x0F, 0x44, 0xC1, // cmove rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmovne_via_cmp_unequal() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xC7, 0xC1, 0x2A, 0x00, 0x00, 0x00, // mov rcx, 42
        0x48, 0x83, 0xF8, 0x00,                   // cmp rax, 0 -> ZF=0
        0x48, 0x0F, 0x45, 0xC1,                   // cmovne rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
}
