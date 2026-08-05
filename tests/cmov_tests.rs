/// Tests for all CMOVcc (conditional move) variants.
///
/// Strategy: pre-seed the relevant flags in RegisterContext so that we can test
/// both the "condition true" (rax changes to rcx=42) and "condition false"
/// (rax stays 0) cases without emitting extra flag-setting instructions.
use microlift::testkit::prelude::*;

/// `cmovCC rax, rcx` with rax=0, rcx=42.
/// All tests follow: if condition true -> rax=42; if false -> rax=0.
///
/// Encoding: REX.W (48) + 0F + opcode + ModRM(C1)
///   ModRM C1 = mod=11, reg=rax(000), rm=rcx(001) -> 0b11_000_001

// ---------------------------------------------------------------------------
// CMOVE / CMOVZ  (ZF=1)
// ---------------------------------------------------------------------------

#[test]
fn test_cmove_true() {
    // ZF=1 -> condition true -> rax = 42
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
    // ZF=0 -> condition false -> rax stays 0
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // zf=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x44, 0xC1, // cmove rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVNE / CMOVNZ  (ZF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovne_true() {
    // ZF=0 -> condition true -> rax = 42
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // zf=0 by default
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
    // ZF=1 -> condition false -> rax stays 0
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

// ---------------------------------------------------------------------------
// CMOVB / CMOVC / CMOVNAE  (CF=1)
// ---------------------------------------------------------------------------

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
    // cf=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x42, 0xC1, // cmovb rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVAE / CMOVNB / CMOVNC  (CF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovae_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // cf=0 by default
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

// ---------------------------------------------------------------------------
// CMOVBE / CMOVNA  (CF=1 OR ZF=1)
// ---------------------------------------------------------------------------

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
    // cf=0, zf=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x46, 0xC1, // cmovbe rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVA / CMOVNBE  (CF=0 AND ZF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmova_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // cf=0, zf=0 by default
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

// ---------------------------------------------------------------------------
// CMOVL / CMOVNGE  (SF != OF)
// ---------------------------------------------------------------------------

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
    // sf=0, of=0 -> SF == OF -> not less
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4C, 0xC1, // cmovl rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVGE / CMOVNL  (SF == OF)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovge_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // sf=0, of=0 -> SF == OF
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

// ---------------------------------------------------------------------------
// CMOVLE / CMOVNG  (ZF=1 OR SF!=OF)
// ---------------------------------------------------------------------------

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
    // zf=0, sf=0, of=0 -> ZF=0 and SF==OF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4E, 0xC1, // cmovle rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVG / CMOVNLE  (ZF=0 AND SF==OF)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovg_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // zf=0, sf=0, of=0 by default
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

// ---------------------------------------------------------------------------
// CMOVS  (SF=1)
// ---------------------------------------------------------------------------

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
    // sf=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x48, 0xC1, // cmovs rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVNS  (SF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovns_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // sf=0 by default
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

// ---------------------------------------------------------------------------
// CMOVO  (OF=1)
// ---------------------------------------------------------------------------

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
    // of=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x40, 0xC1, // cmovo rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVNO  (OF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovno_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // of=0 by default
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

// ---------------------------------------------------------------------------
// CMOVP / CMOVPE  (PF=1)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovp_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.pf = 1;
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
    // pf=0 by default
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4A, 0xC1, // cmovp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// CMOVNP / CMOVPO  (PF=0)
// ---------------------------------------------------------------------------

#[test]
fn test_cmovnp_true() {
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    // pf=0 by default
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
    ctx.pf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x0F, 0x4B, 0xC1, // cmovnp rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0);
}

// ---------------------------------------------------------------------------
// Sanity: CMOVE with inline flag-setting code (no pre-seeding)
// ---------------------------------------------------------------------------

#[test]
fn test_cmove_via_cmp_equal() {
    // cmp rax, rax (rax=0=0 -> ZF=1) then cmove rax, rcx (rcx=42)
    // cmp rax, rax: 48 39 C0  (CMP r/m64, r64; reg=rax, rm=rax -> ModRM 0xC0)
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
    // mov rax, 1; mov rcx, 42; cmp rax, 0 (ZF=0) -> cmovne moves
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
