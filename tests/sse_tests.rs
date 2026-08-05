//! SSE / SSE2 instruction tests.
//!
//! Each test lifts a small x86-64 snippet and verifies the resulting
//! `RegisterContext` has the expected XMM (and where relevant GPR/memory)
//! state after JIT execution.

use microlift::testkit::prelude::*;

/// Scalar-XMM (default) lift-and-run.
fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

/// Vector-XMM (`use_vector_xmm = true`) lift-and-run.
fn run_test_vec(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).vector_xmm(true).ctx(ctx).run().ctx
}

//  ENDBR64

#[test]
fn test_endbr64_nop() {
    // endbr64; ret   ->   F3 0F 1E FA  C3
    let bytes = [0xF3, 0x0F, 0x1E, 0xFA, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x42;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x42, "endbr64 must not modify registers");
    assert_eq!(ctx.xmm[0], [0, 0], "endbr64 must not modify xmm0");
}

//  XORPS / XORPD / PXOR

#[test]
fn test_xorps_self_zeros() {
    // xorps xmm0, xmm0; ret   ->   0F 57 C0  C3
    let bytes = [0x0F, 0x57, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0, 0], "xorps xmm0,xmm0 must zero xmm0");
}

#[test]
fn test_xorps_two_regs() {
    // xorps xmm0, xmm1; ret   ->   0F 57 C1  C3
    let bytes = [0x0F, 0x57, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xF0F0_F0F0_F0F0_F0F0, 0x0000_FFFF_0000_FFFF];
    ctx.xmm[1] = [0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            0xF0F0_F0F0_F0F0_F0F0 ^ 0xFF00_FF00_FF00_FF00,
            0x0000_FFFF_0000_FFFF ^ 0xFFFF_0000_FFFF_0000,
        ],
        "xorps xmm0,xmm1 must XOR all 128 bits"
    );
}

#[test]
fn test_pxor_same_as_xorps() {
    // pxor xmm0, xmm1; ret   ->   66 0F EF C1  C3
    let bytes = [0x66, 0x0F, 0xEF, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555];
    ctx.xmm[1] = [0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]);
}

//  ANDPS / ANDPD / PAND

#[test]
fn test_andps() {
    // andps xmm0, xmm1; ret   ->   0F 54 C1  C3
    let bytes = [0x0F, 0x54, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFF00_FF00_FF00_FF00, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.xmm[1] = [0xF0F0_F0F0_F0F0_F0F0, 0xAAAA_AAAA_AAAA_AAAA];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            0xFF00_FF00_FF00_FF00 & 0xF0F0_F0F0_F0F0_F0F0,
            0x0F0F_0F0F_0F0F_0F0F & 0xAAAA_AAAA_AAAA_AAAA,
        ],
        "andps xmm0,xmm1 must AND all 128 bits"
    );
}

#[test]
fn test_pand_same_as_andps() {
    // pand xmm0, xmm1; ret   ->   66 0F DB C1  C3
    let bytes = [0x66, 0x0F, 0xDB, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0000];
    ctx.xmm[1] = [0xAAAA_BBBB_CCCC_DDDD, 0xEEEE_FFFF_1111_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xAAAA_BBBB_CCCC_DDDD);
    assert_eq!(ctx.xmm[0][1], 0x0000_0000_0000_0000);
}

//  ORPS / ORPD / POR

#[test]
fn test_orps() {
    // orps xmm0, xmm1; ret   ->   0F 56 C1  C3
    let bytes = [0x0F, 0x56, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_AAAA_AAAA_AAAA, 0x0000_0000_0000_0000];
    ctx.xmm[1] = [0x5555_5555_5555_5555, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]);
}

#[test]
fn test_por_same_as_orps() {
    // por xmm0, xmm1; ret   ->   66 0F EB C1  C3
    let bytes = [0x66, 0x0F, 0xEB, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1234_0000_0000_0000, 0x0000_0000_0000_5678];
    ctx.xmm[1] = [0x0000_ABCD_0000_0000, 0x0000_0000_EF01_0000];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x1234_ABCD_0000_0000);
    assert_eq!(ctx.xmm[0][1], 0x0000_0000_EF01_5678);
}

//  ANDNPS / ANDNPD / PANDN

#[test]
fn test_andnps() {
    // ANDNPS: result = ~xmm0 & xmm1
    // andnps xmm0, xmm1; ret   ->   0F 55 C1  C3
    let bytes = [0x0F, 0x55, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xF0F0_F0F0_F0F0_F0F0, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.xmm[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    // ~0xF0F0... = 0x0F0F...;  ~0x0F0F... = 0xF0F0...
    assert_eq!(ctx.xmm[0][0], 0x0F0F_0F0F_0F0F_0F0F);
    assert_eq!(ctx.xmm[0][1], 0xF0F0_F0F0_F0F0_F0F0);
}

#[test]
fn test_andnps_selective_mask() {
    // andnps xmm0, xmm1; ret   ->   0F 55 C1  C3
    let bytes = [0x0F, 0x55, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // dst (xmm0) is a mask; source (xmm1) is data.
    ctx.xmm[0] = [0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF];
    ctx.xmm[1] = [0xAAAA_AAAA_BBBB_BBBB, 0xCCCC_CCCC_DDDD_DDDD];
    let ctx = run_test(&bytes, ctx);
    // result = ~xmm0 & xmm1
    assert_eq!(ctx.xmm[0][0], 0x0000_0000_BBBB_BBBB);
    assert_eq!(ctx.xmm[0][1], 0xCCCC_CCCC_0000_0000);
}

#[test]
fn test_pandn_same_as_andnps() {
    // pandn xmm0, xmm1; ret   ->   66 0F DF C1  C3
    let bytes = [0x66, 0x0F, 0xDF, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0000];
    ctx.xmm[1] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    let ctx = run_test(&bytes, ctx);
    // ~xmm0[lo] & xmm1[lo] = 0 & anything = 0
    assert_eq!(ctx.xmm[0][0], 0x0000_0000_0000_0000);
    // ~xmm0[hi] & xmm1[hi] = 0xFFFF... & 0x1111... = 0x1111...
    assert_eq!(ctx.xmm[0][1], 0x1111_2222_3333_4444);
}

//  MOVAPS / MOVUPS (register-to-register and memory)

#[test]
fn test_movaps_reg_reg() {
    // movaps xmm0, xmm1; ret   ->   0F 28 C1  C3
    let bytes = [0x0F, 0x28, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210]);
}

#[test]
fn test_movups_mem_to_reg() {
    // movups xmm0, [rax]; ret   ->   0F 10 00  C3
    let bytes = [0x0F, 0x10, 0x00, 0xC3];
    let mem: [u64; 2] = [0xDEAD_BEEF_CAFE_BABE, 0x1111_2222_3333_4444];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_ptr() as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(ctx.xmm[0][1], 0x1111_2222_3333_4444);
}

#[test]
fn test_movups_reg_to_mem() {
    // movups [rax], xmm0; ret   ->   0F 11 00  C3
    let bytes = [0x0F, 0x11, 0x00, 0xC3];
    let mut mem = [0u64; 2];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_mut_ptr() as u64;
    ctx.xmm[0] = [0xAAAA_BBBB_CCCC_DDDD, 0xEEEE_FFFF_1111_2222];
    run_test(&bytes, ctx);
    assert_eq!(mem[0], 0xAAAA_BBBB_CCCC_DDDD);
    assert_eq!(mem[1], 0xEEEE_FFFF_1111_2222);
}

#[test]
fn test_movdqu_reg_reg() {
    // movdqu xmm0, xmm1; ret   ->   F3 0F 6F C1  C3
    let bytes = [0xF3, 0x0F, 0x6F, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0xABCD_EF01_2345_6789, 0x9876_5432_1012_3456];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xABCD_EF01_2345_6789, 0x9876_5432_1012_3456]);
}

//  MOVQ

#[test]
fn test_movq_gpr_to_xmm() {
    // movq xmm0, rax; ret   ->   66 48 0F 6E C0  C3
    let bytes = [0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_CAFE_BABE;
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(
        ctx.xmm[0][1], 0,
        "high 64 bits must be zeroed by movq xmm,r64"
    );
}

#[test]
fn test_movq_xmm_to_gpr() {
    // movq rax, xmm0; ret   ->   66 48 0F 7E C0  C3
    let bytes = [0x66, 0x48, 0x0F, 0x7E, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1234_5678_ABCD_EF01, 0x9999_9999_9999_9999];
    ctx.rax = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rax, 0x1234_5678_ABCD_EF01,
        "movq r64,xmm must copy low 64 bits"
    );
}

#[test]
fn test_movq_mem_to_xmm() {
    // movq xmm0, [rax]; ret   ->   66 48 0F 6E 00  C3
    let bytes = [0x66, 0x48, 0x0F, 0x6E, 0x00, 0xC3];
    let val: u64 = 0xBEEF_CAFE_DEAD_1234;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const u64 as u64;
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xBEEF_CAFE_DEAD_1234);
    assert_eq!(
        ctx.xmm[0][1], 0,
        "high 64 bits must be zeroed by movq xmm,m64"
    );
}

#[test]
fn test_movq_xmm_to_mem() {
    // movq [rax], xmm0; ret   ->   66 48 0F 7E 00  C3
    let bytes = [0x66, 0x48, 0x0F, 0x7E, 0x00, 0xC3];
    let mut mem: u64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut mem as *mut u64 as u64;
    ctx.xmm[0] = [0xCAFE_BABE_1234_5678, 0xFFFF_FFFF_FFFF_FFFF];
    run_test(&bytes, ctx);
    assert_eq!(
        mem, 0xCAFE_BABE_1234_5678,
        "movq m64,xmm must store low 64 bits"
    );
}

//  MOVSS

#[test]
fn test_movss_reg_reg_merges_low32() {
    // MOVSS xmm0, xmm1: bits[31:0] of dst replaced; upper 96 bits preserved.
    // movss xmm0, xmm1; ret   ->   F3 0F 10 C1  C3
    let bytes = [0xF3, 0x0F, 0x10, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    // low 32 bits of xmm1 lane 0 = 0x0000_1111
    ctx.xmm[1] = [0xFFFF_FFFF_0000_1111, 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    // bits[31:0] of xmm0 lane 0 come from xmm1; bits[63:32] of lane 0 preserved
    assert_eq!(
        ctx.xmm[0][0], 0xAAAA_BBBB_0000_1111,
        "movss xmm,xmm: low 32 merged"
    );
    // high lane (bits[127:64]) unchanged
    assert_eq!(
        ctx.xmm[0][1], 0x1111_2222_3333_4444,
        "movss xmm,xmm: high lane preserved"
    );
}

#[test]
fn test_movss_load_from_mem() {
    // MOVSS xmm0, [rax]: load 32 bits; upper 96 bits zeroed.
    // movss xmm0, [rax]; ret   ->   F3 0F 10 00  C3
    let bytes = [0xF3, 0x0F, 0x10, 0x00, 0xC3];
    let val: u32 = 0xDEAD_BEEF;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const u32 as u64;
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xDEAD_BEEF as u64);
    assert_eq!(
        ctx.xmm[0][1], 0,
        "upper 96 bits must be zero after movss xmm,m32"
    );
}

#[test]
fn test_movss_store_to_mem() {
    // MOVSS [rax], xmm0: store low 32 bits.
    // movss [rax], xmm0; ret   ->   F3 0F 11 00  C3
    let bytes = [0xF3, 0x0F, 0x11, 0x00, 0xC3];
    let mut mem: u32 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut mem as *mut u32 as u64;
    // low 32 bits of xmm0 lane 0 = 0x1234_5678
    ctx.xmm[0] = [0xAAAA_BBBB_1234_5678, 0x9999_9999_9999_9999];
    run_test(&bytes, ctx);
    assert_eq!(mem, 0x1234_5678u32, "movss m32,xmm must store low 32 bits");
}

//  MOVSD (SSE2)

#[test]
fn test_movsd_reg_reg_merges_low64() {
    // MOVSD xmm0, xmm1: lane 0 of dst replaced; lane 1 preserved.
    // movsd xmm0, xmm1; ret   ->   F2 0F 10 C1  C3
    let bytes = [0xF2, 0x0F, 0x10, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_AAAA_AAAA_AAAA, 0x9999_9999_9999_9999];
    ctx.xmm[1] = [0x1234_5678_ABCD_EF01, 0x0000_0000_0000_0000];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0][0], 0x1234_5678_ABCD_EF01,
        "movsd xmm,xmm: lane 0 from src"
    );
    assert_eq!(
        ctx.xmm[0][1], 0x9999_9999_9999_9999,
        "movsd xmm,xmm: lane 1 preserved"
    );
}

#[test]
fn test_movsd_load_from_mem() {
    // MOVSD xmm0, [rax]: load 64 bits; lane 1 zeroed.
    // movsd xmm0, [rax]; ret   ->   F2 0F 10 00  C3
    let bytes = [0xF2, 0x0F, 0x10, 0x00, 0xC3];
    let val: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut ctx = RegisterContext::default();
    ctx.rax = &val as *const u64 as u64;
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(ctx.xmm[0][1], 0, "lane 1 must be zero after movsd xmm,m64");
}

#[test]
fn test_movsd_store_to_mem() {
    // MOVSD [rax], xmm0: store lane 0.
    // movsd [rax], xmm0; ret   ->   F2 0F 11 00  C3
    let bytes = [0xF2, 0x0F, 0x11, 0x00, 0xC3];
    let mut mem: u64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut mem as *mut u64 as u64;
    ctx.xmm[0] = [0xCAFE_BABE_DEAD_BEEF, 0xFFFF_FFFF_FFFF_FFFF];
    run_test(&bytes, ctx);
    assert_eq!(
        mem, 0xCAFE_BABE_DEAD_BEEF,
        "movsd m64,xmm must store lane 0"
    );
}

//  Integration

#[test]
fn test_pxor_self_zeros_any_value() {
    // pxor xmm0, xmm0; ret   ->   66 0F EF C0  C3
    let bytes = [0x66, 0x0F, 0xEF, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [u64::MAX, u64::MAX];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0, 0]);
}

#[test]
fn test_xmm_bitwise_chain() {
    // orps xmm0, xmm1; andps xmm0, xmm2; ret
    // 0F 56 C1  0F 54 C2  C3
    let bytes = [0x0F, 0x56, 0xC1, 0x0F, 0x54, 0xC2, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0000_0000_0000_0000, 0x0000_0000_0000_0000];
    ctx.xmm[1] = [0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555];
    ctx.xmm[2] = [0xFF00_FF00_FF00_FF00, 0x00FF_00FF_00FF_00FF];
    let ctx = run_test(&bytes, ctx);
    // orps  xmm0, xmm1 -> xmm0 = xmm1
    // andps xmm0, xmm2 -> xmm0 = xmm1 & xmm2
    assert_eq!(ctx.xmm[0][0], 0xAAAA_AAAA_AAAA_AAAA & 0xFF00_FF00_FF00_FF00);
    assert_eq!(ctx.xmm[0][1], 0x5555_5555_5555_5555 & 0x00FF_00FF_00FF_00FF);
}

#[test]
fn test_xmm_copy_and_xor() {
    // Copy xmm1 into xmm0 then XOR xmm0 with itself to zero it.
    // movaps xmm0, xmm1; xorps xmm0, xmm0; ret
    // 0F 28 C1  0F 57 C0  C3
    let bytes = [0x0F, 0x28, 0xC1, 0x0F, 0x57, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF, 0x0];
    ctx.xmm[1] = [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210];
    let ctx = run_test(&bytes, ctx);
    // After movaps: xmm0 = xmm1; after xorps xmm0,xmm0: xmm0 = 0
    assert_eq!(ctx.xmm[0], [0, 0]);
    // xmm1 must be unchanged
    assert_eq!(ctx.xmm[1], [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210]);
}

#[test]
fn test_movq_roundtrip_gpr_xmm() {
    // Move rax into xmm0, then move xmm0 back into rcx.
    // movq xmm0, rax; movq rcx, xmm0; ret
    // 66 48 0F 6E C0   66 48 0F 7E C1   C3
    //   xmm0 ← rax          rcx ← xmm0
    let bytes = [
        0x66, 0x48, 0x0F, 0x6E, 0xC0, // movq xmm0, rax
        0x66, 0x48, 0x0F, 0x7E, 0xC1, // movq rcx, xmm0
        0xC3, // ret
    ];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xCAFE_BABE_1234_5678;
    ctx.rcx = 0;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rcx, 0xCAFE_BABE_1234_5678);
    assert_eq!(ctx.xmm[0][1], 0, "movq must zero high lane");
}

//  Vector-mode (use_vector_xmm = true) regression tests.
//
//  Mirror a representative subset of the bitwise / move handlers above
//  but flip on the I64X2 backing representation. This exercises the
//  vector load/store paths in `load_state_from_context` /
//  `sync_to_context_selective` and the vector fast paths in
//  handle_xorps / handle_andps / handle_orps / handle_andnps /
//  handle_movaps.

#[test]
fn test_vec_xorps_two_regs() {
    let bytes = [0x0F, 0x57, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xF0F0_F0F0_F0F0_F0F0, 0x0000_FFFF_0000_FFFF];
    ctx.xmm[1] = [0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            0xF0F0_F0F0_F0F0_F0F0 ^ 0xFF00_FF00_FF00_FF00,
            0x0000_FFFF_0000_FFFF ^ 0xFFFF_0000_FFFF_0000,
        ],
    );
}

#[test]
fn test_vec_andps() {
    let bytes = [0x0F, 0x54, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFF00_FF00_FF00_FF00, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.xmm[1] = [0xF0F0_F0F0_F0F0_F0F0, 0xAAAA_AAAA_AAAA_AAAA];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            0xFF00_FF00_FF00_FF00 & 0xF0F0_F0F0_F0F0_F0F0,
            0x0F0F_0F0F_0F0F_0F0F & 0xAAAA_AAAA_AAAA_AAAA,
        ],
    );
}

#[test]
fn test_vec_orps() {
    // orps xmm0, xmm1; ret   ->   0F 56 C1  C3
    let bytes = [0x0F, 0x56, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x00FF_00FF_00FF_00FF, 0xAAAA_0000_AAAA_0000];
    ctx.xmm[1] = [0xFF00_FF00_FF00_FF00, 0x0000_5555_0000_5555];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            0x00FF_00FF_00FF_00FF | 0xFF00_FF00_FF00_FF00,
            0xAAAA_0000_AAAA_0000 | 0x0000_5555_0000_5555,
        ],
    );
}

#[test]
fn test_vec_andnps() {
    // andnps xmm0, xmm1; ret   ->   0F 55 C1  C3   (result = ~xmm0 & xmm1)
    let bytes = [0x0F, 0x55, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFF00_FF00_FF00_FF00, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.xmm[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xAAAA_AAAA_AAAA_AAAA];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [
            !0xFF00_FF00_FF00_FF00u64 & 0xFFFF_FFFF_FFFF_FFFF,
            !0x0F0F_0F0F_0F0F_0F0Fu64 & 0xAAAA_AAAA_AAAA_AAAA,
        ],
    );
}

#[test]
fn test_vec_movaps_reg_reg() {
    // movaps xmm0, xmm1; ret   ->   0F 28 C1  C3
    let bytes = [0x0F, 0x28, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0, 0];
    ctx.xmm[1] = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0]);
}

#[test]
fn test_vec_movq_roundtrip_gpr_xmm() {
    // Exercises lo/hi accessors in vector mode (extractlane / insertlane).
    let bytes = [
        0x66, 0x48, 0x0F, 0x6E, 0xC0, // movq xmm0, rax
        0x66, 0x48, 0x0F, 0x7E, 0xC1, // movq rcx, xmm0
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xCAFE_BABE_1234_5678;
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.rcx, 0xCAFE_BABE_1234_5678);
    assert_eq!(ctx.xmm[0][1], 0, "movq must zero high lane");
}

//  MOVD

#[test]
fn test_movd_gpr_to_xmm() {
    // movd xmm0, eax; ret  ->  66 0F 6E C0  C3
    let bytes = [0x66, 0x0F, 0x6E, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xFFFF_FFFF_DEAD_BEEF; // only eax (0xDEADBEEF) participates
    ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0xDEAD_BEEF, 0],
        "movd must load low dword and zero the rest"
    );
}

#[test]
fn test_movd_xmm_to_gpr() {
    // movd eax, xmm0; ret  ->  66 0F 7E C0  C3
    let bytes = [0x66, 0x0F, 0x7E, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xAAAA_AAAA_AAAA_AAAA;
    ctx.xmm[0] = [0x1122_3344_5566_7788, 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rax, 0x5566_7788,
        "movd r32,xmm extracts low dword and zero-extends"
    );
}

//  MOVHPS

#[test]
fn test_movhps_load_high() {
    // movhps xmm0, [rdi]; ret  ->  0F 16 07  C3
    let bytes = [0x0F, 0x16, 0x07, 0xC3];
    let mut buf = [0u64; 2];
    buf[0] = 0xCAFE_BABE_DEAD_BEEF;
    let mut ctx = RegisterContext::default();
    ctx.rdi = buf.as_ptr() as u64;
    ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x1111_1111_1111_1111, 0xCAFE_BABE_DEAD_BEEF],
        "movhps load must replace the high quadword and preserve the low"
    );
}

#[test]
fn test_movhps_store_high() {
    // movhps [rdi], xmm0; ret  ->  0F 17 07  C3
    let bytes = [0x0F, 0x17, 0x07, 0xC3];
    let mut buf = [0u64; 2];
    let mut ctx = RegisterContext::default();
    ctx.rdi = buf.as_mut_ptr() as u64;
    ctx.xmm[0] = [0x1234_5678_9ABC_DEF0, 0xFEED_FACE_CAFE_F00D];
    let _ = run_test(&bytes, ctx);
    assert_eq!(
        buf[0], 0xFEED_FACE_CAFE_F00D,
        "movhps store must write the high quadword"
    );
}

//  MOVLPS / MOVLPD - merging load / store of the low quadword.
//
//  Both XMM representations are exercised, since the lo/hi accessors take
//  different paths in vector mode.

#[test]
fn test_movlps_load_low() {
    // movlps xmm0, [rdi]; ret  ->  0F 12 07  C3
    let bytes = [0x0F, 0x12, 0x07, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let buf = [0xCAFE_BABE_DEAD_BEEFu64, 0x0BAD_0BAD_0BAD_0BAD];
        let mut ctx = RegisterContext::default();
        ctx.rdi = buf.as_ptr() as u64;
        ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        let ctx = run(&bytes, ctx);
        assert_eq!(
            ctx.xmm[0],
            [0xCAFE_BABE_DEAD_BEEF, 0x2222_2222_2222_2222],
            "movlps load must replace the low quadword and preserve the high"
        );
    }
}

#[test]
fn test_movlpd_load_low_preserves_high() {
    // movlpd xmm0, [rdi]; ret  ->  66 0F 12 07  C3
    // Unlike movsd xmm, m64, this must NOT zero the high quadword.
    let bytes = [0x66, 0x0F, 0x12, 0x07, 0xC3];
    let buf = [0x1234_5678_9ABC_DEF0u64];
    let mut ctx = RegisterContext::default();
    ctx.rdi = buf.as_ptr() as u64;
    ctx.xmm[0] = [0, 0xFEED_FACE_CAFE_F00D];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x1234_5678_9ABC_DEF0, 0xFEED_FACE_CAFE_F00D],
        "movlpd is a merging load, not a zeroing one"
    );
}

#[test]
fn test_movlps_store_low() {
    // movlps [rdi], xmm0; ret  ->  0F 13 07  C3
    let bytes = [0x0F, 0x13, 0x07, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut buf = [0u64; 2];
        let mut ctx = RegisterContext::default();
        ctx.rdi = buf.as_mut_ptr() as u64;
        ctx.xmm[0] = [0x1234_5678_9ABC_DEF0, 0xFEED_FACE_CAFE_F00D];
        let _ = run(&bytes, ctx);
        assert_eq!(
            buf[0], 0x1234_5678_9ABC_DEF0,
            "movlps store must write the low quadword"
        );
        assert_eq!(buf[1], 0, "movlps store must only write 8 bytes");
    }
}

//  MOVHLPS / MOVLHPS - the register-to-register half moves that share the
//  0F 12 / 0F 16 opcodes with MOVLPS / MOVHPS.

#[test]
fn test_movhlps() {
    // movhlps xmm0, xmm1; ret  ->  0F 12 C1  C3
    let bytes = [0x0F, 0x12, 0xC1, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        ctx.xmm[1] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
        let ctx = run(&bytes, ctx);
        assert_eq!(
            ctx.xmm[0],
            [0x4444_4444_4444_4444, 0x2222_2222_2222_2222],
            "movhlps must move src high -> dst low, preserving dst high"
        );
        assert_eq!(
            ctx.xmm[1],
            [0x3333_3333_3333_3333, 0x4444_4444_4444_4444],
            "movhlps must not modify the source"
        );
    }
}

#[test]
fn test_movlhps() {
    // movlhps xmm0, xmm1; ret  ->  0F 16 C1  C3
    let bytes = [0x0F, 0x16, 0xC1, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        ctx.xmm[1] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
        let ctx = run(&bytes, ctx);
        assert_eq!(
            ctx.xmm[0],
            [0x1111_1111_1111_1111, 0x3333_3333_3333_3333],
            "movlhps must move src low -> dst high, preserving dst low"
        );
    }
}

#[test]
fn test_movhlps_movlhps_swap_halves() {
    // A round trip through xmm2 swaps the halves of xmm0:
    //   movhlps xmm2, xmm0   -> xmm2.lo = xmm0.hi   (0F 12 D0)
    //   movlhps xmm2, xmm0   -> xmm2.hi = xmm0.lo   (0F 16 D0)
    //   movaps  xmm0, xmm2                          (0F 28 C2)
    let bytes = [0x0F, 0x12, 0xD0, 0x0F, 0x16, 0xD0, 0x0F, 0x28, 0xC2, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xBBBB_BBBB_BBBB_BBBB, 0xAAAA_AAAA_AAAA_AAAA]);
}

//  MOVDDUP

#[test]
fn test_movddup_reg() {
    // movddup xmm0, xmm1; ret  ->  F2 0F 12 C1  C3
    let bytes = [0xF2, 0x0F, 0x12, 0xC1, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        ctx.xmm[1] = [0xDEAD_BEEF_CAFE_BABE, 0x9999_9999_9999_9999];
        let ctx = run(&bytes, ctx);
        assert_eq!(
            ctx.xmm[0],
            [0xDEAD_BEEF_CAFE_BABE, 0xDEAD_BEEF_CAFE_BABE],
            "movddup must broadcast the source's low quadword to both halves"
        );
    }
}

#[test]
fn test_movddup_mem() {
    // movddup xmm0, [rdi]; ret  ->  F2 0F 12 07  C3
    let bytes = [0xF2, 0x0F, 0x12, 0x07, 0xC3];
    let buf = [0x0102_0304_0506_0708u64, 0x0BAD_0BAD_0BAD_0BAD];
    let mut ctx = RegisterContext::default();
    ctx.rdi = buf.as_ptr() as u64;
    ctx.xmm[0] = [0, 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x0102_0304_0506_0708, 0x0102_0304_0506_0708],
        "movddup m64 must load 8 bytes and broadcast them"
    );
}

//  PSRLQ

#[test]
fn test_psrlq_imm() {
    // psrlq xmm0, 4; ret  ->  66 0F 73 D0 04  C3
    let bytes = [0x66, 0x0F, 0x73, 0xD0, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0000_0000_0000_FF00, 0xF000_0000_0000_0000];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x0000_0000_0000_0FF0, 0x0F00_0000_0000_0000],
        "psrlq imm must shift each 64-bit lane right"
    );
}

#[test]
fn test_psrlq_imm_ge_64_zeroes() {
    // psrlq xmm0, 64; ret  ->  66 0F 73 D0 40  C3
    let bytes = [0x66, 0x0F, 0x73, 0xD0, 0x40, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0, 0], "psrlq by >=64 must zero both lanes");
}

#[test]
fn test_psrlq_xmm_count() {
    // psrlq xmm0, xmm1; ret  ->  66 0F D3 C1  C3  (count = low 64 bits of xmm1)
    let bytes = [0x66, 0x0F, 0xD3, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0000_0000_0000_FF00, 0xF000_0000_0000_0000];
    ctx.xmm[1] = [8, 0]; // shift by 8
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x0000_0000_0000_00FF, 0x00F0_0000_0000_0000],
        "psrlq by register count must shift both lanes"
    );

    // A count >= 64 from the register form must also zero the lanes.
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0];
    ctx.xmm[1] = [100, 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0, 0],
        "psrlq by >=64 (register) must zero both lanes"
    );
}

//  Packed per-lane shifts: PSRLW/PSRLD/PSRLQ, PSLLW/PSLLD/PSLLQ, PSRAW/PSRAD.
//
//  Every case is checked in both XMM representations (scalar lo/hi pair with
//  lane masking vs native vector shifts) via `check_packed`, which puts the
//  shifted value in xmm0 and the register-form count in xmm1.

/// Word-lane test pattern: lanes cover 0x0001, all-ones, sign-bit-set and a
/// mixed value, so lane-boundary bleed and sign fill are both visible.
const SHIFT_W: [u64; 2] = [0x000F_FF00_0F00_FFFF, 0x8000_1234_FFFF_0001];
/// Dword-lane test pattern.
const SHIFT_D: [u64; 2] = [0x0000_000F_FF00_0000, 0xF000_0001_8000_0000];
/// Qword-lane test pattern.
const SHIFT_Q: [u64; 2] = [0x0000_0000_0000_FF00, 0xF000_0000_0000_0000];

//  PSRLW

#[test]
fn test_psrlw_imm() {
    // psrlw xmm0, 4; ret  ->  66 0F 71 D0 04  C3
    let bytes = [0x66, 0x0F, 0x71, 0xD0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [0, 0],
        [0x0000_0FF0_00F0_0FFF, 0x0800_0123_0FFF_0000],
        "psrlw imm",
    );
}

#[test]
fn test_psrlw_imm_ge_16_zeroes() {
    // psrlw xmm0, 16; ret  ->  66 0F 71 D0 10  C3
    let bytes = [0x66, 0x0F, 0x71, 0xD0, 0x10, 0xC3];
    check_packed(&bytes, SHIFT_W, [0, 0], [0, 0], "psrlw by >=16");
}

#[test]
fn test_psrlw_xmm_count() {
    // psrlw xmm0, xmm1; ret  ->  66 0F D1 C1  C3
    let bytes = [0x66, 0x0F, 0xD1, 0xC1, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [8, 0],
        [0x0000_00FF_000F_00FF, 0x0080_0012_00FF_0000],
        "psrlw xmm count",
    );
    check_packed(
        &bytes,
        SHIFT_W,
        [100, 0],
        [0, 0],
        "psrlw by >=16 (register)",
    );
}

#[test]
fn test_psrlw_mem_count() {
    // psrlw xmm0, [rdi]; ret  ->  66 0F D1 07  C3
    let bytes = [0x66, 0x0F, 0xD1, 0x07, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let count = [8u64, 0];
        let mut ctx = RegisterContext::default();
        ctx.rdi = count.as_ptr() as u64;
        ctx.xmm[0] = SHIFT_W;
        let ctx = run(&bytes, ctx);
        assert_eq!(
            ctx.xmm[0],
            [0x0000_00FF_000F_00FF, 0x0080_0012_00FF_0000],
            "psrlw must take the count from the low 64 bits of m128"
        );
    }
}

//  PSRLD

#[test]
fn test_psrld_imm() {
    // psrld xmm0, 4; ret  ->  66 0F 72 D0 04  C3
    let bytes = [0x66, 0x0F, 0x72, 0xD0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [0, 0],
        [0x0000_0000_0FF0_0000, 0x0F00_0000_0800_0000],
        "psrld imm",
    );
}

#[test]
fn test_psrld_imm_ge_32_zeroes() {
    // psrld xmm0, 32; ret  ->  66 0F 72 D0 20  C3
    let bytes = [0x66, 0x0F, 0x72, 0xD0, 0x20, 0xC3];
    check_packed(
        &bytes,
        [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0],
        [0, 0],
        [0, 0],
        "psrld by >=32",
    );
}

#[test]
fn test_psrld_xmm_count() {
    // psrld xmm0, xmm1; ret  ->  66 0F D2 C1  C3
    let bytes = [0x66, 0x0F, 0xD2, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_00FF_FF00_0000, 0xF000_0001_0000_00FF],
        [8, 0],
        [0x0000_0000_00FF_0000, 0x00F0_0000_0000_0000],
        "psrld xmm count",
    );
    check_packed(
        &bytes,
        [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0],
        [100, 0],
        [0, 0],
        "psrld by >=32 (register)",
    );
}

#[test]
fn test_psrld_zero_count_is_identity() {
    // psrld xmm0, 0; ret  ->  66 0F 72 D0 00  C3
    let bytes = [0x66, 0x0F, 0x72, 0xD0, 0x00, 0xC3];
    check_packed(&bytes, SHIFT_D, [0, 0], SHIFT_D, "psrld by 0");
}

//  PSLLW / PSLLD / PSLLQ

#[test]
fn test_psllw_imm() {
    // psllw xmm0, 4; ret  ->  66 0F 71 F0 04  C3
    let bytes = [0x66, 0x0F, 0x71, 0xF0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [0, 0],
        [0x00F0_F000_F000_FFF0, 0x0000_2340_FFF0_0010],
        "psllw imm",
    );
}

#[test]
fn test_psllw_xmm_count() {
    // psllw xmm0, xmm1; ret  ->  66 0F F1 C1  C3
    let bytes = [0x66, 0x0F, 0xF1, 0xC1, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [8, 0],
        [0x0F00_0000_0000_FF00, 0x0000_3400_FF00_0100],
        "psllw xmm count",
    );
    check_packed(&bytes, SHIFT_W, [16, 0], [0, 0], "psllw by >=16 (register)");
}

#[test]
fn test_pslld_imm() {
    // pslld xmm0, 4; ret  ->  66 0F 72 F0 04  C3
    let bytes = [0x66, 0x0F, 0x72, 0xF0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [0, 0],
        [0x0000_00F0_F000_0000, 0x0000_0010_0000_0000],
        "pslld imm",
    );
}

#[test]
fn test_pslld_xmm_count() {
    // pslld xmm0, xmm1; ret  ->  66 0F F2 C1  C3
    let bytes = [0x66, 0x0F, 0xF2, 0xC1, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [4, 0],
        [0x0000_00F0_F000_0000, 0x0000_0010_0000_0000],
        "pslld xmm count",
    );
    check_packed(&bytes, SHIFT_D, [32, 0], [0, 0], "pslld by >=32 (register)");
}

#[test]
fn test_psllq_imm_and_reg() {
    // psllq xmm0, 4; ret  ->  66 0F 73 F0 04  C3
    let imm = [0x66, 0x0F, 0x73, 0xF0, 0x04, 0xC3];
    let expected = [0x0000_0000_000F_F000, 0x0000_0000_0000_0000];
    check_packed(&imm, SHIFT_Q, [0, 0], expected, "psllq imm");

    // psllq xmm0, xmm1; ret  ->  66 0F F3 C1  C3
    let reg = [0x66, 0x0F, 0xF3, 0xC1, 0xC3];
    check_packed(&reg, SHIFT_Q, [4, 0], expected, "psllq xmm count");
    check_packed(&reg, SHIFT_Q, [64, 0], [0, 0], "psllq by >=64 (register)");
}

//  PSRAW / PSRAD - an out-of-range count saturates to each lane's sign
//  instead of zeroing the lane.

#[test]
fn test_psraw_imm() {
    // psraw xmm0, 4; ret  ->  66 0F 71 E0 04  C3
    let bytes = [0x66, 0x0F, 0x71, 0xE0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [0, 0],
        [0x0000_FFF0_00F0_FFFF, 0xF800_0123_FFFF_0000],
        "psraw imm",
    );
}

#[test]
fn test_psraw_imm_ge_16_saturates_to_sign() {
    // psraw xmm0, 20; ret  ->  66 0F 71 E0 14  C3
    let bytes = [0x66, 0x0F, 0x71, 0xE0, 0x14, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [0, 0],
        [0x0000_FFFF_0000_FFFF, 0xFFFF_0000_FFFF_0000],
        "psraw by >=16",
    );
}

#[test]
fn test_psraw_xmm_count() {
    // psraw xmm0, xmm1; ret  ->  66 0F E1 C1  C3
    let bytes = [0x66, 0x0F, 0xE1, 0xC1, 0xC3];
    check_packed(
        &bytes,
        SHIFT_W,
        [8, 0],
        [0x0000_FFFF_000F_FFFF, 0xFF80_0012_FFFF_0000],
        "psraw xmm count",
    );
    check_packed(
        &bytes,
        SHIFT_W,
        [99, 0],
        [0x0000_FFFF_0000_FFFF, 0xFFFF_0000_FFFF_0000],
        "psraw by >=16 (register)",
    );
}

#[test]
fn test_psrad_imm() {
    // psrad xmm0, 4; ret  ->  66 0F 72 E0 04  C3
    let bytes = [0x66, 0x0F, 0x72, 0xE0, 0x04, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [0, 0],
        [0x0000_0000_FFF0_0000, 0xFF00_0000_F800_0000],
        "psrad imm",
    );
}

#[test]
fn test_psrad_imm_ge_32_saturates_to_sign() {
    // psrad xmm0, 32; ret  ->  66 0F 72 E0 20  C3
    let bytes = [0x66, 0x0F, 0x72, 0xE0, 0x20, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [0, 0],
        [0x0000_0000_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        "psrad by >=32",
    );
}

#[test]
fn test_psrad_xmm_count() {
    // psrad xmm0, xmm1; ret  ->  66 0F E2 C1  C3
    let bytes = [0x66, 0x0F, 0xE2, 0xC1, 0xC3];
    check_packed(
        &bytes,
        SHIFT_D,
        [8, 0],
        [0x0000_0000_FFFF_0000, 0xFFF0_0000_FF80_0000],
        "psrad xmm count",
    );
    check_packed(
        &bytes,
        SHIFT_D,
        [1 << 40, 0],
        [0x0000_0000_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        "psrad by a huge count (register)",
    );
}

//  PSHUFD

#[test]
fn test_pshufd_reverse() {
    // pshufd xmm0, xmm1, 0x1B; ret  ->  66 0F 70 C1 1B  C3
    // imm 0x1B = 00_01_10_11 -> dst dwords [d3, d2, d1, d0] (reverse).
    let bytes = [0x66, 0x0F, 0x70, 0xC1, 0x1B, 0xC3];
    let mut ctx = RegisterContext::default();
    // src dwords: d0=0, d1=0x11111111, d2=0x22222222, d3=0x33333333
    ctx.xmm[1] = [0x1111_1111_0000_0000, 0x3333_3333_2222_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0],
        [0x2222_2222_3333_3333, 0x0000_0000_1111_1111],
        "pshufd 0x1B must reverse the four dwords"
    );
}

#[test]
fn test_pshufd_identity() {
    // pshufd xmm0, xmm1, 0xE4 (11_10_01_00) is the identity permutation.
    let bytes = [0x66, 0x0F, 0x70, 0xC1, 0xE4, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x1111_1111_0000_0000, 0x3333_3333_2222_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1111_1111_0000_0000, 0x3333_3333_2222_2222]);
}

#[test]
fn test_pshufd_broadcast_dword2() {
    // pshufd xmm0, xmm1, 0xAA (10_10_10_10) broadcasts source dword 2.
    let bytes = [0x66, 0x0F, 0x70, 0xC1, 0xAA, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x1111_1111_0000_0000, 0x3333_3333_2222_2222]; // d2 = 0x22222222
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x2222_2222_2222_2222, 0x2222_2222_2222_2222]);
}

//  PUNPCKLDQ

#[test]
fn test_punpckldq() {
    // punpckldq xmm0, xmm1; ret  ->  66 0F 62 C1  C3
    // result dwords = [dst.d0, src.d0, dst.d1, src.d1]
    let bytes = [0x66, 0x0F, 0x62, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // dst dwords: d0=0xAAAA_AAAA, d1=0xBBBB_BBBB (d2/d3 ignored)
    ctx.xmm[0] = [0xBBBB_BBBB_AAAA_AAAA, 0x9999_9999_8888_8888];
    // src dwords: s0=0xCCCC_CCCC, s1=0xDDDD_DDDD
    ctx.xmm[1] = [0xDDDD_DDDD_CCCC_CCCC, 0x7777_7777_6666_6666];
    let ctx = run_test(&bytes, ctx);
    // lo = dst.d0 | src.d0<<32 ; hi = dst.d1 | src.d1<<32
    assert_eq!(ctx.xmm[0][0], 0xCCCC_CCCC_AAAA_AAAA);
    assert_eq!(ctx.xmm[0][1], 0xDDDD_DDDD_BBBB_BBBB);
}

//  UNPCKHPD

#[test]
fn test_unpckhpd() {
    // unpckhpd xmm0, xmm1; ret  ->  66 0F 15 C1  C3
    // result: dst[63:0] = dst[127:64], dst[127:64] = src[127:64]
    let bytes = [0x66, 0x0F, 0x15, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.xmm[1] = [0xCCCC_CCCC_CCCC_CCCC, 0xDDDD_DDDD_DDDD_DDDD];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xBBBB_BBBB_BBBB_BBBB, "lo = old dst hi");
    assert_eq!(ctx.xmm[0][1], 0xDDDD_DDDD_DDDD_DDDD, "hi = src hi");
}

#[test]
fn test_unpckhpd_mem() {
    // unpckhpd xmm0, [rax]; ret  ->  66 0F 15 00  C3
    let bytes = [0x66, 0x0F, 0x15, 0x00, 0xC3];
    let mem: [u64; 2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_ptr() as u64;
    ctx.xmm[0] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x4444_4444_4444_4444, "lo = old dst hi");
    assert_eq!(ctx.xmm[0][1], 0x2222_2222_2222_2222, "hi = src[127:64]");
}

//  SUBPD (packed double subtract)

#[test]
fn test_subpd() {
    // subpd xmm0, xmm1; ret  ->  66 0F 5C C1  C3
    let bytes = [0x66, 0x0F, 0x5C, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [10.0f64.to_bits(), 3.5f64.to_bits()];
    ctx.xmm[1] = [4.0f64.to_bits(), 1.25f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 6.0, "lane 0: 10 - 4");
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 2.25, "lane 1: 3.5 - 1.25");
}

//  ADDSD (scalar double add)

#[test]
fn test_addsd_reg() {
    // addsd xmm0, xmm1; ret  ->  F2 0F 58 C1  C3
    let bytes = [0xF2, 0x0F, 0x58, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [2.5f64.to_bits(), 0x1111_2222_3333_4444];
    ctx.xmm[1] = [7.0f64.to_bits(), 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 9.5, "low lane: 2.5 + 7.0");
    assert_eq!(
        ctx.xmm[0][1], 0x1111_2222_3333_4444,
        "high lane must be preserved by addsd"
    );
}

#[test]
fn test_addsd_mem() {
    // addsd xmm0, [rax]; ret  ->  F2 0F 58 00  C3  (m64 source)
    let bytes = [0xF2, 0x0F, 0x58, 0x00, 0xC3];
    let src: f64 = 0.5;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.xmm[0] = [1.0f64.to_bits(), 0xDEAD_BEEF_CAFE_BABE];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1.5);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE);
}

//  MULSD / DIVSD (scalar double multiply / divide)

#[test]
fn test_mulsd_reg() {
    // mulsd xmm0, xmm1; ret  ->  F2 0F 59 C1  C3
    let bytes = [0xF2, 0x0F, 0x59, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [6.0f64.to_bits(), 0x1111_2222_3333_4444];
    ctx.xmm[1] = [7.0f64.to_bits(), 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 42.0, "low lane: 6 * 7");
    assert_eq!(
        ctx.xmm[0][1], 0x1111_2222_3333_4444,
        "high lane must be preserved by mulsd"
    );
}

#[test]
fn test_divsd_reg() {
    // divsd xmm0, xmm1; ret  ->  F2 0F 5E C1  C3
    let bytes = [0xF2, 0x0F, 0x5E, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [10.0f64.to_bits(), 0xAAAA_BBBB_CCCC_DDDD];
    ctx.xmm[1] = [4.0f64.to_bits(), 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 2.5, "low lane: 10 / 4");
    assert_eq!(
        ctx.xmm[0][1], 0xAAAA_BBBB_CCCC_DDDD,
        "high lane must be preserved by divsd"
    );
}

#[test]
fn test_divsd_mem() {
    // divsd xmm0, [rax]; ret  ->  F2 0F 5E 00  C3  (m64 source)
    let bytes = [0xF2, 0x0F, 0x5E, 0x00, 0xC3];
    let src: f64 = 2.0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &src as *const f64 as u64;
    ctx.xmm[0] = [9.0f64.to_bits(), 0xDEAD_BEEF_CAFE_BABE];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 4.5);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE);
}

//  UCOMISD (scalar double compare -> EFLAGS)

#[test]
fn test_ucomisd_less() {
    // ucomisd xmm0, xmm1; ret  ->  66 0F 2E C1  C3
    let bytes = [0x66, 0x0F, 0x2E, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [1.0f64.to_bits(), 0];
    ctx.xmm[1] = [2.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 1, "1 < 2 -> CF set");
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.pf, 0, "ordered -> PF clear");
}

#[test]
fn test_ucomisd_equal() {
    let bytes = [0x66, 0x0F, 0x2E, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [5.0f64.to_bits(), 0];
    ctx.xmm[1] = [5.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.zf, 1, "equal -> ZF set");
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.pf, 0);
}

#[test]
fn test_ucomisd_greater() {
    let bytes = [0x66, 0x0F, 0x2E, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [9.0f64.to_bits(), 0];
    ctx.xmm[1] = [4.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 0);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.pf, 0);
}

#[test]
fn test_ucomisd_unordered() {
    // NaN in xmm0 -> unordered: ZF=PF=CF=1
    let bytes = [0x66, 0x0F, 0x2E, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [f64::NAN.to_bits(), 0];
    ctx.xmm[1] = [1.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 1);
    assert_eq!(ctx.pf, 1, "unordered -> PF set");
}

//  MOVAPD (already dispatched via handle_movaps)

#[test]
fn test_movapd_reg_reg() {
    // movapd xmm0, xmm1; ret  ->  66 0F 28 C1  C3
    let bytes = [0x66, 0x0F, 0x28, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210]);
}

#[test]
fn test_vec_pshufd_and_psrlq() {
    // Same ops as the scalar tests, but in vector-xmm mode, to exercise the
    // lo/hi accessors' insertlane/extractlane paths.
    // pshufd xmm0, xmm1, 0x1B; ret
    let bytes = [0x66, 0x0F, 0x70, 0xC1, 0x1B, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x1111_1111_0000_0000, 0x3333_3333_2222_2222];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x2222_2222_3333_3333, 0x0000_0000_1111_1111]);

    // psrlq xmm0, 4; ret
    let bytes = [0x66, 0x0F, 0x73, 0xD0, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0000_0000_0000_FF00, 0xF000_0000_0000_0000];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x0000_0000_0000_0FF0, 0x0F00_0000_0000_0000]);
}

//  Packed integer add / subtract
//
//  PADDB/W/D/Q, PSUBB/W/D/Q and the saturating PADDS*/PADDUS*/PSUBS*/PSUBUS*
//  forms. Every case is checked in both XMM representations: the scalar lo/hi
//  pair (SWAR / per-lane expansion) and native vectors.

/// Set `xmm0 = dst`, `xmm1 = src`, run `bytes` in both scalar and vector XMM
/// mode, and assert `xmm0 == expected` either way.
fn check_packed(bytes: &[u8], dst: [u64; 2], src: [u64; 2], expected: [u64; 2], what: &str) {
    for (mode, run) in [
        (
            "scalar",
            run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        ),
        ("vector", run_test_vec),
    ] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = dst;
        ctx.xmm[1] = src;
        let ctx = run(bytes, ctx);
        assert_eq!(
            ctx.xmm[0], expected,
            "{what} ({mode} xmm): dst={dst:#018x?} src={src:#018x?}"
        );
        assert_eq!(
            ctx.xmm[1], src,
            "{what} ({mode} xmm) must not modify the source"
        );
    }
}

#[test]
fn test_paddd() {
    // paddd xmm0, xmm1; ret  ->  66 0F FE C1  C3
    let bytes = [0x66, 0x0F, 0xFE, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_0002_FFFF_FFFF, 0x8000_0000_7FFF_FFFF],
        [0x0000_0003_0000_0001, 0x8000_0000_0000_0001],
        // d0 wraps to 0 without carrying into d1; d3 (0x80000000 * 2) wraps too.
        [0x0000_0005_0000_0000, 0x0000_0000_8000_0000],
        "paddd",
    );
}

#[test]
fn test_paddd_other_registers() {
    // paddd xmm2, xmm3; ret  ->  66 0F FE D3  C3
    let bytes = [0x66, 0x0F, 0xFE, 0xD3, 0xC3];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[2] = [0x0000_0001_0000_0002, 0x0000_0003_0000_0004];
        ctx.xmm[3] = [0x0000_0010_0000_0020, 0x0000_0030_0000_0040];
        let ctx = run(&bytes, ctx);
        assert_eq!(ctx.xmm[2], [0x0000_0011_0000_0022, 0x0000_0033_0000_0044]);
        assert_eq!(ctx.xmm[0], [0, 0], "paddd xmm2,xmm3 must not touch xmm0");
    }
}

#[test]
fn test_paddd_mem() {
    // paddd xmm0, [rax]; ret  ->  66 0F FE 00  C3
    let bytes = [0x66, 0x0F, 0xFE, 0x00, 0xC3];
    let mem: [u64; 2] = [0x0000_0003_0000_0001, 0x0000_0000_FFFF_FFFF];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.rax = mem.as_ptr() as u64;
        ctx.xmm[0] = [0x0000_0002_FFFF_FFFF, 0x0000_0001_0000_0002];
        let ctx = run(&bytes, ctx);
        assert_eq!(ctx.xmm[0], [0x0000_0005_0000_0000, 0x0000_0001_0000_0001]);
    }
}

#[test]
fn test_paddb() {
    // paddb xmm0, xmm1; ret  ->  66 0F FC C1  C3
    let bytes = [0x66, 0x0F, 0xFC, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0102_0304_0506_07FF, 0x7F7F_7F7F_8080_8080],
        [0x0101_0101_0101_0101, 0x0101_0101_0101_0101],
        // The low byte wraps 0xFF -> 0x00 without carrying into the next lane.
        [0x0203_0405_0607_0800, 0x8080_8080_8181_8181],
        "paddb",
    );
}

#[test]
fn test_paddw() {
    // paddw xmm0, xmm1; ret  ->  66 0F FD C1  C3
    let bytes = [0x66, 0x0F, 0xFD, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0xFFFF_0001_1234_FFFF, 0x0000_FFFF_8000_7FFF],
        [0x0001_0001_0001_0001, 0x0001_0001_8000_0001],
        [0x0000_0002_1235_0000, 0x0001_0000_0000_8000],
        "paddw",
    );
}

#[test]
fn test_paddq() {
    // paddq xmm0, xmm1; ret  ->  66 0F D4 C1  C3
    let bytes = [0x66, 0x0F, 0xD4, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0005],
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0007],
        // The low quadword wraps to 0; no carry crosses into the high lane.
        [0x0000_0000_0000_0000, 0x0000_0000_0000_000C],
        "paddq",
    );
}

#[test]
fn test_psubd() {
    // psubd xmm0, xmm1; ret  ->  66 0F FA C1  C3
    let bytes = [0x66, 0x0F, 0xFA, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_0005_0000_0000, 0x0000_0000_8000_0000],
        [0x0000_0002_0000_0001, 0x0000_0001_8000_0000],
        // d0's borrow must not reach d1; likewise d2 -> d3.
        [0x0000_0003_FFFF_FFFF, 0xFFFF_FFFF_0000_0000],
        "psubd",
    );
}

#[test]
fn test_psubb() {
    // psubb xmm0, xmm1; ret  ->  66 0F F8 C1  C3
    let bytes = [0x66, 0x0F, 0xF8, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x1020_3040_5060_7000, 0x0000_0000_0000_0000],
        [0x0101_0101_0101_0101, 0x0101_0101_0101_0101],
        [0x0F1F_2F3F_4F5F_6FFF, 0xFFFF_FFFF_FFFF_FFFF],
        "psubb",
    );
}

#[test]
fn test_psubw() {
    // psubw xmm0, xmm1; ret  ->  66 0F F9 C1  C3
    let bytes = [0x66, 0x0F, 0xF9, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_0000_0001_0000, 0x8000_1234_0000_ABCD],
        [0x0001_0001_0001_0001, 0x0001_0234_0001_00CD],
        [0xFFFF_FFFF_0000_FFFF, 0x7FFF_1000_FFFF_AB00],
        "psubw",
    );
}

#[test]
fn test_psubw_mem() {
    // psubw xmm0, [rax]; ret  ->  66 0F F9 00  C3
    let bytes = [0x66, 0x0F, 0xF9, 0x00, 0xC3];
    let mem: [u64; 2] = [0x0001_0001_0001_0001, 0x0001_0001_0001_0001];
    for run in [
        run_test as fn(&[u8], RegisterContext) -> RegisterContext,
        run_test_vec,
    ] {
        let mut ctx = RegisterContext::default();
        ctx.rax = mem.as_ptr() as u64;
        ctx.xmm[0] = [0x0000_0000_0001_0000, 0x0002_0003_0004_0005];
        let ctx = run(&bytes, ctx);
        assert_eq!(ctx.xmm[0], [0xFFFF_FFFF_0000_FFFF, 0x0001_0002_0003_0004]);
    }
}

#[test]
fn test_psubq() {
    // psubq xmm0, xmm1; ret  ->  66 0F FB C1  C3
    let bytes = [0x66, 0x0F, 0xFB, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_0000_0000_0000, 0x0000_0000_0000_000A],
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0003],
        [0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0007],
        "psubq",
    );
}

#[test]
fn test_paddsb_saturates_signed() {
    // paddsb xmm0, xmm1; ret  ->  66 0F EC C1  C3
    let bytes = [0x66, 0x0F, 0xEC, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x7080_01FF_7F80_0000, 0x7F7F_7F7F_7F7F_7F7F],
        [0x7080_0201_01FF_0000, 0x0101_0101_0101_0101],
        // byte7: 0x70+0x70 -> +0x7F   byte6: 0x80+0x80 -> -0x80
        // byte5: 0x01+0x02 = 0x03     byte4: 0xFF+0x01 = 0x00
        // byte3: 0x7F+0x01 -> +0x7F   byte2: 0x80+0xFF -> -0x80
        [0x7F80_0300_7F80_0000, 0x7F7F_7F7F_7F7F_7F7F],
        "paddsb",
    );
}

#[test]
fn test_paddsw_saturates_signed() {
    // paddsw xmm0, xmm1; ret  ->  66 0F ED C1  C3
    let bytes = [0x66, 0x0F, 0xED, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x7FFF_8000_0100_FFFF, 0x8000_8000_8000_8000],
        [0x0001_8000_0200_0001, 0xFFFF_FFFF_FFFF_FFFF],
        [0x7FFF_8000_0300_0000, 0x8000_8000_8000_8000],
        "paddsw",
    );
}

#[test]
fn test_paddusb_saturates_unsigned() {
    // paddusb xmm0, xmm1; ret  ->  66 0F DC C1  C3
    let bytes = [0x66, 0x0F, 0xDC, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0xFF80_1000_FF01_0203, 0x0102_0304_0506_0708],
        [0x0180_2000_0002_0304, 0x0101_0101_0101_0101],
        // byte7: 0xFF+0x01 -> 0xFF    byte6: 0x80+0x80 -> 0xFF
        [0xFFFF_3000_FF03_0507, 0x0203_0405_0607_0809],
        "paddusb",
    );
}

#[test]
fn test_paddusw_saturates_unsigned() {
    // paddusw xmm0, xmm1; ret  ->  66 0F DD C1  C3
    let bytes = [0x66, 0x0F, 0xDD, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0xFFFF_8000_1234_0001, 0x0000_1111_2222_3333],
        [0x0001_8000_1000_0002, 0x0001_1111_1111_1111],
        [0xFFFF_FFFF_2234_0003, 0x0001_2222_3333_4444],
        "paddusw",
    );
}

#[test]
fn test_psubsb_saturates_signed() {
    // psubsb xmm0, xmm1; ret  ->  66 0F E8 C1  C3
    let bytes = [0x66, 0x0F, 0xE8, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x807F_0010_0000_0000, 0x7F7F_7F7F_7F7F_7F7F],
        [0x0180_0102_0000_0000, 0x8080_8080_8080_8080],
        // byte7: -0x80-0x01 -> -0x80   byte6: +0x7F-(-0x80) -> +0x7F
        // byte5: 0x00-0x01 = 0xFF      byte4: 0x10-0x02 = 0x0E
        [0x807F_FF0E_0000_0000, 0x7F7F_7F7F_7F7F_7F7F],
        "psubsb",
    );
}

#[test]
fn test_psubsw_saturates_signed() {
    // psubsw xmm0, xmm1; ret  ->  66 0F E9 C1  C3
    let bytes = [0x66, 0x0F, 0xE9, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x8000_7FFF_0005_0000, 0x0000_0000_0000_0000],
        [0x0001_8000_0003_0001, 0x0001_0001_0001_0001],
        [0x8000_7FFF_0002_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        "psubsw",
    );
}

#[test]
fn test_psubusb_saturates_at_zero() {
    // psubusb xmm0, xmm1; ret  ->  66 0F D8 C1  C3
    let bytes = [0x66, 0x0F, 0xD8, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0010_FF80_0102_0304, 0x0203_0405_0607_0809],
        [0x0102_FF81_0201_0004, 0x0101_0101_0101_0101],
        // Every underflowing lane clamps to 0x00 rather than wrapping.
        [0x000E_0000_0001_0300, 0x0102_0304_0506_0708],
        "psubusb",
    );
}

#[test]
fn test_psubusw_saturates_at_zero() {
    // psubusw xmm0, xmm1; ret  ->  66 0F D9 C1  C3
    let bytes = [0x66, 0x0F, 0xD9, 0xC1, 0xC3];
    check_packed(
        &bytes,
        [0x0000_1234_FFFF_0005, 0x1000_2000_3000_4000],
        [0x0001_0234_FFFF_0006, 0x0001_0001_0001_0001],
        [0x0000_1000_0000_0000, 0x0FFF_1FFF_2FFF_3FFF],
        "psubusw",
    );
}

#[test]
fn test_packed_add_sub_roundtrip() {
    // paddd xmm0, xmm1; psubd xmm0, xmm1; ret  -> the pair must be an identity
    // even when individual lanes wrap.
    let bytes = [0x66, 0x0F, 0xFE, 0xC1, 0x66, 0x0F, 0xFA, 0xC1, 0xC3];
    let start = [0xDEAD_BEEF_CAFE_BABE, 0x0000_0001_FFFF_FFFF];
    check_packed(
        &bytes,
        start,
        [0x1234_5678_9ABC_DEF0, 0xFFFF_FFFF_0000_0001],
        start,
        "paddd+psubd roundtrip",
    );
}

//  CVTSI2SS

/// Raw bits of `v as f32`, as they should appear in `xmm[dst][0]`'s low dword.
fn f32_bits(v: f32) -> u64 {
    v.to_bits() as u64
}

#[test]
fn test_cvtsi2ss_r32() {
    // cvtsi2ss xmm0, eax; ret   ->   F3 0F 2A C0  C3
    let bytes = [0xF3, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 42;
    ctx.xmm[0] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.xmm[0][0],
        0xAAAA_BBBB_0000_0000 | f32_bits(42.0),
        "cvtsi2ss r32: low dword converted, bits[63:32] preserved"
    );
    assert_eq!(
        ctx.xmm[0][1], 0x1111_2222_3333_4444,
        "cvtsi2ss must preserve bits[127:64]"
    );
}

#[test]
fn test_cvtsi2ss_r32_negative() {
    // The source is *signed*: 0xFFFFFFFF as i32 is -1, not 4294967295.
    // cvtsi2ss xmm0, eax; ret
    let bytes = [0xF3, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    // Upper 32 bits of rax must be ignored entirely.
    ctx.rax = 0xDEAD_BEEF_FFFF_FFFF;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0] as u32 as u64, f32_bits(-1.0));
}

#[test]
fn test_cvtsi2ss_r64() {
    // cvtsi2ss xmm0, rax; ret   ->   F3 48 0F 2A C0  C3
    let bytes = [0xF3, 0x48, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = (-123_456_789_012i64) as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], f32_bits(-123_456_789_012i64 as f32));
}

#[test]
fn test_cvtsi2ss_r64_rounds_to_nearest_even() {
    // 2^24 + 1 is not representable in f32; the default rounding mode rounds it
    // back down to 2^24.
    // cvtsi2ss xmm0, rax; ret
    let bytes = [0xF3, 0x48, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = (1u64 << 24) + 1;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], f32_bits(16_777_216.0));
}

#[test]
fn test_cvtsi2ss_m32() {
    // cvtsi2ss xmm0, dword ptr [rdi]; ret   ->   F3 0F 2A 07  C3
    let bytes = [0xF3, 0x0F, 0x2A, 0x07, 0xC3];
    let val: i32 = -7;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &val as *const i32 as u64;
    ctx.xmm[0] = [0x0123_4567_89AB_CDEF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x0123_4567_0000_0000 | f32_bits(-7.0));
    assert_eq!(ctx.xmm[0][1], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_cvtsi2ss_m64() {
    // cvtsi2ss xmm0, qword ptr [rdi]; ret   ->   F3 48 0F 2A 07  C3
    let bytes = [0xF3, 0x48, 0x0F, 0x2A, 0x07, 0xC3];
    let val: i64 = 1 << 40;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &val as *const i64 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], f32_bits((1i64 << 40) as f32));
}

#[test]
fn test_cvtsi2ss_vector_xmm() {
    // Same as `test_cvtsi2ss_r32` but with the native-vector XMM backing.
    // cvtsi2ss xmm3, ecx; ret   ->   F3 0F 2A D9  C3
    let bytes = [0xF3, 0x0F, 0x2A, 0xD9, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rcx = 42;
    ctx.xmm[3] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[3][0], 0xAAAA_BBBB_0000_0000 | f32_bits(42.0));
    assert_eq!(ctx.xmm[3][1], 0x1111_2222_3333_4444);
}

//  Scalar single-precision arithmetic: ADDSS / SUBSS / MULSS / DIVSS

/// Pack two `f32`s into one 64-bit XMM half (lane 0 in the low dword).
fn pack_f32(l0: f32, l1: f32) -> u64 {
    (l0.to_bits() as u64) | ((l1.to_bits() as u64) << 32)
}

/// Run a scalar-single op on `xmm0 <- xmm1` and return the resulting `xmm[0]`.
///
/// `xmm0` and `xmm1` are seeded so every bit outside lane 0 is distinctive,
/// letting the caller assert bits[127:32] of the destination survive.
fn run_scalar_single(bytes: &[u8], a: f32, b: f32) -> [u64; 2] {
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [
        pack_f32(a, f32::from_bits(0xAAAA_AAAA)),
        0x1111_2222_3333_4444,
    ];
    ctx.xmm[1] = [
        pack_f32(b, f32::from_bits(0xBBBB_BBBB)),
        0x9999_9999_9999_9999,
    ];
    run_test(bytes, ctx).xmm[0]
}

/// Assert a scalar-single op produced `expected` in lane 0 and touched nothing
/// else in the destination.
fn check_scalar_single(got: [u64; 2], expected: f32, what: &str) {
    assert_eq!(
        got[0],
        pack_f32(expected, f32::from_bits(0xAAAA_AAAA)),
        "{what}: lane 0 result, bits[63:32] preserved"
    );
    assert_eq!(
        got[1], 0x1111_2222_3333_4444,
        "{what}: bits[127:64] preserved"
    );
}

#[test]
fn test_addss_reg_reg() {
    // addss xmm0, xmm1; ret   ->   F3 0F 58 C1  C3
    let got = run_scalar_single(&[0xF3, 0x0F, 0x58, 0xC1, 0xC3], 1.5, 2.25);
    check_scalar_single(got, 3.75, "addss");
}

#[test]
fn test_subss_reg_reg_is_dst_minus_src() {
    // subss xmm0, xmm1; ret   ->   F3 0F 5C C1  C3
    // Asymmetric operands pin down the operand order.
    let got = run_scalar_single(&[0xF3, 0x0F, 0x5C, 0xC1, 0xC3], 1.5, 10.0);
    check_scalar_single(got, -8.5, "subss");
}

#[test]
fn test_mulss_reg_reg() {
    // mulss xmm0, xmm1; ret   ->   F3 0F 59 C1  C3
    let got = run_scalar_single(&[0xF3, 0x0F, 0x59, 0xC1, 0xC3], 1.5, 3.0);
    check_scalar_single(got, 4.5, "mulss");
}

#[test]
fn test_divss_reg_reg_is_dst_over_src() {
    // divss xmm0, xmm1; ret   ->   F3 0F 5E C1  C3
    let got = run_scalar_single(&[0xF3, 0x0F, 0x5E, 0xC1, 0xC3], 7.0, 2.0);
    check_scalar_single(got, 3.5, "divss");
}

#[test]
fn test_divss_by_zero_is_infinity() {
    // IEEE division by zero yields an infinity, not a trap.
    // divss xmm0, xmm1; ret
    let got = run_scalar_single(&[0xF3, 0x0F, 0x5E, 0xC1, 0xC3], 1.0, 0.0);
    check_scalar_single(got, f32::INFINITY, "divss by zero");
}

#[test]
fn test_mulss_inexact_operands_round_like_f32() {
    // 0.1 is not representable in binary; the product must match what native
    // f32 multiplication produces, bit for bit.
    // mulss xmm0, xmm1; ret
    let got = run_scalar_single(&[0xF3, 0x0F, 0x59, 0xC1, 0xC3], 0.1, 3.0);
    check_scalar_single(got, 0.1f32 * 3.0f32, "mulss inexact operands");
}

#[test]
fn test_addss_rounds_at_single_precision() {
    // 2^24 + 1 has no f32 representation: the add must saturate the 24-bit
    // significand back to 2^24 rather than keeping the extra bit in a wider
    // intermediate lane.
    // addss xmm0, xmm1; ret
    let got = run_scalar_single(&[0xF3, 0x0F, 0x58, 0xC1, 0xC3], 16_777_216.0, 1.0);
    check_scalar_single(got, 16_777_216.0, "addss single-precision rounding");
}

#[test]
fn test_mulss_load_from_mem() {
    // mulss xmm0, dword ptr [rdi]; ret   ->   F3 0F 59 07  C3
    let bytes = [0xF3, 0x0F, 0x59, 0x07, 0xC3];
    let val: f32 = 3.0;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &val as *const f32 as u64;
    ctx.xmm[0] = [
        pack_f32(1.5, f32::from_bits(0xAAAA_AAAA)),
        0x1111_2222_3333_4444,
    ];
    let got = run_test(&bytes, ctx).xmm[0];
    check_scalar_single(got, 4.5, "mulss m32");
}

#[test]
fn test_addss_vector_xmm() {
    // Same op under the native-vector XMM backing, on a non-zero register pair.
    // addss xmm2, xmm1; ret   ->   F3 0F 58 D1  C3
    let bytes = [0xF3, 0x0F, 0x58, 0xD1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[2] = [
        pack_f32(1.5, f32::from_bits(0xAAAA_AAAA)),
        0x1111_2222_3333_4444,
    ];
    ctx.xmm[1] = [pack_f32(2.25, 0.0), 0x9999_9999_9999_9999];
    let got = run_test_vec(&bytes, ctx).xmm[2];
    check_scalar_single(got, 3.75, "addss (vector xmm)");
}

//  Packed arithmetic: MULPS / ADDPD

/// 16-byte-aligned backing store for an `m128` memory operand.
///
/// The payload is only ever read by the lifted code through its address, which
/// Rust cannot see - hence the `dead_code` allowance.
#[repr(align(16))]
struct Aligned128(#[allow(dead_code)] [u64; 2]);

#[test]
fn test_mulps_reg_reg_all_four_lanes() {
    // mulps xmm0, xmm1; ret   ->   0F 59 C1  C3
    let bytes = [0x0F, 0x59, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(1.5, 2.0), pack_f32(-3.0, 0.5)];
    ctx.xmm[1] = [pack_f32(2.0, 4.0), pack_f32(3.0, 8.0)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(3.0, 8.0), "mulps lanes 0,1");
    assert_eq!(ctx.xmm[0][1], pack_f32(-9.0, 4.0), "mulps lanes 2,3");
}

#[test]
fn test_mulps_load_from_mem() {
    // mulps xmm0, xmmword ptr [rdi]; ret   ->   0F 59 07  C3
    let bytes = [0x0F, 0x59, 0x07, 0xC3];
    let src = Aligned128([pack_f32(2.0, 4.0), pack_f32(3.0, 8.0)]);
    let mut ctx = RegisterContext::default();
    ctx.rdi = &src as *const Aligned128 as u64;
    ctx.xmm[0] = [pack_f32(1.5, 2.0), pack_f32(-3.0, 0.5)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(3.0, 8.0), "mulps m128 lanes 0,1");
    assert_eq!(ctx.xmm[0][1], pack_f32(-9.0, 4.0), "mulps m128 lanes 2,3");
}

#[test]
fn test_mulps_vector_xmm() {
    // mulps xmm0, xmm1; ret, under the native-vector XMM backing.
    let bytes = [0x0F, 0x59, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(1.5, 2.0), pack_f32(-3.0, 0.5)];
    ctx.xmm[1] = [pack_f32(2.0, 4.0), pack_f32(3.0, 8.0)];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(3.0, 8.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(-9.0, 4.0));
}

#[test]
fn test_addpd_reg_reg_both_lanes() {
    // addpd xmm0, xmm1; ret   ->   66 0F 58 C1  C3
    let bytes = [0x66, 0x0F, 0x58, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [1.5f64.to_bits(), (-3.0f64).to_bits()];
    ctx.xmm[1] = [2.25f64.to_bits(), 0.5f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 3.75f64.to_bits(), "addpd low lane");
    assert_eq!(ctx.xmm[0][1], (-2.5f64).to_bits(), "addpd high lane");
}

#[test]
fn test_addpd_load_from_mem() {
    // addpd xmm0, xmmword ptr [rdi]; ret   ->   66 0F 58 07  C3
    let bytes = [0x66, 0x0F, 0x58, 0x07, 0xC3];
    let src = Aligned128([2.25f64.to_bits(), 0.5f64.to_bits()]);
    let mut ctx = RegisterContext::default();
    ctx.rdi = &src as *const Aligned128 as u64;
    ctx.xmm[0] = [1.5f64.to_bits(), (-3.0f64).to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 3.75f64.to_bits());
    assert_eq!(ctx.xmm[0][1], (-2.5f64).to_bits());
}

#[test]
fn test_addpd_vector_xmm() {
    // addpd xmm0, xmm1; ret, under the native-vector XMM backing.
    let bytes = [0x66, 0x0F, 0x58, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [1.5f64.to_bits(), (-3.0f64).to_bits()];
    ctx.xmm[1] = [2.25f64.to_bits(), 0.5f64.to_bits()];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 3.75f64.to_bits());
    assert_eq!(ctx.xmm[0][1], (-2.5f64).to_bits());
}

#[test]
fn test_addpd_subpd_roundtrip() {
    // addpd xmm0, xmm1; subpd xmm0, xmm1; ret - exact for these operands.
    let bytes = [0x66, 0x0F, 0x58, 0xC1, 0x66, 0x0F, 0x5C, 0xC1, 0xC3];
    let start = [1.5f64.to_bits(), (-3.0f64).to_bits()];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = start;
    ctx.xmm[1] = [2.25f64.to_bits(), 0.5f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], start, "addpd+subpd must be an identity here");
}

//  CVTTSS2SI

/// x86's "integer indefinite" results, produced whenever a truncated float is
/// unrepresentable in the destination.
const INDEF32: u64 = 0x8000_0000;
const INDEF64: u64 = 0x8000_0000_0000_0000;

/// Run a `cvttss2si <reg>, xmm0` form with `v` in lane 0 of xmm0, returning rax.
///
/// rax is pre-seeded with garbage so the 32-bit form's zero-extension of the
/// upper half is observable, and the unused parts of xmm0 are non-zero so a
/// handler reading the wrong lane would show up.
fn run_cvttss2si(bytes: &[u8], v: f32) -> u64 {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_DEAD_BEEF;
    ctx.xmm[0] = [
        pack_f32(v, f32::from_bits(0xAAAA_AAAA)),
        0x1111_2222_3333_4444,
    ];
    run_test(bytes, ctx).rax
}

/// `cvttss2si eax, xmm0 ; ret`
const CVTTSS2SI_EAX: [u8; 5] = [0xF3, 0x0F, 0x2C, 0xC0, 0xC3];
/// `cvttss2si rax, xmm0 ; ret`
const CVTTSS2SI_RAX: [u8; 6] = [0xF3, 0x48, 0x0F, 0x2C, 0xC0, 0xC3];

#[test]
fn test_cvttss2si_r32_truncates_toward_zero() {
    // Truncation, not rounding: both magnitudes drop their fraction.
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, 2.9), 2);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, 2.1), 2);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, -2.9), (-2i32) as u32 as u64);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, -0.5), 0);
}

#[test]
fn test_cvttss2si_r32_zero_extends_into_rax() {
    // Writing a 32-bit destination clears the upper 32 bits of the GPR.
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, 7.0), 7);
}

#[test]
fn test_cvttss2si_r32_range_boundaries() {
    // -2^31 is exactly representable and in range; +2^31 is one past the top.
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, -2147483648.0), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, 2147483648.0), INDEF32);
    // The largest f32 strictly below 2^31 still converts normally.
    assert_eq!(
        run_cvttss2si(&CVTTSS2SI_EAX, 2147483520.0),
        2147483520u32 as u64
    );
}

#[test]
fn test_cvttss2si_r32_invalid_gives_integer_indefinite() {
    // x86 returns INT_MIN for NaN and both infinities - it does *not* saturate
    // the way Rust's `as` cast or Cranelift's fcvt_to_sint_sat would.
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, f32::NAN), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, -f32::NAN), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, f32::INFINITY), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, f32::NEG_INFINITY), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, 1e10), INDEF32);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_EAX, -1e10), INDEF32);
}

#[test]
fn test_cvttss2si_r64() {
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, 2.9), 2);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, -2.9), (-2i64) as u64);
    // Out of range for 32 bits, comfortably in range for 64.
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, 1e10), 10_000_000_000);
    assert_eq!(
        run_cvttss2si(&CVTTSS2SI_RAX, -1e10),
        (-10_000_000_000i64) as u64
    );
    // 2^62 is exactly representable in f32.
    assert_eq!(
        run_cvttss2si(&CVTTSS2SI_RAX, 4611686018427387904.0),
        1u64 << 62
    );
}

#[test]
fn test_cvttss2si_r64_invalid_gives_integer_indefinite() {
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, f32::NAN), INDEF64);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, f32::INFINITY), INDEF64);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, f32::NEG_INFINITY), INDEF64);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, 1e30), INDEF64);
    assert_eq!(run_cvttss2si(&CVTTSS2SI_RAX, -1e30), INDEF64);
    // -2^63 is exactly representable but still one past the bottom of the range
    // once truncated, so it too yields the indefinite value.
    assert_eq!(
        run_cvttss2si(&CVTTSS2SI_RAX, -9223372036854775808.0),
        INDEF64
    );
}

#[test]
fn test_cvttss2si_m32() {
    // cvttss2si ecx, dword ptr [rdi]; ret   ->   F3 0F 2C 0F  C3
    let bytes = [0xF3, 0x0F, 0x2C, 0x0F, 0xC3];
    let val: f32 = -12.75;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &val as *const f32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rcx, (-12i32) as u32 as u64, "cvttss2si m32 truncates");
}

#[test]
fn test_cvttss2si_m32_r64() {
    // cvttss2si rcx, dword ptr [rdi]; ret   ->   F3 48 0F 2C 0F  C3
    let bytes = [0xF3, 0x48, 0x0F, 0x2C, 0x0F, 0xC3];
    let val: f32 = 1e10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &val as *const f32 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rcx, 10_000_000_000);
}

#[test]
fn test_cvttss2si_vector_xmm() {
    // cvttss2si ecx, xmm3; ret   ->   F3 0F 2C CB  C3
    let bytes = [0xF3, 0x0F, 0x2C, 0xCB, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[3] = [
        pack_f32(-2.9, f32::from_bits(0xAAAA_AAAA)),
        0x1111_2222_3333_4444,
    ];
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.rcx, (-2i32) as u32 as u64);
}

#[test]
fn test_cvtsi2ss_cvttss2si_roundtrip() {
    // cvtsi2ss xmm0, eax ; cvttss2si eax, xmm0 ; ret
    // Integers below 2^24 survive the round trip exactly.
    let bytes = [
        0xF3, 0x0F, 0x2A, 0xC0, // cvtsi2ss xmm0, eax
        0xF3, 0x0F, 0x2C, 0xC0, // cvttss2si eax, xmm0
        0xC3,
    ];
    for v in [0i32, 1, -1, 12345, -12345, 16_777_215, -16_777_215] {
        let mut ctx = RegisterContext::default();
        ctx.rax = v as u32 as u64;
        let ctx = run_test(&bytes, ctx);
        assert_eq!(ctx.rax, v as u32 as u64, "roundtrip of {v}");
    }
}

//  Packed integer compare: PCMPEQB/W/D/Q and PCMPGTB/W/D/Q

/// Run a `<op> xmm0, xmm1` snippet in both XMM representations and assert both
/// agree on `xmm0`. Returns nothing - the assertion is the point.
fn assert_packed_both(bytes: &[u8], xmm0: [u64; 2], xmm1: [u64; 2], expected: [u64; 2]) {
    let seed = || {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = xmm0;
        ctx.xmm[1] = xmm1;
        ctx
    };
    assert_eq!(run_test(bytes, seed()).xmm[0], expected, "scalar-xmm mode");
    assert_eq!(
        run_test_vec(bytes, seed()).xmm[0],
        expected,
        "vector-xmm mode"
    );
}

#[test]
fn test_pcmpeqd() {
    // pcmpeqd xmm0, xmm1; ret  ->  66 0F 76 C1  C3
    let bytes = [0x66, 0x0F, 0x76, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [0x0000_0001_0000_0002, 0x0000_0003_0000_0004],
        [0x0000_0001_FFFF_FFFF, 0x0000_0003_0000_0004],
        // lo: dword0 differs -> 0, dword1 equal -> ~0; hi: both dwords equal.
        [0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_FFFF_FFFF],
    );
}

#[test]
fn test_pcmpeqd_self_is_all_ones() {
    // The `pcmpeqd xmm0, xmm0` idiom compilers use to materialise an all-ones
    // register.  ->  66 0F 76 C0  C3
    let bytes = [0x66, 0x0F, 0x76, 0xC0, 0xC3];
    assert_packed_both(
        &bytes,
        [0x0123_4567_89AB_CDEF, 0],
        [0, 0],
        [u64::MAX, u64::MAX],
    );
}

#[test]
fn test_pcmpeqd_m128() {
    // pcmpeqd xmm0, [rdi]; ret  ->  66 0F 76 07  C3
    let bytes = [0x66, 0x0F, 0x76, 0x07, 0xC3];
    let src: [u64; 2] = [0x0000_0001_0000_0002, 0];
    let mut ctx = RegisterContext::default();
    ctx.rdi = src.as_ptr() as u64;
    ctx.xmm[0] = [0x0000_0001_0000_0009, 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0xFFFF_FFFF_0000_0000, u64::MAX]);
}

#[test]
fn test_pcmpeqb() {
    // pcmpeqb xmm0, xmm1; ret  ->  66 0F 74 C1  C3
    let bytes = [0x66, 0x0F, 0x74, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [0x0011_2233_4455_6677, 0xAABB_CCDD_EEFF_0011],
        [0x0011_2233_4455_6600, 0x0000_0000_0000_0000],
        // lo: only byte 0 differs. hi: only byte 1 (0x00) matches the zero src.
        [0xFFFF_FFFF_FFFF_FF00, 0x0000_0000_0000_FF00],
    );
}

#[test]
fn test_pcmpeqw() {
    // pcmpeqw xmm0, xmm1; ret  ->  66 0F 75 C1  C3
    let bytes = [0x66, 0x0F, 0x75, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [0x0001_0002_0003_0004, 0],
        [0x0001_9999_0003_9999, 0],
        [0xFFFF_0000_FFFF_0000, u64::MAX],
    );
}

#[test]
fn test_pcmpeqq() {
    // pcmpeqq xmm0, xmm1; ret  ->  66 0F 38 29 C1  C3  (SSE4.1)
    let bytes = [0x66, 0x0F, 0x38, 0x29, 0xC1, 0xC3];
    assert_packed_both(&bytes, [5, 7], [5, 8], [u64::MAX, 0]);
}

#[test]
fn test_pcmpgtb() {
    // pcmpgtb xmm0, xmm1; ret  ->  66 0F 64 C1  C3
    let bytes = [0x66, 0x0F, 0x64, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        // byte 0 of lo is 1, byte 0 of hi is 0x80 (-128)
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0080],
        [0x0000_0000_0000_00FF, 0x0000_0000_0000_00FF],
        // 1 > -1 in lo; -128 > -1 is false in hi. Zero lanes are never >.
        [0x0000_0000_0000_00FF, 0x0000_0000_0000_0000],
    );
}

#[test]
fn test_pcmpgtw() {
    // pcmpgtw xmm0, xmm1; ret  ->  66 0F 65 C1  C3
    let bytes = [0x66, 0x0F, 0x65, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [0x0000_0000_8000_7FFF, 0],
        [0x0000_0000_7FFF_8000, 0],
        // lane0: 32767 > -32768 -> ones; lane1: -32768 > 32767 -> zero.
        [0x0000_0000_0000_FFFF, 0],
    );
}

#[test]
fn test_pcmpgtd() {
    // pcmpgtd xmm0, xmm1; ret  ->  66 0F 66 C1  C3
    let bytes = [0x66, 0x0F, 0x66, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [0xFFFF_FFFF_0000_0005, 0],
        [0x0000_0001_FFFF_FFFF, 0],
        // lane0: 5 > -1; lane1: -1 > 1 is false.
        [0x0000_0000_FFFF_FFFF, 0],
    );
}

#[test]
fn test_pcmpgtq() {
    // pcmpgtq xmm0, xmm1; ret  ->  66 0F 38 37 C1  C3  (SSE4.2)
    let bytes = [0x66, 0x0F, 0x38, 0x37, 0xC1, 0xC3];
    assert_packed_both(
        &bytes,
        [1, u64::MAX],
        [u64::MAX, 1],
        // 1 > -1; -1 > 1 is false.
        [u64::MAX, 0],
    );
}

//  Sign-mask extraction: PMOVMSKB / MOVMSKPS / MOVMSKPD

#[test]
fn test_pmovmskb() {
    // pmovmskb eax, xmm0; ret  ->  66 0F D7 C0  C3
    let bytes = [0x66, 0x0F, 0xD7, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    // lo bytes (lane 0 first): 00 80 00 80 00 80 00 80 -> bits 1,3,5,7
    ctx.xmm[0] = [0x8000_8000_8000_8000, 0x00FF_00FF_00FF_00FF];
    // hi bytes: FF 00 FF 00 FF 00 FF 00 -> bits 8,10,12,14
    ctx.rax = 0xDEAD_BEEF_DEAD_BEEF;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x55AA, "pmovmskb zero-extends into the full GPR");
}

#[test]
fn test_pmovmskb_all_and_none() {
    let bytes = [0x66, 0x0F, 0xD7, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [u64::MAX, u64::MAX];
    assert_eq!(run_test(&bytes, ctx).rax, 0xFFFF);

    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x7F7F_7F7F_7F7F_7F7F, 0x7F7F_7F7F_7F7F_7F7F];
    assert_eq!(run_test(&bytes, ctx).rax, 0);
}

#[test]
fn test_movmskps() {
    // movmskps eax, xmm0; ret  ->  0F 50 C0  C3
    let bytes = [0x0F, 0x50, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    // dwords (lane 0 first): 00000000 80000000 7FFFFFFF FFFFFFFF -> bits 1 and 3
    ctx.xmm[0] = [0x8000_0000_0000_0000, 0xFFFF_FFFF_7FFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0b1010);
}

#[test]
fn test_movmskpd() {
    // movmskpd eax, xmm0; ret  ->  66 0F 50 C0  C3
    let bytes = [0x66, 0x0F, 0x50, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0b01);
}

//  Packed integer min / max / average / multiply

#[test]
fn test_pminub_pmaxub() {
    let a = [0x7766_5544_3322_1100u64, 0];
    let b = [0x0011_2233_4455_6677u64, 0];
    // pminub xmm0, xmm1  ->  66 0F DA C1
    assert_packed_both(
        &[0x66, 0x0F, 0xDA, 0xC1, 0xC3],
        a,
        b,
        [0x0011_2233_3322_1100, 0],
    );
    // pmaxub xmm0, xmm1  ->  66 0F DE C1
    assert_packed_both(
        &[0x66, 0x0F, 0xDE, 0xC1, 0xC3],
        a,
        b,
        [0x7766_5544_4455_6677, 0],
    );
}

#[test]
fn test_pminsw_pmaxsw() {
    // words, lane 0 first: dst = [-32768, -1, 2, 1], src = [32767, 1, -2, 0]
    let a = [0x0001_0002_FFFF_8000u64, 0];
    let b = [0x0000_FFFE_0001_7FFFu64, 0];
    // pminsw xmm0, xmm1  ->  66 0F EA C1
    assert_packed_both(
        &[0x66, 0x0F, 0xEA, 0xC1, 0xC3],
        a,
        b,
        [0x0000_FFFE_FFFF_8000, 0],
    );
    // pmaxsw xmm0, xmm1  ->  66 0F EE C1
    assert_packed_both(
        &[0x66, 0x0F, 0xEE, 0xC1, 0xC3],
        a,
        b,
        [0x0001_0002_0001_7FFF, 0],
    );
}

#[test]
fn test_pavgb_rounds_up() {
    // pavgb xmm0, xmm1  ->  66 0F E0 C1
    // lane0 = (1 + 2 + 1) >> 1 = 2; lane1 = (255 + 1 + 1) >> 1 = 128.
    assert_packed_both(
        &[0x66, 0x0F, 0xE0, 0xC1, 0xC3],
        [0x0000_0000_0000_FF01, 0],
        [0x0000_0000_0000_0102, 0],
        [0x0000_0000_0000_8002, 0],
    );
}

#[test]
fn test_pavgw_rounds_up() {
    // pavgw xmm0, xmm1  ->  66 0F E3 C1
    // (65535 + 1 + 1) >> 1 = 32768 - the rounding carry must not be truncated.
    assert_packed_both(
        &[0x66, 0x0F, 0xE3, 0xC1, 0xC3],
        [0x0000_0000_0000_FFFF, 0],
        [0x0000_0000_0000_0001, 0],
        [0x0000_0000_0000_8000, 0],
    );
}

#[test]
fn test_pmullw() {
    // pmullw xmm0, xmm1  ->  66 0F D5 C1
    // lane0 = 3 * 5 = 0x000F; lane1 = 0x1234 * 2 = 0x2468.
    assert_packed_both(
        &[0x66, 0x0F, 0xD5, 0xC1, 0xC3],
        [0x0000_0000_1234_0003, 0],
        [0x0000_0000_0002_0005, 0],
        [0x0000_0000_2468_000F, 0],
    );
}

#[test]
fn test_pmulhw_signed() {
    // pmulhw xmm0, xmm1  ->  66 0F E5 C1
    // lane0 = (-1 * 2) >> 16 = 0xFFFF; lane1 = (16384 * 4) >> 16 = 1.
    assert_packed_both(
        &[0x66, 0x0F, 0xE5, 0xC1, 0xC3],
        [0x0000_0000_4000_FFFF, 0],
        [0x0000_0000_0004_0002, 0],
        [0x0000_0000_0001_FFFF, 0],
    );
}

#[test]
fn test_pmulhuw_unsigned() {
    // pmulhuw xmm0, xmm1  ->  66 0F E4 C1
    // Same inputs as PMULHW, but lane0 is 65535 * 2 = 0x0001_FFFE -> 1.
    assert_packed_both(
        &[0x66, 0x0F, 0xE4, 0xC1, 0xC3],
        [0x0000_0000_4000_FFFF, 0],
        [0x0000_0000_0004_0002, 0],
        [0x0000_0000_0001_0001, 0],
    );
}

//  Unpack / interleave: PUNPCK{L,H}{BW,WD,DQ,QDQ} and the ps/pd aliases

/// dst/src seeds whose every byte is distinct, so an unpack test pins down the
/// exact lane order: dst bytes are 0x00..0x0F, src bytes 0x10..0x1F.
const UNPACK_DST: [u64; 2] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
const UNPACK_SRC: [u64; 2] = [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918];

#[test]
fn test_punpcklbw() {
    // punpcklbw xmm0, xmm1  ->  66 0F 60 C1
    assert_packed_both(
        &[0x66, 0x0F, 0x60, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1303_1202_1101_1000, 0x1707_1606_1505_1404],
    );
}

#[test]
fn test_punpckhbw() {
    // punpckhbw xmm0, xmm1  ->  66 0F 68 C1
    assert_packed_both(
        &[0x66, 0x0F, 0x68, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1B0B_1A0A_1909_1808, 0x1F0F_1E0E_1D0D_1C0C],
    );
}

#[test]
fn test_punpcklwd() {
    // punpcklwd xmm0, xmm1  ->  66 0F 61 C1
    assert_packed_both(
        &[0x66, 0x0F, 0x61, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1312_0302_1110_0100, 0x1716_0706_1514_0504],
    );
}

#[test]
fn test_punpckhwd() {
    // punpckhwd xmm0, xmm1  ->  66 0F 69 C1
    assert_packed_both(
        &[0x66, 0x0F, 0x69, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1B1A_0B0A_1918_0908, 0x1F1E_0F0E_1D1C_0D0C],
    );
}

#[test]
fn test_punpckhdq() {
    // punpckhdq xmm0, xmm1  ->  66 0F 6A C1
    assert_packed_both(
        &[0x66, 0x0F, 0x6A, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1B1A_1918_0B0A_0908, 0x1F1E_1D1C_0F0E_0D0C],
    );
}

#[test]
fn test_punpcklqdq() {
    // punpcklqdq xmm0, xmm1  ->  66 0F 6C C1
    assert_packed_both(
        &[0x66, 0x0F, 0x6C, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [UNPACK_DST[0], UNPACK_SRC[0]],
    );
}

#[test]
fn test_punpckhqdq() {
    // punpckhqdq xmm0, xmm1  ->  66 0F 6D C1
    assert_packed_both(
        &[0x66, 0x0F, 0x6D, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [UNPACK_DST[1], UNPACK_SRC[1]],
    );
}

#[test]
fn test_unpcklps_aliases_punpckldq() {
    // unpcklps xmm0, xmm1  ->  0F 14 C1
    assert_packed_both(
        &[0x0F, 0x14, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1312_1110_0302_0100, 0x1716_1514_0706_0504],
    );
}

#[test]
fn test_unpckhps_aliases_punpckhdq() {
    // unpckhps xmm0, xmm1  ->  0F 15 C1
    assert_packed_both(
        &[0x0F, 0x15, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [0x1B1A_1918_0B0A_0908, 0x1F1E_1D1C_0F0E_0D0C],
    );
}

#[test]
fn test_unpcklpd_aliases_punpcklqdq() {
    // unpcklpd xmm0, xmm1  ->  66 0F 14 C1
    assert_packed_both(
        &[0x66, 0x0F, 0x14, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [UNPACK_DST[0], UNPACK_SRC[0]],
    );
}

#[test]
fn test_unpckhpd_still_works() {
    // unpckhpd xmm0, xmm1  ->  66 0F 15 C1  (now routed through handle_punpck)
    assert_packed_both(
        &[0x66, 0x0F, 0x15, 0xC1, 0xC3],
        UNPACK_DST,
        UNPACK_SRC,
        [UNPACK_DST[1], UNPACK_SRC[1]],
    );
}

//  Whole-register byte shifts: PSLLDQ / PSRLDQ

#[test]
fn test_pslldq() {
    // pslldq xmm0, 3; ret  ->  66 0F 73 F8 03  C3
    let bytes = [0x66, 0x0F, 0x73, 0xF8, 0x03, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0807_0605_0403_0201, 0x100F_0E0D_0C0B_0A09];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x0504_0302_0100_0000, 0x0D0C_0B0A_0908_0706]);
}

#[test]
fn test_pslldq_crosses_the_halves() {
    // pslldq xmm0, 9; ret  ->  66 0F 73 F8 09  C3  (shift >= 8 bytes)
    let bytes = [0x66, 0x0F, 0x73, 0xF8, 0x09, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0807_0605_0403_0201, 0x100F_0E0D_0C0B_0A09];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0, 0x0706_0504_0302_0100]);
}

#[test]
fn test_psrldq() {
    // psrldq xmm0, 3; ret  ->  66 0F 73 D8 03  C3
    let bytes = [0x66, 0x0F, 0x73, 0xD8, 0x03, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x0807_0605_0403_0201, 0x100F_0E0D_0C0B_0A09];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x0B0A_0908_0706_0504, 0x0000_0010_0F0E_0D0C]);
}

#[test]
fn test_psrldq_overshift_clears() {
    // psrldq xmm0, 20; ret  ->  66 0F 73 D8 14  C3  (count > 15)
    let bytes = [0x66, 0x0F, 0x73, 0xD8, 0x14, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [u64::MAX, u64::MAX];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0, 0]);
}

//  Remaining scalar/packed floating-point arithmetic

#[test]
fn test_subsd() {
    // subsd xmm0, xmm1; ret  ->  F2 0F 5C C1  C3
    let bytes = [0xF2, 0x0F, 0x5C, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [9.0f64.to_bits(), 0xDEAD_BEEF_CAFE_BABE];
    ctx.xmm[1] = [4.0f64.to_bits(), 0x1111_1111_1111_1111];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 5.0);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE, "high half preserved");
}

#[test]
fn test_mulpd_divpd() {
    let seed = || {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [8.0f64.to_bits(), 3.0f64.to_bits()];
        ctx.xmm[1] = [4.0f64.to_bits(), 0.5f64.to_bits()];
        ctx
    };
    // mulpd xmm0, xmm1  ->  66 0F 59 C1
    let ctx = run_test(&[0x66, 0x0F, 0x59, 0xC1, 0xC3], seed());
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 32.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 1.5);
    // divpd xmm0, xmm1  ->  66 0F 5E C1
    let ctx = run_test(&[0x66, 0x0F, 0x5E, 0xC1, 0xC3], seed());
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 2.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 6.0);
}

#[test]
fn test_addps_subps_divps() {
    let seed = || {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(1.5, 8.0), pack_f32(-3.0, 9.0)];
        ctx.xmm[1] = [pack_f32(2.5, 4.0), pack_f32(3.0, 0.5)];
        ctx
    };
    // addps xmm0, xmm1  ->  0F 58 C1
    let ctx = run_test(&[0x0F, 0x58, 0xC1, 0xC3], seed());
    assert_eq!(ctx.xmm[0][0], pack_f32(4.0, 12.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(0.0, 9.5));
    // subps xmm0, xmm1  ->  0F 5C C1
    let ctx = run_test(&[0x0F, 0x5C, 0xC1, 0xC3], seed());
    assert_eq!(ctx.xmm[0][0], pack_f32(-1.0, 4.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(-6.0, 8.5));
    // divps xmm0, xmm1  ->  0F 5E C1
    let ctx = run_test(&[0x0F, 0x5E, 0xC1, 0xC3], seed());
    assert_eq!(ctx.xmm[0][0], pack_f32(0.6, 2.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(-1.0, 18.0));
}

//  MIN / MAX - x86's asymmetric, non-IEEE flavour

#[test]
fn test_minsd_maxsd() {
    let seed = |a: f64, b: f64| {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [a.to_bits(), 0xDEAD_BEEF_CAFE_BABE];
        ctx.xmm[1] = [b.to_bits(), 0];
        ctx
    };
    // minsd xmm0, xmm1  ->  F2 0F 5D C1
    let min = [0xF2, 0x0F, 0x5D, 0xC1, 0xC3];
    // maxsd xmm0, xmm1  ->  F2 0F 5F C1
    let max = [0xF2, 0x0F, 0x5F, 0xC1, 0xC3];

    let ctx = run_test(&min, seed(3.0, 7.0));
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 3.0);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE, "high half preserved");
    assert_eq!(
        f64::from_bits(run_test(&min, seed(7.0, 3.0)).xmm[0][0]),
        3.0
    );
    assert_eq!(
        f64::from_bits(run_test(&max, seed(3.0, 7.0)).xmm[0][0]),
        7.0
    );
    assert_eq!(
        f64::from_bits(run_test(&max, seed(7.0, 3.0)).xmm[0][0]),
        7.0
    );
}

#[test]
fn test_minsd_returns_source_on_nan() {
    let min = [0xF2, 0x0F, 0x5D, 0xC1, 0xC3];
    let seed = |a: u64, b: u64| {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [a, 0];
        ctx.xmm[1] = [b, 0];
        ctx
    };

    // A NaN in *either* operand makes the comparison false, so the source
    // operand is returned - the documented x86 rule, and the reason this is not
    // an IEEE minNum.
    let nan = f64::NAN.to_bits();
    assert_eq!(
        run_test(&min, seed(3.0f64.to_bits(), nan)).xmm[0][0],
        nan,
        "NaN source is returned unchanged"
    );
    assert_eq!(
        f64::from_bits(run_test(&min, seed(nan, 1.0f64.to_bits())).xmm[0][0]),
        1.0,
        "NaN destination loses to the source"
    );
    // min(-0.0, +0.0) likewise returns the source, +0.0.
    assert_eq!(
        run_test(&min, seed((-0.0f64).to_bits(), 0.0f64.to_bits())).xmm[0][0],
        0.0f64.to_bits(),
        "signed zeros: the source wins"
    );
}

#[test]
fn test_minss_maxss() {
    let seed = |a: f32, b: f32| {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(a, f32::from_bits(0xAAAA_AAAA)), 0x1111_2222];
        ctx.xmm[1] = [pack_f32(b, 0.0), 0];
        ctx
    };
    // minss xmm0, xmm1  ->  F3 0F 5D C1 ; maxss  ->  F3 0F 5F C1
    let ctx = run_test(&[0xF3, 0x0F, 0x5D, 0xC1, 0xC3], seed(2.5, 4.0));
    assert_eq!(
        ctx.xmm[0][0],
        pack_f32(2.5, f32::from_bits(0xAAAA_AAAA)),
        "lane 1 preserved"
    );
    assert_eq!(ctx.xmm[0][1], 0x1111_2222, "high half preserved");
    let ctx = run_test(&[0xF3, 0x0F, 0x5F, 0xC1, 0xC3], seed(2.5, 4.0));
    assert_eq!(f32::from_bits(ctx.xmm[0][0] as u32), 4.0);
}

#[test]
fn test_minps_maxps() {
    let seed = || {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(1.0, 5.0), pack_f32(-2.0, 8.0)];
        ctx.xmm[1] = [pack_f32(3.0, 2.0), pack_f32(-7.0, 9.0)];
        ctx
    };
    // minps xmm0, xmm1  ->  0F 5D C1
    let ctx = run_test(&[0x0F, 0x5D, 0xC1, 0xC3], seed());
    assert_eq!(ctx.xmm[0][0], pack_f32(1.0, 2.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(-7.0, 8.0));
    // maxps xmm0, xmm1  ->  0F 5F C1
    let ctx = run_test(&[0x0F, 0x5F, 0xC1, 0xC3], seed());
    assert_eq!(ctx.xmm[0][0], pack_f32(3.0, 5.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(-2.0, 9.0));
}

#[test]
fn test_minpd_maxpd() {
    let seed = || {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [1.0f64.to_bits(), 8.0f64.to_bits()];
        ctx.xmm[1] = [3.0f64.to_bits(), 2.0f64.to_bits()];
        ctx
    };
    // minpd xmm0, xmm1  ->  66 0F 5D C1
    let ctx = run_test(&[0x66, 0x0F, 0x5D, 0xC1, 0xC3], seed());
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 2.0);
    // maxpd xmm0, xmm1  ->  66 0F 5F C1
    let ctx = run_test(&[0x66, 0x0F, 0x5F, 0xC1, 0xC3], seed());
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 3.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 8.0);
}

//  SQRT

#[test]
fn test_sqrtsd() {
    // sqrtsd xmm0, xmm1; ret  ->  F2 0F 51 C1  C3
    let bytes = [0xF2, 0x0F, 0x51, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1234_5678_9ABC_DEF0, 0xDEAD_BEEF_CAFE_BABE];
    ctx.xmm[1] = [16.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        f64::from_bits(ctx.xmm[0][0]),
        4.0,
        "reads the *source* lane"
    );
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE, "high half preserved");
}

#[test]
fn test_sqrtss() {
    // sqrtss xmm0, xmm1; ret  ->  F3 0F 51 C1  C3
    let bytes = [0xF3, 0x0F, 0x51, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(0.0, f32::from_bits(0xAAAA_AAAA)), 0x1111_2222];
    ctx.xmm[1] = [pack_f32(9.0, 0.0), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(3.0, f32::from_bits(0xAAAA_AAAA)));
    assert_eq!(ctx.xmm[0][1], 0x1111_2222);
}

#[test]
fn test_sqrtps() {
    // sqrtps xmm0, xmm1; ret  ->  0F 51 C1  C3
    let bytes = [0x0F, 0x51, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_f32(4.0, 9.0), pack_f32(16.0, 25.0)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(2.0, 3.0));
    assert_eq!(ctx.xmm[0][1], pack_f32(4.0, 5.0));
}

#[test]
fn test_sqrtpd() {
    // sqrtpd xmm0, xmm1; ret  ->  66 0F 51 C1  C3
    let bytes = [0x66, 0x0F, 0x51, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [2.25f64.to_bits(), 100.0f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1.5);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 10.0);
}

//  UCOMISS / COMISS (scalar single compare -> EFLAGS)

#[test]
fn test_ucomiss_flags() {
    // ucomiss xmm0, xmm1; ret  ->  0F 2E C1  C3
    let bytes = [0x0F, 0x2E, 0xC1, 0xC3];
    let seed = |a: f32, b: f32| {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(a, 0.0), 0];
        ctx.xmm[1] = [pack_f32(b, 0.0), 0];
        ctx
    };

    let ctx = run_test(&bytes, seed(1.0, 2.0));
    assert_eq!((ctx.cf, ctx.zf, ctx.pf), (1, 0, 0), "less");
    let ctx = run_test(&bytes, seed(5.0, 5.0));
    assert_eq!((ctx.cf, ctx.zf, ctx.pf), (0, 1, 0), "equal");
    let ctx = run_test(&bytes, seed(9.0, 4.0));
    assert_eq!((ctx.cf, ctx.zf, ctx.pf), (0, 0, 0), "greater");
    let ctx = run_test(&bytes, seed(f32::NAN, 1.0));
    assert_eq!((ctx.cf, ctx.zf, ctx.pf), (1, 1, 1), "unordered");
}

#[test]
fn test_comiss_matches_ucomiss() {
    // comiss xmm0, xmm1; ret  ->  0F 2F C1  C3
    let bytes = [0x0F, 0x2F, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(1.0, 0.0), 0];
    ctx.xmm[1] = [pack_f32(2.0, 0.0), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!((ctx.cf, ctx.zf, ctx.pf), (1, 0, 0));
}

//  SSE2 conversions

#[test]
fn test_cvtsi2sd_r32() {
    // cvtsi2sd xmm0, eax; ret  ->  F2 0F 2A C0  C3
    let bytes = [0xF2, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = (-5i32) as u32 as u64;
    ctx.xmm[0] = [0, 0xDEAD_BEEF_CAFE_BABE];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), -5.0);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE, "high half preserved");
}

#[test]
fn test_cvtsi2sd_r64() {
    // cvtsi2sd xmm0, rax; ret  ->  F2 48 0F 2A C0  C3
    let bytes = [0xF2, 0x48, 0x0F, 0x2A, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 1_000_000_000_000;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1e12);
}

#[test]
fn test_cvttsd2si() {
    // cvttsd2si eax, xmm0; ret  ->  F2 0F 2C C0  C3
    let bytes = [0xF2, 0x0F, 0x2C, 0xC0, 0xC3];
    let run = |v: f64| {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [v.to_bits(), 0];
        run_test(&bytes, ctx).rax
    };
    assert_eq!(run(-12.75), (-12i32) as u32 as u64, "truncates toward zero");
    assert_eq!(run(12.75), 12);
    // Out-of-range and NaN both yield the 32-bit integer indefinite value.
    const INDEF32: u64 = 0x8000_0000;
    assert_eq!(run(f64::NAN), INDEF32);
    assert_eq!(run(1e30), INDEF32);
    assert_eq!(run(-1e30), INDEF32);
}

#[test]
fn test_cvttsd2si_r64() {
    // cvttsd2si rax, xmm0; ret  ->  F2 48 0F 2C C0  C3
    let bytes = [0xF2, 0x48, 0x0F, 0x2C, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [1e12f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 1_000_000_000_000);
}

#[test]
fn test_cvtss2sd() {
    // cvtss2sd xmm0, xmm1; ret  ->  F3 0F 5A C1  C3
    let bytes = [0xF3, 0x0F, 0x5A, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0, 0xDEAD_BEEF_CAFE_BABE];
    ctx.xmm[1] = [pack_f32(1.5, 0.0), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1.5);
    assert_eq!(ctx.xmm[0][1], 0xDEAD_BEEF_CAFE_BABE, "high half preserved");
}

#[test]
fn test_cvtsd2ss() {
    // cvtsd2ss xmm0, xmm1; ret  ->  F2 0F 5A C1  C3
    let bytes = [0xF2, 0x0F, 0x5A, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(0.0, f32::from_bits(0xAAAA_AAAA)), 0x1111_2222];
    ctx.xmm[1] = [2.5f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], pack_f32(2.5, f32::from_bits(0xAAAA_AAAA)));
    assert_eq!(ctx.xmm[0][1], 0x1111_2222, "bits[127:32] preserved");
}

#[test]
fn test_cvtsd2ss_cvtss2sd_roundtrip() {
    // cvtsd2ss xmm0, xmm0 ; cvtss2sd xmm0, xmm0 ; ret
    let bytes = [
        0xF2, 0x0F, 0x5A, 0xC0, // cvtsd2ss xmm0, xmm0
        0xF3, 0x0F, 0x5A, 0xC0, // cvtss2sd xmm0, xmm0
        0xC3,
    ];
    // Values exactly representable as f32 survive the narrowing round trip.
    for v in [0.0f64, 1.0, -1.0, 0.5, 1024.0, -8388608.0] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [v.to_bits(), 0];
        let ctx = run_test(&bytes, ctx);
        assert_eq!(f64::from_bits(ctx.xmm[0][0]), v, "roundtrip of {v}");
    }
}

//  Packed conversions: CVTPD2PS / CVTPS2PD / CVTDQ2PS / CVT[T]PS2DQ /
//  CVTDQ2PD / CVT[T]PD2DQ.

/// Split a 64-bit XMM half into its two `f32` lanes.
fn unpack_f32(half: u64) -> (f32, f32) {
    (
        f32::from_bits(half as u32),
        f32::from_bits((half >> 32) as u32),
    )
}

/// Pack two signed dwords into a 64-bit XMM half.
fn pack_i32(l0: i32, l1: i32) -> u64 {
    (l0 as u32 as u64) | ((l1 as u32 as u64) << 32)
}

/// Split a 64-bit XMM half into its two signed dword lanes.
fn unpack_i32(half: u64) -> (i32, i32) {
    (half as u32 as i32, (half >> 32) as u32 as i32)
}

#[test]
fn test_cvtpd2ps() {
    // cvtpd2ps xmm0, xmm1; ret  ->  66 0F 5A C1  C3
    let bytes = [0x66, 0x0F, 0x5A, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1122_3344_5566_7788];
    ctx.xmm[1] = [1.5f64.to_bits(), (-2.25f64).to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_f32(ctx.xmm[0][0]), (1.5, -2.25));
    assert_eq!(
        ctx.xmm[0][1], 0,
        "cvtpd2ps must zero bits[127:64] of the destination"
    );
}

#[test]
fn test_cvtpd2ps_rounds_and_overflows() {
    let bytes = [0x66, 0x0F, 0x5A, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // A double too large for f32 becomes +inf; a tiny one becomes +0.
    ctx.xmm[1] = [1e300f64.to_bits(), 1e-300f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    let (l0, l1) = unpack_f32(ctx.xmm[0][0]);
    assert_eq!(l0, f32::INFINITY);
    assert_eq!(l1, 0.0);
    assert!(l1.is_sign_positive());
}

#[test]
fn test_cvtpd2ps_mem() {
    // cvtpd2ps xmm0, [rdi]; ret  ->  66 0F 5A 07  C3
    let bytes = [0x66, 0x0F, 0x5A, 0x07, 0xC3];
    let src = [3.5f64, -0.5];
    let mut ctx = RegisterContext::default();
    ctx.rdi = src.as_ptr() as u64;
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_f32(ctx.xmm[0][0]), (3.5, -0.5));
    assert_eq!(ctx.xmm[0][1], 0);
}

#[test]
fn test_cvtps2pd() {
    // cvtps2pd xmm0, xmm1; ret  ->  0F 5A C1  C3
    let bytes = [0x0F, 0x5A, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1122_3344_5566_7788];
    // Only the low two lanes of the source participate.
    ctx.xmm[1] = [pack_f32(1.5, -8.0), pack_f32(99.0, 99.0)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 1.5);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), -8.0);
}

#[test]
fn test_cvtps2pd_mem_reads_only_8_bytes() {
    // cvtps2pd xmm0, [rdi]; ret  ->  0F 5A 07  C3
    let bytes = [0x0F, 0x5A, 0x07, 0xC3];
    // A 2-element f32 array: reading 16 bytes here would run off the end.
    let src = [2.5f32, -0.25];
    let mut ctx = RegisterContext::default();
    ctx.rdi = src.as_ptr() as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 2.5);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), -0.25);
}

#[test]
fn test_cvtpd2ps_cvtps2pd_roundtrip() {
    // cvtpd2ps xmm0, xmm0 ; cvtps2pd xmm0, xmm0 ; ret
    let bytes = [
        0x66, 0x0F, 0x5A, 0xC0, // cvtpd2ps xmm0, xmm0
        0x0F, 0x5A, 0xC0, // cvtps2pd xmm0, xmm0
        0xC3,
    ];
    for (a, b) in [(0.0f64, 1.0f64), (-1.0, 1024.0), (0.5, -8388608.0)] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [a.to_bits(), b.to_bits()];
        let ctx = run_test(&bytes, ctx);
        assert_eq!(f64::from_bits(ctx.xmm[0][0]), a, "lane 0 roundtrip of {a}");
        assert_eq!(f64::from_bits(ctx.xmm[0][1]), b, "lane 1 roundtrip of {b}");
    }
}

#[test]
fn test_cvtdq2ps() {
    // cvtdq2ps xmm0, xmm1; ret  ->  0F 5B C1  C3
    let bytes = [0x0F, 0x5B, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_i32(0, -1), pack_i32(1 << 24, i32::MIN)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_f32(ctx.xmm[0][0]), (0.0, -1.0));
    assert_eq!(
        unpack_f32(ctx.xmm[0][1]),
        (16777216.0, -2147483648.0),
        "large magnitudes round to nearest f32"
    );
}

#[test]
fn test_cvtps2dq_rounds_to_nearest_even() {
    // cvtps2dq xmm0, xmm1; ret  ->  66 0F 5B C1  C3
    let bytes = [0x66, 0x0F, 0x5B, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // Ties round to even under the default MXCSR rounding mode.
    ctx.xmm[1] = [pack_f32(1.5, 2.5), pack_f32(-1.5, -0.4)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (2, 2));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (-2, 0));
}

#[test]
fn test_cvttps2dq_truncates() {
    // cvttps2dq xmm0, xmm1; ret  ->  F3 0F 5B C1  C3
    let bytes = [0xF3, 0x0F, 0x5B, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_f32(1.9, -1.9), pack_f32(0.5, -0.5)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (1, -1));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (0, 0));
}

#[test]
fn test_cvttps2dq_indefinite_on_out_of_range() {
    let bytes = [0xF3, 0x0F, 0x5B, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [
        pack_f32(f32::NAN, f32::INFINITY),
        pack_f32(f32::NEG_INFINITY, 1e30),
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        unpack_i32(ctx.xmm[0][0]),
        (i32::MIN, i32::MIN),
        "NaN and +inf yield the integer indefinite value"
    );
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (i32::MIN, i32::MIN));
}

#[test]
fn test_cvtdq2pd() {
    // cvtdq2pd xmm0, xmm1; ret  ->  F3 0F E6 C1  C3
    let bytes = [0xF3, 0x0F, 0xE6, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // Only the low quadword (two dwords) of the source participates.
    ctx.xmm[1] = [pack_i32(7, -9), pack_i32(1234, 5678)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), 7.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), -9.0);
}

#[test]
fn test_cvtdq2pd_mem_reads_only_8_bytes() {
    // cvtdq2pd xmm0, [rdi]; ret  ->  F3 0F E6 07  C3
    let bytes = [0xF3, 0x0F, 0xE6, 0x07, 0xC3];
    let src = [i32::MIN, i32::MAX];
    let mut ctx = RegisterContext::default();
    ctx.rdi = src.as_ptr() as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(ctx.xmm[0][0]), -2147483648.0);
    assert_eq!(f64::from_bits(ctx.xmm[0][1]), 2147483647.0);
}

#[test]
fn test_cvtpd2dq_rounds_and_zeroes_high() {
    // cvtpd2dq xmm0, xmm1; ret  ->  F2 0F E6 C1  C3
    let bytes = [0xF2, 0x0F, 0xE6, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.xmm[1] = [2.5f64.to_bits(), (-3.5f64).to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (2, -4), "ties round to even");
    assert_eq!(
        ctx.xmm[0][1], 0,
        "cvtpd2dq must zero bits[127:64] of the destination"
    );
}

#[test]
fn test_cvttpd2dq_truncates() {
    // cvttpd2dq xmm0, xmm1; ret  ->  66 0F E6 C1  C3
    let bytes = [0x66, 0x0F, 0xE6, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [2.9f64.to_bits(), f64::NAN.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (2, i32::MIN));
    assert_eq!(ctx.xmm[0][1], 0);
}

//  CVTSS2SI / CVTSD2SI - the rounding scalar-to-integer forms.

#[test]
fn test_cvtss2si_rounds_to_nearest_even() {
    // cvtss2si eax, xmm0; ret  ->  F3 0F 2D C0  C3
    let bytes = [0xF3, 0x0F, 0x2D, 0xC0, 0xC3];
    for (input, want) in [(1.5f32, 2i64), (2.5, 2), (-1.5, -2), (0.4, 0), (-0.6, -1)] {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(input, 0.0), 0];
        let ctx = run_test(&bytes, ctx);
        assert_eq!(ctx.rax as i32 as i64, want, "cvtss2si of {input}");
    }
}

#[test]
fn test_cvtsd2si_64bit_and_indefinite() {
    // cvtsd2si rax, xmm0; ret  ->  F2 48 0F 2D C0  C3
    let bytes = [0xF2, 0x48, 0x0F, 0x2D, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [(1e12f64 + 0.5).to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax as i64, 1_000_000_000_000);

    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [f64::NAN.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax as i64, i64::MIN, "NaN yields integer indefinite");
}

//  CMPPS / CMPPD / CMPSS / CMPSD

/// All-ones / all-zeros lane masks, as the `CMP*` family produces them.
const T32: u64 = 0xFFFF_FFFF;
const T64: u64 = 0xFFFF_FFFF_FFFF_FFFF;

#[test]
fn test_cmpps_predicates() {
    // cmpps xmm0, xmm1, imm8; ret  ->  0F C2 C1 ib  C3
    // Lanes: (1<2), (2>1), (equal), (NaN vs 1).
    for (imm, want) in [
        (0u8, [0, 0, T32, 0]),   // EQ
        (1, [T32, 0, 0, 0]),     // LT
        (2, [T32, 0, T32, 0]),   // LE
        (3, [0, 0, 0, T32]),     // UNORD
        (4, [T32, T32, 0, T32]), // NEQ (true for unordered)
        (5, [0, T32, T32, T32]), // NLT
        (6, [0, T32, 0, T32]),   // NLE
        (7, [T32, T32, T32, 0]), // ORD
    ] {
        let bytes = [0x0F, 0xC2, 0xC1, imm, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [pack_f32(1.0, 2.0), pack_f32(3.0, f32::NAN)];
        ctx.xmm[1] = [pack_f32(2.0, 1.0), pack_f32(3.0, 1.0)];
        let ctx = run_test(&bytes, ctx);
        let got = [
            ctx.xmm[0][0] & 0xFFFF_FFFF,
            ctx.xmm[0][0] >> 32,
            ctx.xmm[0][1] & 0xFFFF_FFFF,
            ctx.xmm[0][1] >> 32,
        ];
        assert_eq!(got, want, "cmpps predicate {imm}");
    }
}

#[test]
fn test_cmppd_lt() {
    // cmppd xmm0, xmm1, 1; ret  ->  66 0F C2 C1 01  C3
    let bytes = [0x66, 0x0F, 0xC2, 0xC1, 0x01, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [1.0f64.to_bits(), 5.0f64.to_bits()];
    ctx.xmm[1] = [2.0f64.to_bits(), 5.0f64.to_bits()];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [T64, 0]);
}

#[test]
fn test_cmpss_preserves_high_lanes() {
    // cmpss xmm0, xmm1, 0; ret  ->  F3 0F C2 C1 00  C3
    let bytes = [0xF3, 0x0F, 0xC2, 0xC1, 0x00, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(4.0, 7.0), 0x1111_2222_3333_4444];
    ctx.xmm[1] = [pack_f32(4.0, 9.0), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0] & 0xFFFF_FFFF, T32, "lane 0 compares equal");
    assert_eq!(ctx.xmm[0][0] >> 32, 7.0f32.to_bits() as u64);
    assert_eq!(ctx.xmm[0][1], 0x1111_2222_3333_4444);
}

#[test]
fn test_cmpsd_sse_preserves_high_quadword() {
    // cmpsd xmm0, xmm1, 6; ret  ->  F2 0F C2 C1 06  C3   (NLE)
    let bytes = [0xF2, 0x0F, 0xC2, 0xC1, 0x06, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [9.0f64.to_bits(), 0xDEAD_BEEF];
    ctx.xmm[1] = [3.0f64.to_bits(), 0];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [T64, 0xDEAD_BEEF]);
}

//  SHUFPS / SHUFPD / PSHUFLW / PSHUFHW

#[test]
fn test_shufps() {
    // shufps xmm0, xmm1, 0x1B; ret  ->  0F C6 C1 1B  C3
    // imm 0x1B = 00 01 10 11 -> lanes [dst3, dst2, src1, src0].
    let bytes = [0x0F, 0xC6, 0xC1, 0x1B, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_i32(0, 1), pack_i32(2, 3)];
    ctx.xmm[1] = [pack_i32(10, 11), pack_i32(12, 13)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (3, 2));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (11, 10));
}

#[test]
fn test_shufpd() {
    // shufpd xmm0, xmm1, imm8; ret  ->  66 0F C6 C1 ib  C3
    // imm[0] selects the destination's quadword, imm[1] the source's.
    const D0: u64 = 0xD000_0000_0000_0000;
    const D1: u64 = 0xD111_1111_1111_1111;
    const S0: u64 = 0x5000_0000_0000_0000;
    const S1: u64 = 0x5111_1111_1111_1111;
    for (imm, want) in [(0u8, [D0, S0]), (1, [D1, S0]), (2, [D0, S1]), (3, [D1, S1])] {
        let bytes = [0x66, 0x0F, 0xC6, 0xC1, imm, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [D0, D1];
        ctx.xmm[1] = [S0, S1];
        let ctx = run_test(&bytes, ctx);
        assert_eq!(ctx.xmm[0], want, "shufpd imm8={imm}");
    }
}

#[test]
fn test_pshuflw_and_pshufhw() {
    // pshuflw xmm0, xmm1, 0x1B; ret  ->  F2 0F 70 C1 1B  C3  (reverse low words)
    let bytes = [0xF2, 0x0F, 0x70, 0xC1, 0x1B, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x0004_0003_0002_0001, 0xAAAA_BBBB_CCCC_DDDD];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x0001_0002_0003_0004);
    assert_eq!(ctx.xmm[0][1], 0xAAAA_BBBB_CCCC_DDDD, "high half copied");

    // pshufhw xmm0, xmm1, 0x1B; ret  ->  F3 0F 70 C1 1B  C3
    let bytes = [0xF3, 0x0F, 0x70, 0xC1, 0x1B, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0xAAAA_BBBB_CCCC_DDDD, 0x0004_0003_0002_0001];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0xAAAA_BBBB_CCCC_DDDD, "low half copied");
    assert_eq!(ctx.xmm[0][1], 0x0001_0002_0003_0004);
}

//  PINSRW

#[test]
fn test_pinsrw_reg_all_lanes() {
    for lane in 0..8u8 {
        // pinsrw xmm0, ecx, imm8; ret  ->  66 0F C4 C1 ib  C3
        let bytes = [0x66, 0x0F, 0xC4, 0xC1, lane, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
        // Only the low 16 bits of the GPR are used.
        ctx.rcx = 0xDEAD_BEEF_0000_ABCD;
        let ctx = run_test(&bytes, ctx);

        let mut want = [0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888];
        let half = usize::from(lane) / 4;
        let shift = (u32::from(lane) % 4) * 16;
        want[half] = (want[half] & !(0xFFFFu64 << shift)) | (0xABCDu64 << shift);
        assert_eq!(ctx.xmm[0], want, "pinsrw lane {lane}");
    }
}

#[test]
fn test_pinsrw_imm_selector_masks_to_three_bits() {
    // pinsrw xmm0, ecx, 0x0A; ret - imm8 bits above [2:0] are ignored, so
    // 0x0A selects word 2.
    let bytes = [0x66, 0x0F, 0xC4, 0xC1, 0x0A, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
    ctx.rcx = 0xABCD;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1111_ABCD_3333_4444, 0x5555_6666_7777_8888]);
}

#[test]
fn test_pinsrw_from_mem() {
    // pinsrw xmm0, word ptr [rax], 5; ret  ->  66 0F C4 00 05  C3
    let bytes = [0x66, 0x0F, 0xC4, 0x00, 0x05, 0xC3];
    let mem: u16 = 0xBEEF;
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
    ctx.rax = &mem as *const u16 as u64;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1111_2222_3333_4444, 0x5555_6666_BEEF_8888]);
}

#[test]
fn test_pinsrw_vector_mode() {
    // Same as the register form above, but with native I64X2 XMM vars.
    let bytes = [0x66, 0x0F, 0xC4, 0xC1, 0x07, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
    ctx.rcx = 0xABCD;
    let ctx = run_test_vec(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x1111_2222_3333_4444, 0xABCD_6666_7777_8888]);
}

//  INSERTPS

#[test]
fn test_insertps_selects_source_and_destination_lanes() {
    // insertps xmm0, xmm1, 0xB5; ret
    // source lane 2 -> destination lane 3; zero destination lanes 0 and 2.
    let bytes = [0x66, 0x0F, 0x3A, 0x21, 0xC1, 0xB5, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x3333_3333_2222_2222, 0x5555_5555_4444_4444];
    ctx.xmm[1] = [0xBBBB_BBBB_AAAA_AAAA, 0xDDDD_DDDD_CCCC_CCCC];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.xmm[0], [0x3333_3333_0000_0000, 0xCCCC_CCCC_0000_0000]);
}

#[test]
fn test_insertps_zero_mask_overrides_insert_and_vector_mode_agrees() {
    // Source lane 1 -> destination lane 2, but bit 2 of the zero mask clears
    // that destination lane after the insert.
    let bytes = [0x66, 0x0F, 0x3A, 0x21, 0xC1, 0x64, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x3333_3333_2222_2222, 0x5555_5555_4444_4444];
    ctx.xmm[1] = [0xBBBB_BBBB_AAAA_AAAA, 0xDDDD_DDDD_CCCC_CCCC];
    let scalar = run_test(&bytes, ctx);
    let vector = run_test_vec(&bytes, ctx);
    assert_eq!(
        scalar.xmm[0],
        [0x3333_3333_2222_2222, 0x5555_5555_0000_0000]
    );
    assert_eq!(scalar.xmm, vector.xmm);
}

#[test]
fn test_insertps_memory_source_reads_one_dword() {
    // insertps xmm0, [rax], 0x10; ret -> memory dword into destination lane 1.
    let bytes = [0x66, 0x0F, 0x3A, 0x21, 0x00, 0x10, 0xC3];
    let mem = [0xAABB_CCDD_u32, 0xEEEE_FFFF];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_ptr() as u64;
    ctx.xmm[0] = [0x3333_3333_2222_2222, 0x5555_5555_4444_4444];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.xmm[0], [0xAABB_CCDD_2222_2222, 0x5555_5555_4444_4444]);
}

//  HADDPD / HSUBPD / HADDPS / HSUBPS - SSE3 horizontal arithmetic. The lanes
//  combined are *adjacent within one operand*, which is the whole difference
//  from ADDPD, and reading them the vertical way is a result that is plausible
//  rather than wrong-looking - so every case below uses four distinct values.

#[test]
fn test_haddpd_adds_adjacent_lanes_of_each_operand() {
    // haddpd xmm0, xmm1; ret
    let bytes = [0x66, 0x0F, 0x7C, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // xmm0 = [1.5, 2.25], xmm1 = [3.0, 4.0]
    ctx.xmm[0] = [0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000];
    ctx.xmm[1] = [0x4008_0000_0000_0000, 0x4010_0000_0000_0000];
    let out = run_test(&bytes, ctx);
    // [1.5 + 2.25, 3.0 + 4.0] = [3.75, 7.0]
    assert_eq!(out.xmm[0], [0x400E_0000_0000_0000, 0x401C_0000_0000_0000]);
}

#[test]
fn test_haddpd_with_the_destination_as_its_own_source() {
    // haddpd xmm0, xmm0; ret - the shape a compiler emits to sum a pair, and
    // the one that goes wrong if a result lane is written before both operands
    // have been read.
    let bytes = [0x66, 0x0F, 0x7C, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.xmm[0], [0x400E_0000_0000_0000, 0x400E_0000_0000_0000]);
}

#[test]
fn test_hsubpd_subtracts_the_high_lane_from_the_low_one() {
    // hsubpd xmm0, xmm1; ret
    let bytes = [0x66, 0x0F, 0x7D, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000];
    ctx.xmm[1] = [0x4008_0000_0000_0000, 0x4010_0000_0000_0000];
    let out = run_test(&bytes, ctx);
    // [1.5 - 2.25, 3.0 - 4.0] = [-0.75, -1.0]
    assert_eq!(out.xmm[0], [0xBFE8_0000_0000_0000, 0xBFF0_0000_0000_0000]);
}

#[test]
fn test_haddpd_memory_source() {
    // haddpd xmm0, [rax]; ret
    let bytes = [0x66, 0x0F, 0x7C, 0x00, 0xC3];
    let mem = [3.0_f64, 4.0];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_ptr() as u64;
    ctx.xmm[0] = [0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.xmm[0], [0x400E_0000_0000_0000, 0x401C_0000_0000_0000]);
}

#[test]
fn test_haddps_pairs_all_four_lanes_of_both_operands() {
    // haddps xmm0, xmm1; ret
    let bytes = [0xF2, 0x0F, 0x7C, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // xmm0 = [1.5, 2.5, 3.0, 4.0], xmm1 = [1.0, 2.0, 5.0, 6.0]
    ctx.xmm[0] = [0x4020_0000_3FC0_0000, 0x4080_0000_4040_0000];
    ctx.xmm[1] = [0x4000_0000_3F80_0000, 0x40C0_0000_40A0_0000];
    let out = run_test(&bytes, ctx);
    // [1.5+2.5, 3.0+4.0, 1.0+2.0, 5.0+6.0] = [4.0, 7.0, 3.0, 11.0]
    assert_eq!(out.xmm[0], [0x40E0_0000_4080_0000, 0x4130_0000_4040_0000]);
}

#[test]
fn test_hsubps_subtracts_within_each_adjacent_pair() {
    // hsubps xmm0, xmm1; ret
    let bytes = [0xF2, 0x0F, 0x7D, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x4020_0000_3FC0_0000, 0x4080_0000_4040_0000];
    ctx.xmm[1] = [0x4000_0000_3F80_0000, 0x40C0_0000_40A0_0000];
    let out = run_test(&bytes, ctx);
    // [1.5-2.5, 3.0-4.0, 1.0-2.0, 5.0-6.0] = [-1.0, -1.0, -1.0, -1.0]
    assert_eq!(out.xmm[0], [0xBF80_0000_BF80_0000, 0xBF80_0000_BF80_0000]);
}

#[test]
fn test_horizontal_arithmetic_agrees_in_vector_mode() {
    // The vector-XMM lift is a different path through the same semantics, so it
    // has to produce the same registers.
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000];
    ctx.xmm[1] = [0x4008_0000_0000_0000, 0x4010_0000_0000_0000];
    for bytes in [
        [0x66, 0x0F, 0x7C, 0xC1, 0xC3],
        [0x66, 0x0F, 0x7D, 0xC1, 0xC3],
        [0xF2, 0x0F, 0x7C, 0xC1, 0xC3],
        [0xF2, 0x0F, 0x7D, 0xC1, 0xC3],
    ] {
        assert_eq!(run_test(&bytes, ctx).xmm, run_test_vec(&bytes, ctx).xmm);
    }
}

//  ROUNDSD / ROUNDSS / ROUNDPD / ROUNDPS

/// Every mode of the scalar double form, against the same input, with the upper
/// quadword of the destination checked to be untouched.
#[test]
fn test_roundsd_modes() {
    for (imm, expected) in [
        (0x00u8, 2.0f64), // nearest, ties to even: 2.5 -> 2
        (0x01, 2.0),      // floor
        (0x02, 3.0),      // ceil
        (0x03, 2.0),      // truncate
    ] {
        // roundsd xmm0, xmm1, imm; ret
        let bytes = [0x66, 0x0F, 0x3A, 0x0B, 0xC1, imm, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0, 0xDEAD_BEEF_DEAD_BEEF];
        ctx.xmm[1] = [2.5f64.to_bits(), 0];
        let out = run_test(&bytes, ctx);
        assert_eq!(f64::from_bits(out.xmm[0][0]), expected, "imm {imm:#04x}");
        assert_eq!(out.xmm[0][1], 0xDEAD_BEEF_DEAD_BEEF, "imm {imm:#04x}");
    }
}

/// Negative inputs, where floor and truncate stop agreeing.
#[test]
fn test_roundsd_negative_floor_and_truncate_differ() {
    for (imm, expected) in [(0x01u8, -3.0f64), (0x03, -2.0)] {
        let bytes = [0x66, 0x0F, 0x3A, 0x0B, 0xC1, imm, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[1] = [(-2.5f64).to_bits(), 0];
        let out = run_test(&bytes, ctx);
        assert_eq!(f64::from_bits(out.xmm[0][0]), expected, "imm {imm:#04x}");
    }
}

/// Bit 2 means "use MXCSR", whose reset mode is round-to-nearest-even; see
/// `round_mode`.
#[test]
fn test_roundsd_mxcsr_bit_rounds_to_nearest_even() {
    let bytes = [0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [3.5f64.to_bits(), 0];
    let out = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(out.xmm[0][0]), 4.0);
}

/// ROUNDSS preserves bits[127:32], which is the whole difference from ROUNDPS.
#[test]
fn test_roundss_preserves_the_upper_lanes() {
    // roundss xmm0, xmm1, 1; ret
    let bytes = [0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x01, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1111_1111_0000_0000, 0x2222_2222_3333_3333];
    ctx.xmm[1] = [u64::from(2.75f32.to_bits()), 0];
    let out = run_test(&bytes, ctx);
    assert_eq!(f32::from_bits(out.xmm[0][0] as u32), 2.0);
    assert_eq!(out.xmm[0][0] >> 32, 0x1111_1111);
    assert_eq!(out.xmm[0][1], 0x2222_2222_3333_3333);
}

/// Both lanes of the packed double form, one per rounding direction.
#[test]
fn test_roundpd_rounds_both_lanes() {
    // roundpd xmm0, xmm1, 2 (ceil); ret
    let bytes = [0x66, 0x0F, 0x3A, 0x09, 0xC1, 0x02, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [1.25f64.to_bits(), (-1.25f64).to_bits()];
    let out = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(out.xmm[0][0]), 2.0);
    assert_eq!(f64::from_bits(out.xmm[0][1]), -1.0);
}

/// All four lanes of the packed single form.
#[test]
fn test_roundps_rounds_four_lanes() {
    // roundps xmm0, xmm1, 1 (floor); ret
    let bytes = [0x66, 0x0F, 0x3A, 0x08, 0xC1, 0x01, 0xC3];
    let lanes = [1.5f32, -1.5, 9.75, -0.25];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [
        u64::from(lanes[0].to_bits()) | (u64::from(lanes[1].to_bits()) << 32),
        u64::from(lanes[2].to_bits()) | (u64::from(lanes[3].to_bits()) << 32),
    ];
    let out = run_test(&bytes, ctx);
    let got = [
        f32::from_bits(out.xmm[0][0] as u32),
        f32::from_bits((out.xmm[0][0] >> 32) as u32),
        f32::from_bits(out.xmm[0][1] as u32),
        f32::from_bits((out.xmm[0][1] >> 32) as u32),
    ];
    assert_eq!(got, [1.0, -2.0, 9.0, -1.0]);
}

/// A memory source, which is the form a constant operand compiles to.
#[test]
fn test_roundsd_memory_source() {
    // roundsd xmm0, [rax], 2 (ceil); ret
    let bytes = [0x66, 0x0F, 0x3A, 0x0B, 0x00, 0x02, 0xC3];
    let mem = [7.125f64];
    let mut ctx = RegisterContext::default();
    ctx.rax = mem.as_ptr() as u64;
    let out = run_test(&bytes, ctx);
    assert_eq!(f64::from_bits(out.xmm[0][0]), 8.0);
}

//  RCPPS / RSQRTPS / RCPSS / RSQRTSS - approximate, so compared with a
//  tolerance well inside the architectural 1.5 * 2^-12 relative error bound.

#[test]
fn test_rcpps_and_rsqrtps() {
    // rcpps xmm0, xmm1; ret  ->  0F 53 C1  C3
    let bytes = [0x0F, 0x53, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_f32(1.0, 2.0), pack_f32(4.0, -8.0)];
    let ctx = run_test(&bytes, ctx);
    let got = [unpack_f32(ctx.xmm[0][0]), unpack_f32(ctx.xmm[0][1])];
    for (got, want) in [
        (got[0].0, 1.0f32),
        (got[0].1, 0.5),
        (got[1].0, 0.25),
        (got[1].1, -0.125),
    ] {
        assert!(
            (got - want).abs() <= want.abs() * 4e-4,
            "rcpps {got} vs {want}"
        );
    }

    // rsqrtps xmm0, xmm1; ret  ->  0F 52 C1  C3
    let bytes = [0x0F, 0x52, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_f32(1.0, 4.0), pack_f32(16.0, 64.0)];
    let ctx = run_test(&bytes, ctx);
    let got = [unpack_f32(ctx.xmm[0][0]), unpack_f32(ctx.xmm[0][1])];
    for (got, want) in [
        (got[0].0, 1.0f32),
        (got[0].1, 0.5),
        (got[1].0, 0.25),
        (got[1].1, 0.125),
    ] {
        assert!((got - want).abs() <= want * 4e-4, "rsqrtps {got} vs {want}");
    }
}

#[test]
fn test_rcpss_scalar_preserves_high_lanes() {
    // rcpss xmm0, xmm1; ret  ->  F3 0F 53 C1  C3
    let bytes = [0xF3, 0x0F, 0x53, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(99.0, 7.0), 0x1111_2222_3333_4444];
    ctx.xmm[1] = [pack_f32(4.0, 0.0), 0];
    let ctx = run_test(&bytes, ctx);
    let (l0, l1) = unpack_f32(ctx.xmm[0][0]);
    assert!((l0 - 0.25).abs() <= 1e-4, "rcpss lane 0 = {l0}");
    assert_eq!(l1, 7.0, "lane 1 preserved");
    assert_eq!(ctx.xmm[0][1], 0x1111_2222_3333_4444);
}

//  MOVSLDUP / MOVSHDUP / MOVNT* stores

#[test]
fn test_movsldup_movshdup() {
    // movsldup xmm0, xmm1; ret  ->  F3 0F 12 C1  C3
    let bytes = [0xF3, 0x0F, 0x12, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_i32(0, 1), pack_i32(2, 3)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (0, 0));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (2, 2));

    // movshdup xmm0, xmm1; ret  ->  F3 0F 16 C1  C3
    let bytes = [0xF3, 0x0F, 0x16, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_i32(0, 1), pack_i32(2, 3)];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (1, 1));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (3, 3));
}

#[test]
fn test_movntdq_store() {
    // movntdq [rdi], xmm0; ret  ->  66 0F E7 07  C3
    let bytes = [0x66, 0x0F, 0xE7, 0x07, 0xC3];
    let mut buf = [0u64; 2];
    let mut ctx = RegisterContext::default();
    ctx.rdi = buf.as_mut_ptr() as u64;
    ctx.xmm[0] = [0x1234_5678_9ABC_DEF0, 0xFEED_FACE_CAFE_F00D];
    let _ = run_test(&bytes, ctx);
    assert_eq!(buf, [0x1234_5678_9ABC_DEF0, 0xFEED_FACE_CAFE_F00D]);
}

/// The new packed conversions and shuffles under the vector XMM representation,
/// where the lo/hi accessors take a different path.
#[test]
fn test_conversions_and_shuffles_vector_xmm() {
    // cvtpd2ps xmm0, xmm1
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.xmm[1] = [1.5f64.to_bits(), (-2.25f64).to_bits()];
    let ctx = run_test_vec(&[0x66, 0x0F, 0x5A, 0xC1, 0xC3], ctx);
    assert_eq!(unpack_f32(ctx.xmm[0][0]), (1.5, -2.25));
    assert_eq!(ctx.xmm[0][1], 0);

    // cvttps2dq xmm0, xmm1
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [pack_f32(1.9, -1.9), pack_f32(3.0, -4.0)];
    let ctx = run_test_vec(&[0xF3, 0x0F, 0x5B, 0xC1, 0xC3], ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (1, -1));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (3, -4));

    // shufps xmm0, xmm1, 0x1B
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_i32(0, 1), pack_i32(2, 3)];
    ctx.xmm[1] = [pack_i32(10, 11), pack_i32(12, 13)];
    let ctx = run_test_vec(&[0x0F, 0xC6, 0xC1, 0x1B, 0xC3], ctx);
    assert_eq!(unpack_i32(ctx.xmm[0][0]), (3, 2));
    assert_eq!(unpack_i32(ctx.xmm[0][1]), (11, 10));

    // cmpps xmm0, xmm1, 1 (LT)
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [pack_f32(1.0, 5.0), pack_f32(1.0, 5.0)];
    ctx.xmm[1] = [pack_f32(2.0, 5.0), pack_f32(2.0, 5.0)];
    let ctx = run_test_vec(&[0x0F, 0xC2, 0xC1, 0x01, 0xC3], ctx);
    assert_eq!(ctx.xmm[0], [T32, T32]);
}

//  SSSE3 / SSE4.1 packed integer ops
//
//  These pin down concrete results (and the vector-XMM representation); the
//  exhaustive hardware cross-check lives in `tests/unicorn_oracle_tests.rs`.

/// Bytes 0x00..0x0F laid out lane 0 first.
const IDENTITY_BYTES: [u64; 2] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];

#[test]
fn test_pshufb_selects_zeroes_and_masks() {
    // pshufb xmm0, xmm1 ; ret  ->  66 0F 38 00 C1  C3
    let bytes = [0x66, 0x0F, 0x38, 0x00, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = IDENTITY_BYTES;
    // Control bytes, lane 0 first: 0x0F..0x08 reverse the low half from the
    // destination's high bytes; 0x80 zeroes; 0x31 must be masked to index 1.
    ctx.xmm[1] = [0x0809_0A0B_0C0D_0E0F, 0x3180_0000_0000_0007];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x0809_0A0B_0C0D_0E0F, "reversed high bytes");
    assert_eq!(
        ctx.xmm[0][1], 0x0100_0000_0000_0007,
        "bit 7 zeroes the byte; index 0x31 masks to 1"
    );
}

#[test]
fn test_pmuludq_multiplies_even_dwords() {
    // pmuludq xmm0, xmm1 ; ret  ->  66 0F F4 C1  C3
    let bytes = [0x66, 0x0F, 0xF4, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    // Odd dwords (the high halves) must be ignored entirely.
    ctx.xmm[0] = [0xFFFF_FFFF_0000_0003, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.xmm[1] = [0xFFFF_FFFF_0000_0007, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 21);
    // 0xFFFF_FFFF * 0xFFFF_FFFF, unsigned - the signed PMULDQ would give 1.
    assert_eq!(ctx.xmm[0][1], 0xFFFF_FFFE_0000_0001);
}

#[test]
fn test_pmuldq_multiplies_signed() {
    // pmuldq xmm0, xmm1 ; ret  ->  66 0F 38 28 C1  C3
    let bytes = [0x66, 0x0F, 0x38, 0x28, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0xFFFF_FFFF, 0x0000_0000_FFFF_FFFE];
    ctx.xmm[1] = [0xFFFF_FFFF, 0x0000_0000_0000_0003];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 1, "-1 * -1");
    assert_eq!(ctx.xmm[0][1], (-6i64) as u64, "-2 * 3");
}

#[test]
fn test_palignr_shifts_across_the_pair() {
    // palignr xmm0, xmm1, imm8 ; ret  ->  66 0F 3A 0F C1 ib  C3
    let run = |imm: u8| {
        let bytes = [0x66, 0x0F, 0x3A, 0x0F, 0xC1, imm, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918];
        ctx.xmm[1] = IDENTITY_BYTES;
        run_test(&bytes, ctx).xmm[0]
    };
    assert_eq!(run(0), IDENTITY_BYTES, "count 0 is the source unchanged");
    assert_eq!(run(4), [0x0B0A_0908_0706_0504, 0x1312_1110_0F0E_0D0C]);
    assert_eq!(
        run(16),
        [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918],
        "count 16 is the destination unchanged"
    );
    assert_eq!(
        run(20),
        [0x1B1A_1918_1716_1514, 0x0000_0000_1F1E_1D1C],
        "the destination shifts down and the top bytes zero-fill"
    );
    assert_eq!(run(32), [0, 0], "counts past 31 clear the register");
}

#[test]
fn test_packsswb_and_packuswb_saturate() {
    // The destination's eight words, lane 0 first: 0, -128, 127, 128,
    // -1, 1, 32767, -32768. The source is zero, so the result's high half is
    // whatever those words saturate zero to.
    let dst = [0x0080_007F_FF80_0000u64, 0x8000_7FFF_0001_FFFF];

    // packsswb xmm0, xmm1 ; ret  ->  66 0F 63 C1  C3  (signed, clamps to i8)
    let bytes = [0x66, 0x0F, 0x63, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = dst;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x807F_01FF_7F7F_8000);
    assert_eq!(ctx.xmm[0][1], 0, "the zero source packs to zero bytes");

    // packuswb xmm0, xmm1 ; ret  ->  66 0F 67 C1  C3  (clamps to u8)
    let bytes = [0x66, 0x0F, 0x67, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = dst;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0][0], 0x00FF_0100_807F_0000, "negatives clamp to 0");
    assert_eq!(ctx.xmm[0][1], 0);
}

#[test]
fn test_pmovzx_and_pmovsx_widen() {
    // pmovzxbw xmm0, xmm1 ; ret  ->  66 0F 38 30 C1  C3
    let bytes = [0x66, 0x0F, 0x38, 0x30, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x8070_6050_4030_2010, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x0040_0030_0020_0010, 0x0080_0070_0060_0050]);

    // pmovsxbw xmm0, xmm1 ; ret  ->  66 0F 38 20 C1  C3
    let bytes = [0x66, 0x0F, 0x38, 0x20, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x8070_6050_4030_2010, 0xFFFF_FFFF_FFFF_FFFF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [0x0040_0030_0020_0010, 0xFF80_0070_0060_0050]);

    // pmovsxdq xmm0, xmm1 ; ret  ->  66 0F 38 25 C1  C3  (only the low 8 bytes)
    let bytes = [0x66, 0x0F, 0x38, 0x25, 0xC1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[1] = [0x0000_0007_FFFF_FFFE, 0xDEAD_BEEF_DEAD_BEEF];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[0], [(-2i64) as u64, 7]);
}

#[test]
fn test_ptest_sets_zf_and_cf() {
    // ptest xmm0, xmm1 ; ret  ->  66 0F 38 17 C1  C3
    let bytes = [0x66, 0x0F, 0x38, 0x17, 0xC1, 0xC3];
    let cases = [
        // (dst, src, zf, cf)
        ([0u64; 2], [0u64; 2], 1, 1),
        ([0xFF, 0], [0x0F, 0], 0, 1),
        ([0xFF, 0], [0x0F00, 0], 1, 0),
        ([0xFF, 0], [0x0FF0, 0], 0, 0),
        ([0, 0xFF], [0, 0x0F], 0, 1),
    ];
    for (dst, src, zf, cf) in cases {
        let mut ctx = RegisterContext::default();
        ctx.xmm[0] = dst;
        ctx.xmm[1] = src;
        // Seed the flags PTEST must clear.
        ctx.sf = 1;
        ctx.of = 1;
        ctx.pf = 1;
        ctx.af = 1;
        let ctx = run_test(&bytes, ctx);
        assert_eq!(ctx.zf, zf, "ZF for {dst:#x?} & {src:#x?}");
        assert_eq!(ctx.cf, cf, "CF for {dst:#x?} & ~{src:#x?}");
        assert_eq!((ctx.sf, ctx.of, ctx.pf, ctx.af), (0, 0, 0, 0), "cleared");
        assert_eq!(ctx.xmm[0], dst, "ptest must not write its operand");
    }
}

#[test]
fn test_lane_extract_to_gpr() {
    // pextrw eax, xmm1, 5 ; ret  ->  66 0F C5 C1 05  C3
    let bytes = [0x66, 0x0F, 0xC5, 0xC1, 0x05, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_DEAD_BEEF;
    ctx.xmm[1] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x7777, "zero-extends into the full GPR");

    // pextrq rcx, xmm0, 1 ; ret  ->  66 48 0F 3A 16 C1 01  C3
    // (the SSE4.1 extract forms put the GPR destination in the r/m field)
    let bytes = [0x66, 0x48, 0x0F, 0x3A, 0x16, 0xC1, 0x01, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rcx, 0x5555_6666_7777_8888);
}

#[test]
fn test_pblendvb_uses_xmm0_mask() {
    // pblendvb xmm2, xmm1 ; ret  ->  66 0F 38 10 D1  C3   (mask is xmm0)
    let bytes = [0x66, 0x0F, 0x38, 0x10, 0xD1, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.xmm[0] = [0x8000_0080_0000_0080, 0x0000_0000_0000_0000];
    ctx.xmm[1] = [0xAAAA_AAAA_AAAA_AAAA, 0xAAAA_AAAA_AAAA_AAAA];
    ctx.xmm[2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.xmm[2][0], 0xAA11_11AA_1111_11AA);
    assert_eq!(ctx.xmm[2][1], 0x2222_2222_2222_2222, "zero mask keeps dst");
}

/// The scalar lo/hi and native-vector XMM representations must agree on every
/// new op - the lane plumbing is shared, but the reads and writes are not.
#[test]
fn test_new_packed_ops_match_in_vector_mode() {
    let progs: &[(&str, &[u8])] = &[
        ("pshufb", &[0x66, 0x0F, 0x38, 0x00, 0xC1, 0xC3]),
        ("phaddw", &[0x66, 0x0F, 0x38, 0x01, 0xC1, 0xC3]),
        ("phsubd", &[0x66, 0x0F, 0x38, 0x06, 0xC1, 0xC3]),
        ("psignb", &[0x66, 0x0F, 0x38, 0x08, 0xC1, 0xC3]),
        ("pmulhrsw", &[0x66, 0x0F, 0x38, 0x0B, 0xC1, 0xC3]),
        ("pabsw", &[0x66, 0x0F, 0x38, 0x1D, 0xC1, 0xC3]),
        ("pmaddubsw", &[0x66, 0x0F, 0x38, 0x04, 0xC1, 0xC3]),
        ("pmuludq", &[0x66, 0x0F, 0xF4, 0xC1, 0xC3]),
        ("pmuldq", &[0x66, 0x0F, 0x38, 0x28, 0xC1, 0xC3]),
        ("pmulld", &[0x66, 0x0F, 0x38, 0x40, 0xC1, 0xC3]),
        ("pmaddwd", &[0x66, 0x0F, 0xF5, 0xC1, 0xC3]),
        ("psadbw", &[0x66, 0x0F, 0xF6, 0xC1, 0xC3]),
        ("packsswb", &[0x66, 0x0F, 0x63, 0xC1, 0xC3]),
        ("packusdw", &[0x66, 0x0F, 0x38, 0x2B, 0xC1, 0xC3]),
        ("pmovzxbw", &[0x66, 0x0F, 0x38, 0x30, 0xC1, 0xC3]),
        ("pmovsxwq", &[0x66, 0x0F, 0x38, 0x24, 0xC1, 0xC3]),
        ("pminsb", &[0x66, 0x0F, 0x38, 0x38, 0xC1, 0xC3]),
        ("pmaxud", &[0x66, 0x0F, 0x38, 0x3F, 0xC1, 0xC3]),
        ("palignr", &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x05, 0xC3]),
        ("pblendw", &[0x66, 0x0F, 0x3A, 0x0E, 0xC1, 0x5A, 0xC3]),
        ("pblendvb", &[0x66, 0x0F, 0x38, 0x10, 0xD1, 0xC3]),
        ("pinsrd", &[0x66, 0x0F, 0x3A, 0x22, 0xC1, 0x02, 0xC3]),
        ("pextrb", &[0x66, 0x0F, 0x3A, 0x14, 0xC1, 0x0B, 0xC3]),
        ("ptest", &[0x66, 0x0F, 0x38, 0x17, 0xC1, 0xC3]),
    ];
    for (name, bytes) in progs {
        let mut ctx = RegisterContext::default();
        ctx.rcx = 0x1234_5678_9ABC_DEF0;
        ctx.xmm[0] = [0x8F0E_0D0C_0B0A_0908, 0x0706_0504_0302_0100];
        ctx.xmm[1] = [0xDEAD_BEEF_CAFE_BABE, 0x1122_3344_5566_7788];
        ctx.xmm[2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        let scalar = run_test(bytes, ctx);

        let mut ctx = RegisterContext::default();
        ctx.rcx = 0x1234_5678_9ABC_DEF0;
        ctx.xmm[0] = [0x8F0E_0D0C_0B0A_0908, 0x0706_0504_0302_0100];
        ctx.xmm[1] = [0xDEAD_BEEF_CAFE_BABE, 0x1122_3344_5566_7788];
        ctx.xmm[2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
        let vector = run_test_vec(bytes, ctx);

        assert_eq!(scalar.xmm, vector.xmm, "{name}: xmm differs across modes");
        assert_eq!(
            (scalar.rax, scalar.rcx),
            (vector.rax, vector.rcx),
            "{name}: GPR result differs across modes"
        );
        assert_eq!(
            (scalar.zf, scalar.cf),
            (vector.zf, vector.cf),
            "{name}: flags differ across modes"
        );
    }
}

/// The MMX encodings of the SSE integer opcodes (same mnemonic, no 0x66
/// prefix) operate on `MM0..MM7`, which this frontend does not model. They must
/// exit to the host as unimplemented - a resumable, shimmable exit - rather
/// than failing the lift of the whole region.
#[test]
fn test_mmx_encodings_exit_to_host() {
    // pshufb mm0, mm1 ; ret  ->  0F 38 00 C1  C3
    let out = LiftTest::new(&[0x0F, 0x38, 0x00, 0xC1, 0xC3])
        .try_run()
        .expect("mmx forms must lift, not error");
    assert_eq!(out.exit, ExitReason::UnimplementedInstruction);

    // pmuludq mm0, mm1 ; ret  ->  0F F4 C1  C3
    let out = LiftTest::new(&[0x0F, 0xF4, 0xC1, 0xC3])
        .try_run()
        .expect("mmx forms must lift, not error");
    assert_eq!(out.exit, ExitReason::UnimplementedInstruction);
}
