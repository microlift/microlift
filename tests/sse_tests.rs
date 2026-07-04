//! SSE / SSE2 instruction tests.
//!
//! Each test lifts a small x86-64 snippet and verifies the resulting
//! `RegisterContext` has the expected XMM (and where relevant GPR/memory)
//! state after JIT execution.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::{Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    run_test_with_config(bytes, ctx, LifterConfig::default())
}

fn run_test_vec(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    let mut cfg = LifterConfig::default();
    cfg.use_vector_xmm = true;
    run_test_with_config(bytes, ctx, cfg)
}

fn run_test_with_config(
    bytes: &[u8],
    mut ctx: RegisterContext,
    config: LifterConfig,
) -> RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);

    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
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
    assert_eq!(ctx.xmm[0], [0xDEAD_BEEF, 0], "movd must load low dword and zero the rest");
}

#[test]
fn test_movd_xmm_to_gpr() {
    // movd eax, xmm0; ret  ->  66 0F 7E C0  C3
    let bytes = [0x66, 0x0F, 0x7E, 0xC0, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xAAAA_AAAA_AAAA_AAAA;
    ctx.xmm[0] = [0x1122_3344_5566_7788, 0x9999_9999_9999_9999];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x5566_7788, "movd r32,xmm extracts low dword and zero-extends");
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
    assert_eq!(buf[0], 0xFEED_FACE_CAFE_F00D, "movhps store must write the high quadword");
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
    assert_eq!(ctx.xmm[0], [0, 0], "psrlq by >=64 (register) must zero both lanes");
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
