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
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

fn run_test(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::new(builder);

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
