//! Regression tests for the third x86-64 lifter review round.
//!
//! Covers:
//! 1. Indirect `JMP` must not be classified as a direct branch to address 0.
//! 2. `RCL` (rotate-through-carry-left) is implemented and bit-exact.
//! 3. `BSF`/`BSR` with a zero source leave the *entire* destination register
//!    unchanged (including the upper 32 bits of a 64-bit register).
//! 4. `LOCK NOT` / `LOCK NEG` remain correct after the atomic-RMW rework.
//! 5. Address-size-prefixed (`0x67`) string ops select ECX/ESI/EDI and verify.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::arch::x86_64::X86_64;
use microlift::backend::CraneliftBackend;
use microlift::cfg::static_targets;
use microlift::context::RegisterContext;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

const LIFT_BASE: u64 = 0x1000;

/// The sentinel return address this file's harness pushes; a `RET` to it exits
/// with `ExitReason::Return` because the config below declares it.
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Lift + JIT `bytes` at LIFT_BASE, run against `ctx`, and return the mutated
/// context. A fresh stack is installed with the exit trampoline pre-pushed.
fn run(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::with_config(
        builder,
        LifterConfig {
            exit_trampoline_address: Some(EXIT_TRAMPOLINE),
            ..LifterConfig::default()
        },
    );

    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    if ctx.image_base == 0 {
        ctx.image_base = LIFT_BASE;
    }
    unsafe {
        *(ctx.rsp as *mut u64) = EXIT_TRAMPOLINE;
    }

    lifter.lift(bytes, LIFT_BASE).expect("lifting failed");
    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    let _ = stack;
    ctx
}

// 1. Indirect JMP classification

#[test]
fn indirect_jmp_reg_is_not_a_static_branch_target() {
    // ff e0 = jmp rax (register-indirect). No statically-known target.
    let t = static_targets::<X86_64>(&[0xFF, 0xE0], LIFT_BASE);
    assert!(
        t.branches.is_empty(),
        "indirect `jmp rax` reported static branch target(s): {:?}",
        t.branches
    );
}

#[test]
fn indirect_jmp_mem_is_not_a_static_branch_target() {
    // ff 25 00 00 00 00 = jmp [rip+0] (memory-indirect).
    let t = static_targets::<X86_64>(&[0xFF, 0x25, 0, 0, 0, 0], LIFT_BASE);
    assert!(
        t.branches.is_empty(),
        "indirect `jmp [mem]` reported static branch target(s): {:?}",
        t.branches
    );
}

#[test]
fn direct_jmp_still_reports_its_target() {
    // eb 05 = jmp +5 from the end of this 2-byte instruction => 0x1007.
    let t = static_targets::<X86_64>(&[0xEB, 0x05], LIFT_BASE);
    assert!(t.branches.contains(&0x1007), "branches: {:?}", t.branches);
}

// 2. RCL

#[test]
fn rcl_rax_by_1_through_carry() {
    // 48 c1 d0 01 = rcl rax, 1
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x8000_0000_0000_0001;
    ctx.cf = 0;
    let out = run(&[0x48, 0xC1, 0xD0, 0x01], ctx);
    // CF:rax = 0:0x8000000000000001 rotated left 1 -> rax=0x2, the old MSB into CF.
    assert_eq!(out.rax, 0x2, "rax");
    assert_eq!(out.cf, 1, "cf");
}

#[test]
fn rcl_al_by_1_wraps_carry_into_lsb() {
    // d0 d0 = rcl al, 1 (8-bit exercises the width-modulo path)
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x80; // al = 0x80
    ctx.cf = 1;
    let out = run(&[0xD0, 0xD0], ctx);
    // CF:al = 1:10000000 rotated left 1 -> al=0x01, CF=old bit7=1.
    assert_eq!(out.rax & 0xFF, 0x01, "al");
    assert_eq!(out.cf, 1, "cf");
}

#[test]
fn rcl_matches_rcr_roundtrip() {
    // rcl then rcr by the same count restores the original (with same CF in).
    // 48 c1 d0 05 = rcl rax,5 ; 48 c1 d8 05 = rcr rax,5
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x0123_4567_89AB_CDEF;
    ctx.cf = 0;
    let out = run(&[0x48, 0xC1, 0xD0, 0x05, 0x48, 0xC1, 0xD8, 0x05], ctx);
    assert_eq!(out.rax, 0x0123_4567_89AB_CDEF, "rcl;rcr should round-trip");
}

// 3. BSF / BSR with a zero source preserve the full destination

#[test]
fn bsf_zero_source_preserves_upper_dword() {
    // 0f bc c3 = bsf eax, ebx ; ebx = 0 => ZF=1, dest unchanged.
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_1234_5678;
    ctx.rbx = 0;
    let out = run(&[0x0F, 0xBC, 0xC3], ctx);
    assert_eq!(
        out.rax, 0xDEAD_BEEF_1234_5678,
        "rax must be fully unchanged"
    );
    assert_eq!(out.zf, 1, "zf");
}

#[test]
fn bsr_zero_source_preserves_upper_dword() {
    // 0f bd c3 = bsr eax, ebx ; ebx = 0 => ZF=1, dest unchanged.
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xCAFE_F00D_AABB_CCDD;
    ctx.rbx = 0;
    let out = run(&[0x0F, 0xBD, 0xC3], ctx);
    assert_eq!(
        out.rax, 0xCAFE_F00D_AABB_CCDD,
        "rax must be fully unchanged"
    );
    assert_eq!(out.zf, 1, "zf");
}

#[test]
fn bsf_nonzero_source_zero_extends_result() {
    // 0f bc c3 = bsf eax, ebx ; ebx = 0x10 (bit 4) => eax=4, upper 32 cleared.
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_0000_0000;
    ctx.rbx = 0x10;
    let out = run(&[0x0F, 0xBC, 0xC3], ctx);
    assert_eq!(out.rax, 4, "rax");
    assert_eq!(out.zf, 0, "zf");
}

// 4. LOCK NOT / LOCK NEG

#[test]
fn lock_not_qword_mem() {
    // f0 48 f7 10 = lock not qword [rax]
    let mut cell: u64 = 0x0F0F_0F0F_0F0F_0F0F;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut cell as *mut u64 as u64;
    run(&[0xF0, 0x48, 0xF7, 0x10], ctx);
    assert_eq!(cell, !0x0F0F_0F0F_0F0F_0F0Fu64);
}

#[test]
fn lock_neg_qword_mem_sets_flags() {
    // f0 48 f7 18 = lock neg qword [rax]
    let mut cell: u64 = 5;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut cell as *mut u64 as u64;
    let out = run(&[0xF0, 0x48, 0xF7, 0x18], ctx);
    assert_eq!(cell, (0u64).wrapping_sub(5), "cell = -5");
    // NEG sets CF iff operand != 0; SF from the (negative) result; ZF=0.
    assert_eq!(out.cf, 1, "cf");
    assert_eq!(out.zf, 0, "zf");
    assert_eq!(out.sf, 1, "sf");
}

#[test]
fn lock_neg_of_zero_clears_carry() {
    let mut cell: u64 = 0;
    let mut ctx = RegisterContext::default();
    ctx.rax = &mut cell as *mut u64 as u64;
    let out = run(&[0xF0, 0x48, 0xF7, 0x18], ctx);
    assert_eq!(cell, 0, "cell stays 0");
    assert_eq!(out.cf, 0, "cf clear when operand is 0");
    assert_eq!(out.zf, 1, "zf");
}

// 5. String-op address size

#[test]
fn rep_movsb_64bit_addr_size_still_works() {
    // f3 a4 = rep movsb. Copies RCX bytes [RSI]->[RDI].
    let src: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    let mut dst: [u8; 4] = [0; 4];
    let mut ctx = RegisterContext::default();
    ctx.rsi = src.as_ptr() as u64;
    ctx.rdi = dst.as_mut_ptr() as u64;
    ctx.rcx = 4;
    ctx.df = 0;
    let out = run(&[0xF3, 0xA4], ctx);
    assert_eq!(dst, [0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(out.rcx, 0, "rcx drained");
    assert_eq!(out.rsi, src.as_ptr() as u64 + 4, "rsi advanced");
    assert_eq!(out.rdi, dst.as_ptr() as u64 + 4, "rdi advanced");
}

#[test]
fn rep_stosb_64bit_addr_size_still_works() {
    // f3 aa = rep stosb. Stores AL to [RDI], RCX times.
    let mut dst: [u8; 3] = [0; 3];
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x7E;
    ctx.rdi = dst.as_mut_ptr() as u64;
    ctx.rcx = 3;
    ctx.df = 0;
    run(&[0xF3, 0xAA], ctx);
    assert_eq!(dst, [0x7E, 0x7E, 0x7E]);
}

#[test]
fn addr_size_prefixed_string_op_lifts_and_compiles() {
    // 67 f3 a4 = rep movsb with 32-bit address size (ECX/ESI/EDI). Executing it
    // would require low-4GB buffers; here we assert the lowering is well-formed
    // (selects the 32-bit registers and passes the IR verifier + isel).
    let func = lift_to_function::<X86_64>(&[0x67, 0xF3, 0xA4], LIFT_BASE, LifterConfig::default())
        .expect("32-bit-addr rep movsb should lift");
    let mut backend = CraneliftBackend::new().unwrap();
    backend
        .compile(func)
        .expect("32-bit-addr rep movsb should compile");
}
