use microlift::arch::x86_64::context::X86_64Callbacks;
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn test_mfence() {
    let bytes = [0x0F, 0xAE, 0xF0, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_sfence() {
    let bytes = [0x0F, 0xAE, 0xF8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_lfence() {
    let bytes = [0x0F, 0xAE, 0xE8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_pause() {
    let bytes = [0xF3, 0x90, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_cmpxchg_reg_equal() {
    let bytes = [
        0x48, 0xC7, 0xC0, 10, 0, 0, 0, // mov rax, 10
        0x48, 0xC7, 0xC1, 10, 0, 0, 0, // mov rcx, 10
        0x48, 0xC7, 0xC3, 99, 0, 0, 0, // mov rbx, 99
        0x48, 0x0F, 0xB1, 0xD9, // cmpxchg rcx, rbx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rcx, 99, "rcx should be updated to rbx (99) when equal");
    assert_eq!(ctx.zf, 1, "ZF=1 when equal");
}

#[test]
fn test_cmpxchg_reg_not_equal() {
    let bytes = [
        0x48, 0xC7, 0xC0, 5, 0, 0, 0, // mov rax, 5
        0x48, 0xC7, 0xC1, 10, 0, 0, 0, // mov rcx, 10
        0x48, 0xC7, 0xC3, 99, 0, 0, 0, // mov rbx, 99
        0x48, 0x0F, 0xB1, 0xD9, // cmpxchg rcx, rbx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 10,
        "rax should be loaded with old rcx (10) when not equal"
    );
    assert_eq!(ctx.rcx, 10, "rcx unchanged when not equal");
    assert_eq!(ctx.zf, 0, "ZF=0 when not equal");
}

#[test]
fn test_cmpxchg_mem_equal() {
    let mut mem_val: u64 = 42;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC0, 42, 0, 0, 0, // mov rax, 42
        0x48, 0xC7, 0xC3, 99, 0, 0, 0, // mov rbx, 99
        0x48, 0x0F, 0xB1, 0x1F, // cmpxchg [rdi], rbx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 99, "[rdi] should be updated to 99");
    assert_eq!(ctx.zf, 1, "ZF=1 on success");
}

#[test]
fn test_cmpxchg_mem_not_equal() {
    let mut mem_val: u64 = 7;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC0, 42, 0, 0, 0, // mov rax, 42
        0x48, 0xC7, 0xC3, 99, 0, 0, 0, // mov rbx, 99
        0x48, 0x0F, 0xB1, 0x1F, // cmpxchg [rdi], rbx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 7, "[rdi] should be unchanged");
    assert_eq!(ctx.rax, 7, "rax should be loaded with old [rdi]");
    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
}

#[test]
fn test_cmpxchg8b_equal() {
    let mut mem_val: u64 = (2u64 << 32) | 1u64;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 3;
    ctx.rcx = 4;

    let bytes = [
        0x0F, 0xC7, 0x0F, // cmpxchg8b [rdi]
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, (4u64 << 32) | 3u64, "[rdi] should be updated");
    assert_eq!(ctx.zf, 1, "ZF=1 on success");
}

#[test]
fn test_cmpxchg8b_not_equal() {
    let mut mem_val: u64 = (0x99u64 << 32) | 0x88u64;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 3;
    ctx.rcx = 4;

    let bytes = [
        0x0F, 0xC7, 0x0F, // cmpxchg8b [rdi]
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        mem_val,
        (0x99u64 << 32) | 0x88u64,
        "[rdi] should be unchanged"
    );
    assert_eq!(ctx.rax, 0x88, "eax should be low 32 bits of old [rdi]");
    assert_eq!(ctx.rdx, 0x99, "edx should be high 32 bits of old [rdi]");
    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
}

#[test]
fn test_lock_add_mem() {
    let mut mem_val: u64 = 10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC6, 5, 0, 0, 0, // mov rsi, 5
        0xF0, 0x48, 0x01, 0x37, // lock add [rdi], rsi
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 15);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_lock_add_mem_zero_result() {
    let mut mem_val: u64 = u64::MAX;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC6, 1, 0, 0, 0, // mov rsi, 1
        0xF0, 0x48, 0x01, 0x37, // lock add [rdi], rsi
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0);
    assert_eq!(ctx.zf, 1);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_lock_xadd() {
    let mut mem_val: u64 = 100;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC6, 5, 0, 0, 0, // mov rsi, 5
        0xF0, 0x48, 0x0F, 0xC1, 0x37, // lock xadd [rdi], rsi
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 105, "[rdi] should be old + src");
    assert_eq!(ctx.rsi, 100, "rsi should hold old [rdi]");
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_lock_xchg() {
    let mut mem_val: u64 = 77;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0x48, 0xC7, 0xC6, 99, 0, 0, 0, // mov rsi, 99
        0xF0, 0x48, 0x87, 0x37, // lock xchg [rdi], rsi
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 99, "[rdi] should become rsi");
    assert_eq!(ctx.rsi, 77, "rsi should become old [rdi]");
}

#[test]
fn test_lock_sub() {
    let mut mem_val: u64 = 20;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0xF0, 0x48, 0x83, 0x2F, 7, // lock sub qword [rdi], 7
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 13);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_lock_and() {
    let mut mem_val: u64 = 0b1111;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0xF0, 0x48, 0x83, 0x27, 0b1010, // lock and qword [rdi], 0b1010
        0xC3,
    ];
    run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b1010);
}

#[test]
fn test_lock_or() {
    let mut mem_val: u64 = 0b1010;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0xF0, 0x48, 0x83, 0x0F, 0b0101, // lock or qword [rdi], 5
        0xC3,
    ];
    run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b1111);
}

#[test]
fn test_lock_xor() {
    let mut mem_val: u64 = 0xFF;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [
        0xF0, 0x48, 0x83, 0x37, 0x0F, // lock xor qword [rdi], 0x0F
        0xC3,
    ];
    run_test(&bytes, ctx);
    assert_eq!(mem_val, 0xF0);
}

#[test]
fn test_lock_inc() {
    let mut mem_val: u64 = 41;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [0xF0, 0x48, 0xFF, 0x07, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 42);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_lock_dec() {
    let mut mem_val: u64 = 1;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [0xF0, 0x48, 0xFF, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0);
    assert_eq!(ctx.zf, 1);
}

#[test]
fn test_lock_bts() {
    let mut mem_val: u64 = 0b0101;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rcx = 1;

    let bytes = [0xF0, 0x48, 0x0F, 0xAB, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b0111, "bit 1 should be set");
    assert_eq!(ctx.cf, 0, "CF=0: bit was originally 0");
}

#[test]
fn test_lock_bts_already_set() {
    let mut mem_val: u64 = 0b0111;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rcx = 1;

    let bytes = [0xF0, 0x48, 0x0F, 0xAB, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b0111, "mem unchanged when bit already set");
    assert_eq!(ctx.cf, 1, "CF=1: bit was originally set");
}

#[test]
fn test_lock_btr() {
    let mut mem_val: u64 = 0b0111;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rcx = 1;

    let bytes = [0xF0, 0x48, 0x0F, 0xB3, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b0101, "bit 1 should be cleared");
    assert_eq!(ctx.cf, 1, "CF=1: bit was originally set");
}

#[test]
fn test_lock_btc() {
    let mut mem_val: u64 = 0b0101;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rcx = 2;

    let bytes = [0xF0, 0x48, 0x0F, 0xBB, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b0001, "bit 2 should be flipped");
    assert_eq!(ctx.cf, 1, "CF=1: bit was originally set");
}

/// 16-byte-aligned pair of qwords; hardware faults #GP on a misaligned CMPXCHG16B.
#[repr(C, align(16))]
struct Aligned128 {
    lo: u64,
    hi: u64,
}

#[test]
fn test_cmpxchg16b_success() {
    let mut mem = Aligned128 { lo: 1, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 0xAAAA_BBBB_CCCC_DDDD;
    ctx.rcx = 0x1111_2222_3333_4444;

    let bytes = [0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(mem.lo, 0xAAAA_BBBB_CCCC_DDDD, "low half takes RBX");
    assert_eq!(mem.hi, 0x1111_2222_3333_4444, "high half takes RCX");
    assert_eq!(ctx.zf, 1, "ZF=1 on success");
    assert_eq!(ctx.rax, 1, "RAX unchanged on success");
    assert_eq!(ctx.rdx, 2, "RDX unchanged on success");
}

/// A high-half-only mismatch must fail the compare.
#[test]
fn test_cmpxchg16b_failure_high_half() {
    let mut mem = Aligned128 { lo: 1, hi: 0xDEAD };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 99;
    ctx.rcx = 99;

    let bytes = [0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(mem.lo, 1, "memory unchanged on failure");
    assert_eq!(mem.hi, 0xDEAD, "memory unchanged on failure");
    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
    assert_eq!(ctx.rax, 1, "RAX loaded with the observed low half");
    assert_eq!(ctx.rdx, 0xDEAD, "RDX loaded with the observed high half");
}

/// A low-half-only mismatch must fail too.
#[test]
fn test_cmpxchg16b_failure_low_half() {
    let mut mem = Aligned128 { lo: 0xBEEF, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 99;
    ctx.rcx = 99;

    let bytes = [0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(mem.lo, 0xBEEF, "memory unchanged on failure");
    assert_eq!(mem.hi, 2, "memory unchanged on failure");
    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
    assert_eq!(ctx.rax, 0xBEEF);
    assert_eq!(ctx.rdx, 2);
}

#[test]
fn test_cmpxchg16b_without_lock_prefix() {
    let mut mem = Aligned128 { lo: 5, hi: 6 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 5;
    ctx.rdx = 6;
    ctx.rbx = 7;
    ctx.rcx = 8;

    let bytes = [0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!((mem.lo, mem.hi), (7, 8));
    assert_eq!(ctx.zf, 1);
}

/// CMPXCHG16B touches ZF only (unlike CMPXCHG, which updates all flags).
#[test]
fn test_cmpxchg16b_leaves_other_flags_alone() {
    let mut mem = Aligned128 { lo: 0, hi: 0 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1; // != mem.lo, so the CAS fails
    ctx.rdx = 0;

    let bytes = [0xF9, 0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
    assert_eq!(ctx.cf, 1, "CF must be untouched by CMPXCHG16B");
}

/// `Native` needs an x64 host; no other backend lowers `atomic_cas.i128`.
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn test_cmpxchg16b_native_mode() {
    let mut mem = Aligned128 { lo: 1, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 30;
    ctx.rcx = 40;

    let bytes = [0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let out = LiftTest::new(&bytes)
        .ctx(ctx)
        .atomic128(Atomic128Mode::Native)
        .run();

    assert_eq!((mem.lo, mem.hi), (30, 40));
    assert_eq!(out.ctx.zf, 1);

    let mut mem = Aligned128 { lo: 11, hi: 22 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    let out = LiftTest::new(&bytes)
        .ctx(ctx)
        .atomic128(Atomic128Mode::Native)
        .run();
    assert_eq!((mem.lo, mem.hi), (11, 22), "memory unchanged");
    assert_eq!(out.ctx.zf, 0);
    assert_eq!((out.ctx.rax, out.ctx.rdx), (11, 22));
}

/// `Interpret` hands the instruction to the interpreter callback; the region resumes at the `RET`.
static EXIT_MODE_RIP: AtomicU64 = AtomicU64::new(0);
static EXIT_MODE_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn interpret_cmpxchg16b(ctx: *mut RegisterContext, num_instructions: u64) {
    let ctx = unsafe { &mut *ctx };
    EXIT_MODE_RIP.store(ctx.rip, Ordering::SeqCst);
    EXIT_MODE_COUNT.store(num_instructions, Ordering::SeqCst);

    let mem = unsafe { &mut *(ctx.rdi as *mut Aligned128) };
    if (mem.lo, mem.hi) == (ctx.rax, ctx.rdx) {
        mem.lo = ctx.rbx;
        mem.hi = ctx.rcx;
        ctx.zf = 1;
    } else {
        ctx.rax = mem.lo;
        ctx.rdx = mem.hi;
        ctx.zf = 0;
    }
    ctx.rip += 5;
}

#[test]
fn test_cmpxchg16b_exit_mode() {
    let mut mem = Aligned128 { lo: 1, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 9;
    ctx.rcx = 9;
    ctx.callbacks = X86_64Callbacks {
        interpreter_fallback: interpret_cmpxchg16b,
        ..X86_64Callbacks::default()
    };

    // nop makes the callback PC distinguishable from region entry.
    let bytes = [0x90, 0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let out = LiftTest::new(&bytes)
        .no_unicorn()
        .ctx(ctx)
        .atomic128(Atomic128Mode::Interpret)
        .run();

    assert_eq!(out.exit, ExitReason::Return, "the region ran on to its RET");
    assert_eq!(
        EXIT_MODE_RIP.load(Ordering::SeqCst),
        DEFAULT_LIFT_BASE + 1,
        "the PC points at the CMPXCHG16B, not past it"
    );
    assert_eq!(EXIT_MODE_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(
        (mem.lo, mem.hi),
        (9, 9),
        "the callback performed the exchange"
    );
    assert_eq!(out.ctx.zf, 1);
}

/// Compiler-style 128-bit CAS loop: save RBX, retry CMPXCHG16B, restore RBX.
#[test]
fn test_cmpxchg16b_retry_loop() {
    let mut mem = Aligned128 { lo: 100, hi: 200 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rbx = 0xCAFE; // must be preserved across the region

    #[rustfmt::skip]
    let bytes = [
        0x53,                                     // push rbx
        0x48, 0x8B, 0x07,                         // mov rax, [rdi]
        0x48, 0x8B, 0x57, 0x08,                   // mov rdx, [rdi+8]
        0x48, 0x8D, 0x58, 0x01,                   // lea rbx, [rax+1]
        0x48, 0x8D, 0x4A, 0x01,                   // lea rcx, [rdx+1]
        0xF0, 0x48, 0x0F, 0xC7, 0x0F,             // lock cmpxchg16b [rdi]
        0x75, 0xF1,                               // jne .retry
        0x5B,                                     // pop rbx
        0xC3,                                     // ret
    ];
    let ctx = run_test(&bytes, ctx);

    assert_eq!((mem.lo, mem.hi), (101, 201), "the increment landed");
    assert_eq!(ctx.zf, 1, "loop exits with ZF=1");
    assert_eq!(ctx.rbx, 0xCAFE, "RBX restored by the pop");
}

/// LOCK ADC/SBB depend on incoming CF, so they lower to a CAS retry loop.
#[test]
fn test_lock_adc() {
    let mut mem_val: u64 = 10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 5;

    let bytes = [0xF9, 0xF0, 0x48, 0x11, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 16, "10 + 5 + CF");
    assert_eq!(ctx.cf, 0, "no carry out");
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_lock_adc_carry_out() {
    let mut mem_val: u64 = u64::MAX;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 1;

    let bytes = [0xF8, 0xF0, 0x48, 0x11, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0);
    assert_eq!(ctx.cf, 1, "carry out of the wrap");
    assert_eq!(ctx.zf, 1, "result is zero");
}

#[test]
fn test_lock_sbb() {
    let mut mem_val: u64 = 10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 5;

    let bytes = [0xF9, 0xF0, 0x48, 0x19, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 4, "10 - 5 - CF");
    assert_eq!(ctx.cf, 0, "no borrow out");
}

#[test]
fn test_lock_sbb_borrow_out() {
    let mut mem_val: u64 = 1;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 2;

    let bytes = [0xF8, 0xF0, 0x48, 0x19, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, u64::MAX);
    assert_eq!(ctx.cf, 1, "borrow out");
}

#[test]
fn test_lock_adc_imm_dword() {
    let mut mem_val: u64 = 0xFFFF_FFFF_0000_000A;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    let bytes = [0xF9, 0xF0, 0x83, 0x17, 0x05, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        mem_val, 0xFFFF_FFFF_0000_0010,
        "only the low dword is written: 0xA + 5 + CF"
    );
    assert_eq!(ctx.cf, 0);
}

/// Fences and prefetch hints must not abort the lift.
#[test]
fn test_cache_hints_are_transparent() {
    let mut mem_val: u64 = 7;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    #[rustfmt::skip]
    let bytes = [
        0x0F, 0x18, 0x0F,             // prefetcht0 [rdi]
        0x0F, 0x18, 0x07,             // prefetchnta [rdi]
        0x0F, 0x0D, 0x0F,             // prefetchw [rdi]
        0x48, 0x8B, 0x07,             // mov rax, [rdi]
        0x0F, 0xAE, 0x3F,             // clflush [rdi]
        0x66, 0x0F, 0xAE, 0x3F,       // clflushopt [rdi]
        0x66, 0x0F, 0xAE, 0x37,       // clwb [rdi]
        0x0F, 0x01, 0xE8,             // serialize
        0xC3,                         // ret
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 7, "the load between the hints still happened");
    assert_eq!(mem_val, 7);
}
