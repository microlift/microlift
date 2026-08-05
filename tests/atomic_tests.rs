use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;

#[test]
fn test_mfence() {
    // mfence; ret
    let bytes = [0x0F, 0xAE, 0xF0, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    // Just verifies no crash / correct exit
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_sfence() {
    // sfence; ret
    let bytes = [0x0F, 0xAE, 0xF8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_lfence() {
    // lfence; ret
    let bytes = [0x0F, 0xAE, 0xE8, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_pause() {
    // pause; ret
    let bytes = [0xF3, 0x90, 0xC3];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
}

#[test]
fn test_cmpxchg_reg_equal() {
    // mov rax, 10; mov rcx, 10; mov rbx, 99;
    // cmpxchg rcx, rbx  (compare RAX==RCX -> equal, so RCX <- RBX=99)
    // ret
    //
    // Encoding: 48 0F B1 D9  = CMPXCHG rcx, rbx (REX.W 0F B1 /r)
    // ModRM for rcx (dst) / rbx (src): mod=11 reg=rbx(3) rm=rcx(1) -> 0xD9
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
    // mov rax, 5; mov rcx, 10; mov rbx, 99;
    // cmpxchg rcx, rbx  (compare RAX(5) != RCX(10) -> RAX <- RCX=10, ZF=0)
    // ret
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
    // RDI = pointer to mem_val = 42
    // mov rax, 42; mov rbx, 99;
    // cmpxchg [rdi], rbx   (42==42 -> [rdi]=99, ZF=1)
    // ret
    //
    // CMPXCHG [RDI], RBX (REX.W 0F B1 1F): 48 0F B1 1F
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
    // RDI = pointer to mem_val = 7
    // mov rax, 42; mov rbx, 99;
    // cmpxchg [rdi], rbx   (42 != 7 -> rax=7, ZF=0, [rdi] unchanged)
    // ret
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
    // mem = 0x0000_0002_0000_0001  (EDX:EAX expected = 2:1)
    // ECX:EBX replacement = 4:3  -> mem should become 0x0000_0004_0000_0003
    // ZF = 1
    //
    // CMPXCHG8B [RDI]: 0F C7 0F
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
    // mem = 0x0000_0099_0000_0088  (EDX:EAX expected mismatch)
    // EDX:EAX = 2:1 (doesn't match), so ZF=0 and EDX:EAX <- mem
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
    // RDI = pointer to mem_val = 10
    // mov rsi, 5
    // lock add [rdi], rsi    -> mem = 15, flags for 10+5
    // ret
    //
    // F0 48 01 37: LOCK ADD [RDI], RSI
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
    // mem = u64::MAX, add 1 -> wraps to 0, ZF=1, CF=1
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
    // RDI = pointer to mem_val = 100
    // RSI = 5
    // lock xadd [rdi], rsi  -> mem = 105, rsi = 100 (old value)
    //
    // F0 48 0F C1 37: LOCK XADD [RDI], RSI
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
    // RDI = pointer to mem_val = 77
    // RSI = 99
    // lock xchg [rdi], rsi  -> mem = 99, rsi = 77
    //
    // F0 48 87 37: LOCK XCHG [RDI], RSI
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
    // mem = 20; lock sub [rdi], 7 -> mem = 13
    // F0 48 83 2F 07: LOCK SUB QWORD [RDI], 7  (imm8 sign-extended)
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
    // mem = 0b1111; lock and [rdi], 0b1010 -> mem = 0b1010
    // F0 48 83 27 0A: LOCK AND QWORD [RDI], 10
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
    // mem = 0b1010; lock or [rdi], 0b0101 -> mem = 0b1111
    // F0 48 83 0F 05: LOCK OR QWORD [RDI], 5
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
    // mem = 0xFF; lock xor [rdi], 0x0F -> mem = 0xF0
    // F0 48 83 37 0F: LOCK XOR QWORD [RDI], 15
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
    // mem = 41; lock inc [rdi] -> mem = 42
    // F0 48 FF 07: LOCK INC QWORD [RDI]
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
    // mem = 1; lock dec [rdi] -> mem = 0, ZF=1
    // F0 48 FF 0F: LOCK DEC QWORD [RDI]
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
    // mem = 0b0101; lock bts [rdi], 1  -> sets bit 1 -> mem = 0b0111, CF = 0 (was 0)
    // F0 48 0F AB 0F: LOCK BTS QWORD [RDI], RCX  (with RCX=1)
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
    // mem = 0b0111; lock bts [rdi], 1  -> bit 1 already set -> CF=1, mem unchanged
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
    // mem = 0b0111; lock btr [rdi], 1  -> clears bit 1 -> mem = 0b0101, CF=1 (was 1)
    // F0 48 0F B3 0F: LOCK BTR QWORD [RDI], RCX  (with RCX=1)
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
    // mem = 0b0101; lock btc [rdi], 2  -> complements bit 2 -> mem = 0b0001, CF=1 (was 1)
    // F0 48 0F BB 0F: LOCK BTC QWORD [RDI], RCX  (with RCX=2)
    let mut mem_val: u64 = 0b0101;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rcx = 2;

    let bytes = [0xF0, 0x48, 0x0F, 0xBB, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0b0001, "bit 2 should be flipped");
    assert_eq!(ctx.cf, 1, "CF=1: bit was originally set");
}

/// A 16-byte-aligned pair of qwords, standing in for the `__int128` a guest's
/// `CMPXCHG16B` operates on. Real hardware raises `#GP` on a misaligned
/// operand, so the tests use an operand the guest would actually be given.
#[repr(C, align(16))]
struct Aligned128 {
    lo: u64,
    hi: u64,
}

/// `lock cmpxchg16b [rdi]` where RDX:RAX matches: the memory takes RCX:RBX,
/// ZF is set, and RDX:RAX keep the expected value.
#[test]
fn test_cmpxchg16b_success() {
    let mut mem = Aligned128 { lo: 1, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 0xAAAA_BBBB_CCCC_DDDD;
    ctx.rcx = 0x1111_2222_3333_4444;

    // F0 48 0F C7 0F: lock cmpxchg16b [rdi]
    let bytes = [0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(mem.lo, 0xAAAA_BBBB_CCCC_DDDD, "low half takes RBX");
    assert_eq!(mem.hi, 0x1111_2222_3333_4444, "high half takes RCX");
    assert_eq!(ctx.zf, 1, "ZF=1 on success");
    assert_eq!(ctx.rax, 1, "RAX unchanged on success");
    assert_eq!(ctx.rdx, 2, "RDX unchanged on success");
}

/// A mismatch in the *high* half alone must still fail the comparison - the
/// case a lowering that only compares 64 bits would get wrong.
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

/// The low half alone mismatching must fail too.
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

/// Without the LOCK prefix the instruction is still valid (just not atomic) and
/// must lift to the same architectural result.
#[test]
fn test_cmpxchg16b_without_lock_prefix() {
    let mut mem = Aligned128 { lo: 5, hi: 6 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 5;
    ctx.rdx = 6;
    ctx.rbx = 7;
    ctx.rcx = 8;

    // 48 0F C7 0F: cmpxchg16b [rdi]
    let bytes = [0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!((mem.lo, mem.hi), (7, 8));
    assert_eq!(ctx.zf, 1);
}

/// CMPXCHG16B affects ZF only; CF/SF/OF/PF/AF must survive it untouched.
/// (`CMPXCHG` differs here - it updates all of them as if a `CMP` had run.)
#[test]
fn test_cmpxchg16b_leaves_other_flags_alone() {
    let mut mem = Aligned128 { lo: 0, hi: 0 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    // stc sets CF, then a failing cmpxchg16b must leave it set.
    ctx.rax = 1; // != mem.lo, so the CAS fails
    ctx.rdx = 0;

    // F9: stc ; F0 48 0F C7 0F: lock cmpxchg16b [rdi] ; C3
    let bytes = [0xF9, 0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let ctx = run_test(&bytes, ctx);

    assert_eq!(ctx.zf, 0, "ZF=0 on failure");
    assert_eq!(ctx.cf, 1, "CF must be untouched by CMPXCHG16B");
}

/// `Atomic128Mode::Native` lowers to a real `lock cmpxchg16b`. It only compiles
/// on an x64 host (no other Cranelift backend lowers `atomic_cas.i128`), which
/// is exactly why the mode is opt-in.
#[test]
#[cfg(target_arch = "x86_64")]
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

    // ...and the failing case leaves the observed value in RDX:RAX.
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

/// `Atomic128Mode::Exit` hands the instruction to the host as an ordinary
/// unimplemented-instruction exit: memory is untouched, and the PC is already
/// past the CMPXCHG16B so the host emulates it and resumes.
#[test]
fn test_cmpxchg16b_exit_mode() {
    let mut mem = Aligned128 { lo: 1, hi: 2 };
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem as *mut Aligned128 as u64;
    ctx.rax = 1;
    ctx.rdx = 2;
    ctx.rbx = 9;
    ctx.rcx = 9;

    // 90: nop (so the exit PC is distinguishable from the region entry)
    let bytes = [0x90, 0xF0, 0x48, 0x0F, 0xC7, 0x0F, 0xC3];
    let out = LiftTest::new(&bytes)
        .ctx(ctx)
        .atomic128(Atomic128Mode::Exit)
        .run();

    assert_eq!(
        out.ctx.exit_reason,
        ExitReason::UnimplementedInstruction as u32
    );
    assert_eq!(
        out.ctx.rip,
        DEFAULT_LIFT_BASE + 6,
        "PC is past the CMPXCHG16B (nop + 5-byte instruction)"
    );
    assert_eq!(
        (mem.lo, mem.hi),
        (1, 2),
        "memory untouched - the host runs it"
    );
}

/// The shape a compiler emits for a 128-bit compare-exchange loop: save RBX,
/// load the expected pair, retry the CAS until it succeeds, restore RBX. This
/// exercises CMPXCHG16B together with the PUSH/POP/JNE/MOV around it.
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
        // .retry:
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

/// LOCK ADC / LOCK SBB have no atomic-RMW opcode (the result depends on the
/// incoming CF), so they lower to a CAS retry loop.
#[test]
fn test_lock_adc() {
    // stc ; lock adc qword [rdi], rsi  ->  mem = 10 + 5 + 1
    let mut mem_val: u64 = 10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 5;

    // F9: stc ; F0 48 11 37: lock adc [rdi], rsi
    let bytes = [0xF9, 0xF0, 0x48, 0x11, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 16, "10 + 5 + CF");
    assert_eq!(ctx.cf, 0, "no carry out");
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_lock_adc_carry_out() {
    // lock adc [rdi], rsi with CF=0 and a sum that wraps: CF out = 1, result 0.
    let mut mem_val: u64 = u64::MAX;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 1;

    // F8: clc ; F0 48 11 37: lock adc [rdi], rsi
    let bytes = [0xF8, 0xF0, 0x48, 0x11, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 0);
    assert_eq!(ctx.cf, 1, "carry out of the wrap");
    assert_eq!(ctx.zf, 1, "result is zero");
}

#[test]
fn test_lock_sbb() {
    // stc ; lock sbb qword [rdi], rsi  ->  mem = 10 - 5 - 1
    let mut mem_val: u64 = 10;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 5;

    // F9: stc ; F0 48 19 37: lock sbb [rdi], rsi
    let bytes = [0xF9, 0xF0, 0x48, 0x19, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, 4, "10 - 5 - CF");
    assert_eq!(ctx.cf, 0, "no borrow out");
}

#[test]
fn test_lock_sbb_borrow_out() {
    // clc ; lock sbb [rdi], rsi with mem < rsi: borrow out, result wraps.
    let mut mem_val: u64 = 1;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;
    ctx.rsi = 2;

    let bytes = [0xF8, 0xF0, 0x48, 0x19, 0x37, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(mem_val, u64::MAX);
    assert_eq!(ctx.cf, 1, "borrow out");
}

/// LOCK ADC with an immediate source, at a narrower width.
#[test]
fn test_lock_adc_imm_dword() {
    let mut mem_val: u64 = 0xFFFF_FFFF_0000_000A;
    let mut ctx = RegisterContext::default();
    ctx.rdi = &mut mem_val as *mut u64 as u64;

    // F9: stc ; F0 83 17 05: lock adc dword [rdi], 5
    let bytes = [0xF9, 0xF0, 0x83, 0x17, 0x05, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        mem_val, 0xFFFF_FFFF_0000_0010,
        "only the low dword is written: 0xA + 5 + CF"
    );
    assert_eq!(ctx.cf, 0);
}

/// Cache maintenance and prefetch hints have no architectural effect here, but
/// they routinely appear next to atomics and must not abort the lift.
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
