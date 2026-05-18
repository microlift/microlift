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
    let stack_top = unsafe { stack.as_mut_ptr().add(1024) } as u64;
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
