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

    // Provide a stack buffer that we can also use for string op testing
    let mut stack = [0u8; 8192];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;

    // Initialize RSP and push the exit trampoline address
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(bytes, 0x1000).expect("lifting failed");
    println!("IR: {}", func);

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

#[test]
fn test_shifts_rotates() {
    // SHL rax, 1 (rax = 1)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xD1, 0xE0, // shl rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 2);
    assert_eq!(ctx.cf, 0);

    // SHR rax, 1 (rax = 2)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00, // mov rax, 2
        0x48, 0xD1, 0xE8, // shr rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 0);

    // ROR rax, 1 (rax = 1) -> 0x8000
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xD1, 0xC8, // ror rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1u64 << 63);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_bit_ops() {
    // BTS rax, 3 (rax = 0)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x0F, 0xBA, 0xE8, 0x03, // bts rax, 3
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
    assert_eq!(ctx.cf, 0);

    // BTR rax, 3 (rax = 8)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xF0, 0x03, // btr rax, 3
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 1);

    // BTC rax, 1 (rax = 0)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x0F, 0xBA, 0xF8, 0x01, // btc rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 2);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_bit_scans() {
    // BSF rax, rcx (rcx = 0x10) -> rax = 4
    let bytes = [
        0x48, 0xC7, 0xC1, 0x10, 0x00, 0x00, 0x00, // mov rcx, 16
        0x48, 0x0F, 0xBC, 0xC1, // bsf rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 4);
    assert_eq!(ctx.zf, 0);

    // BSR rax, rcx (rcx = 0x11) -> rax = 4
    let bytes = [
        0x48, 0xC7, 0xC1, 0x11, 0x00, 0x00, 0x00, // mov rcx, 17
        0x48, 0x0F, 0xBD, 0xC1, // bsr rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 4);
    assert_eq!(ctx.zf, 0);

    // BSF with zero
    let bytes = [
        0x48, 0x31, 0xC9, // xor rcx, rcx
        0x48, 0x0F, 0xBC, 0xC1, // bsf rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.zf, 1);
}

#[test]
fn test_movsx() {
    // MOVSX rax, cl (cl = 0xFF) -> rax = 0xFFFFFFFFFFFFFFFF
    let bytes = [
        0x48, 0xC7, 0xC1, 0xFF, 0x00, 0x00, 0x00, // mov rcx, 0xFF
        0x48, 0x0F, 0xBE, 0xC1, // movsx rax, cl
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFF);

    // MOVSXD rax, ecx (ecx = 0x80000000) -> rax = 0xFFFFFFFF80000000
    let bytes = [
        0x48, 0xC7, 0xC1, 0x00, 0x00, 0x00, 0x80, // mov rcx, 0x80000000
        0x48, 0x63, 0xC1, // movsxd rax, ecx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFF80000000);
}

#[test]
fn test_string_ops() {
    let mut ctx = RegisterContext::default();

    let mut mem = [0u8; 256];
    let mem_ptr = mem.as_mut_ptr() as u64;

    // MOVSB: copy from [rsi] to [rdi]
    mem[0] = 0xCC;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 8;

    let bytes = [
        0xA4, // movsb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[8], 0xCC);
    assert_eq!(ctx_out.rsi, mem_ptr + 1);
    assert_eq!(ctx_out.rdi, mem_ptr + 9);

    // STOSB: store AL to [rdi]
    ctx.rax = 0x42;
    ctx.rdi = mem_ptr + 16;
    let bytes = [
        0xAA, // stosb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x42);
    assert_eq!(ctx_out.rdi, mem_ptr + 17);

    // LODSB: load from [rsi] to AL
    mem[32] = 0x77;
    ctx.rsi = mem_ptr + 32;
    let bytes = [
        0xAC, // lodsb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(ctx_out.rax & 0xFF, 0x77);
    assert_eq!(ctx_out.rsi, mem_ptr + 33);
}

#[test]
fn test_string_ops_sizes() {
    let mut mem = [0u8; 256];
    let mem_ptr = mem.as_mut_ptr() as u64;
    let mut ctx = RegisterContext::default();

    // MOVSW
    mem[0] = 0x11;
    mem[1] = 0x22;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 16;
    let bytes = [0x66, 0xA5, 0xC3]; // movsw, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x11);
    assert_eq!(mem[17], 0x22);
    assert_eq!(ctx_out.rsi, mem_ptr + 2);

    // MOVSD (32-bit)
    mem[0] = 0x33;
    mem[1] = 0x44;
    mem[2] = 0x55;
    mem[3] = 0x66;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 32;
    let bytes = [0xA5, 0xC3]; // movsd (32-bit in 64-bit mode?), ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[32], 0x33);
    assert_eq!(mem[35], 0x66);
    assert_eq!(ctx_out.rsi, mem_ptr + 4);

    // MOVSQ
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 64;
    unsafe {
        *(mem_ptr as *mut u64) = 0xDEADBEEFCAFEBABE;
    }
    let bytes = [0x48, 0xA5, 0xC3]; // movsq, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(
        unsafe { *((mem_ptr + 64) as *const u64) },
        0xDEADBEEFCAFEBABE
    );
    assert_eq!(ctx_out.rsi, mem_ptr + 8);

    // STOSW
    ctx.rax = 0xCCAA;
    ctx.rdi = mem_ptr + 80;
    let bytes = [0x66, 0xAB, 0xC3]; // stosw, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[80], 0xAA);
    assert_eq!(mem[81], 0xCC);
    assert_eq!(ctx_out.rdi, mem_ptr + 82);

    // STOSD
    ctx.rax = 0x11223344;
    ctx.rdi = mem_ptr + 96;
    let bytes = [0xAB, 0xC3]; // stosd, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[96], 0x44);
    assert_eq!(mem[99], 0x11);
    assert_eq!(ctx_out.rdi, mem_ptr + 100);

    // STOSQ
    ctx.rax = 0x1122334455667788;
    ctx.rdi = mem_ptr + 128;
    let bytes = [0x48, 0xAB, 0xC3]; // stosq, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(
        unsafe { *((mem_ptr + 128) as *const u64) },
        0x1122334455667788
    );
    assert_eq!(ctx_out.rdi, mem_ptr + 136);
}
