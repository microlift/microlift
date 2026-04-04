use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;

fn run_test(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::Fast));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    ctx.image_base = 0x1000;

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    println!("IR: {}", func);

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

#[test]
fn test_and_or() {
    // mov rax, 0b1010; mov rcx, 0b1100; and rax, rcx; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, // mov rax, 10
        0x48, 0xC7, 0xC1, 0x0C, 0x00, 0x00, 0x00, // mov rcx, 12
        0x48, 0x21, 0xC8, // and rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8); // 10 & 12 = 8
    assert_eq!(ctx.zf, 0); // ZF should be 0

    // mov rax, 0b1010; mov rcx, 0b1100; or rax, rcx; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, // mov rax, 10
        0x48, 0xC7, 0xC1, 0x0C, 0x00, 0x00, 0x00, // mov rcx, 12
        0x48, 0x09, 0xC8, // or rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 14); // 10 | 12 = 14
}

#[test]
fn test_inc_dec() {
    // mov rax, 41; inc rax; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x29, 0x00, 0x00, 0x00, // mov rax, 41
        0x48, 0xFF, 0xC0, // inc rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);

    // mov rax, 43; dec rax; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2B, 0x00, 0x00, 0x00, // mov rax, 43
        0x48, 0xFF, 0xC8, // dec rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_cmp_jnz() {
    // mov rax, 10; cmp rax, 10; jne target; mov rax, 20; target: ret
    // 0: 48 c7 c0 0a 00 00 00    mov rax, 10
    // 7: 48 83 f8 0a             cmp rax, 10
    // b: 75 07                   jne +7 (to ret)
    // d: 48 c7 c0 14 00 00 00    mov rax, 20
    // 14: c3                     ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, 0x48, 0x83, 0xF8, 0x0A, 0x75, 0x07, 0x48, 0xC7,
        0xC0, 0x14, 0x00, 0x00, 0x00, 0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 20); // jne should NOT jump

    // mov rax, 10; cmp rax, 11; jne target; mov rax, 20; target: ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, 0x48, 0x83, 0xF8, 0x0B, 0x75, 0x07, 0x48, 0xC7,
        0xC0, 0x14, 0x00, 0x00, 0x00, 0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 10); // jne SHOULD jump
}

#[test]
fn test_test_jnz() {
    // mov rax, 0; test rax, rax; jnz target; mov rax, 20; target: ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x85, 0xC0, // test rax, rax
        0x75, 0x07, // jnz +7
        0x48, 0xC7, 0xC0, 0x14, 0x00, 0x00, 0x00, // mov rax, 20
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 20); // jnz should NOT jump

    // mov rax, 1; test rax, rax; jnz target; mov rax, 20; target: ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x85, 0xC0, // test rax, rax
        0x75, 0x07, // jnz +7
        0x48, 0xC7, 0xC0, 0x14, 0x00, 0x00, 0x00, // mov rax, 20
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1); // jnz SHOULD jump
}

#[test]
fn test_lea() {
    // mov rax, 1000; lea rbx, [rax + 42]; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0xE8, 0x03, 0x00, 0x00, // mov rax, 1000
        0x48, 0x8D, 0x58, 0x2A, // lea rbx, [rax + 42]
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rbx, 1042);
}
