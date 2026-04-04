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
    let mut lifter = Lifter::new(builder, 0x1000);

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    println!("IR: {}", func);

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

#[test]
fn test_mov_32_zero_extend() {
    // mov rax, 0xFFFFFFFFFFFFFFFF; mov eax, 0x12345678; ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xB8, 0x78, 0x56, 0x34, 0x12, // mov eax, 0x12345678
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x12345678);
}

#[test]
fn test_mov_16_partial() {
    // mov rax, 0x1122334455667788; mov ax, 0xAAAA; ret
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
        0x11, // mov rax, 0x1122334455667788
        0x66, 0xB8, 0xAA, 0xAA, // mov ax, 0xAAAA
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566AAAA);
}

#[test]
fn test_mov_8_partial() {
    // mov rax, 0x1122334455667788; mov al, 0xBB; ret
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, 11...
        0xB0, 0xBB, // mov al, 0xBB
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x11223344556677BB);
}

#[test]
fn test_mov_8_high_partial() {
    // mov rax, 0x1122334455667788; mov ah, 0xCC; ret
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, 11...
        0xB4, 0xCC, // mov ah, 0xCC
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566CC88);
}

#[test]
fn test_add_32_flags() {
    // mov eax, 0x7FFFFFFF; add eax, 1; ret
    let bytes = [
        0xB8, 0xFF, 0xFF, 0xFF, 0x7F, // mov eax, 0x7FFFFFFF
        0x83, 0xC0, 0x01, // add eax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x80000000);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_sub_8_carry() {
    // mov al, 0; sub al, 1; ret
    let bytes = [
        0xB0, 0x00, // mov al, 0
        0x2C, 0x01, // sub al, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0xFF);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.sf, 1);
}

#[test]
fn test_inc_8_overflow() {
    // mov al, 0x7F; inc al; ret
    let bytes = [
        0xB0, 0x7F, // mov al, 0x7F
        0xFE, 0xC0, // inc al
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0x80);
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.sf, 1);
}
