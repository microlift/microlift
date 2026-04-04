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

    let mut stack = [0u64; 1024];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
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
fn test_mov_32_zero_extend() {
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
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, ...
        0x66, 0xB8, 0xAA, 0xAA, // mov ax, 0xAAAA
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566AAAA);
}

#[test]
fn test_mov_8_partial() {
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0xB0, 0xBB, // mov al, 0xBB
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x11223344556677BB);
}

#[test]
fn test_mov_8_high_partial() {
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0xB4, 0xCC, // mov ah, 0xCC
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x112233445566CC88);
}

#[test]
fn test_add_32_flags() {
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
