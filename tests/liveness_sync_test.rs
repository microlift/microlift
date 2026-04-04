use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;

#[test]
fn test_liveness_sync_add() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::Fast));
    // Input: *mut RegisterContext (I64)
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);

    let mut lifter = Lifter::new(builder);

    // add rax, rbx; ret
    let bytes = [0x48, 0x01, 0xD8, 0xC3];

    let mut ctx = RegisterContext::default();
    let mut stack = [0u64; 1024];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;
    ctx.rax = 100;
    ctx.rbx = 42;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    // Verify sync
    assert_eq!(ctx.rax, 142);
    assert_eq!(ctx.rbx, 42);
    assert_eq!(ctx.zf, 0);
}

#[test]
fn test_liveness_sync_flags() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::Fast));
    // Input: *mut RegisterContext (I64)
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);

    let mut lifter = Lifter::new(builder);

    // xor rax, rax; ret (should set ZF=1)
    let bytes = [0x48, 0x31, 0xC0, 0xC3];

    let mut ctx = RegisterContext::default();
    let mut stack = [0u64; 1024];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x2000;
    ctx.rax = 999;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(&bytes, 0x2000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    // Verify sync
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.zf, 1);
}
