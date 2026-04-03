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
    // 48 01 D8 (add rax, rbx)
    // C3 (ret)
    let bytes = [0x48, 0x01, 0xD8, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let mut ctx = RegisterContext::default();
    ctx.rax = 100;
    ctx.rbx = 42;

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    // Verify sync
    assert_eq!(ctx.rax, 142);
    // RBX should remain 42
    assert_eq!(ctx.rbx, 42);
    // ZF should be 0 because result is not zero
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
    // 48 31 C0 (xor rax, rax)
    // C3 (ret)
    let bytes = [0x48, 0x31, 0xC0, 0xC3];
    lifter.lift(&bytes, 0x2000).expect("lifting failed");

    println!("IR:\n{}", func.display());

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let mut ctx = RegisterContext::default();
    ctx.rax = 999;

    println!("CTX before: {:?}", ctx);
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    println!("CTX after: {:?}", ctx);

    // Verify sync
    assert_eq!(ctx.rax, 0);
    // ZF should be 1
    assert_eq!(ctx.zf, 1);
}
