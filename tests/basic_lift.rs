use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

#[test]
fn test_mov_rax_imm() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let _isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(flag_builder))
        .unwrap();

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    // Input: *mut RegisterContext (I64)
    func.signature.params.push(AbiParam::new(types::I64));
    // Output: void

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);

    let mut lifter = Lifter::new(builder);

    let mut ctx = RegisterContext::default();
    let mut stack = [0u64; 1024];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;

    // Push exit trampoline address
    unsafe {
        *(ctx.rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF;
    }

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    assert_eq!(ctx.rax, 42);
}
