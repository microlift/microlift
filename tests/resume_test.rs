use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::lifter::{ExecutionMode, Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

// FIXME: When bswap is implemented, this test will fail.
// The test should be changed to use a made up, completely invalid instruction.
#[test]
fn test_resumable_unimplemented_fallback() {
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
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);

    let config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        interpreter_fallback: true,
        emit_resume_blocks: true,
        ..Default::default()
    };

    let mut lifter = Lifter::with_config(builder, config);

    let mut ctx = RegisterContext::default();
    let mut stack = [0u64; 1024];
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;
    ctx.rip = 0x1000;

    // Push exit trampoline address
    unsafe {
        *(ctx.rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF;
    }

    // mov rax, 42;
    // bswap rax (unimplemented)
    // ret
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0x48, 0x0F, 0xC8, // bswap rax (unimplemented)
        0xC3, // ret
    ];

    lifter.lift(&bytes, 0x1000).expect("lifting failed");
    drop(lifter);

    println!("IR: {:#?}", &func);
    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let entry: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };

    // First run: should hit unimplemented instruction
    entry(&mut ctx);

    assert_eq!(ctx.exit_reason, ExitReason::UnimplementedInstruction as u32);
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rip, 0x100A); // IP at next instruction (ret)
    assert_ne!(ctx.exit_code, 0); // Should have an opcode

    // Host shims bswap: rax = swap(42)
    ctx.rax = ctx.rax.swap_bytes();
    // ctx.rip is already at 0x100A, no need to update

    // Resume by calling entry again
    entry(&mut ctx);

    // Should now hit ret and exit with Return (via trampoline)
    assert!(
        ctx.exit_reason == ExitReason::Return as u32 || ctx.exit_reason == ExitReason::None as u32
    );
    assert_eq!(ctx.rax, 42u64.swap_bytes());
}
