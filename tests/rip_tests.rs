use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

fn run_rip_test(
    bytes: &[u8],
    ip: u64,
    base_address: u64,
    mut ctx: RegisterContext,
) -> RegisterContext {
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
    let stack_ptr = stack.as_mut_ptr() as u64;
    let stack_top = stack_ptr + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = base_address;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(bytes, ip).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

#[test]
fn test_rip_relative_lea() {
    // lea rax, [rip + 0x10]; ret
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    // Case 1: base_address matches ip
    let ip = 0x1000;
    let base_address = 0x1000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rax, 0x1017);
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF); // Popped trampoline
}

#[test]
fn test_rip_sync_on_exit() {
    // mov rax, 1; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let ip = 0x2000;
    let base_address = 0x500000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_rip_relative_jmp() {
    // jmp 0x1010; nop; ret
    let bytes = [0xEB, 0x0E, 0x90, 0xC3];
    let ip = 0x1000;
    let base_address = 0x8000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());

    // reconciled_rip = 0x8000 + (0x1010 - 0x1000) = 0x8010
    assert_eq!(ctx.rip, 0x8010);
}

#[test]
fn test_rip_relative_cond_jmp() {
    // xor rax, rax; jz 0x1010; ret
    let bytes = [0x48, 0x31, 0xC0, 0x74, 0x0B, 0xC3];
    let ip = 0x1000;
    let base_address = 0x9000;

    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0x9010);

    let bytes_jnz = [0x48, 0x31, 0xC0, 0x75, 0x0B, 0xC3];
    let ctx = run_rip_test(&bytes_jnz, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF);
}
