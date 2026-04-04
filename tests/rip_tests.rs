use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;

fn run_rip_test(
    bytes: &[u8],
    ip: u64,
    base_address: u64,
    mut ctx: RegisterContext,
) -> RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::Fast));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder, base_address);

    lifter.lift(bytes, ip).expect("lifting failed");

    println!("IR:\n{}", func.display());

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

#[test]
fn test_rip_relative_lea() {
    // lea rax, [rip + 0x10]; ret
    // 48 8d 05 10 00 00 00 (lea rax, [rip+16])
    // c3 (ret)
    // instruction length is 7 bytes
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    // Case 1: base_address matches ip
    let ip = 0x1000;
    let base_address = 0x1000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    // next_ip = 0x1007
    // rax = next_ip + 0x10 = 0x1007 + 0x10 = 0x1017
    assert_eq!(ctx.rax, 0x1017);
    assert_eq!(ctx.rip, 0x1008); // next_ip after ret is 0x1008

    // Case 2: different base_address
    let ip = 0x1000;
    let base_address = 0x400000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    // lift_base = 0x1000, base_address = 0x400000
    // current_ip = 0x1007
    // reconciled_rip = 0x400000 + (0x1007 - 0x1000) = 0x400007
    // rax = reconciled_rip + 0x10 = 0x400007 + 0x10 = 0x400017
    assert_eq!(ctx.rax, 0x400017);
}

#[test]
fn test_rip_sync_on_exit() {
    // mov rax, 1; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let ip = 0x2000;
    let base_address = 0x500000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());

    // The last instruction lifted is 'ret' (length 1).
    // Before 'ret', current_ip = 0x2007 (after mov).
    // Then 'ret' is decoded, current_ip = 0x2008.
    // ret handler calls sync_to_context.
    // reconciled_rip = 0x500000 + (0x2008 - 0x2000) = 0x500008.
    assert_eq!(ctx.rip, 0x500008);
}
