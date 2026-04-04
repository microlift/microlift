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
    let mut lifter = Lifter::new(builder);

    ctx.image_base = base_address;

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

#[test]
fn test_rip_relative_jmp() {
    // jmp 0x1010; nop; ret
    // eb 0e (jmp 0x1010) - next_ip is 0x1002, 0x1002 + 0x0e = 0x1010
    // 90 (nop)
    // c3 (ret)
    let bytes = [0xEB, 0x0E, 0x90, 0xC3];
    let ip = 0x1000;
    // We only lift 4 bytes. Target is 0x1010, which is outside the lifted range.
    // The lifter should sync and return.
    let base_address = 0x8000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());

    // reconciled_rip = 0x8000 + (0x1010 - 0x1000) = 0x8010
    assert_eq!(ctx.rip, 0x8010);
}

#[test]
fn test_rip_relative_cond_jmp() {
    // xor rax, rax; jz 0x1010; ret
    // 48 31 c0 (xor rax, rax)
    // 74 0b (jz 0x1010) - next_ip is 1005, 1005+0b = 1010
    // c3 (ret)
    let bytes = [0x48, 0x31, 0xC0, 0x74, 0x0B, 0xC3];
    let ip = 0x1000;
    let base_address = 0x9000;

    // Test Case 1: ZF=1 (jz taken)
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    // reconciled_rip = 0x9000 + (0x1010 - 0x1000) = 0x9010
    assert_eq!(ctx.rip, 0x9010);

    // Test Case 2: ZF=0 (jz not taken)
    // We can't easily force ZF=0 without changing code, so let's use 'jnz' instead.
    // 75 0b (jnz 0x1010)
    let bytes_jnz = [0x48, 0x31, 0xC0, 0x75, 0x0B, 0xC3];
    let ctx = run_rip_test(&bytes_jnz, ip, base_address, RegisterContext::default());
    // next_ip after ret is 1006. Reconciled: 0x9000 + 6 = 0x9006.
    assert_eq!(ctx.rip, 0x9006);
}
