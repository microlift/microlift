use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

fn run_test(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
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
fn test_overlapping_jmp_into_self() {
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,       // 0x1000: xor eax, eax
        0xEB, 0xFF,       // 0x1002: jmp 0x1003  (into the 0xFF byte below)
        0xC0,             // 0x1004: (this byte is 'C0'; with 0x1003's 'FF' -> inc eax)
        0xC3,             // 0x1005: ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1, "overlapping `FF C0` should decode as inc eax");
}

/// Sub-encoded instruction hidden inside a `mov` immediate.
///
/// A linear disassembler decodes a 5-byte `mov eax, imm32` at 0x1004, but control
/// flow jumps one byte into it, so the immediate bytes decode as `add eax, imm32`.
#[test]
fn test_subencoded_in_mov_immediate() {
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,                   // 0x1000: xor eax, eax
        0xEB, 0x01,                   // 0x1002: jmp 0x1005 (into the mov immediate)
        0xB8,                         // 0x1004: mov-eax opcode (skipped byte)
        0x05, 0x2A, 0x00, 0x00, 0x00, // 0x1005: add eax, 0x2A (overlaps the mov)
        0xC3,                         // 0x100A: ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x2A, "bytes entered mid-mov should decode as add eax, 0x2A");
}
