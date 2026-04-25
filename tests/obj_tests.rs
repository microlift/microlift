use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use microlift::backend::ArtifactBackend;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::lifter::{ExecutionMode, Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;
use object::Object;

#[test]
fn test_compile_host_object() {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let obj_bytes = ArtifactBackend::compile_to_object(func.clone(), None, None)
        .expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());

    // Check for some common object magic (ELF \x7fELF, Mach-O \xce\xfa\xed\xfe or \xcf\xfa\xed\xfe, PE MZ)
    let magic = &obj_bytes[0..4];
    let is_elf = magic == b"\x7fELF";
    let is_macho = magic == b"\xce\xfa\xed\xfe" || magic == b"\xcf\xfa\xed\xfe";
    let is_pe = &obj_bytes[0..2] == b"MZ";

    assert!(
        is_elf || is_macho || is_pe,
        "Should be a valid object file format, got magic: {:02X?}",
        magic
    );

    let raw_bytes =
        ArtifactBackend::compile_to_raw(func, None, None).expect("failed to compile to raw");
    assert!(!raw_bytes.is_empty());

    #[cfg(feature = "disasm")]
    {
        let disasm = ArtifactBackend::disassemble(&raw_bytes, None).expect("disassembly failed");
        println!("Host Disassembly:\n{}", disasm);
    }
}

#[test]
fn test_compile_riscv64_object() {
    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::SystemV));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let triple = "riscv64-unknown-linux-gnu";
    let obj_bytes = ArtifactBackend::compile_to_object(func.clone(), Some(triple), None)
        .expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());

    // RISC-V Linux should be ELF
    assert_eq!(&obj_bytes[0..4], b"\x7fELF");

    let raw_bytes = ArtifactBackend::compile_to_raw(func, Some(triple), None)
        .expect("failed to compile to raw");
    assert!(!raw_bytes.is_empty());

    #[cfg(feature = "disasm")]
    {
        let disasm =
            ArtifactBackend::disassemble(&raw_bytes, Some(triple)).expect("disassembly failed");
        println!("RISC-V 64 Disassembly:\n{}", disasm);
    }
}

#[test]
fn test_compile_aarch64_object() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "true").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::SystemV));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let triple = "aarch64-unknown-linux-gnu";
    let obj_bytes =
        ArtifactBackend::compile_to_object(func.clone(), Some(triple), Some(flags.clone()))
            .expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());

    let obj = object::File::parse(&*obj_bytes).expect("failed to parse object");
    assert_eq!(obj.architecture(), object::Architecture::Aarch64);

    let raw_bytes = ArtifactBackend::compile_to_raw(func, Some(triple), Some(flags.clone()))
        .expect("failed to compile to raw");
    assert!(!raw_bytes.is_empty());

    #[cfg(feature = "disasm")]
    {
        // Raw .text disassembly starts at address 0, so local PC-relative branches
        // print as section-relative targets like `je 0x114` rather than runtime VAs.
        let disasm =
            ArtifactBackend::disassemble(&raw_bytes, Some(triple)).expect("disassembly failed");
        println!("Disassembly:\n{}", disasm);
    }
}

#[test]
fn test_execute_emitted_host_function() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    // `cranelift-jit` requires non-PIC code, so this runnable example uses
    // JIT-compatible flags while the object test above verifies PIC emission.
    flag_builder.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let mut ctx = RegisterContext {
        image_base: 0x1000,
        ..RegisterContext::default()
    };
    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + (stack.len() * std::mem::size_of::<u64>()) as u64;
    ctx.rsp = stack_top - 8;

    unsafe {
        *(ctx.rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF;
    }

    let mut backend = CraneliftBackend::with_flags(flags).expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let emitted: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    emitted(&mut ctx);

    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_compile_runtime_assisted_syscall_object() {
    let mut func =
        Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::SystemV));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(
        builder,
        LifterConfig {
            execution_mode: ExecutionMode::RuntimeAssisted,
            syscall_handler_call_conv: CallConv::SystemV,
            ..Default::default()
        },
    );

    // mov rax, 42; syscall; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0x0F, 0x05, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let obj_bytes =
        ArtifactBackend::compile_to_object(func.clone(), Some("aarch64-unknown-linux-gnu"), None)
            .expect("failed to compile runtime-assisted syscall object");
    assert!(!obj_bytes.is_empty());
    assert_eq!(&obj_bytes[0..4], b"\x7fELF");

    let raw_bytes = ArtifactBackend::compile_to_raw(func, Some("aarch64-unknown-linux-gnu"), None)
        .expect("failed to compile runtime-assisted syscall raw code");
    assert!(!raw_bytes.is_empty());
}
