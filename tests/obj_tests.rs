use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use microlift::backend::ArtifactBackend;
use microlift::lifter::Lifter;

#[test]
fn test_compile_host_object() {
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::SystemV));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let obj_bytes = ArtifactBackend::compile_to_object(func.clone(), None).expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());
    
    // Check for some common object magic (ELF \x7fELF, Mach-O \xce\xfa\xed\xfe or \xcf\xfa\xed\xfe, PE MZ)
    let magic = &obj_bytes[0..4];
    let is_elf = magic == b"\x7fELF";
    let is_macho = magic == b"\xce\xfa\xed\xfe" || magic == b"\xcf\xfa\xed\xfe";
    let is_pe = &obj_bytes[0..2] == b"MZ";
    
    assert!(is_elf || is_macho || is_pe, "Should be a valid object file format, got magic: {:02X?}", magic);

    let raw_bytes = ArtifactBackend::compile_to_raw(func, None).expect("failed to compile to raw");
    assert!(!raw_bytes.is_empty());

    #[cfg(feature = "disasm")]
    {
        let disasm =
            ArtifactBackend::disassemble(&raw_bytes, None).expect("disassembly failed");
        println!("Host Disassembly:\n{}", disasm);
    }
}

#[test]
fn test_compile_riscv64_object() {
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), Signature::new(CallConv::SystemV));
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let triple = "riscv64-unknown-linux-gnu";
    let obj_bytes = ArtifactBackend::compile_to_object(func.clone(), Some(triple)).expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());
    
    // RISC-V Linux should be ELF
    assert_eq!(&obj_bytes[0..4], b"\x7fELF");

    let raw_bytes = ArtifactBackend::compile_to_raw(func, Some(triple)).expect("failed to compile to raw");
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
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(CallConv::SystemV),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    // mov rax, 42; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    lifter.lift(&bytes, 0x1000).expect("lifting failed");

    let triple = "aarch64-unknown-linux-gnu";
    let obj_bytes = ArtifactBackend::compile_to_object(func.clone(), Some(triple))
        .expect("failed to compile to object");
    assert!(!obj_bytes.is_empty());

    // AArch64 Linux should be ELF
    assert_eq!(&obj_bytes[0..4], b"\x7fELF");

    let raw_bytes =
        ArtifactBackend::compile_to_raw(func, Some(triple)).expect("failed to compile to raw");
    assert!(!raw_bytes.is_empty());

    #[cfg(feature = "disasm")]
    {
        let disasm =
            ArtifactBackend::disassemble(&raw_bytes, Some(triple)).expect("disassembly failed");
        println!("AArch64 Disassembly:\n{}", disasm);
    }
}
