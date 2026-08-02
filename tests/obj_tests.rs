use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use microlift::backend::ArtifactBackend;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::core::abi::CallAbi;
use microlift::function_traits::{DynamicSymbol, trampoline_func};
use microlift::lifter::{ExecutionMode, Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;
use object::{Object, ObjectSymbol};

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

    let mut backend = CraneliftBackend::with_flags(flags, &[]).expect("backend init failed");
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

/// A `RuntimeAssisted`-mode function with one fast import compiles to an object
/// that exports `lifted_func` and lists the import symbol as undefined, meaning
/// the object module emitted a named relocation rather than an embedded
/// absolute address.
#[test]
fn test_compile_runtime_assisted_fast_import_has_relocation() {
    unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
        let mut n = 0u64;
        while unsafe { *s.add(n as usize) } != 0 {
            n += 1;
        }
        n
    }

    const LIFT_BASE: u64 = 0x1000;
    const IMPORT_VA: u64 = 0x2000;

    // Build: mov rdi, 0; call IMPORT_VA; ret
    let call_next_ip = LIFT_BASE + 10 + 5;
    let rel32 = (IMPORT_VA as i64 - call_next_ip as i64) as i32;
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&[0x48, 0xBF]); // mov rdi, 0  (REX.W + MOVABS)
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.push(0xE8); // call rel32
    bytes.extend_from_slice(&rel32.to_le_bytes());
    bytes.push(0xC3); // ret

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        ..LifterConfig::default()
    };
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::sysv_amd64(),
    );

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);
    lifter.lift(&bytes, LIFT_BASE).expect("lifting failed");

    let aot_imports: Vec<_> = lifter.aot_imports().to_vec();
    assert_eq!(aot_imports.len(), 1, "should have exactly one AOT import");
    drop(lifter);

    let obj_bytes =
        ArtifactBackend::compile_to_object_with_fast_imports(func, &aot_imports, None, None, &[])
            .expect("failed to compile object with fast imports");
    assert!(!obj_bytes.is_empty());

    let obj = object::File::parse(&*obj_bytes).expect("failed to parse object");

    // Mach-O prepends an underscore to C symbol names; strip it for portability.
    let bare: fn(&str) -> &str = |name| name.strip_prefix('_').unwrap_or(name);

    // `lifted_func` must be a defined export.
    let has_lifted_func = obj
        .symbols()
        .any(|s| !s.is_undefined() && s.name().map(|n| bare(n) == "lifted_func").unwrap_or(false));
    assert!(has_lifted_func, "object must export 'lifted_func'");

    // `host_strlen` must appear as an undefined symbol so the linker can
    // emit a relocation for it - not an embedded absolute address.
    let has_undefined_strlen = obj
        .symbols()
        .any(|s| s.is_undefined() && s.name().map(|n| bare(n) == "host_strlen").unwrap_or(false));
    assert!(
        has_undefined_strlen,
        "object must have 'host_strlen' as an undefined/import symbol; found: {:?}",
        obj.symbols()
            .filter_map(|s| s.name().ok())
            .collect::<Vec<_>>(),
    );
}

/// Two fast imports both appear as undefined symbols in the compiled object,
/// verifying that the `ExternalName::user(0, i)` index ordering correctly maps
/// each symbol to its own relocation entry regardless of `HashMap` iteration order.
#[test]
fn test_compile_runtime_assisted_two_fast_imports_have_relocations() {
    unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
        let mut n = 0u64;
        while unsafe { *s.add(n as usize) } != 0 {
            n += 1;
        }
        n
    }
    unsafe extern "C" fn host_strcmp(s1: *const u8, s2: *const u8) -> i64 {
        let mut i = 0usize;
        loop {
            let c1 = unsafe { *s1.add(i) };
            let c2 = unsafe { *s2.add(i) };
            if c1 != c2 {
                return (c1 as i64) - (c2 as i64);
            }
            if c1 == 0 {
                return 0;
            }
            i += 1;
        }
    }

    const LIFT_BASE: u64 = 0x1000;
    const STRLEN_VA: u64 = 0x2000;
    const STRCMP_VA: u64 = 0x3000;

    // Build: mov rdi, 0; call STRLEN_VA; mov rdi, 0; mov rsi, 0; call STRCMP_VA; ret
    let mut bytes: Vec<u8> = Vec::new();
    let encode_mov_r64 = |rex_opcode: u8, val: u64| -> Vec<u8> {
        let mut v = vec![0x48u8, rex_opcode];
        v.extend_from_slice(&val.to_le_bytes());
        v
    };
    let encode_call = |target: u64, next_ip: u64| -> Vec<u8> {
        let rel = (target as i64 - next_ip as i64) as i32;
        let mut v = vec![0xE8u8];
        v.extend_from_slice(&rel.to_le_bytes());
        v
    };

    bytes.extend_from_slice(&encode_mov_r64(0xBF, 0)); // mov rdi, 0  (10 B)
    let call1_next = LIFT_BASE + 10 + 5;
    bytes.extend_from_slice(&encode_call(STRLEN_VA, call1_next)); // call STRLEN_VA (5 B)
    bytes.extend_from_slice(&encode_mov_r64(0xBF, 0)); // mov rdi, 0  (10 B)
    bytes.extend_from_slice(&encode_mov_r64(0xBE, 0)); // mov rsi, 0  (10 B)
    let call2_next = LIFT_BASE + 10 + 5 + 10 + 10 + 5;
    bytes.extend_from_slice(&encode_call(STRCMP_VA, call2_next)); // call STRCMP_VA (5 B)
    bytes.push(0xC3); // ret

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        ..LifterConfig::default()
    };
    config.register_fast_import(
        STRLEN_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::sysv_amd64(),
    );
    config.register_fast_import(
        STRCMP_VA,
        DynamicSymbol::new(
            "host_strcmp",
            host_strcmp as unsafe extern "C" fn(*const u8, *const u8) -> i64,
        ),
        CallAbi::sysv_amd64(),
    );

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);
    lifter.lift(&bytes, LIFT_BASE).expect("lifting failed");

    let aot_imports: Vec<_> = lifter.aot_imports().to_vec();
    assert_eq!(aot_imports.len(), 2, "should have exactly 2 AOT imports");
    drop(lifter);

    let obj_bytes =
        ArtifactBackend::compile_to_object_with_fast_imports(func, &aot_imports, None, None, &[])
            .expect("failed to compile object with fast imports");

    let obj = object::File::parse(&*obj_bytes).expect("failed to parse object");

    // Mach-O prepends an underscore to C symbol names; strip it for portability.
    let bare: fn(&str) -> &str = |name| name.strip_prefix('_').unwrap_or(name);

    let undefined_bare_names: Vec<&str> = obj
        .symbols()
        .filter(|s| s.is_undefined())
        .filter_map(|s| s.name().ok().map(bare))
        .collect();

    assert!(
        undefined_bare_names.contains(&"host_strlen"),
        "object must list 'host_strlen' as undefined; undefined symbols: {undefined_bare_names:?}",
    );
    assert!(
        undefined_bare_names.contains(&"host_strcmp"),
        "object must list 'host_strcmp' as undefined; undefined symbols: {undefined_bare_names:?}",
    );
}
