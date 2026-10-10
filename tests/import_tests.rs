//! Import interception: fast-call (inline `call_indirect`, one `Return` exit) vs slow import (`ImportedFunctionCall` exit, host services and resumes).
//! Both call a host `strlen` from the same `call 0x2000` byte sequence; only `LifterConfig` differs.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::core::abi::CallAbi;
use microlift::function_traits::{DynamicSymbol, trampoline_func};
use microlift::lifter::{ExecutionMode, Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

const LIFT_BASE: u64 = 0x1000;
const IMPORT_VA: u64 = 0x2000;
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// `mov rdi, <ptr> ; call 0x2000 ; ret`.
fn strlen_call_bytes(string_ptr: u64) -> Vec<u8> {
    let mut b = Vec::new();

    b.push(0x48);
    b.push(0xBF);
    b.extend_from_slice(&string_ptr.to_le_bytes());

    let next_ip = LIFT_BASE + 10 + 5;
    let rel32 = (IMPORT_VA as i64 - next_ip as i64) as i32;
    b.push(0xE8);
    b.extend_from_slice(&rel32.to_le_bytes());

    b.push(0xC3);

    b
}

fn lift_and_compile(bytes: &[u8], mut config: LifterConfig) -> (CraneliftBackend, *const u8) {
    // Sentinel declared here, not in each caller's config.
    config.exit_trampoline_address = Some(EXIT_TRAMPOLINE);

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);

    lifter.lift(bytes, LIFT_BASE).expect("lifting failed");
    drop(lifter); // finalises the FunctionBuilder, releasing &mut func

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");
    (backend, code_ptr)
}

fn make_ctx_and_stack() -> (RegisterContext, Vec<u64>) {
    let mut stack = vec![0u64; 64];
    let rsp = stack.as_mut_ptr() as u64 + (stack.len() as u64) * 8 - 8;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };

    let mut ctx = RegisterContext::default();
    ctx.rsp = rsp;
    ctx.image_base = LIFT_BASE;
    ctx.rip = LIFT_BASE; // start of the lifted region

    (ctx, stack)
}

/// Fast-call: full `DynamicSymbol` metadata, so the import is an inline `call_indirect` with no VM exit.
#[test]
fn test_import_fast_call_strlen() {
    let test_string = b"hello, world!\0"; // strlen == 13
    let string_ptr = test_string.as_ptr() as u64;
    let bytes = strlen_call_bytes(string_ptr);

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::sysv_amd64(),
    );

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "expected a clean Return exit; got {:?}",
        ctx.exit_reason
    );
    assert_eq!(ctx.rax, 13, "strlen(\"hello, world!\") should be 13");
}

/// Slow import: exits `ImportedFunctionCall` with `rip` at the `ret`; the host writes `rax` and re-enters.
#[test]
fn test_import_slow_call_strlen_with_resume() {
    let test_string = b"microlift\0"; // strlen == 9
    let string_ptr = test_string.as_ptr() as u64;
    let bytes = strlen_call_bytes(string_ptr);

    let mut config = LifterConfig::default();
    config.register_slow_import(IMPORT_VA);

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };

    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::ImportedFunctionCall as u32,
        "expected ImportedFunctionCall on first run"
    );
    assert_eq!(
        ctx.imported_function_va, IMPORT_VA,
        "imported_function_va should identify the callee"
    );
    assert_eq!(
        ctx.rip,
        LIFT_BASE + 15,
        "rip should point to the instruction after the CALL (the ret)"
    );
    assert_eq!(
        ctx.rdi, string_ptr,
        "rdi should hold the string pointer arg"
    );

    let len = unsafe { host_strlen(ctx.rdi as *const u8) };
    ctx.rax = len; // write return value where the ABI says (RAX)

    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "expected Return exit after resuming"
    );
    assert_eq!(ctx.rax, 9, "strlen(\"microlift\") should be 9");
}

/// `RuntimeAssisted` fast-call via `compile_with_fast_imports`: the call is an `ExternalName::user(0, i)` relocation.
#[test]
fn test_runtime_assisted_fast_call_strlen() {
    let test_string = b"hello, world!\0"; // strlen == 13
    let string_ptr = test_string.as_ptr() as u64;
    let bytes = strlen_call_bytes(string_ptr);

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
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
    drop(lifter);

    let code_ptr = CraneliftBackend::compile_with_fast_imports(
        func,
        &aot_imports,
        settings::Flags::new(settings::builder()),
        &[],
    )
    .expect("JIT compilation with fast imports failed");

    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "expected a clean Return exit (no ImportedFunctionCall); got {}",
        ctx.exit_reason,
    );
    assert_eq!(ctx.rax, 13, "strlen(\"hello, world!\") should be 13");
}

/// Two fast imports check the `ExternalName::user(0, i)` index assignment.
#[test]
fn test_runtime_assisted_fast_call_two_imports() {
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

    const STRCMP_VA: u64 = 0x3000;

    let s1 = b"abc\0";
    let s2 = b"abc\0";
    let p1 = s1.as_ptr() as u64;
    let p2 = s2.as_ptr() as u64;

    let mut bytes: Vec<u8> = Vec::new();
    let encode_call = |target: u64, next_ip: u64| -> Vec<u8> {
        let rel = (target as i64 - next_ip as i64) as i32;
        let mut v = vec![0xE8u8];
        v.extend_from_slice(&rel.to_le_bytes());
        v
    };
    let encode_mov_rdi = |val: u64| -> Vec<u8> {
        let mut v = vec![0x48u8, 0xBF];
        v.extend_from_slice(&val.to_le_bytes());
        v
    };
    let encode_mov_rsi = |val: u64| -> Vec<u8> {
        let mut v = vec![0x48u8, 0xBE];
        v.extend_from_slice(&val.to_le_bytes());
        v
    };

    let mov1 = encode_mov_rdi(p1); // 10 bytes
    let call1_next = LIFT_BASE + 10 + 5;
    let call1 = encode_call(IMPORT_VA, call1_next); // 5 bytes
    let mov2 = encode_mov_rdi(p1); // 10 bytes
    let mov3 = encode_mov_rsi(p2); // 10 bytes
    let call2_next = LIFT_BASE + 10 + 5 + 10 + 10 + 5;
    let call2 = encode_call(STRCMP_VA, call2_next); // 5 bytes

    bytes.extend_from_slice(&mov1);
    bytes.extend_from_slice(&call1);
    bytes.extend_from_slice(&mov2);
    bytes.extend_from_slice(&mov3);
    bytes.extend_from_slice(&call2);
    bytes.push(0xC3); // ret

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..LifterConfig::default()
    };
    config.register_fast_import(
        IMPORT_VA,
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

    let code_ptr = CraneliftBackend::compile_with_fast_imports(
        func,
        &aot_imports,
        settings::Flags::new(settings::builder()),
        &[],
    )
    .expect("JIT compilation with fast imports failed");

    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "expected a clean Return; got {}",
        ctx.exit_reason
    );
    assert_eq!(ctx.rax as i64, 0, "strcmp(\"abc\", \"abc\") should be 0");
}

// The guest ABI belongs to the lifted code, not the host: same function under different `CallAbi`s.

fn call_bytes(target: u64, next_ip: u64) -> Vec<u8> {
    let mut b = vec![0xE8];
    b.extend_from_slice(&((target as i64 - next_ip as i64) as i32).to_le_bytes());
    b
}

/// Microsoft x64 guest: first integer argument in RCX.
#[test]
fn test_import_fast_call_win64_abi() {
    let test_string = b"hello, world!\0"; // strlen == 13
    let mut bytes = vec![0x48, 0xB9];
    bytes.extend_from_slice(&(test_string.as_ptr() as u64).to_le_bytes());
    bytes.extend_from_slice(&call_bytes(IMPORT_VA, LIFT_BASE + 10 + 5));
    bytes.push(0xC3);

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::win64(),
    );

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(
        ctx.rax, 13,
        "win64 fast call must read its argument from RCX"
    );
}

/// Two imports with different conventions in one region (RDI for SysV, RCX for Win64).
#[test]
fn test_import_fast_calls_with_mixed_abis() {
    const WIN64_VA: u64 = 0x3000;
    let long_string = b"hello, world!\0"; // 13
    let short_string = b"abc\0"; // 3

    let mut bytes = vec![0x48, 0xBF];
    bytes.extend_from_slice(&(long_string.as_ptr() as u64).to_le_bytes());
    bytes.extend_from_slice(&call_bytes(IMPORT_VA, LIFT_BASE + 15));
    bytes.extend_from_slice(&[0x48, 0x89, 0xC3]);
    bytes.extend_from_slice(&[0x48, 0xB9]);
    bytes.extend_from_slice(&(short_string.as_ptr() as u64).to_le_bytes());
    bytes.extend_from_slice(&call_bytes(WIN64_VA, LIFT_BASE + 33));
    bytes.extend_from_slice(&[0x48, 0x01, 0xD8]);
    bytes.push(0xC3);

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::sysv_amd64(),
    );
    config.register_fast_import(
        WIN64_VA,
        DynamicSymbol::new("host_strlen_win64", trampoline_func!(host_strlen, _)),
        CallAbi::win64(),
    );

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 13 + 3, "each import must use its own convention");
}

/// Stack-based convention (32-bit cdecl style): every argument pushed by the caller.
#[test]
fn test_import_fast_call_stack_passed_argument() {
    let test_string = b"hello, world!\0"; // strlen == 13

    let mut bytes = vec![0x48, 0xB8];
    bytes.extend_from_slice(&(test_string.as_ptr() as u64).to_le_bytes());
    bytes.push(0x50);
    bytes.extend_from_slice(&call_bytes(IMPORT_VA, LIFT_BASE + 16));
    bytes.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]);
    bytes.push(0xC3);

    let stack_abi = CallAbi {
        name: "all_stack",
        int_arg_regs: Vec::new(),
        fp_arg_regs: Vec::new(),
        ..CallAbi::sysv_amd64()
    };

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        stack_abi,
    );

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 13, "argument must be read from the guest stack");
}

/// An ABI with no stack rule cannot place spilled arguments, so the import demotes to a VM exit.
#[test]
fn test_import_without_stack_rule_is_demoted() {
    let test_string = b"hello, world!\0";
    let bytes = strlen_call_bytes(test_string.as_ptr() as u64);

    let no_arg_regs = CallAbi {
        name: "registerless",
        int_arg_regs: Vec::new(),
        fp_arg_regs: Vec::new(),
        stack_args: None,
        ..CallAbi::sysv_amd64()
    };

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        no_arg_regs,
    );

    let (_backend, code_ptr) = lift_and_compile(&bytes, config);
    let (mut ctx, _stack) = make_ctx_and_stack();
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::ImportedFunctionCall as u32,
        "an unplaceable argument must fall back to the slow exit"
    );
    assert_eq!(ctx.imported_function_va, IMPORT_VA);
}
