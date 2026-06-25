//! Tests for the import interception feature.
//!
//! Both tests call into a host-side `strlen` reimplementation (no libc) from
//! recompiled x86-64 code.  The same "call strlen at 0x2000" byte sequence is
//! used in both tests; only the `LifterConfig` differs:
//!
//! - **Fast-call**: `strlen` is registered with full `DynamicSymbol` metadata.
//!   The lifter emits an inline `call_indirect`; the test completes in one shot.
//! - **Exit-based**: `strlen` is registered as a slow import.  The lifted code
//!   exits with `ImportedFunctionCall`; the test reads the args from the context,
//!   calls `strlen` manually, writes the result back, and resumes.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::function_traits::{trampoline_func, DynamicSymbol};
use microlift::lifter::{ExecutionMode, Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

/// Pure Rust reimplementation of C `strlen` - counts bytes until the NUL
/// terminator.  No libc dependency.
unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

/// VA of the first lifted instruction.
const LIFT_BASE: u64 = 0x1000;
/// VA used for the fake "strlen import" stub.
const IMPORT_VA: u64 = 0x2000;
/// The exit-trampoline sentinel used by the lifter (always this value).
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Builds the x86-64 byte sequence for:
///
/// ```asm
/// 0x1000:  mov rdi, <string_ptr>  ; 10 bytes (REX.W + B8/BF + imm64)
/// 0x100A:  call 0x2000            ;  5 bytes (E8 + rel32)
/// 0x100F:  ret                    ;  1 byte
/// ```
///
/// The `CALL` target is `IMPORT_VA` (`0x2000`); the relative displacement is
/// computed from the known next-IP (`0x100F`).
fn strlen_call_bytes(string_ptr: u64) -> Vec<u8> {
    let mut b = Vec::new();

    // mov rdi, imm64  (48 BF <8 bytes LE>)
    b.push(0x48);
    b.push(0xBF);
    b.extend_from_slice(&string_ptr.to_le_bytes());

    // call rel32  - rel32 = target - next_ip
    //   next_ip  = 0x1000 + 10 (mov) + 5 (call) = 0x100F
    //   target   = 0x2000
    //   rel32    = 0x2000 - 0x100F = 0xFF1
    let next_ip = LIFT_BASE + 10 + 5;
    let rel32 = (IMPORT_VA as i64 - next_ip as i64) as i32;
    b.push(0xE8);
    b.extend_from_slice(&rel32.to_le_bytes());

    // ret
    b.push(0xC3);

    b
}

/// Lift `bytes` with the given config, compile with the JIT backend, and
/// return the live backend (which keeps the code allocation alive) together
/// with the raw code pointer.
fn lift_and_compile(bytes: &[u8], config: LifterConfig) -> (CraneliftBackend, *const u8) {
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

/// Allocate a 512-byte guest stack, push the exit trampoline, and return a
/// ready-to-use `RegisterContext` together with the backing allocation (the
/// `Vec` must stay live for the duration of the test).
fn make_ctx_and_stack() -> (RegisterContext, Vec<u64>) {
    let mut stack = vec![0u64; 64];
    // Stack grows downward; point RSP at the topmost slot.
    let rsp = stack.as_mut_ptr() as u64 + (stack.len() as u64) * 8 - 8;
    // Pre-push the exit trampoline so the first RET exits cleanly.
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };

    let mut ctx = RegisterContext::default();
    ctx.rsp = rsp;
    ctx.image_base = LIFT_BASE;
    ctx.rip = LIFT_BASE; // start of the lifted region

    (ctx, stack)
}

/// Fast-call path: `strlen` is registered with full `DynamicSymbol` metadata.
///
/// The lifter emits a Cranelift `call_indirect` inline (SysV AMD64 ABI):
/// - Reads the string pointer from `RDI`.
/// - Calls `host_strlen` directly.
/// - Writes the result into `RAX`.
///
/// Execution reaches `ret` without any VM exit for the import, producing a
/// single `Return` exit at the end.
#[test]
fn test_import_fast_call_strlen() {
    let test_string = b"hello, world!\0"; // strlen == 13
    let string_ptr = test_string.as_ptr() as u64;
    let bytes = strlen_call_bytes(string_ptr);

    let mut config = LifterConfig::default();
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
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

/// Exit-based path: `strlen` is registered as a slow import (no ABI metadata).
///
/// Round 1: the lifter exits with `ImportedFunctionCall`.
///   - `ctx.imported_function_va` == `IMPORT_VA`.
///   - `ctx.rip` == 0x100F (the `ret` instruction - the resume point).
///   - `ctx.rdi` holds the string pointer (synced by `sync_to_context_selective`).
///
/// The host reads `ctx.rdi`, calls `host_strlen` itself, writes the result
/// into `ctx.rax`, and invokes the lifted function a second time.
///
/// Round 2: the dispatch switch routes to the `ret` block (RIP == 0x100F).
///   `ret` pops the exit trampoline -> `Return` exit.  `ctx.rax` still holds
///   the value the host wrote.
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
    // RIP == 0x100F: the `ret` instruction, i.e. the resume point.
    assert_eq!(
        ctx.rip,
        LIFT_BASE + 15,
        "rip should point to the instruction after the CALL (the ret)"
    );
    // RDI was synced by sync_to_context_selective and holds the string ptr.
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

/// `RuntimeAssisted` fast-call path via `CraneliftBackend::compile_with_fast_imports`.
///
/// In `RuntimeAssisted` mode the lifter emits `call(func_ref, ...)` backed by
/// `ExternalName::user(0, 0)` rather than `call_indirect` with a baked-in
/// pointer.  `compile_with_fast_imports` creates a fresh JIT module that:
///   1. Registers the host symbol address so the linker can patch the call site.
///   2. Declares the import with the same `FuncId` order used in the IR.
///
/// The test asserts that execution completes with a single `Return` exit -
/// no `ImportedFunctionCall` exits - and that `rax` holds the correct length,
/// proving that the call went through `host_strlen` directly.
#[test]
fn test_runtime_assisted_fast_call_strlen() {
    let test_string = b"hello, world!\0"; // strlen == 13
    let string_ptr = test_string.as_ptr() as u64;
    let bytes = strlen_call_bytes(string_ptr);

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        ..LifterConfig::default()
    };
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
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

    // Capture the ordered import list before dropping the lifter.
    let aot_imports: Vec<_> = lifter.aot_imports().to_vec();
    drop(lifter);

    let code_ptr = CraneliftBackend::compile_with_fast_imports(
        func,
        &aot_imports,
        settings::Flags::new(settings::builder()),
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

/// Same as above but with two registered fast imports to verify that the
/// `ExternalName::user(0, i)` index assignment correctly maps both symbols.
///
/// The lifted code calls `host_strlen` then immediately `host_strcmp`, and the
/// final result in `rax` is the `strcmp` return value.
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

    // Build:
    //   mov rdi, p1
    //   call IMPORT_VA     ; strlen(p1) -> rax (unused, just exercises the call)
    //   mov rdi, p1
    //   mov rsi, p2
    //   call STRCMP_VA     ; strcmp(p1, p2) -> rax
    //   ret
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

    // ip=LIFT_BASE
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
        ..LifterConfig::default()
    };
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
    );
    config.register_fast_import(
        STRCMP_VA,
        DynamicSymbol::new(
            "host_strcmp",
            host_strcmp as unsafe extern "C" fn(*const u8, *const u8) -> i64,
        ),
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
    // strcmp("abc", "abc") == 0
    assert_eq!(ctx.rax as i64, 0, "strcmp(\"abc\", \"abc\") should be 0");
}
