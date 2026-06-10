//! Tests for [`LifterConfig::direct_native_call`].
//!
//! The lifter, with this option enabled, replaces what would normally be a
//! VM exit (slow import, out-of-region direct call, unregistered indirect
//! call) with a call to an x86-64 trampoline that syncs the virtual CPU
//! state onto the real CPU, calls the native target, then writes the
//! post-call state back into the `RegisterContext`.
//!
//! All four call shapes are exercised end-to-end against a pure-Rust
//! re-implementation of `strlen`:
//!
//! - Indirect `CALL reg`         (target known only at runtime)
//! - Indirect `CALL [mem]`       (target read from memory at runtime)
//! - Direct `CALL rel32` out of region (static target, not a registered import)
//! - Direct `CALL rel32` to a registered slow import
//!
//! A final test verifies that [`LifterConfig::set_direct_native_call`] is
//! gated on host architecture.

#![cfg(target_arch = "x86_64")]

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::lifter::{Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;

unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

/// Mutates one of its caller-saved arg registers - used to confirm that the
/// trampoline writes register changes back into the `RegisterContext`.
unsafe extern "C" fn host_add(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

const LIFT_BASE: u64 = 0x1000;
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Lift `bytes` with `config`, JIT-compile, and return both the live backend
/// (which owns the code allocation) and the entry pointer.
fn lift_and_compile(
    bytes: &[u8],
    lift_base: u64,
    config: LifterConfig,
) -> (CraneliftBackend, *const u8) {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);

    lifter.lift(bytes, lift_base).expect("lifting failed");
    drop(lifter);

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    let code_ptr = backend.compile(func).expect("compilation failed");
    (backend, code_ptr)
}

/// Allocate a 16-byte-aligned 4 KiB guest stack and pre-push the exit
/// trampoline so a single `ret` at the end of the lifted region exits cleanly.
fn make_ctx_and_stack(lift_base: u64) -> (RegisterContext, Vec<u64>) {
    let mut stack = vec![0u64; 512];
    let top = stack.as_mut_ptr() as u64 + (stack.len() as u64) * 8;
    // Align rsp so that after pushing the exit trampoline below, rsp % 16 == 0
    // (the standard SysV "call site" alignment that the trampoline's own
    // `call qword ptr [rsp + 8]` depends on).
    let mut rsp = top & !0xF;
    rsp -= 8;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };

    let mut ctx = RegisterContext::default();
    ctx.rsp = rsp;
    ctx.image_base = lift_base;
    ctx.rip = lift_base;
    (ctx, stack)
}

/// Build `mov rax, imm64` (10 bytes: REX.W + B8 + imm64).
fn mov_rax_imm64(imm: u64) -> [u8; 10] {
    let mut b = [0u8; 10];
    b[0] = 0x48;
    b[1] = 0xB8;
    b[2..10].copy_from_slice(&imm.to_le_bytes());
    b
}

/// Build `mov rdi, imm64` (10 bytes: REX.W + BF + imm64).
fn mov_rdi_imm64(imm: u64) -> [u8; 10] {
    let mut b = [0u8; 10];
    b[0] = 0x48;
    b[1] = 0xBF;
    b[2..10].copy_from_slice(&imm.to_le_bytes());
    b
}

/// Build `mov rsi, imm64` (10 bytes: REX.W + BE + imm64).
fn mov_rsi_imm64(imm: u64) -> [u8; 10] {
    let mut b = [0u8; 10];
    b[0] = 0x48;
    b[1] = 0xBE;
    b[2..10].copy_from_slice(&imm.to_le_bytes());
    b
}

// indirect CALL via register, routed through the trampoline.
#[test]
fn test_direct_native_call_indirect_register() {
    let s = b"hello, world!\0"; // strlen == 13
    let s_ptr = s.as_ptr() as u64;
    let strlen_addr = host_strlen as *const () as u64;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&mov_rax_imm64(strlen_addr)); // rax = &host_strlen
    bytes.extend_from_slice(&mov_rdi_imm64(s_ptr)); //       rdi = s
    bytes.extend_from_slice(&[0xFF, 0xD0]); //               call rax
    bytes.push(0xC3); //                                    ret

    let mut config = LifterConfig::default();
    config
        .set_direct_native_call(true)
        .expect("direct_native_call should succeed on x86-64");

    let (_backend, code_ptr) = lift_and_compile(&bytes, LIFT_BASE, config);
    let (mut ctx, _stack) = make_ctx_and_stack(LIFT_BASE);

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "expected Return; the trampoline must not generate any VM exit"
    );
    assert_eq!(ctx.rax, 13, "strlen of test string must reach RAX");
}

// indirect CALL via memory operand, routed through the trampoline.
#[test]
fn test_direct_native_call_indirect_memory() {
    let s = b"microlift\0"; // strlen == 9
    let s_ptr = s.as_ptr() as u64;

    // A pointer-to-pointer indirection that the lifted code will dereference
    // via `call qword ptr [rax]`.
    let strlen_addr = host_strlen as *const () as u64;
    let slot: Box<u64> = Box::new(strlen_addr);
    let slot_ptr = Box::into_raw(slot) as u64;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&mov_rax_imm64(slot_ptr)); // rax = &slot
    bytes.extend_from_slice(&mov_rdi_imm64(s_ptr)); //    rdi = s
    bytes.extend_from_slice(&[0xFF, 0x10]); //            call qword ptr [rax]
    bytes.push(0xC3); //                                 ret

    let mut config = LifterConfig::default();
    config.set_direct_native_call(true).unwrap();

    let (_backend, code_ptr) = lift_and_compile(&bytes, LIFT_BASE, config);
    let (mut ctx, _stack) = make_ctx_and_stack(LIFT_BASE);

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    // Free the slot now that the lifted code has consumed it.
    let _ = unsafe { Box::from_raw(slot_ptr as *mut u64) };

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 9);
}

// Test 3: direct CALL rel32 to a static out-of-region target.
//
// We choose LIFT_BASE close enough to the real `host_add` function that
// rel32 reaches it.  `host_add` lets us check that two argument registers
// (rdi, rsi) are correctly delivered and that the return value (rax) is
// written back into the context.
#[test]
fn test_direct_native_call_out_of_region_direct() {
    let add_addr = host_add as *const () as u64;

    // Lay out a code region 0x1000 bytes below `host_add` so that the
    // `call rel32` displacement always fits in i32 regardless of ASLR.
    let lift_base = add_addr.wrapping_sub(0x1000);

    // mov rdi, 100
    // mov rsi, 23
    // call rel32 (-> host_add)
    // ret
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&mov_rdi_imm64(100));
    bytes.extend_from_slice(&mov_rsi_imm64(23));
    let call_ip = lift_base + bytes.len() as u64;
    let next_ip = call_ip + 5;
    let rel32 = (add_addr as i64 - next_ip as i64) as i32;
    bytes.push(0xE8);
    bytes.extend_from_slice(&rel32.to_le_bytes());
    bytes.push(0xC3);

    let mut config = LifterConfig::default();
    config.set_direct_native_call(true).unwrap();

    let (_backend, code_ptr) = lift_and_compile(&bytes, lift_base, config);
    let (mut ctx, _stack) = make_ctx_and_stack(lift_base);

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "out-of-region direct call must not VM-exit when direct_native_call is on"
    );
    assert_eq!(ctx.rax, 123, "return value of host_add(100, 23)");
    // rdi/rsi are SysV caller-saved — the callee may leave anything in them.
    // We only assert RAX, which carries the return value.
}

// a CALL rel32 to a registered slow import, routed through the
// trampoline.  This proves that `register_slow_import` + `direct_native_call`
// composes correctly: the lifter picks the trampoline instead of the VM
// exit because the slow import has no fast-call metadata.
#[test]
fn test_direct_native_call_slow_import_path() {
    let s = b"abc\0"; // strlen == 3
    let s_ptr = s.as_ptr() as u64;
    let strlen_addr = host_strlen as *const () as u64;
    let lift_base = strlen_addr.wrapping_sub(0x1000);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&mov_rdi_imm64(s_ptr));
    let call_ip = lift_base + bytes.len() as u64;
    let next_ip = call_ip + 5;
    let rel32 = (strlen_addr as i64 - next_ip as i64) as i32;
    bytes.push(0xE8);
    bytes.extend_from_slice(&rel32.to_le_bytes());
    bytes.push(0xC3);

    let mut config = LifterConfig::default();
    config.register_slow_import(strlen_addr);
    config.set_direct_native_call(true).unwrap();

    let (_backend, code_ptr) = lift_and_compile(&bytes, lift_base, config);
    let (mut ctx, _stack) = make_ctx_and_stack(lift_base);

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);

    assert_eq!(
        ctx.exit_reason,
        ExitReason::Return as u32,
        "slow import + direct_native_call should resolve inline with no VM exit"
    );
    assert_eq!(ctx.rax, 3);
}

// Test 5: feature-gate check.  `set_direct_native_call(true)` must succeed
// on x86-64 (this whole file is cfg'd to x86-64 so we only positive-test
// here).
#[test]
fn test_set_direct_native_call_ok_on_x86_64() {
    let mut config = LifterConfig::default();
    assert!(config.set_direct_native_call(true).is_ok());
    assert!(config.direct_native_call);
    assert!(config.set_direct_native_call(false).is_ok());
    assert!(!config.direct_native_call);
}
