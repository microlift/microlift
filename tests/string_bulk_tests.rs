//! `REP MOVS`/`REP STOS` bulk lowering against element-wise semantics.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::core::LifterConfig;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

const LIFT_BASE: u64 = 0x1000;

/// Sentinel return address; a `RET` to it exits with `Return`.
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Lift + JIT `bytes` at LIFT_BASE and return the mutated context.
fn run(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::with_config(
        builder,
        LifterConfig {
            exit_trampoline_address: Some(EXIT_TRAMPOLINE),
            ..LifterConfig::default()
        },
    );

    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    if ctx.image_base == 0 {
        ctx.image_base = LIFT_BASE;
    }
    unsafe {
        *(ctx.rsp as *mut u64) = EXIT_TRAMPOLINE;
    }

    lifter.lift(bytes, LIFT_BASE).expect("lifting failed");
    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    let _ = stack;
    ctx
}


/// `REP MOVS`/`REP STOS` runs in 16-byte chunks ahead of the element loop.
/// These pin the element-wise answer at every length and overlap that matters.

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + 3) as u8).collect()
}

fn rep_movs(code: &[u8], buf: &mut [u8], src: usize, dst: usize, count: u64, df: u8) -> RegisterContext {
    let mut ctx = RegisterContext::default();
    ctx.rsi = buf.as_ptr() as u64 + src as u64;
    ctx.rdi = buf.as_mut_ptr() as u64 + dst as u64;
    ctx.rcx = count;
    ctx.df = df;
    run(code, ctx)
}

fn element_wise(buf: &mut [u8], src: usize, dst: usize, bytes: usize) {
    for i in 0..bytes {
        buf[dst + i] = buf[src + i];
    }
}

#[test]
fn rep_movsb_long_copy_matches_element_wise() {
    for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 512, 1000] {
        let mut buf = pattern(4096);
        let mut want = buf.clone();
        element_wise(&mut want, 0, 2048, len);
        let out = rep_movs(&[0xF3, 0xA4], &mut buf, 0, 2048, len as u64, 0);
        assert_eq!(buf, want, "len {len}");
        assert_eq!(out.rcx, 0, "len {len}: rcx drained");
        assert_eq!(out.rsi, buf.as_ptr() as u64 + len as u64, "len {len}: rsi");
        assert_eq!(out.rdi, buf.as_ptr() as u64 + 2048 + len as u64, "len {len}: rdi");
    }
}

#[test]
fn rep_movsb_overlapping_forward_replicates_like_element_wise() {
    for gap in 1..=20usize {
        let mut buf = pattern(256);
        let mut want = buf.clone();
        element_wise(&mut want, 10, 10 + gap, 100);
        rep_movs(&[0xF3, 0xA4], &mut buf, 10, 10 + gap, 100, 0);
        assert_eq!(buf, want, "destination {gap} bytes after the source");
    }
}

#[test]
fn rep_movsb_destination_before_source_matches_element_wise() {
    for gap in 1..=20usize {
        let mut buf = pattern(256);
        let mut want = buf.clone();
        element_wise(&mut want, 50, 50 - gap, 100);
        rep_movs(&[0xF3, 0xA4], &mut buf, 50, 50 - gap, 100, 0);
        assert_eq!(buf, want, "destination {gap} bytes before the source");
    }
}

#[test]
fn rep_movsq_counts_elements_not_bytes() {
    // f3 48 a5 = rep movsq
    let mut buf = pattern(1024);
    let mut want = buf.clone();
    element_wise(&mut want, 8, 512, 7 * 8);
    let out = rep_movs(&[0xF3, 0x48, 0xA5], &mut buf, 8, 512, 7, 0);
    assert_eq!(buf, want);
    assert_eq!(out.rcx, 0);
    assert_eq!(out.rdi, buf.as_ptr() as u64 + 512 + 56);
}

#[test]
fn rep_movsd_with_a_remainder_matches_element_wise() {
    // f3 a5 = rep movsd (dwords)
    let mut buf = pattern(1024);
    let mut want = buf.clone();
    element_wise(&mut want, 4, 600, 11 * 4);
    rep_movs(&[0xF3, 0xA5], &mut buf, 4, 600, 11, 0);
    assert_eq!(buf, want);
}

#[test]
fn rep_movsb_backward_is_untouched_by_the_bulk_path() {
    // DF=1: copies downward from the given addresses.
    let mut buf = pattern(256);
    let mut want = buf.clone();
    for i in 0..40 {
        want[200 - i] = want[100 - i];
    }
    let out = rep_movs(&[0xF3, 0xA4], &mut buf, 100, 200, 40, 1);
    assert_eq!(buf, want);
    assert_eq!(out.rsi, buf.as_ptr() as u64 + 60);
}

#[test]
fn rep_stos_fills_every_element_width() {
    // f3 aa / f3 66 ab / f3 ab / f3 48 ab
    let cases: [(&[u8], usize); 4] = [
        (&[0xF3, 0xAA], 1),
        (&[0xF3, 0x66, 0xAB], 2),
        (&[0xF3, 0xAB], 4),
        (&[0xF3, 0x48, 0xAB], 8),
    ];
    for (code, size) in cases {
        for count in [0usize, 1, 3, 16, 37, 100] {
            let mut buf = vec![0u8; 1024];
            let value: u64 = 0x1122_3344_5566_7788;
            let mut ctx = RegisterContext::default();
            ctx.rax = value;
            ctx.rdi = buf.as_mut_ptr() as u64 + 8;
            ctx.rcx = count as u64;
            let out = run(code, ctx);
            let element = &value.to_le_bytes()[..size];
            for (i, byte) in buf.iter().enumerate() {
                let filled = i >= 8 && i < 8 + count * size;
                let want = if filled { element[(i - 8) % size] } else { 0 };
                assert_eq!(*byte, want, "size {size} count {count} byte {i}");
            }
            assert_eq!(out.rcx, 0);
            assert_eq!(out.rdi, buf.as_ptr() as u64 + 8 + (count * size) as u64);
        }
    }
}
