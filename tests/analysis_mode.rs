//! Tests for `LifterConfig::analysis_mode` - the "inspection IR" lift mode used
//! by static-analysis consumers (e.g. Microreverse).
//!
//! Analysis mode must (a) still produce a verifiable Cranelift function, and
//! (b) be substantially leaner than execution IR: no entry PC-dispatch
//! `br_table`, and no per-exit register write-back stores.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use cranelift_codegen::verifier::verify_function;
use microlift::core::LifterConfig;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

/// Lift `bytes` at `base` and return the Cranelift IR text, optionally in
/// analysis mode.
fn lift_ir(bytes: &[u8], base: u64, analysis_mode: bool) -> String {
    lift_ir_with(bytes, base, analysis_mode, true)
}

/// [`lift_ir`], with control over [`LifterConfig::lazy_register_sync`].
fn lift_ir_with(bytes: &[u8], base: u64, analysis_mode: bool, lazy: bool) -> String {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);

    let config = LifterConfig {
        analysis_mode,
        lazy_register_sync: lazy,
        ..Default::default()
    };
    let mut lifter = Lifter::with_config(builder, config);
    lifter.lift(bytes, base).expect("lifting failed");
    drop(lifter);

    // The produced function must verify in either mode.
    let flags = settings::Flags::new(settings::builder());
    verify_function(&func, &flags).expect("verifier failed");

    func.display().to_string()
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// mov rax, 42 ; ret
const MOV_RET: [u8; 8] = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

#[test]
fn analysis_mode_drops_dispatch_switch() {
    let analysis = lift_ir(&MOV_RET, 0x1000, true);

    // Analysis IR enters directly with an unconditional jump and never builds a
    // PC-dispatch jump table.
    assert!(
        !analysis.contains("br_table"),
        "analysis IR must not emit a dispatch table:\n{analysis}"
    );
    assert!(
        analysis.contains("jump block"),
        "analysis IR should enter via a direct jump:\n{analysis}"
    );
}

#[test]
fn analysis_mode_omits_context_writeback() {
    let exec = lift_ir(&MOV_RET, 0x1000, false);
    let analysis = lift_ir(&MOV_RET, 0x1000, true);

    // Execution IR writes every register/flag back to the context on exit;
    // analysis IR performs no write-back at all.
    assert!(count(&exec, "store") > 0, "exec IR should write state back");
    assert_eq!(
        count(&analysis, "store"),
        0,
        "analysis IR must not store guest state back:\n{analysis}"
    );

    // And the write-back is most of what execution IR carries. Measured
    // against a lift with the lazy-sync prune off, because with it on the
    // execution IR has already shed the stores for registers this snippet
    // never wrote - which is a different saving, not this one. Line counts
    // exclude the alias-region preamble, which both modes declare.
    let eager = lift_ir_with(&MOV_RET, 0x1000, false, false);
    assert!(
        body_lines(&analysis) * 2 < body_lines(&eager),
        "analysis IR ({} lines) should be much smaller than eager exec IR ({} lines)",
        body_lines(&analysis),
        body_lines(&eager)
    );
}

fn body_lines(ir: &str) -> usize {
    ir.lines()
        .filter(|l| !l.trim_start().starts_with("region"))
        .count()
}

#[test]
fn analysis_mode_preserves_semantics_instructions() {
    // The actual computation (loading rax with 42) must survive: an iconst of
    // 0x2a (42) should still appear.
    let analysis = lift_ir(&MOV_RET, 0x1000, true);
    assert!(
        analysis.contains("0x2a") || analysis.contains("42"),
        "analysis IR should still contain the mov immediate:\n{analysis}"
    );
}
