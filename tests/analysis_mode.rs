//! `analysis_mode`: verifiable, leaner IR with no PC-dispatch `br_table` and no exit write-back.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use cranelift_codegen::verifier::verify_function;
use microlift::core::LifterConfig;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

fn lift_ir(bytes: &[u8], base: u64, analysis_mode: bool) -> String {
    lift_ir_with(bytes, base, analysis_mode, true)
}

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

    let flags = settings::Flags::new(settings::builder());
    verify_function(&func, &flags).expect("verifier failed");

    func.display().to_string()
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

const MOV_RET: [u8; 8] = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

#[test]
fn analysis_mode_drops_dispatch_switch() {
    let analysis = lift_ir(&MOV_RET, 0x1000, true);

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

    assert!(count(&exec, "store") > 0, "exec IR should write state back");
    assert_eq!(
        count(&analysis, "store"),
        0,
        "analysis IR must not store guest state back:\n{analysis}"
    );

    // Compared against lazy sync off, since the prune already sheds untouched-register stores.
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
    let analysis = lift_ir(&MOV_RET, 0x1000, true);
    assert!(
        analysis.contains("0x2a") || analysis.contains("42"),
        "analysis IR should still contain the mov immediate:\n{analysis}"
    );
}
