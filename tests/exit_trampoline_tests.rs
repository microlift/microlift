use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::exit::TransferKind;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::EXIT_TRAMPOLINE;
use microlift::testkit::prelude::*;

const MOV_RET: [u8; 8] = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

const ELSEWHERE: u64 = 0x4321_0000;

#[test]
fn a_declared_sentinel_still_returns() {
    let out = LiftTest::new(&MOV_RET)
        .exit_trampoline(Some(EXIT_TRAMPOLINE))
        .ret_addr(RetAddr::Sentinel(EXIT_TRAMPOLINE))
        .run();

    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rip, EXIT_TRAMPOLINE);
    assert_eq!(out.ctx.rax, 42);
}

/// With a sentinel declared, any other out-of-region `RET` is still `UnknownJump`.
#[test]
fn a_declared_sentinel_does_not_swallow_other_returns() {
    let out = LiftTest::new(&MOV_RET)
        .exit_trampoline(Some(EXIT_TRAMPOLINE))
        .ret_addr(RetAddr::Sentinel(ELSEWHERE))
        .run();

    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(out.ctx.exit_code, TransferKind::Return as u32);
    assert_eq!(out.ctx.rip, ELSEWHERE);
}

/// Default: no sentinel, every out-of-region `RET` is `UnknownJump` (the exit block linking resolves).
#[test]
fn without_a_sentinel_a_ret_is_an_unknown_jump() {
    for ret_to in [ELSEWHERE, EXIT_TRAMPOLINE] {
        let out = LiftTest::new(&MOV_RET)
            .exit_trampoline(None)
            .ret_addr(RetAddr::Sentinel(ret_to))
            .run();

        assert_eq!(
            out.exit,
            ExitReason::UnknownJump,
            "returning to {ret_to:#x}"
        );
        assert_eq!(out.ctx.exit_code, TransferKind::Return as u32);
        assert_eq!(out.ctx.rip, ret_to);
        assert_eq!(out.ctx.rax, 42, "the region still ran");
    }
}

/// An in-region `RET` target dispatches as before.
#[test]
fn an_in_region_ret_still_dispatches_in_region() {
    let bytes = [
        0xE8, 0x08, 0x00, 0x00, 0x00, // 0x1000: call 0x100D
        0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, // 0x1005: mov rbx, 7
        0xC3, // 0x100C: ret
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // 0x100D: mov rax, 42
        0xC3, // 0x1014: ret -> 0x1005, in region
    ];

    let out = LiftTest::new(&bytes)
        .exit_trampoline(None)
        .ret_addr(RetAddr::Sentinel(ELSEWHERE))
        .run();

    assert_eq!(out.ctx.rax, 42, "the callee ran");
    assert_eq!(out.ctx.rbx, 7, "the inner ret dispatched back in region");
    assert_eq!(
        out.exit,
        ExitReason::UnknownJump,
        "only the outer ret exits"
    );
    assert_eq!(out.ctx.rip, ELSEWHERE);
}

#[test]
fn no_sentinel_sheds_the_compare() {
    for triple in ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"] {
        let with = code_len(&MOV_RET, Some(EXIT_TRAMPOLINE), triple);
        let without = code_len(&MOV_RET, None, triple);

        assert!(
            without < with,
            "{triple}: no sentinel should shed code, got {without} bytes against {with}"
        );
    }
}

fn code_len(bytes: &[u8], trampoline: Option<u64>, triple: &str) -> usize {
    const SAMPLES: usize = 8;

    (0..SAMPLES)
        .map(|_| {
            let config = LifterConfig {
                exit_trampoline_address: trampoline,
                ..Default::default()
            };
            let mut code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
            // Override the baked host convention so this measures lowering, not the target ABI.
            code.func.signature.call_conv = cranelift_codegen::isa::CallConv::SystemV;
            ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
                .expect("compile")
                .len()
        })
        .min()
        .expect("at least one sample")
}
