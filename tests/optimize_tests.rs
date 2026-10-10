use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::optimize::{Optimization, Optimizations};
use microlift::core::{GeneratedNan, LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;

const F32_INF: u32 = 0x7F80_0000;

/// `mulss xmm0, xmm1 ; ret`
const MULSS: &[u8] = &[0xF3, 0x0F, 0x59, 0xC1, 0xC3];
/// `mulps xmm0, xmm1 ; ret`
const MULPS: &[u8] = &[0x0F, 0x59, 0xC1, 0xC3];
/// `sqrtss xmm0, xmm1 ; ret`
const SQRTSS: &[u8] = &[0xF3, 0x0F, 0x51, 0xC1, 0xC3];

const REPRS: [bool; 2] = [false, true];

#[test]
fn the_default_has_nothing_enabled() {
    assert_eq!(LifterConfig::default().optimizations, Optimizations::none());
    assert!(!Optimizations::none().allows(Optimization::UnsafeImpreciseNaN));
}

#[test]
fn unsafe_imprecise_nan_still_produces_a_nan() {
    // Still a NaN, so downstream arithmetic behaves the same.
    for vector in REPRS {
        let r = LiftTest::new(MULSS)
            .generated_nan(GeneratedNan::X86Exact)
            .optimizations(Optimizations::none().with(Optimization::UnsafeImpreciseNaN))
            .vector_xmm(vector)
            .no_unicorn()
            .reg(|c| {
                c.xmm[0] = [0, 0];
                c.xmm[1] = [F32_INF as u64, 0];
            })
            .run()
            .ctx
            .xmm[0][0] as u32;
        assert!(
            f32::from_bits(r).is_nan(),
            "mulss 0*inf (vector={vector}) should still be a NaN, got {r:#010x}"
        );
    }
}

/// Compile `bytes` and return the `.text` length.
fn code_len(bytes: &[u8], optimizations: Optimizations, triple: &str) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        optimizations,
        ..Default::default()
    };
    let mut func = lift_to_function::<X86_64>(bytes, 0x1000, config).expect("lift");
    func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_to_raw(func, Some(triple), None)
        .expect("compile")
        .len()
}

/// Size check: an implementation ignoring the optimization would pass the tests above.
#[test]
fn unsafe_imprecise_nan_sheds_the_fixup() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for (name, bytes) in [("mulss", MULSS), ("mulps", MULPS), ("sqrtss", SQRTSS)] {
            let off = code_len(bytes, Optimizations::none(), triple);
            let on = code_len(
                bytes,
                Optimizations::none().with(Optimization::UnsafeImpreciseNaN),
                triple,
            );
            assert!(
                on < off,
                "{name} on {triple}: UnsafeImpreciseNaN ({on} bytes) should be smaller \
                 than disabled ({off} bytes)"
            );
        }
    }
}

/// `div bl ; ret` - AL = AX / BL, AH = AX % BL.
const DIV_BL: &[u8] = &[0xF6, 0xF3, 0xC3];
/// `div rcx ; ret` - RDX:RAX / RCX, the 128-bit software-loop width.
const DIV_RCX: &[u8] = &[0x48, 0xF7, 0xF1, 0xC3];

fn no_de() -> Optimizations {
    Optimizations::none().with(Optimization::UnsafeImpreciseDivisionExceptions)
}

#[test]
fn a_valid_division_is_unaffected_by_dropping_the_de_guards() {
    // Guards only reject, so removing them leaves non-faulting divisions unchanged.
    for opts in [Optimizations::none(), no_de()] {
        let out = LiftTest::new(DIV_RCX)
            .optimizations(opts)
            .reg(|c| {
                c.rax = 43;
                c.rdx = 0;
                c.rcx = 7;
            })
            .run();
        assert_eq!(out.ctx.rax, 6, "{opts:?}");
        assert_eq!(out.ctx.rdx, 1, "{opts:?}");
        assert_eq!(out.ctx.exit_reason, ExitReason::Return as u32, "{opts:?}");
    }
}

#[test]
fn a_quotient_overflow_faults_by_default_and_wraps_with_the_optimization() {
    // AX/BL = 0x2000 overflows AL: the guard-2 case, the only #DE safe to execute without guards.
    let seed = |c: &mut RegisterContext| {
        c.rax = 0x2000;
        c.rbx = 1;
    };

    let faulted = LiftTest::new(DIV_BL)
        .optimizations(Optimizations::none())
        .no_unicorn()
        .reg(seed)
        .run();
    assert_eq!(
        faulted.ctx.exit_reason,
        ExitReason::DivideError as u32,
        "the overflow guard should exit with #DE"
    );
    assert_eq!(faulted.ctx.rax, 0x2000, "a fault commits no result");

    let wrapped = LiftTest::new(DIV_BL)
        .optimizations(no_de())
        .no_unicorn()
        .reg(seed)
        .run();
    assert_eq!(
        wrapped.ctx.exit_reason,
        ExitReason::Return as u32,
        "with the guards gone the same division must not fault"
    );
    // AL = quotient, AH = remainder; rest of RAX untouched.
    assert_eq!(wrapped.ctx.rax & 0xFFFF, 0x0000);
}

/// Size check, as above.
#[test]
fn unsafe_imprecise_division_exceptions_sheds_the_guards() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for (name, bytes) in [("div bl", DIV_BL), ("div rcx", DIV_RCX)] {
            let off = code_len(bytes, Optimizations::none(), triple);
            let on = code_len(bytes, no_de(), triple);
            assert!(
                on < off,
                "{name} on {triple}: UnsafeImpreciseDivisionExceptions ({on} bytes) \
                 should be smaller than disabled ({off} bytes)"
            );
        }
    }
}

#[test]
fn without_undoes_with() {
    let opts = Optimizations::none()
        .with(Optimization::UnsafeImpreciseNaN)
        .without(Optimization::UnsafeImpreciseNaN);
    assert_eq!(opts, Optimizations::none());
}

#[test]
fn only_replaces_the_whole_set() {
    let opts = Optimizations::all().only(&[Optimization::UnsafeImpreciseNaN]);
    assert!(opts.allows(Optimization::UnsafeImpreciseNaN));
    assert_eq!(opts.active().count(), 1);
}
