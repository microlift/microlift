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
    // The one guarantee this optimization keeps even though it gives up the
    // bit pattern: the value is still a NaN, so downstream arithmetic and
    // comparisons behave the same.
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

/// The half a behavioural test cannot reach: an implementation that ignored
/// the optimization entirely would pass everything above.
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
