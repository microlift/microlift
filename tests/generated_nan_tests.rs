//! `generated_nan`: `X86Exact` must give x86's indefinite for *generated* NaNs; `HostDefault` must shed the fix-up.
//! Operand NaNs propagate with sign and payload intact under both modes.

use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

/// Sign bit set; the only difference from the AArch64 default NaN.
const F32_INDEFINITE: u32 = 0xFFC0_0000;
const F64_INDEFINITE: u64 = 0xFFF8_0000_0000_0000;

const F32_INF: u32 = 0x7F80_0000;
const F64_INF: u64 = 0x7FF0_0000_0000_0000;

const MULSS: &[u8] = &[0xF3, 0x0F, 0x59, 0xC1, 0xC3];
const MULSD: &[u8] = &[0xF2, 0x0F, 0x59, 0xC1, 0xC3];
const DIVSS: &[u8] = &[0xF3, 0x0F, 0x5E, 0xC1, 0xC3];
const SQRTSS: &[u8] = &[0xF3, 0x0F, 0x51, 0xC1, 0xC3];
const MULPS: &[u8] = &[0x0F, 0x59, 0xC1, 0xC3];

fn run_lo(bytes: &[u8], mode: GeneratedNan, vector: bool, a: u64, b: u64) -> u64 {
    let test = LiftTest::new(bytes)
        .generated_nan(mode)
        .vector_xmm(vector)
        .reg(|c| {
            c.xmm[0] = [a, 0];
            c.xmm[1] = [b, 0];
        });
    // HostDefault may differ from hardware on generated NaN bits, so the Unicorn diff is off.
    let test = match mode {
        GeneratedNan::X86Exact => test,
        GeneratedNan::HostDefault => test.no_unicorn(),
    };
    test.run().ctx.xmm[0][0]
}

/// The fix-up has separate scalar and vector implementations.
const REPRS: [bool; 2] = [false, true];

#[test]
fn x86_exact_generates_the_floating_indefinite() {
    for vector in REPRS {
        // Invalid operation: the NaN is generated, not propagated.
        let r = run_lo(MULSS, GeneratedNan::X86Exact, vector, 0, F32_INF as u64);
        assert_eq!(
            r as u32, F32_INDEFINITE,
            "mulss 0*inf (vector={vector}) should give x86's indefinite"
        );

        let r = run_lo(MULSD, GeneratedNan::X86Exact, vector, 0, F64_INF);
        assert_eq!(
            r, F64_INDEFINITE,
            "mulsd 0*inf (vector={vector}) should give x86's indefinite"
        );

        let r = run_lo(DIVSS, GeneratedNan::X86Exact, vector, 0, 0);
        assert_eq!(
            r as u32, F32_INDEFINITE,
            "divss 0/0 (vector={vector}) should give x86's indefinite"
        );

        let neg_one = 0xBF80_0000u32;
        let r = run_lo(
            SQRTSS,
            GeneratedNan::X86Exact,
            vector,
            0,
            u64::from(neg_one),
        );
        assert_eq!(
            r as u32, F32_INDEFINITE,
            "sqrtss -1 (vector={vector}) should give x86's indefinite"
        );
    }
}

#[test]
fn host_default_still_produces_a_nan() {
    // Still a NaN, so comparisons and onward arithmetic agree; only the bit pattern differs.
    for vector in REPRS {
        let r = run_lo(MULSS, GeneratedNan::HostDefault, vector, 0, F32_INF as u64) as u32;
        assert!(
            f32::from_bits(r).is_nan(),
            "mulss 0*inf (vector={vector}) should still be a NaN, got {r:#010x}"
        );

        let r = run_lo(MULSD, GeneratedNan::HostDefault, vector, 0, F64_INF);
        assert!(
            f64::from_bits(r).is_nan(),
            "mulsd 0*inf (vector={vector}) should still be a NaN, got {r:#018x}"
        );
    }
}

#[test]
fn the_mode_only_touches_generated_nans() {
    // Operand NaNs are propagated by the host as x86 does, so removing the fix-up must not change them.
    let payloads: &[u32] = &[
        0x7FC0_1234, // quiet, positive, with a payload
        0xFFC0_5678, // quiet, negative
        0x7F80_0001, // signalling, positive
        0xFF80_0BAD, // signalling, negative
    ];
    let two = 0x4000_0000u32;

    for vector in REPRS {
        for &nan in payloads {
            let exact = run_lo(
                MULSS,
                GeneratedNan::X86Exact,
                vector,
                u64::from(nan),
                u64::from(two),
            );
            let host = run_lo(
                MULSS,
                GeneratedNan::HostDefault,
                vector,
                u64::from(nan),
                u64::from(two),
            );
            assert_eq!(
                exact, host,
                "propagated NaN {nan:#010x} (vector={vector}) diverged between modes"
            );
        }
    }
}

#[test]
fn ordinary_arithmetic_is_unchanged() {
    // Non-NaN-generating inputs must be bit-identical.
    let mut rng = Rng::new(0xFC00);
    let mut pairs: Vec<(u64, u64)> = (0..64).map(|_| (rng.next_u64(), rng.next_u64())).collect();
    pairs.extend([
        (u64::from(F32_INF), u64::from(F32_INF)), // inf * inf = inf
        (u64::from(F32_INF), 0x3F80_0000),        // inf * 1 = inf
        (0, 0),                                   // 0 * 0 = 0
        (0x8000_0000, 0x3F80_0000),               // -0 * 1 = -0
    ]);

    for vector in REPRS {
        for (a, b) in &pairs {
            let exact = run_lo(MULSS, GeneratedNan::X86Exact, vector, *a, *b);
            let host = run_lo(MULSS, GeneratedNan::HostDefault, vector, *a, *b);
            // Only pairs that generate a NaN may differ.
            let generated = f32::from_bits(exact as u32).is_nan() && exact as u32 == F32_INDEFINITE;
            if !generated {
                assert_eq!(
                    exact, host,
                    "mulss({a:#x}, {b:#x}) (vector={vector}) diverged outside a generated NaN"
                );
            }
        }
    }
}

fn code_len(bytes: &[u8], mode: GeneratedNan, triple: &str) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        generated_nan: mode,
        ..Default::default()
    };
    let mut func = lift_to_function::<X86_64>(bytes, 0x1000, config).expect("lift");
    func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_to_raw(func, Some(triple), None)
        .expect("compile")
        .len()
}

/// Behavioural tests can't tell whether the mode is ignored; code size can (the fix-up costs on every backend).
#[test]
fn host_default_sheds_the_fixup() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for (name, bytes) in [("mulss", MULSS), ("mulps", MULPS), ("sqrtss", SQRTSS)] {
            let exact = code_len(bytes, GeneratedNan::X86Exact, triple);
            let host = code_len(bytes, GeneratedNan::HostDefault, triple);
            assert!(
                host < exact,
                "{name} on {triple}: HostDefault ({host} bytes) should be smaller \
                 than X86Exact ({exact} bytes)"
            );
        }
    }
}

/// The default must stay exact.
#[test]
fn the_default_is_x86_exact() {
    assert_eq!(
        LifterConfig::default().generated_nan,
        GeneratedNan::X86Exact
    );
}
