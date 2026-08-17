//! `LifterConfig::generated_nan` - what bit pattern an SSE instruction's
//! *generated* NaN gets.
//!
//! Two halves, and both are required. [`GeneratedNan::X86Exact`] (the default)
//! has to keep producing x86's floating indefinite, or the knob has broken the
//! path everything else in the suite depends on;
//! [`GeneratedNan::HostDefault`] has to actually *shed* the fix-up, since a
//! mode that emits the same IR would pass every behavioural test here while
//! buying nothing.
//!
//! What the mode may change is narrow, and the propagation tests are what pin
//! that down: only a NaN the instruction *generated* out of an invalid
//! operation is affected. A NaN that arrived in an operand is propagated with
//! its sign and payload intact under both modes, because the host FPU already
//! does that the way x86 does.

use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

/// x86's single-precision floating indefinite - note the set sign bit, which is
/// the only thing distinguishing it from the AArch64 default NaN.
const F32_INDEFINITE: u32 = 0xFFC0_0000;
/// The double-precision [`F32_INDEFINITE`].
const F64_INDEFINITE: u64 = 0xFFF8_0000_0000_0000;

const F32_INF: u32 = 0x7F80_0000;
const F64_INF: u64 = 0x7FF0_0000_0000_0000;

/// `mulss xmm0, xmm1 ; ret`
const MULSS: &[u8] = &[0xF3, 0x0F, 0x59, 0xC1, 0xC3];
/// `mulsd xmm0, xmm1 ; ret`
const MULSD: &[u8] = &[0xF2, 0x0F, 0x59, 0xC1, 0xC3];
/// `divss xmm0, xmm1 ; ret`
const DIVSS: &[u8] = &[0xF3, 0x0F, 0x5E, 0xC1, 0xC3];
/// `sqrtss xmm0, xmm1 ; ret`
const SQRTSS: &[u8] = &[0xF3, 0x0F, 0x51, 0xC1, 0xC3];
/// `mulps xmm0, xmm1 ; ret`
const MULPS: &[u8] = &[0x0F, 0x59, 0xC1, 0xC3];

/// Run `bytes` with `xmm0`/`xmm1` seeded from raw bit patterns, and return the
/// low 64 bits of `xmm0`.
fn run_lo(bytes: &[u8], mode: GeneratedNan, vector: bool, a: u64, b: u64) -> u64 {
    let test = LiftTest::new(bytes)
        .generated_nan(mode)
        .vector_xmm(vector)
        .reg(|c| {
            c.xmm[0] = [a, 0];
            c.xmm[1] = [b, 0];
        });
    // Under `HostDefault` the lift is deliberately allowed to disagree with
    // hardware on a generated NaN's bits, so the automatic Unicorn differential
    // would be asserting the thing this mode exists to give up.
    let test = match mode {
        GeneratedNan::X86Exact => test,
        GeneratedNan::HostDefault => test.no_unicorn(),
    };
    test.run().ctx.xmm[0][0]
}

/// Both XMM backing representations, since the fix-up has a separate
/// implementation on each (`fp_arith_x86` and `vec_fp_arith_x86`).
const REPRS: [bool; 2] = [false, true];

#[test]
fn x86_exact_generates_the_floating_indefinite() {
    for vector in REPRS {
        // 0 * inf is an invalid operation: the NaN is generated, not
        // propagated, so its bits are the lifter's to get right.
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

        // 0 / 0 likewise.
        let r = run_lo(DIVSS, GeneratedNan::X86Exact, vector, 0, 0);
        assert_eq!(
            r as u32, F32_INDEFINITE,
            "divss 0/0 (vector={vector}) should give x86's indefinite"
        );

        // sqrt of a negative is the unary form of the same rule.
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
    // The weaker guarantee this mode makes: the result is *a* NaN, so every
    // comparison, classification and onward arithmetic behaves the same. Only
    // the bit pattern is the host's rather than x86's.
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
    // A NaN that arrived in an operand is propagated by the host exactly as x86
    // does - quieted, sign and payload intact - so the fix-up never applied to
    // it and removing the fix-up must not change it.
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
    // Everything that does not generate a NaN has to be bit-identical, which is
    // most of the instruction's input space. Seeded random 64-bit patterns
    // cover the ordinary finite cases; the pairs below are the ones that reach
    // an infinity or a zero without being invalid.
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
            // A pair that did generate a NaN is allowed to differ; that is
            // the whole point of the mode. Everything else must not.
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

/// Compile `bytes` for `triple` under `mode` and return the `.text` length.
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

/// The half a behavioural test cannot reach: an implementation that ignored the
/// mode entirely would pass everything above.
///
/// Both targets are checked because the fix-up is emitted from portable
/// Cranelift ops, so it costs instructions on every backend - including x86-64,
/// where the sequence recomputes a NaN the machine was going to produce anyway.
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

/// The default must stay the exact one - this is the knob's whole safety story,
/// and it is one line to assert.
#[test]
fn the_default_is_x86_exact() {
    assert_eq!(
        LifterConfig::default().generated_nan,
        GeneratedNan::X86Exact
    );
}
