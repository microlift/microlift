//! Specialized-vs-portable equivalence for the target-directed fast paths.
//!
//! The portable expansion is the authority on semantics; the sequences in
//! `semantics/sse_spec.rs` are a lowering choice that must not change a single
//! result bit. Every instruction with a specialization is run here under each
//! [`SpecTarget`] and under none, from identical seeded pseudo-random state,
//! and the full architectural context is compared.
//!
//! This is the regression net for the whole feature. Unlike the jump-table
//! recovery - which is safe by construction, because a wrong guess only ever
//! lifts unreachable code - a wrong specialization silently computes the wrong
//! answer. Adding one without listing its encoding here leaves it unproven.
//!
//! Three properties are checked, and they are not the same property:
//!
//! 1. **Results agree** with the unspecialized vector path, and with the scalar
//!    `(lo, hi)` path underneath it (`agrees_with_portable`), including on the
//!    saturation and NaN edges hardware and x86 disagree about.
//! 2. **The compiled code shrinks**, per instruction and per target, exactly
//!    where the support table says it should and nowhere else
//!    (`specializations_shrink_the_compiled_code`). Without this a
//!    silently-inert `try_*` would pass (1) perfectly.
//! 3. **Each target's IR lowers** for the machine it names, cross-compiled
//!    (`each_target_lowers_its_own_ir`). This is where an opcode with no rule on
//!    one backend would be caught.

use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

/// A guest instruction with a specialization: its encoding, the [`Spec`] that
/// covers it, and the targets on which that specialization is expected to
/// shrink the compiled code.
struct Case {
    name: String,
    bytes: Vec<u8>,
    spec: Spec,
    /// Targets where this instruction must get *smaller*. On any other target
    /// it must be byte-identical to the portable lowering - a specialization
    /// that fires and does not pay is a bug, not a wash.
    improves: &'static [SpecTarget],
}

/// Both families.
const BOTH: &[SpecTarget] = &[SpecTarget::X86_64, SpecTarget::Aarch64];
/// aarch64 only - the portable path is already good on x86-64, or the
/// replacement is a measured regression there. See [`Spec::supported_on`] and
/// `Lifter::try_horizontal_int_spec` for which is which.
const ARM: &[SpecTarget] = &[SpecTarget::Aarch64];

/// Every specialization, paired with a guest encoding that reaches it.
///
/// The `imm`-carrying entries are swept over their immediates where the
/// immediate selects a distinct lift-time path (`ROUNDPS`/`ROUNDPD`, whose imm8
/// picks the rounding mode).
fn cases() -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();

    let mut push = |name: &str, bytes: &[u8], spec: Spec, improves: &'static [SpecTarget]| {
        let mut v = bytes.to_vec();
        v.push(0xC3); // ret
        out.push(Case {
            name: name.to_string(),
            bytes: v,
            spec,
            improves,
        });
    };

    // ROUNDPS / ROUNDPD - imm8[2:0] selects the mode; bit 2 means "use MXCSR",
    // which resolves to nearest. Sweep all eight so every mode is covered.
    for imm in 0..8u8 {
        let m = Spec::PackedFpRound;
        push(&format!("roundps-{imm}"), &[0x66, 0x0F, 0x3A, 0x08, 0xC1, imm], m, BOTH);
        push(&format!("roundpd-{imm}"), &[0x66, 0x0F, 0x3A, 0x09, 0xC1, imm], m, BOTH);
    }

    // Horizontal floating-point. The even/odd gather is `shufps`-shaped at
    // dword and qword width, which baseline x86-64 has, so all four pay off on
    // both families.
    let h = Spec::PackedFpHorizontal;
    push("haddps", &[0xF2, 0x0F, 0x7C, 0xC1], h, BOTH);
    push("hsubps", &[0xF2, 0x0F, 0x7D, 0xC1], h, BOTH);
    push("haddpd", &[0x66, 0x0F, 0x7C, 0xC1], h, BOTH);
    push("hsubpd", &[0x66, 0x0F, 0x7D, 0xC1], h, BOTH);

    // Horizontal integer. The plain adds are `iadd_pairwise` and pay off
    // everywhere; the rest gather, so the word-width ones are aarch64-only -
    // a 16-bit permute needs SSSE3's `pshufb`, which baseline x86-64 lacks.
    let p = Spec::PackedIntHorizontal;
    push("phaddw", &[0x66, 0x0F, 0x38, 0x01, 0xC1], p, BOTH);
    push("phaddd", &[0x66, 0x0F, 0x38, 0x02, 0xC1], p, BOTH);
    push("phaddsw", &[0x66, 0x0F, 0x38, 0x03, 0xC1], p, ARM);
    push("phsubw", &[0x66, 0x0F, 0x38, 0x05, 0xC1], p, ARM);
    push("phsubd", &[0x66, 0x0F, 0x38, 0x06, 0xC1], p, BOTH);
    push("phsubsw", &[0x66, 0x0F, 0x38, 0x07, 0xC1], p, ARM);

    // Pairwise and widening multiplies.
    push("pmaddwd", &[0x66, 0x0F, 0xF5, 0xC1], Spec::PairwiseMultiplyAdd, BOTH);
    push(
        "pmaddubsw",
        &[0x66, 0x0F, 0x38, 0x04, 0xC1],
        Spec::MultiplyAddUnsignedSigned,
        BOTH,
    );
    push(
        "psadbw",
        &[0x66, 0x0F, 0xF6, 0xC1],
        Spec::SumOfAbsoluteDifferences,
        BOTH,
    );
    // The portable `I64X2` imul is fine on a machine that has `pmuludq`, so
    // these two are aarch64-only in the support table.
    push("pmuludq", &[0x66, 0x0F, 0xF4, 0xC1], Spec::WideningEvenMultiply, ARM);
    push(
        "pmuldq",
        &[0x66, 0x0F, 0x38, 0x28, 0xC1],
        Spec::WideningEvenMultiply,
        ARM,
    );
    push(
        "pmulhrsw",
        &[0x66, 0x0F, 0x38, 0x0B, 0xC1],
        Spec::RoundedMultiplyHigh,
        ARM,
    );

    out
}

/// Seeded XMM/GPR state, identical across every configuration under test.
fn seeded_state(seed: u64) -> RegisterContext {
    let mut rng = Rng::new(seed);
    let mut init = RegisterContext::default();
    for xmm in init.xmm.iter_mut() {
        xmm[0] = rng.next_u64();
        xmm[1] = rng.next_u64();
    }
    init.rax = rng.next_u64();
    init.rcx = rng.next_u64();
    init.rdx = rng.next_u64();
    init
}

/// Run `bytes` from `init` under a given specialization, vector-backed XMM.
fn run_spec(bytes: &[u8], init: RegisterContext, spec: Specialization) -> RegisterContext {
    LiftTest::new(bytes)
        .ctx(init)
        .vector_xmm(true)
        .specialize(spec)
        .run()
        .ctx
}

#[track_caller]
fn assert_same(name: &str, seed: u64, what: &str, a: &RegisterContext, b: &RegisterContext) {
    assert_eq!(a.xmm, b.xmm, "{name}: xmm diverged {what} (seed {seed})");
    assert_eq!(a.rax, b.rax, "{name}: rax diverged {what} (seed {seed})");
    let flags = |c: &RegisterContext| (c.cf, c.pf, c.af, c.zf, c.sf, c.of);
    assert_eq!(
        flags(a),
        flags(b),
        "{name}: flags diverged {what} (seed {seed})"
    );
}

/// Every specialization computes exactly what the portable paths compute.
///
/// Both portable paths are compared against, not just one: the unspecialized
/// vector path (the direct sibling) *and* the scalar `(lo, hi)` expansion,
/// which is the ultimate authority and the only one some of these instructions
/// had before this feature existed.
#[test]
fn agrees_with_portable() {
    let targets = [SpecTarget::X86_64, SpecTarget::Aarch64];

    for case in cases() {
        for seed in 0..24u64 {
            let init = seeded_state(seed);

            let scalar = LiftTest::new(&case.bytes)
                .ctx(init)
                .vector_xmm(false)
                .run()
                .ctx;
            let vector = run_spec(&case.bytes, init, Specialization::generic());
            assert_same(&case.name, seed, "scalar vs vector", &scalar, &vector);

            for target in targets {
                let spec = run_spec(&case.bytes, init, Specialization::for_target(target));
                assert_same(&case.name, seed, target.name(), &scalar, &spec);
            }
        }
    }
}

/// The edge cases the specializations were specifically reasoned about, where
/// hardware and x86 nearly-but-not-quite agree.
///
/// Random state finds these only by luck - `-32768 * -32768` is one pair out of
/// 2^32 - so they are seeded by hand. Each is a case where a plausible-looking
/// specialization is wrong.
#[test]
fn agrees_on_the_saturation_edges() {
    // (name, encoding, xmm0, xmm1)
    let edges: &[(&str, &[u8], [u64; 2], [u64; 2])] = &[
        // PMULHRSW: aarch64's SQRDMULH saturates to 0x7FFF where x86 wraps to
        // 0x8000. Every word is the minimum, so every lane hits it.
        (
            "pmulhrsw-min",
            &[0x66, 0x0F, 0x38, 0x0B, 0xC1, 0xC3],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
        ),
        // PMADDWD: -32768*-32768 twice overflows a dword and must *wrap*, not
        // saturate.
        (
            "pmaddwd-overflow",
            &[0x66, 0x0F, 0xF5, 0xC1, 0xC3],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
        ),
        // PMADDUBSW: 255*-128 + 255*-128 = -65280, below i16::MIN, so the
        // saturating narrow has to clamp.
        (
            "pmaddubsw-saturate-low",
            &[0x66, 0x0F, 0x38, 0x04, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0x8080_8080_8080_8080, 0x8080_8080_8080_8080],
        ),
        // PMADDUBSW: 255*127 + 255*127 = 64770, above i16::MAX.
        (
            "pmaddubsw-saturate-high",
            &[0x66, 0x0F, 0x38, 0x04, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0x7F7F_7F7F_7F7F_7F7F, 0x7F7F_7F7F_7F7F_7F7F],
        ),
        // PSADBW: the maximum possible sum, 8 * 255 per half.
        (
            "psadbw-max",
            &[0x66, 0x0F, 0xF6, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0, 0],
        ),
        // PSADBW the other way round: the absolute difference must be
        // symmetric, which is the whole point of the two saturating subtracts.
        (
            "psadbw-max-reversed",
            &[0x66, 0x0F, 0xF6, 0xC1, 0xC3],
            [0, 0],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        ),
        // PHADDSW / PHSUBSW: adjacent extremes, so the pairwise result leaves
        // the i16 range in both directions.
        (
            "phaddsw-saturate",
            &[0x66, 0x0F, 0x38, 0x03, 0xC1, 0xC3],
            [0x7FFF_7FFF_8000_8000, 0x8000_8000_7FFF_7FFF],
            [0x7FFF_8000_8000_7FFF, 0x0001_7FFF_8000_FFFF],
        ),
        (
            "phsubsw-saturate",
            &[0x66, 0x0F, 0x38, 0x07, 0xC1, 0xC3],
            [0x8000_7FFF_7FFF_8000, 0x7FFF_8000_8000_7FFF],
            [0x8000_7FFF_0000_FFFF, 0x7FFF_8000_0001_0000],
        ),
        // PMULDQ: the most negative dword squared, which needs the full 64 bits
        // and the sign extension to be right.
        (
            "pmuldq-min",
            &[0x66, 0x0F, 0x38, 0x28, 0xC1, 0xC3],
            [0x0000_0000_8000_0000, 0x0000_0000_8000_0000],
            [0x0000_0000_8000_0000, 0x0000_0000_8000_0000],
        ),
        // PMULUDQ: the maximum unsigned dword squared. The *odd* dwords are all
        // ones too, and must be ignored.
        (
            "pmuludq-max",
            &[0x66, 0x0F, 0xF4, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        ),
    ];

    for (name, bytes, x0, x1) in edges {
        let mut init = RegisterContext::default();
        init.xmm[0] = *x0;
        init.xmm[1] = *x1;

        let scalar = LiftTest::new(bytes).ctx(init).vector_xmm(false).run().ctx;
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            let spec = run_spec(bytes, init, Specialization::for_target(target));
            assert_same(name, 0, target.name(), &scalar, &spec);
        }
    }
}

/// NaN and infinity handling in the horizontal float forms.
///
/// `HADDPS`/`HSUBPS` feed the ordinary packed arithmetic, which owes x86's
/// sign-set floating indefinite for an invalid operation rather than the host's
/// default NaN. `inf + -inf` and `inf - inf` are the cases that produce one.
#[test]
fn agrees_on_float_horizontal_nans() {
    const INF: u32 = 0x7F80_0000;
    const NINF: u32 = 0xFF80_0000;
    const QNAN: u32 = 0x7FC0_0000;
    // Distinct payloads, so a lane that propagates the wrong operand's NaN is
    // visible rather than coincidentally right.
    const QNAN_A: u32 = 0x7FC0_0003;
    const QNAN_B: u32 = 0xFFC0_0005;

    // No signaling NaNs here. Quieting one is a property of the shared packed
    // arithmetic, not of the horizontal gather this test is about - and under
    // `--features unicorn-oracle` every `LiftTest::run` is cross-checked against
    // Unicorn, which disagrees with x86 (and with microlift) about SNaN payload
    // propagation: `HADDPS` of `0x7F800001` and a quiet NaN gives the quieted
    // source, `0x7FC00001`, on hardware and in microlift, and `0x7FC00000` in
    // Unicorn. That is a real divergence and worth chasing, but it belongs to
    // neither the portable nor the specialized path here, both of which agree.
    let pack = |a: u32, b: u32| u64::from(a) | (u64::from(b) << 32);

    let states: &[([u64; 2], [u64; 2])] = &[
        // inf + -inf in the low pair, inf - inf in the next: the invalid
        // operations, which owe x86's sign-set indefinite rather than the
        // host's default NaN.
        (
            [pack(INF, NINF), pack(INF, INF)],
            [pack(NINF, INF), pack(QNAN_A, QNAN)],
        ),
        // NaN operands, which must propagate rather than become indefinite.
        (
            [pack(QNAN_A, 0x3F80_0000), pack(QNAN, INF)],
            [pack(QNAN_B, QNAN_A), pack(NINF, QNAN_B)],
        ),
    ];

    for (i, (x0, x1)) in states.iter().enumerate() {
        for (name, bytes) in [
            ("haddps", &[0xF2u8, 0x0F, 0x7C, 0xC1, 0xC3][..]),
            ("hsubps", &[0xF2, 0x0F, 0x7D, 0xC1, 0xC3][..]),
            ("haddpd", &[0x66, 0x0F, 0x7C, 0xC1, 0xC3][..]),
            ("hsubpd", &[0x66, 0x0F, 0x7D, 0xC1, 0xC3][..]),
        ] {
            let mut init = RegisterContext::default();
            init.xmm[0] = *x0;
            init.xmm[1] = *x1;

            let scalar = LiftTest::new(bytes).ctx(init).vector_xmm(false).run().ctx;
            for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
                let spec = run_spec(bytes, init, Specialization::for_target(target));
                assert_same(name, i as u64, target.name(), &scalar, &spec);
            }
        }
    }
}

/// Count the IR instructions a lift emits, which is the closest lift-time proxy
/// for what the specialization buys.
fn ir_len(bytes: &[u8], spec: Specialization) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        specialization: spec,
        ..Default::default()
    };
    let code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    code.func
        .layout
        .blocks()
        .map(|b| code.func.layout.block_insts(b).count())
        .sum()
}

/// One compile of `bytes` under `spec` for `triple`, returning `.text` size.
fn compile_once(bytes: &[u8], spec: Specialization, triple: &str) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        specialization: spec,
        ..Default::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    // The lifter bakes in the *host* calling convention; override it so this
    // measures instruction lowering rather than tripping the target's ABI code
    // on a convention it does not accept.
    code.func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
        .expect("compile")
        .len()
}

/// Compiled `.text` size for `bytes` under `spec`, cross-compiled for `triple`.
///
/// Machine code size, not IR instruction count, is the metric these
/// specializations are actually about - and the difference is not academic.
/// `PMULUDQ`'s portable path is a single `imul` on `I64X2`, which is *one* IR
/// instruction and a seven-instruction `rev64`/`mul`/`xtn`/`addp`/`shll`/`umlal`
/// sequence on aarch64; the specialization replaces it with five IR
/// instructions that lower to three. Counting IR would score that a regression.
///
/// ## Why the minimum of several compiles
///
/// Two lifts of the same bytes with the same config do not produce identical
/// code. The lifter's context sync walks a `HashMap` of registers, and each
/// `HashMap` instance gets its own hash seed, so the closing run of stores is
/// emitted in a different order each time - same instructions, different
/// x86 displacement encodings, and up to **26 bytes** of difference in `.text`.
/// (aarch64 is exactly stable: fixed-width encodings make store order free.)
///
/// That noise is the same size as the smallest genuine effect measured here, so
/// a single sample cannot tell them apart. Taking the minimum of several
/// compiles pulls each measurement most of the way to the floor for its IR
/// shape; the residue is absorbed by [`noise_floor`], which is what the
/// comparisons are made modulo.
fn code_len(bytes: &[u8], spec: Specialization, triple: &str) -> usize {
    (0..SAMPLES)
        .map(|_| compile_once(bytes, spec, triple))
        .min()
        .expect("at least one sample")
}

/// Compiles per measurement.
const SAMPLES: usize = 8;

/// How many bytes of difference to treat as measurement noise on `target`.
///
/// Measured, not guessed: twelve compiles of one unchanged configuration span
/// up to 26 bytes on x86-64 and **exactly zero** on aarch64. Fixed-width
/// aarch64 encodings make the store reordering described on [`code_len`] free,
/// while x86 displacement encodings shrink and grow with it. Taking the minimum
/// of [`SAMPLES`] compiles removes most of that; 16 covers the rest with room
/// to spare, and is still well under the smallest real effect in the table
/// (35 bytes on x86-64).
fn noise_floor(target: SpecTarget) -> i64 {
    match target {
        SpecTarget::Aarch64 => 0,
        _ => 16,
    }
}

/// Each specialization actually fires on every target that claims it and
/// shrinks the compiled code - and does nothing at all on a target that does
/// not claim it.
///
/// Without the first half, a `try_*` that returned `false` unconditionally - a
/// mistyped gate, a `Spec` left off a target - would pass every equivalence
/// check above while doing nothing at all. Without the second, a
/// [`Spec::supported_on`] entry could quietly claim a target it has no
/// implementation for.
///
/// This is what decided the support table: the claim "specialization X helps
/// target Y" is measured here, per pair, rather than assumed. `PMULUDQ` on
/// x86-64 is the pair that failed - the portable `I64X2` `imul` is perfectly
/// reasonable on a machine that has `pmuludq`, and the shuffle-based
/// replacement is a net loss - which is why it is aarch64-only.
///
/// The compile is against each target's **baseline** feature set, no
/// `isa_settings`. That is the weakest configuration an AOT caller can hand the
/// backend, so a specialization that only pays off with optional ISA features
/// enabled does not count.
#[test]
fn specializations_shrink_the_compiled_code() {
    let targets = [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ];

    for case in cases() {
        let Case {
            name,
            bytes,
            spec,
            improves,
        } = case;
        for (target, triple) in targets {
            let base = code_len(&bytes, Specialization::generic(), triple);
            // Only the one specialization under test, so the measurement cannot
            // be explained by a different one firing.
            let only = Specialization::for_target(target).only(&[spec]);
            let n = code_len(&bytes, only, triple);

            let delta = n as i64 - base as i64;
            let noise = noise_floor(target);

            if improves.contains(&target) {
                assert!(
                    delta < -noise,
                    "{name}: {} compiled to {n} bytes with {}, {base} without ({delta:+}) - it \
                     did not fire, or it is not an improvement",
                    target.name(),
                    spec.name()
                );
            } else {
                assert!(
                    delta.abs() <= noise,
                    "{name}: {} is not expected to benefit from {}, but the compiled code \
                     changed by {delta:+} bytes ({base} -> {n})",
                    target.name(),
                    spec.name()
                );
            }
        }
    }
}

/// Disabling a specialization restores the portable IR exactly.
///
/// This is what makes `without()` a usable bisection tool: it has to be a true
/// off switch, not a different third path.
#[test]
fn disabling_restores_the_portable_ir() {
    for case in cases() {
        let (name, bytes, spec) = (case.name, case.bytes, case.spec);
        let base = ir_len(&bytes, Specialization::generic());
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            let off = Specialization::for_target(target).without(spec);
            // Other specializations do not apply - each snippet is one guest
            // instruction - so the whole set minus this one must be inert.
            assert_eq!(
                ir_len(&bytes, off),
                base,
                "{name}: disabling {} on {} changed the IR",
                spec.name(),
                target.name()
            );
        }
    }
}

/// Each target's specialized IR lowers for that target.
///
/// The one that matters: `x86_pmulhrsw` and `x86_pmaddubsw` have no aarch64
/// rule, and `sqmul_round_sat` is only reachable through the aarch64 path. A
/// specialization wired to the wrong family compiles fine and then fails to
/// lower, which nothing above would notice.
#[test]
fn each_target_lowers_its_own_ir() {
    let pairs = [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ];

    for case in cases() {
        let (name, bytes) = (case.name, case.bytes);
        for (target, triple) in pairs {
            let config = LifterConfig {
                use_vector_xmm: true,
                specialization: Specialization::for_target(target),
                ..Default::default()
            };
            let mut code = lift_to_code::<X86_64>(&bytes, 0x1000, config).expect("lift");
            // The lifter bakes in the *host* calling convention; override it so
            // this exercises instruction lowering rather than tripping the
            // target's ABI code on a convention it does not accept.
            code.func.signature.call_conv = CallConv::SystemV;
            ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
                .unwrap_or_else(|e| panic!("{name} failed to lower for {triple}: {e:?}"));
        }
    }
}

/// A specialization is inert without vector-backed XMM registers.
///
/// Every sequence here is a whole-register vector one, so there is nothing to
/// build it out of in scalar mode. Asking for it anyway must be a no-op rather
/// than an error or a partial application.
#[test]
fn inert_without_vector_xmm() {
    for case in cases() {
        let (name, bytes) = (case.name, case.bytes);
        let config = |spec| LifterConfig {
            use_vector_xmm: false,
            specialization: spec,
            ..Default::default()
        };
        let base = lift_to_code::<X86_64>(&bytes, 0x1000, config(Specialization::generic()))
            .expect("lift");
        let spec = lift_to_code::<X86_64>(
            &bytes,
            0x1000,
            config(Specialization::for_target(SpecTarget::Aarch64)),
        )
        .expect("lift");

        let count = |c: &microlift::core::LiftedCode| -> usize {
            c.func
                .layout
                .blocks()
                .map(|b| c.func.layout.block_insts(b).count())
                .sum()
        };
        assert_eq!(
            count(&base),
            count(&spec),
            "{name}: specialization was not inert in scalar mode"
        );
    }
}

/// Not a test: prints the measurement table `specializations_shrink_the_compiled_code`
/// asserts on. Run with `--ignored --nocapture` when adding a specialization.
#[test]
#[ignore]
fn dump_code_sizes() {
    for (target, triple) in [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ] {
        println!("== {}", target.name());
        for case in cases() {
            let (name, bytes, spec) = (case.name, case.bytes, case.spec);
            let base = code_len(&bytes, Specialization::generic(), triple);
            let only = Specialization::for_target(target).none().with(spec);
            // Bypass supported_on so unsupported pairs are measured too.
            let n = code_len(&bytes, only, triple);
            println!(
                "{name:24} {spec:?} base={base} spec={n} delta={}",
                n as i64 - base as i64
            );
        }
    }
}
