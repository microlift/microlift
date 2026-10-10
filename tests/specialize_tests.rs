//! Specialized-vs-portable equivalence: a specialization may not change a single result bit.
//! Checks (1) results agree with the vector and scalar paths, (2) code shrinks exactly where the support table says (`specializations_shrink_the_compiled_code`), (3) each target's IR lowers (`each_target_lowers_its_own_ir`).
//! Adding a specialization without listing its encoding here leaves it unproven.

use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

/// A guest instruction, its [`Spec`], and the targets where it must shrink.
struct Case {
    name: String,
    bytes: Vec<u8>,
    spec: Spec,
    /// Targets where this must get smaller; elsewhere it must be byte-identical to portable.
    improves: &'static [SpecTarget],
    /// Lift with [`LifterConfig::eliminate_dead_flags`] on; [`Spec::IntegerOverflowFlags`] declines without flag liveness.
    lean: bool,
}

/// Both families.
const BOTH: &[SpecTarget] = &[SpecTarget::X86_64, SpecTarget::Aarch64];
/// aarch64 only: portable is already good on x86-64 or the replacement regresses. See [`Spec::supported_on`].
const ARM: &[SpecTarget] = &[SpecTarget::Aarch64];
/// x86-64 only: sub-word scalar widths. See `Lifter::native_overflow_width_ok`.
const X64: &[SpecTarget] = &[SpecTarget::X86_64];

/// Every specialization with an encoding that reaches it; `imm`-carrying entries are swept over their lift-time-distinct immediates.
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
            lean: false,
        });
    };

    // ROUNDPS/ROUNDPD imm8[2:0] selects the mode (bit 2 = MXCSR -> nearest); sweep all eight.
    for imm in 0..8u8 {
        let m = Spec::PackedFpRound;
        push(
            &format!("roundps-{imm}"),
            &[0x66, 0x0F, 0x3A, 0x08, 0xC1, imm],
            m,
            BOTH,
        );
        push(
            &format!("roundpd-{imm}"),
            &[0x66, 0x0F, 0x3A, 0x09, 0xC1, imm],
            m,
            BOTH,
        );
    }

    // Horizontal float: dword/qword gathers are `shufps`-shaped, so all four pay on both families.
    let h = Spec::PackedFpHorizontal;
    push("haddps", &[0xF2, 0x0F, 0x7C, 0xC1], h, BOTH);
    push("hsubps", &[0xF2, 0x0F, 0x7D, 0xC1], h, BOTH);
    push("haddpd", &[0x66, 0x0F, 0x7C, 0xC1], h, BOTH);
    push("hsubpd", &[0x66, 0x0F, 0x7D, 0xC1], h, BOTH);

    // Horizontal int: plain adds are `iadd_pairwise` (pay everywhere); word gathers need `pshufb`, so aarch64-only.
    let p = Spec::PackedIntHorizontal;
    push("phaddw", &[0x66, 0x0F, 0x38, 0x01, 0xC1], p, BOTH);
    push("phaddd", &[0x66, 0x0F, 0x38, 0x02, 0xC1], p, BOTH);
    push("phaddsw", &[0x66, 0x0F, 0x38, 0x03, 0xC1], p, ARM);
    push("phsubw", &[0x66, 0x0F, 0x38, 0x05, 0xC1], p, ARM);
    push("phsubd", &[0x66, 0x0F, 0x38, 0x06, 0xC1], p, BOTH);
    push("phsubsw", &[0x66, 0x0F, 0x38, 0x07, 0xC1], p, ARM);

    // Pairwise and widening multiplies.
    push(
        "pmaddwd",
        &[0x66, 0x0F, 0xF5, 0xC1],
        Spec::PairwiseMultiplyAdd,
        BOTH,
    );
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
    // Portable `I64X2` imul is fine with `pmuludq`: aarch64-only.
    push(
        "pmuludq",
        &[0x66, 0x0F, 0xF4, 0xC1],
        Spec::WideningEvenMultiply,
        ARM,
    );
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

    // `<insn> ; seto dl ; setb bl` x REPEATS: the setccs give CF/OF a reader so the specialization is reached.
    let mut push_n = |name: &str, bytes: &[u8], spec: Spec, improves: &'static [SpecTarget]| {
        let mut unit = bytes.to_vec();
        unit.extend_from_slice(&[0x0F, 0x90, 0xC2]); // seto dl
        unit.extend_from_slice(&[0x0F, 0x92, 0xC3]); // setb bl
        let mut v: Vec<u8> = unit.repeat(REPEATS);
        v.push(0xC3); // ret
        out.push(Case {
            name: name.to_string(),
            bytes: v,
            spec,
            improves,
            lean: true,
        });
    };

    // Scalar integer CF/OF, register forms only (locked forms live in `overflow_flags_tests.rs`), destination RAX so `assert_same` sees result and flags.
    // Repeated [`REPEATS`] times. Widths swept on ADD alone; narrow ones are x86-64-only (see `Lifter::native_overflow_width_ok`).
    let o = Spec::IntegerOverflowFlags;
    push_n("add-r64", &[0x48, 0x01, 0xC8], o, BOTH);
    push_n("add-r32", &[0x01, 0xC8], o, BOTH);
    push_n("add-r16", &[0x66, 0x01, 0xC8], o, X64);
    push_n("add-r8", &[0x00, 0xC8], o, X64);
    push_n("sub-r64", &[0x48, 0x29, 0xC8], o, BOTH);
    push_n("cmp-r64", &[0x48, 0x39, 0xC8], o, BOTH);
    push_n("adc-r64", &[0x48, 0x11, 0xC8], o, BOTH);
    push_n("sbb-r64", &[0x48, 0x19, 0xC8], o, BOTH);
    push_n("inc-r64", &[0x48, 0xFF, 0xC0], o, BOTH);
    push_n("dec-r64", &[0x48, 0xFF, 0xC8], o, BOTH);
    push_n("neg-r64", &[0x48, 0xF7, 0xD8], o, BOTH);
    push_n("xadd-r64", &[0x48, 0x0F, 0xC1, 0xC8], o, BOTH);
    // `smul_overflow` is x64's one-operand `imul` (pins RAX/RDX), a net loss: aarch64-only. See `try_native_imul_flags`.
    push_n("imul-r64", &[0x48, 0x0F, 0xAF, 0xC1], o, ARM);
    push_n("imul-imm32", &[0x69, 0xC1, 0x34, 0x12, 0x00, 0x00], o, ARM);

    out
}

/// Copies of the instruction per snippet: this spec saves only ~3 host instructions against ~1500 bytes of fixed overhead, so repeating scales the effect.
const REPEATS: usize = 32;

/// Seeded XMM/GPR state, identical across configurations.
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

/// Run `bytes` from `init` under a specialization, vector-backed XMM.
fn run_spec(
    bytes: &[u8],
    init: RegisterContext,
    spec: Specialization,
    lean: bool,
) -> RegisterContext {
    let t = LiftTest::new(bytes)
        .ctx(init)
        .vector_xmm(true)
        .specialize(spec);
    let t = if lean { t.eliminate_dead_flags() } else { t };
    t.run().ctx
}

#[track_caller]
fn assert_same(name: &str, seed: u64, what: &str, a: &RegisterContext, b: &RegisterContext) {
    assert_eq!(a.xmm, b.xmm, "{name}: xmm diverged {what} (seed {seed})");
    let gprs = |c: &RegisterContext| (c.rax, c.rcx, c.rdx);
    assert_eq!(
        gprs(a),
        gprs(b),
        "{name}: GPRs diverged {what} (seed {seed})"
    );
    let flags = |c: &RegisterContext| (c.cf, u8::from(c.pf()), u8::from(c.af()), c.zf, c.sf, c.of);
    assert_eq!(
        flags(a),
        flags(b),
        "{name}: flags diverged {what} (seed {seed})"
    );
}

/// Every specialization equals both the unspecialized vector path and the scalar `(lo, hi)` expansion (the authority).
#[test]
fn agrees_with_portable() {
    let targets = [SpecTarget::X86_64, SpecTarget::Aarch64];

    for case in cases() {
        for seed in 0..24u64 {
            let init = seeded_state(seed);

            let plain = LiftTest::new(&case.bytes).ctx(init).vector_xmm(false);
            let scalar = if case.lean {
                plain.eliminate_dead_flags().run().ctx
            } else {
                plain.run().ctx
            };
            let vector = run_spec(&case.bytes, init, Specialization::generic(), case.lean);
            assert_same(&case.name, seed, "scalar vs vector", &scalar, &vector);

            for target in targets {
                let spec = run_spec(
                    &case.bytes,
                    init,
                    Specialization::for_target(target),
                    case.lean,
                );
                assert_same(&case.name, seed, target.name(), &scalar, &spec);
            }
        }
    }
}

/// Hand-seeded edges where plausible specializations are wrong (`-32768 * -32768` is one pair in 2^32).
#[test]
fn agrees_on_the_saturation_edges() {
    // (name, encoding, xmm0, xmm1)
    let edges: &[(&str, &[u8], [u64; 2], [u64; 2])] = &[
        // PMULHRSW: aarch64 SQRDMULH saturates to 0x7FFF where x86 wraps to 0x8000.
        (
            "pmulhrsw-min",
            &[0x66, 0x0F, 0x38, 0x0B, 0xC1, 0xC3],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
        ),
        // PMADDWD: -32768*-32768 twice overflows a dword and must wrap.
        (
            "pmaddwd-overflow",
            &[0x66, 0x0F, 0xF5, 0xC1, 0xC3],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
            [0x8000_8000_8000_8000, 0x8000_8000_8000_8000],
        ),
        // PMADDUBSW: 255*-128 twice = -65280, below i16::MIN: must clamp.
        (
            "pmaddubsw-saturate-low",
            &[0x66, 0x0F, 0x38, 0x04, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0x8080_8080_8080_8080, 0x8080_8080_8080_8080],
        ),
        // PMADDUBSW: 255*127 twice = 64770, above i16::MAX.
        (
            "pmaddubsw-saturate-high",
            &[0x66, 0x0F, 0x38, 0x04, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0x7F7F_7F7F_7F7F_7F7F, 0x7F7F_7F7F_7F7F_7F7F],
        ),
        // PSADBW: maximum sum, 8 * 255 per half.
        (
            "psadbw-max",
            &[0x66, 0x0F, 0xF6, 0xC1, 0xC3],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0, 0],
        ),
        // PSADBW reversed: absolute difference must be symmetric.
        (
            "psadbw-max-reversed",
            &[0x66, 0x0F, 0xF6, 0xC1, 0xC3],
            [0, 0],
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        ),
        // PHADDSW/PHSUBSW: adjacent extremes leave the i16 range both ways.
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
        // PMULDQ: most negative dword squared needs full 64 bits.
        (
            "pmuldq-min",
            &[0x66, 0x0F, 0x38, 0x28, 0xC1, 0xC3],
            [0x0000_0000_8000_0000, 0x0000_0000_8000_0000],
            [0x0000_0000_8000_0000, 0x0000_0000_8000_0000],
        ),
        // PMULUDQ: max dword squared; the odd dwords are ones and must be ignored.
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
            let spec = run_spec(bytes, init, Specialization::for_target(target), false);
            assert_same(name, 0, target.name(), &scalar, &spec);
        }
    }
}

/// `HADDPS`/`HSUBPS` owe x86's sign-set indefinite for `inf + -inf`, not the host's default NaN.
#[test]
fn agrees_on_float_horizontal_nans() {
    const INF: u32 = 0x7F80_0000;
    const NINF: u32 = 0xFF80_0000;
    const QNAN: u32 = 0x7FC0_0000;
    // Distinct payloads expose a wrong-operand NaN.
    const QNAN_A: u32 = 0x7FC0_0003;
    const QNAN_B: u32 = 0xFFC0_0005;

    // No SNaNs: quieting belongs to shared packed arithmetic, and Unicorn disagrees with hardware on SNaN payloads here (`0x7FC00000` vs `0x7FC00001`).
    let pack = |a: u32, b: u32| u64::from(a) | (u64::from(b) << 32);

    let states: &[([u64; 2], [u64; 2])] = &[
        // inf + -inf, inf - inf: invalid, owe the sign-set indefinite.
        (
            [pack(INF, NINF), pack(INF, INF)],
            [pack(NINF, INF), pack(QNAN_A, QNAN)],
        ),
        // NaN operands propagate rather than become indefinite.
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
                let spec = run_spec(bytes, init, Specialization::for_target(target), false);
                assert_same(name, i as u64, target.name(), &scalar, &spec);
            }
        }
    }
}

/// IR instruction count: a lift-time proxy for what the specialization buys.
fn ir_len(bytes: &[u8], spec: Specialization, lean: bool) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        specialization: spec,
        eliminate_dead_flags: lean,
        ..Default::default()
    };
    let code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    code.func
        .layout
        .blocks()
        .map(|b| code.func.layout.block_insts(b).count())
        .sum()
}

/// One compile of `bytes` under `spec` for `triple`; returns `.text` size.
fn compile_once(bytes: &[u8], spec: Specialization, triple: &str, lean: bool) -> usize {
    let config = LifterConfig {
        use_vector_xmm: true,
        specialization: spec,
        eliminate_dead_flags: lean,
        ..Default::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    // Override the host calling convention to measure lowering, not ABI.
    code.func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
        .expect("compile")
        .len()
}

/// Compiled `.text` size under `spec`, cross-compiled for `triple`.
/// Machine code, not IR count, is the metric (PMULUDQ: 1 IR op vs 7 aarch64 instructions portable; 5 IR ops -> 3 specialized).
/// Lifts are not byte-identical: HashMap-ordered context stores vary x86 `.text` by up to 26 bytes, so take the minimum of several compiles modulo [`noise_floor`].
fn code_len(bytes: &[u8], spec: Specialization, triple: &str, lean: bool) -> usize {
    (0..SAMPLES)
        .map(|_| compile_once(bytes, spec, triple, lean))
        .min()
        .expect("at least one sample")
}

/// Compiles per measurement.
const SAMPLES: usize = 8;

/// Bytes of difference treated as noise on `target`: 0 on aarch64 (fixed-width), up to 26 on x86-64 before taking the minimum; 16 covers the rest and is below the smallest real effect (35).
fn noise_floor(target: SpecTarget) -> i64 {
    match target {
        SpecTarget::Aarch64 => 0,
        _ => 16,
    }
}

/// Each specialization fires and shrinks code on every target claiming it, and does nothing elsewhere.
/// This decides the support table per pair (PMULUDQ on x86-64 failed: portable `imul` is fine there).
/// Compiles against each target's baseline feature set, no `isa_settings`.
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
            lean,
        } = case;
        for (target, triple) in targets {
            // Only the spec under test.
            let only = Specialization::for_target(target).only(&[spec]);

            if improves.contains(&target) {
                let base = code_len(&bytes, Specialization::generic(), triple, lean);
                let n = code_len(&bytes, only, triple, lean);
                let delta = n as i64 - base as i64;
                assert!(
                    delta < -noise_floor(target),
                    "{name}: {} compiled to {n} bytes with {}, {base} without ({delta:+}) - it \
                     did not fire, or it is not an improvement",
                    target.name(),
                    spec.name()
                );
            } else {
                // "Did not fire" is checked in IR (exact, the lifter is deterministic); `.text` noise exceeds the floor on repeated snippets. IR count cannot check the improving direction (see [`code_len`]).
                assert_eq!(
                    ir_len(&bytes, only, lean),
                    ir_len(&bytes, Specialization::generic(), lean),
                    "{name}: {} is not expected to benefit from {}, but enabling it changed \
                     the emitted IR",
                    target.name(),
                    spec.name()
                );
            }
        }
    }
}

/// Disabling a specialization restores the portable IR exactly: `without()` is a true off switch.
#[test]
fn disabling_restores_the_portable_ir() {
    for case in cases() {
        let (name, bytes, spec, lean) = (case.name, case.bytes, case.spec, case.lean);
        let base = ir_len(&bytes, Specialization::generic(), lean);
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            let off = Specialization::for_target(target).without(spec);
            // Each snippet is one guest instruction, so the rest of the set is inert.
            assert_eq!(
                ir_len(&bytes, off, lean),
                base,
                "{name}: disabling {} on {} changed the IR",
                spec.name(),
                target.name()
            );
        }
    }
}

/// Each target's specialized IR lowers for it: `x86_pmulhrsw`/`x86_pmaddubsw` have no aarch64 rule and `sqmul_round_sat` is aarch64-only.
#[test]
fn each_target_lowers_its_own_ir() {
    let pairs = [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ];

    for case in cases() {
        let (name, bytes, lean) = (case.name, case.bytes, case.lean);
        for (target, triple) in pairs {
            let config = LifterConfig {
                use_vector_xmm: true,
                specialization: Specialization::for_target(target),
                eliminate_dead_flags: lean,
                ..Default::default()
            };
            let mut code = lift_to_code::<X86_64>(&bytes, 0x1000, config).expect("lift");
            // Override the host calling convention to exercise lowering, not ABI.
            code.func.signature.call_conv = CallConv::SystemV;
            ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
                .unwrap_or_else(|e| panic!("{name} failed to lower for {triple}: {e:?}"));
        }
    }
}

/// Vector specs are inert without vector-backed XMM; [`Spec::IntegerOverflowFlags`] is exempt (scalar integer) and checked for the opposite.
#[test]
fn inert_without_vector_xmm() {
    for case in cases() {
        let (name, bytes, needs_vec) = (case.name, case.bytes, case.spec.requires_vector_xmm());
        let lean = case.lean;
        // The target the case is claimed on: a spec can decline for target-specific reasons (sub-word widths on aarch64).
        let target = case.improves[0];
        let config = |spec| LifterConfig {
            use_vector_xmm: false,
            specialization: spec,
            eliminate_dead_flags: lean,
            ..Default::default()
        };
        let base = lift_to_code::<X86_64>(&bytes, 0x1000, config(Specialization::generic()))
            .expect("lift");
        let spec =
            lift_to_code::<X86_64>(&bytes, 0x1000, config(Specialization::for_target(target)))
                .expect("lift");

        let count = |c: &microlift::core::LiftedCode| -> usize {
            c.func
                .layout
                .blocks()
                .map(|b| c.func.layout.block_insts(b).count())
                .sum()
        };
        if needs_vec {
            assert_eq!(
                count(&base),
                count(&spec),
                "{name}: specialization was not inert in scalar mode"
            );
        } else {
            assert_ne!(
                count(&base),
                count(&spec),
                "{name}: specialization declares it does not need vector XMM, but did \
                 nothing in scalar mode"
            );
        }
    }
}

/// Not a test: prints the measurement table. Run with `--ignored --nocapture` when adding a specialization.
#[test]
#[ignore]
fn dump_code_sizes() {
    for (target, triple) in [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ] {
        println!("== {}", target.name());
        for case in cases() {
            let (name, bytes, spec, lean) = (case.name, case.bytes, case.spec, case.lean);
            let base = code_len(&bytes, Specialization::generic(), triple, lean);
            let only = Specialization::for_target(target).none().with(spec);
            // Bypass supported_on so unsupported pairs are measured.
            let n = code_len(&bytes, only, triple, lean);
            println!(
                "{name:24} {spec:?} base={base} spec={n} delta={}",
                n as i64 - base as i64
            );
        }
    }
}
