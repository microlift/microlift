use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;

/// `pause` ; `ret`
const PAUSE_RET: &[u8] = &[0xF3, 0x90, 0xC3];

/// `pause`, `F3 90` (`rep nop`).
const X86_PAUSE: &[u8] = &[0xF3, 0x90];

/// `yield`, `HINT #1`, `0xD503203F` little-endian.
const AARCH64_YIELD: &[u8] = &[0x3F, 0x20, 0x03, 0xD5];

fn compile_for(bytes: &[u8], spec: Specialization, triple: &str) -> Vec<u8> {
    let config = LifterConfig {
        specialization: spec,
        ..Default::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    // Override the host calling convention to measure lowering, not ABI.
    code.func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None).expect("compile")
}

fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .filter(|w| *w == needle)
        .count()
}

/// Emitted IR instruction count over every block.
fn ir_len(bytes: &[u8], spec: Specialization) -> usize {
    let config = LifterConfig {
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

/// Targets claiming the spec, with the encoding each emits.
const TARGETS: &[(SpecTarget, &str, &[u8])] = &[
    (SpecTarget::X86_64, "x86_64-unknown-linux-gnu", X86_PAUSE),
    (
        SpecTarget::Aarch64,
        "aarch64-unknown-linux-gnu",
        AARCH64_YIELD,
    ),
];

/// Each target emits its hint only with the spec on. Counted, since `F3 90` could occur by coincidence: enabling must add exactly one occurrence.
#[test]
fn each_target_emits_its_own_hint() {
    for &(target, triple, needle) in TARGETS {
        let only = Specialization::for_target(target).only(&[Spec::SpinLoopHint]);

        let off = occurrences(
            &compile_for(PAUSE_RET, Specialization::generic(), triple),
            needle,
        );
        let on = occurrences(&compile_for(PAUSE_RET, only, triple), needle);

        assert_eq!(
            on,
            off + 1,
            "{}: expected one more {needle:02X?} with spin-loop-hint on ({off} without, {on} with)",
            target.name()
        );
    }
}

/// Unclaimed targets and the generic target emit nothing; generic is the default, so unaware callers see identical output.
#[test]
fn inert_where_unclaimed() {
    let base = ir_len(PAUSE_RET, Specialization::generic());

    // Enabling on a target with no lowering stays inert.
    assert_eq!(
        ir_len(
            PAUSE_RET,
            Specialization::generic().with(Spec::SpinLoopHint)
        ),
        base,
        "spin-loop-hint fired under the generic target"
    );

    // A true off switch where claimed.
    for &(target, _, _) in TARGETS {
        assert_eq!(
            ir_len(
                PAUSE_RET,
                Specialization::for_target(target).without(Spec::SpinLoopHint)
            ),
            base,
            "{}: disabling spin-loop-hint changed the IR",
            target.name()
        );
    }
}

/// Enabling changes the IR on a claiming target; mirror of [`inert_where_unclaimed`], catching a spec that never fires.
#[test]
fn fires_where_claimed() {
    let base = ir_len(PAUSE_RET, Specialization::generic());
    for &(target, _, _) in TARGETS {
        assert_eq!(
            ir_len(
                PAUSE_RET,
                Specialization::for_target(target).only(&[Spec::SpinLoopHint])
            ),
            base + 1,
            "{}: spin-loop-hint did not emit its one instruction",
            target.name()
        );
    }
}

/// No `use_vector_xmm` needed (exempt via [`Spec::requires_vector_xmm`]); see `specialize_tests::inert_without_vector_xmm`.
#[test]
fn does_not_need_vector_xmm() {
    assert!(!Spec::SpinLoopHint.requires_vector_xmm());

    let config = |spec| LifterConfig {
        use_vector_xmm: false,
        specialization: spec,
        ..Default::default()
    };
    let count = |spec| -> usize {
        let code = lift_to_code::<X86_64>(PAUSE_RET, 0x1000, config(spec)).expect("lift");
        code.func
            .layout
            .blocks()
            .map(|b| code.func.layout.block_insts(b).count())
            .sum()
    };
    assert_ne!(
        count(Specialization::generic()),
        count(Specialization::host()),
        "spin-loop-hint declares it does not need vector XMM, but did nothing in scalar mode"
    );
}

/// Executed for real: a `PAUSE` between instructions must leave the result and flags unchanged.
#[test]
fn changes_no_architectural_state() {
    // mov rax, 7 ; add rax, 5 ; pause ; sub rax, 1 ; ret
    let bytes = &[
        0x48, 0xC7, 0xC0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x48, 0x83, 0xC0, 0x05, // add rax, 5
        0xF3, 0x90, // pause
        0x48, 0x83, 0xE8, 0x01, // sub rax, 1
        0xC3, // ret
    ];

    let plain = LiftTest::new(bytes).run().ctx;
    let hinted = LiftTest::new(bytes)
        .specialize(Specialization::host().only(&[Spec::SpinLoopHint]))
        .run()
        .ctx;

    assert_eq!(hinted.rax, 11);
    assert_eq!(plain.rax, hinted.rax);
    let flags = |c: &RegisterContext| (c.cf, u8::from(c.pf()), u8::from(c.af()), c.zf, c.sf, c.of);
    assert_eq!(flags(&plain), flags(&hinted));
}

/// Each target's IR lowers for it; `spin_hint` is a new CLIF opcode, so a backend lacking a rule fails to lower.
#[test]
fn each_target_lowers_its_own_ir() {
    for &(target, triple, _) in TARGETS {
        compile_for(PAUSE_RET, Specialization::for_target(target), triple);
    }
}
