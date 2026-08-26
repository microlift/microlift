use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;

/// `pause` ; `ret`
const PAUSE_RET: &[u8] = &[0xF3, 0x90, 0xC3];

/// `pause`, `F3 90` - a `rep nop`.
const X86_PAUSE: &[u8] = &[0xF3, 0x90];

/// `yield`, `HINT #1` - `0xD503203F`, little-endian.
const AARCH64_YIELD: &[u8] = &[0x3F, 0x20, 0x03, 0xD5];

fn compile_for(bytes: &[u8], spec: Specialization, triple: &str) -> Vec<u8> {
    let config = LifterConfig {
        specialization: spec,
        ..Default::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, 0x1000, config).expect("lift");
    // The lifter bakes in the *host* calling convention; override it so this
    // measures instruction lowering rather than tripping the target's ABI code
    // on a convention it does not accept.
    code.func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None).expect("compile")
}

fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .filter(|w| *w == needle)
        .count()
}

/// Emitted IR instruction count, over every block.
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

/// The targets that claim the spec, with the encoding each is expected to emit.
const TARGETS: &[(SpecTarget, &str, &[u8])] = &[
    (SpecTarget::X86_64, "x86_64-unknown-linux-gnu", X86_PAUSE),
    (
        SpecTarget::Aarch64,
        "aarch64-unknown-linux-gnu",
        AARCH64_YIELD,
    ),
];

/// Each target emits its own hint instruction, and only with the spec enabled.
///
/// Counted rather than merely searched for: `F3 90` is two very ordinary bytes,
/// and a displacement or an immediate elsewhere in the region could contain
/// them by coincidence. What must hold is that enabling the spec adds exactly
/// one more occurrence than the portable lowering has.
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

/// A target that does not claim the spec, and the generic target, emit nothing.
///
/// The generic half is the one that matters: `Specialization::generic()` is the
/// default, so this is the promise that a caller who never touches the knob
/// gets byte-identical output to a lifter without it.
#[test]
fn inert_where_unclaimed() {
    let base = ir_len(PAUSE_RET, Specialization::generic());

    // Enabling it on a target that has no lowering must stay inert rather than
    // emit an opcode that target cannot lower.
    assert_eq!(
        ir_len(
            PAUSE_RET,
            Specialization::generic().with(Spec::SpinLoopHint)
        ),
        base,
        "spin-loop-hint fired under the generic target"
    );

    // And it is a true off switch on a target that does claim it.
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

/// Enabling it does change the IR on a target that claims it.
///
/// The mirror of [`inert_where_unclaimed`]: without this, a gate mistyped so
/// that the spec never fires would pass every other test in this file except
/// the machine-code one, and would pass that one too if the encoding search
/// were ever loosened.
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

/// The spec needs no `use_vector_xmm`, and must fire without it.
///
/// [`Spec::requires_vector_xmm`] declares this one exempt - it is a single
/// operand-free instruction with no opinion about how XMM registers are stored
/// - and `specialize_tests::inert_without_vector_xmm` checks the exemption for
/// the other specs. This is the same check for this one.
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

/// The hint perturbs no architectural state, executed for real on the host.
///
/// A `PAUSE` between two ordinary instructions must leave the region's result
/// exactly as it is without one - including the flags, which is the half a
/// careless lowering would break.
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

/// Each target's specialized IR lowers for the machine it names.
///
/// `spin_hint` is a new CLIF opcode, so a backend with no rule for it fails to
/// lower rather than producing wrong code - loudly, but only if something
/// actually compiles for that backend.
#[test]
fn each_target_lowers_its_own_ir() {
    for &(target, triple, _) in TARGETS {
        compile_for(PAUSE_RET, Specialization::for_target(target), triple);
    }
}
