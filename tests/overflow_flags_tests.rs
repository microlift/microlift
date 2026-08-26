use cranelift::codegen::ir::Opcode;
use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;

const OVERFLOW_OPS: &[Opcode] = &[
    Opcode::UaddOverflow,
    Opcode::SaddOverflow,
    Opcode::UsubOverflow,
    Opcode::SsubOverflow,
    Opcode::SmulOverflow,
];

fn config(spec: Specialization, lean: bool) -> LifterConfig {
    LifterConfig {
        specialization: spec,
        eliminate_dead_flags: lean,
        ..Default::default()
    }
}

/// How many overflow-producing instructions a lift of `bytes` emits.
fn overflow_ops(bytes: &[u8], spec: Specialization, lean: bool) -> usize {
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config(spec, lean)).expect("lift");
    code.func
        .layout
        .blocks()
        .flat_map(|b| code.func.layout.block_insts(b))
        .filter(|&i| OVERFLOW_OPS.contains(&code.func.dfg.insts[i].opcode()))
        .count()
}

/// Every add-, subtract- and multiply-shaped mnemonic, in a shape whose CF and
/// OF are read - `seto`/`setb` right after, so the liveness pass reports both
/// live and the specialization is actually reached.
///
/// `LOCK` forms address `[rsp]`, which the harness guarantees is mapped.
fn snippets() -> Vec<(&'static str, Vec<u8>)> {
    // seto dl ; setb bl - the readers that make CF and OF live.
    const READ: &[u8] = &[0x0F, 0x90, 0xC2, 0x0F, 0x92, 0xC3];
    let cases: &[(&str, &[u8])] = &[
        ("add r64", &[0x48, 0x01, 0xC8]),
        ("add r32", &[0x01, 0xC8]),
        ("add r16", &[0x66, 0x01, 0xC8]),
        ("add r8", &[0x00, 0xC8]),
        ("sub r64", &[0x48, 0x29, 0xC8]),
        ("sub r8", &[0x28, 0xC8]),
        ("cmp r64", &[0x48, 0x39, 0xC8]),
        ("adc r64", &[0x48, 0x11, 0xC8]),
        ("adc r8", &[0x10, 0xC8]),
        ("sbb r64", &[0x48, 0x19, 0xC8]),
        ("inc r64", &[0x48, 0xFF, 0xC0]),
        ("dec r64", &[0x48, 0xFF, 0xC8]),
        ("neg r64", &[0x48, 0xF7, 0xD8]),
        ("neg r8", &[0xF6, 0xD8]),
        ("xadd r64", &[0x48, 0x0F, 0xC1, 0xC8]),
        ("imul r64,r64", &[0x48, 0x0F, 0xAF, 0xC1]),
        ("imul r32,r32,imm", &[0x69, 0xC1, 0x34, 0x12, 0x00, 0x00]),
        // Memory destinations: `add [rsp], rcx` and friends. The stack is the
        // one address the harness always has mapped.
        ("add [rsp], rcx", &[0x48, 0x01, 0x0C, 0x24]),
        ("sub [rsp], rcx", &[0x48, 0x29, 0x0C, 0x24]),
        ("adc [rsp], rcx", &[0x48, 0x11, 0x0C, 0x24]),
        ("sbb [rsp], rcx", &[0x48, 0x19, 0x0C, 0x24]),
        ("inc qword [rsp]", &[0x48, 0xFF, 0x04, 0x24]),
        ("neg qword [rsp]", &[0x48, 0xF7, 0x1C, 0x24]),
        // The locked forms, which route through `atomic_rmw` / `atomic_cas`.
        // LOCK ADC and LOCK SBB are the retry loop.
        ("lock add [rsp], rcx", &[0xF0, 0x48, 0x01, 0x0C, 0x24]),
        ("lock sub [rsp], rcx", &[0xF0, 0x48, 0x29, 0x0C, 0x24]),
        ("lock adc [rsp], rcx", &[0xF0, 0x48, 0x11, 0x0C, 0x24]),
        ("lock sbb [rsp], rcx", &[0xF0, 0x48, 0x19, 0x0C, 0x24]),
        ("lock inc qword [rsp]", &[0xF0, 0x48, 0xFF, 0x04, 0x24]),
        ("lock dec qword [rsp]", &[0xF0, 0x48, 0xFF, 0x0C, 0x24]),
        ("lock neg qword [rsp]", &[0xF0, 0x48, 0xF7, 0x1C, 0x24]),
        (
            "lock xadd [rsp], rcx",
            &[0xF0, 0x48, 0x0F, 0xC1, 0x0C, 0x24],
        ),
        (
            "lock cmpxchg [rsp], rcx",
            &[0xF0, 0x48, 0x0F, 0xB1, 0x0C, 0x24],
        ),
    ];
    cases
        .iter()
        .map(|(name, bytes)| {
            let mut v = bytes.to_vec();
            v.extend_from_slice(READ);
            v.push(0xC3); // ret
            (*name, v)
        })
        .collect()
}

/// Operand pairs straddling the boundaries CF and OF disagree about.
///
/// `(rax, rcx, cf_in)`. The unsigned wrap and the signed one are different
/// events - `0x7FFF..F + 1` overflows signed and not unsigned, `-1 + 1` the
/// other way round - so a `uadd_overflow` where a `sadd_overflow` belonged, or
/// either with its operands swapped, shows up here and nowhere in random state.
const OPERANDS: &[(u64, u64, u8)] = &[
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 1),
    (u64::MAX, 1, 0),
    (u64::MAX, 1, 1),
    (u64::MAX, u64::MAX, 1),
    (0x7FFF_FFFF_FFFF_FFFF, 1, 0),
    (0x7FFF_FFFF_FFFF_FFFF, 1, 1),
    (0x8000_0000_0000_0000, 1, 0),
    (0x8000_0000_0000_0000, u64::MAX, 0),
    (0x8000_0000_0000_0000, 0x8000_0000_0000_0000, 0),
    // 8- and 16-bit boundaries, in the low bytes of the same registers.
    (0x0000_0000_0000_007F, 1, 0),
    (0x0000_0000_0000_0080, 0xFF, 0),
    (0x0000_0000_0000_00FF, 1, 1),
    (0x0000_0000_0000_7FFF, 1, 0),
    (0x0000_0000_0000_8000, 0xFFFF, 0),
    (0x0000_0000_7FFF_FFFF, 1, 0),
    (0x0000_0000_FFFF_FFFF, 1, 0),
    (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, 1),
];

/// Everything a guest could observe afterwards.
type State = ([u64; 6], [u8; 6], u32);

fn state(c: &RegisterContext) -> State {
    (
        [c.rax, c.rbx, c.rcx, c.rdx, c.rsi, c.rdi],
        [c.cf, u8::from(c.pf()), u8::from(c.af()), c.zf, c.sf, c.of],
        c.exit_reason,
    )
}

fn run(bytes: &[u8], spec: Specialization, lean: bool, ops: (u64, u64, u8)) -> State {
    let (rax, rcx, cf) = ops;
    let t = LiftTest::new(bytes)
        .specialize(spec)
        .reg(move |c| {
            c.rax = rax;
            c.rcx = rcx;
            c.cf = cf;
        })
        .stack_words(8);
    let t = if lean { t.eliminate_dead_flags() } else { t };
    state(&t.run().ctx)
}

/// The specialized lowering computes exactly what the portable one computes,
/// for every mnemonic, every width and every boundary operand pair.
///
/// The portable run is with the specialization off and the liveness pass *on*,
/// which is the configuration this feature is an optimisation of - so a
/// divergence here is this feature's, not the dead-flag pass's.
#[test]
fn agrees_with_the_portable_paths() {
    for (name, bytes) in snippets() {
        for &ops in OPERANDS {
            let portable = run(&bytes, Specialization::generic(), true, ops);
            for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
                let spec = run(&bytes, Specialization::for_target(target), true, ops);
                assert_eq!(
                    portable,
                    spec,
                    "{name}: {} diverged from the portable path on {ops:x?}",
                    target.name()
                );
            }
        }
    }
}

/// The dead-flag pass itself is not perturbed: with the specialization on, the
/// answer still matches a lift that has neither.
#[test]
fn agrees_with_the_unoptimised_lifter() {
    for (name, bytes) in snippets() {
        for &ops in OPERANDS {
            let plain = run(&bytes, Specialization::generic(), false, ops);
            let both = run(&bytes, Specialization::host(), true, ops);
            assert_eq!(
                plain, both,
                "{name}: diverged from a plain lift on {ops:x?}"
            );
        }
    }
}

/// Without flag liveness the specialization emits nothing.
#[test]
fn no_overflow_ops_without_liveness() {
    for (name, bytes) in snippets() {
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            let spec = Specialization::for_target(target);
            assert_eq!(
                overflow_ops(&bytes, spec, false),
                0,
                "{name}: emitted an overflow op on {} without the liveness pass",
                target.name()
            );
        }
        assert_eq!(
            overflow_ops(&bytes, Specialization::generic(), true),
            0,
            "{name}: emitted an overflow op with no target"
        );
    }
}

/// A snippet whose operands are narrower than 32 bits.
///
/// aarch64's `ADDS`/`SUBS` start at 32 bits, so Cranelift has to synthesize the
/// overflow ops below that and the result is a measured regression; the lifter
/// declines there. See `Lifter::native_overflow_width_ok` and
/// `specialize_tests::specializations_shrink_the_compiled_code`.
fn is_sub_word(name: &str) -> bool {
    name.ends_with(" r16") || name.ends_with(" r8")
}

/// The width gate, in both directions: a sub-word site declines on aarch64 and
/// still fires on x86-64, with CF and OF live either way.
///
/// The two halves are both required. Without the first, a lifter that ignored
/// the width would pass every equivalence test while emitting the sequence that
/// was measured as a regression; without the second, a gate that was accidentally
/// unconditional would look exactly the same from here.
#[test]
fn sub_word_overflow_ops_are_aarch64_only() {
    for (name, bytes) in snippets().into_iter().filter(|(n, _)| is_sub_word(n)) {
        assert_eq!(
            overflow_ops(
                &bytes,
                Specialization::for_target(SpecTarget::Aarch64),
                true
            ),
            0,
            "{name}: aarch64 has no flag-setting arithmetic at this width"
        );
        assert!(
            overflow_ops(&bytes, Specialization::for_target(SpecTarget::X86_64), true) > 0,
            "{name}: x86-64 has ADD/SUB at every width and should still specialize"
        );
    }
}

#[test]
fn liveness_lets_the_overflow_ops_through() {
    for (name, bytes) in snippets() {
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            if name.starts_with("imul") && target == SpecTarget::X86_64 {
                continue;
            }
            if is_sub_word(name) && target == SpecTarget::Aarch64 {
                continue;
            }
            let n = overflow_ops(&bytes, Specialization::for_target(target), true);
            assert!(
                n > 0,
                "{name}: no overflow op on {} even with CF and OF live",
                target.name()
            );
        }
    }
}

/// A site whose CF and OF nothing reads gets a plain `iadd`, not an
/// overflow-producing add whose condition byte is thrown away.
///
/// This is the request the whole gate exists to satisfy, and the snippets are
/// chosen so the answer is unambiguous: the flags of the *first* instruction
/// are overwritten by the second before anything looks at them, and the second
/// instruction is one this specialization has no opinion about.
#[test]
fn dead_flags_leave_a_plain_add() {
    // <insn> ; test rax, rax ; sete dl ; ret
    //
    // `TEST` writes CF, OF, ZF, SF and PF, so the first instruction's CF and OF
    // are both dead. AF is *not* written by TEST (it is left unchanged), which
    // is why the first instruction still emits its AF - and why this checks the
    // overflow opcodes specifically rather than the total IR length.
    const KILL: &[u8] = &[0x48, 0x85, 0xC0, 0x0F, 0x94, 0xC2, 0xC3];
    let cases: &[(&str, &[u8])] = &[
        ("add r64", &[0x48, 0x01, 0xC8]),
        ("sub r64", &[0x48, 0x29, 0xC8]),
        ("cmp r64", &[0x48, 0x39, 0xC8]),
        ("inc r64", &[0x48, 0xFF, 0xC0]),
        ("dec r64", &[0x48, 0xFF, 0xC8]),
        ("neg r64", &[0x48, 0xF7, 0xD8]),
        ("imul r64,r64", &[0x48, 0x0F, 0xAF, 0xC1]),
    ];

    for (name, insn) in cases {
        let mut bytes = insn.to_vec();
        bytes.extend_from_slice(KILL);
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            assert_eq!(
                overflow_ops(&bytes, Specialization::for_target(target), true),
                0,
                "{name}: emitted an overflow op on {} for flags nothing reads",
                target.name()
            );
        }
    }
}

/// `ADC`/`SBB` need their carry *in* whatever happens, so a dead carry *out*
/// still leaves the two-step arithmetic - but no overflow ops, and no carry
/// chain.
#[test]
fn a_dead_carry_out_still_leaves_the_carry_in() {
    // adc rax, rcx ; test rax, rax ; sete dl ; ret
    let bytes = &[
        0x48, 0x11, 0xC8, // adc rax, rcx
        0x48, 0x85, 0xC0, // test rax, rax
        0x0F, 0x94, 0xC2, // sete dl
        0xC3,
    ];
    for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
        assert_eq!(
            overflow_ops(bytes, Specialization::for_target(target), true),
            0,
            "{}: ADC emitted an overflow op for a carry out nothing reads",
            target.name()
        );
    }
    // And it still adds the carry in: 1 + 1 + CF(1) == 3.
    let out = LiftTest::new(bytes)
        .specialize(Specialization::host())
        .eliminate_dead_flags()
        .reg(|c| {
            c.rax = 1;
            c.rcx = 1;
            c.cf = 1;
        })
        .run();
    assert_eq!(out.ctx.rax, 3, "ADC dropped its carry in");
}

/// The specialized IR lowers for the target it names, cross-compiled.
///
/// riscv64 is the reason [`SpecTarget::Generic`] gets none of this: it lowers
/// `uadd_overflow` alone, at `I32`/`I64` only. Compiling the *generic* lift for
/// it is the check that nothing leaks out of the gate.
#[test]
fn each_target_lowers_its_own_ir() {
    let pairs = [
        (SpecTarget::X86_64, "x86_64-unknown-linux-gnu"),
        (SpecTarget::Aarch64, "aarch64-unknown-linux-gnu"),
    ];

    for (name, bytes) in snippets() {
        for (target, triple) in pairs {
            let cfg = config(Specialization::for_target(target), true);
            let mut code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, cfg).expect("lift");
            // The lifter bakes in the host calling convention; override it so
            // this measures instruction lowering rather than the target's ABI
            // code rejecting a convention.
            code.func.signature.call_conv = CallConv::SystemV;
            ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
                .unwrap_or_else(|e| panic!("{name} failed to lower for {triple}: {e:?}"));
        }
    }
}

#[test]
fn generic_still_lowers_for_riscv64() {
    for (name, bytes) in snippets() {
        let cfg = config(Specialization::generic(), true);
        let mut code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, cfg).expect("lift");
        code.func.signature.call_conv = CallConv::SystemV;
        ArtifactBackend::compile_lifted_to_raw(code, Some("riscv64gc-unknown-linux-gnu"), None)
            .unwrap_or_else(|e| panic!("{name} failed to lower for riscv64: {e:?}"));
    }
}
