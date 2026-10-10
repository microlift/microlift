//! Metamorphic check: the same bytes give the same state under every opt level and XMM representation.
//! Run: `cargo test --features fuzzing --test fuzz_cross_config`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use fuzz::{AluOp, alu_snippet, fingerprint};
use microlift::testkit::prelude::*;
use proptest::prelude::*;

fn any_alu_op() -> impl Strategy<Value = AluOp> {
    prop::sample::select(AluOp::ALL.to_vec())
}

fn run_cfg(bytes: &[u8], opt: Opt, vector_xmm: bool) -> fuzz::StateFingerprint {
    let out = LiftTest::new(bytes).opt(opt).vector_xmm(vector_xmm).run();
    fingerprint(&out.ctx)
}

proptest! {
    #[test]
    fn opt_and_xmm_agree(op in any_alu_op(), a in any::<u64>(), b in any::<u64>()) {
        let bytes = alu_snippet(op, a, b);
        let base = run_cfg(&bytes, Opt::None, false);

        for (opt, vec) in [
            (Opt::Speed, false),
            (Opt::SpeedAndSize, false),
            (Opt::None, true),
            (Opt::Speed, true),
        ] {
            let got = run_cfg(&bytes, opt, vec);
            prop_assert_eq!(
                &got, &base,
                "config divergence for {:?} {:#x},{:#x} at opt={:?} vector_xmm={}",
                op, a, b, opt, vec
            );
        }
    }

    /// Dropping unused context traffic may not change any state.
    #[test]
    fn lazy_register_sync_changes_nothing(
        op in any_alu_op(),
        a in any::<u64>(),
        b in any::<u64>(),
    ) {
        let bytes = alu_snippet(op, a, b);
        let base = LiftTest::new(&bytes)
            .opt(Opt::None)
            .with_config(|c| c.lazy_register_sync = false)
            .run();
        let base = fingerprint(&base.ctx);

        for opt in [Opt::None, Opt::Speed, Opt::SpeedAndSize] {
            for lazy in [false, true] {
                let out = LiftTest::new(&bytes)
                    .opt(opt)
                    .with_config(|c| c.lazy_register_sync = lazy)
                    .run();
                prop_assert_eq!(
                    &fingerprint(&out.ctx), &base,
                    "context-sync divergence for {:?} {:#x},{:#x} at opt={:?} lazy={}",
                    op, a, b, opt, lazy
                );
            }
        }
    }

    /// Dead flag removal may not change state, including flags published at exits.
    #[test]
    fn dead_flag_elimination_changes_nothing(
        op in any_alu_op(),
        a in any::<u64>(),
        b in any::<u64>(),
    ) {
        let bytes = alu_snippet(op, a, b);
        let run = |lean: bool| {
            let test = LiftTest::new(&bytes).opt(Opt::Speed);
            let test = if lean { test.eliminate_dead_flags() } else { test };
            fingerprint(&test.run().ctx)
        };
        prop_assert_eq!(
            run(true), run(false),
            "dead-flag divergence for {:?} {:#x},{:#x}",
            op, a, b
        );
    }

    /// A lone ALU+RET keeps all flags live; mixed programs cover overwrites, carry chains and joins.
    #[test]
    fn dead_flags_agree_over_mixed_programs(
        instructions in prop::collection::vec(0usize..18, 2..32),
        a in any::<u64>(), b in any::<u64>(), count in any::<u8>(), bits in any::<u8>(),
    ) {
        const OPS: &[&[u8]] = &[
            &[0x48, 0x01, 0xD8], // add rax, rbx
            &[0x48, 0x29, 0xD8], // sub rax, rbx
            &[0x48, 0x11, 0xD8], // adc rax, rbx
            &[0x48, 0x19, 0xD8], // sbb rax, rbx
            &[0x48, 0xFF, 0xC0], // inc rax
            &[0x48, 0xFF, 0xC8], // dec rax
            &[0x48, 0x31, 0xD8], // xor rax, rbx
            &[0x48, 0x85, 0xC0], // test rax, rax
            &[0x48, 0xD3, 0xE0], // shl rax, cl
            &[0x48, 0xD3, 0xE8], // shr rax, cl
            &[0x48, 0x0F, 0xAF, 0xC3], // imul rax, rbx
            &[0x9F], // lahf
            &[0x9E], // sahf
            &[0xF9], // stc
            &[0xF5], // cmc
            &[0x0F, 0x94, 0xC2], // sete dl
            &[0x48, 0x0F, 0x43, 0xFB], // cmovae rdi, rbx
            &[0x74, 0x03, 0x48, 0x31, 0xF6], // je over xor rsi, rsi
        ];
        let mut bytes = Vec::new();
        for index in instructions { bytes.extend_from_slice(OPS[index]); }
        bytes.push(0xC3);
        let run = |lean, opt, lazy, spec| {
            LiftTest::new(&bytes).opt(opt).specialize(spec)
                .reg(|c| {
                    c.rax = a; c.rbx = b; c.rcx = u64::from(count);
                    c.cf = bits & 1; c.set_pf(bits & 2 != 0); c.set_af(bits & 4 != 0);
                    c.zf = (bits >> 3) & 1; c.sf = (bits >> 4) & 1; c.of = (bits >> 5) & 1;
                })
                .with_config(|c| { c.eliminate_dead_flags = lean; c.lazy_register_sync = lazy; })
                .run()
        };
        let baseline = fingerprint(&run(false, Opt::None, false, Specialization::generic()).ctx);
        for opt in [Opt::None, Opt::Speed] {
            for lazy in [false, true] {
                for spec in [Specialization::generic(), Specialization::host()] {
                    prop_assert_eq!(&fingerprint(&run(true, opt, lazy, spec).ctx), &baseline);
                }
            }
        }
    }

    /// Native CF/OF may not change state. `eliminate_dead_flags` is on in every arm, or the spec is inert.
    /// Both targets run on any host; this checks the IR each path emits.
    #[test]
    fn native_overflow_flags_change_nothing(
        op in any_alu_op(),
        a in any::<u64>(),
        b in any::<u64>(),
    ) {
        let bytes = alu_snippet(op, a, b);
        let run = |spec| {
            fingerprint(
                &LiftTest::new(&bytes)
                    .opt(Opt::Speed)
                    .eliminate_dead_flags()
                    .specialize(spec)
                    .run()
                    .ctx,
            )
        };
        let base = run(Specialization::generic());
        for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
            prop_assert_eq!(
                run(Specialization::for_target(target)),
                base.clone(),
                "overflow-flag divergence on {} for {:?} {:#x},{:#x}",
                target.name(), op, a, b
            );
        }
    }

    /// `Direct` and `ImageBaseRelative` must agree when the image sits at its lift base.
    #[test]
    fn address_modes_agree_at_lift_base(op in any_alu_op(), a in any::<u64>(), b in any::<u64>()) {
        let bytes = alu_snippet(op, a, b);

        let run = |mode| {
            fingerprint(
                &LiftTest::new(&bytes)
                    .address_mode(mode)
                    .run()
                    .ctx,
            )
        };

        prop_assert_eq!(
            run(AddressMode::Direct),
            run(AddressMode::ImageBaseRelative),
            "address-mode divergence for {:?} {:#x},{:#x}",
            op, a, b
        );
    }
}
