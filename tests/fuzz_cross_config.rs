//! Cross-configuration equivalence - the "semantics must survive
//! optimization" requirement expressed as a metamorphic relation: the same
//! lifted bytes must produce the same architectural state regardless of
//! Cranelift optimization level or XMM representation. Needs no oracle; any
//! divergence is a miscompile.
//!
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

/// Run one snippet under a specific (opt, vector_xmm) configuration and return
/// its config-invariant fingerprint.
fn run_cfg(bytes: &[u8], opt: Opt, vector_xmm: bool) -> fuzz::StateFingerprint {
    let out = LiftTest::new(bytes).opt(opt).vector_xmm(vector_xmm).run();
    fingerprint(&out.ctx)
}

proptest! {
    /// Every optimization level and XMM representation must agree on the final
    /// state for the same bytes.
    #[test]
    fn opt_and_xmm_agree(op in any_alu_op(), a in any::<u64>(), b in any::<u64>()) {
        let bytes = alu_snippet(op, a, b);
        // Baseline: unoptimized, scalar XMM.
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

    /// Skipping the flag writes nothing reads may not change a single bit of
    /// state - flags very much included, since a mistake in the classification
    /// shows up there and nowhere else.
    ///
    /// The dedicated suite (`dead_flag_tests`) covers the seams by hand; this
    /// is the half that tries operands nobody thought of.
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

    /// `AddressMode::Direct` bakes lift-time VAs in as constants instead of
    /// rebasing them onto `image_base`. When the image is loaded exactly where
    /// it was lifted - which `LiftTest` does by default - the two must be
    /// indistinguishable, PC included.
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
