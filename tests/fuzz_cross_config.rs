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

use fuzz::{alu_snippet, fingerprint, AluOp};
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
}
