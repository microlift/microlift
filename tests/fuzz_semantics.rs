//! (a) Semantics fuzzing - randomized operands for the diffable ALU ops,
//! checked against a pure-Rust reference model, plus a decoder-robustness
//! property (arbitrary bytes must never panic the lifter).
//!
//! Run: `cargo test --features fuzzing --test fuzz_semantics`
//! Deeper: `PROPTEST_CASES=20000 cargo test --features fuzzing --test fuzz_semantics`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use fuzz::{AluOp, alu_reference, alu_snippet};
use microlift::arch::x86_64::X86_64;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;
use proptest::prelude::*;

/// Strategy selecting one of the five ALU ops.
fn any_alu_op() -> impl Strategy<Value = AluOp> {
    prop_oneof![
        Just(AluOp::Add),
        Just(AluOp::Sub),
        Just(AluOp::And),
        Just(AluOp::Or),
        Just(AluOp::Xor),
    ]
}

proptest! {
    /// `op rax, rcx` must match the reference model on both result and flags.
    #[test]
    fn alu_matches_reference(op in any_alu_op(), a in any::<u64>(), b in any::<u64>()) {
        let bytes = alu_snippet(op, a, b);
        let out = LiftTest::new(&bytes).run();
        let want = alu_reference(op, a, b);

        prop_assert_eq!(out.ctx.rax, want.result, "{:?} {:#x},{:#x} result", op, a, b);
        prop_assert_eq!(out.ctx.cf, want.cf, "{:?} {:#x},{:#x} CF", op, a, b);
        prop_assert_eq!(out.ctx.zf, want.zf, "{:?} {:#x},{:#x} ZF", op, a, b);
        prop_assert_eq!(out.ctx.sf, want.sf, "{:?} {:#x},{:#x} SF", op, a, b);
        prop_assert_eq!(out.ctx.of, want.of, "{:?} {:#x},{:#x} OF", op, a, b);
        prop_assert_eq!(u8::from(out.ctx.pf()), want.pf, "{:?} {:#x},{:#x} PF", op, a, b);
        // AF is architecturally defined for ADD/SUB; undefined for logic ops.
        if matches!(op, AluOp::Add | AluOp::Sub) {
            prop_assert_eq!(u8::from(out.ctx.af()), want.af, "{:?} {:#x},{:#x} AF", op, a, b);
        }
    }
}

proptest! {
    /// Decoder / IR-emission robustness: lifting arbitrary bytes must return a
    /// clean `Ok`/`Err` and never panic or invoke UB. (We only lift to IR here;
    /// we never execute random bytes.)
    #[test]
    fn lifting_arbitrary_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..24)) {
        let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
    }
}

/// Regression test for a lifter panic the fuzzer surfaced (see
/// `lifting_arbitrary_bytes_never_panics`). Straight-line code after a leader is
/// tail-duplicated into every block that reaches it; here `jo 0x100E` at 0x100C
/// is emitted both as the tail of block 0x1006 and as the tail of block 0x1007.
/// The second copy used to `switch_to_block` on its fallthrough block, which the
/// first copy had already filled - "you cannot switch to a block which is
/// already filled". `continue_in_fallthrough` now terminates instead.
#[test]
fn lifter_block_fill_panic_reproducer() {
    // jo 0x1007 ; (bad) ; sbb [100Ch],cl / or eax,0 ; jo 0x100E - overlapping decode.
    let bytes = [
        0x70, 0x05, 0xC4, 0x04, 0x00, 0x00, 0x18, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x70, 0x00,
    ];
    let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
}
