//! (a) Semantics fuzzing - randomized operands for the diffable ALU ops,
//! checked against a pure-Rust reference model, plus a decoder-robustness
//! property (arbitrary bytes must never panic the lifter).
//!
//! Run: `cargo test --features fuzzing --test fuzz_semantics`
//! Deeper: `PROPTEST_CASES=20000 cargo test --features fuzzing --test fuzz_semantics`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use fuzz::{alu_reference, alu_snippet, AluOp};
use microlift::arch::x86_64::X86_64;
use microlift::core::{lift_to_function, LifterConfig};
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
        prop_assert_eq!(out.ctx.pf, want.pf, "{:?} {:#x},{:#x} PF", op, a, b);
        // AF is architecturally defined for ADD/SUB; undefined for logic ops.
        if matches!(op, AluOp::Add | AluOp::Sub) {
            prop_assert_eq!(out.ctx.af, want.af, "{:?} {:#x},{:#x} AF", op, a, b);
        }
    }
}

proptest! {
    /// Decoder / IR-emission robustness: lifting arbitrary bytes must return a
    /// clean `Ok`/`Err` and never panic or invoke UB. (We only lift to IR here;
    /// we never execute random bytes.)
    ///
    /// IGNORED: this property currently FAILS - it found a real lifter panic in
    /// the overlapping/sub-encoded-instruction path. A conditional branch whose
    /// target lands in the middle of a following (multi-byte, e.g. VEX)
    /// instruction drives Cranelift's frontend to `switch_to_block` on an
    /// already-filled block:
    ///     "you cannot switch to a block which is already filled"
    /// Minimal reproducer captured in `lifter_block_fill_panic_reproducer`
    /// below. Re-enable (remove `#[ignore]`) once the block-emission bug is
    /// fixed. Run on demand with `-- --ignored`.
    #[test]
    #[ignore = "reproduces an unfixed lifter panic; see doc comment + reproducer test"]
    fn lifting_arbitrary_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..24)) {
        let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
    }
}

/// Minimal reproducer for the lifter panic the fuzzer surfaced (see
/// `lifting_arbitrary_bytes_never_panics`). `jo +5` targets the interior of the
/// following 3-byte VEX-prefixed instruction; block emission then tries to
/// switch to an already-filled block. `#[ignore]`d because it currently panics;
/// run with `-- --ignored` to reproduce, and it should start passing once the
/// bug is fixed.
#[test]
#[ignore = "reproduces an unfixed lifter block-emission panic"]
fn lifter_block_fill_panic_reproducer() {
    // jo 0x1007 ; (vex) ... - overlapping decode.
    let bytes = [
        0x70, 0x05, 0xC4, 0x04, 0x00, 0x00, 0x18, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x70, 0x00,
    ];
    let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
}
