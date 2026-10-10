//! Randomized ALU operands against a Rust reference model, plus a never-panics decoder property.
//! Deeper: `PROPTEST_CASES=20000 cargo test --features fuzzing --test fuzz_semantics`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use fuzz::{AluOp, alu_reference, alu_snippet};
use microlift::arch::x86_64::X86_64;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;
use proptest::prelude::*;

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
        // AF is undefined for logic ops.
        if matches!(op, AluOp::Add | AluOp::Sub) {
            prop_assert_eq!(u8::from(out.ctx.af()), want.af, "{:?} {:#x},{:#x} AF", op, a, b);
        }
    }
}

proptest! {
    /// Lifting arbitrary bytes must return `Ok`/`Err`, never panic (bytes are not executed).
    #[test]
    fn lifting_arbitrary_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..24)) {
        let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
    }
}

/// Regression: tail-duplicated `jo` hit `switch_to_block` on an already-filled fallthrough block.
#[test]
fn lifter_block_fill_panic_reproducer() {
    let bytes = [
        0x70, 0x05, 0xC4, 0x04, 0x00, 0x00, 0x18, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x70, 0x00,
    ];
    let _ = lift_to_function::<X86_64>(&bytes, 0x1000, LifterConfig::default());
}
