//! Minimal smoke test and canonical template for the shared testkit harness.

use microlift::testkit::prelude::*;

#[test]
fn test_mov_rax_imm() {
    // mov rax, 42 ; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.ctx.rax, 42);
}
