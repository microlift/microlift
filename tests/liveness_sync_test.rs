use microlift::testkit::prelude::*;

#[test]
fn test_liveness_sync_add() {
    // add rax, rbx; ret
    let bytes = [0x48, 0x01, 0xD8, 0xC3];
    let out = LiftTest::new(&bytes)
        .reg(|c| {
            c.rax = 100;
            c.rbx = 42;
        })
        .run();

    // Verify sync
    assert_eq!(out.ctx.rax, 142);
    assert_eq!(out.ctx.rbx, 42);
    assert_eq!(out.ctx.zf, 0);
}

#[test]
fn test_liveness_sync_flags() {
    // xor rax, rax; ret (should set ZF=1)
    let bytes = [0x48, 0x31, 0xC0, 0xC3];
    let out = LiftTest::new(&bytes)
        .at(0x2000)
        .image_base(0x2000)
        .reg(|c| c.rax = 999)
        .run();

    // Verify sync
    assert_eq!(out.ctx.rax, 0);
    assert_eq!(out.ctx.zf, 1);
}
