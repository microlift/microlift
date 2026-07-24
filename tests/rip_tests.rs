use microlift::testkit::prelude::*;

fn run_rip_test(
    bytes: &[u8],
    ip: u64,
    base_address: u64,
    ctx: RegisterContext,
) -> RegisterContext {
    LiftTest::new(bytes)
        .at(ip)
        .image_base(base_address)
        .ctx(ctx)
        .run()
        .ctx
}

#[test]
fn test_rip_relative_lea() {
    // lea rax, [rip + 0x10]; ret
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    // Case 1: base_address matches ip
    let ip = 0x1000;
    let base_address = 0x1000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rax, 0x1017);
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF); // Popped trampoline
}

#[test]
fn test_rip_sync_on_exit() {
    // mov rax, 1; ret
    let bytes = [0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let ip = 0x2000;
    let base_address = 0x500000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_rip_relative_jmp() {
    // jmp 0x1010; nop; ret
    let bytes = [0xEB, 0x0E, 0x90, 0xC3];
    let ip = 0x1000;
    let base_address = 0x8000;
    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());

    // reconciled_rip = 0x8000 + (0x1010 - 0x1000) = 0x8010
    assert_eq!(ctx.rip, 0x8010);
}

#[test]
fn test_rip_relative_cond_jmp() {
    // xor rax, rax; jz 0x1010; ret
    let bytes = [0x48, 0x31, 0xC0, 0x74, 0x0B, 0xC3];
    let ip = 0x1000;
    let base_address = 0x9000;

    let ctx = run_rip_test(&bytes, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0x9010);

    let bytes_jnz = [0x48, 0x31, 0xC0, 0x75, 0x0B, 0xC3];
    let ctx = run_rip_test(&bytes_jnz, ip, base_address, RegisterContext::default());
    assert_eq!(ctx.rip, 0xFFFF_FFFF_FFFF_FFFF);
}
