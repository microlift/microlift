//! `LifterConfig::address_mode`: ImageBaseRelative survives relocation, Direct bakes lift-time VAs.
//! The two must agree when `image_base == lift_base`.

use microlift::testkit::prelude::*;

const LIFT_BASE: u64 = 0x1000;
// Differs from LIFT_BASE so passing under both modes can't be coincidence.
const RELOC_BASE: u64 = 0x8000_0000;

#[test]
fn direct_rip_relative_lea_ignores_image_base() {
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .run();

    assert_eq!(out.ctx.rax, 0x1017);
}

#[test]
fn relative_rip_relative_lea_tracks_image_base() {
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::ImageBaseRelative)
        // Unicorn has no relocation base.
        .no_unicorn()
        .run();

    assert_eq!(out.ctx.rax, RELOC_BASE + 0x17);
}

#[test]
fn direct_out_of_region_jmp_reports_lift_time_va() {
    let bytes = [0xEB, 0x0E, 0x90, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .run();

    assert_eq!(out.ctx.rip, 0x1010);

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::ImageBaseRelative)
        // Unicorn has no relocation base.
        .no_unicorn()
        .run();

    assert_eq!(out.ctx.rip, RELOC_BASE + 0x10);
}

#[test]
fn direct_call_pushes_lift_time_return_address() {
    let bytes = [
        0xE8, 0x00, 0x00, 0x00, 0x00, // call 0x1005
        0x58, // pop rax
        0xC3, // ret
    ];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .run();

    assert_eq!(
        out.ctx.rax, 0x1005,
        "Direct mode must push the lift-time return address"
    );

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::ImageBaseRelative)
        // Unicorn has no relocation base.
        .no_unicorn()
        .run();

    assert_eq!(out.ctx.rax, RELOC_BASE + 5);
}

#[test]
fn direct_exit_pc_is_a_lift_time_va() {
    let bytes = [0x0F, 0x0B];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .run();

    assert_eq!(out.exit, ExitReason::Ud2);
    assert_eq!(out.ctx.rip, LIFT_BASE);
}

#[test]
fn direct_reentry_dispatches_on_lift_time_pc() {
    let bytes = [0xCC, 0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .with_config(|c| c.resumption = VmResumptionPoints::All)
        .run();

    assert_eq!(out.exit, ExitReason::Breakpoint);
    assert_eq!(out.ctx.rip, LIFT_BASE);

    let out = out.resume_with(|ctx| ctx.rip = LIFT_BASE + 1);
    assert_eq!(out.ctx.rax, 42);
}

#[test]
fn modes_agree_when_image_base_equals_lift_base() {
    let bytes = [
        0x48, 0x8D, 0x05, 0x08, 0x00, 0x00, 0x00, // lea rax, [rip+8]
        0xE8, 0x00, 0x00, 0x00, 0x00, // call next
        0x59, // pop rcx
        0x48, 0x31, 0xD2, // xor rdx, rdx
        0xC3, // ret
    ];

    let run = |mode| {
        LiftTest::new(&bytes)
            .at(LIFT_BASE)
            .image_base(LIFT_BASE)
            .address_mode(mode)
            .run()
            .ctx
    };

    let rel = run(AddressMode::ImageBaseRelative);
    let dir = run(AddressMode::Direct);

    assert_eq!(rel.rax, dir.rax);
    assert_eq!(rel.rcx, dir.rcx);
    assert_eq!(rel.rdx, dir.rdx);
    assert_eq!(rel.rip, dir.rip);
    assert_eq!(rel.exit_reason, dir.exit_reason);
}
