//! `LifterConfig::address_mode`: how guest virtual addresses are materialized.
//!
//! The default [`AddressMode::ImageBaseRelative`] forms every guest address as
//! `image_base + (va - lift_base)` so a compiled region survives being loaded
//! at a different base (and can therefore be cached on disk). [`AddressMode::Direct`]
//! bakes the lift-time VA in as a constant instead - cheaper, but valid only at
//! the base it was lifted for. These tests pin both behaviours, and that the two
//! agree whenever `image_base == lift_base`.

use microlift::testkit::prelude::*;

const LIFT_BASE: u64 = 0x1000;
/// A relocation base deliberately different from `LIFT_BASE`, so a test that
/// passes under both modes cannot be passing by coincidence.
const RELOC_BASE: u64 = 0x8000_0000;

#[test]
fn direct_rip_relative_lea_ignores_image_base() {
    // lea rax, [rip + 0x10]; ret
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        .run();

    // Direct: the lift-time target address is baked in, so the seeded
    // `image_base` has no effect at all.
    assert_eq!(out.ctx.rax, 0x1017);
}

#[test]
fn relative_rip_relative_lea_tracks_image_base() {
    // lea rax, [rip + 0x10]; ret
    let bytes = [0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::ImageBaseRelative)
        .run();

    assert_eq!(out.ctx.rax, RELOC_BASE + 0x17);
}

#[test]
fn direct_out_of_region_jmp_reports_lift_time_va() {
    // jmp 0x1010 ; nop ; ret  - the target is outside the 4-byte region.
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
        .run();

    assert_eq!(out.ctx.rip, RELOC_BASE + 0x10);
}

#[test]
fn direct_call_pushes_lift_time_return_address() {
    // The classic "get my own address": call the next instruction, then pop the
    // pushed return address into rax.
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
        .run();

    assert_eq!(out.ctx.rax, RELOC_BASE + 5);
}

#[test]
fn direct_exit_pc_is_a_lift_time_va() {
    // ud2 - the reported RIP points *at* the faulting instruction.
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
    // int3            @ 0x1000 (Breakpoint exit, RIP at the INT3)
    // mov rax, 0x2A   @ 0x1001
    // ret             @ 0x1008
    let bytes = [0xCC, 0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

    let out = LiftTest::new(&bytes)
        .at(LIFT_BASE)
        .image_base(RELOC_BASE)
        .address_mode(AddressMode::Direct)
        // Make every instruction boundary a block so there is a resume point
        // just past the INT3.
        .with_config(|c| c.emit_resume_blocks = true)
        .run();

    assert_eq!(out.exit, ExitReason::Breakpoint);
    assert_eq!(out.ctx.rip, LIFT_BASE);

    // Step over the INT3 and re-enter: the entry dispatch must route a plain
    // lift-time VA back to the right block.
    let out = out.resume_with(|ctx| ctx.rip = LIFT_BASE + 1);
    assert_eq!(out.ctx.rax, 42);
}

/// The two modes are interchangeable when the image is loaded exactly where it
/// was lifted - the case an in-process JIT actually has.
#[test]
fn modes_agree_when_image_base_equals_lift_base() {
    // lea rax, [rip + 8] ; call +0 ; pop rcx ; xor rdx, rdx ; ret
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
