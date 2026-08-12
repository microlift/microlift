//! Segment *selector* register tests: `MOV r/m16, Sreg`, `MOV Sreg, r/m16`,
//! and `PUSH`/`POP FS`/`GS`.
//!
//! Distinct from `tls_tests.rs`, which covers the segment *bases* - the two are
//! independent state here exactly as they are on 64-bit hardware, and one of
//! the tests below pins that down.
//!
//! There is no Unicorn differential for any of this: seeding `CS`/`SS` in
//! Unicorn means writing a real descriptor selector into an emulated CPU that
//! is already in long mode, and the values a guest sees are a property of the
//! guest OS rather than of the code, so there is nothing an oracle could agree
//! with us about.

use microlift::arch::x86_64::context::{
    CS_USER64_DARWIN, CS_USER64_LINUX, SS_USER64_DARWIN, SS_USER64_LINUX,
};
use microlift::testkit::prelude::*;

#[test]
fn default_context_holds_user_mode_selectors() {
    let ctx = RegisterContext::default();
    assert_eq!(ctx.cs, CS_USER64_DARWIN);
    assert_eq!(ctx.ss, SS_USER64_DARWIN);
    // A real 64-bit user process reports 0 for these, base or no base.
    assert_eq!((ctx.ds, ctx.es, ctx.fs, ctx.gs), (0, 0, 0, 0));
}

/// The tail of Crashpad's `CaptureContext`, which is what motivated all of
/// this: three selector reads straight into a diagnostic struct.
#[test]
fn crashpad_capture_context_tail() {
    // mov ax, cs   ; 66 8C C8
    // mov cx, fs   ; 66 8C E1
    // mov dx, gs   ; 66 8C EA
    // ret          ; C3
    let out = LiftTest::new(&[0x66, 0x8C, 0xC8, 0x66, 0x8C, 0xE1, 0x66, 0x8C, 0xEA, 0xC3])
        .reg(|c| {
            c.cs = CS_USER64_LINUX;
            c.ss = SS_USER64_LINUX;
            c.fs = 0;
            c.gs = 0;
            // Nonsense in the upper bits, to prove a 16-bit write preserves them.
            c.rax = 0xAAAA_AAAA_AAAA_0000;
            c.rcx = 0xBBBB_BBBB_BBBB_1111;
            c.rdx = 0xCCCC_CCCC_CCCC_2222;
        })
        .run();

    assert_eq!(out.ctx.rax, 0xAAAA_AAAA_AAAA_0000 | CS_USER64_LINUX as u64);
    assert_eq!(out.ctx.rcx, 0xBBBB_BBBB_BBBB_0000);
    assert_eq!(out.ctx.rdx, 0xCCCC_CCCC_CCCC_0000);
}

#[test]
fn read_every_selector_16_bit() {
    // mov ax, es ; 66 8C C0     mov cx, cs ; 66 8C C9
    // mov dx, ss ; 66 8C D2     mov bx, ds ; 66 8C DB
    // mov si, fs ; 66 8C E6     mov di, gs ; 66 8C EF
    // ret        ; C3
    let out = LiftTest::new(&[
        0x66, 0x8C, 0xC0, 0x66, 0x8C, 0xC9, 0x66, 0x8C, 0xD2, 0x66, 0x8C, 0xDB, 0x66, 0x8C, 0xE6,
        0x66, 0x8C, 0xEF, 0xC3,
    ])
    .reg(|c| {
        c.es = 0x0011;
        c.cs = 0x0022;
        c.ss = 0x0033;
        c.ds = 0x0044;
        c.fs = 0x0055;
        c.gs = 0x0066;
    })
    .run();

    assert_eq!(out.ctx.rax, 0x11);
    assert_eq!(out.ctx.rcx, 0x22);
    assert_eq!(out.ctx.rdx, 0x33);
    assert_eq!(out.ctx.rbx, 0x44);
    assert_eq!(out.ctx.rsi, 0x55);
    assert_eq!(out.ctx.rdi, 0x66);
}

/// `MOV r32, Sreg` / `MOV r64, Sreg`: the selector is zero-extended into the
/// destination, so the upper bits are *cleared* rather than preserved.
#[test]
fn read_selector_into_wider_register_zero_extends() {
    // mov eax, cs ; 8C C8
    // mov rcx, gs ; 48 8C E9
    // ret         ; C3
    let out = LiftTest::new(&[0x8C, 0xC8, 0x48, 0x8C, 0xE9, 0xC3])
        .reg(|c| {
            c.cs = 0x002B;
            c.gs = 0x1234;
            c.rax = 0xFFFF_FFFF_FFFF_FFFF;
            c.rcx = 0xFFFF_FFFF_FFFF_FFFF;
        })
        .run();

    assert_eq!(out.ctx.rax, 0x2B);
    assert_eq!(out.ctx.rcx, 0x1234);
}

/// A memory destination is 16 bits wide whatever the operand size says.
#[test]
fn store_selector_to_memory() {
    // mov [rax], cs ; 8C 08
    // ret           ; C3
    let mut buf: [u16; 2] = [0xDEAD, 0xBEEF];
    let out = LiftTest::new(&[0x8C, 0x08, 0xC3])
        .reg(|c| {
            c.cs = 0x0033;
            c.rax = buf.as_mut_ptr() as u64;
        })
        .run();
    let _ = out;

    assert_eq!(buf[0], 0x0033);
    assert_eq!(buf[1], 0xBEEF, "only 16 bits may be written");
}

#[test]
fn write_selector_from_register() {
    // mov ds, ax ; 66 8E D8
    // mov es, cx ; 66 8E C1
    // ret        ; C3
    let out = LiftTest::new(&[0x66, 0x8E, 0xD8, 0x66, 0x8E, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0xFFFF_FFFF_0000_1234;
            c.rcx = 0x5678;
        })
        .run();

    assert_eq!(out.ctx.ds, 0x1234);
    assert_eq!(out.ctx.es, 0x5678);
}

/// Loading the `FS` selector does not move the `FS` *base*. Hardware would
/// reload the base from the descriptor; the lifter models selectors as storage
/// only, and a guest that changes its TLS this way is out of scope.
#[test]
fn selector_write_leaves_segment_base_alone() {
    // mov fs, ax ; 66 8E E0
    // ret        ; C3
    let out = LiftTest::new(&[0x66, 0x8E, 0xE0, 0xC3])
        .reg(|c| {
            c.rax = 0x0053;
            c.fs = 0;
            c.fs_base = 0x7FFF_0000_1234_5678;
        })
        .run();

    assert_eq!(out.ctx.fs, 0x53);
    assert_eq!(out.ctx.fs_base, 0x7FFF_0000_1234_5678);
}

/// `PUSH FS`/`PUSH GS` move `RSP` by 8, not by the 2 bytes the register is
/// wide, and zero-extend the selector into the slot.
#[test]
fn push_segment_uses_an_eight_byte_slot() {
    // push rcx ; 51   - sentinel, as in the 16-bit case below
    // push gs  ; 0F A8
    // pop rax  ; 58
    // pop rcx  ; 59
    // ret      ; C3
    let out = LiftTest::new(&[0x51, 0x0F, 0xA8, 0x58, 0x59, 0xC3])
        .reg(|c| {
            c.gs = 0x1234;
            c.rax = 0xFFFF_FFFF_FFFF_FFFF;
            c.rcx = 0x0BAD_C0DE_0BAD_C0DE;
        })
        .run();

    assert_eq!(
        out.ctx.rax, 0x1234,
        "the slot is the zero-extended selector"
    );
    assert_eq!(
        out.ctx.rcx, 0x0BAD_C0DE_0BAD_C0DE,
        "RSP must be back where it started"
    );
}

/// The `0x66`-prefixed spelling is the 2-byte one.
#[test]
fn push_segment_16_bit_uses_a_two_byte_slot() {
    // push rcx     ; 51        - sentinel, to catch a wrong slot width
    // push fs      ; 66 0F A0
    // pop ax       ; 66 58
    // pop rcx      ; 59
    // ret          ; C3
    //
    // An 8-byte `push fs` here would leave `RSP` 6 bytes low after the 2-byte
    // `pop ax`, and `pop rcx` would come back with something other than the
    // sentinel.
    let out = LiftTest::new(&[0x51, 0x66, 0x0F, 0xA0, 0x66, 0x58, 0x59, 0xC3])
        .reg(|c| {
            c.fs = 0x4321;
            c.rax = 0xFFFF_FFFF_FFFF_0000;
            c.rcx = 0x0BAD_C0DE_0BAD_C0DE;
        })
        .run();

    assert_eq!(out.ctx.rax, 0xFFFF_FFFF_FFFF_4321);
    assert_eq!(
        out.ctx.rcx, 0x0BAD_C0DE_0BAD_C0DE,
        "RSP must be back where it started"
    );
}

#[test]
fn pop_segment_loads_the_low_16_bits() {
    // push rax ; 50
    // pop fs   ; 0F A1
    // ret      ; C3
    let out = LiftTest::new(&[0x50, 0x0F, 0xA1, 0xC3])
        .reg(|c| {
            c.rax = 0xDEAD_BEEF_CAFE_0053;
            c.fs = 0;
            c.fs_base = 0xAAAA_BBBB_CCCC_DDDD;
        })
        .run();

    assert_eq!(out.ctx.fs, 0x0053);
    assert_eq!(out.ctx.fs_base, 0xAAAA_BBBB_CCCC_DDDD);
}

/// A selector read is a context read, so a value the host changes *between*
/// two entries into the same compiled region is seen by the second one. (An
/// SSA-variable-backed selector would be reloaded at entry too; this pins the
/// behaviour either way.)
#[test]
fn selector_reads_track_the_host_between_entries() {
    // mov ax, cs ; 66 8C C8
    // ret        ; C3
    let out = LiftTest::new(&[0x66, 0x8C, 0xC8, 0xC3])
        .resumable()
        .reg(|c| c.cs = 0x0033)
        .run();
    assert_eq!(out.ctx.rax & 0xFFFF, 0x33);

    let out = out.resume_with(|c| {
        c.cs = 0x002B;
        c.rip = c.image_base + microlift::testkit::DEFAULT_LIFT_BASE;
    });
    assert_eq!(out.ctx.rax & 0xFFFF, 0x2B);
}
