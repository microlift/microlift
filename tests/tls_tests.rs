//! FS/GS segment-base tests: prefixed accesses add the base; `rd/wr{fs,gs}base` move it to/from a GPR.
//! Bases are pre-loaded with a local buffer address; no multithreading needed.

use microlift::testkit::prelude::*;

#[test]
fn test_fs_segment_read_via_reg() {
    // mov rax, qword ptr fs:[rcx]  ; 64 48 8B 01
    // ret                           ; C3
    let bytes = [0x64, 0x48, 0x8B, 0x01, 0xC3];

    let tls_data: [u64; 2] = [0xDEAD_BEEF_1234_5678, 0xCAFE_BABE_0000_0001];
    let mut ctx = RegisterContext::default();
    ctx.fs_base = tls_data.as_ptr() as u64;
    ctx.rcx = 0; // offset 0 -> first element

    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0xDEAD_BEEF_1234_5678, "fs:[rcx+0] read mismatch");
}

#[test]
fn test_fs_segment_read_with_offset() {
    // mov rax, qword ptr fs:[rcx]  ; 64 48 8B 01 (rcx = 8: second element)
    // ret                           ; C3
    let bytes = [0x64, 0x48, 0x8B, 0x01, 0xC3];

    let tls_data: [u64; 2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let mut ctx = RegisterContext::default();
    ctx.fs_base = tls_data.as_ptr() as u64;
    ctx.rcx = 8; // offset 8 -> second element

    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x2222_2222_2222_2222, "fs:[rcx+8] read mismatch");
}

#[test]
fn test_gs_segment_read_via_reg() {
    // mov rax, qword ptr gs:[rcx]  ; 65 48 8B 01
    // ret                           ; C3
    let bytes = [0x65, 0x48, 0x8B, 0x01, 0xC3];

    let tls_data: [u64; 2] = [0xAAAA_BBBB_CCCC_DDDD, 0xEEEE_FFFF_0000_1111];
    let mut ctx = RegisterContext::default();
    ctx.gs_base = tls_data.as_ptr() as u64;
    ctx.rcx = 8; // second element

    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0xEEEE_FFFF_0000_1111, "gs:[rcx+8] read mismatch");
}

#[test]
fn test_fs_segment_write_via_reg() {
    // mov qword ptr fs:[rcx], rax  ; 64 48 89 01
    // ret                           ; C3
    let bytes = [0x64, 0x48, 0x89, 0x01, 0xC3];

    let mut tls_data: [u64; 2] = [0, 0];
    let mut ctx = RegisterContext::default();
    ctx.fs_base = tls_data.as_mut_ptr() as u64;
    ctx.rcx = 0;
    ctx.rax = 0x1234_5678_9ABC_DEF0;

    run_test(&bytes, ctx);
    // The JIT wrote directly into our buffer.
    assert_eq!(
        tls_data[0], 0x1234_5678_9ABC_DEF0,
        "fs:[rcx] write mismatch"
    );
}

#[test]
fn test_fs_segment_write_with_offset() {
    // mov qword ptr fs:[rcx], rax  ; 64 48 89 01
    // ret                           ; C3
    let bytes = [0x64, 0x48, 0x89, 0x01, 0xC3];

    let mut tls_data: [u64; 2] = [0xDEAD, 0xDEAD];
    let mut ctx = RegisterContext::default();
    ctx.fs_base = tls_data.as_mut_ptr() as u64;
    ctx.rcx = 8; // second slot
    ctx.rax = 0x0BAD_C0DE_CAFE_BABE;

    run_test(&bytes, ctx);
    assert_eq!(tls_data[0], 0xDEAD, "first slot must be untouched");
    assert_eq!(
        tls_data[1], 0x0BAD_C0DE_CAFE_BABE,
        "fs:[rcx+8] write mismatch"
    );
}

#[test]
fn test_gs_segment_write_via_reg() {
    // mov qword ptr gs:[rcx], rax  ; 65 48 89 01
    // ret                           ; C3
    let bytes = [0x65, 0x48, 0x89, 0x01, 0xC3];

    let mut tls_data: [u64; 1] = [0];
    let mut ctx = RegisterContext::default();
    ctx.gs_base = tls_data.as_mut_ptr() as u64;
    ctx.rcx = 0;
    ctx.rax = 0xFEED_FACE_DEAD_BEEF;

    run_test(&bytes, ctx);
    assert_eq!(
        tls_data[0], 0xFEED_FACE_DEAD_BEEF,
        "gs:[rcx] write mismatch"
    );
}

#[test]
fn test_rdfsbase() {
    // rdfsbase rax  ; F3 48 0F AE C0
    // ret           ; C3
    let bytes = [0xF3, 0x48, 0x0F, 0xAE, 0xC0, 0xC3];

    let mut ctx = RegisterContext::default();
    ctx.fs_base = 0x7FFF_0000_DEAD_CAFE;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rax, 0x7FFF_0000_DEAD_CAFE,
        "rdfsbase must copy fs_base -> rax"
    );
}

#[test]
fn test_wrfsbase() {
    // wrfsbase rax  ; F3 48 0F AE D0
    // ret           ; C3
    let bytes = [0xF3, 0x48, 0x0F, 0xAE, 0xD0, 0xC3];

    let mut ctx = RegisterContext::default();
    ctx.rax = 0x1234_5678_9ABC_DEF0;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.fs_base, 0x1234_5678_9ABC_DEF0,
        "wrfsbase must copy rax -> fs_base"
    );
}

#[test]
fn test_wrfsbase_then_read() {
    // wrfsbase rax        ; F3 48 0F AE D0
    // mov rbx, [fs:rcx]  ; 64 48 8B 19  (read via the new base)
    // ret                 ; C3
    let bytes = [
        0xF3, 0x48, 0x0F, 0xAE, 0xD0, // wrfsbase rax
        0x64, 0x48, 0x8B, 0x19, // mov rbx, [fs:rcx]
        0xC3, // ret
    ];

    let tls_data: [u64; 1] = [0xABCD_EF01_2345_6789];
    let mut ctx = RegisterContext::default();
    ctx.rax = tls_data.as_ptr() as u64; // new fs_base
    ctx.rcx = 0;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rbx, 0xABCD_EF01_2345_6789,
        "fs read after wrfsbase mismatch"
    );
    assert_eq!(
        ctx.fs_base,
        tls_data.as_ptr() as u64,
        "fs_base not persisted"
    );
}

#[test]
fn test_rdgsbase() {
    // rdgsbase rax  ; F3 48 0F AE C8
    // ret           ; C3
    let bytes = [0xF3, 0x48, 0x0F, 0xAE, 0xC8, 0xC3];

    let mut ctx = RegisterContext::default();
    ctx.gs_base = 0xBEEF_CAFE_1234_5678;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rax, 0xBEEF_CAFE_1234_5678,
        "rdgsbase must copy gs_base -> rax"
    );
}

#[test]
fn test_wrgsbase() {
    // wrgsbase rax  ; F3 48 0F AE D8
    // ret           ; C3
    let bytes = [0xF3, 0x48, 0x0F, 0xAE, 0xD8, 0xC3];

    let mut ctx = RegisterContext::default();
    ctx.rax = 0xDEAD_BEEF_0000_CAFE;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.gs_base, 0xDEAD_BEEF_0000_CAFE,
        "wrgsbase must copy rax -> gs_base"
    );
}

#[test]
fn test_fs_gs_independent() {
    // rdfsbase rax   ; F3 48 0F AE C0
    // rdgsbase rcx   ; F3 48 0F AE C9  (C9 = mod 11, reg 001, rm 001)
    // ret            ; C3
    let bytes = [
        0xF3, 0x48, 0x0F, 0xAE, 0xC0, // rdfsbase rax
        0xF3, 0x48, 0x0F, 0xAE, 0xC9, // rdgsbase rcx
        0xC3, // ret
    ];

    let mut ctx = RegisterContext::default();
    ctx.fs_base = 0x1111_1111_1111_1111;
    ctx.gs_base = 0x2222_2222_2222_2222;

    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x1111_1111_1111_1111, "rax must hold fs_base");
    assert_eq!(ctx.rcx, 0x2222_2222_2222_2222, "rcx must hold gs_base");
}
