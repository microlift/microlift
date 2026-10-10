use microlift::testkit::prelude::*;

#[test]
fn test_shifts_rotates() {
    // SHL rax, 1 (rax = 1)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xD1, 0xE0, // shl rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 2);
    assert_eq!(ctx.cf, 0);

    // SHR rax, 1 (rax = 2)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00, // mov rax, 2
        0x48, 0xD1, 0xE8, // shr rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 0);

    // ROR rax, 1 (rax = 1) -> 0x8000
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xD1, 0xC8, // ror rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1u64 << 63);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_rcr() {
    // RCR rax, 1 with rax=1, CF=0: LSB goes to CF, old CF to bit 63 -> rax=0, CF=1.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xD1, 0xD8, // rcr rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 1);

    // RCR rax, 1 with rax=0, CF=1: the carry rotates into bit 63, LSB (0) -> CF.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0xD1, 0xD8, // rcr rax, 1
        0xC3, // ret
    ];
    let mut ctx_in = RegisterContext::default();
    ctx_in.cf = 1;
    let ctx = run_test(&bytes, ctx_in);
    assert_eq!(ctx.rax, 1u64 << 63);
    assert_eq!(ctx.cf, 0);

    // RCR rax, CL (CL=4), rax=0xF0, CF=0: 0xF0 >> 4 = 0x0F, CF = bit 3 = 0.
    let bytes = [
        0x48, 0xC7, 0xC0, 0xF0, 0x00, 0x00, 0x00, // mov rax, 0xF0
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0xD3, 0xD8, // rcr rax, cl
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0F);
    assert_eq!(ctx.cf, 0);

    // 8-bit RCR al, 2 (al=1, CF=0) exercises the modulo-(N+1) path -> al=0x80, CF=0.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0xC0, 0xD8, 0x02, // rcr al, 2
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF, 0x80);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_pushfq() {
    // CLC then PUSHFQ: only reserved bit 1 set -> 0x2.
    let bytes = [
        0xF8, // clc
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 0x2,
        "only the always-set reserved bit should be present"
    );

    // STC then PUSHFQ -> CF | reserved = 0x3.
    let bytes = [
        0xF9, // stc
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 0x3,
        "CF (bit 0) and the reserved bit (bit 1) should be set"
    );

    // Zero result sets ZF (bit 6) and PF (bit 2): reserved | ZF | PF.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x2B, 0xC0, // sub rax, rax  (-> 0, ZF=1, PF=1)
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & (1 << 6), 1 << 6, "ZF should be set");
    assert_eq!(ctx.rax & (1 << 2), 1 << 2, "PF should be set");
    assert_eq!(ctx.rax & 1, 0, "CF should be clear");
    assert_eq!(ctx.rax & (1 << 7), 0, "SF should be clear");
    assert_eq!(ctx.rax & (1 << 11), 0, "OF should be clear");
}

#[test]
fn test_popfq() {
    // POPFQ of 0xC3 (CF | reserved | ZF | SF) loads the tracked bits; PUSHFQ reproduces them.
    let bytes = [
        0x48, 0xC7, 0xC0, 0xC3, 0x00, 0x00, 0x00, // mov rax, 0xC3
        0x50, // push rax
        0x9D, // popfq
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 0xC3,
        "POPFQ should load CF/ZF/SF and PUSHFQ reproduce them"
    );

    // POPFQ also clears flags that are 0 in the popped image: CF preset, popped 0x2 -> CF reads 0.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00, // mov rax, 0x2
        0x50, // push rax
        0x9D, // popfq
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let mut ctx_in = RegisterContext::default();
    ctx_in.cf = 1;
    let ctx = run_test(&bytes, ctx_in);
    assert_eq!(
        ctx.rax, 0x2,
        "POPFQ should clear CF when its source bit is 0"
    );
}

#[test]
fn test_lahf() {
    // STC then LAHF: AH = reserved | CF = 0x03; rest of RAX unmoved.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0xF9, // stc
        0x9F, // lahf
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0300, "AH should hold CF and the reserved bit");

    // Zero result: ZF, PF set, CF clear -> AH = 0x46.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x2B, 0xC0, // sub rax, rax
        0x9F, // lahf
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFF00, 0x4600, "ZF, PF and the reserved bit");
    assert_eq!(ctx.rax & 0xFF, 0, "AL is not part of AH");

    // OF is bit 11, outside LAHF's byte: 0x7fff_ffff + 1 gives OF, SF, PF, AF -> AH = 0x96 with no OF.
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0x7F, // mov rax, 0x7fffffff
        0x83, 0xC0, 0x01, // add eax, 1  (-> 0x80000000, OF=1, SF=1)
        0x9F, // lahf
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.of, 1, "the fixture must actually set OF");
    assert_eq!(
        ctx.rax & 0xFF00,
        0x9600,
        "SF, AF, PF and the reserved bit only"
    );
}

#[test]
fn test_sahf() {
    // AH = CF | reserved | ZF: SAHF sets those and clears the rest. AH is loaded with `mov` since `shl` would set OF itself.
    let bytes = [
        0xB4, 0x43, // mov ah, 0x43
        0x9E, // sahf
        0xC3, // ret
    ];
    let mut ctx_in = RegisterContext::default();
    ctx_in.sf = 1;
    ctx_in.set_pf(1 != 0);
    let ctx = run_test(&bytes, ctx_in);
    assert_eq!(ctx.cf, 1);
    assert_eq!(ctx.zf, 1);
    assert_eq!(ctx.sf, 0, "SAHF clears a flag whose bit is 0");
    assert_eq!(u8::from(ctx.pf()), 0, "SAHF clears a flag whose bit is 0");

    // SAHF leaves OF as found (inverse of LAHF, not POPFQ).
    let mut ctx_in = RegisterContext::default();
    ctx_in.of = 1;
    let ctx = run_test(&bytes, ctx_in);
    assert_eq!(ctx.of, 1, "SAHF must not touch OF");
}

#[test]
fn test_lahf_sahf_round_trip() {
    // LAHF then SAHF is the identity on the five flags it carries.
    let bytes = [
        0xF9, // stc
        0x9F, // lahf
        0xF8, // clc
        0x9E, // sahf
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1, "the CF LAHF saved should come back");
}

#[test]
fn test_loopne() {
    // LOOPNE runs while RCX != 0 and ZF == 0; INC keeps ZF=0 so the full count runs.
    let bytes = [
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0xFF, 0xC0, // inc rax   <- loop target
        0xE0, 0xFB, // loopne -5
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(
        ctx.rax, 4,
        "loopne should run the full count when ZF stays 0"
    );
    assert_eq!(ctx.rcx, 0);

    // LOOPNE exits early when ZF becomes 1: DEC RAX from 3 hits zero on iteration 3 with RCX still 2.
    let bytes = [
        0x48, 0xC7, 0xC1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
        0x48, 0xC7, 0xC0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xFF, 0xC8, // dec rax   <- loop target
        0xE0, 0xFB, // loopne -5
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0, "loopne should stop once dec rax sets ZF");
    assert_eq!(
        ctx.rcx, 2,
        "loopne should exit early with RCX still nonzero"
    );
}

#[test]
fn test_loope() {
    // LOOPE runs while RCX != 0 and ZF == 1: DEC RAX from 1 gives ZF=1 once, then ZF=0 stops it.
    let bytes = [
        0x48, 0xC7, 0xC1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00, // mov rax, 2
        0x48, 0xFF, 0xC8, // dec rax   <- loop target
        0xE1, 0xFB, // loope -5
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    // iter1: rax=1 (ZF=0) -> loope: rcx=4, ZF=0 -> stop.
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.rcx, 4, "loope should exit as soon as ZF clears");
}

#[test]
fn test_bit_ops() {
    // BTS rax, 3 (rax = 0)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x0F, 0xBA, 0xE8, 0x03, // bts rax, 3
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
    assert_eq!(ctx.cf, 0);

    // BTR rax, 3 (rax = 8)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xF0, 0x03, // btr rax, 3
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 1);

    // BTC rax, 1 (rax = 0)
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x0F, 0xBA, 0xF8, 0x01, // btc rax, 1
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 2);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_bit_scans() {
    // BSF rax, rcx (rcx = 0x10) -> rax = 4
    let bytes = [
        0x48, 0xC7, 0xC1, 0x10, 0x00, 0x00, 0x00, // mov rcx, 16
        0x48, 0x0F, 0xBC, 0xC1, // bsf rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 4);
    assert_eq!(ctx.zf, 0);

    // BSR rax, rcx (rcx = 0x11) -> rax = 4
    let bytes = [
        0x48, 0xC7, 0xC1, 0x11, 0x00, 0x00, 0x00, // mov rcx, 17
        0x48, 0x0F, 0xBD, 0xC1, // bsr rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 4);
    assert_eq!(ctx.zf, 0);

    // BSF with zero
    let bytes = [
        0x48, 0x31, 0xC9, // xor rcx, rcx
        0x48, 0x0F, 0xBC, 0xC1, // bsf rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.zf, 1);
}

/// TZCNT/LZCNT are defined for a zero source, unlike BSF/BSR.
#[test]
fn test_count_leading_and_trailing_zeros() {
    // TZCNT rax, rcx (rcx = 0x10) -> 4, CF = 0
    let bytes = [
        0x48, 0xC7, 0xC1, 0x10, 0x00, 0x00, 0x00, // mov rcx, 16
        0xF3, 0x48, 0x0F, 0xBC, 0xC1, // tzcnt rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 4);
    assert_eq!((ctx.cf, ctx.zf), (0, 0));

    // TZCNT of zero -> the operand width, with CF set and ZF clear.
    let bytes = [
        0x48, 0x31, 0xC9, // xor rcx, rcx
        0xF3, 0x48, 0x0F, 0xBC, 0xC1, // tzcnt rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 64, "TZCNT(0) is the operand width, not undefined");
    assert_eq!((ctx.cf, ctx.zf), (1, 0));

    // 32-bit TZCNT of zero -> 32, and the upper half of the register zeroed.
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x31, 0xC9, // xor ecx, ecx
        0xF3, 0x0F, 0xBC, 0xC1, // tzcnt eax, ecx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 32);

    // LZCNT counts zeros above the highest set bit: 0x11 -> 64 - 1 - 4 = 59.
    let bytes = [
        0x48, 0xC7, 0xC1, 0x11, 0x00, 0x00, 0x00, // mov rcx, 17
        0xF3, 0x48, 0x0F, 0xBD, 0xC1, // lzcnt rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 59);
    assert_eq!(ctx.cf, 0);

    // LZCNT of zero -> operand width, CF set, ZF clear (ZF follows the result).
    let bytes = [
        0x48, 0x31, 0xC9, // xor rcx, rcx
        0xF3, 0x48, 0x0F, 0xBD, 0xC1, // lzcnt rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 64);
    assert_eq!((ctx.cf, ctx.zf), (1, 0));

    // ZF is set only when the result is zero (every bit already set).
    let bytes = [
        0x48, 0xC7, 0xC1, 0x01, 0x00, 0x00, 0x00, // mov rcx, 1
        0xF3, 0x48, 0x0F, 0xBC, 0xC1, // tzcnt rax, rcx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!((ctx.rax, ctx.zf, ctx.cf), (0, 1, 0));
}

#[test]
fn test_movsx() {
    // MOVSX rax, cl (cl = 0xFF) -> rax = 0xFFFFFFFFFFFFFFFF
    let bytes = [
        0x48, 0xC7, 0xC1, 0xFF, 0x00, 0x00, 0x00, // mov rcx, 0xFF
        0x48, 0x0F, 0xBE, 0xC1, // movsx rax, cl
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFF);

    // MOVSXD rax, ecx (ecx = 0x80000000) -> rax = 0xFFFFFFFF80000000
    let bytes = [
        0x48, 0xC7, 0xC1, 0x00, 0x00, 0x00, 0x80, // mov rcx, 0x80000000
        0x48, 0x63, 0xC1, // movsxd rax, ecx
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFF80000000);
}

/// MOVSXD's narrow forms do not extend: they are ordinary same-width moves. Mono's JIT emits the 32-bit form.
#[test]
fn test_movsxd_narrow_forms_do_not_extend() {
    // MOVSXD ecx, eax without REX.W: 32-bit move zeroing RCX's top half.
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, ...
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF, // mov rcx, -1
        0x63, 0xC8, // movsxd ecx, eax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rcx, 0x5566_7788);

    // MOVSXD from memory, the form Mono generates.
    let mem = [0x8899AABB_u32, 0];
    let mut ctx = RegisterContext::default();
    ctx.rdi = mem.as_ptr() as u64 + 0x30;
    ctx.rcx = 0xFFFF_FFFF_FFFF_FFFF;
    // movsxd ecx, dword [rdi-0x30] ; ret
    let bytes = [0x63, 0x4F, 0xD0, 0xC3];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(
        ctx.rcx, 0x8899AABB,
        "no sign extension, and the top half zeroed"
    );

    // MOVSXD cx, ax under 0x66: 16-bit move, upper bits preserved.
    let bytes = [
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, ...
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF, // mov rcx, -1
        0x66, 0x63, 0xC8, // movsxd cx, ax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rcx, 0xFFFF_FFFF_FFFF_7788);
}

#[test]
fn test_string_ops() {
    let mut ctx = RegisterContext::default();

    let mut mem = [0u8; 256];
    let mem_ptr = mem.as_mut_ptr() as u64;

    // MOVSB: copy from [rsi] to [rdi]
    mem[0] = 0xCC;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 8;

    let bytes = [
        0xA4, // movsb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[8], 0xCC);
    assert_eq!(ctx_out.rsi, mem_ptr + 1);
    assert_eq!(ctx_out.rdi, mem_ptr + 9);

    // STOSB: store AL to [rdi]
    ctx.rax = 0x42;
    ctx.rdi = mem_ptr + 16;
    let bytes = [
        0xAA, // stosb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x42);
    assert_eq!(ctx_out.rdi, mem_ptr + 17);

    // LODSB: load from [rsi] to AL
    mem[32] = 0x77;
    ctx.rsi = mem_ptr + 32;
    let bytes = [
        0xAC, // lodsb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(ctx_out.rax & 0xFF, 0x77);
    assert_eq!(ctx_out.rsi, mem_ptr + 33);
    // Buffer is read back through `mem` after the lifted code used `mem_ptr`.
    assert_eq!(mem[32], 0x77, "lodsb must not modify its source");
}

#[test]
fn test_direction_flag() {
    let mut mem = [0u8; 256];
    let mem_ptr = mem.as_mut_ptr() as u64;

    // STD: STOSB at [rdi] then rdi -= 1.
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x42;
    ctx.rdi = mem_ptr + 16;
    let bytes = [
        0xFD, // std
        0xAA, // stosb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x42, "stosb should still write [rdi]");
    assert_eq!(ctx_out.rdi, mem_ptr + 15, "DF=1 should decrement rdi");

    // CLD restores incrementing even if DF was set on entry.
    let mut ctx = RegisterContext::default();
    ctx.df = 1; // start with DF set
    ctx.rax = 0x99;
    ctx.rdi = mem_ptr + 16;
    let bytes = [
        0xFC, // cld
        0xAA, // stosb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x99);
    assert_eq!(ctx_out.rdi, mem_ptr + 17, "DF=0 should increment rdi");

    // MOVSB with DF=1 set via the incoming context decrements both pointers.
    let mut ctx = RegisterContext::default();
    ctx.df = 1;
    mem[64] = 0xAB;
    ctx.rsi = mem_ptr + 64;
    ctx.rdi = mem_ptr + 128;
    let bytes = [
        0xA4, // movsb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[128], 0xAB);
    assert_eq!(ctx_out.rsi, mem_ptr + 63, "DF=1 should decrement rsi");
    assert_eq!(ctx_out.rdi, mem_ptr + 127, "DF=1 should decrement rdi");
}

#[test]
fn test_direction_flag_in_rflags() {
    // STD sets DF: PUSHFQ reports bit 10 plus reserved -> 0x402.
    let bytes = [
        0xFD, // std
        0x9C, // pushfq
        0x58, // pop rax
        0xC3, // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x402, "PUSHFQ should report DF at bit 10");

    // POPFQ loads DF from the popped image (0x400); STOSB then decrements rdi.
    let mut mem = [0u8; 64];
    let mem_ptr = mem.as_mut_ptr() as u64;
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x400; // value to load into RFLAGS (DF set)
    ctx.rdi = mem_ptr + 8;
    let bytes = [
        0x50, // push rax (RFLAGS image)
        0x9D, // popfq
        0xB0, 0x7E, // mov al, 0x7E
        0xAA, // stosb
        0xC3, // ret
    ];
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[8], 0x7E);
    assert_eq!(
        ctx_out.rdi,
        mem_ptr + 7,
        "POPFQ should have set DF -> rdi decremented"
    );
    assert_eq!(ctx_out.df, 1, "DF should be synced back to the context");
}

#[test]
fn test_string_ops_sizes() {
    let mut mem = [0u8; 256];
    let mem_ptr = mem.as_mut_ptr() as u64;
    let mut ctx = RegisterContext::default();

    // MOVSW
    mem[0] = 0x11;
    mem[1] = 0x22;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 16;
    let bytes = [0x66, 0xA5, 0xC3]; // movsw, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[16], 0x11);
    assert_eq!(mem[17], 0x22);
    assert_eq!(ctx_out.rsi, mem_ptr + 2);

    // MOVSD (32-bit)
    mem[0] = 0x33;
    mem[1] = 0x44;
    mem[2] = 0x55;
    mem[3] = 0x66;
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 32;
    let bytes = [0xA5, 0xC3]; // movsd (32-bit in 64-bit mode?), ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[32], 0x33);
    assert_eq!(mem[35], 0x66);
    assert_eq!(ctx_out.rsi, mem_ptr + 4);

    // MOVSQ
    ctx.rsi = mem_ptr;
    ctx.rdi = mem_ptr + 64;
    unsafe {
        *(mem_ptr as *mut u64) = 0xDEADBEEFCAFEBABE;
    }
    let bytes = [0x48, 0xA5, 0xC3]; // movsq, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(
        unsafe { *((mem_ptr + 64) as *const u64) },
        0xDEADBEEFCAFEBABE
    );
    assert_eq!(ctx_out.rsi, mem_ptr + 8);

    // STOSW
    ctx.rax = 0xCCAA;
    ctx.rdi = mem_ptr + 80;
    let bytes = [0x66, 0xAB, 0xC3]; // stosw, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[80], 0xAA);
    assert_eq!(mem[81], 0xCC);
    assert_eq!(ctx_out.rdi, mem_ptr + 82);

    // STOSD
    ctx.rax = 0x11223344;
    ctx.rdi = mem_ptr + 96;
    let bytes = [0xAB, 0xC3]; // stosd, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(mem[96], 0x44);
    assert_eq!(mem[99], 0x11);
    assert_eq!(ctx_out.rdi, mem_ptr + 100);

    // STOSQ
    ctx.rax = 0x1122334455667788;
    ctx.rdi = mem_ptr + 128;
    let bytes = [0x48, 0xAB, 0xC3]; // stosq, ret
    let ctx_out = run_test(&bytes, ctx);
    assert_eq!(
        unsafe { *((mem_ptr + 128) as *const u64) },
        0x1122334455667788
    );
    assert_eq!(ctx_out.rdi, mem_ptr + 136);
}
