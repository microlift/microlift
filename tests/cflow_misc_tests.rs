use microlift::testkit::prelude::*;


#[test]
fn test_jl() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00, // mov rcx, 2
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx  -> rax-rcx=-1, SF=1, OF=0
        0x7C, 0x07,                                // jl +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1); // jump taken

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx  -> SF=0, OF=0
        0x7C, 0x07,                                // jl +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99); // jump not taken
}


#[test]
fn test_jge() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx
        0x7D, 0x07,                                // jge +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 5);

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,
        0x7D, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jg() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,
        0x7F, 0x07,                                // jg +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 5);

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,
        0x7F, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jle() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,
        0x7E, 0x07,                                // jle +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 2);

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,
        0x7E, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_js() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xE8, 0x01, // sub rax, 1  (rax=0 -> rax=-1, SF=1, CF=1)
        0x78, 0x07,             // js +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX); // -1 as u64; jump taken

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x83, 0xE8, 0x01,                   // sub rax, 1  -> rax=4, SF=0
        0x78, 0x07,                               // js +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jns() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx -> SF=0
        0x79, 0x07,                                // jns +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 5);

    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xE8, 0x01,                   // sub rax, 1 -> SF=1
        0x79, 0x07,                               // jns +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jo() {
    // 0x7FFF...FFFF + 1 sets OF.
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,             // xor eax, eax
        0x48, 0xFF, 0xC8,       // dec rax
        0x48, 0xD1, 0xE8,       // shr rax, 1
        0x48, 0x83, 0xC0, 0x01, // add rax, 1  -> OF=1
        0x70, 0x07,             // jo +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x8000000000000000u64); // jump taken, rax = i64::MIN

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1 -> OF=0
        0x70, 0x07,                               // jo +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jno() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1 -> rax=6, OF=0
        0x71, 0x07,                               // jno +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6); // jump taken

    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,
        0x48, 0xFF, 0xC8,
        0x48, 0xD1, 0xE8,
        0x48, 0x83, 0xC0, 0x01, // produce OF=1; rax=0x8000...
        0x71, 0x07,             // jno +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jp() {
    let mut ctx = RegisterContext::default();
    ctx.set_pf(1 != 0);
    #[rustfmt::skip]
    let bytes = [
        0x7A, 0x07,                                // jp +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0); // jump taken

    #[rustfmt::skip]
    let bytes = [
        0x7A, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default()); // pf=0
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jnp() {
    #[rustfmt::skip]
    let bytes = [
        0x7B, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);

    let mut ctx = RegisterContext::default();
    ctx.set_pf(1 != 0);
    #[rustfmt::skip]
    let bytes = [
        0x7B, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_jrcxz() {
    #[rustfmt::skip]
    let bytes = [
        0xE3, 0x07,                                // jrcxz +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0); // jump taken

    let mut ctx = RegisterContext::default();
    ctx.rcx = 1;
    #[rustfmt::skip]
    let bytes = [
        0xE3, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 99);
}


#[test]
fn test_xchg() {
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xAA, 0x00, 0x00, 0x00, // mov rax, 0xAA
        0x48, 0xC7, 0xC1, 0xBB, 0x00, 0x00, 0x00, // mov rcx, 0xBB
        0x48, 0x91,                                // xchg rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xBB);
    assert_eq!(ctx.rcx, 0xAA);
}

#[test]
fn test_xchg_memory() {
    // xchg [rsp-16], rax needs a SIB byte (0x24).
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x5678;

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x11, 0x00, 0x00, 0x00, // mov rax, 0x11
        0x48, 0xC7, 0xC2, 0x22, 0x00, 0x00, 0x00, // mov rdx, 0x22
        0x48, 0x92,                                // xchg rax, rdx
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x22);
    assert_eq!(ctx.rdx, 0x11);
}


#[test]
fn test_leave() {
    // push rbp; mov rbp, rsp; sub rsp, 16; mov rax, 42; leave; ret
    #[rustfmt::skip]
    let bytes = [
        0x55,                                      // push rbp
        0x48, 0x89, 0xE5,                          // mov rbp, rsp
        0x48, 0x83, 0xEC, 0x10,                    // sub rsp, 16
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0xC9,                                      // leave
        0xC3,                                      // ret
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rbp, 0);
}


#[test]
fn test_lodsw() {
    let mut mem = [0u8; 16];
    mem[0] = 0x34;
    mem[1] = 0x12; // little-endian 0x1234
    let mem_ptr = mem.as_ptr() as u64;

    let mut ctx = RegisterContext::default();
    ctx.rsi = mem_ptr;

    #[rustfmt::skip]
    let bytes = [
        0x66, 0xAD, // lodsw  -> AX = [rsi], rsi += 2
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax & 0xFFFF, 0x1234);
    assert_eq!(ctx.rsi, mem_ptr + 2);
}


#[test]
fn test_lodsd() {
    let mut mem = [0u8; 16];
    mem[0] = 0x78;
    mem[1] = 0x56;
    mem[2] = 0x34;
    mem[3] = 0x12; // little-endian 0x12345678
    let mem_ptr = mem.as_ptr() as u64;

    let mut ctx = RegisterContext::default();
    ctx.rsi = mem_ptr;

    #[rustfmt::skip]
    let bytes = [
        0xAD, // lodsd  -> EAX = [rsi] (zero-extended to RAX), rsi += 4
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x12345678u64); // zero-extended
    assert_eq!(ctx.rsi, mem_ptr + 4);
}


#[test]
fn test_lodsq() {
    let val: u64 = 0xDEADBEEFCAFEBABE;
    let mem_ptr = &val as *const u64 as u64;

    let mut ctx = RegisterContext::default();
    ctx.rsi = mem_ptr;

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xAD, // lodsq  -> RAX = [rsi], rsi += 8
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0xDEADBEEFCAFEBABEu64);
    assert_eq!(ctx.rsi, mem_ptr + 8);
}

/// Regression: a guest-to-guest CALL's return address must be a block leader, or `RET` exits `UnknownJump`.
#[test]
fn test_in_region_call_returns_to_resume_point() {
    #[rustfmt::skip]
    let bytes = [
        // call +5 -> 0x100A
        0xE8, 0x05, 0x00, 0x00, 0x00,
        // add rax, 1 (resume point)
        0x48, 0x83, 0xC0, 0x01,
        // ret
        0xC3,
        // mov rax, 41 (callee)
        0x48, 0xC7, 0xC0, 0x29, 0x00, 0x00, 0x00,
        // ret
        0xC3,
    ];

    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.exit, ExitReason::Return, "call/ret must halt cleanly");
    assert_eq!(
        out.ctx.rax, 42,
        "callee ran and returned to the resume point"
    );
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}

/// Inner callee placed before the outer, so resume points are out of address order.
#[test]
fn test_nested_in_region_calls() {
    #[rustfmt::skip]
    let bytes = [
        // call +10 -> 0x100F
        0xE8, 0x0A, 0x00, 0x00, 0x00,
        // add rax, 100
        0x48, 0x83, 0xC0, 0x64,
        // ret
        0xC3,
        // inner: add rax, 3 ; ret
        0x48, 0x83, 0xC0, 0x03,
        0xC3,
        // outer: add rax, 10 ; call -14 ; add rax, 1 ; ret
        0x48, 0x83, 0xC0, 0x0A,
        0xE8, 0xF2, 0xFF, 0xFF, 0xFF,
        0x48, 0x83, 0xC0, 0x01,
        0xC3,
    ];

    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(out.ctx.rax, 114, "10 + 3 + 1 + 100");
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}

// RETF pops offset and selector in operand-size slots; the selector is discarded (flat model).
// `.no_unicorn()`: Unicorn faults on the synthetic CS selectors.

/// `RETFQ` pops a 64-bit offset, so the exit trampoline is recognised; RSP advances by 16.
#[test]
fn test_retfq_to_trampoline() {
    let out = LiftTest::new(&[0x48, 0xCB]).no_unicorn().run(); // retfq
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(
        out.ctx.rsp,
        out.initial_rsp + 16,
        "far return pops offset + selector"
    );
}

/// `RETF` without REX.W pops a 4-byte offset; the truncated trampoline is an `UnknownJump`.
#[test]
fn test_retfd_pops_32bit_offset() {
    let out = LiftTest::new(&[0xCB]).no_unicorn().run(); // retf (retfd)
    assert_eq!(out.exit, ExitReason::UnknownJump);
    assert_eq!(
        out.ctx.rip, 0xFFFF_FFFF,
        "only the low 32 bits are popped, zero-extended"
    );
    assert_eq!(
        out.ctx.rsp,
        out.initial_rsp + 8,
        "4-byte offset + 4-byte CS"
    );
}

/// A 32-bit far return to an in-region block resumes there like a near `RET`.
#[test]
fn test_retfd_in_region_target() {
    #[rustfmt::skip]
    let bytes = [
        // sub rsp, 8
        0x48, 0x83, 0xEC, 0x08,
        // mov dword [rsp], 0x1016
        0xC7, 0x04, 0x24, 0x16, 0x10, 0x00, 0x00,
        // mov dword [rsp+4], 0x33 (CS, discarded)
        0xC7, 0x44, 0x24, 0x04, 0x33, 0x00, 0x00, 0x00,
        // retf
        0xCB,
        // jmp 0x1016: makes it a leader for the dispatch switch
        0xEB, 0x00,
        // mov rax, 42
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00,
        // ret
        0xC3,
    ];

    let out = LiftTest::new(&bytes).no_unicorn().run();
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(
        out.ctx.rax, 42,
        "execution resumed at the far-return target"
    );
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8, "stack is balanced");
}

/// `RETF imm16` also releases `imm16` argument bytes.
#[test]
fn test_retfd_imm16_releases_args() {
    #[rustfmt::skip]
    let bytes = [
        // sub rsp, 0x18
        0x48, 0x83, 0xEC, 0x18,
        // mov dword [rsp], 0x1018
        0xC7, 0x04, 0x24, 0x18, 0x10, 0x00, 0x00,
        // mov dword [rsp+4], 0x33
        0xC7, 0x44, 0x24, 0x04, 0x33, 0x00, 0x00, 0x00,
        // retf 0x10
        0xCA, 0x10, 0x00,
        // jmp 0x1018 (leader)
        0xEB, 0x00,
        // mov rbx, rsp
        0x48, 0x89, 0xE3,
        // ret
        0xC3,
    ];

    let out = LiftTest::new(&bytes).no_unicorn().run();
    assert_eq!(out.exit, ExitReason::Return);
    assert_eq!(
        out.ctx.rbx, out.initial_rsp,
        "RETF imm16 must release the argument bytes too"
    );
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}
