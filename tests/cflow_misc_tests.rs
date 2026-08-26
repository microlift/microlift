use microlift::testkit::prelude::*;

// JL (signed less: SF != OF)  opcode 7C

#[test]
fn test_jl() {
    // rax=1 < rcx=2 (signed) -> SF=1, OF=0 -> jl taken -> rax stays 1
    // cmp rax, rcx: 48 3B C1 (CMP r64, r/m64; reg=rax, rm=rcx -> ModRM C1)
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

    // rax=5 > rcx=2 -> SF=0, OF=0 -> jl NOT taken -> rax = 99
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

// JGE (signed >=: SF == OF)  opcode 7D

#[test]
fn test_jge() {
    // rax=5 >= rcx=2 -> SF=0, OF=0 -> SF==OF -> jge taken -> rax stays 5
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

    // rax=1 < rcx=5 -> SF=1, OF=0 -> SF!=OF -> jge NOT taken
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

// JG (signed >: ZF=0 AND SF==OF)  opcode 7F

#[test]
fn test_jg() {
    // rax=5 > rcx=2 -> ZF=0, SF=0, OF=0 -> jg taken -> rax stays 5
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

    // rax=2 == rcx=2 -> ZF=1 -> jg NOT taken
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

// JLE (signed <=: ZF=1 OR SF!=OF)  opcode 7E

#[test]
fn test_jle() {
    // rax=2 <= rcx=5 -> SF=1, OF=0 -> SF!=OF -> jle taken -> rax stays 2
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

    // rax=5 > rcx=2 -> ZF=0, SF=0=OF -> jle NOT taken
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

// JS (SF=1)  opcode 78

#[test]
fn test_js() {
    // sub rax, 1 with rax=0 -> rax=-1, SF=1 -> js taken -> rax stays -1
    // sub rax, 1: 48 83 E8 01 (SUB r/m64, imm8; /5 + rax -> E8)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xE8, 0x01, // sub rax, 1  (rax=0 -> rax=-1, SF=1, CF=1)
        0x78, 0x07,             // js +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX); // -1 as u64; jump taken

    // mov rax, 5; sub rax, 1 -> rax=4, SF=0 -> js NOT taken
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

// JNS (SF=0)  opcode 79

#[test]
fn test_jns() {
    // cmp rax=5, rcx=3 -> 5-3=2, SF=0 -> jns taken -> rax stays 5
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

    // sub rax, 1 with rax=0 -> SF=1 -> jns NOT taken
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

// JO (OF=1)  opcode 70

#[test]
fn test_jo() {
    // Produce signed overflow: 0x7FFFFFFFFFFFFFFF + 1 -> OF=1
    // xor eax,eax (31 C0) -> rax=0
    // dec rax      (48 FF C8) -> rax=-1=0xFFFF...
    // shr rax, 1   (48 D1 E8) -> rax=0x7FFF...
    // add rax, 1   (48 83 C0 01) -> rax=0x8000..., OF=1
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

    // No overflow: add rax=5, 1 -> OF=0 -> jo NOT taken
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

// JNO (OF=0)  opcode 71

#[test]
fn test_jno() {
    // add rax=5, 1 -> OF=0 -> jno taken -> rax=6
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

    // Overflow -> OF=1 -> jno NOT taken
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

// JP (PF=1)  opcode 7A  - parity not computed, pre-seed in context

#[test]
fn test_jp() {
    // PF=1 pre-seeded -> jp taken -> rax stays 0
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

    // PF=0 -> jp NOT taken -> rax = 99
    #[rustfmt::skip]
    let bytes = [
        0x7A, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default()); // pf=0
    assert_eq!(ctx.rax, 99);
}

// JNP (PF=0)  opcode 7B

#[test]
fn test_jnp() {
    // PF=0 -> jnp taken -> rax stays 0
    #[rustfmt::skip]
    let bytes = [
        0x7B, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);

    // PF=1 -> jnp NOT taken -> rax = 99
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

// JRCXZ (branch if RCX == 0)  opcode E3

#[test]
fn test_jrcxz() {
    // rcx=0 (default) -> jrcxz taken -> rax stays 0
    #[rustfmt::skip]
    let bytes = [
        0xE3, 0x07,                                // jrcxz +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0); // jump taken

    // rcx=1 -> jrcxz NOT taken -> rax = 99
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

// XCHG

#[test]
fn test_xchg() {
    // xchg rax, rcx: swap the two registers
    // Use short encoding: REX.W + 91 = xchg rax, rcx
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
    // xchg [mem], rax where mem holds a known value
    // Store 0x1234 at [rsp-16], load it into rax via xchg
    // xchg [rsp-16], rax: 48 87 44 24 F0
    //   ModRM: mod=01, reg=rax(000), rm=rsp->SIB, disp8=-16=0xF0
    //   Actually for [rsp+disp8] we need SIB: ModRM=01_000_100, SIB=ss=00,idx=100(none),base=rsp(100) = 0x24
    //   Full: 48 87 44 24 F0
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x5678;

    // Use a simpler register form: xchg rax, rdx (48 92 = xchg rax, rdx)
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

// LEAVE

#[test]
fn test_leave() {
    // Standard function prologue/epilogue:
    //   push rbp; mov rbp, rsp; sub rsp, 16; mov rax, 42; leave; ret
    // After leave: rsp restored to post-push value, rbp restored to 0 (default).
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
    // rbp should be restored to 0 (the original rbp pushed onto the stack)
    assert_eq!(ctx.rbp, 0);
}

// LODSW (load word from [RSI] into AX, RSI += 2)  opcode 66 AD

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

// LODSD (load dword from [RSI] into EAX, RSI += 4)  opcode AD

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

// LODSQ (load qword from [RSI] into RAX, RSI += 8)  opcode 48 AD

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

/// A direct `CALL` to a callee inside the lifted region, whose `RET` must land
/// back on the instruction after the call.
///
/// The return address is only reachable if it was marked a block leader during
/// discovery: `RET` dispatches through a switch over known blocks, so a missing
/// leader turns the return into an `UnknownJump` VM exit instead. Regression
/// test for exactly that (guest-to-guest calls used to only register the
/// *callee* as a leader).
#[test]
fn test_in_region_call_returns_to_resume_point() {
    #[rustfmt::skip]
    let bytes = [
        // 0x1000: call +5 -> 0x100A
        0xE8, 0x05, 0x00, 0x00, 0x00,
        // 0x1005: add rax, 1   (the resume point)
        0x48, 0x83, 0xC0, 0x01,
        // 0x1009: ret          (back to the exit trampoline)
        0xC3,
        // 0x100A: mov rax, 41  (the callee)
        0x48, 0xC7, 0xC0, 0x29, 0x00, 0x00, 0x00,
        // 0x1011: ret          (back to 0x1005)
        0xC3,
    ];

    let out = LiftTest::new(&bytes).run();
    assert_eq!(out.exit, ExitReason::Return, "call/ret must halt cleanly");
    assert_eq!(
        out.ctx.rax, 42,
        "callee ran and returned to the resume point"
    );
    // Both frames were popped: the guest's stack is exactly balanced.
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}

/// Two levels deep, with the inner callee placed before the outer one, so the
/// resume points are not in address order.
#[test]
fn test_nested_in_region_calls() {
    #[rustfmt::skip]
    let bytes = [
        // 0x1000: call +10 -> 0x100F  (outer)
        0xE8, 0x0A, 0x00, 0x00, 0x00,
        // 0x1005: add rax, 100
        0x48, 0x83, 0xC0, 0x64,
        // 0x1009: ret
        0xC3,
        // 0x100A: inner: add rax, 3 ; ret
        0x48, 0x83, 0xC0, 0x03,
        0xC3,
        // 0x100F: outer: add rax, 10 ; call -14 -> 0x100A ; add rax, 1 ; ret
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

// RETF / RETFQ (far return)
//
// A far return pops the return offset and then the code selector, both in
// operand-size-wide slots (2 / 4 / 8 bytes for RETFW / RETFD / RETFQ). Microlift
// models a flat address space, so the selector is popped-and-discarded; only the
// stack adjustment and the branch to the popped offset are observable.
//
// These are `.no_unicorn()`: Unicorn emulates the real segmentation machinery
// and faults on the synthetic CS selectors these tests push, which is exactly
// the part Microlift deliberately does not model.

/// `RETFQ` (REX.W) pops a full 64-bit offset, so the harness's exit trampoline
/// at `[rsp]` is recognised and the region returns cleanly. RSP advances by 16:
/// 8 for the offset plus 8 for the discarded selector.
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

/// `RETF` with no REX.W uses the 64-bit-mode default 32-bit operand size, so it
/// pops a *4-byte* offset (zero-extended) and a 4-byte selector. The trampoline
/// (`0xFFFF_FFFF_FFFF_FFFF`) therefore truncates to `0xFFFF_FFFF`, which is
/// neither the trampoline nor an in-region block: an `UnknownJump` exit.
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

/// A 32-bit far return whose popped offset lands on a block inside the region
/// resumes there, exactly like a near `RET`.
#[test]
fn test_retfd_in_region_target() {
    #[rustfmt::skip]
    let bytes = [
        // 0x1000: sub rsp, 8                (room for the far-return frame)
        0x48, 0x83, 0xEC, 0x08,
        // 0x1004: mov dword [rsp], 0x1016   (return offset)
        0xC7, 0x04, 0x24, 0x16, 0x10, 0x00, 0x00,
        // 0x100B: mov dword [rsp+4], 0x33   (CS selector - discarded)
        0xC7, 0x44, 0x24, 0x04, 0x33, 0x00, 0x00, 0x00,
        // 0x1013: retf
        0xCB,
        // 0x1014: jmp 0x1016 - unreachable, but it makes 0x1016 a block leader
        //         so the far return's dispatch switch has an entry for it.
        0xEB, 0x00,
        // 0x1016: mov rax, 42
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00,
        // 0x101D: ret
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

/// `RETF imm16` additionally releases `imm16` bytes of arguments, on top of the
/// offset and selector slots.
#[test]
fn test_retfd_imm16_releases_args() {
    #[rustfmt::skip]
    let bytes = [
        // 0x1000: sub rsp, 0x18   (8-byte frame + 16 bytes of "arguments")
        0x48, 0x83, 0xEC, 0x18,
        // 0x1004: mov dword [rsp], 0x1018
        0xC7, 0x04, 0x24, 0x18, 0x10, 0x00, 0x00,
        // 0x100B: mov dword [rsp+4], 0x33
        0xC7, 0x44, 0x24, 0x04, 0x33, 0x00, 0x00, 0x00,
        // 0x1013: retf 0x10
        0xCA, 0x10, 0x00,
        // 0x1016: jmp 0x1018 (leader-maker, as above)
        0xEB, 0x00,
        // 0x1018: mov rbx, rsp   (capture RSP right after the far return)
        0x48, 0x89, 0xE3,
        // 0x101B: ret
        0xC3,
    ];

    let out = LiftTest::new(&bytes).no_unicorn().run();
    assert_eq!(out.exit, ExitReason::Return);
    // 0x18 subtracted, then 8 (frame) + 0x10 (args) released: back to entry RSP.
    assert_eq!(
        out.ctx.rbx, out.initial_rsp,
        "RETF imm16 must release the argument bytes too"
    );
    assert_eq!(out.ctx.rsp, out.initial_rsp + 8);
}
