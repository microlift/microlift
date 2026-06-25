/// Tests for conditional jump variants lacking coverage (jl, jge, jg, jle,
/// js, jns, jo, jno, jp, jnp, jrcxz) plus xchg, leave, and lods variants.
///
/// Jump test pattern: `jCC +7; mov rax, 99 (7 bytes); ret`
/// - condition true  → jump over mov → rax = initial value
/// - condition false → fall through → rax = 99
use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::backend::CraneliftBackend;
use microlift::context::RegisterContext;
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

fn run_test(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::new(builder);

    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    ctx.image_base = 0x1000;

    let exit_trampoline = lifter.exit_trampoline_address;
    unsafe {
        *(ctx.rsp as *mut u64) = exit_trampoline;
    }

    lifter.lift(bytes, 0x1000).expect("lifting failed");

    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");

    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    ctx
}

// ---------------------------------------------------------------------------
// JL (signed less: SF != OF)  opcode 7C
// ---------------------------------------------------------------------------

#[test]
fn test_jl() {
    // rax=1 < rcx=2 (signed) → SF=1, OF=0 → jl taken → rax stays 1
    // cmp rax, rcx: 48 3B C1 (CMP r64, r/m64; reg=rax, rm=rcx → ModRM C1)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00, // mov rcx, 2
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx  → rax-rcx=-1, SF=1, OF=0
        0x7C, 0x07,                                // jl +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1); // jump taken

    // rax=5 > rcx=2 → SF=0, OF=0 → jl NOT taken → rax = 99
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx  → SF=0, OF=0
        0x7C, 0x07,                                // jl +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99); // jump not taken
}

// ---------------------------------------------------------------------------
// JGE (signed >=: SF == OF)  opcode 7D
// ---------------------------------------------------------------------------

#[test]
fn test_jge() {
    // rax=5 >= rcx=2 → SF=0, OF=0 → SF==OF → jge taken → rax stays 5
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

    // rax=1 < rcx=5 → SF=1, OF=0 → SF!=OF → jge NOT taken
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

// ---------------------------------------------------------------------------
// JG (signed >: ZF=0 AND SF==OF)  opcode 7F
// ---------------------------------------------------------------------------

#[test]
fn test_jg() {
    // rax=5 > rcx=2 → ZF=0, SF=0, OF=0 → jg taken → rax stays 5
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

    // rax=2 == rcx=2 → ZF=1 → jg NOT taken
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

// ---------------------------------------------------------------------------
// JLE (signed <=: ZF=1 OR SF!=OF)  opcode 7E
// ---------------------------------------------------------------------------

#[test]
fn test_jle() {
    // rax=2 <= rcx=5 → SF=1, OF=0 → SF!=OF → jle taken → rax stays 2
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

    // rax=5 > rcx=2 → ZF=0, SF=0=OF → jle NOT taken
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

// ---------------------------------------------------------------------------
// JS (SF=1)  opcode 78
// ---------------------------------------------------------------------------

#[test]
fn test_js() {
    // sub rax, 1 with rax=0 → rax=-1, SF=1 → js taken → rax stays -1
    // sub rax, 1: 48 83 E8 01 (SUB r/m64, imm8; /5 + rax → E8)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xE8, 0x01, // sub rax, 1  (rax=0 → rax=-1, SF=1, CF=1)
        0x78, 0x07,             // js +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, u64::MAX); // -1 as u64; jump taken

    // mov rax, 5; sub rax, 1 → rax=4, SF=0 → js NOT taken
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x83, 0xE8, 0x01,                   // sub rax, 1  → rax=4, SF=0
        0x78, 0x07,                               // js +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}

// ---------------------------------------------------------------------------
// JNS (SF=0)  opcode 79
// ---------------------------------------------------------------------------

#[test]
fn test_jns() {
    // cmp rax=5, rcx=3 → 5-3=2, SF=0 → jns taken → rax stays 5
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00,
        0x48, 0x3B, 0xC1,                          // cmp rax, rcx → SF=0
        0x79, 0x07,                                // jns +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 5);

    // sub rax, 1 with rax=0 → SF=1 → jns NOT taken
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x83, 0xE8, 0x01,                   // sub rax, 1 → SF=1
        0x79, 0x07,                               // jns +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}

// ---------------------------------------------------------------------------
// JO (OF=1)  opcode 70
// ---------------------------------------------------------------------------

#[test]
fn test_jo() {
    // Produce signed overflow: 0x7FFFFFFFFFFFFFFF + 1 → OF=1
    // xor eax,eax (31 C0) → rax=0
    // dec rax      (48 FF C8) → rax=-1=0xFFFF...
    // shr rax, 1   (48 D1 E8) → rax=0x7FFF...
    // add rax, 1   (48 83 C0 01) → rax=0x8000..., OF=1
    #[rustfmt::skip]
    let bytes = [
        0x31, 0xC0,             // xor eax, eax
        0x48, 0xFF, 0xC8,       // dec rax
        0x48, 0xD1, 0xE8,       // shr rax, 1
        0x48, 0x83, 0xC0, 0x01, // add rax, 1  → OF=1
        0x70, 0x07,             // jo +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x8000000000000000u64); // jump taken, rax = i64::MIN

    // No overflow: add rax=5, 1 → OF=0 → jo NOT taken
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1 → OF=0
        0x70, 0x07,                               // jo +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 99);
}

// ---------------------------------------------------------------------------
// JNO (OF=0)  opcode 71
// ---------------------------------------------------------------------------

#[test]
fn test_jno() {
    // add rax=5, 1 → OF=0 → jno taken → rax=6
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1 → rax=6, OF=0
        0x71, 0x07,                               // jno +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6); // jump taken

    // Overflow → OF=1 → jno NOT taken
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

// ---------------------------------------------------------------------------
// JP (PF=1)  opcode 7A  - parity not computed, pre-seed in context
// ---------------------------------------------------------------------------

#[test]
fn test_jp() {
    // PF=1 pre-seeded → jp taken → rax stays 0
    let mut ctx = RegisterContext::default();
    ctx.pf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x7A, 0x07,                                // jp +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0); // jump taken

    // PF=0 → jp NOT taken → rax = 99
    #[rustfmt::skip]
    let bytes = [
        0x7A, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default()); // pf=0
    assert_eq!(ctx.rax, 99);
}

// ---------------------------------------------------------------------------
// JNP (PF=0)  opcode 7B
// ---------------------------------------------------------------------------

#[test]
fn test_jnp() {
    // PF=0 → jnp taken → rax stays 0
    #[rustfmt::skip]
    let bytes = [
        0x7B, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);

    // PF=1 → jnp NOT taken → rax = 99
    let mut ctx = RegisterContext::default();
    ctx.pf = 1;
    #[rustfmt::skip]
    let bytes = [
        0x7B, 0x07,
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00,
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 99);
}

// ---------------------------------------------------------------------------
// JRCXZ (branch if RCX == 0)  opcode E3
// ---------------------------------------------------------------------------

#[test]
fn test_jrcxz() {
    // rcx=0 (default) → jrcxz taken → rax stays 0
    #[rustfmt::skip]
    let bytes = [
        0xE3, 0x07,                                // jrcxz +7
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0); // jump taken

    // rcx=1 → jrcxz NOT taken → rax = 99
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

// ---------------------------------------------------------------------------
// XCHG
// ---------------------------------------------------------------------------

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
    //   ModRM: mod=01, reg=rax(000), rm=rsp→SIB, disp8=-16=0xF0
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

// ---------------------------------------------------------------------------
// LEAVE
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LODSW (load word from [RSI] into AX, RSI += 2)  opcode 66 AD
// ---------------------------------------------------------------------------

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
        0x66, 0xAD, // lodsw  → AX = [rsi], rsi += 2
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax & 0xFFFF, 0x1234);
    assert_eq!(ctx.rsi, mem_ptr + 2);
}

// ---------------------------------------------------------------------------
// LODSD (load dword from [RSI] into EAX, RSI += 4)  opcode AD
// ---------------------------------------------------------------------------

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
        0xAD, // lodsd  → EAX = [rsi] (zero-extended to RAX), rsi += 4
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0x12345678u64); // zero-extended
    assert_eq!(ctx.rsi, mem_ptr + 4);
}

// ---------------------------------------------------------------------------
// LODSQ (load qword from [RSI] into RAX, RSI += 8)  opcode 48 AD
// ---------------------------------------------------------------------------

#[test]
fn test_lodsq() {
    let val: u64 = 0xDEADBEEFCAFEBABE;
    let mem_ptr = &val as *const u64 as u64;

    let mut ctx = RegisterContext::default();
    ctx.rsi = mem_ptr;

    #[rustfmt::skip]
    let bytes = [
        0x48, 0xAD, // lodsq  → RAX = [rsi], rsi += 8
        0xC3,
    ];
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.rax, 0xDEADBEEFCAFEBABEu64);
    assert_eq!(ctx.rsi, mem_ptr + 8);
}
