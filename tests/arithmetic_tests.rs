/// Tests for arithmetic instructions lacking coverage:
/// neg, imul (3 forms), mul, idiv, div,
/// cbw/cwde/cdqe/cwd/cdq/cqo, sar, rol, shld/shrd, bt.
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
// NEG
// ---------------------------------------------------------------------------

#[test]
fn test_neg() {
    // neg rax when rax=5 -> rax=-5, CF=1 (src!=0), ZF=0, SF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xF7, 0xD8,                          // neg rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-5i64) as u64);
    assert_eq!(ctx.sf, 1);
    assert_eq!(ctx.zf, 0);
    assert_eq!(ctx.cf, 1); // CF=1 when src != 0

    // neg rax when rax=0 -> rax=0, CF=0, ZF=1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0x31, 0xC0, // xor rax, rax -> rax=0
        0x48, 0xF7, 0xD8, // neg rax
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0);
    assert_eq!(ctx.cf, 0); // CF=0 when src == 0
    assert_eq!(ctx.zf, 1);
}

// ---------------------------------------------------------------------------
// IMUL
// ---------------------------------------------------------------------------

#[test]
fn test_imul_two_operand() {
    // imul rax, rcx  (2-operand: dst = dst * src)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0x0F, 0xAF, 0xC1,                    // imul rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    // OF=CF=0 since result fits (42 == sign-extend of 42)
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_imul_two_operand_negative() {
    // imul rax, rcx with rax=-3, rcx=4 -> rax=-12
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFD, 0xFF, 0xFF, 0xFF, // mov rax, -3 (sign-extended)
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0x0F, 0xAF, 0xC1,                    // imul rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-12i64) as u64);
}

#[test]
fn test_imul_three_operand() {
    // imul rax, rcx, 6  (3-operand: dst = src * imm)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0x6B, 0xC1, 0x06,                    // imul rax, rcx, 6
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
}

#[test]
fn test_imul_one_operand() {
    // imul rcx  (1-operand: rdx:rax = rax * rcx, signed)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xE9,                          // imul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42); // low half in rax
    assert_eq!(ctx.rdx, 0);  // high half = 0 (result fits in 64 bits)
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_imul_one_operand_overflow() {
    // i64::MAX * 2 overflows: rdx should be non-zero, OF/CF should be set
    // rax = 0x7FFFFFFFFFFFFFFF; imul rcx (rcx=2) -> rdx:rax = 0x0000...FFFE
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, // mov rax, i64::MAX
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,                   // mov rcx, 2
        0x48, 0xF7, 0xE9,                                            // imul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    // rdx:rax = i64::MAX * 2 = 0x0000000000000000_FFFFFFFFFFFFFFFE
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFEu64);
    assert_eq!(ctx.rdx, 0); // high half is 0 for this particular product
    // Actually i64::MAX * 2 = 2^63 - 2, which fits in 64 unsigned bits so rdx=0
    // but the signed high half: smulhi(i64::MAX, 2) = 0
    // OF/CF should be 1 since the result doesn't equal sign-extend of the low half
    // (low half = 0xFFFFFFFFFFFFFFFE = -2 as signed, but sign-extend of that = 0xFFFF... != 0)
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}

// ---------------------------------------------------------------------------
// MUL
// ---------------------------------------------------------------------------

#[test]
fn test_mul() {
    // mul rcx  (1-operand: rdx:rax = rax * rcx, unsigned)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x06, 0x00, 0x00, 0x00, // mov rax, 6
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xE1,                          // mul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 42);
    assert_eq!(ctx.rdx, 0);  // no overflow
    assert_eq!(ctx.of, 0);
    assert_eq!(ctx.cf, 0);
}

#[test]
fn test_mul_overflow() {
    // u64::MAX * 2 = 2^65 - 2 = rdx=1, rax=0xFFFFFFFFFFFFFFFE
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, u64::MAX
        0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,                   // mov rcx, 2
        0x48, 0xF7, 0xE1,                                            // mul rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFFFFFFFFFEu64);
    assert_eq!(ctx.rdx, 1); // high half = 1
    assert_eq!(ctx.of, 1);
    assert_eq!(ctx.cf, 1);
}

// ---------------------------------------------------------------------------
// IDIV / DIV
// ---------------------------------------------------------------------------

#[test]
fn test_idiv() {
    // idiv rcx: rdx:rax / rcx (signed)
    // 43 / 7 = 6 remainder 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2B, 0x00, 0x00, 0x00, // mov rax, 43
        0x48, 0x31, 0xD2,                          // xor rdx, rdx  (rdx=0, sign-extend)
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF9,                          // idiv rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6); // quotient
    assert_eq!(ctx.rdx, 1); // remainder
}

#[test]
fn test_idiv_negative() {
    // -43 / 7 = -6 remainder -1  (truncation toward zero)
    // mov rax, -43: 48 C7 C0 D5 FF FF FF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xD5, 0xFF, 0xFF, 0xFF, // mov rax, -43
        0x48, 0xC7, 0xC2, 0xFF, 0xFF, 0xFF, 0xFF, // mov rdx, -1  (sign extension of -43)
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF9,                          // idiv rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-6i64) as u64); // quotient = -6
    assert_eq!(ctx.rdx, (-1i64) as u64); // remainder = -1
}

#[test]
fn test_div() {
    // div rcx: rdx:rax / rcx (unsigned)
    // 43 / 7 = 6 remainder 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2B, 0x00, 0x00, 0x00, // mov rax, 43
        0x48, 0x31, 0xD2,                          // xor rdx, rdx
        0x48, 0xC7, 0xC1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xF7, 0xF1,                          // div rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 6);
    assert_eq!(ctx.rdx, 1);
}

// ---------------------------------------------------------------------------
// Sign-extension conversions
// ---------------------------------------------------------------------------

#[test]
fn test_cbw() {
    // CBW: AX = sign_extend(AL)
    // AL = 0xFF (-1 as i8) -> AX = 0xFFFF
    #[rustfmt::skip]
    let bytes = [
        0xB0, 0xFF, // mov al, 0xFF
        0x66, 0x98, // cbw
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0xFFFF);

    // AL = 0x7F (127) -> AX = 0x007F (positive, no sign extension)
    #[rustfmt::skip]
    let bytes = [
        0xB0, 0x7F, // mov al, 0x7F
        0x66, 0x98, // cbw
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x007F);
}

#[test]
fn test_cwde() {
    // CWDE: EAX = sign_extend(AX)
    // AX = 0x8000 (-32768) -> EAX = 0xFFFF8000 -> RAX = 0x00000000FFFF8000
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x98,                   // cwde
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x00000000FFFF8000u64);

    // AX = 0x0080 (positive) -> EAX = 0x00000080 -> RAX = 0x0000000000000080
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x80, 0x00, // mov ax, 0x0080
        0x98,                   // cwde
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000000000080u64);
}

#[test]
fn test_cdqe() {
    // CDQE: RAX = sign_extend(EAX)
    // EAX = 0x80000000 -> RAX = 0xFFFFFFFF80000000
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x48, 0x98,                    // cdqe
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xFFFFFFFF80000000u64);

    // EAX = 0x7FFFFFFF -> RAX = 0x000000007FFFFFFF (positive, no change)
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0xFF, 0xFF, 0xFF, 0x7F, // mov eax, 0x7FFFFFFF
        0x48, 0x98,                    // cdqe
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x000000007FFFFFFFu64);
}

#[test]
fn test_cwd() {
    // CWD: DX:AX = sign_extend(AX)
    // AX = 0x8000 (-32768) -> DX = 0xFFFF
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x00, 0x80, // mov ax, 0x8000
        0x66, 0x99,             // cwd
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax & 0xFFFF, 0x8000); // AX unchanged
    assert_eq!(ctx.rdx & 0xFFFF, 0xFFFF); // DX = -1

    // AX = 0x0001 (positive) -> DX = 0x0000
    #[rustfmt::skip]
    let bytes = [
        0x66, 0xB8, 0x01, 0x00, // mov ax, 0x0001
        0x66, 0x99,             // cwd
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx & 0xFFFF, 0x0000);
}

#[test]
fn test_cdq() {
    // CDQ: EDX:EAX = sign_extend(EAX)
    // EAX = 0x80000000 -> EDX = 0xFFFFFFFF -> RDX = 0x00000000FFFFFFFF
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x99,                          // cdq
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000080000000u64);
    assert_eq!(ctx.rdx, 0x00000000FFFFFFFFu64);

    // EAX = 0x00000001 -> EDX = 0x00000000
    #[rustfmt::skip]
    let bytes = [
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0x99,                          // cdq
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, 0);
}

#[test]
fn test_cqo() {
    // CQO: RDX:RAX = sign_extend(RAX)
    // RAX = 0x8000000000000000 -> RDX = 0xFFFFFFFFFFFFFFFF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 0x8000...
        0x48, 0x99,                                                    // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, u64::MAX); // 0xFFFFFFFFFFFFFFFF

    // RAX = 1 -> RDX = 0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x99,                                // cqo
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rdx, 0);
}

// ---------------------------------------------------------------------------
// SAR (arithmetic right shift)
// ---------------------------------------------------------------------------

#[test]
fn test_sar_positive() {
    // sar rax, 4 when rax=128: 128 >> 4 = 8 (preserves sign=0)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x80, 0x00, 0x00, 0x00, // mov rax, 128
        0x48, 0xC1, 0xF8, 0x04,                    // sar rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
    assert_eq!(ctx.sf, 0);
}

#[test]
fn test_sar_negative() {
    // sar rax, 4 when rax=-128 (0xFFFFFFFFFFFFFF80):
    // -128 >> 4 (arithmetic) = -8 (0xFFFFFFFFFFFFFFF8)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x80, 0xFF, 0xFF, 0xFF, // mov rax, -128 (sign-extended)
        0x48, 0xC1, 0xF8, 0x04,                    // sar rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, (-8i64) as u64);
    assert_eq!(ctx.sf, 1); // result is negative
}

#[test]
fn test_sar_by_cl() {
    // sar rax, cl when rax=64, cl=3: 64 >> 3 = 8
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x40, 0x00, 0x00, 0x00, // mov rax, 64
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0xD3, 0xF8,                          // sar rax, cl
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 8);
}

// ---------------------------------------------------------------------------
// ROL (rotate left)
// ---------------------------------------------------------------------------

#[test]
fn test_rol_msb_rotates_to_lsb() {
    // rol rax, 1 when rax=0x8000000000000000:
    // result = 1, CF = 1 (the MSB that was rotated out)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 0x8000...
        0x48, 0xD1, 0xC0,                                             // rol rax, 1
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
    assert_eq!(ctx.cf, 1);
}

#[test]
fn test_rol_preserves_value_at_full_rotation() {
    // rol rax, 4 when rax=0x0F0F0F0F0F0F0F0F
    // result = 0xF0F0F0F0F0F0F0F0  (nibbles shifted left)
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xB8, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, // mov rax, 0x0F0F...
        0x48, 0xC1, 0xC0, 0x04,                                       // rol rax, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xF0F0F0F0F0F0F0F0u64);
}

// ---------------------------------------------------------------------------
// SHLD / SHRD (double-precision shift)
// ---------------------------------------------------------------------------

#[test]
fn test_shld() {
    // shld rax, rcx, 4: result = (rax << 4) | (rcx >> 60)
    // rax = 0x00000000000000FF, rcx = 0xF000000000000000
    // rax << 4 = 0x0000000000000FF0
    // rcx >> 60 = 0xF
    // result = 0x0000000000000FFF
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xFF, 0x00, 0x00, 0x00, // mov rax, 0xFF
        0x48, 0xB9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, // mov rcx, 0xF000000000000000
        0x48, 0x0F, 0xA4, 0xC8, 0x04,             // shld rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x0000000000000FFFu64);
}

#[test]
fn test_shld_simple() {
    // shld rax, rcx, 4 when rcx=0: result = rax << 4 = 0x10
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x31, 0xC9,                          // xor rcx, rcx
        0x48, 0x0F, 0xA4, 0xC8, 0x04,             // shld rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0x10);
}

#[test]
fn test_shrd() {
    // shrd rax, rcx, 4: result = (rcx << 60) | (rax >> 4)
    // rax = 0x00000000000000F0, rcx = 0x000000000000000F
    // rcx << 60 = 0xF000000000000000
    // rax >> 4  = 0x000000000000000F
    // result = 0xF00000000000000F
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0xF0, 0x00, 0x00, 0x00, // mov rax, 0xF0
        0x48, 0xC7, 0xC1, 0x0F, 0x00, 0x00, 0x00, // mov rcx, 0x0F
        0x48, 0x0F, 0xAC, 0xC8, 0x04,             // shrd rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 0xF00000000000000Fu64);
}

#[test]
fn test_shrd_simple() {
    // shrd rax, rcx, 4 when rcx=0: result = rax >> 4 = 1
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0x31, 0xC9,                          // xor rcx, rcx
        0x48, 0x0F, 0xAC, 0xC8, 0x04,             // shrd rax, rcx, 4
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.rax, 1);
}

// ---------------------------------------------------------------------------
// BT (bit test, read-only)
// ---------------------------------------------------------------------------

#[test]
fn test_bt_immediate() {
    // bt rax, 3 when rax=8 (bit 3 is set): CF=1, rax unchanged
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xE0, 0x03,             // bt rax, 3
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1); // bit 3 was set
    assert_eq!(ctx.rax, 8); // rax unchanged

    // bt rax, 2 when rax=8 (bit 2 is NOT set): CF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // mov rax, 8
        0x48, 0x0F, 0xBA, 0xE0, 0x02,             // bt rax, 2
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0); // bit 2 was clear
    assert_eq!(ctx.rax, 8); // rax unchanged
}

#[test]
fn test_bt_register() {
    // bt rax, rcx when rax=0x10, rcx=4: bit 4 is set -> CF=1
    // bt rax, rcx: 48 0F A3 C8
    // ModRM: mod=11, reg=rcx(001), rm=rax(000) -> 0b11_001_000 = 0xC8
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10 (bit 4 set)
        0x48, 0xC7, 0xC1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0x48, 0x0F, 0xA3, 0xC8,                    // bt rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 1);

    // bt rax, rcx when rax=0x10, rcx=3: bit 3 is NOT set -> CF=0
    #[rustfmt::skip]
    let bytes = [
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0x0F, 0xA3, 0xC8,                    // bt rax, rcx
        0xC3,
    ];
    let ctx = run_test(&bytes, RegisterContext::default());
    assert_eq!(ctx.cf, 0);
}
