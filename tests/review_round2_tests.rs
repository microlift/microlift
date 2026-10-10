//! Regression tests for the second x86-64 lifter review round (SHLD/SHRD count 0, XADD, BT memory forms, FST specials, shift flags, 0x67, UD2/INT3 RIP, stack-arg imports, BSF/BSR zero source).

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use cranelift_codegen::settings;
use microlift::arch::x86_64::X86_64;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::core::abi::CallAbi;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::function_traits::{DynamicSymbol, FunctionArgumentType};
use microlift::lifter::Lifter;
use microlift::os::get_host_calling_conv;

const LIFT_BASE: u64 = 0x1000;

/// Sentinel return address; a `RET` to it exits with `Return`.
const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Lift + JIT `bytes` at LIFT_BASE and return the mutated context. Override `image_base` to test relocation.
fn run(bytes: &[u8], mut ctx: RegisterContext) -> RegisterContext {
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
    let mut lifter = Lifter::with_config(
        builder,
        LifterConfig {
            exit_trampoline_address: Some(EXIT_TRAMPOLINE),
            ..LifterConfig::default()
        },
    );

    let mut stack = [0u64; 1024];
    let stack_top = stack.as_mut_ptr() as u64 + 8192;
    ctx.rsp = stack_top - 8;
    if ctx.image_base == 0 {
        ctx.image_base = LIFT_BASE;
    }
    unsafe {
        *(ctx.rsp as *mut u64) = EXIT_TRAMPOLINE;
    }

    lifter.lift(bytes, LIFT_BASE).expect("lifting failed");
    let mut backend = CraneliftBackend::new().expect("backend init failed");
    drop(lifter);
    let code_ptr = backend.compile(func).expect("compilation failed");
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    f(&mut ctx);
    let _ = stack;
    ctx
}

// 1. SHLD / SHRD with count 0

#[test]
fn shld_count_zero_is_noop() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x0000_0000_1111_1111;
    ctx.rbx = 0xFFFF_FFFF_FFFF_FFFF;
    // shld eax, ebx, 0 ; ret
    let out = run(&[0x0F, 0xA4, 0xD8, 0x00, 0xC3], ctx);
    assert_eq!(out.rax, 0x1111_1111, "shld dst,src,0 must not modify dst");
}

#[test]
fn shrd_count_zero_is_noop() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x0000_0000_1111_1111;
    ctx.rbx = 0xFFFF_FFFF_FFFF_FFFF;
    // shrd eax, ebx, 0 ; ret
    let out = run(&[0x0F, 0xAC, 0xD8, 0x00, 0xC3], ctx);
    assert_eq!(out.rax, 0x1111_1111, "shrd dst,src,0 must not modify dst");
}

#[test]
fn shld_count_four_shifts_in_src_bits() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x1234_5678;
    ctx.rbx = 0xF000_0000; // top nibble feeds into the low bits
    // shld eax, ebx, 4 ; ret -> 0x2345678F
    let out = run(&[0x0F, 0xA4, 0xD8, 0x04, 0xC3], ctx);
    assert_eq!(out.rax, 0x2345_678F, "shld eax,ebx,4 result");
}

// 2. XADD same-register operand ordering

#[test]
fn xadd_same_register_doubles() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 21;
    // xadd rax, rax ; ret
    let out = run(&[0x48, 0x0F, 0xC1, 0xC0, 0xC3], ctx);
    assert_eq!(out.rax, 42, "xadd rax, rax must double rax");
}

#[test]
fn xadd_distinct_registers_swap_and_add() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 100; // dst
    ctx.rbx = 7; // src
    // xadd rax, rbx ; ret  -> rax = 107, rbx = 100 (old dst)
    let out = run(&[0x48, 0x0F, 0xC1, 0xD8, 0xC3], ctx);
    assert_eq!(out.rax, 107, "xadd dst gets the sum");
    assert_eq!(out.rbx, 100, "xadd src gets the old dst");
}

// 3. BT-family memory bit-string addressing

#[test]
fn bt_mem_bit_index_beyond_operand() {
    let mut buf: [u32; 2] = [0, 1]; // bit 32 of the bit string is set
    let mut ctx = RegisterContext::default();
    ctx.rax = buf.as_mut_ptr() as u64;
    ctx.rbx = 32;
    // bt [rax], ebx ; setc cl ; ret
    let out = run(&[0x0F, 0xA3, 0x18, 0x0F, 0x92, 0xC1, 0xC3], ctx);
    let _ = buf;
    assert_eq!(
        out.rcx & 0xFF,
        1,
        "bt [mem], reg must address beyond the dword"
    );
}

#[test]
fn bts_mem_bit_index_sets_correct_word() {
    let mut buf: [u32; 2] = [0, 0];
    let mut ctx = RegisterContext::default();
    ctx.rax = buf.as_mut_ptr() as u64;
    ctx.rbx = 33; // second dword, bit 1
    // bts [rax], ebx ; ret
    let out = run(&[0x0F, 0xAB, 0x18, 0xC3], ctx);
    let _ = &out;
    assert_eq!(buf[0], 0, "first dword untouched");
    assert_eq!(buf[1], 0b10, "bts must set bit 1 of the second dword");
}

#[test]
fn bt_mem_negative_index_addresses_below_base() {
    // Word[-1] has bit 31 set; index -1 targets it.
    let mut buf: [u32; 2] = [0x8000_0000, 0];
    let mut ctx = RegisterContext::default();
    // rax = buf[1] so a -1 index reaches buf[0] bit 31.
    ctx.rax = unsafe { buf.as_mut_ptr().add(1) } as u64;
    ctx.rbx = 0xFFFF_FFFF_FFFF_FFFF; // -1
    // bt [rax], rbx ; setc cl ; ret  (64-bit index)
    let out = run(&[0x48, 0x0F, 0xA3, 0x18, 0x0F, 0x92, 0xC1, 0xC3], ctx);
    let _ = buf;
    assert_eq!(
        out.rcx & 0xFF,
        1,
        "negative bt index must address below the base"
    );
}

// 4. FST m32/m64 of specials and rounding

fn fld_fstp_roundtrip_f64(value: f64) -> f64 {
    let mut buf: [u64; 2] = [value.to_bits(), 0];
    let mut ctx = RegisterContext::default();
    ctx.rax = buf.as_mut_ptr() as u64;
    // fld qword [rax] ; fstp qword [rax+8] ; ret
    run(&[0xDD, 0x00, 0xDD, 0x58, 0x08, 0xC3], ctx);
    f64::from_bits(buf[1])
}

#[test]
fn fstp_m64_infinity_roundtrip() {
    let r = fld_fstp_roundtrip_f64(f64::INFINITY);
    assert!(r.is_infinite() && r.is_sign_positive(), "got {r}");
    let r = fld_fstp_roundtrip_f64(f64::NEG_INFINITY);
    assert!(r.is_infinite() && r.is_sign_negative(), "got {r}");
}

#[test]
fn fstp_m64_nan_roundtrip() {
    let r = fld_fstp_roundtrip_f64(f64::NAN);
    assert!(r.is_nan(), "fld/fstp of NaN must stay NaN, got {r}");
}

#[test]
fn fstp_m64_finite_roundtrip() {
    for v in [0.0f64, -0.0, 1.0, -2.5, 3.141592653589793, 1e300, 5e-324] {
        let r = fld_fstp_roundtrip_f64(v);
        assert_eq!(
            r.to_bits(),
            v.to_bits(),
            "fld/fstp of {v} must round-trip exactly"
        );
    }
}

#[test]
fn fstp_m32_infinity_roundtrip() {
    // fld qword [rax] ; fstp dword [rax+8] ; ret
    let mut buf: [u64; 2] = [f64::INFINITY.to_bits(), 0];
    let mut ctx = RegisterContext::default();
    ctx.rax = buf.as_mut_ptr() as u64;
    run(&[0xDD, 0x00, 0xD9, 0x58, 0x08, 0xC3], ctx);
    let stored = f32::from_bits(buf[1] as u32);
    assert!(
        stored.is_infinite() && stored.is_sign_positive(),
        "got {stored}"
    );
}

// 5. Shift flags: OF for count==1, PF update

#[test]
fn shl_by_one_defines_of() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x40; // shl al,1 -> 0x80, CF=0, so OF=1
    // shl al, 1 ; seto bl ; ret
    let out = run(&[0xD0, 0xE0, 0x0F, 0x90, 0xC3, 0xC3], ctx);
    assert_eq!(out.rbx & 0xFF, 1, "shl al,1 of 0x40 must set OF");
}

#[test]
fn shl_by_one_clears_of_when_no_sign_change() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0x10; // shl al,1 -> 0x20, MSB stays 0, OF=0
    let out = run(&[0xD0, 0xE0, 0x0F, 0x90, 0xC3, 0xC3], ctx);
    assert_eq!(out.rbx & 0xFF, 0, "shl al,1 of 0x10 must clear OF");
}

#[test]
fn shr_updates_pf() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xF3;
    ctx.set_pf(0 != 0); // stale PF a correct shift overwrites
    // shr al, 4 ; setp bl ; ret  (result 0x0F -> 4 set bits -> PF=1)
    let out = run(&[0xC0, 0xE8, 0x04, 0x0F, 0x9A, 0xC3, 0xC3], ctx);
    assert_eq!(out.rbx & 0xFF, 1, "shr must set PF from the result");
}

#[test]
fn shift_count_zero_preserves_flags() {
    let mut ctx = RegisterContext::default();
    ctx.rax = 0xFF;
    ctx.of = 1;
    ctx.set_pf(1 != 0);
    ctx.cf = 1;
    // shl al, cl with cl=0 -> no flags change
    ctx.rcx = 0;
    // shl al, cl ; seto bl ; ret
    let out = run(&[0xD2, 0xE0, 0x0F, 0x90, 0xC3, 0xC3], ctx);
    assert_eq!(out.rbx & 0xFF, 1, "shl by 0 must preserve OF");
    assert_eq!(out.rax & 0xFF, 0xFF, "shl by 0 must not change the value");
}

// 6. 0x67 32-bit address-size operands must verify

#[test]
fn addr32_operand_produces_verifiable_ir() {
    // 67 8b 01 = mov eax, dword [ecx] ; ret
    let bytes = [0x67, 0x8B, 0x01, 0xC3];
    let func = lift_to_function::<X86_64>(&bytes, LIFT_BASE, LifterConfig::default()).unwrap();
    let mut backend = CraneliftBackend::new().unwrap();
    backend
        .compile(func)
        .expect("32-bit address-size operand must produce verifiable IR");
}

// 7. UD2 / INT3 relocation-correct RIP

#[test]
fn ud2_reports_relocated_rip() {
    // nop ; ud2   (ud2 at region offset 1)
    let bytes = [0x90, 0x0F, 0x0B];
    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x5555_0000; // relocated away from LIFT_BASE
    ctx.rip = ctx.image_base; // enter at region start
    let out = run(&bytes, ctx);
    assert_eq!(out.exit_reason, ExitReason::Ud2 as u32);
    assert_eq!(
        out.rip,
        0x5555_0000 + 1,
        "UD2 RIP must be image_base + region offset, not the lift-time VA"
    );
}

#[test]
fn int3_reports_relocated_rip() {
    // nop ; int3
    let bytes = [0x90, 0xCC];
    let mut ctx = RegisterContext::default();
    ctx.image_base = 0x5555_0000;
    ctx.rip = ctx.image_base;
    let out = run(&bytes, ctx);
    assert_eq!(out.exit_reason, ExitReason::Breakpoint as u32);
    assert_eq!(out.rip, 0x5555_0000 + 1, "INT3 RIP must be relocated");
}

// 8. Fast imports needing stack args are placed on the stack, not miscompiled

#[test]
fn fast_import_with_seven_int_args_compiles() {
    // Seven integer args: the 7th spills past SysV's six registers and must be read from the guest stack.
    let sym = DynamicSymbol {
        name: "seven_args",
        value: 0x1 as *mut core::ffi::c_void,
        arg_types: vec![
            FunctionArgumentType::None, // void return
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
            FunctionArgumentType::Pointer,
        ],
        is_ellipsis_function: false,
        has_execution_side_effects: false,
    };

    const IMPORT_VA: u64 = 0x2000;
    let mut config = LifterConfig::default();
    config.register_fast_import(IMPORT_VA, sym, CallAbi::sysv_amd64());

    // call 0x2000 ; ret   (E8 rel32 to 0x2000 from 0x1005)
    let rel = (IMPORT_VA as i64 - 0x1005) as i32;
    let mut bytes = vec![0xE8];
    bytes.extend_from_slice(&rel.to_le_bytes());
    bytes.push(0xC3);

    // Must lift and compile without a verifier error.
    let func = lift_to_function::<X86_64>(&bytes, LIFT_BASE, config).unwrap();
    let mut backend = CraneliftBackend::new().unwrap();
    backend
        .compile(func)
        .expect("import with a stack-passed argument must compile");
}

// 9. BSF / BSR leave the destination unchanged when src == 0

#[test]
fn bsf_zero_source_preserves_destination() {
    let mut ctx = RegisterContext::default();
    ctx.rbx = 0; // source
    ctx.rax = 0xDEAD_BEEF; // pre-existing destination
    // bsf eax, ebx ; ret
    let out = run(&[0x0F, 0xBC, 0xC3, 0xC3], ctx);
    assert_eq!(
        out.rax, 0xDEAD_BEEF,
        "bsf with src=0 must leave dst unchanged"
    );
}

#[test]
fn bsr_zero_source_preserves_destination() {
    let mut ctx = RegisterContext::default();
    ctx.rbx = 0;
    ctx.rax = 0x1234_5678;
    // bsr eax, ebx ; ret
    let out = run(&[0x0F, 0xBD, 0xC3, 0xC3], ctx);
    assert_eq!(
        out.rax, 0x1234_5678,
        "bsr with src=0 must leave dst unchanged"
    );
}

#[test]
fn bsr_nonzero_source_reports_high_bit() {
    let mut ctx = RegisterContext::default();
    ctx.rbx = 0x0000_0100; // bit 8
    ctx.rax = 0;
    let out = run(&[0x0F, 0xBD, 0xC3, 0xC3], ctx);
    assert_eq!(out.rax, 8, "bsr of 0x100 must report bit 8");
}
