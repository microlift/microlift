//! Outlining of the 80-bit x87 softfloat kernels into shared helper functions.
//!
//! The kernels in `semantics/softfloat.rs` are large (the divide and square
//! root each contain a 128-bit shift/subtract loop). Emitting them once per
//! region and `call`ing them, rather than inlining a copy at every x87 site, is
//! what keeps a float-heavy region's code size sane. These tests pin what that
//! has to satisfy: the same answers as inlining, a real code-size win, one
//! helper per kernel however many sites use it, helper bodies that reach the
//! object file, and a namespace-0 index space that stays consistent when a
//! region has both imports and helpers.

use microlift::arch::x86_64::X86_64;
use microlift::arch::x86_64::semantics::softfloat::F80Kernel;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;
use object::{Object, ObjectSymbol};

/// `fldt [rdi]; fldt [rsi]; f<op> st1, st0; fstpt [rdx]` - one x87 arithmetic
/// op on two 80-bit memory operands, result stored back to memory.
///
/// Registers are seeded with the addresses at run time, so the guest program
/// needs no relocations of its own.
fn f80_program(op: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&[0xDB, 0x2F]); // fld tbyte [rdi]
    b.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
    b.extend_from_slice(op);
    b.extend_from_slice(&[0xDB, 0x3A]); // fstp tbyte [rdx]
    b.push(0xC3); // ret
    b
}

/// `faddp st1, st0` / `fmulp` / `fdivp` - pop-form binary arithmetic.
const FADDP: [u8; 2] = [0xDE, 0xC1];
const FMULP: [u8; 2] = [0xDE, 0xC9];
const FDIVP: [u8; 2] = [0xDE, 0xF9];

/// Run one `f80_program` over two 80-bit inputs and return the 80-bit result.
fn run_f80(op: &[u8], a: [u64; 2], b: [u64; 2], outline: bool) -> [u64; 2] {
    let bytes = f80_program(op);
    let mut lhs = a;
    let mut rhs = b;
    let mut out = [0u64; 2];

    LiftTest::new(&bytes)
        .with_config(|c| c.outline_helpers = outline)
        .reg(|c| {
            c.rdi = lhs.as_mut_ptr() as u64;
            c.rsi = rhs.as_mut_ptr() as u64;
            c.rdx = out.as_mut_ptr() as u64;
        })
        // The x87 stack is not part of the oracle's comparable state.
        .no_unicorn()
        .run();

    out
}

/// 80-bit encodings of a few exact values (`[mantissa, sign_exp]`).
const F80_ONE: [u64; 2] = [0x8000_0000_0000_0000, 0x3FFF];
const F80_TWO: [u64; 2] = [0x8000_0000_0000_0000, 0x4000];
const F80_THREE: [u64; 2] = [0xC000_0000_0000_0000, 0x4000];
const F80_SIX: [u64; 2] = [0xC000_0000_0000_0000, 0x4001];

#[test]
fn outlined_kernels_match_inline_emission() {
    // Every binary kernel, over operands chosen so the answer is exact and the
    // two emission strategies must agree bit-for-bit.
    let cases: [(&[u8], [u64; 2], [u64; 2]); 3] = [
        (&FADDP, F80_ONE, F80_TWO),   // 1 + 2
        (&FMULP, F80_TWO, F80_THREE), // 2 * 3
        (&FDIVP, F80_SIX, F80_THREE), // 6 / 3  (dst=st1=6, src=st0=3)
    ];

    for (op, a, b) in cases {
        let outlined = run_f80(op, a, b, true);
        let inlined = run_f80(op, a, b, false);
        assert_eq!(
            outlined, inlined,
            "outlined and inlined emission disagree for op {op:02X?}"
        );
    }

    // Spot-check one absolute value so this is not just two identical bugs:
    // 2 * 3 == 6.
    assert_eq!(run_f80(&FMULP, F80_TWO, F80_THREE, true), F80_SIX);
    // 6 / 3 == 2.
    assert_eq!(run_f80(&FDIVP, F80_SIX, F80_THREE, true), F80_TWO);
}

/// Compile a region both ways and return `(outlined_bytes, inlined_bytes)` of
/// emitted machine code.
fn text_sizes(bytes: &[u8]) -> (usize, usize) {
    let mut sizes = Vec::new();
    for outline in [true, false] {
        let config = LifterConfig {
            outline_helpers: outline,
            ..LifterConfig::default()
        };
        let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).expect("lift");
        let text = ArtifactBackend::compile_lifted_to_raw(code, None, None).expect("compile");
        sizes.push(text.len());
    }
    (sizes[0], sizes[1])
}

/// `fld tbyte [rsi]; fdivp st1, st0` repeated `n` times, bracketed by a load
/// and a store.
fn n_divides(n: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xDB, 0x2F]); // fld tbyte [rdi]
    for _ in 0..n {
        bytes.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
        bytes.extend_from_slice(&FDIVP);
    }
    bytes.extend_from_slice(&[0xDB, 0x3A]); // fstp tbyte [rdx]
    bytes.push(0xC3);
    bytes
}

#[test]
fn outlining_shrinks_code_for_repeated_x87_ops() {
    let (outlined, inlined) = text_sizes(&n_divides(10));
    assert!(
        outlined * 2 < inlined,
        "outlining should more than halve code size for ten divides, \
         got outlined={outlined} inlined={inlined}"
    );
}

#[test]
fn outlining_makes_each_extra_x87_site_cheap() {
    // The point of outlining is the *marginal* cost: one more divide should add
    // a call sequence, not another copy of the long-division kernel. Comparing
    // the 2-site and 12-site regions isolates that from the fixed cost both
    // strategies pay (prologue, dispatch, the x87 stack plumbing around each
    // op, and - for the outlined build - the single kernel body).
    let (outlined_small, inlined_small) = text_sizes(&n_divides(2));
    let (outlined_big, inlined_big) = text_sizes(&n_divides(12));

    let outlined_marginal = (outlined_big - outlined_small) / 10;
    let inlined_marginal = (inlined_big - inlined_small) / 10;
    assert!(
        outlined_marginal * 3 < inlined_marginal,
        "each extra outlined site should cost a fraction of an inlined one, \
         got {outlined_marginal} vs {inlined_marginal} bytes per site"
    );
}

#[test]
fn one_helper_per_kernel_however_many_sites() {
    // Two adds, two divides, one square root: five sites, three kernels.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xDB, 0x2F]); // fld tbyte [rdi]
    bytes.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
    for op in [&FADDP, &FADDP, &FDIVP, &FDIVP] {
        bytes.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
        bytes.extend_from_slice(op);
    }
    bytes.extend_from_slice(&[0xD9, 0xFA]); // fsqrt
    bytes.push(0xC3);

    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");

    let mut names: Vec<&str> = code.helpers.iter().map(|h| h.name).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            F80Kernel::Add.symbol(),
            F80Kernel::Div.symbol(),
            F80Kernel::Sqrt.symbol(),
        ]
    );
}

#[test]
fn helpers_reach_the_object_file() {
    let bytes = f80_program(&FDIVP);
    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    assert!(!code.helpers.is_empty(), "FDIVP should outline a helper");

    let obj_bytes = ArtifactBackend::compile_lifted_to_object(code, None, None, &[])
        .expect("compile to object");
    let obj = object::File::parse(&*obj_bytes).expect("parse object");

    let symbols: Vec<String> = obj
        .symbols()
        .filter_map(|s| s.name().ok().map(|n| n.to_string()))
        .collect();
    let has = |needle: &str| symbols.iter().any(|s| s.contains(needle));

    assert!(has("lifted_func"), "symbols: {symbols:?}");
    assert!(has(F80Kernel::Div.symbol()), "symbols: {symbols:?}");

    // Helper bodies are *declared* before `lifted_func` (that is what fixes the
    // namespace-0 index space) but *defined* after it, so the lifted function
    // still starts the text section - which is all `compile_to_raw` callers,
    // who get bytes with no symbol table, have to go on.
    let lifted = obj
        .symbols()
        .find(|s| s.name().is_ok_and(|n| n.contains("lifted_func")))
        .expect("lifted_func symbol");
    assert_eq!(lifted.address(), 0, "lifted_func should start .text");
}

#[test]
fn lift_to_function_refuses_to_drop_helpers() {
    // The bare-`Function` entry point cannot carry helper bodies, so rather
    // than hand back IR with unresolvable calls it must fail loudly.
    let bytes = f80_program(&FADDP);
    let err = microlift::core::lift_to_function::<X86_64>(
        &bytes,
        DEFAULT_LIFT_BASE,
        LifterConfig::default(),
    )
    .expect_err("should refuse");
    assert!(
        err.to_string().contains("lift_to_code"),
        "error should point at the fix, got: {err}"
    );

    // ...and with outlining off it succeeds, self-contained.
    let config = LifterConfig {
        outline_helpers: false,
        ..LifterConfig::default()
    };
    microlift::core::lift_to_function::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config)
        .expect("inline emission should yield a standalone function");
}

/// Pure-Rust `strlen` for the import in the mixed test below.
unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

/// Imports and helpers share one `ExternalName::user(0, i)` index space, and a
/// helper's final index depends on how many imports the *rest* of the lift
/// turns up - so a region that has both is the case where the renumbering in
/// `finish()` can go wrong. Silently: the call would just land on the other
/// callee.
#[test]
fn imports_and_helpers_share_one_index_space() {
    use microlift::core::ExecutionMode;
    use microlift::core::abi::CallAbi;
    use microlift::function_traits::{DynamicSymbol, trampoline_func};

    const LIFT_BASE: u64 = 0x1000;
    const IMPORT_VA: u64 = 0x2000;

    let text = b"microlift\0"; // strlen == 9
    let mut lhs = F80_ONE;
    let mut rhs = F80_TWO;
    let mut out = [0u64; 2];

    // mov rdi, text ; call strlen ; mov rsi,&lhs ; mov rdx,&rhs ; mov rcx,&out
    // fld tbyte [rsi] ; fld tbyte [rdx] ; faddp st1,st0 ; fstp tbyte [rcx] ; ret
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0x48, 0xBF]);
    bytes.extend_from_slice(&(text.as_ptr() as u64).to_le_bytes());
    let call_next_ip = LIFT_BASE + bytes.len() as u64 + 5;
    bytes.push(0xE8);
    bytes.extend_from_slice(&((IMPORT_VA as i64 - call_next_ip as i64) as i32).to_le_bytes());
    for (reg, ptr) in [
        (0xBEu8, lhs.as_mut_ptr() as u64),
        (0xBA, rhs.as_mut_ptr() as u64),
        (0xB9, out.as_mut_ptr() as u64),
    ] {
        bytes.extend_from_slice(&[0x48, reg]);
        bytes.extend_from_slice(&ptr.to_le_bytes());
    }
    bytes.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
    bytes.extend_from_slice(&[0xDB, 0x2A]); // fld tbyte [rdx]
    bytes.extend_from_slice(&FADDP);
    bytes.extend_from_slice(&[0xDB, 0x39]); // fstp tbyte [rcx]
    bytes.push(0xC3);

    let mut config = LifterConfig {
        execution_mode: ExecutionMode::RuntimeAssisted,
        ..LifterConfig::default()
    };
    config.register_fast_import(
        IMPORT_VA,
        DynamicSymbol::new("host_strlen", trampoline_func!(host_strlen, _)),
        CallAbi::sysv_amd64(),
    );

    let code = lift_to_code::<X86_64>(&bytes, LIFT_BASE, config).expect("lift");
    assert_eq!(code.imports.len(), 1, "one fast import");
    assert_eq!(code.helpers.len(), 1, "one softfloat helper (add)");
    // The import took index 0, so the helper must have been renumbered to 1.
    assert!(
        code.func.display().to_string().contains("u0:1"),
        "helper call should reference u0:1:\n{}",
        code.func.display()
    );

    let code_ptr = microlift::backend::CraneliftBackend::compile_lifted_with_fast_imports(
        code,
        cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        &[],
    )
    .expect("compile");

    let stack = vec![0u64; 1024];
    let rsp = stack.as_ptr() as u64 + 1024 * 8 - 8;
    unsafe { *(rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF };
    let mut ctx = RegisterContext {
        rsp,
        image_base: LIFT_BASE,
        ..RegisterContext::default()
    };

    let f: unsafe extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    unsafe { f(&mut ctx) };

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 9, "the import call went somewhere else");
    assert_eq!(out, F80_THREE, "the helper call went somewhere else");
}

#[test]
fn analysis_mode_stays_self_contained() {
    // Inspection IR must not sprout calls into helper functions a consumer
    // reading the CFG has no way to resolve.
    let bytes = f80_program(&FADDP);
    let config = LifterConfig {
        analysis_mode: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    assert!(code.helpers.is_empty());
}
