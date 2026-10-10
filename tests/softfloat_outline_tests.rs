//! Outlining of the 80-bit softfloat kernels into shared helpers.
//! Pins: same answers as inlining, a code-size win, one helper per kernel, helper bodies in the object, and a consistent namespace-0 index space with imports.

use microlift::arch::x86_64::X86_64;
use microlift::arch::x86_64::semantics::softfloat::F80Kernel;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;
use object::{Object, ObjectSymbol};

/// `fldt [rdi]; fldt [rsi]; f<op> st1, st0; fstpt [rdx]`: one x87 op on two memory operands. Registers hold addresses at run time.
fn f80_program(op: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&[0xDB, 0x2F]); // fld tbyte [rdi]
    b.extend_from_slice(&[0xDB, 0x2E]); // fld tbyte [rsi]
    b.extend_from_slice(op);
    b.extend_from_slice(&[0xDB, 0x3A]); // fstp tbyte [rdx]
    b.push(0xC3); // ret
    b
}

/// Pop-form binary arithmetic.
const FADDP: [u8; 2] = [0xDE, 0xC1];
const FMULP: [u8; 2] = [0xDE, 0xC9];
const FDIVP: [u8; 2] = [0xDE, 0xF9];

/// Run one `f80_program` over two inputs and return the 80-bit result.
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
        // The x87 stack is not in the oracle's comparable state.
        .no_unicorn()
        .run();

    out
}

/// 80-bit encodings `[mantissa, sign_exp]` of exact values.
const F80_ONE: [u64; 2] = [0x8000_0000_0000_0000, 0x3FFF];
const F80_TWO: [u64; 2] = [0x8000_0000_0000_0000, 0x4000];
const F80_THREE: [u64; 2] = [0xC000_0000_0000_0000, 0x4000];
const F80_SIX: [u64; 2] = [0xC000_0000_0000_0000, 0x4001];

#[test]
fn outlined_kernels_match_inline_emission() {
    // Exact operands: both emission strategies must agree bit for bit.
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

    // Absolute spot-check, so this is not two identical bugs: 2 * 3 == 6.
    assert_eq!(run_f80(&FMULP, F80_TWO, F80_THREE, true), F80_SIX);
    // 6 / 3 == 2.
    assert_eq!(run_f80(&FDIVP, F80_SIX, F80_THREE, true), F80_TWO);
}

#[test]
fn outlined_trig_matches_inline_emission() {
    // Unary kernel: `fld tbyte [rdi]; <op>; fstp tbyte [rdx]; ret`.
    let run = |op: u8, arg: [u64; 2], outline: bool| {
        let bytes = [0xDB, 0x2F, 0xD9, op, 0xDB, 0x3A, 0xC3];
        let mut input = arg;
        let mut out = [0u64; 2];
        LiftTest::new(&bytes)
            .with_config(|c| c.outline_helpers = outline)
            .reg(|c| {
                c.rdi = input.as_mut_ptr() as u64;
                c.rdx = out.as_mut_ptr() as u64;
            })
            .no_unicorn()
            .run();
        out
    };

    // 0xFF = fcos, 0xFE = fsin: one body, two phases.
    for op in [0xFFu8, 0xFE] {
        for arg in [F80_ONE, F80_TWO, F80_THREE, F80_SIX] {
            assert_eq!(
                run(op, arg, true),
                run(op, arg, false),
                "outlined and inlined {op:#04x} disagree for {arg:#x?}"
            );
        }
    }
    // cos(0) == 1, sin(0) == 0 in both strategies.
    let zero = [0u64, 0];
    assert_eq!(run(0xFF, zero, true), F80_ONE);
    assert_eq!(run(0xFF, zero, false), F80_ONE);
    assert_eq!(run(0xFE, zero, true), zero);
    assert_eq!(run(0xFE, zero, false), zero);
}

#[test]
fn trig_kernel_cross_compiles() {
    // Kernel ops must lower everywhere (no host x86-64 needed).
    let bytes = [0xDB, 0x2F, 0xD9, 0xFF, 0xDB, 0x3A, 0xC3];
    // aarch64 is the target that rejects x64-only ops.
    let triple = "aarch64-unknown-linux-gnu";
    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    assert!(
        code.helpers
            .iter()
            .any(|h| h.name == F80Kernel::Trig.symbol()),
        "FCOS should outline the trig kernel"
    );
    ArtifactBackend::compile_lifted_to_object(code, Some(triple), None, &[])
        .unwrap_or_else(|e| panic!("compile for {triple}: {e}"));
}

#[test]
fn every_trig_instruction_shares_one_helper() {
    // FSIN/FCOS/FSINCOS/FPTAN differ only in a phase argument: one trig body plus the FPTAN divide.
    let bytes = [0xD9, 0xFE, 0xD9, 0xFF, 0xD9, 0xFB, 0xD9, 0xF2, 0xC3];
    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    let names: Vec<&str> = code.helpers.iter().map(|h| h.name).collect();
    assert_eq!(
        names
            .iter()
            .filter(|n| **n == F80Kernel::Trig.symbol())
            .count(),
        1,
        "one shared trig body for all four instructions, got {names:?}"
    );
    assert!(
        names.contains(&F80Kernel::Div.symbol()),
        "FPTAN divides sin by cos, got {names:?}"
    );
}

/// Compile both ways; return `(outlined_bytes, inlined_bytes)`.
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

/// `fld tbyte [rsi]; fdivp st1, st0` repeated `n` times between a load and a store.
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
    // Outlining pays in marginal cost: 2-site vs 12-site isolates the per-site call from fixed costs.
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
    // Two adds, two divides, one sqrt: five sites, three kernels.
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

    // Helpers are declared before but defined after `lifted_func`, so it still starts the text section (all `compile_to_raw` callers have to go on).
    let lifted = obj
        .symbols()
        .find(|s| s.name().is_ok_and(|n| n.contains("lifted_func")))
        .expect("lifted_func symbol");
    assert_eq!(lifted.address(), 0, "lifted_func should start .text");
}

#[test]
fn lift_to_function_refuses_to_drop_helpers() {
    // The bare-`Function` entry cannot carry helpers and must fail loudly.
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

    // With outlining off it succeeds, self-contained.
    let config = LifterConfig {
        outline_helpers: false,
        ..LifterConfig::default()
    };
    microlift::core::lift_to_function::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config)
        .expect("inline emission should yield a standalone function");
}

/// Pure-Rust `strlen` for the mixed test.
unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

/// Imports and helpers share one `ExternalName::user(0, i)` space; renumbering in `finish()` fails silently (the call lands on the other callee).
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
        // Makes the closing `ret` a clean `Return` exit.
        exit_trampoline_address: Some(0xFFFF_FFFF_FFFF_FFFF),
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
    // Import took index 0, so the helper must be renumbered to 1.
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
        ..RegisterContext::default()
    };
    ctx.image_base = LIFT_BASE;

    let f: unsafe extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };
    unsafe { f(&mut ctx) };

    assert_eq!(ctx.exit_reason, ExitReason::Return as u32);
    assert_eq!(ctx.rax, 9, "the import call went somewhere else");
    assert_eq!(out, F80_THREE, "the helper call went somewhere else");
}

#[test]
fn analysis_mode_stays_self_contained() {
    // Inspection IR must not call helpers a CFG consumer cannot resolve.
    let bytes = f80_program(&FADDP);
    let config = LifterConfig {
        analysis_mode: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    assert!(code.helpers.is_empty());
}
