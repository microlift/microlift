//! Scalar-vs-vector XMM equivalence for the SSE handlers.
//!
//! The scalar `(lo, hi)` expansion is the authority on semantics; the
//! `use_vector_xmm` fast paths in `semantics/sse_vec.rs` are a performance
//! choice that must not change a single result bit. Every instruction with a
//! vector fast path is run here under both backing representations over seeded
//! pseudo-random XMM state, and the full architectural state is compared.
//!
//! This is the regression net for the fast paths: adding one without listing
//! its encoding here leaves it unproven.

use cranelift::codegen::isa::CallConv;
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_function};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

/// Run `bytes` under both XMM backing representations from identical state and
/// return the two resulting contexts.
fn both_modes(bytes: &[u8], seed: u64) -> (RegisterContext, RegisterContext) {
    let mut rng = Rng::new(seed);
    let mut init = RegisterContext::default();
    for xmm in init.xmm.iter_mut() {
        xmm[0] = rng.next_u64();
        xmm[1] = rng.next_u64();
    }
    // GPRs matter for the handlers that write one (PMOVMSKB, PEXTR*, CVT*2SI).
    init.rax = rng.next_u64();
    init.rcx = rng.next_u64();
    init.rdx = rng.next_u64();

    let scalar = LiftTest::new(bytes).ctx(init).vector_xmm(false).run().ctx;
    let vector = LiftTest::new(bytes).ctx(init).vector_xmm(true).run().ctx;
    (scalar, vector)
}

/// Assert that both representations agree on XMM state, GPRs and flags.
///
/// Each seed costs two JIT compiles, so the immediate sweeps below pass a small
/// `seeds` - an immediate selects a distinct lift-time code path, and a handful
/// of random 128-bit inputs per path already pins it down.
#[track_caller]
fn assert_modes_agree_n(name: &str, bytes: &[u8], seeds: u64) {
    for seed in 0..seeds {
        let (s, v) = both_modes(bytes, seed);
        assert_eq!(s.xmm, v.xmm, "{name}: xmm diverged (seed {seed})");
        assert_eq!(s.rax, v.rax, "{name}: rax diverged (seed {seed})");
        assert_eq!(s.rcx, v.rcx, "{name}: rcx diverged (seed {seed})");
        assert_eq!(s.rdx, v.rdx, "{name}: rdx diverged (seed {seed})");
        let flags = |c: &RegisterContext| (c.cf, u8::from(c.pf()), u8::from(c.af()), c.zf, c.sf, c.of);
        assert_eq!(flags(&s), flags(&v), "{name}: flags diverged (seed {seed})");
    }
}

/// [`assert_modes_agree_n`] at the default seed count, for the forms whose
/// behaviour is fixed at lift time (no immediate to sweep).
#[track_caller]
fn assert_modes_agree(name: &str, bytes: &[u8]) {
    assert_modes_agree_n(name, bytes, 16);
}

/// Build `prefix + opcode + modrm(xmm0, xmm1)` followed by `ret`, with an
/// optional trailing `imm8`.
fn snippet(bytes: &[u8], imm: Option<u8>) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if let Some(i) = imm {
        out.push(i);
    }
    out.push(0xC3);
    out
}

#[test]
fn shuffles_agree_across_xmm_representations() {
    // Every imm8 is exercised for the immediate-controlled shuffles: the mask
    // is built at lift time, so each immediate is a distinct code path.
    for imm in 0..=255u8 {
        // pshufd xmm0, xmm1, imm
        assert_modes_agree_n("pshufd", &snippet(&[0x66, 0x0F, 0x70, 0xC1], Some(imm)), 2);
        // shufps xmm0, xmm1, imm
        assert_modes_agree_n("shufps", &snippet(&[0x0F, 0xC6, 0xC1], Some(imm)), 2);
        // shufpd xmm0, xmm1, imm
        assert_modes_agree_n("shufpd", &snippet(&[0x66, 0x0F, 0xC6, 0xC1], Some(imm)), 2);
        // pshuflw xmm0, xmm1, imm
        assert_modes_agree_n("pshuflw", &snippet(&[0xF2, 0x0F, 0x70, 0xC1], Some(imm)), 2);
        // pshufhw xmm0, xmm1, imm
        assert_modes_agree_n("pshufhw", &snippet(&[0xF3, 0x0F, 0x70, 0xC1], Some(imm)), 2);
    }

    // Byte shifts and align: counts past the register width are the
    // interesting edges (all-zero result, and PALIGNR's 16..31 window).
    for imm in 0..=40u8 {
        // pslldq xmm0, imm
        assert_modes_agree_n("pslldq", &snippet(&[0x66, 0x0F, 0x73, 0xF8], Some(imm)), 3);
        // psrldq xmm0, imm
        assert_modes_agree_n("psrldq", &snippet(&[0x66, 0x0F, 0x73, 0xD8], Some(imm)), 3);
        // palignr xmm0, xmm1, imm
        assert_modes_agree_n(
            "palignr",
            &snippet(&[0x66, 0x0F, 0x3A, 0x0F, 0xC1], Some(imm)),
            3,
        );
    }

    let cases: &[(&str, &[u8])] = &[
        ("punpcklbw", &[0x66, 0x0F, 0x60, 0xC1]),
        ("punpcklwd", &[0x66, 0x0F, 0x61, 0xC1]),
        ("punpckldq", &[0x66, 0x0F, 0x62, 0xC1]),
        ("punpcklqdq", &[0x66, 0x0F, 0x6C, 0xC1]),
        ("punpckhbw", &[0x66, 0x0F, 0x68, 0xC1]),
        ("punpckhwd", &[0x66, 0x0F, 0x69, 0xC1]),
        ("punpckhdq", &[0x66, 0x0F, 0x6A, 0xC1]),
        ("punpckhqdq", &[0x66, 0x0F, 0x6D, 0xC1]),
        ("unpcklps", &[0x0F, 0x14, 0xC1]),
        ("unpckhps", &[0x0F, 0x15, 0xC1]),
        ("unpcklpd", &[0x66, 0x0F, 0x14, 0xC1]),
        ("unpckhpd", &[0x66, 0x0F, 0x15, 0xC1]),
        ("movsldup", &[0xF3, 0x0F, 0x12, 0xC1]),
        ("movshdup", &[0xF3, 0x0F, 0x16, 0xC1]),
        ("movddup", &[0xF2, 0x0F, 0x12, 0xC1]),
        ("movhlps", &[0x0F, 0x12, 0xC1]),
        ("movlhps", &[0x0F, 0x16, 0xC1]),
    ];
    for (name, bytes) in cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }
}

#[test]
fn packed_fp_ops_agree_across_xmm_representations() {
    let cases: &[(&str, &[u8])] = &[
        ("addps", &[0x0F, 0x58, 0xC1]),
        ("subps", &[0x0F, 0x5C, 0xC1]),
        ("mulps", &[0x0F, 0x59, 0xC1]),
        ("divps", &[0x0F, 0x5E, 0xC1]),
        ("addpd", &[0x66, 0x0F, 0x58, 0xC1]),
        ("subpd", &[0x66, 0x0F, 0x5C, 0xC1]),
        ("mulpd", &[0x66, 0x0F, 0x59, 0xC1]),
        ("divpd", &[0x66, 0x0F, 0x5E, 0xC1]),
        ("minps", &[0x0F, 0x5D, 0xC1]),
        ("maxps", &[0x0F, 0x5F, 0xC1]),
        ("minpd", &[0x66, 0x0F, 0x5D, 0xC1]),
        ("maxpd", &[0x66, 0x0F, 0x5F, 0xC1]),
        ("sqrtps", &[0x0F, 0x51, 0xC1]),
        ("sqrtpd", &[0x66, 0x0F, 0x51, 0xC1]),
        ("rcpps", &[0x0F, 0x53, 0xC1]),
        ("rsqrtps", &[0x0F, 0x52, 0xC1]),
        ("addsubps", &[0xF2, 0x0F, 0xD0, 0xC1]),
        ("addsubpd", &[0x66, 0x0F, 0xD0, 0xC1]),
    ];
    for (name, bytes) in cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }

    // All eight legacy compare predicates, both widths.
    for imm in 0..8u8 {
        assert_modes_agree_n("cmpps", &snippet(&[0x0F, 0xC2, 0xC1], Some(imm)), 8);
        assert_modes_agree_n("cmppd", &snippet(&[0x66, 0x0F, 0xC2, 0xC1], Some(imm)), 8);
    }
}

/// The scalar single/double arithmetic forms, whose vector fast path keeps
/// lane 0 in the vector register file instead of routing it through a GPR.
///
/// The merge is the whole risk here: these instructions write bits[31:0] (or
/// [63:0]) and *preserve* everything above, so the fast path's `insertlane`
/// has to leave lanes 1.. exactly as the portable mask-and-OR expansion does.
/// Seeded random XMM state is what catches a lane getting clobbered.
#[test]
fn scalar_fp_ops_agree_across_xmm_representations() {
    let cases: &[(&str, &[u8])] = &[
        ("addss", &[0xF3, 0x0F, 0x58, 0xC1]),
        ("subss", &[0xF3, 0x0F, 0x5C, 0xC1]),
        ("mulss", &[0xF3, 0x0F, 0x59, 0xC1]),
        ("divss", &[0xF3, 0x0F, 0x5E, 0xC1]),
        ("addsd", &[0xF2, 0x0F, 0x58, 0xC1]),
        ("subsd", &[0xF2, 0x0F, 0x5C, 0xC1]),
        ("mulsd", &[0xF2, 0x0F, 0x59, 0xC1]),
        ("divsd", &[0xF2, 0x0F, 0x5E, 0xC1]),
    ];
    for (name, bytes) in cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }

    // The memory-source forms take a separate path in the fast lane read (a
    // float-typed load rather than an `extractlane`), so they need their own
    // cases. `[rsp]` is the harness's stack, which holds the exit trampoline
    // address - an arbitrary but identical bit pattern under both modes.
    let mem_cases: &[(&str, &[u8])] = &[
        ("mulss-m32", &[0xF3, 0x0F, 0x59, 0x04, 0x24]),
        ("addss-m32", &[0xF3, 0x0F, 0x58, 0x04, 0x24]),
        ("divsd-m64", &[0xF2, 0x0F, 0x5E, 0x04, 0x24]),
        ("subsd-m64", &[0xF2, 0x0F, 0x5C, 0x04, 0x24]),
    ];
    for (name, bytes) in mem_cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }
}

#[test]
fn movmsk_and_conversions_agree_across_xmm_representations() {
    let cases: &[(&str, &[u8])] = &[
        ("pmovmskb", &[0x66, 0x0F, 0xD7, 0xC1]),
        ("movmskps", &[0x0F, 0x50, 0xC1]),
        ("movmskpd", &[0x66, 0x0F, 0x50, 0xC1]),
        ("cvtdq2ps", &[0x0F, 0x5B, 0xC1]),
        ("cvtps2dq", &[0x66, 0x0F, 0x5B, 0xC1]),
        ("cvttps2dq", &[0xF3, 0x0F, 0x5B, 0xC1]),
        ("cvtpd2ps", &[0x66, 0x0F, 0x5A, 0xC1]),
        ("cvtps2pd", &[0x0F, 0x5A, 0xC1]),
        ("cvtdq2pd", &[0xF3, 0x0F, 0xE6, 0xC1]),
    ];
    for (name, bytes) in cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }
}

#[test]
fn packed_integer_ops_agree_across_xmm_representations() {
    let cases: &[(&str, &[u8])] = &[
        ("pshufb", &[0x66, 0x0F, 0x38, 0x00, 0xC1]),
        // The blendv forms take their mask from the implicit XMM0, which is
        // also the destination here - the mask is read before the write.
        ("pblendvb", &[0x66, 0x0F, 0x38, 0x10, 0xC1]),
        ("blendvps", &[0x66, 0x0F, 0x38, 0x14, 0xC1]),
        ("blendvpd", &[0x66, 0x0F, 0x38, 0x15, 0xC1]),
        ("pabsb", &[0x66, 0x0F, 0x38, 0x1C, 0xC1]),
        ("pabsw", &[0x66, 0x0F, 0x38, 0x1D, 0xC1]),
        ("pabsd", &[0x66, 0x0F, 0x38, 0x1E, 0xC1]),
        ("packsswb", &[0x66, 0x0F, 0x63, 0xC1]),
        ("packssdw", &[0x66, 0x0F, 0x6B, 0xC1]),
        ("packuswb", &[0x66, 0x0F, 0x67, 0xC1]),
        ("packusdw", &[0x66, 0x0F, 0x38, 0x2B, 0xC1]),
        ("pmovsxbw", &[0x66, 0x0F, 0x38, 0x20, 0xC1]),
        ("pmovsxbd", &[0x66, 0x0F, 0x38, 0x21, 0xC1]),
        ("pmovsxbq", &[0x66, 0x0F, 0x38, 0x22, 0xC1]),
        ("pmovsxwd", &[0x66, 0x0F, 0x38, 0x23, 0xC1]),
        ("pmovsxwq", &[0x66, 0x0F, 0x38, 0x24, 0xC1]),
        ("pmovsxdq", &[0x66, 0x0F, 0x38, 0x25, 0xC1]),
        ("pmovzxbw", &[0x66, 0x0F, 0x38, 0x30, 0xC1]),
        ("pmovzxbd", &[0x66, 0x0F, 0x38, 0x31, 0xC1]),
        ("pmovzxbq", &[0x66, 0x0F, 0x38, 0x32, 0xC1]),
        ("pmovzxwd", &[0x66, 0x0F, 0x38, 0x33, 0xC1]),
        ("pmovzxwq", &[0x66, 0x0F, 0x38, 0x34, 0xC1]),
        ("pmovzxdq", &[0x66, 0x0F, 0x38, 0x35, 0xC1]),
        ("pmulhw", &[0x66, 0x0F, 0xE5, 0xC1]),
        ("pmulhuw", &[0x66, 0x0F, 0xE4, 0xC1]),
        ("pmullw", &[0x66, 0x0F, 0xD5, 0xC1]),
        ("pmulhrsw", &[0x66, 0x0F, 0x38, 0x0B, 0xC1]),
        ("pmaddubsw", &[0x66, 0x0F, 0x38, 0x04, 0xC1]),
        ("psignb", &[0x66, 0x0F, 0x38, 0x08, 0xC1]),
        ("psignw", &[0x66, 0x0F, 0x38, 0x09, 0xC1]),
        ("psignd", &[0x66, 0x0F, 0x38, 0x0A, 0xC1]),
        ("pmuludq", &[0x66, 0x0F, 0xF4, 0xC1]),
        ("pmuldq", &[0x66, 0x0F, 0x38, 0x28, 0xC1]),
        ("ptest", &[0x66, 0x0F, 0x38, 0x17, 0xC1]),
    ];
    for (name, bytes) in cases {
        assert_modes_agree(name, &snippet(bytes, None));
    }

    // The imm8-controlled blends: every lane-selection pattern.
    for imm in 0..=255u8 {
        // pblendw xmm0, xmm1, imm
        assert_modes_agree_n(
            "pblendw",
            &snippet(&[0x66, 0x0F, 0x3A, 0x0E, 0xC1], Some(imm)),
            2,
        );
        // blendps xmm0, xmm1, imm
        assert_modes_agree_n(
            "blendps",
            &snippet(&[0x66, 0x0F, 0x3A, 0x0C, 0xC1], Some(imm)),
            2,
        );
        // blendpd xmm0, xmm1, imm
        assert_modes_agree_n(
            "blendpd",
            &snippet(&[0x66, 0x0F, 0x3A, 0x0D, 0xC1], Some(imm)),
            2,
        );
    }
}

/// Every vector fast path must lower on every backend microlift can be pointed
/// at, not just the host.
///
/// This is not hypothetical: `smulhi`/`umulhi` are documented as
/// vector-polymorphic and `x86_pmulhrsw` / `x86_pmaddubsw` model their x86
/// instructions exactly, but none of the three has an aarch64 rule; the
/// `UnorderedOrGreaterThan{,OrEqual}` vector compares have no aarch64 rule
/// either. Each looked correct until compiled for the other target.
#[test]
fn vector_fast_paths_lower_on_every_target() {
    // One encoding per distinct vector fast path.
    let snippets: &[(&str, &[u8])] = &[
        ("pshufd", &[0x66, 0x0F, 0x70, 0xC1, 0x1B]),
        ("shufps", &[0x0F, 0xC6, 0xC1, 0x1B]),
        ("shufpd", &[0x66, 0x0F, 0xC6, 0xC1, 0x03]),
        ("pshuflw", &[0xF2, 0x0F, 0x70, 0xC1, 0x1B]),
        ("pshufhw", &[0xF3, 0x0F, 0x70, 0xC1, 0x1B]),
        ("punpcklbw", &[0x66, 0x0F, 0x60, 0xC1]),
        ("punpckhqdq", &[0x66, 0x0F, 0x6D, 0xC1]),
        ("pslldq", &[0x66, 0x0F, 0x73, 0xF8, 0x03]),
        ("psrldq", &[0x66, 0x0F, 0x73, 0xD8, 0x03]),
        ("palignr-lo", &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x05]),
        ("palignr-hi", &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x14]),
        ("movsldup", &[0xF3, 0x0F, 0x12, 0xC1]),
        ("movddup", &[0xF2, 0x0F, 0x12, 0xC1]),
        ("pshufb", &[0x66, 0x0F, 0x38, 0x00, 0xC1]),
        ("pblendvb", &[0x66, 0x0F, 0x38, 0x10, 0xC1]),
        ("pblendw", &[0x66, 0x0F, 0x3A, 0x0E, 0xC1, 0x5A]),
        ("pabsb", &[0x66, 0x0F, 0x38, 0x1C, 0xC1]),
        ("psignb", &[0x66, 0x0F, 0x38, 0x08, 0xC1]),
        ("packsswb", &[0x66, 0x0F, 0x63, 0xC1]),
        ("packusdw", &[0x66, 0x0F, 0x38, 0x2B, 0xC1]),
        ("pmovsxbq", &[0x66, 0x0F, 0x38, 0x22, 0xC1]),
        ("pmovzxwd", &[0x66, 0x0F, 0x38, 0x33, 0xC1]),
        ("pmulhw", &[0x66, 0x0F, 0xE5, 0xC1]),
        ("pmulhuw", &[0x66, 0x0F, 0xE4, 0xC1]),
        ("pmulhrsw", &[0x66, 0x0F, 0x38, 0x0B, 0xC1]),
        ("pmuludq", &[0x66, 0x0F, 0xF4, 0xC1]),
        ("pmuldq", &[0x66, 0x0F, 0x38, 0x28, 0xC1]),
        ("ptest", &[0x66, 0x0F, 0x38, 0x17, 0xC1]),
        ("pmovmskb", &[0x66, 0x0F, 0xD7, 0xC1]),
        ("movmskpd", &[0x66, 0x0F, 0x50, 0xC1]),
        ("addps", &[0x0F, 0x58, 0xC1]),
        ("divpd", &[0x66, 0x0F, 0x5E, 0xC1]),
        ("minps", &[0x0F, 0x5D, 0xC1]),
        ("maxpd", &[0x66, 0x0F, 0x5F, 0xC1]),
        ("sqrtps", &[0x0F, 0x51, 0xC1]),
        ("rsqrtps", &[0x0F, 0x52, 0xC1]),
        ("addsubps", &[0xF2, 0x0F, 0xD0, 0xC1]),
        ("addsubpd", &[0x66, 0x0F, 0xD0, 0xC1]),
        ("mulss", &[0xF3, 0x0F, 0x59, 0xC1]),
        ("divsd", &[0xF2, 0x0F, 0x5E, 0xC1]),
        ("mulss-m32", &[0xF3, 0x0F, 0x59, 0x04, 0x24]),
        ("divsd-m64", &[0xF2, 0x0F, 0x5E, 0x04, 0x24]),
        ("cvtdq2ps", &[0x0F, 0x5B, 0xC1]),
        ("cvttps2dq", &[0xF3, 0x0F, 0x5B, 0xC1]),
        ("cvtpd2ps", &[0x66, 0x0F, 0x5A, 0xC1]),
        ("cvtps2pd", &[0x0F, 0x5A, 0xC1]),
        ("cvtdq2pd", &[0xF3, 0x0F, 0xE6, 0xC1]),
        // These five have no vector fast path - their cross-lane reductions
        // have no Cranelift primitive behind them - but under `use_vector_xmm`
        // they still reach the XMM file through `extractlane`/`insertlane`, so
        // they emit vector IR and have to lower like everything else here.
        ("extractps", &[0x66, 0x0F, 0x3A, 0x17, 0xC8, 0x02]),
        ("dpps", &[0x66, 0x0F, 0x3A, 0x40, 0xC1, 0xFF]),
        ("dppd", &[0x66, 0x0F, 0x3A, 0x41, 0xC1, 0x33]),
        ("phminposuw", &[0x66, 0x0F, 0x38, 0x41, 0xC1]),
        ("mpsadbw", &[0x66, 0x0F, 0x3A, 0x42, 0xC1, 0x05]),
    ];

    // The `NLT` / `NLE` compare predicates are the ones without an aarch64
    // vector-float rule, so sweep all eight rather than picking one.
    let mut cases: Vec<(String, Vec<u8>)> = snippets
        .iter()
        .map(|(n, b)| ((*n).to_string(), snippet(b, None)))
        .collect();
    for imm in 0..8u8 {
        cases.push((
            format!("cmpps-{imm}"),
            snippet(&[0x0F, 0xC2, 0xC1], Some(imm)),
        ));
        cases.push((
            format!("cmppd-{imm}"),
            snippet(&[0x66, 0x0F, 0xC2, 0xC1], Some(imm)),
        ));
    }

    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for (name, bytes) in &cases {
            let config = LifterConfig {
                use_vector_xmm: true,
                ..Default::default()
            };
            let mut func = lift_to_function::<X86_64>(bytes, 0x1000, config).expect("lift");
            // The lifter bakes in the *host* calling convention; override it so
            // this exercises instruction lowering rather than tripping the
            // target's ABI code on a convention it does not accept.
            func.signature.call_conv = CallConv::SystemV;
            ArtifactBackend::compile_to_raw(func, Some(triple), None)
                .unwrap_or_else(|e| panic!("{name} failed to lower for {triple}: {e:?}"));
        }
    }
}
