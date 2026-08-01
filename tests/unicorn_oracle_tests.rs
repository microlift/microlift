//! Differential tests: the Microlift JIT vs the Unicorn engine, sharing
//! Microlift's guest VA == host VA memory model (Unicorn identity-maps each
//! touched page onto its own host address; see `testkit::oracle::unicorn`).
//!
//! Every case runs the same program on both engines from an identical seed and
//! asserts their architectural state matches. Divergence is a bug in one of the
//! two engines - which is the whole point of the oracle.
//!
//! Run: `cargo test --features unicorn-oracle --test unicorn_oracle_tests`

#![cfg(feature = "unicorn-oracle")]

use microlift::testkit::oracle::unicorn::{UnicornOracle, differential_unicorn};
use microlift::testkit::oracle::*;
use microlift::testkit::prelude::*;

const CODE_VA: u64 = 0x1000;
const BUDGET: u64 = 10_000;

/// Assert Microlift and Unicorn agree on `code` seeded from `init`.
fn agree(code: &[u8], init: &CpuState) {
    let diff = differential_unicorn(code, CODE_VA, init, &[], BUDGET, DiffOptions::default());
    let diff = diff.expect("both engines should reach a clean halt");
    assert!(diff.is_empty(), "microlift vs unicorn diverged: {diff:?}");
}

/// `mov rax, imm ; add rax, rcx ; ret` - the canonical smoke test.
#[test]
fn add_reg_reg_agrees() {
    let code = [
        0x48, 0xC7, 0xC0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100
        0x48, 0x01, 0xC8, // add rax, rcx
        0xC3, // ret
    ];
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RCX] = 0x2A;
    agree(&code, &init);
}

/// A pile of flag-defining ALU ops over seeded operands: the two engines must
/// agree bit-for-bit on CF/PF/AF/ZF/SF/OF, not just the result.
#[test]
fn alu_flags_agree_over_operands() {
    // op rax, rcx ; ret  for each of add/sub/and/or/xor/cmp.
    let ops: &[(&str, [u8; 2])] = &[
        ("add", [0x48, 0x01]),
        ("sub", [0x48, 0x29]),
        ("and", [0x48, 0x21]),
        ("or", [0x48, 0x09]),
        ("xor", [0x48, 0x31]),
    ];
    let operands: &[(u64, u64)] = &[
        (0, 0),
        (1, 0xFFFF_FFFF_FFFF_FFFF),
        (0x7FFF_FFFF_FFFF_FFFF, 1),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        (0xDEAD_BEEF, 0xCAFE),
        (0x1_0000_0000, 0x1_0000_0000),
    ];

    for (name, opc) in ops {
        for &(a, b) in operands {
            // <op> rax, rcx (opc[1] is the /r ModRM for rax,rcx = 0xC8) ; ret
            let code = [opc[0], opc[1], 0xC8, 0xC3];
            let mut init = CpuState::zeroed();
            init.gpr[gpr::RAX] = a;
            init.gpr[gpr::RCX] = b;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {a:#x},{b:#x} diverged: {diff:?}");
        }
    }
}

/// Shifts, `imul`, `neg`, `not`, `bswap`, `lea`, `movzx/movsx`, `setcc`, `cmov`
/// through a single straight-line block. Exercises a broad instruction slice in
/// one comparison.
#[test]
fn mixed_straightline_block_agrees() {
    let code = [
        0x48, 0xC7, 0xC0, 0x0F, 0x00, 0x00, 0x00, // mov rax, 15
        0x48, 0xC1, 0xE0, 0x04, // shl rax, 4
        0x48, 0x0F, 0xAF, 0xC1, // imul rax, rcx
        0x48, 0xF7, 0xD8, // neg rax
        0x48, 0xF7, 0xD0, // not rax
        0x48, 0x0F, 0xC8, // bswap rax
        0x48, 0x8D, 0x14, 0x40, // lea rdx, [rax+rax*2]
        0x0F, 0x94, 0xC3, // sete bl
        0x48, 0x0F, 0x44, 0xF2, // cmove rsi, rdx
        0xC3, // ret
    ];
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RCX] = 7;
    agree(&code, &init);
}

/// A non-diffable instruction Microlift turns into a VM exit (`cpuid`) must make
/// the differential decline (returns `None`) - Unicorn executes it, Microlift
/// exits, so there is nothing sound to compare.
#[test]
fn non_diffable_declines() {
    let code = [0x0F, 0xA2, 0xC3]; // cpuid ; ret
    let init = CpuState::zeroed();
    let diff = differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default());
    assert!(diff.is_none(), "cpuid must not be compared");
}

/// Identity-mapped memory: a store then load through a shared arena VA must read
/// back the same value under both engines, and the surrounding registers must
/// agree. Skipped if the fixed mapping is unavailable on this host.
#[cfg(unix)]
#[test]
fn identity_memory_roundtrip_agrees() {
    const ARENA_VA: u64 = 0x3_4000_0000;
    const ARENA_LEN: usize = 0x1000;

    let Some(mut arena) = IdentityArena::map(ARENA_VA, ARENA_LEN) else {
        eprintln!("identity arena unavailable at {ARENA_VA:#x}; skipping");
        return;
    };
    // Seed a known qword the guest will load.
    let seed: u64 = 0x0123_4567_89AB_CDEF;
    arena.write_at(0, &seed.to_le_bytes());

    // mov rax, [ARENA_VA] ; add rax, 1 ; mov [ARENA_VA+8], rax ; ret
    let mut code = vec![0x48, 0xA1]; // mov rax, moffs64
    code.extend_from_slice(&ARENA_VA.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01]); // add rax, 1
    code.extend_from_slice(&[0x48, 0xA3]); // mov moffs64, rax
    code.extend_from_slice(&(ARENA_VA + 8).to_le_bytes());
    code.push(0xC3); // ret

    let init = CpuState::zeroed();

    let mut ml = MicroliftOracle::default();
    let out_ml = ml.run(&code, CODE_VA, &init, &[], BUDGET);
    assert!(out_ml.is_halted());
    assert_eq!(out_ml.state().gpr[gpr::RAX], seed.wrapping_add(1));

    let mut uc = UnicornOracle::new();
    let out_uc = uc.run(&code, CODE_VA, &init, &[], BUDGET);
    assert!(out_uc.is_halted(), "unicorn should halt at the trampoline");

    let diff = diff_states(out_ml.state(), out_uc.state(), DiffOptions::default());
    assert!(
        diff.is_empty(),
        "engines diverged on memory roundtrip: {diff:?}"
    );

    let stored = u64::from_le_bytes(arena.as_mut_slice()[8..16].try_into().unwrap());
    assert_eq!(stored, seed.wrapping_add(1));
}

/// The testkit's `LiftTest::assert_matches_unicorn` hook: a program that both
/// asserts a concrete Microlift result *and* cross-checks Unicorn in one line.
#[test]
fn liftttest_hook_cross_checks() {
    // mov rax, 42 ; add rax, rax ; ret  -> 84
    let bytes = [
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0x48, 0x01, 0xC0, // add rax, rax
        0xC3, // ret
    ];
    let t = LiftTest::new(&bytes).at(CODE_VA).image_base(CODE_VA);
    t.assert_matches_unicorn(DiffOptions::default());

    let out = LiftTest::new(&bytes).at(CODE_VA).image_base(CODE_VA).run();
    assert_eq!(out.ctx.rax, 84);
}

/// Packed integer add/sub (`PADDB..PADDQ`, `PSUBB..PSUBQ` and the saturating
/// forms) over adversarial lane patterns.
///
/// The oracle runs the default (scalar lo/hi) XMM representation, which
/// implements these with SWAR / per-lane expansion - so this is the check that
/// that lane math matches hardware exactly, including carry/borrow isolation at
/// lane boundaries and the saturation clamps. The vector representation is
/// cross-checked against the same expectations in `tests/sse_tests.rs`.
#[test]
fn packed_int_add_sub_agrees() {
    // <op> xmm0, xmm1 ; ret
    let ops: &[(&str, u8)] = &[
        ("paddb", 0xFC),
        ("paddw", 0xFD),
        ("paddd", 0xFE),
        ("paddq", 0xD4),
        ("psubb", 0xF8),
        ("psubw", 0xF9),
        ("psubd", 0xFA),
        ("psubq", 0xFB),
        ("paddsb", 0xEC),
        ("paddsw", 0xED),
        ("paddusb", 0xDC),
        ("paddusw", 0xDD),
        ("psubsb", 0xE8),
        ("psubsw", 0xE9),
        ("psubusb", 0xD8),
        ("psubusw", 0xD9),
    ];
    let operands: &[([u64; 2], [u64; 2])] = &[
        ([0, 0], [0, 0]),
        (
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
            [0x0101_0101_0101_0101, 0x0000_0000_0000_0001],
        ),
        (
            [0x8080_8080_8080_8080, 0x7F7F_7F7F_7F7F_7F7F],
            [0x8080_8080_8080_8080, 0x0101_0101_0101_0101],
        ),
        (
            [0x8000_8000_8000_8000, 0x7FFF_7FFF_7FFF_7FFF],
            [0xFFFF_0001_8000_7FFF, 0x0001_FFFF_8000_7FFF],
        ),
        (
            [0x0000_0002_FFFF_FFFF, 0x8000_0000_7FFF_FFFF],
            [0x0000_0003_0000_0001, 0x8000_0000_0000_0001],
        ),
        (
            [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0],
            [0x1122_3344_5566_7788, 0xFEDC_BA98_7654_3210],
        ),
    ];

    for (name, opcode) in ops {
        for &(dst, src) in operands {
            let code = [0x66, 0x0F, *opcode, 0xC1, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// PCMPEQ/PCMPGT, PMIN/PMAX, PAVG and the PMUL forms against hardware.
///
/// The compares in particular are worth pinning down: they produce all-ones
/// lane masks, so any lane-boundary slip in the scalar expansion shows up as a
/// glaringly wrong result rather than an off-by-one.
#[test]
fn packed_lane_ops_agree() {
    // <op> xmm0, xmm1 ; ret   (all `66 0F <op> C1`)
    let ops: &[(&str, u8)] = &[
        ("pcmpeqb", 0x74),
        ("pcmpeqw", 0x75),
        ("pcmpeqd", 0x76),
        ("pcmpgtb", 0x64),
        ("pcmpgtw", 0x65),
        ("pcmpgtd", 0x66),
        ("pminub", 0xDA),
        ("pmaxub", 0xDE),
        ("pminsw", 0xEA),
        ("pmaxsw", 0xEE),
        ("pavgb", 0xE0),
        ("pavgw", 0xE3),
        ("pmullw", 0xD5),
        ("pmulhw", 0xE5),
        ("pmulhuw", 0xE4),
    ];
    for (name, opcode) in ops {
        for &(dst, src) in PACKED_OPERANDS {
            let code = [0x66, 0x0F, *opcode, 0xC1, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// PCMPEQQ / PCMPGTQ - the SSE4 quadword compares (three-byte opcodes).
#[test]
fn packed_qword_compares_agree() {
    for (name, opcode) in [("pcmpeqq", 0x29u8), ("pcmpgtq", 0x37)] {
        for &(dst, src) in PACKED_OPERANDS {
            let code = [0x66, 0x0F, 0x38, opcode, 0xC1, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// The full PUNPCK{L,H}{BW,WD,DQ,QDQ} family - pure lane permutation, so any
/// mis-ordered interleave is caught exactly.
#[test]
fn packed_unpack_agrees() {
    let ops: &[(&str, u8)] = &[
        ("punpcklbw", 0x60),
        ("punpcklwd", 0x61),
        ("punpckldq", 0x62),
        ("punpcklqdq", 0x6C),
        ("punpckhbw", 0x68),
        ("punpckhwd", 0x69),
        ("punpckhdq", 0x6A),
        ("punpckhqdq", 0x6D),
    ];
    for (name, opcode) in ops {
        for &(dst, src) in PACKED_OPERANDS {
            let code = [0x66, 0x0F, *opcode, 0xC1, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// PMOVMSKB / MOVMSKPS / MOVMSKPD - sign-bit gathering into a GPR.
#[test]
fn sign_mask_extraction_agrees() {
    // pmovmskb eax, xmm0 / movmskps eax, xmm0 / movmskpd eax, xmm0
    let ops: &[(&str, &[u8])] = &[
        ("pmovmskb", &[0x66, 0x0F, 0xD7, 0xC0, 0xC3]),
        ("movmskps", &[0x0F, 0x50, 0xC0, 0xC3]),
        ("movmskpd", &[0x66, 0x0F, 0x50, 0xC0, 0xC3]),
    ];
    for (name, code) in ops {
        for &(dst, _) in PACKED_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            // Seed a dirty destination so a failure to zero the upper bits shows.
            init.gpr[gpr::RAX] = 0xDEAD_BEEF_DEAD_BEEF;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {dst:#018x?} diverged: {diff:?}");
        }
    }
}

/// PSLLDQ / PSRLDQ across every interesting byte count: within a half, exactly
/// at the half boundary, across it, and past the end of the register.
#[test]
fn byte_shift_dq_agrees() {
    for count in [0u8, 1, 3, 7, 8, 9, 15, 16, 31, 255] {
        // pslldq xmm0, imm8 -> 66 0F 73 F8 ib ; psrldq -> 66 0F 73 D8 ib
        for (name, modrm) in [("pslldq", 0xF8u8), ("psrldq", 0xD8)] {
            let code = [0x66, 0x0F, 0x73, modrm, count, 0xC3];
            for &(dst, _) in PACKED_OPERANDS {
                let mut init = CpuState::zeroed();
                init.xmm[0] = dst;
                let diff = differential_unicorn(
                    &code,
                    CODE_VA,
                    &init,
                    &[],
                    BUDGET,
                    DiffOptions::default(),
                )
                .expect("clean halt");
                assert!(
                    diff.is_empty(),
                    "{name} {dst:#018x?}, {count} diverged: {diff:?}"
                );
            }
        }
    }
}

/// The newly added floating-point arithmetic, including the MIN/MAX forms whose
/// NaN and signed-zero behaviour is deliberately *not* IEEE.
#[test]
fn sse_float_arith_agrees() {
    let ops: &[(&str, &[u8])] = &[
        ("subsd", &[0xF2, 0x0F, 0x5C, 0xC1, 0xC3]),
        ("mulpd", &[0x66, 0x0F, 0x59, 0xC1, 0xC3]),
        ("divpd", &[0x66, 0x0F, 0x5E, 0xC1, 0xC3]),
        ("addps", &[0x0F, 0x58, 0xC1, 0xC3]),
        ("subps", &[0x0F, 0x5C, 0xC1, 0xC3]),
        ("divps", &[0x0F, 0x5E, 0xC1, 0xC3]),
        ("sqrtsd", &[0xF2, 0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtss", &[0xF3, 0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtps", &[0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtpd", &[0x66, 0x0F, 0x51, 0xC1, 0xC3]),
        ("minss", &[0xF3, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxss", &[0xF3, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("minsd", &[0xF2, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxsd", &[0xF2, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("minps", &[0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxps", &[0x0F, 0x5F, 0xC1, 0xC3]),
        ("minpd", &[0x66, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxpd", &[0x66, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("cvtss2sd", &[0xF3, 0x0F, 0x5A, 0xC1, 0xC3]),
        ("cvtsd2ss", &[0xF2, 0x0F, 0x5A, 0xC1, 0xC3]),
    ];
    for (name, code) in ops {
        for &(dst, src) in FLOAT_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// UCOMISS / COMISS - the flag-setting single-precision compares.
#[test]
fn ucomiss_agrees() {
    for (name, code) in [
        ("ucomiss", [0x0F, 0x2E, 0xC1, 0xC3]),
        ("comiss", [0x0F, 0x2F, 0xC1, 0xC3]),
    ] {
        for &(dst, src) in FLOAT_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// CVTSI2SD / CVTTSD2SI in both the 32- and 64-bit operand forms, including the
/// NaN and out-of-range inputs that must yield the integer indefinite value.
#[test]
fn sse2_int_float_conversions_agree() {
    let doubles: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        12.75,
        -12.75,
        1e12,
        -1e12,
        1e30,
        -1e30,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9223372036854775808.0,
        -9223372036854775808.0,
    ];
    // cvttsd2si eax, xmm0 / cvttsd2si rax, xmm0
    for (name, code) in [
        ("cvttsd2si r32", &[0xF2, 0x0F, 0x2C, 0xC0, 0xC3][..]),
        ("cvttsd2si r64", &[0xF2, 0x48, 0x0F, 0x2C, 0xC0, 0xC3][..]),
    ] {
        for &v in doubles {
            let mut init = CpuState::zeroed();
            init.xmm[0] = [v.to_bits(), 0];
            init.gpr[gpr::RAX] = 0xDEAD_BEEF_DEAD_BEEF;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {v} diverged: {diff:?}");
        }
    }

    // cvtsi2sd xmm0, eax / cvtsi2sd xmm0, rax
    for (name, code) in [
        ("cvtsi2sd r32", &[0xF2, 0x0F, 0x2A, 0xC0, 0xC3][..]),
        ("cvtsi2sd r64", &[0xF2, 0x48, 0x0F, 0x2A, 0xC0, 0xC3][..]),
    ] {
        for v in [
            0i64,
            1,
            -1,
            12345,
            -12345,
            i32::MIN as i64,
            i64::MIN,
            i64::MAX,
        ] {
            let mut init = CpuState::zeroed();
            init.gpr[gpr::RAX] = v as u64;
            init.xmm[0] = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0];
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {v} diverged: {diff:?}");
        }
    }
}

/// Bit patterns that stress lane boundaries, sign bits and saturation edges.
const PACKED_OPERANDS: &[([u64; 2], [u64; 2])] = &[
    ([0, 0], [0, 0]),
    (
        [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF],
        [0x0101_0101_0101_0101, 0x0000_0000_0000_0001],
    ),
    (
        [0x8080_8080_8080_8080, 0x7F7F_7F7F_7F7F_7F7F],
        [0x8080_8080_8080_8080, 0x0101_0101_0101_0101],
    ),
    (
        [0x8000_8000_8000_8000, 0x7FFF_7FFF_7FFF_7FFF],
        [0xFFFF_0001_8000_7FFF, 0x0001_FFFF_8000_7FFF],
    ),
    (
        [0x0000_0002_FFFF_FFFF, 0x8000_0000_7FFF_FFFF],
        [0x0000_0003_0000_0001, 0x8000_0000_0000_0001],
    ),
    (
        [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0],
        [0x1122_3344_5566_7788, 0xFEDC_BA98_7654_3210],
    ),
    (
        [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908],
        [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918],
    ),
    // Byte patterns aimed at PSHUFB's control rules: indices with bit 7 set
    // (which zero the output byte) and indices whose bits 4..6 are set (which
    // hardware masks away rather than treating as out of range).
    (
        [0x0F0E_0D0C_0B0A_0908, 0x0706_0504_0302_0100],
        [0x8898_A8B8_C8D8_E8F8, 0x3021_1203_7F40_5160],
    ),
];

/// XMM seeds interpreted as floats, for the arithmetic differential.
///
/// Every *dword* of the **source** operand is a positive finite normal `f32`
/// and the whole quadword is a positive finite normal `f64`, while the
/// destination ranges freely over signed
/// zeros, NaNs, infinities, negatives and denormals. That asymmetry is
/// deliberate: it keeps every case free of an operation that *generates* a
/// fresh default NaN (`0/0`, `inf-inf`, `0*inf`, `sqrt(-x)`).
///
/// Generated default NaNs are excluded because x86 and AArch64 disagree on
/// their sign - x86's integer/floating indefinite is `0xFFF8_0000_0000_0000`,
/// the AArch64 default NaN is `0x7FF8_0000_0000_0000` - and Microlift lowers
/// these ops to Cranelift `fdiv`/`fsub`/`sqrt`, so the *host* FPU picks the
/// sign. That divergence is pre-existing and general (untouched instructions
/// such as `DIVSD` show it identically); it is not a property of the
/// instructions added here. *Propagated* input NaNs keep their exact bits and
/// are covered - the NaN destinations below flow through every op.
const FLOAT_OPERANDS: &[([u64; 2], [u64; 2])] = &[
    // 1.0 / 2.0 (f64) destinations against 2.0 / 1.0 sources
    (
        [0x3FF0_0000_0000_0000, 0x4000_0000_0000_0000],
        [0x4000_0000_4000_0000, 0x3F80_0000_3F80_0000],
    ),
    // f32 lanes 1.5, 2.5, -3.0, 4.0 against 2.0, 2.0, 4.0, 8.0
    (
        [0x4020_0000_3FC0_0000, 0x4080_0000_C040_0000],
        [0x4000_0000_4000_0000, 0x4100_0000_4080_0000],
    ),
    // signed zeros in the destination
    (
        [0x8000_0000_0000_0000, 0x0000_0000_0000_0000],
        [0x3F80_0000_3F80_0000, 0x4049_0FDB_4049_0FDB],
    ),
    // QNaN destinations: the payload and sign must propagate unchanged
    (
        [0x7FF8_0000_0000_0000, 0xFFF8_0000_0000_0000],
        [0x4000_0000_4000_0000, 0x3F80_0000_3F80_0000],
    ),
    // infinities and negatives in the destination
    (
        [0xFFF0_0000_0000_0000, 0x7FF0_0000_0000_0000],
        [0x4049_0FDB_4049_0FDB, 0x4000_0000_4000_0000],
    ),
    // denormal and near-overflow magnitudes
    (
        [0x0000_0000_0000_0001, 0x7FEF_FFFF_FFFF_FFFF],
        [0x4100_0000_4080_0000, 0x3FC0_0000_4020_0000],
    ),
];

/// MIN/MAX and the compares against NaN- and zero-heavy operands.
///
/// These never generate a NaN - they only *select* an operand - so unlike
/// [`FLOAT_OPERANDS`] they can put NaNs and zeros on both sides, which is
/// exactly where x86's asymmetric "return the source unless the strict
/// comparison holds" rule differs from an IEEE `minNum`/`maxNum`.
const SELECT_OPERANDS: &[([u64; 2], [u64; 2])] = &[
    // -0.0 vs +0.0, both orders
    (
        [0x8000_0000_0000_0000, 0x0000_0000_0000_0000],
        [0x0000_0000_0000_0000, 0x8000_0000_0000_0000],
    ),
    // NaN destination against a finite source, and the reverse
    (
        [0x7FF8_0000_0000_0000, 0x3FF0_0000_0000_0000],
        [0x3FF0_0000_0000_0000, 0xFFF8_0000_0000_0000],
    ),
    // NaN on both sides: the source must win
    (
        [0x7FF8_0000_DEAD_BEEF, 0xFFF8_0000_CAFE_BABE],
        [0xFFF8_0000_1234_5678, 0x7FF8_0000_9ABC_DEF0],
    ),
    // ordinary ordered pairs, plus infinities
    (
        [0x3FF0_0000_0000_0000, 0xFFF0_0000_0000_0000],
        [0x4000_0000_0000_0000, 0x7FF0_0000_0000_0000],
    ),
    (
        [0x4020_0000_3FC0_0000, 0x4080_0000_C040_0000],
        [0xBF80_0000_3F00_0000, 0x4100_0000_4000_0000],
    ),
];

/// MINSS/MAXSS/MINPS/MAXPS/MINSD/MAXSD/MINPD/MAXPD over NaNs and signed zeros.
///
/// Split out from [`sse_float_arith_agrees`] so it can use the NaN-on-both-sides
/// operands that a NaN-generating op could not.
#[test]
fn sse_minmax_selection_agrees() {
    let ops: &[(&str, &[u8])] = &[
        ("minss", &[0xF3, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxss", &[0xF3, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("minsd", &[0xF2, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxsd", &[0xF2, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("minps", &[0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxps", &[0x0F, 0x5F, 0xC1, 0xC3]),
        ("minpd", &[0x66, 0x0F, 0x5D, 0xC1, 0xC3]),
        ("maxpd", &[0x66, 0x0F, 0x5F, 0xC1, 0xC3]),
        ("ucomiss", &[0x0F, 0x2E, 0xC1, 0xC3]),
        ("ucomisd", &[0x66, 0x0F, 0x2E, 0xC1, 0xC3]),
    ];
    for (name, code) in ops {
        for &(dst, src) in SELECT_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// The packed conversions - `CVTPD2PS`, `CVTPS2PD`, `CVTDQ2PS`,
/// `CVT[T]PS2DQ`, `CVTDQ2PD`, `CVT[T]PD2DQ` - over the shared float operand
/// pool plus integer bit patterns that exercise the dword-lane forms.
///
/// This is the check that the rounding mode (nearest-even), the
/// integer-indefinite result for unrepresentable inputs, and the zeroing of
/// bits[127:64] by the narrowing forms all match hardware.
#[test]
fn packed_conversions_agree() {
    let ops: &[(&str, &[u8])] = &[
        ("cvtpd2ps", &[0x66, 0x0F, 0x5A, 0xC1, 0xC3]),
        ("cvtps2pd", &[0x0F, 0x5A, 0xC1, 0xC3]),
        ("cvtdq2ps", &[0x0F, 0x5B, 0xC1, 0xC3]),
        ("cvtps2dq", &[0x66, 0x0F, 0x5B, 0xC1, 0xC3]),
        ("cvttps2dq", &[0xF3, 0x0F, 0x5B, 0xC1, 0xC3]),
        ("cvtdq2pd", &[0xF3, 0x0F, 0xE6, 0xC1, 0xC3]),
        ("cvtpd2dq", &[0xF2, 0x0F, 0xE6, 0xC1, 0xC3]),
        ("cvttpd2dq", &[0x66, 0x0F, 0xE6, 0xC1, 0xC3]),
    ];

    // Dword patterns: zero, +-1, the signed extremes, and values whose f32
    // conversion must round, plus halfway cases for the float -> int direction.
    let int_operands: &[([u64; 2], [u64; 2])] = &[
        (
            [0x0000_0001_0000_0000, 0xFFFF_FFFF_7FFF_FFFF],
            [0x8000_0000_0000_0002, 0x0100_0001_00FF_FFFF],
        ),
        (
            [0x3FC0_0000_4020_0000, 0xBFC0_0000_3F00_0000], // 1.5 / 2.5 / 0.5 / -1.5
            [0x4700_0000_C700_0000, 0x7F80_0000_FF80_0000], // +-32768.0, +-inf
        ),
    ];

    for (name, code) in ops {
        for &(dst, src) in FLOAT_OPERANDS.iter().chain(int_operands) {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// CVTSS2SI / CVTSD2SI - the rounding (not truncating) scalar-to-integer forms,
/// in both operand widths.
#[test]
fn rounding_scalar_to_int_conversions_agree() {
    // Halfway cases in both directions, plus the unrepresentable inputs.
    let doubles: &[f64] = &[
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        12.75,
        -12.75,
        1e12,
        1e30,
        -1e30,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2147483647.5,
        -2147483648.5,
    ];

    for (name, code) in [
        ("cvtsd2si r32", &[0xF2, 0x0F, 0x2D, 0xC0, 0xC3][..]),
        ("cvtsd2si r64", &[0xF2, 0x48, 0x0F, 0x2D, 0xC0, 0xC3][..]),
    ] {
        for &v in doubles {
            let mut init = CpuState::zeroed();
            init.xmm[0] = [v.to_bits(), 0];
            init.gpr[gpr::RAX] = 0xDEAD_BEEF_DEAD_BEEF;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {v} diverged: {diff:?}");
        }
    }

    for (name, code) in [
        ("cvtss2si r32", &[0xF3, 0x0F, 0x2D, 0xC0, 0xC3][..]),
        ("cvtss2si r64", &[0xF3, 0x48, 0x0F, 0x2D, 0xC0, 0xC3][..]),
    ] {
        for &v in doubles {
            let v = v as f32;
            let mut init = CpuState::zeroed();
            init.xmm[0] = [u64::from(v.to_bits()), 0];
            init.gpr[gpr::RAX] = 0xDEAD_BEEF_DEAD_BEEF;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {v} diverged: {diff:?}");
        }
    }
}

/// CMPPS / CMPPD / CMPSS / CMPSD across all eight legacy predicates, and the
/// lane shuffles (SHUFPS / SHUFPD / PSHUFLW / PSHUFHW) and dword duplications
/// (MOVSLDUP / MOVSHDUP) compilers emit alongside them.
#[test]
fn packed_compare_and_shuffle_agree() {
    for imm in 0..8u8 {
        let ops: &[(&str, &[u8])] = &[
            ("cmpps", &[0x0F, 0xC2, 0xC1, imm, 0xC3]),
            ("cmppd", &[0x66, 0x0F, 0xC2, 0xC1, imm, 0xC3]),
            ("cmpss", &[0xF3, 0x0F, 0xC2, 0xC1, imm, 0xC3]),
            ("cmpsd", &[0xF2, 0x0F, 0xC2, 0xC1, imm, 0xC3]),
        ];
        for (name, code) in ops {
            for &(dst, src) in FLOAT_OPERANDS {
                let mut init = CpuState::zeroed();
                init.xmm[0] = dst;
                init.xmm[1] = src;
                let diff =
                    differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                        .expect("clean halt");
                assert!(
                    diff.is_empty(),
                    "{name} imm={imm} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
                );
            }
        }
    }

    let lanes = (
        [0x0004_0003_0002_0001u64, 0x0008_0007_0006_0005u64],
        [0x1004_1003_1002_1001u64, 0x1008_1007_1006_1005u64],
    );
    for imm in [0x00u8, 0x1B, 0x55, 0xE4, 0xFF] {
        let ops: &[(&str, &[u8])] = &[
            ("shufps", &[0x0F, 0xC6, 0xC1, imm, 0xC3]),
            ("shufpd", &[0x66, 0x0F, 0xC6, 0xC1, imm, 0xC3]),
            ("pshuflw", &[0xF2, 0x0F, 0x70, 0xC1, imm, 0xC3]),
            ("pshufhw", &[0xF3, 0x0F, 0x70, 0xC1, imm, 0xC3]),
        ];
        for (name, code) in ops {
            let mut init = CpuState::zeroed();
            init.xmm[0] = lanes.0;
            init.xmm[1] = lanes.1;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} imm={imm:#04x} diverged: {diff:?}");
        }
    }

    for (name, code) in [
        ("movsldup", &[0xF3, 0x0F, 0x12, 0xC1, 0xC3][..]),
        ("movshdup", &[0xF3, 0x0F, 0x16, 0xC1, 0xC3][..]),
    ] {
        let mut init = CpuState::zeroed();
        init.xmm[0] = lanes.0;
        init.xmm[1] = lanes.1;
        let diff = differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
            .expect("clean halt");
        assert!(diff.is_empty(), "{name} diverged: {diff:?}");
    }
}

/// PINSRW against hardware: every word selector (including the imm8 bits above
/// `[2:0]` that hardware ignores), for both the GPR and the m16 source form.
#[test]
fn pinsrw_agrees() {
    for imm in 0..16u8 {
        // pinsrw xmm0, ecx, imm8 ; ret
        let code = [0x66, 0x0F, 0xC4, 0xC1, imm, 0xC3];
        let mut init = CpuState::zeroed();
        init.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
        init.gpr[gpr::RCX] = 0xDEAD_BEEF_0000_ABCD;
        let diff = differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
            .expect("clean halt");
        assert!(diff.is_empty(), "pinsrw reg imm={imm} diverged: {diff:?}");
    }
}

/// The `PINSRW xmm, m16, imm8` form, which needs an identity-mapped source
/// word both engines can read. Skipped if the fixed mapping is unavailable.
#[cfg(unix)]
#[test]
fn pinsrw_mem_agrees() {
    const ARENA_VA: u64 = 0x3_5000_0000;

    let Some(mut arena) = IdentityArena::map(ARENA_VA, 0x1000) else {
        eprintln!("identity arena unavailable at {ARENA_VA:#x}; skipping");
        return;
    };
    arena.write_at(0, &0xBEEFu16.to_le_bytes());
    let seed = [MemImage {
        base: ARENA_VA,
        bytes: 0xBEEFu16.to_le_bytes().to_vec(),
    }];

    for imm in 0..8u8 {
        // pinsrw xmm0, word ptr [rax], imm8 ; ret
        let code = [0x66, 0x0F, 0xC4, 0x00, imm, 0xC3];
        let mut init = CpuState::zeroed();
        init.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
        init.gpr[gpr::RAX] = ARENA_VA;
        let diff =
            differential_unicorn(&code, CODE_VA, &init, &seed, BUDGET, DiffOptions::default())
                .expect("clean halt");
        assert!(diff.is_empty(), "pinsrw mem imm={imm} diverged: {diff:?}");
    }
}

/// Run `code` against Unicorn for every [`PACKED_OPERANDS`] pair, seeded into
/// `xmm0` (destination) and `xmm1` (source).
fn agree_packed(name: &str, code: &[u8]) {
    for &(dst, src) in PACKED_OPERANDS {
        let mut init = CpuState::zeroed();
        init.xmm[0] = dst;
        init.xmm[1] = src;
        let diff = differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
            .expect("clean halt");
        assert!(
            diff.is_empty(),
            "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
        );
    }
}

/// The SSSE3 block: `PSHUFB`, the horizontal add/sub forms, the sign ops, the
/// mixed-signedness multiply-accumulates and `PABS*`.
///
/// `PSHUFB` gets the shuffle-heavy operand set for free - every pair above is
/// used as a control vector as well as data, so the byte selectors cover
/// in-range indices, the high-bit zeroing rule and indices whose bits 4..6 are
/// set (which hardware masks away and Cranelift's `swizzle` would not).
#[test]
fn ssse3_lane_ops_agree() {
    // <op> xmm0, xmm1 ; ret   (all `66 0F 38 <op> C1`)
    let ops: &[(&str, u8)] = &[
        ("pshufb", 0x00),
        ("phaddw", 0x01),
        ("phaddd", 0x02),
        ("phaddsw", 0x03),
        ("pmaddubsw", 0x04),
        ("phsubw", 0x05),
        ("phsubd", 0x06),
        ("phsubsw", 0x07),
        ("psignb", 0x08),
        ("psignw", 0x09),
        ("psignd", 0x0A),
        ("pmulhrsw", 0x0B),
        ("pabsb", 0x1C),
        ("pabsw", 0x1D),
        ("pabsd", 0x1E),
    ];
    for (name, op) in ops {
        agree_packed(name, &[0x66, 0x0F, 0x38, *op, 0xC1, 0xC3]);
    }
}

/// PALIGNR over every byte count, including the 16..31 range that shifts the
/// destination down into the result and the >= 32 range that clears it.
#[test]
fn palignr_agrees() {
    for imm in 0..=34u8 {
        agree_packed("palignr", &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, imm, 0xC3]);
    }
}

/// The widening multiplies, the dot product, sum-of-absolute-differences and
/// the saturating narrowing packs.
#[test]
fn packed_multiply_and_pack_agree() {
    // `66 0F <op> C1`
    for (name, op) in [
        ("pmuludq", 0xF4u8),
        ("pmaddwd", 0xF5),
        ("psadbw", 0xF6),
        ("packsswb", 0x63),
        ("packssdw", 0x6B),
        ("packuswb", 0x67),
    ] {
        agree_packed(name, &[0x66, 0x0F, op, 0xC1, 0xC3]);
    }
    // `66 0F 38 <op> C1`
    for (name, op) in [("pmuldq", 0x28u8), ("pmulld", 0x40), ("packusdw", 0x2B)] {
        agree_packed(name, &[0x66, 0x0F, 0x38, op, 0xC1, 0xC3]);
    }
}

/// The SSE4.1 min/max set and `PTEST` (whose whole observable effect is
/// flags - the `rflags` comparison in the diff is the test).
#[test]
fn sse41_minmax_and_ptest_agree() {
    let ops: &[(&str, u8)] = &[
        ("pminsb", 0x38),
        ("pminsd", 0x39),
        ("pminuw", 0x3A),
        ("pminud", 0x3B),
        ("pmaxsb", 0x3C),
        ("pmaxsd", 0x3D),
        ("pmaxuw", 0x3E),
        ("pmaxud", 0x3F),
        ("ptest", 0x17),
    ];
    for (name, op) in ops {
        agree_packed(name, &[0x66, 0x0F, 0x38, *op, 0xC1, 0xC3]);
    }
    // PTEST's two flag-setting edges: CF is set only when src & ~dst is zero.
    for (dst, src) in [
        ([0u64; 2], [0u64; 2]),
        ([!0, !0], [0x1234, 0]),
        ([0x00FF_00FF, 0], [0x00FF_0000, 0]),
    ] {
        let mut init = CpuState::zeroed();
        init.xmm[0] = dst;
        init.xmm[1] = src;
        let code = [0x66, 0x0F, 0x38, 0x17, 0xC1, 0xC3];
        let diff = differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
            .expect("clean halt");
        assert!(diff.is_empty(), "ptest {dst:?},{src:?} diverged: {diff:?}");
    }
}

/// The sign- and zero-extending widening loads (register source form).
#[test]
fn pmovsx_pmovzx_agree() {
    let ops: &[(&str, u8)] = &[
        ("pmovsxbw", 0x20),
        ("pmovsxbd", 0x21),
        ("pmovsxbq", 0x22),
        ("pmovsxwd", 0x23),
        ("pmovsxwq", 0x24),
        ("pmovsxdq", 0x25),
        ("pmovzxbw", 0x30),
        ("pmovzxbd", 0x31),
        ("pmovzxbq", 0x32),
        ("pmovzxwd", 0x33),
        ("pmovzxwq", 0x34),
        ("pmovzxdq", 0x35),
    ];
    for (name, op) in ops {
        agree_packed(name, &[0x66, 0x0F, 0x38, *op, 0xC1, 0xC3]);
    }
}

/// Blends: the `imm8`-selected forms and the variable forms that take their
/// mask from the implicit `XMM0` - which is also the destination here, so the
/// mask is read before it is overwritten.
#[test]
fn blends_agree() {
    for imm in [0x00u8, 0x01, 0x55, 0xAA, 0xF0, 0xFF] {
        for (name, op) in [("pblendw", 0x0Eu8), ("blendps", 0x0C), ("blendpd", 0x0D)] {
            agree_packed(name, &[0x66, 0x0F, 0x3A, op, 0xC1, imm, 0xC3]);
        }
    }
    // pblendvb/blendvps/blendvpd xmm2, xmm1  - mask in xmm0, dst xmm2.
    for (name, op) in [("pblendvb", 0x10u8), ("blendvps", 0x14), ("blendvpd", 0x15)] {
        for &(mask, src) in PACKED_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = mask;
            init.xmm[1] = src;
            init.xmm[2] = [0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555];
            // modrm D1: reg = xmm2, rm = xmm1
            let code = [0x66, 0x0F, 0x38, op, 0xD1, 0xC3];
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "{name} mask={mask:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// Lane insert and extract at every lane width and index, in both directions
/// (`PINSR*` from a GPR, `PEXTR*` into a GPR).
#[test]
fn lane_insert_extract_agree() {
    for lane in 0..16u8 {
        let mut init = CpuState::zeroed();
        init.xmm[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
        init.gpr[gpr::RCX] = 0xDEAD_BEEF_1234_ABCD;

        // pinsrb / pinsrw / pinsrd / pinsrq xmm0, rcx, lane ; ret
        let progs: Vec<(&str, Vec<u8>)> = vec![
            ("pinsrb", vec![0x66, 0x0F, 0x3A, 0x20, 0xC1, lane, 0xC3]),
            ("pinsrw", vec![0x66, 0x0F, 0xC4, 0xC1, lane, 0xC3]),
            ("pinsrd", vec![0x66, 0x0F, 0x3A, 0x22, 0xC1, lane, 0xC3]),
            (
                "pinsrq",
                vec![0x66, 0x48, 0x0F, 0x3A, 0x22, 0xC1, lane, 0xC3],
            ),
            ("pextrb", vec![0x66, 0x0F, 0x3A, 0x14, 0xC1, lane, 0xC3]),
            ("pextrw", vec![0x66, 0x0F, 0xC5, 0xC1, lane, 0xC3]),
            ("pextrd", vec![0x66, 0x0F, 0x3A, 0x16, 0xC1, lane, 0xC3]),
            (
                "pextrq",
                vec![0x66, 0x48, 0x0F, 0x3A, 0x16, 0xC1, lane, 0xC3],
            ),
        ];
        for (name, code) in progs {
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} lane={lane} diverged: {diff:?}");
        }
    }
}

/// The memory-source forms: a 128-bit operand for PSHUFB/PMULUDQ and the
/// *narrower* loads PMOVZX/PMOVSX and PINSRB use, which must read only the
/// bytes they consume. Skipped if the fixed mapping is unavailable.
#[cfg(unix)]
#[test]
fn packed_memory_sources_agree() {
    const ARENA_VA: u64 = 0x3_6000_0000;

    let Some(mut arena) = IdentityArena::map(ARENA_VA, 0x1000) else {
        eprintln!("identity arena unavailable at {ARENA_VA:#x}; skipping");
        return;
    };
    let data: [u8; 16] = [
        0x81, 0x02, 0xF3, 0x04, 0x85, 0x06, 0x77, 0x08, 0x09, 0x8A, 0x0B, 0xCC, 0x0D, 0x0E, 0x0F,
        0x90,
    ];
    arena.write_at(0, &data);
    let seed = [MemImage {
        base: ARENA_VA,
        bytes: data.to_vec(),
    }];

    // <op> xmm0, [rax] ; ret   - modrm 00 selects [rax] with reg = xmm0.
    let progs: &[(&str, &[u8])] = &[
        ("pshufb", &[0x66, 0x0F, 0x38, 0x00, 0x00, 0xC3]),
        ("pmuludq", &[0x66, 0x0F, 0xF4, 0x00, 0xC3]),
        ("palignr", &[0x66, 0x0F, 0x3A, 0x0F, 0x00, 0x07, 0xC3]),
        ("packsswb", &[0x66, 0x0F, 0x63, 0x00, 0xC3]),
        ("pmovzxbw", &[0x66, 0x0F, 0x38, 0x30, 0x00, 0xC3]),
        ("pmovsxbq", &[0x66, 0x0F, 0x38, 0x22, 0x00, 0xC3]),
        ("pmovsxwd", &[0x66, 0x0F, 0x38, 0x23, 0x00, 0xC3]),
        ("pmovzxdq", &[0x66, 0x0F, 0x38, 0x35, 0x00, 0xC3]),
        ("pinsrb", &[0x66, 0x0F, 0x3A, 0x20, 0x00, 0x05, 0xC3]),
        ("ptest", &[0x66, 0x0F, 0x38, 0x17, 0x00, 0xC3]),
    ];
    for (name, code) in progs {
        let mut init = CpuState::zeroed();
        init.xmm[0] = [0x0706_0504_0302_0100, 0x8F0E_0D0C_0B0A_0908];
        init.gpr[gpr::RAX] = ARENA_VA;
        let diff =
            differential_unicorn(code, CODE_VA, &init, &seed, BUDGET, DiffOptions::default())
                .expect("clean halt");
        assert!(diff.is_empty(), "{name} (mem) diverged: {diff:?}");
    }
}
