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
        ("addsubps", &[0xF2, 0x0F, 0xD0, 0xC1, 0xC3]),
        ("addsubpd", &[0x66, 0x0F, 0xD0, 0xC1, 0xC3]),
        ("haddps", &[0xF2, 0x0F, 0x7C, 0xC1, 0xC3]),
        ("hsubps", &[0xF2, 0x0F, 0x7D, 0xC1, 0xC3]),
        ("haddpd", &[0x66, 0x0F, 0x7C, 0xC1, 0xC3]),
        ("hsubpd", &[0x66, 0x0F, 0x7D, 0xC1, 0xC3]),
        ("movsldup", &[0xF3, 0x0F, 0x12, 0xC1, 0xC3]),
        ("movshdup", &[0xF3, 0x0F, 0x16, 0xC1, 0xC3]),
        ("movddup", &[0xF2, 0x0F, 0x12, 0xC1, 0xC3]),
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

/// Operands for the ADDSUB differential, chosen so the two lane kinds are
/// forced apart.
///
/// The point of every entry is that an even lane and an odd lane must disagree
/// on it: `inf - inf` invalidates where `inf + inf` does not, and the two swap
/// over when the source sign flips. A handler running one operation across all
/// four lanes passes every finite test and fails these. The signed zeros do the
/// same job with no NaN involved at all - `-0.0 - +0.0` is `-0.0` while
/// `-0.0 + +0.0` is `+0.0`.
///
/// NaNs appear on one side only, never both: with a NaN in each operand the
/// oracle itself is unreliable, since Unicorn does not reproduce x86's
/// "destination NaN wins" rule (`MULPS` shows the same divergence with no
/// ADDSUB involved). See the caveat on [`FLOAT_OPERANDS`].
const ADDSUB_OPERANDS: &[([u64; 2], [u64; 2])] = &[
    // f64 +inf against +inf then -inf: lane 0 subtracts, lane 1 adds.
    (
        [0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000],
        [0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000],
    ),
    // the mirror: the invalid operation lands in the other lane.
    (
        [0xFFF0_0000_0000_0000, 0x7FF0_0000_0000_0000],
        [0xFFF0_0000_0000_0000, 0x7FF0_0000_0000_0000],
    ),
    // f32 lanes +inf,+inf,+inf,-inf against +inf,-inf,-inf,-inf.
    (
        [0x7F80_0000_7F80_0000, 0xFF80_0000_7F80_0000],
        [0xFF80_0000_7F80_0000, 0xFF80_0000_FF80_0000],
    ),
    // signed zeros, both orders - no NaN, pure sign-of-zero.
    (
        [0x8000_0000_0000_0000, 0x0000_0000_0000_0000],
        [0x0000_0000_0000_0000, 0x8000_0000_0000_0000],
    ),
    (
        [0x8000_0000_8000_0000, 0x0000_0000_0000_0000],
        [0x0000_0000_8000_0000, 0x8000_0000_0000_0000],
    ),
    // QNaN destination against finite sources: sign and payload propagate.
    (
        [0x7FF8_0000_DEAD_BEEF, 0xFFF8_0000_CAFE_BABE],
        [0x4000_0000_4000_0000, 0x3F80_0000_3F80_0000],
    ),
    // QNaN source against a finite destination, the direction that would break
    // under a "negate the source and add" implementation.
    (
        [0x3FF0_0000_0000_0000, 0x3F80_0000_3F80_0000],
        [0xFFF8_0000_1234_5678, 0x7FF8_0000_9ABC_DEF0],
    ),
    // SNaNs, which the instruction quiets rather than replaces.
    (
        [0x3FF0_0000_0000_0000, 0x3F80_0000_3F80_0000],
        [0x7FF4_0000_0000_0001, 0xFFA0_0000_FFA0_0000],
    ),
    (
        [0x7FF4_0000_0000_0001, 0xFFA0_0000_FFA0_0000],
        [0x3FF0_0000_0000_0000, 0x3F80_0000_3F80_0000],
    ),
    // ordinary finite lanes, so the common path is covered here too.
    (
        [0x4020_0000_3FC0_0000, 0x4080_0000_C040_0000],
        [0xBF80_0000_3F00_0000, 0x4100_0000_4000_0000],
    ),
    (
        [0x0000_0000_0000_0001, 0x7FEF_FFFF_FFFF_FFFF],
        [0x4100_0000_4080_0000, 0x3FC0_0000_4020_0000],
    ),
];

/// ADDSUBPS / ADDSUBPD against hardware over [`ADDSUB_OPERANDS`].
#[test]
fn sse3_addsub_agrees() {
    let ops: &[(&str, &[u8])] = &[
        ("addsubps", &[0xF2, 0x0F, 0xD0, 0xC1, 0xC3]),
        ("addsubpd", &[0x66, 0x0F, 0xD0, 0xC1, 0xC3]),
    ];
    for (name, code) in ops {
        for &(dst, src) in ADDSUB_OPERANDS {
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

/// The sequence a compiler emits around ADDSUBPS: a complex multiply built
/// from MOVSLDUP / MOVSHDUP / SHUFPS / MULPS / ADDSUBPS, checked against
/// hardware as a whole rather than one instruction at a time.
///
/// Operands are finite: the MULPS pair ahead of the ADDSUBPS can put a NaN
/// into *both* of its operands, which is the one case the oracle cannot
/// adjudicate (see the caveat on [`ADDSUB_OPERANDS`]). The NaN and infinity
/// behaviour of ADDSUBPS itself is pinned down by
/// [`sse3_addsub_agrees`] instead.
#[test]
fn complex_multiply_sequence_agrees() {
    let code = [
        0xF3, 0x0F, 0x12, 0xD1, // movsldup xmm2, xmm1
        0xF3, 0x0F, 0x16, 0xD9, // movshdup xmm3, xmm1
        0x0F, 0x59, 0xD0, // mulps xmm2, xmm0
        0x0F, 0xC6, 0xC0, 0xB1, // shufps xmm0, xmm0, 0xB1
        0x0F, 0x59, 0xD8, // mulps xmm3, xmm0
        0xF2, 0x0F, 0xD0, 0xD3, // addsubps xmm2, xmm3
        0xC3, // ret
    ];
    // (a + bi) * (c + di), four complex pairs per run: signed zeros, a
    // denormal, and magnitudes whose product both underflows and overflows.
    let pairs: &[[f32; 4]] = &[
        [1.0, 2.0, 3.0, 4.0],
        [-1.5, 0.25, 2.0, -8.0],
        [0.0, -0.0, -0.0, 0.0],
        [f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.5, 0.5],
        [f32::MAX, f32::MAX, 2.0, -2.0],
        [1e-30, 1e30, 1e30, 1e-30],
    ];
    for lanes in pairs {
        let pack = |a: f32, b: f32| u64::from(a.to_bits()) | (u64::from(b.to_bits()) << 32);
        let mut init = CpuState::zeroed();
        init.xmm[0] = [pack(lanes[0], lanes[1]), pack(lanes[1], lanes[0])];
        init.xmm[1] = [pack(lanes[2], lanes[3]), pack(lanes[3], lanes[2])];
        let diff = differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
            .expect("clean halt");
        assert!(
            diff.is_empty(),
            "complex multiply {lanes:?} diverged: {diff:?}"
        );
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
/// fresh default NaN (`0/0`, `inf-inf`, `0*inf`).
///
/// Generated default NaNs are excluded because x86 and AArch64 disagree on
/// their sign - x86's integer/floating indefinite is `0xFFF8_0000_0000_0000`,
/// the AArch64 default NaN is `0x7FF8_0000_0000_0000` - and Microlift lowers
/// these ops to Cranelift `fdiv`/`fsub`, so the *host* FPU picks the sign.
/// That divergence is general, not a property of any one instruction:
/// `DIVSD`, `SUBPS` and the rest show it identically.
///
/// `sqrt(-x)` used to be on that list and no longer is: the SQRT handlers
/// materialise x86's indefinite themselves rather than deferring to the host,
/// so [`sse_sqrt_negative_agrees`] covers negative operands directly. The same
/// treatment would lift the restriction for the others.
///
/// *Propagated* input NaNs keep their exact bits and are covered here - the NaN
/// destinations below flow through every op.
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

/// Sources for the SQRT differential, centred on the negative inputs whose
/// result is a *generated* NaN.
///
/// `sqrt(-x)` is the one case where the general default-NaN caveat on
/// [`FLOAT_OPERANDS`] would otherwise bite: on an AArch64 host the FPU's
/// default NaN is `0x7FF8_..`, while x86 produces the indefinite `0xFFF8_..`.
/// The SQRT handlers therefore materialise x86's indefinite themselves instead
/// of letting the host FPU pick, and these operands are the proof - every lane
/// is exercised as both an `f32` quadruple and an `f64` pair.
const SQRT_OPERANDS: &[[u64; 2]] = &[
    // negative normals: -1.0 / -2.0 as f64, and -1.0 / -2.0 / -0.5 / -4.0 f32
    [0xBFF0_0000_0000_0000, 0xC000_0000_0000_0000],
    [0xC000_0000_BF80_0000, 0xC080_0000_BF00_0000],
    // negative zero (result is -0.0, not a NaN) beside a negative normal
    [0x8000_0000_0000_0000, 0xBF80_0000_8000_0000],
    // negative infinity, and negative denormals
    [0xFFF0_0000_0000_0000, 0x8000_0001_8000_0001],
    // NaNs: propagated unchanged, sign and payload intact, never replaced
    [0x7FF8_0000_DEAD_BEEF, 0xFFF8_0000_CAFE_BABE],
    // signalling NaNs, which the instruction quiets rather than indefinites
    [0x7FF4_0000_0000_0001, 0xFFA0_0000_FFA0_0000],
    // positive values alongside, so the non-NaN path stays covered here too
    [0x4000_0000_0000_0000, 0x4080_0000_3F80_0000],
    // mixed sign per f32 lane: only the negative lanes may become indefinite
    [0x3F80_0000_BF80_0000, 0xBF00_0000_4000_0000],
];

/// SQRTPS/SQRTPD/SQRTSS/SQRTSD against negative inputs.
///
/// Split out from [`sse_float_arith_agrees`] for the same reason
/// [`sse_minmax_selection_agrees`] is: it needs operands that suite
/// deliberately excludes. See [`SQRT_OPERANDS`].
#[test]
fn sse_sqrt_negative_agrees() {
    let ops: &[(&str, &[u8])] = &[
        ("sqrtps", &[0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtpd", &[0x66, 0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtss", &[0xF3, 0x0F, 0x51, 0xC1, 0xC3]),
        ("sqrtsd", &[0xF2, 0x0F, 0x51, 0xC1, 0xC3]),
    ];
    for (name, code) in ops {
        for &src in SQRT_OPERANDS {
            // The destination is a distinct pattern so a handler that wrote the
            // wrong lane, or failed to preserve the scalar forms' upper bits,
            // shows up as a divergence rather than a coincidental match.
            let mut init = CpuState::zeroed();
            init.xmm[0] = [0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321];
            init.xmm[1] = src;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} {src:#018x?} diverged: {diff:?}");
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

/// EXTRACTPS against hardware: every lane selector (including the imm8 bits
/// above `[1:0]` that hardware ignores), for both the 32-bit and the `REX.W`
/// destination spelling.
#[test]
fn extractps_agrees() {
    for imm in 0..8u8 {
        let progs: &[(&str, &[u8])] = &[
            // extractps eax, xmm1, imm8 ; ret
            ("extractps r32", &[0x66, 0x0F, 0x3A, 0x17, 0xC8, imm, 0xC3]),
            // extractps rax, xmm1, imm8 ; ret
            (
                "extractps r64",
                &[0x66, 0x48, 0x0F, 0x3A, 0x17, 0xC8, imm, 0xC3],
            ),
        ];
        for (name, code) in progs {
            let mut init = CpuState::zeroed();
            init.xmm[1] = [0x2222_2222_1111_1111, 0x4444_4444_3333_3333];
            // A dirty destination catches a merge where hardware zero-extends.
            init.gpr[gpr::RAX] = 0xDEAD_BEEF_DEAD_BEEF;
            let diff =
                differential_unicorn(code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(diff.is_empty(), "{name} imm={imm} diverged: {diff:?}");
        }
    }
}

/// Dot-product destinations at *single* precision: signed zeros, infinities,
/// denormals and near-overflow magnitudes, and no NaN in any dword.
///
/// [`FLOAT_OPERANDS`] cannot be reused here. It keeps the source NaN-free but
/// lets the destination hold NaNs, which is safe for a lane-wise op - only one
/// operand of each addition can then be a NaN. `DPPS` reduces *across* lanes,
/// so two NaN destination lanes meet each other in the summation tree, and the
/// two-QNaN case is a standing disagreement between the engines: x86 returns
/// the first (destination) operand, which is what Microlift gets by deferring
/// to the host, while Unicorn's older softfloat applies the x87 larger-
/// significand rule. Plain `ADDPS` with a QNaN in both operands shows the same
/// divergence, so it is not a property of these instructions.
///
/// Propagation of a *single* NaN is covered by
/// [`dot_product_nan_propagation_agrees`]; the infinities and zeros below still
/// exercise the *generated* NaNs (`inf - inf`, `0 * inf`), where Microlift
/// substitutes x86's floating indefinite rather than the host's default NaN.
const DPPS_OPERANDS: &[([u64; 2], [u64; 2])] = &[
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
    // +/-inf lanes: with two enabled multiplies of opposite sign the tree hits
    // `inf + -inf`, an invalid operation that *generates* a NaN.
    (
        [0x7F80_0000_FF80_0000, 0xFF80_0000_7F80_0000],
        [0x3F80_0000_3F80_0000, 0x4000_0000_4000_0000],
    ),
    // an infinity against a +0.0 source lane: `0 * inf`, the other invalid
    // operation reachable here.
    (
        [0x0000_0000_7F80_0000, 0x7F80_0000_0000_0000],
        [0x3F80_0000_0000_0000, 0x0000_0000_3F80_0000],
    ),
    // denormals and near-overflow magnitudes, so the sum can round to infinity
    (
        [0x0000_0001_8000_0001, 0x7F7F_FFFF_FF7F_FFFF],
        [0x0080_0000_4000_0000, 0x4000_0000_3F80_0000],
    ),
];

/// The double-precision companion of [`DPPS_OPERANDS`], NaN-free per quadword.
const DPPD_OPERANDS: &[([u64; 2], [u64; 2])] = &[
    (
        [0x3FF0_0000_0000_0000, 0x4000_0000_0000_0000],
        [0x4000_0000_0000_0000, 0x3FF0_0000_0000_0000],
    ),
    // signed zeros
    (
        [0x8000_0000_0000_0000, 0x0000_0000_0000_0000],
        [0x3FF0_0000_0000_0000, 0x4009_21FB_5444_2D18],
    ),
    // +inf and -inf: `inf + -inf` when both multiplies are enabled
    (
        [0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000],
        [0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000],
    ),
    // an infinity against a +0.0 source lane: `0 * inf`
    (
        [0x7FF0_0000_0000_0000, 0x3FF0_0000_0000_0000],
        [0x0000_0000_0000_0000, 0x4000_0000_0000_0000],
    ),
    // denormal and near-overflow magnitudes
    (
        [0x0000_0000_0000_0001, 0x7FEF_FFFF_FFFF_FFFF],
        [0x4000_0000_0000_0000, 0x3FF0_0000_0000_0000],
    ),
];

/// DPPS / DPPD over every `imm8`, on operands whose summation is **exact**.
///
/// The exactness is a hard requirement, not a convenience. Unicorn 2.x carries
/// a pre-fix QEMU `helper_dpps`/`helper_dppd` that folds the products into a
/// running accumulator, where the SDM (and current QEMU) specify the pairwise
/// tree `(t0 + t1) + (t2 + t3)`. The two agree only when no intermediate
/// rounds, so every value in the tables above is a small power-of-two multiple,
/// an infinity, or a magnitude that overflows or underflows identically either
/// way. Adding an operand that rounds turns this test into an assertion about
/// the oracle's bug; `test_dpps_sums_as_a_pairwise_tree_not_left_to_right` in
/// `tests/sse_tests.rs` pins the order against the SDM instead.
///
/// What the oracle *does* prove here is everything else, and the full 256-value
/// sweep is why: the two mask nibbles interact, a masked-off multiply
/// substituting `+0.0` is the only thing keeping an infinity in a disabled lane
/// out of the sum, and which `imm8` values reach an `inf + -inf` - so a
/// *generated* NaN, whose sign is x86's floating indefinite and not the host
/// FPU's default - depends on the mask too.
#[test]
fn dot_product_agrees() {
    for (name, opcode, operands) in [
        ("dpps", 0x40u8, DPPS_OPERANDS),
        ("dppd", 0x41u8, DPPD_OPERANDS),
    ] {
        for imm in 0..=255u8 {
            // dpps/dppd xmm0, xmm1, imm8 ; ret
            let code = [0x66, 0x0F, 0x3A, opcode, 0xC1, imm, 0xC3];
            for &(dst, src) in operands {
                let mut init = CpuState::zeroed();
                init.xmm[0] = dst;
                init.xmm[1] = src;
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
                    "{name} imm={imm:#04x} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
                );
            }
        }
    }
}

/// Dot-product NaN *propagation*, with exactly one NaN in play.
///
/// One multiply is enabled and its destination lane holds a QNaN, so every
/// addition in the tree sees at most one NaN operand and the result must be
/// that NaN, bits intact. This is the coverage
/// [`DPPS_OPERANDS`] / [`DPPD_OPERANDS`] deliberately give up to stay clear of
/// the two-QNaN disagreement.
#[test]
fn dot_product_nan_propagation_agrees() {
    // Sign, payload and signalling-ness all have to survive.
    const NANS: [u64; 3] = [0x7FC0_0000, 0xFFC0_1234, 0x7F80_0001];

    for lane in 0..4u32 {
        for nan in NANS {
            // Enable only `lane`'s multiply; broadcast to all four lanes.
            let imm = ((1u8 << lane) << 4) | 0x0F;
            let code = [0x66, 0x0F, 0x3A, 0x40, 0xC1, imm, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = [0x4000_0000_3F80_0000, 0x4080_0000_4040_0000];
            init.xmm[0][(lane / 2) as usize] &= !(0xFFFF_FFFFu64 << ((lane % 2) * 32));
            init.xmm[0][(lane / 2) as usize] |= nan << ((lane % 2) * 32);
            init.xmm[1] = [0x4000_0000_4000_0000, 0x4000_0000_4000_0000];
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "dpps nan={nan:#010x} lane={lane} diverged: {diff:?}"
            );
        }
    }

    for lane in 0..2u32 {
        for nan in [
            0x7FF8_0000_0000_0000u64,
            0xFFF8_0000_0000_1234,
            0x7FF0_0000_0000_0001,
        ] {
            let imm = ((1u8 << lane) << 4) | 0x03;
            let code = [0x66, 0x0F, 0x3A, 0x41, 0xC1, imm, 0xC3];
            let mut init = CpuState::zeroed();
            init.xmm[0] = [0x3FF0_0000_0000_0000, 0x4000_0000_0000_0000];
            init.xmm[0][lane as usize] = nan;
            init.xmm[1] = [0x4000_0000_0000_0000, 0x4000_0000_0000_0000];
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "dppd nan={nan:#018x} lane={lane} diverged: {diff:?}"
            );
        }
    }
}

/// PHMINPOSUW against hardware. The unsigned comparison and the lowest-index
/// tie rule are both invisible in the common case, so this leans on
/// [`PACKED_OPERANDS`], which contains all-equal vectors and words whose high
/// bit is set.
#[test]
fn phminposuw_agrees() {
    // phminposuw xmm0, xmm1 ; ret
    agree_packed("phminposuw", &[0x66, 0x0F, 0x38, 0x41, 0xC1, 0xC3]);
}

/// MPSADBW over every block/window selector `imm8` encodes, plus the bits above
/// `[2:0]` that hardware ignores.
#[test]
fn mpsadbw_agrees() {
    for imm in 0..16u8 {
        // mpsadbw xmm0, xmm1, imm8 ; ret
        let code = [0x66, 0x0F, 0x3A, 0x42, 0xC1, imm, 0xC3];
        for &(dst, src) in PACKED_OPERANDS {
            let mut init = CpuState::zeroed();
            init.xmm[0] = dst;
            init.xmm[1] = src;
            let diff =
                differential_unicorn(&code, CODE_VA, &init, &[], BUDGET, DiffOptions::default())
                    .expect("clean halt");
            assert!(
                diff.is_empty(),
                "mpsadbw imm={imm:#04x} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}

/// `Spec::IntegerOverflowFlags` against real hardware.
///
/// The half `specialize_tests` cannot prove. That file shows the specialized
/// CF and OF match microlift's own portable formulas; this shows the formulas
/// match the CPU - which is what would catch both microlift paths being wrong
/// the same way, and CF/OF are exactly the pair where that is plausible, since
/// the unsigned and signed overflow conditions are easy to conflate and
/// `usub_overflow`'s operand order is easy to get backwards.
///
/// The operands straddle both boundaries at every width. The `seto`/`setb`
/// after each instruction is not decoration: without a reader the liveness pass
/// reports both flags dead and the specialization correctly declines, leaving
/// this measuring the portable path twice.
#[test]
fn integer_overflow_flags_agree_with_unicorn() {
    use microlift::testkit::oracle::unicorn::differential_unicorn_spec;

    // seto dl ; setb bl - readers that keep CF and OF live.
    const READ: &[u8] = &[0x0F, 0x90, 0xC2, 0x0F, 0x92, 0xC3];
    // xor esi, esi - a fully-defined flag write, appended after `READ` for the
    // instructions whose ZF/SF/AF/PF the architecture leaves *undefined*.
    //
    // `IMUL`'s truncating forms are the only ones here. Microlift leaves those
    // four as they were (which is what `flag_effects` classifies, and what
    // `dead_flag_tests` pins down); real silicon sets them from the product.
    // Neither is wrong, and neither is comparable - so this overwrites all five
    // defined flags on both engines and leaves the two this test is about in DL
    // and BL, where the GPR comparison sees them.
    const NORMALIZE: &[u8] = &[0x31, 0xF6];
    let cases: &[(&str, &[u8])] = &[
        ("add r64", &[0x48, 0x01, 0xC8]),
        ("add r32", &[0x01, 0xC8]),
        ("add r16", &[0x66, 0x01, 0xC8]),
        ("add r8", &[0x00, 0xC8]),
        ("sub r64", &[0x48, 0x29, 0xC8]),
        ("sub r32", &[0x29, 0xC8]),
        ("sub r16", &[0x66, 0x29, 0xC8]),
        ("sub r8", &[0x28, 0xC8]),
        ("cmp r64", &[0x48, 0x39, 0xC8]),
        ("adc r64", &[0x48, 0x11, 0xC8]),
        ("adc r8", &[0x10, 0xC8]),
        ("sbb r64", &[0x48, 0x19, 0xC8]),
        ("sbb r8", &[0x18, 0xC8]),
        ("inc r64", &[0x48, 0xFF, 0xC0]),
        ("dec r64", &[0x48, 0xFF, 0xC8]),
        ("neg r64", &[0x48, 0xF7, 0xD8]),
        ("neg r8", &[0xF6, 0xD8]),
        ("xadd r64", &[0x48, 0x0F, 0xC1, 0xC8]),
        ("imul r64,r64", &[0x48, 0x0F, 0xAF, 0xC1]),
        ("imul r32,r32,imm", &[0x69, 0xC1, 0x34, 0x12, 0x00, 0x00]),
    ];

    // (rax, rcx, carry in)
    let operands: &[(u64, u64, bool)] = &[
        (0, 0, false),
        (0, 1, false),
        (1, 0, false),
        (0, 0, true),
        (u64::MAX, 1, false),
        (u64::MAX, 1, true),
        (u64::MAX, u64::MAX, true),
        (0x7FFF_FFFF_FFFF_FFFF, 1, false),
        (0x7FFF_FFFF_FFFF_FFFF, 1, true),
        (0x8000_0000_0000_0000, 1, false),
        (0x8000_0000_0000_0000, u64::MAX, false),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000, false),
        (0x7F, 1, false),
        (0x80, 0xFF, false),
        (0xFF, 1, true),
        (0x7FFF, 1, false),
        (0x8000, 0xFFFF, false),
        (0x7FFF_FFFF, 1, false),
        (0xFFFF_FFFF, 1, false),
        (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, true),
    ];

    for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
        let spec = Specialization::for_target(target);
        for (name, insn) in cases {
            let mut code = insn.to_vec();
            code.extend_from_slice(READ);
            if name.starts_with("imul") {
                code.extend_from_slice(NORMALIZE);
            }
            code.push(0xC3);
            for &(rax, rcx, cf) in operands {
                let mut init = CpuState::zeroed();
                init.gpr[gpr::RAX] = rax;
                init.gpr[gpr::RCX] = rcx;
                if cf {
                    init.rflags |= 1;
                }
                let diff = differential_unicorn_spec(&code, CODE_VA, &init, BUDGET, spec)
                    .expect("clean halt");
                assert!(
                    diff.is_empty(),
                    "{name} under {} with rax={rax:#x} rcx={rcx:#x} cf={cf} diverged: {diff:?}",
                    target.name()
                );
            }
        }
    }
}

/// Every target-specialized SSE lowering, against real hardware semantics.
///
/// `tests/specialize_tests.rs` proves specialized == microlift's portable path;
/// this proves the specialized path == Unicorn. The two together are what make
/// a specialization trustworthy - the first alone would happily certify a
/// specialization that faithfully reproduced a bug in the portable expansion.
///
/// Both [`SpecTarget`]s are exercised on whatever host this runs on. That is
/// deliberate: the JIT compiles for the host either way, so this checks the IR
/// each target's path *emits*, not the machine it was tuned for. Divergence
/// here is a semantic error in the sequence, which is host-independent.
#[test]
fn specialized_lowerings_agree_with_unicorn() {
    use microlift::testkit::oracle::unicorn::differential_unicorn_spec;

    // (name, encoding without the trailing `ret`)
    let cases: &[(&str, &[u8])] = &[
        ("haddps", &[0xF2, 0x0F, 0x7C, 0xC1]),
        ("hsubps", &[0xF2, 0x0F, 0x7D, 0xC1]),
        ("haddpd", &[0x66, 0x0F, 0x7C, 0xC1]),
        ("hsubpd", &[0x66, 0x0F, 0x7D, 0xC1]),
        ("phaddw", &[0x66, 0x0F, 0x38, 0x01, 0xC1]),
        ("phaddd", &[0x66, 0x0F, 0x38, 0x02, 0xC1]),
        ("phaddsw", &[0x66, 0x0F, 0x38, 0x03, 0xC1]),
        ("phsubw", &[0x66, 0x0F, 0x38, 0x05, 0xC1]),
        ("phsubd", &[0x66, 0x0F, 0x38, 0x06, 0xC1]),
        ("phsubsw", &[0x66, 0x0F, 0x38, 0x07, 0xC1]),
        ("pmaddwd", &[0x66, 0x0F, 0xF5, 0xC1]),
        ("pmaddubsw", &[0x66, 0x0F, 0x38, 0x04, 0xC1]),
        ("psadbw", &[0x66, 0x0F, 0xF6, 0xC1]),
        ("pmuludq", &[0x66, 0x0F, 0xF4, 0xC1]),
        ("pmuldq", &[0x66, 0x0F, 0x38, 0x28, 0xC1]),
        ("pmulhrsw", &[0x66, 0x0F, 0x38, 0x0B, 0xC1]),
    ];

    let mut all: Vec<(String, Vec<u8>)> = cases
        .iter()
        .map(|(n, b)| {
            let mut v = b.to_vec();
            v.push(0xC3);
            ((*n).to_string(), v)
        })
        .collect();

    // ROUNDPS / ROUNDPD sweep every rounding mode, since the imm8 selects the
    // Cranelift opcode at lift time.
    for imm in 0..8u8 {
        all.push((
            format!("roundps-{imm}"),
            vec![0x66, 0x0F, 0x3A, 0x08, 0xC1, imm, 0xC3],
        ));
        all.push((
            format!("roundpd-{imm}"),
            vec![0x66, 0x0F, 0x3A, 0x09, 0xC1, imm, 0xC3],
        ));
    }

    for target in [SpecTarget::X86_64, SpecTarget::Aarch64] {
        let spec = Specialization::for_target(target);
        for (name, code) in &all {
            for &(dst, src) in PACKED_OPERANDS {
                let mut init = CpuState::zeroed();
                init.xmm[0] = dst;
                init.xmm[1] = src;
                let diff = differential_unicorn_spec(code, CODE_VA, &init, BUDGET, spec)
                    .expect("clean halt");
                assert!(
                    diff.is_empty(),
                    "{name} under {} {dst:#018x?},{src:#018x?} diverged: {diff:?}",
                    target.name()
                );
            }
        }
    }
}
