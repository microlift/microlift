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

use microlift::testkit::oracle::unicorn::{differential_unicorn, UnicornOracle};
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
            let diff = differential_unicorn(
                &code,
                CODE_VA,
                &init,
                &[],
                BUDGET,
                DiffOptions::default(),
            )
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
    assert!(diff.is_empty(), "engines diverged on memory roundtrip: {diff:?}");

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
                "{name} {dst:#018x?},{src:#018x?} diverged: {diff:?}"
            );
        }
    }
}
