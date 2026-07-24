//! Exercises the `testkit::oracle` differential abstraction end-to-end using
//! two in-process Microlift oracles (compiled at different optimization levels).
//! This is the same machinery an external emulator (Unicorn) will plug into via
//! the `CpuOracle` trait; here it validates the state conversion, the
//! differential driver, and the identity-mapped arena without needing an
//! external dependency.
//!
//! Run: `cargo test --features testkit --test oracle_tests`

#![cfg(feature = "testkit")]

use microlift::testkit::oracle::*;
use microlift::testkit::Opt;

const CODE_VA: u64 = 0x1000;

/// `mov rax, imm ; add rax, rcx ; ret` with rcx seeded - two oracles at
/// different opt levels must agree on the resulting state.
#[test]
fn microlift_vs_microlift_opt_levels_agree() {
    // mov rax, 100 ; add rax, rcx ; ret
    let code = [
        0x48, 0xC7, 0xC0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100
        0x48, 0x01, 0xC8, // add rax, rcx
        0xC3, // ret
    ];

    let mut init = CpuState::zeroed();
    init.gpr[gpr::RCX] = 0x2A; // rcx = 42

    let mut a = MicroliftOracle::new(Opt::None);
    let mut b = MicroliftOracle::new(Opt::SpeedAndSize);

    let diff = differential(
        &mut a,
        &mut b,
        &code,
        CODE_VA,
        &init,
        &[],
        1000,
        DiffOptions::default(),
    )
    .expect("both oracles should halt cleanly");

    assert!(diff.is_empty(), "opt levels disagreed: {diff:?}");

    // And the computed value is correct: 100 + 42 = 142.
    let out = a.run(&code, CODE_VA, &init, &[], 1000);
    assert!(out.is_halted());
    assert_eq!(out.state().gpr[gpr::RAX], 142);
}

/// A non-diffable instruction (`cpuid`) must surface as an exit, and the
/// differential driver must decline to compare (returns `None`).
#[test]
fn non_diffable_exit_is_not_compared() {
    // cpuid ; ret
    let code = [0x0F, 0xA2, 0xC3];
    let init = CpuState::zeroed();

    let mut a = MicroliftOracle::new(Opt::None);
    let out = a.run(&code, CODE_VA, &init, &[], 1000);
    assert!(!out.is_halted(), "cpuid should be a non-diffable exit");
    match out {
        OracleOutcome::Exit {
            reason: OracleExit::NonDiffable(r),
            ..
        } => assert_eq!(r, microlift::context::ExitReason::Cpuid),
        _ => panic!("expected a non-diffable cpuid exit"),
    }

    let mut b = MicroliftOracle::new(Opt::Speed);
    let diff = differential(
        &mut a,
        &mut b,
        &code,
        CODE_VA,
        &init,
        &[],
        1000,
        DiffOptions::default(),
    );
    assert!(diff.is_none(), "non-diffable runs must not be compared");
}

/// The identity-mapped arena places host memory at a fixed guest VA so a lifted
/// load reads exactly what was written there - the reconciliation primitive an
/// external emulator will mirror. Skipped gracefully if the fixed mapping is
/// unavailable on this host.
#[cfg(unix)]
#[test]
fn identity_arena_backs_a_guest_load() {
    const ARENA_VA: u64 = 0x1_2000_0000;
    const ARENA_LEN: usize = 0x1000;

    let Some(mut arena) = IdentityArena::map(ARENA_VA, ARENA_LEN) else {
        eprintln!("identity arena unavailable at {ARENA_VA:#x}; skipping");
        return;
    };
    assert_eq!(arena.base(), ARENA_VA);

    // Put a known 64-bit value at ARENA_VA.
    let sentinel: u64 = 0xDEAD_BEEF_CAFE_F00D;
    arena.write_at(0, &sentinel.to_le_bytes());

    // mov rax, [ARENA_VA] ; ret   (mov rax, moffs64 = 48 A1 <abs64>)
    let mut code = vec![0x48, 0xA1];
    code.extend_from_slice(&ARENA_VA.to_le_bytes());
    code.push(0xC3);

    let init = CpuState::zeroed();
    let mut jit = MicroliftOracle::new(Opt::None);
    let out = jit.run(&code, CODE_VA, &init, &[], 1000);
    assert!(out.is_halted());
    assert_eq!(
        out.state().gpr[gpr::RAX],
        sentinel,
        "guest load through the identity arena must observe the written value"
    );
}
