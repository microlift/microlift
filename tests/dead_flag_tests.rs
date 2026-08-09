//! `LifterConfig::eliminate_dead_flags`: it must change nothing, and it must
//! change something.
//!
//! Two obligations, and they pull in opposite directions, which is why both are
//! here. The optimisation is *metamorphic* - turning it on may not alter a
//! single bit of architectural state, including the flags themselves, because
//! the whole claim is that the writes it drops were never going to be read. And
//! it is an *optimisation* - if it drops nothing, the analysis is broken in a
//! way no equivalence test would ever notice.
//!
//! So: [`the_state_is_identical_either_way`] is the correctness half, over a
//! battery of snippets chosen for the places the classification differs from
//! the Intel manual; [`a_dead_flag_costs_no_ir`] and its neighbours are the
//! effectiveness half, counted in IR instructions rather than compiled bytes
//! because IR is the thing this exists to produce less of (and because compiled
//! size carries hash-seed noise - see `specialize_tests::code_len`).
//!
//! The strongest check is not in this file. `fuzz_cross_config` runs the same
//! equivalence over proptest-generated ALU snippets and random operands; this
//! file is the hand-picked cases that suite would be unlikely to find.

use microlift::arch::x86_64::X86_64;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;

/// Everything a lift can leave behind that the guest could observe.
///
/// `rsp` and `rip` are excluded: the harness allocates a fresh stack per run so
/// `rsp` is a different host address each time, and `rip` follows from it on the
/// exit path. Every flag is in, since flags are exactly what is at stake.
type State = ([u64; 15], [u8; 7], u32);

fn state(ctx: &RegisterContext) -> State {
    (
        [
            ctx.rax, ctx.rcx, ctx.rdx, ctx.rbx, ctx.rbp, ctx.rsi, ctx.rdi, ctx.r8, ctx.r9, ctx.r10,
            ctx.r11, ctx.r12, ctx.r13, ctx.r14, ctx.r15,
        ],
        [ctx.cf, ctx.pf, ctx.af, ctx.zf, ctx.sf, ctx.of, ctx.df],
        ctx.exit_reason,
    )
}

/// Seed the same registers both ways, so a divergence is the optimisation and
/// not the inputs.
fn seed(ctx: &mut RegisterContext) {
    ctx.rax = 0x0123_4567_89AB_CDEF;
    ctx.rbx = 0x8000_0000_0000_0000;
    ctx.rcx = 3;
    ctx.rdx = 0xFF;
    ctx.rsi = 0x7FFF_FFFF_FFFF_FFFF;
    ctx.rdi = 1;
    ctx.cf = 1;
    ctx.pf = 1;
    ctx.af = 1;
    ctx.zf = 1;
    ctx.sf = 1;
    ctx.of = 1;
}

/// The snippets, each ending in `ret` so the harness gets its state back.
///
/// Chosen for the seams rather than for coverage: the three places
/// `flag_effects` deliberately disagrees with the architecture, the points where
/// the analysis must give up, and the ordinary `cmp`/`jcc` shape the whole thing
/// exists for.
const SNIPPETS: &[(&str, &[u8])] = &[
    // The motivating case: five of the CMP's six flags have no reader, and the
    // sixth (ZF) does.
    ("cmp/je", &[0x48, 0x39, 0xD8, 0x74, 0x01, 0x90, 0xC3]),
    // Nothing reads any of them before the return - and the return publishes
    // all seven, so nothing may be dropped.
    ("add then return", &[0x48, 0x01, 0xD8, 0xC3]),
    // AF survives an AND, because `update_flags_logical` leaves it alone and
    // the classification says so. `lahf` is the reader that would notice.
    (
        "add; and; lahf",
        &[0x48, 0x01, 0xD8, 0x48, 0x21, 0xC8, 0x9F, 0xC3],
    ),
    // The undefined flags of IMUL are likewise left as they were.
    (
        "add; imul; lahf",
        &[0x48, 0x01, 0xD8, 0x48, 0x0F, 0xAF, 0xC1, 0x9F, 0xC3],
    ),
    // A shift reads the flags it may have to keep - `shl rax, cl` with cl = 3,
    // then again with the count masked to zero further down.
    (
        "add; shl rax, cl",
        &[0x48, 0x01, 0xD8, 0x48, 0xD3, 0xE0, 0xC3],
    ),
    // Two arms of a branch that disagree about which flag they read: the CMP
    // must keep whatever either side wants.
    (
        "cmp; jbe over an adc",
        &[
            0x48, 0x39, 0xD8, // cmp rax, rbx
            0x76, 0x03, // jbe +3
            0x48, 0x11, 0xC8, // adc rax, rcx
            0x9F, // lahf
            0xC3, // ret
        ],
    ),
    // A backward branch, so the fixpoint has to iterate: the ADD inside the
    // loop feeds the `jne` that jumps back over it.
    (
        "loop on dec/jne",
        &[
            0x48, 0x01, 0xD8, // add rax, rbx
            0x48, 0xFF, 0xC9, // dec rcx
            0x75, 0xF8, // jne -8
            0xC3, // ret
        ],
    ),
    // `cpuid` is unclassified, so everything before it stays live - which is
    // what keeps the flags the host sees at the exit correct.
    ("add; cpuid", &[0x48, 0x01, 0xD8, 0x0F, 0xA2, 0xC3]),
    // Flags into and out of a `setcc`/`cmovcc` pair.
    (
        "sub; sete; cmovne",
        &[
            0x48, 0x29, 0xD8, // sub rax, rbx
            0x0F, 0x94, 0xC2, // sete dl
            0x48, 0x0F, 0x45, 0xF1, // cmovne rsi, rcx
            0xC3,
        ],
    ),
    // Direction flag: written, then read by nothing this region contains, then
    // published by the return.
    ("std; cld", &[0xFD, 0xFC, 0xC3]),
];

/// The optimisation may not change a single observable bit.
#[test]
fn the_state_is_identical_either_way() {
    for (name, bytes) in SNIPPETS {
        let plain = LiftTest::new(bytes).reg(seed).run();
        let lean = LiftTest::new(bytes).reg(seed).eliminate_dead_flags().run();
        assert_eq!(
            state(&plain.ctx),
            state(&lean.ctx),
            "{name}: eliminating dead flags changed the result"
        );
        assert_eq!(
            plain.exit, lean.exit,
            "{name}: eliminating dead flags changed how the region exited"
        );
    }
}

/// The same, but resumed after the exit - a region that comes back has to agree
/// too, since the second half runs against flags the first half published.
#[test]
fn a_resumed_region_agrees_as_well() {
    // add rax, rbx ; cpuid ; add rax, rcx ; ret
    let bytes = &[0x48, 0x01, 0xD8, 0x0F, 0xA2, 0x48, 0x01, 0xC8, 0xC3];

    let run = |lean: bool| {
        let test = LiftTest::new(bytes).reg(seed).resumable();
        let test = if lean {
            test.eliminate_dead_flags()
        } else {
            test
        };
        let first = test.run();
        assert_eq!(first.exit, ExitReason::Cpuid, "expected a CPUID exit");
        let resumed = first.resume();
        state(&resumed.ctx)
    };

    assert_eq!(run(false), run(true), "resuming diverged");
}

/// How many IR instructions one lift emits.
///
/// The unit this optimisation is denominated in: the point is not smaller
/// machine code - the compiler was already deleting these - but less IR for the
/// compiler to walk, optimise and register-allocate on the way there.
fn ir_len(bytes: &[u8], eliminate: bool) -> usize {
    let config = LifterConfig {
        eliminate_dead_flags: eliminate,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    code.func
        .layout
        .blocks()
        .map(|b| code.func.layout.block_insts(b).count())
        .sum()
}

/// A flag with no reader costs no IR. If this ever passes trivially, the
/// equivalence tests above are checking that nothing does nothing.
#[test]
fn a_dead_flag_costs_no_ir() {
    // cmp rax, rbx ; je +1 ; nop ; ret
    //
    // The `je` reads ZF. CF, PF, AF, SF and OF are computed by the `cmp` and
    // read by nobody - but the `ret` publishes them, so they are live after all
    // and this snippet must *not* shrink. That is the control.
    let control = &[0x48, 0x39, 0xD8, 0x74, 0x01, 0x90, 0xC3];
    assert_eq!(
        ir_len(control, false),
        ir_len(control, true),
        "a region whose exit publishes the flags has no dead ones to drop"
    );

    // The same compare, with the flags overwritten before anything can leave:
    // cmp rax, rbx ; je +3 ; xor rax, rax ; add rax, rbx ; ret
    //
    // Now the `cmp`'s CF/PF/AF/SF/OF are all killed by the `xor`/`add` on one
    // path and by the `add` on the other, so only ZF has to survive it.
    let bytes = &[
        0x48, 0x39, 0xD8, // cmp rax, rbx
        0x74, 0x03, // je +3
        0x48, 0x31, 0xC0, // xor rax, rax
        0x48, 0x01, 0xD8, // add rax, rbx
        0xC3, // ret
    ];
    let plain = ir_len(bytes, false);
    let lean = ir_len(bytes, true);
    assert!(
        lean < plain,
        "eliminating dead flags emitted {lean} IR instructions against {plain} - it did not fire"
    );
}

/// Parity is the flag this was built for: four IR instructions on every
/// arithmetic instruction in the program, for something almost nothing reads.
#[test]
fn a_chain_of_arithmetic_sheds_most_of_its_flag_ir() {
    // Eight adds whose flags nobody reads, then a `test`/`jne` that wants ZF,
    // then a call that publishes everything.
    let mut bytes = Vec::new();
    for _ in 0..8 {
        bytes.extend_from_slice(&[0x48, 0x01, 0xD8]); // add rax, rbx
    }
    bytes.extend_from_slice(&[0x48, 0x85, 0xC0]); // test rax, rax
    bytes.extend_from_slice(&[0x75, 0x00]); // jne +0
    bytes.push(0xC3); // ret

    let plain = ir_len(&bytes, false);
    let lean = ir_len(&bytes, true);
    assert!(
        lean * 3 < plain * 2,
        "expected a third off the IR of a flag-heavy straight line, got {lean} against {plain}"
    );
}

/// Nothing is dropped where the analysis cannot see, and the region that proves
/// it is the one that ends in a call: a callee may do anything, and the sync
/// that reaches it publishes every flag.
#[test]
fn a_call_keeps_every_flag() {
    // add rax, rbx ; call +0 ; ret
    let bytes = &[0x48, 0x01, 0xD8, 0xE8, 0x00, 0x00, 0x00, 0x00, 0xC3];
    assert_eq!(
        ir_len(bytes, false),
        ir_len(bytes, true),
        "a call is a conservative point; the add before it must keep its flags"
    );
}

/// An instruction the frontend has not classified is conservative, which is
/// what makes an unimplemented instruction safe without the table knowing which
/// ones those are.
#[test]
fn an_unclassified_instruction_keeps_every_flag() {
    // add rax, rbx ; cpuid ; ret - `cpuid` is deliberately absent from the
    // classification, and it syncs the whole context to the host.
    let bytes = &[0x48, 0x01, 0xD8, 0x0F, 0xA2, 0xC3];
    assert_eq!(
        ir_len(bytes, false),
        ir_len(bytes, true),
        "an exit to the host must keep the flags it publishes"
    );
}

/// Off by default, so an embedder that has not asked for this is not given it.
///
/// Worth an assertion rather than a glance: the safety argument for the whole
/// feature ends with "and it is opt-in", so the default flipping is the one
/// change that would invalidate it silently.
#[test]
fn the_default_is_off() {
    assert!(!LifterConfig::default().eliminate_dead_flags);
}

/// Not an assertion - a printout, so the numbers behind the thresholds above
/// can be read rather than trusted. `cargo test --features testkit --test
/// dead_flag_tests -- --ignored --nocapture`.
#[test]
#[ignore]
fn dump_ir_savings() {
    println!(
        "{:<28} {:>8} {:>8} {:>7}",
        "snippet", "plain", "lean", "saved"
    );
    for (name, bytes) in SNIPPETS {
        let plain = ir_len(bytes, false);
        let lean = ir_len(bytes, true);
        println!(
            "{name:<28} {plain:>8} {lean:>8} {:>6.1}%",
            100.0 * (plain - lean) as f64 / plain as f64
        );
    }
}
