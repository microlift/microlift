//! The call-return ring: predicting where a guest `RET` goes.
//!
//! A port of FEX's call-return shadow stack, with a link cell where FEX keeps a
//! raw host PC - see [`microlift::core::callret`] for why Cranelift forces that
//! substitution and what it costs.
//!
//! Two properties carry this file, and neither is a happy path.
//! [`a_return_is_predicted_with_no_table_at_all`] removes every other mechanism,
//! so a `RET` that lands must have landed through the ring; and
//! [`an_unbalanced_guest_gets_correct_answers_with_a_worse_hit_rate`] is the
//! property the whole no-guard-page argument rests on.

use std::sync::{Arc, RwLock};

use microlift::core::callret::CallRetConfig;
use microlift::core::exit::ExitReason;
use microlift::core::link::{BlockLinkConfig, LinkSiteKind};
use microlift::core::{LifterConfig, lift_to_code};
use microlift::arch::x86_64::X86_64;
use microlift::runtime::{CodeCache, Step, ThreadRuntime, Translation};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

/// Where the callee lives.
const CALLEE: u64 = 0x2000;
/// The address the call comes back to - `DEFAULT_LIFT_BASE` plus the `call`'s
/// five bytes.
const RETURN_TO: u64 = DEFAULT_LIFT_BASE + 5;
const ROWS: u32 = 256;
const FRAMES: u32 = 4;

/// `call 0x2000 ; mov rax, 1 ; ret`
#[rustfmt::skip]
const CALLER: &[u8] = &[
    0xE8, 0xFB, 0x0F, 0x00, 0x00,              // call 0x2000
    0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
    0xC3,                                      // ret
];

/// `mov rbx, 7 ; ret` - returns to [`RETURN_TO`].
const CALLEE_CODE: &[u8] = &[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3];

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

fn ring() -> CallRetConfig {
    CallRetConfig::new(FRAMES)
}

/// `ring` on, or only cells - the control for every behavioural case here.
fn config(with_ring: bool) -> impl Fn(&mut LifterConfig) {
    move |c: &mut LifterConfig| {
        c.block_link = Some(link_config());
        c.link_cells = true;
        c.call_return_stack = with_ring.then(ring);
    }
}

fn caller(with_ring: bool) -> Translation {
    LiftTest::new(CALLER)
        .with_config(config(with_ring))
        .translation()
        .expect("compiles")
}

fn callee(with_ring: bool) -> Translation {
    LiftTest::new(CALLEE_CODE)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(config(with_ring))
        .translation()
        .expect("compiles")
}

struct Guest {
    runtime: ThreadRuntime,
    _stack: Vec<u64>,
}

impl Guest {
    fn new(cache: &Arc<RwLock<CodeCache>>, with_ring: bool) -> Self {
        let stack = vec![0u64; 1024];
        let top = stack.as_ptr() as u64 + (stack.len() as u64) * 8;
        let mut runtime = ThreadRuntime::new(link_config(), Arc::clone(cache));
        if with_ring {
            runtime = runtime.with_call_return(ring());
        }
        // The guest `call` pushes below this, so leave it a word of headroom.
        runtime.ctx.rsp = top - 8;
        // SAFETY: inside `stack`, which this struct owns.
        unsafe { *(runtime.ctx.rsp as *mut u64) = EXIT_TRAMPOLINE };
        runtime.ctx.rip = DEFAULT_LIFT_BASE;
        Self {
            runtime,
            _stack: stack,
        }
    }
}

/// Both regions in the cache, the caller published, everything installed.
fn seeded(with_ring: bool) -> (Arc<RwLock<CodeCache>>, Guest) {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, with_ring);
    let caller = cache.write().unwrap().insert(caller(with_ring));
    cache.write().unwrap().insert(callee(with_ring));
    guest.runtime.publish(&caller);
    (cache, guest)
}

#[test]
fn a_call_gets_a_cell_for_the_address_it_returns_to() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        call_return_stack: Some(ring()),
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();

    let returns: Vec<_> = code
        .link_sites
        .iter()
        .filter(|s| s.kind == LinkSiteKind::CallReturn)
        .collect();
    assert_eq!(returns.len(), 1);
    assert_eq!(returns[0].target, RETURN_TO);

    // The whole design rests on this: a return address is always a *call resume
    // point*, so the region's entry dispatch can already be entered there and
    // the cell needs no finer granularity than a whole region.
    assert!(
        code.resume_points.contains(&RETURN_TO),
        "the return address must be enterable: {:x?}",
        code.resume_points
    );
}

#[test]
fn the_ring_being_off_emits_no_return_cell() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        call_return_stack: None,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(
        code.link_sites
            .iter()
            .all(|s| s.kind == LinkSiteKind::StaticEdge)
    );
}

#[test]
fn the_ring_needs_cells_to_have_anything_to_name() {
    // A hit hands back a cell, so the ring is inert without them rather than
    // half-working - the same silent-fallback shape as `pinned_abi_for`.
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: false,
        call_return_stack: Some(ring()),
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(code.link_sites.is_empty());
}

/// With `link_table` null, a `RET` - whose target is a runtime value and so has no static cell - can only be
/// resolved by the ring.
#[test]
fn a_return_is_predicted_with_no_table_at_all() {
    let (_cache, mut guest) = seeded(true);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee ran");
    assert_eq!(
        guest.runtime.ctx.rax, 1,
        "and its `ret` came back into the caller at the resume point"
    );
    assert_eq!(
        guest.runtime.stats().lookups,
        1,
        "one host dispatch, for the initial entry only"
    );
}

#[test]
fn without_the_ring_the_same_return_needs_the_host() {
    // The other half of the control: identical program, ring off. The callee's
    // `RET` has no cell and no table, so it must exit.
    let (_cache, mut guest) = seeded(false);
    guest.runtime.detach_table();

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee still ran");
    assert_eq!(
        guest.runtime.ctx.rax, 0,
        "but the caller was never re-entered"
    );
    assert_eq!(guest.runtime.ctx.rip, RETURN_TO);
}

#[test]
fn a_return_to_an_address_nobody_pushed_is_a_miss() {
    // **The inverse control.** A ring that took the top frame without checking
    // its `guest_pc` would re-enter the caller here, which is a wrong jump - the
    // exact failure the compare exists to prevent. The callee rewrites its own
    // return address, so the address it returns to and the frame the caller
    // pushed disagree.
    const ELSEWHERE: u64 = 0x3000;
    #[rustfmt::skip]
    let rogue: &[u8] = &[
        0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00,  // mov rbx, 7
        0x48, 0xC7, 0x04, 0x24, 0x00, 0x30, 0x00, 0x00,  // mov qword ptr [rsp], 0x3000
        0xC3,                                      // ret
    ];

    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, true);
    let caller = cache.write().unwrap().insert(caller(true));
    let rogue = LiftTest::new(rogue)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(config(true))
        .translation()
        .expect("compiles");
    cache.write().unwrap().insert(rogue);
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee ran");
    assert_eq!(
        guest.runtime.ctx.rax, 0,
        "and the caller was NOT re-entered: the frame did not match"
    );
    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rip, ELSEWHERE);

    // A declining pop must leave the ring alone - the frame it refused still
    // belongs to an outer call that has not returned yet.
    assert_eq!(
        guest.runtime.ctx.callret_off,
        (ring().bytes() - microlift::core::callret::FRAME_SIZE),
        "the push happened and the pop declined, so one frame is still live"
    );
}

/// **The property the no-guard-page argument rests on.** A guest that pushes
/// more frames than the ring holds must get correct answers, losing only hit
/// rate. Four frames, five nested calls.
#[test]
fn an_unbalanced_guest_gets_correct_answers_with_a_worse_hit_rate() {
    // Five regions, each calling the next: 0x1000 -> 0x2000 -> ... -> 0x6000.
    // The innermost returns, and the returns unwind back out through a ring that
    // is one frame too small to have remembered them all.
    const DEPTH: u64 = 5;
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let stack = vec![0u64; 1024];
    let top = stack.as_ptr() as u64 + (stack.len() as u64) * 8;
    let mut runtime = ThreadRuntime::new(link_config(), Arc::clone(&cache)).with_call_return(ring());
    runtime.ctx.rsp = top - 8;
    // SAFETY: inside `stack`, kept alive below.
    unsafe { *(runtime.ctx.rsp as *mut u64) = EXIT_TRAMPOLINE };
    runtime.ctx.rip = DEFAULT_LIFT_BASE;

    for i in 0..DEPTH {
        let at = if i == 0 {
            DEFAULT_LIFT_BASE
        } else {
            (i + 1) * 0x1000
        };
        let bytes: Vec<u8> = if i + 1 == DEPTH {
            // Innermost: `add rax, 1 ; ret`
            vec![0x48, 0x83, 0xC0, 0x01, 0xC3]
        } else {
            // `call next ; add rax, 1 ; ret`
            let next = (i + 2) * 0x1000;
            let disp = (next as i64 - (at as i64 + 5)) as i32;
            let mut v = vec![0xE8];
            v.extend_from_slice(&disp.to_le_bytes());
            v.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01, 0xC3]);
            v
        };
        let t = LiftTest::new(Box::leak(bytes.into_boxed_slice()))
            .at(at)
            .image_base(at)
            .with_config(config(true))
            .translation()
            .expect("compiles");
        let t = cache.write().unwrap().insert(t);
        runtime.publish(&t);
    }

    // SAFETY: every translation was compiled by this test for this context.
    let mut steps = 0;
    loop {
        let step = unsafe { runtime.step() };
        steps += 1;
        assert!(steps < 32, "the unwind should terminate");
        match step {
            Step::Exit(ExitReason::Return) => break,
            Step::Exit(_) => continue,
            Step::NeedsTranslation(pc) => panic!("no translation for {pc:#x}"),
        }
    }

    // Every level ran exactly once. This is the assertion that matters: a ring
    // too small to remember the outermost frames still produced the right
    // answer, because a stale frame *misses* rather than lying.
    assert_eq!(
        runtime.ctx.rax, DEPTH,
        "each of the {DEPTH} levels must have run once"
    );
    drop(stack);
}

#[test]
fn a_thread_with_no_ring_is_not_a_special_case() {
    // Code lifted *with* the ring, run on a thread that never allocated one: a
    // null pointer must mean "no prediction", not a crash - the same shape as a
    // null `link_table`.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, false); // no ring on the thread
    let caller = cache.write().unwrap().insert(caller(true)); // ring in the code
    cache.write().unwrap().insert(callee(true));
    guest.runtime.publish(&caller);
    assert!(guest.runtime.ctx.callret_base.is_null());

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    // It falls back to the table, which is published, so the program completes.
    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rbx, 7);
    assert_eq!(guest.runtime.ctx.rax, 1);
}

#[test]
fn a_matching_pop_advances_the_ring_and_a_missing_one_does_not() {
    let (_cache, mut guest) = seeded(true);
    guest.runtime.detach_table();
    assert_eq!(guest.runtime.ctx.callret_off, 0);

    // SAFETY: as above.
    unsafe { guest.runtime.step() };

    // One push (offset wraps down by a frame) and one matching pop (back up), so
    // a balanced program leaves the pointer exactly where it found it.
    assert_eq!(
        guest.runtime.ctx.callret_off, 0,
        "a balanced call/return is a no-op on the ring"
    );
    assert_eq!(
        guest.runtime.call_return().unwrap().written(),
        1,
        "and the frame it used is still there - a pop does not clear"
    );
}
