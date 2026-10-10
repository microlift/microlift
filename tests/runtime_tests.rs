//! Host side of block linking ([`microlift::runtime`]): cache, link table, dispatch loop.
//! The load-bearing assertions are on [`CacheStats`]: a dispatcher that never links passes every behavioural test.

use std::sync::{Arc, RwLock};

use microlift::core::exit::ExitReason;
use microlift::core::link::BlockLinkConfig;
use microlift::runtime::{CodeCache, Step, ThreadRuntime, Translation};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

/// Far from the first region so tests cannot confuse them.
const CALLEE: u64 = 0x2000;
const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// `mov rax, 1 ; ret` at [`DEFAULT_LIFT_BASE`]: the region that exits.
fn caller() -> Translation {
    LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .with_config(|c| c.block_link = Some(link_config()))
        .translation()
        .expect("the caller region compiles")
}

/// `mov rbx, 7 ; ret` at [`CALLEE`]: the region a link lands in.
fn callee() -> Translation {
    LiftTest::new(&[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3])
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(|c| c.block_link = Some(link_config()))
        .translation()
        .expect("the callee region compiles")
}

/// A guest stack and its thread; the harness-owned `Vec` must outlive every entry.
struct Guest {
    runtime: ThreadRuntime,
    _stack: Vec<u64>,
}

impl Guest {
    /// Stack holds `returns` (innermost first) over the exit trampoline.
    fn new(cache: &Arc<RwLock<CodeCache>>, returns: &[u64]) -> Self {
        let stack = vec![0u64; 1024];
        let top = stack.as_ptr() as u64 + (stack.len() as u64) * 8;

        let mut runtime = ThreadRuntime::new(link_config(), Arc::clone(cache));
        // First `RET` pops `returns[0]`.
        let depth = returns.len() as u64 + 1;
        runtime.ctx.rsp = top - depth * 8;
        // SAFETY: every word written is inside `stack`.
        unsafe {
            for (i, &addr) in returns.iter().enumerate() {
                *((runtime.ctx.rsp + (i as u64) * 8) as *mut u64) = addr;
            }
            *((runtime.ctx.rsp + (returns.len() as u64) * 8) as *mut u64) = EXIT_TRAMPOLINE;
        }
        Self {
            runtime,
            _stack: stack,
        }
    }
}

/// A cache holding both regions, published into `guest`'s table.
fn seeded(guest: &mut Guest, cache: &Arc<RwLock<CodeCache>>) {
    for translation in [caller(), callee()] {
        let translation = cache.write().unwrap().insert(translation);
        guest.runtime.publish(&translation);
    }
}

#[test]
fn a_published_translation_is_entered_and_its_successor_is_followed() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    seeded(&mut guest, &cache);
    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    // One `step` ran both regions: the `RET` resolved its successor from the table.
    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rax, 1, "the caller ran");
    assert_eq!(guest.runtime.ctx.rbx, 7, "and so did the callee");

    let stats = guest.runtime.stats();
    assert_eq!(stats.lookups, 1, "only the initial entry asked the host");
    assert_eq!(stats.links_followed, 1, "the successor came from the table");
}

#[test]
fn the_host_is_asked_once_per_entry_when_nothing_is_published() {
    // Control: with an empty table the host is consulted per region.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    for translation in [caller(), callee()] {
        cache.write().unwrap().insert(translation);
    }
    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;

    // SAFETY: as above.
    let first = unsafe { guest.runtime.step() };
    assert_eq!(first, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rax, 1);
    assert_eq!(guest.runtime.ctx.rbx, 0, "the callee has not run yet");
    assert_eq!(guest.runtime.ctx.rip, CALLEE);

    // SAFETY: as above.
    let second = unsafe { guest.runtime.step() };
    assert_eq!(second, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rbx, 7);

    let stats = guest.runtime.stats();
    assert_eq!(stats.lookups, 2, "one host dispatch per region");
    assert_eq!(stats.links_followed, 0, "and nothing linked");
}

#[test]
fn an_untranslated_address_asks_for_a_translation_rather_than_guessing() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[]);
    guest.runtime.ctx.rip = 0xDEAD_0000;

    // SAFETY: nothing is entered on this path.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::NeedsTranslation(0xDEAD_0000));
    assert_eq!(
        guest.runtime.stats().miss_rate(),
        Some(1.0),
        "one lookup, no hit"
    );
}

#[test]
fn publishing_uses_the_link_target_and_not_the_host_entry() {
    // Addresses coincide without a pinned ABI; this checks `row()` alone decides and names an enterable address (`pinned_link_tests` covers the difference).
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    seeded(&mut guest, &cache);

    let row = guest
        .runtime
        .table()
        .get(CALLEE)
        .expect("the callee is published at its own entry address");
    let translation = cache
        .read()
        .unwrap()
        .peek(CALLEE)
        .expect("and is in the cache")
        .clone();

    assert_eq!(row.code, translation.link_target as u64);
    assert_eq!(row.image_base, CALLEE);
}

#[test]
fn a_translation_is_published_at_every_address_it_can_be_entered_at() {
    // Publishing only the entry would leave links to inner resume points permanently missing.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[]);
    let translation = cache.write().unwrap().insert(caller());
    guest.runtime.publish(&translation);

    assert!(
        !translation.resume_points.is_empty(),
        "the fixture must have somewhere to be entered"
    );
    for &pc in &translation.resume_points {
        assert!(
            guest.runtime.table().get(pc).is_some(),
            "no row published for resume point {pc:#x}"
        );
        assert!(cache.read().unwrap().peek(pc).is_some());
    }
    assert_eq!(
        guest.runtime.table().occupancy(),
        translation.resume_points.len()
    );
}

#[test]
fn detaching_the_table_forces_every_exit_through_the_host() {
    // The documented way to end a chain: one null pointer stops the `RET` resolving.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    seeded(&mut guest, &cache);
    guest.runtime.detach_table();
    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rbx, 0, "the callee did not run");
    assert_eq!(guest.runtime.stats().links_followed, 0);

    // Re-attaching restores it; the state is the pointer alone.
    guest.runtime.attach_table();
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);
}

#[test]
fn a_colliding_row_is_a_miss_rather_than_a_wrong_jump() {
    // Direct-mapped eviction must degrade to a host lookup, never the wrong code.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    seeded(&mut guest, &cache);

    // Same slot as CALLEE (`pc >> 4` masked to ROWS); publishing evicts its row.
    let colliding = CALLEE + (ROWS as u64) * 16;
    assert_eq!(link_config().slot(colliding), link_config().slot(CALLEE));
    let evictor = LiftTest::new(&[0xC3])
        .at(colliding)
        .image_base(colliding)
        .with_config(|c| c.block_link = Some(link_config()))
        .translation()
        .expect("compiles");
    let evictor = cache.write().unwrap().insert(evictor);
    guest.runtime.publish(&evictor);

    assert!(guest.runtime.table().get(CALLEE).is_none(), "evicted");
    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rip, CALLEE);
    assert_eq!(guest.runtime.ctx.rbx, 0, "the evictor must not have run");
    assert_eq!(guest.runtime.stats().links_followed, 0);
}

// ThreadLinks: the per-thread half

use microlift::arch::x86_64::context::RegisterContext;
use microlift::core::callret::CallRetConfig;
use microlift::runtime::{Entered, ThreadLinks};

#[test]
fn attaching_points_a_context_at_both_tables() {
    // One call, not two stores, so hosts need not remember the pair.
    let mut links = ThreadLinks::new(link_config(), Some(CallRetConfig::new(16)));
    let mut ctx = RegisterContext::default();
    assert!(ctx.link_table.is_null());
    assert!(ctx.callret_base.is_null());

    links.attach(&mut ctx);

    assert_eq!(ctx.link_table, links.link_table_ptr());
    assert_eq!(ctx.callret_base, links.callret_base_ptr());
}

#[test]
fn a_thread_without_a_ring_attaches_a_null_one() {
    // Null is the first thing every push and pop tests, so disabling prediction needs no host branch.
    let mut links = ThreadLinks::new(link_config(), None);
    let mut ctx = RegisterContext::default();
    links.attach(&mut ctx);

    assert!(!ctx.link_table.is_null(), "the table is still attached");
    assert!(ctx.callret_base.is_null());
    assert!(links.call_return().is_none());
}

#[test]
fn detaching_leaves_the_ring_alone() {
    // `detach` nulls the table only; the per-thread ring outlives chains and holds frames of unreturned calls.
    let mut links = ThreadLinks::new(link_config(), Some(CallRetConfig::new(16)));
    let mut ctx = RegisterContext::default();
    links.attach(&mut ctx);

    ThreadLinks::detach(&mut ctx);

    assert!(ctx.link_table.is_null());
    assert!(!ctx.callret_base.is_null(), "the ring survives");
}

#[test]
fn a_nested_entry_resumes_below_the_one_that_called_it() {
    // The ring is per thread, a context per entry: a nested entry starting at zero would clobber the outer frame (costs predictions, not correctness).
    let mut outer = RegisterContext::default();
    outer.callret_off = 0x40;

    let mut inner = RegisterContext::default();
    ThreadLinks::resume_call_return(&mut inner, Some(&outer));
    assert_eq!(inner.callret_off, 0x40);

    // The outermost entry starts where an untouched ring starts.
    let mut first = RegisterContext::default();
    first.callret_off = 0xdead;
    ThreadLinks::resume_call_return(&mut first, None);
    assert_eq!(first.callret_off, 0);
}

#[test]
fn publishing_covers_every_resume_point_and_nothing_else() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut links = ThreadLinks::new(link_config(), None);
    let translation = cache.write().unwrap().insert(caller());

    links.publish(&translation);

    for &pc in &translation.resume_points {
        let row = links
            .table()
            .get(pc)
            .unwrap_or_else(|| panic!("no row for resume point {pc:#x}"));
        assert_eq!(row.code, translation.link_target as u64);
    }
    assert_eq!(
        links.table().occupancy(),
        translation.resume_points.len(),
        "a row for an address that is not a resume point would run the region \
         from the top or exit having made no progress"
    );
}

#[test]
fn retracting_reports_only_the_rows_this_thread_held() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut links = ThreadLinks::new(link_config(), None);
    let translation = cache.write().unwrap().insert(callee());
    links.publish(&translation);

    let held = translation.resume_points.clone();
    let mut asked = held.clone();
    // An address this thread never published must not count as lost, even if another thread's table had it.
    asked.push(0xdead_0000);

    assert_eq!(links.retract_all(&asked), held.len());
    assert_eq!(links.table().occupancy(), 0);
    assert_eq!(links.retract_all(&asked), 0, "and again is a no-op");
}

#[test]
fn each_link_is_reported_once_with_the_exit_that_resolved_it() {
    // Observer gets the context too: a linked exit is almost always `UnknownJump`, and the transfer kind is in `exit_code`.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache, &[CALLEE]);
    seeded(&mut guest, &cache);
    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;

    let translation = cache
        .read()
        .unwrap()
        .peek(DEFAULT_LIFT_BASE)
        .expect("the caller is published")
        .clone();

    let mut seen: Vec<(ExitReason, u32)> = Vec::new();
    // SAFETY: both translations were compiled by this test for this context.
    let entered: Entered = unsafe {
        microlift::runtime::enter_with(&translation, &mut guest.runtime.ctx, |exit, ctx| {
            seen.push((exit, ctx.exit_code));
        })
    };

    assert_eq!(entered.links, 1, "one hop: the caller's `RET` into the callee");
    assert_eq!(seen.len(), 1, "reported once, for the hop and not the tail");
    assert_eq!(seen[0].0, ExitReason::UnknownJump);
    assert_eq!(
        entered.exit,
        ExitReason::Return,
        "the tail's exit is returned rather than reported"
    );
}
