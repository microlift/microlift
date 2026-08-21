//! Retracting a translation when the guest bytes under it change.

use std::sync::{Arc, RwLock};

use microlift::core::exit::ExitReason;
use microlift::core::link::BlockLinkConfig;
use microlift::runtime::{CodeCache, Step, ThreadRuntime, Translation};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

const CALLEE: u64 = 0x2000;
const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// `mov rax, 1 ; ret` - the region that exits to [`CALLEE`].
fn caller() -> Translation {
    LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .with_config(|c| c.block_link = Some(link_config()))
        .translation()
        .expect("compiles")
}

/// `mov rbx, imm32 ; ret` at [`CALLEE`] - the region whose bytes a test
/// changes, so that re-lifting the same address yields a different answer.
fn callee_writing(value: u32) -> Translation {
    let mut bytes = vec![0x48, 0xC7, 0xC3];
    bytes.extend_from_slice(&value.to_le_bytes());
    bytes.push(0xC3);
    LiftTest::new(Box::leak(bytes.into_boxed_slice()))
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(|c| c.block_link = Some(link_config()))
        .translation()
        .expect("compiles")
}

struct Guest {
    runtime: ThreadRuntime,
    _stack: Vec<u64>,
}

impl Guest {
    fn new(cache: &Arc<RwLock<CodeCache>>) -> Self {
        let stack = vec![0u64; 1024];
        let top = stack.as_ptr() as u64 + (stack.len() as u64) * 8;
        let mut runtime = ThreadRuntime::new(link_config(), Arc::clone(cache));
        runtime.ctx.rsp = top - 16;
        // SAFETY: both words are inside `stack`, which this struct owns.
        unsafe {
            *(runtime.ctx.rsp as *mut u64) = CALLEE;
            *((runtime.ctx.rsp + 8) as *mut u64) = EXIT_TRAMPOLINE;
        }
        Self {
            runtime,
            _stack: stack,
        }
    }

    /// Reset the stack and PC for another run from the top.
    fn rewind(&mut self) {
        let top = self._stack.as_ptr() as u64 + (self._stack.len() as u64) * 8;
        self.runtime.ctx.rsp = top - 16;
        // SAFETY: as in `new`.
        unsafe {
            *(self.runtime.ctx.rsp as *mut u64) = CALLEE;
            *((self.runtime.ctx.rsp + 8) as *mut u64) = EXIT_TRAMPOLINE;
        }
        self.runtime.ctx.rip = DEFAULT_LIFT_BASE;
        self.runtime.ctx.rbx = 0;
    }
}

fn install(guest: &mut Guest, cache: &Arc<RwLock<CodeCache>>, translation: Translation) {
    let translation = cache.write().unwrap().insert(translation);
    guest.runtime.publish(&translation);
}

/// The guest rewrites the callee, and a host that does not retract keeps running the translation of the bytes that
/// are gone.
#[test]
fn a_stale_translation_is_not_run_after_its_bytes_change() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    install(&mut guest, &cache, caller());
    install(&mut guest, &cache, callee_writing(7));

    guest.rewind();
    // SAFETY: every translation here was compiled by this test for this context.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);

    // The guest rewrites the eight bytes at CALLEE. The host is told.
    let retracted = guest.runtime.invalidate_range(CALLEE, 8);
    assert_eq!(retracted, 1, "the callee's row should have been retracted");

    guest.rewind();
    // Nothing is published for CALLEE any more, so the caller's `RET` cannot
    // link and the host is asked - which is the whole observable effect.
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::UnknownJump)
    );
    assert_eq!(guest.runtime.ctx.rip, CALLEE);
    assert_eq!(guest.runtime.ctx.rbx, 0, "the stale translation did not run");

    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::NeedsTranslation(CALLEE)
    );

    // And once the host lifts the new bytes, the *new* answer is what runs.
    install(&mut guest, &cache, callee_writing(9));
    guest.rewind();
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(
        guest.runtime.ctx.rbx, 9,
        "a no-op invalidator would still be answering 7"
    );
}

/// **The negative-negative.** Without this, "retract everything, always" passes
/// the test above - and would turn every invalidation into a full cache flush.
#[test]
fn a_write_that_misses_the_covered_range_retracts_nothing() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    install(&mut guest, &cache, caller());
    install(&mut guest, &cache, callee_writing(7));

    // A page nothing was lifted from.
    let retracted = guest.runtime.invalidate_range(0x9000, 0x1000);
    assert_eq!(retracted, 0);
    assert_eq!(cache.read().unwrap().stats().retractions, 0);

    guest.rewind();
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7, "both translations survived");
    assert_eq!(guest.runtime.stats().links_followed, 1);
}

/// A write inside the *same page* as a translation, but outside the bytes it
/// was lifted from, must also spare it. The page index narrows the sweep;
/// `covered` is what decides.
#[test]
fn the_page_index_narrows_the_sweep_but_does_not_decide_it() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    install(&mut guest, &cache, caller());
    install(&mut guest, &cache, callee_writing(7));

    // CALLEE is 0x2000 and the region is eight bytes; 0x2800 is the same page.
    assert_eq!(0x2800 >> 12, CALLEE >> 12);
    let retracted = guest.runtime.invalidate_range(0x2800, 8);

    assert_eq!(retracted, 0, "same page, different bytes");
    guest.rewind();
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);
}

/// A one-byte write anywhere inside `covered` is enough - the translation
/// depends on all of it, not just its first instruction.
#[test]
fn a_single_byte_inside_the_region_is_enough() {
    for offset in [0u64, 3, 7] {
        let cache = Arc::new(RwLock::new(CodeCache::new()));
        let mut guest = Guest::new(&cache);
        install(&mut guest, &cache, callee_writing(7));

        assert_eq!(
            guest.runtime.invalidate_range(CALLEE + offset, 1),
            1,
            "a write at +{offset} must retract"
        );
    }
}

/// Retiring is not freeing. The translation stays alive until the host says
/// every thread has passed a safepoint, because another one may be inside it.
#[test]
fn an_invalidated_translation_is_retired_before_it_is_freed() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    install(&mut guest, &cache, callee_writing(7));

    guest.runtime.invalidate_range(CALLEE, 8);

    assert_eq!(
        cache.read().unwrap().retired_count(),
        1,
        "unpublished, not yet freed"
    );
    assert!(
        cache.read().unwrap().peek(CALLEE).is_none(),
        "and unreachable by address"
    );

    assert_eq!(cache.write().unwrap().reclaim(), 1);
    assert_eq!(cache.read().unwrap().retired_count(), 0);
}

/// A translation is published at every resume point it has and must be
/// retracted from all of them - not just the one the page index happened to
/// name first.
#[test]
fn every_resume_point_is_retracted_not_just_the_first() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let translation = callee_writing(7);
    let points = translation.resume_points.clone();
    install(&mut guest, &cache, translation);

    guest.runtime.invalidate_range(CALLEE, 8);

    for pc in points {
        assert!(
            cache.read().unwrap().peek(pc).is_none(),
            "resume point {pc:#x} is still published"
        );
        assert!(
            guest.runtime.table().get(pc).is_none(),
            "resume point {pc:#x} still has a row"
        );
    }
    assert_eq!(guest.runtime.table().occupancy(), 0);
}
