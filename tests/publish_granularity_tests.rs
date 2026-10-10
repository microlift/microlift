//! `ThreadLinks::publish` vs `publish_at`: the narrow form writes only one row, sparing a direct-mapped table.
//! Pins which row it writes, which it does not, and what it refuses.

use std::sync::{Arc, RwLock};

use microlift::core::link::BlockLinkConfig;
use microlift::runtime::{CodeCache, ThreadRuntime, Translation};
use microlift::testkit::DEFAULT_LIFT_BASE;
use microlift::testkit::prelude::*;

const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// Second entry point far enough away to land in a different row (adjacent addresses share a slot).
const SECOND_ENTRY: u64 = DEFAULT_LIFT_BASE + 16;

/// Sixteen `nop`s and a `ret`, entered at the top or the `ret`: two resume points.
fn two_entry_points() -> Translation {
    let mut bytes = vec![0x90; 16];
    bytes.push(0xC3);
    LiftTest::new(Box::leak(bytes.into_boxed_slice()))
        .with_config(|c| {
            c.block_link = Some(link_config());
            c.resume_points = vec![DEFAULT_LIFT_BASE, SECOND_ENTRY];
        })
        .translation()
        .expect("compiles")
}

/// Whether this thread's table offers `pc`.
fn linkable(runtime: &ThreadRuntime, pc: u64) -> bool {
    runtime.links().table().get(pc).is_some()
}

fn runtime(cache: &Arc<RwLock<CodeCache>>) -> ThreadRuntime {
    ThreadRuntime::new(link_config(), Arc::clone(cache))
}

/// Baseline: every resume point becomes linkable.
#[test]
fn publishing_broadly_offers_every_resume_point() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut runtime = runtime(&cache);
    let translation = cache.write().unwrap().insert(two_entry_points());

    runtime.publish(&translation);

    assert!(linkable(&runtime, DEFAULT_LIFT_BASE));
    assert!(linkable(&runtime, SECOND_ENTRY));
}

/// Narrow form offers only the one asked for; the other row stays unspent.
#[test]
fn publishing_narrowly_offers_only_the_address_asked_for() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut runtime = runtime(&cache);
    let translation = cache.write().unwrap().insert(two_entry_points());

    assert!(runtime.publish_at(&translation, SECOND_ENTRY));

    assert!(linkable(&runtime, SECOND_ENTRY));
    assert!(
        !linkable(&runtime, DEFAULT_LIFT_BASE),
        "the other resume point must not have been published"
    );
}

/// An address the region cannot be entered at is refused, not published.
#[test]
fn an_address_that_is_not_a_resume_point_is_refused() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut runtime = runtime(&cache);
    let translation = cache.write().unwrap().insert(two_entry_points());

    let inside_but_not_an_entry = DEFAULT_LIFT_BASE + 7;
    assert!(!runtime.publish_at(&translation, inside_but_not_an_entry));
    assert!(!linkable(&runtime, inside_but_not_an_entry));

    // An address unrelated to the region.
    assert!(!runtime.publish_at(&translation, 0x9999_0000));
    assert!(!linkable(&runtime, 0x9999_0000));
}
