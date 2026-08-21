//! The host side of block linking: a translation cache, a per-thread link
//! table, and the dispatch loop that ties them together.
//!
//! Everything in [`crate::core::link`] describes what lifted code does with a
//! [`BlockLinkEntry`]. This module is the other half - what a host must do so
//! that those rows are true. It exists because the contract is easy to state and
//! easy to get subtly wrong, and prose in a doc comment cannot be run: before
//! this module the only implementation of it was a `follow()` helper in a test
//! file, and every real embedder rewrote it.
//!
//! # Trust model
//!
//! Lifted code performs no validation beyond an address compare. A row that
//! matches is taken to name executable code enterable at that guest PC, with the
//! `image_base` it wants, staying mapped for as long as the row is reachable.
//! Nothing here is atomic. So:
//!
//! - One [`ThreadRuntime`] per guest thread, each owning its own table. The
//!   [`CodeCache`] behind them is shared.
//! - A translation is published at every address it can be entered at**, which
//!   is exactly [`LiftedCode::resume_points`](crate::core::LiftedCode::resume_points)
//!   and nothing else. Publishing an address that is not a resume point either
//!   runs the region from the top or exits with no progress.
//! - The address published is [`CompiledRegion::link_target`], not `entry`.
//!   Under a pinned ABI those are different functions and using one where the
//!   other belongs is a wrong-code bug nothing catches. [`CodeCache::insert`] is
//!   the only place that choice is made, which is what stops it being made wrong.
//!
//! # Retraction and the safepoint rule
//!
//! Retracting a row is a single naturally-aligned 8-byte store, so a concurrent
//! reader sees the old value or the new one and nothing between. But "the old
//! value" means a thread can enter a translation that was retracted a moment
//! ago, and therefore **host code must not be freed at retraction time**.
//!
//! [`CodeCache::invalidate_range`] unpublishes and *retires*; the retired
//! translation is held alive by its `Arc` until [`CodeCache::reclaim`] is told
//! every thread has passed a safepoint since. The top of [`ThreadRuntime::step`]
//! is that safepoint: no lifted code is executing there. A single-threaded host
//! may reclaim immediately.
//!
//! # What is deliberately not here
//!
//! FEX's `LookupCache` has three tiers; this has one. FEX's L1 must be lockless
//! because its dispatcher performs a lookup on *every* region entry. Here the
//! hit path is inside guest code, reading a table the thread owns exclusively,
//! and the dispatcher runs only on a **miss** - so the lock is on the cold path
//! by construction, and a second lockless tier in front of it would buy nothing.
//! [`CacheStats`] is how that assumption stays checkable.
//!
//! FEX also resizes its L1 at runtime against a hit-rate heuristic. Microlift
//! bakes [`BlockLinkConfig`] into every region it compiles, so a resize means
//! discarding all code; pick the size once, and read [`CacheStats::miss_rate`]
//! to find out if you picked badly.
//!
//! # A known cost
//!
//! Each [`Translation`] owns its own [`CraneliftBackend`], because
//! `compile_lifted*` requires a fresh module (outlined helper symbols are named
//! per body, so two regions in one module would redefine them). That is one
//! mmap per translation. Fine for tests, expensive for a long-running cache; the
//! fix is to give helper symbols a region discriminant so one module can hold
//! many, and it is not done here.

#[rustfmt::skip]
#[allow(unused_imports)]
use crate::{
    backend::{CompiledRegion, CraneliftBackend},
    core::link::{BlockLinkConfig, BlockLinkEntry},
};

mod cache;
mod callret;
mod links;
mod table;
mod thread;

pub use cache::{CacheStats, CodeCache, Translation};
pub use callret::CallRetStack;
pub use links::{Entered, ThreadLinks, enter, enter_with, follow_links, follow_links_with};
pub use table::LinkTable;
pub use thread::{Step, ThreadRuntime};
