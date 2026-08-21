//! One guest thread: its register context, its tables, and the dispatch loop
//! that enters lifted code and services what comes back.
//!
//! The tables and the entry sequence are [`ThreadLinks`]; this is that plus a
//! context and a loop. A host with a dispatch loop of its own - one that services
//! imports, syscalls and callbacks, and owns the context across all of them -
//! wants `ThreadLinks` directly and not this.

use std::sync::{Arc, RwLock};

use super::{CacheStats, CallRetStack, CodeCache, LinkTable, ThreadLinks, Translation};
use crate::arch::x86_64::context::RegisterContext;
use crate::core::callret::CallRetConfig;
use crate::core::exit::ExitReason;
use crate::core::link::BlockLinkConfig;

/// What one turn of the dispatch loop did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Lifted code ran and exited for a reason the host must service. The
    /// context holds the guest PC and whatever else the exit carried.
    Exit(ExitReason),
    /// There is no translation for the current guest PC. The host must compile
    /// one, [`CodeCache::insert`] it, and step again.
    NeedsTranslation(u64),
}

/// One guest thread's execution state.
///
/// Owns the [`RegisterContext`] and the [`LinkTable`] lifted code reads, and
/// shares a [`CodeCache`] with every other thread. The table pointer is
/// installed into the context on construction and stays installed - which is
/// what makes `ctx.link_table` non-null and linking live.
pub struct ThreadRuntime {
    /// The live guest register context.
    pub ctx: RegisterContext,
    /// The tables lifted code reads. Boxed so the pointers `attach` puts in the
    /// context survive this struct being moved.
    links: Box<ThreadLinks>,
    cache: Arc<RwLock<CodeCache>>,
}

impl ThreadRuntime {
    /// A thread with a table of `config`'s geometry sharing `cache`.
    ///
    /// `config` must be the same one every region reachable through this table
    /// was lifted under.
    pub fn new(config: BlockLinkConfig, cache: Arc<RwLock<CodeCache>>) -> Self {
        let mut runtime = Self {
            ctx: RegisterContext::default(),
            links: Box::new(ThreadLinks::new(config, None)),
            cache,
        };
        runtime.attach_table();
        runtime
    }

    /// Allocate a call-return ring for this thread and install it.
    ///
    /// `config` must be the one every region on this thread was lifted under.
    /// Without this the ring pointer stays null and every `RET` falls through to
    /// the probe - which is a valid configuration, not a broken one.
    ///
    /// Boxed so the pointer handed to the context survives this struct being
    /// moved.
    pub fn with_call_return(mut self, config: CallRetConfig) -> Self {
        let table = self.links.table().config();
        self.links = Box::new(ThreadLinks::new(table, Some(config)));
        self.ctx.callret_off = 0;
        self.attach_table();
        self
    }

    /// This thread's call-return ring, if it has one.
    pub fn call_return(&self) -> Option<&CallRetStack> {
        self.links.call_return()
    }

    /// This thread's tables.
    pub fn links(&self) -> &ThreadLinks {
        &self.links
    }

    /// Point `ctx.link_table` at this thread's table, enabling linking.
    ///
    /// Called by [`new`](Self::new); public because it is the counterpart to
    /// [`detach_table`](Self::detach_table).
    pub fn attach_table(&mut self) {
        self.links.attach(&mut self.ctx);
    }

    /// Null `ctx.link_table`, so the next probe declines and control returns
    /// here.
    ///
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned)
    /// this is the way to end a chain of tail calls: a linked entry is
    /// invisible to the host, so there is no counter to trip and no budget to
    /// exhaust. Under `HostResume` it merely forces every exit through the host.
    pub fn detach_table(&mut self) {
        ThreadLinks::detach(&mut self.ctx);
    }

    /// This thread's table.
    pub fn table(&self) -> &LinkTable {
        self.links.table()
    }

    /// The shared cache.
    pub fn cache(&self) -> &Arc<RwLock<CodeCache>> {
        &self.cache
    }

    /// Statistics from the shared cache.
    pub fn stats(&self) -> CacheStats {
        self.cache.read().unwrap().stats()
    }

    /// Publish `translation` into this thread's table at every address it can
    /// be entered at.
    ///
    /// Separate from [`CodeCache::insert`] because the cache is shared and a
    /// table is not: inserting makes a translation *findable*, publishing makes
    /// it *linkable from this thread*. A host with several threads inserts once
    /// and publishes per thread - or publishes lazily, on the miss that would
    /// otherwise have gone to the cache.
    pub fn publish(&mut self, translation: &Translation) {
        self.links.publish(translation);
    }

    /// Run one turn: enter the translation for the current guest PC, then
    /// follow as many links as the region resolves for itself.
    ///
    /// This is the safepoint. No lifted code is executing at the moment this
    /// is entered, which is what makes it the point a host may
    /// [`reclaim`](CodeCache::reclaim) from.
    ///
    /// The lookup takes the cache's **read** lock: the counters are atomic, so a
    /// miss - which is on its way to a compile - does not queue behind another
    /// thread's miss on a different address.
    ///
    /// # Safety
    ///
    /// Enters JIT-compiled code with `ctx` as its argument. Every translation
    /// reachable from the cache or the table must be one this thread's context
    /// layout, guest architecture and lifter config actually match.
    pub unsafe fn step(&mut self) -> Step {
        let pc = self.ctx.rip;
        let Some(translation) = self.cache.read().unwrap().lookup(pc) else {
            return Step::NeedsTranslation(pc);
        };

        // SAFETY: the caller's obligation, documented above. `super::enter` is
        // the shared sequence - the fields an entry owes the region, the call,
        // and the chain of links after it.
        let entered = unsafe { super::enter(&translation, &mut self.ctx) };
        // Recorded here rather than inside the chain, because the counter is the
        // shared cache's and the chain runs guest code: reaching it from in there
        // would hold the cache's lock across the whole of it.
        self.cache
            .read()
            .unwrap()
            .note_links_followed(entered.links);
        Step::Exit(entered.exit)
    }

    /// Invalidate `[start, start + len)` for the cache *and* this thread's
    /// table.
    ///
    /// The two halves are separate because the cache is shared and the table is
    /// not: this retracts only from **this** thread. A multi-threaded host must
    /// call [`CodeCache::invalidate_range`] once and then retract the addresses
    /// it returns from every thread's table, at a point where that thread is not
    /// inside lifted code.
    ///
    /// Returns how many rows this thread's table lost.
    pub fn invalidate_range(&mut self, start: u64, len: u64) -> usize {
        let unpublished = self.cache.write().unwrap().invalidate_range(start, len);
        self.links.retract_all(&unpublished)
    }

    /// Retract addresses another thread's [`CodeCache::invalidate_range`]
    /// unpublished, from this thread's table.
    pub fn retract_all(&mut self, addresses: &[u64]) -> usize {
        self.links.retract_all(addresses)
    }

    /// Step until `stop` says otherwise or a step needs a translation.
    ///
    /// # Safety
    ///
    /// As [`step`](Self::step).
    pub unsafe fn run_until(&mut self, stop: impl Fn(&RegisterContext, ExitReason) -> bool) -> Step {
        loop {
            // SAFETY: the caller's obligation.
            let step = unsafe { self.step() };
            match step {
                Step::Exit(exit) if stop(&self.ctx, exit) => return step,
                Step::Exit(_) => {}
                Step::NeedsTranslation(_) => return step,
            }
        }
    }
}
