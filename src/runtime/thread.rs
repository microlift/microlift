//! One guest thread: context, [`ThreadLinks`] and a simple dispatch loop.
//! A host with its own dispatch loop wants `ThreadLinks` directly.

use std::sync::{Arc, RwLock};

use super::{CacheStats, CallRetStack, CodeCache, LinkTable, ThreadLinks, Translation};
use crate::arch::x86_64::context::RegisterContext;
use crate::core::callret::CallRetConfig;
use crate::core::exit::ExitReason;
use crate::core::link::BlockLinkConfig;

/// What one turn of the dispatch loop did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Lifted code exited for a reason the host must service.
    Exit(ExitReason),
    /// No translation for this PC: compile, [`CodeCache::insert`], step again.
    NeedsTranslation(u64),
}

/// One guest thread: owns the [`RegisterContext`] and [`LinkTable`], shares a [`CodeCache`].
/// The table pointer is installed on construction (linking is live).
pub struct ThreadRuntime {
    pub ctx: RegisterContext,
    // Boxed so pointers `attach` put in the context survive moves.
    links: Box<ThreadLinks>,
    cache: Arc<RwLock<CodeCache>>,
}

impl ThreadRuntime {
    /// `config` must equal the one every region reachable through this table was lifted under.
    pub fn new(config: BlockLinkConfig, cache: Arc<RwLock<CodeCache>>) -> Self {
        let mut runtime = Self {
            ctx: RegisterContext::default(),
            links: Box::new(ThreadLinks::new(config, None)),
            cache,
        };
        runtime.attach_table();
        runtime
    }

    /// Allocates and installs a call-return ring; `config` must match the lift.
    /// Without it every `RET` falls through to the probe.
    pub fn with_call_return(mut self, config: CallRetConfig) -> Self {
        let table = self.links.table().config();
        self.links = Box::new(ThreadLinks::new(table, Some(config)));
        self.ctx.callret_off = 0;
        self.attach_table();
        self
    }

    pub fn call_return(&self) -> Option<&CallRetStack> {
        self.links.call_return()
    }

    pub fn links(&self) -> &ThreadLinks {
        &self.links
    }

    /// Points `ctx.link_table` at this thread's table; counterpart of [`detach_table`](Self::detach_table).
    pub fn attach_table(&mut self) {
        self.links.attach(&mut self.ctx);
    }

    /// Nulls `ctx.link_table` so the next probe declines.
    ///
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned) this is the only
    /// way to end a tail-call chain.
    pub fn detach_table(&mut self) {
        ThreadLinks::detach(&mut self.ctx);
    }

    pub fn table(&self) -> &LinkTable {
        self.links.table()
    }

    pub fn cache(&self) -> &Arc<RwLock<CodeCache>> {
        &self.cache
    }

    pub fn stats(&self) -> CacheStats {
        self.cache.read().unwrap().stats()
    }

    /// Makes `translation` linkable from this thread at every entry address.
    ///
    /// [`CodeCache::insert`] makes it findable; the cache is shared, tables are per thread.
    pub fn publish(&mut self, translation: &Translation) {
        self.links.publish(translation);
    }

    /// See [`ThreadLinks::publish_at`].
    pub fn publish_at(&mut self, translation: &Translation, pc: u64) -> bool {
        self.links.publish_at(translation, pc)
    }

    /// Enters the translation for the current PC and follows resolved links.
    ///
    /// This is a safepoint: no lifted code is running on entry, so a host may
    /// [`reclaim`](CodeCache::reclaim) here. The lookup takes only the cache's read lock.
    ///
    /// # Safety
    ///
    /// Enters JIT code with `ctx`; every reachable translation must match this thread's context
    /// layout, guest architecture and lifter config.
    pub unsafe fn step(&mut self) -> Step {
        let pc = self.ctx.rip;
        let Some(translation) = self.cache.read().unwrap().lookup(pc) else {
            return Step::NeedsTranslation(pc);
        };

        // SAFETY: the caller's obligation.
        let entered = unsafe { super::enter(&translation, &mut self.ctx) };
        // Recorded outside the chain so the cache lock is not held across guest code.
        self.cache
            .read()
            .unwrap()
            .note_links_followed(entered.links);
        Step::Exit(entered.exit)
    }

    /// Invalidates `[start, start + len)` in the cache and **this** thread's table; returns rows lost.
    ///
    /// A multi-threaded host calls [`CodeCache::invalidate_range`] once, then retracts the returned
    /// addresses from each thread's table while that thread is outside lifted code.
    pub fn invalidate_range(&mut self, start: u64, len: u64) -> usize {
        let unpublished = self.cache.write().unwrap().invalidate_range(start, len);
        self.links.retract_all(&unpublished)
    }

    /// Retracts addresses unpublished by [`CodeCache::invalidate_range`] from this thread's table.
    pub fn retract_all(&mut self, addresses: &[u64]) -> usize {
        self.links.retract_all(addresses)
    }

    /// Steps until `stop` returns true or a translation is needed.
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
