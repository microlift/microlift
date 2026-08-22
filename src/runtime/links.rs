//! The per-thread half of block linking: the two tables lifted code reads, and
//! the sequence for entering a region and running out the chain it resolves.
//!
//! Split out of [`ThreadRuntime`](super::ThreadRuntime) because those are two
//! separable jobs and only one of them is generic. A recompiler's dispatch loop
//! is *its own* - it services imports, syscalls, signals, bridges and callbacks,
//! and it owns the [`RegisterContext`] across all of them. What is not its own is
//! this: which pointers a context must carry before it may enter lifted code,
//! which rows a translation must be published under, and what to do with a
//! [`BlockLinkEntry`] the region left in `ctx.link_hit`. Getting any of those
//! subtly wrong is a wrong-code bug with no diagnostic, so they live here and are
//! written once.
//!
//! `ThreadRuntime` is this plus a context and a loop.

use super::{CallRetStack, LinkTable, Translation};
use crate::arch::x86_64::context::RegisterContext;
use crate::core::callret::{CallRetConfig, CallRetEntry};
use crate::core::exit::ExitReason;
use crate::core::link::{BlockLinkConfig, BlockLinkEntry};

/// One guest thread's link table and call-return ring.
///
/// Both are read by *compiled code* through raw pointers in the
/// [`RegisterContext`], so both must outlive every region entered with them -
/// which is what makes [`attach`](Self::attach) take `&mut self` and hand out
/// addresses rather than returning copies.
pub struct ThreadLinks {
    table: LinkTable,
    callret: Option<CallRetStack>,
}

impl ThreadLinks {
    /// A thread's tables, with `link`'s geometry and optionally `callret`'s ring.
    ///
    /// Both configs must be the ones every region reachable from this thread was
    /// lifted under: the row count, the shift and the frame count are baked into
    /// each emitted probe. `None` for `callret` leaves the ring pointer null,
    /// which is not a degraded mode but simply "do not predict returns" - the
    /// emitted push and pop test for it.
    pub fn new(link: BlockLinkConfig, callret: Option<CallRetConfig>) -> Self {
        Self {
            table: LinkTable::new(link),
            callret: callret.map(CallRetStack::new),
        }
    }

    /// The rows lifted code probes.
    pub fn table(&self) -> &LinkTable {
        &self.table
    }

    /// The ring, if this thread predicts returns.
    pub fn call_return(&self) -> Option<&CallRetStack> {
        self.callret.as_ref()
    }

    /// The address `ctx.link_table` wants.
    pub fn link_table_ptr(&mut self) -> *mut BlockLinkEntry {
        self.table.as_mut_ptr()
    }

    /// The address `ctx.callret_base` wants, or null when there is no ring.
    ///
    /// Null is not a special case anywhere: it is the first thing every emitted
    /// push and pop tests, exactly as a null `link_table` is for the probe.
    pub fn callret_base_ptr(&mut self) -> *mut CallRetEntry {
        match &mut self.callret {
            Some(stack) => stack.as_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Point `ctx` at these tables.
    ///
    /// **Every context entering lifted code on this thread must have had this
    /// called on it**, and a host with more than one way in wants one place that
    /// does it rather than a store per site. `callret_off` is left alone: it is
    /// per-*entry* state, and a nested entry wants to resume where the entry
    /// below it left off rather than at the top - see
    /// [`resume_call_return`](Self::resume_call_return).
    pub fn attach(&mut self, ctx: &mut RegisterContext) {
        ctx.link_table = self.link_table_ptr();
        ctx.callret_base = self.callret_base_ptr();
    }

    /// Start `ctx`'s call-return frames below `outer`'s, rather than at the top.
    ///
    /// The ring is per *thread* and a `RegisterContext` is per *entry*, so a
    /// nested entry - a host callback into guest code - that started at zero would
    /// push its first frame over the outermost frame of the entry that called it.
    /// That is not a correctness problem, because a clobbered frame fails its
    /// `guest_pc` compare and falls back to the probe, but it costs the outer chain
    /// its predictions for nothing. Pass the outer entry's live context, or `None`
    /// at the outermost entry.
    pub fn resume_call_return(ctx: &mut RegisterContext, outer: Option<&RegisterContext>) {
        ctx.callret_off = outer.map_or(0, |outer| outer.callret_off);
    }

    /// Stop linking through `ctx`: the next probe declines and control returns to
    /// the host.
    ///
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned)
    /// this is **the** way to end a chain of tail calls, since a linked entry is
    /// invisible to the host and there is no counter to trip.
    pub fn detach(ctx: &mut RegisterContext) {
        ctx.link_table = std::ptr::null_mut();
    }

    /// Make `translation` linkable from this thread at **every** address it can
    /// be entered at.
    ///
    /// Not only at the address the host happened to look it up by: publishing the
    /// whole set is what lets a link to an address this thread has never entered
    /// hit anyway. And not at any address outside the set, which is why this takes
    /// a `&Translation` rather than a `(pc, row)` pair - a row offering an address
    /// that is not a resume point either runs the region from the top or exits
    /// having made no progress, and both were found the hard way.
    ///
    /// A row whose slot is taken evicts the row that was there. That costs a
    /// lookup and nothing else, which is what makes publishing the extra rows safe
    /// to do unconditionally.
    pub fn publish(&mut self, translation: &Translation) {
        for &pc in &translation.resume_points {
            self.table.publish(translation.row(pc));
        }
    }

    /// Make `translation` linkable from this thread at `pc` **alone**, reporting
    /// whether a row was written.
    ///
    /// The narrow half of [`publish`](Self::publish), for a host that knows which
    /// address it is about to link on and would rather not spend the rest of the
    /// table saying so. The two are the same trade seen from opposite ends: the
    /// broad form wins the links it could not have predicted, and pays for them in
    /// evictions - one region with many resume points can rewrite a large share of
    /// a direct-mapped table, and every row it displaces belonged to something the
    /// thread was already using.
    ///
    /// Which way that trade falls is a property of the guest, not of this crate,
    /// so both are offered and neither is the default here. A host with a
    /// **declining** probe and a **hitting** entry cache of the same geometry is
    /// looking at the eviction end of it.
    ///
    /// `covers` is checked rather than assumed, because the invariant
    /// [`publish`](Self::publish) gets for free from walking the resume points has
    /// to be paid for here: a row offering an address that is not a resume point
    /// either runs the region from the top or exits having made no progress.
    pub fn publish_at(&mut self, translation: &Translation, pc: u64) -> bool {
        if !translation.covers(pc) {
            return false;
        }
        self.table.publish(translation.row(pc));
        true
    }

    /// Drop the rows for `addresses`, returning how many this thread actually
    /// held.
    ///
    /// The per-thread half of invalidation: [`CodeCache::invalidate_range`] does
    /// the shared half and reports the addresses it unpublished, and each thread
    /// applies them to its own table at a point where it is not inside lifted
    /// code.
    ///
    /// [`CodeCache::invalidate_range`]: super::CodeCache::invalidate_range
    pub fn retract_all(&mut self, addresses: &[u64]) -> usize {
        addresses
            .iter()
            .filter(|&&pc| self.table.retract(pc))
            .count()
    }
}

/// How an entry into lifted code ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entered {
    /// Why the last region to run came back to the host.
    pub exit: ExitReason,
    /// How many successors were resolved without the host being asked - i.e. how
    /// many region entries block linking spared it.
    ///
    /// Returned rather than counted internally because the counter belongs to the
    /// shared [`CodeCache`](super::CodeCache), and reaching it from here would
    /// mean holding its lock across the execution of guest code. Always zero under
    /// a pinned ABI, where a link is a branch the host never sees.
    pub links: usize,
}

/// Enter `translation` at `ctx.rip`, and run out the chain of successors it and
/// they resolve for themselves.
///
/// The one place a host calls into lifted code. It sets the three fields an entry
/// owes the region - `image_base`, a cleared `exit_reason`, a cleared `link_hit` -
/// then enters, then follows links until one exit reports none.
///
/// `ctx.link_table` and `ctx.callret_base` are **not** set here: they are facts
/// about the *thread*, not about this entry, and a host that set them per entry
/// would be paying two stores for something that has not changed since the thread
/// started. See [`ThreadLinks::attach`].
///
/// Clearing `link_hit` here rather than having a missing probe clear it is what
/// puts that store on the once-per-entry path instead of the once-per-exit one,
/// and is why an exit with no probe at all reports "no link" by doing nothing.
///
/// # Safety
///
/// `translation` must have been compiled for this guest architecture, this
/// `RegisterContext` layout and this thread's table geometry, and every row
/// reachable from `ctx.link_table` must name code that is still mapped. This is
/// the [module](super)'s trust model, and past the address compare lifted code
/// checks none of it.
pub unsafe fn enter(translation: &Translation, ctx: &mut RegisterContext) -> Entered {
    // SAFETY: the caller's obligation.
    unsafe { enter_with(translation, ctx, |_, _| {}) }
}

/// [`enter`], reporting each link before it is taken.
///
/// `on_link` is called with the exit reason of the region that just finished and
/// the context it left behind, at the moment it turns out to have resolved a
/// successor - so once per hop, and never for the last region in a chain, whose
/// exit is the returned [`Entered::exit`]. The context comes with it because the
/// reason alone does not say what happened: a linked exit is almost always
/// `UnknownJump`, and which *kind* of transfer it was is in `exit_code`.
///
/// A host profiling which kinds of exit are being linked needs this; one that only
/// wants the total has [`Entered::links`].
///
/// # Safety
///
/// As [`enter`].
pub unsafe fn enter_with<F>(
    translation: &Translation,
    ctx: &mut RegisterContext,
    on_link: F,
) -> Entered
where
    F: FnMut(ExitReason, &RegisterContext),
{
    ctx.image_base = translation.image_base;
    ctx.exit_reason = ExitReason::None as u32;
    ctx.link_hit = 0;
    // SAFETY: the caller's obligation. `entry` and never `link_target`: under a
    // pinned ABI those are different functions and the second is not callable
    // from Rust.
    let exit = unsafe { call(translation.entry, ctx) };
    // SAFETY: as above.
    unsafe { follow_links_with(exit, ctx, on_link) }
}

/// Take every link the exiting region resolved, until one exit reports none.
///
/// `exit` is what the region that just returned reported. Under a pinned ABI this
/// returns it untouched and does not loop: a pinned link is a *branch*, the chain
/// has already run itself out, and `link_hit` is never written. That is why there
/// is no "is this host-resolved" test here - the empty loop is the test, and it
/// costs one load of a field the entry just zeroed.
///
/// # Safety
///
/// As [`enter`].
pub unsafe fn follow_links(exit: ExitReason, ctx: &mut RegisterContext) -> Entered {
    // SAFETY: the caller's obligation.
    unsafe { follow_links_with(exit, ctx, |_, _| {}) }
}

/// [`follow_links`], reporting each link before it is taken. See
/// [`enter_with`].
///
/// # Safety
///
/// As [`enter`].
pub unsafe fn follow_links_with<F>(
    exit: ExitReason,
    ctx: &mut RegisterContext,
    mut on_link: F,
) -> Entered
where
    F: FnMut(ExitReason, &RegisterContext),
{
    let mut entered = Entered { exit, links: 0 };
    while ctx.link_hit != 0 {
        // SAFETY: `link_hit` holds the address of a `BlockLinkEntry` lifted code
        // resolved for itself - either a row of the table in `ctx.link_table`,
        // which this thread owns, or a link cell in some region's own array. Both
        // are alive for as long as the host keeps the translations they name, which
        // is this module's trust model.
        let row = unsafe { *(ctx.link_hit as *const BlockLinkEntry) };
        debug_assert_eq!(
            row.guest_pc, ctx.rip,
            "a link resolved a row that is not about the PC it committed"
        );
        on_link(entered.exit, ctx);
        ctx.image_base = row.image_base;
        ctx.exit_reason = ExitReason::None as u32;
        ctx.link_hit = 0;
        // SAFETY: `code` is a translation's `link_target`, written by
        // `Translation::row`. Under a host-resolved ABI - the only one that reaches
        // this loop - `link_target` and `entry` are the same function.
        entered.exit = unsafe { call(row.code as *const u8, ctx) };
        entered.links += 1;
    }
    entered
}

/// Call compiled code at `code` with `ctx`, and decode why it came back.
///
/// # Safety
///
/// `code` must be an `extern "C" fn(*mut RegisterContext)` a host compiled for
/// this context.
unsafe fn call(code: *const u8, ctx: &mut RegisterContext) -> ExitReason {
    let entry: unsafe extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(code) };
    unsafe { entry(ctx as *mut RegisterContext) };
    ExitReason::try_from(ctx.exit_reason).unwrap_or(ExitReason::None)
}
