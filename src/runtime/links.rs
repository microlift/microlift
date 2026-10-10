//! Per-thread block linking: the tables lifted code reads, plus entering a region and
//! following the links it resolves. Getting these wrong is silent wrong code.
//!
//! [`ThreadRuntime`](super::ThreadRuntime) is this plus a context and a loop.

use super::{CallRetStack, LinkTable, Translation};
use crate::arch::x86_64::context::RegisterContext;
use crate::core::callret::{CallRetConfig, CallRetEntry};
use crate::core::exit::ExitReason;
use crate::core::link::{BlockLinkConfig, BlockLinkEntry};

/// One guest thread's link table and call-return ring.
///
/// Compiled code reads both through raw context pointers, so both must outlive every region entered with them.
pub struct ThreadLinks {
    table: LinkTable,
    callret: Option<CallRetStack>,
}

impl ThreadLinks {
    /// Both configs must equal those every reachable region was lifted under.
    /// `None` for `callret` leaves the ring pointer null (no return prediction).
    pub fn new(link: BlockLinkConfig, callret: Option<CallRetConfig>) -> Self {
        Self {
            table: LinkTable::new(link),
            callret: callret.map(CallRetStack::new),
        }
    }

    pub fn table(&self) -> &LinkTable {
        &self.table
    }

    pub fn call_return(&self) -> Option<&CallRetStack> {
        self.callret.as_ref()
    }

    /// For `ctx.link_table`.
    pub fn link_table_ptr(&mut self) -> *mut BlockLinkEntry {
        self.table.as_mut_ptr()
    }

    /// For `ctx.callret_base`; null when there is no ring.
    pub fn callret_base_ptr(&mut self) -> *mut CallRetEntry {
        match &mut self.callret {
            Some(stack) => stack.as_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Points `ctx` at these tables; **every context entering lifted code on this thread needs it**.
    ///
    /// `callret_off` is per-entry state and left alone ([`resume_call_return`](Self::resume_call_return)).
    pub fn attach(&mut self, ctx: &mut RegisterContext) {
        ctx.link_table = self.link_table_ptr();
        ctx.callret_base = self.callret_base_ptr();
    }

    /// Starts `ctx`'s frames below `outer`'s; the ring is per thread, contexts per entry.
    ///
    /// A nested entry starting at zero would clobber outer frames (a miss, not wrong).
    /// Pass `None` at the outermost entry.
    pub fn resume_call_return(ctx: &mut RegisterContext, outer: Option<&RegisterContext>) {
        ctx.callret_off = outer.map_or(0, |outer| outer.callret_off);
    }

    /// Nulls `ctx.link_table` so the next probe declines.
    ///
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned) this is the only
    /// way to end a tail-call chain.
    pub fn detach(ctx: &mut RegisterContext) {
        ctx.link_table = std::ptr::null_mut();
    }

    /// Makes `translation` linkable at **every** address it can be entered at.
    ///
    /// Never publish a non-resume-point address: it runs the region from the top or exits with
    /// no progress. Evictions cost only a lookup.
    pub fn publish(&mut self, translation: &Translation) {
        for &pc in &translation.resume_points {
            self.table.publish(translation.row(pc));
        }
    }

    /// Makes `translation` linkable at `pc` **alone**; returns whether a row was written.
    ///
    /// Narrow form of [`publish`](Self::publish): fewer evictions in a direct-mapped table, fewer
    /// unpredicted links. `pc` must be a resume point (checked via `covers`).
    pub fn publish_at(&mut self, translation: &Translation, pc: u64) -> bool {
        if !translation.covers(pc) {
            return false;
        }
        self.table.publish(translation.row(pc));
        true
    }

    /// Drops the rows for `addresses`; returns how many were held.
    ///
    /// Per-thread half of [`CodeCache::invalidate_range`](super::CodeCache::invalidate_range);
    /// call while outside lifted code.
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
    pub exit: ExitReason,
    /// Successors resolved without the host; always zero under a pinned ABI.
    ///
    /// Returned so the shared [`CodeCache`](super::CodeCache) lock is not held across guest code.
    pub links: usize,
}

/// Enters `translation` at `ctx.rip` and follows the links it resolves.
///
/// Sets `image_base`, clears `exit_reason` and `link_hit`, enters, then follows links. Does
/// **not** set `link_table`/`callret_base`: see [`ThreadLinks::attach`].
///
/// # Safety
///
/// `translation` must match this guest architecture, context layout and table geometry, and
/// every row reachable from `ctx.link_table` must name mapped code (nothing is checked past the
/// address compare).
pub unsafe fn enter(translation: &Translation, ctx: &mut RegisterContext) -> Entered {
    // SAFETY: the caller's obligation.
    unsafe { enter_with(translation, ctx, |_, _| {}) }
}

/// [`enter`], calling `on_link(exit, ctx)` once per link taken (never for the final region).
///
/// A linked exit is usually `UnknownJump`; the transfer kind is in `exit_code`.
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
    // SAFETY: the caller's obligation. `entry`, not `link_target`: they differ under a pinned ABI.
    let exit = unsafe { call(translation.entry, ctx) };
    // SAFETY: as above.
    unsafe { follow_links_with(exit, ctx, on_link) }
}

/// Takes every link the exiting region resolved until one exit reports none.
///
/// Under a pinned ABI this returns `exit` untouched (`link_hit` is never written).
///
/// # Safety
///
/// As [`enter`].
pub unsafe fn follow_links(exit: ExitReason, ctx: &mut RegisterContext) -> Entered {
    // SAFETY: the caller's obligation.
    unsafe { follow_links_with(exit, ctx, |_, _| {}) }
}

/// [`follow_links`] with the [`enter_with`] callback.
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
        // SAFETY: `link_hit` is a table row or a region's link cell, alive while the host keeps its translations.
        let row = unsafe { *(ctx.link_hit as *const BlockLinkEntry) };
        debug_assert_eq!(
            row.guest_pc, ctx.rip,
            "a link resolved a row that is not about the PC it committed"
        );
        on_link(entered.exit, ctx);
        ctx.image_base = row.image_base;
        ctx.exit_reason = ExitReason::None as u32;
        ctx.link_hit = 0;
        // SAFETY: `code` is `link_target` from `Translation::row`; equals `entry` under HostResume.
        entered.exit = unsafe { call(row.code as *const u8, ctx) };
        entered.links += 1;
    }
    entered
}

/// # Safety
///
/// `code` must be an `extern "C" fn(*mut RegisterContext)` compiled for this context.
unsafe fn call(code: *const u8, ctx: &mut RegisterContext) -> ExitReason {
    let entry: unsafe extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(code) };
    unsafe { entry(ctx as *mut RegisterContext) };
    ExitReason::try_from(ctx.exit_reason).unwrap_or(ExitReason::None)
}
