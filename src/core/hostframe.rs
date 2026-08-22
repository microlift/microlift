//! Host call frames: turning a guest `CALL`/`RET` pair into a real host one.
//!
//! The second shape of return prediction, and the one [`crate::core::callret`]
//! was measured against and lost to. Where the ring keeps a software shadow stack
//! and turns a `RET` into a table lookup by another name, this makes the guest's
//! call *be* a host call: an out-of-region direct `CALL` becomes an ordinary
//! `call_indirect` to the callee's compiled code, and the callee's `RET` is the
//! `return_` it already was. The host's own stack is the shadow stack.
//!
//! What that buys, and the ring did not:
//!
//! - **The return-address stack predicts it.** The `bl` and the `ret` are a
//!   matched pair the CPU recognises, which is the whole of why FEX pays for a
//!   shadow stack. The ring could not have this - Cranelift has no
//!   `ret`-flavoured indirect branch - and without it the ring was left paying
//!   FEX's bookkeeping for none of FEX's payoff.
//! - **No entry dispatch on the way back.** A linked return re-enters the calling
//!   region through its entry `Switch` and prologue; a host return lands at the
//!   instruction after the call.
//! - **No shadow stack at all**, which is the part worth saying twice. See below.
//!
//! # Why there is no ring here
//!
//! A returning region has to answer "is this return the one my caller is waiting
//! for?", and in the ring that question is asked in the *callee*, which does not
//! know its caller and therefore needs a per-frame record to compare against.
//!
//! Asked in the **caller** it needs no record at all: the address a call comes
//! back to is a lift-time constant there. So the callee always does what it
//! already did for an unlinked `RET` - sync, and `return_` with the guest's
//! target in `ctx.rip` - and the caller compares that against its own resume
//! address. A match continues in place. A mismatch is propagated by returning
//! again, which unwinds one host frame at a time until some caller recognises the
//! address or the entry thunk hands it to the host.
//!
//! That fallback is not a special case bolted on: it is exactly what a `longjmp`,
//! a hand-rolled coroutine or a computed return *should* do, and it costs
//! unwinding proportional to how far the guest actually jumped.
//!
//! # The two things that bound it
//!
//! **The host stack is finite.** Every live guest call is a host frame, so a guest
//! recursing a hundred thousand deep would exhaust it. [`HostFrameConfig::max_depth`]
//! is the ceiling; past it a call takes the ordinary linked exit instead, which
//! costs a host dispatch and nothing else.
//!
//! **A callee must exist to be called.** The address comes from a
//! [link cell](crate::core::link::LinkSite), so an edge whose target has not been
//! translated yet takes the ordinary exit until it has - the same fallback, and
//! the same reason, as every other cell.
//!
//! # It was measured, and where it stands
//!
//! Through rosetta3, against the same run with cells and no nesting:
//!
//! | guest | B2 |
//! | --- | --- |
//! | 20M iterations, two calls each, **leaf** callees | **-11.8%** |
//! | mutual recursion, 9.6M calls, depth 4 | +12.8% |
//! | mutual recursion, depth 24 | +9.5% |
//!
//! So it wins where a callee is a leaf and loses where the *caller* is itself
//! hot, and the dividing line is not depth but whether the region making the call
//! is one the guest spends time in.
//!
//! The cause is named in this crate's own notes on register-context traffic:
//! **a call is a memory fence.** Cranelift's alias analysis recognises the exit
//! store of a register the region never wrote as idempotent against the entry
//! load and deletes both; a `call_indirect` between them invalidates that, so a
//! region containing a nested call keeps the whole ~55-store sync *and* has ~55
//! values live from the post-call reload to its own exit. In a leaf-calling loop
//! that is paid once per iteration and the saved dispatch more than covers it. In
//! a recursive region it is paid per level, and the spills land on a host stack
//! that is already deep.
//!
//! Narrowing that reload was tried and was **not** where the fix was. Its loads
//! were not dead-but-broad; they were kept alive by a chain the prune left
//! behind - a region's own `RET` dispatch has an in-region arm back to its
//! earlier blocks, so every register variable is a block parameter there and the
//! reloaded values are the branch arguments feeding them.
//!
//! [`crate::core::dce`] is what fixed it, and the numbers above are already
//! after it: before, the same fixtures ran **+68%** and **+101%**, and the leaf
//! case was only -3.7%.
//!
//! What remains is **not** a fence an annotation could lift, which is where this
//! note used to point. A per-callee "writes only region R" cannot help here: the
//! callee is a lifted region that writes the whole context, so the fence a nested
//! call raises is semantically required. (That annotation was measured on its own
//! terms and dropped; see the register-context-traffic section of `CLAUDE.md`.)
//! So the remaining ~10% is unexplained, and finding it starts with a profile
//! rather than a change.
//!
//! Until then this is off by default: a clear win on leaf-calling code and a ~10%
//! loss on nested code, and which of those a guest is cannot be decided here.
//!
//! One thing that was *not* worth trying, recorded so it is not retried: moving
//! the reload to the top of the resume block, where it would be the single
//! definition reaching everything below and the loads would be plainly dead. The
//! resume point is an ordinary in-region address, so this region's own `RET`
//! dispatch can branch to that block without having synced anything, and the
//! reload would overwrite live registers with a stale context.
//!
//! What the measurement *does* settle is the comparison it was built for: against
//! the call-return ring, which was +12% to +20% everywhere, this is better on
//! every fixture tried and is the only one of the two that is ever a win.

/// How deep guest calls may nest as host frames before falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameConfig {
    /// The most host frames guest calls may occupy at once.
    ///
    /// Each is a real stack frame of a compiled region, so this is a bound on
    /// host stack use and wants setting against the thread's stack size rather
    /// than against how deep the guest is expected to go. Exceeding it is not an
    /// error and not a fault path: the call takes the linked exit it would have
    /// taken with this mechanism off.
    pub max_depth: u32,
}

impl HostFrameConfig {
    /// Nest at most `max_depth` guest calls as host frames.
    ///
    /// # Panics
    ///
    /// If `max_depth` is zero, which would make every call take the fallback and
    /// is a configuration mistake rather than a way to switch the feature off -
    /// `None` is that.
    pub fn new(max_depth: u32) -> Self {
        assert!(max_depth > 0, "a host-frame depth of zero disables nothing");
        Self { max_depth }
    }
}

/// Where a guest's `RegisterContext` keeps the host-frame depth.
///
/// Returned by [`GuestArch::host_frame_offsets`](crate::core::GuestArch::host_frame_offsets);
/// a guest that has not made room for it returns `None` and never nests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameOffsets {
    /// Byte offset of the `u32` counting how many guest calls are currently live
    /// as host frames.
    ///
    /// Incremented before a nested call and restored after it, by the region
    /// making the call - so it is correct however that call ends, including the
    /// unwinding path. A host initialises it to zero and never touches it again.
    pub depth: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "disables nothing")]
    fn a_zero_depth_is_refused_rather_than_taken_as_off() {
        HostFrameConfig::new(0);
    }
}
