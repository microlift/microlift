//! The call-return stack: predicting where a guest `RET` goes.
//!
//! A guest `RET` is the most common linkable exit in a real program, and it is
//! the one the [table probe](crate::core::link) serves worst: the return address
//! is a runtime value, so there is nothing to specialize, and every return pays
//! a shift, a mask, a scale, an add, a load and a compare. FEX answers this with
//! a shadow stack of `<GuestReturnRIP, HostReturnPC>` pairs pushed at each call
//! and popped at each return (`BranchOps.cpp:164-196`). This is the same idea,
//! with the one substitution Cranelift forces.
//!
//! FEX pushes `adr TMP1, &l_CallReturn` - the address of a label inside the
//! function being emitted - and returns with `ret(TMP2)`, an indirect branch
//! the CPU's return-address-stack predictor accepts because the matching `bl`
//! trained it. None of that is expressible here. Cranelift can materialize the
//! address of a function or a data symbol and nothing finer; it has no
//! `ret`-flavoured indirect branch, and no way to reach an arbitrary host
//! address except a call or a tail call.
//!
//! So the second word is a [link cell](crate::core::link::LinkSite) instead
//! of a host PC: the row naming the region that contains the return address,
//! installed by the host like any other. The region's entry dispatch makes up
//! the difference, which works because a return address is always a *call resume
//! point* and therefore already an arm of that dispatch - see
//! [`VmResumptionPoints::ExitOnly`](crate::core::VmResumptionPoints).
//!
//! What this saves us is the lookup: one load and one compare instead of the probe's
//! six, and no collisions. What it does not do is FEX's branch prediction.
//! Do not read a FEX benchmark and expect it here.
//!
//! Storing a cell rather than a raw host PC has a second consequence, and it is
//! the reason there is nothing here resembling FEX's `Core.cpp:992`: FEX must
//! wipe its entire shadow stack on any successful invalidation, because the
//! stack holds addresses of code that may have just been freed. A cell is
//! retracted by the same reverse map as every other link, so this stack is never
//! wiped.
//!
//! # Imbalance design
//!
//! 1. Overflow is a miss. Wrapping overwrites the oldest frame. When that
//!    frame's `RET` eventually arrives it reads whatever is there, fails the
//!    `guest_pc` compare, and falls through to the probe.
//! 2. Underflow cannot happen. [`CallRetEntry::guest_pc`] advances the
//!    pointer only on a match, and an untouched frame holds zero, which no
//!    guest returns to - the same argument [`crate::core::link`] makes for an
//!    empty row.
//! 3. A mispaired pop is still correct. A frame is used only when its
//!    `guest_pc` matches, and a cell for a guest PC names a translation
//!    enterable there. That is the interchangeability invariant again, applied
//!    to a second table.
//!
//! So a guest that unbalances its calls and returns - `longjmp`, a hand-rolled
//! coroutine, a `push`/`ret` thunk - costs hit rate and nothing else.

/// One frame: *"if the guest returns to `guest_pc`, the row naming where to go
/// is `cell`."*
///
/// `#[repr(C)]` and exactly 16 bytes because lifted code indexes the ring with a
/// mask - the layout is ABI, not an implementation detail.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CallRetEntry {
    /// The return address the matching `RET` must pop for this frame to apply,
    /// as a run-time VA - the same space `ctx.rip` is in.
    ///
    /// Zero in an untouched frame, which is what makes an unbalanced stack miss
    /// rather than lie.
    pub guest_pc: u64,
    /// Address of the [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) for
    /// `guest_pc` - a [`LinkSiteKind::CallReturn`] cell belonging to the region
    /// that pushed this frame.
    ///
    /// Zero, or naming an uninstalled cell, is safe: the pop reads the cell's
    /// `code` and treats zero as "not linked", exactly as a static edge does.
    ///
    /// [`LinkSiteKind::CallReturn`]: crate::core::link::LinkSiteKind::CallReturn
    pub cell: u64,
}

/// The size of one [`CallRetEntry`], as lifted code computes it.
pub const FRAME_SIZE: u64 = 16;

/// Byte offset of [`CallRetEntry::cell`] within a frame.
pub const CELL_OFFSET: i32 = 8;

/// The shape of the host's call-return ring, as the lifter must know it.
///
/// Baked into every region compiled under it, like
/// [`BlockLinkConfig`](crate::core::link::BlockLinkConfig): changing it means
/// discarding those translations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallRetConfig {
    /// How many frames the ring holds. Must be a power of two - the wrap is a
    /// mask, and that is what makes overflow a miss instead of a stray write.
    ///
    /// Sized against expected call depth: a guest that recurses deeper than this
    /// loses its oldest frames and pays a probe for them on the way back out.
    pub entries: u32,
}

impl CallRetConfig {
    /// A ring of `entries` frames.
    ///
    /// # Panics
    ///
    /// If `entries` is not a power of two, for the same reason
    /// [`BlockLinkConfig::new`](crate::core::link::BlockLinkConfig::new) panics:
    /// the lifter emits a mask, and a non-power-of-two size would let it index
    /// outside the ring.
    pub fn new(entries: u32) -> Self {
        assert!(
            entries.is_power_of_two(),
            "a call-return ring must have a power-of-two frame count, not {entries}"
        );
        Self { entries }
    }

    /// The ring's size in bytes - what a host must allocate, and the modulus the
    /// emitted mask applies.
    pub fn bytes(&self) -> u64 {
        self.entries as u64 * FRAME_SIZE
    }

    /// The mask lifted code applies to a byte offset after adding or
    /// subtracting a frame.
    pub(crate) fn mask(&self) -> i64 {
        (self.bytes() - 1) as i64
    }
}

/// Where a guest's `RegisterContext` keeps the two fields the ring needs.
///
/// Returned by [`GuestArch::call_return_offsets`](crate::core::GuestArch::call_return_offsets);
/// a guest that has not made room for them returns `None` and never predicts a
/// return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallRetOffsets {
    /// Byte offset of the pointer to the host's ring. A null pointer disables
    /// prediction for that thread, so a host that has not set one up is not a
    /// special case - exactly as with
    /// [`BlockLinkOffsets::table`](crate::core::link::BlockLinkOffsets::table).
    pub base: i32,
    /// Byte offset of the **byte offset** of the top frame within the ring.
    ///
    /// A byte offset rather than an index so the emitted code masks once and
    /// adds, instead of masking, scaling and adding. Any value is valid on
    /// entry, including a host's zero: the ring has no invalid state.
    pub offset: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_the_size_lifted_code_indexes_by() {
        assert_eq!(std::mem::size_of::<CallRetEntry>(), FRAME_SIZE as usize);
        assert_eq!(std::mem::offset_of!(CallRetEntry, guest_pc), 0);
        assert_eq!(
            std::mem::offset_of!(CallRetEntry, cell),
            CELL_OFFSET as usize
        );
    }

    #[test]
    fn the_mask_wraps_the_ring_and_nothing_wider() {
        let config = CallRetConfig::new(8);
        assert_eq!(config.bytes(), 128);
        assert_eq!(config.mask(), 127);
        // Pushing eight frames from zero returns to zero: the ring is exactly
        // as big as the mask says, so overflow overwrites rather than escapes.
        let mut off: i64 = 0;
        for _ in 0..8 {
            off = (off - FRAME_SIZE as i64) & config.mask();
        }
        assert_eq!(off, 0);
    }

    #[test]
    fn every_offset_the_mask_can_produce_is_inside_the_ring() {
        let config = CallRetConfig::new(4);
        for raw in [0i64, 1, -16, 16, 1 << 40, -(1 << 40)] {
            let off = raw & config.mask();
            assert!(
                off >= 0 && (off as u64) < config.bytes(),
                "{raw} masked to {off}, outside a {}-byte ring",
                config.bytes()
            );
        }
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn a_non_power_of_two_ring_is_refused_at_construction() {
        CallRetConfig::new(100);
    }
}
