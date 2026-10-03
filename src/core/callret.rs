/// One frame: *"if the guest returns to `guest_pc`, the row naming where to go
/// is `cell`."*
///
/// `#[repr(C)]` and exactly 16 bytes because lifted code indexes the ring with a
/// mask.
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
