//! Block linking: letting a translated region resolve its own successor.
//!
//! The lifted code performs **no validation** beyond the address compare, so for
//! every [`BlockLinkEntry`] it publishes the host must guarantee:
//!
//! - `code` is a compiled translation that *can be entered at* `guest_pc`
//!   ([`LiftedCode::resume_points`] is the exact set), with the `image_base` that
//!   translation wants (`pc = image_base + (ip - lift_base)`).
//! - `code` stays mapped while the entry is reachable. Overwriting it is the only
//!   way to retract it, and nothing here is atomic - one table per guest thread
//!   is the intended shape.
//! - The table is `entries` rows long and at least 8-byte aligned; the probe's
//!   loads are `notrap`/`aligned`, so a misaligned table is undefined.
//!
//! An entry may be stale in the one way that does not matter: naming an older
//! translation that also covers the PC. Two translations of the same address are
//! interchangeable, which is what makes a direct-mapped table with no
//! invalidation sound.
//!
//! [`LinkDispatch`] decides what a region does with a match; everything above
//! holds for both shapes.
//!
//! [`RegisterContext`]: crate::arch::x86_64::context::RegisterContext
//! [`LifterConfig::block_link`]: crate::core::LifterConfig::block_link
//! [`LiftedCode::resume_points`]: crate::core::LiftedCode::resume_points

use crate::core::pin::PinnedAbi;

/// One row of a host's block-link table: *"guest `guest_pc` is compiled, it is
/// at `code`, and it wants `image_base`"*.
///
/// `#[repr(C)]` and exactly 32 bytes because lifted code indexes it with a
/// shift - the layout is ABI, not an implementation detail. The trailing word
/// is padding and is never read.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockLinkEntry {
    /// The guest address this row is about, as a **run-time** virtual address -
    /// the same space `ctx.rip` is in. Zero for an empty row, which is safe
    /// because a guest PC of zero is not code anyone links to.
    pub guest_pc: u64,
    /// Host address of the compiled entry point for `guest_pc`.
    pub code: u64,
    /// What the host must put in `ctx.image_base` before entering `code`.
    pub image_base: u64,
    /// Padding to 32 bytes. Never read by lifted code.
    pub reserved: u64,
}

/// The size of one [`BlockLinkEntry`], as lifted code computes it.
pub const ENTRY_SIZE: u64 = 32;

/// Byte offset of [`BlockLinkEntry::code`] within a row, as lifted code loads
/// it. Read only under [`LinkDispatch::Pinned`], where the region branches to
/// the code itself instead of naming the row for the host.
pub const CODE_OFFSET: i32 = 8;

/// Byte offset of [`BlockLinkEntry::image_base`] within a row. Read only under
/// [`LinkDispatch::Pinned`], for the same reason: there is no host on that path
/// to set it from the row.
pub const IMAGE_BASE_OFFSET: i32 = 16;

/// The shape of the host's link table, as the lifter must know it.
///
/// Both fields are baked into the emitted probe, so changing either means
/// discarding the translations lifted under the old value. Keeping the
/// geometry here makes [`BlockLinkConfig::slot`] the single definition of
/// where a PC lives, so the host fills the row lifted code reads by
/// construction rather than by agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkConfig {
    /// How many rows the table has. Must be a power of two, so the index is a
    /// mask rather than a division.
    pub entries: u32,
    /// How far to shift a guest PC right before masking.
    ///
    /// The low bits of a code address are its least distinguishing (functions are
    /// aligned), so dropping them spreads a working set over the table. Four is
    /// the usual answer.
    pub shift: u32,
    /// What the probe does once it has found a row. See [`LinkDispatch`].
    pub dispatch: LinkDispatch,
}

/// What a region does with the successor it resolved: hand it to the host, or
/// branch to it.
///
/// **The boundary between the two shapes of block linking.** Both use the same
/// table, geometry and probe up to the address compare, and differ only in the
/// three instructions after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkDispatch {
    /// Publish the matching row in [`BlockLinkOffsets::hit`] and return.
    ///
    /// The host reads the row and calls the code it names, so the register file
    /// still goes out to the context and back on every region entry - what is
    /// deleted is the host's *decision*, not its spill. A region under this mode
    /// is an ordinary `extern "C" fn(*mut RegisterContext)`.
    HostResume,
    /// Tail-call the code the row names, carrying the pinned registers with it.
    ///
    /// The host does not see the link: the successor is entered by a
    /// `return_call_indirect` with the guest state in [`PinnedAbi`] still in
    /// machine registers, so the fill and spill are paid once per *chain* instead
    /// of once per region. See [`crate::core::pin`].
    ///
    /// Inert, falling back to [`HostResume`](Self::HostResume), when the ABI is
    /// empty or the guest frontend cannot type its slots.
    Pinned(PinnedAbi),
}

impl BlockLinkConfig {
    /// A table of `entries` rows indexed on `pc >> 4`.
    ///
    /// # Panics
    ///
    /// If `entries` is not a power of two: the lifter emits a mask, which for a
    /// non-power-of-two size would silently read outside the table.
    pub fn new(entries: u32) -> Self {
        assert!(
            entries.is_power_of_two(),
            "a block-link table must have a power-of-two row count, not {entries}"
        );
        Self {
            entries,
            shift: 4,
            dispatch: LinkDispatch::HostResume,
        }
    }

    /// The same table, resolved by **branching** to the successor with `abi`
    /// held in registers rather than by returning the row to the host.
    ///
    /// Every region reachable through the table must be lifted under this exact
    /// config - the geometry *and* the ABI. See [`crate::core::pin`].
    #[must_use]
    pub fn pinned(mut self, abi: PinnedAbi) -> Self {
        self.dispatch = LinkDispatch::Pinned(abi);
        self
    }

    /// The pinned ABI this config asks for, if any.
    ///
    /// Says nothing about whether the *guest* can supply it - that is
    /// [`crate::core::pinned_abi_for`], the question the engine actually asks.
    pub fn pinned_abi(&self) -> Option<PinnedAbi> {
        match self.dispatch {
            LinkDispatch::HostResume => None,
            LinkDispatch::Pinned(abi) => (!abi.is_empty()).then_some(abi),
        }
    }

    /// The row `pc` belongs to. **The** definition, shared by the lifter's
    /// emitted arithmetic and by the host filling the table.
    pub fn slot(&self, pc: u64) -> usize {
        ((pc >> self.shift) as usize) & (self.entries as usize - 1)
    }

    /// The mask lifted code applies after the shift.
    pub(crate) fn mask(&self) -> i64 {
        self.entries as i64 - 1
    }
}

/// Where a guest's `RegisterContext` keeps the two fields block linking needs.
///
/// Returned by [`GuestArch::block_link_offsets`](crate::core::GuestArch::block_link_offsets);
/// a guest that has not made room for them returns `None` and never links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkOffsets {
    /// Byte offset of the pointer to the host's [`BlockLinkEntry`] table. Read
    /// on every linkable exit; a null pointer disables linking for that entry,
    /// so a host that has not set it up is not a special case.
    pub table: i32,
    /// Byte offset of the word the probe writes the matching entry's **address**
    /// into. Zero means "no link; service this exit yourself".
    ///
    /// The host clears it before entering rather than the probe clearing it on a
    /// miss: that puts the store on the once-per-entry path, and lets an exit with
    /// no probe at all report "no link" without knowing linking exists.
    pub hit: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_is_the_size_lifted_code_indexes_by() {
        assert_eq!(std::mem::size_of::<BlockLinkEntry>(), ENTRY_SIZE as usize);
        assert_eq!(std::mem::offset_of!(BlockLinkEntry, guest_pc), 0);
        assert_eq!(
            std::mem::offset_of!(BlockLinkEntry, code),
            CODE_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(BlockLinkEntry, image_base),
            IMAGE_BASE_OFFSET as usize
        );
    }

    #[test]
    fn slots_are_the_masked_shift_the_lifter_emits() {
        let config = BlockLinkConfig::new(8192);
        assert_eq!(config.mask(), 8191);
        // The arithmetic the probe performs, written out independently.
        for pc in [0u64, 0x1000, 0x1004, 0x1_0000_0000, 0xdead_beef] {
            assert_eq!(config.slot(pc), ((pc >> 4) & 8191) as usize);
            assert!(config.slot(pc) < 8192);
        }
    }

    #[test]
    fn the_shift_is_what_spreads_aligned_functions_out() {
        let config = BlockLinkConfig::new(16);
        // Sixteen bytes apart is one slot apart; within sixteen bytes is a
        // collision, which is a miss and not an error.
        assert_ne!(config.slot(0x400), config.slot(0x410));
        assert_eq!(config.slot(0x400), config.slot(0x40f));
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn a_non_power_of_two_table_is_refused_at_construction() {
        BlockLinkConfig::new(1000);
    }
}
