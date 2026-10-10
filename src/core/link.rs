//! Block linking: a translated region resolves its own successor from a host table.
//!
//! Lifted code checks only that a row's `guest_pc` matches. The host must guarantee:
//!
//! - `code` is a translation enterable at `guest_pc` with the row's `image_base`.
//! - `code` stays mapped while the row is reachable; overwriting the row retracts it.
//! - The table has `entries` rows and is at least 8-byte aligned (loads are `notrap`).
//!
//! A row naming an older translation of unchanged bytes is harmless, so a
//! direct-mapped collision is a miss, not an error. A row outliving its bytes
//! (self-modifying code, unmapping, eviction) is not: the host retracts it via
//! [`LiftedCode::covered`] and [`LiftedCode::data_dependencies`].
//! [`crate::runtime::CodeCache`] implements this protocol.
//!
//! [`LiftedCode::covered`]: crate::core::LiftedCode::covered
//! [`LiftedCode::data_dependencies`]: crate::core::LiftedCode::data_dependencies

use crate::core::pin::PinnedAbi;

/// One row of the host's link table: `guest_pc` is compiled at `code` and wants `image_base`.
///
/// `#[repr(C)]`, exactly 32 bytes: lifted code indexes it with a shift.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockLinkEntry {
    /// Run-time guest address (same space as `ctx.rip`). Zero marks an empty row.
    pub guest_pc: u64,
    /// Host address of the compiled entry point.
    pub code: u64,
    /// Value the host must put in `ctx.image_base` before entering `code`.
    pub image_base: u64,
    /// Padding; never read.
    pub reserved: u64,
}

/// The size of one [`BlockLinkEntry`], as lifted code computes it.
pub const ENTRY_SIZE: u64 = 32;

/// Byte offset of [`BlockLinkEntry::code`]; read only under [`LinkDispatch::Pinned`].
pub const CODE_OFFSET: i32 = 8;

/// Byte offset of [`BlockLinkEntry::image_base`]; read only under [`LinkDispatch::Pinned`].
pub const IMAGE_BASE_OFFSET: i32 = 16;

/// A dedicated cell for an exit whose destination is known at lift time.
///
/// A [`BlockLinkEntry`] the host installs at a lifter-chosen index instead of
/// [`BlockLinkConfig::slot`]. No hashing, no collisions; `code == 0` means not
/// installed, so a zeroed array is valid and clearing a cell retracts the link.
/// `guest_pc` is never read by lifted code.
///
/// Reported as [`LiftedCode::link_sites`], in cell order.
///
/// [`LiftedCode::link_sites`]: crate::core::LiftedCode::link_sites
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkSite {
    /// Lift-time VA; relocate it as `runtime_addr` would before matching.
    pub target: u64,
    /// Why the cell exists. Lifted code treats every kind identically.
    pub kind: LinkSiteKind,
}

/// What kind of transfer a [`LinkSite`] serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkSiteKind {
    /// Branch, call or fallthrough leaving the region for a constant address.
    StaticEdge,
    /// A call's return address. Reserved for the call-return stack; not yet emitted.
    CallReturn,
}

/// Symbol name of a region's zero-initialised link-cell array.
pub const CELL_SYMBOL: &str = "lifted_cells";

/// Geometry of the host's link table, baked into the emitted probe.
///
/// Changing either field invalidates translations lifted under the old value.
/// [`slot`](Self::slot) is the single definition of where a PC lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkConfig {
    /// Row count; a power of two so the index is a mask.
    pub entries: u32,
    /// Right shift applied to a PC before masking (low bits of code addresses carry little entropy).
    pub shift: u32,
    /// What the probe does once it finds a row.
    pub dispatch: LinkDispatch,
}

/// What a region does with a resolved successor. The probe is identical up to the address compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkDispatch {
    /// Publish the row in [`BlockLinkOffsets::hit`] and return; the host calls its `code`.
    HostResume,
    /// Tail-call the row's `code` with [`PinnedAbi`] registers held in machine registers.
    ///
    /// Falls back to [`HostResume`](Self::HostResume) if the ABI is empty or the
    /// guest cannot type its slots. See [`crate::core::pin`].
    Pinned(PinnedAbi),
}

impl BlockLinkConfig {
    /// A table of `entries` rows indexed on `pc >> 4`.
    ///
    /// # Panics
    ///
    /// If `entries` is not a power of two.
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

    /// The same table, resolved by branching to the successor with `abi` held in registers.
    ///
    /// Every region reachable through the table must be lifted under this exact config.
    #[must_use]
    pub fn pinned(mut self, abi: PinnedAbi) -> Self {
        self.dispatch = LinkDispatch::Pinned(abi);
        self
    }

    /// The pinned ABI requested, if any. The guest may still be unable to supply it.
    pub fn pinned_abi(&self) -> Option<PinnedAbi> {
        match self.dispatch {
            LinkDispatch::HostResume => None,
            LinkDispatch::Pinned(abi) => (!abi.is_empty()).then_some(abi),
        }
    }

    /// The row `pc` belongs to; shared by emitted code and the host.
    pub fn slot(&self, pc: u64) -> usize {
        ((pc >> self.shift) as usize) & (self.entries as usize - 1)
    }

    pub(crate) fn mask(&self) -> i64 {
        self.entries as i64 - 1
    }
}

/// Where a guest's `RegisterContext` keeps the two fields block linking needs.
///
/// A guest without room for them returns `None` from
/// [`GuestArch::block_link_offsets`](crate::core::GuestArch::block_link_offsets) and never links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkOffsets {
    /// Byte offset of the pointer to the host's [`BlockLinkEntry`] table; null disables linking.
    pub table: i32,
    /// Byte offset of the word the probe writes the matching entry's address into.
    ///
    /// Zero means no link. The host clears it before entering; dereference it
    /// rather than assuming it lies inside [`table`](Self::table).
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
        for pc in [0u64, 0x1000, 0x1004, 0x1_0000_0000, 0xdead_beef] {
            assert_eq!(config.slot(pc), ((pc >> 4) & 8191) as usize);
            assert!(config.slot(pc) < 8192);
        }
    }

    #[test]
    fn the_shift_is_what_spreads_aligned_functions_out() {
        let config = BlockLinkConfig::new(16);
        assert_ne!(config.slot(0x400), config.slot(0x410));
        assert_eq!(config.slot(0x400), config.slot(0x40f));
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn a_non_power_of_two_table_is_refused_at_construction() {
        BlockLinkConfig::new(1000);
    }
}
