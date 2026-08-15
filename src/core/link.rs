//! Block linking: letting a translated region resolve its own successor.
//!
//! The lifted code performs **no validation** beyond the address compare: a
//! matching [`BlockLinkEntry`] is taken to name executable code that may be
//! entered at that guest PC. So the host must guarantee, for every entry it
//! publishes:
//!
//! - `code` is the entry point of a compiled translation that *can be entered
//!   at* `guest_pc` (microlift's [`LiftedCode::resume_points`] is the exact
//!   set), and `image_base` is the value that translation wants - the lift base
//!   of the region, reconciled as `pc = image_base + (ip - lift_base)`.
//! - `code` stays mapped for as long as the entry is reachable. Overwriting the
//!   entry is the only way to retract it, and nothing here is atomic: a table
//!   must not be written while another thread's guest code can read it. One
//!   table per guest thread is the intended shape.
//! - The table is `entries` rows long and at least 8-byte aligned, which a
//!   `[BlockLinkEntry]` is by construction - the probe's loads are `notrap` and
//!   `aligned`, so a misaligned table is undefined rather than slow.
//!
//! An entry is otherwise free to be stale in the one way that does not matter:
//! naming an older translation that also covers the PC. Two translations of
//! the same address are interchangeable, which is what makes a direct-mapped
//! table with no invalidation sound.
//!
//! [`RegisterContext`]: crate::arch::x86_64::context::RegisterContext
//! [`LifterConfig::block_link`]: crate::core::LifterConfig::block_link
//! [`LiftedCode::resume_points`]: crate::core::LiftedCode::resume_points

/// One row of a host's block-link table: *"guest `guest_pc` is compiled, it is
/// at `code`, and it wants `image_base`"*.
///
/// `#[repr(C)]` and exactly 32 bytes because lifted code indexes it with a
/// shift: the layout is part of the ABI between a translation and its host, not
/// an implementation detail. The trailing word is padding to that power of two
/// and is never read by lifted code.
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

/// The shape of the host's link table, as the lifter must know it.
///
/// Both fields are baked into the emitted probe, so a host that changes either
/// one has to discard the translations lifted under the old value. Keeping the
/// table's geometry here - rather than letting each side pick its own - is what
/// makes [`BlockLinkConfig::slot`] the single definition of where a PC lives:
/// the host fills the row that lifted code will read, by construction rather
/// than by agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkConfig {
    /// How many rows the table has. Must be a power of two, so the index is a
    /// mask rather than a division.
    pub entries: u32,
    /// How far to shift a guest PC right before masking.
    ///
    /// The low bits of a code address are the least distinguishing ones it has -
    /// functions are aligned - so dropping them spreads a working set over the
    /// table instead of leaving most of it empty. Four is the usual answer.
    pub shift: u32,
}

impl BlockLinkConfig {
    /// A table of `entries` rows indexed on `pc >> 4`.
    ///
    /// # Panics
    ///
    /// If `entries` is not a power of two. This is a programming error rather
    /// than a run-time condition: the lifter emits a mask, and a mask for a
    /// non-power-of-two size would silently read outside the table.
    pub fn new(entries: u32) -> Self {
        assert!(
            entries.is_power_of_two(),
            "a block-link table must have a power-of-two row count, not {entries}"
        );
        Self { entries, shift: 4 }
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
/// a guest that has not made room for them returns `None` and never links,
/// which is why every other part of this is written without reference to a
/// particular guest's context layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLinkOffsets {
    /// Byte offset of the pointer to the host's [`BlockLinkEntry`] table. Read
    /// on every linkable exit; a null pointer disables linking for that entry,
    /// so a host that has not set it up is not a special case.
    pub table: i32,
    /// Byte offset of the word the probe writes the matching entry's **address**
    /// into. Zero means "no link; service this exit yourself".
    ///
    /// The host clears it before entering, rather than the probe clearing it on
    /// a miss: that puts the store on the path taken once per entry instead of
    /// once per exit, and makes an exit with no probe at all (a syscall, an
    /// import call) report "no link" without having to know that linking exists.
    pub hit: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_is_the_size_lifted_code_indexes_by() {
        assert_eq!(std::mem::size_of::<BlockLinkEntry>(), ENTRY_SIZE as usize);
        assert_eq!(std::mem::offset_of!(BlockLinkEntry, guest_pc), 0);
        assert_eq!(std::mem::offset_of!(BlockLinkEntry, code), 8);
        assert_eq!(std::mem::offset_of!(BlockLinkEntry, image_base), 16);
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
