//! Per-thread block-link table probed by lifted code.

use crate::core::link::{BlockLinkConfig, BlockLinkEntry};

/// One guest thread's [`BlockLinkEntry`] array: [`BlockLinkConfig::ways`] rows per set,
/// starting at [`BlockLinkConfig::set`], most recently published first.
///
/// Not atomic: only the owning thread writes, and only outside lifted code.
pub struct LinkTable {
    rows: Box<[BlockLinkEntry]>,
    config: BlockLinkConfig,
}

/// `n` empty rows, from zeroed memory rather than a row-by-row fill: a large table is
/// then demand-zero pages, and a thread that never links much never commits them.
fn zeroed_rows(n: usize) -> Box<[BlockLinkEntry]> {
    if n == 0 {
        return Box::new([]);
    }
    let layout =
        std::alloc::Layout::array::<BlockLinkEntry>(n).expect("a link table fits in memory");
    // SAFETY: `layout` is non-zero-sized. An all-zero `BlockLinkEntry` is four zero
    // `u64`s, which is a valid value and exactly `Default` (the empty row), so the
    // allocation is `n` initialised rows and may become a boxed slice of them.
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout).cast::<BlockLinkEntry>();
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n))
    }
}

impl LinkTable {
    /// `config` must equal the one every region reachable through this table was lifted under.
    pub fn new(config: BlockLinkConfig) -> Self {
        Self {
            rows: zeroed_rows(config.entries as usize),
            config,
        }
    }

    pub fn config(&self) -> BlockLinkConfig {
        self.config
    }

    /// The rows `pc` may occupy.
    fn set(&self, pc: u64) -> std::ops::Range<usize> {
        let first = self.config.set(pc);
        first..first + self.config.ways as usize
    }

    /// Installs `entry` in place if its PC already has a row, else at the front of its set,
    /// silently evicting the set's oldest row (a row is a hint; watch
    /// [`CacheStats::miss_rate`](super::CacheStats::miss_rate)).
    pub fn publish(&mut self, entry: BlockLinkEntry) {
        let set = self.set(entry.guest_pc);
        if let Some(row) = self.rows[set.clone()]
            .iter_mut()
            .find(|row| row.guest_pc == entry.guest_pc)
        {
            *row = entry;
            return;
        }
        // Older rows move back one way; whatever was last falls out.
        self.rows[set.clone()].rotate_right(1);
        self.rows[set.start] = entry;
    }

    /// Removes the row for `pc` if its set holds one; returns whether cleared.
    pub fn retract(&mut self, pc: u64) -> bool {
        let set = self.set(pc);
        match self.rows[set].iter_mut().find(|row| row.guest_pc == pc) {
            Some(row) => {
                *row = BlockLinkEntry::default();
                true
            }
            None => false,
        }
    }

    /// Drops every row; safe at any time.
    pub fn clear(&mut self) {
        self.rows.fill(BlockLinkEntry::default());
    }

    pub fn get(&self, pc: u64) -> Option<&BlockLinkEntry> {
        self.rows[self.set(pc)]
            .iter()
            .find(|row| row.guest_pc == pc)
    }

    /// For `ctx.link_table`. Null declines to link and, under
    /// [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned), ends a tail-call chain
    /// ([`ThreadRuntime::detach_table`](super::ThreadRuntime::detach_table)).
    pub fn as_mut_ptr(&mut self) -> *mut BlockLinkEntry {
        self.rows.as_mut_ptr()
    }

    /// Occupied rows; O(n).
    pub fn occupancy(&self) -> usize {
        self.rows.iter().filter(|r| r.guest_pc != 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pc: u64) -> BlockLinkEntry {
        BlockLinkEntry {
            guest_pc: pc,
            code: pc ^ 0xc0de,
            image_base: 0,
            reserved: 0,
        }
    }

    /// Two PCs one table-span apart share a set; one way makes them evict each other.
    #[test]
    fn two_ways_hold_two_colliding_pcs() {
        // The direct table's span is a multiple of the two-way one's, so these collide in both.
        let (a, b) = (0x1000u64, 0x1000 + BlockLinkConfig::new(16).span());
        let mut direct = LinkTable::new(BlockLinkConfig::new(16));
        direct.publish(row(a));
        direct.publish(row(b));
        assert!(direct.get(a).is_none(), "direct mapping evicts");

        let mut table = LinkTable::new(BlockLinkConfig::new(16).with_ways(2));
        table.publish(row(a));
        table.publish(row(b));
        assert_eq!(table.get(a), Some(&row(a)));
        assert_eq!(table.get(b), Some(&row(b)));
    }

    #[test]
    fn a_full_set_evicts_its_oldest_row() {
        let span = BlockLinkConfig::new(16).with_ways(2).span();
        let mut table = LinkTable::new(BlockLinkConfig::new(16).with_ways(2));
        for pc in [0x1000, 0x1000 + span, 0x1000 + 2 * span] {
            table.publish(row(pc));
        }
        assert!(table.get(0x1000).is_none());
        assert!(table.get(0x1000 + span).is_some());
        assert!(table.get(0x1000 + 2 * span).is_some());
        // The newest row is the first way, where the probe looks first.
        assert_eq!(
            table.rows[table.config.set(0x1000)].guest_pc,
            0x1000 + 2 * span
        );
    }

    #[test]
    fn republishing_a_pc_replaces_its_row_in_place() {
        let span = BlockLinkConfig::new(16).with_ways(2).span();
        let mut table = LinkTable::new(BlockLinkConfig::new(16).with_ways(2));
        table.publish(row(0x1000));
        table.publish(row(0x1000 + span));
        let mut newer = row(0x1000);
        newer.code = 1;
        table.publish(newer);
        assert_eq!(table.get(0x1000), Some(&newer));
        assert!(
            table.get(0x1000 + span).is_some(),
            "not a second copy that evicts"
        );
    }

    #[test]
    fn a_new_table_is_all_empty_rows() {
        let table = LinkTable::new(BlockLinkConfig::new(1 << 16).with_ways(4));
        assert_eq!(table.rows.len(), 1 << 16);
        assert_eq!(table.occupancy(), 0);
        assert!(
            table
                .rows
                .iter()
                .all(|row| *row == BlockLinkEntry::default())
        );
    }

    #[test]
    fn retract_clears_only_the_named_pc() {
        let span = BlockLinkConfig::new(16).with_ways(2).span();
        let mut table = LinkTable::new(BlockLinkConfig::new(16).with_ways(2));
        table.publish(row(0x1000));
        table.publish(row(0x1000 + span));
        assert!(table.retract(0x1000));
        assert!(!table.retract(0x1000));
        assert!(table.get(0x1000).is_none());
        assert!(table.get(0x1000 + span).is_some());
    }
}
