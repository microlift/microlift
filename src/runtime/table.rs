//! Per-thread block-link table probed by lifted code.

use crate::core::link::{BlockLinkConfig, BlockLinkEntry};

/// One guest thread's direct-mapped [`BlockLinkEntry`] array, indexed by [`BlockLinkConfig::slot`].
///
/// Not atomic: only the owning thread writes, and only outside lifted code.
pub struct LinkTable {
    rows: Box<[BlockLinkEntry]>,
    config: BlockLinkConfig,
}

impl LinkTable {
    /// `config` must equal the one every region reachable through this table was lifted under.
    pub fn new(config: BlockLinkConfig) -> Self {
        Self {
            rows: vec![BlockLinkEntry::default(); config.entries as usize].into_boxed_slice(),
            config,
        }
    }

    pub fn config(&self) -> BlockLinkConfig {
        self.config
    }

    /// Installs `entry`, silently evicting its slot's occupant (a row is a hint; watch
    /// [`CacheStats::miss_rate`](super::CacheStats::miss_rate)).
    pub fn publish(&mut self, entry: BlockLinkEntry) {
        let slot = self.config.slot(entry.guest_pc);
        self.rows[slot] = entry;
    }

    /// Removes the row for `pc` only if the slot's row is actually for `pc`; returns whether cleared.
    pub fn retract(&mut self, pc: u64) -> bool {
        let slot = self.config.slot(pc);
        if self.rows[slot].guest_pc != pc {
            return false;
        }
        self.rows[slot] = BlockLinkEntry::default();
        true
    }

    /// Drops every row; safe at any time.
    pub fn clear(&mut self) {
        self.rows.fill(BlockLinkEntry::default());
    }

    pub fn get(&self, pc: u64) -> Option<&BlockLinkEntry> {
        let row = &self.rows[self.config.slot(pc)];
        (row.guest_pc == pc).then_some(row)
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
