//! The per-thread block-link table: the array lifted code probes.

use crate::core::link::{BlockLinkConfig, BlockLinkEntry};

/// One guest thread's block-link table.
///
/// A direct-mapped array of [`BlockLinkEntry`] indexed by
/// [`BlockLinkConfig::slot`] - the same function the lifter emitted arithmetic
/// for, called rather than reimplemented, so the host fills the row lifted code
/// reads by construction rather than by agreement.
///
/// One per guest thread. Nothing here is atomic and nothing needs to be: the
/// owning thread is the only writer, and it writes only while it is *not* inside
/// lifted code.
pub struct LinkTable {
    rows: Box<[BlockLinkEntry]>,
    config: BlockLinkConfig,
}

impl LinkTable {
    /// An empty table with `config`'s geometry.
    ///
    /// `config` must be byte-identical to the one every region reachable through
    /// this table was lifted under - the shift and the row count are baked into
    /// each probe. A `Box<[_]>` of a `#[repr(C)]` 32-byte type is 8-byte aligned,
    /// which the probe's `aligned` loads require.
    pub fn new(config: BlockLinkConfig) -> Self {
        Self {
            rows: vec![BlockLinkEntry::default(); config.entries as usize].into_boxed_slice(),
            config,
        }
    }

    /// The geometry this table was built with.
    pub fn config(&self) -> BlockLinkConfig {
        self.config
    }

    /// Install `entry`, overwriting whatever occupied its slot.
    ///
    /// Eviction is silent and correct: a row is a *hint*, and losing one costs a
    /// miss. Two hot addresses that collide will evict each other forever, which
    /// is a table-sizing problem rather than a bug - watch
    /// [`CacheStats::miss_rate`](super::CacheStats::miss_rate).
    pub fn publish(&mut self, entry: BlockLinkEntry) {
        let slot = self.config.slot(entry.guest_pc);
        self.rows[slot] = entry;
    }

    /// Remove the row for `pc`, if the row in its slot is actually about `pc`.
    ///
    /// The guard matters: two addresses share a slot, and zeroing on a
    /// collision would retract a live translation that merely hashes the same.
    /// That would be sound - a retraction only ever costs a miss - but it would
    /// silently degrade a table under exactly the workload that can least afford
    /// it.
    ///
    /// Returns whether a row was cleared.
    pub fn retract(&mut self, pc: u64) -> bool {
        let slot = self.config.slot(pc);
        if self.rows[slot].guest_pc != pc {
            return false;
        }
        self.rows[slot] = BlockLinkEntry::default();
        true
    }

    /// Drop every row. Correct at any time, for the same reason `retract` is.
    pub fn clear(&mut self) {
        self.rows.fill(BlockLinkEntry::default());
    }

    /// The row for `pc`, if one is installed for it.
    pub fn get(&self, pc: u64) -> Option<&BlockLinkEntry> {
        let row = &self.rows[self.config.slot(pc)];
        (row.guest_pc == pc).then_some(row)
    }

    /// The pointer to hand to `ctx.link_table`.
    ///
    /// Setting the context field to null is how a host declines to link at all -
    /// and, under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned),
    /// the only way to end a chain of tail calls. See
    /// [`ThreadRuntime::detach_table`](super::ThreadRuntime::detach_table).
    pub fn as_mut_ptr(&mut self) -> *mut BlockLinkEntry {
        self.rows.as_mut_ptr()
    }

    /// How many rows are occupied. For diagnostics and tests; O(n).
    pub fn occupancy(&self) -> usize {
        self.rows.iter().filter(|r| r.guest_pc != 0).count()
    }
}
