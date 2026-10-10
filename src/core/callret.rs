/// One ring frame: if the guest returns to `guest_pc`, `cell` names where to go.
///
/// `#[repr(C)]`, exactly 16 bytes: lifted code indexes the ring with a mask.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CallRetEntry {
    /// Run-time VA the matching `RET` must pop; zero in an untouched frame, so imbalance misses.
    pub guest_pc: u64,
    /// Address of the [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) for `guest_pc`.
    ///
    /// Zero or uninstalled is safe: it reads as "not linked".
    pub cell: u64,
}

pub const FRAME_SIZE: u64 = 16;

/// Byte offset of [`CallRetEntry::cell`] within a frame.
pub const CELL_OFFSET: i32 = 8;

/// Shape of the host's call-return ring; baked into every region, so changing it discards translations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallRetConfig {
    /// Frame count; a power of two (the wrap is a mask). Deeper recursion loses the oldest frames.
    pub entries: u32,
}

impl CallRetConfig {
    /// # Panics
    ///
    /// If `entries` is not a power of two.
    pub fn new(entries: u32) -> Self {
        assert!(
            entries.is_power_of_two(),
            "a call-return ring must have a power-of-two frame count, not {entries}"
        );
        Self { entries }
    }

    /// Ring size in bytes: what the host must allocate.
    pub fn bytes(&self) -> u64 {
        self.entries as u64 * FRAME_SIZE
    }

    pub(crate) fn mask(&self) -> i64 {
        (self.bytes() - 1) as i64
    }
}

/// Context offsets of the ring fields ([`GuestArch::call_return_offsets`](crate::core::GuestArch::call_return_offsets)); `None` means no prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallRetOffsets {
    /// Offset of the ring pointer; null disables prediction for that thread.
    pub base: i32,
    /// Offset of the top frame's **byte offset** in the ring (any value is valid, including zero).
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
