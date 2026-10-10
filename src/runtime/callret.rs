//! Per-thread call-return ring pushed and popped by lifted code.

use crate::core::callret::{CallRetConfig, CallRetEntry};

/// One guest thread's call-return ring.
pub struct CallRetStack {
    frames: Box<[CallRetEntry]>,
    config: CallRetConfig,
}

impl CallRetStack {
    /// A zeroed ring (zero `guest_pc` marks an empty frame).
    ///
    /// `config` must equal the one every region on this thread was lifted under.
    pub fn new(config: CallRetConfig) -> Self {
        Self {
            frames: vec![CallRetEntry::default(); config.entries as usize].into_boxed_slice(),
            config,
        }
    }

    pub fn config(&self) -> CallRetConfig {
        self.config
    }

    /// For `ctx.callret_base`.
    pub fn as_mut_ptr(&mut self) -> *mut CallRetEntry {
        self.frames.as_mut_ptr()
    }

    /// The frame at byte offset `off` (`ctx.callret_off`).
    ///
    /// # Panics
    ///
    /// If `off` is not a frame boundary inside the ring.
    pub fn frame_at(&self, off: u64) -> &CallRetEntry {
        assert!(
            off < self.config.bytes() && off.is_multiple_of(crate::core::callret::FRAME_SIZE),
            "{off} is not a frame boundary in a {}-byte ring",
            self.config.bytes()
        );
        &self.frames[(off / crate::core::callret::FRAME_SIZE) as usize]
    }

    /// Written frames, not live ones (pops do not clear); O(n).
    pub fn written(&self) -> usize {
        self.frames.iter().filter(|f| f.guest_pc != 0).count()
    }
}
