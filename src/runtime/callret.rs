//! The per-thread call-return ring: the frames lifted code pushes and pops.

use crate::core::callret::{CallRetConfig, CallRetEntry};

/// One guest thread's call-return ring.
pub struct CallRetStack {
    frames: Box<[CallRetEntry]>,
    config: CallRetConfig,
}

impl CallRetStack {
    /// A zeroed ring with `config`'s geometry.
    ///
    /// `config` must be byte-identical to the one every region on this thread
    /// was lifted under: the frame count is baked into each mask. Zeroed is the
    /// correct initial state, not merely convenient - a zero `guest_pc` is the
    /// "empty frame" marker every pop tests against.
    pub fn new(config: CallRetConfig) -> Self {
        Self {
            frames: vec![CallRetEntry::default(); config.entries as usize].into_boxed_slice(),
            config,
        }
    }

    /// The geometry this ring was built with.
    pub fn config(&self) -> CallRetConfig {
        self.config
    }

    /// The pointer to hand to `ctx.callret_base`.
    pub fn as_mut_ptr(&mut self) -> *mut CallRetEntry {
        self.frames.as_mut_ptr()
    }

    /// The frame at byte offset `off` - i.e. at `ctx.callret_off`.
    ///
    /// # Panics
    ///
    /// If `off` is not a frame boundary inside the ring. Lifted code cannot
    /// produce such an offset (it masks), so this catches a host that wrote one
    /// by hand.
    pub fn frame_at(&self, off: u64) -> &CallRetEntry {
        assert!(
            off < self.config.bytes() && off.is_multiple_of(crate::core::callret::FRAME_SIZE),
            "{off} is not a frame boundary in a {}-byte ring",
            self.config.bytes()
        );
        &self.frames[(off / crate::core::callret::FRAME_SIZE) as usize]
    }

    /// How many frames hold a return address. For diagnostics and tests; O(n).
    ///
    /// Note this counts written frames, not live ones: a popped frame keeps
    /// its contents, because clearing it would be a store the pop does not need
    /// to make. The pointer, not the contents, is what says how deep the ring is.
    pub fn written(&self) -> usize {
        self.frames.iter().filter(|f| f.guest_pc != 0).count()
    }
}
