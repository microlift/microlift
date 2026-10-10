//! Runtime-supplied bridges: imports with a fixed signature that receive the context.
//!
//! ```text
//! extern "C" fn(ctx: *mut RegisterContext, import_va: u64) -> u32
//! ```
//!
//! - `0`: handled; reload declared clobbers and fall through inline.
//! - non-zero: return now, leaving `exit_reason`/`exit_code` as written.
//! - [`RESUME_LINKED`]: return now via a linkable exit.

use std::sync::atomic::AtomicPtr;

/// Bridge return meaning "exit, and you may link".
///
/// Asserts guest state is complete and `rip` is a destination; a bare non-zero may owe the host work a link would skip.
pub const RESUME_LINKED: u32 = 2;

/// Where the lifted code finds the bridge function's address.
#[derive(Debug, Clone, Copy)]
pub enum BridgeTarget {
    /// Baked in as an `iconst`.
    Direct(*const u8),
    /// Loaded from a runtime-owned slot before each call, so it can be re-pointed without re-lifting.
    ///
    /// The load is deliberately not `readonly`; read is relaxed, publication must be a **release** store.
    Slot(*const AtomicPtr<()>),
}

/// Which guest registers a bridge touches; only `reads` is stored before the call, only `writes` reloaded after.
///
/// Defaults to [`BridgeEffects::full`]. A mask that is too narrow is a miscompile: widen when in doubt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BridgeEffects {
    /// Stored before the call; the stack pointer is always stored.
    pub reads: RegMask,
    /// Reloaded after a `0` return; not re-stored on the non-zero path (the bridge's values win).
    pub writes: RegMask,
}

impl BridgeEffects {
    pub const fn full() -> Self {
        Self {
            reads: RegMask::ALL,
            writes: RegMask::ALL,
        }
    }

    pub const fn new(reads: RegMask, writes: RegMask) -> Self {
        Self { reads, writes }
    }
}

impl Default for BridgeEffects {
    fn default() -> Self {
        Self::full()
    }
}

/// A bitmask over the frontend's register indices (x86-64: `RAX`=0..`R15`=15, `XMM0`..`XMM15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegMask {
    pub gprs: u32,
    pub vecs: u32,
    pub flags: bool,
    pub segment_bases: bool,
}

impl RegMask {
    pub const NONE: Self = Self {
        gprs: 0,
        vecs: 0,
        flags: false,
        segment_bases: false,
    };

    pub const ALL: Self = Self {
        gprs: u32::MAX,
        vecs: u32::MAX,
        flags: true,
        segment_bases: true,
    };

    pub const fn gpr(idx: u8) -> Self {
        Self {
            gprs: 1u32 << idx,
            ..Self::NONE
        }
    }

    pub const fn vec(idx: u8) -> Self {
        Self {
            vecs: 1u32 << idx,
            ..Self::NONE
        }
    }

    pub const fn with_gpr(mut self, idx: u8) -> Self {
        self.gprs |= 1u32 << idx;
        self
    }

    pub const fn with_vec(mut self, idx: u8) -> Self {
        self.vecs |= 1u32 << idx;
        self
    }

    pub const fn with_flags(mut self) -> Self {
        self.flags = true;
        self
    }

    pub const fn with_segment_bases(mut self) -> Self {
        self.segment_bases = true;
        self
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            gprs: self.gprs | other.gprs,
            vecs: self.vecs | other.vecs,
            flags: self.flags || other.flags,
            segment_bases: self.segment_bases || other.segment_bases,
        }
    }

    pub const fn complement(self) -> Self {
        Self {
            gprs: !self.gprs,
            vecs: !self.vecs,
            flags: !self.flags,
            segment_bases: !self.segment_bases,
        }
    }

    pub const fn has_gpr(self, idx: u8) -> bool {
        self.gprs & (1u32 << idx) != 0
    }

    pub const fn has_vec(self, idx: u8) -> bool {
        self.vecs & (1u32 << idx) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.gprs == 0 && self.vecs == 0 && !self.flags && !self.segment_bases
    }
}

impl Default for RegMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// A bridge registered for one import address (see the [module docs](self)).
#[derive(Debug, Clone)]
pub struct BridgeImport {
    /// Diagnostics only.
    pub name: &'static str,
    pub target: BridgeTarget,
    /// Defaults to [`BridgeEffects::full`].
    pub effects: BridgeEffects,
}

impl BridgeImport {
    /// Fixed address; assumed to read and clobber everything.
    pub fn direct(name: &'static str, func: *const u8) -> Self {
        Self {
            name,
            target: BridgeTarget::Direct(func),
            effects: BridgeEffects::full(),
        }
    }

    /// Reached through a slot (see [`BridgeTarget::Slot`] for ordering); assumed to read and clobber everything.
    pub fn slot(name: &'static str, slot: *const AtomicPtr<()>) -> Self {
        Self {
            name,
            target: BridgeTarget::Slot(slot),
            effects: BridgeEffects::full(),
        }
    }

    pub fn with_effects(mut self, effects: BridgeEffects) -> Self {
        self.effects = effects;
        self
    }
}

#[derive(Debug, Clone)]
pub enum ImportKind {
    /// Always an `ImportedFunctionCall` VM exit.
    Slow,
    /// Inline call using the import's [`CallAbi`](crate::core::abi::CallAbi); demoted to slow for variadics and unplaceable arguments.
    Fast(crate::core::FastImport),
    Bridge(BridgeImport),
}

/// Lift-time view of which addresses are imports, queried per call site (so `dlopen` works).
pub trait ImportRegistry: std::fmt::Debug + Send + Sync {
    fn lookup(&self, va: u64) -> Option<ImportKind>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regmask_set_algebra() {
        let m = RegMask::NONE
            .with_gpr(0)
            .with_gpr(7)
            .with_vec(1)
            .with_flags();
        assert!(m.has_gpr(0) && m.has_gpr(7) && !m.has_gpr(1));
        assert!(m.has_vec(1) && !m.has_vec(0));
        assert!(m.flags && !m.segment_bases);

        let c = m.complement();
        assert!(!c.has_gpr(0) && c.has_gpr(1));
        assert!(!c.flags && c.segment_bases);

        assert!(RegMask::NONE.is_empty());
        assert!(!RegMask::ALL.is_empty());
        assert_eq!(RegMask::NONE.union(m), m);
        assert_eq!(RegMask::gpr(3), RegMask::NONE.with_gpr(3));
    }

    #[test]
    fn effects_default_is_full() {
        assert_eq!(BridgeEffects::default(), BridgeEffects::full());
        assert_eq!(BridgeEffects::full().reads, RegMask::ALL);
    }
}
