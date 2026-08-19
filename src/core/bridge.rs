//! Runtime-supplied **bridges**: the third import kind.
//!
//! A fast import inlines a real Cranelift `call`, which requires the callee's
//! signature to be expressible as a
//! [`FunctionArgumentType`](crate::function_traits::FunctionArgumentType) list
//! placeable by a [`CallAbi`](crate::core::abi::CallAbi) - ruling out variadics
//! and struct-by-value. A **bridge** drops that: the lifter calls one
//! runtime-supplied trampoline with a fixed signature and hands it the context.
//!
//! ```text
//! extern "C" fn(ctx: *mut RegisterContext, import_va: u64) -> u32
//! ```
//!
//! The return value is the protocol: **0** = handled, reload the declared
//! clobbers and fall through inline with no VM exit; **non-zero** = return now,
//! leaving `exit_reason`/`exit_code` as the bridge wrote them;
//! [`RESUME_LINKED`] = return now, *and* the exit is linkable, so a bridged
//! indirect call into guest code needs no host round trip.
//!
//! `import_va` is a second argument so one compiled bridge per symbol serves
//! every stub and slot that names it.
//!
//! [`LifterConfig::register_fast_import`]: crate::core::LifterConfig::register_fast_import

use std::sync::atomic::AtomicPtr;

/// The bridge return value meaning "exit, and you may link".
///
/// Distinct from a bare non-zero because the two differ in what the host is
/// still owed: an ordinary exit may have an error parked on the thread, a call
/// left to service, or a `rip` that is a resume point rather than a
/// destination, and a link would skip the host that was supposed to notice.
/// This value asserts guest state is complete and `rip` is where execution
/// continues.
///
/// Two rather than one so a bridge written before this existed keeps its
/// behaviour.
pub const RESUME_LINKED: u32 = 2;

/// Where the lifted code finds the bridge function's address.
#[derive(Debug, Clone, Copy)]
pub enum BridgeTarget {
    /// Baked into the code as an `iconst`. Immutable for the life of the
    /// compiled region, one call, no load.
    Direct(*const u8),
    /// Loaded from a runtime-owned slot immediately before each call: one dependent
    /// load, and the runtime can re-point the slot without re-lifting anything.
    ///
    /// The load is deliberately **not** `readonly`, so Cranelift may not hoist it
    /// out of a loop or CSE it across the call. The read is a relaxed load;
    /// publication must be a **release** store. Sound on both hosts: x86-64 stores
    /// are release-ordered, and on aarch64 the address dependency to the
    /// `call_indirect` orders the callee's code and data against the store.
    Slot(*const AtomicPtr<()>),
}

/// Which parts of the guest register file a bridge touches.
///
/// Only `reads` is stored into the context before the call and only `writes`
/// reloaded after; registers outside `writes` keep their SSA values. For a
/// bridge specialized to `int foo(int, char *)` that is two stores and one
/// reload instead of a full context sync. Defaults to [`BridgeEffects::full`],
/// which is what the slow exit already pays, so a bridge is never slower than
/// the exit it replaces.
///
/// **This is not `sync_to_context_selective`** (`arch/x86_64/sync.rs`, marked
/// *DANGEROUS - DO NOT USE*), which infers a dirty set from emission order and
/// misses loop-carried writes. A `RegMask` is *declared by the callee* and
/// applied at one call site with no CFG reasoning at all.
///
/// Getting a mask wrong is still a correctness bug, just a local one: a
/// register read but not declared in `reads` is stale in the context, and one
/// clobbered but not in `writes` keeps its stale SSA value. When in doubt,
/// widen the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BridgeEffects {
    /// Guest state the bridge reads out of the context. Written to the context
    /// before the call. The stack pointer is always stored regardless of this
    /// mask, since every bridge needs a coherent guest stack.
    pub reads: RegMask,
    /// Guest state the bridge may modify in the context. Reloaded into SSA
    /// after a handled (`0`) return, and *not* re-stored on the exiting
    /// (non-zero) path - the bridge's values win there.
    pub writes: RegMask,
}

impl BridgeEffects {
    /// Reads and clobbers everything: a full context sync around the call. The
    /// conservative default.
    pub const fn full() -> Self {
        Self {
            reads: RegMask::ALL,
            writes: RegMask::ALL,
        }
    }

    /// Reads and clobbers exactly the given masks.
    pub const fn new(reads: RegMask, writes: RegMask) -> Self {
        Self { reads, writes }
    }
}

impl Default for BridgeEffects {
    fn default() -> Self {
        Self::full()
    }
}

/// A set of guest registers, as a bitmask over the frontend's canonical
/// register indices.
///
/// For x86-64 the GPR bit order is the architectural encoding order (`RAX`=0
/// through `R15`=15) and the vector bits are `XMM0`..`XMM15`. Build masks with
/// the helpers in [`crate::arch::x86_64::abi`], not by writing raw bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegMask {
    /// Bit `i` selects general-purpose register `i`.
    pub gprs: u32,
    /// Bit `i` selects vector register `i`.
    pub vecs: u32,
    /// All condition/status flags, as one unit.
    pub flags: bool,
    /// The segment-base pseudo-registers (`fs_base` / `gs_base` on x86-64).
    pub segment_bases: bool,
}

impl RegMask {
    /// The empty set.
    pub const NONE: Self = Self {
        gprs: 0,
        vecs: 0,
        flags: false,
        segment_bases: false,
    };

    /// Every register the guest frontend models.
    pub const ALL: Self = Self {
        gprs: u32::MAX,
        vecs: u32::MAX,
        flags: true,
        segment_bases: true,
    };

    /// A mask holding just general-purpose register `idx`.
    pub const fn gpr(idx: u8) -> Self {
        Self {
            gprs: 1u32 << idx,
            ..Self::NONE
        }
    }

    /// A mask holding just vector register `idx`.
    pub const fn vec(idx: u8) -> Self {
        Self {
            vecs: 1u32 << idx,
            ..Self::NONE
        }
    }

    /// Add general-purpose register `idx`.
    pub const fn with_gpr(mut self, idx: u8) -> Self {
        self.gprs |= 1u32 << idx;
        self
    }

    /// Add vector register `idx`.
    pub const fn with_vec(mut self, idx: u8) -> Self {
        self.vecs |= 1u32 << idx;
        self
    }

    /// Add the condition flags.
    pub const fn with_flags(mut self) -> Self {
        self.flags = true;
        self
    }

    /// Add the segment bases.
    pub const fn with_segment_bases(mut self) -> Self {
        self.segment_bases = true;
        self
    }

    /// Set union.
    pub const fn union(self, other: Self) -> Self {
        Self {
            gprs: self.gprs | other.gprs,
            vecs: self.vecs | other.vecs,
            flags: self.flags || other.flags,
            segment_bases: self.segment_bases || other.segment_bases,
        }
    }

    /// Everything this mask does *not* contain.
    pub const fn complement(self) -> Self {
        Self {
            gprs: !self.gprs,
            vecs: !self.vecs,
            flags: !self.flags,
            segment_bases: !self.segment_bases,
        }
    }

    /// Whether general-purpose register `idx` is in the set.
    pub const fn has_gpr(self, idx: u8) -> bool {
        self.gprs & (1u32 << idx) != 0
    }

    /// Whether vector register `idx` is in the set.
    pub const fn has_vec(self, idx: u8) -> bool {
        self.vecs & (1u32 << idx) != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.gprs == 0 && self.vecs == 0 && !self.flags && !self.segment_bases
    }
}

impl Default for RegMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// A runtime-supplied bridge registered for one import address (or for the
/// indirect-call / tail-jump dispatch as a whole). See the [module docs](self)
/// for the calling contract.
#[derive(Debug, Clone)]
pub struct BridgeImport {
    /// Diagnostic name; not used for linkage.
    pub name: &'static str,
    /// How the lifted code reaches the bridge function.
    pub target: BridgeTarget,
    /// What the bridge reads and clobbers, so the sync around the call can be
    /// narrowed. Defaults to [`BridgeEffects::full`].
    pub effects: BridgeEffects,
}

impl BridgeImport {
    /// A bridge at a fixed address, conservatively assumed to read and clobber
    /// the entire guest register file.
    pub fn direct(name: &'static str, func: *const u8) -> Self {
        Self {
            name,
            target: BridgeTarget::Direct(func),
            effects: BridgeEffects::full(),
        }
    }

    /// A bridge reached through a live-updatable slot, conservatively assumed to
    /// read and clobber the entire guest register file. See
    /// [`BridgeTarget::Slot`] for the memory-ordering contract.
    pub fn slot(name: &'static str, slot: *const AtomicPtr<()>) -> Self {
        Self {
            name,
            target: BridgeTarget::Slot(slot),
            effects: BridgeEffects::full(),
        }
    }

    /// Narrow the sync around this bridge's call sites.
    pub fn with_effects(mut self, effects: BridgeEffects) -> Self {
        self.effects = effects;
        self
    }
}

/// How a registered import address is reached from lifted code.
#[derive(Debug, Clone)]
pub enum ImportKind {
    /// Always emit an `ImportedFunctionCall` VM exit.
    Slow,
    /// Attempt an inline Cranelift call, reading arguments from wherever the
    /// import's [`CallAbi`](crate::core::abi::CallAbi) says the guest put them.
    /// Automatically demoted to [`ImportKind::Slow`] for variadic functions and
    /// for arguments the ABI cannot place.
    Fast(crate::core::FastImport),
    /// Call a runtime-supplied bridge; see the [module docs](self).
    Bridge(BridgeImport),
}

/// A lift-time view of which addresses are imports.
///
/// Queried **per direct-call instruction while lifting** rather than
/// snapshotted when the [`LifterConfig`] is built, which is what makes
/// `dlopen`/`dlsym` work without re-plumbing anything: a region lifted before a
/// `dlopen` contains no calls into that image, so nothing compiled needs
/// patching, and a region lifted after it sees the new sites.
///
/// A slot that rebinds underneath already-compiled code is covered by
/// [`BridgeTarget::Slot`], which reloads before every call.
///
/// Lifting is not the hot path, so a virtual call per call site is free.
///
/// [`LifterConfig`]: crate::core::LifterConfig
pub trait ImportRegistry: std::fmt::Debug + Send + Sync {
    /// What kind of import lives at `va`, or `None` if `va` is not an import.
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
