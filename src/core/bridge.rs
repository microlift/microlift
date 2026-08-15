//! Runtime-supplied **bridges**: the third import kind.
//!
//! A fast import ([`LifterConfig::register_fast_import`]) inlines a real
//! Cranelift `call` to the host function, marshaling arguments out of the guest
//! register file. That requires the callee's signature to be expressible as
//! [`FunctionArgumentType`](crate::function_traits::FunctionArgumentType)s and
//! placeable by a [`CallAbi`](crate::core::abi::CallAbi) - which rules out
//! variadics, struct-by-value, and anything else the descriptor cannot spell.
//! Those imports are demoted to an `ImportedFunctionCall` VM exit today.
//!
//! A **bridge** removes that restriction. Instead of reproducing the callee's
//! signature, the lifter calls one runtime-supplied trampoline with a fixed
//! signature and hands it the whole guest context:
//!
//! ```text
//! extern "C" fn(ctx: *mut RegisterContext, import_va: u64) -> u32
//! ```
//!
//! The bridge reads whatever it needs out of `ctx`, does the call (or the
//! redirect, or the error), writes results back into `ctx`, and returns:
//!
//! - **0** - handled. The lifted code reloads the declared clobbers from the
//!   context and **falls through inline**; the block is not terminated and no
//!   VM exit occurs. This is the common case, and it costs one call.
//! - **non-zero** - the runtime wants out. The lifted code returns immediately,
//!   leaving `exit_reason` / `exit_code` exactly as the bridge wrote them. This
//!   covers everything a slow `ImportedFunctionCall` exit covers today: an ABI
//!   error the runtime stops on, `longjmp`, a control redirect, or a transfer
//!   into another guest image.
//! - [`RESUME_LINKED`] - a particular non-zero: the runtime wants out, and
//!   the only thing left to do is continue at the `rip` it has written. Where
//!   block linking is on, the lifted code probes the link table for that address
//!   before returning, so a bridged indirect call into guest code - a C++
//!   virtual call, which is a quarter of all exits in a real program - can be
//!   resolved without a host round trip. Any other non-zero value returns
//!   unconditionally, which is what a bridge that has parked an error, or that
//!   wants its exit serviced, must do.
//!
//! `import_va` is passed as a second argument so one compiled bridge per symbol
//! serves every stub and slot that names it - the runtime keeps its own side
//! table rather than the lifter compiling N bridges for N call sites.
//!
//! [`LifterConfig::register_fast_import`]: crate::core::LifterConfig::register_fast_import

use std::sync::atomic::AtomicPtr;

/// The bridge return value meaning "exit, and you may link".
///
/// Distinct from a bare non-zero because the two differ in what the host is
/// still owed. An ordinary exit may have an error parked on the thread, a call
/// left to service, or a `rip` that is a resume point rather than a
/// destination - and a link would skip the host that was supposed to notice.
/// This value asserts the opposite: guest state is complete, `rip` is where
/// execution continues, and nothing is waiting.
///
/// Two, not one, so that a bridge written before this existed - returning `1`
/// for every exit - keeps exactly the behaviour it had.
pub const RESUME_LINKED: u32 = 2;

/// Where the lifted code finds the bridge function's address.
#[derive(Debug, Clone, Copy)]
pub enum BridgeTarget {
    /// Baked into the code as an `iconst`. Immutable for the life of the
    /// compiled region, one call, no load.
    Direct(*const u8),
    /// Loaded from a runtime-owned slot immediately before each call: one
    /// dependent load, and the runtime can re-point the slot at any time
    /// without re-lifting anything.
    ///
    /// The emitted load is deliberately **not** marked `readonly`, so Cranelift
    /// may not hoist it out of a loop or CSE it across the call. The read side
    /// is a plain (relaxed) load; publication on the runtime side must be a
    /// **release** store. That is sound on both supported hosts: x86-64 stores
    /// are release-ordered, and on aarch64 the address dependency between this
    /// load and the `call_indirect` that consumes it orders the callee's code
    /// and data against the publishing store.
    Slot(*const AtomicPtr<()>),
}

/// Which parts of the guest register file a bridge touches.
///
/// The lifter stores only `reads` into the `RegisterContext` before the call
/// and reloads only `writes` after it; registers outside `writes` keep their
/// SSA values across the call site. For a bridge specialized to, say,
/// `int foo(int, char *)` that is two stores and one reload instead of a full
/// context sync.
///
/// The default is [`BridgeEffects::full`] - everything read, everything
/// clobbered - which is the same cost the slow `ImportedFunctionCall` exit pays
/// today, so a bridge is never slower than the exit it replaces.
///
/// **This is not `sync_to_context_selective`.** That method (see
/// `arch/x86_64/sync.rs`, marked *DANGEROUS - DO NOT USE*) is unsound because
/// its dirty set is an emission-order side effect that misses loop-carried
/// writes, i.e. it reasons - wrongly - about the whole CFG. A `RegMask` here is
/// *declared by the callee* and applied at exactly one call site with no CFG
/// reasoning at all: "store these variables now, reload these after". Different
/// mechanism entirely.
///
/// Getting a mask wrong is still a correctness bug, just a local one: a
/// register the bridge reads but did not declare in `reads` is stale in the
/// context, and one it clobbers but did not declare in `writes` keeps its stale
/// SSA value afterwards. When in doubt, widen the mask.
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

/// A set of guest registers, as a bitmask over the guest frontend's canonical
/// register indices.
///
/// For x86-64 the GPR bit order is the architectural encoding order
/// (`RAX`=0, `RCX`=1, `RDX`=2, `RBX`=3, `RSP`=4, `RBP`=5, `RSI`=6, `RDI`=7,
/// `R8`..`R15`=8..15) and the vector bits are `XMM0`..`XMM15`. Build masks with
/// the helpers in [`crate::arch::x86_64::abi`] rather than by writing raw bits.
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
/// indirect-call / tail-jump dispatch as a whole).
///
/// See the [module docs](self) for the calling contract and the meaning of the
/// return value.
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
/// The engine queries this **per direct-call instruction while lifting**,
/// rather than snapshotting a map when the [`LifterConfig`] is built. That is
/// what makes `dlopen`/`dlsym` work without re-plumbing the lifter: a region
/// lifted *before* a `dlopen` contains no calls into that image (the image did
/// not exist), so nothing already compiled needs patching - what matters is
/// that a region lifted *after* the `dlopen` sees the new sites, which it does
/// as soon as the runtime's own view has them.
///
/// The residual case - a slot that rebinds underneath code that is already
/// compiled - is covered by [`BridgeTarget::Slot`], which reloads the target
/// before every call and so needs no re-lift either.
///
/// Lifting is not the hot path, so a virtual call per direct call site is free;
/// back this with the runtime's existing loader view instead of copying it.
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
