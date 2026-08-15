//! Pinned register files: keeping guest state in host registers across a link.
//!
//! [`crate::core::link`] lets a region resolve its own successor. This module is
//! what lets it *branch* to that successor instead of handing a code address
//! back to the host - and, in doing so, carry part of the guest register file
//! along in machine registers rather than through the
//! [`RegisterContext`](crate::arch::x86_64::context::RegisterContext).
//!
//! # The mechanism, and why it is the signature
//!
//! Cranelift allocates registers per function, so two translations do not agree
//! on where the guest's registers live. The one portable way to make them agree
//! is to say so in the ABI: a pinned region is compiled with
//! [`CallConv::Tail`](cranelift_codegen::isa::CallConv::Tail) and the signature
//!
//! ```text
//! tail (ctx, slot0, slot1, ...) -> (ctx, slot0, slot1, ...)
//! ```
//!
//! so a link is a real `return_call_indirect` and the pinned slots are already
//! in the registers the callee expects. There is no separate "pin these
//! registers" facility to reach for: Cranelift's `enable_pinned_reg` pins
//! exactly one, which is not a register *file*.
//!
//! **The context pointer is returned as well as passed, and that is
//! load-bearing.** Under the tail convention arguments and results are assigned
//! from the same register sequence, so a signature whose results are its
//! parameters puts every slot in the register it already occupies: forwarding
//! state costs nothing and a link lowers to a branch. Drop `ctx` from the
//! results and every slot shifts by one register, which is a move per slot per
//! link, in both directions.
//!
//! # What the host decides, and what it must not get wrong
//!
//! Microlift has no opinion about which registers are worth pinning - that is a
//! property of the guest program, and the answer for a game is not the answer
//! for a compiler. The host declares an ordered [`PinnedAbi`] and everything
//! else follows from it.
//!
//! Two obligations come with that:
//!
//! - **Every region reachable through one link table must be lifted under the
//!   same `PinnedAbi`,** slot for slot and in the same order. It is the ABI of a
//!   branch between two independently compiled functions; nothing checks it at
//!   run time and a mismatch is a wrong-code bug, not a fault. Changing it means
//!   discarding every translation lifted under the old one - the same rule
//!   [`BlockLinkConfig`](crate::core::link::BlockLinkConfig) already carries for
//!   its geometry, for the same reason.
//! - **Slots are scarce, and overflowing them is silent for arguments.** See
//!   [`RegisterBudget`]: past the budget Cranelift assigns stack locations, and
//!   a "pinned" register that lives in a stack slot is strictly worse than one
//!   that lives in the context. [`PinnedAbi::check_budget`] is how a host asks
//!   before it finds out.
//!
//! # Where the boundary is
//!
//! A pinned slot changes exactly three things, and deliberately nothing else:
//!
//! 1. **Entry.** Its value arrives as a function parameter instead of a load
//!    from the context.
//! 2. **Exit to the host.** Its value is returned instead of stored; the entry
//!    thunk ([`LiftedCode::entry_thunk`](crate::core::LiftedCode::entry_thunk))
//!    performs the store, once per *chain* rather than once per region.
//! 3. **A taken link.** Its value is an argument to the tail call.
//!
//! Everything in between - a bridge call's [`RegMask`](crate::core::bridge::RegMask)
//! sync, an import's argument marshalling, a guest instruction reading the
//! register - is untouched, because those all happen while the region is
//! running and the context is what the host reads *then*. The rule that keeps
//! this straight: **the full [`GuestArch::sync_to_context`] skips pinned slots,
//! and the masked syncs never do.**
//!
//! [`GuestArch::sync_to_context`]: crate::core::GuestArch::sync_to_context

use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use target_lexicon::Architecture;

/// The most slots a [`PinnedAbi`] can hold.
///
/// Comfortably above every target's [`RegisterBudget`], so the cap is never the
/// thing a host runs into first - it exists so the ABI is a `Copy` value that
/// can live inside a [`LifterConfig`](crate::core::LifterConfig) cloned once per
/// region.
pub const MAX_PINNED_SLOTS: usize = 16;

/// Which of a guest's register banks a slot names.
///
/// Deliberately only two: this is the granularity every guest frontend can be
/// expected to answer for, and the granularity a machine ABI actually
/// distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// An integer/pointer register - x86-64's `RAX`..`R15`, aarch64's `X0`..`X30`.
    Gpr,
    /// A vector register - x86-64's `XMM0`..`XMM15`.
    Vector,
}

/// One guest register, in the guest's own architectural numbering.
///
/// The engine never interprets `index`; it hands the pair straight back to the
/// frontend ([`GuestArch::pinned_slot_var`](crate::core::GuestArch::pinned_slot_var)),
/// which is what keeps this file free of any particular guest's register names.
/// For x86-64 the numbering is [`GPR_ORDER`](crate::arch::x86_64::regs::GPR_ORDER)
/// for [`RegClass::Gpr`] and the XMM number for [`RegClass::Vector`]; the named
/// constants in [`crate::arch::x86_64::pin`] are the readable spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedReg {
    /// Which bank the register is in.
    pub class: RegClass,
    /// Its architectural number within that bank.
    pub index: u16,
}

impl PinnedReg {
    /// The general-purpose register numbered `index` in the guest's own order.
    pub const fn gpr(index: u16) -> Self {
        Self {
            class: RegClass::Gpr,
            index,
        }
    }

    /// The vector register numbered `index`.
    pub const fn vector(index: u16) -> Self {
        Self {
            class: RegClass::Vector,
            index,
        }
    }
}

/// The ordered set of guest registers a host pins into the link ABI.
///
/// Order is significant - it is the parameter order of every region compiled
/// under it - so this is a sequence, not a set. Duplicates are refused at
/// construction because a register occupying two slots has two values on entry
/// and no defined answer for which one wins.
///
/// `Copy`, and sized rather than heap-allocated, because a
/// [`LifterConfig`](crate::core::LifterConfig) is cloned once per region and
/// anything in it is paid for at that rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedAbi {
    slots: [PinnedReg; MAX_PINNED_SLOTS],
    len: u8,
}

impl Default for PinnedAbi {
    fn default() -> Self {
        Self::new()
    }
}

impl PinnedAbi {
    /// An ABI with no pinned registers, which is
    /// [inert](PinnedAbi::is_empty): a region lifted under it keeps the whole
    /// register file in the context, exactly as one lifted without pinning.
    pub const fn new() -> Self {
        Self {
            slots: [PinnedReg::gpr(0); MAX_PINNED_SLOTS],
            len: 0,
        }
    }

    /// Append `reg` as the next slot.
    ///
    /// # Panics
    ///
    /// If `reg` is already pinned, or the ABI is full ([`MAX_PINNED_SLOTS`]).
    /// Both are programming errors in a declaration written once at start-up,
    /// not run-time conditions - and a silently dropped slot would be a
    /// wrong-code bug in every region compiled afterwards.
    #[must_use]
    pub fn with(mut self, reg: PinnedReg) -> Self {
        assert!(
            !self.contains(reg),
            "{reg:?} is already pinned; a register cannot occupy two slots"
        );
        assert!(
            (self.len as usize) < MAX_PINNED_SLOTS,
            "a pinned ABI holds at most {MAX_PINNED_SLOTS} slots"
        );
        self.slots[self.len as usize] = reg;
        self.len += 1;
        self
    }

    /// Append every register in `regs`, in order.
    #[must_use]
    pub fn with_all(mut self, regs: impl IntoIterator<Item = PinnedReg>) -> Self {
        for reg in regs {
            self = self.with(reg);
        }
        self
    }

    /// The pinned registers, in slot order.
    pub fn slots(&self) -> &[PinnedReg] {
        &self.slots[..self.len as usize]
    }

    /// How many registers are pinned.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether nothing is pinned, in which case the whole mechanism is inert.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `reg` has a slot.
    pub fn contains(&self, reg: PinnedReg) -> bool {
        self.slots().contains(&reg)
    }

    /// The slot index `reg` occupies, if any. This is its position in the
    /// signature *after* the context pointer, i.e. parameter `1 + i`.
    pub fn position(&self, reg: PinnedReg) -> Option<usize> {
        self.slots().iter().position(|&s| s == reg)
    }

    /// How many slots of `class` there are.
    pub fn count(&self, class: RegClass) -> usize {
        self.slots().iter().filter(|s| s.class == class).count()
    }

    /// [`check_budget`](Self::check_budget) for the machine this process is
    /// running on - the question a JIT host asks.
    pub fn check_host_budget(&self) -> Result<(), BudgetError> {
        self.check_budget(target_lexicon::Triple::host().architecture)
    }

    /// Whether this ABI fits in `arch`'s register budget, and what it would
    /// overflow if not. See [`RegisterBudget`].
    pub fn check_budget(&self, arch: Architecture) -> Result<(), BudgetError> {
        let Some(budget) = RegisterBudget::for_arch(arch) else {
            return Ok(());
        };
        // The context pointer is a parameter and a result like any other slot,
        // and it is the one nobody counts when they count what they pinned.
        let gprs = self.count(RegClass::Gpr) + 1;
        let vectors = self.count(RegClass::Vector);
        if gprs > budget.gpr as usize
            || vectors > budget.vector as usize
            || gprs + vectors > budget.total as usize
        {
            return Err(BudgetError {
                arch,
                budget,
                gprs,
                vectors,
            });
        }
        Ok(())
    }
}

/// What one target can pass in registers under the tail calling convention,
/// counting the context pointer.
///
/// **This mirrors Cranelift's own `compute_arg_locs` and is a lint, not a
/// contract.** The authority is the backend; this is here so a host can ask the
/// question at start-up instead of discovering the answer as a mysterious stack
/// access in a profile. The one direction that is not silent is results: too
/// many of those is a hard compile error out of Cranelift ("Too many return
/// values to fit in registers"), which is the backstop under this lint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterBudget {
    /// Integer values passable in registers, *including* the context pointer.
    pub gpr: u8,
    /// Vector values passable in registers.
    pub vector: u8,
    /// Total register-passed values, across both classes.
    pub total: u8,
}

impl RegisterBudget {
    /// The budget for `arch`, or `None` for a target whose tail-ABI register
    /// assignment is not recorded here - in which case nothing is linted and
    /// Cranelift's own error is the only check.
    pub fn for_arch(arch: Architecture) -> Option<Self> {
        match arch {
            // `x0` is reserved for the return-area pointer and `x1` for the
            // indirect callee, for arguments *and* results, leaving `x2`-`x7`
            // and `v0`-`v7` under a shared cap of sixteen values.
            Architecture::Aarch64(_) => Some(Self {
                gpr: 6,
                vector: 8,
                total: 16,
            }),
            // `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`-`r10`, and `xmm0`-`xmm7`;
            // `r11` and `r15` are reserved as scratch.
            Architecture::X86_64 => Some(Self {
                gpr: 8,
                vector: 8,
                total: 16,
            }),
            _ => None,
        }
    }
}

/// A [`PinnedAbi`] that would not fit in its target's [`RegisterBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetError {
    /// The target the budget belongs to.
    pub arch: Architecture,
    /// What that target can pass.
    pub budget: RegisterBudget,
    /// Integer slots asked for, including the context pointer.
    pub gprs: usize,
    /// Vector slots asked for.
    pub vectors: usize,
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a pinned ABI of {} integer slot(s) (including the context pointer) and {} vector \
             slot(s) does not fit {}'s tail-call budget of {} integer / {} vector / {} total; \
             the overflow would be passed on the stack, which is slower than leaving the \
             register in the context",
            self.gprs,
            self.vectors,
            self.arch,
            self.budget.gpr,
            self.budget.vector,
            self.budget.total
        )
    }
}

impl std::error::Error for BudgetError {}

/// Build the signature every region under `abi` carries: the context pointer
/// followed by one parameter per slot, returning exactly the same list.
///
/// `slot_types` gives the Cranelift type of each slot in order; a frontend that
/// cannot type a slot has already disabled pinning by then (see
/// [`crate::core::pinned_abi_for`]), so this takes them resolved.
pub(crate) fn pinned_signature(slot_types: &[Type]) -> Signature {
    let mut sig = Signature::new(CallConv::Tail);
    sig.params.push(AbiParam::new(types::I64));
    for &ty in slot_types {
        sig.params.push(AbiParam::new(ty));
    }
    // Results are the parameters, which is what makes forwarding free - see the
    // module doc. Not an accident to be tidied up.
    sig.returns = sig.params.clone();
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_keep_the_order_they_were_declared_in() {
        let abi = PinnedAbi::new()
            .with(PinnedReg::gpr(4))
            .with(PinnedReg::gpr(0))
            .with(PinnedReg::vector(1));
        assert_eq!(abi.len(), 3);
        assert_eq!(abi.position(PinnedReg::gpr(0)), Some(1));
        assert_eq!(abi.count(RegClass::Gpr), 2);
        assert_eq!(abi.count(RegClass::Vector), 1);
    }

    #[test]
    #[should_panic(expected = "already pinned")]
    fn a_register_cannot_take_two_slots() {
        let _ = PinnedAbi::new()
            .with(PinnedReg::gpr(4))
            .with(PinnedReg::gpr(4));
    }

    #[test]
    fn the_context_pointer_counts_against_the_budget() {
        let arch = Architecture::Aarch64(target_lexicon::Aarch64Architecture::Aarch64);
        // Five slots plus the context pointer is exactly aarch64's six.
        let five = PinnedAbi::new().with_all((0..5).map(PinnedReg::gpr));
        assert!(five.check_budget(arch).is_ok());
        let six = five.with(PinnedReg::gpr(5));
        assert!(six.check_budget(arch).is_err());
    }

    #[test]
    fn results_are_the_parameters_so_a_link_moves_nothing() {
        let sig = pinned_signature(&[types::I64, types::I64X2]);
        assert_eq!(sig.call_conv, CallConv::Tail);
        assert_eq!(sig.params.len(), 3, "the context pointer plus two slots");
        let params: Vec<Type> = sig.params.iter().map(|p| p.value_type).collect();
        let returns: Vec<Type> = sig.returns.iter().map(|p| p.value_type).collect();
        assert_eq!(params, returns);
    }
}
