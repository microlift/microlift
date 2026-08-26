//! Pinned register files: keeping guest state in host registers across a link.
//!
//! [`crate::core::link`] lets a region resolve its own successor; this lets it
//! *branch* there, carrying part of the guest register file in machine registers
//! rather than through the `RegisterContext`.
//!
//! **The pinned file is the signature.** A pinned region is compiled with
//! [`CallConv::Tail`](cranelift_codegen::isa::CallConv::Tail) as
//! `tail (ctx, slot0, ...) -> (ctx, slot0, ...)`. The results being the
//! parameters is load-bearing: the tail convention assigns both from the same
//! register sequence, so every slot stays where it is and a link lowers to a
//! branch.
//!
//! The host declares the ordered [`PinnedAbi`], and owes two things: every region
//! reachable through one link table must be lifted under the **same** ABI, slot
//! for slot (nothing checks it at run time); and slots are scarce, with overflow
//! past [`RegisterBudget`] silent for arguments ([`PinnedAbi::check_budget`]).
//!
//! A pinned slot changes exactly three things - it arrives as a parameter at
//! entry, is returned rather than stored on exit to the host, and is an argument
//! to a taken link. Hence: **the full [`GuestArch::sync_to_context`] skips pinned
//! slots, and the masked syncs never do.**
//!
//! [`GuestArch::sync_to_context`]: crate::core::GuestArch::sync_to_context

use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use target_lexicon::Architecture;

/// The most slots a [`PinnedAbi`] can hold.
///
/// Comfortably above every target's [`RegisterBudget`]; it exists so the ABI is
/// a `Copy` value that can live in a `LifterConfig` cloned once per region.
pub const MAX_PINNED_SLOTS: usize = 16;

/// Which of a guest's register banks a slot names.
///
/// Only two: the granularity every guest frontend can answer for, and the one
/// a machine ABI actually distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// An integer/pointer register - x86-64's `RAX`..`R15`, aarch64's `X0`..`X30`.
    Gpr,
    /// A vector register - x86-64's `XMM0`..`XMM15`.
    Vector,
}

/// One guest register, in the guest's own architectural numbering.
///
/// The engine never interprets `index`; it hands the pair back to the frontend
/// ([`GuestArch::pinned_slot_var`](crate::core::GuestArch::pinned_slot_var)).
/// For x86-64 that is [`GPR_ORDER`](crate::arch::x86_64::regs::GPR_ORDER) or
/// the XMM number; [`crate::arch::x86_64::pin`] is the readable spelling.
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
/// construction: a register in two slots has two values on entry.
///
/// `Copy` and sized rather than heap-allocated, because a `LifterConfig` is
/// cloned once per region.
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
    /// If `reg` is already pinned, or the ABI is full ([`MAX_PINNED_SLOTS`]). Both
    /// are programming errors in a declaration written once at start-up, and a
    /// silently dropped slot would be a wrong-code bug in every region after it.
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
/// **A lint, not a contract** - it mirrors Cranelift's own `compute_arg_locs`
/// so a host can ask at start-up rather than find a mysterious stack access in
/// a profile; the authority is the backend. Too many *results* is the one case
/// that is not silent: Cranelift errors outright.
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
            // indirect callee, for arguments *and* results; `x8` is the
            // indirect-result register and is skipped. That leaves `x2`-`x7` and
            // `x9`-`x15` - thirteen - plus `v0`-`v7`, under a shared cap of
            // twenty-four values.
            //
            // Past AAPCS64's eight because the `tail` convention in microlift's
            // Cranelift fork was widened to `x15` for exactly this: `tail`
            // clobbers every register, so there is no callee-saved contract to
            // respect, and every integer slot that does not fit here is a store
            // and a reload of a guest register at every region boundary.
            Architecture::Aarch64(_) => Some(Self {
                gpr: 13,
                vector: 8,
                total: 24,
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
/// A frontend that cannot type a slot has already disabled pinning (see
/// [`crate::core::pinned_abi_for`]), so `slot_types` arrives resolved.
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
        // The budget counts the context pointer, so one fewer guest register
        // fits than the figure suggests - which is the whole point of the test
        // and is what stays true as the budget moves.
        let gpr = RegisterBudget::for_arch(arch).expect("aarch64 has a budget").gpr;
        let full = PinnedAbi::new().with_all((0..u16::from(gpr) - 1).map(PinnedReg::gpr));
        assert!(full.check_budget(arch).is_ok(), "{gpr} slots counting the context pointer");
        let one_too_many = full.with(PinnedReg::gpr(u16::from(gpr) - 1));
        assert!(one_too_many.check_budget(arch).is_err());
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
