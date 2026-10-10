//! Pinned register files: guest state kept in machine registers across links ([`crate::core::link`]).
//!
//! - A pinned region is `tail (ctx, slot0, ...) -> (ctx, slot0, ...)`; results == parameters is
//!   load-bearing (a link lowers to a branch).
//! - Every region sharing a link table must use the **same** [`PinnedAbi`]; nothing checks at run time.
//! - Full `sync_to_context` skips pinned slots; masked syncs never do.

use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use target_lexicon::Architecture;

/// Most slots a [`PinnedAbi`] holds (keeps it `Copy`).
pub const MAX_PINNED_SLOTS: usize = 16;

/// Register bank of a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    Gpr,
    Vector,
}

/// A guest register; `index` is uninterpreted by the engine and passed to
/// [`GuestArch::pinned_slot_var`](crate::core::GuestArch::pinned_slot_var).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedReg {
    pub class: RegClass,
    pub index: u16,
}

impl PinnedReg {
    pub const fn gpr(index: u16) -> Self {
        Self {
            class: RegClass::Gpr,
            index,
        }
    }

    pub const fn vector(index: u16) -> Self {
        Self {
            class: RegClass::Vector,
            index,
        }
    }
}

/// Ordered guest registers pinned into the link ABI; order is the parameter order, duplicates are refused.
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
    /// No pinned registers: inert.
    pub const fn new() -> Self {
        Self {
            slots: [PinnedReg::gpr(0); MAX_PINNED_SLOTS],
            len: 0,
        }
    }

    /// # Panics
    ///
    /// If `reg` is already pinned or the ABI is full ([`MAX_PINNED_SLOTS`]).
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

    #[must_use]
    pub fn with_all(mut self, regs: impl IntoIterator<Item = PinnedReg>) -> Self {
        for reg in regs {
            self = self.with(reg);
        }
        self
    }

    pub fn slots(&self) -> &[PinnedReg] {
        &self.slots[..self.len as usize]
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, reg: PinnedReg) -> bool {
        self.slots().contains(&reg)
    }

    /// Slot index of `reg`; its parameter is `1 + i` (after the context pointer).
    pub fn position(&self, reg: PinnedReg) -> Option<usize> {
        self.slots().iter().position(|&s| s == reg)
    }

    pub fn count(&self, class: RegClass) -> usize {
        self.slots().iter().filter(|s| s.class == class).count()
    }

    /// [`check_budget`](Self::check_budget) for the host architecture.
    pub fn check_host_budget(&self) -> Result<(), BudgetError> {
        self.check_budget(target_lexicon::Triple::host().architecture)
    }

    /// Whether this ABI fits `arch`'s [`RegisterBudget`].
    pub fn check_budget(&self, arch: Architecture) -> Result<(), BudgetError> {
        let Some(budget) = RegisterBudget::for_arch(arch) else {
            return Ok(());
        };
        // The context pointer takes a GPR slot.
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

/// What a target passes in registers under the tail convention, counting the context pointer.
///
/// A lint mirroring Cranelift's `compute_arg_locs`; overflow is silent for arguments, an error for results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterBudget {
    /// Includes the context pointer.
    pub gpr: u8,
    pub vector: u8,
    pub total: u8,
}

impl RegisterBudget {
    /// `None` for targets not recorded here (nothing is linted).
    pub fn for_arch(arch: Architecture) -> Option<Self> {
        match arch {
            // x0/x1 reserved, x8 skipped: x2-x7, x9-x15 (the fork widens `tail` past AAPCS64's eight).
            Architecture::Aarch64(_) => Some(Self {
                gpr: 13,
                vector: 8,
                total: 24,
            }),
            // r11 and r15 are reserved as scratch.
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
    pub arch: Architecture,
    pub budget: RegisterBudget,
    /// Includes the context pointer.
    pub gprs: usize,
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

/// Context pointer plus one parameter per slot, returning the same list.
pub(crate) fn pinned_signature(slot_types: &[Type]) -> Signature {
    let mut sig = Signature::new(CallConv::Tail);
    sig.params.push(AbiParam::new(types::I64));
    for &ty in slot_types {
        sig.params.push(AbiParam::new(ty));
    }
    // Results == parameters is deliberate; see the module doc.
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
