//! VSA soundness fuzzing; an indirect `jmp`/`call` whose only possible
//! runtime target is known by construction. Value-Set Analysis is an
//! over-approximation: whenever it resolves the site it must include the real
//! target; leaving the site unresolved is always sound. It must never resolve
//! to a set that omits a reachable target.
//!
//! Run: `cargo test --features fuzzing --test fuzz_vsa`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use microlift::arch::x86_64::X86_64;
use microlift::cfg::resolved_indirect_targets;
use proptest::prelude::*;

const LIFT_BASE: u64 = 0x1000;

/// `mov eax, imm32` (zero-extends into rax), then a 2-byte indirect transfer.
/// The indirect site is at `LIFT_BASE + 5`.
fn snippet(imm: u32, modrm_op: [u8; 2]) -> Vec<u8> {
    let mut bytes = vec![0xB8]; // mov eax, imm32
    bytes.extend_from_slice(&imm.to_le_bytes());
    bytes.extend_from_slice(&modrm_op);
    bytes
}

proptest! {
    /// `jmp rax` with a statically-known `rax`: if VSA resolves the site, the
    /// resolved set must contain the true target.
    #[test]
    fn indirect_jmp_target_is_sound(imm in any::<u32>()) {
        let bytes = snippet(imm, [0xFF, 0xE0]); // jmp rax
        let resolved = resolved_indirect_targets::<X86_64>(&bytes, LIFT_BASE, &[]);
        let site = LIFT_BASE + 5;
        if let Some(set) = resolved.jumps.get(&site) {
            prop_assert!(
                set.contains(&(imm as u64)),
                "VSA resolved jmp@{:#x} to {:?} but omitted the real target {:#x}",
                site, set, imm
            );
        }
    }

    /// Same soundness property for `call rax`.
    #[test]
    fn indirect_call_target_is_sound(imm in any::<u32>()) {
        let bytes = snippet(imm, [0xFF, 0xD0]); // call rax
        let resolved = resolved_indirect_targets::<X86_64>(&bytes, LIFT_BASE, &[]);
        let site = LIFT_BASE + 5;
        if let Some(set) = resolved.calls.get(&site) {
            prop_assert!(
                set.contains(&(imm as u64)),
                "VSA resolved call@{:#x} to {:?} but omitted the real target {:#x}",
                site, set, imm
            );
        }
    }

    /// A register copy between the constant and the branch must not lose the
    /// target: `mov eax,imm ; mov ecx,eax ; jmp rcx`.
    #[test]
    fn indirect_jmp_through_copy_is_sound(imm in any::<u32>()) {
        let mut bytes = vec![0xB8]; // mov eax, imm32
        bytes.extend_from_slice(&imm.to_le_bytes());
        bytes.extend_from_slice(&[0x89, 0xC1]); // mov ecx, eax
        bytes.extend_from_slice(&[0xFF, 0xE1]); // jmp rcx
        let site = LIFT_BASE + 7;
        let resolved = resolved_indirect_targets::<X86_64>(&bytes, LIFT_BASE, &[]);
        if let Some(set) = resolved.jumps.get(&site) {
            prop_assert!(
                set.contains(&(imm as u64)),
                "VSA lost the target through a register copy: {:?} omits {:#x}",
                set, imm
            );
        }
    }
}
