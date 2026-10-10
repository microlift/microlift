//! VSA soundness: a resolved indirect `jmp`/`call` target set must contain the real target; unresolved is always sound.
//! Run: `cargo test --features fuzzing --test fuzz_vsa`

#![cfg(feature = "fuzzing")]

#[path = "fuzz/mod.rs"]
mod fuzz;

use microlift::arch::x86_64::X86_64;
use microlift::cfg::resolved_indirect_targets;
use proptest::prelude::*;

const LIFT_BASE: u64 = 0x1000;

/// `mov eax, imm32` then a 2-byte indirect transfer at `LIFT_BASE + 5`.
fn snippet(imm: u32, modrm_op: [u8; 2]) -> Vec<u8> {
    let mut bytes = vec![0xB8]; // mov eax, imm32
    bytes.extend_from_slice(&imm.to_le_bytes());
    bytes.extend_from_slice(&modrm_op);
    bytes
}

proptest! {
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
