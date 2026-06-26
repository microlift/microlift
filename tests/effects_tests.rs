//! Tests for per-instruction register/memory effects (`GuestArch::instruction_effects`).

use microlift::arch::x86_64::X86_64;
use microlift::arch::x86_64::decode::decode_region;
use microlift::core::GuestArch;

fn effects_of(bytes: &[u8]) -> microlift::arch::InstructionEffects {
    let insns = decode_region(bytes, 0x1000);
    X86_64::instruction_effects(&insns[0]).expect("x86 should report effects")
}

#[test]
fn add_reads_both_writes_dest_and_flags() {
    // add rax, rbx
    let e = effects_of(&[0x48, 0x01, 0xD8]);
    assert!(e.reads.contains(&"rax".to_string()));
    assert!(e.reads.contains(&"rbx".to_string()));
    assert!(e.writes.contains(&"rax".to_string()));
    // ADD writes the arithmetic status flags.
    for flag in ["cf", "zf", "sf", "of", "pf", "af"] {
        assert!(e.writes.contains(&flag.to_string()), "missing flag write: {flag}");
    }
    assert!(!e.reads_memory);
    assert!(!e.writes_memory);
}

#[test]
fn mov_imm_writes_only_dest() {
    // mov rax, 42
    let e = effects_of(&[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]);
    assert!(e.writes.contains(&"rax".to_string()));
    assert!(!e.reads.contains(&"rax".to_string()));
    assert!(!e.reads_memory && !e.writes_memory);
}

#[test]
fn store_to_memory_sets_write_memory() {
    // mov [rax], rbx
    let e = effects_of(&[0x48, 0x89, 0x18]);
    assert!(e.reads.contains(&"rax".to_string())); // address base
    assert!(e.reads.contains(&"rbx".to_string())); // stored value
    assert!(e.writes_memory);
    assert!(!e.reads_memory);
}

#[test]
fn load_from_memory_sets_read_memory() {
    // mov rbx, [rax]
    let e = effects_of(&[0x48, 0x8B, 0x18]);
    assert!(e.reads.contains(&"rax".to_string()));
    assert!(e.writes.contains(&"rbx".to_string()));
    assert!(e.reads_memory);
    assert!(!e.writes_memory);
}
