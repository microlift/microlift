use microlift::arch::x86_64::X86_64;
use microlift::core::{LifterConfig, lift_to_function};

// mov rax, [rdi] ; mov rcx, [rdi+8] ; mov [rsi], rax ; mov [rsi+8], rcx ; ret
const CODE: [u8; 15] = [
    0x48, 0x8B, 0x07, 0x48, 0x8B, 0x4F, 0x08, 0x48, 0x89, 0x06, 0x48, 0x89, 0x4E, 0x08, 0xC3,
];

fn fences(tso: bool) -> usize {
    let config = LifterConfig {
        total_store_order: tso,
        outline_helpers: false,
        ..LifterConfig::default()
    };
    let func = lift_to_function::<X86_64>(&CODE, 0x1000, config).unwrap();
    func.display().to_string().matches("fence").count()
}

#[test]
fn tso_fences_ordered_guest_accesses() {
    assert_eq!(fences(false), 0);
    // Before each of the two loads after the first, and before each store.
    assert!(fences(true) >= 3, "only {} fences", fences(true));
}
