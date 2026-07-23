//! Soundness regression tests for the VSA control-flow modelling: conditional
//! transfers other than `Jcc`/`Loop` (namely `LOOPE`/`LOOPNE`/`JRCXZ`) must
//! propagate their taken edge, and `SYSCALL` must fall through with its
//! ABI-clobbered registers invalidated.

use iced_x86::{Decoder, DecoderOptions, Register};
use microlift::vsa::{ValueSet, ValueSetAnalyzer};
use std::collections::BTreeSet;

fn decode_at(bytes: &[u8], ip: u64) -> Vec<iced_x86::Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut out = Vec::new();
    while decoder.can_decode() {
        out.push(decoder.decode());
    }
    out
}

fn multiple(vals: &[i64]) -> ValueSet {
    ValueSet::Multiple(vals.iter().copied().collect::<BTreeSet<_>>())
}

#[test]
fn loope_back_edge_is_propagated() {
    // 0x1000: mov eax, 1     (B8 01 00 00 00)
    // 0x1005: mov ecx, 5     (B9 05 00 00 00)
    // 0x100A: mov eax, 2     (B8 02 00 00 00)   <- loop_top
    // 0x100F: loope 0x100A   (E1 F9)            back-edge to 0x100A
    //
    // The pre-state of loop_top must be the join of the entry (eax=1) and the
    // loope back-edge (eax=2). If the back-edge is dropped, eax stays a single
    // Constant(1) - an unsound under-approximation.
    let bytes = [
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax,1
        0xB9, 0x05, 0x00, 0x00, 0x00, // mov ecx,5
        0xB8, 0x02, 0x00, 0x00, 0x00, // mov eax,2  (loop_top @0x100A)
        0xE1, 0xF9, // loope 0x100A
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);

    let top = vsa
        .value_sets
        .get(&0x100A)
        .expect("loop_top must be analyzed");
    assert_eq!(
        top.get(Register::RAX),
        multiple(&[1, 2]),
        "loope back-edge must contribute eax=2 to loop_top's state"
    );
}

#[test]
fn syscall_falls_through_and_clobbers_rax() {
    // 0x1000: mov eax, 0x40   (B8 40 00 00 00)
    // 0x1005: syscall         (0F 05)
    // 0x1007: jmp rax         (FF E0)
    //
    // SYSCALL returns to 0x1007 (must be analyzed) and clobbers RAX, so the
    // indirect `jmp rax` must be unresolvable - NOT resolved to the stale
    // pre-syscall value 0x40.
    let bytes = [
        0xB8, 0x40, 0x00, 0x00, 0x00, // mov eax,0x40
        0x0F, 0x05, // syscall
        0xFF, 0xE0, // jmp rax
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);

    assert!(
        vsa.value_sets.contains_key(&0x1007),
        "post-syscall instruction must be reached by fallthrough"
    );
    let jmp = insts.iter().find(|i| i.ip() == 0x1007).unwrap();
    assert_eq!(
        vsa.indirect_targets(jmp),
        None,
        "rax is clobbered by syscall; the jmp must not resolve to 0x40"
    );
    assert_eq!(vsa.value_sets[&0x1007].get(Register::RAX), ValueSet::Top);
}

// RBX is callee-saved under the standard x86-64 C ABI, so a value in it would
// survive a CALL *if* the analysis trusted that ABI. The default is conservative
// (trust nothing), so it must NOT survive; ABI-aware precision is opt-in.
const CALLEE_SAVED_ACROSS_CALL: [u8; 11] = [
    0x48, 0xC7, 0xC3, 0x00, 0x30, 0x00, 0x00, // mov rbx, 0x3000   @0x1000
    0xFF, 0xD0, // call rax          @0x1007 (rax unknown)
    0xFF, 0xE3, // jmp rbx           @0x1009
];

#[test]
fn default_is_conservative_no_value_survives_a_call() {
    let insts = decode_at(&CALLEE_SAVED_ACROSS_CALL, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    assert!(vsa.conservative, "conservative must be the default");
    vsa.analyze(&insts);

    let jmp = insts.iter().find(|i| i.ip() == 0x1009).unwrap();
    // Conservative mode clobbers every GPR across the call, so RBX is unknown
    // afterwards and the jmp must not resolve.
    assert_eq!(vsa.indirect_targets(jmp), None);
}

#[test]
fn standard_abi_opt_in_preserves_callee_saved_across_call() {
    let insts = decode_at(&CALLEE_SAVED_ACROSS_CALL, 0x1000);
    // Opt into ABI-aware precision: RBX is callee-saved under SysV/Win64, so the
    // standard clobber set keeps it across the call and the jmp resolves.
    let mut vsa = ValueSetAnalyzer::new().with_standard_abi();
    vsa.analyze(&insts);

    let jmp = insts.iter().find(|i| i.ip() == 0x1009).unwrap();
    assert_eq!(vsa.indirect_targets(jmp), Some(vec![0x3000]));
}

#[test]
fn custom_call_clobbers_switch_off_conservative_mode() {
    let insts = decode_at(&CALLEE_SAVED_ACROSS_CALL, 0x1000);
    // Supplying an explicit clobber set implies ABI-aware mode. Here RBX is NOT
    // in the set, so it is (deliberately) preserved across the call.
    let mut vsa = ValueSetAnalyzer::new().with_call_clobbers(vec![Register::RAX]);
    assert!(!vsa.conservative, "explicit clobbers imply ABI-aware mode");
    vsa.analyze(&insts);

    let jmp = insts.iter().find(|i| i.ip() == 0x1009).unwrap();
    assert_eq!(vsa.indirect_targets(jmp), Some(vec![0x3000]));
}

#[test]
fn cpuid_clobbers_output_registers() {
    // mov rax, 0x3000 ; cpuid ; jmp rax
    // CPUID (a VM exit in the lifter) architecturally writes RAX/RBX/RCX/RDX,
    // which iced reports, so the catch-all invalidates RAX -> jmp unresolvable.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x30, 0x00, 0x00, // mov rax,0x3000  @0x1000
        0x0F, 0xA2, // cpuid           @0x1007
        0xFF, 0xE0, // jmp rax         @0x1009
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let jmp = insts.iter().find(|i| i.ip() == 0x1009).unwrap();
    assert_eq!(vsa.indirect_targets(jmp), None, "cpuid must clobber rax");
}

#[test]
fn software_interrupt_clobbers_registers_in_conservative_mode() {
    // mov rax, 0x3000 ; int 0x80 ; jmp rax
    // `int 0x80` transfers to a handler (a VM exit) that may return a value in
    // RAX. iced reports no GP writes for INT, so only conservative mode catches
    // this - the default. The jmp must not resolve to the stale 0x3000.
    let bytes = [
        0x48, 0xC7, 0xC0, 0x00, 0x30, 0x00, 0x00, // mov rax,0x3000  @0x1000
        0xCD, 0x80, // int 0x80        @0x1007
        0xFF, 0xE0, // jmp rax         @0x1009
    ];
    let insts = decode_at(&bytes, 0x1000);

    // Default (conservative): the interrupt invalidates rax.
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let jmp = insts.iter().find(|i| i.ip() == 0x1009).unwrap();
    assert_eq!(
        vsa.indirect_targets(jmp),
        None,
        "conservative mode must not carry rax across int 0x80"
    );

    // ABI-aware mode trusts iced's (empty) GP write set for INT, so it would
    // carry the stale value - demonstrating exactly why conservative is default.
    let mut precise = ValueSetAnalyzer::new().with_standard_abi();
    precise.analyze(&insts);
    assert_eq!(precise.indirect_targets(jmp), Some(vec![0x3000]));
}

#[test]
fn jrcxz_taken_edge_is_propagated() {
    // 0x1000: mov eax, 7      (B8 07 00 00 00)
    // 0x1005: jrcxz 0x100C    (67 E3 04)  addr-size CX? use rcx form: E3 rel8
    //   Use `E3 05` at 0x1005 (2 bytes) -> next 0x1007, target 0x1007+5=0x100C.
    // 0x1007: mov eax, 9      (B8 09 00 00 00)   fallthrough sets eax=9
    // 0x100C: <target>        (nop landing)      reached by BOTH the jrcxz taken
    //   edge (eax=7) and the fallthrough of `mov eax,9` (eax=9) => {7,9}.
    //   Without the taken edge it would be a single Constant(9).
    let bytes = [
        0xB8, 0x07, 0x00, 0x00, 0x00, // mov eax,7      @0x1000
        0xE3, 0x05, // jrcxz 0x100C  @0x1005
        0xB8, 0x09, 0x00, 0x00, 0x00, // mov eax,9      @0x1007
        0x90, // nop            @0x100C
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);

    let taken = vsa
        .value_sets
        .get(&0x100C)
        .expect("jrcxz taken target must be analyzed");
    assert_eq!(
        taken.get(Register::RAX),
        multiple(&[7, 9]),
        "jrcxz taken edge (eax=7) must reach its target alongside the fallthrough"
    );
}
