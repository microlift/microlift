//! Comprehensive Value Set Analysis tests covering lattice operations,
//! dataflow propagation, and edge cases in the transfer function.

use iced_x86::{Decoder, DecoderOptions, Register};
use microlift::context::MemorySegment;
use microlift::vsa::{RegisterValueSet, ValueSet, ValueSetAnalyzer};
use std::collections::{BTreeSet, HashSet};

// ── helpers ──────────────────────────────────────────────────────────────────

fn decode_at(bytes: &[u8], ip: u64) -> Vec<iced_x86::Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut out = Vec::new();
    while decoder.can_decode() {
        out.push(decoder.decode());
    }
    out
}

fn make_multiple(vals: &[i64]) -> ValueSet {
    let mut s = BTreeSet::new();
    for &v in vals {
        s.insert(v);
    }
    ValueSet::Multiple(s)
}

// ── ValueSet::join ────────────────────────────────────────────────────────────

#[test]
fn test_join_empty_is_identity() {
    let c = ValueSet::Constant(42);
    assert_eq!(ValueSet::Empty.join(&c), c);
    assert_eq!(c.join(&ValueSet::Empty), c);
    assert_eq!(ValueSet::Empty.join(&ValueSet::Empty), ValueSet::Empty);
    assert_eq!(ValueSet::Empty.join(&ValueSet::Top), ValueSet::Top);
}

#[test]
fn test_join_same_constant_is_idempotent() {
    let a = ValueSet::Constant(7);
    assert_eq!(a.join(&a), ValueSet::Constant(7));
}

#[test]
fn test_join_two_different_constants_becomes_multiple() {
    let result = ValueSet::Constant(1).join(&ValueSet::Constant(2));
    assert_eq!(result, make_multiple(&[1, 2]));
}

#[test]
fn test_join_multiple_and_constant_grows_set() {
    let m = make_multiple(&[1, 2, 3]);
    let result = m.join(&ValueSet::Constant(4));
    assert_eq!(result, make_multiple(&[1, 2, 3, 4]));
}

#[test]
fn test_join_multiple_at_max_size_stays_multiple() {
    // Exactly MAX_SET_SIZE = 64 distinct values — must remain Multiple.
    let mut vs = ValueSet::Empty;
    for i in 0..64i64 {
        vs = vs.join(&ValueSet::Constant(i));
    }
    assert!(
        matches!(vs, ValueSet::Multiple(_)),
        "expected Multiple, got {vs:?}"
    );
}

#[test]
fn test_join_multiple_over_max_size_widens_to_top() {
    // 65 distinct values must widen to Top.
    let mut vs = ValueSet::Empty;
    for i in 0..65i64 {
        vs = vs.join(&ValueSet::Constant(i));
    }
    assert_eq!(vs, ValueSet::Top);
}

#[test]
fn test_join_two_multiples_merge() {
    let a = make_multiple(&[1, 2]);
    let b = make_multiple(&[3, 4]);
    assert_eq!(a.join(&b), make_multiple(&[1, 2, 3, 4]));
}

#[test]
fn test_join_two_multiples_over_limit_widens() {
    // Each side has 33 elements; union = 66 -> Top.
    let mut a = BTreeSet::new();
    let mut b = BTreeSet::new();
    for i in 0..33i64 {
        a.insert(i);
    }
    for i in 33..66i64 {
        b.insert(i);
    }
    let result = ValueSet::Multiple(a).join(&ValueSet::Multiple(b));
    assert_eq!(result, ValueSet::Top);
}

#[test]
fn test_join_interval_union_takes_outer_bounds() {
    let a = ValueSet::Interval { min: 1, max: 5 };
    let b = ValueSet::Interval { min: 3, max: 10 };
    assert_eq!(a.join(&b), ValueSet::Interval { min: 1, max: 10 });
}

#[test]
fn test_join_disjoint_intervals_union() {
    let a = ValueSet::Interval { min: 1, max: 3 };
    let b = ValueSet::Interval { min: 100, max: 200 };
    // Both are intervals, join takes outer bounds.
    assert_eq!(a.join(&b), ValueSet::Interval { min: 1, max: 200 });
}

#[test]
fn test_join_interval_and_constant_widens_to_top() {
    // Mixed Interval + Constant -> Top (conservative).
    let iv = ValueSet::Interval { min: 1, max: 10 };
    let c = ValueSet::Constant(5);
    assert_eq!(iv.join(&c), ValueSet::Top);
    assert_eq!(c.join(&iv), ValueSet::Top);
}

#[test]
fn test_join_top_absorbs_everything() {
    assert_eq!(ValueSet::Top.join(&ValueSet::Constant(1)), ValueSet::Top);
    assert_eq!(ValueSet::Constant(1).join(&ValueSet::Top), ValueSet::Top);
    assert_eq!(ValueSet::Top.join(&ValueSet::Empty), ValueSet::Top);
    assert_eq!(ValueSet::Top.join(&ValueSet::Top), ValueSet::Top);
    assert_eq!(
        ValueSet::Top.join(&ValueSet::Interval { min: 0, max: 99 }),
        ValueSet::Top
    );
}

// ── ValueSet::abstract_add ────────────────────────────────────────────────────

#[test]
fn test_abstract_add_constants() {
    assert_eq!(
        ValueSet::Constant(10).abstract_add(&ValueSet::Constant(5)),
        ValueSet::Constant(15)
    );
}

#[test]
fn test_abstract_add_constant_wraps() {
    assert_eq!(
        ValueSet::Constant(i64::MAX).abstract_add(&ValueSet::Constant(1)),
        ValueSet::Constant(i64::MIN)
    );
}

#[test]
fn test_abstract_add_multiple_plus_constant() {
    let m = make_multiple(&[10, 20]);
    assert_eq!(
        m.abstract_add(&ValueSet::Constant(5)),
        make_multiple(&[15, 25])
    );
    // Commutative
    let m = make_multiple(&[10, 20]);
    assert_eq!(
        ValueSet::Constant(5).abstract_add(&m),
        make_multiple(&[15, 25])
    );
}

#[test]
fn test_abstract_add_interval_plus_interval() {
    let a = ValueSet::Interval { min: 1, max: 3 };
    let b = ValueSet::Interval { min: 10, max: 20 };
    assert_eq!(a.abstract_add(&b), ValueSet::Interval { min: 11, max: 23 });
}

#[test]
fn test_abstract_add_constant_plus_interval() {
    let c = ValueSet::Constant(100);
    let iv = ValueSet::Interval { min: 1, max: 9 };
    assert_eq!(
        c.abstract_add(&iv),
        ValueSet::Interval { min: 101, max: 109 }
    );
    // Commutative
    assert_eq!(
        iv.abstract_add(&c),
        ValueSet::Interval { min: 101, max: 109 }
    );
}

#[test]
fn test_abstract_add_empty_stays_empty() {
    assert_eq!(
        ValueSet::Empty.abstract_add(&ValueSet::Constant(5)),
        ValueSet::Empty
    );
    assert_eq!(
        ValueSet::Constant(5).abstract_add(&ValueSet::Empty),
        ValueSet::Empty
    );
}

#[test]
fn test_abstract_add_top_propagates() {
    assert_eq!(
        ValueSet::Top.abstract_add(&ValueSet::Constant(1)),
        ValueSet::Top
    );
    assert_eq!(
        ValueSet::Constant(1).abstract_add(&ValueSet::Top),
        ValueSet::Top
    );
}

#[test]
fn test_abstract_add_multiple_plus_multiple_falls_to_top() {
    // Multiple + Multiple is not handled specially -> Top.
    let a = make_multiple(&[1, 2]);
    let b = make_multiple(&[3, 4]);
    assert_eq!(a.abstract_add(&b), ValueSet::Top);
}

// ── ValueSet::abstract_sub ────────────────────────────────────────────────────

#[test]
fn test_abstract_sub_constants() {
    assert_eq!(
        ValueSet::Constant(100).abstract_sub(&ValueSet::Constant(30)),
        ValueSet::Constant(70)
    );
}

#[test]
fn test_abstract_sub_constant_wraps() {
    assert_eq!(
        ValueSet::Constant(i64::MIN).abstract_sub(&ValueSet::Constant(1)),
        ValueSet::Constant(i64::MAX)
    );
}

#[test]
fn test_abstract_sub_constant_minus_interval_reverses_bounds() {
    // C - [lo, hi] = [C-hi, C-lo]
    let c = ValueSet::Constant(100);
    let iv = ValueSet::Interval { min: 10, max: 20 };
    assert_eq!(c.abstract_sub(&iv), ValueSet::Interval { min: 80, max: 90 });
}

#[test]
fn test_abstract_sub_interval_minus_constant_is_top() {
    // Interval - Constant is not handled -> Top.
    let iv = ValueSet::Interval { min: 1, max: 10 };
    assert_eq!(iv.abstract_sub(&ValueSet::Constant(1)), ValueSet::Top);
}

#[test]
fn test_abstract_sub_empty_stays_empty() {
    assert_eq!(
        ValueSet::Empty.abstract_sub(&ValueSet::Constant(1)),
        ValueSet::Empty
    );
    assert_eq!(
        ValueSet::Constant(1).abstract_sub(&ValueSet::Empty),
        ValueSet::Empty
    );
}

#[test]
fn test_abstract_sub_top_propagates() {
    assert_eq!(
        ValueSet::Top.abstract_sub(&ValueSet::Constant(1)),
        ValueSet::Top
    );
}

// ── ValueSet::abstract_and ────────────────────────────────────────────────────

#[test]
fn test_abstract_and_constants() {
    assert_eq!(
        ValueSet::Constant(0xFF).abstract_and(&ValueSet::Constant(0x0F)),
        ValueSet::Constant(0x0F)
    );
}

#[test]
fn test_abstract_and_zero_short_circuits_to_zero() {
    // The zero short-circuit fires only when neither operand is Top/Empty.
    // Interval & 0 = 0
    assert_eq!(
        ValueSet::Interval { min: 1, max: 99 }.abstract_and(&ValueSet::Constant(0)),
        ValueSet::Constant(0)
    );
    assert_eq!(
        ValueSet::Constant(0).abstract_and(&ValueSet::Interval { min: 1, max: 99 }),
        ValueSet::Constant(0)
    );
    // Multiple & 0 = 0
    assert_eq!(
        make_multiple(&[1, 2, 3]).abstract_and(&ValueSet::Constant(0)),
        ValueSet::Constant(0)
    );
}

#[test]
fn test_abstract_and_top_with_zero_is_zero() {
    // Top & 0 = 0: the zero arm now fires before the Top arm.
    assert_eq!(
        ValueSet::Top.abstract_and(&ValueSet::Constant(0)),
        ValueSet::Constant(0)
    );
    assert_eq!(
        ValueSet::Constant(0).abstract_and(&ValueSet::Top),
        ValueSet::Constant(0)
    );
}

#[test]
fn test_abstract_and_top_with_non_zero_is_top() {
    assert_eq!(
        ValueSet::Top.abstract_and(&ValueSet::Constant(1)),
        ValueSet::Top
    );
}

#[test]
fn test_abstract_and_empty_stays_empty() {
    assert_eq!(
        ValueSet::Empty.abstract_and(&ValueSet::Constant(0xFF)),
        ValueSet::Empty
    );
}

// ── ValueSet::abstract_xor ───────────────────────────────────────────────────

#[test]
fn test_abstract_xor_constants() {
    assert_eq!(
        ValueSet::Constant(0b1010).abstract_xor(&ValueSet::Constant(0b1100)),
        ValueSet::Constant(0b0110)
    );
}

#[test]
fn test_abstract_xor_self_is_zero() {
    let c = ValueSet::Constant(12345);
    assert_eq!(c.abstract_xor(&c), ValueSet::Constant(0));
}

#[test]
fn test_abstract_xor_top_propagates() {
    assert_eq!(
        ValueSet::Top.abstract_xor(&ValueSet::Constant(1)),
        ValueSet::Top
    );
    assert_eq!(
        ValueSet::Constant(1).abstract_xor(&ValueSet::Top),
        ValueSet::Top
    );
}

#[test]
fn test_abstract_xor_empty_stays_empty() {
    assert_eq!(
        ValueSet::Empty.abstract_xor(&ValueSet::Constant(5)),
        ValueSet::Empty
    );
}

// ── ValueSet::is_resolvable / targets ────────────────────────────────────────

#[test]
fn test_is_resolvable() {
    assert!(ValueSet::Constant(0).is_resolvable());
    assert!(make_multiple(&[1, 2]).is_resolvable());
    assert!(!ValueSet::Top.is_resolvable());
    assert!(!ValueSet::Empty.is_resolvable());
    assert!(!ValueSet::Interval { min: 1, max: 5 }.is_resolvable());
}

#[test]
fn test_targets_constant() {
    assert_eq!(ValueSet::Constant(0x3000).targets(), vec![0x3000u64]);
}

#[test]
fn test_targets_multiple_are_sorted() {
    // BTreeSet is sorted, so targets come out in ascending order.
    let m = make_multiple(&[0x4000, 0x1000, 0x2000]);
    assert_eq!(m.targets(), vec![0x1000u64, 0x2000, 0x4000]);
}

#[test]
fn test_targets_non_resolvable_are_empty() {
    assert!(ValueSet::Top.targets().is_empty());
    assert!(ValueSet::Empty.targets().is_empty());
    assert!(ValueSet::Interval { min: 1, max: 5 }.targets().is_empty());
}

#[test]
fn test_targets_negative_constant_reinterprets_as_u64() {
    // Negative i64 stored -> cast to u64 on output.
    assert_eq!(ValueSet::Constant(-1).targets(), vec![u64::MAX]);
}

// ── RegisterValueSet ─────────────────────────────────────────────────────────

#[test]
fn test_register_value_set_unknown_register_returns_top() {
    let rvs = RegisterValueSet::new();
    assert_eq!(rvs.get(Register::RAX), ValueSet::Top);
    assert_eq!(rvs.get(Register::R15), ValueSet::Top);
}

#[test]
fn test_register_value_set_join_merges_registers() {
    let mut a = RegisterValueSet::new();
    a.set(Register::RAX, ValueSet::Constant(1));

    let mut b = RegisterValueSet::new();
    b.set(Register::RAX, ValueSet::Constant(2));
    b.set(Register::RBX, ValueSet::Constant(5));

    let joined = a.join(&b);
    // RAX: Constant(1) ⊔ Constant(2) = Multiple({1,2})
    assert_eq!(joined.get(Register::RAX), make_multiple(&[1, 2]));
    // RBX: a doesn't know RBX (Top) ⊔ Constant(5) = Top
    assert_eq!(joined.get(Register::RBX), ValueSet::Top);
}

#[test]
fn test_register_value_set_join_preserves_unmodified_register() {
    let mut a = RegisterValueSet::new();
    a.set(Register::RCX, ValueSet::Constant(99));
    // b doesn't touch RCX
    let b = RegisterValueSet::new();
    let joined = a.join(&b);
    // RCX in a is Constant(99); b contributes Top -> join = Top.
    // (a registers not in b are already in result; b's Top for unknown regs
    //  means the join computes existing ⊔ Top = Top for every reg in b —
    //  but b has no explicit entries, so only a's entries survive unchanged.)
    assert_eq!(joined.get(Register::RCX), ValueSet::Constant(99));
}

// ── Analyzer: basic dataflow ──────────────────────────────────────────────────

#[test]
fn test_empty_instructions_no_panic() {
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&[]);
    assert!(vsa.value_sets.is_empty());
}

#[test]
fn test_xor_self_zero_idiom() {
    // xor rax, rax  ->  rax = 0
    let bytes = [0x48u8, 0x31, 0xC0]; // xor rax, rax
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(0));
}

#[test]
fn test_xor_different_registers_is_top() {
    // xor rax, rbx  ->  rax = Top (both operands unknown)
    let bytes = [0x48u8, 0x31, 0xD8]; // xor rax, rbx
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_mov_immediate_constant_propagation() {
    // mov rax, 42  ->  rax = 42
    let bytes = [0x48u8, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(42));
}

#[test]
fn test_mov_register_to_register_propagates_value() {
    // mov rax, 77 ; mov rbx, rax  ->  rbx = 77
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x4D, 0x00, 0x00, 0x00, // mov rax, 77  (ip 0x1000, 7 bytes)
        0x48, 0x89, 0xC3, // mov rbx, rax  (ip 0x1007, 3 bytes)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let state = vsa.value_sets[&0x1007].clone();
    let post = vsa.transfer(&state, &insts[1]);
    assert_eq!(post.get(Register::RBX), ValueSet::Constant(77));
}

#[test]
fn test_add_chain_propagates_constant() {
    // mov rax, 10  ; add rax, 5  ->  rax = 15
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00, // mov rax, 10  (ip 0x1000)
        0x48, 0x83, 0xC0, 0x05, // add rax, 5   (ip 0x1007)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_add = vsa.value_sets[&0x1007].clone();
    assert_eq!(pre_add.get(Register::RAX), ValueSet::Constant(10));
    let post = vsa.transfer(&pre_add, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(15));
}

#[test]
fn test_sub_chain_propagates_constant() {
    // mov rax, 100 ; sub rax, 30  ->  rax = 70
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100  (ip 0x1000)
        0x48, 0x83, 0xE8, 0x1E, // sub rax, 30   (ip 0x1007)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_sub = vsa.value_sets[&0x1007].clone();
    assert_eq!(pre_sub.get(Register::RAX), ValueSet::Constant(100));
    let post = vsa.transfer(&pre_sub, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(70));
}

#[test]
fn test_and_with_constant_mask() {
    // mov rax, 0xFF0F ; and rax, 0x0F  ->  rax = 0x0F
    let bytes = [
        // mov rax, imm64 = 0xFF0F  (ip 0x1000, 10 bytes)
        0x48u8, 0xB8, 0x0F, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // and rax, 0x0F  (ip 0x100A, 4 bytes)
        0x48, 0x83, 0xE0, 0x0F,
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_and = vsa.value_sets[&0x100A].clone();
    assert_eq!(pre_and.get(Register::RAX), ValueSet::Constant(0xFF0F));
    let post = vsa.transfer(&pre_and, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(0x0F));
}

#[test]
fn test_or_kills_destination() {
    // mov rax, 42 ; or rax, rbx  ->  rax = Top
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42  (ip 0x1000)
        0x48, 0x09, 0xD8, // or rax, rbx   (ip 0x1007)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_or = vsa.value_sets[&0x1007].clone();
    let post = vsa.transfer(&pre_or, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_unknown_instruction_kills_destination() {
    // mov rax, 42 ; imul rax, rbx  ->  rax = Top (imul not in transfer)
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42  (ip 0x1000, 7 bytes)
        0x48, 0x0F, 0xAF, 0xC3, // imul rax, rbx (ip 0x1007, 4 bytes)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_imul = vsa.value_sets[&0x1007].clone();
    assert_eq!(pre_imul.get(Register::RAX), ValueSet::Constant(42));
    let post = vsa.transfer(&pre_imul, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_lea_known_base_propagates() {
    // mov rbx, 100 ; lea rax, [rbx+8]  ->  rax = 108
    let bytes = [
        0x48u8, 0xC7, 0xC3, 0x64, 0x00, 0x00, 0x00, // mov rbx, 100  (ip 0x1000, 7 bytes)
        0x48, 0x8D, 0x43, 0x08, // lea rax, [rbx+8]  (ip 0x1007, 4 bytes)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_lea = vsa.value_sets[&0x1007].clone();
    let post = vsa.transfer(&pre_lea, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(108));
}

#[test]
fn test_lea_with_index_register_is_top() {
    // lea rax, [rbx+rcx*2]  ->  rax = Top (index register present)
    // REX.W=0x48, 8D, ModRM(00 000 100)=0x04, SIB(01 001 011)=0x4B
    let bytes = [0x48u8, 0x8D, 0x04, 0x4B];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_lea_with_no_base_register_is_top() {
    // lea rax, [rcx*4 + 0]  ->  Top (no base, only scaled index)
    // REX.W, 8D, ModRM(00 000 100)=0x04, SIB(10 001 101)=0x8D + disp32 zero
    // scale=2(4x), index=001(rcx), base=101(disp32) -> SIB=0b10_001_101=0x8D
    let bytes = [0x48u8, 0x8D, 0x04, 0x8D, 0x00, 0x00, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

// ── Analyzer: control flow ────────────────────────────────────────────────────

#[test]
fn test_conditional_branch_propagates_to_both_successors() {
    // 0x1000: mov rax, 42   [48 C7 C0 2A 00 00 00]  7 bytes
    // 0x1007: je 0x100A     [74 01]                   2 bytes  (fallthrough=0x1009, taken=0x100A)
    // 0x1009: ret           [C3]                      1 byte   (fallthrough target)
    // 0x100A: ret           [C3]                      1 byte   (taken target)
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
        0x74, 0x01, // je +1  ->  0x100A
        0xC3, // ret (fallthrough)
        0xC3, // ret (taken)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let fallthrough = vsa
        .value_sets
        .get(&0x1009)
        .expect("fallthrough state missing");
    let taken = vsa.value_sets.get(&0x100A).expect("taken state missing");
    assert_eq!(fallthrough.get(Register::RAX), ValueSet::Constant(42));
    assert_eq!(taken.get(Register::RAX), ValueSet::Constant(42));
}

#[test]
fn test_ret_does_not_propagate_to_next_instruction() {
    // 0x1000: ret           [C3]
    // 0x1001: mov rax, 99  [48 C7 C0 63 00 00 00]  (unreachable)
    let bytes = [
        0xC3u8, // ret
        0x48, 0xC7, 0xC0, 0x63, 0x00, 0x00, 0x00, // mov rax, 99  (dead)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    // No state should have been propagated to 0x1001.
    assert!(
        vsa.value_sets.get(&0x1001).is_none(),
        "ret must not fall through to next instruction"
    );
}

#[test]
fn test_direct_jump_propagates_state_to_target() {
    // 0x1000: mov rax, 77   [48 C7 C0 4D 00 00 00]  7 bytes
    // 0x1007: jmp 0x100E    [E9 02 00 00 00]          5 bytes  (next=0x100C, target=0x100E)
    // 0x100C: nop           [90]                      dead
    // 0x100D: nop           [90]                      dead
    // 0x100E: ret           [C3]
    let bytes = [
        0x48u8, 0xC7, 0xC0, 0x4D, 0x00, 0x00, 0x00, // mov rax, 77
        0xE9, 0x02, 0x00, 0x00, 0x00, // jmp 0x100E
        0x90, // nop (dead)
        0x90, // nop (dead)
        0xC3, // ret (jump target)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let target_state = vsa
        .value_sets
        .get(&0x100E)
        .expect("jump target state missing");
    assert_eq!(target_state.get(Register::RAX), ValueSet::Constant(77));
    // Dead code must not have been visited.
    assert!(vsa.value_sets.get(&0x100C).is_none());
    assert!(vsa.value_sets.get(&0x100D).is_none());
}

// ── Analyzer: indirect jump resolution ───────────────────────────────────────

#[test]
fn test_indirect_jump_single_constant_target() {
    // jmp rax where rax = Constant(0x3000) injected directly.
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x1000);
    let mut state = RegisterValueSet::new();
    state.set(Register::RAX, ValueSet::Constant(0x3000));
    let mut vsa = ValueSetAnalyzer::new();
    vsa.value_sets.insert(0x1000, state);
    let targets = vsa.indirect_jump_targets(&insts[0]);
    assert_eq!(targets, Some(vec![0x3000u64]));
}

#[test]
fn test_indirect_jump_multiple_targets() {
    // jmp rax where rax = Multiple({0x1000, 0x2000, 0x3000}).
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x1000);
    let mut state = RegisterValueSet::new();
    state.set(Register::RAX, make_multiple(&[0x1000, 0x2000, 0x3000]));
    let mut vsa = ValueSetAnalyzer::new();
    vsa.value_sets.insert(0x1000, state);
    let targets = vsa.indirect_jump_targets(&insts[0]).unwrap();
    assert_eq!(targets, vec![0x1000u64, 0x2000, 0x3000]);
}

#[test]
fn test_indirect_jump_top_returns_none() {
    // jmp rax where rax = Top -> not resolvable.
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x1000);
    let mut state = RegisterValueSet::new();
    state.set(Register::RAX, ValueSet::Top);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.value_sets.insert(0x1000, state);
    assert!(vsa.indirect_jump_targets(&insts[0]).is_none());
}

#[test]
fn test_indirect_jump_interval_returns_none() {
    // jmp rax where rax = Interval -> not resolvable.
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x1000);
    let mut state = RegisterValueSet::new();
    state.set(
        Register::RAX,
        ValueSet::Interval {
            min: 0x1000,
            max: 0x2000,
        },
    );
    let mut vsa = ValueSetAnalyzer::new();
    vsa.value_sets.insert(0x1000, state);
    assert!(vsa.indirect_jump_targets(&insts[0]).is_none());
}

#[test]
fn test_indirect_jump_non_jmp_instruction_returns_none() {
    // ret — not Jmp, so indirect_jump_targets must return None.
    let bytes = [0xC3u8];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    assert!(vsa.indirect_jump_targets(&insts[0]).is_none());
}

#[test]
fn test_indirect_jump_no_pre_state_returns_none() {
    // jmp rax but no entry in value_sets for that IP -> None.
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x1000);
    let vsa = ValueSetAnalyzer::new(); // empty, no analyze()
    assert!(vsa.indirect_jump_targets(&insts[0]).is_none());
}

// ── Analyzer: memory eval ─────────────────────────────────────────────────────

#[test]
fn test_memory_read_from_rodata_resolves_constant() {
    // mov rax, [0x2000]  where 0x2000 is a read-only segment containing 0x5678.
    // [48 8B 04 25 00 20 00 00] = mov rax, [0x2000]
    let bytes = [0x48u8, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let data = 0x5678u64.to_le_bytes().to_vec();
    let seg = MemorySegment::new(0x2000, data, false);
    let mut vsa = ValueSetAnalyzer::with_segments(vec![seg]);
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(0x5678));
}

#[test]
fn test_memory_read_from_writable_segment_is_top() {
    // Same address, but writable -> Top.
    let bytes = [0x48u8, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let data = 0x5678u64.to_le_bytes().to_vec();
    let seg = MemorySegment::new(0x2000, data, true);
    let mut vsa = ValueSetAnalyzer::with_segments(vec![seg]);
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_memory_read_with_unknown_base_is_top() {
    // mov rax, [rbx]  — rbx is Top on entry.
    // [48 8B 03] = mov rax, [rbx]
    let bytes = [0x48u8, 0x8B, 0x03];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_memory_read_with_constant_base_resolves_segment() {
    // mov rbx, 0x2000 ; mov rax, [rbx]  — base is known, segment contains value.
    let bytes = [
        0x48u8, 0xC7, 0xC3, 0x00, 0x20, 0x00, 0x00, // mov rbx, 0x2000  (ip 0x1000, 7 bytes)
        0x48, 0x8B, 0x03, // mov rax, [rbx]  (ip 0x1007, 3 bytes)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let data = 0xDEAD_BEEFu64.to_le_bytes().to_vec();
    let seg = MemorySegment::new(0x2000, data, false);
    let mut vsa = ValueSetAnalyzer::with_segments(vec![seg]);
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1007].clone();
    let post = vsa.transfer(&pre, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(0xDEAD_BEEF));
}

#[test]
fn test_relocation_address_suppresses_segment_read() {
    // mov rax, [0x2000]  where 0x2000 holds 0x5678 in a segment,
    // but 0x2000 is also in the relocation set -> Top.
    let bytes = [0x48u8, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let data = 0x5678u64.to_le_bytes().to_vec();
    let seg = MemorySegment::new(0x2000, data, false);
    let mut relocs = HashSet::new();
    relocs.insert(0x2000u64);
    let mut vsa = ValueSetAnalyzer::with_segments(vec![seg]).with_relocation_addresses(relocs);
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

#[test]
fn test_relocation_address_makes_indirect_jump_unresolvable() {
    // Without relocation: jmp target resolves to 0x5000.
    // With relocation: jmp target is Top -> None.
    let bytes = [
        0x48u8, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00, // mov rax, [0x2000]  (ip 0x1000)
        0xFF, 0xE0, // jmp rax               (ip 0x1008)
    ];
    let data = 0x5000u64.to_le_bytes().to_vec();
    let seg = MemorySegment::new(0x2000, data.clone(), false);

    // Without relocation: target resolved.
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa_no_reloc = ValueSetAnalyzer::with_segments(vec![seg]);
    vsa_no_reloc.analyze(&insts);
    let targets = vsa_no_reloc.indirect_jump_targets(&insts[1]);
    assert_eq!(
        targets,
        Some(vec![0x5000u64]),
        "should resolve without relocation"
    );

    // With relocation: target must be Top.
    let seg2 = MemorySegment::new(0x2000, data, false);
    let mut relocs = HashSet::new();
    relocs.insert(0x2000u64);
    let insts2 = decode_at(&bytes, 0x1000);
    let mut vsa_reloc =
        ValueSetAnalyzer::with_segments(vec![seg2]).with_relocation_addresses(relocs);
    vsa_reloc.analyze(&insts2);
    let targets2 = vsa_reloc.indirect_jump_targets(&insts2[1]);
    assert!(targets2.is_none(), "should not resolve with relocation");
}

#[test]
fn test_memory_read_address_outside_any_segment_is_top() {
    // mov rax, [0x9999]  — no segment covers this address.
    let bytes = [0x48u8, 0x8B, 0x04, 0x25, 0x99, 0x99, 0x00, 0x00];
    let insts = decode_at(&bytes, 0x1000);
    let seg = MemorySegment::new(0x2000, vec![0u8; 8], false);
    let mut vsa = ValueSetAnalyzer::with_segments(vec![seg]);
    vsa.analyze(&insts);
    let pre = vsa.value_sets[&0x1000].clone();
    let post = vsa.transfer(&pre, &insts[0]);
    assert_eq!(post.get(Register::RAX), ValueSet::Top);
}

// ── Analyzer: xor-based constant propagation through AND ─────────────────────

#[test]
fn test_xor_zero_then_add_propagates() {
    // xor rax, rax ; add rax, 7  ->  rax = 7
    let bytes = [
        0x48u8, 0x31, 0xC0, // xor rax, rax  (ip 0x1000, 3 bytes)
        0x48, 0x83, 0xC0, 0x07, // add rax, 7    (ip 0x1003, 4 bytes)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_add = vsa.value_sets[&0x1003].clone();
    assert_eq!(pre_add.get(Register::RAX), ValueSet::Constant(0));
    let post = vsa.transfer(&pre_add, &insts[1]);
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(7));
}

#[test]
fn test_xor_zero_then_and_with_zero_mask() {
    // xor rax, rax ; and rax, 0xFF  ->  rax = 0 (0 & anything = 0)
    let bytes = [
        0x48u8, 0x31, 0xC0, // xor rax, rax  (ip 0x1000, 3 bytes)
        0x48, 0x83, 0xE0, 0xFF, // and rax, 0xFF  (imm8 0xFF sign-extended = -1, ip 0x1003)
    ];
    let insts = decode_at(&bytes, 0x1000);
    let mut vsa = ValueSetAnalyzer::new();
    vsa.analyze(&insts);
    let pre_and = vsa.value_sets[&0x1003].clone();
    assert_eq!(pre_and.get(Register::RAX), ValueSet::Constant(0));
    let post = vsa.transfer(&pre_and, &insts[1]);
    // 0 & -1 = 0
    assert_eq!(post.get(Register::RAX), ValueSet::Constant(0));
}

// ── Analyzer: multiple paths converge ────────────────────────────────────────

#[test]
fn test_two_paths_with_different_constants_join_to_multiple() {
    // Two separate paths load different constants into rax, then both reach the same jmp rax.
    // We simulate this by injecting a Multiple state directly and checking resolution.
    //
    //   path A: rax = 0x1000
    //   path B: rax = 0x2000
    //   -> at jmp rax, rax = Multiple({0x1000, 0x2000})
    let bytes = [0xFFu8, 0xE0]; // jmp rax
    let insts = decode_at(&bytes, 0x5000);
    let mut state = RegisterValueSet::new();
    state.set(Register::RAX, make_multiple(&[0x1000, 0x2000]));
    let mut vsa = ValueSetAnalyzer::new();
    vsa.value_sets.insert(0x5000, state);
    let targets = vsa.indirect_jump_targets(&insts[0]).unwrap();
    assert_eq!(targets, vec![0x1000u64, 0x2000]);
}
