//! AES-NI vectors from FIPS-197's AES-128 example, plus end-to-end rounds.

use microlift::testkit::prelude::*;

/// FIPS-197 A.1 round keys for key `000102...0f`.
const ROUND_KEYS: [[u64; 2]; 11] = [
    [0x0706050403020100, 0x0f0e0d0c0b0a0908],
    [0xfa72afd2fd74aad6, 0xfe76abd6f178a6da],
    [0xf1bd3d640bcf92b6, 0xfeb3306800c59bbe],
    [0xbfc9c2d24e74ffb6, 0x41bf6904bf0c596c],
    [0x033e3595bcf7f747, 0xfd8d05fdbc326cf9],
    [0xeb9d9fa9e8a3aa3c, 0xaa22f6ad57aff350],
    [0x9692a6f77d0f395e, 0x6b1fa30ac13d55a7],
    [0x8ce25fe31a70f914, 0x26c0a94e4ddf0a44],
    [0xb9651ca435874347, 0xd27abfaef4ba16e0],
    [0x685785f0d1329954, 0x4e972cbe9ced9310],
    [0x174a94e37f1d1113, 0xc5302b4d8ba707f3],
];

const PLAINTEXT: [u64; 2] = [0x7766554433221100, 0xffeeddccbbaa9988];
const CIPHERTEXT: [u64; 2] = [0x30047b6ad8e0c469, 0x5ac5b47080b7cdd8];

const A: [u64; 2] = [0x6556747365545b7b, 0x5d47535d726f7463];
const B: [u64; 2] = [0x2979616853286948, 0x5d6e6f726575475b];

/// `66 [REX] 0F 38 <op> <modrm>`.
fn aes(op: u8, dst: u8, src: u8) -> Vec<u8> {
    let mut out = vec![0x66];
    if dst >= 8 || src >= 8 {
        out.push(0x40 | ((dst >= 8) as u8) << 2 | (src >= 8) as u8);
    }
    out.extend_from_slice(&[0x0F, 0x38, op, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
    out
}

fn aesenc(dst: u8, src: u8) -> Vec<u8> {
    aes(0xDC, dst, src)
}
fn aesenclast(dst: u8, src: u8) -> Vec<u8> {
    aes(0xDD, dst, src)
}
fn aesdec(dst: u8, src: u8) -> Vec<u8> {
    aes(0xDE, dst, src)
}
fn aesdeclast(dst: u8, src: u8) -> Vec<u8> {
    aes(0xDF, dst, src)
}
fn aesimc(dst: u8, src: u8) -> Vec<u8> {
    aes(0xDB, dst, src)
}

fn movdqa(dst: u8, src: u8) -> Vec<u8> {
    let mut out = vec![0x66];
    if dst >= 8 || src >= 8 {
        out.push(0x40 | ((dst >= 8) as u8) << 2 | (src >= 8) as u8);
    }
    out.extend_from_slice(&[0x0F, 0x6F, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
    out
}

fn pxor(dst: u8, src: u8) -> Vec<u8> {
    let mut out = vec![0x66];
    if dst >= 8 || src >= 8 {
        out.push(0x40 | ((dst >= 8) as u8) << 2 | (src >= 8) as u8);
    }
    out.extend_from_slice(&[0x0F, 0xEF, 0xC0 | ((dst & 7) << 3) | (src & 7)]);
    out
}

fn aeskeygenassist(dst: u8, src: u8, rcon: u8) -> Vec<u8> {
    let mut out = vec![0x66];
    if dst >= 8 || src >= 8 {
        out.push(0x40 | ((dst >= 8) as u8) << 2 | (src >= 8) as u8);
    }
    out.extend_from_slice(&[0x0F, 0x3A, 0xDF, 0xC0 | ((dst & 7) << 3) | (src & 7), rcon]);
    out
}

/// Runs under every config that must not change the answer: XMM backing, inline vs outlined kernels.
fn run(code: &[u8], seed: impl Fn(&mut RegisterContext) + Copy) -> RunOutcome {
    let mut first: Option<[[u64; 2]; 16]> = None;
    let mut last = None;
    for (vector, outline) in [(false, true), (true, true), (false, false), (true, false)] {
        let out = LiftTest::new(code)
            .vector_xmm(vector)
            .with_config(|c| c.outline_helpers = outline)
            .reg(seed)
            .run();
        match &first {
            None => first = Some(*out.ctx.xmm),
            Some(f) => assert_eq!(
                *f, *out.ctx.xmm,
                "vector_xmm = {vector}, outline_helpers = {outline} disagreed"
            ),
        }
        last = Some(out);
    }
    last.unwrap()
}

fn one(code: Vec<u8>) -> [u64; 2] {
    let mut program = code;
    program.push(0xC3);
    let out = run(&program, |c| {
        c.xmm[0] = A;
        c.xmm[1] = B;
    });
    out.ctx.xmm[0]
}

#[test]
fn the_four_round_instructions_match_their_reference_values() {
    assert_eq!(one(aesenc(0, 1)), [0x0be8672d9204d843, 0x81a2d95024f450f7]);
    assert_eq!(
        one(aesenclast(0, 1)),
        [0x64d9f3e71f80fb69, 0x1ddf563e2855aaa0]
    );
    assert_eq!(one(aesdec(0, 1)), [0x3b278114f7b5e819, 0x2469e4c5f3acf910]);
    assert_eq!(
        one(aesdeclast(0, 1)),
        [0x376f36e7ef2e394b, 0xe1d7a5ffe8888d5b]
    );
}

#[test]
fn aesimc_matches_its_reference_value() {
    assert_eq!(one(aesimc(0, 1)), [0x373acd9931f9bf2d, 0x8e3de07dfd81c7b7]);
}

/// The round constant must land in the right two dwords.
#[test]
fn aeskeygenassist_applies_rcon_to_the_rotated_words() {
    let go = |rcon: u8| {
        let mut program = aeskeygenassist(0, 1, rcon);
        program.push(0xC3);
        run(&program, |c| {
            c.xmm[0] = A;
            c.xmm[1] = B;
        })
        .ctx
        .xmm[0]
    };
    assert_eq!(go(0x00), [0x45a5b6efa5b6ef45, 0x404c9fa84c9fa840]);
    assert_eq!(go(0x01), [0x45a5b6eea5b6ef45, 0x404c9fa94c9fa840]);
    assert_eq!(go(0x8b), [0x45a5b664a5b6ef45, 0x404c9f234c9fa840]);
}

#[test]
fn a_round_key_can_come_from_memory() {
    let program = [0x66, 0x0F, 0x38, 0xDC, 0x07, 0xC3]; // aesenc xmm0, [rdi] ; ret
    let key = B;

    let out = run(&program, |c| {
        c.xmm[0] = A;
        c.rdi = key.as_ptr() as u64;
    });
    assert_eq!(out.ctx.xmm[0], [0x0be8672d9204d843, 0x81a2d95024f450f7]);
}

// Source must be read before the destination is redefined.
#[test]
fn the_state_can_also_be_the_round_key() {
    assert_eq!(one(aesenc(0, 0)), [0x47c77236a478ea70, 0x818be57f33ee63cf]);
}

#[test]
fn fips_197_aes128_encrypts() {
    let mut code = pxor(0, 11);
    for k in 1..10 {
        code.extend(aesenc(0, k));
    }
    code.extend(aesenclast(0, 10));
    code.push(0xC3);

    let out = run(&code, |c| {
        c.xmm[0] = PLAINTEXT;
        for (i, k) in ROUND_KEYS.iter().enumerate().skip(1) {
            c.xmm[i] = *k;
        }
        c.xmm[11] = ROUND_KEYS[0];
    });
    assert_eq!(out.ctx.xmm[0], CIPHERTEXT);
}

#[test]
fn fips_197_aes128_decrypts() {
    let mut code = pxor(0, 10);
    for k in (1..10).rev() {
        code.extend(movdqa(15, k));
        code.extend(aesimc(15, 15));
        code.extend(aesdec(0, 15));
    }
    code.extend(aesdeclast(0, 11));
    code.push(0xC3);

    let out = run(&code, |c| {
        c.xmm[0] = CIPHERTEXT;
        for (i, k) in ROUND_KEYS.iter().enumerate().skip(1) {
            c.xmm[i] = *k;
        }
        c.xmm[11] = ROUND_KEYS[0];
    });
    assert_eq!(out.ctx.xmm[0], PLAINTEXT);
}

/// Dword 3 is the word the key schedule XORs in; checks SubWord, RotWord and RCON together.
#[test]
fn aeskeygenassist_drives_the_key_schedule() {
    const RCONS: [u8; 10] = [1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

    for (round, rcon) in RCONS.iter().enumerate() {
        let prev = ROUND_KEYS[round];
        let mut code = aeskeygenassist(1, 0, *rcon);
        code.push(0xC3);
        let assist = run(&code, |c| c.xmm[0] = prev).ctx.xmm[1];

        let mut words = [
            prev[0] as u32,
            (prev[0] >> 32) as u32,
            prev[1] as u32,
            (prev[1] >> 32) as u32,
        ];
        let t = (assist[1] >> 32) as u32;
        words[0] ^= t;
        for i in 1..4 {
            words[i] ^= words[i - 1];
        }
        let next = [
            words[0] as u64 | (words[1] as u64) << 32,
            words[2] as u64 | (words[3] as u64) << 32,
        ];
        assert_eq!(next, ROUND_KEYS[round + 1], "round key {}", round + 1);
    }
}

#[test]
fn random_operands_agree_across_configurations() {
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let programs: Vec<Vec<u8>> = vec![
        aesenc(0, 1),
        aesenclast(0, 1),
        aesdec(0, 1),
        aesdeclast(0, 1),
        aesimc(0, 1),
        aeskeygenassist(0, 1, 0x36),
    ];
    for _ in 0..8 {
        let a = [next(), next()];
        let b = [next(), next()];
        for p in &programs {
            let mut code = p.clone();
            code.push(0xC3);
            run(&code, |c| {
                c.xmm[0] = a;
                c.xmm[1] = b;
            });
        }
    }
}

/// Each instruction gets its own helper; a symbol collision would otherwise go unnoticed.
#[test]
fn each_instruction_outlines_exactly_one_shared_kernel() {
    use microlift::arch::x86_64::X86_64;
    use microlift::arch::x86_64::semantics::aes::{AES_KERNELS, AesKernel};
    use microlift::core::{LifterConfig, lift_to_code};
    use microlift::testkit::DEFAULT_LIFT_BASE;

    let mut bytes = Vec::new();
    for k in 1..10 {
        bytes.extend(aesenc(0, k));
    }
    bytes.extend(aesenclast(0, 1));
    bytes.extend(aesdec(0, 1));
    bytes.extend(aesdeclast(0, 1));
    bytes.extend(aesimc(0, 1));
    bytes.extend(aeskeygenassist(0, 1, 1));
    bytes.push(0xC3);

    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    let names: Vec<&str> = code.helpers.iter().map(|h| h.name).collect();
    for kernel in AES_KERNELS {
        assert_eq!(
            names.iter().filter(|n| **n == kernel.symbol()).count(),
            1,
            "{kernel:?} should outline exactly one body, got {names:?}"
        );
    }
    assert_eq!(
        names
            .iter()
            .filter(|n| **n == AesKernel::Enc.symbol())
            .count(),
        1,
        "nine AESENCs share one body, got {names:?}"
    );
}

/// Tables are emitted once however many AES instructions reference them.
#[test]
fn the_aes_tables_are_emitted_once_as_data() {
    use microlift::arch::x86_64::X86_64;
    use microlift::arch::x86_64::semantics::aes::{AES_TABLE_LEN, AES_TABLE_SYMBOL};
    use microlift::core::{LifterConfig, lift_to_code};
    use microlift::testkit::DEFAULT_LIFT_BASE;

    let mut bytes = Vec::new();
    bytes.extend(aesenc(0, 1));
    bytes.extend(aesdec(0, 1));
    bytes.extend(aesimc(0, 1));
    bytes.extend(aeskeygenassist(0, 1, 1));
    bytes.push(0xC3);

    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    assert_eq!(code.data_objects.len(), 1);
    let tables = &code.data_objects[0];
    assert_eq!(tables.name, AES_TABLE_SYMBOL);
    assert_eq!(tables.bytes.len(), AES_TABLE_LEN);

    let (sbox, rest) = tables.bytes.split_at(256);
    let (inv_sbox, xtime) = rest.split_at(256);

    assert_eq!(sbox[0x00], 0x63);
    assert_eq!(sbox[0x01], 0x7c);
    assert_eq!(sbox[0x53], 0xed);
    assert_eq!(sbox[0xff], 0x16);

    for i in 0..256 {
        assert_eq!(inv_sbox[sbox[i] as usize] as usize, i);
        // xtime must not be derived from the shift-and-reduce it implements.
        let expected = ((i as u8) << 1) ^ if i & 0x80 != 0 { 0x1B } else { 0 };
        assert_eq!(xtime[i], expected, "xtime[{i:#04x}]");
    }
}

/// AES and CRC32 tables share a DataId space; each site must reach its own.
#[test]
fn aes_and_crc32_tables_coexist_in_one_region() {
    use microlift::arch::x86_64::X86_64;
    use microlift::core::{LifterConfig, lift_to_code};
    use microlift::testkit::DEFAULT_LIFT_BASE;

    let mut bytes = aesenc(0, 1);
    bytes.extend_from_slice(&[0xF2, 0x0F, 0x38, 0xF0, 0xC1]);
    bytes.push(0xC3);

    let code =
        lift_to_code::<X86_64>(&bytes, DEFAULT_LIFT_BASE, LifterConfig::default()).expect("lift");
    assert_eq!(code.data_objects.len(), 2);

    let out = run(&bytes, |c| {
        c.xmm[0] = A;
        c.xmm[1] = B;
        c.rax = 0;
        c.rcx = 0x61;
    });
    assert_eq!(out.ctx.xmm[0], [0x0be8672d9204d843, 0x81a2d95024f450f7]);
    assert_eq!(out.ctx.rax, 0x93AD_1061);
}
