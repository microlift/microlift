use microlift::testkit::prelude::*;

/// Pack up to 16 bytes into an XMM `[lo, hi]` pair, zero-filling the rest.
fn xmm(bytes: &[u8]) -> [u64; 2] {
    assert!(bytes.len() <= 16);
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    [
        u64::from_le_bytes(buf[..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..].try_into().unwrap()),
    ]
}

// CRC32

/// `crc32 eax, cl` over the digits "123456789", the standard CRC-32C check
/// value. The instruction is the raw remainder step, so the caller supplies the
/// `0xFFFF_FFFF` pre-conditioning and the final inversion.
#[test]
fn crc32_byte_matches_the_published_check_value() {
    let mut code = vec![0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]; // mov rax, -1 (eax = 0xFFFFFFFF)
    for b in b"123456789" {
        // mov cl, imm8 ; crc32 eax, cl
        code.extend_from_slice(&[0xB1, *b]);
        code.extend_from_slice(&[0xF2, 0x0F, 0x38, 0xF0, 0xC1]);
    }
    code.push(0xC3);

    let out = LiftTest::new(&code).run();
    assert_eq!(out.ctx.rax as u32 ^ 0xFFFF_FFFF, 0xE306_9283);
}

/// The three narrower forms against values derived from the same reference.
#[test]
fn crc32_widths() {
    // crc32 eax, cl ; ret   with eax = 0, cl = 'a'
    let out = LiftTest::new(&[0xF2, 0x0F, 0x38, 0xF0, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0;
            c.rcx = 0x61;
        })
        .run();
    assert_eq!(out.ctx.rax, 0x93AD_1061);

    // crc32 eax, cx ; ret
    let out = LiftTest::new(&[0x66, 0xF2, 0x0F, 0x38, 0xF1, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0;
            c.rcx = 0x1234;
        })
        .run();
    assert_eq!(out.ctx.rax, 0xFFA1_C4C7);

    // crc32 eax, ecx ; ret
    let out = LiftTest::new(&[0xF2, 0x0F, 0x38, 0xF1, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0xDEAD_BEEF;
            c.rcx = 0x1234_5678;
        })
        .run();
    assert_eq!(out.ctx.rax, 0xF3ED_4B20);

    // crc32 rax, rcx ; ret
    let out = LiftTest::new(&[0xF2, 0x48, 0x0F, 0x38, 0xF1, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0;
            c.rcx = 0x0123_4567_89AB_CDEF;
        })
        .run();
    assert_eq!(out.ctx.rax, 0xE998_6AA9);
}

/// A quadword fold must equal eight byte folds of the same data - the chunking
/// identity the 64-bit expansion is built on. If the two 32-bit halves were
/// folded in the wrong order, or the high half XORed in at the wrong point,
/// this is what catches it.
#[test]
fn crc32_qword_equals_eight_byte_folds() {
    const DATA: u64 = 0x0123_4567_89AB_CDEF;

    // crc32 rax, rcx ; ret
    let wide = LiftTest::new(&[0xF2, 0x48, 0x0F, 0x38, 0xF1, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0;
            c.rcx = DATA;
        })
        .run();

    let mut code = Vec::new();
    for i in 0..8 {
        // mov cl, imm8 ; crc32 rax, cl
        code.extend_from_slice(&[0xB1, (DATA >> (8 * i)) as u8]);
        code.extend_from_slice(&[0xF2, 0x48, 0x0F, 0x38, 0xF0, 0xC1]);
    }
    code.push(0xC3);
    let bytewise = LiftTest::new(&code).reg(|c| c.rax = 0).run();

    assert_eq!(wide.ctx.rax, bytewise.ctx.rax);
    assert_eq!(wide.ctx.rax, 0xE998_6AA9);
}

/// The `r64` forms produce a 32-bit value: the upper half of the destination is
/// zeroed, not preserved.
#[test]
fn crc32_r64_zeroes_the_upper_half() {
    // crc32 rax, cl ; ret
    let out = LiftTest::new(&[0xF2, 0x48, 0x0F, 0x38, 0xF0, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0xFFFF_FFFF_0000_0000;
            c.rcx = 0;
        })
        .run();
    // The remainder starts from the destination's *low* 32 bits, which are zero.
    assert_eq!(out.ctx.rax, 0);
}

/// A memory source, and the inlined (non-outlined) expansion, must agree with
/// the register source and the outlined helper.
#[test]
fn crc32_memory_source_and_inline_expansion_agree() {
    // crc32 eax, byte [rsi] ; ret
    let code = [0xF2, 0x0F, 0x38, 0xF0, 0x06, 0xC3];
    let byte = 0x61u8;
    let addr = &byte as *const u8 as u64;

    for outline in [true, false] {
        let out = LiftTest::new(&code)
            .with_config(|c| c.outline_helpers = outline)
            .reg(|c| {
                c.rax = 0;
                c.rsi = addr;
            })
            .run();
        assert_eq!(out.ctx.rax, 0x93AD_1061, "outline_helpers = {outline}");
    }
}

/// `CRC32` touches no flag - including under the dead-flag pass, where it is
/// classified as transparent and so must not be allowed to shadow an earlier
/// write.
#[test]
fn crc32_preserves_flags() {
    // cmp rax, rax (sets ZF) ; crc32 eax, cl ; ret
    let code = [0x48, 0x39, 0xC0, 0xF2, 0x0F, 0x38, 0xF0, 0xC1, 0xC3];
    for elim in [false, true] {
        let mut t = LiftTest::new(&code).reg(|c| c.rcx = 0x61);
        if elim {
            t = t.eliminate_dead_flags();
        }
        let out = t.run();
        assert_eq!(out.ctx.zf, 1, "eliminate_dead_flags = {elim}");
        assert_eq!(out.ctx.cf, 0);
    }
}

// POPCNT

#[test]
fn popcnt_counts_and_sets_flags() {
    // popcnt rax, rcx ; ret
    let code = [0xF3, 0x48, 0x0F, 0xB8, 0xC1];
    for (src, expect) in [
        (0u64, 0u64),
        (1, 1),
        (0xFFFF_FFFF_FFFF_FFFF, 64),
        (0x8000_0000_0000_0001, 2),
        (0x0F0F_0F0F_0F0F_0F0F, 32),
    ] {
        let mut program = code.to_vec();
        program.push(0xC3);
        let out = LiftTest::new(&program)
            .reg(|c| {
                c.rcx = src;
                // Seed every flag set so the clears are observable.
                c.cf = 1;
                c.pf = 1;
                c.af = 1;
                c.sf = 1;
                c.of = 1;
            })
            .run();
        assert_eq!(out.ctx.rax, expect, "popcnt of {src:#x}");
        assert_eq!(out.ctx.zf, u8::from(src == 0));
        // POPCNT *clears* the other five rather than leaving them undefined.
        assert_eq!(
            (out.ctx.cf, out.ctx.pf, out.ctx.af, out.ctx.sf, out.ctx.of),
            (0, 0, 0, 0, 0)
        );
    }
}

#[test]
fn popcnt_32_bit_zeroes_the_upper_half() {
    // popcnt eax, ecx ; ret
    let out = LiftTest::new(&[0xF3, 0x0F, 0xB8, 0xC1, 0xC3])
        .reg(|c| {
            c.rax = 0xFFFF_FFFF_FFFF_FFFF;
            c.rcx = 0xFFFF_FFFF_FFFF_FFFF;
        })
        .run();
    assert_eq!(out.ctx.rax, 32);
}

// PCMPISTRI / PCMPISTRM / PCMPESTRI / PCMPESTRM

/// `pcmpistri xmm0, xmm1, imm8 ; ret`
fn istri(imm: u8) -> Vec<u8> {
    vec![0x66, 0x0F, 0x3A, 0x63, 0xC1, imm, 0xC3]
}

/// `pcmpistrm xmm0, xmm1, imm8 ; ret`
fn istrm(imm: u8) -> Vec<u8> {
    vec![0x66, 0x0F, 0x3A, 0x62, 0xC1, imm, 0xC3]
}

/// `pcmpestri xmm0, xmm1, imm8 ; ret` (`wide` adds `REX.W`, taking the lengths
/// from RAX/RDX rather than EAX/EDX).
fn estri(imm: u8, wide: bool) -> Vec<u8> {
    let mut v = vec![0x66];
    if wide {
        v.push(0x48);
    }
    v.extend_from_slice(&[0x0F, 0x3A, 0x61, 0xC1, imm, 0xC3]);
    v
}

/// `pcmpestrm xmm0, xmm1, imm8 ; ret`
fn estrm(imm: u8) -> Vec<u8> {
    vec![0x66, 0x0F, 0x3A, 0x60, 0xC1, imm, 0xC3]
}

/// Run a string compare over `(xmm0, xmm1)`, under every configuration that is
/// supposed to make no difference to the answer.
fn strcmp_run(
    code: &[u8],
    a: &[u8],
    b: &[u8],
    seed: impl Fn(&mut RegisterContext) + Copy,
) -> RunOutcome {
    strcmp_run_inner(code, a, b, seed, true)
}

/// [`strcmp_run`] with the automatic Unicorn cross-check suppressed, for the
/// one case where the oracle is the engine that is wrong. See
/// [`pcmpestri_length_absolute_value_and_saturation`].
fn strcmp_run_no_unicorn(
    code: &[u8],
    a: &[u8],
    b: &[u8],
    seed: impl Fn(&mut RegisterContext) + Copy,
) -> RunOutcome {
    strcmp_run_inner(code, a, b, seed, false)
}

fn strcmp_run_inner(
    code: &[u8],
    a: &[u8],
    b: &[u8],
    seed: impl Fn(&mut RegisterContext) + Copy,
    unicorn: bool,
) -> RunOutcome {
    let mut first: Option<(u64, [u64; 2], (u8, u8, u8, u8, u8, u8))> = None;
    let mut last = None;
    for (vector, outline) in [(false, true), (true, true), (false, false), (true, false)] {
        let mut t = LiftTest::new(code)
            .vector_xmm(vector)
            .with_config(|c| c.outline_helpers = outline);
        if !unicorn {
            t = t.no_unicorn();
        }
        let out = t
            .reg(|c| {
                c.xmm[0] = xmm(a);
                c.xmm[1] = xmm(b);
                seed(c);
            })
            .run();
        let snapshot = (
            out.ctx.rcx,
            out.ctx.xmm[0],
            (
                out.ctx.cf, out.ctx.zf, out.ctx.sf, out.ctx.of, out.ctx.af, out.ctx.pf,
            ),
        );
        match &first {
            None => first = Some(snapshot),
            Some(f) => assert_eq!(
                *f, snapshot,
                "vector_xmm = {vector}, outline_helpers = {outline} disagreed"
            ),
        }
        last = Some(out);
    }
    last.unwrap()
}

/// The `strlen` idiom: equal-each against an all-zero operand. Every position
/// of the (empty) first operand is invalid, so a position matches only where
/// the *second* operand has also run out - making the first set bit the index
/// of the terminator.
#[test]
fn pcmpistri_is_strlen() {
    for (s, len) in [
        (&b""[..], 0usize),
        (&b"a"[..], 1),
        (&b"abc"[..], 3),
        (&b"0123456789abcde"[..], 15),
    ] {
        let out = strcmp_run(&istri(0x08), &[], s, |_| {});
        assert_eq!(out.ctx.rcx, len as u64, "strlen({s:?})");
        assert_eq!(out.ctx.cf, 1);
        assert_eq!(out.ctx.zf, 1, "the string operand held a null");
        assert_eq!(out.ctx.sf, 1, "the empty operand is shorter than 16");
    }

    // A full 16 bytes with no terminator: no match, index 16, and neither
    // exhaustion flag.
    let out = strcmp_run(&istri(0x08), &[], b"0123456789abcdef", |_| {});
    assert_eq!(out.ctx.rcx, 16);
    assert_eq!(out.ctx.cf, 0);
    assert_eq!(out.ctx.zf, 0);
    assert_eq!(out.ctx.sf, 1);
}

/// The `strchr` idiom: equal-any against a one-character set.
#[test]
fn pcmpistri_is_strchr() {
    let out = strcmp_run(&istri(0x00), b"a", b"zza", |_| {});
    assert_eq!(out.ctx.rcx, 2);
    assert_eq!(out.ctx.cf, 1);
    assert_eq!(out.ctx.of, 0, "bit 0 of the result is clear");

    // Not present anywhere.
    let out = strcmp_run(&istri(0x00), b"a", b"zzz", |_| {});
    assert_eq!(out.ctx.rcx, 16);
    assert_eq!(out.ctx.cf, 0);

    // Present at position 0, which is the one case that also sets OF.
    let out = strcmp_run(&istri(0x00), b"a", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 0);
    assert_eq!(out.ctx.of, 1);

    // A multi-character set: the *first* member found wins, not the first
    // member listed.
    let out = strcmp_run(&istri(0x00), b"xyz", b"aazaa", |_| {});
    assert_eq!(out.ctx.rcx, 2);
}

/// The `strcmp` idiom: equal-each with the result inverted, so the first set
/// bit is the first *difference*. Two equal strings must produce no set bit at
/// all - which is only true because two exhausted positions force a match.
#[test]
fn pcmpistri_is_strcmp() {
    let out = strcmp_run(&istri(0x18), b"abc", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 16, "equal strings have no differing position");
    assert_eq!(out.ctx.cf, 0);

    let out = strcmp_run(&istri(0x18), b"abd", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 2);
    assert_eq!(out.ctx.cf, 1);

    // Differing lengths: the terminator itself is the difference.
    let out = strcmp_run(&istri(0x18), b"ab", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 2);
}

/// The `strstr` idiom: equal-ordered reports where the first operand occurs as
/// a substring of the second. A needle shorter than the register only matches
/// because running off its end forces a match.
#[test]
fn pcmpistri_is_strstr() {
    let out = strcmp_run(&istri(0x0C), b"bc", b"abcd", |_| {});
    assert_eq!(out.ctx.rcx, 1);
    assert_eq!(out.ctx.cf, 1);

    let out = strcmp_run(&istri(0x0C), b"bc", b"abdc", |_| {});
    assert_eq!(out.ctx.rcx, 16);
    assert_eq!(out.ctx.cf, 0);

    // A match that would run past the end of the haystack is not a match: "cd"
    // starts at index 3 of "abc", but index 4 is past the terminator.
    let out = strcmp_run(&istri(0x0C), b"cd", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 16);
}

/// The character-class idiom: ranges, with pairs of the first operand read as
/// inclusive bounds.
#[test]
fn pcmpistri_scans_ranges() {
    // ['a'..'z']
    let out = strcmp_run(&istri(0x04), b"az", b"A9m", |_| {});
    assert_eq!(out.ctx.rcx, 2);

    // Two ranges: ['a'..'z'] and ['0'..'9'].
    let out = strcmp_run(&istri(0x04), b"az09", b"!!7", |_| {});
    assert_eq!(out.ctx.rcx, 2);

    // Nothing in class.
    let out = strcmp_run(&istri(0x04), b"az", b"!!!", |_| {});
    assert_eq!(out.ctx.rcx, 16);
}

/// `imm8[6]` on an index form selects the *most* significant set bit.
#[test]
fn pcmpistri_most_significant_index() {
    // equal any, 'a' anywhere in "aXa".
    let out = strcmp_run(&istri(0x40), b"a", b"aXa", |_| {});
    assert_eq!(out.ctx.rcx, 2);
    let out = strcmp_run(&istri(0x00), b"a", b"aXa", |_| {});
    assert_eq!(out.ctx.rcx, 0);

    // No match still reports the element count, not 31.
    let out = strcmp_run(&istri(0x40), b"a", b"XXX", |_| {});
    assert_eq!(out.ctx.rcx, 16);
}

/// Word-granularity elements: 8 per register, and the "no match" index is 8.
#[test]
fn pcmpistri_word_elements() {
    // equal any over words: find 0x2211 in [0x0001, 0x2211, ...].
    let needle = xmm(&[0x11, 0x22]);
    let hay = xmm(&[0x01, 0x00, 0x11, 0x22]);
    let out = LiftTest::new(&istri(0x01))
        .reg(|c| {
            c.xmm[0] = needle;
            c.xmm[1] = hay;
        })
        .run();
    assert_eq!(out.ctx.rcx, 1);

    // Nothing matches: the element count is 8 here, not 16.
    let out = LiftTest::new(&istri(0x01))
        .reg(|c| {
            c.xmm[0] = xmm(&[0xFF, 0xFE]);
            c.xmm[1] = hay;
        })
        .run();
    assert_eq!(out.ctx.rcx, 8);
}

/// Signed elements (`imm8[1]`) change the range comparison: `0x80` is below
/// `'a'` as a signed byte and above `'z'` as an unsigned one.
#[test]
fn pcmpistri_signed_ranges() {
    // Unsigned ['a'..'z'] does not contain 0x80.
    let out = strcmp_run(&istri(0x04), b"az", &[0x80, 0x00], |_| {});
    assert_eq!(out.ctx.rcx, 16);

    // Signed [0x80..0x7F] - i.e. every byte - does.
    let out = LiftTest::new(&istri(0x06))
        .reg(|c| {
            c.xmm[0] = xmm(&[0x80, 0x7F]);
            c.xmm[1] = xmm(&[0x41, 0x00]);
        })
        .run();
    assert_eq!(out.ctx.rcx, 0);
}

/// `PCMPISTRM` with `imm8[6] == 0`: `XMM0` receives the raw bitmask, and
/// everything above it is zeroed.
///
/// The destination is `XMM0` whatever the operands are, so this uses
/// `pcmpistrm xmm2, xmm1` - with `XMM0` pre-filled with ones - to show the
/// whole register is written rather than merged into.
#[test]
fn pcmpistrm_bit_mask() {
    // pcmpistrm xmm2, xmm1, 0x00 ; ret
    let out = LiftTest::new(&[0x66, 0x0F, 0x3A, 0x62, 0xD1, 0x00, 0xC3])
        .reg(|c| {
            c.xmm[0] = [u64::MAX, u64::MAX];
            c.xmm[2] = xmm(b"a");
            c.xmm[1] = xmm(b"zza");
        })
        .run();
    assert_eq!(out.ctx.xmm[0], [0b100, 0]);
}

/// `PCMPISTRM` with `imm8[6] == 1`: each set bit becomes an all-ones element.
#[test]
fn pcmpistrm_expanded_mask() {
    let out = strcmp_run(&istrm(0x40), b"a", b"zza", |_| {});
    assert_eq!(out.ctx.xmm[0], [0x0000_0000_00FF_0000, 0]);

    // Word elements expand to 16 bits each.
    let out = LiftTest::new(&istrm(0x41))
        .reg(|c| {
            c.xmm[0] = xmm(&[0x11, 0x22]);
            c.xmm[1] = xmm(&[0x01, 0x00, 0x11, 0x22]);
        })
        .run();
    assert_eq!(out.ctx.xmm[0], [0x0000_0000_FFFF_0000, 0]);
}

/// The explicit-length forms take their lengths from `EAX`/`EDX` instead of
/// scanning, so a null byte is just another byte.
#[test]
fn pcmpestri_uses_explicit_lengths() {
    // equal any: 'a' in "z\0a", with an explicit length of 3 that reaches past
    // the embedded null the implicit form would stop at.
    let out = strcmp_run(&estri(0x00, false), b"a", &[b'z', 0, b'a'], |c| {
        c.rax = 1;
        c.rdx = 3;
    });
    assert_eq!(out.ctx.rcx, 2);

    // The same data with the implicit form stops at the null and finds nothing.
    let out = strcmp_run(&istri(0x00), b"a", &[b'z', 0, b'a'], |_| {});
    assert_eq!(out.ctx.rcx, 16);

    // A length of 2 excludes the match.
    let out = strcmp_run(&estri(0x00, false), b"a", &[b'z', 0, b'a'], |c| {
        c.rax = 1;
        c.rdx = 2;
    });
    assert_eq!(out.ctx.rcx, 16);
}

/// Explicit lengths are absolute values saturated to the element count, and
/// only the low 32 bits count without `REX.W`.
#[test]
fn pcmpestri_length_absolute_value_and_saturation() {
    let data = &[b'z', b'z', b'a'][..];

    // -3 behaves as 3.
    let out = strcmp_run(&estri(0x00, false), b"a", data, |c| {
        c.rax = 1;
        c.rdx = (-3i64) as u64;
    });
    assert_eq!(out.ctx.rcx, 2);

    // A length past 16 saturates rather than wrapping, and clears ZF/SF.
    let out = strcmp_run(&estri(0x00, false), b"a", data, |c| {
        c.rax = 99;
        c.rdx = 99;
    });
    assert_eq!(out.ctx.rcx, 2);
    assert_eq!(out.ctx.zf, 0);
    assert_eq!(out.ctx.sf, 0);

    // Without REX.W the upper half of RDX is ignored.
    let out = strcmp_run(&estri(0x00, false), b"a", data, |c| {
        c.rax = 1;
        c.rdx = 0xFFFF_FFFF_0000_0003;
    });
    assert_eq!(out.ctx.rcx, 2);

    // With REX.W it is not - and this value's absolute value saturates to 16.
    //
    // No Unicorn differential on this one case: QEMU's `pcmp_elen` computes the
    // 64-bit absolute value and then stores it in an `int`, so `RDX` here comes
    // out as -3 rather than 4294967293 and it reports `ZF = |RDX| < 16 = 1`.
    // The SDM takes the whole of RDX under REX.W, which saturates - so the
    // divergence is Unicorn's, and this is the encoding that proves the
    // 64-bit half of `explicit_len` is not silently truncating the same way.
    let out = strcmp_run_no_unicorn(&estri(0x00, true), b"a", data, |c| {
        c.rax = 1;
        c.rdx = 0xFFFF_FFFF_0000_0003;
    });
    assert_eq!(out.ctx.rcx, 2);
    assert_eq!(out.ctx.zf, 0, "the saturated length is a full register");

    // A zero length makes every position of that operand invalid.
    let out = strcmp_run(&estri(0x00, false), b"a", data, |c| {
        c.rax = 1;
        c.rdx = 0;
    });
    assert_eq!(out.ctx.rcx, 16);
    assert_eq!(out.ctx.cf, 0);
}

/// `PCMPESTRM` writes `XMM0` under explicit lengths like its implicit twin.
#[test]
fn pcmpestrm_writes_xmm0() {
    let out = strcmp_run(&estrm(0x40), b"a", &[b'z', 0, b'a'], |c| {
        c.rax = 1;
        c.rdx = 3;
    });
    assert_eq!(out.ctx.xmm[0], [0x0000_0000_00FF_0000, 0]);
}

/// A memory second operand is a plain 128-bit load, including the bytes past
/// the string's terminator.
#[test]
fn pcmpistri_memory_operand() {
    // pcmpistri xmm0, [rsi], 0x00 ; ret
    let code = [0x66, 0x0F, 0x3A, 0x63, 0x06, 0x00, 0xC3];
    let hay: [u8; 16] = *b"zza\0\0\0\0\0\0\0\0\0\0\0\0\0";
    let out = LiftTest::new(&code)
        .reg(|c| {
            c.xmm[0] = xmm(b"a");
            c.rsi = hay.as_ptr() as u64;
        })
        .run();
    assert_eq!(out.ctx.rcx, 2);
}

/// The two negating polarities differ in what they do to the positions *past*
/// the second operand's end, and the difference is exactly why `strcmp` uses
/// one of them and not the other.
///
/// Plain negation (`imm8[5:4] == 01`) inverts all 16 positions, so the
/// forced-true positions past both strings' ends become false and drop out.
/// Masked negation (`11`) inverts only the positions valid in the second
/// operand, so those forced trues survive and the first set bit is the
/// terminator rather than a difference.
#[test]
fn pcmpistri_negation_polarities_differ_past_the_end() {
    // Equal strings, equal-each. Plain negation reports no difference at all;
    // masked negation reports the first position past the second operand.
    let out = strcmp_run(&istri(0x18), b"abc", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 16);
    let out = strcmp_run(&istri(0x38), b"abc", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 3);

    // A longer first operand: position 3 is (valid, invalid), which forces
    // false and so is *not* a match under either polarity - plain negation
    // turns it into the reported difference, masked negation leaves the
    // terminator at 3 clear and reports 4.
    let out = strcmp_run(&istri(0x18), b"abcd", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 3);
    let out = strcmp_run(&istri(0x38), b"abcd", b"abc", |_| {});
    assert_eq!(out.ctx.rcx, 4);
}

/// Every string compare defines the same six flags, and two of them are always
/// cleared.
#[test]
fn pcmpistri_clears_af_and_pf() {
    let out = strcmp_run(&istri(0x00), b"a", b"zza", |c| {
        c.af = 1;
        c.pf = 1;
    });
    assert_eq!((out.ctx.af, out.ctx.pf), (0, 0));
}

/// The dead-flag pass must not elide any of them: the string compares are
/// hand-classified as writing all six.
#[test]
fn pcmpistri_flags_survive_dead_flag_elimination() {
    let code = istri(0x00);
    let plain = LiftTest::new(&code)
        .reg(|c| {
            c.xmm[0] = xmm(b"a");
            c.xmm[1] = xmm(b"zza");
        })
        .run();
    let elided = LiftTest::new(&code)
        .eliminate_dead_flags()
        .reg(|c| {
            c.xmm[0] = xmm(b"a");
            c.xmm[1] = xmm(b"zza");
        })
        .run();
    assert_eq!(plain.ctx.rcx, elided.ctx.rcx);
    assert_eq!(
        (plain.ctx.cf, plain.ctx.zf, plain.ctx.sf, plain.ctx.of),
        (elided.ctx.cf, elided.ctx.zf, elided.ctx.sf, elided.ctx.of)
    );
}
