//! `MXCSR` and x87 environment instructions: `STMXCSR`, `LDMXCSR` (plus their
//! AVX spellings), `FNSTENV`, `FLDENV`, `FNCLEX`, `FNINIT` and `FWAIT`.
//!
//! These are the instructions a compiler emits around a floating-point control
//! change - `fesetround`, `feholdexcept`, `fegetenv`/`fesetenv`,
//! `_mm_setcsr`/`_mm_getcsr` - so the tests are written as much against those
//! *sequences* as against the individual opcodes.
//!
//! Run: `cargo test --features testkit --test mxcsr_tests`

use microlift::context::{MXCSR_RESET, X87_CW_RESET};
use microlift::testkit::prelude::*;

fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

/// A 64-byte host buffer the guest addresses through `RSI` (guest VA == host
/// VA), returned along with its address.
fn scratch() -> (Box<[u8; 64]>, u64) {
    let buf = Box::new([0u8; 64]);
    let va = buf.as_ptr() as u64;
    (buf, va)
}

fn rd32(buf: &[u8; 64], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn rd16(buf: &[u8; 64], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().unwrap())
}

// ---------------------------------------------------------------- STMXCSR --

/// A fresh context must report the architectural reset `MXCSR` (0x1F80: all
/// exceptions masked, round-to-nearest), not zero.
#[test]
fn test_stmxcsr_stores_reset_value() {
    let (buf, va) = scratch();
    // stmxcsr [rsi] ; ret
    let bytes = [0x0F, 0xAE, 0x1E, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    run_test(&bytes, ctx);
    assert_eq!(rd32(&buf, 0), MXCSR_RESET);
}

/// `STMXCSR` writes exactly four bytes and nothing else.
#[test]
fn test_stmxcsr_writes_four_bytes_only() {
    let (buf, va) = scratch();
    let bytes = [0x0F, 0xAE, 0x1E, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.mxcsr = 0xFFFF;
    run_test(&bytes, ctx);
    assert_eq!(rd32(&buf, 0), 0xFFFF);
    assert!(
        buf[4..].iter().all(|&b| b == 0),
        "stmxcsr must not touch memory past the m32 operand"
    );
}

/// The context field is the authority: whatever the host puts there is what
/// `STMXCSR` reports.
#[test]
fn test_stmxcsr_reflects_context_field() {
    let (buf, va) = scratch();
    // stmxcsr [rsi+4] ; ret
    let bytes = [0x0F, 0xAE, 0x5E, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.mxcsr = 0x0000_9FA5;
    run_test(&bytes, ctx);
    assert_eq!(rd32(&buf, 4), 0x0000_9FA5);
    assert_eq!(rd32(&buf, 0), 0, "displacement must be honoured");
}

// ---------------------------------------------------------------- LDMXCSR --

/// Every defined bit round-trips through `LDMXCSR` into the context.
#[test]
fn test_ldmxcsr_loads_all_defined_bits() {
    for value in [0x0000u32, 0x1F80, 0xFFFF, 0x6000, 0x8040, 0x003F] {
        let (mut buf, va) = scratch();
        buf[0..4].copy_from_slice(&value.to_le_bytes());
        // ldmxcsr [rsi] ; ret
        let bytes = [0x0F, 0xAE, 0x16, 0xC3];
        let mut ctx = RegisterContext::default();
        ctx.rsi = va;
        let ctx = run_test(&bytes, ctx);
        assert_eq!(ctx.mxcsr, value, "ldmxcsr {value:#x}");
    }
}

/// Bits 31:16 are reserved; hardware would `#GP(0)` on them and Microlift
/// (which models no such fault) drops them, leaving the state a legal
/// `LDMXCSR` would have produced.
#[test]
fn test_ldmxcsr_drops_reserved_bits() {
    let (mut buf, va) = scratch();
    buf[0..4].copy_from_slice(&0xDEAD_1F80u32.to_le_bytes());
    let bytes = [0x0F, 0xAE, 0x16, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let ctx = run_test(&bytes, ctx);
    assert_eq!(ctx.mxcsr, 0x1F80);
}

/// `LDMXCSR` then `STMXCSR` is the identity on the defined bits.
#[test]
fn test_ldmxcsr_stmxcsr_roundtrip() {
    let (mut buf, va) = scratch();
    buf[0..4].copy_from_slice(&0x0000_BFC1u32.to_le_bytes());
    // ldmxcsr [rsi] ; stmxcsr [rsi+4] ; ret
    let bytes = [0x0F, 0xAE, 0x16, 0x0F, 0xAE, 0x5E, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);
    assert_eq!(out.mxcsr, 0xBFC1);
    assert_eq!(rd32(&buf, 4), 0xBFC1);
}

/// The AVX encodings (`C5 F8 AE /2`, `/3`) are the same instruction.
#[test]
fn test_vldmxcsr_vstmxcsr() {
    let (mut buf, va) = scratch();
    buf[0..4].copy_from_slice(&0x0000_5F80u32.to_le_bytes());
    // vldmxcsr [rsi] ; vstmxcsr [rsi+4] ; ret
    let bytes = [0xC5, 0xF8, 0xAE, 0x16, 0xC5, 0xF8, 0xAE, 0x5E, 0x04, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);
    assert_eq!(out.mxcsr, 0x5F80);
    assert_eq!(rd32(&buf, 4), 0x5F80);
}

/// The `fesetround(FE_DOWNWARD)` shape: read `MXCSR`, rewrite `RC`, write it
/// back. Exercises `STMXCSR`/`LDMXCSR` against the GPR/memory instructions a
/// compiler puts between them.
#[test]
fn test_fesetround_sequence() {
    let (buf, va) = scratch();
    #[rustfmt::skip]
    let bytes = [
        0x0F, 0xAE, 0x1E,                   // stmxcsr [rsi]
        0x8B, 0x06,                         // mov eax, [rsi]
        0x25, 0xFF, 0x9F, 0xFF, 0xFF,       // and eax, ~0x6000  (clear RC)
        0x0D, 0x00, 0x20, 0x00, 0x00,       // or  eax, 0x2000   (RC = down)
        0x89, 0x06,                         // mov [rsi], eax
        0x0F, 0xAE, 0x16,                   // ldmxcsr [rsi]
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);
    assert_eq!(out.mxcsr, 0x3F80, "RC=01 (toward -inf) on top of the reset");
    assert_eq!(rd32(&buf, 0), 0x3F80);
}

// ----------------------------------------------------------------- FNSTENV --

/// The 28-byte (32-bit operand size) environment layout: control word at +0,
/// status word at +4, tag word at +8, everything else zero.
#[test]
fn test_fnstenv_28byte_layout() {
    let (buf, va) = scratch();
    // fnstenv [rsi] ; ret
    let bytes = [0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x0300; // extended precision, all exceptions *unmasked*
    ctx.fpu_sw = 0x4100; // C3 | C0
    ctx.fpu_top = 3;
    ctx.fpu_tag = 0; // every register empty
    run_test(&bytes, ctx);

    assert_eq!(rd16(&buf, 0), 0x0300, "FCW at +0 (pre-masking value)");
    assert_eq!(
        rd16(&buf, 4),
        0x4100 | (3 << 11),
        "FSW at +4, with the live TOP merged in"
    );
    assert_eq!(rd16(&buf, 8), 0xFFFF, "FTW at +8: all eight regs empty");
    // Reserved halves and the (unmodelled) pointer fields.
    for off in [2, 6, 10, 12, 16, 20, 24] {
        assert_eq!(rd16(&buf, off), 0, "environment offset {off} must be zero");
    }
    assert!(
        buf[28..].iter().all(|&b| b == 0),
        "fnstenv must write exactly 28 bytes"
    );
}

/// The 14-byte (16-bit operand size) layout packs the same three words at +0,
/// +2 and +4.
#[test]
fn test_fnstenv_14byte_layout() {
    let (buf, va) = scratch();
    // o16 fnstenv [rsi] ; ret
    let bytes = [0x66, 0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x027F;
    ctx.fpu_sw = 0x0400; // C2
    ctx.fpu_top = 5;
    run_test(&bytes, ctx);

    assert_eq!(rd16(&buf, 0), 0x027F);
    assert_eq!(rd16(&buf, 2), 0x0400 | (5 << 11));
    assert_eq!(rd16(&buf, 4), 0xFFFF);
    assert!(
        buf[14..].iter().all(|&b| b == 0),
        "the 16-bit form must write exactly 14 bytes"
    );
}

/// `FNSTENV` masks all six floating-point exceptions *after* saving - the
/// property `feholdexcept` relies on.
#[test]
fn test_fnstenv_masks_exceptions_after_saving() {
    let (buf, va) = scratch();
    let bytes = [0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x0300; // no mask bits set
    let out = run_test(&bytes, ctx);
    assert_eq!(rd16(&buf, 0), 0x0300, "the *saved* CW is the old one");
    assert_eq!(
        out.fpu_cw, 0x033F,
        "and the live CW comes back fully masked"
    );
}

/// The architectural tag word is recomputed from the register contents:
/// `00` valid, `01` zero, `10` special, `11` empty.
#[test]
fn test_fnstenv_tag_word_encoding() {
    let (buf, va) = scratch();
    let bytes = [0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    // phys 0: 1.0            -> valid   (00)
    // phys 1: +0.0           -> zero    (01)
    // phys 2: QNaN           -> special (10)
    // phys 3: denormal       -> special (10)
    // phys 4: unnormal       -> special (10)
    // phys 5-7: not live     -> empty   (11)
    ctx.fpr[0] = [0x8000_0000_0000_0000, 0x3FFF];
    ctx.fpr[1] = [0, 0];
    ctx.fpr[2] = [0xC000_0000_0000_0000, 0x7FFF];
    ctx.fpr[3] = [0x0000_0000_0000_0001, 0x0000];
    ctx.fpr[4] = [0x4000_0000_0000_0000, 0x4000]; // integer bit clear
    ctx.fpu_tag = 0b0001_1111;
    run_test(&bytes, ctx);

    // reg7..reg0 = 11 11 11 10 10 10 01 00
    let expected: u16 = 0xFEA4;
    assert_eq!(rd16(&buf, 8), expected, "got {:#06x}", rd16(&buf, 8));
}

/// A negative zero is still tagged `01`, and the sign bit does not leak into
/// the exponent comparison.
#[test]
fn test_fnstenv_negative_zero_tagged_zero() {
    let (buf, va) = scratch();
    let bytes = [0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpr[0] = [0, 0x8000]; // -0.0
    ctx.fpu_tag = 0b0000_0001;
    run_test(&bytes, ctx);
    assert_eq!(rd16(&buf, 8) & 0b11, 0b01);
}

/// An `FLD` of a real value leaves a tag word `FNSTENV` reports as `valid`.
#[test]
fn test_fnstenv_after_fld1() {
    let (buf, va) = scratch();
    // fld1 ; fnstenv [rsi] ; ret
    let bytes = [0xD9, 0xE8, 0xD9, 0x36, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);
    // fld1 pushed: TOP = 7, physical register 7 holds 1.0.
    assert_eq!(out.fpu_top, 7);
    assert_eq!(rd16(&buf, 4) >> 11 & 7, 7, "saved FSW carries TOP = 7");
    assert_eq!(
        rd16(&buf, 8) >> 14,
        0b00,
        "physical register 7 must be tagged valid"
    );
    assert_eq!(rd16(&buf, 8) & 0x3FFF, 0x3FFF, "the rest stay empty");
}

// ------------------------------------------------------------------ FLDENV --

/// `FLDENV` restores the control word, the status word, `TOP` (out of the
/// status word's bits 13-11) and the tag word.
#[test]
fn test_fldenv_restores_environment() {
    let (mut buf, va) = scratch();
    buf[0..2].copy_from_slice(&0x0272u16.to_le_bytes()); // FCW
    buf[4..6].copy_from_slice(&(0x4500u16 | (6 << 11)).to_le_bytes()); // FSW, TOP = 6
    buf[8..10].copy_from_slice(&0xFF34u16.to_le_bytes()); // FTW
    // fldenv [rsi] ; ret
    let bytes = [0xD9, 0x26, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);

    assert_eq!(out.fpu_cw, 0x0272);
    assert_eq!(out.fpu_top, 6, "TOP comes from FSW bits 13-11");
    assert_eq!(out.fpu_sw & !0x3800, 0x4500, "condition codes restored");
    // FTW 0xFF34 tags regs 0..3 as 00/01/11/00 and regs 4..7 as empty, so the
    // abridged (live) mask is 0b0000_1011.
    assert_eq!(out.fpu_tag, 0b0000_1011);
}

/// The 14-byte form reads the same three words from the packed layout.
#[test]
fn test_fldenv_14byte() {
    let (mut buf, va) = scratch();
    buf[0..2].copy_from_slice(&0x137Fu16.to_le_bytes());
    buf[2..4].copy_from_slice(&(0x0200u16 | (2 << 11)).to_le_bytes());
    buf[4..6].copy_from_slice(&0x0FFFu16.to_le_bytes()); // regs 0..5 empty (11)
    // o16 fldenv [rsi] ; ret
    let bytes = [0x66, 0xD9, 0x26, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    let out = run_test(&bytes, ctx);
    assert_eq!(out.fpu_cw, 0x137F);
    assert_eq!(out.fpu_top, 2);
    assert_eq!(out.fpu_tag, 0b1100_0000, "only regs 6 and 7 are non-empty");
}

/// `FLDENV` leaves the data registers alone - it restores the environment, not
/// the stack (that is `FRSTOR`).
#[test]
fn test_fldenv_preserves_data_registers() {
    let (mut buf, va) = scratch();
    buf[0..2].copy_from_slice(&X87_CW_RESET.to_le_bytes());
    buf[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes()); // all empty
    let bytes = [0xD9, 0x26, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpr[2] = [0xDEAD_BEEF_CAFE_F00D, 0x4001];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.fpr[2], [0xDEAD_BEEF_CAFE_F00D, 0x4001]);
    assert_eq!(out.fpu_tag, 0, "but the tag word says they are empty");
}

/// The `feholdexcept` / `fesetenv` pair: save the environment, work under it,
/// then restore. The restored state must equal the saved one exactly.
#[test]
fn test_fnstenv_fldenv_roundtrip() {
    let (buf, va) = scratch();
    // fnstenv [rsi] ; fninit ; fldenv [rsi] ; ret
    let bytes = [0xD9, 0x36, 0xDB, 0xE3, 0xD9, 0x26, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x0300;
    ctx.fpu_sw = 0x4200;
    ctx.fpu_top = 4;
    ctx.fpr[4] = [0x8000_0000_0000_0000, 0x3FFF]; // ST(0) = 1.0
    ctx.fpr[5] = [0, 0]; // ST(1) = +0.0
    ctx.fpu_tag = 0b0011_0000;
    let out = run_test(&bytes, ctx);

    assert_eq!(out.fpu_cw, 0x0300, "the saved CW came back over FNINIT's");
    assert_eq!(out.fpu_top, 4);
    assert_eq!(out.fpu_sw & !0x3800, 0x4200);
    assert_eq!(out.fpu_tag, 0b0011_0000, "live registers restored");
    // The saved tag word: phys 4 = 1.0 (valid, 00), phys 5 = +0.0 (zero, 01),
    // every other register empty (11).
    // reg7..reg0 = 11 11 01 00 11 11 11 11
    let expected: u16 = 0xF4FF;
    assert_eq!(rd16(&buf, 8), expected);
}

// ------------------------------------------------------------ FNCLEX/FNINIT --

/// `FNCLEX` clears the exception flags (7:0) and the busy bit, keeping the
/// condition codes and `TOP`.
#[test]
fn test_fnclex_clears_exception_flags_only() {
    // fnclex ; ret
    let bytes = [0xDB, 0xE2, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.fpu_sw = 0xC5BF; // B | C3 | C2 | C1 | C0 | every exception flag
    ctx.fpu_top = 5;
    ctx.fpu_cw = 0x0300;
    let out = run_test(&bytes, ctx);
    assert_eq!(out.fpu_sw, 0x4500, "only C3/C2/C0 survive (bits 14/10/8)");
    assert_eq!(out.fpu_top, 5, "TOP is untouched");
    assert_eq!(out.fpu_cw, 0x0300, "the control word is untouched");
}

/// `FNINIT` resets the control word, the status word and `TOP`, and tags every
/// register empty - without clearing the register data.
#[test]
fn test_fninit_resets_control_state() {
    // fninit ; ret
    let bytes = [0xDB, 0xE3, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.fpu_cw = 0x0000;
    ctx.fpu_sw = 0xFFFF;
    ctx.fpu_top = 6;
    ctx.fpu_tag = 0xFF;
    ctx.fpr[1] = [0x1234_5678_9ABC_DEF0, 0x4000];
    let out = run_test(&bytes, ctx);
    assert_eq!(out.fpu_cw, X87_CW_RESET);
    assert_eq!(out.fpu_sw, 0);
    assert_eq!(out.fpu_top, 0);
    assert_eq!(out.fpu_tag, 0);
    assert_eq!(out.fpr[1], [0x1234_5678_9ABC_DEF0, 0x4000]);
}

/// A `9B`-prefixed `FSTENV`/`FCLEX` decodes as `WAIT` + the `FN…` form; the
/// `WAIT` must lift as a no-op rather than an unsupported instruction.
#[test]
fn test_wait_prefixed_forms() {
    let (buf, va) = scratch();
    // fwait ; fstenv [rsi] (= 9B D9 36) ; fclex (= 9B DB E2) ; ret
    let bytes = [0x9B, 0x9B, 0xD9, 0x36, 0x9B, 0xDB, 0xE2, 0xC3];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x0300;
    ctx.fpu_sw = 0x00BF; // exception flags set
    let out = run_test(&bytes, ctx);
    assert_eq!(rd16(&buf, 0), 0x0300, "fstenv saved the environment");
    assert_eq!(out.fpu_cw, 0x033F, "fstenv masked the exceptions");
    assert_eq!(out.fpu_sw, 0, "fclex cleared the exception flags");
}

/// The full `feholdexcept()` prologue as GCC/glibc emits it on x86-64: save and
/// mask the x87 environment, then save and mask `MXCSR`.
#[test]
fn test_feholdexcept_shape() {
    let (buf, va) = scratch();
    #[rustfmt::skip]
    let bytes = [
        0xD9, 0x36,                         // fnstenv [rsi]        (x87 env -> +0)
        0xDB, 0xE2,                         // fnclex
        0x0F, 0xAE, 0x5E, 0x1C,             // stmxcsr [rsi+28]
        0x8B, 0x46, 0x1C,                   // mov eax, [rsi+28]
        0x0D, 0x80, 0x1F, 0x00, 0x00,       // or eax, 0x1F80       (mask all, clear nothing)
        0x25, 0xC0, 0xFF, 0xFF, 0xFF,       // and eax, ~0x3F       (clear the sticky flags)
        0x89, 0x46, 0x20,                   // mov [rsi+32], eax
        0x0F, 0xAE, 0x56, 0x20,             // ldmxcsr [rsi+32]
        0xC3,
    ];
    let mut ctx = RegisterContext::default();
    ctx.rsi = va;
    ctx.fpu_cw = 0x0300;
    ctx.mxcsr = 0x0000_0021; // PE | IE raised, everything unmasked
    let out = run_test(&bytes, ctx);

    assert_eq!(rd16(&buf, 0), 0x0300, "the old x87 CW is saved");
    assert_eq!(out.fpu_cw, 0x033F, "and the live one is masked");
    assert_eq!(rd32(&buf, 28), 0x0000_0021, "the old MXCSR is saved");
    assert_eq!(out.mxcsr, 0x1F80, "and the live one is masked and cleared");
}

// ------------------------------------------------------ Unicorn differential --

/// `MXCSR` is not part of the oracle's [`CpuState`], so these cases move it
/// through `EAX` to make it diffable.
#[cfg(feature = "unicorn-oracle")]
mod unicorn {
    use microlift::testkit::oracle::unicorn::differential_unicorn;
    use microlift::testkit::oracle::*;

    const CODE_VA: u64 = 0x1000;
    const BUDGET: u64 = 10_000;

    fn agree(code: &[u8], init: &CpuState) {
        let diff = differential_unicorn(code, CODE_VA, init, &[], BUDGET, DiffOptions::default())
            .expect("both engines should reach a clean halt");
        assert!(diff.is_empty(), "microlift vs unicorn diverged: {diff:?}");
    }

    /// Both engines must start from the same architectural `MXCSR` and report
    /// it identically through `STMXCSR`.
    #[test]
    fn stmxcsr_initial_value_agrees() {
        // The red zone below the harness stack is scratch both engines share.
        #[rustfmt::skip]
        let code = [
            0x0F, 0xAE, 0x5C, 0x24, 0xF8,   // stmxcsr [rsp-8]
            0x8B, 0x44, 0x24, 0xF8,         // mov eax, [rsp-8]
            0xC3,
        ];
        agree(&code, &CpuState::zeroed());
    }

    /// A rounding-mode change round-trips identically on both engines.
    #[test]
    fn ldmxcsr_roundtrip_agrees() {
        #[rustfmt::skip]
        let code = [
            0x0F, 0xAE, 0x5C, 0x24, 0xF8,   // stmxcsr [rsp-8]
            0x8B, 0x44, 0x24, 0xF8,         // mov eax, [rsp-8]
            0x25, 0xFF, 0x9F, 0xFF, 0xFF,   // and eax, ~0x6000
            0x0D, 0x00, 0x40, 0x00, 0x00,   // or eax, 0x4000  (RC = toward +inf)
            0x89, 0x44, 0x24, 0xF8,         // mov [rsp-8], eax
            0x0F, 0xAE, 0x54, 0x24, 0xF8,   // ldmxcsr [rsp-8]
            0x0F, 0xAE, 0x5C, 0x24, 0xF4,   // stmxcsr [rsp-12]
            0x8B, 0x44, 0x24, 0xF4,         // mov eax, [rsp-12]
            0xC3,
        ];
        agree(&code, &CpuState::zeroed());
    }

    /// The x87 environment `FNSTENV` writes - control word, status word (with
    /// `TOP`) and the recomputed tag word - must match Unicorn's, both for a
    /// loaded stack and after `FNINIT`. The pointer fields are not compared:
    /// Unicorn fills in a real `FIP`/`FDP` and Microlift writes zeros.
    #[test]
    fn fnstenv_environment_agrees() {
        #[rustfmt::skip]
        let code = [
            0xD9, 0xE8,                         // fld1        (ST0 = 1.0, valid)
            0xD9, 0xEE,                         // fldz        (ST0 = +0.0, zero)
            0xD9, 0x74, 0x24, 0xC0,             // fnstenv [rsp-64]
            0x8B, 0x44, 0x24, 0xC0,             // mov eax, [rsp-64]         (FCW)
            0x0F, 0xB7, 0x54, 0x24, 0xC4,       // movzx edx, word [rsp-60]  (FSW)
            0x0F, 0xB7, 0x4C, 0x24, 0xC8,       // movzx ecx, word [rsp-56]  (FTW)
            0xDB, 0xE2,                         // fnclex
            0xDB, 0xE3,                         // fninit
            0xD9, 0x74, 0x24, 0xE0,             // fnstenv [rsp-32]
            0x0F, 0xB7, 0x5C, 0x24, 0xE0,       // movzx ebx, word [rsp-32]  (reset FCW)
            0xC3,
        ];
        agree(&code, &CpuState::zeroed());
    }
}
