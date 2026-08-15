//! SSE4.2 packed string compare - `PCMPISTRI`, `PCMPISTRM`, `PCMPESTRI`,
//! `PCMPESTRM`.
//!
//! Four instructions and one operation. Each compares two 128-bit vectors of
//! byte or word *elements* under one of four aggregation rules, folds the
//! per-element answers into a bitmask, and reports it as either an index (the
//! `...I` forms, into `ECX`) or a mask (the `...M` forms, into `XMM0`) - plus
//! the same five flags in every case. The two families differ only in where the
//! operand *lengths* come from: the implicit (`...ISTR...`) forms scan for the
//! first null element, the explicit (`...ESTR...`) forms take them from
//! `EAX`/`EDX` (`RAX`/`RDX` under `REX.W`).
//!
//! ## The `imm8` is a lift-time constant, and that is the whole design
//!
//! There is no register-operand form of the control byte, so every one of its
//! four decisions - element width and signedness, aggregation rule, polarity,
//! index-vs-mask output - is known while lifting. None of them becomes a
//! runtime branch or select: the handler *is* the interpreter for `imm8`, and
//! what it emits is straight-line code for exactly the one variant in front of
//! it. Only the element data and the two lengths stay dynamic.
//!
//! ## Validity, and why it is not an optimisation
//!
//! The aggregation is defined over all 8 or 16 element positions of both
//! operands, including the ones past the operand's length. Those positions do
//! not compare - they take a *forced* value, and which value depends on the
//! aggregation rule and on which side is out of range. The table is the one in
//! the SDM's `PCMPxSTRx` operation section:
//!
//! | `xmm1[i]` | `xmm2[j]` | equal any | ranges | equal each | equal ordered |
//! | --- | --- | --- | --- | --- | --- |
//! | invalid | invalid | false | false | **true** | **true** |
//! | invalid | valid   | false | false | false | **true** |
//! | valid   | invalid | false | false | false | false |
//! | valid   | valid   | compare | compare | compare | compare |
//!
//! The two `true`s in the "equal ordered" column are what make a substring
//! search report a match that runs off the end of the needle, and the `true` in
//! "equal each" is what makes two exhausted strings compare equal. Neither is a
//! don't-care: a lift that treated invalid positions as merely "not matching"
//! computes the wrong answer for `strcmp` and `strstr`, which is most of what
//! these instructions are used for.
//!
//! "Invalid" also covers a position that does not exist. In the equal-ordered
//! aggregation, needle element `i` is paired with haystack element `j + i`,
//! which for a late `j` runs past the end of the register - and that pairing
//! takes the same forced value as one past the operand's length, rather than
//! being skipped. The difference is observable: `PCMPESTRM` with a two-element
//! first operand and a full second one reports `0x7FFF`, not `0xFFFF`, because
//! a two-element needle cannot start at the last element. The tables above were
//! checked against hardware through the Unicorn oracle; where the operation
//! section admits a second reading, the hardware is what this implements.
//!
//! ## Why the aggregation is outlined
//!
//! "Equal any" and "equal ordered" compare every element of one operand against
//! every element of the other: 256 comparisons at byte width, each a handful of
//! IR instructions on top of the 32 lane extractions. That is softfloat-kernel
//! territory ([`softfloat`](super::softfloat)), so it uses the same mechanism -
//! except that the body depends on the control byte, so there is one helper per
//! distinct *aggregation-relevant* nibble rather than one per instruction.
//!
//! Only `imm8[3:0]` reaches the helper. Polarity (`imm8[5:4]`) is a mask over
//! the result and output select (`imm8[6]`) picks between two ways of reporting
//! it, and both are a few instructions at the call site - keeping them out of
//! the body is what holds the helper count at 16 rather than 128, and lets one
//! `PCMPISTRI` and one `PCMPESTRM` with the same element format and rule share
//! a body.

use crate::arch::x86_64::regs::X86Flag;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::{Code, Instruction, Register};

/// Which of the four instructions is being lifted - i.e. where the lengths come
/// from and where the result goes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct PcmpstrForm {
    /// Explicit-length (`PCMPESTRx`): lengths from `rAX`/`rDX` rather than from
    /// a scan for the first null element.
    pub explicit: bool,
    /// `REX.W`: the explicit lengths are read from `RAX`/`RDX` rather than
    /// `EAX`/`EDX`. Meaningless for the implicit forms.
    pub wide: bool,
    /// Index output into `ECX` (`PCMPxSTRI`) rather than a mask into `XMM0`.
    pub index: bool,
}

impl PcmpstrForm {
    /// Classify an instruction, or `None` if it is not one of the four.
    pub(super) fn of(instruction: &Instruction) -> Option<PcmpstrForm> {
        let (explicit, wide, index) = match instruction.code() {
            Code::Pcmpistri_xmm_xmmm128_imm8 => (false, false, true),
            Code::Pcmpistrm_xmm_xmmm128_imm8 => (false, false, false),
            Code::Pcmpestri_xmm_xmmm128_imm8 => (true, false, true),
            Code::Pcmpestri64_xmm_xmmm128_imm8 => (true, true, true),
            Code::Pcmpestrm_xmm_xmmm128_imm8 => (true, false, false),
            Code::Pcmpestrm64_xmm_xmmm128_imm8 => (true, true, false),
            _ => return None,
        };
        Some(PcmpstrForm {
            explicit,
            wide,
            index,
        })
    }
}

/// Element width in bits for a control byte: `imm8[0]` selects words.
fn element_bits(ctl: u8) -> u32 {
    if ctl & 1 != 0 { 16 } else { 8 }
}

/// Number of elements in a 128-bit operand under `ctl`: 8 words or 16 bytes.
fn element_count(ctl: u8) -> usize {
    (128 / element_bits(ctl)) as usize
}

/// `imm8[1]` - elements are signed.
fn signed(ctl: u8) -> bool {
    ctl & 2 != 0
}

// The aggregation kernel.

/// The 16 helper symbols, indexed by `imm8[3:0]`.
///
/// Spelled out rather than formatted because
/// [`outlined_helper`](crate::core::LiftEngine::outlined_helper) keys and
/// de-duplicates on a `&'static str`, and because the nibble is exactly what
/// distinguishes two bodies - a name that did not encode it would silently
/// share one body between two different aggregations.
const AGG_SYMBOLS: [&str; 16] = [
    "__microlift_pcmpstr_agg_0",
    "__microlift_pcmpstr_agg_1",
    "__microlift_pcmpstr_agg_2",
    "__microlift_pcmpstr_agg_3",
    "__microlift_pcmpstr_agg_4",
    "__microlift_pcmpstr_agg_5",
    "__microlift_pcmpstr_agg_6",
    "__microlift_pcmpstr_agg_7",
    "__microlift_pcmpstr_agg_8",
    "__microlift_pcmpstr_agg_9",
    "__microlift_pcmpstr_agg_a",
    "__microlift_pcmpstr_agg_b",
    "__microlift_pcmpstr_agg_c",
    "__microlift_pcmpstr_agg_d",
    "__microlift_pcmpstr_agg_e",
    "__microlift_pcmpstr_agg_f",
];

/// One outlinable aggregation body, identified by `imm8[3:0]` - element format
/// (`[1:0]`) and aggregation rule (`[3:2]`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct PcmpstrKernel(pub u8);

impl PcmpstrKernel {
    /// The symbol the helper function is declared under.
    pub fn symbol(self) -> &'static str {
        AGG_SYMBOLS[(self.0 & 0xF) as usize]
    }

    /// `(lo1, hi1, lo2, hi2: I64, len1, len2: I32) -> I32`, the last being
    /// `IntRes1`. `call_conv` should match the calling function's, so a call
    /// needs no convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.params.push(AbiParam::new(types::I32));
        sig.params.push(AbiParam::new(types::I32));
        sig.returns.push(AbiParam::new(types::I32));
        sig
    }

    /// Emit the kernel's body into `b`, whose function must already have been
    /// given [`signature`](Self::signature).
    pub fn build_body(self, b: &mut FunctionBuilder) {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let p: Vec<Value> = b.block_params(entry).to_vec();
        let r = emit_pcmpstr_agg(b, self.0, p[0], p[1], p[2], p[3], p[4], p[5]);
        b.ins().return_(&[r]);
    }
}

/// Extract the `128 / bits` elements of a `(lo, hi)` operand, each widened into
/// a full `I64` and sign- or zero-extended per `signed`, lowest element first.
///
/// A free-function twin of [`Lifter::xmm_lanes`](super::sse_int) - the helper
/// bodies have a `FunctionBuilder` and no `Lifter`.
fn elements(b: &mut FunctionBuilder, lo: Value, hi: Value, bits: u32, signed: bool) -> Vec<Value> {
    let per_half = 64 / bits;
    let mut out = Vec::with_capacity((per_half * 2) as usize);
    for v in [lo, hi] {
        for lane in 0..per_half {
            let shift = i64::from(lane * bits);
            out.push(if signed {
                let left = 64 - shift - i64::from(bits);
                let aligned = if left == 0 {
                    v
                } else {
                    b.ins().ishl_imm_u(v, left)
                };
                b.ins().sshr_imm_u(aligned, 64 - i64::from(bits))
            } else {
                let shifted = if shift == 0 {
                    v
                } else {
                    b.ins().ushr_imm_u(v, shift)
                };
                b.ins().band_imm_u(shifted, ((1u64 << bits) - 1) as i64)
            });
        }
    }
    out
}

/// `valid[k] = k < len`, as `n` `I8` booleans.
fn validity(b: &mut FunctionBuilder, len: Value, n: usize) -> Vec<Value> {
    (0..n)
        .map(|k| b.ins().icmp_imm_s(IntCC::SignedGreaterThan, len, k as i64))
        .collect()
}

/// Logical negation of a 0/1 `I8`.
fn not(b: &mut FunctionBuilder, v: Value) -> Value {
    b.ins().bxor_imm_u(v, 1)
}

/// Pack `n` 0/1 `I8` bits into an `I32`, bit `j` from `bits[j]`.
fn pack_bits(b: &mut FunctionBuilder, bits: &[Value]) -> Value {
    let mut acc = b.ins().iconst(types::I32, 0);
    for (j, &bit) in bits.iter().enumerate() {
        let wide = b.ins().uextend(types::I32, bit);
        let placed = if j == 0 {
            wide
        } else {
            b.ins().ishl_imm_u(wide, j as i64)
        };
        acc = b.ins().bor(acc, placed);
    }
    acc
}

/// Emit the aggregation for control nibble `ctl`, producing `IntRes1` as an
/// `I32` with one bit per element position of the *second* operand.
///
/// `len1` / `len2` are element counts in `0..=n`, already saturated by the
/// caller; everything at or above them is an invalid position and takes the
/// forced value from the module's table rather than a comparison.
pub fn emit_pcmpstr_agg(
    b: &mut FunctionBuilder,
    ctl: u8,
    lo1: Value,
    hi1: Value,
    lo2: Value,
    hi2: Value,
    len1: Value,
    len2: Value,
) -> Value {
    let bits = element_bits(ctl);
    let n = element_count(ctl);
    let sign = signed(ctl);

    // Both operands' elements are extended the same way, so a signed compare on
    // the widened `I64`s is the right ordering for both formats: a zero-extended
    // unsigned element can never look negative at 64 bits.
    let a = elements(b, lo1, hi1, bits, sign);
    let c = elements(b, lo2, hi2, bits, sign);
    let v1 = validity(b, len1, n);
    let v2 = validity(b, len2, n);

    let result: Vec<Value> = match (ctl >> 2) & 3 {
        // Equal any: does element j of the second operand appear anywhere in the
        // first? Every forced value is false, so an invalid position on either
        // side simply contributes nothing - which is why the whole `v2[j]` term
        // factors out of the OR.
        0 => (0..n)
            .map(|j| {
                let mut any = b.ins().iconst(types::I8, 0);
                for i in 0..n {
                    let eq = b.ins().icmp(IntCC::Equal, a[i], c[j]);
                    let live = b.ins().band(eq, v1[i]);
                    any = b.ins().bor(any, live);
                }
                b.ins().band(any, v2[j])
            })
            .collect(),

        // Ranges: consecutive pairs of first-operand elements are `[lo, hi]`
        // bounds, and element j is in range if it is within any of them. Forced
        // values are false as above.
        1 => (0..n)
            .map(|j| {
                let mut any = b.ins().iconst(types::I8, 0);
                for k in (0..n).step_by(2) {
                    let ge = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, c[j], a[k]);
                    let le = b.ins().icmp(IntCC::SignedLessThanOrEqual, c[j], a[k + 1]);
                    let within = b.ins().band(ge, le);
                    // The upper bound being a valid element implies the lower one
                    // is (validity is a prefix), but both are named so the
                    // correspondence with the forcing table stays readable.
                    let bounded = b.ins().band(v1[k], v1[k + 1]);
                    let live = b.ins().band(within, bounded);
                    any = b.ins().bor(any, live);
                }
                b.ins().band(any, v2[j])
            })
            .collect(),

        // Equal each: element-wise equality. Two positions that are *both* past
        // their operand's end force true - this is what makes two strings of the
        // same length compare equal all the way to element 15.
        2 => (0..n)
            .map(|j| {
                let eq = b.ins().icmp(IntCC::Equal, a[j], c[j]);
                let both_valid = b.ins().band(v1[j], v2[j]);
                let matched = b.ins().band(eq, both_valid);
                let n1 = not(b, v1[j]);
                let n2 = not(b, v2[j]);
                let both_invalid = b.ins().band(n1, n2);
                b.ins().bor(matched, both_invalid)
            })
            .collect(),

        // Equal ordered: substring search. Bit j asks whether the first operand
        // occurs in the second starting at j. Running off the *needle* (an
        // invalid `a[i]`) forces true, so a short needle still matches; running
        // off the *haystack* forces false, so a match may not extend past the
        // end of the string being searched.
        //
        // The pairing runs over all `n` positions of the needle, not the
        // `n - j` that stay inside the register: a needle element whose
        // haystack position would be element `n` or beyond is compared against
        // an out-of-range - hence invalid - element, and so forces false unless
        // the needle has run out first. Stopping the loop at the register edge
        // instead would report a match for a needle that demonstrably does not
        // fit, which is the one place the natural reading of the operation
        // section is wrong.
        _ => (0..n)
            .map(|j| {
                let mut all = b.ins().iconst(types::I8, 1);
                for i in 0..(n - j) {
                    let eq = b.ins().icmp(IntCC::Equal, a[i], c[j + i]);
                    let live = b.ins().band(eq, v2[j + i]);
                    let exhausted = not(b, v1[i]);
                    let term = b.ins().bor(live, exhausted);
                    all = b.ins().band(all, term);
                }
                if j > 0 {
                    // The tail `i >= n - j`, whose every term is "the needle has
                    // run out". Validity is a prefix, so all of them hold
                    // exactly when the first of them does.
                    let fits = not(b, v1[n - j]);
                    all = b.ins().band(all, fits);
                }
                all
            })
            .collect(),
    };

    pack_bits(b, &result)
}

impl<'a> Lifter<'a> {
    /// `PCMPISTRI` / `PCMPISTRM` / `PCMPESTRI` / `PCMPESTRM`.
    pub(super) fn handle_pcmpstr(
        &mut self,
        instruction: &Instruction,
        form: PcmpstrForm,
    ) -> crate::Result<()> {
        let imm = instruction.immediate8();
        let ctl = imm & 0x0F;
        let bits = element_bits(ctl);
        let n = element_count(ctl);

        let src1_idx = self.xmm_operand_index(instruction, 0, "pcmpstr")?;
        let lo1 = self.get_xmm_lo(src1_idx);
        let hi1 = self.get_xmm_hi(src1_idx);
        let (lo2, hi2) = self.read_xmm_src(instruction, 1)?;

        let (len1, len2) = if form.explicit {
            let (a, d) = if form.wide {
                (Register::RAX, Register::RDX)
            } else {
                (Register::EAX, Register::EDX)
            };
            (self.explicit_len(a, n)?, self.explicit_len(d, n)?)
        } else {
            (
                self.implicit_len(lo1, hi1, bits),
                self.implicit_len(lo2, hi2, bits),
            )
        };

        let int_res1 = self.pcmpstr_agg(ctl, lo1, hi1, lo2, hi2, len1, len2);
        let int_res2 = self.apply_polarity(imm, int_res1, len2, n);

        self.pcmpstr_flags(int_res2, len1, len2, n)?;

        if form.index {
            self.pcmpstr_index_out(imm, int_res2, n)
        } else {
            self.pcmpstr_mask_out(imm, int_res2, bits, n)
        }
    }

    /// Emit the aggregation, as a call to the shared per-nibble helper or -
    /// when outlining is off - inline at this site. Both paths run the identical
    /// [`emit_pcmpstr_agg`] builder.
    fn pcmpstr_agg(
        &mut self,
        ctl: u8,
        lo1: Value,
        hi1: Value,
        lo2: Value,
        hi2: Value,
        len1: Value,
        len2: Value,
    ) -> Value {
        if !self.outline_helpers() {
            return emit_pcmpstr_agg(&mut self.builder, ctl, lo1, hi1, lo2, hi2, len1, len2);
        }

        let kernel = PcmpstrKernel(ctl);
        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self
            .builder
            .ins()
            .call(func_ref, &[lo1, hi1, lo2, hi2, len1, len2]);
        self.builder.inst_results(call)[0]
    }

    /// The implicit-length rule: the number of elements before the first null
    /// one, or all `n` if there is none.
    ///
    /// Built as a fold from the top so that the *lowest* null wins, which is the
    /// C string rule - a later null past the terminator must not extend the
    /// length.
    fn implicit_len(&mut self, lo: Value, hi: Value, bits: u32) -> Value {
        let lanes = self.xmm_lanes(lo, hi, bits, false);
        let n = lanes.len();
        let mut len = self.builder.ins().iconst(types::I32, n as i64);
        for j in (0..n).rev() {
            let is_null = self.builder.ins().icmp_imm_s(IntCC::Equal, lanes[j], 0);
            let here = self.builder.ins().iconst(types::I32, j as i64);
            len = self.builder.ins().select(is_null, here, len);
        }
        len
    }

    /// The explicit-length rule: `|reg|`, saturated to `n`.
    ///
    /// The comparison against `n` is *unsigned* on purpose. `iabs` of the most
    /// negative value is itself, and reading that as unsigned makes it larger
    /// than `n` - so the one input whose absolute value does not exist
    /// saturates, rather than passing through as a huge negative length.
    fn explicit_len(&mut self, reg: Register, n: usize) -> crate::Result<Value> {
        let v = self.get_reg_val(reg)?;
        let wide = if self.builder.func.dfg.value_type(v) == types::I64 {
            v
        } else {
            self.builder.ins().sextend(types::I64, v)
        };
        let abs = self.builder.ins().iabs(wide);
        let cap = self.builder.ins().iconst(types::I64, n as i64);
        let over = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThan, abs, cap);
        let clamped = self.builder.ins().select(over, cap, abs);
        Ok(self.builder.ins().ireduce(types::I32, clamped))
    }

    /// `imm8[5:4]`: leave `IntRes1` alone, invert every element position, or
    /// invert only the positions that are valid in the second operand.
    ///
    /// The masked form (`11`) is the one a `strcmp` uses: inverting the invalid
    /// positions too would report a mismatch past the end of both strings.
    fn apply_polarity(&mut self, imm: u8, int_res1: Value, len2: Value, n: usize) -> Value {
        let full = ((1u64 << n) - 1) as i64;
        match (imm >> 4) & 3 {
            1 => {
                let inverted = self.builder.ins().bnot(int_res1);
                self.builder.ins().band_imm_u(inverted, full)
            }
            3 => {
                let one = self.builder.ins().iconst(types::I32, 1);
                let shifted = self.builder.ins().ishl(one, len2);
                let mask = self.builder.ins().iadd_imm_s(shifted, -1);
                let mask = self.builder.ins().band_imm_u(mask, full);
                self.builder.ins().bxor(int_res1, mask)
            }
            _ => int_res1,
        }
    }

    /// The five flags every `PCMPxSTRx` defines, plus the two it clears.
    ///
    /// `ZF` and `SF` report *operand exhaustion* rather than anything about the
    /// comparison: each is set when the corresponding operand held fewer than a
    /// full register of elements, which for the implicit forms means it
    /// contained a null.
    fn pcmpstr_flags(
        &mut self,
        int_res2: Value,
        len1: Value,
        len2: Value,
        n: usize,
    ) -> crate::Result<()> {
        let cf = self.builder.ins().icmp_imm_s(IntCC::NotEqual, int_res2, 0);
        self.set_flag(X86Flag::Carry, cf)?;

        let zf = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, len2, n as i64);
        self.set_flag(X86Flag::Zero, zf)?;

        let sf = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, len1, n as i64);
        self.set_flag(X86Flag::Sign, sf)?;

        let of = self.builder.ins().band_imm_u(int_res2, 1);
        self.set_flag(X86Flag::Overflow, of)?;

        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Auxiliary, zero)?;
        self.set_flag(X86Flag::Parity, zero)
    }

    /// `PCMPxSTRI`: `ECX` gets the index of the lowest (`imm8[6] == 0`) or
    /// highest set bit of `IntRes2`, or `n` when there is none.
    fn pcmpstr_index_out(&mut self, imm: u8, int_res2: Value, n: usize) -> crate::Result<()> {
        let index = if imm & 0x40 == 0 {
            self.builder.ins().ctz(int_res2)
        } else {
            // `clz` counts from bit 31 of the I32, so the set-bit index is its
            // complement within the word.
            let lead = self.builder.ins().clz(int_res2);
            let top = self.builder.ins().iconst(types::I32, 31);
            self.builder.ins().isub(top, lead)
        };
        let empty = self.builder.ins().icmp_imm_s(IntCC::Equal, int_res2, 0);
        let none = self.builder.ins().iconst(types::I32, n as i64);
        let ecx = self.builder.ins().select(empty, none, index);
        self.set_reg_val(Register::ECX, ecx)
    }

    /// `PCMPxSTRM`: `XMM0` gets `IntRes2` either as a bitmask in its low bits
    /// (`imm8[6] == 0`) or expanded to one all-ones element per set bit.
    ///
    /// Both forms write the whole register - the bitmask form zeroes everything
    /// above bit `n`.
    fn pcmpstr_mask_out(
        &mut self,
        imm: u8,
        int_res2: Value,
        bits: u32,
        n: usize,
    ) -> crate::Result<()> {
        let (lo, hi) = if imm & 0x40 == 0 {
            let lo = self.builder.ins().uextend(types::I64, int_res2);
            let hi = self.builder.ins().iconst(types::I64, 0);
            (lo, hi)
        } else {
            let lanes: Vec<Value> = (0..n)
                .map(|j| {
                    let shifted = if j == 0 {
                        int_res2
                    } else {
                        self.builder.ins().ushr_imm_u(int_res2, j as i64)
                    };
                    let bit = self.builder.ins().band_imm_u(shifted, 1);
                    let wide = self.builder.ins().uextend(types::I64, bit);
                    // 0 or 1 negated is 0 or all-ones; `pack_xmm_lanes` keeps
                    // only the low `bits` of each.
                    self.builder.ins().ineg(wide)
                })
                .collect();
            self.pack_xmm_lanes(&lanes, bits)
        };
        self.set_xmm_lo(0, lo);
        self.set_xmm_hi(0, hi);
        Ok(())
    }
}
