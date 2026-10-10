//! SSE4.2 packed string compare: `PCMPISTRI`, `PCMPISTRM`, `PCMPESTRI`, `PCMPESTRM`.
//!
//! Compares two 128-bit vectors of byte/word elements under one of four aggregation rules and reports a bitmask as an index
//! (`...I`, `ECX`) or mask (`...M`, `XMM0`) plus five flags. Implicit forms scan for the first null for lengths; explicit forms use `EAX`/`EDX`.
//! The control byte is an immediate, so every decision is known at lift time.
//!
//! ## Validity
//!
//! Positions past an operand's length take a *forced* value (SDM `PCMPxSTRx`):
//!
//! | `xmm1[i]` | `xmm2[j]` | equal any | ranges | equal each | equal ordered |
//! | --- | --- | --- | --- | --- | --- |
//! | invalid | invalid | false | false | **true** | **true** |
//! | invalid | valid   | false | false | false | **true** |
//! | valid   | invalid | false | false | false | false |
//! | valid   | valid   | compare | compare | compare | compare |
//!
//! The `true`s are not don't-cares: they make a substring search match a needle running off the end, and exhausted strings compare equal.
//! In equal-ordered, needle element `i` pairs with haystack `j + i`, which past the register takes the same forced value
//! (`PCMPESTRM` with a two-element first operand and a full second reports `0x7FFF`; checked against hardware).
//!
//! The exhaustive aggregations are **outlined**, one helper per `imm8[3:0]`; polarity and output select stay at the call site (16 helpers).

use crate::arch::x86_64::regs::X86Flag;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::{Code, Instruction, Register};

/// Which of the four instructions: where lengths come from and where the result goes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct PcmpstrForm {
    /// Explicit-length (`PCMPESTRx`): lengths from `rAX`/`rDX`, not a null scan.
    pub explicit: bool,
    /// `REX.W`: explicit lengths from `RAX`/`RDX` rather than `EAX`/`EDX`.
    pub wide: bool,
    /// Index output into `ECX` (`PCMPxSTRI`) rather than a mask into `XMM0`.
    pub index: bool,
}

impl PcmpstrForm {
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


/// The 16 helper symbols, indexed by `imm8[3:0]`; spelled out because [`outlined_helper`](crate::core::LiftEngine::outlined_helper) de-duplicates on a `&'static str`.
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
    pub fn symbol(self) -> &'static str {
        AGG_SYMBOLS[(self.0 & 0xF) as usize]
    }

    /// `(lo1, hi1, lo2, hi2: I64, len1, len2: I32) -> I32`, the last being `IntRes1`.
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

/// Extract the `128 / bits` elements of a `(lo, hi)` operand, each widened to `I64` (sign- or zero-extended per `signed`), lowest first.
/// Free-function twin of [`Lifter::xmm_lanes`](super::sse_int).
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

/// Emit the aggregation for control nibble `ctl`, producing `IntRes1` as an `I32` with one bit per element of the *second* operand.
/// `len1`/`len2` are saturated element counts in `0..=n`; positions at or above them take the forced value.
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

    // Both operands extend the same way, so a signed compare on `I64`s orders both formats.
    let a = elements(b, lo1, hi1, bits, sign);
    let c = elements(b, lo2, hi2, bits, sign);
    let v1 = validity(b, len1, n);
    let v2 = validity(b, len2, n);

    let result: Vec<Value> = match (ctl >> 2) & 3 {
        // Equal any: does v2[j] appear in the first operand? Forced values are false, so invalid positions contribute nothing.
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

        // Ranges: consecutive first-operand pairs are `[lo, hi]` bounds; element j is in range if within any. Forced values are false.
        1 => (0..n)
            .map(|j| {
                let mut any = b.ins().iconst(types::I8, 0);
                for k in (0..n).step_by(2) {
                    let ge = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, c[j], a[k]);
                    let le = b.ins().icmp(IntCC::SignedLessThanOrEqual, c[j], a[k + 1]);
                    let within = b.ins().band(ge, le);
                    // A valid upper bound implies a valid lower (validity is a prefix); both are named to match the forcing table.
                    let bounded = b.ins().band(v1[k], v1[k + 1]);
                    let live = b.ins().band(within, bounded);
                    any = b.ins().bor(any, live);
                }
                b.ins().band(any, v2[j])
            })
            .collect(),

        // Equal each: element-wise; two positions both past their end force true, so equal-length strings match to element 15.
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

        // Equal ordered: bit j asks whether the first operand occurs in the second at j. Running off the needle forces true, off the haystack false.
        // The pairing runs over all `n` needle positions, not just those inside the register: a needle element whose
        // haystack position is `n` or beyond forces false unless the needle ran out first.
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
                    // The tail `i >= n - j` is all "needle has run out"; validity is a prefix, so all hold when the first does.
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

    /// Emit the aggregation as a call to the shared per-nibble helper, or inline when outlining is off.
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

    /// The implicit-length rule: elements before the first null, or `n` if none; folded from the top so the lowest null wins.
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

    /// The explicit-length rule: `|reg|` saturated to `n`; the compare is *unsigned* so `iabs` of the minimum (itself) saturates.
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

    /// `imm8[5:4]`: leave `IntRes1`, invert every position, or invert only positions valid in the second operand.
    /// The masked form (`11`) is `strcmp`'s: inverting invalid positions would report a mismatch past both ends.
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
    /// `ZF`/`SF` report operand exhaustion (fewer than a full register of elements; for implicit forms, a null), not the comparison.
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
            // `clz` counts from bit 31, so the index is its complement.
            let lead = self.builder.ins().clz(int_res2);
            let top = self.builder.ins().iconst(types::I32, 31);
            self.builder.ins().isub(top, lead)
        };
        let empty = self.builder.ins().icmp_imm_s(IntCC::Equal, int_res2, 0);
        let none = self.builder.ins().iconst(types::I32, n as i64);
        let ecx = self.builder.ins().select(empty, none, index);
        self.set_reg_val(Register::ECX, ecx)
    }

    /// `PCMPxSTRM`: `XMM0` gets `IntRes2` as a bitmask in the low bits (`imm8[6] == 0`) or one all-ones element per set bit;
    /// the whole register is written.
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
                    // 0 or 1 negated is 0 or all-ones; `pack_xmm_lanes` keeps the low `bits`.
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
