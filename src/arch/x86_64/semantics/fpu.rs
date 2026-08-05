//! x87 FPU instruction semantics.
//!
//! The x87 register file is a stack of eight registers `ST(0)`-`ST(7)`. The
//! architectural register `ST(i)` is not a fixed location: it names the slot
//! `(TOP + i) & 7`, where `TOP` is a 3-bit pointer that decrements on a push
//! (`FLD`, `FLDZ`, ...) and increments on a pop (`FSTP`, `FADDP`, ...).
//!
//! Because `TOP` is a *runtime* value, this module keeps the whole x87 state in
//! the [`RegisterContext`] (rather than in SSA variables like the GPRs/XMMs) and
//! accesses it through computed pointers. Each `ST(i)` reference loads `TOP`,
//! forms the address `&fpr[(TOP + i) & 7]`, and loads/stores a 16-byte 80-bit
//! slot. This is self-contained: no other instruction category touches the x87
//! fields, so no entry/exit sync integration is required.
//!
//! Registers are stored in true 80-bit extended precision, carried in IR as a
//! `(mant: i64, se: i32)` pair (64-bit significand with explicit integer bit +
//! sign/15-bit-exponent). Data movement (`FLD`/`FST`/`FSTP`/`FXCH`/`FCMOVcc`),
//! the `m80` memory forms, and the integer conversions (`FILD`/`FIST`) are exact
//! 80-bit. The arithmetic (`FADD`/`FSUB`/`FMUL`/`FDIV`/`FSQRT`) and comparisons
//! are computed by the bit-exact software kernels in
//! [`crate::arch::x86_64::semantics::softfloat`] (verified against LLVM's
//! APFloat / the host x87 unit). `f64` is used only for the `m32`/`m64` memory
//! narrowing/widening in `FLD`/`FST`, where it is the correct storage width.

use crate::MicroliftError;
use crate::arch::x86_64::context::RegisterContext;
use crate::arch::x86_64::semantics::softfloat::{self, F80Kernel};
use crate::lifter::{Lifter, X86Flag};
use cranelift::prelude::*;
use iced_x86::{Instruction, MemorySize, OpKind};

/// The binary x87 arithmetic operations, parameterised so a single handler
/// covers `FADD`/`FSUB`/`FSUBR`/`FMUL`/`FDIV`/`FDIVR` and their popping forms.
#[derive(Clone, Copy)]
pub(super) enum FpuOp {
    Add,
    Sub,
    /// Reverse subtract: `dst = src - dst`.
    Subr,
    Mul,
    Div,
    /// Reverse divide: `dst = src / dst`.
    Divr,
}

/// A raw 80-bit x87 value in IR: `(mantissa: i64, sign_exp: i32)`.
type F80 = (Value, Value);

// Status-word condition-code bit positions.
const SW_C0: i64 = 1 << 8;
const SW_C1: i64 = 1 << 9;
const SW_C2: i64 = 1 << 10;
const SW_C3: i64 = 1 << 14;
const SW_CC_MASK: i64 = SW_C0 | SW_C1 | SW_C2 | SW_C3;

impl<'a> Lifter<'a> {
    // --- low-level x87 state access -------------------------------------

    fn fpu_ctx_ptr(&mut self) -> Value {
        self.builder.use_var(self.state.context_ptr)
    }

    /// Load the top-of-stack pointer as an `I32`.
    fn fpu_load_top(&mut self) -> Value {
        let ctx = self.fpu_ctx_ptr();
        let off = std::mem::offset_of!(RegisterContext, fpu_top) as i32;
        let t = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), ctx, off);
        self.builder.ins().uextend(types::I32, t)
    }

    /// Store a new top-of-stack pointer (`I32`, only the low 3 bits matter).
    fn fpu_store_top(&mut self, top: Value) {
        let ctx = self.fpu_ctx_ptr();
        let off = std::mem::offset_of!(RegisterContext, fpu_top) as i32;
        let t16 = self.builder.ins().ireduce(types::I16, top);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), t16, ctx, off);
    }

    /// Pointer to the physical 16-byte slot backing architectural `ST(i)`.
    fn fpu_st_ptr(&mut self, i: u32) -> Value {
        let top = self.fpu_load_top();
        let raw = self.builder.ins().iadd_imm_s(top, i as i64);
        let phys = self.builder.ins().band_imm_u(raw, 7);
        let byte_off = self.builder.ins().imul_imm_s(phys, 16);
        let byte_off = self.builder.ins().uextend(types::I64, byte_off);
        let ctx = self.fpu_ctx_ptr();
        let fpr_off = std::mem::offset_of!(RegisterContext, fpr) as i64;
        let base = self.builder.ins().iadd_imm_s(ctx, fpr_off);
        self.builder.ins().iadd(base, byte_off)
    }

    /// Read `ST(i)` as a raw 80-bit `(mant: i64, se: i32)` pair.
    fn fpu_read_st_raw(&mut self, i: u32) -> (Value, Value) {
        let p = self.fpu_st_ptr(i);
        let mant = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), p, 0);
        let se16 = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), p, 8);
        let se = self.builder.ins().uextend(types::I32, se16);
        (mant, se)
    }

    /// Write a raw 80-bit `(mant, se)` pair into `ST(i)`.
    fn fpu_write_st_raw(&mut self, i: u32, mant: Value, se: Value) {
        let p = self.fpu_st_ptr(i);
        let se16 = self.builder.ins().ireduce(types::I16, se);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), mant, p, 0);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), se16, p, 8);
    }

    /// Decrement `TOP`, store a raw `(mant, se)` pair into the new `ST(0)`, and
    /// mark that physical register valid in the tag word.
    fn fpu_push_raw(&mut self, mant: Value, se: Value) {
        let top = self.fpu_load_top();
        let dec = self.builder.ins().iadd_imm_s(top, -1);
        let new_top = self.builder.ins().band_imm_u(dec, 7);
        self.fpu_store_top(new_top);
        self.fpu_write_st_raw(0, mant, se);
        self.fpu_set_tag_phys(new_top, true);
    }

    /// Mark the current `ST(0)` empty in the tag word and increment `TOP`
    /// (discarding it).
    fn fpu_pop(&mut self) {
        let top = self.fpu_load_top();
        self.fpu_set_tag_phys(top, false);
        let inc = self.builder.ins().iadd_imm_s(top, 1);
        let new_top = self.builder.ins().band_imm_u(inc, 7);
        self.fpu_store_top(new_top);
    }

    /// Load the abridged tag word as an `I32` (bit `i` = physical register `i`
    /// is valid; matches the FXSAVE `ftw` encoding, `1` = valid, `0` = empty).
    fn fpu_load_tag(&mut self) -> Value {
        let ctx = self.fpu_ctx_ptr();
        let off = std::mem::offset_of!(RegisterContext, fpu_tag) as i32;
        let t = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), ctx, off);
        self.builder.ins().uextend(types::I32, t)
    }

    /// Store the abridged tag word (`I32`, low 8 bits significant).
    fn fpu_store_tag(&mut self, tag: Value) {
        let ctx = self.fpu_ctx_ptr();
        let off = std::mem::offset_of!(RegisterContext, fpu_tag) as i32;
        let t16 = self.builder.ins().ireduce(types::I16, tag);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), t16, ctx, off);
    }

    /// Set physical register `phys` (an `I32` value, 0-7) valid or empty in the
    /// tag word.
    fn fpu_set_tag_phys(&mut self, phys: Value, valid: bool) {
        let one = self.builder.ins().iconst(types::I32, 1);
        let bit = self.builder.ins().ishl(one, phys);
        let tag = self.fpu_load_tag();
        let new = if valid {
            self.builder.ins().bor(tag, bit)
        } else {
            self.builder.ins().band_not(tag, bit)
        };
        self.fpu_store_tag(new);
    }

    /// Resolve operand `op_index` to a stack index (`ST(n)` -> `n`).
    fn fpu_st_index(&self, instruction: &Instruction, op_index: u32) -> crate::Result<u32> {
        let reg = self.operand_register(instruction, op_index);
        if reg.is_st() {
            Ok(reg.number() as u32)
        } else {
            Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("fpu-op{}-not-st-{:?}", op_index, reg),
            })
        }
    }

    // --- 80-bit extended conversions ------------------------------------

    /// Convert a raw 80-bit `(mant, se)` value to an `f64`, rounding the
    /// significand to nearest-even (the x87 default; the control-word rounding
    /// mode is not modelled). Handles the full exponent range:
    ///
    /// - `exp80 == 0x7FFF` (inf / NaN) maps to the f64 special exponent with
    ///   the top 52 payload bits preserved (the quiet bit survives).
    /// - Exponents above the f64 range overflow to infinity; the `0x7FE`
    ///   round-up case is produced naturally by the rounding carry.
    /// - Exponents at or below the f64 normal range produce denormals (or
    ///   zero), again with correct rounding; a denormal that rounds up past
    ///   the fraction becomes the smallest normal via the same carry.
    fn f80_to_f64(&mut self, mant: Value, se: Value) -> Value {
        let sign_i32 = self.builder.ins().ushr_imm_u(se, 15);
        let sign_i32 = self.builder.ins().band_imm_u(sign_i32, 1);
        let sign = self.builder.ins().uextend(types::I64, sign_i32);
        let sign_bit = self.builder.ins().ishl_imm_u(sign, 63);

        let exp80 = self.builder.ins().band_imm_u(se, 0x7FFF);
        let exp80_64 = self.builder.ins().uextend(types::I64, exp80);
        // Biased f64 exponent of the integer bit: exp80 - 16383 + 1023.
        let e = self.builder.ins().iadd_imm_s(exp80_64, -15360);

        let is_special = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::Equal, exp80_64, 0x7FFF);
        let is_zero_mant = self.builder.ins().icmp_imm_s(IntCC::Equal, mant, 0);
        let overflow = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, e, 0x7FF);
        let subnormal = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThanOrEqual, e, 0);

        // The 80-bit significand keeps 53 bits after the shift (explicit
        // integer bit at position 52 + 52 fraction bits). Normal values drop
        // 11 bits; a subnormal result drops 11 + (1 - e) so the value lines up
        // against the fixed 2^-1022 scale.
        let c11 = self.builder.ins().iconst(types::I64, 11);
        let shift_sub = {
            let twelve = self.builder.ins().iconst(types::I64, 12);
            self.builder.ins().isub(twelve, e)
        };
        let shift = self.builder.ins().select(subnormal, shift_sub, c11);
        // Shifts of 64+ would be UB-ish in the IR; the value rounds to zero.
        let too_small = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThan, shift, 63);
        let c63 = self.builder.ins().iconst(types::I64, 63);
        let sh = self.builder.ins().select(too_small, c63, shift);

        let keep = self.builder.ins().ushr(mant, sh);
        // Round to nearest, ties to even, on the dropped bits.
        let one = self.builder.ins().iconst(types::I64, 1);
        let mask = {
            let m = self.builder.ins().ishl(one, sh);
            self.builder.ins().iadd_imm_s(m, -1)
        };
        let rem = self.builder.ins().band(mant, mask);
        let half = {
            let shm1 = self.builder.ins().iadd_imm_s(sh, -1);
            self.builder.ins().ishl(one, shm1)
        };
        let gt = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThan, rem, half);
        let eq = self.builder.ins().icmp(IntCC::Equal, rem, half);
        let odd = self.builder.ins().band(keep, one);
        let odd = self.builder.ins().icmp_imm_s(IntCC::NotEqual, odd, 0);
        let tie_up = self.builder.ins().band(eq, odd);
        let round_up = self.builder.ins().bor(gt, tie_up);
        let inc = self.builder.ins().uextend(types::I64, round_up);
        let keep_r = self.builder.ins().iadd(keep, inc);
        let zero64 = self.builder.ins().iconst(types::I64, 0);
        let keep_final = self.builder.ins().select(too_small, zero64, keep_r);

        // combined = (base_exp << 52) + significand. For normals base_exp is
        // e - 1 because the significand still contains the integer bit at
        // position 52 (adding 1 << 52 bumps the field to e). A rounding carry
        // out of the significand increments the exponent field - exactly the
        // IEEE behaviour, including 0x7FE rounding up to infinity and the
        // largest denormal rounding up to the smallest normal.
        let e_minus_1 = self.builder.ins().iadd_imm_s(e, -1);
        let base_exp = self.builder.ins().select(subnormal, zero64, e_minus_1);
        let exp_field = self.builder.ins().ishl_imm_u(base_exp, 52);
        let combined = self.builder.ins().iadd(exp_field, keep_final);

        // Specials: exponent 0x7FF, top 52 significand bits as the payload.
        let inf_bits = self.builder.ins().iconst(types::I64, 0x7FFi64 << 52);
        let frac = self.builder.ins().ushr_imm_u(mant, 11);
        let frac = self.builder.ins().band_imm_u(frac, 0x000F_FFFF_FFFF_FFFF);
        let special_bits = self.builder.ins().bor(inf_bits, frac);

        let magnitude = self.builder.ins().select(overflow, inf_bits, combined);
        let magnitude = self
            .builder
            .ins()
            .select(is_special, special_bits, magnitude);
        // Zero mantissa -> signed zero (also covers the true-zero encoding).
        let magnitude = self.builder.ins().select(is_zero_mant, zero64, magnitude);

        let bits = self.builder.ins().bor(sign_bit, magnitude);
        self.builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), bits)
    }

    /// Convert an `f64` to a raw 80-bit `(mant, se)` value. Exact (`f64` is a
    /// subset of the 80-bit format).
    fn f64_to_f80(&mut self, val: Value) -> (Value, Value) {
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), val);

        let sign = self.builder.ins().ushr_imm_u(bits, 63);
        let sign = self.builder.ins().band_imm_u(sign, 1);
        let exp64 = self.builder.ins().ushr_imm_u(bits, 52);
        let exp64 = self.builder.ins().band_imm_u(exp64, 0x7FF);
        let frac = self.builder.ins().band_imm_u(bits, 0x000F_FFFF_FFFF_FFFF);

        // Normal mantissa: explicit integer bit + fraction shifted up by 11.
        let frac_sh = self.builder.ins().ishl_imm_u(frac, 11);
        let int_bit = self.builder.ins().iconst(types::I64, (1u64 << 63) as i64);
        let mant_normal = self.builder.ins().bor(int_bit, frac_sh);

        let is_exp_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, exp64, 0);
        let is_frac_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, frac, 0);
        let is_zero = self.builder.ins().band(is_exp_zero, is_frac_zero);
        let is_subnormal = self.builder.ins().band_not(is_exp_zero, is_frac_zero);
        let is_special = self.builder.ins().icmp_imm_s(IntCC::Equal, exp64, 0x7FF);

        // Subnormal f64: no implicit integer bit. Normalise the fraction so its
        // top set bit lands at bit 63; the 80-bit format has room to represent
        // it as a normal value. value = frac * 2^-1074, and after shifting left
        // by clz(frac) the integer-bit exponent is 15372 - clz (biased).
        let clz = self.builder.ins().clz(frac);
        let mant_sub = self.builder.ins().ishl(frac, clz);
        let exp80_sub = {
            let base = self.builder.ins().iconst(types::I64, 15372);
            self.builder.ins().isub(base, clz)
        };

        let exp80_normal = self.builder.ins().iadd_imm_s(exp64, 15360);
        let special_exp = self.builder.ins().iconst(types::I64, 0x7FFF);
        let zero_val = self.builder.ins().iconst(types::I64, 0);
        let exp80 = self
            .builder
            .ins()
            .select(is_special, special_exp, exp80_normal);
        let exp80 = self.builder.ins().select(is_subnormal, exp80_sub, exp80);
        let exp80 = self.builder.ins().select(is_zero, zero_val, exp80);
        let mant = self
            .builder
            .ins()
            .select(is_subnormal, mant_sub, mant_normal);
        let mant = self.builder.ins().select(is_zero, zero_val, mant);

        let sign15 = self.builder.ins().ishl_imm_u(sign, 15);
        let se = self.builder.ins().bor(sign15, exp80);
        let se = self.builder.ins().ireduce(types::I32, se);
        (mant, se)
    }

    /// Convert a signed 64-bit integer to a raw 80-bit `(mant, se)` value.
    /// Exact - a 64-bit integer fits in the 64-bit significand.
    fn i64_to_f80(&mut self, v: Value) -> (Value, Value) {
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, v, 0);
        let neg = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, v, 0);
        // Magnitude (|i64::MIN| wraps to 0x8000_0000_0000_0000, the correct
        // unsigned magnitude 2^63).
        let negated = self.builder.ins().ineg(v);
        let mag = self.builder.ins().select(neg, negated, v);

        // Normalise so the most-significant set bit lands at bit 63.
        let lz = self.builder.ins().clz(mag);
        let mant_norm = self.builder.ins().ishl(mag, lz);
        // Unbiased exponent of the integer bit = 63 - lz; biased += 16383.
        let bias = self.builder.ins().iconst(types::I64, 16383 + 63);
        let exp = self.builder.ins().isub(bias, lz);

        let sign_i64 = self.builder.ins().uextend(types::I64, neg);
        let sign15 = self.builder.ins().ishl_imm_u(sign_i64, 15);
        let se = self.builder.ins().bor(sign15, exp);
        let se = self.builder.ins().ireduce(types::I32, se);

        let zero = self.builder.ins().iconst(types::I64, 0);
        let mant = self.builder.ins().select(is_zero, zero, mant_norm);
        let se = {
            let se_zero = self.builder.ins().iconst(types::I32, 0);
            self.builder.ins().select(is_zero, se_zero, se)
        };
        (mant, se)
    }

    /// Convert a raw 80-bit `(mant, se)` value to a signed 64-bit integer.
    /// `truncate` selects round-toward-zero (`FISTTP`); otherwise
    /// round-to-nearest-even (`FIST`/`FISTP`). Out-of-range magnitudes saturate
    /// to the x87 integer indefinite (`0x8000_0000_0000_0000`).
    fn f80_to_i64(&mut self, mant: Value, se: Value, truncate: bool) -> Value {
        let sign_sh = self.builder.ins().ushr_imm_u(se, 15);
        let sign = self.builder.ins().band_imm_u(sign_sh, 1);
        let exp80 = self.builder.ins().band_imm_u(se, 0x7FFF);
        // Unbiased exponent of the integer bit.
        let e = self.builder.ins().iadd_imm_s(exp80, -16383);
        let e = self.builder.ins().sextend(types::I64, e);

        // shift = 63 - e : right-shift applied to `mant` to drop the fraction.
        let c63 = self.builder.ins().iconst(types::I64, 63);
        let shift = self.builder.ins().isub(c63, e);

        // Truncated magnitude for the in-range case (0 <= shift <= 63).
        let shift_clamped = {
            // Cap shift at 63 so the shift is well-defined; the >=64 case is
            // selected away below.
            let sixty3 = self.builder.ins().iconst(types::I64, 63);
            let too_small = self
                .builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, shift, sixty3);
            self.builder.ins().select(too_small, sixty3, shift)
        };
        let trunc_mag = self.builder.ins().ushr(mant, shift_clamped);

        let mag = if truncate {
            trunc_mag
        } else {
            // Round-to-nearest-even using the shifted-out bits.
            // rem = mant & ((1<<shift)-1); half = 1<<(shift-1).
            let one = self.builder.ins().iconst(types::I64, 1);
            let mask = {
                let m = self.builder.ins().ishl(one, shift_clamped);
                self.builder.ins().iadd_imm_s(m, -1)
            };
            let rem = self.builder.ins().band(mant, mask);
            let half = {
                let sh = self.builder.ins().iadd_imm_s(shift_clamped, -1);
                self.builder.ins().ishl(one, sh)
            };
            let gt = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, rem, half);
            let eq = self.builder.ins().icmp(IntCC::Equal, rem, half);
            let odd = self.builder.ins().band_imm_u(trunc_mag, 1);
            let odd = self.builder.ins().icmp_imm_s(IntCC::NotEqual, odd, 0);
            let tie_up = self.builder.ins().band(eq, odd);
            let round_up = self.builder.ins().bor(gt, tie_up);
            let inc = self.builder.ins().uextend(types::I64, round_up);
            // No shift (shift==0) -> already integral, don't round.
            let no_frac = self
                .builder
                .ins()
                .icmp_imm_s(IntCC::Equal, shift_clamped, 0);
            let inc = {
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().select(no_frac, zero, inc)
            };
            self.builder.ins().iadd(trunc_mag, inc)
        };

        // Apply the sign.
        let neg_mag = self.builder.ins().ineg(mag);
        let is_neg = self.builder.ins().icmp_imm_s(IntCC::NotEqual, sign, 0);
        let signed = self.builder.ins().select(is_neg, neg_mag, mag);

        // Out-of-range: value < 1 (shift >= 64) -> 0; magnitude too large
        // (shift < 0, i.e. exponent >= 63) -> integer indefinite.
        let value_lt_one =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, shift, 64);
        let too_large = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, shift, 0);
        let zero = self.builder.ins().iconst(types::I64, 0);
        let indef = self.builder.ins().iconst(types::I64, i64::MIN);
        let r = self.builder.ins().select(value_lt_one, zero, signed);
        self.builder.ins().select(too_large, indef, r)
    }

    // --- memory operand helpers -----------------------------------------

    /// Load a floating-point memory operand (`m32`/`m64`/`m80`) as a raw 80-bit
    /// `(mant, se)` pair. All widths convert exactly (each is a subset of the
    /// 80-bit format).
    fn fpu_load_mem_f80(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<(Value, Value)> {
        let ptr = self.get_mem_ptr(instruction, op_index)?;
        match instruction.memory_size() {
            MemorySize::Float32 => {
                let f32v = self
                    .builder
                    .ins()
                    .load(types::F32, MemFlagsData::new(), ptr, 0);
                let f64v = self.builder.ins().fpromote(types::F64, f32v);
                Ok(self.f64_to_f80(f64v))
            }
            MemorySize::Float64 => {
                let f64v = self
                    .builder
                    .ins()
                    .load(types::F64, MemFlagsData::new(), ptr, 0);
                Ok(self.f64_to_f80(f64v))
            }
            MemorySize::Float80 => {
                let mant = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                let se16 = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlagsData::new(), ptr, 8);
                let se = self.builder.ins().uextend(types::I32, se16);
                Ok((mant, se))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("fpu-load-mem-{:?}", other),
            }),
        }
    }

    /// Load an integer memory operand (`m16`/`m32`/`m64`) as a raw 80-bit value.
    fn fpu_load_mem_int_f80(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<(Value, Value)> {
        let ptr = self.get_mem_ptr(instruction, op_index)?;
        let int64 = match instruction.memory_size() {
            MemorySize::Int16 => {
                let v = self
                    .builder
                    .ins()
                    .load(types::I16, MemFlagsData::new(), ptr, 0);
                self.builder.ins().sextend(types::I64, v)
            }
            MemorySize::Int32 => {
                let v = self
                    .builder
                    .ins()
                    .load(types::I32, MemFlagsData::new(), ptr, 0);
                self.builder.ins().sextend(types::I64, v)
            }
            MemorySize::Int64 => self
                .builder
                .ins()
                .load(types::I64, MemFlagsData::new(), ptr, 0),
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("fpu-load-int-{:?}", other),
                });
            }
        };
        Ok(self.i64_to_f80(int64))
    }

    /// FLD - push a floating-point value onto the x87 stack.
    ///
    /// Covers `FLD m32/m64/m80` (`flds`/`fldl`/`fldt`) and `FLD ST(i)`, all
    /// exact in 80-bit precision.
    pub(super) fn handle_fld(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (mant, se) = match self.operand_kind(instruction, 0) {
            OpKind::Memory => self.fpu_load_mem_f80(instruction, 0)?,
            OpKind::Register => {
                let i = self.fpu_st_index(instruction, 0)?;
                self.fpu_read_st_raw(i)
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("fld-{:?}", other),
                });
            }
        };
        self.fpu_push_raw(mant, se);
        Ok(())
    }

    /// FILD - convert an integer in memory to 80-bit and push it (exact).
    pub(super) fn handle_fild(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (mant, se) = self.fpu_load_mem_int_f80(instruction, 0)?;
        self.fpu_push_raw(mant, se);
        Ok(())
    }

    /// Push an implicit constant (`FLDZ`, `FLD1`, `FLDPI`, ...). The constant is
    /// converted to its exact 80-bit encoding at lift time.
    pub(super) fn handle_fld_const(&mut self, value: f64) -> crate::Result<()> {
        let (mant_bits, se_bits) = crate::arch::x86_64::context::f64_bits_to_f80(value.to_bits());
        let mant = self.builder.ins().iconst(types::I64, mant_bits as i64);
        let se = self.builder.ins().iconst(types::I32, se_bits as i64);
        self.fpu_push_raw(mant, se);
        Ok(())
    }

    /// FST / FSTP - store `ST(0)`; `pop` selects the popping form.
    ///
    /// Covers `FST(P) m32/m64`, `FSTP m80` (`fstpt`) and `FST(P) ST(i)`.
    pub(super) fn handle_fst(&mut self, instruction: &Instruction, pop: bool) -> crate::Result<()> {
        let (mant, se) = self.fpu_read_st_raw(0);
        match self.operand_kind(instruction, 0) {
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                match instruction.memory_size() {
                    MemorySize::Float32 => {
                        // Narrow via f64 (double-rounding is possible only for
                        // values with >53 significant bits; refined with the
                        // softfloat kernels).
                        let f64v = self.f80_to_f64(mant, se);
                        let f32v = self.builder.ins().fdemote(types::F32, f64v);
                        self.builder.ins().store(MemFlagsData::new(), f32v, ptr, 0);
                    }
                    MemorySize::Float64 => {
                        let f64v = self.f80_to_f64(mant, se);
                        self.builder.ins().store(MemFlagsData::new(), f64v, ptr, 0);
                    }
                    MemorySize::Float80 => {
                        let se16 = self.builder.ins().ireduce(types::I16, se);
                        self.builder.ins().store(MemFlagsData::new(), mant, ptr, 0);
                        self.builder.ins().store(MemFlagsData::new(), se16, ptr, 8);
                    }
                    other => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("fst-mem-{:?}", other),
                        });
                    }
                }
            }
            OpKind::Register => {
                let i = self.fpu_st_index(instruction, 0)?;
                self.fpu_write_st_raw(i, mant, se);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("fst-{:?}", other),
                });
            }
        }
        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// FIST / FISTP / FISTTP - store `ST(0)` as an integer.
    ///
    /// `pop` selects the popping form; `truncate` selects `FISTTP` (round
    /// toward zero) versus round-to-nearest for `FIST`/`FISTP`.
    pub(super) fn handle_fist(
        &mut self,
        instruction: &Instruction,
        pop: bool,
        truncate: bool,
    ) -> crate::Result<()> {
        let (mant, se) = self.fpu_read_st_raw(0);
        let int64 = self.f80_to_i64(mant, se, truncate);

        let ptr = self.get_mem_ptr(instruction, 0)?;
        match instruction.memory_size() {
            MemorySize::Int16 => {
                let v = self.builder.ins().ireduce(types::I16, int64);
                self.builder.ins().store(MemFlagsData::new(), v, ptr, 0);
            }
            MemorySize::Int32 => {
                let v = self.builder.ins().ireduce(types::I32, int64);
                self.builder.ins().store(MemFlagsData::new(), v, ptr, 0);
            }
            MemorySize::Int64 => {
                self.builder.ins().store(MemFlagsData::new(), int64, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("fist-mem-{:?}", other),
                });
            }
        }
        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// FXCH - exchange `ST(0)` with `ST(i)` (default `ST(1)`).
    ///
    /// iced decodes `fxch st(i)` as two operands (`op0 = ST0`, `op1 = ST(i)`),
    /// so the exchange partner is the *last* operand.
    pub(super) fn handle_fxch(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let i = match instruction.op_count() {
            0 => 1,
            n => self.fpu_st_index(instruction, n - 1)?,
        };
        let (a_mant, a_se) = self.fpu_read_st_raw(0);
        let (b_mant, b_se) = self.fpu_read_st_raw(i);
        self.fpu_write_st_raw(0, b_mant, b_se);
        self.fpu_write_st_raw(i, a_mant, a_se);
        Ok(())
    }

    /// Resolve the `(dst_index, dst, src)` triple for a binary x87 arithmetic
    /// instruction across its register/register, register/memory and implicit
    /// (no-operand `FADDP`-style) forms. Operands are raw 80-bit `(mant, se)`
    /// pairs.
    fn fpu_binop_operands_raw(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<(u32, F80, F80)> {
        match instruction.op_count() {
            0 => {
                // Implicit popping form, e.g. `faddp` == `faddp st(1), st(0)`.
                let dst = self.fpu_read_st_raw(1);
                let src = self.fpu_read_st_raw(0);
                Ok((1, dst, src))
            }
            1 => {
                // Single operand: `fadd m32/m64` (dst = ST0) or `fadd st(i)`
                // (== `fadd st(0), st(i)`).
                match self.operand_kind(instruction, 0) {
                    OpKind::Memory => {
                        let dst = self.fpu_read_st_raw(0);
                        let src = self.fpu_load_mem_f80(instruction, 0)?;
                        Ok((0, dst, src))
                    }
                    OpKind::Register => {
                        let i = self.fpu_st_index(instruction, 0)?;
                        let dst = self.fpu_read_st_raw(0);
                        let src = self.fpu_read_st_raw(i);
                        Ok((0, dst, src))
                    }
                    other => Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: format!("fpu-binop-{:?}", other),
                    }),
                }
            }
            _ => {
                let dst_idx = self.fpu_st_index(instruction, 0)?;
                let src_idx = self.fpu_st_index(instruction, 1)?;
                let dst = self.fpu_read_st_raw(dst_idx);
                let src = self.fpu_read_st_raw(src_idx);
                Ok((dst_idx, dst, src))
            }
        }
    }

    /// Emit one softfloat kernel over raw 80-bit operands, as a call to the
    /// shared helper function or - when outlining is off - inline at this site.
    ///
    /// Both paths run the identical `emit_f80_*` builder; the helper body is
    /// just that builder wrapped in its own function (see [`F80Kernel`]).
    fn f80_kernel(&mut self, kernel: F80Kernel, args: &[Value]) -> Vec<Value> {
        if !self.outline_helpers() {
            let b = &mut self.builder;
            let (m, se) = match kernel {
                F80Kernel::Add => softfloat::emit_f80_add(b, args[0], args[1], args[2], args[3]),
                F80Kernel::Sub => softfloat::emit_f80_sub(b, args[0], args[1], args[2], args[3]),
                F80Kernel::Mul => softfloat::emit_f80_mul(b, args[0], args[1], args[2], args[3]),
                F80Kernel::Div => softfloat::emit_f80_div(b, args[0], args[1], args[2], args[3]),
                F80Kernel::Sqrt => softfloat::emit_f80_sqrt(b, args[0], args[1]),
                F80Kernel::Cmp => {
                    let (unord, eq, lt) =
                        softfloat::emit_f80_cmp(b, args[0], args[1], args[2], args[3]);
                    return vec![unord, eq, lt];
                }
            };
            return vec![m, se];
        }

        // The helper shares the lifted function's calling convention, so the
        // call needs no ABI bridging.
        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self.builder.ins().call(func_ref, args);
        self.builder.inst_results(call).to_vec()
    }

    /// [`f80_kernel`](Self::f80_kernel) for the kernels returning one 80-bit
    /// value.
    fn f80_value_kernel(&mut self, kernel: F80Kernel, args: &[Value]) -> F80 {
        let r = self.f80_kernel(kernel, args);
        (r[0], r[1])
    }

    /// FADD/FSUB/FSUBR/FMUL/FDIV/FDIVR and their `...P` popping variants, all via
    /// the bit-exact 80-bit software kernels.
    pub(super) fn handle_farith(
        &mut self,
        instruction: &Instruction,
        op: FpuOp,
        pop: bool,
    ) -> crate::Result<()> {
        let (dst_idx, (dm, dse), (sm, sse)) = self.fpu_binop_operands_raw(instruction)?;
        // The reverse forms swap the operands rather than needing their own
        // kernel - `a - b` and `b - a` are the same code with the arguments
        // exchanged, so both share one helper.
        let (kernel, args) = match op {
            FpuOp::Add => (F80Kernel::Add, [dm, dse, sm, sse]),
            FpuOp::Sub => (F80Kernel::Sub, [dm, dse, sm, sse]),
            FpuOp::Subr => (F80Kernel::Sub, [sm, sse, dm, dse]),
            FpuOp::Mul => (F80Kernel::Mul, [dm, dse, sm, sse]),
            FpuOp::Div => (F80Kernel::Div, [dm, dse, sm, sse]),
            FpuOp::Divr => (F80Kernel::Div, [sm, sse, dm, dse]),
        };
        let (rm, rse) = self.f80_value_kernel(kernel, &args);
        self.fpu_write_st_raw(dst_idx, rm, rse);
        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// FABS - clear the sign bit of `ST(0)` (exact).
    pub(super) fn handle_fabs(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let (mant, se) = self.fpu_read_st_raw(0);
        let se = self.builder.ins().band_imm_u(se, 0x7FFF);
        self.fpu_write_st_raw(0, mant, se);
        Ok(())
    }

    /// FCHS - flip the sign bit of `ST(0)` (exact).
    pub(super) fn handle_fchs(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let (mant, se) = self.fpu_read_st_raw(0);
        let se = self.builder.ins().bxor_imm_u(se, 0x8000);
        self.fpu_write_st_raw(0, mant, se);
        Ok(())
    }

    /// FSQRT - square root of `ST(0)` (bit-exact 80-bit).
    pub(super) fn handle_fsqrt(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let (m, se) = self.fpu_read_st_raw(0);
        let (rm, rse) = self.f80_value_kernel(F80Kernel::Sqrt, &[m, se]);
        self.fpu_write_st_raw(0, rm, rse);
        Ok(())
    }

    /// Compute the three ordered/equal/unordered predicates for `a <=> b`,
    /// where each operand is a raw 80-bit `(mant, se)` pair. Exact (no `f64`
    /// rounding).
    fn fpu_cmp_predicates(
        &mut self,
        a: (Value, Value),
        b: (Value, Value),
    ) -> (Value, Value, Value) {
        let r = self.f80_kernel(F80Kernel::Cmp, &[a.0, a.1, b.0, b.1]);
        (r[0], r[1], r[2])
    }

    /// Resolve the raw 80-bit source (and its stack index, if any) of a compare.
    /// `FCOM`/`FUCOM` may take an `ST(i)` or a memory operand; the `...I`
    /// forms always take two `ST` operands (`ST0`, `ST(i)`).
    fn fpu_cmp_src(&mut self, instruction: &Instruction) -> crate::Result<(Value, Value)> {
        if instruction.op_count() == 0 {
            // `fcom`/`fucom` with no operand compare against ST(1).
            return Ok(self.fpu_read_st_raw(1));
        }
        // The source is the last operand (`FCOMI ST0, ST(i)` -> op1;
        // `FCOM ST(i)`/`FCOM m32` -> op0).
        let op_index = instruction.op_count() - 1;
        match self.operand_kind(instruction, op_index) {
            OpKind::Memory => self.fpu_load_mem_f80(instruction, op_index),
            OpKind::Register => {
                let i = self.fpu_st_index(instruction, op_index)?;
                Ok(self.fpu_read_st_raw(i))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("fpu-cmp-src-{:?}", other),
            }),
        }
    }

    /// FCOM/FCOMP/FUCOM/FUCOMP/FTST - compare `ST(0)` and set the status-word
    /// condition codes (C0/C2/C3). `pop` selects the popping variant.
    ///
    /// (Both the ordered and unordered forms are lifted identically here: the
    /// only architectural difference is which operand classes raise an invalid
    /// exception, and this lifter does not model x87 exception flags.)
    pub(super) fn handle_fcom(
        &mut self,
        instruction: &Instruction,
        pop: bool,
    ) -> crate::Result<()> {
        let a = self.fpu_read_st_raw(0);
        let b = self.fpu_cmp_src(instruction)?;
        let (unord, eq, lt) = self.fpu_cmp_predicates(a, b);

        // C3 = ZF-like (equal or unordered), C2 = unordered, C0 = CF-like
        // (less-than or unordered); C1 is cleared.
        let c3 = self.builder.ins().bor(eq, unord);
        let c0 = self.builder.ins().bor(lt, unord);
        let c3 = self.builder.ins().uextend(types::I32, c3);
        let c2 = self.builder.ins().uextend(types::I32, unord);
        let c0 = self.builder.ins().uextend(types::I32, c0);
        let c3 = self.builder.ins().ishl_imm_u(c3, 14);
        let c2 = self.builder.ins().ishl_imm_u(c2, 10);
        let c0 = self.builder.ins().ishl_imm_u(c0, 8);
        let cc = self.builder.ins().bor(c3, c2);
        let cc = self.builder.ins().bor(cc, c0);

        self.fpu_store_cc(cc);
        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// FCOMI/FCOMIP/FUCOMI/FUCOMIP - compare `ST(0)` with `ST(i)` and set the
    /// integer flags (ZF/PF/CF); OF/SF/AF are cleared. `pop` selects the
    /// popping variant.
    pub(super) fn handle_fcomi(
        &mut self,
        instruction: &Instruction,
        pop: bool,
    ) -> crate::Result<()> {
        let a = self.fpu_read_st_raw(0);
        let b = self.fpu_cmp_src(instruction)?;
        let (unord, eq, lt) = self.fpu_cmp_predicates(a, b);

        let zf = self.builder.ins().bor(eq, unord);
        let cf = self.builder.ins().bor(lt, unord);
        self.set_flag(X86Flag::Zero, zf)?;
        self.set_flag(X86Flag::Parity, unord)?;
        self.set_flag(X86Flag::Carry, cf)?;
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Overflow, zero)?;
        self.set_flag(X86Flag::Sign, zero)?;
        self.set_flag(X86Flag::Auxiliary, zero)?;

        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// FTST - compare `ST(0)` with `+0.0`, setting the status-word codes.
    pub(super) fn handle_ftst(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let a = self.fpu_read_st_raw(0);
        let zero_mant = self.builder.ins().iconst(types::I64, 0);
        let zero_se = self.builder.ins().iconst(types::I32, 0);
        let (unord, eq, lt) = self.fpu_cmp_predicates(a, (zero_mant, zero_se));
        let c3 = self.builder.ins().bor(eq, unord);
        let c0 = self.builder.ins().bor(lt, unord);
        let c3 = self.builder.ins().uextend(types::I32, c3);
        let c2 = self.builder.ins().uextend(types::I32, unord);
        let c0 = self.builder.ins().uextend(types::I32, c0);
        let c3 = self.builder.ins().ishl_imm_u(c3, 14);
        let c2 = self.builder.ins().ishl_imm_u(c2, 10);
        let c0 = self.builder.ins().ishl_imm_u(c0, 8);
        let cc = self.builder.ins().bor(c3, c2);
        let cc = self.builder.ins().bor(cc, c0);
        self.fpu_store_cc(cc);
        Ok(())
    }

    /// Merge freshly-computed condition-code bits (`cc`, an `I32` already in
    /// status-word bit positions) into the stored status word.
    fn fpu_store_cc(&mut self, cc: Value) {
        let ctx = self.fpu_ctx_ptr();
        let off = std::mem::offset_of!(RegisterContext, fpu_sw) as i32;
        let sw = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), ctx, off);
        let sw = self.builder.ins().uextend(types::I32, sw);
        let cleared = self.builder.ins().band_imm_u(sw, !SW_CC_MASK);
        let merged = self.builder.ins().bor(cleared, cc);
        let merged16 = self.builder.ins().ireduce(types::I16, merged);
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), merged16, ctx, off);
    }

    // --- conditional moves ----------------------------------------------

    /// FCMOVcc - conditionally copy `ST(i)` into `ST(0)` based on the integer
    /// flags. `cond` is a non-zero `I8` when the move should occur.
    fn handle_fcmov_cond(&mut self, instruction: &Instruction, cond: Value) -> crate::Result<()> {
        // Source is `ST(i)` - op1 for the two-operand `FCMOVcc ST0, ST(i)`.
        let op_index = instruction.op_count().saturating_sub(1);
        let i = self.fpu_st_index(instruction, op_index)?;
        let (cur_mant, cur_se) = self.fpu_read_st_raw(0);
        let (new_mant, new_se) = self.fpu_read_st_raw(i);
        let sel_mant = self.builder.ins().select(cond, new_mant, cur_mant);
        let sel_se = self.builder.ins().select(cond, new_se, cur_se);
        self.fpu_write_st_raw(0, sel_mant, sel_se);
        Ok(())
    }

    fn fpu_flag(&mut self, flag: X86Flag) -> Value {
        self.builder.use_var(self.state.flags[&flag])
    }

    /// FCMOVB - move if CF=1 (below).
    pub(super) fn handle_fcmovb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.fpu_flag(X86Flag::Carry);
        self.handle_fcmov_cond(instruction, cf)
    }

    /// FCMOVNB - move if CF=0 (not below).
    pub(super) fn handle_fcmovnb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.fpu_flag(X86Flag::Carry);
        let cond = self.builder.ins().icmp_imm_s(IntCC::Equal, cf, 0);
        self.handle_fcmov_cond(instruction, cond)
    }

    /// FCMOVE - move if ZF=1 (equal).
    pub(super) fn handle_fcmove(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.fpu_flag(X86Flag::Zero);
        self.handle_fcmov_cond(instruction, zf)
    }

    /// FCMOVNE - move if ZF=0 (not equal).
    pub(super) fn handle_fcmovne(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self.fpu_flag(X86Flag::Zero);
        let cond = self.builder.ins().icmp_imm_s(IntCC::Equal, zf, 0);
        self.handle_fcmov_cond(instruction, cond)
    }

    /// FCMOVBE - move if CF=1 or ZF=1 (below or equal).
    pub(super) fn handle_fcmovbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.fpu_flag(X86Flag::Carry);
        let zf = self.fpu_flag(X86Flag::Zero);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_fcmov_cond(instruction, cond)
    }

    /// FCMOVNBE - move if CF=0 and ZF=0 (above).
    pub(super) fn handle_fcmovnbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self.fpu_flag(X86Flag::Carry);
        let zf = self.fpu_flag(X86Flag::Zero);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        let cond = self.builder.ins().icmp_imm_s(IntCC::Equal, cf_or_zf, 0);
        self.handle_fcmov_cond(instruction, cond)
    }

    /// FCMOVU - move if PF=1 (unordered).
    pub(super) fn handle_fcmovu(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self.fpu_flag(X86Flag::Parity);
        self.handle_fcmov_cond(instruction, pf)
    }

    /// FCMOVNU - move if PF=0 (not unordered).
    pub(super) fn handle_fcmovnu(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self.fpu_flag(X86Flag::Parity);
        let cond = self.builder.ins().icmp_imm_s(IntCC::Equal, pf, 0);
        self.handle_fcmov_cond(instruction, cond)
    }

    // --- control / misc -------------------------------------------------

    /// FNSTSW - store the status word (with the live `TOP` field) to `AX` or a
    /// 16-bit memory operand.
    pub(super) fn handle_fnstsw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let ctx = self.fpu_ctx_ptr();
        let sw_off = std::mem::offset_of!(RegisterContext, fpu_sw) as i32;
        let sw = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), ctx, sw_off);
        let sw = self.builder.ins().uextend(types::I32, sw);
        // Insert TOP into bits 11-13.
        let sw = self.builder.ins().band_imm_u(sw, !0x3800);
        let top = self.fpu_load_top();
        let top = self.builder.ins().band_imm_u(top, 7);
        let top_field = self.builder.ins().ishl_imm_u(top, 11);
        let sw = self.builder.ins().bor(sw, top_field);
        let sw16 = self.builder.ins().ireduce(types::I16, sw);

        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, 0);
                self.set_reg_val(reg, sw16)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), sw16, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("fnstsw-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// FNSTCW - store the control word to a 16-bit memory operand.
    pub(super) fn handle_fnstcw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let ctx = self.fpu_ctx_ptr();
        let cw_off = std::mem::offset_of!(RegisterContext, fpu_cw) as i32;
        let cw = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::trusted(), ctx, cw_off);
        let ptr = self.get_mem_ptr(instruction, 0)?;
        self.builder.ins().store(MemFlagsData::new(), cw, ptr, 0);
        Ok(())
    }

    /// FLDCW - load the control word from a 16-bit memory operand.
    pub(super) fn handle_fldcw(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let ptr = self.get_mem_ptr(instruction, 0)?;
        let cw = self
            .builder
            .ins()
            .load(types::I16, MemFlagsData::new(), ptr, 0);
        let ctx = self.fpu_ctx_ptr();
        let cw_off = std::mem::offset_of!(RegisterContext, fpu_cw) as i32;
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), cw, ctx, cw_off);
        Ok(())
    }

    /// FINCSTP - increment `TOP` without touching the data registers.
    pub(super) fn handle_fincstp(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let top = self.fpu_load_top();
        let inc = self.builder.ins().iadd_imm_s(top, 1);
        let new_top = self.builder.ins().band_imm_u(inc, 7);
        self.fpu_store_top(new_top);
        Ok(())
    }

    /// FDECSTP - decrement `TOP` without touching the data registers.
    pub(super) fn handle_fdecstp(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let top = self.fpu_load_top();
        let dec = self.builder.ins().iadd_imm_s(top, -1);
        let new_top = self.builder.ins().band_imm_u(dec, 7);
        self.fpu_store_top(new_top);
        Ok(())
    }

    /// FFREE - mark `ST(i)` empty in the tag word without changing `TOP` or the
    /// register data.
    pub(super) fn handle_ffree(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let i = self.fpu_st_index(instruction, 0)?;
        let top = self.fpu_load_top();
        let raw = self.builder.ins().iadd_imm_s(top, i as i64);
        let phys = self.builder.ins().band_imm_u(raw, 7);
        self.fpu_set_tag_phys(phys, false);
        Ok(())
    }

    /// FNOP / FWAIT - no observable effect in this FPU model.
    pub(super) fn handle_fpu_nop(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }
}
