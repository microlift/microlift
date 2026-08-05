//! SSE / SSE2 / XMM instruction semantics.
//!
//! XMM registers have two interchangeable backing representations,
//! selected by [`crate::lifter::LifterConfig::use_vector_xmm`]:
//!  - scalar (default): two `I64` SSA variables per register (lo + hi).
//!    Portable to AOT targets without vector lowering (e.g. RISC-V64).
//!  - vector: a single native `I64X2` SSA variable per register, which
//!    lowers directly to host XMM registers on x86-64.
//!
//! The scalar expansion is the authority on semantics and always works.
//! When the vector representation is enabled, most handlers take a native
//! fast path instead - see [`sse_vec`](super::sse_vec) for the shared
//! plumbing and `tests/sse_vec_equiv_tests.rs` for the equivalence proof.
//! Handlers with no fast path (and every handler in scalar mode) reach
//! lo/hi lane access via [`crate::lifter::Lifter::get_xmm_lo`] /
//! [`crate::lifter::Lifter::get_xmm_hi`], which transparently handle both
//! representations.

use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::xmm_register_index;
use crate::arch::x86_64::semantics::sse_vec::{B, PackedFpOp};
use crate::lifter::{Lifter, X86Flag};
use cranelift::codegen::ir::immediates::{Ieee32, Ieee64};
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

/// x86's single-precision *floating indefinite*: the QNaN an SSE instruction
/// produces when it generates a NaN of its own (as opposed to propagating an
/// operand's). Note the set sign bit - that is what distinguishes it from the
/// AArch64 default NaN, and why it cannot be left to the host FPU.
pub(super) const F32_INDEFINITE: Ieee32 = Ieee32::with_bits(0xFFC0_0000);

/// The double-precision [`F32_INDEFINITE`].
pub(super) const F64_INDEFINITE: Ieee64 = Ieee64::with_bits(0xFFF8_0000_0000_0000);

/// Square root of one `F32` or `F64` with x86's NaN behaviour.
///
/// For a negative operand the result is a NaN the instruction *generates*
/// rather than propagates, and the two ISAs disagree on its sign: x86 yields
/// the floating indefinite (see [`F32_INDEFINITE`]) while the AArch64 default
/// NaN is positive. A bare Cranelift `sqrt` lets the *host* FPU pick, so lifted
/// code produces the wrong bits on a non-x86 host; the indefinite is
/// materialised explicitly instead.
///
/// Only the strict `x < 0` case is patched, which is exactly the input set that
/// generates a NaN: `-0.0` still gives `-0.0`, and a NaN operand falls through
/// to `sqrt`, which propagates it - quieting a signalling NaN - with its sign
/// and payload intact, as x86 does.
///
/// A free function rather than a [`Lifter`] method so it can be passed directly
/// as the `op` closure of the packed lane plumbing, which is handed only the
/// builder.
pub(super) fn fp_sqrt_x86(b: &mut FunctionBuilder, x: Value) -> Value {
    let r = b.ins().sqrt(x);
    let (zero, indefinite) = if b.func.dfg.value_type(x) == types::F32 {
        (b.ins().f32const(0.0f32), b.ins().f32const(F32_INDEFINITE))
    } else {
        (b.ins().f64const(0.0f64), b.ins().f64const(F64_INDEFINITE))
    };
    let negative = b.ins().fcmp(FloatCC::LessThan, x, zero);
    b.ins().select(negative, indefinite, r)
}

/// One `F32`/`F64` arithmetic lane with x86's *generated*-NaN bits.
///
/// The companion of [`fp_sqrt_x86`] for the binary operations: `inf + -inf`,
/// `inf - inf`, `0 * inf`, `0 / 0` and `inf / inf` are *invalid operations*,
/// and the NaN they produce is one the instruction generates rather than
/// propagates - so it must be x86's sign-set floating indefinite, not whatever
/// default NaN the host FPU picks (AArch64's is positive).
///
/// The condition is stated once for all four operations rather than enumerated
/// per operation: a NaN result that neither operand supplied *is* the invalid
/// case, and it is exactly the set above. When an operand is already a NaN the
/// host propagates it correctly - quieted, sign and payload intact, as x86
/// does - so `Ordered(x, y)` leaves that path untouched.
pub(super) fn fp_arith_x86(
    b: &mut FunctionBuilder,
    x: Value,
    y: Value,
    op: impl FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
) -> Value {
    let r = op(b, x, y);
    let indefinite = if b.func.dfg.value_type(r) == types::F32 {
        b.ins().f32const(F32_INDEFINITE)
    } else {
        b.ins().f64const(F64_INDEFINITE)
    };
    let generated = b.ins().fcmp(FloatCC::Unordered, r, r);
    let operands_ok = b.ins().fcmp(FloatCC::Ordered, x, y);
    let invalid = b.ins().band(generated, operands_ok);
    b.ins().select(invalid, indefinite, r)
}

/// [`fp_arith_x86`] over `FADD` - the lane closure the packed/scalar plumbing
/// expects.
pub(super) fn fadd_x86(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    fp_arith_x86(b, x, y, |b, x, y| b.ins().fadd(x, y))
}

/// [`fp_arith_x86`] over `FSUB`.
pub(super) fn fsub_x86(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    fp_arith_x86(b, x, y, |b, x, y| b.ins().fsub(x, y))
}

/// [`fp_arith_x86`] over `FMUL`.
pub(super) fn fmul_x86(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    fp_arith_x86(b, x, y, |b, x, y| b.ins().fmul(x, y))
}

/// [`fp_arith_x86`] over `FDIV`.
pub(super) fn fdiv_x86(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    fp_arith_x86(b, x, y, |b, x, y| b.ins().fdiv(x, y))
}

/// The rounding a `ROUND*` instruction's `imm8` selects.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum RoundMode {
    /// Ties to even - the IEEE default, and x86's reset MXCSR mode.
    Nearest,
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Toward zero.
    Trunc,
}

/// Decode `imm8[2:0]` of a `ROUND*` instruction.
///
/// Bit 2 means "ignore bits[1:0] and use MXCSR's current rounding mode". This
/// recompiler does not model MXCSR and nothing it lifts writes one - `LDMXCSR`
/// is not implemented - so the mode is MXCSR's reset value, round-to-nearest-even,
/// and that is what bit 2 resolves to. A guest that changed its rounding mode and
/// then rounded would differ; it is the same standing gap `setjmp`'s missing
/// `MXCSR` save is, recorded in the same place rather than papered over.
///
/// Bit 3 suppresses the precision exception, which is unobservable here: no
/// floating-point exception is raised in the first place.
pub(super) fn round_mode(imm: u8) -> RoundMode {
    if imm & 0b100 != 0 {
        return RoundMode::Nearest;
    }
    match imm & 0b11 {
        0b00 => RoundMode::Nearest,
        0b01 => RoundMode::Floor,
        0b10 => RoundMode::Ceil,
        _ => RoundMode::Trunc,
    }
}

/// Apply `mode` to one float value.
///
/// A free function for the reason [`fp_sqrt_x86`] is: the packed lane plumbing
/// hands its `op` closure only the builder.
pub(super) fn apply_round(b: &mut FunctionBuilder, mode: RoundMode, x: Value) -> Value {
    match mode {
        RoundMode::Nearest => b.ins().nearest(x),
        RoundMode::Floor => b.ins().floor(x),
        RoundMode::Ceil => b.ins().ceil(x),
        RoundMode::Trunc => b.ins().trunc(x),
    }
}

/// Lane width of a packed-integer SSE operation - the `b`/`w`/`d`/`q` suffix
/// of mnemonics like `PADDB` / `PADDW` / `PADDD` / `PADDQ`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum PackedWidth {
    Byte,
    Word,
    Dword,
    Qword,
}

impl PackedWidth {
    /// Lane width in bits.
    pub(super) fn bits(self) -> u32 {
        match self {
            PackedWidth::Byte => 8,
            PackedWidth::Word => 16,
            PackedWidth::Dword => 32,
            PackedWidth::Qword => 64,
        }
    }

    /// The 128-bit Cranelift vector type with these lanes.
    pub(super) fn vector_type(self) -> Type {
        match self {
            PackedWidth::Byte => types::I8X16,
            PackedWidth::Word => types::I16X8,
            PackedWidth::Dword => types::I32X4,
            PackedWidth::Qword => types::I64X2,
        }
    }

    /// A 64-bit mask with the most significant bit of every lane set
    /// (`0x8080..` for byte lanes, `0x8000_8000..` for word lanes, ...).
    ///
    /// This is the "carry barrier" used by the SWAR add/sub in scalar mode.
    fn lane_sign_mask(self) -> i64 {
        let bits = self.bits();
        let mut mask = 0u64;
        let mut shift = 0;
        while shift < 64 {
            mask |= 1u64 << (shift + bits - 1);
            shift += bits;
        }
        mask as i64
    }

    /// A 64-bit mask with all bits of the lowest lane set.
    fn lane_mask(self) -> i64 {
        match self.bits() {
            64 => -1,
            bits => ((1u64 << bits) - 1) as i64,
        }
    }
}

/// Which packed-integer shift a `PSRL*` / `PSLL*` / `PSRA*` form performs.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum PackedShiftOp {
    /// `PSRLW`/`PSRLD`/`PSRLQ` - per-lane logical (zero-filling) right shift.
    Srl,
    /// `PSLLW`/`PSLLD`/`PSLLQ` - per-lane left shift.
    Sll,
    /// `PSRAW`/`PSRAD` - per-lane arithmetic (sign-filling) right shift.
    Sra,
}

impl PackedShiftOp {
    /// True for the two zero-filling shifts, which produce all-zero lanes for
    /// an out-of-range count (`PSRA*` saturates to the sign instead).
    fn zeroes_on_overshift(self) -> bool {
        self != PackedShiftOp::Sra
    }

    /// The mnemonic this op/width pair came from, for error messages.
    fn mnemonic(self, width: PackedWidth) -> &'static str {
        match (self, width) {
            (PackedShiftOp::Srl, PackedWidth::Word) => "psrlw",
            (PackedShiftOp::Srl, PackedWidth::Dword) => "psrld",
            (PackedShiftOp::Srl, PackedWidth::Qword) => "psrlq",
            (PackedShiftOp::Sll, PackedWidth::Word) => "psllw",
            (PackedShiftOp::Sll, PackedWidth::Dword) => "pslld",
            (PackedShiftOp::Sll, PackedWidth::Qword) => "psllq",
            (PackedShiftOp::Sra, PackedWidth::Word) => "psraw",
            (PackedShiftOp::Sra, PackedWidth::Dword) => "psrad",
            (PackedShiftOp::Sra, PackedWidth::Qword) => "psraq",
            (_, PackedWidth::Byte) => "packed-byte-shift",
        }
    }
}

/// The shift count of a packed shift: a decoded `imm8`, or a value read from
/// the low 64 bits of an `xmm/m128` source operand.
#[derive(Copy, Clone)]
enum ShiftCount {
    Imm(u64),
    Dynamic(Value),
}

/// Broadcast the lane mask `m` (confined to the lowest lane) to every lane of
/// a 64-bit pack with `bits`-wide lanes.
fn replicate_lane_mask(m: u64, bits: u32) -> u64 {
    let mut out = m;
    let mut shift = bits;
    while shift < 64 {
        out |= out << shift;
        shift *= 2;
    }
    out
}

/// Which packed-integer add/sub variant to emit.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum PackedIntOp {
    /// `PADDB`/`PADDW`/`PADDD`/`PADDQ` - wrapping per-lane add.
    Add,
    /// `PSUBB`/`PSUBW`/`PSUBD`/`PSUBQ` - wrapping per-lane subtract.
    Sub,
    /// `PADDSB`/`PADDSW` - add, saturating to the signed lane range.
    AddSatSigned,
    /// `PADDUSB`/`PADDUSW` - add, saturating to the unsigned lane range.
    AddSatUnsigned,
    /// `PSUBSB`/`PSUBSW` - subtract, saturating to the signed lane range.
    SubSatSigned,
    /// `PSUBUSB`/`PSUBUSW` - subtract, saturating to the unsigned lane range.
    SubSatUnsigned,
}

/// A lane-wise packed-integer operation whose lanes are independent and fit in
/// a widened `I64` - compares, min/max, average and multiply.
///
/// These share one expansion ([`Lifter::lanewise_64`]) in scalar mode and, for
/// most of them, a single native Cranelift vector op in vector mode.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum PackedLaneOp {
    /// `PCMPEQB`/`PCMPEQW`/`PCMPEQD`/`PCMPEQQ` - lanes equal -> all-ones mask.
    CmpEq,
    /// `PCMPGTB`/`PCMPGTW`/`PCMPGTD`/`PCMPGTQ` - signed greater-than -> all-ones.
    CmpGt,
    /// `PMINUB`/`PMINUW` - unsigned per-lane minimum.
    MinU,
    /// `PMINSB`/`PMINSW`/`PMINSD` - signed per-lane minimum.
    MinS,
    /// `PMAXUB`/`PMAXUW` - unsigned per-lane maximum.
    MaxU,
    /// `PMAXSB`/`PMAXSW`/`PMAXSD` - signed per-lane maximum.
    MaxS,
    /// `PAVGB`/`PAVGW` - unsigned rounded average, `(a + b + 1) >> 1`.
    Avg,
    /// `PMULLW` - low half of the per-lane product.
    MulLow,
    /// `PMULHW` - high half of the signed per-lane product.
    MulHighS,
    /// `PMULHUW` - high half of the unsigned per-lane product.
    MulHighU,
}

impl PackedLaneOp {
    /// True when lanes must be *sign*-extended as they are widened to `I64` for
    /// the scalar expansion.
    fn is_signed(self) -> bool {
        matches!(
            self,
            PackedLaneOp::CmpGt | PackedLaneOp::MinS | PackedLaneOp::MaxS | PackedLaneOp::MulHighS
        )
    }

    /// True when Cranelift has a single native vector instruction for this op,
    /// so vector mode can take the fast path instead of expanding lanes.
    ///
    /// Every [`PackedLaneOp`] now has one - the high-multiply forms map onto
    /// the vector-polymorphic `smulhi` / `umulhi`.
    fn has_vector_form(self) -> bool {
        true
    }
}

impl PackedIntOp {
    /// True for the two ops that interpret lanes as signed and clamp to
    /// `[-2^(n-1), 2^(n-1)-1]`.
    fn is_signed(self) -> bool {
        matches!(self, PackedIntOp::AddSatSigned | PackedIntOp::SubSatSigned)
    }

    /// True for the subtracting ops.
    fn is_subtract(self) -> bool {
        matches!(
            self,
            PackedIntOp::Sub | PackedIntOp::SubSatSigned | PackedIntOp::SubSatUnsigned
        )
    }
}

impl<'a> Lifter<'a> {
    /// Load a 128-bit XMM source operand as a `(lo, hi)` scalar pair.
    ///
    /// Supports:
    ///  - XMM register source
    ///  - GPR source (zero-extends 64-bit value into lo; hi = 0)
    ///  - Memory source (two 8-byte loads)
    pub(super) fn read_xmm_src(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<(Value, Value)> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                if let Some(idx) = xmm_register_index(reg) {
                    Ok((self.get_xmm_lo(idx), self.get_xmm_hi(idx)))
                } else {
                    // GPR source (e.g. MOVQ xmm, r64): zero-extend to 128 bits.
                    let lo = self.get_reg_val(reg)?;
                    let hi = self.builder.ins().iconst(types::I64, 0);
                    Ok((lo, hi))
                }
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                let lo = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                let hi = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 8);
                Ok((lo, hi))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xmm-src-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// Write a `(lo, hi)` scalar pair to an XMM destination operand.
    ///
    /// Supports:
    ///  - XMM register destination
    ///  - GPR destination (uses lo bits only)
    ///  - Memory destination (two 8-byte stores)
    fn write_xmm_dst(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        lo: Value,
        hi: Value,
    ) -> crate::Result<()> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                if let Some(idx) = xmm_register_index(reg) {
                    self.set_xmm_lo(idx, lo);
                    self.set_xmm_hi(idx, hi);
                    Ok(())
                } else {
                    // GPR destination: use lo (low 64 bits).
                    self.set_reg_val(reg, lo)
                }
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                self.builder.ins().store(MemFlagsData::new(), lo, ptr, 0);
                self.builder.ins().store(MemFlagsData::new(), hi, ptr, 8);
                Ok(())
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xmm-dst-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// Load a 128-bit XMM source operand as a native `I64X2` value.
    ///
    /// - XMM register source: direct vec read (or assembled from lo/hi
    ///   in scalar mode).
    /// - GPR source (e.g. MOVQ xmm, r64): zero-extends 64-bit value
    ///   into lane 0; lane 1 = 0.
    /// - Memory source: single 128-bit unaligned load.
    pub(super) fn read_xmm_src_vec(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                if let Some(idx) = xmm_register_index(reg) {
                    Ok(self.get_xmm_vec(idx))
                } else {
                    let lo = self.get_reg_val(reg)?;
                    let v = self.builder.ins().scalar_to_vector(types::I64X2, lo);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    Ok(self.builder.ins().insertlane(v, zero, 1))
                }
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                Ok(self
                    .builder
                    .ins()
                    .load(types::I64X2, MemFlagsData::new(), ptr, 0))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xmm-src-vec-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// Write a 128-bit `I64X2` value to an XMM destination operand.
    ///
    /// - XMM register destination: direct vec write (or split into
    ///   lo/hi in scalar mode).
    /// - GPR destination: uses lane 0 (low 64 bits).
    /// - Memory destination: single 128-bit unaligned store.
    fn write_xmm_dst_vec(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        val: Value,
    ) -> crate::Result<()> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                if let Some(idx) = xmm_register_index(reg) {
                    self.set_xmm_vec(idx, val);
                    Ok(())
                } else {
                    let lo = self.builder.ins().extractlane(val, 0);
                    self.set_reg_val(reg, lo)
                }
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                self.builder.ins().store(MemFlagsData::new(), val, ptr, 0);
                Ok(())
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xmm-dst-vec-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// ENDBR64 / ENDBR32 - CET end-branch marker, a NOP for lifting purposes.
    pub(super) fn handle_endbr64(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    /// MOVAPS / MOVAPD / MOVUPS / MOVUPD / MOVDQA / MOVDQU - 128-bit XMM move.
    pub(super) fn handle_movaps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.config.use_vector_xmm {
            let v = self.read_xmm_src_vec(instruction, 1)?;
            return self.write_xmm_dst_vec(instruction, 0, v);
        }
        let (lo, hi) = self.read_xmm_src(instruction, 1)?;
        self.write_xmm_dst(instruction, 0, lo, hi)
    }

    /// MOVQ (SSE2 form) - move 64 bits to/from an XMM register.
    ///
    /// When the destination is an XMM register the high 64 bits are zeroed
    /// (matching real hardware behaviour).
    pub(super) fn handle_movq_sse(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zero = self.builder.ins().iconst(types::I64, 0);

        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                if let Some(dst_idx) = xmm_register_index(dst_reg) {
                    // MOVQ xmm, xmm/r64/m64 - move 64 bits, zero high 64.
                    let lo = match self.operand_kind(instruction, 1) {
                        OpKind::Register => {
                            let src = self.operand_register(instruction, 1);
                            if let Some(src_idx) = xmm_register_index(src) {
                                self.get_xmm_lo(src_idx)
                            } else {
                                self.get_reg_val(src)?
                            }
                        }
                        OpKind::Memory => {
                            let ptr = self.get_mem_ptr(instruction, 1)?;
                            self.builder
                                .ins()
                                .load(types::I64, MemFlagsData::new(), ptr, 0)
                        }
                        other => {
                            return Err(MicroliftError::UnsupportedInstruction {
                                mnemonic: format!("movq-src-{:?}", other),
                            });
                        }
                    };
                    self.set_xmm_lo(dst_idx, lo);
                    self.set_xmm_hi(dst_idx, zero);
                } else {
                    // MOVQ r64, xmm - extract low 64 bits to GPR.
                    let src_xmm = self.operand_register(instruction, 1);
                    let src_idx = xmm_register_index(src_xmm).ok_or_else(|| {
                        MicroliftError::UnsupportedInstruction {
                            mnemonic: "movq-r64-non-xmm-src".into(),
                        }
                    })?;
                    let lo = self.get_xmm_lo(src_idx);
                    self.set_reg_val(dst_reg, lo)?;
                }
            }
            OpKind::Memory => {
                // MOVQ m64, xmm - store low 64 bits.
                let src_xmm = self.operand_register(instruction, 1);
                let src_idx = xmm_register_index(src_xmm).ok_or_else(|| {
                    MicroliftError::UnsupportedInstruction {
                        mnemonic: "movq-mem-non-xmm-src".into(),
                    }
                })?;
                let lo = self.get_xmm_lo(src_idx);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), lo, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movq-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// XORPS / XORPD - bitwise XOR of two 128-bit XMM values.
    pub(super) fn handle_xorps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx =
            xmm_register_index(dst_reg).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xorps-dst-not-xmm-{:?}", dst_reg),
            })?;
        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let res = self.builder.ins().bxor(dst, src);
            self.set_xmm_vec(dst_idx, res);
            return Ok(());
        }
        let (dst_lo, dst_hi) = (self.get_xmm_lo(dst_idx), self.get_xmm_hi(dst_idx));
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let res_lo = self.builder.ins().bxor(dst_lo, src_lo);
        let res_hi = self.builder.ins().bxor(dst_hi, src_hi);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PXOR - same bit-level semantics as XORPS.
    pub(super) fn handle_pxor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_xorps(instruction)
    }

    /// ANDPS / ANDPD / PAND - bitwise AND of two 128-bit XMM values.
    pub(super) fn handle_andps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx =
            xmm_register_index(dst_reg).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("andps-dst-not-xmm-{:?}", dst_reg),
            })?;
        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let res = self.builder.ins().band(dst, src);
            self.set_xmm_vec(dst_idx, res);
            return Ok(());
        }
        let (dst_lo, dst_hi) = (self.get_xmm_lo(dst_idx), self.get_xmm_hi(dst_idx));
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let res_lo = self.builder.ins().band(dst_lo, src_lo);
        let res_hi = self.builder.ins().band(dst_hi, src_hi);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ORPS / ORPD / POR - bitwise OR of two 128-bit XMM values.
    pub(super) fn handle_orps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx =
            xmm_register_index(dst_reg).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("orps-dst-not-xmm-{:?}", dst_reg),
            })?;
        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let res = self.builder.ins().bor(dst, src);
            self.set_xmm_vec(dst_idx, res);
            return Ok(());
        }
        let (dst_lo, dst_hi) = (self.get_xmm_lo(dst_idx), self.get_xmm_hi(dst_idx));
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let res_lo = self.builder.ins().bor(dst_lo, src_lo);
        let res_hi = self.builder.ins().bor(dst_hi, src_hi);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ANDNPS / ANDNPD / PANDN - bitwise AND-NOT: `result = ~dst & src`.
    pub(super) fn handle_andnps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx =
            xmm_register_index(dst_reg).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("andnps-dst-not-xmm-{:?}", dst_reg),
            })?;
        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // band_not(a, b) = a & !b - so band_not(src, dst) = src & !dst = ~dst & src
            let res = self.builder.ins().band_not(src, dst);
            self.set_xmm_vec(dst_idx, res);
            return Ok(());
        }
        let (dst_lo, dst_hi) = (self.get_xmm_lo(dst_idx), self.get_xmm_hi(dst_idx));
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        // band_not(a, b) = a & !b - so band_not(src, dst) = src & !dst = ~dst & src
        let res_lo = self.builder.ins().band_not(src_lo, dst_lo);
        let res_hi = self.builder.ins().band_not(src_hi, dst_hi);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// MOVSS - move scalar single-precision float (low 32 bits of XMM).
    ///
    /// - `MOVSS xmm1, xmm2`: replace bits[31:0] of dst with bits[31:0] of src;
    ///   upper 96 bits of dst are preserved.
    /// - `MOVSS xmm, m32`: load 32 bits into lo; upper 96 bits zeroed.
    /// - `MOVSS m32, xmm`: store bits[31:0] of src lo to memory.
    pub(super) fn handle_movss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
                    MicroliftError::UnsupportedInstruction {
                        mnemonic: "movss-dst-not-xmm".into(),
                    }
                })?;

                match self.operand_kind(instruction, 1) {
                    OpKind::Register => {
                        // MOVSS xmm1, xmm2 - merge low 32 bits, preserve upper 96.
                        let src_reg = self.operand_register(instruction, 1);
                        let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                            MicroliftError::UnsupportedInstruction {
                                mnemonic: "movss-src-not-xmm".into(),
                            }
                        })?;
                        let src_lo = self.get_xmm_lo(src_idx);
                        let dst_lo = self.get_xmm_lo(dst_idx);
                        // Replace bits[31:0] of dst_lo with src_lo[31:0]; preserve bits[63:32].
                        let mask = self.builder.ins().iconst(types::I64, !0xFFFF_FFFFi64);
                        let cleared = self.builder.ins().band(dst_lo, mask);
                        let src32 = self.builder.ins().band_imm_u(src_lo, 0xFFFF_FFFF);
                        let merged_lo = self.builder.ins().bor(cleared, src32);
                        // hi 64 bits of dst are preserved (no write needed).
                        self.set_xmm_lo(dst_idx, merged_lo);
                    }
                    OpKind::Memory => {
                        // MOVSS xmm, m32 - load 32 bits, zero upper 96 bits.
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        let val32 =
                            self.builder
                                .ins()
                                .load(types::I32, MemFlagsData::new(), ptr, 0);
                        let lo = self.builder.ins().uextend(types::I64, val32);
                        let hi = self.builder.ins().iconst(types::I64, 0);
                        self.set_xmm_lo(dst_idx, lo);
                        self.set_xmm_hi(dst_idx, hi);
                    }
                    other => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("movss-src-{:?}", other),
                        });
                    }
                }
            }
            OpKind::Memory => {
                // MOVSS m32, xmm - store low 32 bits.
                let src_reg = self.operand_register(instruction, 1);
                let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                    MicroliftError::UnsupportedInstruction {
                        mnemonic: "movss-mem-src-not-xmm".into(),
                    }
                })?;
                let lo = self.get_xmm_lo(src_idx);
                let val32 = self.builder.ins().ireduce(types::I32, lo);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), val32, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movss-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// MOVD - move a 32-bit value between a GPR/memory operand and an XMM
    /// register's low dword.
    ///
    /// - `MOVD xmm, r/m32`: low 32 bits = src; upper 96 bits zeroed.
    /// - `MOVD r/m32, xmm`: dst = xmm[31:0].
    ///
    /// (The 64-bit `r/m64` forms decode as `MOVQ`, handled separately.)
    pub(super) fn handle_movd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                if let Some(dst_idx) = xmm_register_index(dst_reg) {
                    // MOVD xmm, r/m32 - load low dword, zero the upper 96 bits.
                    let val32 = match self.operand_kind(instruction, 1) {
                        OpKind::Register => {
                            let src = self.operand_register(instruction, 1);
                            self.get_reg_val(src)?
                        }
                        OpKind::Memory => {
                            let ptr = self.get_mem_ptr(instruction, 1)?;
                            self.builder
                                .ins()
                                .load(types::I32, MemFlagsData::new(), ptr, 0)
                        }
                        other => {
                            return Err(MicroliftError::UnsupportedInstruction {
                                mnemonic: format!("movd-src-{:?}", other),
                            });
                        }
                    };
                    let lo = self.builder.ins().uextend(types::I64, val32);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.set_xmm_lo(dst_idx, lo);
                    self.set_xmm_hi(dst_idx, zero);
                } else {
                    // MOVD r32, xmm - extract low dword into a GPR (zero-extended).
                    let src_idx = self.xmm_operand_index(instruction, 1, "movd")?;
                    let lo = self.get_xmm_lo(src_idx);
                    let val32 = self.builder.ins().ireduce(types::I32, lo);
                    self.set_reg_val(dst_reg, val32)?;
                }
            }
            OpKind::Memory => {
                // MOVD m32, xmm - store the low dword.
                let src_idx = self.xmm_operand_index(instruction, 1, "movd")?;
                let lo = self.get_xmm_lo(src_idx);
                let val32 = self.builder.ins().ireduce(types::I32, lo);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), val32, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movd-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// MOVHPS / MOVHPD - move 64 bits to/from the high quadword of an XMM.
    ///
    /// - `MOVHPS xmm, m64`: high 64 bits = `[mem]`; low 64 bits preserved.
    /// - `MOVHPS m64, xmm`: store the high 64 bits of `xmm` to memory.
    pub(super) fn handle_movhps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                // Load form: dst is XMM, src must be m64.
                let dst_idx = self.xmm_operand_index(instruction, 0, "movhps")?;
                if self.operand_kind(instruction, 1) != OpKind::Memory {
                    return Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: "movhps-reg-src (use movlhps)".into(),
                    });
                }
                let ptr = self.get_mem_ptr(instruction, 1)?;
                let hi = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                self.set_xmm_hi(dst_idx, hi);
            }
            OpKind::Memory => {
                // Store form: dst is m64, src is XMM.
                let src_idx = self.xmm_operand_index(instruction, 1, "movhps")?;
                let hi = self.get_xmm_hi(src_idx);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), hi, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movhps-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// MOVLPS / MOVLPD - move 64 bits to/from the low quadword of an XMM.
    ///
    /// - `MOVLPS xmm, m64`: low 64 bits = `[mem]`; high 64 bits preserved.
    /// - `MOVLPS m64, xmm`: store the low 64 bits of `xmm` to memory.
    ///
    /// Note the preserved high half: unlike `MOVSD xmm, m64`, this is a merging
    /// load, not a zeroing one. There is no register-to-register form - those
    /// encodings decode as `MOVHLPS` / `MOVLHPS`.
    pub(super) fn handle_movlps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                // Load form: dst is XMM, src must be m64.
                let dst_idx = self.xmm_operand_index(instruction, 0, "movlps")?;
                if self.operand_kind(instruction, 1) != OpKind::Memory {
                    return Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: "movlps-reg-src (use movhlps)".into(),
                    });
                }
                let ptr = self.get_mem_ptr(instruction, 1)?;
                let lo = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                self.set_xmm_lo(dst_idx, lo);
            }
            OpKind::Memory => {
                // Store form: dst is m64, src is XMM.
                let src_idx = self.xmm_operand_index(instruction, 1, "movlps")?;
                let lo = self.get_xmm_lo(src_idx);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), lo, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movlps-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// MOVHLPS - `xmm1[63:0] = xmm2[127:64]`; the high half of `xmm1` is
    /// preserved. Register-to-register only (`0F 12` with a register operand).
    pub(super) fn handle_movhlps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "movhlps")?;
        let src_idx = self.xmm_operand_index(instruction, 1, "movhlps")?;
        let src_hi = self.get_xmm_hi(src_idx);
        self.set_xmm_lo(dst_idx, src_hi);
        Ok(())
    }

    /// MOVLHPS - `xmm1[127:64] = xmm2[63:0]`; the low half of `xmm1` is
    /// preserved. Register-to-register only (`0F 16` with a register operand).
    pub(super) fn handle_movlhps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "movlhps")?;
        let src_idx = self.xmm_operand_index(instruction, 1, "movlhps")?;
        let src_lo = self.get_xmm_lo(src_idx);
        self.set_xmm_hi(dst_idx, src_lo);
        Ok(())
    }

    /// MOVDDUP - `xmm1 = broadcast(src[63:0])`: the source's low quadword is
    /// written to both halves of the destination.
    pub(super) fn handle_movddup(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "movddup")?;
        let lo = match self.operand_kind(instruction, 1) {
            OpKind::Register => {
                let src_idx = self.xmm_operand_index(instruction, 1, "movddup")?;
                self.get_xmm_lo(src_idx)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                self.builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0)
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movddup-src-{:?}", other),
                });
            }
        };
        self.set_xmm_lo(dst_idx, lo);
        self.set_xmm_hi(dst_idx, lo);
        Ok(())
    }

    /// PSRLW/PSRLD/PSRLQ, PSLLW/PSLLD/PSLLQ, PSRAW/PSRAD - packed per-lane
    /// shift by a scalar count.
    ///
    /// - `<op> xmm, imm8`: every lane shifts by the immediate.
    /// - `<op> xmm, xmm/m128`: the count is the low 64 bits of the source
    ///   (all 64 bits participate, so a huge count is out of range, not
    ///   truncated).
    ///
    /// An out-of-range count (>= the lane width) is *not* the modulo-width
    /// behaviour of a raw machine shift: the zero-filling shifts produce
    /// all-zero lanes, while `PSRA*` saturates to each lane's sign, which is
    /// the same as shifting by `lane_bits - 1`.
    pub(super) fn handle_packed_shift(
        &mut self,
        instruction: &Instruction,
        op: PackedShiftOp,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let who = op.mnemonic(width);
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let count = self.read_packed_shift_count(instruction, who)?;

        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let r = self.packed_shift_vec(dst, count, op, width);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let res_lo = self.packed_shift_64(dst_lo, count, op, width);
        let res_hi = self.packed_shift_64(dst_hi, count, op, width);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Read operand 1 of a packed shift as a shift count.
    fn read_packed_shift_count(
        &mut self,
        instruction: &Instruction,
        who: &str,
    ) -> crate::Result<ShiftCount> {
        match self.operand_kind(instruction, 1) {
            OpKind::Immediate8 => Ok(ShiftCount::Imm(instruction.immediate8() as u64)),
            OpKind::Register => {
                let src_idx = self.xmm_operand_index(instruction, 1, who)?;
                Ok(ShiftCount::Dynamic(self.get_xmm_lo(src_idx)))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                let v = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                Ok(ShiftCount::Dynamic(v))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-src-{:?}", who, other),
            }),
        }
    }

    /// Packed shift in vector mode: bitcast to the lane type and use
    /// Cranelift's native lane-wise shift, which lowers to a single host SIMD
    /// instruction.
    ///
    /// Cranelift masks the shift amount to the lane width, so the out-of-range
    /// count has to be handled explicitly: `PSRA*` clamps the count, the
    /// zero-filling shifts mask the result away.
    fn packed_shift_vec(
        &mut self,
        dst: Value,
        count: ShiftCount,
        op: PackedShiftOp,
        width: PackedWidth,
    ) -> Value {
        let ty = width.vector_type();
        let bits = width.bits() as u64;
        let a = self.vec_bitcast(dst, ty);

        let r = match count {
            ShiftCount::Imm(c) => {
                if op.zeroes_on_overshift() && c >= bits {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    return self.builder.ins().splat(types::I64X2, zero);
                }
                let c = c.min(bits - 1) as i64;
                match op {
                    PackedShiftOp::Srl => self.builder.ins().ushr_imm_u(a, c),
                    PackedShiftOp::Sll => self.builder.ins().ishl_imm_u(a, c),
                    PackedShiftOp::Sra => self.builder.ins().sshr_imm_u(a, c),
                }
            }
            ShiftCount::Dynamic(cv) => {
                let limit = self.builder.ins().iconst(types::I64, bits as i64 - 1);
                let clamped = self.builder.ins().umin(cv, limit);
                match op {
                    PackedShiftOp::Sra => self.builder.ins().sshr(a, clamped),
                    _ => {
                        let shifted = if op == PackedShiftOp::Sll {
                            self.builder.ins().ishl(a, clamped)
                        } else {
                            self.builder.ins().ushr(a, clamped)
                        };
                        // Zero every lane when the count was out of range. A
                        // splat of a scalar select keeps the mask off the
                        // vector unit's critical path.
                        let too_big =
                            self.builder
                                .ins()
                                .icmp(IntCC::UnsignedGreaterThan, cv, limit);
                        let lane_ty = ty.lane_type();
                        let ones = self.builder.ins().iconst(lane_ty, -1);
                        let zero = self.builder.ins().iconst(lane_ty, 0);
                        let keep = self.builder.ins().select(too_big, zero, ones);
                        let keep_vec = self.builder.ins().splat(ty, keep);
                        self.builder.ins().band(shifted, keep_vec)
                    }
                }
            }
        };
        self.vec_bitcast(r, types::I64X2)
    }

    /// Packed shift of one 64-bit half of an XMM register (scalar mode).
    fn packed_shift_64(
        &mut self,
        a: Value,
        count: ShiftCount,
        op: PackedShiftOp,
        width: PackedWidth,
    ) -> Value {
        let bits = width.bits() as u64;

        if op == PackedShiftOp::Sra {
            // Per lane, `sra(x, c) == srl(x ^ S, c) - srl(S, c)` where `S` is
            // the lane's sign bit: the xor turns the sign-fill into a
            // zero-fill, and subtracting the shifted sign bit restores it.
            // `swar_sub` keeps the subtraction from borrowing across lanes.
            // A count >= bits saturates to the sign, i.e. a shift by bits - 1.
            let clamped = self.clamp_shift_count(count, bits - 1);
            let sign_imm = width.lane_sign_mask();
            let sign = self.builder.ins().iconst(types::I64, sign_imm);
            let flipped = self.builder.ins().bxor(a, sign);
            let shifted = self.packed_logical_shift_64(flipped, clamped, false, width);
            let sign_shifted = self.packed_logical_shift_64(sign, clamped, false, width);
            return self.swar_sub(shifted, sign_shifted, width);
        }

        let left = op == PackedShiftOp::Sll;
        match count {
            ShiftCount::Imm(c) if c >= bits => self.builder.ins().iconst(types::I64, 0),
            ShiftCount::Imm(_) => self.packed_logical_shift_64(a, count, left, width),
            ShiftCount::Dynamic(cv) => {
                let shifted = self.packed_logical_shift_64(a, count, left, width);
                let zero = self.builder.ins().iconst(types::I64, 0);
                let too_big =
                    self.builder
                        .ins()
                        .icmp_imm_s(IntCC::UnsignedGreaterThan, cv, bits as i64 - 1);
                self.builder.ins().select(too_big, zero, shifted)
            }
        }
    }

    /// Clamp a shift count to at most `max`, so the shift stays in range.
    fn clamp_shift_count(&mut self, count: ShiftCount, max: u64) -> ShiftCount {
        match count {
            ShiftCount::Imm(c) => ShiftCount::Imm(c.min(max)),
            ShiftCount::Dynamic(cv) => {
                let limit = self.builder.ins().iconst(types::I64, max as i64);
                ShiftCount::Dynamic(self.builder.ins().umin(cv, limit))
            }
        }
    }

    /// Lane-wise logical shift of one 64-bit lane pack by a count that must
    /// already be in range (`< lane width`).
    ///
    /// The raw 64-bit shift lets bits cross lane boundaries, so the result is
    /// masked with the shifted lane mask replicated across every lane - e.g.
    /// for `PSRLD` by 4 the mask is `0x0FFF_FFFF_0FFF_FFFF`.
    fn packed_logical_shift_64(
        &mut self,
        a: Value,
        count: ShiftCount,
        left: bool,
        width: PackedWidth,
    ) -> Value {
        let bits = width.bits();
        let lane = width.lane_mask() as u64;

        match count {
            ShiftCount::Imm(c) => {
                let c = c as i64;
                let lane_keep = if left { (lane << c) & lane } else { lane >> c };
                let mask = replicate_lane_mask(lane_keep, bits) as i64;
                let shifted = if left {
                    self.builder.ins().ishl_imm_u(a, c)
                } else {
                    self.builder.ins().ushr_imm_u(a, c)
                };
                self.builder.ins().band_imm_u(shifted, mask)
            }
            ShiftCount::Dynamic(cv) => {
                // Build the same mask at runtime: shift the lowest lane's mask,
                // then double it up into every lane.
                let lane_const = self.builder.ins().iconst(types::I64, lane as i64);
                let mut mask = if left {
                    let s = self.builder.ins().ishl(lane_const, cv);
                    self.builder.ins().band_imm_u(s, lane as i64)
                } else {
                    self.builder.ins().ushr(lane_const, cv)
                };
                let mut shift = bits;
                while shift < 64 {
                    let up = self.builder.ins().ishl_imm_u(mask, shift as i64);
                    mask = self.builder.ins().bor(mask, up);
                    shift *= 2;
                }
                let shifted = if left {
                    self.builder.ins().ishl(a, cv)
                } else {
                    self.builder.ins().ushr(a, cv)
                };
                self.builder.ins().band(shifted, mask)
            }
        }
    }

    /// PSHUFD - shuffle the four 32-bit dwords of `xmm2/m128` into `xmm1`
    /// according to `imm8` (2 bits per destination dword select a source dword).
    pub(super) fn handle_pshufd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "pshufd")?;
        let imm = instruction.immediate8();

        if self.vec_mode() {
            // Every result dword comes from the source, so the shuffle's second
            // operand is unused - pass the source again rather than materialize
            // a dead value.
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let mask = Self::lane_shuffle_mask(32, |j| u32::from((imm >> (2 * j)) & 0b11));
            let r = self.vec_shuffle(src, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        // The four source dwords, each isolated into the low 32 bits of an I64.
        let dwords = [
            self.builder.ins().band_imm_u(src_lo, 0xFFFF_FFFF),
            self.builder.ins().ushr_imm_u(src_lo, 32),
            self.builder.ins().band_imm_u(src_hi, 0xFFFF_FFFF),
            self.builder.ins().ushr_imm_u(src_hi, 32),
        ];

        let pick = |j: u32| dwords[((imm >> (2 * j)) & 0b11) as usize];
        let (r0, r1, r2, r3) = (pick(0), pick(1), pick(2), pick(3));

        let r1_hi = self.builder.ins().ishl_imm_u(r1, 32);
        let res_lo = self.builder.ins().bor(r0, r1_hi);
        let r3_hi = self.builder.ins().ishl_imm_u(r3, 32);
        let res_hi = self.builder.ins().bor(r2, r3_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Resolve operand `op_index` to an XMM register index, or error with a
    /// handler-specific mnemonic tag.
    pub(super) fn xmm_operand_index(
        &self,
        instruction: &Instruction,
        op_index: u32,
        who: &str,
    ) -> crate::Result<u8> {
        let reg = self.operand_register(instruction, op_index);
        xmm_register_index(reg).ok_or_else(|| MicroliftError::UnsupportedInstruction {
            mnemonic: format!("{}-op{}-not-xmm-{:?}", who, op_index, reg),
        })
    }

    /// MOVSD (SSE2 form) - move scalar double-precision float (low 64 bits of XMM).
    ///
    /// - `MOVSD xmm1, xmm2`: replace lo of dst with lo of src;
    ///   hi (high 64 bits) of dst is preserved.
    /// - `MOVSD xmm, m64`: load 64 bits into lo; hi zeroed.
    /// - `MOVSD m64, xmm`: store lo of src to memory.
    pub(super) fn handle_movsd_sse(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
                    MicroliftError::UnsupportedInstruction {
                        mnemonic: "movsd-sse-dst-not-xmm".into(),
                    }
                })?;

                match self.operand_kind(instruction, 1) {
                    OpKind::Register => {
                        // MOVSD xmm1, xmm2 - copy lo of src, preserve hi of dst.
                        let src_reg = self.operand_register(instruction, 1);
                        let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                            MicroliftError::UnsupportedInstruction {
                                mnemonic: "movsd-sse-src-not-xmm".into(),
                            }
                        })?;
                        let src_lo = self.get_xmm_lo(src_idx);
                        // hi of dst is preserved (no write needed).
                        self.set_xmm_lo(dst_idx, src_lo);
                    }
                    OpKind::Memory => {
                        // MOVSD xmm, m64 - load 64 bits, zero hi.
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        let lo = self
                            .builder
                            .ins()
                            .load(types::I64, MemFlagsData::new(), ptr, 0);
                        let hi = self.builder.ins().iconst(types::I64, 0);
                        self.set_xmm_lo(dst_idx, lo);
                        self.set_xmm_hi(dst_idx, hi);
                    }
                    other => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("movsd-sse-src-{:?}", other),
                        });
                    }
                }
            }
            OpKind::Memory => {
                // MOVSD m64, xmm - store lo.
                let src_reg = self.operand_register(instruction, 1);
                let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                    MicroliftError::UnsupportedInstruction {
                        mnemonic: "movsd-sse-mem-src-not-xmm".into(),
                    }
                })?;
                let lo = self.get_xmm_lo(src_idx);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlagsData::new(), lo, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movsd-sse-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// Read the low 64 bits (a scalar `f64`'s raw bits) of a source operand.
    ///
    /// Unlike [`Self::read_xmm_src`], memory sources load only 8 bytes, which is
    /// correct for the `m64` operands of scalar-double instructions (`ADDSD`,
    /// `UCOMISD`, ...).
    fn read_scalar_f64_bits(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                if let Some(idx) = xmm_register_index(reg) {
                    Ok(self.get_xmm_lo(idx))
                } else {
                    self.get_reg_val(reg)
                }
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                Ok(self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("scalar-f64-src-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// Read the low 32 bits (a scalar `f32`'s raw bits, as an `I32`) of a source
    /// operand.
    ///
    /// Memory sources load only 4 bytes, which is correct for the `m32` operands
    /// of scalar-single instructions (`ADDSS`, `MULSS`, ...). Register sources
    /// must be XMM - no scalar-single instruction takes a GPR operand.
    fn read_scalar_f32_bits(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        who: &str,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let idx = self.xmm_operand_index(instruction, op_index, who)?;
                let lo = self.get_xmm_lo(idx);
                Ok(self.builder.ins().ireduce(types::I32, lo))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                Ok(self
                    .builder
                    .ins()
                    .load(types::I32, MemFlagsData::new(), ptr, 0))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("scalar-f32-src-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// Replace bits[31:0] of XMM `idx` with `bits32` (an `I32`), leaving
    /// bits[127:32] untouched - the merge behaviour every scalar-single
    /// destination write shares.
    fn merge_xmm_dword0(&mut self, idx: u8, bits32: Value) {
        let bits = self.builder.ins().uextend(types::I64, bits32);
        let dst_lo = self.get_xmm_lo(idx);
        let mask = self.builder.ins().iconst(types::I64, !0xFFFF_FFFFi64);
        let cleared = self.builder.ins().band(dst_lo, mask);
        let merged = self.builder.ins().bor(cleared, bits);
        // The high quadword is preserved (no write needed).
        self.set_xmm_lo(idx, merged);
    }

    /// The whole `PUNPCK{L,H}{BW,WD,DQ,QDQ}` family, plus their floating-point
    /// aliases `UNPCK{L,H}P{S,D}`.
    ///
    /// Every form interleaves `width`-wide lanes taken from one 64-bit half of
    /// each operand: the *low* half for `PUNPCKL*`, the *high* half for
    /// `PUNPCKH*`. Result lane `2i` comes from the destination, lane `2i+1` from
    /// the source, so the selected half of each operand expands to fill all 128
    /// bits of the destination.
    ///
    /// For example `PUNPCKLDQ` yields `[dst.d0, src.d0, dst.d1, src.d1]`.
    pub(super) fn handle_punpck(
        &mut self,
        instruction: &Instruction,
        who: &str,
        width: PackedWidth,
        high: bool,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let lane_bits = width.bits();

        if self.vec_mode() {
            // Result lane `2i` is destination lane `off + i`, lane `2i+1` the
            // source's - `off` selecting which half of each operand feeds the
            // interleave. Source lanes are numbered past the first operand's.
            let lanes = 128 / lane_bits;
            let off = if high { lanes / 2 } else { 0 };
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let mask = Self::lane_shuffle_mask(lane_bits, |j| {
                let lane = off + j / 2;
                if j % 2 == 0 { lane } else { lanes + lane }
            });
            let r = self.vec_shuffle(dst, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let (a, b) = if high {
            (self.get_xmm_hi(dst_idx), src_hi)
        } else {
            (self.get_xmm_lo(dst_idx), src_lo)
        };

        if lane_bits == 64 {
            // A single lane from each operand: no interleaving to do.
            self.set_xmm_lo(dst_idx, a);
            self.set_xmm_hi(dst_idx, b);
            return Ok(());
        }

        // `n` lanes per source half expand into `2n` result lanes, i.e. exactly
        // the two 64-bit halves of the destination.
        let n = 64 / lane_bits;
        let mut acc = [
            self.builder.ins().iconst(types::I64, 0),
            self.builder.ins().iconst(types::I64, 0),
        ];
        for i in 0..n {
            let shift = (i * lane_bits) as i64;
            let x = self.extract_lane(a, shift, lane_bits, false);
            let y = self.extract_lane(b, shift, lane_bits, false);
            for (val, out) in [(x, 2 * i), (y, 2 * i + 1)] {
                let half = (out / n) as usize;
                let pos = ((out % n) * lane_bits) as i64;
                let placed = if pos == 0 {
                    val
                } else {
                    self.builder.ins().ishl_imm_u(val, pos)
                };
                acc[half] = self.builder.ins().bor(acc[half], placed);
            }
        }

        self.set_xmm_lo(dst_idx, acc[0]);
        self.set_xmm_hi(dst_idx, acc[1]);
        Ok(())
    }

    /// Apply `op` to two `f64` values given as raw `I64` bit patterns, returning
    /// the result's bits.
    fn f64_bits_arith(
        &mut self,
        a_bits: Value,
        b_bits: Value,
        op: &impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> Value {
        let a = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), a_bits);
        let b = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), b_bits);
        let r = op(&mut self.builder, a, b);
        self.builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), r)
    }

    /// Apply `op` to the two packed `f32` lanes held in a 64-bit half of an XMM
    /// register (lane 0 = bits[31:0], lane 1 = bits[63:32]), returning the
    /// repacked 64-bit result.
    fn f32_pair_arith(
        &mut self,
        a_bits: Value,
        b_bits: Value,
        op: &impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> Value {
        let a0 = self.builder.ins().ireduce(types::I32, a_bits);
        let b0 = self.builder.ins().ireduce(types::I32, b_bits);
        let a1_64 = self.builder.ins().ushr_imm_u(a_bits, 32);
        let b1_64 = self.builder.ins().ushr_imm_u(b_bits, 32);
        let a1 = self.builder.ins().ireduce(types::I32, a1_64);
        let b1 = self.builder.ins().ireduce(types::I32, b1_64);

        let r0 = self.f32_bits_arith(a0, b0, op);
        let r1 = self.f32_bits_arith(a1, b1, op);

        let r0_64 = self.builder.ins().uextend(types::I64, r0);
        let r1_64 = self.builder.ins().uextend(types::I64, r1);
        let r1_shifted = self.builder.ins().ishl_imm_u(r1_64, 32);
        self.builder.ins().bor(r0_64, r1_shifted)
    }

    /// Apply `op` to two `f32` values given as raw `I32` bit patterns, returning
    /// the result's bits.
    fn f32_bits_arith(
        &mut self,
        a_bits: Value,
        b_bits: Value,
        op: &impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> Value {
        let a = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), a_bits);
        let b = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), b_bits);
        let r = op(&mut self.builder, a, b);
        self.builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), r)
    }

    /// Shared implementation for the scalar double-precision arithmetic
    /// instructions (`ADDSD` / `SUBSD` / `MULSD` / `DIVSD`): apply `op` to the
    /// low 64-bit lanes of dst and src, preserving the high 64 bits of dst.
    ///
    /// `op` receives the two operands as `F64` values and returns the `F64`
    /// result.
    fn scalar_double_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let src_lo = self.read_scalar_f64_bits(instruction, 1)?;
        let res_lo = self.f64_bits_arith(dst_lo, src_lo, &op);

        // hi 64 bits are preserved (no write needed).
        self.set_xmm_lo(dst_idx, res_lo);
        Ok(())
    }

    /// Shared implementation for the SSE3 horizontal double-precision
    /// instructions (`HADDPD` / `HSUBPD`): the destination's two lanes are
    /// combined into the result's low lane and the source's two into its high
    /// one.
    ///
    /// Both operands are read before either result is written, because the
    /// source is very often the destination - `haddpd xmm0, xmm0` is how a
    /// compiler sums a pair - and a lane written early would then be an input to
    /// the other half.
    fn horizontal_double_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.f64_bits_arith(dst_lo, dst_hi, &op);
        let res_hi = self.f64_bits_arith(src_lo, src_hi, &op);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Shared implementation for the SSE3 horizontal single-precision
    /// instructions (`HADDPS` / `HSUBPS`): each 64-bit half contributes one
    /// 32-bit result lane, the destination's two halves filling the low half of
    /// the result and the source's two the high half.
    fn horizontal_single_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let r0 = self.f32_pair_horizontal(dst_lo, &op);
        let r1 = self.f32_pair_horizontal(dst_hi, &op);
        let r2 = self.f32_pair_horizontal(src_lo, &op);
        let r3 = self.f32_pair_horizontal(src_hi, &op);

        let res_lo = self.join_dwords(r0, r1);
        let res_hi = self.join_dwords(r2, r3);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Apply `op` to the two packed `f32` lanes held in one 64-bit half of an
    /// XMM register, returning the single 32-bit result's bits.
    fn f32_pair_horizontal(
        &mut self,
        bits: Value,
        op: &impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> Value {
        let lane0 = self.builder.ins().ireduce(types::I32, bits);
        let lane1_64 = self.builder.ins().ushr_imm_u(bits, 32);
        let lane1 = self.builder.ins().ireduce(types::I32, lane1_64);
        self.f32_bits_arith(lane0, lane1, op)
    }

    /// Two 32-bit values as one 64-bit half, `lo` in bits [31:0].
    fn join_dwords(&mut self, lo: Value, hi: Value) -> Value {
        let lo64 = self.builder.ins().uextend(types::I64, lo);
        let hi64 = self.builder.ins().uextend(types::I64, hi);
        let hi_shifted = self.builder.ins().ishl_imm_u(hi64, 32);
        self.builder.ins().bor(lo64, hi_shifted)
    }

    /// HADDPD - horizontal add of the two double-precision lanes of each operand.
    pub(super) fn handle_haddpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.horizontal_double_arith(instruction, "haddpd", fadd_x86)
    }

    /// HSUBPD - horizontal subtract: `dst[0] - dst[1]`, then `src[0] - src[1]`.
    pub(super) fn handle_hsubpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.horizontal_double_arith(instruction, "hsubpd", fsub_x86)
    }

    /// HADDPS - horizontal add of adjacent single-precision lane pairs.
    pub(super) fn handle_haddps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.horizontal_single_arith(instruction, "haddps", fadd_x86)
    }

    /// HSUBPS - horizontal subtract of adjacent single-precision lane pairs.
    pub(super) fn handle_hsubps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.horizontal_single_arith(instruction, "hsubps", fsub_x86)
    }

    /// Shared implementation for the packed double-precision arithmetic
    /// instructions (`ADDPD` / `SUBPD` / ...): apply `op` to both 64-bit lanes.
    fn packed_double_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.f64_bits_arith(dst_lo, src_lo, &op);
        let res_hi = self.f64_bits_arith(dst_hi, src_hi, &op);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Shared implementation for the scalar single-precision arithmetic
    /// instructions (`ADDSS` / `SUBSS` / `MULSS` / `DIVSS`): apply `op` to the
    /// low 32-bit lanes of dst and src, preserving bits[127:32] of dst.
    ///
    /// `op` receives the two operands as `F32` values and returns the `F32`
    /// result.
    fn scalar_single_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_bits = self.builder.ins().ireduce(types::I32, dst_lo);
        let src_bits = self.read_scalar_f32_bits(instruction, 1, who)?;

        let res = self.f32_bits_arith(dst_bits, src_bits, &op);
        self.merge_xmm_dword0(dst_idx, res);
        Ok(())
    }

    /// Shared implementation for the packed single-precision arithmetic
    /// instructions (`MULPS` / `ADDPS` / ...): apply `op` to all four 32-bit
    /// lanes.
    fn packed_single_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.f32_pair_arith(dst_lo, src_lo, &op);
        let res_hi = self.f32_pair_arith(dst_hi, src_hi, &op);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ADDSD - scalar double-precision add of the low 64 bits; the high 64 bits
    /// of the destination are preserved.
    pub(super) fn handle_addsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "addsd", fadd_x86)
    }

    /// MULSD - scalar double-precision multiply of the low 64 bits; the high 64
    /// bits of the destination are preserved.
    pub(super) fn handle_mulsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "mulsd", fmul_x86)
    }

    /// DIVSD - scalar double-precision divide of the low 64 bits (`dst / src`);
    /// the high 64 bits of the destination are preserved.
    pub(super) fn handle_divsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "divsd", fdiv_x86)
    }

    /// ADDPD - packed double-precision add of both 64-bit lanes.
    pub(super) fn handle_addpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "addpd", false, PackedFpOp::Add)? {
            return Ok(());
        }
        self.packed_double_arith(instruction, "addpd", fadd_x86)
    }

    /// SUBPD - packed double-precision subtract of both 64-bit lanes.
    pub(super) fn handle_subpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "subpd", false, PackedFpOp::Sub)? {
            return Ok(());
        }
        self.packed_double_arith(instruction, "subpd", fsub_x86)
    }

    /// ADDSS - scalar single-precision add of the low 32 bits; bits[127:32] of
    /// the destination are preserved.
    pub(super) fn handle_addss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_single_arith(instruction, "addss", fadd_x86)
    }

    /// SUBSS - scalar single-precision subtract of the low 32 bits (`dst - src`);
    /// bits[127:32] of the destination are preserved.
    pub(super) fn handle_subss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_single_arith(instruction, "subss", fsub_x86)
    }

    /// MULSS - scalar single-precision multiply of the low 32 bits; bits[127:32]
    /// of the destination are preserved.
    pub(super) fn handle_mulss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_single_arith(instruction, "mulss", fmul_x86)
    }

    /// DIVSS - scalar single-precision divide of the low 32 bits (`dst / src`);
    /// bits[127:32] of the destination are preserved.
    pub(super) fn handle_divss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_single_arith(instruction, "divss", fdiv_x86)
    }

    /// MULPS - packed single-precision multiply of all four 32-bit lanes.
    pub(super) fn handle_mulps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "mulps", true, PackedFpOp::Mul)? {
            return Ok(());
        }
        self.packed_single_arith(instruction, "mulps", fmul_x86)
    }

    /// CVTSI2SS - convert a signed integer to a scalar single-precision float.
    ///
    /// - `CVTSI2SS xmm, r/m32`: `dst[31:0] = (f32)(i32)src`
    /// - `CVTSI2SS xmm, r/m64` (REX.W): `dst[31:0] = (f32)(i64)src`
    ///
    /// Bits [127:32] of the destination are preserved.
    pub(super) fn handle_cvtsi2ss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtsi2ss")?;
        let src = self.read_int32_or_64_src(instruction, 1, "cvtsi2ss")?;
        let f = self.builder.ins().fcvt_from_sint(types::F32, src);
        let bits32 = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), f);
        self.merge_xmm_dword0(dst_idx, bits32);
        Ok(())
    }

    /// CVTSS2SI / CVTTSS2SI - convert a scalar single-precision float to a
    /// signed integer, rounding to nearest-even (`CVTSS2SI`) or truncating
    /// toward zero (`CVTTSS2SI`).
    ///
    /// - `CVT[T]SS2SI r32, xmm/m32`: `dst = (i32)round(src)`
    /// - `CVT[T]SS2SI r64, xmm/m32` (REX.W): `dst = (i64)round(src)`
    pub(super) fn handle_cvtss2si(
        &mut self,
        instruction: &Instruction,
        truncate: bool,
    ) -> crate::Result<()> {
        let who = if truncate { "cvttss2si" } else { "cvtss2si" };
        self.scalar_fp_to_si(instruction, types::F32, truncate, who)
    }

    /// UCOMISD / COMISD - compare the low scalar doubles of two operands and set
    /// the integer flags (ZF/PF/CF); OF/SF/AF are cleared.
    ///
    /// The ordered (`COMISD`) and unordered (`UCOMISD`) forms differ only in
    /// which operands raise an invalid-operation exception, which this lifter
    /// does not model, so both are handled identically.
    pub(super) fn handle_ucomisd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "ucomisd")?;
        let a_bits = self.get_xmm_lo(dst_idx);
        let b_bits = self.read_scalar_f64_bits(instruction, 1)?;

        let a = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), a_bits);
        let b = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), b_bits);

        let unord = self.builder.ins().fcmp(FloatCC::Unordered, a, b);
        let eq = self.builder.ins().fcmp(FloatCC::Equal, a, b);
        let lt = self.builder.ins().fcmp(FloatCC::LessThan, a, b);

        let zf = self.builder.ins().bor(eq, unord);
        let cf = self.builder.ins().bor(lt, unord);
        self.set_flag(X86Flag::Zero, zf)?;
        self.set_flag(X86Flag::Parity, unord)?;
        self.set_flag(X86Flag::Carry, cf)?;
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Overflow, zero)?;
        self.set_flag(X86Flag::Sign, zero)?;
        self.set_flag(X86Flag::Auxiliary, zero)?;
        Ok(())
    }

    //  Packed integer add / subtract (PADDB..PADDQ, PSUBB..PSUBQ and the
    //  saturating PADDS*/PADDUS*/PSUBS*/PSUBUS* forms).

    /// Shared implementation of every packed-integer add/sub form.
    ///
    /// In vector mode the 128-bit value is bitcast to the lane type and handed
    /// to Cranelift's native `iadd`/`isub`/`*_sat`, which lower to a single host
    /// SIMD instruction. In scalar mode each 64-bit half is processed
    /// independently: wrapping ops use a SWAR sequence that blocks carries at
    /// the lane boundaries, saturating ops are expanded per lane.
    ///
    /// Note that `dst` is always the XMM register operand 0; the source may be
    /// an XMM register or `m128`.
    fn packed_int_arith(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: PackedIntOp,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.config.use_vector_xmm {
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let ty = width.vector_type();
            let a = self.vec_bitcast(dst, ty);
            let b = self.vec_bitcast(src, ty);
            let r = match op {
                PackedIntOp::Add => self.builder.ins().iadd(a, b),
                PackedIntOp::Sub => self.builder.ins().isub(a, b),
                PackedIntOp::AddSatSigned => self.builder.ins().sadd_sat(a, b),
                PackedIntOp::AddSatUnsigned => self.builder.ins().uadd_sat(a, b),
                PackedIntOp::SubSatSigned => self.builder.ins().ssub_sat(a, b),
                PackedIntOp::SubSatUnsigned => self.builder.ins().usub_sat(a, b),
            };
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let res_lo = self.packed_int_arith_64(dst_lo, src_lo, op, width);
        let res_hi = self.packed_int_arith_64(dst_hi, src_hi, op, width);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Reinterpret a 128-bit vector value as `ty` (a no-op if it already has
    /// that type). Lane-count changes require an explicit byte order.
    pub(super) fn vec_bitcast(&mut self, v: Value, ty: Type) -> Value {
        if self.builder.func.dfg.value_type(v) == ty {
            return v;
        }
        let flags = MemFlagsData::new().with_endianness(codegen::ir::Endianness::Little);
        self.builder.ins().bitcast(ty, flags, v)
    }

    /// Apply one packed-integer op to a single 64-bit half of an XMM register
    /// (scalar mode).
    fn packed_int_arith_64(
        &mut self,
        a: Value,
        b: Value,
        op: PackedIntOp,
        width: PackedWidth,
    ) -> Value {
        match op {
            PackedIntOp::Add => self.swar_add(a, b, width),
            PackedIntOp::Sub => self.swar_sub(a, b, width),
            _ => self.saturating_lanes(a, b, op, width),
        }
    }

    /// Lane-wise wrapping add of two 64-bit lane packs, without cross-lane
    /// carry propagation.
    ///
    /// Adds the lanes with their high bits cleared (so no carry can escape a
    /// lane) and then re-applies each lane's high bit as `a ^ b`, which is the
    /// correct high bit of the sum.
    fn swar_add(&mut self, a: Value, b: Value, width: PackedWidth) -> Value {
        if width == PackedWidth::Qword {
            return self.builder.ins().iadd(a, b);
        }
        let high = width.lane_sign_mask();
        let a_low = self.builder.ins().band_imm_u(a, !high);
        let b_low = self.builder.ins().band_imm_u(b, !high);
        let sum = self.builder.ins().iadd(a_low, b_low);
        let xor = self.builder.ins().bxor(a, b);
        let high_bits = self.builder.ins().band_imm_u(xor, high);
        self.builder.ins().bxor(sum, high_bits)
    }

    /// Lane-wise wrapping subtract of two 64-bit lane packs, without cross-lane
    /// borrow propagation.
    ///
    /// The mirror of [`Self::swar_add`]: setting every lane's high bit in `a`
    /// guarantees each lane can absorb its own borrow, and the high bits are
    /// then corrected to `~(a ^ b)`.
    fn swar_sub(&mut self, a: Value, b: Value, width: PackedWidth) -> Value {
        if width == PackedWidth::Qword {
            return self.builder.ins().isub(a, b);
        }
        let high_imm = width.lane_sign_mask();
        let high = self.builder.ins().iconst(types::I64, high_imm);
        let a_high_set = self.builder.ins().bor(a, high);
        let b_low = self.builder.ins().band_imm_u(b, !high_imm);
        let diff = self.builder.ins().isub(a_high_set, b_low);
        let xor = self.builder.ins().bxor(a, b);
        // band_not(high, xor) = high & ~(a ^ b) = ((a ^ ~b) & high)
        let correction = self.builder.ins().band_not(high, xor);
        self.builder.ins().bxor(diff, correction)
    }

    /// Saturating lane-wise add/sub of two 64-bit lane packs, expanded one lane
    /// at a time (scalar mode).
    ///
    /// Each lane is widened into a full `I64`, the arithmetic performed there
    /// (which cannot overflow, since x86 only defines saturating packed add/sub
    /// for 8- and 16-bit lanes), clamped to the lane's representable range, and
    /// shifted back into place.
    fn saturating_lanes(
        &mut self,
        a: Value,
        b: Value,
        op: PackedIntOp,
        width: PackedWidth,
    ) -> Value {
        let lane_bits = width.bits();
        debug_assert!(
            lane_bits <= 16,
            "saturating packed arithmetic is only defined for 8/16-bit lanes"
        );
        let lane_mask = width.lane_mask();
        let signed = op.is_signed();
        let (min, max) = if signed {
            (-(1i64 << (lane_bits - 1)), (1i64 << (lane_bits - 1)) - 1)
        } else {
            (0, lane_mask)
        };
        let min = self.builder.ins().iconst(types::I64, min);
        let max = self.builder.ins().iconst(types::I64, max);

        let mut acc = self.builder.ins().iconst(types::I64, 0);
        for lane in 0..(64 / lane_bits) {
            let shift = (lane * lane_bits) as i64;
            let x = self.extract_lane(a, shift, lane_bits, signed);
            let y = self.extract_lane(b, shift, lane_bits, signed);
            let raw = if op.is_subtract() {
                self.builder.ins().isub(x, y)
            } else {
                self.builder.ins().iadd(x, y)
            };
            // `raw` is small enough that signed clamping is correct for the
            // unsigned forms too (unsigned lanes widen to non-negative I64s).
            let low_clamped = self.builder.ins().smax(raw, min);
            let clamped = self.builder.ins().smin(low_clamped, max);
            let lane_val = self.builder.ins().band_imm_u(clamped, lane_mask);
            let placed = if shift == 0 {
                lane_val
            } else {
                self.builder.ins().ishl_imm_u(lane_val, shift)
            };
            acc = self.builder.ins().bor(acc, placed);
        }
        acc
    }

    /// Widen the `lane_bits`-wide lane at bit offset `shift` of `v` into a full
    /// `I64`, sign- or zero-extending as requested.
    pub(super) fn extract_lane(
        &mut self,
        v: Value,
        shift: i64,
        lane_bits: u32,
        signed: bool,
    ) -> Value {
        if signed {
            let left = 64 - shift - lane_bits as i64;
            let aligned = if left == 0 {
                v
            } else {
                self.builder.ins().ishl_imm_u(v, left)
            };
            self.builder
                .ins()
                .sshr_imm_u(aligned, 64 - lane_bits as i64)
        } else {
            let shifted = if shift == 0 {
                v
            } else {
                self.builder.ins().ushr_imm_u(v, shift)
            };
            let mask = match lane_bits {
                64 => return shifted,
                bits => ((1u64 << bits) - 1) as i64,
            };
            self.builder.ins().band_imm_u(shifted, mask)
        }
    }

    /// PADDB / PADDW / PADDD / PADDQ - wrapping packed integer add.
    pub(super) fn handle_padd(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "padd", PackedIntOp::Add, width)
    }

    /// PSUBB / PSUBW / PSUBD / PSUBQ - wrapping packed integer subtract.
    pub(super) fn handle_psub(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "psub", PackedIntOp::Sub, width)
    }

    /// PADDSB / PADDSW - packed signed-saturating add.
    pub(super) fn handle_padds(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "padds", PackedIntOp::AddSatSigned, width)
    }

    /// PADDUSB / PADDUSW - packed unsigned-saturating add.
    pub(super) fn handle_paddus(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "paddus", PackedIntOp::AddSatUnsigned, width)
    }

    /// PSUBSB / PSUBSW - packed signed-saturating subtract.
    pub(super) fn handle_psubs(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "psubs", PackedIntOp::SubSatSigned, width)
    }

    /// PSUBUSB / PSUBUSW - packed unsigned-saturating subtract.
    pub(super) fn handle_psubus(
        &mut self,
        instruction: &Instruction,
        width: PackedWidth,
    ) -> crate::Result<()> {
        self.packed_int_arith(instruction, "psubus", PackedIntOp::SubSatUnsigned, width)
    }

    //  Lane-wise packed integer compare / min / max / average / multiply.

    /// Shared implementation of every [`PackedLaneOp`] form.
    ///
    /// Vector mode maps most ops straight onto a native Cranelift vector
    /// instruction; the high-multiply forms (which have no such instruction)
    /// and scalar mode both fall back to the per-lane expansion in
    /// [`Self::lanewise_64`].
    pub(super) fn handle_packed_lane_op(
        &mut self,
        instruction: &Instruction,
        who: &str,
        op: PackedLaneOp,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.config.use_vector_xmm && op.has_vector_form() {
            let ty = width.vector_type();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;

            // The high-multiply forms are 16-bit-lane only and need their own
            // expansion; everything else is a single native vector op.
            if let PackedLaneOp::MulHighS | PackedLaneOp::MulHighU = op {
                let signed = op == PackedLaneOp::MulHighS;
                let r = self.vec_mulhi_word(dst, src, signed);
                self.set_xmm_vec(dst_idx, r);
                return Ok(());
            }

            let a = self.vec_bitcast(dst, ty);
            let b = self.vec_bitcast(src, ty);
            let r = match op {
                PackedLaneOp::CmpEq => self.builder.ins().icmp(IntCC::Equal, a, b),
                PackedLaneOp::CmpGt => self.builder.ins().icmp(IntCC::SignedGreaterThan, a, b),
                PackedLaneOp::MinU => self.builder.ins().umin(a, b),
                PackedLaneOp::MinS => self.builder.ins().smin(a, b),
                PackedLaneOp::MaxU => self.builder.ins().umax(a, b),
                PackedLaneOp::MaxS => self.builder.ins().smax(a, b),
                PackedLaneOp::Avg => self.builder.ins().avg_round(a, b),
                PackedLaneOp::MulLow => self.builder.ins().imul(a, b),
                PackedLaneOp::MulHighS | PackedLaneOp::MulHighU => unreachable!("handled above"),
            };
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let res_lo = self.lanewise_64(dst_lo, src_lo, op, width);
        let res_hi = self.lanewise_64(dst_hi, src_hi, op, width);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Apply a [`PackedLaneOp`] to every lane of one 64-bit half, by widening
    /// each lane to a full `I64`, operating there, and repacking.
    fn lanewise_64(&mut self, a: Value, b: Value, op: PackedLaneOp, width: PackedWidth) -> Value {
        let lane_bits = width.bits();
        let lane_mask = width.lane_mask();
        let signed = op.is_signed();

        let mut acc = self.builder.ins().iconst(types::I64, 0);
        for lane in 0..(64 / lane_bits) {
            let shift = (lane * lane_bits) as i64;
            let x = self.extract_lane(a, shift, lane_bits, signed);
            let y = self.extract_lane(b, shift, lane_bits, signed);
            let r = self.lane_op_64(x, y, op, lane_bits);
            let lane_val = self.builder.ins().band_imm_u(r, lane_mask);
            let placed = if shift == 0 {
                lane_val
            } else {
                self.builder.ins().ishl_imm_u(lane_val, shift)
            };
            acc = self.builder.ins().bor(acc, placed);
        }
        acc
    }

    /// Apply a [`PackedLaneOp`] to a single pair of already-widened lanes.
    ///
    /// Only the low `lane_bits` of the result matter - the caller masks it back
    /// into place - so the multiply forms need no overflow care beyond keeping
    /// the (at most 64-bit) product exact, which holds for every lane width x86
    /// defines these ops at.
    fn lane_op_64(&mut self, x: Value, y: Value, op: PackedLaneOp, lane_bits: u32) -> Value {
        match op {
            PackedLaneOp::CmpEq | PackedLaneOp::CmpGt => {
                let cc = if op == PackedLaneOp::CmpEq {
                    IntCC::Equal
                } else {
                    IntCC::SignedGreaterThan
                };
                let c = self.builder.ins().icmp(cc, x, y);
                let ones = self.builder.ins().iconst(types::I64, -1);
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().select(c, ones, zero)
            }
            // Lanes are zero-extended for the unsigned forms and sign-extended
            // for the signed ones, so a signed min/max on the widened values is
            // correct in both cases - but use the matching op for clarity.
            PackedLaneOp::MinU => self.builder.ins().umin(x, y),
            PackedLaneOp::MinS => self.builder.ins().smin(x, y),
            PackedLaneOp::MaxU => self.builder.ins().umax(x, y),
            PackedLaneOp::MaxS => self.builder.ins().smax(x, y),
            PackedLaneOp::Avg => {
                let sum = self.builder.ins().iadd(x, y);
                let rounded = self.builder.ins().iadd_imm_s(sum, 1);
                self.builder.ins().ushr_imm_u(rounded, 1)
            }
            PackedLaneOp::MulLow => self.builder.ins().imul(x, y),
            PackedLaneOp::MulHighS => {
                let p = self.builder.ins().imul(x, y);
                self.builder.ins().sshr_imm_u(p, lane_bits as i64)
            }
            PackedLaneOp::MulHighU => {
                let p = self.builder.ins().imul(x, y);
                self.builder.ins().ushr_imm_u(p, lane_bits as i64)
            }
        }
    }

    /// PMOVMSKB / MOVMSKPS / MOVMSKPD - gather the sign bit of every `width`
    /// lane into the low bits of a GPR, zeroing the rest of the destination.
    ///
    /// The lane width is what distinguishes the three: byte lanes give the 16
    /// bits of `PMOVMSKB`, dword lanes the 4 of `MOVMSKPS`, qword lanes the 2
    /// of `MOVMSKPD`.
    pub(super) fn handle_movmsk(
        &mut self,
        instruction: &Instruction,
        who: &str,
        width: PackedWidth,
    ) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let src_idx = self.xmm_operand_index(instruction, 1, who)?;

        if self.vec_mode() {
            // `vhigh_bits` is precisely this instruction: the concatenated sign
            // bits of every lane.
            let src = self.get_xmm_vec(src_idx);
            let v = self.vec_bitcast(src, width.vector_type());
            let acc = self.builder.ins().vhigh_bits(types::I64, v);
            return self.write_movmsk_dst(dst_reg, acc);
        }

        let halves = [self.get_xmm_lo(src_idx), self.get_xmm_hi(src_idx)];

        let lane_bits = width.bits();
        let per_half = 64 / lane_bits;
        let mut acc = self.builder.ins().iconst(types::I64, 0);
        for (half, v) in halves.into_iter().enumerate() {
            for lane in 0..per_half {
                let sign_bit = (lane * lane_bits + lane_bits - 1) as i64;
                let out = half as u32 * per_half + lane;
                let shifted = self.builder.ins().ushr_imm_u(v, sign_bit);
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                let placed = if out == 0 {
                    bit
                } else {
                    self.builder.ins().ishl_imm_u(bit, out as i64)
                };
                acc = self.builder.ins().bor(acc, placed);
            }
        }

        self.write_movmsk_dst(dst_reg, acc)
    }

    /// Write a gathered `MOVMSK` bitmask (an `I64`) to its GPR destination.
    ///
    /// The 32-bit destination form zero-extends to the full GPR anyway, so both
    /// widths just write the same value at their own type.
    fn write_movmsk_dst(&mut self, dst_reg: iced_x86::Register, acc: Value) -> crate::Result<()> {
        if dst_reg.size() == 8 {
            self.set_reg_val(dst_reg, acc)
        } else {
            let acc32 = self.builder.ins().ireduce(types::I32, acc);
            self.set_reg_val(dst_reg, acc32)
        }
    }

    /// PSLLDQ / PSRLDQ - shift the whole 128-bit register left/right by an
    /// *byte* count, zero-filling. A count above 15 clears the register.
    ///
    /// Only the `imm8` form exists, so the count is a lift-time constant and
    /// this compiles down to at most a couple of shifts and an OR.
    pub(super) fn handle_byte_shift_dq(
        &mut self,
        instruction: &Instruction,
        left: bool,
    ) -> crate::Result<()> {
        let who = if left { "pslldq" } else { "psrldq" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.vec_mode() {
            let count = i32::from(instruction.immediate8());
            let dst = self.get_xmm_vec(dst_idx);
            let zero = self.vec_zero();
            // Result byte `i` is destination byte `i -/+ count`, or a zero
            // byte - byte 0 of the zero operand - when that falls outside the
            // register. A count above 15 leaves every byte pointing at zero.
            let mut mask = [B; 16];
            for (i, m) in mask.iter_mut().enumerate() {
                let from = if left {
                    i as i32 - count
                } else {
                    i as i32 + count
                };
                if (0..16).contains(&from) {
                    *m = from as u8;
                }
            }
            let r = self.vec_shuffle(dst, zero, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let lo = self.get_xmm_lo(dst_idx);
        let hi = self.get_xmm_hi(dst_idx);
        let count = instruction.immediate8() as u32;

        let (res_lo, res_hi) = if count > 15 {
            let zero = self.builder.ins().iconst(types::I64, 0);
            (zero, zero)
        } else if count == 0 {
            (lo, hi)
        } else {
            let bits = (count * 8) as i64;
            match (left, bits < 64) {
                (true, true) => {
                    let hi_shifted = self.builder.ins().ishl_imm_u(hi, bits);
                    let carry = self.builder.ins().ushr_imm_u(lo, 64 - bits);
                    let new_hi = self.builder.ins().bor(hi_shifted, carry);
                    (self.builder.ins().ishl_imm_u(lo, bits), new_hi)
                }
                (true, false) => {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    (zero, self.builder.ins().ishl_imm_u(lo, bits - 64))
                }
                (false, true) => {
                    let lo_shifted = self.builder.ins().ushr_imm_u(lo, bits);
                    let carry = self.builder.ins().ishl_imm_u(hi, 64 - bits);
                    let new_lo = self.builder.ins().bor(lo_shifted, carry);
                    (new_lo, self.builder.ins().ushr_imm_u(hi, bits))
                }
                (false, false) => {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    (self.builder.ins().ushr_imm_u(hi, bits - 64), zero)
                }
            }
        };

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    //  Floating-point: the remaining scalar/packed arithmetic, MIN/MAX, SQRT,
    //  the single-precision compare and the SSE2 conversions.

    /// SUBSD - scalar double-precision subtract of the low 64 bits (`dst - src`);
    /// the high 64 bits of the destination are preserved.
    pub(super) fn handle_subsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "subsd", fsub_x86)
    }

    /// MULPD - packed double-precision multiply of both 64-bit lanes.
    pub(super) fn handle_mulpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "mulpd", false, PackedFpOp::Mul)? {
            return Ok(());
        }
        self.packed_double_arith(instruction, "mulpd", fmul_x86)
    }

    /// DIVPD - packed double-precision divide of both 64-bit lanes.
    pub(super) fn handle_divpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "divpd", false, PackedFpOp::Div)? {
            return Ok(());
        }
        self.packed_double_arith(instruction, "divpd", fdiv_x86)
    }

    /// ADDPS - packed single-precision add of all four 32-bit lanes.
    pub(super) fn handle_addps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "addps", true, PackedFpOp::Add)? {
            return Ok(());
        }
        self.packed_single_arith(instruction, "addps", fadd_x86)
    }

    /// SUBPS - packed single-precision subtract of all four 32-bit lanes.
    pub(super) fn handle_subps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "subps", true, PackedFpOp::Sub)? {
            return Ok(());
        }
        self.packed_single_arith(instruction, "subps", fsub_x86)
    }

    /// DIVPS - packed single-precision divide of all four 32-bit lanes.
    pub(super) fn handle_divps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "divps", true, PackedFpOp::Div)? {
            return Ok(());
        }
        self.packed_single_arith(instruction, "divps", fdiv_x86)
    }

    /// MINSS / MINSD / MINPS / MINPD, MAXSS / MAXSD / MAXPS / MAXPD.
    ///
    /// x86's MIN/MAX are **not** IEEE `minNum`/`maxNum` and are not commutative:
    /// the result is the destination operand only when the strict comparison
    /// holds, and the *source* operand otherwise. That single rule covers all
    /// three documented special cases at once - if either operand is a NaN the
    /// comparison is false and the source is returned, and `min(+0.0, -0.0)`
    /// likewise returns the source. So `select(dst < src, dst, src)` (and its
    /// `>` mirror) is the exact semantics, which is why this cannot use
    /// Cranelift's NaN-propagating `fmin`/`fmax`.
    fn fp_min_max(min: bool) -> impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value {
        move |b, x, y| {
            let cc = if min {
                FloatCC::LessThan
            } else {
                FloatCC::GreaterThan
            };
            let c = b.ins().fcmp(cc, x, y);
            b.ins().select(c, x, y)
        }
    }

    /// MINSD / MAXSD - scalar double-precision minimum/maximum of the low 64
    /// bits; the high 64 bits of the destination are preserved.
    pub(super) fn handle_minmaxsd(
        &mut self,
        instruction: &Instruction,
        min: bool,
    ) -> crate::Result<()> {
        let who = if min { "minsd" } else { "maxsd" };
        self.scalar_double_arith(instruction, who, Self::fp_min_max(min))
    }

    /// MINPD / MAXPD - packed double-precision minimum/maximum of both lanes.
    pub(super) fn handle_minmaxpd(
        &mut self,
        instruction: &Instruction,
        min: bool,
    ) -> crate::Result<()> {
        let who = if min { "minpd" } else { "maxpd" };
        let op = if min {
            PackedFpOp::Min
        } else {
            PackedFpOp::Max
        };
        if self.try_packed_fp_vec(instruction, who, false, op)? {
            return Ok(());
        }
        self.packed_double_arith(instruction, who, Self::fp_min_max(min))
    }

    /// MINSS / MAXSS - scalar single-precision minimum/maximum of the low 32
    /// bits; bits[127:32] of the destination are preserved.
    pub(super) fn handle_minmaxss(
        &mut self,
        instruction: &Instruction,
        min: bool,
    ) -> crate::Result<()> {
        let who = if min { "minss" } else { "maxss" };
        self.scalar_single_arith(instruction, who, Self::fp_min_max(min))
    }

    /// MINPS / MAXPS - packed single-precision minimum/maximum of all four lanes.
    pub(super) fn handle_minmaxps(
        &mut self,
        instruction: &Instruction,
        min: bool,
    ) -> crate::Result<()> {
        let who = if min { "minps" } else { "maxps" };
        let op = if min {
            PackedFpOp::Min
        } else {
            PackedFpOp::Max
        };
        if self.try_packed_fp_vec(instruction, who, true, op)? {
            return Ok(());
        }
        self.packed_single_arith(instruction, who, Self::fp_min_max(min))
    }

    /// SQRTSD - scalar double-precision square root.
    ///
    /// Note the shape difference from the binary forms: the operand read is the
    /// *source*, not the destination, and only the destination's low lane is
    /// written (bits[127:64] are preserved).
    pub(super) fn handle_sqrtsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "sqrtsd")?;
        let src_bits = self.read_scalar_f64_bits(instruction, 1)?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), src_bits);
        let r = fp_sqrt_x86(&mut self.builder, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), r);
        self.set_xmm_lo(dst_idx, bits);
        Ok(())
    }

    /// SQRTSS - scalar single-precision square root; bits[127:32] of the
    /// destination are preserved.
    pub(super) fn handle_sqrtss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "sqrtss")?;
        let src_bits = self.read_scalar_f32_bits(instruction, 1, "sqrtss")?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), src_bits);
        let r = fp_sqrt_x86(&mut self.builder, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), r);
        self.merge_xmm_dword0(dst_idx, bits);
        Ok(())
    }

    /// SQRTPD - packed double-precision square root of both 64-bit lanes of the
    /// source into the destination.
    pub(super) fn handle_sqrtpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "sqrtpd", false, PackedFpOp::Sqrt)? {
            return Ok(());
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "sqrtpd")?;
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        // The unary op is expressed as a binary one that ignores its second
        // argument, reusing the shared bit-level plumbing.
        // The unary op is expressed as a binary one that ignores its second
        // argument, reusing the shared bit-level plumbing.
        let sqrt = |b: &mut FunctionBuilder<'a>, x: Value, _y: Value| fp_sqrt_x86(b, x);
        let res_lo = self.f64_bits_arith(src_lo, src_lo, &sqrt);
        let res_hi = self.f64_bits_arith(src_hi, src_hi, &sqrt);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// SQRTPS - packed single-precision square root of all four lanes of the
    /// source into the destination.
    pub(super) fn handle_sqrtps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.try_packed_fp_vec(instruction, "sqrtps", true, PackedFpOp::Sqrt)? {
            return Ok(());
        }
        let dst_idx = self.xmm_operand_index(instruction, 0, "sqrtps")?;
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let sqrt = |b: &mut FunctionBuilder<'a>, x: Value, _y: Value| fp_sqrt_x86(b, x);
        let res_lo = self.f32_pair_arith(src_lo, src_lo, &sqrt);
        let res_hi = self.f32_pair_arith(src_hi, src_hi, &sqrt);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ROUNDSD / ROUNDSS / ROUNDPD / ROUNDPS - SSE4.1 rounding with the mode
    /// chosen by `imm8` rather than by MXCSR.
    ///
    /// The scalar forms preserve the destination's upper bits, exactly as
    /// `SQRTSD` and `SQRTSS` do, so they reuse the same two writers.
    ///
    /// Reached by every `Math.Floor`, `Math.Ceiling` and `Math.Round` a .NET
    /// program performs: clang and the Mono JIT both emit `ROUNDSD` for
    /// `floor`/`ceil`/`trunc` instead of calling libm, so a runtime without it
    /// stops a managed program at its first rounded number.
    pub(super) fn handle_roundsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "roundsd")?;
        let mode = round_mode(instruction.immediate8());
        let src_bits = self.read_scalar_f64_bits(instruction, 1)?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), src_bits);
        let r = apply_round(&mut self.builder, mode, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), r);
        self.set_xmm_lo(dst_idx, bits);
        Ok(())
    }

    /// ROUNDSS - see [`Self::handle_roundsd`]; bits[127:32] are preserved.
    pub(super) fn handle_roundss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "roundss")?;
        let mode = round_mode(instruction.immediate8());
        let src_bits = self.read_scalar_f32_bits(instruction, 1, "roundss")?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), src_bits);
        let r = apply_round(&mut self.builder, mode, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), r);
        self.merge_xmm_dword0(dst_idx, bits);
        Ok(())
    }

    /// ROUNDPD - both 64-bit lanes, see [`Self::handle_roundsd`].
    pub(super) fn handle_roundpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "roundpd")?;
        let mode = round_mode(instruction.immediate8());
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        // The unary op as a binary one that ignores its second argument, the
        // shape `SQRTPD` already established for this plumbing.
        let round = |b: &mut FunctionBuilder<'a>, x: Value, _y: Value| apply_round(b, mode, x);
        let res_lo = self.f64_bits_arith(src_lo, src_lo, &round);
        let res_hi = self.f64_bits_arith(src_hi, src_hi, &round);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ROUNDPS - all four 32-bit lanes, see [`Self::handle_roundsd`].
    pub(super) fn handle_roundps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "roundps")?;
        let mode = round_mode(instruction.immediate8());
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let round = |b: &mut FunctionBuilder<'a>, x: Value, _y: Value| apply_round(b, mode, x);
        let res_lo = self.f32_pair_arith(src_lo, src_lo, &round);
        let res_hi = self.f32_pair_arith(src_hi, src_hi, &round);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// UCOMISS / COMISS - the single-precision twin of [`Self::handle_ucomisd`]:
    /// compare the low scalar floats and set ZF/PF/CF, clearing OF/SF/AF.
    pub(super) fn handle_ucomiss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "ucomiss")?;
        let a_lo = self.get_xmm_lo(dst_idx);
        let a_bits = self.builder.ins().ireduce(types::I32, a_lo);
        let b_bits = self.read_scalar_f32_bits(instruction, 1, "ucomiss")?;

        let a = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), a_bits);
        let b = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), b_bits);

        let unord = self.builder.ins().fcmp(FloatCC::Unordered, a, b);
        let eq = self.builder.ins().fcmp(FloatCC::Equal, a, b);
        let lt = self.builder.ins().fcmp(FloatCC::LessThan, a, b);

        let zf = self.builder.ins().bor(eq, unord);
        let cf = self.builder.ins().bor(lt, unord);
        self.set_flag(X86Flag::Zero, zf)?;
        self.set_flag(X86Flag::Parity, unord)?;
        self.set_flag(X86Flag::Carry, cf)?;
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(X86Flag::Overflow, zero)?;
        self.set_flag(X86Flag::Sign, zero)?;
        self.set_flag(X86Flag::Auxiliary, zero)?;
        Ok(())
    }

    /// Read an `r/m32` or `r/m64` integer source operand, as used by the
    /// `CVTSI2S*` conversions.
    fn read_int32_or_64_src(
        &mut self,
        instruction: &Instruction,
        op_index: u32,
        who: &str,
    ) -> crate::Result<Value> {
        let src = match self.operand_kind(instruction, op_index) {
            OpKind::Register => {
                let reg = self.operand_register(instruction, op_index);
                self.get_reg_val(reg)?
            }
            OpKind::Memory => {
                let ty = match instruction.memory_size().size() {
                    4 => types::I32,
                    8 => types::I64,
                    other => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("{}-mem-size-{}", who, other),
                        });
                    }
                };
                let ptr = self.get_mem_ptr(instruction, op_index)?;
                self.builder.ins().load(ty, MemFlagsData::new(), ptr, 0)
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("{}-src-{:?}", who, other),
                });
            }
        };

        let src_ty = self.builder.func.dfg.value_type(src);
        if src_ty != types::I32 && src_ty != types::I64 {
            return Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-src-width-{}", who, src_ty.bits()),
            });
        }
        Ok(src)
    }

    /// CVTSI2SD - convert a signed integer to a scalar double-precision float.
    ///
    /// - `CVTSI2SD xmm, r/m32`: `dst[63:0] = (f64)(i32)src`
    /// - `CVTSI2SD xmm, r/m64` (REX.W): `dst[63:0] = (f64)(i64)src`
    ///
    /// Bits [127:64] of the destination are preserved.
    pub(super) fn handle_cvtsi2sd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtsi2sd")?;
        let src = self.read_int32_or_64_src(instruction, 1, "cvtsi2sd")?;
        let f = self.builder.ins().fcvt_from_sint(types::F64, src);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), f);
        self.set_xmm_lo(dst_idx, bits);
        Ok(())
    }

    /// CVTSD2SI / CVTTSD2SI - convert a scalar double-precision float to a
    /// signed integer, rounding to nearest-even or truncating toward zero.
    ///
    /// The double-precision twin of [`Self::handle_cvtss2si`].
    pub(super) fn handle_cvtsd2si(
        &mut self,
        instruction: &Instruction,
        truncate: bool,
    ) -> crate::Result<()> {
        let who = if truncate { "cvttsd2si" } else { "cvtsd2si" };
        self.scalar_fp_to_si(instruction, types::F64, truncate, who)
    }

    /// CVTSS2SD - widen the source's low single to a double in `dst[63:0]`;
    /// bits[127:64] of the destination are preserved.
    pub(super) fn handle_cvtss2sd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtss2sd")?;
        let src_bits = self.read_scalar_f32_bits(instruction, 1, "cvtss2sd")?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), src_bits);
        let r = self.builder.ins().fpromote(types::F64, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), r);
        self.set_xmm_lo(dst_idx, bits);
        Ok(())
    }

    /// CVTSD2SS - narrow the source's low double to a single in `dst[31:0]`;
    /// bits[127:32] of the destination are preserved.
    pub(super) fn handle_cvtsd2ss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtsd2ss")?;
        let src_bits = self.read_scalar_f64_bits(instruction, 1)?;
        let x = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), src_bits);
        let r = self.builder.ins().fdemote(types::F32, x);
        let bits = self
            .builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), r);
        self.merge_xmm_dword0(dst_idx, bits);
        Ok(())
    }

    //  Packed conversions (CVTPD2PS and friends) plus the lane shuffles,
    //  compares and approximate reciprocals compilers emit around them.

    /// Split a 64-bit XMM half into its two dwords, as `I32` values.
    pub(super) fn split_dwords(&mut self, half: Value) -> (Value, Value) {
        let d0 = self.builder.ins().ireduce(types::I32, half);
        let upper = self.builder.ins().ushr_imm_u(half, 32);
        let d1 = self.builder.ins().ireduce(types::I32, upper);
        (d0, d1)
    }

    /// Pack two `I32` dwords into a 64-bit XMM half (`d0` becomes bits[31:0]).
    pub(super) fn pack_dwords(&mut self, d0: Value, d1: Value) -> Value {
        let d0 = self.builder.ins().uextend(types::I64, d0);
        let d1 = self.builder.ins().uextend(types::I64, d1);
        let d1 = self.builder.ins().ishl_imm_u(d1, 32);
        self.builder.ins().bor(d0, d1)
    }

    /// Convert one float to a signed integer with x86's out-of-range behaviour.
    ///
    /// `x` is an `F32` or `F64`; `ty` is `I32` or `I64`. When the result is
    /// unrepresentable - NaN, an infinity, or simply out of range - x86 produces
    /// the *integer indefinite* value (`INT_MIN` for the destination width)
    /// rather than saturating. Cranelift's `fcvt_to_sint_sat` already yields
    /// `INT_MIN` for the too-negative side, so only the "NaN or too positive"
    /// side needs patching, which is exactly the
    /// `UnorderedOrGreaterThanOrEqual` predicate against `2^(width-1)`.
    ///
    /// `truncate` selects the `CVTT*` (toward zero) forms; the plain `CVT*`
    /// forms round per MXCSR.RC, whose reset value - the only rounding mode this
    /// lifter models - is round-to-nearest-even, i.e. Cranelift's `nearest`.
    fn fp_to_sint_x86(&mut self, x: Value, ty: Type, truncate: bool) -> Value {
        let src_ty = self.builder.func.dfg.value_type(x);
        let x = if truncate {
            x
        } else {
            self.builder.ins().nearest(x)
        };

        let sat = self.builder.ins().fcvt_to_sint_sat(ty, x);
        let (limit_f, indefinite) = if ty == types::I32 {
            (2147483648.0f64, i32::MIN as u32 as i64)
        } else {
            (9223372036854775808.0f64, i64::MIN)
        };
        let limit = if src_ty == types::F32 {
            self.builder.ins().f32const(limit_f as f32)
        } else {
            self.builder.ins().f64const(limit_f)
        };

        let invalid = self
            .builder
            .ins()
            .fcmp(FloatCC::UnorderedOrGreaterThanOrEqual, x, limit);
        let indefinite = self.builder.ins().iconst(ty, indefinite);
        self.builder.ins().select(invalid, indefinite, sat)
    }

    /// Shared body of `CVT[T]SS2SI` / `CVT[T]SD2SI`: convert the source's low
    /// scalar float to the GPR destination's width.
    fn scalar_fp_to_si(
        &mut self,
        instruction: &Instruction,
        src_ty: Type,
        truncate: bool,
        who: &str,
    ) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let ty = match dst_reg.size() {
            4 => types::I32,
            8 => types::I64,
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("{}-dst-size-{}", who, other),
                });
            }
        };

        let src_bits = if src_ty == types::F32 {
            self.read_scalar_f32_bits(instruction, 1, who)?
        } else {
            self.read_scalar_f64_bits(instruction, 1)?
        };
        let x = self
            .builder
            .ins()
            .bitcast(src_ty, MemFlagsData::new(), src_bits);

        let res = self.fp_to_sint_x86(x, ty, truncate);
        self.set_reg_val(dst_reg, res)
    }

    /// Convert the `f64` in `bits` (an `I64`) to an `f32`, returning its `I32`
    /// bit pattern.
    fn f64_bits_to_f32_bits(&mut self, bits: Value) -> Value {
        let x = self
            .builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), bits);
        let r = self.builder.ins().fdemote(types::F32, x);
        self.builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), r)
    }

    /// Convert the `f32` in `bits` (an `I32`) to an `f64`, returning its `I64`
    /// bit pattern.
    fn f32_bits_to_f64_bits(&mut self, bits: Value) -> Value {
        let x = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), bits);
        let r = self.builder.ins().fpromote(types::F64, x);
        self.builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), r)
    }

    /// Read the low 64 bits of the `xmm/m64` source of the "half-width source"
    /// conversions (`CVTPS2PD`, `CVTDQ2PD`), which touch only 8 bytes of memory.
    fn read_low_qword_src(&mut self, instruction: &Instruction, who: &str) -> crate::Result<Value> {
        match self.operand_kind(instruction, 1) {
            OpKind::Register => {
                let idx = self.xmm_operand_index(instruction, 1, who)?;
                Ok(self.get_xmm_lo(idx))
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 1)?;
                Ok(self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("{}-src-{:?}", who, other),
            }),
        }
    }

    /// [`Self::read_low_qword_src`] as a 128-bit vector, the consumed quadword
    /// in lane 0 and lane 1 zeroed.
    ///
    /// Used by the "half-width source" conversions in vector mode: their
    /// `xmm/m64` operand must not be read with a full `m128` load, but every
    /// vector op that consumes it (`fvpromote_low`, `swiden_low`) looks only at
    /// the low half anyway.
    fn read_low_qword_src_vec(
        &mut self,
        instruction: &Instruction,
        who: &str,
    ) -> crate::Result<Value> {
        match self.operand_kind(instruction, 1) {
            OpKind::Register => {
                let idx = self.xmm_operand_index(instruction, 1, who)?;
                Ok(self.get_xmm_vec(idx))
            }
            _ => {
                let q = self.read_low_qword_src(instruction, who)?;
                Ok(self.builder.ins().scalar_to_vector(types::I64X2, q))
            }
        }
    }

    /// CVTPD2PS - narrow two packed doubles to two singles in `dst[63:0]`.
    ///
    /// Note the asymmetry with the scalar `CVTSD2SS`: because the result is
    /// narrower than the source, bits[127:64] of the destination are **zeroed**,
    /// not preserved.
    pub(super) fn handle_cvtpd2ps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtpd2ps")?;

        if self.vec_mode() {
            // `fvdemote` narrows `F64X2` into the low two lanes of an `F32X4`
            // and zeroes the upper half - exactly this instruction.
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let x = self.vec_bitcast(src, types::F64X2);
            let f = self.builder.ins().fvdemote(x);
            let r = self.vec_bitcast(f, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let d0 = self.f64_bits_to_f32_bits(src_lo);
        let d1 = self.f64_bits_to_f32_bits(src_hi);
        let res_lo = self.pack_dwords(d0, d1);
        let zero = self.builder.ins().iconst(types::I64, 0);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, zero);
        Ok(())
    }

    /// CVTPS2PD - widen the two singles in the source's low 64 bits (`xmm/m64`)
    /// to two doubles filling the whole destination.
    pub(super) fn handle_cvtps2pd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtps2pd")?;

        if self.vec_mode() {
            let src = self.read_low_qword_src_vec(instruction, "cvtps2pd")?;
            let x = self.vec_bitcast(src, types::F32X4);
            let f = self.builder.ins().fvpromote_low(x);
            let r = self.vec_bitcast(f, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let src = self.read_low_qword_src(instruction, "cvtps2pd")?;
        let (s0, s1) = self.split_dwords(src);

        let res_lo = self.f32_bits_to_f64_bits(s0);
        let res_hi = self.f32_bits_to_f64_bits(s1);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// CVTDQ2PS - convert four packed signed dwords to four singles.
    pub(super) fn handle_cvtdq2ps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtdq2ps")?;

        if self.vec_mode() {
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let a = self.vec_bitcast(src, types::I32X4);
            let f = self.builder.ins().fcvt_from_sint(types::F32X4, a);
            let r = self.vec_bitcast(f, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut halves = [src_lo, src_hi];
        for half in &mut halves {
            let (a, b) = self.split_dwords(*half);
            let a = self.sint32_to_f32_bits(a);
            let b = self.sint32_to_f32_bits(b);
            *half = self.pack_dwords(a, b);
        }

        self.set_xmm_lo(dst_idx, halves[0]);
        self.set_xmm_hi(dst_idx, halves[1]);
        Ok(())
    }

    /// Convert the signed `I32` in `v` to an `f32`, returning its `I32` bits.
    fn sint32_to_f32_bits(&mut self, v: Value) -> Value {
        let f = self.builder.ins().fcvt_from_sint(types::F32, v);
        self.builder
            .ins()
            .bitcast(types::I32, MemFlagsData::new(), f)
    }

    /// CVTPS2DQ / CVTTPS2DQ - convert four packed singles to four signed dwords,
    /// rounding to nearest-even or truncating toward zero.
    pub(super) fn handle_cvtps2dq(
        &mut self,
        instruction: &Instruction,
        truncate: bool,
    ) -> crate::Result<()> {
        let who = if truncate { "cvttps2dq" } else { "cvtps2dq" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.vec_mode() {
            let src = self.read_xmm_src_vec(instruction, 1)?;
            let x = self.vec_bitcast(src, types::F32X4);
            let r = self.vec_fp_to_sint_x86(x, types::I32X4, truncate);
            let r = self.vec_bitcast(r, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut halves = [src_lo, src_hi];
        for half in &mut halves {
            let (a, b) = self.split_dwords(*half);
            let a = self.f32_bits_to_sint32(a, truncate);
            let b = self.f32_bits_to_sint32(b, truncate);
            *half = self.pack_dwords(a, b);
        }

        self.set_xmm_lo(dst_idx, halves[0]);
        self.set_xmm_hi(dst_idx, halves[1]);
        Ok(())
    }

    /// Convert the `f32` bit pattern in `bits` (an `I32`) to a signed `I32`.
    fn f32_bits_to_sint32(&mut self, bits: Value, truncate: bool) -> Value {
        let x = self
            .builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), bits);
        self.fp_to_sint_x86(x, types::I32, truncate)
    }

    /// CVTDQ2PD - convert the two signed dwords in the source's low 64 bits
    /// (`xmm/m64`) to two doubles filling the whole destination.
    pub(super) fn handle_cvtdq2pd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "cvtdq2pd")?;

        if self.vec_mode() {
            let src = self.read_low_qword_src_vec(instruction, "cvtdq2pd")?;
            let a = self.vec_bitcast(src, types::I32X4);
            // Widen the low two dwords to qwords, then convert - `fcvt_from_sint`
            // is lane-preserving, so the widening is what selects the low half.
            let w = self.builder.ins().swiden_low(a);
            let f = self.builder.ins().fcvt_from_sint(types::F64X2, w);
            let r = self.vec_bitcast(f, types::I64X2);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let src = self.read_low_qword_src(instruction, "cvtdq2pd")?;
        let (s0, s1) = self.split_dwords(src);

        let mut out = [s0, s1];
        for v in &mut out {
            let f = self.builder.ins().fcvt_from_sint(types::F64, *v);
            *v = self
                .builder
                .ins()
                .bitcast(types::I64, MemFlagsData::new(), f);
        }

        self.set_xmm_lo(dst_idx, out[0]);
        self.set_xmm_hi(dst_idx, out[1]);
        Ok(())
    }

    /// CVTPD2DQ / CVTTPD2DQ - convert two packed doubles to two signed dwords in
    /// `dst[63:0]`, rounding to nearest-even or truncating toward zero.
    ///
    /// As with [`Self::handle_cvtpd2ps`], the narrowing result zeroes
    /// bits[127:64] of the destination.
    pub(super) fn handle_cvtpd2dq(
        &mut self,
        instruction: &Instruction,
        truncate: bool,
    ) -> crate::Result<()> {
        let who = if truncate { "cvttpd2dq" } else { "cvtpd2dq" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut out = [src_lo, src_hi];
        for v in &mut out {
            let x = self
                .builder
                .ins()
                .bitcast(types::F64, MemFlagsData::new(), *v);
            *v = self.fp_to_sint_x86(x, types::I32, truncate);
        }
        let res_lo = self.pack_dwords(out[0], out[1]);
        let zero = self.builder.ins().iconst(types::I64, 0);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, zero);
        Ok(())
    }

    /// The comparison predicate of a `CMPPS` / `CMPPD` / `CMPSS` / `CMPSD`
    /// `imm8`, as a Cranelift condition.
    ///
    /// Only the low three bits are defined for the legacy SSE encodings. The
    /// negated predicates (`NEQ`, `NLT`, `NLE`) are true for unordered operands,
    /// which is exactly what Cranelift's `Unordered*` conditions express.
    fn cmp_predicate(imm: u8) -> FloatCC {
        match imm & 0b111 {
            0 => FloatCC::Equal,                         // EQ  (ordered)
            1 => FloatCC::LessThan,                      // LT  (ordered)
            2 => FloatCC::LessThanOrEqual,               // LE  (ordered)
            3 => FloatCC::Unordered,                     // UNORD
            4 => FloatCC::NotEqual,                      // NEQ (unordered)
            5 => FloatCC::UnorderedOrGreaterThanOrEqual, // NLT
            6 => FloatCC::UnorderedOrGreaterThan,        // NLE
            _ => FloatCC::Ordered,                       // ORD
        }
    }

    /// The lane operation of the `CMP*` family: an all-ones mask when the
    /// predicate holds, all-zeros otherwise.
    ///
    /// The mask is bitcast back to the lane's float type so this can reuse the
    /// shared `f32`/`f64` lane plumbing, which expects a float result and
    /// bitcasts it to bits again; the round trip folds away.
    fn fp_cmp_mask(cc: FloatCC) -> impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value {
        move |b, x, y| {
            let fty = b.func.dfg.value_type(x);
            let ity = if fty == types::F32 {
                types::I32
            } else {
                types::I64
            };
            let c = b.ins().fcmp(cc, x, y);
            let ones = b.ins().iconst(ity, -1);
            let zeros = b.ins().iconst(ity, 0);
            let mask = b.ins().select(c, ones, zeros);
            b.ins().bitcast(fty, MemFlagsData::new(), mask)
        }
    }

    /// CMPPS - packed single-precision compare; each lane becomes all-ones or
    /// all-zeros according to the `imm8` predicate.
    pub(super) fn handle_cmpps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cc = Self::cmp_predicate(instruction.immediate8());
        if self.try_packed_fp_vec(instruction, "cmpps", true, PackedFpOp::Cmp(cc))? {
            return Ok(());
        }
        self.packed_single_arith(instruction, "cmpps", Self::fp_cmp_mask(cc))
    }

    /// CMPPD - packed double-precision compare.
    pub(super) fn handle_cmppd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cc = Self::cmp_predicate(instruction.immediate8());
        if self.try_packed_fp_vec(instruction, "cmppd", false, PackedFpOp::Cmp(cc))? {
            return Ok(());
        }
        self.packed_double_arith(instruction, "cmppd", Self::fp_cmp_mask(cc))
    }

    /// CMPSS - scalar single-precision compare; bits[127:32] of the destination
    /// are preserved.
    pub(super) fn handle_cmpss(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cc = Self::cmp_predicate(instruction.immediate8());
        self.scalar_single_arith(instruction, "cmpss", Self::fp_cmp_mask(cc))
    }

    /// CMPSD (SSE2 form) - scalar double-precision compare; the high 64 bits of
    /// the destination are preserved.
    pub(super) fn handle_cmpsd_sse(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cc = Self::cmp_predicate(instruction.immediate8());
        self.scalar_double_arith(instruction, "cmpsd", Self::fp_cmp_mask(cc))
    }

    /// SHUFPS - the destination's low two dwords are selected from the
    /// destination, its high two from the source, two `imm8` bits each.
    pub(super) fn handle_shufps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "shufps")?;

        if self.vec_mode() {
            let imm = instruction.immediate8();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // The low two dwords are selected from the destination, the high
            // two from the source (lane numbers 4..7 in the shared mask
            // builder's cross-operand numbering).
            let mask = Self::lane_shuffle_mask(32, |j| {
                let sel = u32::from((imm >> (2 * j)) & 0b11);
                if j < 2 { sel } else { 4 + sel }
            });
            let r = self.vec_shuffle(dst, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let (d0, d1) = self.split_dwords(dst_lo);
        let (d2, d3) = self.split_dwords(dst_hi);
        let (s0, s1) = self.split_dwords(src_lo);
        let (s2, s3) = self.split_dwords(src_hi);
        let d = [d0, d1, d2, d3];
        let s = [s0, s1, s2, s3];

        let imm = instruction.immediate8();
        let sel = |n: u8| ((imm >> (2 * n)) & 0b11) as usize;
        let (r0, r1, r2, r3) = (d[sel(0)], d[sel(1)], s[sel(2)], s[sel(3)]);

        let res_lo = self.pack_dwords(r0, r1);
        let res_hi = self.pack_dwords(r2, r3);
        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// SHUFPD - the destination's low quadword is selected from the destination
    /// by `imm8[0]`, its high quadword from the source by `imm8[1]`.
    pub(super) fn handle_shufpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "shufpd")?;

        if self.vec_mode() {
            let imm = instruction.immediate8();
            let dst = self.get_xmm_vec(dst_idx);
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // Low qword from the destination by imm[0], high qword from the
            // source by imm[1] (source qwords are lanes 2 and 3).
            let mask = Self::lane_shuffle_mask(64, |j| {
                if j == 0 {
                    u32::from(imm & 1)
                } else {
                    2 + u32::from((imm >> 1) & 1)
                }
            });
            let r = self.vec_shuffle(dst, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let imm = instruction.immediate8();
        let res_lo = if imm & 1 != 0 { dst_hi } else { dst_lo };
        let res_hi = if imm & 2 != 0 { src_hi } else { src_lo };

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PSHUFLW / PSHUFHW - shuffle the four words of one half of `xmm2/m128`
    /// into the same half of `xmm1` (two `imm8` bits per destination word); the
    /// other half is copied through unchanged.
    pub(super) fn handle_pshuflhw(
        &mut self,
        instruction: &Instruction,
        high: bool,
    ) -> crate::Result<()> {
        let who = if high { "pshufhw" } else { "pshuflw" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.vec_mode() {
            let imm = instruction.immediate8();
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // Only the shuffled half is permuted; the other half is copied
            // through from the source unchanged. Both come from operand `a`.
            let base = if high { 4 } else { 0 };
            let mask = Self::lane_shuffle_mask(16, |j| {
                if (j >= 4) == high {
                    base + u32::from((imm >> (2 * (j - base))) & 0b11)
                } else {
                    j
                }
            });
            let r = self.vec_shuffle(src, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let shuffled = if high { src_hi } else { src_lo };
        let words: Vec<Value> = (0..4)
            .map(|i| self.extract_lane(shuffled, i * 16, 16, false))
            .collect();

        let imm = instruction.immediate8();
        let mut acc = self.builder.ins().iconst(types::I64, 0);
        for out in 0..4u8 {
            let w = words[((imm >> (2 * out)) & 0b11) as usize];
            let placed = if out == 0 {
                w
            } else {
                self.builder.ins().ishl_imm_u(w, i64::from(out) * 16)
            };
            acc = self.builder.ins().bor(acc, placed);
        }

        if high {
            self.set_xmm_lo(dst_idx, src_lo);
            self.set_xmm_hi(dst_idx, acc);
        } else {
            self.set_xmm_lo(dst_idx, acc);
            self.set_xmm_hi(dst_idx, src_hi);
        }
        Ok(())
    }

    /// The lane operation of `RCP*` / `RSQRT*`, applied to the *source* lane
    /// (`y`) - these are unary instructions expressed through the shared binary
    /// lane plumbing, so the destination lane `x` is ignored.
    ///
    /// Hardware computes an approximation with a relative error up to
    /// `1.5 * 2^-12`; the exact quotient emitted here is within that bound but
    /// will not be bit-identical to a real CPU. Compilers only emit these under
    /// fast-math, where that difference is already licensed.
    ///
    /// The square root goes through [`fp_sqrt_x86`] rather than a bare
    /// Cranelift `sqrt`: a negative lane makes `RSQRT*` *generate* a NaN, and
    /// that NaN must be x86's sign-set floating indefinite rather than the
    /// host's default NaN. The following `fdiv` merely propagates it.
    fn fp_reciprocal(sqrt: bool) -> impl Fn(&mut FunctionBuilder<'a>, Value, Value) -> Value {
        move |b, _x, y| {
            let y = if sqrt { fp_sqrt_x86(b, y) } else { y };
            let one = b.ins().f32const(1.0);
            b.ins().fdiv(one, y)
        }
    }

    /// RCPPS / RSQRTPS - packed approximate reciprocal (of the square root) of
    /// all four single-precision lanes of the source.
    pub(super) fn handle_rcpps(
        &mut self,
        instruction: &Instruction,
        sqrt: bool,
    ) -> crate::Result<()> {
        let who = if sqrt { "rsqrtps" } else { "rcpps" };
        if self.try_packed_fp_vec(instruction, who, true, PackedFpOp::Rcp { sqrt })? {
            return Ok(());
        }
        self.packed_single_arith(instruction, who, Self::fp_reciprocal(sqrt))
    }

    /// RCPSS / RSQRTSS - scalar approximate reciprocal (of the square root);
    /// bits[127:32] of the destination are preserved.
    pub(super) fn handle_rcpss(
        &mut self,
        instruction: &Instruction,
        sqrt: bool,
    ) -> crate::Result<()> {
        let who = if sqrt { "rsqrtss" } else { "rcpss" };
        self.scalar_single_arith(instruction, who, Self::fp_reciprocal(sqrt))
    }

    /// MOVSLDUP / MOVSHDUP - duplicate the source's even (`sldup`) or odd
    /// (`shdup`) dwords: `[d0,d0,d2,d2]` or `[d1,d1,d3,d3]`.
    pub(super) fn handle_movsldup(
        &mut self,
        instruction: &Instruction,
        odd: bool,
    ) -> crate::Result<()> {
        let who = if odd { "movshdup" } else { "movsldup" };
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;

        if self.vec_mode() {
            let src = self.read_xmm_src_vec(instruction, 1)?;
            // [d0,d0,d2,d2] or [d1,d1,d3,d3].
            let mask = Self::lane_shuffle_mask(32, |j| (j & !1) + u32::from(odd));
            let r = self.vec_shuffle(src, src, mask);
            self.set_xmm_vec(dst_idx, r);
            return Ok(());
        }

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let mut halves = [src_lo, src_hi];
        for half in &mut halves {
            let (even, oddv) = self.split_dwords(*half);
            let d = if odd { oddv } else { even };
            *half = self.pack_dwords(d, d);
        }

        self.set_xmm_lo(dst_idx, halves[0]);
        self.set_xmm_hi(dst_idx, halves[1]);
        Ok(())
    }
}
