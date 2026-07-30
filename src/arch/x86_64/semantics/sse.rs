//! SSE / SSE2 / XMM instruction semantics.
//!
//! XMM registers have two interchangeable backing representations,
//! selected by [`crate::lifter::LifterConfig::use_vector_xmm`]:
//!  - scalar (default): two `I64` SSA variables per register (lo + hi).
//!    Portable to AOT targets without vector lowering (e.g. RISC-V64).
//!  - vector: a single native `I64X2` SSA variable per register, which
//!    lowers directly to host XMM registers on x86-64.
//!
//! Whole-register bitwise handlers (movaps / xorps / andps / orps /
//! andnps and their `pd`/integer aliases) take a vector fast path when
//! the option is enabled; scalar move/extract helpers fall back to lo/hi
//! lane access via [`crate::lifter::Lifter::get_xmm_lo`] /
//! [`crate::lifter::Lifter::get_xmm_hi`] which transparently handle both
//! representations.

use crate::MicroliftError;
use crate::lifter::{Lifter, X86Flag};
use crate::arch::x86_64::semantics::helpers::xmm_register_index;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

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
    fn bits(self) -> u32 {
        match self {
            PackedWidth::Byte => 8,
            PackedWidth::Word => 16,
            PackedWidth::Dword => 32,
            PackedWidth::Qword => 64,
        }
    }

    /// The 128-bit Cranelift vector type with these lanes.
    fn vector_type(self) -> Type {
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
    fn read_xmm_src(
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
                let lo = self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0);
                let hi = self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 8);
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
    fn read_xmm_src_vec(
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
                            self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0)
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
                        let src32 = self.builder.ins().band_imm(src_lo, 0xFFFF_FFFF);
                        let merged_lo = self.builder.ins().bor(cleared, src32);
                        // hi 64 bits of dst are preserved (no write needed).
                        self.set_xmm_lo(dst_idx, merged_lo);
                    }
                    OpKind::Memory => {
                        // MOVSS xmm, m32 - load 32 bits, zero upper 96 bits.
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        let val32 = self.builder.ins().load(types::I32, MemFlagsData::new(), ptr, 0);
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
                            self.builder.ins().load(types::I32, MemFlagsData::new(), ptr, 0)
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
                let hi = self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0);
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
                let v = self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0);
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
                    PackedShiftOp::Srl => self.builder.ins().ushr_imm(a, c),
                    PackedShiftOp::Sll => self.builder.ins().ishl_imm(a, c),
                    PackedShiftOp::Sra => self.builder.ins().sshr_imm(a, c),
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
                        let too_big = self
                            .builder
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
                let too_big = self.builder.ins().icmp_imm(
                    IntCC::UnsignedGreaterThan,
                    cv,
                    bits as i64 - 1,
                );
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
                    self.builder.ins().ishl_imm(a, c)
                } else {
                    self.builder.ins().ushr_imm(a, c)
                };
                self.builder.ins().band_imm(shifted, mask)
            }
            ShiftCount::Dynamic(cv) => {
                // Build the same mask at runtime: shift the lowest lane's mask,
                // then double it up into every lane.
                let lane_const = self.builder.ins().iconst(types::I64, lane as i64);
                let mut mask = if left {
                    let s = self.builder.ins().ishl(lane_const, cv);
                    self.builder.ins().band_imm(s, lane as i64)
                } else {
                    self.builder.ins().ushr(lane_const, cv)
                };
                let mut shift = bits;
                while shift < 64 {
                    let up = self.builder.ins().ishl_imm(mask, shift as i64);
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
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        // The four source dwords, each isolated into the low 32 bits of an I64.
        let dwords = [
            self.builder.ins().band_imm(src_lo, 0xFFFF_FFFF),
            self.builder.ins().ushr_imm(src_lo, 32),
            self.builder.ins().band_imm(src_hi, 0xFFFF_FFFF),
            self.builder.ins().ushr_imm(src_hi, 32),
        ];

        let imm = instruction.immediate8();
        let pick = |j: u32| dwords[((imm >> (2 * j)) & 0b11) as usize];
        let (r0, r1, r2, r3) = (pick(0), pick(1), pick(2), pick(3));

        let r1_hi = self.builder.ins().ishl_imm(r1, 32);
        let res_lo = self.builder.ins().bor(r0, r1_hi);
        let r3_hi = self.builder.ins().ishl_imm(r3, 32);
        let res_hi = self.builder.ins().bor(r2, r3_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Resolve operand `op_index` to an XMM register index, or error with a
    /// handler-specific mnemonic tag.
    fn xmm_operand_index(
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
                        let lo = self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0);
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
                Ok(self.builder.ins().load(types::I64, MemFlagsData::new(), ptr, 0))
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("scalar-f64-src-op{}-{:?}", op_index, other),
            }),
        }
    }

    /// PUNPCKLDQ - interleave the low doublewords of dst and src.
    ///
    /// The result's four dwords are `[dst.d0, src.d0, dst.d1, src.d1]`, so only
    /// the low quadword of each operand participates.
    pub(super) fn handle_punpckldq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "punpckldq")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let (src_lo, _src_hi) = self.read_xmm_src(instruction, 1)?;

        let dst_d0 = self.builder.ins().band_imm(dst_lo, 0xFFFF_FFFF);
        let dst_d1 = self.builder.ins().ushr_imm(dst_lo, 32);
        let src_d0 = self.builder.ins().band_imm(src_lo, 0xFFFF_FFFF);
        let src_d1 = self.builder.ins().ushr_imm(src_lo, 32);

        let src_d0_hi = self.builder.ins().ishl_imm(src_d0, 32);
        let res_lo = self.builder.ins().bor(dst_d0, src_d0_hi);
        let src_d1_hi = self.builder.ins().ishl_imm(src_d1, 32);
        let res_hi = self.builder.ins().bor(dst_d1, src_d1_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// UNPCKHPD - unpack and interleave the high 64-bit lanes:
    /// `dst[63:0] = dst[127:64]`, `dst[127:64] = src[127:64]`.
    pub(super) fn handle_unpckhpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "unpckhpd")?;
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (_src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        self.set_xmm_lo(dst_idx, dst_hi);
        self.set_xmm_hi(dst_idx, src_hi);
        Ok(())
    }

    /// SUBPD - packed double-precision subtract of both 64-bit lanes.
    pub(super) fn handle_subpd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "subpd")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.fsub_f64_bits(dst_lo, src_lo);
        let res_hi = self.fsub_f64_bits(dst_hi, src_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// Subtract two `f64` values given as raw `I64` bit patterns, returning the
    /// difference's bits.
    fn fsub_f64_bits(&mut self, a_bits: Value, b_bits: Value) -> Value {
        let a = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), a_bits);
        let b = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), b_bits);
        let r = self.builder.ins().fsub(a, b);
        self.builder.ins().bitcast(types::I64, MemFlagsData::new(), r)
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
        op: impl FnOnce(&mut FunctionBuilder<'a>, Value, Value) -> Value,
    ) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, who)?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let src_lo = self.read_scalar_f64_bits(instruction, 1)?;

        let a = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), dst_lo);
        let b = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), src_lo);
        let r = op(&mut self.builder, a, b);
        let res_lo = self.builder.ins().bitcast(types::I64, MemFlagsData::new(), r);

        // hi 64 bits are preserved (no write needed).
        self.set_xmm_lo(dst_idx, res_lo);
        Ok(())
    }

    /// ADDSD - scalar double-precision add of the low 64 bits; the high 64 bits
    /// of the destination are preserved.
    pub(super) fn handle_addsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "addsd", |b, x, y| b.ins().fadd(x, y))
    }

    /// MULSD - scalar double-precision multiply of the low 64 bits; the high 64
    /// bits of the destination are preserved.
    pub(super) fn handle_mulsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "mulsd", |b, x, y| b.ins().fmul(x, y))
    }

    /// DIVSD - scalar double-precision divide of the low 64 bits (`dst / src`);
    /// the high 64 bits of the destination are preserved.
    pub(super) fn handle_divsd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.scalar_double_arith(instruction, "divsd", |b, x, y| b.ins().fdiv(x, y))
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

        let a = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), a_bits);
        let b = self.builder.ins().bitcast(types::F64, MemFlagsData::new(), b_bits);

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
    fn vec_bitcast(&mut self, v: Value, ty: Type) -> Value {
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
        let a_low = self.builder.ins().band_imm(a, !high);
        let b_low = self.builder.ins().band_imm(b, !high);
        let sum = self.builder.ins().iadd(a_low, b_low);
        let xor = self.builder.ins().bxor(a, b);
        let high_bits = self.builder.ins().band_imm(xor, high);
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
        let b_low = self.builder.ins().band_imm(b, !high_imm);
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
            let lane_val = self.builder.ins().band_imm(clamped, lane_mask);
            let placed = if shift == 0 {
                lane_val
            } else {
                self.builder.ins().ishl_imm(lane_val, shift)
            };
            acc = self.builder.ins().bor(acc, placed);
        }
        acc
    }

    /// Widen the `lane_bits`-wide lane at bit offset `shift` of `v` into a full
    /// `I64`, sign- or zero-extending as requested.
    fn extract_lane(&mut self, v: Value, shift: i64, lane_bits: u32, signed: bool) -> Value {
        if signed {
            let left = 64 - shift - lane_bits as i64;
            let aligned = if left == 0 {
                v
            } else {
                self.builder.ins().ishl_imm(v, left)
            };
            self.builder.ins().sshr_imm(aligned, 64 - lane_bits as i64)
        } else {
            let shifted = if shift == 0 {
                v
            } else {
                self.builder.ins().ushr_imm(v, shift)
            };
            let mask = match lane_bits {
                64 => return shifted,
                bits => ((1u64 << bits) - 1) as i64,
            };
            self.builder.ins().band_imm(shifted, mask)
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
}
