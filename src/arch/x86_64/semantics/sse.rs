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
                let lo = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
                let hi = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 8);
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
                self.builder.ins().store(MemFlags::new(), lo, ptr, 0);
                self.builder.ins().store(MemFlags::new(), hi, ptr, 8);
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
                    .load(types::I64X2, MemFlags::new(), ptr, 0))
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
                self.builder.ins().store(MemFlags::new(), val, ptr, 0);
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
                            self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0)
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
                self.builder.ins().store(MemFlags::new(), lo, ptr, 0);
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
                        let val32 = self.builder.ins().load(types::I32, MemFlags::new(), ptr, 0);
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
                self.builder.ins().store(MemFlags::new(), val32, ptr, 0);
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
                            self.builder.ins().load(types::I32, MemFlags::new(), ptr, 0)
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
                self.builder.ins().store(MemFlags::new(), val32, ptr, 0);
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
                let hi = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
                self.set_xmm_hi(dst_idx, hi);
            }
            OpKind::Memory => {
                // Store form: dst is m64, src is XMM.
                let src_idx = self.xmm_operand_index(instruction, 1, "movhps")?;
                let hi = self.get_xmm_hi(src_idx);
                let ptr = self.get_mem_ptr(instruction, 0)?;
                self.builder.ins().store(MemFlags::new(), hi, ptr, 0);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("movhps-dst-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// PSRLQ - packed shift right logical of the two 64-bit lanes.
    ///
    /// - `PSRLQ xmm, imm8`: shift both lanes right by the immediate.
    /// - `PSRLQ xmm, xmm/m128`: the shift count is the low 64 bits of the source.
    ///
    /// In both cases a count of 64 or more produces all-zero lanes (rather than
    /// the modulo-width behaviour of a raw shift).
    pub(super) fn handle_psrlq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_idx = self.xmm_operand_index(instruction, 0, "psrlq")?;
        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (res_lo, res_hi) = match self.operand_kind(instruction, 1) {
            OpKind::Immediate8 => {
                let count = instruction.immediate8() as i64;
                if count >= 64 {
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    (zero, zero)
                } else {
                    (
                        self.builder.ins().ushr_imm(dst_lo, count),
                        self.builder.ins().ushr_imm(dst_hi, count),
                    )
                }
            }
            _ => {
                // Count = low 64 bits of the xmm/m128 source.
                let count = match self.operand_kind(instruction, 1) {
                    OpKind::Register => {
                        let src_idx = self.xmm_operand_index(instruction, 1, "psrlq")?;
                        self.get_xmm_lo(src_idx)
                    }
                    OpKind::Memory => {
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0)
                    }
                    other => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("psrlq-src-{:?}", other),
                        });
                    }
                };
                let shifted_lo = self.builder.ins().ushr(dst_lo, count);
                let shifted_hi = self.builder.ins().ushr(dst_hi, count);
                let zero = self.builder.ins().iconst(types::I64, 0);
                // A count > 63 zeroes the lane (the raw shift would wrap mod 64).
                let too_big = self
                    .builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, count, 63);
                (
                    self.builder.ins().select(too_big, zero, shifted_lo),
                    self.builder.ins().select(too_big, zero, shifted_hi),
                )
            }
        };

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
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
                        let lo = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
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
                self.builder.ins().store(MemFlags::new(), lo, ptr, 0);
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
                Ok(self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0))
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
        let a = self.builder.ins().bitcast(types::F64, MemFlags::new(), a_bits);
        let b = self.builder.ins().bitcast(types::F64, MemFlags::new(), b_bits);
        let r = self.builder.ins().fsub(a, b);
        self.builder.ins().bitcast(types::I64, MemFlags::new(), r)
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

        let a = self.builder.ins().bitcast(types::F64, MemFlags::new(), dst_lo);
        let b = self.builder.ins().bitcast(types::F64, MemFlags::new(), src_lo);
        let r = op(&mut self.builder, a, b);
        let res_lo = self.builder.ins().bitcast(types::I64, MemFlags::new(), r);

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

        let a = self.builder.ins().bitcast(types::F64, MemFlags::new(), a_bits);
        let b = self.builder.ins().bitcast(types::F64, MemFlags::new(), b_bits);

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
}
