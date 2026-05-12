//! SSE / SSE2 / XMM instruction semantics.
//!
//! XMM registers are tracked as two I64 SSA variables each (lo / hi).
//! This sidesteps the need for 128-bit SIMD types in Cranelift while still
//! correctly modelling the data flow for the most common patterns.

use crate::MicroliftError;
use crate::lifter::Lifter;
use crate::semantics::helpers::xmm_register_index;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

impl<'a> Lifter<'a> {
    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Load the low and high 64-bit halves of an XMM source operand.
    ///
    /// Supports:
    ///  - XMM register source (reads SSA variables)
    ///  - Memory source (two 8-byte loads at `ptr` and `ptr+8`)
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
                    // GPR used as source (e.g. MOVQ xmm, r64): zero-extend to 128 bits.
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

    /// Write the low and high 64-bit halves to an XMM destination operand.
    ///
    /// Supports:
    ///  - XMM register destination (writes SSA variables)
    ///  - Memory destination (two 8-byte stores at `ptr` and `ptr+8`)
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
                    // GPR destination: only use the low 64 bits.
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

    // ── Instruction handlers ─────────────────────────────────────────────────

    /// ENDBR64 — CET end-branch marker, a NOP for lifting purposes.
    pub(super) fn handle_endbr64(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        Ok(())
    }

    /// MOVAPS / MOVAPD / MOVUPS / MOVUPD / MOVDQA / MOVDQU — 128-bit XMM move.
    pub(super) fn handle_movaps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (lo, hi) = self.read_xmm_src(instruction, 1)?;
        self.write_xmm_dst(instruction, 0, lo, hi)
    }

    /// MOVQ (SSE2 form) — move 64 bits to/from an XMM register.
    ///
    /// When the destination is an XMM register the high 64 bits are zeroed
    /// (matching real hardware behaviour).
    pub(super) fn handle_movq_sse(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zero = self.builder.ins().iconst(types::I64, 0);

        match self.operand_kind(instruction, 0) {
            OpKind::Register => {
                let dst_reg = self.operand_register(instruction, 0);
                if let Some(dst_idx) = xmm_register_index(dst_reg) {
                    // MOVQ xmm, xmm/r64/m64
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
                    // MOVQ r64, xmm
                    let src_xmm = self.operand_register(instruction, 1);
                    let lo = if let Some(src_idx) = xmm_register_index(src_xmm) {
                        self.get_xmm_lo(src_idx)
                    } else {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: "movq-r64-non-xmm-src".into(),
                        });
                    };
                    self.set_reg_val(dst_reg, lo)?;
                }
            }
            OpKind::Memory => {
                // MOVQ m64, xmm
                let src_xmm = self.operand_register(instruction, 1);
                let lo = if let Some(src_idx) = xmm_register_index(src_xmm) {
                    self.get_xmm_lo(src_idx)
                } else {
                    return Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: "movq-mem-non-xmm-src".into(),
                    });
                };
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

    /// XORPS / XORPD — bitwise XOR of two 128-bit XMM values.
    pub(super) fn handle_xorps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!("xorps-dst-not-xmm-{:?}", dst_reg),
            }
        })?;

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.builder.ins().bxor(dst_lo, src_lo);
        let res_hi = self.builder.ins().bxor(dst_hi, src_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// PXOR — same bit-level semantics as XORPS.
    pub(super) fn handle_pxor(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_xorps(instruction)
    }

    /// ANDPS / ANDPD / PAND — bitwise AND of two 128-bit XMM values.
    pub(super) fn handle_andps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!("andps-dst-not-xmm-{:?}", dst_reg),
            }
        })?;

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.builder.ins().band(dst_lo, src_lo);
        let res_hi = self.builder.ins().band(dst_hi, src_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ORPS / ORPD / POR — bitwise OR of two 128-bit XMM values.
    pub(super) fn handle_orps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!("orps-dst-not-xmm-{:?}", dst_reg),
            }
        })?;

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.builder.ins().bor(dst_lo, src_lo);
        let res_hi = self.builder.ins().bor(dst_hi, src_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// ANDNPS / ANDNPD / PANDN — bitwise AND-NOT: dst = ~dst & src.
    pub(super) fn handle_andnps(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst_reg = self.operand_register(instruction, 0);
        let dst_idx = xmm_register_index(dst_reg).ok_or_else(|| {
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!("andnps-dst-not-xmm-{:?}", dst_reg),
            }
        })?;

        let dst_lo = self.get_xmm_lo(dst_idx);
        let dst_hi = self.get_xmm_hi(dst_idx);

        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;

        let res_lo = self.builder.ins().band_not(src_lo, dst_lo);
        let res_hi = self.builder.ins().band_not(src_hi, dst_hi);

        self.set_xmm_lo(dst_idx, res_lo);
        self.set_xmm_hi(dst_idx, res_hi);
        Ok(())
    }

    /// MOVSS — move scalar single-precision float (low 32 bits of XMM).
    ///
    /// MOVSS xmm1, xmm2: merges low 32 bits of src into dst, preserving
    /// the upper 96 bits of dst.
    /// MOVSS xmm, m32: loads 32 bits, zeroes upper 96 bits of dst.
    /// MOVSS m32, xmm: stores low 32 bits of src to memory.
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
                        // MOVSS xmm1, xmm2 — merge low 32 bits, preserve upper 96
                        let src_reg = self.operand_register(instruction, 1);
                        let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                            MicroliftError::UnsupportedInstruction {
                                mnemonic: "movss-src-not-xmm".into(),
                            }
                        })?;
                        let src_lo = self.get_xmm_lo(src_idx);
                        let dst_lo = self.get_xmm_lo(dst_idx);

                        // Replace bits [31:0] of dst_lo with src_lo[31:0]
                        let mask = self.builder.ins().iconst(types::I64, !0xFFFF_FFFFi64);
                        let cleared = self.builder.ins().band(dst_lo, mask);
                        let src32 = self.builder.ins().band_imm(src_lo, 0xFFFF_FFFF);
                        let merged = self.builder.ins().bor(cleared, src32);
                        self.set_xmm_lo(dst_idx, merged);
                        // hi unchanged
                    }
                    OpKind::Memory => {
                        // MOVSS xmm, m32 — load 32 bits, zero upper 96
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        let val32 =
                            self.builder
                                .ins()
                                .load(types::I32, MemFlags::new(), ptr, 0);
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
                // MOVSS m32, xmm — store low 32 bits
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

    /// MOVSD (SSE2 form) — move scalar double-precision float (low 64 bits of XMM).
    ///
    /// MOVSD xmm1, xmm2: merges low 64 bits of src into dst, preserving
    /// the upper 64 bits of dst.
    /// MOVSD xmm, m64: loads 64 bits, zeroes high 64 bits of dst.
    /// MOVSD m64, xmm: stores low 64 bits of src to memory.
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
                        // MOVSD xmm1, xmm2 — copy lo64 of src, preserve hi64 of dst
                        let src_reg = self.operand_register(instruction, 1);
                        let src_idx = xmm_register_index(src_reg).ok_or_else(|| {
                            MicroliftError::UnsupportedInstruction {
                                mnemonic: "movsd-sse-src-not-xmm".into(),
                            }
                        })?;
                        let src_lo = self.get_xmm_lo(src_idx);
                        self.set_xmm_lo(dst_idx, src_lo);
                        // hi64 of dst is preserved — no write needed
                    }
                    OpKind::Memory => {
                        // MOVSD xmm, m64 — load 64 bits, zero high 64 bits
                        let ptr = self.get_mem_ptr(instruction, 1)?;
                        let lo =
                            self.builder
                                .ins()
                                .load(types::I64, MemFlags::new(), ptr, 0);
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
                // MOVSD m64, xmm — store low 64 bits
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
}
