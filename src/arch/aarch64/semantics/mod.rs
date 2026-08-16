//! aarch64 instruction semantics (vertical slice).
//!
//! Covers a representative subset proving the `GuestArch` abstraction
//! end-to-end: moves (MOVZ/MOVK/MOVN, MOV-via-ORR), integer data-processing
//! (ADD/SUB/ADDS/SUBS incl. CMP, AND/ORR/EOR/ANDS, ORN/EON/BIC incl. MVN),
//! multiply (MUL/MADD/MSUB), division (UDIV/SDIV with A64's non-trapping
//! semantics), variable shifts (LSLV/LSRV/ASRV/RORV), bit ops (CLZ/RBIT/REV),
//! conditional select family (CSEL/CSINC/CSINV/CSNEG), PC-relative addressing
//! (ADR/ADRP), base+offset and pair loads/stores (LDR/STR/LDP/STP), branches
//! (B/B.cond/CBZ/CBNZ/TBZ/TBNZ), RET, NOP, and SVC.
//!
//! Implemented as inherent methods on the concrete `Lifter64`; the dispatch
//! entry point is reached from `GuestArch::lift_instruction`.

use crate::MicroliftError;
use crate::arch::aarch64::Lifter64;
use crate::arch::aarch64::decode::A64Instr;
use crate::arch::aarch64::regs::{Nzcv, ZR};
use crate::core::exit::{ExitReason, TransferKind};
use cranelift::prelude::*;
use yaxpeax_arm::armv8::a64::{Opcode, Operand, ShiftStyle, SizeCode};
use crate::arch::aarch64::Aarch64;
use crate::core::GuestArch;

/// Decode a register operand into `(reg_number, size, sp_form)`.
/// `sp_form` is `true` when register 31 denotes SP rather than the zero register.
fn decode_reg(op: Operand) -> Option<(u16, SizeCode, bool)> {
    match op {
        Operand::Register(sz, n) => Some((n, sz, false)),
        Operand::RegisterOrSP(sz, n) => Some((n, sz, true)),
        _ => None,
    }
}

fn size_ty(sz: SizeCode) -> Type {
    match sz {
        SizeCode::X => types::I64,
        SizeCode::W => types::I32,
    }
}

impl<'a> Lifter64<'a> {
    pub fn handle_instruction(&mut self, instr: &A64Instr) -> crate::Result<()> {
        let op0 = instr.inst.operands[0];
        let op1 = instr.inst.operands[1];
        let op2 = instr.inst.operands[2];
        let op3 = instr.inst.operands[3];

        match instr.inst.opcode {
            Opcode::MOVZ => self.handle_movz(op0, op1),
            Opcode::MOVN => self.handle_movn(op0, op1),
            Opcode::MOVK => self.handle_movk(op0, op1),
            Opcode::ADD => self.handle_addsub(op0, op1, op2, false, false),
            Opcode::ADDS => self.handle_addsub(op0, op1, op2, false, true),
            Opcode::SUB => self.handle_addsub(op0, op1, op2, true, false),
            Opcode::SUBS => self.handle_addsub(op0, op1, op2, true, true),
            Opcode::AND => self.handle_logical(op0, op1, op2, LogicOp::And),
            Opcode::ORR => self.handle_logical(op0, op1, op2, LogicOp::Or),
            Opcode::EOR => self.handle_logical(op0, op1, op2, LogicOp::Xor),
            // ANDS also covers the TST alias (ANDS to the zero register).
            Opcode::ANDS => self.handle_ands(op0, op1, op2),
            Opcode::ORN => self.handle_logical_neg(op0, op1, op2, LogicOp::Or),
            Opcode::EON => self.handle_logical_neg(op0, op1, op2, LogicOp::Xor),
            // BIC/BICS: AND with the inverted second operand; BICS also sets NZ.
            Opcode::BIC => self.handle_logical_neg(op0, op1, op2, LogicOp::And),
            Opcode::CSEL => self.handle_csel(op0, op1, op2, op3),
            Opcode::CSINC => self.handle_cs_variant(op0, op1, op2, op3, CsVariant::Inc),
            Opcode::CSINV => self.handle_cs_variant(op0, op1, op2, op3, CsVariant::Inv),
            Opcode::CSNEG => self.handle_cs_variant(op0, op1, op2, op3, CsVariant::Neg),
            Opcode::MADD => self.handle_madd(op0, op1, op2, op3, false),
            Opcode::MSUB => self.handle_madd(op0, op1, op2, op3, true),
            Opcode::UDIV => self.handle_div(op0, op1, op2, false),
            Opcode::SDIV => self.handle_div(op0, op1, op2, true),
            Opcode::LSLV => self.handle_shift_var(op0, op1, op2, ShiftStyle::LSL),
            Opcode::LSRV => self.handle_shift_var(op0, op1, op2, ShiftStyle::LSR),
            Opcode::ASRV => self.handle_shift_var(op0, op1, op2, ShiftStyle::ASR),
            Opcode::RORV => self.handle_shift_var(op0, op1, op2, ShiftStyle::ROR),
            Opcode::CLZ => self.handle_bit_unary(op0, op1, BitUnary::Clz),
            Opcode::RBIT => self.handle_bit_unary(op0, op1, BitUnary::Rbit),
            Opcode::REV => self.handle_bit_unary(op0, op1, BitUnary::Rev),
            Opcode::ADRP => self.handle_adrp(instr, op0),
            Opcode::ADR => self.handle_adr(instr, op0),
            Opcode::LDR => self.handle_ldr(op0, op1),
            Opcode::STR => self.handle_str(op0, op1),
            Opcode::STP => self.handle_stp(op0, op1, op2),
            Opcode::LDP => self.handle_ldp(op0, op1, op2),
            Opcode::B => self.handle_b(instr),
            Opcode::Bcc(cond) => self.handle_bcc(instr, cond),
            Opcode::CBZ => self.handle_cb(instr, op0, true),
            Opcode::CBNZ => self.handle_cb(instr, op0, false),
            Opcode::TBZ => self.handle_tb(instr, op0, op1, true),
            Opcode::TBNZ => self.handle_tb(instr, op0, op1, false),
            Opcode::BL => self.handle_bl(instr),
            Opcode::BLR => self.handle_blr(op0),
            Opcode::RET => self.handle_ret(op0),
            Opcode::SVC => self.handle_svc(),
            Opcode::HINT => Ok(()),
            // Pointer authentication: without a PAC-enforcing environment the
            // sign/authenticate pair round-trips to identity, so we model both
            // as no-ops (the link register is left unchanged).
            Opcode::PACIASP | Opcode::AUTIASP => Ok(()),
            other => {
                if self.config.interpreter_fallback {
                    self.exit_unimplemented()
                } else {
                    Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: format!("{other:?}"),
                    })
                }
            }
        }
    }

    /// Read register `num` at width `ty`. Register 31 is the zero register
    /// (yields 0) unless `sp_form` is set, in which case it is SP.
    fn read_reg_val(&mut self, num: u16, sp_form: bool, ty: Type) -> Value {
        if num == ZR && !sp_form {
            return self.builder.ins().iconst(ty, 0);
        }
        let var = if num == ZR {
            self.state.sp
        } else {
            self.state.gprs[&num]
        };
        let full = self.builder.use_var(var);
        if ty == types::I32 {
            self.builder.ins().ireduce(types::I32, full)
        } else {
            full
        }
    }

    /// Write `val` (of width `ty`) to register `num`. A W-width write
    /// zero-extends into the full 64-bit register. Writes to the zero register
    /// are discarded.
    fn write_reg_val(&mut self, num: u16, sp_form: bool, ty: Type, val: Value) {
        if num == ZR && !sp_form {
            return;
        }
        let full = if ty == types::I32 {
            self.builder.ins().uextend(types::I64, val)
        } else {
            val
        };
        if num == ZR {
            self.builder.def_var(self.state.sp, full);
            self.state.dirty_sp = true;
        } else {
            self.builder.def_var(self.state.gprs[&num], full);
            self.state.dirty_gprs.insert(num);
        }
    }

    /// Read a source operand (register, immediate, or shifted register) as a
    /// value of width `ty`.
    fn read_source(&mut self, op: Operand, ty: Type) -> crate::Result<Value> {
        match op {
            Operand::Register(_, n) => Ok(self.read_reg_val(n, false, ty)),
            Operand::RegisterOrSP(_, n) => Ok(self.read_reg_val(n, true, ty)),
            Operand::Immediate(v) => Ok(self.builder.ins().iconst(ty, v as i64)),
            Operand::Imm64(v) => Ok(self.builder.ins().iconst(ty, v as i64)),
            Operand::ImmShift(imm, shift) => Ok(self
                .builder
                .ins()
                .iconst(ty, ((imm as u64) << shift) as i64)),
            Operand::RegShift(style, amount, _sz, n) => {
                let base = self.read_reg_val(n, false, ty);
                Ok(match style {
                    ShiftStyle::LSL => self.builder.ins().ishl_imm_u(base, amount as i64),
                    ShiftStyle::LSR => self.builder.ins().ushr_imm_u(base, amount as i64),
                    ShiftStyle::ASR => self.builder.ins().sshr_imm_u(base, amount as i64),
                    ShiftStyle::ROR => self.builder.ins().rotr_imm_u(base, amount as i64),
                    _ => {
                        return Err(MicroliftError::UnsupportedInstruction {
                            mnemonic: format!("shift-{style:?}"),
                        });
                    }
                })
            }
            _ => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("source-operand-{op:?}"),
            }),
        }
    }

    fn set_flag(&mut self, flag: Nzcv, val: Value) {
        let v8 = if self.builder.func.dfg.value_type(val) == types::I8 {
            val
        } else {
            self.builder.ins().ireduce(types::I8, val)
        };
        self.builder.def_var(self.state.flags[&flag], v8);
        self.state.dirty_flags.insert(flag);
    }

    fn get_flag(&mut self, flag: Nzcv) -> Value {
        self.builder.use_var(self.state.flags[&flag])
    }

    /// Set N and Z from a result value of width `ty`.
    fn set_nz(&mut self, result: Value, ty: Type) {
        let n = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, result, 0);
        self.set_flag(Nzcv::N, n);
        let z = self.builder.ins().icmp_imm_s(IntCC::Equal, result, 0);
        self.set_flag(Nzcv::Z, z);
        let _ = ty;
    }

    /// Set all of NZCV for an add (`is_sub=false`) or subtract (`is_sub=true`)
    /// of `a` and `b` producing `result`, all of width `ty`.
    fn set_nzcv(&mut self, a: Value, b: Value, result: Value, ty: Type, is_sub: bool) {
        self.set_nz(result, ty);

        let shift = if ty == types::I64 { 63 } else { 31 };

        // Carry.
        let c = if is_sub {
            // No borrow: a >=u b.
            self.builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, a, b)
        } else {
            // Unsigned wrap: result <u a.
            self.builder.ins().icmp(IntCC::UnsignedLessThan, result, a)
        };
        self.set_flag(Nzcv::C, c);

        // Overflow: top bit of ((a^b') & (a^result)), where b' = b for sub and
        // ~b for add (i.e. add uses ~(a^b)).
        let a_xor_r = self.builder.ins().bxor(a, result);
        let v_term = if is_sub {
            let a_xor_b = self.builder.ins().bxor(a, b);
            self.builder.ins().band(a_xor_b, a_xor_r)
        } else {
            let a_xor_b = self.builder.ins().bxor(a, b);
            let not_axb = self.builder.ins().bnot(a_xor_b);
            self.builder.ins().band(not_axb, a_xor_r)
        };
        let v_top = self.builder.ins().ushr_imm_u(v_term, shift);
        let v_bit = self.builder.ins().band_imm_u(v_top, 1);
        self.set_flag(Nzcv::V, v_bit);
    }

    fn handle_movz(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("movz", op0))?;
        let ty = size_ty(sz);
        let val = self.read_source(op1, ty)?;
        self.write_reg_val(d, false, ty, val);
        Ok(())
    }

    fn handle_movn(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("movn", op0))?;
        let ty = size_ty(sz);
        let imm = match op1 {
            Operand::ImmShift(imm, shift) => (imm as u64) << shift,
            Operand::Immediate(v) => v as u64,
            Operand::Imm64(v) => v,
            _ => return Err(Self::bad_operand("movn", op1)),
        };
        let val = self.builder.ins().iconst(ty, !imm as i64);
        self.write_reg_val(d, false, ty, val);
        Ok(())
    }

    fn handle_movk(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("movk", op0))?;
        let ty = size_ty(sz);
        let (imm, shift) = match op1 {
            Operand::ImmShift(imm, shift) => (imm as u64, shift),
            _ => return Err(Self::bad_operand("movk", op1)),
        };
        let old = self.read_reg_val(d, false, ty);
        let mask = !(0xFFFFu64 << shift);
        let cleared = self.builder.ins().band_imm_u(old, mask as i64);
        let insert = self.builder.ins().iconst(ty, (imm << shift) as i64);
        let new = self.builder.ins().bor(cleared, insert);
        self.write_reg_val(d, false, ty, new);
        Ok(())
    }

    fn handle_addsub(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        is_sub: bool,
        set_flags: bool,
    ) -> crate::Result<()> {
        let (d, sz, d_sp) = decode_reg(op0).ok_or_else(|| Self::bad_operand("add/sub", op0))?;
        let ty = size_ty(sz);
        let (n, _, n_sp) = decode_reg(op1).ok_or_else(|| Self::bad_operand("add/sub", op1))?;
        let a = self.read_reg_val(n, n_sp, ty);
        let b = self.read_source(op2, ty)?;

        let result = if is_sub {
            self.builder.ins().isub(a, b)
        } else {
            self.builder.ins().iadd(a, b)
        };

        if set_flags {
            self.set_nzcv(a, b, result, ty, is_sub);
        }
        // For flag-setting forms (e.g. CMP = SUBS to XZR) the destination may be
        // the zero register, in which case the write is discarded.
        self.write_reg_val(d, d_sp, ty, result);
        Ok(())
    }

    fn handle_logical(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        logic: LogicOp,
    ) -> crate::Result<()> {
        let (d, sz, d_sp) = decode_reg(op0).ok_or_else(|| Self::bad_operand("logical", op0))?;
        let ty = size_ty(sz);
        let (n, _, n_sp) = decode_reg(op1).ok_or_else(|| Self::bad_operand("logical", op1))?;
        let a = self.read_reg_val(n, n_sp, ty);
        let b = self.read_source(op2, ty)?;
        let result = match logic {
            LogicOp::And => self.builder.ins().band(a, b),
            LogicOp::Or => self.builder.ins().bor(a, b),
            LogicOp::Xor => self.builder.ins().bxor(a, b),
        };
        self.write_reg_val(d, d_sp, ty, result);
        Ok(())
    }

    /// `ANDS` (and its `TST` alias): bitwise AND that sets NZ from the result
    /// and clears C and V. `TST` decodes as `ANDS` to the zero register, so the
    /// discarded-write path in `write_reg_val` handles it transparently.
    fn handle_ands(&mut self, op0: Operand, op1: Operand, op2: Operand) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("ands", op0))?;
        let ty = size_ty(sz);
        let (n, _, n_sp) = decode_reg(op1).ok_or_else(|| Self::bad_operand("ands", op1))?;
        let a = self.read_reg_val(n, n_sp, ty);
        let b = self.read_source(op2, ty)?;
        let result = self.builder.ins().band(a, b);
        self.set_nz(result, ty);
        let zero = self.builder.ins().iconst(types::I8, 0);
        self.set_flag(Nzcv::C, zero);
        self.set_flag(Nzcv::V, zero);
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `CSEL Rd, Rn, Rm, cond`: write `Rn` if `cond` holds, otherwise `Rm`.
    fn handle_csel(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        op3: Operand,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("csel", op0))?;
        let ty = size_ty(sz);
        let cond = match op3 {
            Operand::ConditionCode(c) => c,
            _ => return Err(Self::bad_operand("csel", op3)),
        };
        let a = self.read_source(op1, ty)?;
        let b = self.read_source(op2, ty)?;
        let cc = self.eval_condition(cond);
        let result = self.builder.ins().select(cc, a, b);
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `ORN`/`EON`/`BIC` (and the `MVN` alias of `ORN`): a logical op whose
    /// second operand is bitwise-inverted before combining. `MVN Rd, Rm` decodes
    /// as `ORN Rd, XZR, Rm`, which the zero-register read handles transparently.
    fn handle_logical_neg(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        logic: LogicOp,
    ) -> crate::Result<()> {
        let (d, sz, d_sp) = decode_reg(op0).ok_or_else(|| Self::bad_operand("logical-neg", op0))?;
        let ty = size_ty(sz);
        let (n, _, n_sp) = decode_reg(op1).ok_or_else(|| Self::bad_operand("logical-neg", op1))?;
        let a = self.read_reg_val(n, n_sp, ty);
        let b = self.read_source(op2, ty)?;
        let b_inv = self.builder.ins().bnot(b);
        let result = match logic {
            LogicOp::And => self.builder.ins().band(a, b_inv),
            LogicOp::Or => self.builder.ins().bor(a, b_inv),
            LogicOp::Xor => self.builder.ins().bxor(a, b_inv),
        };
        self.write_reg_val(d, d_sp, ty, result);
        Ok(())
    }

    /// `CSINC`/`CSINV`/`CSNEG Rd, Rn, Rm, cond`: write `Rn` when `cond` holds,
    /// otherwise a transformed `Rm` (`Rm+1`, `~Rm`, or `-Rm`). These also back the
    /// `CSET`/`CSETM`/`CINC`/`CINV`/`CNEG` aliases.
    fn handle_cs_variant(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        op3: Operand,
        variant: CsVariant,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("cs-variant", op0))?;
        let ty = size_ty(sz);
        let cond = match op3 {
            Operand::ConditionCode(c) => c,
            _ => return Err(Self::bad_operand("cs-variant", op3)),
        };
        let a = self.read_source(op1, ty)?;
        let b = self.read_source(op2, ty)?;
        let b_xform = match variant {
            CsVariant::Inc => self.builder.ins().iadd_imm_s(b, 1),
            CsVariant::Inv => self.builder.ins().bnot(b),
            CsVariant::Neg => self.builder.ins().ineg(b),
        };
        let cc = self.eval_condition(cond);
        let result = self.builder.ins().select(cc, a, b_xform);
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `MADD`/`MSUB Rd, Rn, Rm, Ra`: `Ra ± (Rn * Rm)`. `MUL Rd, Rn, Rm` decodes
    /// as `MADD` with `Ra = XZR`, so the multiply-only form needs no special case.
    fn handle_madd(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        op3: Operand,
        is_sub: bool,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("madd", op0))?;
        let ty = size_ty(sz);
        let n = self.read_source(op1, ty)?;
        let m = self.read_source(op2, ty)?;
        let a = self.read_source(op3, ty)?;
        let prod = self.builder.ins().imul(n, m);
        let result = if is_sub {
            self.builder.ins().isub(a, prod)
        } else {
            self.builder.ins().iadd(a, prod)
        };
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `UDIV`/`SDIV Rd, Rn, Rm`: integer division that, per the A64 spec, yields
    /// zero on divide-by-zero instead of trapping. Signed division additionally
    /// returns `INT_MIN` for the `INT_MIN / -1` overflow case rather than
    /// trapping. Cranelift's `udiv`/`sdiv` trap on both, so the divisor is forced
    /// to a safe non-trapping value and the spec result is selected back in.
    fn handle_div(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        signed: bool,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("div", op0))?;
        let ty = size_ty(sz);
        let n = self.read_source(op1, ty)?;
        let m = self.read_source(op2, ty)?;

        let div_by_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, m, 0);

        if signed {
            // Detect the INT_MIN / -1 overflow: numerator == INT_MIN && divisor == -1.
            let int_min = if ty == types::I64 {
                i64::MIN
            } else {
                i32::MIN as i64
            };
            let n_is_min = self.builder.ins().icmp_imm_s(IntCC::Equal, n, int_min);
            let m_is_neg1 = self.builder.ins().icmp_imm_s(IntCC::Equal, m, -1);
            let overflow = self.builder.ins().band(n_is_min, m_is_neg1);
            // Force the divisor to 1 whenever dividing would trap, then patch the
            // result: 0 for divide-by-zero, INT_MIN for the overflow case.
            let trap = self.builder.ins().bor(div_by_zero, overflow);
            let one = self.builder.ins().iconst(ty, 1);
            let safe_m = self.builder.ins().select(trap, one, m);
            let q = self.builder.ins().sdiv(n, safe_m);
            let min_val = self.builder.ins().iconst(ty, int_min);
            let q = self.builder.ins().select(overflow, min_val, q);
            let zero = self.builder.ins().iconst(ty, 0);
            let result = self.builder.ins().select(div_by_zero, zero, q);
            self.write_reg_val(d, false, ty, result);
        } else {
            let one = self.builder.ins().iconst(ty, 1);
            let safe_m = self.builder.ins().select(div_by_zero, one, m);
            let q = self.builder.ins().udiv(n, safe_m);
            let zero = self.builder.ins().iconst(ty, 0);
            let result = self.builder.ins().select(div_by_zero, zero, q);
            self.write_reg_val(d, false, ty, result);
        }
        Ok(())
    }

    /// `LSLV`/`LSRV`/`ASRV`/`RORV Rd, Rn, Rm`: shift `Rn` by a variable amount
    /// taken from `Rm`. A64 masks the shift amount modulo the datasize, which is
    /// exactly how Cranelift's shift instructions interpret their amount operand,
    /// so the mapping is direct.
    fn handle_shift_var(
        &mut self,
        op0: Operand,
        op1: Operand,
        op2: Operand,
        style: ShiftStyle,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("shift", op0))?;
        let ty = size_ty(sz);
        let a = self.read_source(op1, ty)?;
        let amount = self.read_source(op2, ty)?;
        let result = match style {
            ShiftStyle::LSL => self.builder.ins().ishl(a, amount),
            ShiftStyle::LSR => self.builder.ins().ushr(a, amount),
            ShiftStyle::ASR => self.builder.ins().sshr(a, amount),
            ShiftStyle::ROR => self.builder.ins().rotr(a, amount),
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("shift-var-{style:?}"),
                });
            }
        };
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `CLZ`/`RBIT`/`REV Rd, Rn`: count-leading-zeros, bit-reverse, and byte-swap
    /// of a single source register.
    fn handle_bit_unary(
        &mut self,
        op0: Operand,
        op1: Operand,
        kind: BitUnary,
    ) -> crate::Result<()> {
        let (d, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("bit-unary", op0))?;
        let ty = size_ty(sz);
        let a = self.read_source(op1, ty)?;
        let result = match kind {
            BitUnary::Clz => self.builder.ins().clz(a),
            BitUnary::Rbit => self.builder.ins().bitrev(a),
            BitUnary::Rev => self.builder.ins().bswap(a),
        };
        self.write_reg_val(d, false, ty, result);
        Ok(())
    }

    /// `ADR Rd, label`: form the runtime byte address `PC + imm`. Like `ADRP`
    /// (but without page alignment) the PC is the runtime value so the result
    /// survives ASLR. See [`Self::handle_adrp`].
    fn handle_adr(&mut self, instr: &A64Instr, op0: Operand) -> crate::Result<()> {
        let (d, _, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("adr", op0))?;
        let off = self.branch_target_offset(instr)?;
        let runtime_pc = self.runtime_addr(instr.ip);
        let result = self.builder.ins().iadd_imm_s(runtime_pc, off);
        self.write_reg_val(d, false, types::I64, result);
        Ok(())
    }

    /// `ADRP Rd, label`: form the runtime address of the 4 KiB page that is
    /// `imm` pages away from the page containing this instruction.
    ///
    /// The page is computed against the *runtime* PC, so under the default
    /// [`AddressMode::ImageBaseRelative`](crate::core::AddressMode) the result
    /// stays correct under ASLR rather than baking in the lift-time virtual
    /// address.
    fn handle_adrp(&mut self, instr: &A64Instr, op0: Operand) -> crate::Result<()> {
        let (d, _, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("adrp", op0))?;
        let off = self.branch_target_offset(instr)?;
        let runtime_pc = self.runtime_addr(instr.ip);
        let page = self.builder.ins().band_imm_u(runtime_pc, !0xFFFi64);
        let result = self.builder.ins().iadd_imm_s(page, off);
        self.write_reg_val(d, false, types::I64, result);
        Ok(())
    }

    // --- memory ---------------------------------------------------------------

    /// Resolve a memory operand to `(address, optional writeback)`.
    ///
    /// The writeback, if present, is the new base-register value to store after
    /// the access (pre-index with `!`, or post-index).
    fn mem_address(&mut self, op: Operand) -> crate::Result<(Value, Option<(u16, Value)>)> {
        match op {
            Operand::RegPreIndex(base, imm, writeback) => {
                let base_val = self.read_addr_base(base);
                let addr = self.builder.ins().iadd_imm_s(base_val, imm as i64);
                let wb = if writeback { Some((base, addr)) } else { None };
                Ok((addr, wb))
            }
            Operand::RegPostIndex(base, imm) => {
                let base_val = self.read_addr_base(base);
                let new_base = self.builder.ins().iadd_imm_s(base_val, imm as i64);
                // Access happens at the original base; writeback updates it.
                Ok((base_val, Some((base, new_base))))
            }
            Operand::RegRegOffset(base, index, _idx_sz, style, amount) => {
                let base_val = self.read_addr_base(base);
                let idx_val = self.read_reg_val(index, false, types::I64);
                let scaled = match style {
                    ShiftStyle::LSL => self.builder.ins().ishl_imm_u(idx_val, amount as i64),
                    _ => idx_val,
                };
                let addr = self.builder.ins().iadd(base_val, scaled);
                Ok((addr, None))
            }
            _ => Err(Self::bad_operand("memory", op)),
        }
    }

    /// Read a memory base register (register 31 is SP, never the zero register).
    fn read_addr_base(&mut self, num: u16) -> Value {
        self.read_reg_val(num, true, types::I64)
    }

    fn apply_writeback(&mut self, wb: Option<(u16, Value)>) {
        if let Some((base, val)) = wb {
            self.write_reg_val(base, true, types::I64, val);
        }
    }

    fn handle_ldr(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (t, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("ldr", op0))?;
        let ty = size_ty(sz);
        let (addr, wb) = self.mem_address(op1)?;
        let val = self.builder.ins().load(ty, MemFlagsData::new(), addr, 0);
        self.write_reg_val(t, false, ty, val);
        self.apply_writeback(wb);
        Ok(())
    }

    fn handle_str(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (t, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("str", op0))?;
        let ty = size_ty(sz);
        let val = self.read_reg_val(t, false, ty);
        let (addr, wb) = self.mem_address(op1)?;
        self.builder.ins().store(MemFlagsData::new(), val, addr, 0);
        self.apply_writeback(wb);
        Ok(())
    }

    /// `STP Rt, Rt2, [addr]`: store a register pair to consecutive slots. The
    /// second register goes one element-width above the first. Any pre/post-index
    /// writeback on the address operand is applied once, after both stores.
    fn handle_stp(&mut self, op0: Operand, op1: Operand, op2: Operand) -> crate::Result<()> {
        let (t1, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("stp", op0))?;
        let (t2, _, _) = decode_reg(op1).ok_or_else(|| Self::bad_operand("stp", op1))?;
        let ty = size_ty(sz);
        let elem = ty.bytes() as i32;
        let v1 = self.read_reg_val(t1, false, ty);
        let v2 = self.read_reg_val(t2, false, ty);
        let (addr, wb) = self.mem_address(op2)?;
        self.builder.ins().store(MemFlagsData::new(), v1, addr, 0);
        self.builder
            .ins()
            .store(MemFlagsData::new(), v2, addr, elem);
        self.apply_writeback(wb);
        Ok(())
    }

    /// `LDP Rt, Rt2, [addr]`: load a register pair from consecutive slots, the
    /// complement of [`Self::handle_stp`]. Both loads use the original address;
    /// any pre/post-index writeback is applied once afterwards.
    fn handle_ldp(&mut self, op0: Operand, op1: Operand, op2: Operand) -> crate::Result<()> {
        let (t1, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("ldp", op0))?;
        let (t2, _, _) = decode_reg(op1).ok_or_else(|| Self::bad_operand("ldp", op1))?;
        let ty = size_ty(sz);
        let elem = ty.bytes() as i32;
        let (addr, wb) = self.mem_address(op2)?;
        let v1 = self.builder.ins().load(ty, MemFlagsData::new(), addr, 0);
        let v2 = self.builder.ins().load(ty, MemFlagsData::new(), addr, elem);
        self.write_reg_val(t1, false, ty, v1);
        self.write_reg_val(t2, false, ty, v2);
        self.apply_writeback(wb);
        Ok(())
    }

    fn handle_b(&mut self, instr: &A64Instr) -> crate::Result<()> {
        let target = self.branch_target(instr)?;
        let block = self.get_block_with_kind(target, TransferKind::DirectJump);
        self.builder.ins().jump(block, &[]);
        self.block_terminated = true;
        Ok(())
    }

    fn handle_bcc(&mut self, instr: &A64Instr, cond: u8) -> crate::Result<()> {
        let target = self.branch_target(instr)?;
        let taken = self.get_block_with_kind(target, TransferKind::ConditionalJump);
        let next = self.get_block(instr.ip + 4);
        let cc = self.eval_condition(cond);
        self.builder.ins().brif(cc, taken, &[], next, &[]);
        self.block_terminated = true;
        Ok(())
    }

    /// `CBZ`/`CBNZ Rt, label`: branch when `Rt` is zero (`branch_if_zero`) or
    /// non-zero. The register is compared at its operand width (W or X).
    fn handle_cb(
        &mut self,
        instr: &A64Instr,
        op0: Operand,
        branch_if_zero: bool,
    ) -> crate::Result<()> {
        let (t, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("cbz", op0))?;
        let ty = size_ty(sz);
        let val = self.read_reg_val(t, false, ty);
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, val, 0);

        let target = self.branch_target(instr)?;
        let taken = self.get_block_with_kind(target, TransferKind::ConditionalJump);
        let next = self.get_block(instr.ip + 4);
        if branch_if_zero {
            self.builder.ins().brif(is_zero, taken, &[], next, &[]);
        } else {
            self.builder.ins().brif(is_zero, next, &[], taken, &[]);
        }
        self.block_terminated = true;
        Ok(())
    }

    /// `TBZ`/`TBNZ Rt, #bit, label`: branch when the selected bit of `Rt` is
    /// zero (`branch_if_zero`) or non-zero. The test register is read at full
    /// 64-bit width so any tested bit position is valid.
    fn handle_tb(
        &mut self,
        instr: &A64Instr,
        op0: Operand,
        op1: Operand,
        branch_if_zero: bool,
    ) -> crate::Result<()> {
        let (t, _, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("tbz", op0))?;
        let bit = match op1 {
            Operand::Imm16(b) => b as i64,
            Operand::Immediate(b) => b as i64,
            _ => return Err(Self::bad_operand("tbz", op1)),
        };
        let val = self.read_reg_val(t, false, types::I64);
        let masked = self.builder.ins().band_imm_u(val, 1i64 << bit);
        let is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, masked, 0);

        let target = self.branch_target(instr)?;
        let taken = self.get_block_with_kind(target, TransferKind::ConditionalJump);
        let next = self.get_block(instr.ip + 4);
        if branch_if_zero {
            self.builder.ins().brif(is_zero, taken, &[], next, &[]);
        } else {
            self.builder.ins().brif(is_zero, next, &[], taken, &[]);
        }
        self.block_terminated = true;
        Ok(())
    }

    /// `BL label`: direct call. In analysis mode this emits a real Cranelift
    /// `call` to an external function representing the callee and falls through
    /// (see the x86 `handle_call_analysis` for rationale). Execution-mode `BL`
    /// is not yet implemented.
    fn handle_bl(&mut self, instr: &A64Instr) -> crate::Result<()> {
        if self.config.analysis_mode {
            let target = self.branch_target(instr)?;
            let fref = self.analysis_call_funcref(target);
            self.builder.ins().call(fref, &[]);
            self.block_terminated = false;
            return Ok(());
        }
        Err(MicroliftError::UnsupportedInstruction {
            mnemonic: "bl".into(),
        })
    }

    /// `BLR Rn`: indirect call through a register. Sets the link register (X30)
    /// to the return address and exits with `IndirectCall` so the host can
    /// dispatch the runtime-resolved callee. See [`ExitReason::IndirectCall`].
    fn handle_blr(&mut self, op0: Operand) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Indirect call: emit `call_indirect` on the runtime target and fall
            // through; no statically-known callee to record.
            let target_reg = match op0 {
                Operand::Register(_, n) => n,
                _ => return Err(Self::bad_operand("blr", op0)),
            };
            let target = self.read_reg_val(target_reg, false, types::I64);
            let sig = self.analysis_call_sig();
            self.builder.ins().call_indirect(sig, target, &[]);
            self.block_terminated = false;
            return Ok(());
        }
        let target_reg = match op0 {
            Operand::Register(_, n) => n,
            _ => return Err(Self::bad_operand("blr", op0)),
        };
        let target = self.read_reg_val(target_reg, false, types::I64);

        // The return address (and PC resume point) is the next instruction:
        // current_ip already points there.
        let ret_addr = self.runtime_current_ip();
        self.write_reg_val(30, false, types::I64, ret_addr);

        // Ensure the resume block exists so the entry dispatch can route back.
        self.get_block(self.current_ip);

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = <Aarch64 as GuestArch>::imported_function_va_offset();
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), target, ctx_ptr, va_offset);

        self.sync_to_context(None, ExitReason::IndirectCall)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    fn handle_ret(&mut self, op0: Operand) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Inspection IR: a return terminates the function (see the x86 RET
            // handler for rationale).
            self.emit_host_return();
            self.block_terminated = true;
            return Ok(());
        }

        // RET branches to the link register (X30) by default, or to an explicit
        // register operand. The target is runtime-only, so we exit like x86 RET.
        let target_reg = match op0 {
            Operand::Register(_, n) => n,
            _ => 30,
        };
        let target = self.read_reg_val(target_reg, false, types::I64);
        self.sync_to_context(Some(target), ExitReason::Return)?;
        self.write_exit_code(TransferKind::Return as u32);
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    fn handle_svc(&mut self) -> crate::Result<()> {
        self.sync_to_context(None, ExitReason::Syscall)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    fn exit_unimplemented(&mut self) -> crate::Result<()> {
        self.sync_to_context(None, ExitReason::UnimplementedInstruction)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Resolve an instruction's PC-relative branch target.
    fn branch_target(&self, instr: &A64Instr) -> crate::Result<u64> {
        for op in &instr.inst.operands {
            if let Operand::PCOffset(off) = op {
                return Ok(instr.ip.wrapping_add(*off as u64));
            }
        }
        Err(MicroliftError::UnsupportedInstruction {
            mnemonic: "branch-without-pcoffset".into(),
        })
    }

    /// The raw PC-relative offset operand of an instruction, *not* added to the
    /// IP. Used by `ADRP`, whose offset is a page delta applied to the masked PC.
    fn branch_target_offset(&self, instr: &A64Instr) -> crate::Result<i64> {
        for op in &instr.inst.operands {
            if let Operand::PCOffset(off) = op {
                return Ok(*off);
            }
        }
        Err(MicroliftError::UnsupportedInstruction {
            mnemonic: "adrp-without-pcoffset".into(),
        })
    }

    /// Evaluate an A64 condition code (0–15) into an I8 boolean.
    fn eval_condition(&mut self, cond: u8) -> Value {
        let n = self.get_flag(Nzcv::N);
        let z = self.get_flag(Nzcv::Z);
        let c = self.get_flag(Nzcv::C);
        let v = self.get_flag(Nzcv::V);
        // Condition codes come in pairs differing only by their low bit, which
        // inverts the result.
        let base = cond >> 1;
        let result = match base {
            0 => z, // EQ: Z==1
            1 => c, // CS: C==1
            2 => n, // MI: N==1
            3 => v, // VS: V==1
            4 => {
                // HI: C==1 && Z==0
                let z0 = self.builder.ins().icmp_imm_s(IntCC::Equal, z, 0);
                self.builder.ins().band(c, z0)
            }
            5 => {
                // GE: N==V
                self.builder.ins().icmp(IntCC::Equal, n, v)
            }
            6 => {
                // GT: Z==0 && N==V
                let z0 = self.builder.ins().icmp_imm_s(IntCC::Equal, z, 0);
                let nv = self.builder.ins().icmp(IntCC::Equal, n, v);
                self.builder.ins().band(z0, nv)
            }
            _ => {
                // AL / NV: always true.
                self.builder.ins().iconst(types::I8, 1)
            }
        };

        if cond & 1 == 1 && base != 7 {
            // Odd condition code: invert (NE, CC, PL, VC, LS, LT, LE).
            let one = self.builder.ins().iconst(types::I8, 1);
            self.builder.ins().bxor(result, one)
        } else {
            result
        }
    }

    fn bad_operand(ctx: &str, op: Operand) -> MicroliftError {
        MicroliftError::UnsupportedInstruction {
            mnemonic: format!("{ctx}-operand-{op:?}"),
        }
    }
}

#[derive(Clone, Copy)]
enum LogicOp {
    And,
    Or,
    Xor,
}

/// Which transform `CSINC`/`CSINV`/`CSNEG` applies to their else-operand.
#[derive(Clone, Copy)]
enum CsVariant {
    /// `Rm + 1` (CSINC).
    Inc,
    /// `~Rm` (CSINV).
    Inv,
    /// `-Rm` (CSNEG).
    Neg,
}

/// Single-source bit-manipulation unary ops.
#[derive(Clone, Copy)]
enum BitUnary {
    /// Count leading zeros (CLZ).
    Clz,
    /// Reverse bit order (RBIT).
    Rbit,
    /// Reverse byte order (REV).
    Rev,
}
