//! aarch64 instruction semantics, as inherent methods on `Lifter64` reached via `GuestArch::lift_instruction`.
//!
//! Subset: moves, integer ALU/logic, multiply/divide, shifts, CLZ/RBIT/REV, CSEL family,
//! ADR/ADRP, LDR/STR/LDP/STP, branches, RET, NOP, SVC.

use crate::MicroliftError;
use crate::arch::aarch64::Aarch64;
use crate::arch::aarch64::Lifter64;
use crate::arch::aarch64::decode::A64Instr;
use crate::arch::aarch64::regs::{FLAG_ORDER, Nzcv, ZR};
use crate::arch::aarch64::sync::SP_SLOT;
use crate::core::GuestArch;
use crate::core::exit::{ExitReason, TransferKind};
use crate::core::regsync::RegSlot;
use cranelift::prelude::*;
use yaxpeax_arm::armv8::a64::{Opcode, Operand, ShiftStyle, SizeCode};

/// Decodes a register operand to `(reg_number, size, sp_form)`; `sp_form` means register 31 is SP, not XZR.
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
            // TST is ANDS to the zero register.
            Opcode::ANDS => self.handle_ands(op0, op1, op2),
            Opcode::ORN => self.handle_logical_neg(op0, op1, op2, LogicOp::Or),
            Opcode::EON => self.handle_logical_neg(op0, op1, op2, LogicOp::Xor),
            // BICS also sets NZ.
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
            // Without PAC enforcement sign/authenticate round-trips; modelled as no-ops.
            Opcode::PACIASP | Opcode::AUTIASP => Ok(()),
            other => {
                if self.config.interpreter_fallback {
                    self.interpret_unimplemented(instr)
                } else {
                    Err(MicroliftError::UnsupportedInstruction {
                        mnemonic: format!("{other:?}"),
                    })
                }
            }
        }
    }

    /// Reads register `num`; 31 is XZR (0) unless `sp_form`.
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

    /// Writes `val`; W-width writes zero-extend, XZR writes are discarded.
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
            self.mark_written(SP_SLOT);
        } else {
            self.builder.def_var(self.state.gprs[&num], full);
            self.mark_written(RegSlot::Gpr(num as u8));
        }
    }

    /// Reads a register, immediate, or shifted-register source at width `ty`.
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
        let idx = FLAG_ORDER
            .iter()
            .position(|&f| f == flag)
            .expect("a flag variable has a FLAG_ORDER index");
        self.mark_written(RegSlot::Flag(idx as u8));
    }

    fn get_flag(&mut self, flag: Nzcv) -> Value {
        self.builder.use_var(self.state.flags[&flag])
    }

    /// Sets N and Z from `result`.
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

    /// Sets NZCV for `a + b` (or `a - b` when `is_sub`) producing `result`.
    fn set_nzcv(&mut self, a: Value, b: Value, result: Value, ty: Type, is_sub: bool) {
        self.set_nz(result, ty);

        let shift = if ty == types::I64 { 63 } else { 31 };

        let c = if is_sub {
            self.builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, a, b)
        } else {
            self.builder.ins().icmp(IntCC::UnsignedLessThan, result, a)
        };
        self.set_flag(Nzcv::C, c);

        // OF: top bit of (a^b') & (a^result), b' = b for sub, ~b for add.
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
        // The destination may be XZR (CMP); the write is then discarded.
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

    /// `ANDS` (and `TST`): AND that sets NZ and clears C and V.
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

    /// `CSEL Rd, Rn, Rm, cond`: `Rn` if `cond` holds, else `Rm`.
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

    /// `ORN`/`EON`/`BIC` (and `MVN`): logical op with the second operand inverted.
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

    /// `CSINC`/`CSINV`/`CSNEG Rd, Rn, Rm, cond`: `Rn` if `cond`, else `Rm+1`, `~Rm`, `-Rm`.
    /// Also backs `CSET`/`CSETM`/`CINC`/`CINV`/`CNEG`.
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

    /// `MADD`/`MSUB Rd, Rn, Rm, Ra`: `Ra ± (Rn * Rm)`; `MUL` is `MADD` with `Ra = XZR`.
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

    /// `UDIV`/`SDIV Rd, Rn, Rm`: A64 yields 0 on divide-by-zero and `INT_MIN` for
    /// `INT_MIN / -1`, where Cranelift traps, so the divisor is forced safe and the result patched.
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
            // INT_MIN / -1 overflow.
            let int_min = if ty == types::I64 {
                i64::MIN
            } else {
                i32::MIN as i64
            };
            let n_is_min = self.builder.ins().icmp_imm_s(IntCC::Equal, n, int_min);
            let m_is_neg1 = self.builder.ins().icmp_imm_s(IntCC::Equal, m, -1);
            let overflow = self.builder.ins().band(n_is_min, m_is_neg1);
            // Force divisor to 1 where it would trap; patch the result.
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

    /// `LSLV`/`LSRV`/`ASRV`/`RORV Rd, Rn, Rm`: shift by `Rm`; the datasize masking matches Cranelift's.
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

    /// `CLZ`/`RBIT`/`REV Rd, Rn`: count leading zeros, bit-reverse, byte-swap.
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

    /// `ADR Rd, label`: runtime `PC + imm` (ASLR-safe). See [`Self::handle_adrp`].
    fn handle_adr(&mut self, instr: &A64Instr, op0: Operand) -> crate::Result<()> {
        let (d, _, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("adr", op0))?;
        let off = self.branch_target_offset(instr)?;
        let runtime_pc = self.runtime_addr(instr.ip);
        let result = self.builder.ins().iadd_imm_s(runtime_pc, off);
        self.write_reg_val(d, false, types::I64, result);
        Ok(())
    }

    /// `ADRP Rd, label`: runtime address of the 4 KiB page `imm` pages from this instruction's page.
    /// Computed against the runtime PC so [`AddressMode::ImageBaseRelative`](crate::core::AddressMode) survives ASLR.
    fn handle_adrp(&mut self, instr: &A64Instr, op0: Operand) -> crate::Result<()> {
        let (d, _, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("adrp", op0))?;
        let off = self.branch_target_offset(instr)?;
        let runtime_pc = self.runtime_addr(instr.ip);
        let page = self.builder.ins().band_imm_u(runtime_pc, !0xFFFi64);
        let result = self.builder.ins().iadd_imm_s(page, off);
        self.write_reg_val(d, false, types::I64, result);
        Ok(())
    }


    /// Resolves a memory operand to `(address, optional writeback)`; the writeback is the new base value to store after the access.
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
                // Access uses the original base.
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

    /// Reads a memory base register; 31 is SP, never XZR.
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
        let val = self.builder.ins().load(ty, self.guest_flags, addr, 0);
        self.write_reg_val(t, false, ty, val);
        self.apply_writeback(wb);
        Ok(())
    }

    fn handle_str(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (t, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("str", op0))?;
        let ty = size_ty(sz);
        let val = self.read_reg_val(t, false, ty);
        let (addr, wb) = self.mem_address(op1)?;
        self.builder.ins().store(self.guest_flags, val, addr, 0);
        self.apply_writeback(wb);
        Ok(())
    }

    /// `STP Rt, Rt2, [addr]`: store a pair to consecutive slots; writeback applied once after both.
    fn handle_stp(&mut self, op0: Operand, op1: Operand, op2: Operand) -> crate::Result<()> {
        let (t1, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("stp", op0))?;
        let (t2, _, _) = decode_reg(op1).ok_or_else(|| Self::bad_operand("stp", op1))?;
        let ty = size_ty(sz);
        let elem = ty.bytes() as i32;
        let v1 = self.read_reg_val(t1, false, ty);
        let v2 = self.read_reg_val(t2, false, ty);
        let (addr, wb) = self.mem_address(op2)?;
        self.builder.ins().store(self.guest_flags, v1, addr, 0);
        self.builder.ins().store(self.guest_flags, v2, addr, elem);
        self.apply_writeback(wb);
        Ok(())
    }

    /// `LDP Rt, Rt2, [addr]`: load a pair from consecutive slots, original address; writeback applied once after. See [`Self::handle_stp`].
    fn handle_ldp(&mut self, op0: Operand, op1: Operand, op2: Operand) -> crate::Result<()> {
        let (t1, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("ldp", op0))?;
        let (t2, _, _) = decode_reg(op1).ok_or_else(|| Self::bad_operand("ldp", op1))?;
        let ty = size_ty(sz);
        let elem = ty.bytes() as i32;
        let (addr, wb) = self.mem_address(op2)?;
        let v1 = self.builder.ins().load(ty, self.guest_flags, addr, 0);
        let v2 = self.builder.ins().load(ty, self.guest_flags, addr, elem);
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

    /// `CBZ`/`CBNZ Rt, label`: branch if `Rt` is zero (`branch_if_zero`) or non-zero, at operand width.
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

    /// `TBZ`/`TBNZ Rt, #bit, label`: branch on the selected bit of `Rt` (read at 64 bits).
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

    /// `BL label`: direct call. Analysis mode emits a `call` to an external callee; execution mode is unimplemented.
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

    /// `BLR Rn`: sets X30 to the return address and exits with [`ExitReason::IndirectCall`].
    fn handle_blr(&mut self, op0: Operand) -> crate::Result<()> {
        if self.config.analysis_mode {
            // No statically-known callee to record.
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

        // `current_ip` already points at the return address.
        let ret_addr = self.runtime_current_ip();
        self.write_reg_val(30, false, types::I64, ret_addr);

        self.get_block(self.current_ip);

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = <Aarch64 as GuestArch>::imported_function_va_offset();
        self.ctx_store(target, ctx_ptr, va_offset);

        self.sync_to_context(None, ExitReason::IndirectCall)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    fn handle_ret(&mut self, op0: Operand) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Inspection IR: a return terminates the function.
            self.emit_host_return();
            self.block_terminated = true;
            return Ok(());
        }

        // Target is X30 or an explicit register; runtime-only, so exit like x86 RET.
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

    /// `SVC`: calls the host syscall callback inline ([`crate::core::callback`]); resumes at `current_ip`.
    fn handle_svc(&mut self) -> crate::Result<()> {
        self.emit_syscall_callback(None)
    }

    /// Hands an unlifted instruction to the host interpreter inline ([`crate::core::callback`]).
    fn interpret_unimplemented(&mut self, instr: &A64Instr) -> crate::Result<()> {
        let pc = self.runtime_addr(instr.ip);
        let resume = self.runtime_addr(instr.ip + 4);
        self.emit_interpreter_callback(Some(pc), Some(resume), 1)
    }

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

    /// Raw PC-relative offset operand, not added to the IP (`ADRP` applies it as a page delta).
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

    /// Evaluates an A64 condition code (0-15) to an I8 boolean.
    fn eval_condition(&mut self, cond: u8) -> Value {
        let n = self.get_flag(Nzcv::N);
        let z = self.get_flag(Nzcv::Z);
        let c = self.get_flag(Nzcv::C);
        let v = self.get_flag(Nzcv::V);
        // Codes pair up; the low bit inverts.
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
