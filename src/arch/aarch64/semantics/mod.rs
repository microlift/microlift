//! aarch64 instruction semantics (vertical slice).
//!
//! Covers a minimal but representative subset proving the `GuestArch`
//! abstraction end-to-end: moves (MOVZ/MOVK/MOVN, MOV-via-ORR), integer
//! data-processing (ADD/SUB/ADDS/SUBS incl. CMP, AND/ORR), base+offset
//! loads/stores (LDR/STR), branches (B/B.cond), RET, NOP, and SVC.
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
            Opcode::LDR => self.handle_ldr(op0, op1),
            Opcode::STR => self.handle_str(op0, op1),
            Opcode::B => self.handle_b(instr),
            Opcode::Bcc(cond) => self.handle_bcc(instr, cond),
            Opcode::RET => self.handle_ret(op0),
            Opcode::SVC => self.handle_svc(),
            Opcode::HINT => Ok(()),
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
            Operand::ImmShift(imm, shift) => {
                Ok(self.builder.ins().iconst(ty, ((imm as u64) << shift) as i64))
            }
            Operand::RegShift(style, amount, _sz, n) => {
                let base = self.read_reg_val(n, false, ty);
                Ok(match style {
                    ShiftStyle::LSL => self.builder.ins().ishl_imm(base, amount as i64),
                    ShiftStyle::LSR => self.builder.ins().ushr_imm(base, amount as i64),
                    ShiftStyle::ASR => self.builder.ins().sshr_imm(base, amount as i64),
                    ShiftStyle::ROR => self.builder.ins().rotr_imm(base, amount as i64),
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
        let n = self.builder.ins().icmp_imm(IntCC::SignedLessThan, result, 0);
        self.set_flag(Nzcv::N, n);
        let z = self.builder.ins().icmp_imm(IntCC::Equal, result, 0);
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
        let v_top = self.builder.ins().ushr_imm(v_term, shift);
        let v_bit = self.builder.ins().band_imm(v_top, 1);
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
        let cleared = self.builder.ins().band_imm(old, mask as i64);
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

    // --- memory ---------------------------------------------------------------

    /// Resolve a memory operand to `(address, optional writeback)`.
    ///
    /// The writeback, if present, is the new base-register value to store after
    /// the access (pre-index with `!`, or post-index).
    fn mem_address(&mut self, op: Operand) -> crate::Result<(Value, Option<(u16, Value)>)> {
        match op {
            Operand::RegPreIndex(base, imm, writeback) => {
                let base_val = self.read_addr_base(base);
                let addr = self.builder.ins().iadd_imm(base_val, imm as i64);
                let wb = if writeback { Some((base, addr)) } else { None };
                Ok((addr, wb))
            }
            Operand::RegPostIndex(base, imm) => {
                let base_val = self.read_addr_base(base);
                let new_base = self.builder.ins().iadd_imm(base_val, imm as i64);
                // Access happens at the original base; writeback updates it.
                Ok((base_val, Some((base, new_base))))
            }
            Operand::RegRegOffset(base, index, _idx_sz, style, amount) => {
                let base_val = self.read_addr_base(base);
                let idx_val = self.read_reg_val(index, false, types::I64);
                let scaled = match style {
                    ShiftStyle::LSL => self.builder.ins().ishl_imm(idx_val, amount as i64),
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
        let val = self.builder.ins().load(ty, MemFlags::new(), addr, 0);
        self.write_reg_val(t, false, ty, val);
        self.apply_writeback(wb);
        Ok(())
    }

    fn handle_str(&mut self, op0: Operand, op1: Operand) -> crate::Result<()> {
        let (t, sz, _) = decode_reg(op0).ok_or_else(|| Self::bad_operand("str", op0))?;
        let ty = size_ty(sz);
        let val = self.read_reg_val(t, false, ty);
        let (addr, wb) = self.mem_address(op1)?;
        self.builder.ins().store(MemFlags::new(), val, addr, 0);
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

    fn handle_ret(&mut self, op0: Operand) -> crate::Result<()> {
        // RET branches to the link register (X30) by default, or to an explicit
        // register operand. The target is runtime-only, so we exit like x86 RET.
        let target_reg = match op0 {
            Operand::Register(_, n) => n,
            _ => 30,
        };
        let target = self.read_reg_val(target_reg, false, types::I64);
        self.sync_to_context_selective(Some(target), ExitReason::Return)?;
        self.write_exit_code(TransferKind::Return as u32);
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    fn handle_svc(&mut self) -> crate::Result<()> {
        self.sync_to_context_selective(None, ExitReason::Syscall)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    fn exit_unimplemented(&mut self) -> crate::Result<()> {
        self.sync_to_context_selective(None, ExitReason::UnimplementedInstruction)?;
        self.builder.ins().return_(&[]);
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
            0 => z,                                                  // EQ: Z==1
            1 => c,                                                  // CS: C==1
            2 => n,                                                  // MI: N==1
            3 => v,                                                  // VS: V==1
            4 => {
                // HI: C==1 && Z==0
                let z0 = self.builder.ins().icmp_imm(IntCC::Equal, z, 0);
                self.builder.ins().band(c, z0)
            }
            5 => {
                // GE: N==V
                self.builder.ins().icmp(IntCC::Equal, n, v)
            }
            6 => {
                // GT: Z==0 && N==V
                let z0 = self.builder.ins().icmp_imm(IntCC::Equal, z, 0);
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
