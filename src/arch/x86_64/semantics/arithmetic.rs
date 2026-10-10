use crate::MicroliftError;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::ExitReason;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::{Instruction, Register};

impl<'a> Lifter<'a> {
    pub(super) fn handle_add(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let (place, dst, src) = self.read_binary_rmw(instruction, size)?;
        let result = self.add_with_flags(dst, src, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_sub(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let (place, dst, src) = self.read_binary_rmw(instruction, size)?;
        let result = self.sub_with_flags(dst, src, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_adc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        // CF is read before the operands so a memory load is not reordered ahead of a flag this instruction redefines.
        let cf = self.carry_in(size);
        let (place, dst, src) = self.read_binary_rmw(instruction, size)?;
        let result = self.adc_sbb_with_flags(dst, src, cf, true, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_sbb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let cf = self.carry_in(size);
        let (place, dst, src) = self.read_binary_rmw(instruction, size)?;
        let result = self.adc_sbb_with_flags(dst, src, cf, false, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_xadd(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (size, place, dst_val) = self.read_unary_rmw(instruction)?;
        let src_reg = instruction.op1_register();
        let src_val = self.get_reg_val(src_reg)?;

        // Flags first: they depend only on the two values just read.
        let result = self.add_with_flags(dst_val, src_val, size)?;
        // Hardware order: TEMP <- SRC + DEST; SRC <- DEST; DEST <- TEMP; the destination write is last so `xadd r, r` holds the sum.
        self.set_reg_val(src_reg, dst_val)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_cmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let (dst, src) = self.read_binary_operands(instruction, size)?;
        // CMP is a SUB that keeps only the flags.
        self.sub_with_flags(dst, src, size)?;
        Ok(())
    }

    pub(super) fn handle_inc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (size, place, dst_val) = self.read_unary_rmw(instruction)?;
        let ty = size_to_type(size);
        let one = self.builder.ins().iconst(ty, 1);
        let result = self.inc_dec_with_flags(dst_val, one, true, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_dec(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (size, place, dst_val) = self.read_unary_rmw(instruction)?;
        let ty = size_to_type(size);
        let one = self.builder.ins().iconst(ty, 1);
        let result = self.inc_dec_with_flags(dst_val, one, false, size)?;
        self.write_place(place, result)
    }

    pub(super) fn handle_neg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let (size, place, val) = self.read_unary_rmw(instruction)?;
        let ty = size_to_type(size);
        let zero = self.builder.ins().iconst(ty, 0);
        // NEG is `0 - val`: CF is the borrow, OF the signed overflow.
        let result = self.sub_with_flags(zero, val, size)?;
        self.write_place(place, result)
    }

    /// IMUL - handles 1, 2, and 3-operand forms.
    ///
    /// - 1-operand: `imul r/m` -> RAX * src -> RDX:RAX (signed 128-bit)
    /// - 2-operand: `imul dst, src` -> dst = dst * src (truncated, flags set)
    /// - 3-operand: `imul dst, src, imm` -> dst = src * imm (truncated, flags set)
    pub(super) fn handle_imul(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op_count() {
            1 => {
                let size = self.operand_size(instruction, 0)?;
                let ty = size_to_type(size);
                let rax = self.get_reg_val(Register::RAX)?;
                let src = self.read_operand(instruction, 0, size)?;

                // `rax` is the whole register and needs narrowing; `src` is already operand width.
                let rax_narrow = self.narrow_to(ty, rax);
                let src_narrow = self.narrow_to(ty, src);

                let lo = self.builder.ins().imul(rax_narrow, src_narrow);
                let hi = self.builder.ins().smulhi(rax_narrow, src_narrow);

                // 8-bit -> AX, 16-bit -> DX:AX, 32-bit -> EDX:EAX, 64-bit -> RDX:RAX
                let (lo_reg, hi_reg) = match size {
                    1 => {
                        let lo_64 = self.builder.ins().uextend(types::I64, lo);
                        let hi_64 = self.builder.ins().uextend(types::I64, hi);
                        let hi_shifted = self.builder.ins().ishl_imm_u(hi_64, 8);
                        let ax_val = self.builder.ins().bor(lo_64, hi_shifted);
                        let ax16 = self.builder.ins().ireduce(types::I16, ax_val);
                        self.set_reg_val(Register::AX, ax16)?;
                        // CF/OF set if AH != sign extension of AL
                        let sign_ext = self.builder.ins().sshr_imm_u(lo, 7);
                        let sign_ext8 = self.builder.ins().ireduce(types::I8, sign_ext);
                        let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext8, hi);
                        self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                        self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
                        return Ok(());
                    }
                    2 => (Register::AX, Register::DX),
                    4 => (Register::EAX, Register::EDX),
                    8 => (Register::RAX, Register::RDX),
                    _ => unreachable!(),
                };
                self.set_reg_val(lo_reg, lo)?;
                self.set_reg_val(hi_reg, hi)?;

                // CF/OF: set if hi != sign extension of lo
                let sign_ext = self.builder.ins().sshr_imm_u(lo, (size * 8 - 1) as i64);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, sign_ext, hi);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            2 => {
                let size = self.operand_size(instruction, 0)?;
                let dst = self.read_operand(instruction, 0, size)?;
                let src = self.read_operand(instruction, 1, size)?;
                let result = self.imul_with_flags(dst, src, size)?;
                self.write_operand(instruction, 0, result)?;
            }
            3 => {
                let size = self.operand_size(instruction, 0)?;
                let src = self.read_operand(instruction, 1, size)?;
                let imm = self.read_operand(instruction, 2, size)?;
                let result = self.imul_with_flags(src, imm, size)?;
                self.write_operand(instruction, 0, result)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "imul (unexpected op count)".into(),
                });
            }
        }
        Ok(())
    }

    /// MUL - unsigned multiply, 1-operand form.
    /// 8-bit: AX = AL * r/m8
    /// 16-bit: DX:AX = AX * r/m16
    /// 32-bit: EDX:EAX = EAX * r/m32
    /// 64-bit: RDX:RAX = RAX * r/m64
    pub(super) fn handle_mul(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let ty = size_to_type(size);
        let src = self.read_operand(instruction, 0, size)?;

        match size {
            1 => {
                let rax_val = self.get_reg_val(Register::RAX)?;
                let al = self.builder.ins().ireduce(types::I8, rax_val);
                let lo = self.builder.ins().imul(al, src);
                let hi = self.builder.ins().umulhi(al, src);
                let lo_64 = self.builder.ins().uextend(types::I64, lo);
                let hi_64 = self.builder.ins().uextend(types::I64, hi);
                let hi_shifted = self.builder.ins().ishl_imm_u(hi_64, 8);
                let ax_val = self.builder.ins().bor(lo_64, hi_shifted);
                let ax16 = self.builder.ins().ireduce(types::I16, ax_val);
                self.set_reg_val(Register::AX, ax16)?;
                // CF/OF set if AH != 0
                let of_cf = self.builder.ins().icmp_imm_s(IntCC::NotEqual, hi, 0);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            2 | 4 | 8 => {
                let (acc_reg, lo_reg, hi_reg) = match size {
                    2 => (Register::AX, Register::AX, Register::DX),
                    4 => (Register::EAX, Register::EAX, Register::EDX),
                    8 => (Register::RAX, Register::RAX, Register::RDX),
                    _ => unreachable!(),
                };
                let acc = self.get_reg_val(acc_reg)?;
                let acc_narrow = self.narrow_to(ty, acc);
                let lo = self.builder.ins().imul(acc_narrow, src);
                let hi = self.builder.ins().umulhi(acc_narrow, src);
                self.set_reg_val(lo_reg, lo)?;
                self.set_reg_val(hi_reg, hi)?;
                // CF/OF set if hi != 0
                let zero = self.builder.ins().iconst(ty, 0);
                let of_cf = self.builder.ins().icmp(IntCC::NotEqual, hi, zero);
                self.set_flag(crate::lifter::X86Flag::Carry, of_cf)?;
                self.set_flag(crate::lifter::X86Flag::Overflow, of_cf)?;
            }
            _ => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("mul-size-{}", size),
                });
            }
        }
        Ok(())
    }

    /// Assemble the double-width dividend `high:low` in `work_ty` for DIV/IDIV (bitwise, so signed and unsigned share it).
    fn assemble_dividend(
        &mut self,
        low: Register,
        high: Register,
        work_ty: Type,
    ) -> crate::Result<Value> {
        let shift = (low.size() * 8) as i64;
        let lo = self.get_reg_val(low)?;
        let hi = self.get_reg_val(high)?;
        let lo_w = self.builder.ins().uextend(work_ty, lo);
        let hi_w = self.builder.ins().uextend(work_ty, hi);
        let hi_shifted = self.builder.ins().ishl_imm_u(hi_w, shift);
        Ok(self.builder.ins().bor(hi_shifted, lo_w))
    }

    /// The `(low_acc, high_acc, working_type)` for DIV/IDIV of `size`: dividend `high:low` (e.g. `DX:AX`), divided in twice the operand width.
    fn div_regs(size: usize) -> Option<(Register, Register, Type)> {
        match size {
            1 => Some((Register::AL, Register::AH, types::I16)),
            2 => Some((Register::AX, Register::DX, types::I32)),
            4 => Some((Register::EAX, Register::EDX, types::I64)),
            8 => Some((Register::RAX, Register::RDX, types::I128)),
            _ => None,
        }
    }

    /// A zero of width `ty`; `I128` is a zero-extended `I64`.
    fn wide_zero(&mut self, ty: Type) -> Value {
        if ty == types::I128 {
            let z = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().uextend(types::I128, z)
        } else {
            self.builder.ins().iconst(ty, 0)
        }
    }

    /// Unsigned `dividend / divisor` in `work_ty` as `(quot, rem)`: native `udiv`/`urem`, or the portable loop for `I128`. The divisor must be nonzero.
    fn emit_udivmod(&mut self, dividend: Value, divisor: Value, work_ty: Type) -> (Value, Value) {
        if work_ty == types::I128 {
            self.emit_udivmod128(dividend, divisor)
        } else {
            let q = self.builder.ins().udiv(dividend, divisor);
            let r = self.builder.ins().urem(dividend, divisor);
            (q, r)
        }
    }

    /// IDIV - signed divide, full `high:low` dividend.
    /// 8-bit:  AL = AX / r/m8, AH = AX % r/m8
    /// 16-bit: AX = DX:AX / r/m16, DX = DX:AX % r/m16
    /// 32/64:  EDX:EAX / r/m32 and RDX:RAX / r/m64 respectively.
    pub(super) fn handle_idiv(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_division(instruction, true)
    }

    /// DIV - unsigned divide (same `high:low` dividend layout as IDIV).
    pub(super) fn handle_div(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_division(instruction, false)
    }

    /// Shared DIV/IDIV lowering with #DE checks.
    ///
    /// A zero divisor or a quotient that does not fit branches to a [`ExitReason::DivideError`] VM exit at the faulting
    /// instruction, before any result is committed. [`Optimization::UnsafeImpreciseDivisionExceptions`] drops the guards;
    /// see [`Lifter::divide_error_checks`].
    ///
    /// [`Optimization::UnsafeImpreciseDivisionExceptions`]:
    ///     crate::core::optimize::Optimization::UnsafeImpreciseDivisionExceptions
    fn emit_division(&mut self, instruction: &Instruction, signed: bool) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let (low, high, work_ty) =
            Self::div_regs(size).ok_or_else(|| MicroliftError::UnsupportedInstruction {
                mnemonic: format!("div-size-{}", size),
            })?;
        let ty = size_to_type(size);
        let n = (size * 8) as i64;

        let divisor_n = self.read_operand(instruction, 0, size)?;
        let dividend = self.assemble_dividend(low, high, work_ty)?;

        let checked = self.divide_error_checks();

        // Fault path, shared by the divide-by-zero and overflow guards.
        let de_block = if checked {
            Some(self.builder.create_block())
        } else {
            None
        };

        if let Some(de_block) = de_block {
            let div0 = self.builder.ins().icmp_imm_s(IntCC::Equal, divisor_n, 0);
            let after_div0 = self.builder.create_block();
            self.builder
                .ins()
                .brif(div0, de_block, &[], after_div0, &[]);
            self.builder.switch_to_block(after_div0);
            self.builder.seal_block(after_div0);
        }

        let (quot, rem) = if signed {
            let divisor_w = self.builder.ins().sextend(work_ty, divisor_n);
            let zero = self.wide_zero(work_ty);
            let n_neg = self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, dividend, zero);
            let d_neg = self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, divisor_w, zero);
            // Divide magnitudes and reapply signs (truncate toward zero, remainder takes the dividend's sign); avoids `INT_MIN / -1`.
            let neg_n = self.builder.ins().isub(zero, dividend);
            let abs_n = self.builder.ins().select(n_neg, neg_n, dividend);
            let neg_d = self.builder.ins().isub(zero, divisor_w);
            let abs_d = self.builder.ins().select(d_neg, neg_d, divisor_w);

            let (uq, ur) = self.emit_udivmod(abs_n, abs_d, work_ty);

            let q_neg = self.builder.ins().bxor(n_neg, d_neg);
            let neg_uq = self.builder.ins().isub(zero, uq);
            let quot = self.builder.ins().select(q_neg, neg_uq, uq);
            let neg_ur = self.builder.ins().isub(zero, ur);
            let rem = self.builder.ins().select(n_neg, neg_ur, ur);
            (quot, rem)
        } else {
            let divisor_w = self.builder.ins().uextend(work_ty, divisor_n);
            self.emit_udivmod(dividend, divisor_w, work_ty)
        };

        // Guard 2: quotient overflow.
        //   signed:   the low N bits, sign-extended, must reproduce the quotient.
        //   unsigned: the bits above N must all be zero.
        if let Some(de_block) = de_block {
            let overflow = if signed {
                let q_trunc = self.builder.ins().ireduce(ty, quot);
                let q_back = self.builder.ins().sextend(work_ty, q_trunc);
                self.builder.ins().icmp(IntCC::NotEqual, q_back, quot)
            } else {
                let hi = self.builder.ins().ushr_imm_u(quot, n);
                self.builder.ins().icmp_imm_s(IntCC::NotEqual, hi, 0)
            };
            let after_ovf = self.builder.create_block();
            self.builder
                .ins()
                .brif(overflow, de_block, &[], after_ovf, &[]);

            // Terminate `de_block` before switching away.
            self.builder.switch_to_block(de_block);
            self.builder.seal_block(de_block);
            self.emit_divide_error(instruction)?;

            // Non-faulting path: commit, and leave the builder here for the engine.
            self.builder.switch_to_block(after_ovf);
            self.builder.seal_block(after_ovf);
        }
        let quot_n = self.builder.ins().ireduce(ty, quot);
        let rem_n = self.builder.ins().ireduce(ty, rem);
        self.set_reg_val(low, quot_n)?;
        self.set_reg_val(high, rem_n)?;
        // DIV/IDIV leave the arithmetic flags undefined per the x86 manual.
        self.block_terminated = false;
        Ok(())
    }

    /// Whether `DIV`/`IDIV` still emit their #DE guards.
    ///
    /// [`Optimization::UnsafeImpreciseDivisionExceptions`] turns them off, giving up the fault protocol: an overflowing quotient
    /// is truncated, and a zero divisor traps on the host (native `udiv`) or yields garbage ([`Lifter::emit_udivmod128`]).
    /// Only safe for a guest that never divides by zero or overflows a quotient.
    ///
    /// [`Optimization::UnsafeImpreciseDivisionExceptions`]:
    ///     crate::core::optimize::Optimization::UnsafeImpreciseDivisionExceptions
    fn divide_error_checks(&self) -> bool {
        !self
            .config
            .optimizations
            .allows(crate::core::optimize::Optimization::UnsafeImpreciseDivisionExceptions)
    }

    /// Emit a divide-error VM exit with the PC at the faulting `DIV`/`IDIV` (faults re-execute).
    fn emit_divide_error(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_at = self.runtime_addr(instruction.ip());
        self.sync_to_context(Some(rip_at), ExitReason::DivideError)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Unsigned 128-bit division by restoring shift/subtract, returning `(quotient, remainder)` as `I128`.
    /// The backend lacks `udiv.i128`/`urem.i128`; the dividend is normalized by its leading-zero count and all shifts are
    /// by one (variable `I128` shifts lower expensively on x64). Loop state is carried in block params.
    fn emit_udivmod128(&mut self, n: Value, d: Value) -> (Value, Value) {
        let i128 = types::I128;
        let i64 = types::I64;
        let zero = {
            let z = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().uextend(i128, z)
        };
        let one = {
            let o = self.builder.ins().iconst(types::I64, 1);
            self.builder.ins().uextend(i128, o)
        };

        // A zero dividend has clz=128, handled by the `n < d` fast path; the select keeps the shift in range.
        let hi = self.builder.ins().ushr_imm_u(n, 64);
        let hi = self.builder.ins().ireduce(types::I64, hi);
        let lo = self.builder.ins().ireduce(types::I64, n);
        let hi_nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, hi, 0);
        let clz_hi = self.builder.ins().clz(hi);
        let clz_lo_base = self.builder.ins().clz(lo);
        let clz_lo = self.builder.ins().iadd_imm_s(clz_lo_base, 64);
        let clz = self.builder.ins().select(hi_nonzero, clz_hi, clz_lo);
        let n_is_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, n, 0);
        let zero_i64 = self.builder.ins().iconst(i64, 0);
        let shift = self.builder.ins().select(n_is_zero, zero_i64, clz);
        let n_shifted = self.builder.ins().ishl(n, shift);
        let bit_width = self.builder.ins().iconst(i64, 128);
        let bits = self.builder.ins().isub(bit_width, clz);
        let n_lt_d = self.builder.ins().icmp(IntCC::UnsignedLessThan, n, d);

        let loop_b = self.builder.create_block();
        self.builder.append_block_param(loop_b, i64); // remaining significant bits
        self.builder.append_block_param(loop_b, i128); // normalized dividend
        self.builder.append_block_param(loop_b, i128); // q: quotient so far
        self.builder.append_block_param(loop_b, i128); // r: remainder so far
        let body = self.builder.create_block();
        self.builder.append_block_param(body, i64);
        self.builder.append_block_param(body, i128);
        self.builder.append_block_param(body, i128);
        self.builder.append_block_param(body, i128);
        let exit = self.builder.create_block();
        self.builder.append_block_param(exit, i128); // final q
        self.builder.append_block_param(exit, i128); // final r
        let fast_exit = self.builder.create_block();
        self.builder.append_block_param(fast_exit, i128); // q = 0
        self.builder.append_block_param(fast_exit, i128); // r = n

        // `d` is nonzero, so `n < d` is a complete result and skips the loop.
        self.builder.ins().brif(
            n_lt_d,
            fast_exit,
            &[zero.into(), n.into()],
            loop_b,
            &[bits.into(), n_shifted.into(), zero.into(), zero.into()],
        );

        // loop_b(remaining, n, q, r): stop after the significant bits.
        self.builder.switch_to_block(loop_b);
        let remaining = self.builder.block_params(loop_b)[0];
        let n_work = self.builder.block_params(loop_b)[1];
        let q = self.builder.block_params(loop_b)[2];
        let r = self.builder.block_params(loop_b)[3];
        let done = self.builder.ins().icmp_imm_s(IntCC::Equal, remaining, 0);
        self.builder.ins().brif(
            done,
            exit,
            &[q.into(), r.into()],
            body,
            &[remaining.into(), n_work.into(), q.into(), r.into()],
        );

        // body: consume the MSB, shift dividend and quotient left.
        self.builder.switch_to_block(body);
        let remaining = self.builder.block_params(body)[0];
        let n_work = self.builder.block_params(body)[1];
        let q = self.builder.block_params(body)[2];
        let r = self.builder.block_params(body)[3];
        let n_bit = self.builder.ins().ushr_imm_u(n_work, 127);
        let n_next = self.builder.ins().ishl_imm_u(n_work, 1);
        let r_shifted = self.builder.ins().ishl_imm_u(r, 1);
        let r_new = self.builder.ins().bor(r_shifted, n_bit);
        let ge = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, r_new, d);
        let r_sub = self.builder.ins().isub(r_new, d);
        let r_out = self.builder.ins().select(ge, r_sub, r_new);
        let q_shifted = self.builder.ins().ishl_imm_u(q, 1);
        let q_set = self.builder.ins().bor(q_shifted, one);
        let q_out = self.builder.ins().select(ge, q_set, q_shifted);
        let remaining_next = self.builder.ins().iadd_imm_s(remaining, -1);
        self.builder.ins().jump(
            loop_b,
            &[
                remaining_next.into(),
                n_next.into(),
                q_out.into(),
                r_out.into(),
            ],
        );

        self.builder.switch_to_block(exit);
        let quot = self.builder.block_params(exit)[0];
        let rem = self.builder.block_params(exit)[1];

        // fast_exit(q, r): n < d, so q=0 and r=n.
        self.builder.switch_to_block(fast_exit);
        let fast_quot = self.builder.block_params(fast_exit)[0];
        let fast_rem = self.builder.block_params(fast_exit)[1];

        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, i128);
        self.builder.append_block_param(merge, i128);
        self.builder
            .ins()
            .jump(merge, &[fast_quot.into(), fast_rem.into()]);
        self.builder.switch_to_block(exit);
        self.builder.ins().jump(merge, &[quot.into(), rem.into()]);
        self.builder.switch_to_block(merge);
        let merged_q = self.builder.block_params(merge)[0];
        let merged_r = self.builder.block_params(merge)[1];
        (merged_q, merged_r)
    }


    /// CBW - AX = sign_extend(AL)
    pub(super) fn handle_cbw(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let al = self.get_reg_val(Register::AL)?;
        let ax = self.builder.ins().sextend(types::I16, al);
        self.set_reg_val(Register::AX, ax)
    }

    /// CWDE - EAX = sign_extend(AX)  (upper 32 bits of RAX zeroed per x86-64)
    pub(super) fn handle_cwde(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ax = self.get_reg_val(Register::AX)?;
        let eax = self.builder.ins().sextend(types::I32, ax);
        self.set_reg_val(Register::EAX, eax)
    }

    /// CDQE - RAX = sign_extend(EAX)
    pub(super) fn handle_cdqe(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let eax = self.get_reg_val(Register::EAX)?;
        let rax = self.builder.ins().sextend(types::I64, eax);
        self.set_reg_val(Register::RAX, rax)
    }

    /// CWD - DX:AX = sign_extend(AX)  (DX = 0 or 0xFFFF)
    pub(super) fn handle_cwd(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let ax = self.get_reg_val(Register::AX)?;
        let dx = self.builder.ins().sshr_imm_u(ax, 15);
        self.set_reg_val(Register::DX, dx)
    }

    /// CDQ - EDX:EAX = sign_extend(EAX)  (EDX = 0 or 0xFFFFFFFF)
    pub(super) fn handle_cdq(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let eax = self.get_reg_val(Register::EAX)?;
        let edx = self.builder.ins().sshr_imm_u(eax, 31);
        self.set_reg_val(Register::EDX, edx)
    }

    /// CQO - RDX:RAX = sign_extend(RAX)  (RDX = 0 or -1)
    pub(super) fn handle_cqo(&mut self, _instruction: &Instruction) -> crate::Result<()> {
        let rax = self.get_reg_val(Register::RAX)?;
        let rdx = self.builder.ins().sshr_imm_u(rax, 63);
        self.set_reg_val(Register::RDX, rdx)
    }
}
