use crate::MicroliftError;
use crate::context::{ExitReason, RegisterContext, TransferKind};
use crate::function_traits::FunctionArgumentType;
use crate::lifter::{ExecutionMode, Lifter};
use crate::arch::x86_64::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use cranelift_codegen::ir::{FuncRef, SigRef};
use iced_x86::{Code, CodeSize, Instruction, OpKind, Register};

impl<'a> Lifter<'a> {
    pub(super) fn handle_jmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op0_kind() {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = instruction.near_branch_target();
                let target_block = self.get_block_with_kind(target, TransferKind::DirectJump);
                self.builder.ins().jump(target_block, &[]);
                self.block_terminated = true;
            }
            OpKind::Register => {
                // Indirect jump via register - emit a runtime dispatch switch
                let target_val = self.get_reg_val(instruction.op0_register())?;
                self.emit_runtime_indirect_jump(target_val)?;
            }
            OpKind::Memory => {
                // Indirect jump via memory - load the target then dispatch
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
                self.emit_runtime_indirect_jump(target_val)?;
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("jmp-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// Emit a switch-based indirect jump dispatch for a runtime target value.
    /// Jumps to any known block whose offset matches; otherwise exits the VM.
    pub fn emit_runtime_indirect_jump(&mut self, target_val: Value) -> crate::Result<()> {
        let image_base = self.builder.use_var(self.state.image_base);
        // Convert absolute target address -> region offset
        let sub_base = self.builder.ins().isub(target_val, image_base);
        let target_offset = self.builder.ins().iadd_imm(sub_base, self.lift_base as i64);

        let mut switch = cranelift_frontend::Switch::new();
        for (&offset, &block) in &self.blocks {
            switch.set_entry(offset as u128, block);
        }

        let external_block = self.builder.create_block();
        switch.emit(&mut self.builder, target_offset, external_block);

        self.builder.switch_to_block(external_block);
        self.builder.seal_block(external_block);
        self.sync_to_context(Some(target_val), ExitReason::UnknownJump)?;
        self.write_exit_code(TransferKind::IndirectJump as u32);
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    pub(super) fn handle_ret(&mut self, instruction: &Instruction) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Inspection IR: a return terminates the function. The runtime stack
            // pop + dispatch switch + trampoline check are execution-only
            // machinery that only obscure the control-flow view.
            self.builder.ins().return_(&[]);
            self.block_terminated = true;
            return Ok(());
        }

        let imm = instruction.immediate16(); // for RET n

        let rsp = self.get_reg_val(Register::RSP)?;
        let target_ip = self.builder.ins().load(types::I64, MemFlags::new(), rsp, 0);

        let mut new_rsp = self.builder.ins().iadd_imm(rsp, 8);
        if imm > 0 {
            new_rsp = self.builder.ins().iadd_imm(new_rsp, imm as i64);
        }
        self.set_reg_val(Register::RSP, new_rsp)?;

        // Internal Dispatch via Switch
        let mut switch = cranelift_frontend::Switch::new();
        let image_base = self.builder.use_var(self.state.image_base);

        // target_offset = target_ip - image_base + lift_base
        let sub_base = self.builder.ins().isub(target_ip, image_base);
        let target_offset = self.builder.ins().iadd_imm(sub_base, self.lift_base as i64);

        for (&offset, &block) in &self.blocks {
            switch.set_entry(offset as u128, block);
        }

        let external_check_block = self.builder.create_block();
        switch.emit(&mut self.builder, target_offset, external_check_block);

        // Handle Return via Trampoline or Unknown
        self.builder.switch_to_block(external_check_block);
        self.builder.seal_block(external_check_block);

        let trampoline = self
            .builder
            .ins()
            .iconst(types::I64, self.exit_trampoline_address as i64);
        let is_trampoline = self.builder.ins().icmp(IntCC::Equal, target_ip, trampoline);

        let exit_block = self.builder.create_block();
        let unknown_block = self.builder.create_block();

        self.builder
            .ins()
            .brif(is_trampoline, exit_block, &[], unknown_block, &[]);

        // Handle Return via Trampoline
        self.builder.switch_to_block(exit_block);
        self.sync_to_context(Some(target_ip), ExitReason::Return)?;
        self.builder.ins().return_(&[]);

        // Handle Unknown Jump/Return
        self.builder.switch_to_block(unknown_block);
        self.sync_to_context(Some(target_ip), ExitReason::UnknownJump)?;
        self.write_exit_code(TransferKind::Return as u32);
        self.builder.ins().return_(&[]);

        self.block_terminated = true;
        Ok(())
    }

    pub(super) fn handle_call(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op0_kind() {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = instruction.near_branch_target();

                if self.import_vas.contains(&target) {
                    // Registered import - fast-call or slow VM exit.
                    return self.emit_import_call(target);
                }

                // Out-of-region direct call. Emit UnknownJump + DirectCall so
                // the runtime can distinguish this from a real ImportedFunctionCall
                // (a registered import whose callee is always a native function).
                // For DirectCall the callee may itself be a virtualized function,
                // which the runtime will have to account for appropriately.
                if target < self.lift_base || target >= self.lift_end {
                    self.get_block(self.current_ip);

                    let image_base = self.builder.use_var(self.state.image_base);
                    let callee_offset = target.wrapping_sub(self.lift_base);
                    let rebased_callee = self
                        .builder
                        .ins()
                        .iadd_imm(image_base, callee_offset as i64);

                    let ctx_ptr = self.builder.use_var(self.state.context_ptr);
                    let va_offset =
                        std::mem::offset_of!(RegisterContext, imported_function_va) as i32;
                    self.builder.ins().store(
                        MemFlags::trusted(),
                        rebased_callee,
                        ctx_ptr,
                        va_offset,
                    );

                    // Push the return address onto the guest stack exactly as a
                    // real CALL does, so the callee can RET correctly.
                    let ret_offset = self.current_ip.wrapping_sub(self.lift_base);
                    let ret_addr = self.builder.ins().iadd_imm(image_base, ret_offset as i64);
                    let rsp = self.get_reg_val(Register::RSP)?;
                    let new_rsp = self.builder.ins().iadd_imm(rsp, -8);
                    self.set_reg_val(Register::RSP, new_rsp)?;
                    self.builder.ins().store(MemFlags::new(), ret_addr, new_rsp, 0);

                    self.sync_to_context_selective(None, ExitReason::UnknownJump)?;
                    self.write_exit_code(TransferKind::DirectCall as u32);
                    self.builder.ins().return_(&[]);
                    self.block_terminated = true;
                    return Ok(());
                }

                // Internal direct call - push return address and jump.
                let next_ip = instruction.next_ip();
                self.get_block(next_ip);

                let image_base = self.builder.use_var(self.state.image_base);
                let offset = self.current_ip.wrapping_sub(self.lift_base);
                let ret_addr = self.builder.ins().iadd_imm(image_base, offset as i64);

                let rsp = self.get_reg_val(Register::RSP)?;
                let new_rsp = self.builder.ins().iadd_imm(rsp, -8);
                self.set_reg_val(Register::RSP, new_rsp)?;
                self.builder
                    .ins()
                    .store(MemFlags::new(), ret_addr, new_rsp, 0);

                let target_block = self.get_block_with_kind(target, TransferKind::DirectCall);
                self.builder.ins().jump(target_block, &[]);
                self.block_terminated = true;
                Ok(())
            }
            OpKind::Register => {
                let target_val = self.get_reg_val(instruction.op0_register())?;
                self.emit_indirect_call_exit(target_val)
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
                self.emit_indirect_call_exit(target_val)
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("call-{:?}", other),
            }),
        }
    }

    /// Dispatch a call to a registered import address: either emit a fast
    /// inline call or a slow `ImportedFunctionCall` VM exit.
    fn emit_import_call(&mut self, import_va: u64) -> crate::Result<()> {
        match self.config.execution_mode {
            ExecutionMode::Jit => {
                if let Some((sig_ref, fn_ptr, arg_types)) =
                    self.fast_call_data.get(&import_va).cloned()
                {
                    return self.emit_fast_import_call(sig_ref, fn_ptr, arg_types);
                }
            }
            ExecutionMode::RuntimeAssisted => {
                if let Some((func_ref, arg_types)) =
                    self.fast_call_funcrefs.get(&import_va).cloned()
                {
                    return self.emit_fast_import_call_relocatable(func_ref, arg_types);
                }
            }
        }
        self.emit_import_slow_exit(import_va)
    }

    /// Emit an `ImportedFunctionCall` VM exit for a direct import call.
    ///
    /// `self.current_ip` already holds the instruction after the CALL (the
    /// resume point), so we sync and store that as RIP.
    fn emit_import_slow_exit(&mut self, import_va: u64) -> crate::Result<()> {
        // Ensure the resume block exists even without emit_resume_blocks.
        self.get_block(self.current_ip);

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_val = self.builder.ins().iconst(types::I64, import_va as i64);
        let va_offset = std::mem::offset_of!(RegisterContext, imported_function_va) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), va_val, ctx_ptr, va_offset);

        self.sync_to_context_selective(None, ExitReason::ImportedFunctionCall)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// Emit an `IndirectCall` VM exit for an indirect call whose target
    /// address is only known at runtime (unregistered `CALL reg` / `CALL [mem]`).
    ///
    /// `target_val` is the runtime callee address (I64). `self.current_ip`
    /// holds the resume point (next instruction after the CALL).
    ///
    /// RSP is intentionally left unchanged. See `ExitReason::IndirectCall` for
    /// the contract the host must uphold when dispatching the callee.
    fn emit_indirect_call_exit(&mut self, target_val: Value) -> crate::Result<()> {
        self.get_block(self.current_ip);

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = std::mem::offset_of!(RegisterContext, imported_function_va) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), target_val, ctx_ptr, va_offset);

        self.sync_to_context_selective(None, ExitReason::IndirectCall)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// Emit an inline `call_indirect` for a non-variadic fast-call import,
    /// following the SysV AMD64 calling convention:
    ///
    /// - Integer / pointer args: RDI, RSI, RDX, RCX, R8, R9 (up to 6).
    /// - Float args: XMM0–XMM7 (F32 from lo-32 bits; F64 from full lo-64).
    /// - Return value: RAX for integer/pointer, XMM0 lo for floats.
    ///
    /// Arguments beyond the 6th integer register or 8th XMM register are
    /// not yet supported (they would require stack placement).
    fn emit_fast_import_call(
        &mut self,
        sig_ref: SigRef,
        fn_ptr: i64,
        arg_types: Vec<FunctionArgumentType>,
    ) -> crate::Result<()> {
        const INT_ARG_REGS: [Register; 6] = [
            Register::RDI,
            Register::RSI,
            Register::RDX,
            Register::RCX,
            Register::R8,
            Register::R9,
        ];

        let ret_type = arg_types[0];
        let mut int_idx = 0usize;
        let mut xmm_idx = 0u8;
        let mut call_args: Vec<Value> = Vec::new();

        for &ty in arg_types.iter().skip(1) {
            match ty {
                FunctionArgumentType::VariadicList | FunctionArgumentType::None => break,
                FunctionArgumentType::SinglePrecisionFp => {
                    // XMM lo (I64) -> ireduce to I32 -> bitcast to F32.
                    let lo = self.get_xmm_lo(xmm_idx);
                    let i32_val = self.builder.ins().ireduce(types::I32, lo);
                    let f32_val = self
                        .builder
                        .ins()
                        .bitcast(types::F32, MemFlags::new(), i32_val);
                    call_args.push(f32_val);
                    xmm_idx += 1;
                }
                FunctionArgumentType::DoublePrecisionFp => {
                    // XMM lo (I64) -> bitcast to F64.
                    let lo = self.get_xmm_lo(xmm_idx);
                    let f64_val = self.builder.ins().bitcast(types::F64, MemFlags::new(), lo);
                    call_args.push(f64_val);
                    xmm_idx += 1;
                }
                _ => {
                    if int_idx >= INT_ARG_REGS.len() {
                        // Stack-based args are not yet supported; skip them.
                        int_idx += 1;
                        continue;
                    }
                    let reg = INT_ARG_REGS[int_idx];
                    let val = self.get_reg_val(reg)?;
                    let narrowed = match ty {
                        FunctionArgumentType::SignedByte | FunctionArgumentType::UnsignedByte => {
                            self.builder.ins().ireduce(types::I8, val)
                        }
                        FunctionArgumentType::SignedHalfword
                        | FunctionArgumentType::UnsignedHalfword => {
                            self.builder.ins().ireduce(types::I16, val)
                        }
                        FunctionArgumentType::SignedWord | FunctionArgumentType::UnsignedWord => {
                            self.builder.ins().ireduce(types::I32, val)
                        }
                        _ => val, // I64 / Pointer - already correct width
                    };
                    call_args.push(narrowed);
                    int_idx += 1;
                }
            }
        }

        let fn_ptr_val = self.builder.ins().iconst(types::I64, fn_ptr);
        let call = self
            .builder
            .ins()
            .call_indirect(sig_ref, fn_ptr_val, &call_args);
        let results: Vec<Value> = self.builder.inst_results(call).to_vec();

        if !results.is_empty() {
            let ret_val = results[0];
            match ret_type {
                FunctionArgumentType::None => {}
                FunctionArgumentType::SinglePrecisionFp => {
                    // F32 -> bitcast to I32 -> zero-extend to I64 -> XMM0 lo.
                    let i32_val = self
                        .builder
                        .ins()
                        .bitcast(types::I32, MemFlags::new(), ret_val);
                    let i64_val = self.builder.ins().uextend(types::I64, i32_val);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.set_xmm_lo(0, i64_val);
                    self.set_xmm_hi(0, zero);
                }
                FunctionArgumentType::DoublePrecisionFp => {
                    // F64 -> bitcast to I64 -> XMM0 lo.
                    let i64_val = self
                        .builder
                        .ins()
                        .bitcast(types::I64, MemFlags::new(), ret_val);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.set_xmm_lo(0, i64_val);
                    self.set_xmm_hi(0, zero);
                }
                ty => {
                    let i64_val = match ty {
                        FunctionArgumentType::SignedByte => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedByte => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::SignedHalfword => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedHalfword => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::SignedWord => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedWord => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        _ => ret_val, // Already I64 / Pointer
                    };
                    self.set_reg_val(Register::RAX, i64_val)?;
                }
            }
        }

        // Fast calls do NOT terminate the block; execution continues inline.
        Ok(())
    }

    /// Emit an inline `call` for a non-variadic fast-call import in
    /// `RuntimeAssisted` mode.
    ///
    /// Uses `call(func_ref, args)` instead of `call_indirect`, so the
    /// `ObjectModule` backend emits a named relocation rather than an
    /// embedded absolute address. Argument marshaling and return-value
    /// handling follow the same SysV AMD64 ABI rules as `emit_fast_import_call`.
    fn emit_fast_import_call_relocatable(
        &mut self,
        func_ref: FuncRef,
        arg_types: Vec<FunctionArgumentType>,
    ) -> crate::Result<()> {
        const INT_ARG_REGS: [Register; 6] = [
            Register::RDI,
            Register::RSI,
            Register::RDX,
            Register::RCX,
            Register::R8,
            Register::R9,
        ];

        let ret_type = arg_types[0];
        let mut int_idx = 0usize;
        let mut xmm_idx = 0u8;
        let mut call_args: Vec<Value> = Vec::new();

        for &ty in arg_types.iter().skip(1) {
            match ty {
                FunctionArgumentType::VariadicList | FunctionArgumentType::None => break,
                FunctionArgumentType::SinglePrecisionFp => {
                    let lo = self.get_xmm_lo(xmm_idx);
                    let i32_val = self.builder.ins().ireduce(types::I32, lo);
                    let f32_val = self
                        .builder
                        .ins()
                        .bitcast(types::F32, MemFlags::new(), i32_val);
                    call_args.push(f32_val);
                    xmm_idx += 1;
                }
                FunctionArgumentType::DoublePrecisionFp => {
                    let lo = self.get_xmm_lo(xmm_idx);
                    let f64_val = self.builder.ins().bitcast(types::F64, MemFlags::new(), lo);
                    call_args.push(f64_val);
                    xmm_idx += 1;
                }
                _ => {
                    if int_idx >= INT_ARG_REGS.len() {
                        int_idx += 1;
                        continue;
                    }
                    let reg = INT_ARG_REGS[int_idx];
                    let val = self.get_reg_val(reg)?;
                    let narrowed = match ty {
                        FunctionArgumentType::SignedByte | FunctionArgumentType::UnsignedByte => {
                            self.builder.ins().ireduce(types::I8, val)
                        }
                        FunctionArgumentType::SignedHalfword
                        | FunctionArgumentType::UnsignedHalfword => {
                            self.builder.ins().ireduce(types::I16, val)
                        }
                        FunctionArgumentType::SignedWord | FunctionArgumentType::UnsignedWord => {
                            self.builder.ins().ireduce(types::I32, val)
                        }
                        _ => val,
                    };
                    call_args.push(narrowed);
                    int_idx += 1;
                }
            }
        }

        let call = self.builder.ins().call(func_ref, &call_args);
        let results: Vec<Value> = self.builder.inst_results(call).to_vec();

        if !results.is_empty() {
            let ret_val = results[0];
            match ret_type {
                FunctionArgumentType::None => {}
                FunctionArgumentType::SinglePrecisionFp => {
                    let i32_val = self
                        .builder
                        .ins()
                        .bitcast(types::I32, MemFlags::new(), ret_val);
                    let i64_val = self.builder.ins().uextend(types::I64, i32_val);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.set_xmm_lo(0, i64_val);
                    self.set_xmm_hi(0, zero);
                }
                FunctionArgumentType::DoublePrecisionFp => {
                    let i64_val = self
                        .builder
                        .ins()
                        .bitcast(types::I64, MemFlags::new(), ret_val);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    self.set_xmm_lo(0, i64_val);
                    self.set_xmm_hi(0, zero);
                }
                ty => {
                    let i64_val = match ty {
                        FunctionArgumentType::SignedByte => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedByte => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::SignedHalfword => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedHalfword => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::SignedWord => {
                            self.builder.ins().sextend(types::I64, ret_val)
                        }
                        FunctionArgumentType::UnsignedWord => {
                            self.builder.ins().uextend(types::I64, ret_val)
                        }
                        _ => ret_val,
                    };
                    self.set_reg_val(Register::RAX, i64_val)?;
                }
            }
        }

        Ok(())
    }

    // Conditional branches

    pub(super) fn handle_jb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_conditional_branch(instruction, cf)
    }

    pub(super) fn handle_jbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_conditional_branch(instruction, cond)
    }

    /// JA (above): CF=0 and ZF=0
    pub(super) fn handle_ja(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        // JA = branch when CF=0 AND ZF=0 (inverted: don't branch when CF OR ZF)
        self.handle_conditional_branch_inverted(instruction, cf_or_zf)
    }

    /// JAE (above or equal): CF=0
    pub(super) fn handle_jae(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_conditional_branch_inverted(instruction, cf)
    }

    pub(super) fn handle_jz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_conditional_branch(instruction, zf)
    }

    pub(super) fn handle_jnz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_conditional_branch_inverted(instruction, zf)
    }

    /// JG (greater, signed): ZF=0 and SF=OF
    pub(super) fn handle_jg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        // SF == OF
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // cond_not_g = ZF | (SF ^ OF)  - branch when this is 0
        let not_g = self.builder.ins().bor(zf, sf_xor_of);
        self.handle_conditional_branch_inverted(instruction, not_g)
    }

    /// JGE (greater or equal, signed): SF=OF
    pub(super) fn handle_jge(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // JGE: branch when SF == OF (i.e. SF^OF == 0)
        self.handle_conditional_branch_inverted(instruction, sf_xor_of)
    }

    /// JL (less, signed): SF != OF
    pub(super) fn handle_jl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // JL: SF != OF
        self.handle_conditional_branch(instruction, sf_xor_of)
    }

    /// JLE (less or equal, signed): ZF=1 or SF!=OF
    pub(super) fn handle_jle(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bor(zf, sf_xor_of);
        self.handle_conditional_branch(instruction, cond)
    }

    /// JS (sign set): SF=1
    pub(super) fn handle_js(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_conditional_branch(instruction, sf)
    }

    /// JNS (sign clear): SF=0
    pub(super) fn handle_jns(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_conditional_branch_inverted(instruction, sf)
    }

    /// JO (overflow set): OF=1
    pub(super) fn handle_jo(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_conditional_branch(instruction, of)
    }

    /// JNO (overflow clear): OF=0
    pub(super) fn handle_jno(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_conditional_branch_inverted(instruction, of)
    }

    /// JP (parity set): PF=1
    pub(super) fn handle_jp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_conditional_branch(instruction, pf)
    }

    /// JNP (parity clear): PF=0
    pub(super) fn handle_jnp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_conditional_branch_inverted(instruction, pf)
    }

    /// JRCXZ / JECXZ / JCXZ - branch if CX/ECX/RCX == 0
    pub(super) fn handle_jrcxz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let counter = match instruction.code() {
            Code::Jcxz_rel8_16 => self.get_reg_val(Register::CX)?,
            Code::Jecxz_rel8_32 => self.get_reg_val(Register::ECX)?,
            _ => self.get_reg_val(Register::RCX)?,
        };
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, counter, 0);
        self.handle_conditional_branch(instruction, cond)
    }

    pub(super) fn handle_loop(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let counter_reg = match instruction.code() {
            Code::Loop_rel8_16_CX | Code::Loop_rel8_16_ECX | Code::Loop_rel8_16_RCX => Register::CX,
            Code::Loop_rel8_32_CX | Code::Loop_rel8_32_ECX => Register::ECX,
            Code::Loop_rel8_64_ECX => Register::ECX,
            Code::Loop_rel8_64_RCX => Register::RCX,
            _ => match instruction.code_size() {
                CodeSize::Code16 => Register::CX,
                CodeSize::Code32 => Register::ECX,
                _ => Register::RCX,
            },
        };

        let counter = self.get_reg_val(counter_reg)?;
        let one = self
            .builder
            .ins()
            .iconst(size_to_type(counter_reg.size()), 1);
        let decremented = self.builder.ins().isub(counter, one);
        self.set_reg_val(counter_reg, decremented)?;
        self.handle_conditional_branch(instruction, decremented)
    }

    // CMOV family

    /// Generic CMOVcc - move if condition is true.
    fn handle_cmov_cond(&mut self, instruction: &Instruction, cond: Value) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let dst = self.read_operand(instruction, 0, size)?;
        let src = self.read_operand(instruction, 1, size)?;
        let result = self.builder.ins().select(cond, src, dst);
        self.write_operand(instruction, 0, result)
    }

    pub(super) fn handle_cmove(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_cmov_cond(instruction, zf)
    }

    pub(super) fn handle_cmovne(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, zf, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_cmov_cond(instruction, cf)
    }

    pub(super) fn handle_cmovae(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, cf, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmova(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, cf_or_zf, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let cond = self.builder.ins().bxor(sf, of);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovge(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, sf_xor, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovle(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bor(zf, sf_xor);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let not_g = self.builder.ins().bor(zf, sf_xor);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, not_g, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovs(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_cmov_cond(instruction, sf)
    }

    pub(super) fn handle_cmovns(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, sf, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovo(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_cmov_cond(instruction, of)
    }

    pub(super) fn handle_cmovno(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, of, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_cmov_cond(instruction, pf)
    }

    pub(super) fn handle_cmovnp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, pf, 0);
        self.handle_cmov_cond(instruction, cond)
    }

    // Shared branch helpers

    pub fn handle_conditional_branch(
        &mut self,
        instruction: &Instruction,
        cond: Value,
    ) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block_with_kind(target, TransferKind::ConditionalJump);
        let next_block = self.get_block(next_ip);

        self.builder
            .ins()
            .brif(cond, target_block, &[], next_block, &[]);

        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

        Ok(())
    }

    pub fn handle_conditional_branch_inverted(
        &mut self,
        instruction: &Instruction,
        cond: Value,
    ) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block_with_kind(target, TransferKind::ConditionalJump);
        let next_block = self.get_block(next_ip);

        self.builder
            .ins()
            .brif(cond, next_block, &[], target_block, &[]);

        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

        Ok(())
    }
}
