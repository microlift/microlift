use crate::MicroliftError;
use crate::arch::x86_64::abi::x86_reg;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::{ExitReason, RegisterContext, TransferKind};
use crate::core::abi::ArgLocation;
use crate::core::{FastCall, FastCallTarget, fat_to_cl_type};
use crate::function_traits::FunctionArgumentType;
use crate::lifter::Lifter;
use cranelift::prelude::*;
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
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
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
        let target_ip = self
            .builder
            .ins()
            .load(types::I64, MemFlagsData::new(), rsp, 0);

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
        if self.config.analysis_mode {
            return self.handle_call_analysis(instruction);
        }
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
                        MemFlagsData::trusted(),
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
                    self.builder
                        .ins()
                        .store(MemFlagsData::new(), ret_addr, new_rsp, 0);

                    self.sync_to_context(None, ExitReason::UnknownJump)?;
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
                    .store(MemFlagsData::new(), ret_addr, new_rsp, 0);

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
                if instruction.memory_base() == Register::RIP
                    && instruction.memory_index() == Register::None
                {
                    let slot_va = instruction.memory_displacement64();
                    if self.import_vas.contains(&slot_va) {
                        return self.emit_import_call(slot_va);
                    }
                }
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                self.emit_indirect_call_exit(target_val)
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("call-{:?}", other),
            }),
        }
    }

    /// Analysis-mode `CALL`: emit a real Cranelift `call`/`call_indirect` to an
    /// external function representing the callee, then **fall through** to the
    /// next instruction. Unlike the execution path this does not touch RSP, push
    /// a return address, jump into the callee, or terminate the block - so the
    /// IR reads as a clean call with its continuation intact, and in-region
    /// callees are no longer inlined as jumps.
    fn handle_call_analysis(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op0_kind() {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = instruction.near_branch_target();
                let fref = self.analysis_call_funcref(target);
                self.builder.ins().call(fref, &[]);
            }
            OpKind::Register => {
                // Indirect call: no statically-known callee, so no external
                // function is declared (and no `analysis_call_targets` entry);
                // `call_indirect` carries the runtime target value directly.
                let target_val = self.get_reg_val(instruction.op0_register())?;
                let sig = self.analysis_call_sig();
                self.builder.ins().call_indirect(sig, target_val, &[]);
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlagsData::new(), ptr, 0);
                let sig = self.analysis_call_sig();
                self.builder.ins().call_indirect(sig, target_val, &[]);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("call-{:?}", other),
                });
            }
        }
        // Execution continues at the next instruction; do not terminate.
        self.block_terminated = false;
        Ok(())
    }

    /// Dispatch a call to a registered import address: either emit a fast
    /// inline call or a slow `ImportedFunctionCall` VM exit.
    fn emit_import_call(&mut self, import_va: u64) -> crate::Result<()> {
        match self.fast_calls.get(&import_va).cloned() {
            Some(fast_call) => self.emit_fast_import_call(fast_call),
            None => self.emit_import_slow_exit(import_va),
        }
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
            .store(MemFlagsData::trusted(), va_val, ctx_ptr, va_offset);

        self.sync_to_context(None, ExitReason::ImportedFunctionCall)?;
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
            .store(MemFlagsData::trusted(), target_val, ctx_ptr, va_offset);

        self.sync_to_context(None, ExitReason::IndirectCall)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// Emit an inline call for a non-variadic fast-call import.
    ///
    /// Argument values are read from wherever the import's [`CallAbi`] says the
    /// guest placed them - a general-purpose register, an XMM register, or a
    /// stack slot relative to RSP - and the return value is written back to the
    /// ABI's return register. Both execution modes share this path; only the
    /// call instruction differs:
    ///
    /// - [`FastCallTarget::Absolute`] (`Jit`): `call_indirect` through the
    ///   symbol's address.
    /// - [`FastCallTarget::Relocatable`] (`RuntimeAssisted`): `call` a declared
    ///   external function, so the object backend emits a named relocation
    ///   rather than an embedded absolute address.
    ///
    /// Stack arguments are read at `RSP + offset`: the fast path replaces the
    /// `CALL` outright, so no return address has been pushed and RSP still
    /// points where the guest's caller left it.
    fn emit_fast_import_call(&mut self, fast_call: FastCall) -> crate::Result<()> {
        let FastCall {
            target,
            arg_types,
            abi,
        } = fast_call;

        let placement = abi.place_args(&arg_types).ok_or_else(|| {
            // Registration demotes such imports to the slow path; emitting a
            // shorter arg list than the signature would fail the IR verifier,
            // so error out loudly instead of miscompiling.
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!(
                    "fast-import-call with unplaceable arguments under {}",
                    abi.name
                ),
            }
        })?;

        let mut call_args: Vec<Value> = Vec::with_capacity(placement.len());
        for (ty, loc) in placement {
            call_args.push(self.load_fast_call_arg(ty, loc)?);
        }

        let call = match target {
            FastCallTarget::Absolute { sig_ref, fn_ptr } => {
                let fn_ptr_val = self.builder.ins().iconst(types::I64, fn_ptr);
                self.builder
                    .ins()
                    .call_indirect(sig_ref, fn_ptr_val, &call_args)
            }
            FastCallTarget::Relocatable(func_ref) => self.builder.ins().call(func_ref, &call_args),
        };

        let results: Vec<Value> = self.builder.inst_results(call).to_vec();
        let ret_ty = arg_types[0];
        if let (Some(&ret_val), Some(loc)) = (results.first(), abi.return_location(ret_ty)) {
            self.store_fast_call_result(ret_ty, loc, ret_val)?;
        }

        // Fast calls do NOT terminate the block; execution continues inline.
        Ok(())
    }

    /// Read one fast-call argument of type `ty` from the guest location `loc`,
    /// converted to the Cranelift type the callee's signature declares.
    fn load_fast_call_arg(
        &mut self,
        ty: FunctionArgumentType,
        loc: ArgLocation,
    ) -> crate::Result<Value> {
        let cl_ty = fat_to_cl_type(ty).ok_or_else(|| MicroliftError::UnsupportedInstruction {
            mnemonic: format!("fast-import-call argument of type {:?}", ty),
        })?;

        let val = match loc {
            ArgLocation::IntReg(reg) => {
                let val = self.get_reg_val(x86_reg(reg)?)?;
                // Guest GPRs are held as I64; narrow to the declared width.
                if cl_ty == types::I64 {
                    val
                } else {
                    self.builder.ins().ireduce(cl_ty, val)
                }
            }
            ArgLocation::FpReg(idx) => {
                // XMM lo (I64) -> narrow for F32 -> bitcast to the float type.
                let lo = self.get_xmm_lo(idx);
                let bits = if cl_ty == types::F32 {
                    self.builder.ins().ireduce(types::I32, lo)
                } else {
                    lo
                };
                self.builder.ins().bitcast(cl_ty, MemFlagsData::new(), bits)
            }
            ArgLocation::Stack(offset) => {
                let rsp = self.get_reg_val(Register::RSP)?;
                self.builder
                    .ins()
                    .load(cl_ty, MemFlagsData::new(), rsp, offset as i32)
            }
        };
        Ok(val)
    }

    /// Write a fast call's return value back into the guest register the ABI
    /// designates, widening integers to the full 64-bit guest register.
    fn store_fast_call_result(
        &mut self,
        ty: FunctionArgumentType,
        loc: ArgLocation,
        ret_val: Value,
    ) -> crate::Result<()> {
        match loc {
            ArgLocation::FpReg(idx) => {
                // Float -> raw bits, zero-extended into the low half of the
                // vector register; the high half is cleared.
                let bits = if ty == FunctionArgumentType::SinglePrecisionFp {
                    let i32_val =
                        self.builder
                            .ins()
                            .bitcast(types::I32, MemFlagsData::new(), ret_val);
                    self.builder.ins().uextend(types::I64, i32_val)
                } else {
                    self.builder
                        .ins()
                        .bitcast(types::I64, MemFlagsData::new(), ret_val)
                };
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.set_xmm_lo(idx, bits);
                self.set_xmm_hi(idx, zero);
            }
            ArgLocation::IntReg(reg) => {
                let i64_val = match ty {
                    FunctionArgumentType::SignedByte
                    | FunctionArgumentType::SignedHalfword
                    | FunctionArgumentType::SignedWord => {
                        self.builder.ins().sextend(types::I64, ret_val)
                    }
                    FunctionArgumentType::UnsignedByte
                    | FunctionArgumentType::UnsignedHalfword
                    | FunctionArgumentType::UnsignedWord => {
                        self.builder.ins().uextend(types::I64, ret_val)
                    }
                    _ => ret_val, // Already I64 / Pointer
                };
                self.set_reg_val(x86_reg(reg)?, i64_val)?;
            }
            // `CallAbi::return_location` never yields a stack location.
            ArgLocation::Stack(_) => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: "fast-import-call with a memory return value".into(),
                });
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

    /// Decrement the LOOP counter register (CX/ECX/RCX, chosen by encoding) and
    /// return an I8 boolean that is true when the decremented counter is nonzero.
    fn decrement_loop_counter(&mut self, instruction: &Instruction) -> crate::Result<Value> {
        // In iced's Code naming the *suffix* names the counter register (the
        // `rel8_16/32/64` part is the branch operand size, which does not
        // affect which counter is decremented).
        let counter_reg = match instruction.code() {
            Code::Loop_rel8_16_CX
            | Code::Loope_rel8_16_CX
            | Code::Loopne_rel8_16_CX
            | Code::Loop_rel8_32_CX
            | Code::Loope_rel8_32_CX
            | Code::Loopne_rel8_32_CX => Register::CX,
            Code::Loop_rel8_16_ECX
            | Code::Loope_rel8_16_ECX
            | Code::Loopne_rel8_16_ECX
            | Code::Loop_rel8_32_ECX
            | Code::Loope_rel8_32_ECX
            | Code::Loopne_rel8_32_ECX
            | Code::Loop_rel8_64_ECX
            | Code::Loope_rel8_64_ECX
            | Code::Loopne_rel8_64_ECX => Register::ECX,
            Code::Loop_rel8_16_RCX
            | Code::Loope_rel8_16_RCX
            | Code::Loopne_rel8_16_RCX
            | Code::Loop_rel8_64_RCX
            | Code::Loope_rel8_64_RCX
            | Code::Loopne_rel8_64_RCX => Register::RCX,
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
        Ok(self.builder.ins().icmp_imm(IntCC::NotEqual, decremented, 0))
    }

    /// LOOP - decrement the counter; branch while it is nonzero.
    pub(super) fn handle_loop(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let nonzero = self.decrement_loop_counter(instruction)?;
        self.handle_conditional_branch(instruction, nonzero)
    }

    /// LOOPE / LOOPZ - decrement the counter; branch while it is nonzero AND ZF=1.
    pub(super) fn handle_loope(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let nonzero = self.decrement_loop_counter(instruction)?;
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().band(nonzero, zf);
        self.handle_conditional_branch(instruction, cond)
    }

    /// LOOPNE / LOOPNZ - decrement the counter; branch while it is nonzero AND ZF=0.
    pub(super) fn handle_loopne(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let nonzero = self.decrement_loop_counter(instruction)?;
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let zf_clear = self.builder.ins().icmp_imm(IntCC::Equal, zf, 0);
        let cond = self.builder.ins().band(nonzero, zf_clear);
        self.handle_conditional_branch(instruction, cond)
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

        self.continue_in_fallthrough(next_block);

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

        self.continue_in_fallthrough(next_block);

        Ok(())
    }
}

#[cfg(test)]
mod analysis_call_tests {
    use crate::arch::x86_64::X86_64;
    use crate::core::{LifterConfig, lift_to_function_with_call_targets};

    fn analysis_ir(bytes: &[u8], base: u64) -> (String, Vec<u64>) {
        let cfg = LifterConfig {
            analysis_mode: true,
            ..Default::default()
        };
        let (func, targets) =
            lift_to_function_with_call_targets::<X86_64>(bytes, base, cfg).unwrap();
        (func.display().to_string(), targets)
    }

    #[test]
    fn direct_call_emits_real_call_op_not_jump() {
        // call 0x100A ; ret ; padding ; (callee) mov eax,1 ; ret
        let bytes = [
            0xE8, 0x05, 0x00, 0x00, 0x00, // call 0x100A
            0xC3, 0x90, 0x90, 0x90, 0x90, // ret + padding
            0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3, // mov eax,1 ; ret
        ];
        let (ir, targets) = analysis_ir(&bytes, 0x1000);
        // The call is a first-class IR op referencing an analysis funcref...
        assert!(ir.contains("call fn0()"), "ir was:\n{ir}");
        assert!(ir.contains("fn0 = u1:0"), "ir was:\n{ir}");
        // ...and the index->VA map resolves fn0 to the callee.
        assert_eq!(targets, vec![0x100A]);
    }

    #[test]
    fn indirect_call_emits_call_indirect_without_target_entry() {
        // call rax (ff d0) ; ret
        let bytes = [0xFF, 0xD0, 0xC3];
        let (ir, targets) = analysis_ir(&bytes, 0x1000);
        assert!(ir.contains("call_indirect"), "ir was:\n{ir}");
        // No statically-known callee, so no analysis_call_targets entry.
        assert!(targets.is_empty());
    }
}
