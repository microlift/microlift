use crate::MicroliftError;
use crate::arch::x86_64::X86_64;
use crate::arch::x86_64::abi::x86_reg;
use crate::arch::x86_64::semantics::helpers::size_to_type;
use crate::context::{ExitReason, TransferKind};
use crate::core::GuestArch;
use crate::core::abi::ArgLocation;
use crate::core::bridge::ImportKind;
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
                // Tail `JMP` to a known import: jump straight into it; its `RET` returns to our caller.
                if !self.config.analysis_mode
                    && self.is_import(target)
                    && self.emit_import_tail_jump(target)?
                {
                    return Ok(());
                }
                let target_block = self.get_block_with_kind(target, TransferKind::DirectJump);
                self.builder.ins().jump(target_block, &[]);
                self.block_terminated = true;
            }
            OpKind::Register => {
                let target_val = self.get_reg_val(instruction.op0_register())?;
                self.emit_runtime_indirect_jump(target_val)?;
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, self.guest_flags, ptr, 0);
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

    pub(super) fn handle_ret(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_return(instruction, 8, false)
    }

    /// `RETF`/`RETFQ` - far return: pops offset then CS, each an operand-size slot (2 for `RETFW`, 4 for `RETFD`, 8 for `RETFQ`).
    /// The selector is discarded (flat address space); the rest is identical to a near `RET`.
    pub(super) fn handle_retf(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let operand_size = match instruction.code() {
            Code::Retfw | Code::Retfw_imm16 => 2,
            Code::Retfq | Code::Retfq_imm16 => 8,
            // Retfd/Retfd_imm16, and anything unexpected: 32-bit default.
            _ => 4,
        };
        self.emit_return(instruction, operand_size, true)
    }

    /// Shared `RET`/`RETF` emitter: `slot_size` is the popped slot width, `pop_cs` adds a discarded selector slot; `imm16` forms release argument bytes.
    fn emit_return(
        &mut self,
        instruction: &Instruction,
        slot_size: i64,
        pop_cs: bool,
    ) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Inspection IR: a return terminates the function.
            self.emit_host_return();
            self.block_terminated = true;
            return Ok(());
        }

        let imm = instruction.immediate16(); // for RET n

        let rsp = self.get_reg_val(Register::RSP)?;
        // A narrow far return leaves the target's upper bits undefined; zero-extend like the CPU.
        let slot_ty = size_to_type(slot_size as usize);
        let popped = self.builder.ins().load(slot_ty, self.guest_flags, rsp, 0);
        let target_ip = if slot_ty == types::I64 {
            popped
        } else {
            self.builder.ins().uextend(types::I64, popped)
        };

        let popped_bytes = if pop_cs { slot_size * 2 } else { slot_size };
        let mut new_rsp = self.builder.ins().iadd_imm_s(rsp, popped_bytes);
        if imm > 0 {
            new_rsp = self.builder.ins().iadd_imm_s(new_rsp, imm as i64);
        }
        self.set_reg_val(Register::RSP, new_rsp)?;

        self.emit_return_dispatch(target_ip)
    }

    /// Transfer to a popped return address: dispatch in-region if known, else exit as `UnknownJump`/`Return`,
    /// or via the [`exit_trampoline_address`](crate::core::LifterConfig::exit_trampoline_address) sentinel.
    /// Split from [`Self::emit_return`] so an import tail `JMP` can perform its callee's return.
    fn emit_return_dispatch(&mut self, target_ip: Value) -> crate::Result<()> {
        let target_offset = self.to_lift_va(target_ip);

        let external_check_block = self.builder.create_block();
        self.emit_region_switch(target_offset, external_check_block);

        self.builder.switch_to_block(external_check_block);
        self.builder.seal_block(external_check_block);

        // The trampoline compare is only emitted when the host declared a sentinel.
        let unknown_block = if let Some(trampoline_addr) = self.config.exit_trampoline_address {
            let trampoline = self
                .builder
                .ins()
                .iconst(types::I64, trampoline_addr as i64);
            let is_trampoline = self.builder.ins().icmp(IntCC::Equal, target_ip, trampoline);

            let exit_block = self.builder.create_block();
            let unknown_block = self.builder.create_block();

            self.builder
                .ins()
                .brif(is_trampoline, exit_block, &[], unknown_block, &[]);

            // Handle Return via Trampoline. Not linkable.
            self.builder.switch_to_block(exit_block);
            self.sync_to_context(Some(target_ip), ExitReason::Return)?;
            self.emit_host_return();

            unknown_block
        } else {
            external_check_block
        };

        // Handle Unknown Jump/Return. Linkable.
        self.builder.switch_to_block(unknown_block);
        self.sync_to_context(Some(target_ip), ExitReason::UnknownJump)?;
        self.write_exit_code(TransferKind::Return as u32);
        if self.returns_through_host_frames() {
            // A caller may be waiting on the host stack; linking would strand its frame (see `LiftEngine::returns_through_host_frames`).
            self.emit_host_return();
        } else {
            // After the trampoline test, so a `RET` to the sentinel stays unlinkable, and after the sync.
            self.emit_callret_pop(target_ip);
            self.emit_exit(Some(target_ip));
        }

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

                if self.is_import(target) {
                    return self.emit_import_call(target);
                }

                // Out-of-region direct call: `UnknownJump` + `DirectCall`, since a `DirectCall` callee may be virtualized.
                if !self.in_region(target) {
                    self.get_block(self.current_ip);

                    let rebased_callee = self.runtime_addr(target);

                    let ctx_ptr = self.builder.use_var(self.state.context_ptr);
                    let va_offset = <X86_64 as GuestArch>::imported_function_va_offset();
                    self.ctx_store(rebased_callee, ctx_ptr, va_offset);

                    let ret_addr = self.runtime_current_ip();
                    let rsp = self.get_reg_val(Register::RSP)?;
                    let new_rsp = self.builder.ins().iadd_imm_s(rsp, -8);
                    self.set_reg_val(Register::RSP, new_rsp)?;
                    self.builder
                        .ins()
                        .store(self.guest_flags, ret_addr, new_rsp, 0);

                    // Record the return resume point so the callee's `RET` finds it without a lookup.
                    self.emit_callret_push(instruction.next_ip(), ret_addr);

                    self.sync_to_context(None, ExitReason::UnknownJump)?;
                    self.write_exit_code(TransferKind::DirectCall as u32);

                    // Prefer a real host call so the callee's `ret` is predicted; emits nothing and returns `false` when off.
                    if self.emit_host_frame_call(target, rebased_callee, self.current_ip)? {
                        // Call not possible (no cell yet, or host stack too deep): take the ordinary exit.
                        self.emit_static_exit(
                            target,
                            Some(rebased_callee),
                            crate::core::link::LinkSiteKind::StaticEdge,
                        );
                        self.block_terminated = true;
                        return Ok(());
                    }

                    // Statically linkable: the callee is a lift-time constant, so it earns a cell.
                    // `current_ip` is the *return* address, hence `Some(rebased_callee)`.
                    self.emit_static_exit(
                        target,
                        Some(rebased_callee),
                        crate::core::link::LinkSiteKind::StaticEdge,
                    );
                    self.block_terminated = true;
                    return Ok(());
                }

                let next_ip = instruction.next_ip();
                self.get_block(next_ip);

                let ret_addr = self.runtime_current_ip();

                let rsp = self.get_reg_val(Register::RSP)?;
                let new_rsp = self.builder.ins().iadd_imm_s(rsp, -8);
                self.set_reg_val(Register::RSP, new_rsp)?;
                self.builder
                    .ins()
                    .store(self.guest_flags, ret_addr, new_rsp, 0);

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
                    if self.is_import(slot_va) {
                        return self.emit_import_call(slot_va);
                    }
                }
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, self.guest_flags, ptr, 0);
                self.emit_indirect_call_exit(target_val)
            }
            other => Err(MicroliftError::UnsupportedInstruction {
                mnemonic: format!("call-{:?}", other),
            }),
        }
    }

    /// Analysis-mode `CALL`: a real `call`/`call_indirect` that falls through; no RSP push, jump or terminator.
    fn handle_call_analysis(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op0_kind() {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = instruction.near_branch_target();
                let fref = self.analysis_call_funcref(target);
                self.builder.ins().call(fref, &[]);
            }
            OpKind::Register => {
                // Indirect: no declared function; `call_indirect` carries the target.
                let target_val = self.get_reg_val(instruction.op0_register())?;
                let sig = self.analysis_call_sig();
                self.builder.ins().call_indirect(sig, target_val, &[]);
            }
            OpKind::Memory => {
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self
                    .builder
                    .ins()
                    .load(types::I64, self.guest_flags, ptr, 0);
                let sig = self.analysis_call_sig();
                self.builder.ins().call_indirect(sig, target_val, &[]);
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("call-{:?}", other),
                });
            }
        }
        self.block_terminated = false;
        Ok(())
    }

    /// Dispatch a call to a registered import: bridge, fast inline call or slow `ImportedFunctionCall` exit, in that order.
    fn emit_import_call(&mut self, import_va: u64) -> crate::Result<()> {
        if let Some(ImportKind::Bridge(bridge)) = self.lookup_import(import_va) {
            // A bridge that exits parks the resume PC at the next instruction, which needs a block.
            self.get_block(self.current_ip);
            let callee = self.builder.ins().iconst(types::I64, import_va as i64);
            return self.emit_bridge_call(&bridge, callee, None, ExitReason::ImportedFunctionCall);
        }
        match self.fast_call_for(import_va) {
            Some(fast_call) => self.emit_fast_import_call(fast_call),
            None => self.emit_import_slow_exit(import_va),
        }
    }

    /// Emit an import reached by tail `JMP`: the call inline, then the return its `RET` would perform.
    /// Returns `false` for a slow import or a fast one whose ABI cannot place its arguments; the caller emits the ordinary exit.
    /// The return address is popped before the call so stack arguments sit at `RSP+0`.
    fn emit_import_tail_jump(&mut self, import_va: u64) -> crate::Result<bool> {
        let bridge = match self.lookup_import(import_va) {
            Some(ImportKind::Bridge(b)) => Some(b),
            _ => None,
        };
        let fast = match bridge {
            Some(_) => None,
            None => self.fast_call_for(import_va),
        };
        if bridge.is_none() && fast.is_none() {
            return Ok(false);
        }

        let rsp = self.get_reg_val(Register::RSP)?;
        let ret_ip = self
            .builder
            .ins()
            .load(types::I64, self.guest_flags, rsp, 0);
        let new_rsp = self.builder.ins().iadd_imm_s(rsp, 8);
        self.set_reg_val(Register::RSP, new_rsp)?;

        match (bridge, fast) {
            (Some(bridge), _) => {
                let callee = self.builder.ins().iconst(types::I64, import_va as i64);
                // An exiting bridge leaves the guest past the stub: resume at the popped return address.
                self.emit_bridge_call(
                    &bridge,
                    callee,
                    Some(ret_ip),
                    ExitReason::ImportedFunctionCall,
                )?;
            }
            (None, Some(fast_call)) => self.emit_fast_import_call(fast_call)?,
            (None, None) => unreachable!("checked above"),
        }

        self.emit_return_dispatch(ret_ip)?;
        Ok(true)
    }

    /// `ImportedFunctionCall` VM exit for a direct import call; `current_ip` (the next instruction) is stored as RIP.
    fn emit_import_slow_exit(&mut self, import_va: u64) -> crate::Result<()> {
        // Ensure the resume block exists even without VmResumptionPoints::All.
        self.get_block(self.current_ip);

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_val = self.builder.ins().iconst(types::I64, import_va as i64);
        let va_offset = <X86_64 as GuestArch>::imported_function_va_offset();
        self.ctx_store(va_val, ctx_ptr, va_offset);

        self.sync_to_context(None, ExitReason::ImportedFunctionCall)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// `IndirectCall` VM exit for a runtime-only target (`CALL reg` / `CALL [mem]`); `current_ip` is the resume point.
    /// RSP is left unchanged (see `ExitReason::IndirectCall`) unless a link row hits: then the push happens in the hit arm
    /// and execution continues at the callee; see [`LiftEngine::emit_dynamic_link`].
    ///
    /// [`LiftEngine::emit_dynamic_link`]: crate::core::LiftEngine::emit_dynamic_link
    fn emit_indirect_call_exit(&mut self, target_val: Value) -> crate::Result<()> {
        self.get_block(self.current_ip);

        // An indirect-call bridge usually handles the call inline; it returns non-zero for guest code.
        if let Some(bridge) = self.config.indirect_call_bridge.clone() {
            return self.emit_bridge_call(&bridge, target_val, None, ExitReason::IndirectCall);
        }

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = <X86_64 as GuestArch>::imported_function_va_offset();
        self.ctx_store(target_val, ctx_ptr, va_offset);

        let ret_addr = self.runtime_current_ip();
        let resume = self.current_ip;
        self.emit_dynamic_link(target_val, |lifter| {
            let rsp = lifter.get_reg_val(Register::RSP)?;
            let new_rsp = lifter.builder.ins().iadd_imm_s(rsp, -8);
            lifter.set_reg_val(Register::RSP, new_rsp)?;
            lifter
                .builder
                .ins()
                .store(lifter.guest_flags, ret_addr, new_rsp, 0);
            lifter.emit_callret_push(resume, ret_addr);
            lifter.sync_to_context(Some(target_val), ExitReason::UnknownJump)?;
            lifter.write_exit_code(TransferKind::IndirectCall as u32);
            Ok(())
        })?;

        // Nothing links: the plain exit, stack untouched.
        self.sync_to_context(None, ExitReason::IndirectCall)?;
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Emit an inline call for a non-variadic fast-call import.
    ///
    /// Arguments are read per the import's [`CallAbi`] (GPR, XMM, or stack slot at `RSP + offset`; no return address was pushed)
    /// and the result is written to the ABI's return register.
    /// [`FastCallTarget::Absolute`] (`Jit`) uses `call_indirect`; [`FastCallTarget::Relocatable`] (`RuntimeAssisted`)
    /// calls a declared function so the object gets a named relocation.
    ///
    /// # A fast import must not touch the `RegisterContext`
    ///
    /// Reading or writing it, directly or by re-entering guest code against it, is **undefined behaviour**:
    /// nothing is published or reloaded, and the region overwrites whatever the callee left there.
    /// Use a [bridge](crate::core::bridge) or a slow import instead. The rule also underwrites [`crate::core::regsync`]:
    /// a register the region never defined still holds its entry value past a fast call.
    fn emit_fast_import_call(&mut self, fast_call: FastCall) -> crate::Result<()> {
        let FastCall {
            target,
            arg_types,
            abi,
        } = fast_call;

        let placement = abi.place_args(&arg_types).ok_or_else(|| {
            // Registration demotes such imports to slow; a short arg list would fail the verifier.
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

        // Pinned registers must be in the context while host code runs; it may read the stack or re-enter guest code.
        self.publish_pinned()?;

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

        Ok(())
    }

    /// Read one fast-call argument of type `ty` from `loc`, converted to the callee's declared type.
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
                // `bitcast`'s memflags carry endianness, not aliasing.
                self.builder.ins().bitcast(cl_ty, MemFlagsData::new(), bits)
            }
            ArgLocation::Stack(offset) => {
                // A stack argument is guest memory, not context.
                let rsp = self.get_reg_val(Register::RSP)?;
                let flags = self.guest_flags;
                self.builder.ins().load(cl_ty, flags, rsp, offset as i32)
            }
        };
        Ok(val)
    }

    /// Write a fast call's return value to the ABI's guest register, widened to 64 bits.
    fn store_fast_call_result(
        &mut self,
        ty: FunctionArgumentType,
        loc: ArgLocation,
        ret_val: Value,
    ) -> crate::Result<()> {
        match loc {
            ArgLocation::FpReg(idx) => {
                // Float -> raw bits in the vector register's low half; the high half is cleared.
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

    /// JRCXZ / JECXZ / JCXZ - branch if CX/ECX/RCX == 0
    pub(super) fn handle_jrcxz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let counter = match instruction.code() {
            Code::Jcxz_rel8_16 => self.get_reg_val(Register::CX)?,
            Code::Jecxz_rel8_32 => self.get_reg_val(Register::ECX)?,
            _ => self.get_reg_val(Register::RCX)?,
        };
        let cond = self.builder.ins().icmp_imm_s(IntCC::Equal, counter, 0);
        self.handle_conditional_branch(instruction, cond)
    }

    /// Decrement the LOOP counter (CX/ECX/RCX per encoding); returns an `I8` true when it is nonzero.
    fn decrement_loop_counter(&mut self, instruction: &Instruction) -> crate::Result<Value> {
        // The encoding suffix names the counter; `rel8_16/32/64` is only the branch operand size.
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
        Ok(self
            .builder
            .ins()
            .icmp_imm_s(IntCC::NotEqual, decremented, 0))
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
        let zf_clear = self.builder.ins().icmp_imm_s(IntCC::Equal, zf, 0);
        let cond = self.builder.ins().band(nonzero, zf_clear);
        self.handle_conditional_branch(instruction, cond)
    }


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
        assert!(ir.contains("call fn0()"), "ir was:\n{ir}");
        assert!(ir.contains("fn0 = u1:0"), "ir was:\n{ir}");
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
