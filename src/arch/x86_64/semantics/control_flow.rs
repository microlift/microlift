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
                // A `JMP` to a known import is a tail call: the stub jumps straight into the
                // imported function, which returns to *our* caller. Handled here rather than
                // exiting to a host that must re-recognise the stub before every entry.
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
        // Near return: pop a full 64-bit return address off the stack.
        self.emit_return(instruction, 8, false)
    }

    /// `RETF`/`RETFQ` - far return.
    ///
    /// Pops the return offset and then the code selector, both stored in
    /// operand-size-wide stack slots: 2 bytes each for `RETFW` (0x66 prefix),
    /// 4 for `RETFD` (the 64-bit-mode default), 8 for `RETFQ` (REX.W).
    ///
    /// Microlift models a flat address space with no segmentation, so the
    /// popped CS selector is discarded (it cannot change the linear address of
    /// the target) - only the stack adjustment and the branch to the popped
    /// offset are architecturally observable here. Everything past the pop
    /// (in-region dispatch, exit trampoline, `UnknownJump`) is identical to a
    /// near `RET`.
    pub(super) fn handle_retf(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let operand_size = match instruction.code() {
            Code::Retfw | Code::Retfw_imm16 => 2,
            Code::Retfq | Code::Retfq_imm16 => 8,
            // Retfd/Retfd_imm16, and anything unexpected: 32-bit default.
            _ => 4,
        };
        self.emit_return(instruction, operand_size, true)
    }

    /// Shared `RET`/`RETF` emitter.
    ///
    /// `slot_size` is the width in bytes of each stack slot popped (8 for a
    /// near return, the operand size for a far one) and `pop_cs` selects
    /// whether a second (discarded) selector slot follows the return offset.
    /// `RET/RETF imm16` additionally releases `imm16` bytes of arguments.
    fn emit_return(
        &mut self,
        instruction: &Instruction,
        slot_size: i64,
        pop_cs: bool,
    ) -> crate::Result<()> {
        if self.config.analysis_mode {
            // Inspection IR: a return terminates the function. The stack pop, dispatch
            // switch and trampoline check are execution-only machinery.
            self.emit_host_return();
            self.block_terminated = true;
            return Ok(());
        }

        let imm = instruction.immediate16(); // for RET n

        let rsp = self.get_reg_val(Register::RSP)?;
        // A narrow far return leaves the upper bits of the target undefined on
        // real hardware; zero-extending the popped slot matches what the CPU
        // loads into RIP.
        let slot_ty = size_to_type(slot_size as usize);
        let popped = self.builder.ins().load(slot_ty, self.guest_flags, rsp, 0);
        let target_ip = if slot_ty == types::I64 {
            popped
        } else {
            self.builder.ins().uextend(types::I64, popped)
        };

        // Pop the return offset, then the discarded CS selector for a far
        // return, then any `imm16` argument bytes.
        let popped_bytes = if pop_cs { slot_size * 2 } else { slot_size };
        let mut new_rsp = self.builder.ins().iadd_imm_s(rsp, popped_bytes);
        if imm > 0 {
            new_rsp = self.builder.ins().iadd_imm_s(new_rsp, imm as i64);
        }
        self.set_reg_val(Register::RSP, new_rsp)?;

        self.emit_return_dispatch(target_ip)
    }

    /// Transfer control to a popped return address: dispatch into the region if
    /// it names a known block, otherwise exit - as an `UnknownJump`/`Return`,
    /// or cleanly via the trampoline when
    /// [`LifterConfig::exit_trampoline_address`](crate::core::LifterConfig::exit_trampoline_address)
    /// named one and the address is it.
    ///
    /// Split out of [`Self::emit_return`] so an import tail `JMP` can perform
    /// exactly the return its callee's `RET` would have, after the stack pointer
    /// has already been adjusted.
    fn emit_return_dispatch(&mut self, target_ip: Value) -> crate::Result<()> {
        let target_offset = self.to_lift_va(target_ip);

        let external_check_block = self.builder.create_block();
        self.emit_region_switch(target_offset, external_check_block);

        self.builder.switch_to_block(external_check_block);
        self.builder.seal_block(external_check_block);

        // The trampoline test - constant, compare, branch - is paid by every
        // out-of-region RET, so it is only emitted for a host that declared a
        // sentinel. Without one, every such return is "keep going at this PC",
        // which is also the linkable exit.
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
            // A caller may be waiting on the host stack for exactly this, and
            // linking instead would strand its frame - see
            // `LiftEngine::returns_through_host_frames`. Whoever catches the
            // return decides whether it was the one they wanted.
            self.emit_host_return();
        } else {
            // After the trampoline test, so a `RET` to the host's sentinel stays
            // deliberately unlinkable - and after the sync, so a miss leaves the
            // context exactly as the host has always seen it.
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
                    // Registered import - fast-call or slow VM exit.
                    return self.emit_import_call(target);
                }

                // Out-of-region direct call. `UnknownJump` + `DirectCall` so the runtime can
                // distinguish it from a real `ImportedFunctionCall`, whose callee is always
                // native - a `DirectCall` callee may itself be virtualized.
                if !self.in_region(target) {
                    self.get_block(self.current_ip);

                    let rebased_callee = self.runtime_addr(target);

                    let ctx_ptr = self.builder.use_var(self.state.context_ptr);
                    let va_offset = <X86_64 as GuestArch>::imported_function_va_offset();
                    self.ctx_store(rebased_callee, ctx_ptr, va_offset);

                    // Push the return address onto the guest stack exactly as a
                    // real CALL does, so the callee can RET correctly.
                    let ret_addr = self.runtime_current_ip();
                    let rsp = self.get_reg_val(Register::RSP)?;
                    let new_rsp = self.builder.ins().iadd_imm_s(rsp, -8);
                    self.set_reg_val(Register::RSP, new_rsp)?;
                    self.builder
                        .ins()
                        .store(self.guest_flags, ret_addr, new_rsp, 0);

                    // Record where this call comes back to, so the callee's
                    // `RET` can find its way here without a lookup. The cell it
                    // names is one of *this* region's - the return address is a
                    // resume point of the region making the call.
                    self.emit_callret_push(instruction.next_ip(), ret_addr);

                    self.sync_to_context(None, ExitReason::UnknownJump)?;
                    self.write_exit_code(TransferKind::DirectCall as u32);

                    // First choice: make it a real host call, so the callee's `RET`
                    // is a `ret` the return-address stack predicts and lands at the
                    // instruction after this one rather than back through this
                    // region's entry dispatch. Emits nothing and returns `false`
                    // when that is off, leaving the exit below as it was.
                    if self.emit_host_frame_call(target, rebased_callee, self.current_ip)? {
                        // The builder is in the arm where the call could not be
                        // made - no cell yet, or the host stack is as deep as it
                        // may go - which takes the ordinary exit.
                        self.emit_static_exit(
                            target,
                            Some(rebased_callee),
                            crate::core::link::LinkSiteKind::StaticEdge,
                        );
                        self.block_terminated = true;
                        return Ok(());
                    }

                    // Linkable, and *statically* linkable: the return address is pushed and
                    // `rip` is the callee, so the host has nothing left to do but enter it -
                    // and the callee is a lift-time constant, which is what earns it a cell.
                    // `Some(rebased_callee)` because `current_ip` here is the *return*
                    // address, not where execution continues.
                    self.emit_static_exit(
                        target,
                        Some(rebased_callee),
                        crate::core::link::LinkSiteKind::StaticEdge,
                    );
                    self.block_terminated = true;
                    return Ok(());
                }

                // Internal direct call - push return address and jump.
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
                // Indirect call: no statically-known callee, so no external function is
                // declared; `call_indirect` carries the runtime target directly.
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
        // Execution continues at the next instruction; do not terminate.
        self.block_terminated = false;
        Ok(())
    }

    /// Dispatch a call to a registered import address: a runtime bridge, a fast
    /// inline call, or a slow `ImportedFunctionCall` VM exit, in that order of
    /// preference.
    fn emit_import_call(&mut self, import_va: u64) -> crate::Result<()> {
        if let Some(ImportKind::Bridge(bridge)) = self.lookup_import(import_va) {
            // A bridge that exits parks the resume PC at the next instruction,
            // so that address needs a block to come back to.
            self.get_block(self.current_ip);
            let callee = self.builder.ins().iconst(types::I64, import_va as i64);
            return self.emit_bridge_call(&bridge, callee, None, ExitReason::ImportedFunctionCall);
        }
        match self.fast_call_for(import_va) {
            Some(fast_call) => self.emit_fast_import_call(fast_call),
            None => self.emit_import_slow_exit(import_va),
        }
    }

    /// Emit an import reached by tail `JMP`: perform the call inline, then the
    /// return the imported function's own `RET` would have performed.
    ///
    /// Returns `false` when the import has no inline form (a slow import, or a
    /// fast one whose ABI cannot place its arguments), leaving the caller to
    /// emit the ordinary out-of-region jump exit - a slow import cannot be
    /// tail-jumped inline, because the host would have to service the call and
    /// then perform a return the lifted code never described.
    ///
    /// The return address is popped *before* the call, which puts the stack in
    /// exactly the state the callee's ABI expects: stack arguments at `RSP+0`,
    /// as at any other call site the fast path replaces.
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
                // If the bridge exits, the guest is architecturally *past* the
                // stub already: resume at the popped return address.
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

    /// Emit an `ImportedFunctionCall` VM exit for a direct import call.
    ///
    /// `self.current_ip` already holds the instruction after the CALL (the
    /// resume point), so we sync and store that as RIP.
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

    /// Emit an `IndirectCall` VM exit for an indirect call whose target
    /// address is only known at runtime (unregistered `CALL reg` / `CALL [mem]`).
    ///
    /// `target_val` is the runtime callee address (I64). `self.current_ip`
    /// holds the resume point (next instruction after the CALL).
    ///
    /// RSP is intentionally left unchanged. See `ExitReason::IndirectCall` for
    /// the contract the host must uphold when dispatching the callee.
    ///
    /// Unless the callee links. The host's decision - native function or guest
    /// code - is what keeps the push out of this exit, and a matching row makes
    /// it for us: it is guest code, and it has been translated. So a hit does
    /// the push here and continues at the callee, exactly as an out-of-region
    /// direct `CALL` does; see [`LiftEngine::emit_dynamic_link`] for why the
    /// commit is in the hit arm rather than before the probe.
    ///
    /// [`LiftEngine::emit_dynamic_link`]: crate::core::LiftEngine::emit_dynamic_link
    fn emit_indirect_call_exit(&mut self, target_val: Value) -> crate::Result<()> {
        self.get_block(self.current_ip);

        // With an indirect-call bridge the runtime dispatches host-side and usually
        // handles the call inline, so a `dlsym`ed pointer or `call *%r12` costs a
        // call rather than a VM exit. It still returns non-zero for guest code.
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

        // Reached when nothing links, which is every lift that is not linking at
        // all: the exit the host has always been given, stack untouched.
        self.sync_to_context(None, ExitReason::IndirectCall)?;
        self.emit_host_return();
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
    ///
    /// # A fast import must not touch the `RegisterContext`
    ///
    /// Reading or writing it - directly, or by re-entering guest code against
    /// the same context - is **undefined behaviour**. This path publishes no
    /// registers and reloads none: the region's state is in machine registers
    /// and stays there across the call, so the context holds whatever it held
    /// at entry and the region will overwrite anything the callee put there.
    /// A callee that needs the whole context is a
    /// [bridge](crate::core::bridge), and one the host must service is a slow
    /// import; both exist precisely for this.
    ///
    /// The rule also underwrites [`crate::core::regsync`]: with it, a register
    /// this region never defined provably still holds its entry-loaded value
    /// even past a fast call, so its store back really is idempotent. (If a
    /// future caller needs the escape hatch, the shape is a per-import flag
    /// asking for full materialisation around the call, not a blanket
    /// pessimisation of every import.)
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

        // Host code is about to run with this region's state in registers, so
        // anything pinned must be in the context for the duration - a host import
        // may read the guest stack, or call back into guest code below the frame.
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
