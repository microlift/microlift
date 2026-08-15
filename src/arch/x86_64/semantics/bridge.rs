//! Emission for runtime-supplied bridges (see [`crate::core::bridge`]).
//!
//! One emitter serves all three bridge sites - a registered import call, an
//! import reached by tail `JMP`, and a runtime-resolved indirect call - because
//! they share one contract:
//!
//! ```text
//! extern "C" fn(ctx: *mut RegisterContext, extra: u64) -> u32
//! ```
//!
//! `extra` is the import VA at an import site and the resolved callee address at
//! an indirect-call site; either way it is also parked in
//! `imported_function_va` before the call, so a bridge that ignores its second
//! argument (or a crash inside one) still finds the callee in the context.
//!
//! The return value is the whole protocol: **0** means handled, and the lifted
//! code falls through inline with the declared clobbers reloaded; **non-zero**
//! means return to the host immediately, with `exit_reason`/`exit_code` left
//! exactly as the bridge wrote them - except
//! [`RESUME_LINKED`](crate::core::bridge::RESUME_LINKED), which additionally
//! declares the exit linkable.

use crate::arch::x86_64::Lifter;
use crate::context::{ExitReason, RegisterContext};
use crate::core::bridge::{BridgeImport, BridgeTarget, RESUME_LINKED, RegMask};
use cranelift::prelude::*;

impl<'a> Lifter<'a> {
    /// Emit a call to `bridge`, and the two-way branch on its result.
    ///
    /// - `callee` is stored into `imported_function_va` and passed as the
    ///   bridge's second argument.
    /// - `pc` is the guest PC published before the call - the address the host
    ///   would resume at if the bridge exits. `None` means "the instruction
    ///   after the current one" (`self.current_ip`), which is the right resume
    ///   point for a call site.
    /// - `reason` seeds `exit_reason` so a bridge that returns non-zero without
    ///   writing anything still produces a well-formed exit.
    ///
    /// On return the builder is positioned in the *handled* continuation and the
    /// block is **not** terminated: the caller keeps emitting inline.
    pub(crate) fn emit_bridge_call(
        &mut self,
        bridge: &BridgeImport,
        callee: Value,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let effects = bridge.effects;
        // RSP is always published: every bridge needs a coherent guest stack,
        // whether or not it declared reading it (stack arguments, the return
        // address, a longjmp target). RIP goes out with it via
        // `flush_rip_and_reason`.
        let reads = effects.reads.union(RegMask::gpr(RSP_GPR_INDEX));

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = std::mem::offset_of!(RegisterContext, imported_function_va) as i32;
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), callee, ctx_ptr, va_offset);

        self.sync_regs_masked(reads)?;
        self.publish_pinned()?;
        self.flush_rip_and_reason(pc, reason)?;

        let bridge_ptr = self.bridge_target_ptr(&bridge.target);
        let sig = self.bridge_sig();
        let call = self
            .builder
            .ins()
            .call_indirect(sig, bridge_ptr, &[ctx_ptr, callee]);
        let handled = self.builder.inst_results(call)[0];

        let exit_block = self.builder.create_block();
        let continue_block = self.builder.create_block();
        // Non-zero -> exit; zero -> fall through inline.
        self.builder
            .ins()
            .brif(handled, exit_block, &[], continue_block, &[]);

        // The bridge wants out. Everything it did *not* declare as a clobber is
        // still live in SSA and is authoritative, so publish that before
        // returning; the declared clobbers are already in the context, written
        // by the bridge itself. `exit_reason`/`exit_code` are untouched here -
        // whatever the bridge stored is what the host sees.
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.sync_regs_masked(effects.writes.complement())?;
        self.reload_pinned()?;
        self.emit_bridge_exit(handled)?;

        // Handled inline: pull the declared clobbers back into SSA and carry on.
        self.builder.switch_to_block(continue_block);
        self.builder.seal_block(continue_block);
        self.load_regs_masked(effects.writes)?;
        self.block_terminated = false;
        Ok(())
    }

    /// Terminate the bridge's exiting path, linking when it asked to.
    ///
    /// The PC is **loaded back out of the context** rather than reused from the
    /// value published before the call, because moving it is exactly what a
    /// bridge that returns [`RESUME_LINKED`] does: an `objc_msgSend` standing in
    /// for the send, a `longjmp`, an indirect call that turned out to target
    /// guest code. The value published before the call is the resume point that
    /// would have applied had the bridge handled it, which is the wrong address
    /// to link on.
    ///
    /// With linking off - or on a guest with nowhere to put the answer - this
    /// collapses to the bare `return_` it has always been, and the comparison
    /// against `RESUME_LINKED` is not emitted at all.
    fn emit_bridge_exit(&mut self, handled: Value) -> crate::Result<()> {
        if self.block_link_active() {
            let link_block = self.builder.create_block();
            let plain_block = self.builder.create_block();
            let linked = self
                .builder
                .ins()
                .icmp_imm_u(IntCC::Equal, handled, RESUME_LINKED as i64);
            self.builder
                .ins()
                .brif(linked, link_block, &[], plain_block, &[]);

            self.builder.switch_to_block(link_block);
            self.builder.seal_block(link_block);
            let ctx_ptr = self.builder.use_var(self.state.context_ptr);
            let pc = self.builder.ins().load(
                types::I64,
                MemFlagsData::trusted(),
                ctx_ptr,
                <crate::arch::x86_64::X86_64 as crate::core::GuestArch>::pc_offset(),
            );
            self.emit_exit(Some(pc));

            self.builder.switch_to_block(plain_block);
            self.builder.seal_block(plain_block);
        }
        self.emit_host_return();
        Ok(())
    }

    /// Materialise the address to `call_indirect` through.
    fn bridge_target_ptr(&mut self, target: &BridgeTarget) -> Value {
        match *target {
            BridgeTarget::Direct(ptr) => self.builder.ins().iconst(types::I64, ptr as i64),
            BridgeTarget::Slot(slot) => {
                let slot_addr = self.builder.ins().iconst(types::I64, slot as i64);
                // `trusted()` is notrap+aligned but deliberately *not* readonly:
                // the slot is live-updatable, so this load must not be hoisted
                // out of a loop or CSE'd across the call it feeds. See
                // `BridgeTarget::Slot` for the ordering argument.
                self.builder
                    .ins()
                    .load(types::I64, MemFlagsData::trusted(), slot_addr, 0)
            }
        }
    }
}

/// `RSP`'s index in [`crate::arch::x86_64::regs::GPR_ORDER`].
const RSP_GPR_INDEX: u8 = 4;

#[cfg(test)]
mod tests {
    use crate::arch::x86_64::regs::{GPR_ORDER, gpr_index};
    use iced_x86::Register;

    #[test]
    fn rsp_index_matches_gpr_order() {
        assert_eq!(GPR_ORDER[super::RSP_GPR_INDEX as usize], Register::RSP);
        assert_eq!(gpr_index(Register::RSP), Some(super::RSP_GPR_INDEX));
        // Sub-registers fold into their containing GPR.
        assert_eq!(gpr_index(Register::ESP), Some(super::RSP_GPR_INDEX));
        assert_eq!(gpr_index(Register::AH), gpr_index(Register::RAX));
        assert_eq!(gpr_index(Register::XMM0), None);
    }
}
