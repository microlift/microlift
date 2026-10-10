//! Emission for runtime-supplied bridges (see [`crate::core::bridge`]).
//!
//! `extern "C" fn(ctx: *mut RegisterContext, extra: u64) -> u32`; `extra` is the callee VA.
//! 0 falls through inline, non-zero exits to the host;
//! [`RESUME_LINKED`](crate::core::bridge::RESUME_LINKED) additionally makes the exit linkable.

use crate::arch::x86_64::Lifter;
use crate::arch::x86_64::X86_64;
use crate::context::ExitReason;
use crate::core::GuestArch;
use crate::core::bridge::{BridgeImport, BridgeTarget, RESUME_LINKED, RegMask};
use cranelift::prelude::*;

impl<'a> Lifter<'a> {
    /// Emit the bridge call and its two-way branch, leaving the builder in the unterminated *handled* continuation.
    /// - `callee` is stored to `imported_function_va` and passed as `extra`.
    /// - `pc` is the PC published before the call; `None` means the next instruction.
    /// - `reason` seeds `exit_reason` for a bridge that exits without writing one.
    pub(crate) fn emit_bridge_call(
        &mut self,
        bridge: &BridgeImport,
        callee: Value,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let effects = bridge.effects;
        // RSP is always published; RIP goes with it via `flush_rip_and_reason`.
        let reads = effects.reads.union(RegMask::gpr(RSP_GPR_INDEX));

        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let va_offset = <X86_64 as GuestArch>::imported_function_va_offset();
        self.ctx_store(callee, ctx_ptr, va_offset);

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
        self.builder
            .ins()
            .brif(handled, exit_block, &[], continue_block, &[]);

        // Undeclared registers are live in SSA and authoritative; declared clobbers are already in the context.
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.sync_regs_masked(effects.writes.complement())?;
        self.reload_pinned()?;
        self.emit_bridge_exit(handled)?;

        self.builder.switch_to_block(continue_block);
        self.builder.seal_block(continue_block);
        self.load_regs_masked(effects.writes)?;
        self.block_terminated = false;
        Ok(())
    }

    /// Terminate the exiting path, linking when it asked to.
    /// The PC is reloaded from the context: a `RESUME_LINKED` bridge moves it.
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
            let pc = self.ctx_load(
                types::I64,
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
                // Not readonly: the slot is live-updatable (see `BridgeTarget::Slot`); own alias region.
                let flags = self.host_slot_flags;
                self.builder.ins().load(types::I64, flags, slot_addr, 0)
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
        assert_eq!(gpr_index(Register::ESP), Some(super::RSP_GPR_INDEX));
        assert_eq!(gpr_index(Register::AH), gpr_index(Register::RAX));
        assert_eq!(gpr_index(Register::XMM0), None);
    }
}
