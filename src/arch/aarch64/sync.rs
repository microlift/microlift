//! aarch64 context state synchronisation.
//!
//! Inherent methods on the concrete `Lifter64` (= `LiftEngine<'a, Aarch64>`);
//! the `GuestArch` trait impl delegates to them. Integer state only for now
//! (V registers are synced once FP/SIMD semantics land).

use crate::arch::aarch64::Lifter64;
use crate::arch::aarch64::regs::{Aarch64RegState, FLAG_ORDER, GPR_ORDER};
use crate::core::exit::ExitReason;
use crate::core::regsync::RegSlot;
use cranelift::prelude::*;

/// `SP`'s [`RegSlot`], in this frontend's `Extra` numbering.
pub const SP_SLOT: RegSlot = RegSlot::Extra(0);
/// `TPIDR_EL0`'s [`RegSlot`].
pub const TPIDR_SLOT: RegSlot = RegSlot::Extra(1);

impl<'a> Lifter64<'a> {
    /// Full sync - writes all GPRs, SP, flags and TPIDR back to the context.
    pub fn sync_to_context(&mut self, pc: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for &reg in &GPR_ORDER {
            let var = self.state.gprs[&reg];
            let val = self.builder.use_var(var);
            let offset = Aarch64RegState::get_gpr_offset(reg);
            self.ctx_store_reg(RegSlot::Gpr(reg as u8), val, ctx_ptr, offset);
        }

        let sp_val = self.builder.use_var(self.state.sp);
        let sp_off = Aarch64RegState::get_sp_offset();
        self.ctx_store_reg(SP_SLOT, sp_val, ctx_ptr, sp_off);

        for (i, &flag) in FLAG_ORDER.iter().enumerate() {
            let var = self.state.flags[&flag];
            let val = self.builder.use_var(var);
            let offset = Aarch64RegState::get_flag_offset(flag);
            self.ctx_store_reg(RegSlot::Flag(i as u8), val, ctx_ptr, offset);
        }

        let tpidr_val = self.builder.use_var(self.state.tpidr_el0);
        let tpidr_off = Aarch64RegState::get_tpidr_offset();
        self.ctx_store_reg(TPIDR_SLOT, tpidr_val, ctx_ptr, tpidr_off);

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for &reg in &GPR_ORDER {
            let var = self.state.gprs[&reg];
            let offset = Aarch64RegState::get_gpr_offset(reg);
            let val = self.ctx_load_reg(types::I64, ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let sp_val = self.ctx_load_reg(types::I64, ctx_ptr, Aarch64RegState::get_sp_offset());
        self.builder.def_var(self.state.sp, sp_val);

        for &flag in &FLAG_ORDER {
            let var = self.state.flags[&flag];
            let offset = Aarch64RegState::get_flag_offset(flag);
            let val = self.ctx_load_reg(types::I8, ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let tpidr_val = self.ctx_load_reg(types::I64, ctx_ptr, Aarch64RegState::get_tpidr_offset());
        self.builder.def_var(self.state.tpidr_el0, tpidr_val);

        let image_base_val = self.image_base_at_entry(ctx_ptr);
        self.builder.def_var(self.state.image_base, image_base_val);

        Ok(())
    }
}
