//! aarch64 context state synchronisation.
//!
//! Inherent methods on the concrete `Lifter64` (= `LiftEngine<'a, Aarch64>`);
//! the `GuestArch` trait impl delegates to them. Integer state only for now
//! (V registers are synced once FP/SIMD semantics land).

use crate::arch::aarch64::Lifter64;
use crate::arch::aarch64::context::Aarch64RegisterContext;
use crate::arch::aarch64::regs::{Aarch64RegState, Nzcv};
use crate::core::exit::ExitReason;
use cranelift::prelude::*;

impl<'a> Lifter64<'a> {
    /// Full sync - writes all GPRs, SP, flags and TPIDR back to the context.
    pub fn sync_to_context(&mut self, pc: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let val = self.builder.use_var(var);
            let offset = Aarch64RegState::get_gpr_offset(reg);
            self.builder
                .ins()
                .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
        }

        let sp_val = self.builder.use_var(self.state.sp);
        self.builder.ins().store(
            MemFlagsData::trusted(),
            sp_val,
            ctx_ptr,
            Aarch64RegState::get_sp_offset(),
        );

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = Aarch64RegState::get_flag_offset(flag);
            self.builder
                .ins()
                .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
        }

        let tpidr_val = self.builder.use_var(self.state.tpidr_el0);
        self.builder.ins().store(
            MemFlagsData::trusted(),
            tpidr_val,
            ctx_ptr,
            Aarch64RegState::get_tpidr_offset(),
        );

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    /// Selective sync - only writes registers/flags dirtied since the last load.
    pub fn sync_to_context_selective(
        &mut self,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        let dirty_gprs: Vec<u16> = self.state.dirty_gprs.iter().cloned().collect();
        for reg in dirty_gprs {
            if let Some(&var) = self.state.gprs.get(&reg) {
                let val = self.builder.use_var(var);
                let offset = Aarch64RegState::get_gpr_offset(reg);
                self.builder
                    .ins()
                    .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
            }
        }

        if self.state.dirty_sp {
            let sp_val = self.builder.use_var(self.state.sp);
            self.builder.ins().store(
                MemFlagsData::trusted(),
                sp_val,
                ctx_ptr,
                Aarch64RegState::get_sp_offset(),
            );
        }

        let dirty_flags: Vec<Nzcv> = self.state.dirty_flags.iter().cloned().collect();
        for flag in dirty_flags {
            if let Some(&var) = self.state.flags.get(&flag) {
                let val = self.builder.use_var(var);
                let offset = Aarch64RegState::get_flag_offset(flag);
                self.builder
                    .ins()
                    .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
            }
        }

        if self.state.dirty_tpidr {
            let tpidr_val = self.builder.use_var(self.state.tpidr_el0);
            self.builder.ins().store(
                MemFlagsData::trusted(),
                tpidr_val,
                ctx_ptr,
                Aarch64RegState::get_tpidr_offset(),
            );
        }

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let offset = Aarch64RegState::get_gpr_offset(reg);
            let val = self
                .builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let sp_val = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            ctx_ptr,
            Aarch64RegState::get_sp_offset(),
        );
        self.builder.def_var(self.state.sp, sp_val);

        for (&flag, &var) in &self.state.flags {
            let offset = Aarch64RegState::get_flag_offset(flag);
            let val = self
                .builder
                .ins()
                .load(types::I8, MemFlagsData::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let tpidr_val = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            ctx_ptr,
            Aarch64RegState::get_tpidr_offset(),
        );
        self.builder.def_var(self.state.tpidr_el0, tpidr_val);

        let image_base_offset = std::mem::offset_of!(Aarch64RegisterContext, image_base) as i32;
        let image_base_val =
            self.builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), ctx_ptr, image_base_offset);
        self.builder.def_var(self.state.image_base, image_base_val);

        self.state.dirty_gprs.clear();
        self.state.dirty_sp = false;
        self.state.dirty_flags.clear();
        self.state.dirty_tpidr = false;

        Ok(())
    }
}
