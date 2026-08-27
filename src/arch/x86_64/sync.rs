//! x86-64 context state synchronisation: loading guest state from the
//! `RegisterContext` on entry and writing it back on VM exits.
//!
//! These are inherent methods on the concrete `Lifter<'a>` (=
//! `LiftEngine<'a, X86_64>`); the `GuestArch` trait impl delegates to them.

use crate::arch::x86_64::Lifter;
use crate::arch::x86_64::regs::{FLAG_ORDER, GPR_ORDER, RegisterState};
use crate::core::bridge::RegMask;
use crate::core::exit::ExitReason;
use crate::core::pin::PinnedReg;
use crate::core::regsync::RegSlot;
use cranelift::prelude::*;
use iced_x86::Register;

/// `fs_base`'s [`RegSlot`], in this frontend's `Extra` numbering.
pub const FS_BASE_SLOT: RegSlot = RegSlot::Extra(0);
/// `gs_base`'s [`RegSlot`].
pub const GS_BASE_SLOT: RegSlot = RegSlot::Extra(1);

impl<'a> Lifter<'a> {
    /// Store exactly the registers in `mask` into the `RegisterContext`, and
    /// nothing else - no PC, no `exit_reason`, no `exit_code`.
    pub fn sync_regs_masked(&mut self, mask: RegMask) -> crate::Result<()> {
        if self.config.analysis_mode || mask.is_empty() {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (i, &reg) in GPR_ORDER.iter().enumerate() {
            if !mask.has_gpr(i as u8) {
                continue;
            }
            let var = self.state.gprs[&reg];
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.ctx_store(val, ctx_ptr, offset);
        }

        if mask.flags {
            for &flag in &FLAG_ORDER {
                let var = self.state.flags[&flag];
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.ctx_store(val, ctx_ptr, offset);
            }
        }

        for i in 0u8..16 {
            if !mask.has_vec(i) {
                continue;
            }
            if self.state.use_vector_xmm {
                let val = self.builder.use_var(self.state.xmms_vec[&i]);
                self.ctx_store(val, ctx_ptr, RegisterState::get_xmm_vec_offset(i));
            } else {
                let lo_val = self.builder.use_var(self.state.xmms_lo[&i]);
                let hi_val = self.builder.use_var(self.state.xmms_hi[&i]);
                self.ctx_store(lo_val, ctx_ptr, RegisterState::get_xmm_offset(i, false));
                self.ctx_store(hi_val, ctx_ptr, RegisterState::get_xmm_offset(i, true));
            }
        }

        if mask.segment_bases {
            let fs_base_val = self.builder.use_var(self.state.fs_base);
            self.ctx_store(fs_base_val, ctx_ptr, RegisterState::get_fs_base_offset());
            let gs_base_val = self.builder.use_var(self.state.gs_base);
            self.ctx_store(gs_base_val, ctx_ptr, RegisterState::get_gs_base_offset());
        }

        Ok(())
    }

    /// Reload exactly the registers in `mask` from the `RegisterContext` into their
    /// SSA variables. The read half of a
    /// [`BridgeEffects`](crate::core::bridge::BridgeEffects) mask; registers
    /// outside `mask` keep the values they had before the call.
    ///
    /// `image_base` is deliberately not reloaded: it is fixed for the lifetime of a
    /// region.
    pub fn load_regs_masked(&mut self, mask: RegMask) -> crate::Result<()> {
        if self.config.analysis_mode || mask.is_empty() {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (i, &reg) in GPR_ORDER.iter().enumerate() {
            if !mask.has_gpr(i as u8) {
                continue;
            }
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val = self.ctx_load(types::I64, ctx_ptr, offset);
            let var = self.state.gprs[&reg];
            self.builder.def_var(var, val);
            self.mark_written(RegSlot::Gpr(i as u8));
        }

        if mask.flags {
            for (i, &flag) in FLAG_ORDER.iter().enumerate() {
                let var = self.state.flags[&flag];
                let offset = RegisterState::get_flag_offset(flag)?;
                let val = self.ctx_load(types::I8, ctx_ptr, offset);
                self.builder.def_var(var, val);
                self.mark_written(RegSlot::Flag(i as u8));
            }
        }

        for i in 0u8..16 {
            if !mask.has_vec(i) {
                continue;
            }
            if self.state.use_vector_xmm {
                let val =
                    self.ctx_load(types::I64X2, ctx_ptr, RegisterState::get_xmm_vec_offset(i));
                self.builder.def_var(self.state.xmms_vec[&i], val);
            } else {
                let lo_val =
                    self.ctx_load(types::I64, ctx_ptr, RegisterState::get_xmm_offset(i, false));
                let hi_val =
                    self.ctx_load(types::I64, ctx_ptr, RegisterState::get_xmm_offset(i, true));
                self.builder.def_var(self.state.xmms_lo[&i], lo_val);
                self.builder.def_var(self.state.xmms_hi[&i], hi_val);
            }
            self.mark_written(RegSlot::Vec(i));
            self.mark_written(RegSlot::VecHi(i));
        }

        if mask.segment_bases {
            let fs_base_val =
                self.ctx_load(types::I64, ctx_ptr, RegisterState::get_fs_base_offset());
            self.builder.def_var(self.state.fs_base, fs_base_val);
            let gs_base_val =
                self.ctx_load(types::I64, ctx_ptr, RegisterState::get_gs_base_offset());
            self.builder.def_var(self.state.gs_base, gs_base_val);
            self.mark_written(FS_BASE_SLOT);
            self.mark_written(GS_BASE_SLOT);
        }

        Ok(())
    }

    /// Whether `reg` is carried out of the region in a machine register rather than
    /// through the `RegisterContext`, and so must be skipped by the full sync and
    /// the entry load.
    ///
    /// Not by the masked ones: those serve a call the host makes while this region
    /// is still running, and the host reads the context then.
    fn pinned_gpr(&self, reg: Register) -> bool {
        match crate::arch::x86_64::regs::gpr_index(reg) {
            Some(i) => self.is_pinned(PinnedReg::gpr(i as u16)),
            None => false,
        }
    }

    /// Whether XMM `idx` is pinned. Only ever true in vector mode - the scalar
    /// pair cannot be a slot, and `pinned_slot_type` refuses it there.
    fn pinned_xmm(&self, idx: u8) -> bool {
        self.is_pinned(PinnedReg::vector(idx as u16))
    }

    /// Full sync - writes all GPRs, flags, XMMs and segment bases back to the
    /// `RegisterContext`. Used on context reload paths and unvisited exit stubs.
    ///
    /// Except the pinned ones ([`crate::core::pin`]), which leave the region in
    /// registers for the entry thunk to store once the chain ends. A store here
    /// would be the very cost pinning exists to delete.
    pub fn sync_to_context(&mut self, pc: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // Every store here is recorded against the register it writes, so the
        // prune in [`crate::core::regsync`] can drop the ones whose register
        // this region never defined.
        for (i, &reg) in GPR_ORDER.iter().enumerate() {
            if self.pinned_gpr(reg) {
                continue;
            }
            let var = self.state.gprs[&reg];
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.ctx_store_reg(RegSlot::Gpr(i as u8), val, ctx_ptr, offset);
        }

        for (i, &flag) in FLAG_ORDER.iter().enumerate() {
            let var = self.state.flags[&flag];
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag)?;
            self.ctx_store_reg(RegSlot::Flag(i as u8), val, ctx_ptr, offset);
        }

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                if self.pinned_xmm(i) {
                    continue;
                }
                let var = *self.state.xmms_vec.get(&i).unwrap();
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_xmm_vec_offset(i);
                self.ctx_store_reg(RegSlot::Vec(i), val, ctx_ptr, offset);
            }
        } else {
            for i in 0u8..16 {
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                let lo_val = self.builder.use_var(lo_var);
                let hi_val = self.builder.use_var(hi_var);
                let lo_off = RegisterState::get_xmm_offset(i, false);
                let hi_off = RegisterState::get_xmm_offset(i, true);
                self.ctx_store_reg(RegSlot::Vec(i), lo_val, ctx_ptr, lo_off);
                self.ctx_store_reg(RegSlot::VecHi(i), hi_val, ctx_ptr, hi_off);
            }
        }

        let fs_base_val = self.builder.use_var(self.state.fs_base);
        let fs_off = RegisterState::get_fs_base_offset();
        self.ctx_store_reg(FS_BASE_SLOT, fs_base_val, ctx_ptr, fs_off);
        let gs_base_val = self.builder.use_var(self.state.gs_base);
        let gs_off = RegisterState::get_gs_base_offset();
        self.ctx_store_reg(GS_BASE_SLOT, gs_base_val, ctx_ptr, gs_off);

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    /// Load the region's entry state out of the `RegisterContext`.
    ///
    /// Pinned registers are skipped: they arrive as function parameters, and the
    /// engine defines their variables from those immediately after this returns.
    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // RIP is materialised from current_ip on sync, not kept as a variable.
        for &reg in &GPR_ORDER {
            if self.pinned_gpr(reg) {
                continue;
            }
            let var = self.state.gprs[&reg];
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val = self.ctx_load_reg(types::I64, ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        for &flag in &FLAG_ORDER {
            let var = self.state.flags[&flag];
            let offset = RegisterState::get_flag_offset(flag)?;
            let val = self.ctx_load_reg(types::I8, ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let image_base_val = self.image_base_at_entry(ctx_ptr);
        self.builder.def_var(self.state.image_base, image_base_val);

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                if self.pinned_xmm(i) {
                    continue;
                }
                let offset = RegisterState::get_xmm_vec_offset(i);
                let val = self.ctx_load_reg(types::I64X2, ctx_ptr, offset);
                let var = *self.state.xmms_vec.get(&i).unwrap();
                self.builder.def_var(var, val);
            }
        } else {
            for i in 0u8..16 {
                let lo_offset = RegisterState::get_xmm_offset(i, false);
                let hi_offset = RegisterState::get_xmm_offset(i, true);
                let lo_val = self.ctx_load_reg(types::I64, ctx_ptr, lo_offset);
                let hi_val = self.ctx_load_reg(types::I64, ctx_ptr, hi_offset);
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                self.builder.def_var(lo_var, lo_val);
                self.builder.def_var(hi_var, hi_val);
            }
        }

        let fs_base_val =
            self.ctx_load_reg(types::I64, ctx_ptr, RegisterState::get_fs_base_offset());
        self.builder.def_var(self.state.fs_base, fs_base_val);

        let gs_base_val =
            self.ctx_load_reg(types::I64, ctx_ptr, RegisterState::get_gs_base_offset());
        self.builder.def_var(self.state.gs_base, gs_base_val);

        Ok(())
    }
}
