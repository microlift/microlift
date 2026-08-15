//! x86-64 context state synchronisation: loading guest state from the
//! `RegisterContext` on entry and writing it back on VM exits.
//!
//! These are inherent methods on the concrete `Lifter<'a>` (=
//! `LiftEngine<'a, X86_64>`); the `GuestArch` trait impl delegates to them.

use crate::arch::x86_64::Lifter;
use crate::arch::x86_64::regs::{GPR_ORDER, RegisterState, X86Flag};
use crate::core::bridge::RegMask;
use crate::core::exit::ExitReason;
use crate::core::pin::PinnedReg;
use cranelift::prelude::*;
use iced_x86::Register;

impl<'a> Lifter<'a> {
    pub(super) fn get_regcontext_flags() -> MemFlagsData {
        MemFlagsData::trusted().with_can_move()
    }

    /// Store exactly the registers in `mask` into the `RegisterContext`, and
    /// nothing else - no PC, no `exit_reason`, no `exit_code`.
    ///
    /// This is the write half of a [`BridgeEffects`](crate::core::bridge::BridgeEffects)
    /// mask. Read its doc comment before using this anywhere new: it is sound
    /// *because* the mask is declared by the callee and applied at a single call
    /// site, not inferred from emission order the way the removed-in-spirit
    /// [`Self::sync_to_context_selective`] does it. Do not try to derive a mask
    /// from `state.dirty_*`.
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
            self.builder
                .ins()
                .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
        }

        if mask.flags {
            for (&flag, &var) in &self.state.flags {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.builder
                    .ins()
                    .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
            }
        }

        for i in 0u8..16 {
            if !mask.has_vec(i) {
                continue;
            }
            if self.state.use_vector_xmm {
                let val = self.builder.use_var(self.state.xmms_vec[&i]);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    val,
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(i),
                );
            } else {
                let lo_val = self.builder.use_var(self.state.xmms_lo[&i]);
                let hi_val = self.builder.use_var(self.state.xmms_hi[&i]);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, false),
                );
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, true),
                );
            }
        }

        if mask.segment_bases {
            let fs_base_val = self.builder.use_var(self.state.fs_base);
            self.builder.ins().store(
                Self::get_regcontext_flags(),
                fs_base_val,
                ctx_ptr,
                RegisterState::get_fs_base_offset(),
            );
            let gs_base_val = self.builder.use_var(self.state.gs_base);
            self.builder.ins().store(
                Self::get_regcontext_flags(),
                gs_base_val,
                ctx_ptr,
                RegisterState::get_gs_base_offset(),
            );
        }

        Ok(())
    }

    /// Reload exactly the registers in `mask` from the `RegisterContext` back
    /// into their SSA variables. The read half of a
    /// [`BridgeEffects`](crate::core::bridge::BridgeEffects) mask; registers
    /// outside `mask` keep the SSA values they had before the call.
    ///
    /// `image_base` is deliberately not reloaded: it is fixed for the lifetime
    /// of a region and a bridge has no business changing it.
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
            let val =
                self.builder
                    .ins()
                    .load(types::I64, Self::get_regcontext_flags(), ctx_ptr, offset);
            let var = self.state.gprs[&reg];
            self.builder.def_var(var, val);
        }

        if mask.flags {
            let flags: Vec<(X86Flag, Variable)> =
                self.state.flags.iter().map(|(&f, &v)| (f, v)).collect();
            for (flag, var) in flags {
                let offset = RegisterState::get_flag_offset(flag)?;
                let val = self.builder.ins().load(
                    types::I8,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    offset,
                );
                self.builder.def_var(var, val);
            }
        }

        for i in 0u8..16 {
            if !mask.has_vec(i) {
                continue;
            }
            if self.state.use_vector_xmm {
                let val = self.builder.ins().load(
                    types::I64X2,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(i),
                );
                self.builder.def_var(self.state.xmms_vec[&i], val);
            } else {
                let lo_val = self.builder.ins().load(
                    types::I64,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, false),
                );
                let hi_val = self.builder.ins().load(
                    types::I64,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, true),
                );
                self.builder.def_var(self.state.xmms_lo[&i], lo_val);
                self.builder.def_var(self.state.xmms_hi[&i], hi_val);
            }
        }

        if mask.segment_bases {
            let fs_base_val = self.builder.ins().load(
                types::I64,
                Self::get_regcontext_flags(),
                ctx_ptr,
                RegisterState::get_fs_base_offset(),
            );
            self.builder.def_var(self.state.fs_base, fs_base_val);
            let gs_base_val = self.builder.ins().load(
                types::I64,
                Self::get_regcontext_flags(),
                ctx_ptr,
                RegisterState::get_gs_base_offset(),
            );
            self.builder.def_var(self.state.gs_base, gs_base_val);
        }

        Ok(())
    }

    /// Whether `reg` is carried out of the region in a machine register rather
    /// than through the `RegisterContext`, and so must be skipped by the full
    /// sync and the entry load.
    ///
    /// Not by the masked ones: those serve a call the host makes while this
    /// region is still running, and the host reads the context then. See
    /// [`crate::core::pin`].
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
    /// Except the pinned ones ([`crate::core::pin`]): those leave the region
    /// in registers, and the entry thunk stores them once the chain ends. A
    /// store here would be the very cost pinning exists to delete, and would
    /// also be stale by the time anything read it.
    pub fn sync_to_context(&mut self, pc: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        let gprs: Vec<(Register, Variable)> = self
            .state
            .gprs
            .iter()
            .map(|(&r, &v)| (r, v))
            .filter(|&(r, _)| !self.pinned_gpr(r))
            .collect();
        for (reg, var) in gprs {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.builder
                .ins()
                .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
        }

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag)?;
            self.builder
                .ins()
                .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
        }

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                if self.pinned_xmm(i) {
                    continue;
                }
                let var = *self.state.xmms_vec.get(&i).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    val,
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(i),
                );
            }
        } else {
            for i in 0u8..16 {
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                let lo_val = self.builder.use_var(lo_var);
                let hi_val = self.builder.use_var(hi_var);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, false),
                );
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, true),
                );
            }
        }

        let fs_base_val = self.builder.use_var(self.state.fs_base);
        self.builder.ins().store(
            Self::get_regcontext_flags(),
            fs_base_val,
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        let gs_base_val = self.builder.use_var(self.state.gs_base);
        self.builder.ins().store(
            Self::get_regcontext_flags(),
            gs_base_val,
            ctx_ptr,
            RegisterState::get_gs_base_offset(),
        );

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    /// Selective sync - only writes registers/flags/xmms marked dirty since the
    /// last `load_state_from_context`.
    ///
    /// DANGEROUS - DO NOT USE. Retained for reference only; all VM-exit paths
    /// now use the full [`Self::sync_to_context`] instead.
    ///
    /// The dirty set (`state.dirty_gprs` etc.) is a single, global, cumulative
    /// set that is only ever cleared by `load_state_from_context` (once at
    /// region entry). It is populated as a side effect of instruction emission,
    /// which happens in block-address order. So at the point a selective sync is
    /// *emitted*, the dirty set reflects the registers written by instructions
    /// emitted *before it in address order* - NOT the registers actually written
    /// along the runtime control-flow paths that reach this exit.
    ///
    /// That is unsound for any non-linear CFG. Concretely: a register written in
    /// a higher-addressed block (emitted later) that reaches a lower-addressed
    /// VM-exit via a backward edge (i.e. a loop) is missing from the dirty set
    /// when the exit is emitted. Its store is omitted, the update never reaches
    /// the `RegisterContext`, and on resume `load_state_from_context` reloads the
    /// stale pre-loop value - silently corrupting a live register across the
    /// exit. (Observed as an out-of-bounds `std::string::substr` in a guest
    /// environ-parsing loop: a loop-carried index reverted to a stale value
    /// across a slow import-call exit.)
    ///
    /// A correct selective sync would need real CFG dataflow (reaching-defs /
    /// liveness over the block graph), not an emission-order side effect. Until
    /// that exists, always use the full sync.
    #[allow(dead_code)]
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

        let dirty_gprs: Vec<Register> = self.state.dirty_gprs.iter().cloned().collect();
        let dirty_flags: Vec<X86Flag> = self.state.dirty_flags.iter().cloned().collect();

        for reg in dirty_gprs {
            if let Some(&var) = self.state.gprs.get(&reg) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_gpr_offset(reg)?;
                self.builder
                    .ins()
                    .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
            }
        }

        for flag in dirty_flags {
            if let Some(&var) = self.state.flags.get(&flag) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.builder
                    .ins()
                    .store(Self::get_regcontext_flags(), val, ctx_ptr, offset);
            }
        }

        let dirty_xmms: Vec<u8> = self.state.dirty_xmms.iter().cloned().collect();
        if self.state.use_vector_xmm {
            for idx in dirty_xmms {
                let var = *self.state.xmms_vec.get(&idx).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    val,
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(idx),
                );
            }
        } else {
            for idx in dirty_xmms {
                let lo_var = *self.state.xmms_lo.get(&idx).unwrap();
                let hi_var = *self.state.xmms_hi.get(&idx).unwrap();
                let lo_val = self.builder.use_var(lo_var);
                let hi_val = self.builder.use_var(hi_var);
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, false),
                );
                self.builder.ins().store(
                    Self::get_regcontext_flags(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, true),
                );
            }
        }

        if self.state.dirty_fs_base {
            let fs_base_val = self.builder.use_var(self.state.fs_base);
            self.builder.ins().store(
                Self::get_regcontext_flags(),
                fs_base_val,
                ctx_ptr,
                RegisterState::get_fs_base_offset(),
            );
        }
        if self.state.dirty_gs_base {
            let gs_base_val = self.builder.use_var(self.state.gs_base);
            self.builder.ins().store(
                Self::get_regcontext_flags(),
                gs_base_val,
                ctx_ptr,
                RegisterState::get_gs_base_offset(),
            );
        }

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
        let gprs: Vec<(Register, Variable)> = self
            .state
            .gprs
            .iter()
            .map(|(&r, &v)| (r, v))
            .filter(|&(r, _)| !self.pinned_gpr(r))
            .collect();
        for (reg, var) in gprs {
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val =
                self.builder
                    .ins()
                    .load(types::I64, Self::get_regcontext_flags(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        for (&flag, &var) in &self.state.flags {
            let offset = RegisterState::get_flag_offset(flag)?;
            let val =
                self.builder
                    .ins()
                    .load(types::I8, Self::get_regcontext_flags(), ctx_ptr, offset);
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
                let val = self.builder.ins().load(
                    types::I64X2,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    offset,
                );
                let var = *self.state.xmms_vec.get(&i).unwrap();
                self.builder.def_var(var, val);
            }
        } else {
            for i in 0u8..16 {
                let lo_offset = RegisterState::get_xmm_offset(i, false);
                let hi_offset = RegisterState::get_xmm_offset(i, true);
                let lo_val = self.builder.ins().load(
                    types::I64,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    lo_offset,
                );
                let hi_val = self.builder.ins().load(
                    types::I64,
                    Self::get_regcontext_flags(),
                    ctx_ptr,
                    hi_offset,
                );
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                self.builder.def_var(lo_var, lo_val);
                self.builder.def_var(hi_var, hi_val);
            }
        }

        let fs_base_val = self.builder.ins().load(
            types::I64,
            Self::get_regcontext_flags(),
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        self.builder.def_var(self.state.fs_base, fs_base_val);

        let gs_base_val = self.builder.ins().load(
            types::I64,
            Self::get_regcontext_flags(),
            ctx_ptr,
            RegisterState::get_gs_base_offset(),
        );
        self.builder.def_var(self.state.gs_base, gs_base_val);

        // After a full reload everything is in sync; clear dirty tracking.
        self.state.dirty_gprs.clear();
        self.state.dirty_flags.clear();
        self.state.dirty_xmms.clear();
        self.state.dirty_fs_base = false;
        self.state.dirty_gs_base = false;

        Ok(())
    }
}
