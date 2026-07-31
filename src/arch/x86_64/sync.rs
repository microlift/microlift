//! x86-64 context state synchronisation: loading guest state from the
//! `RegisterContext` on entry and writing it back on VM exits.
//!
//! These are inherent methods on the concrete `Lifter<'a>` (=
//! `LiftEngine<'a, X86_64>`); the `GuestArch` trait impl delegates to them.

use crate::arch::x86_64::Lifter;
use crate::arch::x86_64::context::RegisterContext;
use crate::arch::x86_64::regs::{RegisterState, X86Flag};
use crate::core::exit::ExitReason;
use cranelift::prelude::*;
use iced_x86::Register;

impl<'a> Lifter<'a> {
    /// Full sync - writes all GPRs, flags, XMMs and segment bases back to the
    /// `RegisterContext`. Used on context reload paths and unvisited exit stubs.
    pub fn sync_to_context(&mut self, pc: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        // Analysis mode produces inspection IR with no context write-back.
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.builder
                .ins()
                .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
        }

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag)?;
            self.builder
                .ins()
                .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
        }

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                let var = *self.state.xmms_vec.get(&i).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    MemFlagsData::new().with_notrap(),
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
                    MemFlagsData::trusted(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, false),
                );
                self.builder.ins().store(
                    MemFlagsData::trusted(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, true),
                );
            }
        }

        let fs_base_val = self.builder.use_var(self.state.fs_base);
        self.builder.ins().store(
            MemFlagsData::trusted(),
            fs_base_val,
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        let gs_base_val = self.builder.use_var(self.state.gs_base);
        self.builder.ins().store(
            MemFlagsData::trusted(),
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
                    .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
            }
        }

        for flag in dirty_flags {
            if let Some(&var) = self.state.flags.get(&flag) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.builder
                    .ins()
                    .store(MemFlagsData::trusted(), val, ctx_ptr, offset);
            }
        }

        let dirty_xmms: Vec<u8> = self.state.dirty_xmms.iter().cloned().collect();
        if self.state.use_vector_xmm {
            for idx in dirty_xmms {
                let var = *self.state.xmms_vec.get(&idx).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    MemFlagsData::new().with_notrap(),
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
                    MemFlagsData::trusted(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, false),
                );
                self.builder.ins().store(
                    MemFlagsData::trusted(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, true),
                );
            }
        }

        if self.state.dirty_fs_base {
            let fs_base_val = self.builder.use_var(self.state.fs_base);
            self.builder.ins().store(
                MemFlagsData::trusted(),
                fs_base_val,
                ctx_ptr,
                RegisterState::get_fs_base_offset(),
            );
        }
        if self.state.dirty_gs_base {
            let gs_base_val = self.builder.use_var(self.state.gs_base);
            self.builder.ins().store(
                MemFlagsData::trusted(),
                gs_base_val,
                ctx_ptr,
                RegisterState::get_gs_base_offset(),
            );
        }

        self.flush_rip_and_reason(pc, reason)?;
        Ok(())
    }

    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // RIP is materialised from current_ip on sync, not kept as a variable.
        for (&reg, &var) in &self.state.gprs {
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val = self
                .builder
                .ins()
                .load(types::I64, MemFlagsData::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        for (&flag, &var) in &self.state.flags {
            let offset = RegisterState::get_flag_offset(flag)?;
            let val = self
                .builder
                .ins()
                .load(types::I8, MemFlagsData::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let image_base_offset = std::mem::offset_of!(RegisterContext, image_base) as i32;
        let image_base_val = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            ctx_ptr,
            image_base_offset,
        );
        self.builder.def_var(self.state.image_base, image_base_val);

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                let offset = RegisterState::get_xmm_vec_offset(i);
                let val = self.builder.ins().load(
                    types::I64X2,
                    MemFlagsData::new().with_notrap(),
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
                    MemFlagsData::trusted(),
                    ctx_ptr,
                    lo_offset,
                );
                let hi_val = self.builder.ins().load(
                    types::I64,
                    MemFlagsData::trusted(),
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
            MemFlagsData::trusted(),
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        self.builder.def_var(self.state.fs_base, fs_base_val);

        let gs_base_val = self.builder.ins().load(
            types::I64,
            MemFlagsData::trusted(),
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
