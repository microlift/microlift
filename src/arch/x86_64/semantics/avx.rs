//! VEX-128 dispatch: each instruction runs through its legacy SSE handler ([`crate::arch::x86_64::vex`]).

use crate::arch::x86_64::regs::XMM_SCRATCH;
use crate::arch::x86_64::semantics::helpers::{XMM_SCRATCH_REGISTER, xmm_register_index};
use crate::arch::x86_64::vex::{VexLowering, legacy_form};
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::{Instruction, OpKind};

const GUEST_XMM_COUNT: u8 = 16;

impl<'a> Lifter<'a> {
    pub(super) fn handle_vex(
        &mut self,
        instruction: &Instruction,
        lowering: VexLowering,
    ) -> crate::Result<()> {
        match lowering {
            VexLowering::ZeroUpper => Ok(()),
            VexLowering::ZeroAll => {
                self.zero_all_xmm();
                Ok(())
            }
            VexLowering::Same(_) | VexLowering::Merge(_) => {
                let mut legacy =
                    legacy_form(instruction, lowering).expect("a lowering with a legacy code");
                if matches!(lowering, VexLowering::Merge(_)) {
                    self.merge_first_source(instruction, &mut legacy)?;
                }
                self.handle_instruction(&legacy)
            }
        }
    }

    /// `dst = src1`, parking any later source that `dst` names so the copy cannot clobber it.
    fn merge_first_source(
        &mut self,
        vex: &Instruction,
        legacy: &mut Instruction,
    ) -> crate::Result<()> {
        let dst = self.xmm_operand_index(vex, 0, "vex-dst")?;
        let src1 = self.xmm_operand_index(vex, 1, "vex-src1")?;
        if dst == src1 {
            return Ok(());
        }
        for op in 1..legacy.op_count() {
            let names_dst = legacy.op_kind(op) == OpKind::Register
                && xmm_register_index(legacy.op_register(op)) == Some(dst);
            if names_dst {
                self.park_in_scratch(dst);
                legacy.set_op_register(op, XMM_SCRATCH_REGISTER);
            }
        }
        self.copy_xmm(src1, dst);
        Ok(())
    }

    fn copy_xmm(&mut self, from: u8, to: u8) {
        if self.state.use_vector_xmm {
            let v = self.get_xmm_vec(from);
            self.set_xmm_vec(to, v);
        } else {
            let lo = self.get_xmm_lo(from);
            let hi = self.get_xmm_hi(from);
            self.set_xmm_lo(to, lo);
            self.set_xmm_hi(to, hi);
        }
    }

    /// Bypasses `set_xmm_*`: the scratch slot is not a guest register and must not be marked written.
    fn park_in_scratch(&mut self, from: u8) {
        if self.state.use_vector_xmm {
            let v = self.get_xmm_vec(from);
            self.builder.def_var(self.state.xmms_vec[&XMM_SCRATCH], v);
        } else {
            let lo = self.get_xmm_lo(from);
            let hi = self.get_xmm_hi(from);
            self.builder.def_var(self.state.xmms_lo[&XMM_SCRATCH], lo);
            self.builder.def_var(self.state.xmms_hi[&XMM_SCRATCH], hi);
        }
    }

    fn zero_all_xmm(&mut self) {
        let zero = self.builder.ins().iconst(types::I64, 0);
        for idx in 0..GUEST_XMM_COUNT {
            if self.state.use_vector_xmm {
                let v = self.builder.ins().splat(types::I64X2, zero);
                self.set_xmm_vec(idx, v);
            } else {
                self.set_xmm_lo(idx, zero);
                self.set_xmm_hi(idx, zero);
            }
        }
    }
}
