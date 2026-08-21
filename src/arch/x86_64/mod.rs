//! x86-64 guest frontend: register file, context struct, decoder/leader
//! classification, state sync, and instruction semantics. Wires them together
//! through the [`GuestArch`] impl for [`X86_64`].

pub mod abi;
pub mod context;
pub mod decode;
pub mod flag_effects;
pub mod jumptable;
pub mod pin;
pub mod regs;
pub mod semantics;
pub mod sync;

use crate::core::exit::ExitReason;
use crate::core::{GuestArch, LiftEngine, LifterConfig};
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::Instruction;

use context::RegisterContext;
use regs::{
    FLAG_NAMES, FLAG_ORDER, GPR_NAMES, GPR_ORDER, RegisterState, SEGMENT_NAMES, SEGMENT_ORDER,
    XMM_HI_NAMES, XMM_NAMES,
};

/// x86-64 guest architecture marker.
pub struct X86_64;

/// The historical `Lifter` type, now the x86-64 monomorphization of the generic
/// engine. Instruction semantics are inherent methods on this type.
pub type Lifter<'a> = LiftEngine<'a, X86_64>;

impl GuestArch for X86_64 {
    /// The architectural limit: no legal x86-64 encoding exceeds fifteen bytes,
    /// however many prefixes it carries, and a decoder rejects anything longer.
    const MAX_INSTRUCTION_LEN: u64 = 15;

    type Instr = Instruction;
    type RegState = RegisterState;

    fn image_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, control.image_base) as i32
    }

    fn pc_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, rip) as i32
    }

    fn exit_reason_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, control.exit_reason) as i32
    }

    fn exit_code_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, control.exit_code) as i32
    }

    fn imported_function_va_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, control.imported_function_va) as i32
    }

    fn block_link_offsets() -> Option<crate::core::link::BlockLinkOffsets> {
        Some(crate::core::link::BlockLinkOffsets {
            table: std::mem::offset_of!(RegisterContext, link_table) as i32,
            hit: std::mem::offset_of!(RegisterContext, link_hit) as i32,
        })
    }

    fn call_return_offsets() -> Option<crate::core::callret::CallRetOffsets> {
        Some(crate::core::callret::CallRetOffsets {
            base: std::mem::offset_of!(RegisterContext, callret_base) as i32,
            offset: std::mem::offset_of!(RegisterContext, callret_off) as i32,
        })
    }

    /// x86-64 pins a GPR as an `I64` and an XMM register as an `I64X2`.
    ///
    /// The XMM answer is `None` without
    /// [`LifterConfig::use_vector_xmm`](crate::core::LifterConfig::use_vector_xmm),
    /// because the scalar representation splits that register into two variables
    /// and a slot carries one value. An index past either bank is `None`.
    fn pinned_slot_type(
        reg: crate::core::pin::PinnedReg,
        config: &LifterConfig,
    ) -> Option<cranelift::prelude::Type> {
        use crate::core::pin::RegClass;
        match reg.class {
            RegClass::Gpr => (reg.index < 16).then_some(types::I64),
            RegClass::Vector => (config.use_vector_xmm && reg.index < 16).then_some(types::I64X2),
        }
    }

    fn pinned_slot_var(
        state: &Self::RegState,
        reg: crate::core::pin::PinnedReg,
    ) -> Option<Variable> {
        use crate::core::pin::RegClass;
        match reg.class {
            RegClass::Gpr => regs::GPR_ORDER
                .get(reg.index as usize)
                .and_then(|r| state.gprs.get(r))
                .copied(),
            RegClass::Vector => state.xmms_vec.get(&(reg.index as u8)).copied(),
        }
    }

    fn pinned_slot_offset(reg: crate::core::pin::PinnedReg) -> Option<i32> {
        use crate::core::pin::RegClass;
        match reg.class {
            RegClass::Gpr => regs::GPR_ORDER
                .get(reg.index as usize)
                .and_then(|&r| RegisterState::get_gpr_offset(r).ok()),
            RegClass::Vector => {
                (reg.index < 16).then(|| RegisterState::get_xmm_vec_offset(reg.index as u8))
            }
        }
    }

    fn context_ptr_var(state: &Self::RegState) -> Variable {
        state.context_ptr
    }

    fn image_base_var(state: &Self::RegState) -> Variable {
        state.image_base
    }

    fn new_state(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self::RegState {
        RegisterState::from_config(builder, config)
    }

    /// Every constant-offset `RegisterContext` field, one alias region each.
    ///
    /// The `fpr` array is deliberately absent: `ST(i)` is `TOP`-relative, so the
    /// same bytes are reached through a computed pointer and through the
    /// context at a constant offset (`fpu_expand_tag_word`), and both must share
    /// the one whole-array region. The segment selectors are absent because they
    /// are read and written in place with no SSA variable, and `_align_xmm` is
    /// padding nothing touches.
    fn context_fields(config: &LifterConfig) -> Vec<(i32, &'static str)> {
        let mut fields: Vec<(i32, &'static str)> = Vec::with_capacity(64);

        for (&reg, name) in GPR_ORDER.iter().zip(GPR_NAMES) {
            fields.push((
                RegisterState::get_gpr_offset(reg).expect("GPR_ORDER holds only GPRs"),
                name,
            ));
        }
        for (&flag, name) in FLAG_ORDER.iter().zip(FLAG_NAMES) {
            fields.push((
                RegisterState::get_flag_offset(flag).expect("FLAG_ORDER holds only flags"),
                name,
            ));
        }
        // One 16-byte field per XMM in vector mode, two 8-byte halves in scalar
        // mode. Reporting both shapes would claim two regions for one address.
        for i in 0u8..16 {
            if config.use_vector_xmm {
                fields.push((RegisterState::get_xmm_vec_offset(i), XMM_NAMES[i as usize]));
            } else {
                fields.push((
                    RegisterState::get_xmm_offset(i, false),
                    XMM_NAMES[i as usize],
                ));
                fields.push((
                    RegisterState::get_xmm_offset(i, true),
                    XMM_HI_NAMES[i as usize],
                ));
            }
        }
        fields.push((RegisterState::get_fs_base_offset(), "fs_base"));
        fields.push((RegisterState::get_gs_base_offset(), "gs_base"));
        // The six selectors have no SSA variable - they are read and written in
        // place - but they are still context fields and still need a region.
        for (&reg, &name) in SEGMENT_ORDER.iter().zip(SEGMENT_NAMES.iter()) {
            fields.push((
                RegisterState::get_segment_offset(reg).expect("SEGMENT_ORDER holds only selectors"),
                name,
            ));
        }

        // Non-register context state: the control header, the host hooks, and
        // the x87/SSE control words.
        fields.push((Self::pc_offset(), "rip"));
        fields.push((Self::image_base_offset(), "image_base"));
        fields.push((Self::exit_reason_offset(), "exit_reason"));
        fields.push((Self::exit_code_offset(), "exit_code"));
        fields.push((Self::imported_function_va_offset(), "imported_function_va"));
        fields.push((
            std::mem::offset_of!(RegisterContext, syscall_handler) as i32,
            "syscall_handler",
        ));
        fields.push((std::mem::offset_of!(RegisterContext, mxcsr) as i32, "mxcsr"));
        fields.push((
            std::mem::offset_of!(RegisterContext, fpu_top) as i32,
            "fpu_top",
        ));
        fields.push((
            std::mem::offset_of!(RegisterContext, fpu_sw) as i32,
            "fpu_sw",
        ));
        fields.push((
            std::mem::offset_of!(RegisterContext, fpu_cw) as i32,
            "fpu_cw",
        ));
        fields.push((
            std::mem::offset_of!(RegisterContext, fpu_tag) as i32,
            "fpu_tag",
        ));
        if let Some(offsets) = Self::block_link_offsets() {
            fields.push((offsets.table, "link_table"));
            fields.push((offsets.hit, "link_hit"));
        }
        if let Some(offsets) = Self::call_return_offsets() {
            fields.push((offsets.base, "callret_base"));
            fields.push((offsets.offset, "callret_off"));
        }

        fields
    }

    fn decode_region(bytes: &[u8], ip: u64) -> Vec<Self::Instr> {
        decode::decode_region(bytes, ip)
    }

    fn decode_one(bytes: &[u8], ip: u64) -> Option<Self::Instr> {
        decode::decode_one(bytes, ip)
    }

    fn instr_ip(i: &Self::Instr) -> u64 {
        decode::instr_ip(i)
    }

    fn instr_next_ip(i: &Self::Instr) -> u64 {
        decode::instr_next_ip(i)
    }

    fn is_invalid(i: &Self::Instr) -> bool {
        decode::is_invalid(i)
    }

    fn control_flow(i: &Self::Instr) -> crate::arch::ControlFlow {
        decode::control_flow(i)
    }

    fn instruction_effects(i: &Self::Instr) -> Option<crate::arch::InstructionEffects> {
        Some(decode::instruction_effects(i))
    }

    fn flag_effects(i: &Self::Instr) -> Option<crate::core::flags::FlagEffects> {
        flag_effects::flag_effects(i)
    }

    fn resolved_indirect_targets(
        bytes: &[u8],
        base: u64,
        segments: &[crate::context::MemorySegment],
    ) -> crate::cfg::ResolvedTargets {
        crate::vsa::resolved_indirect_targets(bytes, base, segments)
    }

    fn jump_table_targets(
        bytes: &[u8],
        base: u64,
        segments: &[crate::context::MemorySegment],
    ) -> std::collections::HashMap<u64, crate::core::JumpTableInfo> {
        jumptable::analyze_jump_tables(bytes, base, segments)
            .into_iter()
            .map(|(ip, table)| {
                let extent = table.extent();
                (
                    ip,
                    crate::core::JumpTableInfo {
                        targets: table.targets,
                        extent,
                    },
                )
            })
            .collect()
    }

    fn lift_instruction(lifter: &mut LiftEngine<'_, Self>, i: &Self::Instr) -> crate::Result<()> {
        lifter.handle_instruction(i)
    }

    fn load_state_from_context(lifter: &mut LiftEngine<'_, Self>) -> crate::Result<()> {
        lifter.load_state_from_context()
    }

    fn sync_to_context(
        lifter: &mut LiftEngine<'_, Self>,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        lifter.sync_to_context(pc, reason)
    }
}
