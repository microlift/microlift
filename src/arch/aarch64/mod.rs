//! aarch64 guest frontend: register file, context struct, yaxpeax-arm
//! decoder/leader classification, state sync, and instruction semantics. Wired
//! together through the [`GuestArch`] impl for [`Aarch64`].

pub mod context;
pub mod decode;
pub mod regs;
pub mod semantics;
pub mod sync;

use crate::core::exit::ExitReason;
use crate::core::{GuestArch, LiftEngine, LifterConfig};
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};

use context::Aarch64RegisterContext;
use decode::A64Instr;
use regs::{Aarch64RegState, FLAG_NAMES, FLAG_ORDER, GPR_NAMES, GPR_ORDER};

/// aarch64 guest architecture marker.
pub struct Aarch64;

/// The aarch64 monomorphization of the generic engine. Instruction semantics
/// are inherent methods on this type.
pub type Lifter64<'a> = LiftEngine<'a, Aarch64>;

impl GuestArch for Aarch64 {
    /// Fixed-width: every A64 instruction is one aligned word. A region cut on
    /// any four-byte boundary therefore never splits one, which is why
    /// [`LifterConfig::region_tail_exit`] can never fire for this guest.
    const MAX_INSTRUCTION_LEN: u64 = 4;

    type Instr = A64Instr;
    type RegState = Aarch64RegState;

    fn image_base_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, control.image_base) as i32
    }

    fn pc_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, pc) as i32
    }

    fn exit_reason_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, control.exit_reason) as i32
    }

    fn exit_code_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, control.exit_code) as i32
    }

    fn imported_function_va_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, control.imported_function_va) as i32
    }

    fn context_ptr_var(state: &Self::RegState) -> Variable {
        state.context_ptr
    }

    fn image_base_var(state: &Self::RegState) -> Variable {
        state.image_base
    }

    fn new_state(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self::RegState {
        Aarch64RegState::from_config(builder, config)
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

    /// Every constant-offset `Aarch64RegisterContext` field, one alias region
    /// each - see [`crate::core::alias`]. The V registers are absent because
    /// they are not SSA variables here yet.
    fn context_fields(_config: &LifterConfig) -> Vec<(i32, &'static str)> {
        let mut fields: Vec<(i32, &'static str)> = Vec::with_capacity(40);
        for (&reg, &name) in GPR_ORDER.iter().zip(GPR_NAMES.iter()) {
            fields.push((Aarch64RegState::get_gpr_offset(reg), name));
        }
        fields.push((Aarch64RegState::get_sp_offset(), "sp"));
        for (&flag, &name) in FLAG_ORDER.iter().zip(FLAG_NAMES.iter()) {
            fields.push((Aarch64RegState::get_flag_offset(flag), name));
        }
        fields.push((Aarch64RegState::get_tpidr_offset(), "tpidr_el0"));
        fields.push((Self::pc_offset(), "pc"));
        fields.push((Self::image_base_offset(), "image_base"));
        fields.push((Self::exit_reason_offset(), "exit_reason"));
        fields.push((Self::exit_code_offset(), "exit_code"));
        fields.push((Self::imported_function_va_offset(), "imported_function_va"));
        let callbacks = Self::callback_offsets();
        fields.push((callbacks.syscall, "callbacks.handle_syscall"));
        fields.push((
            callbacks.interpreter_fallback,
            "callbacks.interpreter_fallback",
        ));
        fields
    }

    fn callback_offsets() -> crate::core::callback::CallbackOffsets {
        crate::core::callback::CallbackOffsets {
            syscall: std::mem::offset_of!(Aarch64RegisterContext, callbacks.handle_syscall) as i32,
            interpreter_fallback: std::mem::offset_of!(
                Aarch64RegisterContext,
                callbacks.interpreter_fallback
            ) as i32,
        }
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
