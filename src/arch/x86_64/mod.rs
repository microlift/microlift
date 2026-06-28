//! x86-64 guest frontend: register file, context struct, decoder/leader
//! classification, state sync, and instruction semantics. Wires them together
//! through the [`GuestArch`] impl for [`X86_64`].

pub mod context;
pub mod decode;
pub mod regs;
pub mod semantics;
pub mod sync;

use crate::core::exit::ExitReason;
use crate::core::{GuestArch, LifterConfig, LiftEngine};
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::Instruction;

use context::RegisterContext;
use regs::RegisterState;

/// x86-64 guest architecture marker.
pub struct X86_64;

/// The historical `Lifter` type, now the x86-64 monomorphization of the generic
/// engine. Instruction semantics are inherent methods on this type.
pub type Lifter<'a> = LiftEngine<'a, X86_64>;

impl GuestArch for X86_64 {
    type Instr = Instruction;
    type RegState = RegisterState;

    fn pc_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, rip) as i32
    }

    fn exit_reason_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, exit_reason) as i32
    }

    fn exit_code_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, exit_code) as i32
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

    fn resolved_indirect_targets(
        bytes: &[u8],
        base: u64,
        segments: &[crate::context::MemorySegment],
    ) -> crate::cfg::ResolvedTargets {
        crate::vsa::resolved_indirect_targets(bytes, base, segments)
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
