use crate::context::ExitReason;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// Hand an unlifted instruction to the host interpreter inline; the PC points *at* it.
    /// Interprets that one instruction only.
    pub(super) fn handle_unimplemented_instruction(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        let rip_after = self.rip_after(instruction);
        self.emit_interpreter_callback(Some(rip_at), Some(rip_after), 1)
    }

    /// `UD2`: `Ud2` VM exit with RIP at the `UD2`.
    pub(super) fn handle_ud2(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        self.sync_to_context(Some(rip_at), ExitReason::Ud2)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// `INT3`: `Breakpoint` VM exit with RIP at the `INT3`.
    pub(super) fn handle_int3(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        self.sync_to_context(Some(rip_at), ExitReason::Breakpoint)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Run-time address of `instruction`, per [`AddressMode`](crate::core::AddressMode).
    fn rip_at(&mut self, instruction: &Instruction) -> Value {
        self.runtime_addr(instruction.ip())
    }

    /// Run-time address of the instruction after `instruction`.
    fn rip_after(&mut self, instruction: &Instruction) -> Value {
        self.runtime_addr(instruction.next_ip())
    }

    /// `Interrupt` VM exit for `vector`, RIP at the next instruction.
    fn emit_interrupt_exit(&mut self, instruction: &Instruction, vector: u32) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Interrupt)?;
        self.resumes_at(instruction.next_ip());
        self.write_exit_code(vector);
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// `INT imm8`: vector in `exit_code`, RIP past the `INT`.
    pub(super) fn handle_int(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let vector = instruction.immediate8() as u32;
        self.emit_interrupt_exit(instruction, vector)
    }

    /// `INT1`/`ICEBP`: raises vector 1.
    /// `INTO` is invalid in 64-bit mode; the decoder rejects it.
    pub(super) fn handle_int1(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.emit_interrupt_exit(instruction, 1)
    }

    pub(super) fn handle_cpuid(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Cpuid)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    pub(super) fn handle_xgetbv(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Xgetbv)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// `RDTSC`: resumable VM exit; the host fills `rdx:rax`, PC is the next instruction.
    pub(super) fn handle_rdtsc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Rdtsc)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }
}
