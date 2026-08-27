use crate::context::ExitReason;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// Hand an instruction this frontend does not lift to the host's
    /// interpreter, inline; see [`crate::core::callback`]. The PC published to
    /// it points *at* the instruction, which is where the interpreter starts.
    ///
    /// One instruction, not the run the callback could do: the region's own
    /// code for everything after it is already compiled.
    pub(super) fn handle_unimplemented_instruction(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        let rip_after = self.rip_after(instruction);
        self.emit_interpreter_callback(Some(rip_at), Some(rip_after), 1)
    }

    /// Handle a `UD2` instruction - emit a `Ud2` VM exit with RIP pointing at
    /// the `UD2` itself (matching real hardware behaviour where #UD points at
    /// the faulting instruction).
    pub(super) fn handle_ud2(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        self.sync_to_context(Some(rip_at), ExitReason::Ud2)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Handle an `INT3` breakpoint - emit a `Breakpoint` VM exit with RIP
    /// pointing at the `INT3` itself so the host can inspect or skip it.
    pub(super) fn handle_int3(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_at = self.rip_at(instruction);
        self.sync_to_context(Some(rip_at), ExitReason::Breakpoint)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Compute the run-time address of `instruction` itself as a Cranelift
    /// value, per the configured
    /// [`AddressMode`](crate::core::AddressMode).
    fn rip_at(&mut self, instruction: &Instruction) -> Value {
        self.runtime_addr(instruction.ip())
    }

    /// Compute the run-time address of the instruction *following*
    /// `instruction`. This is the resume point a software interrupt would push
    /// as its return address.
    fn rip_after(&mut self, instruction: &Instruction) -> Value {
        self.runtime_addr(instruction.next_ip())
    }

    /// Emit an `Interrupt` VM exit for vector `vector`, syncing guest state and
    /// pointing RIP at the instruction after `instruction` (the resume point).
    fn emit_interrupt_exit(&mut self, instruction: &Instruction, vector: u32) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Interrupt)?;
        self.resumes_at(instruction.next_ip());
        self.write_exit_code(vector);
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }

    /// Handle `INT imm8` - raise a software interrupt for the immediate vector.
    /// The vector is reported in `exit_code`; RIP points past the `INT` so the
    /// host can service the vector and resume.
    pub(super) fn handle_int(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let vector = instruction.immediate8() as u32;
        self.emit_interrupt_exit(instruction, vector)
    }

    /// Handle `INT1`/`ICEBP` (`0xF1`) - raise the debug-exception vector (1).
    ///
    /// (`INTO`, the legacy `0xCE` overflow trap, is intentionally not handled:
    /// it is not a valid instruction in 64-bit mode and the decoder rejects it
    /// as `InvalidInstruction`.)
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

    /// RDTSC - read the timestamp counter. The lifted code can't read the host
    /// TSC meaningfully, so this is a resumable VM exit: the PC points at the
    /// instruction *after* `RDTSC` and the host populates `rax`/`rdx` with the
    /// 64-bit counter (low 32 bits in `rax`, high 32 in `rdx`) before resuming.
    pub(super) fn handle_rdtsc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.rip_after(instruction);
        self.sync_to_context(Some(rip_after), ExitReason::Rdtsc)?;
        self.resumes_at(instruction.next_ip());
        self.emit_host_return();
        self.block_terminated = true;
        Ok(())
    }
}
