use crate::lifter::{ExecutionMode, Lifter};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    pub(super) fn handle_syscall(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match self.config.execution_mode {
            ExecutionMode::Jit => {
                let rip_after = self.runtime_addr(instruction.next_ip());

                self.sync_to_context(Some(rip_after), crate::context::ExitReason::Syscall)?;
                self.resumes_at(instruction.next_ip());
                self.emit_host_return();
                self.block_terminated = true;
            }
            ExecutionMode::RuntimeAssisted => {
                let next = instruction.next_ip();
                self.resumes_at(next);
                let in_region = self.in_region(next);

                let ctx_ptr = self.builder.use_var(self.state.context_ptr);
                let handler_offset =
                    std::mem::offset_of!(crate::context::RegisterContext, syscall_handler) as i32;
                let handler_ptr = self.ctx_load(types::I64, ctx_ptr, handler_offset);

                let has_handler = self
                    .builder
                    .ins()
                    .icmp_imm_s(IntCC::NotEqual, handler_ptr, 0);
                let assist_block = self.builder.create_block();
                let exit_block = self.builder.create_block();
                let resume_block = self.get_block(next);

                self.builder
                    .ins()
                    .brif(has_handler, assist_block, &[], exit_block, &[]);

                self.builder.switch_to_block(assist_block);
                self.sync_to_context(None, crate::context::ExitReason::None)?;
                let sig = self
                    .syscall_handler_sig
                    .expect("runtime-assisted syscall mode requires a handler signature");
                self.builder
                    .ins()
                    .call_indirect(sig, handler_ptr, &[ctx_ptr]);
                self.load_state_from_context()?;
                // The handler may have written any field, so nothing past here
                // still holds its entry value; see `Lifter::mark_all_written`.
                self.mark_all_written();
                self.builder.ins().jump(resume_block, &[]);

                self.builder.switch_to_block(exit_block);
                let rip_after = self.runtime_addr(instruction.next_ip());
                self.sync_to_context(Some(rip_after), crate::context::ExitReason::Syscall)?;
                self.emit_host_return();

                if in_region {
                    self.continue_in_fallthrough(resume_block);
                } else {
                    // `finish` turns an out-of-region block into the VM-exit
                    // stub that leaves at `next`; there is nothing to emit.
                    self.block_terminated = true;
                }
            }
        }

        Ok(())
    }
}
