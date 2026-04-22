use crate::context::{ExitReason, RegisterContext};
use crate::lifter::{ExecutionMode, Lifter};
use cranelift::prelude::*;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// Handle an unimplemented instruction by exiting with a fallback exit.
    /// The host can then shim the instruction and either:
    /// 1. (JIT mode) Call lifter again to generate a resume block
    /// 2. (RuntimeAssisted) Resume via pre-generated resume_function
    pub(super) fn handle_unimplemented_instruction(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<()> {
        // Sync full state to RegisterContext - we're exiting to the host
        self.sync_to_context(None, ExitReason::UnimplementedInstruction)?;

        // Store the instruction code for the host to identify what failed
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let exit_code_val = self
            .builder
            .ins()
            .iconst(types::I32, instruction.code() as i64);
        let exit_code_offset = std::mem::offset_of!(RegisterContext, exit_code) as i32;
        self.builder.ins().store(
            MemFlags::trusted(),
            exit_code_val,
            ctx_ptr,
            exit_code_offset,
        );

        // In RuntimeAssisted mode or when emit_resume_blocks is enabled,
        // we can set the resume_function pointer to point back to the entry
        // point of this region (or a block). For now we leave it as 0
        // and expect the host to handle resumption.
        if self.config.execution_mode == ExecutionMode::RuntimeAssisted
            || self.config.emit_resume_blocks
        {
            self.set_resume_placeholder()?;
        }

        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    /// Helper to set a placeholder in the resume_function field.
    fn set_resume_placeholder(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // FIXME: this is not correct, we need to set the address of the current region's entry point
        let resume_addr = self.builder.ins().iconst(types::I64, 0);
        let resume_offset = std::mem::offset_of!(RegisterContext, resume_function) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), resume_addr, ctx_ptr, resume_offset);

        Ok(())
    }
}
