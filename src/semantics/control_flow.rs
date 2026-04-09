use crate::MicroliftError;
use crate::context::ExitReason;
use crate::lifter::Lifter;
use crate::semantics::helpers::size_to_type;
use cranelift::prelude::*;
use iced_x86::{Code, CodeSize, Instruction, OpKind, Register};

impl<'a> Lifter<'a> {
    pub(super) fn handle_jmp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        match instruction.op0_kind() {
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64 => {
                let target = instruction.near_branch_target();
                let target_block = self.get_block(target);
                self.builder.ins().jump(target_block, &[]);
                self.block_terminated = true;
            }
            OpKind::Register => {
                // Indirect jump via register - emit a runtime dispatch switch
                let target_val = self.get_reg_val(instruction.op0_register())?;
                self.emit_runtime_indirect_jump(target_val)?;
            }
            OpKind::Memory => {
                // Indirect jump via memory - load the target then dispatch
                let ptr = self.get_mem_ptr(instruction, 0)?;
                let target_val = self.builder.ins().load(types::I64, MemFlags::new(), ptr, 0);
                self.emit_runtime_indirect_jump(target_val)?;
            }
            other => {
                return Err(MicroliftError::UnsupportedInstruction {
                    mnemonic: format!("jmp-{:?}", other),
                });
            }
        }
        Ok(())
    }

    /// Emit a switch-based indirect jump dispatch for a runtime target value.
    /// Jumps to any known block whose offset matches; otherwise exits the VM.
    pub fn emit_runtime_indirect_jump(&mut self, target_val: Value) -> crate::Result<()> {
        let image_base = self.builder.use_var(self.state.image_base);
        // Convert absolute target address -> region offset
        let sub_base = self.builder.ins().isub(target_val, image_base);
        let target_offset = self.builder.ins().iadd_imm(sub_base, self.lift_base as i64);

        let mut switch = cranelift_frontend::Switch::new();
        for (&offset, &block) in &self.blocks {
            switch.set_entry(offset as u128, block);
        }

        let external_block = self.builder.create_block();
        switch.emit(&mut self.builder, target_offset, external_block);

        self.builder.switch_to_block(external_block);
        self.builder.seal_block(external_block);
        self.sync_to_context(Some(target_val), ExitReason::UnknownJump)?;
        self.builder.ins().return_(&[]);
        self.block_terminated = true;
        Ok(())
    }

    pub(super) fn handle_ret(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let imm = instruction.immediate16(); // for RET n

        let rsp = self.get_reg_val(Register::RSP)?;
        let target_ip = self.builder.ins().load(types::I64, MemFlags::new(), rsp, 0);

        let mut new_rsp = self.builder.ins().iadd_imm(rsp, 8);
        if imm > 0 {
            new_rsp = self.builder.ins().iadd_imm(new_rsp, imm as i64);
        }
        self.set_reg_val(Register::RSP, new_rsp)?;

        // Internal Dispatch via Switch
        let mut switch = cranelift_frontend::Switch::new();
        let image_base = self.builder.use_var(self.state.image_base);

        // target_offset = target_ip - image_base + lift_base
        let sub_base = self.builder.ins().isub(target_ip, image_base);
        let target_offset = self.builder.ins().iadd_imm(sub_base, self.lift_base as i64);

        for (&offset, &block) in &self.blocks {
            switch.set_entry(offset as u128, block);
        }

        let external_check_block = self.builder.create_block();
        switch.emit(&mut self.builder, target_offset, external_check_block);

        // Handle Return via Trampoline or Unknown
        self.builder.switch_to_block(external_check_block);
        self.builder.seal_block(external_check_block);

        let trampoline = self
            .builder
            .ins()
            .iconst(types::I64, self.exit_trampoline_address as i64);
        let is_trampoline = self.builder.ins().icmp(IntCC::Equal, target_ip, trampoline);

        let exit_block = self.builder.create_block();
        let unknown_block = self.builder.create_block();

        self.builder
            .ins()
            .brif(is_trampoline, exit_block, &[], unknown_block, &[]);

        // Handle Return via Trampoline
        self.builder.switch_to_block(exit_block);
        self.sync_to_context(Some(target_ip), ExitReason::Return)?;
        self.builder.ins().return_(&[]);

        // Handle Unknown Jump/Return
        self.builder.switch_to_block(unknown_block);
        self.sync_to_context(Some(target_ip), ExitReason::UnknownJump)?;
        self.builder.ins().return_(&[]);

        self.block_terminated = true;
        Ok(())
    }

    pub(super) fn handle_call(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        // Ensure return address is a known block
        self.get_block(next_ip);

        // Push return address (current_ip) reconciled with image_base
        let image_base = self.builder.use_var(self.state.image_base);
        let offset = self.current_ip.wrapping_sub(self.lift_base);
        let ret_addr = self.builder.ins().iadd_imm(image_base, offset as i64);

        let rsp = self.get_reg_val(Register::RSP)?;
        let new_rsp = self.builder.ins().iadd_imm(rsp, -8);
        self.set_reg_val(Register::RSP, new_rsp)?;
        self.builder
            .ins()
            .store(MemFlags::new(), ret_addr, new_rsp, 0);

        // Jump to target
        let target_block = self.get_block(target);
        self.builder.ins().jump(target_block, &[]);
        self.block_terminated = true;
        Ok(())
    }

    // Conditional branches

    pub(super) fn handle_jb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_conditional_branch(instruction, cf)
    }

    pub(super) fn handle_jbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_conditional_branch(instruction, cond)
    }

    /// JA (above): CF=0 and ZF=0
    pub(super) fn handle_ja(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        // JA = branch when CF=0 AND ZF=0 (inverted: don't branch when CF OR ZF)
        self.handle_conditional_branch_inverted(instruction, cf_or_zf)
    }

    /// JAE (above or equal): CF=0
    pub(super) fn handle_jae(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_conditional_branch_inverted(instruction, cf)
    }

    pub(super) fn handle_jz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_conditional_branch(instruction, zf)
    }

    pub(super) fn handle_jnz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_conditional_branch_inverted(instruction, zf)
    }

    /// JG (greater, signed): ZF=0 and SF=OF
    pub(super) fn handle_jg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        // SF == OF
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // cond_not_g = ZF | (SF ^ OF)  - branch when this is 0
        let not_g = self.builder.ins().bor(zf, sf_xor_of);
        self.handle_conditional_branch_inverted(instruction, not_g)
    }

    /// JGE (greater or equal, signed): SF=OF
    pub(super) fn handle_jge(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // JGE: branch when SF == OF (i.e. SF^OF == 0)
        self.handle_conditional_branch_inverted(instruction, sf_xor_of)
    }

    /// JL (less, signed): SF != OF
    pub(super) fn handle_jl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        // JL: SF != OF
        self.handle_conditional_branch(instruction, sf_xor_of)
    }

    /// JLE (less or equal, signed): ZF=1 or SF!=OF
    pub(super) fn handle_jle(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor_of = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bor(zf, sf_xor_of);
        self.handle_conditional_branch(instruction, cond)
    }

    /// JS (sign set): SF=1
    pub(super) fn handle_js(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_conditional_branch(instruction, sf)
    }

    /// JNS (sign clear): SF=0
    pub(super) fn handle_jns(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_conditional_branch_inverted(instruction, sf)
    }

    /// JO (overflow set): OF=1
    pub(super) fn handle_jo(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_conditional_branch(instruction, of)
    }

    /// JNO (overflow clear): OF=0
    pub(super) fn handle_jno(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_conditional_branch_inverted(instruction, of)
    }

    /// JP (parity set): PF=1
    pub(super) fn handle_jp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_conditional_branch(instruction, pf)
    }

    /// JNP (parity clear): PF=0
    pub(super) fn handle_jnp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_conditional_branch_inverted(instruction, pf)
    }

    /// JRCXZ / JECXZ / JCXZ - branch if CX/ECX/RCX == 0
    pub(super) fn handle_jrcxz(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let counter = match instruction.code() {
            Code::Jcxz_rel8_16 => self.get_reg_val(Register::CX)?,
            Code::Jecxz_rel8_32 => self.get_reg_val(Register::ECX)?,
            _ => self.get_reg_val(Register::RCX)?,
        };
        let cond = self.builder.ins().icmp_imm(IntCC::Equal, counter, 0);
        self.handle_conditional_branch(instruction, cond)
    }

    pub(super) fn handle_loop(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let counter_reg = match instruction.code() {
            Code::Loop_rel8_16_CX | Code::Loop_rel8_16_ECX | Code::Loop_rel8_16_RCX => Register::CX,
            Code::Loop_rel8_32_CX | Code::Loop_rel8_32_ECX => Register::ECX,
            Code::Loop_rel8_64_ECX => Register::ECX,
            Code::Loop_rel8_64_RCX => Register::RCX,
            _ => match instruction.code_size() {
                CodeSize::Code16 => Register::CX,
                CodeSize::Code32 => Register::ECX,
                _ => Register::RCX,
            },
        };

        let counter = self.get_reg_val(counter_reg)?;
        let one = self
            .builder
            .ins()
            .iconst(size_to_type(counter_reg.size()), 1);
        let decremented = self.builder.ins().isub(counter, one);
        self.set_reg_val(counter_reg, decremented)?;
        self.handle_conditional_branch(instruction, decremented)
    }

    // CMOV family

    /// Generic CMOVcc - move if condition is true.
    fn handle_cmov_cond(&mut self, instruction: &Instruction, cond: Value) -> crate::Result<()> {
        let size = self.operand_size(instruction, 0)?;
        let dst = self.read_operand(instruction, 0, size)?;
        let src = self.read_operand(instruction, 1, size)?;
        let result = self.builder.ins().select(cond, src, dst);
        self.write_operand(instruction, 0, result)
    }

    pub(super) fn handle_cmove(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        self.handle_cmov_cond(instruction, zf)
    }

    pub(super) fn handle_cmovne(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let nzf = self.builder.ins().bnot(zf);
        let nzf8 = self.builder.ins().ireduce(types::I8, nzf);
        self.handle_cmov_cond(instruction, nzf8)
    }

    pub(super) fn handle_cmovb(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        self.handle_cmov_cond(instruction, cf)
    }

    pub(super) fn handle_cmovae(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let ncf = self.builder.ins().bnot(cf);
        let ncf8 = self.builder.ins().ireduce(types::I8, ncf);
        self.handle_cmov_cond(instruction, ncf8)
    }

    pub(super) fn handle_cmovbe(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cond = self.builder.ins().bor(cf, zf);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmova(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let cf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Carry]);
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let cf_or_zf = self.builder.ins().bor(cf, zf);
        let cond = self.builder.ins().bnot(cf_or_zf);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    pub(super) fn handle_cmovl(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let cond = self.builder.ins().bxor(sf, of);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovge(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bnot(sf_xor);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    pub(super) fn handle_cmovle(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let cond = self.builder.ins().bor(zf, sf_xor);
        self.handle_cmov_cond(instruction, cond)
    }

    pub(super) fn handle_cmovg(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let zf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Zero]);
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let sf_xor = self.builder.ins().bxor(sf, of);
        let not_g = self.builder.ins().bor(zf, sf_xor);
        let cond = self.builder.ins().bnot(not_g);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    pub(super) fn handle_cmovs(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        self.handle_cmov_cond(instruction, sf)
    }

    pub(super) fn handle_cmovns(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let sf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Sign]);
        let cond = self.builder.ins().bnot(sf);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    pub(super) fn handle_cmovo(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        self.handle_cmov_cond(instruction, of)
    }

    pub(super) fn handle_cmovno(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let of = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Overflow]);
        let cond = self.builder.ins().bnot(of);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    pub(super) fn handle_cmovp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        self.handle_cmov_cond(instruction, pf)
    }

    pub(super) fn handle_cmovnp(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let pf = self
            .builder
            .use_var(self.state.flags[&crate::lifter::X86Flag::Parity]);
        let cond = self.builder.ins().bnot(pf);
        let cond8 = self.builder.ins().ireduce(types::I8, cond);
        self.handle_cmov_cond(instruction, cond8)
    }

    // Shared branch helpers

    pub fn handle_conditional_branch(
        &mut self,
        instruction: &Instruction,
        cond: Value,
    ) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block(target);
        let next_block = self.get_block(next_ip);

        self.builder
            .ins()
            .brif(cond, target_block, &[], next_block, &[]);

        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

        Ok(())
    }

    pub fn handle_conditional_branch_inverted(
        &mut self,
        instruction: &Instruction,
        cond: Value,
    ) -> crate::Result<()> {
        let target = instruction.near_branch_target();
        let next_ip = instruction.next_ip();

        let target_block = self.get_block(target);
        let next_block = self.get_block(next_ip);

        self.builder
            .ins()
            .brif(cond, next_block, &[], target_block, &[]);

        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;

        Ok(())
    }
}
