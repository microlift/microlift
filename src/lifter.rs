use crate::{
    MicroliftError,
    context::{ExitReason, MemorySegment, RegisterContext},
};
use cranelift::prelude::*;
use cranelift_codegen::ir::SigRef;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, Switch, Variable};
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, Register};
use std::collections::{HashMap, HashSet};

/// Mapping of x86 registers to Cranelift variables.
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
    pub context_ptr: Variable,
    pub image_base: Variable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Flag {
    Carry,
    Parity,
    Auxiliary,
    Zero,
    Sign,
    Overflow,
}

impl RegisterState {
    pub fn new(builder: &mut FunctionBuilder) -> Self {
        let mut gprs = HashMap::new();

        // Define all 64-bit GPRs
        let qword_gprs = [
            Register::RAX,
            Register::RCX,
            Register::RDX,
            Register::RBX,
            Register::RSP,
            Register::RBP,
            Register::RSI,
            Register::RDI,
            Register::R8,
            Register::R9,
            Register::R10,
            Register::R11,
            Register::R12,
            Register::R13,
            Register::R14,
            Register::R15,
        ];

        for &reg in &qword_gprs {
            let var = builder.declare_var(types::I64);
            gprs.insert(reg, var);
        }

        let mut flags = HashMap::new();
        let flag_types = [
            X86Flag::Carry,
            X86Flag::Parity,
            X86Flag::Auxiliary,
            X86Flag::Zero,
            X86Flag::Sign,
            X86Flag::Overflow,
        ];

        for &flag in &flag_types {
            let var = builder.declare_var(types::I8);
            flags.insert(flag, var);
        }

        let context_ptr = builder.declare_var(types::I64); // Assume 64-bit pointer
        let image_base = builder.declare_var(types::I64);

        Self {
            gprs,
            flags,
            context_ptr,
            image_base,
        }
    }

    pub fn get_gpr_offset(reg: Register) -> crate::Result<i32> {
        match reg {
            Register::RAX => Ok(std::mem::offset_of!(RegisterContext, rax) as i32),
            Register::RCX => Ok(std::mem::offset_of!(RegisterContext, rcx) as i32),
            Register::RDX => Ok(std::mem::offset_of!(RegisterContext, rdx) as i32),
            Register::RBX => Ok(std::mem::offset_of!(RegisterContext, rbx) as i32),
            Register::RSP => Ok(std::mem::offset_of!(RegisterContext, rsp) as i32),
            Register::RBP => Ok(std::mem::offset_of!(RegisterContext, rbp) as i32),
            Register::RSI => Ok(std::mem::offset_of!(RegisterContext, rsi) as i32),
            Register::RDI => Ok(std::mem::offset_of!(RegisterContext, rdi) as i32),
            Register::R8 => Ok(std::mem::offset_of!(RegisterContext, r8) as i32),
            Register::R9 => Ok(std::mem::offset_of!(RegisterContext, r9) as i32),
            Register::R10 => Ok(std::mem::offset_of!(RegisterContext, r10) as i32),
            Register::R11 => Ok(std::mem::offset_of!(RegisterContext, r11) as i32),
            Register::R12 => Ok(std::mem::offset_of!(RegisterContext, r12) as i32),
            Register::R13 => Ok(std::mem::offset_of!(RegisterContext, r13) as i32),
            Register::R14 => Ok(std::mem::offset_of!(RegisterContext, r14) as i32),
            Register::R15 => Ok(std::mem::offset_of!(RegisterContext, r15) as i32),
            _ => Err(MicroliftError::UnsupportedRegister(reg)),
        }
    }

    pub fn get_flag_offset(flag: X86Flag) -> crate::Result<i32> {
        match flag {
            X86Flag::Carry => Ok(std::mem::offset_of!(RegisterContext, cf) as i32),
            X86Flag::Parity => Ok(std::mem::offset_of!(RegisterContext, pf) as i32),
            X86Flag::Auxiliary => Ok(std::mem::offset_of!(RegisterContext, af) as i32),
            X86Flag::Zero => Ok(std::mem::offset_of!(RegisterContext, zf) as i32),
            X86Flag::Sign => Ok(std::mem::offset_of!(RegisterContext, sf) as i32),
            X86Flag::Overflow => Ok(std::mem::offset_of!(RegisterContext, of) as i32),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Jit,
    RuntimeAssisted,
}

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    pub syscall_handler_call_conv: CallConv,
    pub segments: Vec<MemorySegment>,
    pub interpreter_fallback: bool,
    pub emit_resume_blocks: bool,
}

impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            syscall_handler_call_conv: CallConv::SystemV,
            segments: Vec::new(),
            interpreter_fallback: false,
            emit_resume_blocks: false,
        }
    }
}

pub struct Lifter<'a> {
    pub builder: FunctionBuilder<'a>,
    pub state: RegisterState,
    pub config: LifterConfig,
    pub blocks: HashMap<u64, Block>,
    pub sealed_blocks: HashSet<Block>,
    /// The `ip` value passed to `lift()` - i.e. the original VA of the first
    /// instruction in the lifted region.  RIP offsets are relative to this.
    pub lift_base: u64,
    /// Tracks `instruction.next_ip()` for the instruction currently being
    /// lifted.  This is the value RIP would hold after that instruction
    /// retires, and is what we materialise on `Register::RIP` reads and store
    /// into `RegisterContext::rip` on VM exit.
    pub current_ip: u64,
    /// Blocks that have been filled with instructions during the lift loop.
    pub visited_blocks: HashSet<Block>,
    /// Tracks if the current block has been terminated (jmp, ret, etc.)
    pub block_terminated: bool,
    /// Designated address for controlled VM exit via RET.
    pub exit_trampoline_address: u64,
    pub syscall_handler_sig: Option<SigRef>,
    /// Dirty-bit tracking: GPRs that have been written since the last sync.
    pub dirty_gprs: HashSet<Register>,
    /// Dirty-bit tracking: flags that have been written since the last sync.
    pub dirty_flags: HashSet<X86Flag>,
}

impl<'a> Lifter<'a> {
    /// Create a new `Lifter`.
    ///
    /// RIP is reconciled dynamically at runtime as `image_base + (current_ip - lift_base)`,
    /// where `image_base` is loaded from the `RegisterContext`. This allows the
    /// lifted code to remain correct even if the program image is relocated by ASLR.
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        Self::with_config(builder, LifterConfig::default())
    }

    pub fn with_config(builder: FunctionBuilder<'a>, config: LifterConfig) -> Self {
        let mut builder = builder;
        let state = RegisterState::new(&mut builder);
        let syscall_handler_sig = match config.execution_mode {
            ExecutionMode::Jit => None,
            ExecutionMode::RuntimeAssisted => {
                let mut sig = Signature::new(config.syscall_handler_call_conv);
                sig.params.push(AbiParam::new(types::I64));
                Some(builder.import_signature(sig))
            }
        };

        Self {
            builder,
            state,
            config,
            blocks: HashMap::new(),
            sealed_blocks: HashSet::new(),
            lift_base: 0,
            current_ip: 0,
            visited_blocks: HashSet::new(),
            block_terminated: false,
            exit_trampoline_address: 0xFFFF_FFFF_FFFF_FFFF, // Default to a safe high address
            syscall_handler_sig,
            dirty_gprs: HashSet::new(),
            dirty_flags: HashSet::new(),
        }
    }

    pub fn get_block(&mut self, offset: u64) -> Block {
        if let Some(&block) = self.blocks.get(&offset) {
            block
        } else {
            let block = self.builder.create_block();
            self.blocks.insert(offset, block);
            block
        }
    }

    pub fn lift(&mut self, bytes: &[u8], ip: u64) -> crate::Result<()> {
        let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        let mut instructions = Vec::new();
        let mut leaders = HashSet::from([ip]);

        while decoder.can_decode() {
            let mut instruction = Instruction::default();
            decoder.decode_out(&mut instruction);

            if instruction.is_invalid() {
                // Treat as block terminator - next IP may still be a
                // valid branch target from elsewhere in the function.
                leaders.insert(instruction.next_ip());
                instructions.push(instruction); // keep it, phase 2 will emit a fault stub
                continue; // keep scanning past it
            }

            match instruction.mnemonic() {
                Mnemonic::Jmp
                | Mnemonic::Je
                | Mnemonic::Jne
                | Mnemonic::Jb
                | Mnemonic::Jbe
                | Mnemonic::Loop
                | Mnemonic::Call => {
                    leaders.insert(instruction.near_branch_target());
                }
                _ => {}
            }

            instructions.push(instruction);
        }

        // Record the base IP for RIP-offset arithmetic.
        self.lift_base = ip;

        // Create an entry block that receives function parameters
        let entry_block = self.builder.create_block();
        self.builder
            .append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);

        // Record the base IP for RIP-offset arithmetic.
        self.lift_base = ip;

        // Use the first function parameter as the context pointer
        let ctx_ptr_val = self.builder.block_params(entry_block)[0];
        self.builder.def_var(self.state.context_ptr, ctx_ptr_val);

        // Load state from context *before* dispatching. Logic in blocks will use
        // the SSA variables defined here.
        self.load_state_from_context()?;

        if self.config.emit_resume_blocks {
            for instruction in &instructions {
                leaders.insert(instruction.ip());
                leaders.insert(instruction.next_ip());
            }
        }

        // Populate blocks map for all leaders
        for &leader in &leaders {
            self.get_block(leader);
        }

        // Resumable Dispatch:
        // Instead of always jumping to the 'ip' passed to lift, we look at the
        // current RIP in the context. If it matches one of our blocks, we jump there.
        // This is what enables resumable AOT.
        let rip_val = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx_ptr_val,
            std::mem::offset_of!(RegisterContext, rip) as i32,
        );
        let image_base_val = self.builder.use_var(self.state.image_base);

        // Normalize RIP to a region-relative offset
        let relative_rip = self.builder.ins().isub(rip_val, image_base_val);

        let mut switch = Switch::new();
        for (&offset, &block) in &self.blocks {
            let relative_offset = offset.wrapping_sub(self.lift_base);
            switch.set_entry(relative_offset.into(), block);
        }

        let start_block = self.get_block(ip);
        switch.emit(&mut self.builder, relative_rip, start_block);

        // We don't switch_to_block here yet, we'll do it in the loop
        self.visited_blocks.insert(start_block);
        self.block_terminated = true; // Force switch to first block in loop

        for instruction in instructions {
            let current_ip_dec = instruction.ip();

            // Check if this IP is a jump target
            if let Some(&block) = self.blocks.get(&current_ip_dec) {
                if Some(block) != self.builder.current_block() || self.block_terminated {
                    if !self.block_terminated {
                        self.builder.ins().jump(block, &[]);
                    }
                    self.builder.switch_to_block(block);
                    self.visited_blocks.insert(block);
                    self.block_terminated = false;
                }
            }

            if self.block_terminated {
                // Dead code after a terminator - decode and skip.
                let mut instruction = Instruction::default();
                decoder.decode_out(&mut instruction);
                continue;
            }

            if instruction.is_invalid() {
                // Sync RIP to point AT the invalid instruction so the host
                // sees the correct faulting address.
                self.current_ip = instruction.ip(); // not next_ip - we didn't execute it
                self.sync_to_context(None, ExitReason::InvalidInstruction)?;
                self.builder.ins().return_(&[]);
                self.block_terminated = true;
                continue;
            }

            // Advance current_ip to next_ip *before* dispatching so that any
            // read of Register::RIP inside the handler sees the correct value
            // (RIP always points past the current instruction on real hardware).
            self.current_ip = instruction.next_ip();

            self.lift_instruction(&instruction)?;
        }

        // Handle fallthrough at the end of the lifted region.
        if !self.block_terminated {
            self.sync_to_context(None, ExitReason::None)?;
            self.builder.ins().return_(&[]);
            self.block_terminated = true;
        }

        self.finish()
    }

    /// Fill any blocks that were created (e.g. as jump targets) but never
    /// visited by the decoder. These represent "VM Exits" to addresses
    /// outside the currently lifted region.
    pub fn finish(&mut self) -> crate::Result<()> {
        let mut processed_blocks = HashSet::new();
        let all_offsets: Vec<u64> = self.blocks.keys().cloned().collect();

        for offset in all_offsets {
            let block = self.blocks[&offset];
            if !self.visited_blocks.contains(&block) && !processed_blocks.contains(&block) {
                self.builder.switch_to_block(block);
                // Sync to context at this target IP
                self.current_ip = offset;
                self.sync_to_context(None, ExitReason::None)?;
                self.builder.ins().return_(&[]);
                processed_blocks.insert(block);
            }
        }

        // Final Sealing Pass: Seal all blocks currently known to the builder.
        // This includes all internal targets, exit blocks, and fallthroughs.
        self.builder.seal_all_blocks();

        Ok(())
    }

    fn lift_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_instruction(instruction)
    }

    /// Full sync - writes all GPRs and flags back to the `RegisterContext`.
    /// Used on context reload paths and in unvisited exit stubs where we have
    /// no accurate dirty-set information.
    pub fn sync_to_context(&mut self, rip: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag)?;
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        self.flush_rip_and_reason(rip, reason)?;
        Ok(())
    }

    /// Selective sync - only writes GPRs and flags that have been written since
    /// the last `load_state_from_context` call. RIP and exit_reason are always
    /// written. This is the preferred sync to use on normal VM-exit paths.
    pub fn sync_to_context_selective(
        &mut self,
        rip: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // Clone the dirty sets so we can iterate without borrowing self.
        let dirty_gprs: Vec<Register> = self.dirty_gprs.iter().cloned().collect();
        let dirty_flags: Vec<X86Flag> = self.dirty_flags.iter().cloned().collect();

        for reg in dirty_gprs {
            if let Some(&var) = self.state.gprs.get(&reg) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_gpr_offset(reg)?;
                self.builder
                    .ins()
                    .store(MemFlags::trusted(), val, ctx_ptr, offset);
            }
        }

        for flag in dirty_flags {
            if let Some(&var) = self.state.flags.get(&flag) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.builder
                    .ins()
                    .store(MemFlags::trusted(), val, ctx_ptr, offset);
            }
        }

        self.flush_rip_and_reason(rip, reason)?;
        Ok(())
    }

    /// Shared helper: write RIP and exit_reason to the context. Always called
    /// by both `sync_to_context` and `sync_to_context_selective`.
    fn flush_rip_and_reason(
        &mut self,
        rip: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        let rip_val = if let Some(r) = rip {
            r
        } else {
            // Reconcile statically from current_ip
            let image_base = self.builder.use_var(self.state.image_base);
            let offset = self.current_ip.wrapping_sub(self.lift_base);
            self.builder.ins().iadd_imm(image_base, offset as i64)
        };

        let rip_offset = std::mem::offset_of!(crate::context::RegisterContext, rip) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), rip_val, ctx_ptr, rip_offset);

        // Store exit reason
        let reason_val = self.builder.ins().iconst(types::I32, reason as i64);
        let reason_offset =
            std::mem::offset_of!(crate::context::RegisterContext, exit_reason) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), reason_val, ctx_ptr, reason_offset);

        Ok(())
    }

    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // RIP is not kept as an SSA variable. We materialize it from current_ip
        // when syncing state back to the context.
        for (&reg, &var) in &self.state.gprs {
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val = self
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        for (&flag, &var) in &self.state.flags {
            let offset = RegisterState::get_flag_offset(flag)?;
            let val = self
                .builder
                .ins()
                .load(types::I8, MemFlags::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let image_base_offset = std::mem::offset_of!(RegisterContext, image_base) as i32;
        let image_base_val =
            self.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ctx_ptr, image_base_offset);
        self.builder.def_var(self.state.image_base, image_base_val);

        // After a full reload, the dirty sets are cleared - everything in the
        // context is now in sync with our SSA variables.
        self.dirty_gprs.clear();
        self.dirty_flags.clear();

        Ok(())
    }
}
