use crate::MicroliftError;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::{Decoder, DecoderOptions, Instruction, Register};
use std::collections::{HashMap, HashSet};

/// Mapping of x86nd registers to Cranelift variables.
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
    pub context_ptr: Variable,
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

        Self {
            gprs,
            flags,
            context_ptr,
        }
    }

    pub fn get_gpr_offset(reg: Register) -> i32 {
        match reg {
            Register::RAX => 0,
            Register::RCX => 8,
            Register::RDX => 16,
            Register::RBX => 24,
            Register::RSP => 32,
            Register::RBP => 40,
            Register::RSI => 48,
            Register::RDI => 56,
            Register::R8 => 64,
            Register::R9 => 72,
            Register::R10 => 80,
            Register::R11 => 88,
            Register::R12 => 96,
            Register::R13 => 104,
            Register::R14 => 112,
            Register::R15 => 120,
            _ => panic!("Unsupported register for offset: {:?}", reg),
        }
    }

    pub fn get_flag_offset(flag: X86Flag) -> i32 {
        match flag {
            X86Flag::Carry => 136,
            X86Flag::Parity => 137,
            X86Flag::Auxiliary => 138,
            X86Flag::Zero => 139,
            X86Flag::Sign => 140,
            X86Flag::Overflow => 141,
        }
    }
}

pub struct Lifter<'a> {
    pub builder: FunctionBuilder<'a>,
    pub state: RegisterState,
    pub blocks: HashMap<u64, Block>,
    pub sealed_blocks: HashSet<Block>,
}

impl<'a> Lifter<'a> {
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        let mut builder = builder;
        let state = RegisterState::new(&mut builder);
        Self {
            builder,
            state,
            blocks: HashMap::new(),
            sealed_blocks: HashSet::new(),
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

        // Start initial block
        let initial_block = self.get_block(ip);
        self.builder
            .append_block_params_for_function_params(initial_block);
        self.builder.switch_to_block(initial_block);
        self.builder.seal_block(initial_block);
        self.sealed_blocks.insert(initial_block);

        // Use the first function parameter as the context pointer
        let ctx_ptr_val = self.builder.block_params(initial_block)[0];
        self.builder.def_var(self.state.context_ptr, ctx_ptr_val);

        // Load initial state from context
        for (&reg, &var) in &self.state.gprs {
            let offset = RegisterState::get_gpr_offset(reg);
            let val = self
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ctx_ptr_val, offset);
            self.builder.def_var(var, val);
        }

        for (&flag, &var) in &self.state.flags {
            let offset = RegisterState::get_flag_offset(flag);
            let val = self
                .builder
                .ins()
                .load(types::I8, MemFlags::trusted(), ctx_ptr_val, offset);
            self.builder.def_var(var, val);
        }

        while decoder.can_decode() {
            let current_ip = decoder.ip();

            // Check if this IP is a jump target
            if let Some(&block) = self.blocks.get(&current_ip) {
                if Some(block) != self.builder.current_block() {
                    if !self.builder.is_unreachable() {
                        self.builder.ins().jump(block, &[]);
                    }
                    self.builder.switch_to_block(block);
                }
                if !self.sealed_blocks.contains(&block) {
                    self.builder.seal_block(block);
                    self.sealed_blocks.insert(block);
                }
            }

            if self.builder.is_unreachable() {
                // If we are in an unreachable state, and the start of the loop didn't switch us to a new block,
                // then the following instructions are dead code. We skip them.
                let mut instruction = Instruction::default();
                decoder.decode_out(&mut instruction);
                continue;
            }

            let mut instruction = Instruction::default();
            decoder.decode_out(&mut instruction);

            if instruction.is_invalid() {
                return Err(MicroliftError::DecodeError {
                    offset: decoder.ip(),
                    reason: "invalid instruction".to_string(),
                });
            }

            self.lift_instruction(&instruction)?;
        }

        Ok(())
    }

    fn lift_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_instruction(instruction)
    }

    pub fn sync_to_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg);
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag);
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        Ok(())
    }
}
