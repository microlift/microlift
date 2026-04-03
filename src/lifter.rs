use crate::MicroliftError;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::{Decoder, DecoderOptions, Instruction, Register};
use std::collections::{HashMap, HashSet};

/// Mapping of x86nd registers to Cranelift variables.
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
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

        Self { gprs, flags }
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

        // TODO initial values for registers (e.g. from function arguments)
        let zero = self.builder.ins().iconst(types::I64, 0);
        for &var in self.state.gprs.values() {
            self.builder.def_var(var, zero);
        }

        let zero_flag = self.builder.ins().iconst(types::I8, 0);
        for &var in self.state.flags.values() {
            self.builder.def_var(var, zero_flag);
        }

        while decoder.can_decode() {
            let current_ip = decoder.ip();

            // Check if this IP is a jump target
            if current_ip != ip {
                if let Some(&block) = self.blocks.get(&current_ip) {
                    if !self.builder.is_unreachable() {
                        self.builder.ins().jump(block, &[]);
                    }
                    self.builder.switch_to_block(block);
                    if !self.sealed_blocks.contains(&block) {
                        self.builder.seal_block(block);
                        self.sealed_blocks.insert(block);
                    }
                }
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

            // If instructions were flow-altering (ret, jmp, etc),
            // the builder state might be finished for this block.
        }

        Ok(())
    }

    fn lift_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_instruction(instruction)
    }
}
