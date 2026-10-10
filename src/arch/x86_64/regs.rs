//! x86-64 register file: SSA-variable mapping, flags, dirty tracking, `RegisterContext` offsets.

use crate::MicroliftError;
use crate::arch::x86_64::context::RegisterContext;
use crate::core::LifterConfig;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::Register;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Flag {
    Carry,
    Parity,
    Auxiliary,
    Zero,
    Sign,
    Overflow,
    /// Controls string-operation pointer direction.
    Direction,
}

/// The seven modelled flags in fixed order; iterate this, not a `HashMap`, which would change `.text` between lifts.
pub const FLAG_ORDER: [X86Flag; 7] = [
    X86Flag::Carry,
    X86Flag::Parity,
    X86Flag::Auxiliary,
    X86Flag::Zero,
    X86Flag::Sign,
    X86Flag::Overflow,
    X86Flag::Direction,
];

/// The 16 GPRs in encoding order; also the bit order of [`RegMask::gprs`](crate::core::bridge::RegMask).
pub const GPR_ORDER: [Register; 16] = [
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

/// Names of [`GPR_ORDER`], for alias-region descriptions.
pub const GPR_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// Names of [`FLAG_ORDER`].
pub const FLAG_NAMES: [&str; 7] = ["cf", "pf", "af", "zf", "sf", "of", "df"];

pub const XMM_NAMES: [&str; 16] = [
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
    "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

/// High halves of [`XMM_NAMES`]; scalar mode only.
pub const XMM_HI_NAMES: [&str; 16] = [
    "xmm0.hi", "xmm1.hi", "xmm2.hi", "xmm3.hi", "xmm4.hi", "xmm5.hi", "xmm6.hi", "xmm7.hi",
    "xmm8.hi", "xmm9.hi", "xmm10.hi", "xmm11.hi", "xmm12.hi", "xmm13.hi", "xmm14.hi", "xmm15.hi",
];

/// Segment selectors in encoding order (`ES`=0 ... `GS`=5).
pub const SEGMENT_ORDER: [Register; 6] = [
    Register::ES,
    Register::CS,
    Register::SS,
    Register::DS,
    Register::FS,
    Register::GS,
];

/// Names of [`SEGMENT_ORDER`].
pub const SEGMENT_NAMES: [&str; 6] = ["es", "cs", "ss", "ds", "fs", "gs"];

pub fn is_segment_register(reg: Register) -> bool {
    SEGMENT_ORDER.contains(&reg)
}

/// [`GPR_ORDER`] index of `reg`'s 64-bit form; sub-registers map to their containing GPR.
pub fn gpr_index(reg: Register) -> Option<u8> {
    let full = reg.full_register();
    GPR_ORDER.iter().position(|&r| r == full).map(|i| i as u8)
}

/// Mapping of x86 registers to Cranelift variables. Written-register tracking lives on the engine
/// ([`LiftEngine::mark_written`](crate::core::LiftEngine::mark_written)).
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
    pub context_ptr: Variable,
    pub image_base: Variable,
    /// XMM low halves; scalar mode only.
    pub xmms_lo: HashMap<u8, Variable>,
    /// XMM high halves; scalar mode only.
    pub xmms_hi: HashMap<u8, Variable>,
    /// XMMs as `I64X2`; vector mode only.
    pub xmms_vec: HashMap<u8, Variable>,
    pub use_vector_xmm: bool,
    pub fs_base: Variable,
    pub gs_base: Variable,
}

impl RegisterState {
    pub fn new(builder: &mut FunctionBuilder, use_vector_xmm: bool) -> Self {
        let mut gprs = HashMap::new();

        for &reg in &GPR_ORDER {
            let var = builder.declare_var(types::I64);
            gprs.insert(reg, var);
        }

        let mut flags = HashMap::new();
        for &flag in &FLAG_ORDER {
            let var = builder.declare_var(types::I8);
            flags.insert(flag, var);
        }

        let context_ptr = builder.declare_var(types::I64); // Assume 64-bit pointer
        let image_base = builder.declare_var(types::I64);

        let mut xmms_lo = HashMap::new();
        let mut xmms_hi = HashMap::new();
        let mut xmms_vec = HashMap::new();
        if use_vector_xmm {
            for i in 0u8..16 {
                xmms_vec.insert(i, builder.declare_var(types::I64X2));
            }
        } else {
            for i in 0u8..16 {
                xmms_lo.insert(i, builder.declare_var(types::I64));
                xmms_hi.insert(i, builder.declare_var(types::I64));
            }
        }

        let fs_base = builder.declare_var(types::I64);
        let gs_base = builder.declare_var(types::I64);

        Self {
            gprs,
            flags,
            context_ptr,
            image_base,
            xmms_lo,
            xmms_hi,
            xmms_vec,
            use_vector_xmm,
            fs_base,
            gs_base,
        }
    }

    pub fn from_config(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self {
        Self::new(builder, config.use_vector_xmm)
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
            _ => Err(MicroliftError::UnsupportedRegister(format!("{reg:?}"))),
        }
    }

    pub fn get_flag_offset(flag: X86Flag) -> crate::Result<i32> {
        match flag {
            X86Flag::Carry => Ok(std::mem::offset_of!(RegisterContext, cf) as i32),
            X86Flag::Parity => Ok(std::mem::offset_of!(RegisterContext, pf_src) as i32),
            X86Flag::Auxiliary => Ok(std::mem::offset_of!(RegisterContext, af_src) as i32),
            X86Flag::Zero => Ok(std::mem::offset_of!(RegisterContext, zf) as i32),
            X86Flag::Sign => Ok(std::mem::offset_of!(RegisterContext, sf) as i32),
            X86Flag::Overflow => Ok(std::mem::offset_of!(RegisterContext, of) as i32),
            X86Flag::Direction => Ok(std::mem::offset_of!(RegisterContext, df) as i32),
        }
    }

    /// Offset of the low/high 64-bit half of XMM `idx`.
    pub fn get_xmm_offset(idx: u8, hi: bool) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16 + if hi { 8 } else { 0 }
    }

    pub fn get_xmm_vec_offset(idx: u8) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16
    }

    /// Offset of a segment selector. It has no `Variable`: selectors are read and written in place.
    pub fn get_segment_offset(reg: Register) -> crate::Result<i32> {
        match reg {
            Register::CS => Ok(std::mem::offset_of!(RegisterContext, cs) as i32),
            Register::SS => Ok(std::mem::offset_of!(RegisterContext, ss) as i32),
            Register::DS => Ok(std::mem::offset_of!(RegisterContext, ds) as i32),
            Register::ES => Ok(std::mem::offset_of!(RegisterContext, es) as i32),
            Register::FS => Ok(std::mem::offset_of!(RegisterContext, fs) as i32),
            Register::GS => Ok(std::mem::offset_of!(RegisterContext, gs) as i32),
            _ => Err(MicroliftError::UnsupportedRegister(format!("{reg:?}"))),
        }
    }

    pub fn get_fs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, fs_base) as i32
    }

    pub fn get_gs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, gs_base) as i32
    }
}
