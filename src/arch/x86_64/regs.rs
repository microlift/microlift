//! x86-64 register file: SSA-variable mapping, flag set, dirty tracking, and
//! `RegisterContext` field offsets.

use crate::MicroliftError;
use crate::arch::x86_64::context::RegisterContext;
use crate::core::LifterConfig;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use iced_x86::Register;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Flag {
    Carry,
    Parity,
    Auxiliary,
    Zero,
    Sign,
    Overflow,
    /// Direction flag - controls string-operation pointer direction.
    Direction,
}

/// The 16 general-purpose registers in architectural encoding order.
///
/// This is also the bit order of [`RegMask::gprs`](crate::core::bridge::RegMask)
/// for this guest: bit `i` selects `GPR_ORDER[i]`.
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

/// The [`GPR_ORDER`] index of `reg`'s full 64-bit form, or `None` if `reg` is
/// not a general-purpose register. Sub-registers map to their containing GPR
/// (`EAX`, `AX`, `AL` and `AH` all give `RAX`'s index).
pub fn gpr_index(reg: Register) -> Option<u8> {
    let full = reg.full_register();
    GPR_ORDER.iter().position(|&r| r == full).map(|i| i as u8)
}

/// Mapping of x86 registers to Cranelift variables, plus dirty-bit tracking.
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
    pub context_ptr: Variable,
    pub image_base: Variable,
    /// XMM0–XMM15 low 64 bits. Populated when `use_vector_xmm` is `false`.
    pub xmms_lo: HashMap<u8, Variable>,
    /// XMM0–XMM15 high 64 bits. Populated when `use_vector_xmm` is `false`.
    pub xmms_hi: HashMap<u8, Variable>,
    /// XMM0–XMM15 as a single `I64X2` variable. Populated when `use_vector_xmm`
    /// is `true`.
    pub xmms_vec: HashMap<u8, Variable>,
    /// Selects between the scalar-pair and native vector representations.
    pub use_vector_xmm: bool,
    /// FS segment base address SSA variable.
    pub fs_base: Variable,
    /// GS segment base address SSA variable.
    pub gs_base: Variable,

    // --- dirty-bit tracking (written since the last context reload) ---
    pub dirty_gprs: HashSet<Register>,
    pub dirty_flags: HashSet<X86Flag>,
    pub dirty_xmms: HashSet<u8>,
    pub dirty_fs_base: bool,
    pub dirty_gs_base: bool,
}

impl RegisterState {
    pub fn new(builder: &mut FunctionBuilder, use_vector_xmm: bool) -> Self {
        let mut gprs = HashMap::new();

        for &reg in &GPR_ORDER {
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
            X86Flag::Direction,
        ];

        for &flag in &flag_types {
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
            dirty_gprs: HashSet::new(),
            dirty_flags: HashSet::new(),
            dirty_xmms: HashSet::new(),
            dirty_fs_base: false,
            dirty_gs_base: false,
        }
    }

    /// Build the register state from a [`LifterConfig`].
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
            X86Flag::Parity => Ok(std::mem::offset_of!(RegisterContext, pf) as i32),
            X86Flag::Auxiliary => Ok(std::mem::offset_of!(RegisterContext, af) as i32),
            X86Flag::Zero => Ok(std::mem::offset_of!(RegisterContext, zf) as i32),
            X86Flag::Sign => Ok(std::mem::offset_of!(RegisterContext, sf) as i32),
            X86Flag::Overflow => Ok(std::mem::offset_of!(RegisterContext, of) as i32),
            X86Flag::Direction => Ok(std::mem::offset_of!(RegisterContext, df) as i32),
        }
    }

    /// Byte offset of the low (`hi=false`) or high (`hi=true`) 64-bit half of
    /// XMM register `idx` (0–15) within `RegisterContext`.
    pub fn get_xmm_offset(idx: u8, hi: bool) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16 + if hi { 8 } else { 0 }
    }

    /// Byte offset of the whole 128-bit XMM register `idx` (0–15).
    pub fn get_xmm_vec_offset(idx: u8) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16
    }

    pub fn get_fs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, fs_base) as i32
    }

    pub fn get_gs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, gs_base) as i32
    }
}
