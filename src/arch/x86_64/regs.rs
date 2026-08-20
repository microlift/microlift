//! x86-64 register file: SSA-variable mapping, flag set, dirty tracking, and
//! `RegisterContext` field offsets.

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
    /// Direction flag - controls string-operation pointer direction.
    Direction,
}

/// The seven modelled condition flags, in a fixed order.
///
/// Iterating this rather than `RegisterState::flags` is what makes the context
/// sync deterministic: a `HashMap` walk emits the same stores in a different
/// order under every process, which changes x86 displacement encodings and so
/// changes `.text` between two lifts of identical bytes.
pub const FLAG_ORDER: [X86Flag; 7] = [
    X86Flag::Carry,
    X86Flag::Parity,
    X86Flag::Auxiliary,
    X86Flag::Zero,
    X86Flag::Sign,
    X86Flag::Overflow,
    X86Flag::Direction,
];

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

/// [`GPR_ORDER`]'s names, for the alias-region descriptions that appear in a
/// `.clif` dump.
pub const GPR_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// [`FLAG_ORDER`]'s names, for the same reason.
pub const FLAG_NAMES: [&str; 7] = ["cf", "pf", "af", "zf", "sf", "of", "df"];

/// XMM register names, and the names of their high halves in scalar mode.
pub const XMM_NAMES: [&str; 16] = [
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
    "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

/// High halves of [`XMM_NAMES`], used only when `use_vector_xmm` is off.
pub const XMM_HI_NAMES: [&str; 16] = [
    "xmm0.hi", "xmm1.hi", "xmm2.hi", "xmm3.hi", "xmm4.hi", "xmm5.hi", "xmm6.hi", "xmm7.hi",
    "xmm8.hi", "xmm9.hi", "xmm10.hi", "xmm11.hi", "xmm12.hi", "xmm13.hi", "xmm14.hi", "xmm15.hi",
];

/// The six segment-selector registers, in architectural encoding order
/// (`ES`=0, `CS`=1, `SS`=2, `DS`=3, `FS`=4, `GS`=5).
pub const SEGMENT_ORDER: [Register; 6] = [
    Register::ES,
    Register::CS,
    Register::SS,
    Register::DS,
    Register::FS,
    Register::GS,
];

/// [`SEGMENT_ORDER`]'s names, for the alias-region descriptions in a `.clif`
/// dump.
pub const SEGMENT_NAMES: [&str; 6] = ["es", "cs", "ss", "ds", "fs", "gs"];

/// Whether `reg` is one of the six segment-selector registers.
pub fn is_segment_register(reg: Register) -> bool {
    SEGMENT_ORDER.contains(&reg)
}

/// The [`GPR_ORDER`] index of `reg`'s full 64-bit form, or `None` if `reg` is
/// not a general-purpose register. Sub-registers map to their containing GPR
/// (`EAX`, `AX`, `AL` and `AH` all give `RAX`'s index).
pub fn gpr_index(reg: Register) -> Option<u8> {
    let full = reg.full_register();
    GPR_ORDER.iter().position(|&r| r == full).map(|i| i as u8)
}

/// Mapping of x86 registers to Cranelift variables.
///
/// Which of them a region actually *wrote* is not tracked here: it is a
/// property of the lift rather than of the register file, and it lives on the
/// engine as [`LiftEngine::mark_written`](crate::core::LiftEngine::mark_written)
/// - see [`crate::core::regsync`].
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

    /// Byte offset of a segment *selector* within `RegisterContext`.
    ///
    /// Unlike every other register accessor here this has no matching `Variable`:
    /// the selectors are deliberately not part of the SSA register file and are
    /// loaded/stored in place at each use. A guest reads them roughly once, into a
    /// diagnostic struct and never into control flow, so an SSA variable would buy
    /// nothing and cost six more values live across every block reachable from the
    /// entry dispatch.
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
