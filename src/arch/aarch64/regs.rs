//! aarch64 register file: SSA-variable mapping, NZCV flags, dirty tracking, and
//! `Aarch64RegisterContext` field offsets.
//!
//! SIMD/FP (V) registers are present in the context struct but are not yet
//! materialised as SSA variables - the current semantics slice is
//! integer-only. They will be added alongside FP/SIMD instruction support.

use crate::arch::aarch64::context::Aarch64RegisterContext;
use crate::core::LifterConfig;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use std::collections::HashMap;
use std::collections::HashSet;

/// NZCV condition flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nzcv {
    /// Negative.
    N,
    /// Zero.
    Z,
    /// Carry.
    C,
    /// Overflow.
    V,
}

/// The zero-register index (XZR/WZR). Reads yield 0; writes are discarded.
pub const ZR: u16 = 31;

/// Mapping of aarch64 registers to Cranelift variables, plus dirty tracking.
pub struct Aarch64RegState {
    /// X0–X30 (indices 0–30) as I64 SSA variables. X31 (XZR) is not stored.
    pub gprs: HashMap<u16, Variable>,
    /// Stack pointer.
    pub sp: Variable,
    /// NZCV flags, each an I8 SSA variable.
    pub flags: HashMap<Nzcv, Variable>,
    /// Thread-pointer base (TPIDR_EL0).
    pub tpidr_el0: Variable,
    pub context_ptr: Variable,
    pub image_base: Variable,

    // --- dirty-bit tracking ---
    pub dirty_gprs: HashSet<u16>,
    pub dirty_sp: bool,
    pub dirty_flags: HashSet<Nzcv>,
    pub dirty_tpidr: bool,
}

impl Aarch64RegState {
    pub fn new(builder: &mut FunctionBuilder) -> Self {
        let mut gprs = HashMap::new();
        for i in 0u16..31 {
            gprs.insert(i, builder.declare_var(types::I64));
        }

        let sp = builder.declare_var(types::I64);

        let mut flags = HashMap::new();
        for &flag in &[Nzcv::N, Nzcv::Z, Nzcv::C, Nzcv::V] {
            flags.insert(flag, builder.declare_var(types::I8));
        }

        let tpidr_el0 = builder.declare_var(types::I64);
        let context_ptr = builder.declare_var(types::I64);
        let image_base = builder.declare_var(types::I64);

        Self {
            gprs,
            sp,
            flags,
            tpidr_el0,
            context_ptr,
            image_base,
            dirty_gprs: HashSet::new(),
            dirty_sp: false,
            dirty_flags: HashSet::new(),
            dirty_tpidr: false,
        }
    }

    pub fn from_config(builder: &mut FunctionBuilder, _config: &LifterConfig) -> Self {
        Self::new(builder)
    }

    /// Byte offset of GPR `reg` (0–30) within `Aarch64RegisterContext`.
    pub fn get_gpr_offset(reg: u16) -> i32 {
        let base = std::mem::offset_of!(Aarch64RegisterContext, x) as i32;
        base + (reg as i32) * 8
    }

    pub fn get_sp_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, sp) as i32
    }

    pub fn get_flag_offset(flag: Nzcv) -> i32 {
        match flag {
            Nzcv::N => std::mem::offset_of!(Aarch64RegisterContext, n) as i32,
            Nzcv::Z => std::mem::offset_of!(Aarch64RegisterContext, z) as i32,
            Nzcv::C => std::mem::offset_of!(Aarch64RegisterContext, c) as i32,
            Nzcv::V => std::mem::offset_of!(Aarch64RegisterContext, vf) as i32,
        }
    }

    pub fn get_tpidr_offset() -> i32 {
        std::mem::offset_of!(Aarch64RegisterContext, tpidr_el0) as i32
    }
}
