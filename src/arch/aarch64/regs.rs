//! aarch64 register file: SSA-variable mapping, NZCV flags, dirty tracking, context offsets.
//!
//! V registers are not yet SSA variables (semantics are integer-only).

use crate::arch::aarch64::context::Aarch64RegisterContext;
use crate::core::LifterConfig;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, Variable};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nzcv {
    N,
    Z,
    C,
    V,
}

/// X0-X30 in order; iterate this, not `Aarch64RegState::gprs`, for deterministic sync.
pub const GPR_ORDER: [u16; 31] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
];

/// Names of [`GPR_ORDER`], for alias-region descriptions.
pub const GPR_NAMES: [&str; 31] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30",
];

/// Names of [`FLAG_ORDER`].
pub const FLAG_NAMES: [&str; 4] = ["n", "z", "c", "v"];

/// Fixed order: `HashMap` iteration would change `.text` between lifts of identical bytes.
pub const FLAG_ORDER: [Nzcv; 4] = [Nzcv::N, Nzcv::Z, Nzcv::C, Nzcv::V];

/// XZR/WZR index; reads yield 0, writes are discarded.
pub const ZR: u16 = 31;

pub struct Aarch64RegState {
    /// X0-X30; XZR is not stored.
    pub gprs: HashMap<u16, Variable>,
    pub sp: Variable,
    pub flags: HashMap<Nzcv, Variable>,
    pub tpidr_el0: Variable,
    pub context_ptr: Variable,
    pub image_base: Variable,
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
        }
    }

    pub fn from_config(builder: &mut FunctionBuilder, _config: &LifterConfig) -> Self {
        Self::new(builder)
    }

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
