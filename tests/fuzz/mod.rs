//! Shared building blocks for the differential fuzzing suite.
//!
//! Included per-binary via `#[path = "fuzz/mod.rs"] mod fuzz;` from each
//! `fuzz_*.rs` test target (all gated on the `fuzzing` feature). Holds:
//!
//! - **encoders** - tiny x86-64 snippet builders with known operands,
//! - **reference models** - pure-Rust semantics oracles for the diffable ops,
//! - **state comparison** - a config-invariant fingerprint of a
//!   `RegisterContext` (notably excluding `rsp`, which is a fresh host stack
//!   address on every harness and so legitimately differs run-to-run).

#![allow(dead_code)] // each test binary uses a different subset.

use microlift::testkit::prelude::*;

/// The five diffable 64-bit reg/reg ALU ops we fuzz, with their `/r` opcode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
}

impl AluOp {
    pub const ALL: [AluOp; 5] = [AluOp::Add, AluOp::Sub, AluOp::And, AluOp::Or, AluOp::Xor];

    fn opcode(self) -> u8 {
        match self {
            AluOp::Add => 0x01,
            AluOp::Sub => 0x29,
            AluOp::And => 0x21,
            AluOp::Or => 0x09,
            AluOp::Xor => 0x31,
        }
    }
}

/// Expected architectural result of an ALU op: the destination value plus the
/// six status flags the lifter defines for these instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AluExpect {
    pub result: u64,
    pub cf: u8,
    pub pf: u8,
    pub af: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
}

/// Even-parity of the low byte (x86 `PF`).
fn parity(v: u64) -> u8 {
    ((v as u8).count_ones() % 2 == 0) as u8
}

fn zsp(result: u64) -> (u8, u8, u8) {
    (
        (result == 0) as u8,        // ZF
        ((result >> 63) & 1) as u8, // SF
        parity(result),             // PF
    )
}

/// Pure-Rust reference model for `op rax, rcx` with `rax=a`, `rcx=b`.
pub fn alu_reference(op: AluOp, a: u64, b: u64) -> AluExpect {
    match op {
        AluOp::Add => {
            let result = a.wrapping_add(b);
            let (zf, sf, pf) = zsp(result);
            AluExpect {
                result,
                cf: (result < a) as u8,
                pf,
                af: (((a ^ b ^ result) >> 4) & 1) as u8,
                zf,
                sf,
                of: ((((a ^ result) & (b ^ result)) >> 63) & 1) as u8,
            }
        }
        AluOp::Sub => {
            let result = a.wrapping_sub(b);
            let (zf, sf, pf) = zsp(result);
            AluExpect {
                result,
                cf: (a < b) as u8,
                pf,
                af: (((a ^ b ^ result) >> 4) & 1) as u8,
                zf,
                sf,
                of: ((((a ^ b) & (a ^ result)) >> 63) & 1) as u8,
            }
        }
        AluOp::And | AluOp::Or | AluOp::Xor => {
            let result = match op {
                AluOp::And => a & b,
                AluOp::Or => a | b,
                AluOp::Xor => a ^ b,
                _ => unreachable!(),
            };
            let (zf, sf, pf) = zsp(result);
            // Logical ops clear CF and OF; AF is architecturally undefined.
            AluExpect {
                result,
                cf: 0,
                pf,
                af: 0,
                zf,
                sf,
                of: 0,
            }
        }
    }
}

/// Emit `mov rax, a ; mov rcx, b`.
fn emit_load_ax_cx(a: u64, b: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    out.extend_from_slice(&a.to_le_bytes());
    out.extend_from_slice(&[0x48, 0xB9]); // mov rcx, imm64
    out.extend_from_slice(&b.to_le_bytes());
}

/// Build `mov rax,a ; mov rcx,b ; <op> rax,rcx ; ret`.
pub fn alu_snippet(op: AluOp, a: u64, b: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24);
    emit_load_ax_cx(a, b, &mut bytes);
    bytes.extend_from_slice(&[0x48, op.opcode(), 0xC8]); // op rax, rcx
    bytes.push(0xC3); // ret
    bytes
}

/// A config-invariant fingerprint of the post-run context: everything a pure
/// register/flag computation can affect, **excluding `rsp`** (a fresh host
/// stack pointer each run) and `rip` (trampoline bookkeeping).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateFingerprint {
    pub gpr: [u64; 15], // rax..r15 minus rsp
    pub flags: [u8; 7], // cf,pf,af,zf,sf,of,df
    pub xmm: [[u64; 2]; 16],
    pub exit: u32,
}

pub fn fingerprint(ctx: &RegisterContext) -> StateFingerprint {
    StateFingerprint {
        gpr: [
            ctx.rax, ctx.rcx, ctx.rdx, ctx.rbx, ctx.rbp, ctx.rsi, ctx.rdi, ctx.r8, ctx.r9, ctx.r10,
            ctx.r11, ctx.r12, ctx.r13, ctx.r14, ctx.r15,
        ],
        flags: [
            ctx.cf,
            u8::from(ctx.pf()),
            u8::from(ctx.af()),
            ctx.zf,
            ctx.sf,
            ctx.of,
            ctx.df,
        ],
        xmm: *ctx.xmm,
        exit: ctx.exit_reason,
    }
}
