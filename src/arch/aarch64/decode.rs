//! aarch64 decoding (via yaxpeax-arm) and static control-flow classification.
//!
//! A64 instructions are a fixed 4 bytes wide, which makes leader discovery and
//! IP tracking simple. yaxpeax instructions do not carry their own address, so
//! each decoded instruction is wrapped together with its IP.

use crate::arch::ControlFlow;
use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv8::a64::{InstDecoder, Instruction as RawInst, Opcode, Operand};

/// A decoded A64 instruction tagged with its virtual address.
#[derive(Clone, Copy)]
pub struct A64Instr {
    pub ip: u64,
    pub inst: RawInst,
    /// Whether decoding succeeded. Invalid 4-byte words are retained so phase 2
    /// can emit a fault stub.
    pub valid: bool,
}

/// Decode every 4-byte word in `bytes` starting at virtual address `ip`.
pub fn decode_region(bytes: &[u8], ip: u64) -> Vec<A64Instr> {
    let decoder = InstDecoder::default();
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let word = &bytes[offset..offset + 4];
        let cur_ip = ip + offset as u64;
        let mut reader = U8Reader::new(word);
        match decoder.decode(&mut reader) {
            Ok(inst) => out.push(A64Instr {
                ip: cur_ip,
                inst,
                valid: true,
            }),
            Err(_) => out.push(A64Instr {
                ip: cur_ip,
                inst: RawInst {
                    opcode: Opcode::Invalid,
                    operands: [Operand::Nothing; 4],
                },
                valid: false,
            }),
        }
        offset += 4;
    }
    out
}

/// Decode the single 4-byte instruction at the start of `bytes` (first byte at
/// `ip`). Returns `None` if fewer than 4 bytes are available; an undecodable
/// word yields an `A64Instr` with `valid == false`.
pub fn decode_one(bytes: &[u8], ip: u64) -> Option<A64Instr> {
    if bytes.len() < 4 {
        return None;
    }
    decode_region(&bytes[..4], ip).into_iter().next()
}

/// Find the first PC-relative offset operand and resolve it against `ip`.
fn branch_target(i: &A64Instr) -> Option<u64> {
    for op in &i.inst.operands {
        if let Operand::PCOffset(off) = op {
            return Some(i.ip.wrapping_add(*off as u64));
        }
    }
    None
}

/// Classify an instruction's static control-flow facts for leader discovery.
pub fn control_flow(i: &A64Instr) -> ControlFlow {
    if !i.valid {
        return ControlFlow::invalid();
    }

    match i.inst.opcode {
        Opcode::B => branch_target(i)
            .map(ControlFlow::direct_branch)
            .unwrap_or_else(ControlFlow::indirect_branch),
        Opcode::Bcc(_) | Opcode::CBZ | Opcode::CBNZ | Opcode::TBZ | Opcode::TBNZ => {
            branch_target(i)
                .map(ControlFlow::cond_branch)
                .unwrap_or_else(ControlFlow::fallthrough)
        }
        Opcode::BL => ControlFlow::call(branch_target(i)),
        Opcode::BLR => ControlFlow::call(None),
        Opcode::BR => ControlFlow::indirect_branch(),
        Opcode::RET => ControlFlow::ret(),
        Opcode::Invalid => ControlFlow::invalid(),
        _ => ControlFlow::fallthrough(),
    }
}

#[inline]
pub fn instr_ip(i: &A64Instr) -> u64 {
    i.ip
}

#[inline]
pub fn instr_next_ip(i: &A64Instr) -> u64 {
    i.ip + 4
}

#[inline]
pub fn is_invalid(i: &A64Instr) -> bool {
    !i.valid
}
