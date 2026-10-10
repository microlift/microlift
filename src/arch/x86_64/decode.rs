//! x86-64 decoding (iced-x86) and static control-flow classification for leader discovery.

use crate::arch::{ControlFlow, InstructionEffects};
use iced_x86::{
    Decoder, DecoderOptions, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind,
    Register, RflagsBits,
};

/// Decodes every instruction in `bytes` from `ip`; invalid ones are kept so phase 2 emits a fault stub.
pub fn decode_region(bytes: &[u8], ip: u64) -> Vec<Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        let mut instruction = Instruction::default();
        decoder.decode_out(&mut instruction);
        instructions.push(instruction);
    }
    instructions
}

/// `None` only for an empty slice; a truncated or undecodable encoding is [`is_invalid`].
pub fn decode_one(bytes: &[u8], ip: u64) -> Option<Instruction> {
    if bytes.is_empty() {
        return None;
    }
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut instruction = Instruction::default();
    decoder.decode_out(&mut instruction);
    Some(instruction)
}

/// Control-flow facts for leader discovery: listed `Jmp`/`Jcc`/`Loop` contribute `near_branch_target()`
/// (may be 0 for non-near forms), direct `CALL`s their callee, all else nothing.
pub fn control_flow(instruction: &Instruction) -> ControlFlow {
    if instruction.is_invalid() {
        return ControlFlow::invalid();
    }

    match instruction.mnemonic() {
        Mnemonic::Jmp => {
            // Only near-branch jumps have a static target; indirect ones would report a bogus 0.
            if matches!(
                instruction.op0_kind(),
                OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
            ) {
                ControlFlow::direct_branch(instruction.near_branch_target())
            } else {
                ControlFlow::indirect_branch()
            }
        }
        Mnemonic::Je
        | Mnemonic::Jne
        | Mnemonic::Jb
        | Mnemonic::Jbe
        | Mnemonic::Ja
        | Mnemonic::Jae
        | Mnemonic::Jg
        | Mnemonic::Jge
        | Mnemonic::Jl
        | Mnemonic::Jle
        | Mnemonic::Js
        | Mnemonic::Jns
        | Mnemonic::Jo
        | Mnemonic::Jno
        | Mnemonic::Jp
        | Mnemonic::Jnp
        | Mnemonic::Jrcxz
        | Mnemonic::Loop
        | Mnemonic::Loope
        | Mnemonic::Loopne => ControlFlow::cond_branch(instruction.near_branch_target()),
        Mnemonic::Call => {
            // Only direct calls have a static target.
            if matches!(
                instruction.op0_kind(),
                OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
            ) {
                ControlFlow::call(Some(instruction.near_branch_target()))
            } else {
                ControlFlow::call(None)
            }
        }
        Mnemonic::Ret | Mnemonic::Retf => ControlFlow::ret(),
        _ => ControlFlow::fallthrough(),
    }
}

#[inline]
pub fn instr_ip(i: &Instruction) -> u64 {
    i.ip()
}

#[inline]
pub fn instr_next_ip(i: &Instruction) -> u64 {
    i.next_ip()
}

#[inline]
pub fn is_invalid(i: &Instruction) -> bool {
    i.is_invalid()
}

/// Register/memory effects from iced's instruction-info database.
pub fn instruction_effects(i: &Instruction) -> InstructionEffects {
    let mut factory = InstructionInfoFactory::new();
    let info = factory.info(i);

    let mut reads: Vec<String> = Vec::new();
    let mut writes: Vec<String> = Vec::new();

    for used in info.used_registers() {
        let reg = used.register();
        if reg == Register::None {
            continue;
        }
        let name = format!("{reg:?}").to_ascii_lowercase();
        match used.access() {
            OpAccess::Read | OpAccess::CondRead => push_unique(&mut reads, &name),
            OpAccess::Write | OpAccess::CondWrite => push_unique(&mut writes, &name),
            OpAccess::ReadWrite | OpAccess::ReadCondWrite => {
                push_unique(&mut reads, &name);
                push_unique(&mut writes, &name);
            }
            OpAccess::None | OpAccess::NoMemAccess => {}
        }
    }

    // Status flags are folded in as pseudo-registers (`cf`, `zf`, ...).
    for (bit, name) in RFLAG_NAMES {
        if i.rflags_read() & bit != 0 {
            push_unique(&mut reads, name);
        }
        if (i.rflags_written() | i.rflags_cleared() | i.rflags_set()) & bit != 0 {
            push_unique(&mut writes, name);
        }
    }

    let mut reads_memory = false;
    let mut writes_memory = false;
    for mem in info.used_memory() {
        match mem.access() {
            OpAccess::Read | OpAccess::CondRead => reads_memory = true,
            OpAccess::Write | OpAccess::CondWrite => writes_memory = true,
            OpAccess::ReadWrite | OpAccess::ReadCondWrite => {
                reads_memory = true;
                writes_memory = true;
            }
            OpAccess::None | OpAccess::NoMemAccess => {}
        }
    }

    InstructionEffects {
        reads,
        writes,
        reads_memory,
        writes_memory,
    }
}

fn push_unique(v: &mut Vec<String>, name: &str) {
    if !v.iter().any(|n| n == name) {
        v.push(name.to_string());
    }
}

/// Status-flag bit -> lower-case name, for [`InstructionEffects`].
const RFLAG_NAMES: [(u32, &str); 8] = [
    (RflagsBits::CF, "cf"),
    (RflagsBits::PF, "pf"),
    (RflagsBits::AF, "af"),
    (RflagsBits::ZF, "zf"),
    (RflagsBits::SF, "sf"),
    (RflagsBits::OF, "of"),
    (RflagsBits::DF, "df"),
    (RflagsBits::IF, "if"),
];
