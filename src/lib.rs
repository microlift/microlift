use iced_x86::Register;
use thiserror::Error;

pub mod backend;
pub mod context;
pub mod lifter;
pub mod os;
pub mod semantics;
pub mod vsa;

#[derive(Error, Debug)]
pub enum MicroliftError {
    #[error("failed to decode instruction at {offset}: {reason}")]
    DecodeError { offset: u64, reason: String },

    #[error("unsupported instruction: {mnemonic}")]
    UnsupportedInstruction { mnemonic: String },

    #[error("cranelift error: {0}")]
    CraneliftError(String),

    #[error("unsupported register: {0:?}")]
    UnsupportedRegister(Register),
}

pub type Result<T> = std::result::Result<T, MicroliftError>;
