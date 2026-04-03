use thiserror::Error;

pub mod backend;
pub mod context;
pub mod lifter;
pub mod semantics;

#[derive(Error, Debug)]
pub enum MicroliftError {
    #[error("failed to decode instruction at {offset}: {reason}")]
    DecodeError { offset: u64, reason: String },
    
    #[error("unsupported instruction: {mnemonic}")]
    UnsupportedInstruction { mnemonic: String },
    
    #[error("cranelift error: {0}")]
    CraneliftError(String),
}

pub type Result<T> = std::result::Result<T, MicroliftError>;
