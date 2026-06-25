use thiserror::Error;

pub use function_traits;

pub mod arch;
pub mod backend;
pub mod cfg;
pub mod context;
pub mod core;
pub mod lifter;
pub(crate) mod mapfile;
pub mod os;
pub mod vsa;

#[derive(Error, Debug)]
pub enum MicroliftError {
    #[error("failed to decode instruction at {offset}: {reason}")]
    DecodeError { offset: u64, reason: String },

    #[error("unsupported instruction: {mnemonic}")]
    UnsupportedInstruction { mnemonic: String },

    #[error("cranelift error: {0}")]
    CraneliftError(String),

    #[error("unsupported register: {0}")]
    UnsupportedRegister(String),
}

pub type Result<T> = std::result::Result<T, MicroliftError>;
