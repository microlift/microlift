use thiserror::Error;

pub use function_traits;

// Own Cranelift fork: use these re-exports, not a second crate copy.
pub use cranelift;
pub use cranelift_codegen;
pub use cranelift_frontend;
pub use cranelift_jit;
pub use cranelift_module;
pub use cranelift_native;
pub use cranelift_object;
pub use target_lexicon;

pub mod arch;
pub mod backend;
pub mod cfg;
pub mod context;
pub mod core;
pub mod lifter;
pub mod os;
#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(feature = "testkit")]
pub mod testkit;
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
