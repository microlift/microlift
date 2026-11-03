//! Host references a lift makes by name rather than by address.
//!
//! Lifted code never bakes a host pointer: a fast import, a bridge, a bridge slot and
//! the guest slide are relocations resolved when the code is defined (JIT) or loaded
//! (an [`crate::backend::ImageBuilder`] artifact), so both paths emit the same code.

use std::sync::Arc;

use cranelift_codegen::ir::Signature;

/// What a [`HostSymbol`] names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HostSymbolKind {
    /// Called with the host signature, rendered by Cranelift (`(i64) -> i32 apple_aarch64`).
    Function(Arc<str>),
    /// Addressed, never called.
    Data,
    /// A Cranelift runtime routine (`ceilf`, `memcpy`, ...) the backend chose to call.
    LibCall,
}

/// A host entity lifted code references, keyed by who provides it, its name and its ABI.
///
/// The name alone is not an identity: under a two-level namespace two libraries can
/// both export `_foo`, and a site bound to each must call its own.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostSymbol {
    /// The library that provides it; [`RUNTIME_PROVIDER`] for the embedding runtime itself.
    pub provider: Arc<str>,
    pub name: Arc<str>,
    pub kind: HostSymbolKind,
}

/// Provider of symbols the embedding runtime defines (bridges, bridge slots).
pub const RUNTIME_PROVIDER: &str = "";

/// Provider of [`HostSymbolKind::LibCall`] symbols.
pub const LIBCALL_PROVIDER: &str = "cranelift";

impl HostSymbol {
    pub fn function(provider: &str, name: &str, signature: &Signature) -> Self {
        Self {
            provider: provider.into(),
            name: name.into(),
            kind: HostSymbolKind::Function(signature.to_string().into()),
        }
    }

    pub fn data(provider: &str, name: &str) -> Self {
        Self {
            provider: provider.into(),
            name: name.into(),
            kind: HostSymbolKind::Data,
        }
    }

    pub fn libcall(name: &str) -> Self {
        Self {
            provider: LIBCALL_PROVIDER.into(),
            name: name.into(),
            kind: HostSymbolKind::LibCall,
        }
    }
}

impl std::fmt::Display for HostSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            HostSymbolKind::Function(sig) => write!(f, "{}`{} {sig}", self.provider, self.name),
            HostSymbolKind::Data => write!(f, "{}`{} (data)", self.provider, self.name),
            HostSymbolKind::LibCall => write!(f, "{} (libcall)", self.name),
        }
    }
}

/// A host function the lift calls: `ExternalName::user(0, i)` for `LiftedCode::imports[i]`.
#[derive(Debug, Clone)]
pub struct HostImport {
    pub symbol: HostSymbol,
    /// What the JIT resolves `symbol` to; never part of the code.
    pub address: usize,
    pub signature: Signature,
}

/// What a data relocation names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DataTarget {
    Host(HostSymbol),
    /// The run-time minus lift-time distance of the lifted image
    /// ([`AddressMode::Relocated`](crate::core::AddressMode::Relocated)).
    GuestSlide,
}

/// A data address the lift references, after its link cells and constant tables in
/// namespace 1.
#[derive(Debug, Clone)]
pub struct DataImport {
    pub target: DataTarget,
    /// What the JIT resolves `target` to; never part of the code.
    pub address: u64,
}

/// What the import lookup answered for one address at lift time.
///
/// Recorded per address the lift asked about, so a cached region can be checked
/// against the answers a later run would give.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImportDecision {
    NotImport,
    Slow,
    Fast {
        provider: String,
        name: String,
        abi: String,
    },
    Bridge {
        name: String,
    },
}

impl ImportDecision {
    /// The decision an import lookup answering `kind` amounts to.
    pub fn of(kind: Option<&crate::core::bridge::ImportKind>) -> Self {
        use crate::core::bridge::ImportKind;
        match kind {
            None => Self::NotImport,
            Some(ImportKind::Slow) => Self::Slow,
            Some(ImportKind::Fast(import)) => Self::Fast {
                provider: import.provider.to_string(),
                name: import.symbol.name.to_string(),
                abi: format!("{:?} {}", import.symbol.arg_types, import.abi.name),
            },
            Some(ImportKind::Bridge(bridge)) => Self::Bridge {
                name: bridge.name.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{AbiParam, types};
    use cranelift_codegen::isa::CallConv;

    #[test]
    fn same_name_from_two_providers_is_two_symbols() {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64));
        let a = HostSymbol::function("libA.dylib", "_foo", &sig);
        let b = HostSymbol::function("libB.dylib", "_foo", &sig);
        assert_ne!(a, b);
        assert_eq!(a, HostSymbol::function("libA.dylib", "_foo", &sig));
    }

    #[test]
    fn same_name_with_two_abis_is_two_symbols() {
        let narrow = Signature::new(CallConv::SystemV);
        let mut wide = Signature::new(CallConv::SystemV);
        wide.returns.push(AbiParam::new(types::I64));
        assert_ne!(
            HostSymbol::function("libc", "_foo", &narrow),
            HostSymbol::function("libc", "_foo", &wide)
        );
    }
}
