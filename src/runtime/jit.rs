//! The JIT: a lift compiled position-independent and placed in the shared
//! [`CodeArena`], its GOT entries in front of it.
//!
//! The code is exactly what an image file holds; only where the GOT entries are
//! differs, and the entries hold the addresses the lift recorded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cranelift_codegen::isa::TargetIsa;

use super::CodeOwner;
use super::arena::CodeArena;
use crate::backend::CompiledRegion;
use crate::core::LiftedCode;
use crate::core::link::{BlockLinkEntry, ENTRY_SIZE, LinkSite};
use crate::core::srcloc::GuestCodeMap;
use crate::core::symbol::{DataTarget, HostSymbol, HostSymbolKind};
use crate::image::compile::PicCompiler;
use crate::image::{CodeObject, GotTarget, TEXT_ALIGN, link};

/// A region the JIT compiled, ready to publish.
pub struct JitRegion {
    pub region: CompiledRegion,
    /// Host offsets relative to [`CompiledRegion::link_target`].
    pub map: GuestCodeMap,
    pub resume_points: Vec<u64>,
    pub covered: Vec<std::ops::Range<u64>>,
    pub data_dependencies: Vec<std::ops::Range<u64>>,
    pub link_sites: Vec<LinkSite>,
    /// Bytes of code, GOT entries included.
    pub code_len: usize,
}

/// What keeps arena code mapped: nothing, since the arena never unmaps.
pub struct ArenaCode;

impl CodeOwner for ArenaCode {}

impl JitRegion {
    pub fn owner() -> Arc<dyn CodeOwner> {
        static OWNER: std::sync::LazyLock<Arc<ArenaCode>> =
            std::sync::LazyLock::new(|| Arc::new(ArenaCode));
        Arc::clone(&*OWNER) as Arc<dyn CodeOwner>
    }
}

/// Compiles and places lifts for one ISA.
pub struct JitCompiler {
    compiler: PicCompiler,
    arena: &'static CodeArena,
    /// Helpers and tables already placed, by name, with their addresses.
    placed: Mutex<HashMap<GotTarget, u64>>,
}

impl JitCompiler {
    /// `isa` must be position-independent; see [`PicCompiler::new`].
    pub fn new(isa: Arc<dyn TargetIsa>) -> crate::Result<Self> {
        Ok(Self {
            compiler: PicCompiler::new(isa)?,
            arena: CodeArena::global(),
            placed: Mutex::default(),
        })
    }

    pub fn isa(&self) -> &dyn TargetIsa {
        self.compiler.isa()
    }

    /// The compiler, for code that is written to an image file rather than placed.
    pub fn compiler(&self) -> &PicCompiler {
        &self.compiler
    }

    /// Compiles `code` into the arena. Its host references resolve to the addresses
    /// the lift recorded beside them, and libcalls to the host's own.
    pub fn compile(&self, code: LiftedCode) -> crate::Result<JitRegion> {
        let mut addresses: HashMap<GotTarget, u64> = HashMap::new();
        for import in &code.imports {
            addresses.insert(
                GotTarget::Host(import.symbol.clone()),
                import.address as u64,
            );
        }
        for import in &code.data_imports {
            let target = match &import.target {
                DataTarget::Host(symbol) => GotTarget::Host(symbol.clone()),
                DataTarget::GuestSlide => GotTarget::GuestSlide,
            };
            addresses.insert(target, import.address);
        }
        let object = self.compiler.compile(code)?;
        for (name, helper) in &object.helpers {
            self.place_shared(GotTarget::Helper(name), || self.place_helper(helper))?;
        }
        for table in &object.tables {
            self.place_shared(GotTarget::Table(table.name), || {
                let span = self.span(table.bytes.len(), table.align.max(1) as usize)?;
                self.arena
                    .write(span, |bytes| bytes.copy_from_slice(&table.bytes));
                Ok(span.exec as u64)
            })?;
        }

        let cells = match object.meta.link_sites.len() {
            0 => std::ptr::null_mut(),
            n => self
                .arena
                .data(n * ENTRY_SIZE as usize, ENTRY_SIZE as usize)
                .map_err(arena_error)?
                .cast::<BlockLinkEntry>(),
        };
        let got = object.code.targets.len() * 8;
        let span = self.span(got + object.code.text.len(), TEXT_ALIGN as usize)?;
        let text_va = span.exec as u64 + got as u64;
        let mut values = Vec::with_capacity(object.code.targets.len());
        for target in &object.code.targets {
            values.push(match target {
                GotTarget::Cells => cells as u64,
                GotTarget::Body => text_va + u64::from(object.link_target),
                GotTarget::Helper(_) | GotTarget::Table(_) => self.placed_at(target)?,
                GotTarget::Host(symbol) if symbol.kind == HostSymbolKind::LibCall => {
                    libcall(symbol)?
                }
                other => *addresses.get(other).ok_or_else(|| {
                    crate::MicroliftError::CraneliftError(format!(
                        "{other:?} has no address in this lift"
                    ))
                })?,
            });
        }
        let mut text = object.code.text;
        link(&mut text, &object.code.refs, text_va, |i| {
            span.exec as u64 + u64::from(i) * 8
        })?;
        self.arena.write(span, |bytes| {
            for (slot, value) in bytes[..got].as_chunks_mut::<8>().0.iter_mut().zip(&values) {
                slot.copy_from_slice(&value.to_ne_bytes());
            }
            bytes[got..].copy_from_slice(&text);
        });

        let at = |offset: u32| (text_va + u64::from(offset)) as *const u8;
        Ok(JitRegion {
            region: CompiledRegion {
                entry: at(object.entry),
                link_target: at(object.link_target),
                link_cells: cells,
            },
            map: object.meta.map,
            resume_points: object.meta.resume_points,
            covered: object.meta.covered,
            data_dependencies: object.meta.data_dependencies,
            link_sites: object.meta.link_sites,
            code_len: span.len,
        })
    }

    fn span(&self, len: usize, align: usize) -> crate::Result<super::arena::CodeSpan> {
        self.arena.code(len.max(1), align).map_err(arena_error)
    }

    fn placed_at(&self, target: &GotTarget) -> crate::Result<u64> {
        let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
        placed
            .get(target)
            .copied()
            .ok_or_else(|| crate::MicroliftError::CraneliftError(format!("{target:?} unplaced")))
    }

    /// Places a helper or table once for the life of the process.
    fn place_shared(
        &self,
        target: GotTarget,
        place: impl FnOnce() -> crate::Result<u64>,
    ) -> crate::Result<()> {
        let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
        if let std::collections::hash_map::Entry::Vacant(slot) = placed.entry(target) {
            slot.insert(place()?);
        }
        Ok(())
    }

    fn place_helper(&self, helper: &CodeObject) -> crate::Result<u64> {
        let got = helper.targets.len() * 8;
        let span = self.span(got + helper.text.len(), TEXT_ALIGN as usize)?;
        let values = helper
            .targets
            .iter()
            .map(|target| match target {
                GotTarget::Host(symbol) if symbol.kind == HostSymbolKind::LibCall => {
                    libcall(symbol)
                }
                other => Err(crate::MicroliftError::CraneliftError(format!(
                    "a helper names {other:?}"
                ))),
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let text_va = span.exec as u64 + got as u64;
        let mut text = helper.text.clone();
        link(&mut text, &helper.refs, text_va, |i| {
            span.exec as u64 + u64::from(i) * 8
        })?;
        self.arena.write(span, |bytes| {
            for (slot, value) in bytes[..got].as_chunks_mut::<8>().0.iter_mut().zip(&values) {
                slot.copy_from_slice(&value.to_ne_bytes());
            }
            bytes[got..].copy_from_slice(&text);
        });
        Ok(text_va)
    }
}

fn arena_error(e: std::io::Error) -> crate::MicroliftError {
    crate::MicroliftError::CraneliftError(format!("allocating code memory: {e}"))
}

/// A Cranelift runtime routine, as the host's C library provides it.
pub fn libcall(symbol: &HostSymbol) -> crate::Result<u64> {
    let missing = || crate::MicroliftError::CraneliftError(format!("no libcall {symbol}"));
    let name = std::ffi::CString::new(&*symbol.name).map_err(|_| missing())?;
    // SAFETY: a lookup in the default namespace with a NUL-terminated name.
    let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    match address.is_null() {
        true => Err(missing()),
        false => Ok(address as u64),
    }
}
