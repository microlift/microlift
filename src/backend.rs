use crate::Result;
use crate::core::srcloc::{GuestCodeMap, SourceLocMap};
use crate::core::link::{BlockLinkEntry, CELL_SYMBOL, ENTRY_SIZE};
use crate::core::{DataObject, HelperFunction, LiftedCode};
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, Linkage, Module};
use cranelift_native::builder as native_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSection};
use std::sync::Arc;
use target_lexicon::Triple;

/// An external function the IR references as `ExternalName::user(0, i)`: `(symbol, host_fn_ptr, signature)`.
/// AOT backends need only the name and signature.
pub type AotImport = (&'static str, *const u8, Signature);

/// A declared-but-uncompiled helper ([`declare_externals`] / [`define_helpers`]).
type PendingHelper = (cranelift_module::FuncId, Function);

/// Declares the namespace-0 externals (imports `0..n`, then helpers) before the lifted function,
/// so `FuncId(i) == i`. Helper bodies are compiled later by [`define_helpers`].
///
/// The module must be fresh: helper symbols are per-body.
fn declare_externals<M: Module>(
    module: &mut M,
    imports: &[AotImport],
    helpers: Vec<HelperFunction>,
) -> Result<Vec<PendingHelper>> {
    for &(name, _, ref sig) in imports {
        module
            .declare_function(name, Linkage::Import, sig)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
    }

    helpers
        .into_iter()
        .map(|helper| {
            let id = module
                .declare_function(helper.name, Linkage::Local, &helper.func.signature)
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
            Ok((id, helper.func))
        })
        .collect()
}

/// The guest-instruction map for a function just compiled into `ctx`; empty when
/// [`emit_source_locs`](crate::core::LifterConfig::emit_source_locs) is off.
fn guest_code_map(ctx: &cranelift_codegen::Context, source_map: SourceLocMap) -> GuestCodeMap {
    ctx.compiled_code()
        .map(|compiled| GuestCodeMap::from_compiled(compiled, source_map))
        .unwrap_or_default()
}

fn define_helpers<M: Module>(module: &mut M, pending: Vec<PendingHelper>) -> Result<()> {
    for (id, func) in pending {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
    }
    Ok(())
}

/// Declares and defines this region's zeroed link-cell array, if any.
///
/// Zero is the uninstalled state (`code == 0`), so a `.bss` array links nothing. It must be
/// `DataId(0)` because the IR names it `ExternalName::user(1, 0)`; asserted, as a violation
/// is a relocation to the wrong object.
fn define_cell_array<M: Module>(module: &mut M, sites: usize) -> Result<Option<DataId>> {
    if sites == 0 {
        return Ok(None);
    }
    let id = module
        .declare_data(CELL_SYMBOL, Linkage::Export, true, false)
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
    assert_eq!(
        id.as_u32(),
        0,
        "the link-cell array must be DataId(0); `compile_lifted*` was given a module that \
         already held data"
    );

    let mut description = DataDescription::new();
    description.define_zeroinit(sites * ENTRY_SIZE as usize);
        // The probe's loads are `aligned`; a row is 32 bytes.
    description.set_align(ENTRY_SIZE);
    module
        .define_data(id, &description)
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
    Ok(Some(id))
}

/// Declares and defines the namespace-1 data: the link-cell array, then read-only constant tables.
///
/// Shares one index space (the cell array takes `DataId(0)`); the module must be fresh, asserted.
fn define_module_data<M: Module>(
    module: &mut M,
    sites: usize,
    tables: &[DataObject],
) -> Result<Option<DataId>> {
    let cells = define_cell_array(module, sites)?;
    let mut next = cells.is_some() as u32;
    for table in tables {
        let id = module
            .declare_data(table.name, Linkage::Local, false, false)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        assert_eq!(
            id.as_u32(),
            next,
            "constant table `{}` must be DataId({next}); `compile_lifted*` was given a module \
             that already held data",
            table.name
        );
        let mut description = DataDescription::new();
        description.define(table.bytes.clone().into_boxed_slice());
        description.set_align(table.align);
        module
            .define_data(id, &description)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        next += 1;
    }
    Ok(cells)
}

/// Defines the region body and, if pinned, the entry thunk; returns the host-entry `FuncId` and the body's map.
///
/// - No thunk: the body is `extern "C"` and exported as `lifted_func`.
/// - Thunk ([`crate::core::pin`]): the `tail`-convention body is local `lifted_body`, declared
///   after the helpers, and `lifted_func` is the thunk.
///
/// The body is defined first so map offsets index from the start of `.text`.
fn define_region<M: Module>(
    module: &mut M,
    func: Function,
    entry_thunk: Option<Function>,
    source_map: SourceLocMap,
) -> Result<(DefinedRegion, GuestCodeMap)> {
    let body_linkage = match entry_thunk {
        Some(_) => Linkage::Local,
        None => Linkage::Export,
    };
    let body_name = match entry_thunk {
        Some(_) => "lifted_body",
        None => "lifted_func",
    };

    let mut ctx = cranelift_codegen::Context::for_function(func);
    let body_id = module
        .declare_function(body_name, body_linkage, &ctx.func.signature)
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

    let Some(thunk) = entry_thunk else {
        module
            .define_function(body_id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
        let map = guest_code_map(&ctx, source_map);
        return Ok((
            DefinedRegion {
                entry: body_id,
                link_target: body_id,
            },
            map,
        ));
    };

    let thunk_id = module
        .declare_function("lifted_func", Linkage::Export, &thunk.signature)
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
    module
        .define_function(body_id, &mut ctx)
        .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
    let map = guest_code_map(&ctx, source_map);

    let mut thunk_ctx = cranelift_codegen::Context::for_function(thunk);
    module
        .define_function(thunk_id, &mut thunk_ctx)
        .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;

    Ok((
        DefinedRegion {
            entry: thunk_id,
            link_target: body_id,
        },
        map,
    ))
}

struct DefinedRegion {
    entry: cranelift_module::FuncId,
    link_target: cranelift_module::FuncId,
}

/// The two ways into a compiled region; equal for an ordinary region, different when pinned
/// ([`crate::core::pin`]), where mixing them up is unchecked wrong code.
#[derive(Debug, Clone, Copy)]
pub struct CompiledRegion {
    /// What a **host** calls: always `extern "C" fn(*mut RegisterContext)`.
    pub entry: *const u8,
    /// What a link row ([`BlockLinkEntry::code`](crate::core::link::BlockLinkEntry::code)) names;
    /// under a pinned ABI the tail-called body, not callable from Rust.
    pub link_target: *const u8,
    /// Zeroed link-cell array, one row per [`LiftedCode::link_sites`](crate::core::LiftedCode::link_sites)
    /// in order; null if none. The host installs cell `i` with the row for `link_sites[i].target`
    /// and retracts by zeroing `code`. Writable, unlike the code pointers.
    pub link_cells: *mut BlockLinkEntry,
}

#[cfg(feature = "disasm")]
use capstone::prelude::*;

pub struct CraneliftBackend {
    module: JITModule,
}

impl CraneliftBackend {
    pub fn new() -> Result<Self> {
        Self::with_flags(settings::Flags::new(settings::builder()), &[])
    }

    /// A `TargetIsa` for the host. `isa_settings` are `(name, value)` overrides applied over
    /// detected features (e.g. `has_avx`); an invalid name errors.
    fn native_isa(
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<Arc<dyn isa::TargetIsa>> {
        let mut isa_builder =
            native_builder().map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        for (name, value) in isa_settings {
            isa_builder.set(name, value).map_err(|e| {
                crate::MicroliftError::CraneliftError(format!(
                    "invalid ISA setting {name}={value}: {e}"
                ))
            })?;
        }

        isa_builder
            .finish(flags)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))
    }

    pub fn with_flags(flags: settings::Flags, isa_settings: &[(&str, &str)]) -> Result<Self> {
        let isa = Self::native_isa(flags, isa_settings)?;

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);

        Ok(Self { module })
    }

    /// JIT-compiles a [`LiftedCode`] including outlined helpers (use for x87).
    ///
    /// `code.imports` must be empty; for `RuntimeAssisted` use
    /// [`compile_lifted_with_fast_imports`](Self::compile_lifted_with_fast_imports).
    pub fn compile_lifted(&mut self, code: LiftedCode) -> Result<*const u8> {
        Ok(self.compile_lifted_with_map(code)?.0)
    }

    /// [`compile_lifted`](Self::compile_lifted) plus the host-code -> guest map.
    ///
    /// Offsets are relative to the function pointer ([`GuestCodeMap::lookup_host_addr`]); empty
    /// unless `emit_source_locs` is on.
    pub fn compile_lifted_with_map(
        &mut self,
        code: LiftedCode,
    ) -> Result<(*const u8, GuestCodeMap)> {
        let (region, map) = self.compile_lifted_region(code)?;
        Ok((region.entry, map))
    }

    /// [`compile_lifted_with_map`](Self::compile_lifted_with_map) reporting both entry points.
    ///
    /// Required when publishing link rows: under a pinned ABI they differ ([`CompiledRegion`]).
    pub fn compile_lifted_region(
        &mut self,
        code: LiftedCode,
    ) -> Result<(CompiledRegion, GuestCodeMap)> {
        if !code.imports.is_empty() {
            return Err(crate::MicroliftError::CraneliftError(
                "lifted code has relocatable imports; use compile_lifted_with_fast_imports".into(),
            ));
        }
        let source_map = code.source_map;
        // First, so it is `DataId(0)` (`ExternalName::user(1, 0)`).
        let cells = define_module_data(
            &mut self.module,
            code.link_sites.len(),
            &code.data_objects,
        )?;
        let pending = declare_externals(&mut self.module, &[], code.helpers)?;

        // Read inside `define_region`, before the owning context is recycled.
        let (ids, map) = define_region(&mut self.module, code.func, code.entry_thunk, source_map)?;

        define_helpers(&mut self.module, pending)?;
        self.module
            .finalize_definitions()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok((
            CompiledRegion {
                entry: self.module.get_finalized_function(ids.entry),
                link_target: self.module.get_finalized_function(ids.link_target),
                link_cells: cells.map_or(std::ptr::null_mut(), |id| {
                    self.module.get_finalized_data(id).0 as *mut BlockLinkEntry
                }),
            },
            map,
        ))
    }

    /// JIT-compiles a bare `Function`; cannot carry helpers ([`compile_lifted`](Self::compile_lifted)).
    pub fn compile(&mut self, func: Function) -> Result<*const u8> {
        self.compile_lifted(LiftedCode::from_function(func))
    }

    /// JIT-compiles a `RuntimeAssisted` function with relocatable fast imports.
    ///
    /// `imports` is `Lifter::aot_imports()`: element `i` is `ExternalName::user(0, i)`, declared
    /// first so it gets `FuncId(i)`.
    pub fn compile_with_fast_imports(
        func: Function,
        imports: &[AotImport],
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<*const u8> {
        Self::compile_lifted_with_fast_imports(
            LiftedCode::from_function_with_imports(func, imports.to_vec()),
            flags,
            isa_settings,
        )
    }

    /// [`compile_with_fast_imports`](Self::compile_with_fast_imports) for a full [`LiftedCode`] (with helpers).
    pub fn compile_lifted_with_fast_imports(
        code: LiftedCode,
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<*const u8> {
        Ok(Self::compile_lifted_with_fast_imports_and_map(code, flags, isa_settings)?.0)
    }

    /// [`compile_lifted_with_fast_imports`](Self::compile_lifted_with_fast_imports) plus the map.
    pub fn compile_lifted_with_fast_imports_and_map(
        code: LiftedCode,
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<(*const u8, GuestCodeMap)> {
        let source_map = code.source_map;
        let isa = Self::native_isa(flags, isa_settings)?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        for &(name, ptr, _) in &code.imports {
            jit_builder.symbol(name, ptr);
        }
        let mut module = JITModule::new(jit_builder);

        define_module_data(&mut module, code.link_sites.len(), &code.data_objects)?;
        let pending = declare_externals(&mut module, &code.imports, code.helpers)?;

        let (ids, map) = define_region(&mut module, code.func, code.entry_thunk, source_map)?;

        define_helpers(&mut module, pending)?;
        module
            .finalize_definitions()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        // Owns its module and returns a bare pointer; a linkable region wants `compile_lifted_region`.
        Ok((module.get_finalized_function(ids.entry), map))
    }
}

fn missing_arch_feature(triple: &Triple) -> Option<&'static str> {
    use target_lexicon::Architecture;

    Some(match triple.architecture {
        Architecture::X86_64 | Architecture::X86_64h | Architecture::X86_32(_) => "arch-x86",
        Architecture::Aarch64(_) => "arch-arm64",
        Architecture::Riscv64(_) => "arch-riscv64",
        Architecture::S390x => "arch-s390x",
        Architecture::Pulley32
        | Architecture::Pulley64
        | Architecture::Pulley32be
        | Architecture::Pulley64be => "arch-pulley",
        // No Cranelift backend exists for this target.
        _ => return None,
    })
}

pub struct ArtifactBackend;

impl ArtifactBackend {
    /// A `TargetIsa` for `triple`; `isa_settings` as in the host variant.
    fn get_isa(
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<Arc<dyn isa::TargetIsa>> {
        let triple_obj = if let Some(t) = triple {
            t.parse::<Triple>().map_err(|_| {
                crate::MicroliftError::CraneliftError("invalid target triple".into())
            })?
        } else {
            Triple::host()
        };

        let mut isa_builder = cranelift_codegen::isa::lookup(triple_obj.clone()).map_err(|e| {
            crate::MicroliftError::CraneliftError(match missing_arch_feature(&triple_obj) {
                Some(feature) => format!(
                    "no Cranelift backend for {triple_obj}: {e} \
                     (build microlift with the `{feature}` feature)"
                ),
                None => e.to_string(),
            })
        })?;

        for (name, value) in isa_settings {
            isa_builder.set(name, value).map_err(|e| {
                crate::MicroliftError::CraneliftError(format!(
                    "invalid ISA setting {name}={value}: {e}"
                ))
            })?;
        }

        let isa = isa_builder
            .finish(if let Some(f) = flags {
                f
            } else {
                settings::Flags::new(settings::builder())
            })
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok(isa)
    }

    /// Compiles a `RuntimeAssisted` function to a relocatable object; `imports` is
    /// `Lifter::aot_imports()`, declared before `lifted_func` so `FuncId(i) == i`.
    pub fn compile_to_object_with_fast_imports(
        func: Function,
        imports: &[AotImport],
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<Vec<u8>> {
        Self::compile_lifted_to_object(
            LiftedCode::from_function_with_imports(func, imports.to_vec()),
            triple,
            flags,
            isa_settings,
        )
    }

    /// Compiles a [`LiftedCode`] to a relocatable object: function, import relocations and local helpers.
    pub fn compile_lifted_to_object(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<Vec<u8>> {
        Ok(Self::compile_lifted_to_object_with_map(code, triple, flags, isa_settings)?.0)
    }

    /// [`compile_lifted_to_object`](Self::compile_lifted_to_object) plus the map; offsets are
    /// relative to `lifted_func` (which sits at 0).
    pub fn compile_lifted_to_object_with_map(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<(Vec<u8>, GuestCodeMap)> {
        let source_map = code.source_map;
        let isa = Self::get_isa(triple, flags, isa_settings)?;
        let obj_builder =
            ObjectBuilder::new(isa, "lifted", cranelift_module::default_libcall_names())
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        let mut module = ObjectModule::new(obj_builder);

        // First, so it is `DataId(0)`; in an object it lands in `.bss` (the uninstalled state).
        define_module_data(&mut module, code.link_sites.len(), &code.data_objects)?;
        let pending = declare_externals(&mut module, &code.imports, code.helpers)?;

        let (_id, map) = define_region(&mut module, code.func, code.entry_thunk, source_map)?;

        define_helpers(&mut module, pending)?;

        let product = module.finish();
        let bytes = product
            .emit()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        Ok((bytes, map))
    }

    pub fn compile_to_object(
        func: Function,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        Self::compile_lifted_to_object(
            LiftedCode::from_function(func),
            triple,
            flags,
            &[],
        )
    }

    pub fn compile_to_raw(
        func: Function,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        Self::compile_lifted_to_raw(
            LiftedCode::from_function(func),
            triple,
            flags,
        )
    }

    /// [`compile_to_raw`](Self::compile_to_raw) for a [`LiftedCode`].
    ///
    /// Raw extraction drops the symbol table and all non-code (data tables, their relocations);
    /// prefer [`compile_lifted_to_object`](Self::compile_lifted_to_object) with helpers or data.
    pub fn compile_lifted_to_raw(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        Ok(Self::compile_lifted_to_raw_with_map(code, triple, flags)?.0)
    }

    /// [`compile_lifted_to_raw`](Self::compile_lifted_to_raw) plus the map; `lifted_func` is at
    /// offset 0, so map offsets index the bytes directly.
    pub fn compile_lifted_to_raw_with_map(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<(Vec<u8>, GuestCodeMap)> {
        let (obj_bytes, map) = Self::compile_lifted_to_object_with_map(code, triple, flags, &[])?;
        let obj = object::File::parse(&*obj_bytes)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let section = obj
            .sections()
            .find(|s| s.kind() == object::SectionKind::Text)
            .ok_or_else(|| {
                crate::MicroliftError::CraneliftError("failed to find text section".into())
            })?;

        let bytes = section
            .data()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok((bytes.to_vec(), map))
    }

    /// A Capstone engine for `triple` (host when `None`).
    #[cfg(feature = "disasm")]
    fn capstone_for(triple: Option<&str>) -> Result<Capstone> {
        use target_lexicon::Architecture;

        let triple_obj = if let Some(t) = triple {
            t.parse::<Triple>().map_err(|_| {
                crate::MicroliftError::CraneliftError("invalid target triple".into())
            })?
        } else {
            Triple::host()
        };

        let cs = match triple_obj.architecture {
            Architecture::X86_64 => Capstone::new()
                .x86()
                .mode(capstone::arch::x86::ArchMode::Mode64)
                .build()
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?,
            Architecture::Riscv64(_) => Capstone::new()
                .riscv()
                .mode(capstone::arch::riscv::ArchMode::RiscV64)
                .build()
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?,
            Architecture::Aarch64(_) => Capstone::new()
                .arm64()
                .mode(capstone::arch::arm64::ArchMode::Arm)
                .build()
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?,
            _ => {
                return Err(crate::MicroliftError::CraneliftError(format!(
                    "unsupported architecture for disassembly: {:?}",
                    triple_obj.architecture
                )));
            }
        };

        Ok(cs)
    }

    #[cfg(feature = "disasm")]
    pub fn disassemble(bytes: &[u8], triple: Option<&str>) -> Result<String> {
        let cs = Self::capstone_for(triple)?;
        let insns = cs
            .disasm_all(bytes, 0x0)
            .map_err(|e: capstone::Error| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let mut output = String::new();
        for i in insns.as_ref() {
            output.push_str(&format!("{}\n", i));
        }

        Ok(output)
    }

    /// Disassembles `bytes` (the start of the mapped function), with a header whenever the
    /// responsible guest instruction changes; code with no origin goes under `[no source]`.
    #[cfg(feature = "disasm")]
    pub fn disassemble_annotated(
        bytes: &[u8],
        triple: Option<&str>,
        map: &GuestCodeMap,
    ) -> Result<String> {
        let cs = Self::capstone_for(triple)?;
        let insns = cs
            .disasm_all(bytes, 0x0)
            .map_err(|e: capstone::Error| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let mut output = String::new();
        // `None` is a real state ("no guest origin"), so the "header printed" flag stays separate.
        let mut current: Option<u64> = None;
        let mut started = false;
        for i in insns.as_ref() {
            let guest = u32::try_from(i.address()).ok().and_then(|o| map.lookup(o));
            if !started || guest != current {
                match guest {
                    Some(va) => output.push_str(&format!("; guest {va:#x}\n")),
                    None => output.push_str("; [no source]\n"),
                }
                current = guest;
                started = true;
            }
            output.push_str(&format!("{}\n", i));
        }

        Ok(output)
    }
}
