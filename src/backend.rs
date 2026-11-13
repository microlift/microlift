use crate::Result;
use crate::core::link::{BlockLinkEntry, CELL_SYMBOL, ENTRY_SIZE};
use crate::core::srcloc::{GuestCodeMap, SourceLocMap};
use crate::core::symbol::{DataImport, DataTarget, HostImport};
use crate::core::{DataObject, HelperFunction, LiftedCode};
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, Linkage, Module};
use cranelift_native::builder as native_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSection};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use target_lexicon::Triple;

/// What an object file calls a data target; host symbols keep their own names.
const GUEST_SLIDE_SYMBOL: &str = "__microlift_guest_slide";

fn object_function_name(import: &HostImport) -> String {
    import.symbol.name.to_string()
}

fn object_data_name(import: &DataImport) -> String {
    match &import.target {
        DataTarget::Host(symbol) => symbol.name.to_string(),
        DataTarget::GuestSlide => GUEST_SLIDE_SYMBOL.to_string(),
    }
}

/// A declared-but-uncompiled helper ([`declare_externals`] / [`define_helpers`]).
type PendingHelper = (cranelift_module::FuncId, Function);

/// Declares the namespace-0 externals (imports `0..n`, then helpers) before the lifted function,
/// so `FuncId(i) == i`. Helper bodies are compiled later by [`define_helpers`].
///
/// The module must be fresh: helper symbols are per-body.
/// `names[i]` is what the module calls `imports[i]`; two equal names are one function.
fn declare_externals<M: Module>(
    module: &mut M,
    imports: &[HostImport],
    names: &[String],
    helpers: Vec<HelperFunction>,
) -> Result<Vec<PendingHelper>> {
    for (index, (import, name)) in imports.iter().zip(names).enumerate() {
        let id = module
            .declare_function(name, Linkage::Import, &import.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        if id.as_u32() as usize != index {
            return Err(crate::MicroliftError::CraneliftError(format!(
                "`{}` and an earlier import share the name `{name}`",
                import.symbol
            )));
        }
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

/// Refuses a body naming an external nothing was declared for: a dropped import's
/// `user(0, i)` would otherwise bind to whatever function holds `FuncId(i)`.
pub(crate) fn check_names(code: &LiftedCode) -> Result<()> {
    let functions = code.imports.len() + code.helpers.len();
    let data =
        !code.link_sites.is_empty() as usize + code.data_objects.len() + code.data_imports.len();
    for name in code.func.params.user_named_funcs().values() {
        let declared = match name.namespace {
            0 => functions,
            1 => data,
            _ => 0,
        };
        if name.index as usize >= declared {
            return Err(crate::MicroliftError::CraneliftError(format!(
                "the body names {name:?}, which the lift did not declare; \
                 compile the `LiftedCode` from `lift_to_code`, not a bare `Function`"
            )));
        }
    }
    Ok(())
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

/// Declares and defines the namespace-1 data: the link-cell array, read-only constant
/// tables, then the external data named `data_names`.
///
/// Shares one index space (the cell array takes `DataId(0)`); the module must be fresh, asserted.
fn define_module_data<M: Module>(
    module: &mut M,
    sites: usize,
    tables: &[DataObject],
    data_names: &[String],
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
    for name in data_names {
        let id = module
            .declare_data(name, Linkage::Import, false, false)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        if id.as_u32() != next {
            return Err(crate::MicroliftError::CraneliftError(format!(
                "external data `{name}` must be DataId({next}); the name is taken"
            )));
        }
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
    /// Name -> address for this module's imports, filled before each definition.
    resolved: Arc<Mutex<HashMap<String, usize>>>,
}

impl CraneliftBackend {
    pub fn new() -> Result<Self> {
        Self::with_flags(settings::Flags::new(settings::builder()), &[])
    }

    /// A `TargetIsa` for the host. `isa_settings` are `(name, value)` overrides applied over
    /// detected features (e.g. `has_avx`); an invalid name errors.
    pub fn native_isa(
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

        let resolved: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let lookup = Arc::clone(&resolved);
        jit_builder.symbol_lookup_fn(Box::new(move |name| {
            let resolved = lookup.lock().unwrap_or_else(|e| e.into_inner());
            resolved.get(name).map(|&address| address as *const u8)
        }));
        let module = JITModule::new(jit_builder);

        Ok(Self { module, resolved })
    }

    /// Names this module's imports and records what each resolves to.
    fn resolve_imports(&self, code: &LiftedCode) -> (Vec<String>, Vec<String>) {
        let mut resolved = self.resolved.lock().unwrap_or_else(|e| e.into_inner());
        let functions = code
            .imports
            .iter()
            .enumerate()
            .map(|(i, import)| {
                let name = format!("import{i}");
                resolved.insert(name.clone(), import.address);
                name
            })
            .collect();
        let data = code
            .data_imports
            .iter()
            .enumerate()
            .map(|(i, import)| {
                let name = format!("data{i}");
                resolved.insert(name.clone(), import.address as usize);
                name
            })
            .collect();
        (functions, data)
    }

    /// JIT-compiles a [`LiftedCode`] including outlined helpers (use for x87).
    ///
    /// Imports resolve to the addresses the lift recorded beside their symbols.
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
        check_names(&code)?;
        let source_map = code.source_map;
        let (function_names, data_names) = self.resolve_imports(&code);
        // First, so it is `DataId(0)` (`ExternalName::user(1, 0)`).
        let cells = define_module_data(
            &mut self.module,
            code.link_sites.len(),
            &code.data_objects,
            &data_names,
        )?;
        let pending = declare_externals(
            &mut self.module,
            &code.imports,
            &function_names,
            code.helpers,
        )?;

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

    /// The ISA this backend compiles for, as [`isa_fingerprint`] hashes it.
    pub fn isa(&self) -> &dyn isa::TargetIsa {
        self.module.isa()
    }
}

/// Everything about `isa` compiled code depends on: the triple, the shared flags and
/// the ISA flags, each by name and value.
pub fn isa_fingerprint(isa: &dyn isa::TargetIsa) -> u64 {
    use std::hash::Hasher;
    let mut hasher = crate::core::fingerprint::StableHasher::new();
    hasher.str(&isa.triple().to_string());
    hasher.str(&isa.flags().to_string());
    for value in isa.isa_flags() {
        hasher.str(&value.to_string());
    }
    hasher.finish()
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
    pub fn get_isa(
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

    /// Compiles a function with imports to a relocatable object; `imports` is
    /// `Lifter::aot_imports()`, declared before `lifted_func` so `FuncId(i) == i`.
    pub fn compile_to_object_with_fast_imports(
        func: Function,
        imports: &[HostImport],
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
        check_names(&code)?;
        let data_names: Vec<String> = code.data_imports.iter().map(object_data_name).collect();
        let function_names: Vec<String> = code.imports.iter().map(object_function_name).collect();
        define_module_data(
            &mut module,
            code.link_sites.len(),
            &code.data_objects,
            &data_names,
        )?;
        let pending = declare_externals(&mut module, &code.imports, &function_names, code.helpers)?;

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
        Self::compile_lifted_to_object(LiftedCode::from_function(func), triple, flags, &[])
    }

    pub fn compile_to_raw(
        func: Function,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        Self::compile_lifted_to_raw(LiftedCode::from_function(func), triple, flags)
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
