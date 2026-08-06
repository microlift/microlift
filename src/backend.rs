use crate::Result;
use crate::core::srcloc::{GuestCodeMap, SourceLocMap};
use crate::core::{HelperFunction, LiftedCode};
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use cranelift_native::builder as native_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSection};
use std::sync::Arc;
use target_lexicon::Triple;

/// Metadata for a single external function that the lifted IR references via
/// `ExternalName::user(0, i)`. Produced by `Lifter::aot_imports()`.
///
/// The tuple is `(symbol_name, host_fn_ptr, signature)`:
/// - `symbol_name` - the C symbol the relocation or JIT lookup will resolve.
/// - `host_fn_ptr` - the in-process address; used by JIT backends to register
///   the symbol; AOT backends only need the name and signature.
/// - `signature` - the Cranelift calling-convention signature for the function.
pub type AotImport = (&'static str, *const u8, Signature);

/// A helper that has been declared but not yet compiled - see
/// [`declare_externals`] / [`define_helpers`].
type PendingHelper = (cranelift_module::FuncId, Function);

/// Declare the ordered namespace-0 external functions a [`LiftedCode`]
/// references, *before* the lifted function itself is declared.
///
/// The IR reaches both kinds through `ExternalName::user(0, i)` over one dense
/// index space: imports occupy `0..imports.len()`, helpers continue from there.
/// Module `FuncId`s are handed out in declaration order, so declaring them here
/// in that same order is what makes `FuncId(i) == i` hold.
///
/// Helper *bodies* are not compiled here - the returned list goes to
/// [`define_helpers`] after the lifted function has been defined, which keeps
/// `lifted_func` first in the emitted `.text` (what
/// [`ArtifactBackend::compile_to_raw`] hands back as raw code).
///
/// The module must be fresh: helper symbols are per-body, so compiling two
/// lifted regions into one module would redefine them (and shift the index
/// space out from under the second region's IR).
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

/// Extract the guest-instruction map for a function that was just compiled into
/// `ctx`, decoding the surviving `SourceLoc`s with the lift's `source_map`.
///
/// Empty when the function was lifted with
/// [`LifterConfig::emit_source_locs`](crate::core::LifterConfig::emit_source_locs)
/// off (nothing to decode), and - defensively - when the module reported no
/// compiled code at all.
fn guest_code_map(ctx: &cranelift_codegen::Context, source_map: SourceLocMap) -> GuestCodeMap {
    ctx.compiled_code()
        .map(|compiled| GuestCodeMap::from_compiled(compiled, source_map))
        .unwrap_or_default()
}

/// Compile the helper bodies [`declare_externals`] set aside.
fn define_helpers<M: Module>(module: &mut M, pending: Vec<PendingHelper>) -> Result<()> {
    for (id, func) in pending {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
    }
    Ok(())
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

    /// Build a `TargetIsa` for the host with the shared `flags`.
    ///
    /// `isa_settings` is a list of ISA-specific `(name, value)` pairs applied to
    /// the ISA builder before it is finished - e.g. the x86 target-feature knobs
    /// (`has_avx`, `has_sse41`, ...) used to raise the recompiled code above the
    /// baseline `x86-64` microarchitecture level. This is the JIT counterpart of
    /// `ArtifactBackend::get_isa`; note the builder starts from the *detected*
    /// host features, so these pairs override that detection rather than adding
    /// to a bare baseline. Each name must be valid for the host architecture or
    /// an error is returned.
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

    /// JIT-compile a full [`LiftedCode`] - the lifted region plus any outlined
    /// helper functions it calls (the x87 softfloat kernels).
    ///
    /// This is the entry point for `Jit`-mode output, and the one to use when
    /// the region may contain x87 code; [`compile`](Self::compile) takes a bare
    /// `Function` and so cannot carry helper bodies.
    ///
    /// `code.imports` must be empty: a `RuntimeAssisted` region's imports need
    /// their symbol addresses registered when the module is *built*, which this
    /// backend's module already was - use
    /// [`compile_lifted_with_fast_imports`](Self::compile_lifted_with_fast_imports).
    pub fn compile_lifted(&mut self, code: LiftedCode) -> Result<*const u8> {
        Ok(self.compile_lifted_with_map(code)?.0)
    }

    /// [`compile_lifted`](Self::compile_lifted), additionally returning the
    /// host-code -> guest-instruction map for the compiled region.
    ///
    /// The map's offsets are relative to the returned function pointer, so
    /// [`GuestCodeMap::lookup_host_addr`] turns a host `RIP` sampled inside this
    /// region - from a signal handler, a profiler, a debugger - into the guest
    /// address it is executing. It is empty unless the lift ran with
    /// [`LifterConfig::emit_source_locs`](crate::core::LifterConfig::emit_source_locs)
    /// on (the default).
    pub fn compile_lifted_with_map(
        &mut self,
        code: LiftedCode,
    ) -> Result<(*const u8, GuestCodeMap)> {
        if !code.imports.is_empty() {
            return Err(crate::MicroliftError::CraneliftError(
                "lifted code has relocatable imports; use compile_lifted_with_fast_imports".into(),
            ));
        }
        let source_map = code.source_map;
        let pending = declare_externals(&mut self.module, &[], code.helpers)?;

        let mut ctx = self.module.make_context();
        ctx.func = code.func;

        let id = self
            .module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        self.module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
        // Read the compiled buffer's source locations before the context - and
        // with it the compiled code - is recycled.
        let map = guest_code_map(&ctx, source_map);
        self.module.clear_context(&mut ctx);

        define_helpers(&mut self.module, pending)?;
        self.module
            .finalize_definitions()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok((self.module.get_finalized_function(id), map))
    }

    /// JIT-compile a single lifted `Function`.
    ///
    /// Note this cannot carry outlined helpers; see
    /// [`compile_lifted`](Self::compile_lifted).
    pub fn compile(&mut self, func: Function) -> Result<*const u8> {
        self.compile_lifted(LiftedCode {
            func,
            imports: Vec::new(),
            helpers: Vec::new(),
            analysis_call_targets: Vec::new(),
            source_map: SourceLocMap::default(),
            resume_points: Vec::new(),
        })
    }

    /// JIT-compile a `RuntimeAssisted`-mode function that uses relocatable
    /// fast-call imports (i.e. `call(func_ref, ...)` instructions backed by
    /// `ExternalName::user(0, i)` references).
    ///
    /// `imports` must be the slice returned by `Lifter::aot_imports()` -
    /// element `i` must correspond to `ExternalName::user(0, i)` in the IR.
    /// Imports are declared first so each receives `FuncId(i)`, which the
    /// JIT linker maps to the registered symbol address at finalization time.
    ///
    /// `isa_settings` carries host ISA feature overrides, as in `with_flags`.
    pub fn compile_with_fast_imports(
        func: Function,
        imports: &[AotImport],
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<*const u8> {
        Self::compile_lifted_with_fast_imports(
            LiftedCode {
                func,
                imports: imports.to_vec(),
                helpers: Vec::new(),
                analysis_call_targets: Vec::new(),
                source_map: SourceLocMap::default(),
                resume_points: Vec::new(),
            },
            flags,
            isa_settings,
        )
    }

    /// [`compile_with_fast_imports`](Self::compile_with_fast_imports) for a full
    /// [`LiftedCode`], so a `RuntimeAssisted` region may also carry outlined
    /// helper functions.
    pub fn compile_lifted_with_fast_imports(
        code: LiftedCode,
        flags: settings::Flags,
        isa_settings: &[(&str, &str)],
    ) -> Result<*const u8> {
        Ok(Self::compile_lifted_with_fast_imports_and_map(code, flags, isa_settings)?.0)
    }

    /// [`compile_lifted_with_fast_imports`](Self::compile_lifted_with_fast_imports),
    /// additionally returning the host-code -> guest-instruction map. See
    /// [`compile_lifted_with_map`](Self::compile_lifted_with_map).
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

        let pending = declare_externals(&mut module, &code.imports, code.helpers)?;

        let mut ctx = module.make_context();
        ctx.func = code.func;
        let id = module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
        let map = guest_code_map(&ctx, source_map);

        define_helpers(&mut module, pending)?;
        module
            .finalize_definitions()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok((module.get_finalized_function(id), map))
    }
}

pub struct ArtifactBackend;

impl ArtifactBackend {
    /// Build a `TargetIsa` for `triple` with the shared `flags`.
    ///
    /// `isa_settings` is a list of ISA-specific `(name, value)` pairs applied to
    /// the ISA builder before it is finished - e.g. the x86 target-feature knobs
    /// (`has_avx`, `has_sse41`, ...) used to raise the recompiled code above the
    /// baseline `x86-64` microarchitecture level. Each name must be valid for the
    /// resolved architecture or an error is returned.
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

        let mut isa_builder = cranelift_codegen::isa::lookup(triple_obj)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

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

    /// Compile a `RuntimeAssisted`-mode function to a relocatable object file,
    /// emitting named external relocations for fast-call imports.
    ///
    /// `imports` must be the slice returned by `Lifter::aot_imports()` -
    /// element `i` corresponds to `ExternalName::user(0, i)` in the IR and
    /// will be declared as an imported symbol before `lifted_func`.
    /// Declaring them first ensures `FuncId(i) == i`, so the object module
    /// can resolve `ExternalName::user(0, i)` to the correct symbol name when
    /// emitting relocations.
    pub fn compile_to_object_with_fast_imports(
        func: Function,
        imports: &[AotImport],
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<Vec<u8>> {
        Self::compile_lifted_to_object(
            LiftedCode {
                func,
                imports: imports.to_vec(),
                helpers: Vec::new(),
                analysis_call_targets: Vec::new(),
                source_map: SourceLocMap::default(),
                resume_points: Vec::new(),
            },
            triple,
            flags,
            isa_settings,
        )
    }

    /// Compile a full [`LiftedCode`] to a relocatable object file: the lifted
    /// function, named relocations for its fast-call imports, and a local
    /// definition of every outlined helper it calls.
    pub fn compile_lifted_to_object(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
        isa_settings: &[(&str, &str)],
    ) -> Result<Vec<u8>> {
        Ok(Self::compile_lifted_to_object_with_map(code, triple, flags, isa_settings)?.0)
    }

    /// [`compile_lifted_to_object`](Self::compile_lifted_to_object), additionally
    /// returning the host-code -> guest-instruction map for `lifted_func`.
    ///
    /// Offsets are relative to the start of `lifted_func`, not to the start of
    /// the object's `.text` - look the symbol's address up in the object (or
    /// note that `lifted_func` is defined first, so it sits at offset 0 of the
    /// `.text` that [`compile_lifted_to_raw`](Self::compile_lifted_to_raw)
    /// extracts).
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

        let pending = declare_externals(&mut module, &code.imports, code.helpers)?;

        let mut ctx = cranelift_codegen::Context::for_function(code.func);
        let id = module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{e:?}")))?;
        let map = guest_code_map(&ctx, source_map);

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
            LiftedCode {
                func,
                imports: Vec::new(),
                helpers: Vec::new(),
                analysis_call_targets: Vec::new(),
                source_map: SourceLocMap::default(),
                resume_points: Vec::new(),
            },
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
            LiftedCode {
                func,
                imports: Vec::new(),
                helpers: Vec::new(),
                analysis_call_targets: Vec::new(),
                source_map: SourceLocMap::default(),
                resume_points: Vec::new(),
            },
            triple,
            flags,
        )
    }

    /// [`compile_to_raw`](Self::compile_to_raw) for a full [`LiftedCode`].
    ///
    /// Note the extracted `.text` holds the outlined helper bodies as well as
    /// the lifted function, and the object's symbol table - which says where
    /// each begins - is what raw extraction throws away. Prefer
    /// [`compile_lifted_to_object`](Self::compile_lifted_to_object) when the
    /// region has helpers.
    pub fn compile_lifted_to_raw(
        code: LiftedCode,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        Ok(Self::compile_lifted_to_raw_with_map(code, triple, flags)?.0)
    }

    /// [`compile_lifted_to_raw`](Self::compile_lifted_to_raw), additionally
    /// returning the host-code -> guest-instruction map.
    ///
    /// `lifted_func` is defined before the helpers, so its code starts at offset
    /// 0 of the extracted `.text` and the map's offsets index the returned bytes
    /// directly - which is what makes an annotated disassembly possible; see
    /// [`disassemble_annotated`](Self::disassemble_annotated).
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

    /// Build a Capstone engine for `triple` (host triple when `None`).
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

    /// Disassemble compiled code, interleaving a header line each time the guest
    /// instruction responsible for it changes.
    ///
    /// This is the payoff of source locations for a human reading output: the
    /// lifted code for one guest instruction is not contiguous, not in guest
    /// order, and heavily interleaved by scheduling, so a bare disassembly of a
    /// lifted region is very hard to relate back to the guest program. Host
    /// instructions with no guest origin (prologue, dispatch switch) are grouped
    /// under a `[no source]` header.
    ///
    /// `bytes` must start at the beginning of the function the map came from -
    /// i.e. the output of
    /// [`compile_lifted_to_raw_with_map`](Self::compile_lifted_to_raw_with_map),
    /// whose `lifted_func` sits at offset 0.
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
        // `None` is a real state here ("no guest origin"), so the "have we
        // printed a header yet" flag cannot be folded into the value itself.
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
