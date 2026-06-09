use crate::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use cranelift_native::builder as native_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSection};
use std::sync::Arc;
use target_lexicon::Triple;

#[cfg(feature = "disasm")]
use capstone::prelude::*;

pub struct CraneliftBackend {
    module: JITModule,
}

impl CraneliftBackend {
    pub fn new() -> Result<Self> {
        Self::with_flags(settings::Flags::new(settings::builder()))
    }

    pub fn with_flags(flags: settings::Flags) -> Result<Self> {
        let isa_builder =
            native_builder().map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);

        Ok(Self { module })
    }

    /// Idempotently declare and define the direct-native-call trampoline in
    /// this JIT module, and obtain a `FuncRef` for it within `func` that the
    /// lifter can call.  See [`crate::trampoline::declare_trampoline_in_func`].
    pub fn declare_trampoline_in_func(
        &mut self,
        func: &mut Function,
    ) -> Result<cranelift_codegen::ir::FuncRef> {
        crate::trampoline::declare_trampoline_in_func(&mut self.module, func)
    }

    pub fn compile(&mut self, func: Function) -> Result<*const u8> {
        let mut ctx = self.module.make_context();
        ctx.func = func;

        let id = self
            .module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        self.module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let code = self.module.get_finalized_function(id);
        Ok(code)
    }
}

pub struct ArtifactBackend;

/// Stateful AOT compile session.  Holds the `ObjectModule` across the
/// trampoline-declaration step and the `compile` step so the lifted
/// function's references resolve to the same module that defines them.
pub struct ArtifactSession {
    module: ObjectModule,
}

impl ArtifactSession {
    /// Idempotently declare and define the direct-native-call trampoline in
    /// this session's module, and obtain a `FuncRef` for it within `func`.
    /// See [`crate::trampoline::declare_trampoline_in_func`].
    pub fn declare_trampoline_in_func(
        &mut self,
        func: &mut Function,
    ) -> Result<cranelift_codegen::ir::FuncRef> {
        crate::trampoline::declare_trampoline_in_func(&mut self.module, func)
    }

    /// Declare `func` as `lifted_func`, define it, and emit the final object
    /// bytes.  Any module-level dependencies registered earlier (e.g. the
    /// direct-native-call trampoline) are emitted into the same object.
    pub fn compile(mut self, func: Function) -> Result<Vec<u8>> {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        let id = self
            .module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        self.module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        let product = self.module.finish();
        product
            .emit()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))
    }
}

impl ArtifactBackend {
    fn get_isa(
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Arc<dyn isa::TargetIsa>> {
        let triple_obj = if let Some(t) = triple {
            t.parse::<Triple>().map_err(|_| {
                crate::MicroliftError::CraneliftError("invalid target triple".into())
            })?
        } else {
            Triple::host()
        };

        let isa_builder = cranelift_codegen::isa::lookup(triple_obj)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let isa = isa_builder
            .finish(if let Some(f) = flags {
                f
            } else {
                settings::Flags::new(settings::builder())
            })
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok(isa)
    }

    pub fn compile_to_object(
        func: Function,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        let isa = Self::get_isa(triple, flags)?;
        let obj_builder =
            ObjectBuilder::new(isa, "lifted", cranelift_module::default_libcall_names())
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let mut module = ObjectModule::new(obj_builder);
        let mut ctx = cranelift_codegen::Context::for_function(func);

        let id = module
            .declare_function("lifted_func", Linkage::Export, &ctx.func.signature)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        module
            .define_function(id, &mut ctx)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let product = module.finish();
        Ok(product
            .emit()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?)
    }

    /// Stateful AOT compile session that owns an `ObjectModule` long enough
    /// for the lifter to declare the direct-native-call trampoline in the
    /// function being lifted.  Without this, AOT outputs would be missing
    /// the trampoline bytes that lifted code calls into.
    ///
    /// Workflow:
    ///
    /// ```ignore
    /// let mut sess = ArtifactBackend::session(None, None)?;
    /// let mut func = Function::with_name_signature(...);
    /// // ... configure func.signature ...
    /// let funcref = sess.declare_trampoline_in_func(&mut func)?;
    /// config.direct_native_call_funcref = Some(funcref);
    /// // ... run the Lifter against `func` ...
    /// let obj_bytes = sess.compile(func)?;
    /// ```
    pub fn session(
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<ArtifactSession> {
        let isa = Self::get_isa(triple, flags)?;
        let obj_builder =
            ObjectBuilder::new(isa, "lifted", cranelift_module::default_libcall_names())
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        Ok(ArtifactSession {
            module: ObjectModule::new(obj_builder),
        })
    }

    pub fn compile_to_raw(
        func: Function,
        triple: Option<&str>,
        flags: Option<settings::Flags>,
    ) -> Result<Vec<u8>> {
        let obj_bytes = Self::compile_to_object(func, triple, flags)?;
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

        Ok(bytes.to_vec())
    }

    #[cfg(feature = "disasm")]
    pub fn disassemble(bytes: &[u8], triple: Option<&str>) -> Result<String> {
        use target_lexicon::Architecture;

        let triple_obj = if let Some(t) = triple {
            t.parse::<Triple>().map_err(|_| {
                crate::MicroliftError::CraneliftError("invalid target triple".into())
            })?
        } else {
            Triple::host()
        };

        let (arch, mode) = match triple_obj.architecture {
            Architecture::X86_64 => (capstone::Arch::X86, capstone::Mode::Mode64),
            Architecture::Riscv64(_) => (capstone::Arch::RISCV, capstone::Mode::Mode64),
            Architecture::Aarch64(_) => (capstone::Arch::ARM64, capstone::Mode::Arm),
            _ => {
                return Err(crate::MicroliftError::CraneliftError(format!(
                    "unsupported architecture for disassembly: {:?}",
                    triple_obj.architecture
                )));
            }
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

        let insns = cs
            .disasm_all(bytes, 0x0)
            .map_err(|e: capstone::Error| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let mut output = String::new();
        for i in insns.as_ref() {
            output.push_str(&format!("{}\n", i));
        }

        Ok(output)
    }
}
