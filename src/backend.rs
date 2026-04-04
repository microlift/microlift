use crate::Result;
use cranelift::prelude::*;
use std::sync::Arc;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use cranelift_native::builder as native_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSection};
use target_lexicon::Triple;

pub struct CraneliftBackend {
    module: JITModule,
}

impl CraneliftBackend {
    pub fn new() -> Result<Self> {
        let isa_builder =
            native_builder().map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;
        let flag_builder = settings::builder();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);

        Ok(Self { module })
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

impl ArtifactBackend {
    fn get_isa(triple: Option<&str>) -> Result<Arc<dyn isa::TargetIsa>> {
        let triple_obj = if let Some(t) = triple {
            t.parse::<Triple>()
                .map_err(|_| crate::MicroliftError::CraneliftError("invalid target triple".into()))?
        } else {
            Triple::host()
        };

        let isa_builder = cranelift_codegen::isa::lookup(triple_obj)
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        let flag_builder = settings::builder();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

        Ok(isa)
    }

    pub fn compile_to_object(func: Function, triple: Option<&str>) -> Result<Vec<u8>> {
        let isa = Self::get_isa(triple)?;
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

    pub fn compile_to_raw(func: Function, triple: Option<&str>) -> Result<Vec<u8>> {
        let obj_bytes = Self::compile_to_object(func, triple)?;
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
}
