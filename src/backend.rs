use crate::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use cranelift_native::builder as native_builder;

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
