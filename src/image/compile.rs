//! Compiling a lift to position-independent code.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::{ExternalName, Function, UserExternalName};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::{MachReloc, RelocTarget};

use super::{CodeObject, GotRef, GotRefKind, GotTarget, RegionMeta, RegionObject, TEXT_ALIGN};
use crate::core::srcloc::{GuestCodeMap, SourceLocMap};
use crate::core::symbol::{DataTarget, HostSymbol};
use crate::core::{HelperFunction, LiftedCode};

/// Compiles lifts for one ISA, which must be position-independent (`is_pic`).
///
/// Shared between threads: helpers are compiled once, by whichever thread needs one
/// first, and handed to every region that calls them.
pub struct PicCompiler {
    isa: Arc<dyn TargetIsa>,
    helpers: Mutex<HashMap<&'static str, Arc<CodeObject>>>,
}

impl PicCompiler {
    pub fn new(isa: Arc<dyn TargetIsa>) -> crate::Result<Self> {
        if !isa.flags().is_pic() {
            return Err(crate::MicroliftError::CraneliftError(
                "position-independent compilation needs an ISA with is_pic".into(),
            ));
        }
        Ok(Self {
            isa,
            helpers: Mutex::default(),
        })
    }

    pub fn isa(&self) -> &dyn TargetIsa {
        &*self.isa
    }

    /// Compiles `code`: the body at offset 0, then the entry thunk if it has one.
    pub fn compile(&self, code: LiftedCode) -> crate::Result<RegionObject> {
        crate::backend::check_names(&code)?;
        let helpers: Vec<(&'static str, Arc<CodeObject>)> = code
            .helpers
            .into_iter()
            .map(|helper| Ok((helper.name, self.helper(helper)?)))
            .collect::<crate::Result<_>>()?;

        let functions = code.imports.len() + helpers.len();
        let cells = !code.link_sites.is_empty() as usize;
        let names = |name: UserExternalName| -> crate::Result<GotTarget> {
            let index = name.index as usize;
            let target = match name.namespace {
                0 if index < code.imports.len() => {
                    Some(GotTarget::Host(code.imports[index].symbol.clone()))
                }
                0 if index < functions => {
                    Some(GotTarget::Helper(helpers[index - code.imports.len()].0))
                }
                0 if index == functions => Some(GotTarget::Body),
                1 if index < cells => Some(GotTarget::Cells),
                1 if index < cells + code.data_objects.len() => {
                    Some(GotTarget::Table(code.data_objects[index - cells].name))
                }
                1 => code
                    .data_imports
                    .get(index - cells - code.data_objects.len())
                    .map(|import| match &import.target {
                        DataTarget::Host(symbol) => GotTarget::Host(symbol.clone()),
                        DataTarget::GuestSlide => GotTarget::GuestSlide,
                    }),
                _ => None,
            };
            target.ok_or_else(|| {
                crate::MicroliftError::CraneliftError(format!(
                    "relocation to {name:?} names nothing the lift declared"
                ))
            })
        };

        let mut object = CodeObject::default();
        let map = self.compile_into(&mut object, code.func, Some(code.source_map), &names)?;
        let entry = match code.entry_thunk {
            Some(thunk) => {
                let at = aligned_end(&object);
                self.compile_into(&mut object, thunk, None, &names)?;
                at
            }
            None => 0,
        };
        Ok(RegionObject {
            code: object,
            entry,
            link_target: 0,
            helpers,
            tables: code.data_objects,
            meta: RegionMeta {
                lift_base: code.lift_base,
                resume_points: code.resume_points,
                covered: code.covered,
                data_dependencies: code.data_dependencies,
                link_sites: code.link_sites,
                import_decisions: code.import_decisions,
                map,
            },
        })
    }

    fn helper(&self, helper: HelperFunction) -> crate::Result<Arc<CodeObject>> {
        if let Some(done) = self.lock().get(helper.name) {
            return Ok(Arc::clone(done));
        }
        // Helper bodies call nothing the lift declared.
        let names = |name: UserExternalName| -> crate::Result<GotTarget> {
            Err(crate::MicroliftError::CraneliftError(format!(
                "helper `{}` names {name:?}",
                helper.name
            )))
        };
        let mut object = CodeObject::default();
        self.compile_into(&mut object, helper.func, None, &names)?;
        let object = Arc::new(object);
        // Two threads may both compile a helper; the first one stored is shared.
        Ok(Arc::clone(self.lock().entry(helper.name).or_insert(object)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<&'static str, Arc<CodeObject>>> {
        self.helpers.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Compiles `func` onto the end of `object`; returns its guest map, relative to
    /// the function's start, given a source map.
    fn compile_into(
        &self,
        object: &mut CodeObject,
        func: Function,
        source_map: Option<SourceLocMap>,
        names: &dyn Fn(UserExternalName) -> crate::Result<GotTarget>,
    ) -> crate::Result<GuestCodeMap> {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        let compiled = ctx
            .compile(&*self.isa, &mut ControlPlane::default())
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("{:?}", e.inner)))?;
        let start = aligned_end(object);
        object.text.resize(start as usize, 0);
        object.text.extend_from_slice(compiled.code_buffer());
        let map = source_map
            .map(|source_map| GuestCodeMap::from_compiled(compiled, source_map))
            .unwrap_or_default();
        let relocs: Vec<MachReloc> = compiled.buffer.relocs().to_vec();
        for (index, reloc) in relocs.iter().enumerate() {
            let kind = match reloc.kind {
                Reloc::Aarch64AdrGotPage21 => GotRefKind::Aarch64Page21,
                Reloc::Aarch64Ld64GotLo12Nc => GotRefKind::Aarch64Lo12Ld64,
                Reloc::X86GOTPCRel4 => GotRefKind::X86PcRel32,
                Reloc::RiscvGotHi20 => GotRefKind::Riscv64HiLo,
                // The low half of a RISC-V pair names the `auipc`, which the pair's
                // high half patches together with it.
                Reloc::RiscvPCRelLo12I
                    if index > 0
                        && relocs[index - 1].kind == Reloc::RiscvGotHi20
                        && relocs[index - 1].offset + 4 == reloc.offset =>
                {
                    continue;
                }
                other => {
                    return Err(crate::MicroliftError::CraneliftError(format!(
                        "relocation {other:?} is not position-independent"
                    )));
                }
            };
            let RelocTarget::ExternalName(name) = &reloc.target else {
                return Err(crate::MicroliftError::CraneliftError(
                    "a GOT reference to a label".into(),
                ));
            };
            let target = match name {
                ExternalName::User(user) => {
                    names(ctx.func.params.user_named_funcs()[*user].clone())?
                }
                ExternalName::LibCall(libcall) => GotTarget::Host(HostSymbol::libcall(
                    &(cranelift_module::default_libcall_names())(*libcall),
                )),
                other => {
                    return Err(crate::MicroliftError::CraneliftError(format!(
                        "external {other:?} cannot be placed"
                    )));
                }
            };
            let addend = i32::try_from(reloc.addend)
                .map_err(|_| crate::MicroliftError::CraneliftError("GOT addend".into()))?;
            let target = object.target(target);
            object.refs.push(GotRef {
                offset: start + reloc.offset,
                kind,
                target,
                addend,
            });
        }
        Ok(map)
    }
}

fn aligned_end(object: &CodeObject) -> u32 {
    (object.text.len() as u64).next_multiple_of(TEXT_ALIGN) as u32
}
