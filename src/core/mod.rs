//! Guest-agnostic lift engine: block discovery, PC dispatch, imports, context sync and exits.
//! Guests plug in through [`GuestArch`]; `Lifter<'a>` is `LiftEngine<'a, X86_64>`.

pub mod abi;
pub mod alias;
pub mod align;
pub mod bridge;
pub mod callback;
pub mod callret;
pub mod dce;
pub mod exit;
pub mod flags;
pub mod hostframe;
pub mod link;
pub mod optimize;
pub mod pin;
pub mod regsync;
pub mod specialize;
pub mod srcloc;

use std::collections::BTreeSet;
use std::ops::Range;

use crate::arch::{ControlFlow, ControlFlowKind, InstructionEffects};
use crate::core::abi::CallAbi;
use crate::core::bridge::{BridgeImport, ImportKind, ImportRegistry};
use crate::core::callback::CallbackOffsets;
use crate::core::callret::{CallRetConfig, CallRetOffsets};
use crate::core::exit::{ExitReason, MemorySegment, TransferKind};
use crate::core::flags::FlagEffects;
use crate::core::hostframe::{HostFrameConfig, HostFrameOffsets};
use crate::core::link::{BlockLinkConfig, BlockLinkOffsets};
use crate::core::optimize::Optimizations;
use crate::core::pin::{PinnedAbi, PinnedReg};
use crate::core::specialize::Specialization;
use crate::core::srcloc::SourceLocMap;
use crate::function_traits::{DynamicSymbol, FunctionArgumentType};
use crate::os::get_host_calling_conv;
use cranelift::prelude::*;
use cranelift_codegen::entity::EntitySet;
use cranelift_codegen::ir::{
    ExtFuncData, ExternalName, FuncRef, Function, Inst, SigRef, SourceLoc, UserExternalName,
    UserExternalNameRef, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A guest frontend; semantics are inherent methods on `LiftEngine<'a, Self>`.
pub trait GuestArch: Sized {
    /// Longest encoding in bytes; sizes the window for [`LifterConfig::region_tail_exit`].
    const MAX_INSTRUCTION_LEN: u64;

    type Instr;
    /// SSA register file, including the `context_ptr` / `image_base` variables.
    type RegState;

    fn pc_offset() -> i32;
    fn exit_reason_offset() -> i32;
    fn exit_code_offset() -> i32;
    fn image_base_offset() -> i32;
    fn imported_function_va_offset() -> i32;

    /// Host-callback fields; every guest has them (see [`crate::core::callback`]).
    fn callback_offsets() -> CallbackOffsets;

    /// Block-link fields, or `None`, which makes [`LifterConfig::block_link`] inert.
    fn block_link_offsets() -> Option<BlockLinkOffsets> {
        None
    }

    /// Call-return ring fields, or `None`, which makes [`LifterConfig::call_return_stack`] inert.
    fn call_return_offsets() -> Option<crate::core::callret::CallRetOffsets> {
        None
    }

    /// Host-frame depth field, or `None`, which makes [`LifterConfig::host_call_frames`] inert.
    fn host_frame_offsets() -> Option<crate::core::hostframe::HostFrameOffsets> {
        None
    }

    /// Cranelift type of pinned `reg`, or `None` to refuse it; any `None` disables pinning.
    /// Must agree with [`pinned_slot_var`](GuestArch::pinned_slot_var).
    fn pinned_slot_type(_reg: PinnedReg, _config: &LifterConfig) -> Option<types::Type> {
        None
    }

    /// SSA variable holding pinned `reg`, or `None` if this guest doesn't pin it.
    fn pinned_slot_var(_state: &Self::RegState, _reg: PinnedReg) -> Option<Variable> {
        None
    }

    /// Context offset the entry thunk loads and stores pinned `reg` at.
    fn pinned_slot_offset(_reg: PinnedReg) -> Option<i32> {
        None
    }

    fn context_ptr_var(state: &Self::RegState) -> Variable;
    fn image_base_var(state: &Self::RegState) -> Variable;

    fn new_state(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self::RegState;

    /// Constant-offset `RegisterContext` fields as `(offset, name)`.
    fn context_fields(_config: &LifterConfig) -> Vec<(i32, &'static str)> {
        Vec::new()
    }

    fn decode_region(bytes: &[u8], ip: u64) -> Vec<Self::Instr>;
    /// `None` only when `bytes` is too short; invalid encodings return an instruction
    /// for which [`is_invalid`](GuestArch::is_invalid) is true.
    fn decode_one(bytes: &[u8], ip: u64) -> Option<Self::Instr>;
    fn instr_ip(i: &Self::Instr) -> u64;
    fn instr_next_ip(i: &Self::Instr) -> u64;
    fn is_invalid(i: &Self::Instr) -> bool;
    fn control_flow(i: &Self::Instr) -> ControlFlow;

    /// Register/memory effects, or `None` if the frontend can't compute them.
    fn instruction_effects(_i: &Self::Instr) -> Option<InstructionEffects> {
        None
    }

    /// Legacy flag metadata; dead flag elimination does not use it.
    fn flag_effects(_i: &Self::Instr) -> Option<FlagEffects> {
        None
    }

    /// Resolve indirect `JMP`/`CALL` targets via value-set analysis.
    fn resolved_indirect_targets(
        _bytes: &[u8],
        _base: u64,
        _segments: &[crate::context::MemorySegment],
    ) -> crate::cfg::ResolvedTargets {
        crate::cfg::ResolvedTargets::default()
    }

    /// **Experimental.** Jump-table successors keyed by branch address, consulted under
    /// [`LifterConfig::analyze_jump_tables`]. Only adds leaders: a wrong answer costs dead code.
    fn jump_table_targets(
        _bytes: &[u8],
        _base: u64,
        _segments: &[crate::context::MemorySegment],
    ) -> HashMap<u64, JumpTableInfo> {
        HashMap::new()
    }

    fn lift_instruction(lifter: &mut LiftEngine<'_, Self>, i: &Self::Instr) -> crate::Result<()>;
    fn load_state_from_context(lifter: &mut LiftEngine<'_, Self>) -> crate::Result<()>;
    fn sync_to_context(
        lifter: &mut LiftEngine<'_, Self>,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()>;
}

/// Helper emitted alongside the region ([`LiftEngine::outlined_helper`]).
///
/// Backends define it `Local` before the lifted function, in [`LiftedCode::helpers`]
/// order, so its `FuncId` matches `ExternalName::user(0, i)`.
#[derive(Debug, Clone)]
pub struct HelperFunction {
    /// Unique per helper body.
    pub name: &'static str,
    pub func: Function,
}

/// Read-only constant table ([`LiftEngine::data_constant`]).
///
/// Backends define it in [`LiftedCode::data_objects`] order, after the link-cell
/// array, matching `ExternalName::user(1, i)`.
#[derive(Debug, Clone)]
pub struct DataObject {
    /// Unique per contents.
    pub name: &'static str,
    pub bytes: Vec<u8>,
    pub align: u64,
}

/// One lift's output.
///
/// `ExternalName::user(0, i)` indexes `imports` then `helpers`; backends declare in
/// that order so `FuncId(i) == i`.
pub struct LiftedCode {
    pub func: Function,
    /// Host symbols for `RuntimeAssisted` fast imports; same as [`LiftEngine::aot_imports`].
    pub imports: Vec<(&'static str, *const u8, Signature)>,
    /// Local helper bodies, e.g. outlined x87 kernels.
    pub helpers: Vec<HelperFunction>,
    /// Analysis-mode call targets; see [`lift_to_function_with_call_targets`].
    pub analysis_call_targets: Vec<u64>,
    /// Decodes the `SourceLoc`s in `func`; meaningful only with [`LifterConfig::emit_source_locs`].
    pub source_map: SourceLocMap,
    /// Every guest address the code can be entered at, ascending (the dispatch arms).
    ///
    /// Key a host cache on exactly this: other entries rerun from the top or exit with
    /// `NoResumePoint` under [`LifterConfig::strict_resume`]. Empty under `analysis_mode`.
    pub resume_points: Vec<u64>,
    /// Entry thunk of a **pinned** region, `None` when `func` is host-callable.
    ///
    /// `func` is then the tail-convention body; backends declare it after the helpers
    /// and export the thunk as the entry.
    pub entry_thunk: Option<Function>,
    /// Guest bytes the translation derives from, as lift-time VAs, ascending and disjoint.
    ///
    /// **The invalidation key:** a write in these ranges invalidates the translation.
    /// Over-approximates decoded instructions ([`SourceLocMap::guest_addresses`] is a
    /// subset, so keying on it would miss writes). See [`crate::core::link`].
    pub covered: Vec<Range<u64>>,
    /// Bytes outside `covered` read at lift time (today: [`LifterConfig::analyze_jump_tables`]
    /// tables). Staleness is benign: it only adds leaders and dispatch stays on the runtime address.
    pub data_dependencies: Vec<Range<u64>>,
    /// Link cells in order: cell `i` is `link_sites[i]` at byte `i * 32`. Empty unless
    /// [`LifterConfig::link_cells`]. A host installs a [`BlockLinkEntry`](crate::core::link::BlockLinkEntry)
    /// for the site's target and retracts it by zeroing `code`.
    pub link_sites: Vec<crate::core::link::LinkSite>,
    /// Read-only tables in `ExternalName::user(1, i)` order; `i` starts at 1 when the region has cells.
    pub data_objects: Vec<DataObject>,
}

impl LiftedCode {
    /// Wrap a bare [`Function`], reporting no imports, helpers, resume points or extent.
    ///
    /// Hosts needing resume points or an invalidation key must use [`lift_to_code`].
    pub fn from_function(func: Function) -> Self {
        Self {
            func,
            imports: Vec::new(),
            helpers: Vec::new(),
            analysis_call_targets: Vec::new(),
            source_map: SourceLocMap::default(),
            resume_points: Vec::new(),
            entry_thunk: None,
            covered: Vec::new(),
            data_dependencies: Vec::new(),
            link_sites: Vec::new(),
            data_objects: Vec::new(),
        }
    }

    /// [`from_function`](Self::from_function) with fast-call imports; index `i` is `ExternalName::user(0, i)`.
    pub fn from_function_with_imports(
        func: Function,
        imports: Vec<(&'static str, *const u8, Signature)>,
    ) -> Self {
        Self {
            imports,
            ..Self::from_function(func)
        }
    }
}

/// The pinned ABI this lift runs under, or `None`.
///
/// The one place the question is asked. Falls back silently to `HostResume` unless:
/// linking is `Pinned`, the ABI is non-empty, and the guest has link fields and types every slot.
pub fn pinned_abi_for<A: GuestArch>(config: &LifterConfig) -> Option<PinnedAbi> {
    if config.analysis_mode {
        return None;
    }
    A::block_link_offsets()?;
    let abi = config.block_link?.pinned_abi()?;
    abi.slots()
        .iter()
        .all(|&reg| A::pinned_slot_type(reg, config).is_some())
        .then_some(abi)
}

fn pinned_slot_types<A: GuestArch>(abi: &PinnedAbi, config: &LifterConfig) -> Vec<types::Type> {
    abi.slots()
        .iter()
        .map(|&reg| {
            A::pinned_slot_type(reg, config).expect("pinned_abi_for typed every slot already")
        })
        .collect()
}

/// Entry thunk for a pinned region: loads pinned registers from the context, calls the
/// body, stores the results back. `body_index` follows the imports and helpers.
fn build_entry_thunk<A: GuestArch>(
    abi: &PinnedAbi,
    slot_types: &[types::Type],
    body_sig: Signature,
    body_index: u32,
) -> Function {
    let mut sig = Signature::new(get_host_calling_conv());
    sig.params.push(AbiParam::new(types::I64));
    let mut func = Function::with_name_signature(UserFuncName::user(0, body_index + 1), sig);

    let name_ref = func.declare_imported_user_function(UserExternalName {
        namespace: 0,
        index: body_index,
    });

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);

    let sig_ref = builder.import_signature(body_sig);
    let callee = builder.import_function(ExtFuncData {
        name: ExternalName::user(name_ref),
        signature: sig_ref,
        colocated: false,
        patchable: false,
    });

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let ctx_ptr = builder.block_params(entry)[0];
    let flags = MemFlagsData::trusted().with_can_move();
    let mut args = vec![ctx_ptr];
    for (&reg, &ty) in abi.slots().iter().zip(slot_types) {
        let offset =
            A::pinned_slot_offset(reg).expect("a guest that types a pinned slot must place it");
        args.push(builder.ins().load(ty, flags, ctx_ptr, offset));
    }

    let call = builder.ins().call(callee, &args);
    // Result 0 is the context pointer (see `core::pin`); store via the thunk's own pointer.
    let results: Vec<Value> = builder.inst_results(call)[1..].to_vec();
    for (&reg, value) in abi.slots().iter().zip(results) {
        let offset = A::pinned_slot_offset(reg).expect("placed above");
        builder.ins().store(flags, value, ctx_ptr, offset);
    }
    builder.ins().return_(&[]);
    // Only stack-map layout consults this; the thunk has none.
    builder.finalize(isa::TargetFrontendConfig {
        default_call_conv: get_host_calling_conv(),
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
    func
}

/// Lift a region to a [`Function`] plus its external references.
///
/// The complete form of [`lift_to_function`]; x87 helpers come back in [`LiftedCode::helpers`].
pub fn lift_to_code<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<LiftedCode> {
    // Pinned regions are `tail` functions taking the pinned registers; both start with the context pointer.
    let pinned = pinned_abi_for::<A>(&config);
    let slot_types = pinned
        .map(|abi| pinned_slot_types::<A>(&abi, &config))
        .unwrap_or_default();
    let signature = match pinned {
        Some(_) => pin::pinned_signature(&slot_types),
        None => {
            let mut sig = Signature::new(get_host_calling_conv());
            sig.params.push(AbiParam::new(types::I64));
            sig
        }
    };

    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), signature.clone());

    let mut builder_ctx = FunctionBuilderContext::new();
    let builder = FunctionBuilder::new(&mut func, &mut builder_ctx);

    let mut engine = LiftEngine::<A>::with_config(builder, config);
    engine.lift(bytes, ip)?;
    let analysis_call_targets = std::mem::take(&mut engine.analysis_call_targets);
    let imports = std::mem::take(&mut engine.aot_imports);
    let source_map = engine.source_map;
    let resume_points = std::mem::take(&mut engine.emitted_resume_points);
    let covered = std::mem::take(&mut engine.covered);
    let link_sites = std::mem::take(&mut engine.link_sites);
    let data_objects = engine.take_data_objects();
    let data_dependencies = engine.jump_table_extents();
    let helpers = engine.take_helpers();

    drop(engine);

    // Body index follows the imports and helpers, whose counts are final only now.
    let entry_thunk = pinned.map(|abi| {
        let body_index = (imports.len() + helpers.len()) as u32;
        build_entry_thunk::<A>(&abi, &slot_types, signature, body_index)
    });

    Ok(LiftedCode {
        func,
        imports,
        helpers,
        analysis_call_targets,
        source_map,
        resume_points,
        entry_thunk,
        covered,
        data_dependencies,
        link_sites,
        data_objects,
    })
}

/// Lift a region straight to a [`Function`]; errors if outlined helpers were
/// produced (use [`lift_to_code`]).
///
/// ```
/// use microlift::arch::x86_64::X86_64;
/// use microlift::core::{lift_to_function, LifterConfig};
///
/// // mov rax, 42 ; ret
/// let code = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
/// let func = lift_to_function::<X86_64>(&code, 0x1000, LifterConfig::default()).unwrap();
/// assert!(func.display().to_string().contains("function"));
/// ```
pub fn lift_to_function<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<Function> {
    Ok(lift_to_function_with_call_targets::<A>(bytes, ip, config)?.0)
}

/// Reduce to the main function, erroring if that would drop helper bodies.
fn require_no_helpers(code: LiftedCode) -> crate::Result<(Function, Vec<u64>)> {
    if code.entry_thunk.is_some() {
        return Err(crate::MicroliftError::CraneliftError(
            "a lift with a pinned register file produces a region body plus the entry thunk \
             that calls it, which a bare `Function` cannot carry - use `lift_to_code`"
                .into(),
        ));
    }
    if !code.helpers.is_empty() {
        let names: Vec<&str> = code.helpers.iter().map(|h| h.name).collect();
        return Err(crate::MicroliftError::CraneliftError(format!(
            "lifted region calls outlined helper functions ({}), which a bare \
             `Function` cannot carry - use `lift_to_code` to get them, or set \
             `LifterConfig::outline_helpers = false` to inline them instead",
            names.join(", ")
        )));
    }
    Ok((code.func, code.analysis_call_targets))
}

/// Like [`lift_to_function`], also returning the analysis-mode call-target map
/// ([`LiftEngine::analysis_call_targets`]); empty unless `config.analysis_mode`.
pub fn lift_to_function_with_call_targets<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<(Function, Vec<u64>)> {
    require_no_helpers(lift_to_code::<A>(bytes, ip, config)?)
}

/// `None` for void and `VariadicList`.
pub(crate) fn fat_to_cl_type(ty: FunctionArgumentType) -> Option<types::Type> {
    match ty {
        FunctionArgumentType::None | FunctionArgumentType::VariadicList => None,
        FunctionArgumentType::SignedByte | FunctionArgumentType::UnsignedByte => Some(types::I8),
        FunctionArgumentType::SignedHalfword | FunctionArgumentType::UnsignedHalfword => {
            Some(types::I16)
        }
        FunctionArgumentType::SignedWord | FunctionArgumentType::UnsignedWord => Some(types::I32),
        FunctionArgumentType::SignedDoubleword
        | FunctionArgumentType::UnsignedDoubleword
        | FunctionArgumentType::Pointer => Some(types::I64),
        FunctionArgumentType::SinglePrecisionFp => Some(types::F32),
        FunctionArgumentType::DoublePrecisionFp => Some(types::F64),
    }
}

/// Signature from a `DynamicSymbol`'s `arg_types` (`[0]` return, `[1..]` params),
/// stopping at the first `VariadicList`/`None`.
pub(crate) fn build_import_sig(
    arg_types: &[FunctionArgumentType],
    call_conv: CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);
    if let Some(ret_ty) = arg_types.first().copied().and_then(fat_to_cl_type) {
        sig.returns.push(AbiParam::new(ret_ty));
    }
    for &ty in arg_types.iter().skip(1) {
        match fat_to_cl_type(ty) {
            Some(param_ty) => sig.params.push(AbiParam::new(param_ty)),
            None => break,
        }
    }
    sig
}

/// Build the [`FastCall`] for one fast import, declaring its signature (and, in
/// `RuntimeAssisted`, its external function and `aot_imports` entry).
///
/// `None` when it can't be fast-called; the caller falls back to the slow exit.
/// Keeps one `FuncRef` per symbol name.
fn build_fast_call(
    builder: &mut FunctionBuilder,
    execution_mode: ExecutionMode,
    aot_imports: &mut Vec<(&'static str, *const u8, Signature)>,
    fast_call_by_name: &mut HashMap<&'static str, FastCall>,
    import: &FastImport,
) -> Option<FastCall> {
    let sym = &import.symbol;
    if sym.is_ellipsis_function || !import.abi.can_fast_call(&sym.arg_types) {
        return None;
    }
    // The signature is the host callee; the ABI is where the guest left arguments.
    let host_cc = import
        .abi
        .host_call_conv
        .unwrap_or_else(get_host_calling_conv);
    Some(match execution_mode {
        ExecutionMode::Jit => {
            let sig = build_import_sig(&sym.arg_types, host_cc);
            FastCall {
                target: FastCallTarget::Absolute {
                    sig_ref: builder.import_signature(sig),
                    fn_ptr: sym.value as i64,
                },
                arg_types: sym.arg_types.clone(),
                abi: import.abi.clone(),
            }
        }
        ExecutionMode::RuntimeAssisted => match fast_call_by_name.get(sym.name) {
            Some(existing) => existing.clone(),
            None => {
                let index = aot_imports.len() as u32;
                let sig = build_import_sig(&sym.arg_types, host_cc);
                let sig_ref = builder.import_signature(sig.clone());
                let user_name = builder
                    .func
                    .declare_imported_user_function(UserExternalName {
                        namespace: 0,
                        index,
                    });
                let func_ref = builder.import_function(ExtFuncData {
                    name: ExternalName::user(user_name),
                    signature: sig_ref,
                    colocated: false,
                    patchable: false,
                });
                aot_imports.push((sym.name, sym.value as *const u8, sig));
                let fast_call = FastCall {
                    target: FastCallTarget::Relocatable(func_ref),
                    arg_types: sym.arg_types.clone(),
                    abi: import.abi.clone(),
                };
                fast_call_by_name.insert(sym.name, fast_call.clone());
                fast_call
            }
        },
    })
}

/// A fast-call import: host function plus the guest convention of its call sites.
#[derive(Debug, Clone)]
pub struct FastImport {
    pub symbol: DynamicSymbol,
    pub abi: CallAbi,
}

#[derive(Debug, Clone, Copy)]
pub enum FastCallTarget {
    /// `Jit`: `call_indirect` through the absolute address.
    Absolute { sig_ref: SigRef, fn_ptr: i64 },
    /// `RuntimeAssisted`: `call` a declared external so the object backend emits a relocation.
    Relocatable(FuncRef),
}

#[derive(Debug, Clone)]
pub struct FastCall {
    pub target: FastCallTarget,
    /// `[0]` is the return type; `[1..]` the parameter types.
    pub arg_types: Vec<FunctionArgumentType>,
    pub abi: CallAbi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Jit,
    RuntimeAssisted,
}

/// How lifted code materializes lift-time guest VAs (RIP-relative operands, exit
/// PCs, return addresses, dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    /// `image_base + (va - lift_base)`: relocatable, so compiled code can be cached
    /// across runs. Required by `RuntimeAssisted`.
    #[default]
    ImageBaseRelative,
    /// Baked-in `iconst` VAs, valid only at the lift base; for JITs that re-lift each run.
    /// `RegisterContext::image_base` is ignored and exit PCs are plain lift-time VAs.
    Direct,
}

/// How `CMPXCHG16B` is realized.
///
/// Cranelift lowers `I128` `atomic_cas` only on x64 with `has_cmpxchg16b` (off in
/// the baseline feature set; AOT callers pass it via `isa_settings`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Atomic128Mode {
    /// Plain 128-bit load/compare/store: portable, **not atomic**. Default.
    #[default]
    Emulated,
    /// `atomic_cas` on `I128` -> `lock cmpxchg16b`. The compile **fails** on backends
    /// that can't lower it, or on x64 without `has_cmpxchg16b`.
    Native,
    /// Hand it to the host's interpreter callback: portable and concurrency-safe, one
    /// call per execution.
    ///
    /// Follows the [`crate::core::callback`] contract: state synced, PC *at* the `CMPXCHG16B`.
    Interpret,
}

/// Bit pattern of a *generated* NaN (from an invalid operation).
///
/// x86 yields the floating indefinite (`0xFFC0_0000` single); AArch64's default is
/// `0x7FC0_0000`. Propagated NaNs are untouched either way. Exactness costs about
/// six instructions on every FP operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneratedNan {
    /// Materialise x86's floating indefinite; bit-exact. Default.
    #[default]
    X86Exact,
    /// Keep the host FPU's default NaN.
    ///
    /// Correct unless the guest inspects a generated NaN's sign or raw bits.
    /// Bit-identical on x86-64 hosts.
    HostDefault,
}

/// Which addresses the entry dispatch offers for re-entry.
///
/// The main lever on compile time of large regions: each arm makes every register
/// live into it (128 KiB of `__text`: 1.42s per leader vs 19.36s per instruction).
/// Pick the smallest mode covering everywhere the host can leave from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VmResumptionPoints {
    /// No dispatch: enterable only at the entry, so entering elsewhere *silently* runs
    /// from the top. For hosts that translate a fresh region at every exit PC.
    None,
    /// **Default.** Arms for the entry and every point control leaves and returns to:
    /// call return points and resumable VM exits ([`LiftEngine::resumes_at`]). Add
    /// host-known addresses via [`LifterConfig::resume_points`].
    #[default]
    ExitOnly,
    /// Legacy: every instruction boundary. Correct for any re-entry, slow to compile.
    All,
}

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    /// How guest VAs are materialized. Default `ImageBaseRelative` (relocatable artifacts).
    pub address_mode: AddressMode,
    /// Immutable segments at lift time, letting VSA fold loads to constants.
    ///
    /// **Never list writable data.** `Arc` because cloning per region deep-copied
    /// images (311 us/region); mutate via [`set_segments`](Self::set_segments).
    pub segments: Arc<Vec<MemorySegment>>,
    pub interpreter_fallback: bool,
    /// Addresses the region can be re-entered at. See [`VmResumptionPoints`].
    pub resumption: VmResumptionPoints,
    /// **Experimental.** Recover `JMP reg` successors from compiler jump tables
    /// (read from [`segments`](Self::segments)) so case blocks become leaders.
    ///
    /// Targets are only added as leaders; dispatch stays on the runtime address,
    /// so a bad guess lifts dead code, never wrong code.
    pub analyze_jump_tables: bool,
    /// Statically registered imports, VA -> [`ImportKind`]. Snapshot at config time.
    ///
    /// `Arc` because the map is shared across per-region clones (478 ms of a 935 ms
    /// lift budget). Mutate via [`register_fast_import`](Self::register_fast_import) etc.
    pub imports: Arc<HashMap<u64, ImportKind>>,
    /// Live import lookup queried per call site; consulted only after [`imports`](Self::imports).
    pub import_registry: Option<Arc<dyn ImportRegistry>>,
    /// Bridge for indirect calls to unregistered targets, instead of an `IndirectCall` exit.
    ///
    /// Signature `extern "C" fn(*mut RegisterContext, target: u64) -> u32`:
    /// 0 = handled inline, non-zero = real exit (context already holds a valid
    /// `IndirectCall` exit; the bridge may overwrite `exit_reason`/`exit_code`).
    pub indirect_call_bridge: Option<BridgeImport>,
    /// Native `I64X2` XMM variables and vector fast paths instead of scalar lo/hi pairs.
    ///
    /// Off because AOT targets without SIMD (e.g. RISC-V) can't lower vectors.
    pub use_vector_xmm: bool,
    /// Target-specific lowerings. Never changes results, only speed.
    pub specialization: Specialization,
    /// How `CMPXCHG16B` is lowered. See [`Atomic128Mode`].
    pub atomic128: Atomic128Mode,
    /// Bit pattern of SSE-generated NaNs. See [`GeneratedNan`].
    pub generated_nan: GeneratedNan,
    /// Inspection IR for static analysis: **never execute it.**
    ///
    /// Dispatch becomes a direct jump, no context write-back, exits are bare `return`s.
    pub analysis_mode: bool,
    /// Guest return address meaning "program done, back to host"; `None` if the host has none.
    ///
    /// Costs a second exit block per out-of-region `RET`. x86-64 only.
    pub exit_trampoline_address: Option<u64>,
    /// Emit shared sequences (x87 softfloat kernels) once as helpers and `call` them.
    ///
    /// Off yields one self-contained `Function` (required by [`lift_to_function`]).
    /// Ignored under [`analysis_mode`](Self::analysis_mode).
    pub outline_helpers: bool,
    /// Tag each IR instruction with its guest VA as a `SourceLoc` ([`crate::core::srcloc`]).
    pub emit_source_locs: bool,
    /// Skip context loads/stores for registers the region never writes ([`crate::core::regsync`]).
    pub lazy_register_sync: bool,
    /// Exit instead of decoding when an instruction runs past the end of the region.
    pub region_tail_exit: bool,
    /// Page size for [`page_bounded_extent`](Self::page_bounded_extent).
    pub page_size: u64,
    /// Sub-ranges of the slice that are code. Empty means all of it.
    pub block_ranges: Vec<Range<u64>>,
    /// Extra entry-dispatch addresses, each forced to be a block leader.
    ///
    /// Out-of-region addresses are ignored; nothing is offered under `VmResumptionPoints::None`.
    pub resume_points: Vec<u64>,
    /// Remove unused flag computations from the finished SSA graph.
    ///
    /// Frontends opt in via [`LiftEngine::record_flag_value`]. Needed by `Spec::IntegerOverflowFlags`.
    pub eliminate_dead_flags: bool,
    /// Exit with `TransferKind::NoResumePoint` when the entry PC matches no resume point,
    /// instead of silently running from the region's first block.
    ///
    /// Leaves PC and guest state untouched. Inert under `VmResumptionPoints::None`.
    pub strict_resume: bool,
    /// Resolve successors from a host link table instead of exiting. Off by default.
    /// See [`crate::core::link`].
    pub block_link: Option<BlockLinkConfig>,
    /// Emit the probe at each exit instead of once per region. Off: code size.
    pub inline_link_probes: bool,
    /// Exits with a lift-time-known target get a dedicated [`LinkSite`] cell
    /// instead of hashing into the shared table. Off by default.
    pub link_cells: bool,
    /// Predict `RET`s from a ring of `<return address, cell>` frames. Off by default.
    ///
    /// Needs [`link_cells`](Self::link_cells), [`block_link`](Self::block_link) and
    /// [`GuestArch::call_return_offsets`], or it is inert.
    pub call_return_stack: Option<crate::core::callret::CallRetConfig>,
    /// Make guest `CALL`/`RET` a real host call/return. Off by default.
    ///
    /// Alternative to [`call_return_stack`](Self::call_return_stack). Needs
    /// [`link_cells`](Self::link_cells) and [`GuestArch::host_frame_offsets`], or it is inert.
    pub host_call_frames: Option<crate::core::hostframe::HostFrameConfig>,
    /// Emitted-code optimizations; some trade rarely-relied-on accuracy such as NaN payloads.
    pub optimizations: Optimizations,
}

// `Arc` is deliberate: a host may move a config across threads under its own safety argument.
#[allow(clippy::arc_with_non_send_sync)]
impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            address_mode: AddressMode::ImageBaseRelative,
            segments: Arc::new(Vec::new()),
            interpreter_fallback: false,
            resumption: VmResumptionPoints::default(),
            analyze_jump_tables: false,
            imports: Arc::new(HashMap::new()),
            import_registry: None,
            indirect_call_bridge: None,
            use_vector_xmm: false,
            specialization: Specialization::generic(),
            atomic128: Atomic128Mode::default(),
            generated_nan: GeneratedNan::default(),
            analysis_mode: false,
            exit_trampoline_address: None,
            outline_helpers: true,
            emit_source_locs: true,
            lazy_register_sync: true,
            region_tail_exit: false,
            page_size: 4096,
            block_ranges: Vec::new(),
            resume_points: Vec::new(),
            strict_resume: false,
            eliminate_dead_flags: false,
            block_link: None,
            inline_link_probes: false,
            link_cells: false,
            call_return_stack: None,
            host_call_frames: None,
            optimizations: Optimizations::none(),
        }
    }
}

impl LifterConfig {
    pub fn set_segments(&mut self, segments: Vec<MemorySegment>) {
        self.segments = Arc::new(segments);
    }

    /// Register an import with fast-call metadata; `abi` is the *guest's* convention
    /// (e.g. [`CallAbi::sysv_amd64`](crate::core::abi::CallAbi::sysv_amd64)).
    ///
    /// Demoted to a slow exit when variadic or unplaceable.
    ///
    /// The callee must not touch the `RegisterContext` (UB): a fast call runs with state
    /// in machine registers. Use a [bridge](crate::core::bridge) or
    /// [slow import](Self::register_slow_import) instead.
    pub fn register_fast_import(&mut self, va: u64, symbol: DynamicSymbol, abi: CallAbi) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Fast(FastImport { symbol, abi }));
    }

    /// Register an import that always takes an `ImportedFunctionCall` VM exit.
    pub fn register_slow_import(&mut self, va: u64) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Slow);
    }

    /// Register an import reached through a bridge; see [`crate::core::bridge`].
    ///
    /// More general than a fast import and cheaper than a slow exit. The bridge
    /// address is baked in, so AOT output is valid only where it maps at that address.
    pub fn register_bridge_import(&mut self, va: u64, bridge: BridgeImport) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Bridge(bridge));
    }

    pub fn set_import_registry(&mut self, registry: Arc<dyn ImportRegistry>) {
        self.import_registry = Some(registry);
    }

    pub fn set_indirect_call_bridge(&mut self, bridge: BridgeImport) {
        self.indirect_call_bridge = Some(bridge);
    }

    /// Extent to lift so `pc` can execute, cut on page boundaries instead of following control flow.
    ///
    /// - **Monotone in what the guest wrote:** a page boundary makes no claim about the bytes.
    /// - **Terminates:** spans `pc` plus one maximal instruction, running into the next page near a page end.
    ///
    /// Clamped to `available`; empty when `pc` is outside it. Pair with
    /// [`region_tail_exit`](Self::region_tail_exit).
    pub fn page_bounded_extent<A: GuestArch>(
        &self,
        pc: u64,
        available: core::ops::Range<u64>,
    ) -> core::ops::Range<u64> {
        if !available.contains(&pc) {
            return pc..pc;
        }
        let page = self
            .page_size
            .max(A::MAX_INSTRUCTION_LEN.next_power_of_two());
        // Round up from the last byte the entry instruction could occupy.
        let last = pc.saturating_add(A::MAX_INSTRUCTION_LEN - 1);
        let end = (last / page).saturating_add(1).saturating_mul(page);
        pc..end.min(available.end)
    }
}

/// [`LifterConfig::block_ranges`] clamped to `[base, end)`, sorted, and merged where
/// they touch or overlap; empty input means the whole slice.
///
/// Touching ranges must merge: splitting them would cut an instruction spanning the boundary.
fn normalize_ranges(ranges: &[Range<u64>], base: u64, end: u64) -> Vec<Range<u64>> {
    if ranges.is_empty() {
        // One range, not an address iterator.
        #[allow(clippy::single_range_in_vec_init)]
        return vec![base..end];
    }
    let mut clamped: Vec<Range<u64>> = ranges
        .iter()
        .map(|r| r.start.max(base)..r.end.min(end))
        .filter(|r| r.start < r.end)
        .collect();
    clamped.sort_unstable_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(clamped.len());
    for r in clamped {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

/// The generic lift engine. `A` selects the guest architecture frontend.
pub struct LiftEngine<'a, A: GuestArch = crate::arch::x86_64::X86_64> {
    pub builder: FunctionBuilder<'a>,
    pub state: A::RegState,
    pub config: LifterConfig,
    pub blocks: HashMap<u64, Block>,
    pub sealed_blocks: HashSet<Block>,
    pub lift_base: u64,
    pub lift_end: u64,
    /// Bytes of `[lift_base, lift_end)` this lift covers: normalised [`LifterConfig::block_ranges`]
    /// or the whole slice. Ask [`in_region`](Self::in_region); with holes the bounds aren't enough.
    covered: Vec<Range<u64>>,
    /// `next_ip` of the instruction being lifted.
    pub current_ip: u64,
    pub visited_blocks: HashSet<Block>,
    pub block_terminated: bool,
    /// Condition-flag values before dead-code removal; seeds the candidate graph.
    flag_values: Vec<Value>,
    callback_sigs: HashMap<usize, SigRef>,
    /// Fast-call data for non-variadic imports by VA, filled lazily.
    pub fast_calls: HashMap<u64, FastCall>,
    /// One [`FastCall`] per symbol *name* (a PLT stub and its GOT slot share one); a
    /// duplicate would break `FuncId(i) == i` in the object backend.
    fast_call_by_name: HashMap<&'static str, FastCall>,
    bridge_sig: Option<SigRef>,
    /// External functions; index `i` is `ExternalName::user(0, i)`.
    pub aot_imports: Vec<(&'static str, *const u8, Signature)>,
    /// Transfer kind of each unvisited block, for `finish()`.
    pub block_transfer_kinds: HashMap<Block, TransferKind>,
    /// Analysis mode: callee VA per `call`, indexed by `ExternalName::user(ANALYSIS_CALL_NAMESPACE, i)`;
    /// [`u64::MAX`] for indirect calls.
    pub analysis_call_targets: Vec<u64>,
    analysis_call_sig: Option<SigRef>,
    /// Outlined helpers in first-call order; indices are provisional until [`Self::finish`].
    helpers: Vec<HelperEntry>,
    /// One call target per helper symbol.
    helper_refs: HashMap<&'static str, FuncRef>,
    /// Guest-address encoding for emitted `SourceLoc`s, set from `ip` in [`Self::lift`].
    /// See [`crate::core::srcloc`].
    pub source_map: SourceLocMap,
    /// Jump-table successors by branch address (empty unless [`LifterConfig::analyze_jump_tables`]); inspection only.
    pub jump_tables: HashMap<u64, JumpTableInfo>,
    /// Entry-dispatch arm addresses, ascending ([`LiftedCode::resume_points`]).
    pub emitted_resume_points: Vec<u64>,
    /// In-region addresses resumable VM exits continue at; see [`Self::resumes_at`].
    pub exit_resumes: HashSet<u64>,
    /// The one block-link probe, created on first use and filled in [`Self::finish`].
    /// Its parameter is the PC to look up, which a bridge may have moved off `ctx.rip`.
    link_block: Option<Block>,
    /// Pinned register file from [`pinned_abi_for`], with each slot's variable and type.
    /// Empty when unpinned, making the unpinned shape the degenerate case.
    pinned_slots: Vec<(PinnedReg, Variable, types::Type)>,
    /// Pinned-region signature shared by the link probe and tail calls.
    pinned_sig: Option<SigRef>,
    /// This region's own signature, for nested host calls ([`crate::core::hostframe`]).
    region_sig: Option<SigRef>,

    /// Per-field memflags (one alias region each), interned in [`Self::with_config`];
    /// reach them through [`Self::ctx_store`] / [`Self::ctx_load`].
    ctx_flags: HashMap<i32, MemFlagsData>,
    /// One region for the `TOP`-relative x87 stack.
    x87_flags: MemFlagsData,
    /// A field, not a method, so `builder.ins().load(.., self.guest_flags, ..)` borrow-checks.
    pub guest_flags: MemFlagsData,
    /// Host memory outside the context; today a rebindable bridge slot.
    pub host_slot_flags: MemFlagsData,
    link_table_flags: MemFlagsData,
    /// Separate region from the table: same row type, never the same memory.
    link_cell_flags: MemFlagsData,
    callret_flags: MemFlagsData,
    /// Link-cell array in cell order ([`LiftedCode::link_sites`]).
    link_sites: Vec<crate::core::link::LinkSite>,
    /// Cell index by target, so branches to one address share a cell.
    link_cell_by_target: HashMap<u64, usize>,
    /// Read-only tables in first-use order; indices provisional until [`Self::finish`].
    data_objects: Vec<DataEntry>,
    /// One `GlobalValue` per table symbol.
    data_refs: HashMap<&'static str, cranelift_codegen::ir::GlobalValue>,
    pub const_data_flags: MemFlagsData,

    /// Cell array base, created on first use so its address is materialized once.
    link_cell_base: Option<cranelift_codegen::ir::GlobalValue>,

    /// Every guest register defined during this lift (the [`crate::core::regsync`] write
    /// set); **never cleared**, since it answers "anywhere in the region".
    written_slots: HashSet<regsync::RegSlot>,
    /// Set by [`Self::keep_all_sync`] once the whole register file was reloaded; no store may be pruned then.
    sync_prune_disabled: bool,
    /// Full-sync stores with their register, in emission order; prune candidates.
    ctx_stores: Vec<(Inst, regsync::RegSlot)>,
    /// Entry-state loads; prune candidates decided by a use scan of the finished IR.
    ctx_loads: Vec<Inst>,
}

/// A constant table under construction; [`LiftEngine::finish`] rewrites its name index.
struct DataEntry {
    object: DataObject,
    name_ref: UserExternalNameRef,
}

/// A helper under construction; [`LiftEngine::finish`] rewrites its name index.
struct HelperEntry {
    name: &'static str,
    func: Function,
    name_ref: UserExternalNameRef,
}

/// Provisional namespace-0 helper index, far above any import count so distinct callees
/// can't merge in `declare_imported_user_function`; [`LiftEngine::finish`] renumbers.
const PROVISIONAL_HELPER_INDEX: u32 = 0x8000_0000;

/// Provisional namespace-1 index; index 0 is the link-cell array, whose presence is
/// unknown until the last exit stub.
const PROVISIONAL_DATA_INDEX: u32 = 0x8000_0000;

/// One recovered jump table in guest-agnostic terms, crossing [`GuestArch::jump_table_targets`].
/// `extent` travels with the targets because `targets` is de-duplicated.
#[derive(Debug, Clone)]
pub struct JumpTableInfo {
    /// Successors in table order, de-duplicated; contributed as block leaders.
    pub targets: Vec<u64>,
    /// Guest bytes the table occupies, which the translation now depends on.
    pub extent: Range<u64>,
}

/// Namespace of analysis-mode `call` externals, distinct from namespace 0 (AOT imports).
pub const ANALYSIS_CALL_NAMESPACE: u32 = 1;

impl<'a, A: GuestArch> LiftEngine<'a, A> {
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        Self::with_config(builder, LifterConfig::default())
    }

    pub fn with_config(builder: FunctionBuilder<'a>, config: LifterConfig) -> Self {
        let mut builder = builder;
        let state = A::new_state(&mut builder, &config);
        let pinned_slots: Vec<(PinnedReg, Variable, types::Type)> = pinned_abi_for::<A>(&config)
            .map(|abi| {
                abi.slots()
                    .iter()
                    .map(|&reg| {
                        let var = A::pinned_slot_var(&state, reg)
                            .expect("a guest that types a pinned slot must have a variable for it");
                        let ty =
                            A::pinned_slot_type(reg, &config).expect("typed by pinned_abi_for");
                        (reg, var, ty)
                    })
                    .collect()
            })
            .unwrap_or_default();
        // One alias region per context field plus the whole-object ones, interned up front.
        let mut ctx_flags: HashMap<i32, MemFlagsData> = HashMap::new();
        for (offset, name) in A::context_fields(&config) {
            let flags = alias::region_flags(builder.func, offset as u32, name);
            // The insert must not live in `debug_assert!`: it compiles out in release.
            let previous = ctx_flags.insert(offset, flags);
            debug_assert!(
                previous.is_none(),
                "two RegisterContext fields claim offset {offset} ({name}); \
                 an address may belong to at most one alias region"
            );
        }
        let x87_flags = alias::region_flags(builder.func, alias::X87_STACK_ID, "x87 stack");
        let host_slot_flags =
            alias::region_flags(builder.func, alias::HOST_SLOT_ID, "host bridge slot");
        let link_table_flags =
            alias::region_flags(builder.func, alias::LINK_TABLE_ID, "block-link table");
        let link_cell_flags = alias::region_flags(builder.func, alias::LINK_CELL_ID, "link cells");
        let callret_flags =
            alias::region_flags(builder.func, alias::CALLRET_ID, "call-return ring");
        let const_data_flags =
            alias::region_flags(builder.func, alias::CONST_DATA_ID, "constant tables")
                .with_readonly();
        let guest_flags = alias::guest_memory_flags(builder.func);

        // Built lazily by `fast_call_for`; eager building cost the whole import table per region.
        let fast_calls: HashMap<u64, FastCall> = HashMap::new();
        let aot_imports: Vec<(&'static str, *const u8, Signature)> = Vec::new();
        let fast_call_by_name: HashMap<&'static str, FastCall> = HashMap::new();

        Self {
            builder,
            state,
            config,
            blocks: HashMap::new(),
            sealed_blocks: HashSet::new(),
            lift_base: 0,
            lift_end: 0,
            covered: Vec::new(),
            current_ip: 0,
            visited_blocks: HashSet::new(),
            block_terminated: false,
            flag_values: Vec::new(),
            callback_sigs: HashMap::new(),
            fast_calls,
            fast_call_by_name,
            bridge_sig: None,
            aot_imports,
            block_transfer_kinds: HashMap::new(),
            analysis_call_targets: Vec::new(),
            analysis_call_sig: None,
            helpers: Vec::new(),
            helper_refs: HashMap::new(),
            source_map: SourceLocMap::default(),
            jump_tables: HashMap::new(),
            emitted_resume_points: Vec::new(),
            exit_resumes: HashSet::new(),
            link_block: None,
            pinned_slots,
            pinned_sig: None,
            region_sig: None,
            ctx_flags,
            x87_flags,
            guest_flags,
            host_slot_flags,
            link_table_flags,
            link_cell_flags,
            callret_flags,
            link_sites: Vec::new(),
            link_cell_by_target: HashMap::new(),
            link_cell_base: None,
            data_objects: Vec::new(),
            data_refs: HashMap::new(),
            const_data_flags,
            written_slots: HashSet::new(),
            sync_prune_disabled: false,
            ctx_stores: Vec::new(),
            ctx_loads: Vec::new(),
        }
    }

    /// Store into a context field, tagged with its alias region.
    ///
    /// # Panics
    /// If `offset` is not in [`GuestArch::context_fields`].
    pub fn ctx_store(&mut self, val: Value, ctx_ptr: Value, offset: i32) -> Inst {
        let flags = self.ctx_field_flags(offset);
        self.builder.ins().store(flags, val, ctx_ptr, offset)
    }

    /// Read half of [`Self::ctx_store`].
    pub fn ctx_load(&mut self, ty: types::Type, ctx_ptr: Value, offset: i32) -> Value {
        let flags = self.ctx_field_flags(offset);
        self.builder.ins().load(ty, flags, ctx_ptr, offset)
    }

    /// Memflags for the x87 register stack.
    pub fn x87_flags(&self) -> MemFlagsData {
        self.x87_flags
    }

    pub fn mark_written(&mut self, slot: regsync::RegSlot) {
        self.written_slots.insert(slot);
    }

    /// Disable the lazy-sync prune ([`crate::core::regsync`]) for the rest of the region.
    pub fn keep_all_sync(&mut self) {
        self.sync_prune_disabled = true;
    }

    pub fn slot_written(&self, slot: regsync::RegSlot) -> bool {
        self.written_slots.contains(&slot)
    }

    /// [`Self::ctx_store`] for a guest register in the *full* sync, recorded so the prune
    /// can drop it if `slot` is never written.
    ///
    /// Full sync only: masked syncs serve host calls made while the region runs and must always be emitted.
    pub fn ctx_store_reg(
        &mut self,
        slot: regsync::RegSlot,
        val: Value,
        ctx_ptr: Value,
        offset: i32,
    ) {
        let inst = self.ctx_store(val, ctx_ptr, offset);
        if self.lazy_register_sync() {
            self.ctx_stores.push((inst, slot));
        }
    }

    /// [`Self::ctx_load`] for the entry state, recorded so the prune can drop it.
    pub fn ctx_load_reg(&mut self, ty: types::Type, ctx_ptr: Value, offset: i32) -> Value {
        let val = self.ctx_load(ty, ctx_ptr, offset);
        if self.lazy_register_sync() {
            let inst = self.builder.func.dfg.value_def(val).unwrap_inst();
            self.ctx_loads.push(inst);
        }
        val
    }

    /// Off in analysis mode: that IR emits no write-back and its entry loads keep the register file visible.
    fn lazy_register_sync(&self) -> bool {
        self.config.lazy_register_sync && !self.config.analysis_mode
    }

    /// Drop the context traffic the region turned out not to need.
    fn prune_untouched_sync(&mut self) {
        if !self.lazy_register_sync() || self.sync_prune_disabled {
            return;
        }
        let stores = std::mem::take(&mut self.ctx_stores);
        let loads = std::mem::take(&mut self.ctx_loads);
        let written = std::mem::take(&mut self.written_slots);

        // A never-defined register writes back what the entry load read.
        for (inst, slot) in stores {
            if !written.contains(&slot) {
                self.builder.func.layout.remove_inst(inst);
            }
        }

        // Dropping stores, dead params and loads unlock each other, so run to a fixpoint.
        let mut loads = loads;
        loop {
            let dropped = self.drop_unused_ctx_loads(&mut loads);
            let unhooked = dce::remove_dead_block_params(&mut self.builder.func);
            if !dropped && !unhooked {
                break;
            }
        }
    }

    /// Remove recorded loads nothing reads. Safe because they are `notrap`; asks the IR,
    /// not a frontend read set.
    fn drop_unused_ctx_loads(&mut self, loads: &mut Vec<Inst>) -> bool {
        let func = &mut *self.builder.func;
        // Bitset, not `HashSet`: hashing `Value` showed up in profiles (see `crate::core::dce`).
        let mut used: EntitySet<Value> = EntitySet::with_capacity(func.dfg.num_values());
        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                for arg in func.dfg.inst_values(inst) {
                    used.insert(func.dfg.resolve_aliases(arg));
                }
            }
        }
        let mut dropped = false;
        loads.retain(|&inst| {
            if func.layout.inst_block(inst).is_none() {
                return false;
            }
            let dead = func
                .dfg
                .inst_results(inst)
                .iter()
                .all(|&v| !used.contains(func.dfg.resolve_aliases(v)));
            if dead {
                func.dfg.clear_results(inst);
                func.layout.remove_inst(inst);
                dropped = true;
            }
            !dead
        });
        dropped
    }

    fn ctx_field_flags(&self, offset: i32) -> MemFlagsData {
        match self.ctx_flags.get(&offset) {
            Some(&flags) => flags,
            None => panic!(
                "no alias region for RegisterContext offset {offset}; \
                 add the field to GuestArch::context_fields"
            ),
        }
    }

    /// Whether `reg` lives in a machine register across region boundaries.
    ///
    /// **Full sync and entry load only.** A masked sync must not consult it: the host
    /// reads the context while the region runs.
    pub fn is_pinned(&self, reg: PinnedReg) -> bool {
        self.pinned_slots.iter().any(|&(r, _, _)| r == reg)
    }

    pub fn pinned(&self) -> bool {
        !self.pinned_slots.is_empty()
    }

    /// Context pointer then every pinned slot in ABI order; empty when unpinned.
    fn pinned_call_args(&mut self) -> Vec<Value> {
        if self.pinned_slots.is_empty() {
            return Vec::new();
        }
        let mut args = Vec::with_capacity(self.pinned_slots.len() + 1);
        args.push(self.builder.use_var(A::context_ptr_var(&self.state)));
        let vars: Vec<Variable> = self.pinned_slots.iter().map(|&(_, v, _)| v).collect();
        for var in vars {
            args.push(self.builder.use_var(var));
        }
        args
    }

    /// Return to the caller (entry thunk, or the host directly when unpinned).
    ///
    /// **Every `return_` goes through here:** a bare one under a pinned ABI is a
    /// verifier error or a truncated register file.
    pub fn emit_host_return(&mut self) {
        let args = self.pinned_call_args();
        self.builder.ins().return_(&args);
    }

    /// Store every pinned register into the context **before host code runs inline**.
    ///
    /// Invariant: a pinned register is in the context whenever host code executes.
    /// No-op when unpinned.
    pub fn publish_pinned(&mut self) -> crate::Result<()> {
        if self.pinned_slots.is_empty() {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let slots = self.pinned_slots.clone();
        for (reg, var, _) in slots {
            let offset = A::pinned_slot_offset(reg).expect("a pinned slot has a context offset");
            let val = self.builder.use_var(var);
            self.ctx_store(val, ctx_ptr, offset);
        }
        Ok(())
    }

    /// Reload pinned registers from the context **after host code that may have written one**.
    ///
    /// Other half of [`Self::publish_pinned`]; needed where the host writes guest registers
    /// through the context, e.g. a bridge's exiting path. Untouched registers reload what was published.
    pub fn reload_pinned(&mut self) -> crate::Result<()> {
        if self.pinned_slots.is_empty() {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let slots = self.pinned_slots.clone();
        for (reg, var, ty) in slots {
            let offset = A::pinned_slot_offset(reg).expect("a pinned slot has a context offset");
            let val = self.ctx_load(ty, ctx_ptr, offset);
            self.builder.def_var(var, val);
        }
        Ok(())
    }

    /// Imported signature of a pinned region, for the probe's indirect tail call.
    fn pinned_signature_ref(&mut self) -> SigRef {
        match self.pinned_sig {
            Some(sig) => sig,
            None => {
                let types: Vec<types::Type> =
                    self.pinned_slots.iter().map(|&(_, _, ty)| ty).collect();
                let sig = self.builder.import_signature(pin::pinned_signature(&types));
                self.pinned_sig = Some(sig);
                sig
            }
        }
    }

    /// Attribute IR emitted from here on to guest instruction `va`.
    ///
    /// A mode switch: the builder stamps its current `SourceLoc` on everything it creates.
    /// No-op when [`LifterConfig::emit_source_locs`] is off, and a
    /// [`clear_source_loc`](Self::clear_source_loc) when `va` is too far to encode.
    pub fn set_source_loc(&mut self, va: u64) {
        if !self.config.emit_source_locs {
            return;
        }
        let loc = self.source_map.encode(va);
        self.builder.set_srcloc(loc);
    }

    /// Stop attributing IR to a guest instruction, for scaffolding (prologue, dispatch).
    pub fn clear_source_loc(&mut self) {
        self.builder.set_srcloc(SourceLoc::default());
    }

    /// Declare or reuse an outlined helper `name`, returning a `FuncRef` to `call`.
    ///
    /// The first call hands a fresh `Function` to `build`, which must fill it; later calls
    /// reuse it, so `name` must identify the *body*. Returned in [`LiftedCode::helpers`];
    /// its index is provisional until [`Self::finish`].
    pub fn outlined_helper(
        &mut self,
        name: &'static str,
        signature: Signature,
        build: impl FnOnce(&mut FunctionBuilder),
    ) -> FuncRef {
        if let Some(&func_ref) = self.helper_refs.get(name) {
            return func_ref;
        }

        // The index in the name is cosmetic; `finish` rewrites it.
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), signature.clone());
        let mut helper_ctx = FunctionBuilderContext::new();
        let mut helper_builder = FunctionBuilder::new(&mut func, &mut helper_ctx);
        build(&mut helper_builder);
        // Only stack maps consult the frontend config; helpers request none.
        helper_builder.finalize(isa::TargetFrontendConfig {
            default_call_conv: signature.call_conv,
            pointer_width: target_lexicon::PointerWidth::U64,
            page_size_align_log2: 12,
        });

        let sig_ref = self.builder.import_signature(signature);
        let name_ref = self
            .builder
            .func
            .declare_imported_user_function(UserExternalName {
                namespace: 0,
                index: PROVISIONAL_HELPER_INDEX + self.helpers.len() as u32,
            });
        let func_ref = self.builder.import_function(ExtFuncData {
            name: ExternalName::user(name_ref),
            signature: sig_ref,
            // Matches the fast-import path, whose relocations both backends resolve.
            colocated: false,
            patchable: false,
        });

        self.helpers.push(HelperEntry {
            name,
            func,
            name_ref,
        });
        self.helper_refs.insert(name, func_ref);
        func_ref
    }

    /// Base address of the read-only table `name`, emitted into [`LiftedCode::data_objects`] on first use.
    ///
    /// `name` identifies the contents; `build` runs once per lift. Loads belong in
    /// [`const_data_flags`](Self::const_data_flags). A relocated symbol, not a baked
    /// address, so the artifact survives caching (see `link_cell_base`).
    pub fn data_constant(
        &mut self,
        name: &'static str,
        align: u64,
        build: impl FnOnce() -> Vec<u8>,
    ) -> Value {
        let gv = match self.data_refs.get(name) {
            Some(&gv) => gv,
            None => {
                let name_ref = self
                    .builder
                    .func
                    .declare_imported_user_function(UserExternalName {
                        namespace: 1,
                        index: PROVISIONAL_DATA_INDEX + self.data_objects.len() as u32,
                    });
                let gv = self.builder.func.create_global_value(
                    cranelift_codegen::ir::GlobalValueData::Symbol {
                        name: ExternalName::user(name_ref),
                        offset: 0.into(),
                        // Far, for the same `adrp` range reason as the cell array.
                        colocated: false,
                        tls: false,
                    },
                );
                self.data_objects.push(DataEntry {
                    object: DataObject {
                        name,
                        bytes: build(),
                        align,
                    },
                    name_ref,
                });
                self.data_refs.insert(name, gv);
                gv
            }
        };
        self.builder.ins().symbol_value(types::I64, gv)
    }

    /// Take the finished tables; valid only after [`Self::finish`] assigned final indices.
    pub fn take_data_objects(&mut self) -> Vec<DataObject> {
        std::mem::take(&mut self.data_objects)
            .into_iter()
            .map(|d| d.object)
            .collect()
    }

    /// Give each table its final namespace-1 index. Index 0 is the link-cell array when
    /// the region has one, which is settled only after the exit stubs.
    fn finalize_data_indices(&mut self) {
        let base = !self.link_sites.is_empty() as u32;
        for (ordinal, data) in self.data_objects.iter().enumerate() {
            self.builder.func.params.reset_user_func_name(
                data.name_ref,
                UserExternalName {
                    namespace: 1,
                    index: base + ordinal as u32,
                },
            );
        }
    }

    /// Analysis IR always inlines helpers to stay one self-contained function.
    pub fn outline_helpers(&self) -> bool {
        self.config.outline_helpers && !self.config.analysis_mode
    }

    /// Guest bytes recovered jump tables occupy, normalized like `covered`
    /// ([`LiftedCode::data_dependencies`]).
    pub fn jump_table_extents(&self) -> Vec<Range<u64>> {
        let mut extents: Vec<Range<u64>> = self
            .jump_tables
            .values()
            .map(|t| t.extent.clone())
            .collect();
        extents.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(extents.len());
        for range in extents {
            match merged.last_mut() {
                // Tables can overlap (a table of tables), so coalesce.
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        merged
    }

    /// Take the finished helper bodies; valid only after [`Self::finish`] assigned final indices.
    pub fn take_helpers(&mut self) -> Vec<HelperFunction> {
        std::mem::take(&mut self.helpers)
            .into_iter()
            .map(|h| HelperFunction {
                name: h.name,
                func: h.func,
            })
            .collect()
    }

    /// Give each helper its final namespace-0 index after the AOT imports. Runs at the
    /// end because registry imports can append after a helper was declared.
    fn finalize_helper_indices(&mut self) {
        let base = self.aot_imports.len() as u32;
        for (ordinal, helper) in self.helpers.iter_mut().enumerate() {
            let index = base + ordinal as u32;
            self.builder.func.params.reset_user_func_name(
                helper.name_ref,
                UserExternalName {
                    namespace: 0,
                    index,
                },
            );
            helper.func.name = UserFuncName::user(0, index);
        }
    }

    pub fn analysis_call_sig(&mut self) -> SigRef {
        match self.analysis_call_sig {
            Some(sig) => sig,
            None => {
                let sig = self
                    .builder
                    .import_signature(Signature::new(get_host_calling_conv()));
                self.analysis_call_sig = Some(sig);
                sig
            }
        }
    }

    pub fn analysis_call_funcref(&mut self, callee_va: u64) -> FuncRef {
        let sig = self.analysis_call_sig();
        let index = self.analysis_call_targets.len() as u32;
        self.analysis_call_targets.push(callee_va);
        let user = self
            .builder
            .func
            .declare_imported_user_function(UserExternalName {
                namespace: ANALYSIS_CALL_NAMESPACE,
                index,
            });
        self.builder.import_function(ExtFuncData {
            name: ExternalName::user(user),
            signature: sig,
            colocated: false,
            patchable: false,
        })
    }

    /// How `va` is reached, or `None` if it is not an import.
    ///
    /// Config imports win over the registry, which is queried per call site so a lift
    /// after `dlopen` sees the new stubs.
    pub fn lookup_import(&self, va: u64) -> Option<ImportKind> {
        if let Some(kind) = self.config.imports.get(&va) {
            return Some(kind.clone());
        }
        self.config
            .import_registry
            .as_ref()
            .and_then(|r| r.lookup(va))
    }

    pub fn is_import(&self, va: u64) -> bool {
        self.config.imports.contains_key(&va)
            || self
                .config
                .import_registry
                .as_ref()
                .is_some_and(|r| r.lookup(va).is_some())
    }

    /// The [`FastCall`] for import `va`, built and cached on first use. `None` falls back
    /// to the slow exit (no metadata, variadic, or unplaceable arguments).
    pub fn fast_call_for(&mut self, va: u64) -> Option<FastCall> {
        if let Some(fc) = self.fast_calls.get(&va) {
            return Some(fc.clone());
        }
        let ImportKind::Fast(import) = self.lookup_import(va)? else {
            return None;
        };
        let fast_call = build_fast_call(
            &mut self.builder,
            self.config.execution_mode,
            &mut self.aot_imports,
            &mut self.fast_call_by_name,
            &import,
        )?;
        self.fast_calls.insert(va, fast_call.clone());
        Some(fast_call)
    }

    /// Switch-based indirect jump to a known block, otherwise exit the VM.
    pub fn emit_runtime_indirect_jump(&mut self, target_val: Value) -> crate::Result<()> {
        let target_offset = self.to_lift_va(target_val);

        let external_block = self.builder.create_block();
        self.emit_region_switch(target_offset, external_block);

        self.builder.switch_to_block(external_block);
        self.builder.seal_block(external_block);
        A::sync_to_context(self, Some(target_val), ExitReason::UnknownJump)?;
        self.write_exit_code(TransferKind::IndirectJump as u32);
        // Linkable: an out-of-region register `jmp` is answered by entering whatever covers it.
        self.emit_exit(Some(target_val));
        self.block_terminated = true;
        Ok(())
    }

    fn callback_sig(&mut self, extra: usize) -> SigRef {
        if let Some(&sig) = self.callback_sigs.get(&extra) {
            return sig;
        }
        let mut sig = Signature::new(get_host_calling_conv());
        for _ in 0..=extra {
            sig.params.push(AbiParam::new(types::I64));
        }
        let sig_ref = self.builder.import_signature(sig);
        self.callback_sigs.insert(extra, sig_ref);
        sig_ref
    }

    /// Call the syscall callback inline and resume; `pc` is both what the callback sees
    /// and where it returns. See [`crate::core::callback`].
    pub fn emit_syscall_callback(&mut self, pc: Option<Value>) -> crate::Result<()> {
        let offset = A::callback_offsets().syscall;
        self.emit_callback(offset, pc, pc, &[])
    }

    /// Call the interpreter fallback inline to run `instructions` from `pc`, resuming at `resume`.
    ///
    /// Unlike a syscall it is pointed *at* the instruction and returns past it.
    /// See [`crate::core::callback`].
    pub fn emit_interpreter_callback(
        &mut self,
        pc: Option<Value>,
        resume: Option<Value>,
        instructions: u64,
    ) -> crate::Result<()> {
        let offset = A::callback_offsets().interpreter_fallback;
        self.emit_callback(offset, pc, resume, &[instructions])
    }

    /// Sync state, call the callback at context `offset`, reload, and dispatch on the PC it left.
    ///
    /// The block is **not** terminated on return: the caller keeps emitting inline.
    fn emit_callback(
        &mut self,
        offset: i32,
        pc: Option<Value>,
        resume: Option<Value>,
        extra: &[u64],
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        A::sync_to_context(self, pc, ExitReason::None)?;
        self.publish_pinned()?;

        let target = self.ctx_load(types::I64, ctx_ptr, offset);
        let sig = self.callback_sig(extra.len());
        let mut args = vec![ctx_ptr];
        for &value in extra {
            args.push(self.builder.ins().iconst(types::I64, value as i64));
        }
        self.builder.ins().call_indirect(sig, target, &args);

        A::load_state_from_context(self)?;
        self.reload_pinned()?;
        // The callback may write any field, so no sync store may be pruned.
        self.keep_all_sync();

        self.resume_after_callback(resume)
    }

    /// Fall through inline if the callback left PC at `resume`; otherwise dispatch like an
    /// indirect jump, which may leave the region.
    fn resume_after_callback(&mut self, resume: Option<Value>) -> crate::Result<()> {
        if self.config.analysis_mode {
            return Ok(());
        }
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let landed = self.ctx_load(types::I64, ctx_ptr, A::pc_offset());
        let expected = match resume {
            Some(value) => value,
            None => self.runtime_current_ip(),
        };
        let resumed = self.builder.ins().icmp(IntCC::Equal, landed, expected);

        let continue_block = self.builder.create_block();
        let divert_block = self.builder.create_block();
        self.builder
            .ins()
            .brif(resumed, continue_block, &[], divert_block, &[]);

        self.builder.switch_to_block(divert_block);
        self.builder.seal_block(divert_block);
        self.emit_runtime_indirect_jump(landed)?;

        self.builder.switch_to_block(continue_block);
        self.builder.seal_block(continue_block);
        self.block_terminated = false;
        Ok(())
    }

    pub fn bridge_sig(&mut self) -> SigRef {
        match self.bridge_sig {
            Some(sig) => sig,
            None => {
                let mut sig = Signature::new(get_host_calling_conv());
                sig.params.push(AbiParam::new(types::I64)); // *mut RegisterContext
                sig.params.push(AbiParam::new(types::I64)); // import VA / target
                sig.returns.push(AbiParam::new(types::I32)); // 0 = handled
                let sig_ref = self.builder.import_signature(sig);
                self.bridge_sig = Some(sig_ref);
                sig_ref
            }
        }
    }

    pub fn aot_imports(&self) -> &[(&'static str, *const u8, Signature)] {
        &self.aot_imports
    }

    pub fn get_block(&mut self, offset: u64) -> Block {
        if let Some(&block) = self.blocks.get(&offset) {
            block
        } else {
            let block = self.builder.create_block();
            self.blocks.insert(offset, block);
            block
        }
    }

    /// Like `get_block`, recording the transfer kind that reaches it (first caller wins).
    pub fn get_block_with_kind(&mut self, offset: u64, kind: TransferKind) -> Block {
        let block = self.get_block(offset);
        self.block_transfer_kinds.entry(block).or_insert(kind);
        block
    }

    /// Continue emitting into the fallthrough of a branch just emitted.
    ///
    /// Tail-duplicated code can find the fallthrough already filled; terminate instead of
    /// switching into it, which would trip Cranelift's "already filled" assertion.
    pub fn continue_in_fallthrough(&mut self, next_block: Block) {
        if self.visited_blocks.contains(&next_block) {
            self.block_terminated = true;
            return;
        }
        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;
    }

    pub fn write_exit_code(&mut self, code: u32) {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let val = self.builder.ins().iconst(types::I32, code as i64);
        self.ctx_store(val, ctx_ptr, A::exit_code_offset());
    }

    /// Declare that the VM exit just emitted **resumes at `va`**.
    ///
    /// Every host-serviced, re-entered exit needs both effects:
    ///
    /// - **Block leader**, so the code after the exit is lifted at all.
    /// - **Resume point**, so the entry dispatch has an arm for it.
    ///
    /// Out-of-region addresses are ignored.
    pub fn resumes_at(&mut self, va: u64) {
        if self.in_region(va) {
            self.exit_resumes.insert(va);
            self.get_block(va);
        }
    }

    /// Store `reason` into `exit_reason` with no PC write or register sync; for the exit
    /// before any instruction is lifted.
    pub fn write_exit_reason(&mut self, reason: ExitReason) {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let val = self.builder.ins().iconst(types::I32, reason as i64);
        self.ctx_store(val, ctx_ptr, A::exit_reason_offset());
    }

    /// Whether the instruction at `addr` runs off the end of the lifted bytes:
    ///
    /// - Its `next_ip` is past the end of its range.
    /// - It decoded **invalid** within one maximal instruction of the end: a decoder given
    ///   too few bytes says "invalid", not "truncated", so the resumable answer is safe.
    ///
    /// Never true at a range's first address, so the exit terminates rather than repeats.
    /// "The end" is the range's end, so a block abutting a hole is cut like one at the slice end.
    fn runs_off_region(&self, addr: u64, inst: &A::Instr) -> bool {
        let Some(range) = self.covering(addr) else {
            return false;
        };
        self.config.region_tail_exit
            && addr != range.start
            && (A::instr_next_ip(inst) > range.end
                || (A::is_invalid(inst) && range.end - addr < A::MAX_INSTRUCTION_LEN))
    }

    /// The covered range holding `va`.
    fn covering(&self, va: u64) -> Option<&Range<u64>> {
        // Ascending and disjoint: only the last range starting at or before `va` can hold it.
        let i = self
            .covered
            .partition_point(|r| r.start <= va)
            .checked_sub(1)?;
        self.covered.get(i).filter(|r| va < r.end)
    }

    /// Whether `va` is a byte this lift covers (holes make the plain bounds insufficient).
    pub fn in_region(&self, va: u64) -> bool {
        self.covering(va).is_some()
    }

    /// Always true: dead flag math is removed after the fact. Kept for frontends that
    /// guard flag math through this API.
    pub fn flag_live(&self, _bit: u8) -> bool {
        true
    }

    /// Record a flag's raw SSA value, including source-encoded flags such as x86 PF/AF.
    pub fn record_flag_value(&mut self, value: Value) {
        if self.config.eliminate_dead_flags {
            self.flag_values.push(value);
        }
    }

    /// The `image_base` value at entry: the context's base when `ImageBaseRelative`, the
    /// constant `lift_base` when `Direct` (so `image_base + (va - lift_base) == va` folds).
    pub fn image_base_at_entry(&mut self, ctx_ptr: Value) -> Value {
        match self.config.address_mode {
            AddressMode::ImageBaseRelative => {
                self.ctx_load(types::I64, ctx_ptr, A::image_base_offset())
            }
            AddressMode::Direct => {
                let base = self.lift_base;
                self.builder.ins().iconst(types::I64, base as i64)
            }
        }
    }

    /// Materialize the run-time address of lift-time `va`; the one place a guest VA becomes
    /// a value. Inverse: [`Self::to_lift_va`].
    pub fn runtime_addr(&mut self, va: u64) -> Value {
        match self.config.address_mode {
            AddressMode::ImageBaseRelative => {
                let image_base = self.builder.use_var(A::image_base_var(&self.state));
                let offset = va.wrapping_sub(self.lift_base);
                self.builder.ins().iadd_imm_s(image_base, offset as i64)
            }
            AddressMode::Direct => self.builder.ins().iconst(types::I64, va as i64),
        }
    }

    /// Run-time address of [`Self::current_ip`], the resume point for a re-enterable exit.
    pub fn runtime_current_ip(&mut self) -> Value {
        self.runtime_addr(self.current_ip)
    }

    /// Map a run-time address back to lift-time VA space; inverse of [`Self::runtime_addr`].
    pub fn to_lift_va(&mut self, addr: Value) -> Value {
        match self.config.address_mode {
            AddressMode::ImageBaseRelative => {
                let image_base = self.builder.use_var(A::image_base_var(&self.state));
                let rel = self.builder.ins().isub(addr, image_base);
                self.builder.ins().iadd_imm_s(rel, self.lift_base as i64)
            }
            AddressMode::Direct => addr,
        }
    }

    /// Write the PC and exit_reason to the context. Shared by all sync paths.
    pub fn flush_rip_and_reason(
        &mut self,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));

        let pc_val = if let Some(p) = pc {
            p
        } else {
            self.runtime_current_ip()
        };
        self.ctx_store(pc_val, ctx_ptr, A::pc_offset());

        let reason_val = self.builder.ins().iconst(types::I32, reason as i64);
        self.ctx_store(reason_val, ctx_ptr, A::exit_reason_offset());
        Ok(())
    }

    /// Dispatch lift-time VA `target` to the in-region block starting there, else `external`.
    pub fn emit_region_switch(&mut self, target: Value, external: Block) {
        let span = self.lift_end.wrapping_sub(self.lift_base);
        let inside = self.builder.create_block();
        let rel = self
            .builder
            .ins()
            .iadd_imm_s(target, (self.lift_base as i64).wrapping_neg());
        let outside =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, rel, span as i64);
        self.builder.ins().brif(outside, external, &[], inside, &[]);

        self.builder.switch_to_block(inside);
        self.builder.seal_block(inside);
        let mut switch = Switch::new();
        // Sorted: `HashMap` order is per-instance and `Switch::emit` follows insertion order.
        let mut arms: Vec<(u64, Block)> = self
            .blocks
            .iter()
            .filter(|&(&offset, _)| self.in_region(offset))
            .map(|(&offset, &block)| (offset, block))
            .collect();
        arms.sort_unstable_by_key(|&(offset, _)| offset);
        for (offset, block) in arms {
            switch.set_entry(offset as u128, block);
        }
        switch.emit(&mut self.builder, target, external);
    }

    /// Whether this lift emits link probes at all.
    pub fn block_link_active(&self) -> bool {
        self.block_link().is_some()
    }

    /// Link-table geometry, if this lift is linking. `None` disables every probe path;
    /// analysis mode is excluded since its IR doesn't write the context.
    fn block_link(&self) -> Option<(BlockLinkConfig, BlockLinkOffsets)> {
        if self.config.analysis_mode {
            return None;
        }
        let config = self.config.block_link?;
        Some((config, A::block_link_offsets()?))
    }

    /// Whether lift-time-known exits get their own cell: needs `link_cells` and linking itself.
    fn link_cells(&self) -> bool {
        self.config.link_cells && self.block_link().is_some()
    }

    /// Cell index for `target`, allocated on first use. De-duplicated by target, so a
    /// second branch to an address links the moment the first does.
    fn link_cell_for(&mut self, target: u64, kind: crate::core::link::LinkSiteKind) -> usize {
        if let Some(&index) = self.link_cell_by_target.get(&target) {
            return index;
        }
        let index = self.link_sites.len();
        self.link_sites
            .push(crate::core::link::LinkSite { target, kind });
        self.link_cell_by_target.insert(target, index);
        index
    }

    /// Base address of this region's cell array.
    ///
    /// A relocated symbol so the artifact survives caching at another address. Namespace 1
    /// is `cranelift-module`'s `DataId` space (equal to [`ANALYSIS_CALL_NAMESPACE`], harmless
    /// since analysis IR is never compiled).
    fn link_cell_base(&mut self) -> Value {
        let gv = match self.link_cell_base {
            Some(gv) => gv,
            None => {
                let name = self
                    .builder
                    .func
                    .declare_imported_user_function(UserExternalName {
                        namespace: 1,
                        index: 0,
                    });
                let gv = self.builder.func.create_global_value(
                    cranelift_codegen::ir::GlobalValueData::Symbol {
                        name: ExternalName::user(name),
                        offset: 0.into(),
                        // **Far, though same object:** `colocated` means "within `adrp` range", which
                        // `cranelift-jit`'s separate exec/write `mmap`s can violate (seen 6 GiB apart, overflowing
                        // `Aarch64AdrPrelPgHi21`). Far costs a literal load; `Abs8` can't go out of range.
                        colocated: false,
                        tls: false,
                    },
                );
                self.link_cell_base = Some(gv);
                gv
            }
        };
        self.builder.ins().symbol_value(types::I64, gv)
    }

    /// Host-frame config and offsets, if this lift nests guest calls as host calls.
    ///
    /// Needs cells on (the callee address comes from one) and a guest depth offset;
    /// mirrors [`Self::block_link`] and [`Self::call_return`].
    fn host_call_frames(&self) -> Option<(HostFrameConfig, HostFrameOffsets)> {
        if !self.link_cells() {
            return None;
        }
        let config = self.config.host_call_frames?;
        Some((config, A::host_frame_offsets()?))
    }

    /// Arguments to enter a region with: the context pointer, then one per pinned slot.
    ///
    /// **Not [`Self::pinned_call_args`]**, which is empty when unpinned because its
    /// callers are `return_` sites; a *call* still passes the context pointer.
    fn region_call_args(&mut self) -> Vec<Value> {
        match self.pinned_slots.is_empty() {
            true => vec![self.builder.use_var(A::context_ptr_var(&self.state))],
            false => self.pinned_call_args(),
        }
    }

    /// This region's own signature, re-imported from `func.signature` so it can't drift.
    fn region_signature_ref(&mut self) -> SigRef {
        match self.region_sig {
            Some(sig) => sig,
            None => {
                let signature = self.builder.func.signature.clone();
                let sig = self.builder.import_signature(signature);
                self.region_sig = Some(sig);
                sig
            }
        }
    }

    /// Whether an out-of-region `RET` should **return** rather than link.
    ///
    /// Other half of [`Self::emit_host_frame_call`]: a linking callee would strand the
    /// waiting caller frame and grow the host stack per iteration. Under host frames a
    /// leaving `RET` always returns; with no frame waiting the entry thunk gets it,
    /// costing one host dispatch like a link miss.
    pub fn returns_through_host_frames(&self) -> bool {
        self.host_call_frames().is_some()
    }

    /// Nest an out-of-region direct `CALL` as a real host call, else return `false`.
    ///
    /// ```text
    ///   call(k, callee, resume):
    ///     cell  = &cells[k]
    ///     code  = cell.code;                 if code == 0     goto slow
    ///     depth = ctx.host_frame_depth;      if depth >= max  goto slow
    ///     ctx.host_frame_depth = depth + 1
    ///     ctx.pc = callee;  ctx.image_base = cell.image_base
    ///     r = call_indirect code(ctx, slots...)      ; a real `bl`
    ///     ctx.host_frame_depth = depth
    ///     <slots from r>;  <reload the rest from the context>
    ///     if ctx.exit_reason == UnknownJump/Return && ctx.pc == resume:
    ///         ctx.exit_reason = None;  goto resume       ; the common case
    ///     return r                                       ; unwind one frame
    ///   slow:  <the ordinary linked exit>
    /// ```
    ///
    /// The callee gets no cooperation: its `RET` syncs and returns with the target in the
    /// PC, and the caller (where the target is a lift-time constant) decides if it was
    /// expected. A mismatch returns again, unwinding one host frame per level.
    ///
    /// Returns `true` when emitted; `false` leaves the builder untouched and the caller
    /// must produce its ordinary exit.
    pub fn emit_host_frame_call(
        &mut self,
        target: u64,
        callee: Value,
        resume: u64,
    ) -> crate::Result<bool> {
        let Some((config, offsets)) = self.host_call_frames() else {
            return Ok(false);
        };
        // The resume point must be in-region: past it `finish` makes an exit stub that would
        // sync the pre-call registers over the callee's writes.
        if !self.in_region(resume) {
            return Ok(false);
        }
        let index = self.link_cell_for(target, crate::core::link::LinkSiteKind::StaticEdge);
        let cell_offset = index as i64 * crate::core::link::ENTRY_SIZE as i64;

        let nest = self.builder.create_block();
        let taken = self.builder.create_block();
        let unwind = self.builder.create_block();
        let slow = self.builder.create_block();
        let resume_block = self.get_block(resume);

        let flags = self.link_cell_flags;
        let base = self.link_cell_base();
        let cell = self.builder.ins().iadd_imm_s(base, cell_offset);
        let code = self
            .builder
            .ins()
            .load(types::I64, flags, cell, crate::core::link::CODE_OFFSET);
        // Target not translated yet: nothing to call.
        self.builder.ins().brif(code, nest, &[], slow, &[]);

        self.builder.switch_to_block(nest);
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let depth = self.ctx_load(types::I32, ctx_ptr, offsets.depth);
        let room =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, depth, config.max_depth as i64);
        // The host stack is finite; past the ceiling take the ordinary exit.
        self.builder.ins().brif(room, taken, &[], slow, &[]);

        self.builder.switch_to_block(taken);
        // Saved before the callee's is written: the resume comparison and later addresses use this region's base.
        let my_image_base = self.builder.use_var(A::image_base_var(&self.state));
        let deeper = self.builder.ins().iadd_imm_s(depth, 1);
        self.ctx_store(deeper, ctx_ptr, offsets.depth);
        // The callee enters through its own PC dispatch with its own `image_base`.
        self.ctx_store(callee, ctx_ptr, A::pc_offset());
        let image_base = self.builder.ins().load(
            types::I64,
            flags,
            cell,
            crate::core::link::IMAGE_BASE_OFFSET,
        );
        self.ctx_store(image_base, ctx_ptr, A::image_base_offset());

        let sig = self.region_signature_ref();
        let args = self.region_call_args();
        let call = self.builder.ins().call_indirect(sig, code, &args);
        let results: Vec<Value> = self.builder.inst_results(call).to_vec();

        // Restore this frame's own depth, already in a register.
        self.ctx_store(depth, ctx_ptr, offsets.depth);
        // Restore the image base before the reload so later addresses are this region's.
        self.ctx_store(my_image_base, ctx_ptr, A::image_base_offset());
        // The pinned half returns in order; result 0 is the unmoved context pointer.
        for (i, &(_, var, _)) in self.pinned_slots.clone().iter().enumerate() {
            self.builder.def_var(var, results[i + 1]);
        }
        // Reload the rest here, on the call's exit path only: at the resume block it would
        // overwrite live registers reached by in-region `RET` dispatch without a sync.
        A::load_state_from_context(self)?;
        // No `keep_all_sync`, unlike the syscall reload: a register never written here still
        // holds the value the reload took from the context, so its store stays idempotent
        // (disabling the prune measured +237% vs +89% on a recursive fixture).
        // Held by `host_frame_tests::a_callee_s_writes_survive_a_caller_that_touches_nothing`.

        let reason = self.ctx_load(types::I32, ctx_ptr, A::exit_reason_offset());
        let is_transfer =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::Equal, reason, ExitReason::UnknownJump as i64);
        let code_field = self.ctx_load(types::I32, ctx_ptr, A::exit_code_offset());
        let is_return =
            self.builder
                .ins()
                .icmp_imm_u(IntCC::Equal, code_field, TransferKind::Return as i64);
        let pc = self.ctx_load(types::I64, ctx_ptr, A::pc_offset());
        let want = self.runtime_addr(resume);
        let here = self.builder.ins().icmp(IntCC::Equal, pc, want);
        let ok = self.builder.ins().band(is_transfer, is_return);
        let ok = self.builder.ins().band(ok, here);
        self.builder.ins().brif(ok, resume_block, &[], unwind, &[]);

        self.builder.switch_to_block(unwind);
        // Not our return: hand it up to a lower frame or the entry thunk. Nothing was serviced, so nothing is cleared.
        self.builder.ins().return_(&results);

        self.builder.switch_to_block(slow);
        self.builder.seal_block(nest);
        self.builder.seal_block(taken);
        self.builder.seal_block(unwind);
        self.builder.seal_block(slow);
        Ok(true)
    }

    /// Terminate a linkable exit whose destination is known at **lift time**, trying its
    /// own cell before [`Self::emit_exit`].
    ///
    /// ```text
    ///   static(k, pc):  cells = &lifted_cells        ; one relocated symbol per region
    ///                   cell  = cells + k*32
    ///                   code  = cell.code;  if code == 0 goto slow
    ///   hit:            ctx.pc = pc
    ///                   [pinned]     ctx.image_base = cell.image_base
    ///                                return_call_indirect code(ctx, slots...)
    ///                   [hostresume] ctx.link_hit = cell ; return
    ///   slow:           <exactly emit_exit(pc)>
    /// ```
    ///
    /// Four instructions on the fast path against the probe's nine, and no collisions.
    ///
    /// The fallback is still needed: a cell is empty until the target is translated, and
    /// after [invalidation](crate::runtime::CodeCache::invalidate_range) zeroes it.
    ///
    /// `target` is the lift-time VA the host installs against; `pc` the same address as a
    /// run-time value. Exactly `emit_exit` when [`Self::link_cells`] is off.
    pub fn emit_static_exit(
        &mut self,
        target: u64,
        pc: Option<Value>,
        kind: crate::core::link::LinkSiteKind,
    ) {
        if !self.link_cells() {
            return self.emit_exit(pc);
        }
        let pc = match pc {
            Some(pc) => pc,
            None => self.runtime_current_ip(),
        };
        self.emit_cell_probe(target, pc, kind);
        // Reached only when the cell is empty; `pc` is still in scope (its block dominates).
        self.emit_exit(Some(pc));
    }

    /// One cell probe; terminates the block and leaves the builder in the `slow` arm.
    fn emit_cell_probe(&mut self, target: u64, pc: Value, kind: crate::core::link::LinkSiteKind) {
        let index = self.link_cell_for(target, kind);
        let cell_offset = index as i64 * crate::core::link::ENTRY_SIZE as i64;

        let base = self.link_cell_base();
        let cell = self.builder.ins().iadd_imm_s(base, cell_offset);
        self.emit_cell_dispatch(cell, pc);
    }

    /// Take the link a [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) at `cell`
    /// names, else fall through.
    ///
    /// Terminates the block and leaves the builder in `slow`. `cell` may be a constant
    /// offset into this region's array or a ring-frame pointer into another region's.
    fn emit_cell_dispatch(&mut self, cell: Value, pc: Value) {
        let (_, offsets) = self
            .block_link()
            .expect("a cell is only dispatched when linking is on");

        let hit = self.builder.create_block();
        let slow = self.builder.create_block();

        let flags = self.link_cell_flags;
        let code = self
            .builder
            .ins()
            .load(types::I64, flags, cell, crate::core::link::CODE_OFFSET);
        // Zero is "not installed": a zeroed array is valid and clearing a cell retracts the link.
        self.builder.ins().brif(code, hit, &[], slow, &[]);

        self.builder.switch_to_block(hit);
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        // Committed like the table probe: one exit shape has a `rip` that isn't where it continues.
        self.ctx_store(pc, ctx_ptr, A::pc_offset());
        if self.pinned() {
            let image_base = self.builder.ins().load(
                types::I64,
                flags,
                cell,
                crate::core::link::IMAGE_BASE_OFFSET,
            );
            self.ctx_store(image_base, ctx_ptr, A::image_base_offset());
            let sig = self.pinned_signature_ref();
            let args = self.pinned_call_args();
            self.builder.ins().return_call_indirect(sig, code, &args);
        } else {
            // The cell's address, as the table probe hands back a row's.
            self.ctx_store(cell, ctx_ptr, offsets.hit);
            self.emit_host_return();
        }

        self.builder.switch_to_block(slow);
        self.builder.seal_block(hit);
        self.builder.seal_block(slow);
    }

    /// Call-return ring geometry and offsets, if this lift predicts returns.
    ///
    /// Needs cells on (a hit hands back a cell) and a guest ring offset; mirrors [`Self::block_link`].
    fn call_return(&self) -> Option<(CallRetConfig, CallRetOffsets)> {
        if !self.link_cells() {
            return None;
        }
        let config = self.config.call_return_stack?;
        Some((config, A::call_return_offsets()?))
    }

    /// Push a frame so a return to `ret_pc` enters its region via a cell of this region's array.
    ///
    /// ```text
    ///   push(ret_pc, k):  base = ctx.callret_base;  if base == 0 goto done
    ///                     off  = (ctx.callret_off - 16) & MASK
    ///                     [base+off] = { ret_pc, &cells[k] }
    ///                     ctx.callret_off = off
    ///   done:
    /// ```
    ///
    /// No bounds check: the ring is a power of two and the mask is the wrap, so overflow
    /// overwrites the oldest frame, whose `RET` then misses. See [`crate::core::callret`].
    ///
    /// `ret_pc` is the run-time return address; `ret_va` the lift-time VA the cell
    /// installs against. Inert when [`Self::call_return`] is `None`.
    pub fn emit_callret_push(&mut self, ret_va: u64, ret_pc: Value) {
        let Some((config, offsets)) = self.call_return() else {
            return;
        };
        let index = self.link_cell_for(ret_va, crate::core::link::LinkSiteKind::CallReturn);

        let push = self.builder.create_block();
        let done = self.builder.create_block();

        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let base = self.ctx_load(types::I64, ctx_ptr, offsets.base);
        // No ring is not an error; this also keeps the `notrap` store off a null pointer.
        self.builder.ins().brif(base, push, &[], done, &[]);

        self.builder.switch_to_block(push);
        let old = self.ctx_load(types::I64, ctx_ptr, offsets.offset);
        let grown = self
            .builder
            .ins()
            .iadd_imm_s(old, -(crate::core::callret::FRAME_SIZE as i64));
        let off = self.builder.ins().band_imm_u(grown, config.mask());
        let frame = self.builder.ins().iadd(base, off);

        let cells = self.link_cell_base();
        let cell = self
            .builder
            .ins()
            .iadd_imm_s(cells, index as i64 * crate::core::link::ENTRY_SIZE as i64);

        let flags = self.callret_flags;
        self.builder.ins().store(flags, ret_pc, frame, 0);
        self.builder
            .ins()
            .store(flags, cell, frame, crate::core::callret::CELL_OFFSET);
        self.ctx_store(off, ctx_ptr, offsets.offset);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(done);
        self.builder.seal_block(push);
        self.builder.seal_block(done);
    }

    /// Take the top ring frame if it is about `pc`, or fall through.
    ///
    /// ```text
    ///   pop(pc):  base = ctx.callret_base;  if base == 0 goto slow
    ///             off  = ctx.callret_off
    ///             if [base+off].guest_pc != pc goto slow
    ///             ctx.callret_off = (off + 16) & MASK
    ///             cell = [base+off].cell        ; then the cell hit arm
    ///   slow:     <the caller's existing exit>
    /// ```
    ///
    /// Terminates the block and leaves the builder in `slow`.
    ///
    /// **The pointer only advances on a match**, so underflow is impossible: an unpushed
    /// frame holds zero and no guest returns to zero.
    ///
    /// Inert when [`Self::call_return`] is `None`.
    pub fn emit_callret_pop(&mut self, pc: Value) {
        let Some((config, offsets)) = self.call_return() else {
            return;
        };

        let check = self.builder.create_block();
        let take = self.builder.create_block();
        let slow = self.builder.create_block();

        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let base = self.ctx_load(types::I64, ctx_ptr, offsets.base);
        self.builder.ins().brif(base, check, &[], slow, &[]);

        self.builder.switch_to_block(check);
        let off = self.ctx_load(types::I64, ctx_ptr, offsets.offset);
        let frame = self.builder.ins().iadd(base, off);
        let flags = self.callret_flags;
        let expected = self.builder.ins().load(types::I64, flags, frame, 0);
        let matches = self.builder.ins().icmp(IntCC::Equal, expected, pc);
        self.builder.ins().brif(matches, take, &[], slow, &[]);

        self.builder.switch_to_block(take);
        // Pop only on a match: a declined frame belongs to an outer call.
        let next = self
            .builder
            .ins()
            .iadd_imm_s(off, crate::core::callret::FRAME_SIZE as i64);
        let next = self.builder.ins().band_imm_u(next, config.mask());
        self.ctx_store(next, ctx_ptr, offsets.offset);
        let cell =
            self.builder
                .ins()
                .load(types::I64, flags, frame, crate::core::callret::CELL_OFFSET);
        // Ends in its own `slow` arm, which falls through to ours.
        self.emit_cell_dispatch(cell, pc);
        self.builder.ins().jump(slow, &[]);

        self.builder.switch_to_block(slow);
        self.builder.seal_block(check);
        self.builder.seal_block(take);
        self.builder.seal_block(slow);
    }

    /// Terminate a VM exit the host answers by **continuing at a guest address**, giving
    /// it a chance to resolve its own successor.
    ///
    /// `pc` is the run-time value already written to the context (`None` = [`Self::current_ip`]);
    /// this does not write it.
    ///
    /// Only for exits whose whole servicing is "enter whatever is at `ctx.rip`". Exits the
    /// host must act on first (import call, syscall, `CPUID`, refused resume) must `return_`,
    /// or link via [`Self::emit_dynamic_link`]. With linking off this is exactly `return_`.
    pub fn emit_exit(&mut self, pc: Option<Value>) {
        if self.block_link().is_none() {
            return self.emit_host_return();
        }
        let pc = match pc {
            Some(pc) => pc,
            None => self.runtime_current_ip(),
        };
        if self.config.inline_link_probes {
            // Leaves the builder in the probe's terminated `out` block.
            self.emit_link_probe(pc);
        } else {
            let block = self.link_probe_block();
            self.builder.ins().jump(block, &[pc.into()]);
        }
    }

    /// The link-probe block, created on first use; [`Self::fill_link_probe`] emits its
    /// body after lifting so it isn't filled mid-stub.
    fn link_probe_block(&mut self) -> Block {
        match self.link_block {
            Some(block) => block,
            None => {
                let block = self.builder.create_block();
                self.builder.append_block_param(block, types::I64);
                self.link_block = Some(block);
                block
            }
        }
    }

    /// Emit the probe itself: one masked shift, one address compare, one store.
    ///
    /// ```text
    ///   link(pc):  tbl = ctx.link_table;      if tbl == 0 goto out
    ///              ent = tbl + slot(pc)*32;   if ent.guest_pc != pc goto out
    ///              ctx.pc = pc; ctx.link_hit = ent
    ///   out:       return
    /// ```
    ///
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned) the hit arm
    /// publishes `image_base` from the row and `return_call_indirect`s to `ent.code` with
    /// the pinned registers; a chain of those ends at a null `ctx.link_table`.
    ///
    /// Loads are `trusted` but deliberately **not** `readonly`: the host rewrites rows between entries.
    ///
    /// Nothing is stored on a miss; `link_hit` is the host's to clear before entering
    /// ([`BlockLinkOffsets::hit`]).
    fn fill_link_probe(&mut self) -> crate::Result<()> {
        let Some(block) = self.link_block else {
            return Ok(());
        };
        self.builder.switch_to_block(block);
        let pc = self.builder.block_params(block)[0];
        self.visited_blocks.insert(block);
        self.emit_link_probe(pc);
        Ok(())
    }

    /// One probe; terminates the block and leaves the builder in the `out` arm.
    fn emit_link_probe(&mut self, pc: Value) {
        let (entry, out) = self.emit_link_lookup(pc);

        // A hit also commits the PC so one probe serves exits whose `rip` isn't where they
        // continue (an out-of-region direct `CALL` leaves the callee in `imported_function_va`).
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        self.ctx_store(pc, ctx_ptr, A::pc_offset());
        self.emit_link_taken(entry);

        self.builder.switch_to_block(out);
        self.emit_host_return();
    }

    /// Probe up to the address compare, leaving the builder in the hit arm; returns the row
    /// address and the miss block. Split out so an exit can commit *inside* the hit arm
    /// ([`Self::emit_dynamic_link`]).
    fn emit_link_lookup(&mut self, pc: Value) -> (Value, Block) {
        let (link, offsets) = self
            .block_link()
            .expect("a probe is only emitted when linking is on");

        let probe = self.builder.create_block();
        let hit = self.builder.create_block();
        let out = self.builder.create_block();

        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let table = self.ctx_load(types::I64, ctx_ptr, offsets.table);
        // No table published means it never links, and keeps the `notrap` row loads off null.
        self.builder.ins().brif(table, probe, &[], out, &[]);

        self.builder.switch_to_block(probe);
        let slot = self.builder.ins().ushr_imm_u(pc, link.shift as i64);
        let slot = self.builder.ins().band_imm_u(slot, link.mask());
        let offset = self
            .builder
            .ins()
            .ishl_imm_u(slot, crate::core::link::ENTRY_SIZE.trailing_zeros() as i64);
        let entry = self.builder.ins().iadd(table, offset);
        let row = self.link_table_flags;
        let key = self.builder.ins().load(types::I64, row, entry, 0);
        let matches = self.builder.ins().icmp(IntCC::Equal, key, pc);
        self.builder.ins().brif(matches, hit, &[], out, &[]);

        for block in [probe, hit, out] {
            self.visited_blocks.insert(block);
        }
        self.builder.switch_to_block(hit);
        (entry, out)
    }

    /// Terminate the block by taking the link `entry` names: branch to it, or publish it and return.
    fn emit_link_taken(&mut self, entry: Value) {
        let (_, offsets) = self
            .block_link()
            .expect("a link is only taken when linking is on");
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        if self.pinned() {
            // Branch rather than answer: the successor is entered with pinned registers in place.
            // `image_base` is set here since no host runs to do it. Hoisting these loads beside
            // `key` measured slower.
            let row = self.link_table_flags;
            let image_base = self.builder.ins().load(
                types::I64,
                row,
                entry,
                crate::core::link::IMAGE_BASE_OFFSET,
            );
            self.ctx_store(image_base, ctx_ptr, A::image_base_offset());
            let code =
                self.builder
                    .ins()
                    .load(types::I64, row, entry, crate::core::link::CODE_OFFSET);
            let sig = self.pinned_signature_ref();
            let args = self.pinned_call_args();
            self.builder.ins().return_call_indirect(sig, code, &args);
        } else {
            self.ctx_store(entry, ctx_ptr, offsets.hit);
            self.emit_host_return();
        }
    }

    /// Give an exit with a run-time destination a chance to link, committing only in the hit arm.
    ///
    /// ```text
    ///   dynamic(pc):  <exactly the probe, up to the address compare>
    ///   hit:          <commit>            ; the caller's, emitted here
    ///                 ctx.link_hit = ent  ; or the pinned tail call
    ///                 return
    ///   miss:         <the caller's existing exit, unchanged>
    /// ```
    ///
    /// Unlike [`Self::emit_exit`], for exits whose servicing includes a host *decision* (an
    /// indirect `CALL` pushes its return address only if the callee is guest code): a
    /// matching row settles it, so the commit belongs in the hit arm.
    ///
    /// Leaves the builder in the miss arm; emits nothing when not linking. `commit` must not
    /// terminate the block.
    pub fn emit_dynamic_link(
        &mut self,
        pc: Value,
        commit: impl FnOnce(&mut Self) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if self.block_link().is_none() {
            return Ok(());
        }
        let (entry, miss) = self.emit_link_lookup(pc);
        commit(self)?;
        self.emit_link_taken(entry);
        self.builder.switch_to_block(miss);
        Ok(())
    }

    pub fn lift(&mut self, bytes: &[u8], ip: u64) -> crate::Result<()> {
        self.lift_base = ip;
        self.lift_end = ip + bytes.len() as u64;
        self.covered = normalize_ranges(&self.config.block_ranges, ip, self.lift_end);
        // Guest addresses encode relative to the region base, so this precedes any IR.
        self.source_map = SourceLocMap::new(ip);

        // Phase 1: decode and leader discovery.
        //
        // Address-keyed decoding lets a branch into the middle of an instruction decode fresh
        // there (sub-encoded/overlapping instructions). A linear sweep per covered range primes
        // it, then a worklist follows control flow.
        let mut decoded: HashMap<u64, A::Instr> = HashMap::new();
        for range in &self.covered {
            let (lo, hi) = ((range.start - ip) as usize, (range.end - ip) as usize);
            for inst in A::decode_region(&bytes[lo..hi], range.start) {
                decoded.entry(A::instr_ip(&inst)).or_insert(inst);
            }
        }

        // Jump-table successors let an indirect branch contribute leaders; one pass suffices.
        if self.config.analyze_jump_tables {
            self.jump_tables = A::jump_table_targets(bytes, ip, &self.config.segments);
        }

        let mut leaders: HashSet<u64> = HashSet::from([ip]);
        let mut work: Vec<u64> = decoded.keys().copied().collect();
        work.push(ip);
        // Host resume points are leaders before discovery so code reachable from them is found too.
        if self.config.resumption != VmResumptionPoints::None {
            for &p in &self.config.resume_points {
                if self.in_region(p) {
                    leaders.insert(p);
                    work.push(p);
                }
            }
        }
        // Where calls leaving the region return to; always dispatchable since the host can't name them.
        let mut call_resumes: HashSet<u64> = HashSet::new();
        let mut processed: HashSet<u64> = HashSet::new();

        while let Some(addr) = work.pop() {
            if !self.in_region(addr) {
                continue;
            }
            if !processed.insert(addr) {
                continue;
            }
            // Overlap targets aren't in the linear prime; decode on demand.
            if !decoded.contains_key(&addr) {
                let off = (addr - self.lift_base) as usize;
                match A::decode_one(&bytes[off..], addr) {
                    Some(inst) => {
                        decoded.insert(addr, inst);
                    }
                    None => continue,
                }
            }
            let inst = &decoded[&addr];
            // Truncated encodings give guessed control flow: stop here; phase 2 makes a resumable exit.
            if self.runs_off_region(addr, inst) {
                leaders.insert(addr);
                continue;
            }
            let cf = A::control_flow(inst);
            let next = A::instr_next_ip(inst);
            match cf.kind {
                ControlFlowKind::Invalid => {
                    // Post-invalid bytes are a re-sync point.
                    leaders.insert(next);
                    work.push(next);
                }
                ControlFlowKind::Fallthrough => {
                    work.push(next);
                }
                ControlFlowKind::DirectBranch => {
                    for &t in &cf.static_targets {
                        leaders.insert(t);
                        work.push(t);
                    }
                }
                ControlFlowKind::CondBranch => {
                    for &t in &cf.static_targets {
                        leaders.insert(t);
                        work.push(t);
                    }
                    work.push(next);
                }
                ControlFlowKind::Call { static_target } => {
                    if let Some(target) = static_target {
                        if !self.is_import(target) && self.in_region(target) {
                            leaders.insert(target);
                            work.push(target);
                        }
                    }
                    // The return point is always a leader: completion routes back through a dispatch switch.
                    leaders.insert(next);
                    call_resumes.insert(next);
                    work.push(next);
                }
                ControlFlowKind::Ret => {}
                ControlFlowKind::IndirectBranch => {
                    // Dead end unless the jump-table pass resolved it; out-of-region targets keep their exit path.
                    if let Some(targets) = self.jump_tables.get(&addr).map(|t| &t.targets) {
                        for &t in targets {
                            if self.in_region(t) {
                                leaders.insert(t);
                                work.push(t);
                            }
                        }
                    }
                }
            }
        }

        let entry_block = self.builder.create_block();
        self.builder
            .append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);

        let ctx_ptr_val = self.builder.block_params(entry_block)[0];
        let ctx_ptr_var = A::context_ptr_var(&self.state);
        self.builder.def_var(ctx_ptr_var, ctx_ptr_val);

        A::load_state_from_context(self)?;

        // The pinned half arrives as parameters `1 + i`; the frontend's state load skipped these.
        if self.pinned() {
            let params: Vec<Value> = self.builder.block_params(entry_block)[1..].to_vec();
            let vars: Vec<Variable> = self.pinned_slots.iter().map(|&(_, v, _)| v).collect();
            for (var, value) in vars.into_iter().zip(params) {
                self.builder.def_var(var, value);
            }
        }

        if self.config.resumption == VmResumptionPoints::All {
            for instruction in decoded.values() {
                leaders.insert(A::instr_ip(instruction));
                leaders.insert(A::instr_next_ip(instruction));
            }
        }

        // Sorted: `get_block` creates blocks in call order and `HashSet` order is per-instance,
        // which would make the IR and link-cell indices differ per lift.
        let mut ordered: Vec<u64> = leaders.iter().copied().collect();
        ordered.sort_unstable();
        for leader in ordered {
            self.get_block(leader);
        }

        let start_block = self.get_block(ip);

        // The dispatch is emitted after phase 2, which registers resumable-exit points. Until
        // then the entry jumps to an empty block; the jump folds away.
        let wants_dispatch =
            !self.config.analysis_mode && self.config.resumption != VmResumptionPoints::None;
        let dispatch_block = wants_dispatch.then(|| self.builder.create_block());
        match dispatch_block {
            // No dispatch: analysis IR is never run, and `VmResumptionPoints::None` never re-enters.
            None => self.builder.ins().jump(start_block, &[]),
            Some(dispatch) => self.builder.ins().jump(dispatch, &[]),
        };

        // Phase 2: emit each block from its leader, following `instr_next_ip` (overlap makes list
        // order diverge). Re-scan for the lowest unemitted in-region block since lifting creates
        // blocks; out-of-region ones become exit stubs in `finish()`.
        loop {
            let next_leader = self
                .blocks
                .iter()
                .filter(|(off, b)| self.in_region(**off) && !self.visited_blocks.contains(b))
                .map(|(off, _)| *off)
                .min();
            let Some(leader) = next_leader else { break };
            let block = self.blocks[&leader];

            self.builder.switch_to_block(block);
            self.visited_blocks.insert(block);
            self.block_terminated = false;

            let mut cur = leader;
            let mut first = true;
            loop {
                if !self.in_region(cur) {
                    self.current_ip = cur;
                    self.set_source_loc(cur);
                    // Linkable: the host's answer is to enter a region at `cur`.
                    A::sync_to_context(self, None, ExitReason::None)?;
                    self.emit_static_exit(cur, None, crate::core::link::LinkSiteKind::StaticEdge);
                    self.block_terminated = true;
                    break;
                }
                // Reached another block's leader: jump and lift it on its own iteration, unless it is
                // the starting leader or the block the builder is already in.
                if !first {
                    if let Some(&next_block) = self.blocks.get(&cur) {
                        if self.builder.current_block() != Some(next_block) {
                            // Belongs to the block boundary, not the instruction just lifted.
                            self.set_source_loc(cur);
                            self.builder.ins().jump(next_block, &[]);
                            self.block_terminated = true;
                            break;
                        }
                    }
                }
                first = false;

                if !decoded.contains_key(&cur) {
                    let off = (cur - self.lift_base) as usize;
                    match A::decode_one(&bytes[off..], cur) {
                        Some(inst) => {
                            decoded.insert(cur, inst);
                        }
                        None => {
                            self.current_ip = cur;
                            self.set_source_loc(cur);
                            A::sync_to_context(self, None, ExitReason::None)?;
                            self.emit_static_exit(
                                cur,
                                None,
                                crate::core::link::LinkSiteKind::StaticEdge,
                            );
                            self.block_terminated = true;
                            break;
                        }
                    }
                }
                let inst = &decoded[&cur];

                // Begins here, ends past the bytes given: a transfer *to* `cur`; the host resumes with a
                // region holding all of it. See [`LifterConfig::region_tail_exit`].
                if self.runs_off_region(cur, inst) {
                    self.current_ip = cur;
                    self.set_source_loc(cur);
                    // Linkable: once the host lifts a region at `cur`, later arrivals hit.
                    A::sync_to_context(self, None, ExitReason::UnknownJump)?;
                    self.write_exit_code(TransferKind::RegionTail as u32);
                    self.emit_static_exit(cur, None, crate::core::link::LinkSiteKind::StaticEdge);
                    self.block_terminated = true;
                    break;
                }

                if A::is_invalid(inst) {
                    // PC points AT the invalid instruction. Not linkable: a fault with no translation.
                    self.current_ip = A::instr_ip(inst);
                    self.set_source_loc(self.current_ip);
                    A::sync_to_context(self, None, ExitReason::InvalidInstruction)?;
                    self.emit_host_return();
                    self.block_terminated = true;
                    break;
                }

                // Advance `current_ip` first so PC reads see the post-instruction value; the source
                // location is the pre-advance address.
                let next_ip = A::instr_next_ip(inst);
                self.set_source_loc(A::instr_ip(inst));
                self.current_ip = next_ip;
                A::lift_instruction(self, inst)?;
                if self.block_terminated {
                    break;
                }
                cur = next_ip;
            }
        }

        // Phase 3: the entry dispatch, emitted late (see `dispatch_block`).
        if let Some(dispatch) = dispatch_block {
            self.builder.switch_to_block(dispatch);
            self.visited_blocks.insert(dispatch);

            let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
            let pc_val = self.ctx_load(types::I64, ctx_ptr, A::pc_offset());
            let entry_va = self.to_lift_va(pc_val);

            // Arms: every block under `All`; otherwise the entry, call return points and
            // resumable VM-exit points, plus the host's.
            let host_declared = &self.config.resume_points;
            let arms: BTreeSet<u64> = match self.config.resumption {
                VmResumptionPoints::None => unreachable!("no dispatch block"),
                VmResumptionPoints::All => self.blocks.keys().copied().collect(),
                VmResumptionPoints::ExitOnly => host_declared
                    .iter()
                    .copied()
                    .chain(call_resumes.iter().copied())
                    .chain(self.exit_resumes.iter().copied())
                    .chain(std::iter::once(ip))
                    .collect(),
            };
            let mut switch = Switch::new();
            for offset in arms {
                if let Some(&block) = self.blocks.get(&offset) {
                    switch.set_entry(offset.into(), block);
                    // Report only addresses that make progress. Out-of-region targets (exit stubs) and the
                    // instruction cut by [`LifterConfig::region_tail_exit`] leave the PC unchanged, so a
                    // host would spin on them.
                    let in_region = self.in_region(offset);
                    let tail = decoded
                        .get(&offset)
                        .is_some_and(|inst| self.runs_off_region(offset, inst));
                    if in_region && !tail {
                        self.emitted_resume_points.push(offset);
                    }
                }
            }

            // Default arm: a silent guess unless [`LifterConfig::strict_resume`].
            let refuse = self
                .config
                .strict_resume
                .then(|| self.builder.create_block());
            switch.emit(&mut self.builder, entry_va, refuse.unwrap_or(start_block));

            // Filled now: `switch.emit` terminates its block.
            if let Some(refuse) = refuse {
                self.builder.switch_to_block(refuse);
                // No sync: nothing is lifted yet, so the context holds what the host wrote, PC included.
                self.write_exit_reason(ExitReason::UnknownJump);
                self.write_exit_code(TransferKind::NoResumePoint as u32);
                self.emit_host_return();
                self.visited_blocks.insert(refuse);
            }
        } else if !self.config.analysis_mode {
            // [`VmResumptionPoints::None`]: the entry is the only enter point; report it so
            // `resume_points` is always what a host cache consults.
            self.emitted_resume_points.push(ip);
        }

        self.finish()
    }

    /// Fill created-but-unvisited blocks: VM exits to addresses outside the region.
    pub fn finish(&mut self) -> crate::Result<()> {
        let mut processed_blocks = HashSet::new();
        // **Sorted**, load-bearing: `HashMap` order is per-instance and would permute block
        // numbering and link-cell indices, costing reproducibility.
        let mut all_offsets: Vec<u64> = self.blocks.keys().cloned().collect();
        all_offsets.sort_unstable();

        for offset in all_offsets {
            let block = self.blocks[&offset];
            if !self.visited_blocks.contains(&block) && !processed_blocks.contains(&block) {
                self.builder.switch_to_block(block);
                // The stub is a transfer *to* `offset`, its only honest attribution.
                self.set_source_loc(offset);
                if self.config.analysis_mode {
                    // Inspection IR: bare return, no sync or exit code.
                    self.emit_host_return();
                } else {
                    self.current_ip = offset;
                    A::sync_to_context(self, None, ExitReason::UnknownJump)?;
                    let kind = self
                        .block_transfer_kinds
                        .get(&block)
                        .copied()
                        .unwrap_or(TransferKind::Unknown);
                    self.write_exit_code(kind as u32);
                    // Every stub is a transfer *to* `offset` (jump, taken edge, direct call), answered by
                    // entering a region there; knowing `offset` now lets it be a cell.
                    self.emit_static_exit(
                        offset,
                        None,
                        crate::core::link::LinkSiteKind::StaticEdge,
                    );
                }
                processed_blocks.insert(block);
            }
        }

        // After the stubs, which reach the probe.
        self.fill_link_probe()?;

        self.clear_source_loc();
        self.builder.seal_all_blocks();
        // After sealing: the use scan reads the finished layout.
        self.prune_untouched_sync();
        if self.config.eliminate_dead_flags {
            flags::eliminate(self.builder.func, &self.flag_values);
        }
        self.finalize_helper_indices();
        self.finalize_data_indices();
        Ok(())
    }
}
