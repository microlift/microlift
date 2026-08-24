//! Guest-agnostic lift engine: block discovery, the resumable PC-dispatch
//! switch, import fast-call machinery, context sync and exit handling, all
//! shared across guests via the [`GuestArch`] trait.
//!
//! `Lifter<'a>` is an alias for `LiftEngine<'a, X86_64>`.

pub mod abi;
pub mod alias;
pub mod bridge;
pub mod exit;
pub mod flags;
pub mod callret;
pub mod dce;
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
use crate::core::exit::{ExitReason, MemorySegment, TransferKind};
use crate::core::flags::{FlagEffects, FlagLiveness, FlagSet};
use crate::core::link::{BlockLinkConfig, BlockLinkOffsets};
use crate::core::optimize::Optimizations;
use crate::core::pin::{PinnedAbi, PinnedReg};
use crate::core::specialize::Specialization;
use crate::core::callret::{CallRetConfig, CallRetOffsets};
use crate::core::hostframe::{HostFrameConfig, HostFrameOffsets};
use crate::core::srcloc::SourceLocMap;
use crate::function_traits::{DynamicSymbol, FunctionArgumentType};
use crate::os::get_host_calling_conv;
use cranelift::prelude::*;
use cranelift_codegen::ir::{
    ExtFuncData, ExternalName, FuncRef, Function, Inst, SigRef, SourceLoc, UserExternalName,
    UserExternalNameRef, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A guest instruction-set frontend.
///
/// Instruction semantics are inherent methods on the concrete
/// `LiftEngine<'a, Self>`, reached via `lift_instruction`.
pub trait GuestArch: Sized {
    /// The longest encoding this guest has, in bytes: the width of the window at
    /// the end of a region in which an instruction may not fit. See
    /// [`LifterConfig::region_tail_exit`].
    const MAX_INSTRUCTION_LEN: u64;

    /// A single decoded guest instruction.
    type Instr;
    /// SSA-variable register file (guest registers, flags, dirty tracking, plus
    /// the universal `context_ptr` / `image_base` variables).
    type RegState;

    /// Byte offset of the architectural PC field within the guest's
    /// `RegisterContext`.
    fn pc_offset() -> i32;
    /// Byte offset of the `exit_reason` control field.
    fn exit_reason_offset() -> i32;
    /// Byte offset of the `exit_code` control field.
    fn exit_code_offset() -> i32;
    /// Byte offset of the `image_base` relocation field.
    fn image_base_offset() -> i32;
    /// Byte offset of the `imported_function_va` control field.
    fn imported_function_va_offset() -> i32;

    /// Where this guest's context keeps the two block-linking fields, or `None`,
    /// which makes [`LifterConfig::block_link`] inert for this guest.
    fn block_link_offsets() -> Option<BlockLinkOffsets> {
        None
    }

    /// Where this guest's context keeps the call-return ring's base and offset,
    /// or `None` for a guest that has not made room for them - which makes
    /// [`LifterConfig::call_return_stack`] inert, exactly as
    /// [`Self::block_link_offsets`] makes [`LifterConfig::block_link`] inert.
    fn call_return_offsets() -> Option<crate::core::callret::CallRetOffsets> {
        None
    }

    /// Where this guest's context keeps the host-frame depth, or `None` for a
    /// guest that has not made room for it - which makes
    /// [`LifterConfig::host_call_frames`] inert, exactly as
    /// [`Self::block_link_offsets`] makes [`LifterConfig::block_link`] inert.
    fn host_frame_offsets() -> Option<crate::core::hostframe::HostFrameOffsets> {
        None
    }

    /// The Cranelift type a pinned `reg` is passed as, or `None` to refuse it.
    ///
    /// `None` from any slot disables pinning for the whole lift - x86-64
    /// refuses a vector slot without [`LifterConfig::use_vector_xmm`], where
    /// that register is two variables and a slot is one value. Must agree with
    /// [`pinned_slot_var`](GuestArch::pinned_slot_var).
    fn pinned_slot_type(_reg: PinnedReg, _config: &LifterConfig) -> Option<types::Type> {
        None
    }

    /// The SSA variable holding pinned `reg`'s value, or `None` for a register
    /// this guest does not pin.
    fn pinned_slot_var(_state: &Self::RegState, _reg: PinnedReg) -> Option<Variable> {
        None
    }

    /// Byte offset of pinned `reg` within this guest's `RegisterContext`, where
    /// the entry thunk loads it from and stores it back to.
    fn pinned_slot_offset(_reg: PinnedReg) -> Option<i32> {
        None
    }

    /// The variable holding the pointer to the guest `RegisterContext`.
    fn context_ptr_var(state: &Self::RegState) -> Variable;
    /// The variable holding the runtime image base (for ASLR-resilient PC).
    fn image_base_var(state: &Self::RegState) -> Variable;

    fn new_state(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self::RegState;

    /// Every `RegisterContext` field reachable at a constant offset, as
    /// `(offset, name)`.
    fn context_fields(_config: &LifterConfig) -> Vec<(i32, &'static str)> {
        Vec::new()
    }

    fn decode_region(bytes: &[u8], ip: u64) -> Vec<Self::Instr>;
    /// Decode the single instruction at the start of `bytes`, at virtual address
    /// `ip`. `None` only when `bytes` is too short to hold any instruction;
    /// invalid encodings come back as an instruction for which
    /// [`is_invalid`](GuestArch::is_invalid) is true.
    fn decode_one(bytes: &[u8], ip: u64) -> Option<Self::Instr>;
    fn instr_ip(i: &Self::Instr) -> u64;
    fn instr_next_ip(i: &Self::Instr) -> u64;
    fn is_invalid(i: &Self::Instr) -> bool;
    /// Static control-flow facts for leader discovery.
    fn control_flow(i: &Self::Instr) -> ControlFlow;

    /// Register/memory read-write effects of `i`, or `None` if this frontend
    /// cannot compute them.
    fn instruction_effects(_i: &Self::Instr) -> Option<InstructionEffects> {
        None
    }

    /// What lifting `i` does to the guest's condition flags, in this guest's own
    /// flag numbering, or `None` when this frontend cannot say.
    ///
    /// Drives [`LifterConfig::eliminate_dead_flags`]; `None` is read as
    /// [`FlagEffects::CONSERVATIVE`]. **This describes the lift, not the
    /// architecture** - read [`FlagEffects`] before adding an entry, since a
    /// wrong direction is a miscompile with no diagnostic. Bit indices are the
    /// same ones the frontend passes to [`LiftEngine::flag_live`].
    fn flag_effects(_i: &Self::Instr) -> Option<FlagEffects> {
        None
    }

    /// Resolve indirect `JMP`/`CALL` targets in `bytes` (mapped at `base`) via
    /// value-set analysis, consulting `segments` for reads leaving the region.
    fn resolved_indirect_targets(
        _bytes: &[u8],
        _base: u64,
        _segments: &[crate::context::MemorySegment],
    ) -> crate::cfg::ResolvedTargets {
        crate::cfg::ResolvedTargets::default()
    }

    /// **Experimental.** Recover indirect-branch successors in `bytes` (mapped at
    /// `base`) by recognising jump tables, keyed by branch address. Consulted
    /// only under [`LifterConfig::analyze_jump_tables`]. Results only *add*
    /// leaders, so a wrong answer costs dead code, never correctness.
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

/// A helper function emitted alongside the lifted region - see
/// [`LiftEngine::outlined_helper`].
///
/// Backends must define it with [`Linkage::Local`](cranelift_module::Linkage::Local)
/// *before* the lifted function, in [`LiftedCode::helpers`] order, so its
/// `FuncId` matches the `ExternalName::user(0, i)` baked into the call.
#[derive(Debug, Clone)]
pub struct HelperFunction {
    /// Symbol the helper is declared under. Unique per helper body.
    pub name: &'static str,
    /// The helper's IR.
    pub func: Function,
}

/// Everything one lift produces: the region's function plus the externals it
/// references.
///
/// `ExternalName::user(0, i)` covers both lists, imports first: `i <
/// imports.len()` is `imports[i]`, `imports.len() + j` is `helpers[j]`.
/// Backends must declare them in that order so `FuncId(i) == i`.
pub struct LiftedCode {
    /// The lifted region: `extern fn(*mut RegisterContext)`.
    pub func: Function,
    /// External host symbols the code calls (`RuntimeAssisted` fast imports).
    /// Same contents as [`LiftEngine::aot_imports`].
    pub imports: Vec<(&'static str, *const u8, Signature)>,
    /// Locally-defined helper bodies the code calls, e.g. the outlined x87
    /// softfloat kernels.
    pub helpers: Vec<HelperFunction>,
    /// Analysis-mode call-target map; see
    /// [`lift_to_function_with_call_targets`].
    pub analysis_call_targets: Vec<u64>,
    /// Decodes the [`SourceLoc`](cranelift_codegen::ir::SourceLoc)s carried by
    /// [`func`](Self::func) back to guest addresses. Meaningful only when the
    /// lift ran with [`LifterConfig::emit_source_locs`].
    pub source_map: SourceLocMap,
    /// Every guest address this code can be **entered at**, ascending - the arms
    /// of the entry dispatch.
    ///
    /// A host cache keyed by address must key on exactly this: entering
    /// anywhere else either runs the region from the top or exits with
    /// [`TransferKind::NoResumePoint`](crate::core::exit::TransferKind::NoResumePoint)
    /// under [`LifterConfig::strict_resume`]. Width follows
    /// [`LifterConfig::resumption`]; empty under [`LifterConfig::analysis_mode`].
    pub resume_points: Vec<u64>,
    /// The `extern "C" fn(*mut RegisterContext)` a **pinned** region is entered
    /// through, or `None` when the region is host-callable on its own.
    ///
    /// Present exactly when [`pinned_abi_for`] answered yes. [`func`](Self::func)
    /// is then the region *body*, carrying the tail calling convention; the
    /// thunk loads the pinned slots from the context, calls the body, and stores
    /// the results back. Backends declare the body **after** the helpers,
    /// continuing the same index space, and export the thunk as the entry point.
    pub entry_thunk: Option<Function>,
    /// The guest bytes this translation was derived from, ascending and
    /// disjoint, as **lift-time** VAs.
    ///
    /// **The invalidation key.** A write anywhere in these ranges may change
    /// what the region means, so a host holding the translation must discard
    /// it: the address compare a link performs proves a row is *about* a PC,
    /// never that it is still true. See [`crate::core::link`].
    ///
    /// Exactly [`LifterConfig::block_ranges`] clamped to the region, so a lift
    /// given holes depends on no byte inside one. It over-approximates the
    /// instructions actually decoded, which is the safe direction:
    /// [`SourceLocMap::guest_addresses`] is a *subset* (an instruction whose
    /// result is dead leaves no trace), and keying invalidation on a subset
    /// would miss a write.
    pub covered: Vec<Range<u64>>,
    /// Guest bytes **outside** [`covered`](Self::covered) that were read at lift
    /// time and baked into the code: today only the tables recovered by
    /// [`LifterConfig::analyze_jump_tables`]. Empty otherwise.
    ///
    /// Separate from `covered` because the two differ in what a host may do
    /// about them. `covered` is code, and a host that write-protects guest code
    /// pages to catch self-modification covers these only if it protects them
    /// too.
    ///
    /// Staleness here is benign in the way staleness in `covered` is not: the
    /// jump-table pass only ever *adds* block leaders and dispatch still happens
    /// on the runtime address, so a table whose bytes changed leaves unreachable
    /// code lifted and falls back to the pre-existing exit. It is reported so a
    /// host can be exact, not because it must be.
    pub data_dependencies: Vec<Range<u64>>,
    /// This region's link cells, in cell order - cell `i` is `link_sites[i]`,
    /// at byte `i * 32` of the array.
    ///
    /// Empty unless [`LifterConfig::link_cells`] was on. A host installs a cell
    /// by writing a [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) for
    /// [`LinkSite::target`](crate::core::link::LinkSite::target) into it, and
    /// retracts it by zeroing the row's `code`. See
    /// [`CompiledRegion::link_cells`](crate::backend::CompiledRegion::link_cells)
    /// for where the array itself is.
    pub link_sites: Vec<crate::core::link::LinkSite>,
}

impl LiftedCode {
    /// Wrap a bare [`Function`] as a `LiftedCode` that reports **no** imports,
    /// helpers, source locations, resume points or guest extent.
    ///
    /// For the older `Function`-taking backend entry points, which are handed
    /// IR with no record of where it came from. Every field is empty because
    /// that is the honest answer, not because it is a placeholder: a host that
    /// needs resume points or an invalidation key must lift through
    /// [`lift_to_code`] and compile through
    /// [`compile_lifted_region`](crate::backend::CraneliftBackend::compile_lifted_region).
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
        }
    }

    /// [`from_function`](Self::from_function) with relocatable fast-call
    /// imports, whose index `i` must correspond to `ExternalName::user(0, i)`.
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

/// The pinned ABI a lift will actually run under, or `None`.
///
/// **The one place the question is asked**, so the signature builder, entry
/// prologue, exit paths and link probe cannot disagree. Four conditions, all
/// falling back silently to `HostResume` rather than erroring: linking is on
/// and set to [`link::LinkDispatch::Pinned`]; the ABI is non-empty; this
/// guest has the link fields ([`GuestArch::block_link_offsets`]); and it can
/// type **every** slot ([`GuestArch::pinned_slot_type`]).
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

/// The Cranelift type of each slot of `abi`, in order.
fn pinned_slot_types<A: GuestArch>(abi: &PinnedAbi, config: &LifterConfig) -> Vec<types::Type> {
    abi.slots()
        .iter()
        .map(|&reg| {
            A::pinned_slot_type(reg, config).expect("pinned_abi_for typed every slot already")
        })
        .collect()
}

/// Build the `extern "C"` entry thunk for a pinned region: fill the pinned
/// registers from the context, call the body, spill them back.
///
/// `body_index` is the body's slot in the namespace-0 index space, after the
/// imports and helpers - hence this being built at the end of a lift.
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
    // Result 0 is the context pointer handed back - see `core::pin`. The stores
    // go through the pointer the *thunk* was given: same value, one fewer live range.
    let results: Vec<Value> = builder.inst_results(call)[1..].to_vec();
    for (&reg, value) in abi.slots().iter().zip(results) {
        let offset = A::pinned_slot_offset(reg).expect("placed above");
        builder.ins().store(flags, value, ctx_ptr, offset);
    }
    builder.ins().return_(&[]);
    // `finalize` only consults this to lay out stack maps, which the thunk never requests.
    builder.finalize(isa::TargetFrontendConfig {
        default_call_conv: get_host_calling_conv(),
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
    func
}

/// Lift a region of guest code to a Cranelift [`Function`] plus the external
/// functions it references.
///
/// The complete form of [`lift_to_function`], and the one a backend should
/// use: x87 arithmetic calls into outlined helpers, whose bodies come back
/// in [`LiftedCode::helpers`].
pub fn lift_to_code<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<LiftedCode> {
    // A pinned region is a `tail` function whose parameters *are* the pinned half of
    // the register file; everything else is `extern "C" fn(*mut RegisterContext)`.
    // Both start with the context pointer, so nothing below cares which it got.
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
    let data_dependencies = engine.jump_table_extents();
    let helpers = engine.take_helpers();

    // Release the mutable borrow of `func` held by the builder/engine.
    drop(engine);

    // The body is declared after the imports and the helpers, continuing their
    // index space; only now are both counts final.
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
    })
}

/// Lift a region of guest code straight to a Cranelift [`Function`].
///
/// `A` selects the guest architecture; `ip` is the virtual address the first
/// byte maps to. Errors if the lift produced outlined helpers - use
/// [`lift_to_code`] for those.
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

/// Reduce a [`LiftedCode`] to its main function, erroring if that would drop
/// helper bodies the function calls.
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

/// Like [`lift_to_function`], but also returns the analysis-mode call-target
/// map ([`LiftEngine::analysis_call_targets`]): index `i` is the callee VA
/// for `ExternalName::user(ANALYSIS_CALL_NAMESPACE, i)`. Empty unless
/// `config.analysis_mode` is set.
pub fn lift_to_function_with_call_targets<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<(Function, Vec<u64>)> {
    require_no_helpers(lift_to_code::<A>(bytes, ip, config)?)
}

/// Convert a `FunctionArgumentType` to the corresponding Cranelift IR type.
/// Returns `None` for `None` (void) and `VariadicList` (stop marker).
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

/// Build a Cranelift `Signature` from a `DynamicSymbol`'s `arg_types`:
/// `arg_types[0]` is the return type, `[1..]` the parameters. Stops at the
/// first `VariadicList`/`None` marker, the same cutoff
/// [`CallAbi::place_args`] uses.
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

/// Build the [`FastCall`] data for one fast import, declaring the callee's
/// signature (and in `RuntimeAssisted` mode its external function and
/// `aot_imports` entry) on the way.
///
/// `None` when the import cannot be fast-called - variadic, or an ABI that
/// cannot place every argument - and the caller falls back to the slow exit.
/// Shared by config-time prebuilding and the lazy registry path, so both
/// keep the one-`FuncRef`-per-symbol-name invariant `aot_imports` needs.
fn build_fast_call(
    builder: &mut FunctionBuilder,
    execution_mode: ExecutionMode,
    aot_imports: &mut Vec<(&'static str, *const u8, Signature)>,
    fast_call_by_name: &mut HashMap<&'static str, FastCall>,
    import: &FastImport,
) -> Option<FastCall> {
    let sym = &import.symbol;
    // Variadic functions, and arguments the registered ABI cannot place, demote to slow.
    if sym.is_ellipsis_function || !import.abi.can_fast_call(&sym.arg_types) {
        return None;
    }
    // The signature describes the *host* callee; the ABI descriptor describes
    // where the guest left the arguments.
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

/// A registered fast-call import: the host function to call, plus the guest
/// calling convention its call sites use.
#[derive(Debug, Clone)]
pub struct FastImport {
    pub symbol: DynamicSymbol,
    pub abi: CallAbi,
}

/// How a fast-call import is reached from the lifted code.
#[derive(Debug, Clone, Copy)]
pub enum FastCallTarget {
    /// `Jit` mode: `call_indirect` through the symbol's absolute address, using
    /// the imported signature.
    Absolute { sig_ref: SigRef, fn_ptr: i64 },
    /// `RuntimeAssisted` mode: `call` a declared external function, so the
    /// object backend emits a named relocation instead of baking in an address.
    Relocatable(FuncRef),
}

/// Everything the per-guest emitter needs to inline one fast-call import.
#[derive(Debug, Clone)]
pub struct FastCall {
    pub target: FastCallTarget,
    /// `[0]` is the return type; `[1..]` the parameter types.
    pub arg_types: Vec<FunctionArgumentType>,
    /// Guest convention the call site follows.
    pub abi: CallAbi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Jit,
    RuntimeAssisted,
}

/// How the lifted code forms guest virtual addresses (RIP-relative operands,
/// PCs written on a VM exit, pushed return addresses, and the dispatch
/// mapping a runtime address back to a block).
///
/// Every such address is a *lift-time* VA; this picks how it is
/// materialized at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    /// `image_base + (va - lift_base)`, with `image_base` loaded from the context
    /// at entry. Stays correct when the image is loaded at a different base,
    /// which is what makes compiled code safe to cache on disk and reuse across
    /// runs (and what `RuntimeAssisted` output requires).
    #[default]
    ImageBaseRelative,
    /// Addresses baked in as `iconst`, the reverse mapping the identity. Cheaper,
    /// but valid only at the base it was lifted for.
    ///
    /// For an in-process JIT that re-lifts each run. `RegisterContext::image_base`
    /// is then ignored, and the PC on exit/resume is a plain lift-time VA.
    Direct,
}

/// How a guest 128-bit compare-and-swap (x86-64 `CMPXCHG16B`) is realized.
///
/// Cranelift lowers `atomic_cas` on `I128` only on the x64 backend, and only
/// with the `has_cmpxchg16b` ISA setting (off in the baseline `x86-64`
/// feature set, so AOT callers pass it via `isa_settings`). Elsewhere there
/// is no IR-expressible lock-free 128-bit RMW at all, hence a knob rather
/// than a fixed lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Atomic128Mode {
    /// Plain 128-bit load/compare/store. Portable everywhere and correct for a
    /// single-threaded guest, but **not atomic**.
    ///
    /// The default, on the grounds that it is the lowering that compiles
    /// everywhere. Pick another mode for a location actually shared.
    #[default]
    Emulated,
    /// Emit `atomic_cas` on `I128`, which the x64 backend lowers to a real
    /// `lock cmpxchg16b`. Exact and fast, but the compile **fails** on any
    /// backend that cannot lower it (aarch64, riscv64, s390x) or on an x64 ISA
    /// built without `has_cmpxchg16b`.
    Native,
    /// Do not lower it: exit with [`ExitReason::UnimplementedInstruction`] so the
    /// runtime performs the operation itself, e.g. under a host-side lock. The
    /// only portable option that is also correct under concurrency, at the cost
    /// of a VM exit per execution.
    ///
    /// Follows the ordinary unimplemented-instruction contract: state fully
    /// synced, `exit_code` holds the iced `Code`, and the PC is already past
    /// the `CMPXCHG16B`, so the host emulates and resumes without adjusting it.
    ///
    /// [`ExitReason::UnimplementedInstruction`]: crate::context::ExitReason::UnimplementedInstruction
    Exit,
}

/// What bit pattern a *generated* NaN gets.
///
/// An SSE instruction can **propagate** a NaN an operand held (quieted, sign
/// and payload intact), which every host FPU does x86's way, so nothing is
/// emitted for it. Or it can **generate** one out of an invalid operation
/// (`inf + -inf`, `0 * inf`, `0 / 0`, `sqrt` of a negative), and there the
/// ISAs disagree: x86 yields the *floating indefinite*, sign bit set
/// (`0xFFC0_0000` at single precision), where AArch64's default NaN is the
/// otherwise identical `0x7FC0_0000`.
///
/// Telling them apart at run time is the cost: "result is NaN" is not enough,
/// since a propagated NaN must be left alone, so the test is "result is NaN
/// **and** the operands were ordered" - six or seven instructions wrapped
/// around arithmetic that is one, on every FP operation in the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneratedNan {
    /// Materialise x86's floating indefinite, whatever the host FPU would have
    /// produced. Bit-exact against real hardware, and the default: a lifter
    /// that does not know its guest cannot know the guest will not look.
    #[default]
    X86Exact,
    /// Let the host FPU's own default NaN stand.
    ///
    /// Correct for a guest that never observes the bits of a NaN it generated,
    /// which is most of them. What diverges is a guest that inspects such a
    /// NaN's sign bit, or hashes/serialises its raw bits.
    ///
    /// Host-dependent by construction: on an x86-64 host the FPU's default *is*
    /// the floating indefinite, so this is bit-identical there and only changes
    /// anything where the host's default NaN differs. Propagated NaNs are
    /// unaffected either way.
    HostDefault,
}

/// Which addresses a lifted region's entry dispatch offers for re-entry.
///
/// The dispatch is a `Switch` on the incoming PC, and it is the biggest
/// single lever on what a large region costs to compile: every address with
/// an arm is a block reachable from the function entry, so every guest
/// register is live into it, and regalloc is superlinear in the live ranges
/// that produces. Over 128 KiB of a real program's `__text` in 4 KiB pieces:
/// 1.42s of compile time with an arm per block leader, 19.36s with one per
/// instruction (16 KiB pieces: 5.20s against 134.21s).
///
/// Pick the smallest mode that still lets the host come back to everywhere
/// it can leave from. [`LiftedCode::resume_points`] reports the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VmResumptionPoints {
    /// No dispatch at all - the region is enterable only at its own entry.
    ///
    /// The PC the host wrote is never read, so entering anywhere else is
    /// *silently wrong*: the region runs from the top. Only for a host that
    /// answers every VM exit by translating a fresh region at the exit PC.
    ///
    /// [`LiftedCode::resume_points`] then reports exactly one address.
    None,
    /// **Default.** An arm for every address control can *leave* the region at
    /// and come back to: the region's own entry, the return point of every call
    /// that leaves it, and the resume point of every VM exit that resumes
    /// (`CPUID`, `RDTSC`, `XGETBV`, `SYSCALL`, `INT`, `INT3`, port I/O, an
    /// emulated unimplemented instruction), registered via
    /// [`LiftEngine::resumes_at`].
    ///
    /// Everywhere else, control got there from inside the region and never left
    /// it. Anything the host knows about goes in [`LifterConfig::resume_points`]
    /// and is offered on top of these.
    #[default]
    ExitOnly,
    /// Legacy. Every instruction boundary is a resume point.
    ///
    /// Correct for any re-entry, and expensive - see the timings above. Kept
    /// for hosts that genuinely resume at arbitrary addresses, and for tests
    /// that want to enter anywhere.
    All,
}

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    /// How guest virtual addresses are materialized. See [`AddressMode`];
    /// defaults to [`AddressMode::ImageBaseRelative`], which is required for
    /// relocatable/cacheable artifacts.
    pub address_mode: AddressMode,
    pub syscall_handler_call_conv: CallConv,
    /// The layout of immutable segments in memory at lift time, letting VSA
    /// promote memory values to lift-time constants.
    ///
    /// **Do NOT use this for writable data** - the contents are assumed
    /// immutable.
    ///
    /// Behind an [`Arc`] because a `MemorySegment` owns a copy of the guest
    /// bytes it describes, so cloning the config per region deep-copies whole
    /// mapped images: measured at 311 us per region, 7% of wall clock. Use
    /// [`set_segments`](Self::set_segments), or `Arc::make_mut`.
    pub segments: Arc<Vec<MemorySegment>>,
    pub interpreter_fallback: bool,
    /// Which guest addresses the region can be **re-entered** at. See
    /// [`VmResumptionPoints`]; defaults to
    /// [`ExitOnly`](VmResumptionPoints::ExitOnly).
    ///
    /// Pair with [`resume_points`](Self::resume_points), which adds the host's
    /// own addresses, and [`strict_resume`](Self::strict_resume), which decides
    /// what happens to an address that is not offered.
    pub resumption: VmResumptionPoints,
    /// **Experimental.** Before emitting IR, try to recover each indirect
    /// branch's successors from a compiler-emitted jump table, read out of
    /// [`segments`](Self::segments) (the region itself is always readable, so an
    /// in-function table needs no segment).
    ///
    /// A Clang `switch` otherwise costs a VM exit per dispatch: `JMP reg`
    /// contributes no static targets, so the case blocks never become leaders.
    /// With this on they do, and the [`ExitReason::UnknownJump`] stub survives
    /// only as the default arm.
    ///
    /// Targets are only ever *added* as leaders and dispatch still happens on
    /// the runtime address, so a mispredicted entry lifts unreachable code
    /// rather than producing wrong code. Off by default because the matching is
    /// heuristic and the extra blocks are not free.
    pub analyze_jump_tables: bool,
    /// Statically registered import addresses: guest VA -> how it is reached
    /// (see [`ImportKind`]).
    ///
    /// A snapshot taken when the config is built. Addresses that only become
    /// imports later (a `dlopen`ed image, a lazily bound stub) belong in
    /// [`import_registry`](Self::import_registry), which is queried while
    /// lifting.
    ///
    /// Behind an [`Arc`] because a host clones this config once per region and
    /// the import map is the same for all of them - measured at 478ms of a
    /// 935ms lift budget. Mutate through
    /// [`register_fast_import`](Self::register_fast_import) and friends, which
    /// copy on write.
    pub imports: Arc<HashMap<u64, ImportKind>>,
    /// A live view of which addresses are imports, queried per call site at lift
    /// time rather than snapshotted here.
    ///
    /// Consulted only for addresses absent from [`imports`](Self::imports), so
    /// an explicit registration always wins.
    pub import_registry: Option<Arc<dyn ImportRegistry>>,
    /// Optional bridge for indirect calls whose target is only known at run time
    /// (`CALL reg` / `CALL [mem]` to an unregistered address).
    ///
    /// When set, an indirect call becomes a call to
    /// `extern "C" fn(*mut RegisterContext, target: u64) -> u32` instead of an
    /// [`ExitReason::IndirectCall`] exit: the runtime dispatches on the host
    /// side and returns 0 when it handled the call inline, non-zero when the
    /// target is guest code and a real exit is required.
    ///
    /// On the non-zero path the context already describes a well-formed
    /// `IndirectCall` exit; the bridge may overwrite `exit_reason`/`exit_code`.
    pub indirect_call_bridge: Option<BridgeImport>,
    /// Store vector registers as native `I64X2` SSA variables and take the vector
    /// fast paths, instead of scalar lo/hi 64-bit pairs (the default).
    ///
    /// Off by default because the scalar form avoids relying on Cranelift's
    /// vector lowering, which AOT targets without SIMD (e.g. RISCV-64) need.
    pub use_vector_xmm: bool,
    /// Which target-specific instruction specializations the lifter may apply.
    ///
    /// No specialization changes semantics, it only helps with their speed.
    pub specialization: Specialization,
    /// How a guest 128-bit compare-and-swap (`CMPXCHG16B`) is lowered. See
    /// [`Atomic128Mode`]; defaults to the portable, non-atomic emulation.
    pub atomic128: Atomic128Mode,
    /// What bit pattern an SSE instruction's generated NaN gets. See
    /// [`GeneratedNan`]; defaults to [`GeneratedNan::X86Exact`], which is what
    /// hardware does and costs six or seven instructions per FP operation.
    pub generated_nan: GeneratedNan,
    /// Produce inspection IR for static analysis rather than runnable IR.
    ///
    /// The entry PC-dispatch becomes a direct jump to the first block, context
    /// write-back is skipped so registers stay pure SSA, and out-of-region exit
    /// stubs become bare `return`s.
    ///
    /// The result is a clean CFG of the guest's own blocks, but it does not
    /// write guest state back, so it must never be executed.
    pub analysis_mode: bool,
    /// The guest return address that means "leave the region and hand control
    /// back to the host", or `None` for a host that has no such address.
    pub exit_trampoline_address: Option<u64>,
    /// Emit large shared code sequences - currently the 80-bit x87 softfloat
    /// kernels - once as helper functions and `call` them, rather than inlining
    /// a copy at every site. On by default: the kernels run to hundreds of IR
    /// instructions each.
    ///
    /// Off yields a single self-contained `Function`, which is what
    /// [`lift_to_function`] requires, and is worthwhile for a region with one
    /// x87 operation where the call is pure overhead.
    ///
    /// Ignored under [`analysis_mode`](Self::analysis_mode), which always inlines.
    pub outline_helpers: bool,
    /// Tag every emitted IR instruction with the guest instruction it was lifted
    /// from, as a Cranelift `SourceLoc` (see [`crate::core::srcloc`]).
    pub emit_source_locs: bool,
    /// Keep out of the emitted code the context loads and stores the region had
    /// no use for - see [`crate::core::regsync`].
    pub lazy_register_sync: bool,
    /// Exit rather than decode when an instruction's bytes run past the end of
    /// the region.
    ///
    /// [`ExitReason::InvalidInstruction`]: crate::core::exit::ExitReason::InvalidInstruction
    /// [`ExitReason::UnknownJump`]: crate::core::exit::ExitReason::UnknownJump
    /// [`TransferKind::RegionTail`]: crate::core::exit::TransferKind::RegionTail
    pub region_tail_exit: bool,
    /// The guest page size [`page_bounded_extent`](Self::page_bounded_extent)
    /// cuts regions on.
    pub page_size: u64,
    /// The sub-ranges of the byte slice that are actually code, when the host
    /// knows and the slice is not all of it. Empty - the default - means the
    /// whole slice.
    pub block_ranges: Vec<Range<u64>>,
    /// Extra addresses the host wants the entry dispatch to offer, on top of
    /// whatever [`resumption`](Self::resumption) produces.
    ///
    /// Each becomes a block leader whether or not control flow discovered it, so
    /// a function reached only through a pointer is still lifted and enterable.
    /// Addresses outside the region are ignored, and nothing here is offered
    /// under [`VmResumptionPoints::None`], which emits no dispatch.
    pub resume_points: Vec<u64>,
    /// Do not emit the IR for a condition-flag write nothing goes on to read.
    ///
    /// Guest flags are eager: an `ADD` emits all six whether or not anything
    /// reads them, and PF alone is a `popcnt`/`ireduce`/`band`/`bxor`. Being SSA
    /// the dead ones are deleted at compile time, so this changes no *machine
    /// code* - it changes how much IR the compiler is handed to get there,
    /// which is where a recompiler's time goes.
    ///
    /// Off by default because soundness rests on the per-instruction
    /// classifications in [`GuestArch::flag_effects`] matching what the handlers
    /// do, and a wrong entry there is a wrong answer rather than an error. Inert
    /// on a frontend that does not implement it.
    pub eliminate_dead_flags: bool,
    /// Exit rather than guess when the entry PC matches no resume point.
    ///
    /// The dispatch needs a default arm, and historically that arm is the
    /// region's **first block** - so a host resuming at an address the region
    /// cannot start at silently executes from the top, with no fault and no
    /// diagnostic.
    ///
    /// With this on the default arm is an [`ExitReason::UnknownJump`] carrying
    /// [`TransferKind::NoResumePoint`], leaving the PC exactly as the host set
    /// it and no guest state touched, so the host can translate a region at that
    /// address instead. Inert under [`VmResumptionPoints::None`].
    ///
    /// [`ExitReason::UnknownJump`]: crate::core::exit::ExitReason::UnknownJump
    /// [`TransferKind::NoResumePoint`]: crate::core::exit::TransferKind::NoResumePoint
    pub strict_resume: bool,
    /// Let a region resolve its own successor out of a host-provided table,
    /// instead of handing every guest call and return back to the host to look
    /// up. See [`crate::core::link`] for what is emitted.
    /// Off by default.
    pub block_link: Option<BlockLinkConfig>,
    /// Emit the link probe at each exit site rather than once per region.
    ///
    /// Disabled by default in the name of preventing emitted code size from bloating
    /// to an extreme degree.
    pub inline_link_probes: bool,
    /// Give every exit whose destination is known at **lift time** a
    /// [`LinkSite`] of its own, so it links through a dedicated cell instead of
    /// hashing a PC into the shared table. Off by default.
    ///
    /// Microlift's answer to FEX's backpatched direct branches; see
    /// [`LinkSite`] for why it is a data cell rather than a rewritten call site,
    /// and for what it buys over the probe. Inert unless [`block_link`] is on -
    /// a cell is a [`BlockLinkEntry`] and the whole mechanism is part of block
    /// linking, not an alternative to it.
    ///
    /// **This makes the compiled region bigger, and is meant to.** The probe
    /// stays as the fallback - a cell is empty until its target is compiled, and
    /// empty again after an invalidation - so a cell is a fast path bolted in
    /// front of the slow one rather than a replacement for it. Measured over
    /// reachable out-of-region conditional branches: **+42 bytes of `.text` per
    /// exit on x86-64 and +28 on aarch64**, plus 32 bytes of zero-initialised
    /// data per *distinct* target.
    ///
    /// So the case for it is never code size. It is what the exit *executes*: a
    /// load and a branch instead of a shift, mask, scale, add, load and compare
    /// - and, mattering more, a lookup that cannot collide, where the table
    /// aliases two hot targets sixteen bytes apart into one row and turns a link
    /// into a permanent miss. That is not visible in a size measurement, which
    /// is why this is opt-in and why the tests that prove it works are
    /// behavioural (`link_cell_tests`, which reaches a successor with the table
    /// detached) rather than a size delta.
    ///
    /// Budget accordingly: on a page-bounded region of branch-dense code, cells
    /// can rival the code itself.
    ///
    /// [`LinkSite`]: crate::core::link::LinkSite
    /// [`block_link`]: Self::block_link
    /// [`BlockLinkEntry`]: crate::core::link::BlockLinkEntry
    pub link_cells: bool,
    /// Predict a guest `RET` out of a ring of `<return address, link cell>`
    /// frames pushed at each call, instead of hashing the popped address into
    /// the shared table. Off by default.
    ///
    /// A port of FEX's call-return shadow stack, with the one substitution
    /// Cranelift forces on it - see [`crate::core::callret`] for what that is,
    /// what it buys, and why the ring needs no guard pages.
    ///
    /// Requires [`link_cells`](Self::link_cells): a hit hands back a cell, so
    /// there is nothing for one to name without them. Also inert unless
    /// [`block_link`](Self::block_link) is on and the guest supplies
    /// [`GuestArch::call_return_offsets`] - all three asked in
    /// `LiftEngine::call_return`.
    pub call_return_stack: Option<crate::core::callret::CallRetConfig>,
    /// Nest a guest `CALL`/`RET` pair as a real **host** call and return, instead
    /// of leaving both to the link machinery. Off by default.
    ///
    /// The alternative to [`call_return_stack`](Self::call_return_stack) and not a
    /// companion to it: that one predicts a return out of a software ring, this
    /// one makes the return a `ret` the CPU's return-address stack predicts. They
    /// can be enabled together - nothing breaks, the ring simply never gets to
    /// answer a `RET` the host frame already did - but the point of having both
    /// compiled in is to measure one against the other.
    ///
    /// Requires [`link_cells`](Self::link_cells), because the callee's address
    /// comes from a cell. Also inert unless the guest supplies
    /// [`GuestArch::host_frame_offsets`] - all three asked in
    /// `LiftEngine::host_call_frames`.
    pub host_call_frames: Option<crate::core::hostframe::HostFrameConfig>,
    /// Optimizations for the emitted code. Some optimizations can trade off semantic accuracy
    /// of not commonly relied upon behavior, such as NAN accuracy.
    pub optimizations: Optimizations,
}

// `ImportKind` carries a resolved host address, so the map is not `Send`/`Sync`
// and clippy suggests `Rc`. `Arc` is deliberate: a host may move a config
// between threads under its own safety argument, which a non-atomic refcount
// would make unsound.
#[allow(clippy::arc_with_non_send_sync)]
impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            address_mode: AddressMode::ImageBaseRelative,
            syscall_handler_call_conv: CallConv::SystemV,
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
    /// Replace the immutable-memory hints. See [`segments`](Self::segments).
    pub fn set_segments(&mut self, segments: Vec<MemorySegment>) {
        self.segments = Arc::new(segments);
    }

    /// Register an import with full fast-call metadata.
    ///
    /// `abi` describes how the *guest* passes arguments - the convention the
    /// guest binary was compiled for, which need not match the host's. Use a
    /// preset such as [`CallAbi::sysv_amd64`](crate::core::abi::CallAbi::sysv_amd64).
    ///
    /// Demoted to a slow `ImportedFunctionCall` exit when variadic, or when
    /// `abi` cannot place all the arguments.
    ///
    /// The callee must not touch the `RegisterContext`, directly or by
    /// re-entering guest code against it: a fast call runs inline with the
    /// region's state in machine registers, so nothing is published before it
    /// and nothing is reloaded after. Doing so is undefined behaviour. Register
    /// a [bridge](crate::core::bridge) for a callee that wants the context, or
    /// a [slow import](Self::register_slow_import) for one the host services.
    pub fn register_fast_import(&mut self, va: u64, symbol: DynamicSymbol, abi: CallAbi) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Fast(FastImport { symbol, abi }));
    }

    /// Register an import address with no fast-call metadata. Every call to this
    /// address produces an `ImportedFunctionCall` VM exit.
    pub fn register_slow_import(&mut self, va: u64) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Slow);
    }

    /// Register an import reached through a runtime-supplied bridge.
    ///
    /// Strictly more general than [`register_fast_import`](Self::register_fast_import):
    /// it drops the requirement that the callee's signature be expressible in
    /// [`FunctionArgumentType`]/[`CallAbi`], and is cheaper than the slow exit,
    /// since a bridge returning 0 continues inline with no VM exit at all. A
    /// fast import is still better where the signature *is* describable.
    ///
    /// See [`crate::core::bridge`] for the calling contract.
    ///
    /// Both bridge targets embed a host address in the emitted code, so unlike a
    /// `RuntimeAssisted` fast import a bridge produces no named relocation - AOT
    /// output containing one is only valid for a process that maps the bridge
    /// (or its slot) at the same address.
    pub fn register_bridge_import(&mut self, va: u64, bridge: BridgeImport) {
        Arc::make_mut(&mut self.imports).insert(va, ImportKind::Bridge(bridge));
    }

    /// Install the live import view queried at lift time. See
    /// [`ImportRegistry`].
    pub fn set_import_registry(&mut self, registry: Arc<dyn ImportRegistry>) {
        self.import_registry = Some(registry);
    }

    /// Install the bridge used for runtime-resolved indirect calls. See
    /// [`indirect_call_bridge`](Self::indirect_call_bridge).
    pub fn set_indirect_call_bridge(&mut self, bridge: BridgeImport) {
        self.indirect_call_bridge = Some(bridge);
    }

    /// The extent to lift so that `pc` can execute, cut on page boundaries.
    ///
    /// The alternative to walking control flow to find where a region should
    /// end. Walking predicts which bytes will be executed; this declines to
    /// predict and answers with everything from `pc` to the end of its page.
    /// Two properties make that usable:
    ///
    /// - **Monotone in what the guest has written.** A page boundary makes no
    ///   claim about the bytes, which a prediction ahead of execution does -
    ///   and cannot keep for generated code.
    /// - **It terminates.** The result always spans `pc` plus at least one
    ///   maximal instruction, running into the *next* page when `pc` sits in
    ///   the last few bytes of one, so entering at `pc` always makes progress.
    ///
    /// `available` bounds the answer to memory the host can read; the result is
    /// clamped into it, and is empty when `pc` is not inside it.
    ///
    /// Pair with [`region_tail_exit`](Self::region_tail_exit), which makes the
    /// instruction straddling the far end a resumable exit instead of a fault.
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
        // Round *up* from the last byte the entry instruction could occupy, so a `pc`
        // near a page end takes the following page rather than too short an extent.
        let last = pc.saturating_add(A::MAX_INSTRUCTION_LEN - 1);
        let end = (last / page).saturating_add(1).saturating_mul(page);
        pc..end.min(available.end)
    }
}

/// [`LifterConfig::block_ranges`] made fit to index a slice with: clamped to
/// `[base, end)`, sorted, and merged where they touch or overlap. An empty
/// input means the host named no holes, i.e. the whole slice.
///
/// Merging *touching* ranges matters as much as overlapping ones: two blocks
/// that abut are one run of code, and leaving them as two would cut an
/// instruction that legitimately spans the boundary.
fn normalize_ranges(ranges: &[Range<u64>], base: u64, end: u64) -> Vec<Range<u64>> {
    if ranges.is_empty() {
        // One range covering everything, not an iterator of addresses - which
        // is what clippy's `single_range_in_vec_init` reads this as.
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
    /// The `ip` value passed to `lift()` - the VA of the first instruction.
    pub lift_base: u64,
    /// One past the last byte of the lifted region (`lift_base + bytes.len()`).
    pub lift_end: u64,
    /// The bytes of `[lift_base, lift_end)` this lift covers, ascending and
    /// disjoint - [`LifterConfig::block_ranges`] normalised, or the single range
    /// `lift_base..lift_end` when the host named none.
    ///
    /// Every "is this address in the region" question goes through
    /// [`in_region`](Self::in_region), because with holes the two bounds are no
    /// longer the answer.
    covered: Vec<Range<u64>>,
    /// `next_ip` of the instruction currently being lifted - the value the PC
    /// would hold after that instruction retires.
    pub current_ip: u64,
    pub visited_blocks: HashSet<Block>,
    /// Whether the current block has been terminated (jmp, ret, etc.).
    pub block_terminated: bool,
    /// Per-address condition-flag liveness, when
    /// [`LifterConfig::eliminate_dead_flags`] asked for it. `None` means the
    /// optimisation is off, which [`Self::flag_live`] reads as "all live".
    flag_liveness: Option<FlagLiveness>,
    /// The flags live after the instruction currently being lifted.
    ///
    /// Latched just before each handler runs, because `current_ip` has by then
    /// been advanced to the *next* instruction.
    live_flags: FlagSet,
    pub syscall_handler_sig: Option<SigRef>,
    /// Fast-call data for registered non-variadic imports, keyed by import VA.
    /// The execution mode decides whether the target is absolute (`Jit`) or
    /// relocatable (`RuntimeAssisted`). Filled in lazily and cached here.
    pub fast_calls: HashMap<u64, FastCall>,
    /// One [`FastCall`] per symbol *name*, since the same symbol may be
    /// registered at several VAs (a PLT stub and its GOT slot).
    ///
    /// In `RuntimeAssisted` mode each distinct symbol is one
    /// [`Self::aot_imports`] entry, so emitting the name twice would collide
    /// when the object backend deduplicates and break `FuncId(i) == i`.
    fast_call_by_name: HashMap<&'static str, FastCall>,
    /// Cached signature of the bridge ABI
    /// (`extern "C" fn(*mut RegisterContext, u64) -> u32`), created on first
    /// use.
    bridge_sig: Option<SigRef>,
    /// Ordered external functions for object/JIT backends. Index `i` corresponds
    /// to `ExternalName::user(0, i)`. Entry = `(symbol_name, fn_ptr, signature)`.
    pub aot_imports: Vec<(&'static str, *const u8, Signature)>,
    /// Records what kind of transfer targets each unvisited (out-of-region)
    /// block, so `finish()` can emit the correct `TransferKind`.
    pub block_transfer_kinds: HashMap<Block, TransferKind>,
    /// Analysis-mode only: callee VA for each analysis `call` emitted, indexed by
    /// its external-function index (`ExternalName::user(ANALYSIS_CALL_NAMESPACE, i)`).
    /// Indirect calls record [`u64::MAX`].
    pub analysis_call_targets: Vec<u64>,
    /// Cached empty (no-arg, no-result) signature used by analysis-mode `call`s.
    analysis_call_sig: Option<SigRef>,
    /// Outlined helper functions, in the order they were first called. Their
    /// namespace-0 indices are provisional until [`Self::finish`] renumbers
    /// them to sit immediately after the AOT imports.
    helpers: Vec<HelperEntry>,
    /// One `FuncRef` per helper symbol, so repeated sites share one call
    /// target (and one entry in [`Self::helpers`]).
    helper_refs: HashMap<&'static str, FuncRef>,
    /// Encoding of guest addresses into the `SourceLoc`s tagged onto emitted IR.
    /// Set from `ip` at the top of [`Self::lift`]; the default
    /// (`lift_base == 0`) is only ever observed before then. See
    /// [`Self::set_source_loc`] and [`crate::core::srcloc`].
    pub source_map: SourceLocMap,
    /// Successors recovered from jump tables during phase 1, keyed by the
    /// address of the indirect branch they dispatch. Empty unless
    /// [`LifterConfig::analyze_jump_tables`] is set. Recorded for inspection -
    /// leader discovery has already consumed it by the time the lift returns.
    pub jump_tables: HashMap<u64, JumpTableInfo>,
    /// The guest addresses the entry dispatch was given an arm for, ascending.
    /// Handed out as [`LiftedCode::resume_points`].
    pub emitted_resume_points: Vec<u64>,
    /// In-region addresses a *resumable VM exit* continues at, collected while
    /// lifting rather than declared up front. See [`Self::resumes_at`].
    pub exit_resumes: HashSet<u64>,
    /// The one block-link probe this function has, created on first use and
    /// filled in [`Self::finish`].
    ///
    /// One per function rather than one per exit site. Its block parameter is
    /// the guest PC to look up, which is *not* always `ctx.rip` as the exit left
    /// it - a bridge may have moved it - so each site names the value it means.
    link_block: Option<Block>,
    /// The pinned register file this lift runs under, resolved once by
    /// [`pinned_abi_for`], with each slot's variable and type looked up.
    ///
    /// Empty for a lift that keeps the whole file in the context, which makes
    /// [`Self::emit_host_return`] a bare `return_` and [`Self::pinned_call_args`]
    /// empty - so the unpinned shape is the degenerate case of the pinned one
    /// rather than a branch.
    pinned_slots: Vec<(PinnedReg, Variable, types::Type)>,
    /// Signature of a pinned region, imported once so the link probe's
    /// `return_call_indirect` and the tail-call sites share it. `None` until
    /// first use, and forever when nothing is pinned.
    pinned_sig: Option<SigRef>,
    /// This region's own signature, imported once so a nested host call
    /// ([`crate::core::hostframe`]) can name what it is calling.
    region_sig: Option<SigRef>,

    /// Memflags for each `RegisterContext` field, keyed by its byte offset -
    /// one alias region apiece, interned once in [`Self::with_config`] from
    /// [`GuestArch::context_fields`]. Read through [`Self::ctx_store`] /
    /// [`Self::ctx_load`], never directly.
    ctx_flags: HashMap<i32, MemFlagsData>,
    /// Memflags for the x87 register stack, which is addressed `TOP`-relative
    /// and so gets one region for the whole array.
    x87_flags: MemFlagsData,
    /// Memflags for a guest memory access. A **field** rather than a method so
    /// `self.builder.ins().load(ty, self.guest_flags, ptr, 0)` borrow-checks;
    /// a `&self` method in the same expression would not.
    pub guest_flags: MemFlagsData,
    /// Memflags for host memory reached through this region that is not the
    /// context - today only a rebindable bridge slot.
    pub host_slot_flags: MemFlagsData,
    /// Memflags for a row of the host's block-link table.
    link_table_flags: MemFlagsData,
    /// Memflags for a cell of this region's own link-cell array. A separate
    /// region from the table: same row type, never the same memory.
    link_cell_flags: MemFlagsData,
    /// Memflags for a frame of the host's call-return ring.
    callret_flags: MemFlagsData,
    /// This region's link-cell array, in cell order. Reported as
    /// [`LiftedCode::link_sites`].
    link_sites: Vec<crate::core::link::LinkSite>,
    /// Cell index by target address, so two branches to one address share a
    /// cell and the second links the moment the first does.
    link_cell_by_target: HashMap<u64, usize>,
    /// The `GlobalValue` naming the cell array's base, created on first use.
    ///
    /// One per function, so Cranelift materializes the symbol's address once
    /// and every site shares it - which is what makes the second cell probe in
    /// a region cost a load and a branch and nothing more.
    link_cell_base: Option<cranelift_codegen::ir::GlobalValue>,

    /// Every guest register something defined during this lift, in the
    /// frontend's own numbering. The write set of [`crate::core::regsync`];
    /// **never cleared**, because the question it answers is "anywhere in the
    /// region", not "before this point".
    written_slots: HashSet<regsync::RegSlot>,
    /// Set by [`Self::keep_all_sync`] when a path has reloaded the whole
    /// register file out of the context, after which no store may be pruned.
    sync_prune_disabled: bool,
    /// Full-sync stores, with the register each writes back, in emission order.
    /// Candidates for the prune in [`Self::finish`].
    ctx_stores: Vec<(Inst, regsync::RegSlot)>,
    /// Entry-state loads. Candidates for the prune, decided by a use scan of the
    /// finished IR rather than by a slot, so no read set has to be maintained.
    ctx_loads: Vec<Inst>,
}

/// A helper under construction: its body plus the `UserExternalName` slot whose
/// index [`LiftEngine::finish`] rewrites once the import count is final.
struct HelperEntry {
    name: &'static str,
    func: Function,
    name_ref: UserExternalNameRef,
}

/// Namespace-0 index helpers are declared with before [`LiftEngine::finish`]
/// renumbers them. Imports take indices densely from 0 as they are found, and
/// a helper's final index depends on how many the rest of the lift turns up,
/// so helpers park far above any plausible import count where they cannot
/// collide (which would make `declare_imported_user_function` de-duplicate
/// two distinct callees onto one name).
const PROVISIONAL_HELPER_INDEX: u32 = 0x8000_0000;

/// Namespace used for the external functions referenced by analysis-mode
/// `call`s. Distinct from namespace `0` (AOT imports) so the
/// [`LiftEngine::analysis_call_targets`] index space is unambiguous.
/// One recovered jump table, in the guest-agnostic terms the engine needs.
///
/// The frontend's own type ([`crate::arch::x86_64::jumptable::JumpTable`]) knows
/// far more; this is the part that crosses [`GuestArch::jump_table_targets`].
/// The extent travels with the targets rather than being recomputed because the
/// pass is the only thing that knows how many entries it read - `targets` is
/// de-duplicated and says nothing about the table's size.
#[derive(Debug, Clone)]
pub struct JumpTableInfo {
    /// The recovered successors, in table order, de-duplicated. Contributed as
    /// block leaders.
    pub targets: Vec<u64>,
    /// The guest bytes the table occupies, which the translation now depends on.
    pub extent: Range<u64>,
}

pub const ANALYSIS_CALL_NAMESPACE: u32 = 1;

impl<'a, A: GuestArch> LiftEngine<'a, A> {
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        Self::with_config(builder, LifterConfig::default())
    }

    pub fn with_config(builder: FunctionBuilder<'a>, config: LifterConfig) -> Self {
        let mut builder = builder;
        let state = A::new_state(&mut builder, &config);
        // Resolved here so every later question is a lookup rather than a re-derivation.
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
        let syscall_handler_sig = match config.execution_mode {
            ExecutionMode::Jit => None,
            ExecutionMode::RuntimeAssisted => {
                let mut sig = Signature::new(config.syscall_handler_call_conv);
                sig.params.push(AbiParam::new(types::I64));
                Some(builder.import_signature(sig))
            }
        };

        // One alias region per context field, plus the three whole-object ones.
        // Interned up front rather than on demand: `MemFlagsData` is `Copy`, so
        // a field read borrows disjointly from `self.builder` and an access
        // needs no `let` line to satisfy the borrow checker.
        let mut ctx_flags: HashMap<i32, MemFlagsData> = HashMap::new();
        for (offset, name) in A::context_fields(&config) {
            let flags = alias::region_flags(builder.func, offset as u32, name);
            // NB: the insert is the point, so it must not live inside a
            // `debug_assert!` - that compiles out under `--release` and leaves
            // every context access without a region.
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
        let link_cell_flags =
            alias::region_flags(builder.func, alias::LINK_CELL_ID, "link cells");
        let callret_flags =
            alias::region_flags(builder.func, alias::CALLRET_ID, "call-return ring");
        let guest_flags = alias::guest_memory_flags(builder.func);

        // Built lazily by `fast_call_for` when a call site is actually lifted.
        // Building eagerly cost time proportional to the whole program's import
        // table, per region, and declared every import as an external whether or
        // not anything referenced it.
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
            flag_liveness: None,
            // Nothing has been analysed yet, and the IR the engine emits before
            // phase 2 - the entry block's state load - must not be elided.
            live_flags: FlagSet::ALL,
            syscall_handler_sig,
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
            written_slots: HashSet::new(),
            sync_prune_disabled: false,
            ctx_stores: Vec::new(),
            ctx_loads: Vec::new(),
        }
    }

    /// Store `val` into the `RegisterContext` field at `offset`, tagged with
    /// that field's alias region.
    pub fn ctx_store(&mut self, val: Value, ctx_ptr: Value, offset: i32) -> Inst {
        let flags = self.ctx_field_flags(offset);
        self.builder.ins().store(flags, val, ctx_ptr, offset)
    }

    /// Load the `RegisterContext` field at `offset`. The read half of
    /// [`Self::ctx_store`]; the same completeness rule applies.
    pub fn ctx_load(&mut self, ty: types::Type, ctx_ptr: Value, offset: i32) -> Value {
        let flags = self.ctx_field_flags(offset);
        self.builder.ins().load(ty, flags, ctx_ptr, offset)
    }

    /// Memflags for the x87 register stack, which is addressed `TOP`-relative
    /// and so shares one region across the whole `fpr` array.
    pub fn x87_flags(&self) -> MemFlagsData {
        self.x87_flags
    }

    /// Record that something defined `slot`'s SSA variable in this region.
    pub fn mark_written(&mut self, slot: regsync::RegSlot) {
        self.written_slots.insert(slot);
    }

    /// Give up on the lazy-sync prune ([`crate::core::regsync`]) for the rest of
    /// this region.
    pub fn keep_all_sync(&mut self) {
        self.sync_prune_disabled = true;
    }

    /// Whether anything in this region defined `slot`.
    pub fn slot_written(&self, slot: regsync::RegSlot) -> bool {
        self.written_slots.contains(&slot)
    }

    /// [`Self::ctx_store`] for a guest register in the *full* sync, recording
    /// the store so the prune can drop it if nothing ever wrote `slot`.
    ///
    /// Only the full sync may use this. A masked sync (`sync_regs_masked`,
    /// [`Self::publish_pinned`]) serves a host call happening *while the region
    /// runs*, so its stores are not idempotent in the sense the prune needs and
    /// must always be emitted.
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

    /// [`Self::ctx_load`] for the region's entry state, recording the load so
    /// the prune can drop it once nothing is left that reads it.
    pub fn ctx_load_reg(&mut self, ty: types::Type, ctx_ptr: Value, offset: i32) -> Value {
        let val = self.ctx_load(ty, ctx_ptr, offset);
        if self.lazy_register_sync() {
            let inst = self.builder.func.dfg.value_def(val).unwrap_inst();
            self.ctx_loads.push(inst);
        }
        val
    }

    /// Whether the lazy-sync prune is running for this lift.
    ///
    /// Off in analysis mode: that IR is for reading, it deliberately emits no
    /// write-back at all, and its entry loads are what keep the register file
    /// visible as pure SSA.
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

        // A store whose register nothing ever defined writes back the value the
        // entry load read out of that very field.
        for (inst, slot) in stores {
            if !written.contains(&slot) {
                self.builder.func.layout.remove_inst(inst);
            }
        }

        // With those gone, an entry load may have no consumer left, and a block
        // parameter carrying one may have no reader. The two unlock each other:
        // deleting a store can leave a parameter dead, deleting the parameter
        // drops the branch argument that was the load's last use, and deleting
        // the load drops *its* arguments. So this runs to a fixpoint rather than
        // once - a single pass removes the stores and almost none of the traffic
        // behind them.
        let mut loads = loads;
        loop {
            let dropped = self.drop_unused_ctx_loads(&mut loads);
            let unhooked = dce::remove_dead_block_params(&mut self.builder.func);
            if !dropped && !unhooked {
                break;
            }
        }
    }

    /// Remove every load in `loads` whose results nothing reads, and forget it.
    ///
    /// Only loads this engine recorded, which are the entry state load and the
    /// full sync's - both `notrap`, which is what makes deleting one safe
    /// whatever the address turns out to be. Asks the IR rather than a read set:
    /// a value no instruction names cannot be needed by one, whatever the
    /// frontend believes it did.
    fn drop_unused_ctx_loads(&mut self, loads: &mut Vec<Inst>) -> bool {
        let func = &mut *self.builder.func;
        let mut used: HashSet<Value> = HashSet::new();
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
                .all(|&v| !used.contains(&func.dfg.resolve_aliases(v)));
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

    /// Whether `reg` is carried in a machine register across region boundaries
    /// rather than in the `RegisterContext`.
    ///
    /// **For the full sync and the entry load only** - a frontend calls this to
    /// skip a register in [`GuestArch::load_state_from_context`] and
    /// [`GuestArch::sync_to_context`], because the entry thunk does both around
    /// the outside. A masked sync must *not* consult it: a bridge or import runs
    /// while the region does, and the host reads the context then.
    pub fn is_pinned(&self, reg: PinnedReg) -> bool {
        self.pinned_slots.iter().any(|&(r, _, _)| r == reg)
    }

    /// Whether this lift carries a pinned register file at all.
    pub fn pinned(&self) -> bool {
        !self.pinned_slots.is_empty()
    }

    /// The values a pinned region passes on: the context pointer, then every
    /// pinned slot in ABI order. Empty when nothing is pinned.
    ///
    /// One list for both the return and the linked tail call, which a signature
    /// whose results are its parameters is what makes possible.
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

    /// Return to whoever called this region - the host's entry thunk, or the host
    /// directly when nothing is pinned.
    ///
    /// **Every `return_` in a lifted region goes through here.** With no pinned
    /// file it is exactly `return_(&[])`; with one it hands back the registers
    /// the thunk will store. A hand-written `return_` under a pinned ABI is a
    /// verifier error at best and a truncated register file at worst.
    pub fn emit_host_return(&mut self) {
        let args = self.pinned_call_args();
        self.builder.ins().return_(&args);
    }

    /// Store every pinned register into the `RegisterContext`, **before host code
    /// runs inside this region**.
    ///
    /// The invariant: *a pinned register is in the context whenever host code is
    /// executing.* Pinning moves where a register lives between region entries;
    /// it must not move where a host looks, and a host servicing an inline call
    /// has only the context to read guest state from.
    ///
    /// A no-op when nothing is pinned; one store per slot otherwise.
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

    /// Reload every pinned register from the `RegisterContext`, **after host code
    /// that may have written one**.
    ///
    /// The other half of [`Self::publish_pinned`], needed only where the host
    /// writes guest registers through the context rather than by returning: a
    /// bridge's exiting path, where the region's own SSA values would otherwise
    /// be handed back by the entry thunk and overwrite them.
    ///
    /// A register the host did not touch reloads what was just published, so
    /// this is correct without knowing which ones it wrote.
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

    /// The imported signature of a pinned region, for the probe's indirect tail
    /// call. Created on first use; the types come from this lift's own slots,
    /// which is the same list every region under this ABI has.
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

    /// Attribute the IR emitted from here on to guest instruction `va`.
    ///
    /// A mode switch, not a per-instruction call: the builder stamps its one
    /// current `SourceLoc` onto everything it creates, so the emission loop sets
    /// it once before each [`GuestArch::lift_instruction`] and everything that
    /// handler emits inherits it.
    ///
    /// A no-op when [`LifterConfig::emit_source_locs`] is off, and a
    /// [`clear_source_loc`](Self::clear_source_loc) when `va` is too far from
    /// the region to encode.
    pub fn set_source_loc(&mut self, va: u64) {
        if !self.config.emit_source_locs {
            return;
        }
        let loc = self.source_map.encode(va);
        self.builder.set_srcloc(loc);
    }

    /// Stop attributing emitted IR to any guest instruction, for the region's own
    /// scaffolding (entry prologue, PC dispatch) where a stale location would be
    /// actively misleading.
    pub fn clear_source_loc(&mut self) {
        self.builder.set_srcloc(SourceLoc::default());
    }

    /// Declare (or reuse) an outlined helper function named `name`, returning a
    /// `FuncRef` the caller can `call`.
    ///
    /// The first call for a `name` builds the body: a fresh `Function` with
    /// `signature` is handed to `build`, which must fill it completely. Later
    /// calls reuse that definition and its one `FuncRef` - that reuse is the
    /// point, so `name` must identify the *body*, not the call site.
    ///
    /// The helper is locally defined, not imported, and comes back in
    /// [`LiftedCode::helpers`]. Its namespace-0 index is provisional until
    /// [`Self::finish`], since it lands after however many imports turn up.
    pub fn outlined_helper(
        &mut self,
        name: &'static str,
        signature: Signature,
        build: impl FnOnce(&mut FunctionBuilder),
    ) -> FuncRef {
        if let Some(&func_ref) = self.helper_refs.get(name) {
            return func_ref;
        }

        // Build the body into a standalone function. The index in the name is
        // cosmetic here and rewritten alongside the call's in `finish`.
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), signature.clone());
        let mut helper_ctx = FunctionBuilderContext::new();
        let mut helper_builder = FunctionBuilder::new(&mut func, &mut helper_ctx);
        build(&mut helper_builder);
        // `finalize` only consults the frontend config for stack maps, which helper
        // bodies never request; the pointer width is what every backend here uses.
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
            // Matches the fast-import path, whose relocations both the JIT and
            // the object backend are known to resolve.
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

    /// Whether outlined helpers are in use for this lift. Analysis IR always
    /// inlines so it stays a single self-contained function.
    pub fn outline_helpers(&self) -> bool {
        self.config.outline_helpers && !self.config.analysis_mode
    }

    /// Take the finished helper bodies out of the engine. Valid only after
    /// [`Self::finish`] has assigned their final indices; [`lift_to_code`] is
    /// the normal caller.
    /// The guest bytes recovered jump tables occupy, normalized the way
    /// [`Self::covered`] is: ascending, disjoint, adjacent ranges merged.
    ///
    /// Reported as [`LiftedCode::data_dependencies`]. Empty unless
    /// [`LifterConfig::analyze_jump_tables`] ran.
    pub fn jump_table_extents(&self) -> Vec<Range<u64>> {
        let mut extents: Vec<Range<u64>> =
            self.jump_tables.values().map(|t| t.extent.clone()).collect();
        extents.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(extents.len());
        for range in extents {
            match merged.last_mut() {
                // Two tables can overlap - a `switch` compiled as a table of
                // tables shares bytes - so this coalesces rather than appending.
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        merged
    }

    pub fn take_helpers(&mut self) -> Vec<HelperFunction> {
        std::mem::take(&mut self.helpers)
            .into_iter()
            .map(|h| HelperFunction {
                name: h.name,
                func: h.func,
            })
            .collect()
    }

    /// Give each outlined helper its final namespace-0 index, immediately after
    /// the AOT imports, and rewrite the call sites' `UserExternalName` to match.
    ///
    /// At the end of the lift because the import count is only final then:
    /// [`ImportRegistry`]-supplied imports append to `aot_imports` as their call
    /// sites are lifted, possibly after a helper was declared.
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

    /// The cached empty signature used by analysis-mode `call`/`call_indirect`,
    /// created on first use.
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

    /// How the address `va` is reached, or `None` if it is not an import.
    ///
    /// Checks [`LifterConfig::imports`] first, then the
    /// [`ImportRegistry`](crate::core::bridge::ImportRegistry), so an explicit
    /// registration always wins. Queried per call site *while lifting*, which
    /// is what lets a region lifted after a `dlopen` see the new stubs.
    pub fn lookup_import(&self, va: u64) -> Option<ImportKind> {
        if let Some(kind) = self.config.imports.get(&va) {
            return Some(kind.clone());
        }
        self.config
            .import_registry
            .as_ref()
            .and_then(|r| r.lookup(va))
    }

    /// Whether `va` is a registered import under either source.
    pub fn is_import(&self, va: u64) -> bool {
        self.config.imports.contains_key(&va)
            || self
                .config
                .import_registry
                .as_ref()
                .is_some_and(|r| r.lookup(va).is_some())
    }

    /// The [`FastCall`] for import `va`, building and caching it on first use if
    /// the import came from the registry rather than the config map. `None`
    /// means the import has no fast-call metadata, is variadic, or has arguments
    /// its ABI cannot place - all of which fall back to the slow exit.
    pub fn fast_call_for(&mut self, va: u64) -> Option<FastCall> {
        if let Some(fc) = self.fast_calls.get(&va) {
            return Some(fc.clone());
        }
        // Not prebuilt: either not a fast import, or one the registry only just reported.
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

    /// The signature every bridge is called through:
    /// `extern "C" fn(*mut RegisterContext, u64) -> u32`. Created on first use.
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

    /// Ordered external functions needed for AOT/JIT backends in
    /// `RuntimeAssisted` mode. Index `i` corresponds to `ExternalName::user(0, i)`.
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

    /// Like `get_block`, but records what transfer kind reaches the block (first
    /// caller wins).
    pub fn get_block_with_kind(&mut self, offset: u64, kind: TransferKind) -> Block {
        let block = self.get_block(offset);
        self.block_transfer_kinds.entry(block).or_insert(kind);
        block
    }

    /// Position the builder to keep emitting into `next_block`, the fallthrough
    /// side of a branch just emitted.
    ///
    /// Straight-line code after a leader is tail-duplicated into every block that
    /// reaches it, so when two copies both end in a conditional branch the second
    /// finds its fallthrough already filled. Nothing is left to emit then - the
    /// `brif` edge already points at it - so terminate rather than switch into
    /// it, which would trip Cranelift's "already filled" assertion.
    pub fn continue_in_fallthrough(&mut self, next_block: Block) {
        if self.visited_blocks.contains(&next_block) {
            self.block_terminated = true;
            return;
        }
        self.builder.switch_to_block(next_block);
        self.visited_blocks.insert(next_block);
        self.block_terminated = false;
    }

    /// Emit a store of `code` into the `exit_code` control field.
    pub fn write_exit_code(&mut self, code: u32) {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let val = self.builder.ins().iconst(types::I32, code as i64);
        self.ctx_store(val, ctx_ptr, A::exit_code_offset());
    }

    /// Declare that the VM exit just emitted **resumes at `va`**.
    ///
    /// Called by every exit the host services and then re-enters - `CPUID`,
    /// `RDTSC`, `XGETBV`, `SYSCALL`, `INT`, `INT3`, port I/O, an emulated
    /// unimplemented instruction. It does two things, and both are needed:
    ///
    /// - **Makes `va` a block leader**, so the code after the exit is lifted at
    ///   all; an exit terminates its block, so the phase-2 walk otherwise stops.
    /// - **Makes `va` a resume point**, so the entry dispatch has an arm for it.
    ///
    /// Out-of-region addresses are ignored - the host translates a region there
    /// in the ordinary way.
    pub fn resumes_at(&mut self, va: u64) {
        if self.in_region(va) {
            self.exit_resumes.insert(va);
            self.get_block(va);
        }
    }

    /// Emit a store of `reason` into the `exit_reason` control field, with no PC
    /// write and no register sync - for the one exit that happens before any
    /// guest instruction has been lifted, where the context already holds
    /// exactly what the host put there.
    pub fn write_exit_reason(&mut self, reason: ExitReason) {
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let val = self.builder.ins().iconst(types::I32, reason as i64);
        self.ctx_store(val, ctx_ptr, A::exit_reason_offset());
    }

    /// Whether the instruction at `addr` runs off the end of the lifted bytes.
    ///
    /// Two ways, and the second is easy to miss:
    ///
    /// - The decode says so outright - `next_ip` past the end. That happens when
    ///   the bytes past the cut were readable, i.e. the region was clipped out
    ///   of a longer buffer.
    /// - The decode came back **invalid** from within one maximal instruction of
    ///   the end. A decoder given too few bytes cannot say "truncated"; it says
    ///   "invalid", identically to a byte that really is not an instruction. In
    ///   that window the resumable answer is the safe one - a re-lift at `addr`
    ///   sees the whole encoding and reports a genuine invalid if that is what
    ///   it is.
    ///
    /// Never true at the first address of the range `addr` falls in, which is
    /// what makes the exit terminate rather than repeat. "The end" is the end of
    /// that range rather than `lift_end`, so a block abutting a hole is cut as
    /// one abutting the end of the slice.
    fn runs_off_region(&self, addr: u64, inst: &A::Instr) -> bool {
        let Some(range) = self.covering(addr) else {
            return false;
        };
        self.config.region_tail_exit
            && addr != range.start
            && (A::instr_next_ip(inst) > range.end
                || (A::is_invalid(inst) && range.end - addr < A::MAX_INSTRUCTION_LEN))
    }

    /// The covered range holding `va`, if any. See
    /// [`LifterConfig::block_ranges`].
    fn covering(&self, va: u64) -> Option<&Range<u64>> {
        // Ascending and disjoint, so the last range starting at or before `va`
        // is the only one that can hold it.
        let i = self
            .covered
            .partition_point(|r| r.start <= va)
            .checked_sub(1)?;
        self.covered.get(i).filter(|r| va < r.end)
    }

    /// Whether `va` is a byte this lift covers.
    ///
    /// The replacement for `va >= lift_base && va < lift_end`, which stopped
    /// being the same question once a region could have holes in it.
    pub fn in_region(&self, va: u64) -> bool {
        self.covering(va).is_some()
    }

    /// Whether the instruction being lifted has to compute flag `bit`.
    ///
    /// `bit` is the frontend's own flag index, the same numbering as
    /// [`GuestArch::flag_effects`]. A handler guards each flag with it:
    ///
    /// ```ignore
    /// if self.flag_live(X86Flag::Parity as u8) {
    ///     // emit the popcount
    /// }
    /// ```
    ///
    /// **Always true unless [`LifterConfig::eliminate_dead_flags`] is on.**
    /// Skipping a write leaves the flag's SSA variable holding its previous
    /// value rather than an undefined one, so a downstream sync publishes
    /// something well-formed even if the analysis were wrong.
    pub fn flag_live(&self, bit: u8) -> bool {
        self.live_flags.contains(bit)
    }

    /// The value the `image_base` variable is defined to at region entry.
    ///
    /// Under [`AddressMode::ImageBaseRelative`] the relocation base from the
    /// context; under [`AddressMode::Direct`] the constant `lift_base`, which
    /// makes `image_base + (va - lift_base) == va` hold - so anything still
    /// written in terms of `image_base` folds to the right constant instead of
    /// silently producing a relocated address.
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

    /// Materialize the run-time address of the lift-time virtual address `va`.
    ///
    /// The one place a guest VA becomes a Cranelift value. See [`AddressMode`];
    /// the inverse is [`Self::to_lift_va`].
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

    /// The run-time address of the instruction *after* the one being lifted -
    /// i.e. of [`Self::current_ip`], the resume point for a re-enterable exit.
    pub fn runtime_current_ip(&mut self) -> Value {
        self.runtime_addr(self.current_ip)
    }

    /// Map a run-time address back into lift-time VA space, so it can be compared
    /// against (or switched on) the addresses the decoder saw.
    ///
    /// The inverse of [`Self::runtime_addr`]: `addr - image_base + lift_base`
    /// under [`AddressMode::ImageBaseRelative`], the identity under
    /// [`AddressMode::Direct`].
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

    /// Dispatch `target` - a lift-time VA in a register - to the in-region block
    /// that begins there, falling through to `external` when none does.
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
        // Sorted for the same reason as `finish`'s stub loop: a `HashMap`'s
        // order is per-instance, and `Switch::emit` builds its dispatch from the
        // order entries were added.
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

    /// Whether this lift emits link probes at all, for a semantics handler that
    /// wants to know before building the branch that reaches one.
    pub fn block_link_active(&self) -> bool {
        self.block_link().is_some()
    }

    /// The geometry of the host's link table, if this lift is linking at all.
    ///
    /// `None` disables every path here that would emit a probe, so the two
    /// conditions - the host asked, and this guest's context has somewhere to
    /// put the answer - are asked in exactly one place. Analysis mode is
    /// excluded because its IR does not write to the context at all.
    fn block_link(&self) -> Option<(BlockLinkConfig, BlockLinkOffsets)> {
        if self.config.analysis_mode {
            return None;
        }
        let config = self.config.block_link?;
        Some((config, A::block_link_offsets()?))
    }

    /// Whether this lift gives lift-time-known exits a cell of their own.
    ///
    /// Both conditions in one place, mirroring [`Self::block_link`]: the host
    /// asked for cells, *and* this lift is linking at all. A cell is a
    /// [`BlockLinkEntry`] and there is nothing for one to mean without the rest
    /// of block linking.
    fn link_cells(&self) -> bool {
        self.config.link_cells && self.block_link().is_some()
    }

    /// The cell index for `target`, allocating one if this is the first branch
    /// to that address in this region.
    ///
    /// De-duplicated by target, which is what makes a `switch` whose arms all
    /// leave the region cost one cell per distinct destination rather than one
    /// per edge - and means the second branch to an address links the instant
    /// the first one does.
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

    /// The base address of this region's cell array, as a value.
    ///
    /// A relocated symbol rather than a baked constant: the artifact must
    /// survive being cached to disk and mapped at a different address, which is
    /// the same property [`AddressMode::ImageBaseRelative`] exists to preserve.
    /// Namespace 1 is `cranelift-module`'s `DataId` space, so the backend maps
    /// it with no convention of its own - note it is numerically the same as
    /// [`ANALYSIS_CALL_NAMESPACE`], which is harmless because analysis-mode IR
    /// is never compiled and never emits a cell.
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
                let gv = self
                    .builder
                    .func
                    .create_global_value(cranelift_codegen::ir::GlobalValueData::Symbol {
                        name: ExternalName::user(name),
                        offset: 0.into(),
                        // **Far, though the array is defined in the same object as
                        // the code.** `colocated` does not mean "same object", it
                        // means "within `adrp` range" - and a JIT does not honour
                        // that. `cranelift-jit` hands executable and writable
                        // allocations out of two separate `mmap`s, and over a long
                        // run those drift: measured at 6 GiB apart in a host that
                        // had been translating for half a minute, which overflows
                        // `Aarch64AdrPrelPgHi21` (`compiled_blob.rs` panics rather
                        // than veneering, the way it does for a far `bl`). `Far`
                        // costs a `ldr`/`b` around an 8-byte literal instead of
                        // `adrp`/`add`, and an `Abs8` relocation cannot be out of
                        // range.
                        colocated: false,
                        tls: false,
                    });
                self.link_cell_base = Some(gv);
                gv
            }
        };
        self.builder.ins().symbol_value(types::I64, gv)
    }

    /// The host-frame configuration and offsets, if this lift nests guest calls
    /// as host calls.
    ///
    /// All three conditions in one place, mirroring [`Self::block_link`] and
    /// [`Self::call_return`]: the host asked, cells are on (the callee's address
    /// comes from one), and this guest's context has somewhere to keep the depth.
    fn host_call_frames(&self) -> Option<(HostFrameConfig, HostFrameOffsets)> {
        if !self.link_cells() {
            return None;
        }
        let config = self.config.host_call_frames?;
        Some((config, A::host_frame_offsets()?))
    }

    /// The arguments to enter a region with: the context pointer, then one per
    /// pinned slot.
    ///
    /// **Not [`Self::pinned_call_args`]**, which is empty when nothing is pinned
    /// because its callers are `return_` sites and an unpinned region returns
    /// nothing. A *call* still has to pass the context pointer, so the two differ
    /// in exactly the unpinned case - which is the one where getting it wrong
    /// gives a call with no arguments to a signature that takes one.
    fn region_call_args(&mut self) -> Vec<Value> {
        match self.pinned_slots.is_empty() {
            true => vec![self.builder.use_var(A::context_ptr_var(&self.state))],
            false => self.pinned_call_args(),
        }
    }

    /// This region's own signature, imported so a nested call can name it.
    ///
    /// The region calls something with *its own* shape - every lifted region has
    /// the same one, pinned or not - so this is `func.signature` re-imported
    /// rather than a signature built a second time and kept in agreement by hand.
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
    /// The other half of [`Self::emit_host_frame_call`], and the half without
    /// which that one is worse than useless: a caller that made a host call is
    /// waiting on the stack, and a callee that links instead of returning leaves
    /// that frame stranded. In a loop the host stack then grows by a frame per
    /// iteration until the depth ceiling stops it, after which every call takes
    /// the fallback - paying for the mechanism and using none of it.
    ///
    /// So under host frames a `RET` leaving the region always returns, and the
    /// question of whether anyone wanted it is answered by whoever catches it.
    /// A region with no frame waiting - entered by the host, or at the bottom of
    /// a chain of jump links from one that was called - returns to the entry
    /// thunk, which costs exactly the host dispatch a link *miss* would have.
    pub fn returns_through_host_frames(&self) -> bool {
        self.host_call_frames().is_some()
    }

    /// Nest an out-of-region direct `CALL` as a real host call, falling back to
    /// `fallback` when it cannot be.
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
    /// The callee needs no cooperation and gets none: its `RET` does exactly what
    /// an unlinked `RET` always did - sync, and return with the guest's target in
    /// the PC - and the caller, where that target is a lift-time *constant*, is
    /// what decides whether it was the expected one. This is the whole reason
    /// there is no shadow stack here; see [`crate::core::hostframe`].
    ///
    /// A mismatch returns again rather than handling anything, so a `longjmp` or a
    /// computed return unwinds one host frame per level until a caller recognises
    /// the address or the entry thunk hands it to the host. Correct by the same
    /// argument at every level, and paid in proportion to how far the guest jumped.
    ///
    /// Returns `true` when the nested call was emitted. `false` means nothing was
    /// emitted and the caller must produce its ordinary exit; the builder is left
    /// where it was found.
    pub fn emit_host_frame_call(
        &mut self,
        target: u64,
        callee: Value,
        resume: u64,
    ) -> crate::Result<bool> {
        let Some((config, offsets)) = self.host_call_frames() else {
            return Ok(false);
        };
        // The call has to come back *somewhere this region can continue*. When the
        // resume point is past the end of the region there is no block to return
        // into - `finish` turns it into an exit stub, which would sync the
        // pre-call register file over everything the callee wrote. Nesting is an
        // optimisation, so it declines rather than working around it.
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
        // Not installed yet: the target has not been translated, so there is
        // nothing to call. Exactly the fallback every other cell has.
        self.builder.ins().brif(code, nest, &[], slow, &[]);

        self.builder.switch_to_block(nest);
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let depth = self.ctx_load(types::I32, ctx_ptr, offsets.depth);
        let room = self.builder.ins().icmp_imm_u(
            IntCC::UnsignedLessThan,
            depth,
            config.max_depth as i64,
        );
        // The host stack is finite and every live guest call is a frame of it.
        // Past the ceiling a call takes the ordinary exit, which costs a host
        // dispatch and nothing else.
        self.builder.ins().brif(room, taken, &[], slow, &[]);

        self.builder.switch_to_block(taken);
        // Captured before the callee's is written over it. Every address this
        // region forms is relative to *its* image base under
        // `AddressMode::ImageBaseRelative`, so leaving the callee's behind would
        // make the resume comparison below - and everything after it - arithmetic
        // on the wrong base.
        let my_image_base = self.builder.use_var(A::image_base_var(&self.state));
        let deeper = self.builder.ins().iadd_imm_s(depth, 1);
        self.ctx_store(deeper, ctx_ptr, offsets.depth);
        // The callee enters through its own PC dispatch, so it has to be told
        // where to start - and given the `image_base` that translation wants,
        // which under `ImageBaseRelative` is not this region's.
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

        // Restored rather than decremented: `depth` is this frame's own value, and
        // reading it back would cost a load to learn something already in a
        // register.
        self.ctx_store(depth, ctx_ptr, offsets.depth);
        // Put this region's image base back *before* the reload below, so the
        // reload takes it and every address formed after the call is this
        // region's again.
        self.ctx_store(my_image_base, ctx_ptr, A::image_base_offset());
        // The pinned half comes back in the results, in the order it went out;
        // result 0 is the context pointer, which has not moved.
        for (i, &(_, var, _)) in self.pinned_slots.clone().iter().enumerate() {
            self.builder.def_var(var, results[i + 1]);
        }
        // Everything else the callee may have touched is in the context, so it is
        // read back - **here**, on the path out of the call and nowhere else.
        //
        // Putting it at the top of the resume block instead is tempting, because
        // there it would be the single definition reaching everything below and
        // the ~55 loads a nested call costs would be plainly dead once the prune
        // deletes the stores that consume them. It is also **wrong**: the resume
        // point is an ordinary in-region address, so this region's own `RET`
        // dispatch - and any branch to it - can reach that block without having
        // synced anything, and a reload there would overwrite live register
        // values with whatever the context last happened to hold.
        A::load_state_from_context(self)?;
        // **No `keep_all_sync` here, deliberately**, and the argument is worth
        // spelling out because the syscall handler's reload - which looks like
        // this one - does call it.
        //
        // The prune deletes the sync store of a register nothing in the region
        // wrote, on the grounds that it writes back what the context already
        // holds. A nested call does not break that: a register this region never
        // writes has, at every point after this reload, exactly the value the
        // reload took *out* of the context, so its store is still idempotent. A
        // register the region writes anywhere - before the call or after it - is
        // in the write set and keeps its stores as usual.
        //
        // That matters rather than being a nicety: disabling the prune here
        // measured at +237% against +89% on a recursive fixture, because a region
        // containing a call is exactly the kind that is hot.
        //
        // `host_frame_tests::a_callee_s_writes_survive_a_caller_that_touches_nothing`
        // is what holds this, since being wrong about it is a stale register and
        // not a crash.

        let reason = self.ctx_load(types::I32, ctx_ptr, A::exit_reason_offset());
        let is_transfer = self.builder.ins().icmp_imm_u(
            IntCC::Equal,
            reason,
            ExitReason::UnknownJump as i64,
        );
        let code_field = self.ctx_load(types::I32, ctx_ptr, A::exit_code_offset());
        let is_return = self.builder.ins().icmp_imm_u(
            IntCC::Equal,
            code_field,
            TransferKind::Return as i64,
        );
        let pc = self.ctx_load(types::I64, ctx_ptr, A::pc_offset());
        let want = self.runtime_addr(resume);
        let here = self.builder.ins().icmp(IntCC::Equal, pc, want);
        let ok = self.builder.ins().band(is_transfer, is_return);
        let ok = self.builder.ins().band(ok, here);
        self.builder.ins().brif(ok, resume_block, &[], unwind, &[]);

        self.builder.switch_to_block(unwind);
        // Not our return. Hand it straight back up: the frame below may be the
        // one that wants it, and if none is, the entry thunk gives it to the host
        // with the reason and PC the callee already wrote. Nothing is cleared,
        // because nothing here has been serviced.
        self.builder.ins().return_(&results);

        self.builder.switch_to_block(slow);
        self.builder.seal_block(nest);
        self.builder.seal_block(taken);
        self.builder.seal_block(unwind);
        self.builder.seal_block(slow);
        Ok(true)
    }

    /// Terminate a linkable exit whose destination is known at **lift time**,
    /// trying its own cell before falling back to [`Self::emit_exit`].
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
    /// Four instructions on the fast path against the probe's nine, and - which
    /// matters more than the count - **no collisions**: the table aliases two
    /// targets sixteen bytes apart into one row and turns a link into a
    /// permanent miss, and a cell cannot.
    ///
    /// The fallback is not a formality. A cell is empty until the host installs
    /// it, which it can only do once the target has been translated, so the
    /// probe is what serves every edge whose destination has not been compiled
    /// yet - and the shared table is what serves it after an
    /// [invalidation](crate::runtime::CodeCache::invalidate_range) zeroes the
    /// cell again.
    ///
    /// `target` is the destination as a lift-time VA, which is what the host
    /// installs against; `pc` is the same address as a run-time value, exactly
    /// as [`Self::emit_exit`] wants it. Exactly `emit_exit` when
    /// [`Self::link_cells`] is off.
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
        // Reached only when the cell is empty. `pc` was defined in the block
        // that branched here, which dominates it, so it is still in scope.
        self.emit_exit(Some(pc));
    }

    /// Emit one cell probe, terminating the current block and leaving the
    /// builder in the `slow` arm. See [`Self::emit_static_exit`].
    fn emit_cell_probe(&mut self, target: u64, pc: Value, kind: crate::core::link::LinkSiteKind) {
        let index = self.link_cell_for(target, kind);
        let cell_offset = index as i64 * crate::core::link::ENTRY_SIZE as i64;

        let base = self.link_cell_base();
        let cell = self.builder.ins().iadd_imm_s(base, cell_offset);
        self.emit_cell_dispatch(cell, pc);
    }

    /// Take the link a [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) at
    /// `cell` names, or fall through.
    ///
    /// Terminates the current block and leaves the builder in the `slow` arm.
    /// Shared by the static-edge probe, where `cell` is a constant offset into
    /// this region's own array, and by the call-return pop, where it is a
    /// pointer read out of a ring frame and may belong to another region
    /// entirely - which is fine, because a cell is just an address.
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
        // Zero is "not installed". It is not a code address for the same reason
        // zero is not a guest PC, so a zeroed array is a valid initial state and
        // clearing a cell is how the host retracts the link.
        self.builder.ins().brif(code, hit, &[], slow, &[]);

        self.builder.switch_to_block(hit);
        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        // Committed for the same reason the table probe commits it: one exit
        // shape has a `rip` that is not where it continues.
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
            // The cell's *address*, exactly as the table probe hands back a
            // row's - the host dereferences either without caring which it got.
            self.ctx_store(cell, ctx_ptr, offsets.hit);
            self.emit_host_return();
        }

        self.builder.switch_to_block(slow);
        self.builder.seal_block(hit);
        self.builder.seal_block(slow);
    }

    /// The call-return ring's geometry and context offsets, if this lift
    /// predicts returns at all.
    ///
    /// All three conditions in one place, mirroring [`Self::block_link`]: the
    /// host asked for a ring, cells are on (a hit hands back a cell, so there is
    /// nothing for one to name without them), and this guest's context has
    /// somewhere to keep the pointer.
    fn call_return(&self) -> Option<(CallRetConfig, CallRetOffsets)> {
        if !self.link_cells() {
            return None;
        }
        let config = self.config.call_return_stack?;
        Some((config, A::call_return_offsets()?))
    }

    /// Push a frame recording that a return to `ret_pc` should enter the region
    /// containing it, via a cell of this region's own array.
    ///
    /// ```text
    ///   push(ret_pc, k):  base = ctx.callret_base;  if base == 0 goto done
    ///                     off  = (ctx.callret_off - 16) & MASK
    ///                     [base+off] = { ret_pc, &cells[k] }
    ///                     ctx.callret_off = off
    ///   done:
    /// ```
    ///
    /// No bounds check, and none is missing: the ring is a power of two and the
    /// mask is the wrap, so overflow overwrites the oldest frame - whose `RET`
    /// then misses and falls back to the probe. See [`crate::core::callret`] for
    /// why that is the whole of the overflow story.
    ///
    /// `ret_pc` is the return address as a run-time value; `ret_va` is the same
    /// address as a lift-time VA, which is what the cell is installed against.
    /// Inert when [`Self::call_return`] is `None`.
    pub fn emit_callret_push(&mut self, ret_va: u64, ret_pc: Value) {
        let Some((config, offsets)) = self.call_return() else {
            return;
        };
        let index = self.link_cell_for(ret_va, crate::core::link::LinkSiteKind::CallReturn);

        let push = self.builder.create_block();
        let done = self.builder.create_block();

        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let base = self.ctx_load(types::I64, ctx_ptr, offsets.base);
        // A host that has not allocated a ring is not an error and not a special
        // case, exactly as with a null `link_table`. It also keeps the frame
        // store, which is `notrap`, off a null pointer.
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
    /// Terminates the current block and leaves the builder in `slow`, like
    /// [`Self::emit_cell_dispatch`], which it ends in. Two loads and a compare
    /// against the probe's six instructions - and it cannot collide.
    ///
    /// **The pointer only advances on a match.** That is what makes underflow
    /// impossible: a frame nobody pushed holds zero, and no guest returns to
    /// zero, so a pop that finds one simply misses and leaves the pointer where
    /// it was.
    ///
    /// Inert when [`Self::call_return`] is `None`, in which case the caller's
    /// exit is emitted exactly as before.
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
        // Popped only now, on the match. A miss must leave the ring alone: the
        // frame it declined still belongs to some outer call.
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

    /// Terminate a VM exit the host answers by **continuing at a guest address**,
    /// giving it a chance to resolve its own successor.
    ///
    /// `pc` is that address as a run-time value and must be the one just written
    /// to the context - this does not write it. `None` means
    /// [`Self::current_ip`], i.e. what `sync_to_context(None, ..)` stored.
    ///
    /// Every caller is an exit whose whole servicing is "enter whatever is at
    /// `ctx.rip`". An exit the host must *do* something about first (an import
    /// call, a syscall, `CPUID`, a refused resume point) must return the
    /// ordinary way, or the host would never see it.
    ///
    /// With linking off this is exactly `return_`, which is what makes it safe
    /// to use at every such site unconditionally.
    pub fn emit_exit(&mut self, pc: Option<Value>) {
        if self.block_link().is_none() {
            return self.emit_host_return();
        }
        let pc = match pc {
            Some(pc) => pc,
            None => self.runtime_current_ip(),
        };
        if self.config.inline_link_probes {
            // Leaves the builder in the probe's `out` block, which is terminated.
            // Every caller either stops here or switches somewhere itself.
            self.emit_link_probe(pc);
        } else {
            let block = self.link_probe_block();
            self.builder.ins().jump(block, &[pc.into()]);
        }
    }

    /// The function's link-probe block, created on first use.
    ///
    /// Only its parameter exists at this point; [`Self::fill_link_probe`] emits
    /// the body once lifting is done, since a block created here would be filled
    /// in the middle of whichever exit stub happened to be first.
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
    /// Under [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned)
    /// the hit arm instead publishes `image_base` from the row and
    /// `return_call_indirect`s to `ent.code` with the pinned registers as
    /// arguments. Everything up to the compare is the same either way.
    ///
    /// A chain of those tail calls does not return to the host on its own: the
    /// way to end one is the check at the top - null `ctx.link_table` and the
    /// next probe declines.
    ///
    /// The loads are `trusted` (notrap, aligned) but deliberately **not**
    /// `readonly`: the host rewrites rows between entries.
    ///
    /// Nothing is stored on the miss path. `link_hit` is the host's to clear
    /// before entering (see [`BlockLinkOffsets::hit`]), which lets an exit with
    /// no probe at all report "no link" by doing nothing.
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

    /// Emit one probe, terminating the current block and leaving the builder in
    /// the probe's `out` arm.
    fn emit_link_probe(&mut self, pc: Value) {
        let (link, offsets) = self
            .block_link()
            .expect("a probe is only emitted when linking is on");

        let probe = self.builder.create_block();
        let hit = self.builder.create_block();
        let out = self.builder.create_block();

        let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
        let table = self.ctx_load(types::I64, ctx_ptr, offsets.table);
        // A host that has not published a table for this thread is not an error
        // and not a special case: it simply never links. It is also what keeps the
        // row loads, which are `notrap`, off a null table.
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

        // A hit *commits* the PC as well as naming the successor, which lets one
        // probe serve an exit whose `rip` is not where it continues: the
        // out-of-region direct `CALL` leaves `rip` at the return address and the
        // callee in `imported_function_va`, and a host services it by swapping the
        // two. Redundant at every other site.
        self.builder.switch_to_block(hit);
        self.ctx_store(pc, ctx_ptr, A::pc_offset());
        if self.pinned() {
            // The branch, rather than the answer: the successor is entered with the
            // pinned registers still in place, so nothing is published for a host
            // that is not going to run. `image_base` is the exception - under
            // `HostResume` the host sets it from the row before entering, and here
            // there is no host to do it.
            //
            // Hoisting these beside `key`, so the `br` does not wait on a load
            // started after the branch that selected it, is 0.616 s against
            // 0.589 s - worse. The two loads land on the miss path as well, and
            // the machine was already hiding the latency.
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

        self.builder.switch_to_block(out);
        self.emit_host_return();

        for block in [probe, hit, out] {
            self.visited_blocks.insert(block);
        }
    }

    pub fn lift(&mut self, bytes: &[u8], ip: u64) -> crate::Result<()> {
        self.lift_base = ip;
        self.lift_end = ip + bytes.len() as u64;
        self.covered = normalize_ranges(&self.config.block_ranges, ip, self.lift_end);
        // Guest addresses are encoded relative to the region base, so the map
        // has to be established before any IR is emitted.
        self.source_map = SourceLocMap::new(ip);

        // Phase 1: decode + block-leader discovery.
        //
        // The decode map is address-keyed so the same bytes can be decoded at
        // different starting offsets, which is what makes *sub-encoded* /
        // overlapping instructions work: a branch into the middle of an
        // instruction is decoded fresh at that offset, where a linear sweep would
        // only ever see one interpretation.
        //
        // Primed with a linear sweep (keeping the leader set a superset of the old
        // behaviour), then a worklist follows control flow to a fixpoint so
        // overlapping instructions that themselves branch are handled too.
        //
        // The sweep runs per covered range, so a hole between two blocks is never
        // decoded at all; see [`LifterConfig::block_ranges`].
        let mut decoded: HashMap<u64, A::Instr> = HashMap::new();
        for range in &self.covered {
            let (lo, hi) = ((range.start - ip) as usize, (range.end - ip) as usize);
            for inst in A::decode_region(&bytes[lo..hi], range.start) {
                decoded.entry(A::instr_ip(&inst)).or_insert(inst);
            }
        }

        // Recover jump-table successors so an indirect branch contributes leaders
        // instead of dead-ending at a VM exit. One pass up front is enough - the
        // dispatch sequences it matches are straight-line.
        if self.config.analyze_jump_tables {
            self.jump_tables = A::jump_table_targets(bytes, ip, &self.config.segments);
        }

        let mut leaders: HashSet<u64> = HashSet::from([ip]);
        let mut work: Vec<u64> = decoded.keys().copied().collect();
        work.push(ip);
        // Host-declared resume points are leaders before discovery rather than
        // after it, so that the code reachable *from* one is discovered too. An
        // address the guest only ever reaches through a pointer has no static
        // predecessor and would otherwise be a block with nothing behind it.
        if self.config.resumption != VmResumptionPoints::None {
            for &p in &self.config.resume_points {
                if self.in_region(p) {
                    leaders.insert(p);
                    work.push(p);
                }
            }
        }
        // Where a call that leaves the region comes back to. Collected because
        // these have to stay in the dispatch whatever the host declared: the
        // host cannot name them without decoding the region itself, and a
        // region whose calls cannot return is a region that runs once.
        let mut call_resumes: HashSet<u64> = HashSet::new();
        let mut processed: HashSet<u64> = HashSet::new();

        while let Some(addr) = work.pop() {
            if !self.in_region(addr) {
                continue;
            }
            if !processed.insert(addr) {
                continue;
            }
            // Ensure the instruction at `addr` is decoded. Overlap targets are
            // not present in the linear prime, so decode them on demand.
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
            // Its bytes are not all here, so neither are its control-flow facts:
            // a `next_ip` or a branch target read out of a truncated encoding is
            // a guess. Phase 2 turns this address into a resumable exit, and
            // discovery stops rather than following the guess.
            if self.runs_off_region(addr, inst) {
                leaders.insert(addr);
                continue;
            }
            let cf = A::control_flow(inst);
            let next = A::instr_next_ip(inst);
            match cf.kind {
                ControlFlowKind::Invalid => {
                    // Post-invalid bytes are a re-sync point (as in the old sweep).
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
                            // In-region call: the callee is a block leader.
                            leaders.insert(target);
                            work.push(target);
                        }
                    }
                    // A call returns to the following instruction, and that resume point is
                    // always a leader - whoever completes the call routes back through a
                    // dispatch switch, which can only reach addresses that have a block.
                    leaders.insert(next);
                    call_resumes.insert(next);
                    work.push(next);
                }
                ControlFlowKind::Ret => {}
                ControlFlowKind::IndirectBranch => {
                    // Normally a dead end for discovery. If the jump-table pass
                    // resolved this branch, its in-region targets are leaders
                    // like any other branch target; out-of-region ones stay the
                    // host's problem and keep the exit path they already had.
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

        // Between the phases, and only here: the decode map is complete enough
        // to follow control flow, and no IR exists yet for the answer to be too
        // late for. See [`crate::core::flags`].
        if self.config.eliminate_dead_flags {
            // Asked of the engine up front rather than from inside the
            // analysis, so that the one definition of "this encoding is only
            // partly here" - `runs_off_region`, which phase 2 also uses to
            // decide the exit - is the one the analysis sees.
            let truncated: HashSet<u64> = decoded
                .iter()
                .filter(|(addr, inst)| self.runs_off_region(**addr, inst))
                .map(|(addr, _)| *addr)
                .collect();
            let covered = self.covered.clone();
            let in_region = |va: u64| {
                covered
                    .partition_point(|r: &Range<u64>| r.start <= va)
                    .checked_sub(1)
                    .and_then(|i| covered.get(i))
                    .is_some_and(|r| va < r.end)
            };
            self.flag_liveness = Some(flags::analyze::<A>(&decoded, &in_region, |addr, _| {
                truncated.contains(&addr)
            }));
        }

        // Entry block receives the context pointer as its first parameter.
        let entry_block = self.builder.create_block();
        self.builder
            .append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);

        let ctx_ptr_val = self.builder.block_params(entry_block)[0];
        let ctx_ptr_var = A::context_ptr_var(&self.state);
        self.builder.def_var(ctx_ptr_var, ctx_ptr_val);

        A::load_state_from_context(self)?;

        // The pinned half of the register file arrives in registers instead:
        // parameter `1 + i` is slot `i`, and the frontend's state load skipped
        // exactly these (see [`Self::is_pinned`]). This is the entry half of
        // what the entry thunk does around the outside of the whole chain.
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

        // Sorted, because `get_block` *creates* the block and Cranelift numbers
        // blocks in creation order. `leaders` is a `HashSet` whose iteration
        // order is per-instance, so walking it directly makes the emitted IR -
        // and, downstream, the link-cell indices `link_cell_for` hands out in
        // emission order - different on every lift of the same bytes. The
        // compiled machine code was reproducible anyway; the IR and the cell
        // array were not, and an artifact cached on disk needs both.
        let mut ordered: Vec<u64> = leaders.iter().copied().collect();
        ordered.sort_unstable();
        for leader in ordered {
            self.get_block(leader);
        }

        let start_block = self.get_block(ip);

        // The dispatch is emitted **after** phase 2, because phase 2 is where some
        // of its arms come from: an instruction lifting to a *resumable* VM exit
        // registers its own resume point (see [`Self::resumes_at`]), and nobody
        // knows that address exists until it has been lifted.
        //
        // Cranelift will not let the builder leave a block unfilled, so the entry
        // block is terminated with a jump to one that stays empty until then. The
        // jump folds away.
        let wants_dispatch =
            !self.config.analysis_mode && self.config.resumption != VmResumptionPoints::None;
        let dispatch_block = wants_dispatch.then(|| self.builder.create_block());
        match dispatch_block {
            // No dispatch at all: enter at the region's first instruction and
            // nowhere else. Analysis mode emits inspection IR that is never run,
            // and [`VmResumptionPoints::None`] is a host that never re-enters.
            None => self.builder.ins().jump(start_block, &[]),
            Some(dispatch) => self.builder.ins().jump(dispatch, &[]),
        };

        // Phase 2: emit each block by walking instructions from its leader,
        // following `instr_next_ip` rather than list order - overlapping
        // instructions make the two diverge.
        //
        // Repeatedly take the lowest-addressed unemitted in-region block. New ones
        // can be created *during* lifting (e.g. call resume points), so re-scan
        // each iteration. Out-of-region blocks are left for `finish()` to turn
        // into VM-exit stubs.
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
                // Fell through past the end of the region.
                if !self.in_region(cur) {
                    self.current_ip = cur;
                    self.set_source_loc(cur);
                    // Linkable: the host's whole answer to a fallthrough exit is
                    // to enter a region at `cur`.
                    A::sync_to_context(self, None, ExitReason::None)?;
                    self.emit_static_exit(cur, None, crate::core::link::LinkSiteKind::StaticEdge);
                    self.block_terminated = true;
                    break;
                }
                // Reached another block's leader: emit an explicit fallthrough jump and
                // let that block be lifted on its own iteration. Exempt are the starting
                // leader (the block being filled) and a block an instruction's semantics
                // has already switched the builder into - jumping would target the block
                // we are already in.
                if !first {
                    if let Some(&next_block) = self.blocks.get(&cur) {
                        if self.builder.current_block() != Some(next_block) {
                            // A synthetic terminator, belonging to the block
                            // boundary at `cur` rather than to the instruction
                            // just lifted - whose location is still current.
                            self.set_source_loc(cur);
                            self.builder.ins().jump(next_block, &[]);
                            self.block_terminated = true;
                            break;
                        }
                    }
                }
                first = false;

                // Decode on demand (overlap targets may be absent from phase 1).
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
                            self.emit_static_exit(cur, None, crate::core::link::LinkSiteKind::StaticEdge);
                            self.block_terminated = true;
                            break;
                        }
                    }
                }
                let inst = &decoded[&cur];

                // The instruction begins here and ends past the bytes we were
                // given. Nothing about it has executed, so this is a transfer
                // *to* `cur` and the host resumes there with a region that holds
                // all of it. See [`LifterConfig::region_tail_exit`].
                if self.runs_off_region(cur, inst) {
                    self.current_ip = cur;
                    self.set_source_loc(cur);
                    // Linkable, and worth linking: the host answers this by
                    // lifting a region that begins at `cur` - and once it has,
                    // every later arrival at the same cut is a table hit.
                    A::sync_to_context(self, None, ExitReason::UnknownJump)?;
                    self.write_exit_code(TransferKind::RegionTail as u32);
                    self.emit_static_exit(cur, None, crate::core::link::LinkSiteKind::StaticEdge);
                    self.block_terminated = true;
                    break;
                }

                if A::is_invalid(inst) {
                    // Sync PC to point AT the invalid instruction. Not linkable:
                    // this is a fault the host reports, not an address it
                    // continues at, and a translation for it cannot exist.
                    self.current_ip = A::instr_ip(inst);
                    self.set_source_loc(self.current_ip);
                    A::sync_to_context(self, None, ExitReason::InvalidInstruction)?;
                    self.emit_host_return();
                    self.block_terminated = true;
                    break;
                }

                // Advance current_ip before dispatch so a PC read inside the
                // handler sees the post-instruction value. The source location,
                // by contrast, names the instruction *being* lifted, so it is
                // the pre-advance address.
                let next_ip = A::instr_next_ip(inst);
                self.set_source_loc(A::instr_ip(inst));
                // Latched against the address of the instruction about to be
                // lifted, which is why it happens here and not inside a handler:
                // `current_ip` is the *next* address by the time one runs.
                self.live_flags = match &self.flag_liveness {
                    Some(liveness) => liveness.live_out(A::instr_ip(inst)),
                    None => FlagSet::ALL,
                };
                self.current_ip = next_ip;
                A::lift_instruction(self, inst)?;
                if self.block_terminated {
                    break;
                }
                cur = next_ip;
            }
        }

        // Phase 3: the entry dispatch, now that every block that will exist
        // does. See where `dispatch_block` was created for why it waits.
        if let Some(dispatch) = dispatch_block {
            self.builder.switch_to_block(dispatch);
            self.visited_blocks.insert(dispatch);

            let ctx_ptr = self.builder.use_var(A::context_ptr_var(&self.state));
            let pc_val = self.ctx_load(types::I64, ctx_ptr, A::pc_offset());
            let entry_va = self.to_lift_va(pc_val);

            // Which addresses get an arm. Under [`VmResumptionPoints::All`] every
            // block; otherwise the three sets the engine knows and a host cannot name
            // - the region's entry, the return point of every call leaving it, and
            // the resume point of every VM exit that resumes - plus the host's own.
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
                    // Reported only for addresses control can enter here and then *make
                    // progress from*. Two kinds cannot, and both leave the PC exactly as the
                    // host set it - so a host taking one for a resume point would enter,
                    // exit unchanged, and spin forever:
                    //
                    // - **Out-of-region targets**, whose blocks `finish` turns into exit
                    //   stubs.
                    // - **The instruction the region was cut through** under
                    //   [`LifterConfig::region_tail_exit`], whose block is the tail exit.
                    let in_region = self.in_region(offset);
                    let tail = decoded
                        .get(&offset)
                        .is_some_and(|inst| self.runs_off_region(offset, inst));
                    if in_region && !tail {
                        self.emitted_resume_points.push(offset);
                    }
                }
            }

            // The default arm, which is where a PC that matched nothing lands.
            // Falling into the region's first block is a *guess*, and a silent
            // one; see [`LifterConfig::strict_resume`].
            let refuse = self
                .config
                .strict_resume
                .then(|| self.builder.create_block());
            switch.emit(&mut self.builder, entry_va, refuse.unwrap_or(start_block));

            // Filled only now: `switch.emit` terminates whichever block it was
            // called from, and Cranelift refuses to leave one unfilled.
            if let Some(refuse) = refuse {
                self.builder.switch_to_block(refuse);
                // No sync. The arm runs before any instruction is lifted, so the
                // context still holds exactly what the host wrote - including
                // the PC, which has to be left alone for the host to translate a
                // region at.
                self.write_exit_reason(ExitReason::UnknownJump);
                self.write_exit_code(TransferKind::NoResumePoint as u32);
                self.emit_host_return();
                self.visited_blocks.insert(refuse);
            }
        } else if !self.config.analysis_mode {
            // [`VmResumptionPoints::None`]: the entry jumps straight to the region's
            // first instruction, so that address - and only it - is where this code
            // can be entered. Reporting it keeps [`LiftedCode::resume_points`] the one
            // thing a host cache consults, whatever the mode.
            self.emitted_resume_points.push(ip);
        }

        self.finish()
    }

    /// Fill blocks that were created as jump targets but never visited - these
    /// are VM exits to addresses outside the lifted region.
    pub fn finish(&mut self) -> crate::Result<()> {
        let mut processed_blocks = HashSet::new();
        // **Sorted**, and load-bearing. `self.blocks` is a `HashMap`, and each
        // instance gets its own hash seed, so iterating it directly emits the
        // exit stubs in a different order on every lift - which renumbers the
        // blocks, and, since `link_cell_for` hands out indices in emission
        // order, permutes the link-cell array. Neither is a correctness bug (a
        // host reads `LiftedCode::link_sites` to learn what cell `i` is for),
        // but it costs reproducibility, which an artifact cached on disk needs.
        // Address order is the same order the sweep found them in.
        let mut all_offsets: Vec<u64> = self.blocks.keys().cloned().collect();
        all_offsets.sort_unstable();

        for offset in all_offsets {
            let block = self.blocks[&offset];
            if !self.visited_blocks.contains(&block) && !processed_blocks.contains(&block) {
                self.builder.switch_to_block(block);
                // The stub implements the transfer *to* `offset`, which is the
                // only guest address it can honestly be attributed to (the
                // instruction it belongs to may be any of several branches
                // targeting it).
                self.set_source_loc(offset);
                if self.config.analysis_mode {
                    // Inspection IR: out-of-region targets are bare returns; no
                    // context sync or exit code.
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
                    // Every stub here is a transfer *to* `offset` - a direct jump, a taken
                    // conditional edge, a direct call whose return address is pushed. All
                    // three are answered by entering a region at `offset`, which is what a
                    // link does - and `offset` is known *now*, which is what lets it be a
                    // cell rather than a hash of a runtime PC. The largest population of
                    // static edges in any region.
                    self.emit_static_exit(offset, None, crate::core::link::LinkSiteKind::StaticEdge);
                }
                processed_blocks.insert(block);
            }
        }

        // After the stubs: the probe is reached from them, and a block cannot be
        // jumped to before it exists.
        self.fill_link_probe()?;

        self.clear_source_loc();
        self.builder.seal_all_blocks();
        // After sealing: the use scan reads the finished layout.
        self.prune_untouched_sync();
        self.finalize_helper_indices();
        Ok(())
    }
}
