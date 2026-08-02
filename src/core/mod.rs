//! Guest-agnostic lift engine.
//!
//! Holds the generic `LiftEngine<'a, A: GuestArch>`: block discovery, the
//! resumable PC-dispatch switch, import fast-call machinery, the context-sync
//! skeleton, and exit handling - all written once and shared across guest
//! architectures. Each guest plugs in via the [`GuestArch`] trait.
//!
//! `Lifter<'a>` (the historical name) is a type alias for `LiftEngine<'a,
//! X86_64>`, re-exported from `crate::lifter` for source compatibility.

pub mod abi;
pub mod bridge;
pub mod exit;

use crate::arch::{ControlFlow, ControlFlowKind, InstructionEffects};
use crate::core::abi::CallAbi;
use crate::core::bridge::{BridgeImport, ImportKind, ImportRegistry};
use crate::core::exit::{ExitReason, MemorySegment, TransferKind};
use crate::function_traits::{DynamicSymbol, FunctionArgumentType};
use crate::os::get_host_calling_conv;
use cranelift::prelude::*;
use cranelift_codegen::ir::{
    ExtFuncData, ExternalName, FuncRef, Function, SigRef, UserExternalName, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A guest instruction-set frontend.
///
/// The generic engine calls into this trait for everything architecture
/// specific: decoding, static control-flow facts, the register file, context
/// layout offsets, per-instruction lifting, and state sync. Instruction
/// semantics themselves are implemented as inherent methods on the concrete
/// `LiftEngine<'a, Self>` monomorphization and reached via `lift_instruction`.
pub trait GuestArch: Sized {
    /// A single decoded guest instruction.
    type Instr;
    /// SSA-variable register file (guest registers, flags, dirty tracking, plus
    /// the universal `context_ptr` / `image_base` variables).
    type RegState;

    // --- context layout (drives the shared sync/exit code) ---

    /// Byte offset of the architectural PC field within the guest's
    /// `RegisterContext`.
    fn pc_offset() -> i32;
    /// Byte offset of the `exit_reason` control field.
    fn exit_reason_offset() -> i32;
    /// Byte offset of the `exit_code` control field.
    fn exit_code_offset() -> i32;

    // --- universal SSA variables held inside the register file ---

    /// The variable holding the pointer to the guest `RegisterContext`.
    fn context_ptr_var(state: &Self::RegState) -> Variable;
    /// The variable holding the runtime image base (for ASLR-resilient PC).
    fn image_base_var(state: &Self::RegState) -> Variable;

    // --- register file construction ---
    fn new_state(builder: &mut FunctionBuilder, config: &LifterConfig) -> Self::RegState;

    // --- decode (phase 1) ---
    fn decode_region(bytes: &[u8], ip: u64) -> Vec<Self::Instr>;
    /// Decode the single instruction at the start of `bytes`, whose first byte
    /// is at virtual address `ip`. Returns `None` only when `bytes` is too short
    /// to hold any instruction; invalid encodings are returned as an instruction
    /// for which [`is_invalid`](GuestArch::is_invalid) is true. Used by
    /// recursive-descent decoding ([`crate::cfg::decode_reachable`]).
    fn decode_one(bytes: &[u8], ip: u64) -> Option<Self::Instr>;
    fn instr_ip(i: &Self::Instr) -> u64;
    fn instr_next_ip(i: &Self::Instr) -> u64;
    fn is_invalid(i: &Self::Instr) -> bool;
    /// Static control-flow facts for leader discovery.
    fn control_flow(i: &Self::Instr) -> ControlFlow;

    /// Register/memory read-write effects of `i` for static analysis, or `None`
    /// if this guest frontend cannot compute them yet.
    ///
    /// Defaults to `None`; guest frontends override it where instruction-info is
    /// available (x86-64 does, via iced's instruction-info).
    fn instruction_effects(_i: &Self::Instr) -> Option<InstructionEffects> {
        None
    }

    /// Resolve indirect `JMP`/`CALL` targets in the region `bytes` (mapped at
    /// `base`) via value-set analysis, consulting `segments` for memory reads
    /// that leave the region.
    ///
    /// Defaults to an empty result; guest frontends override it where a value-set
    /// analyzer exists (x86-64 does, via [`crate::vsa`]). Surfaced through
    /// [`crate::cfg::resolved_indirect_targets`].
    fn resolved_indirect_targets(
        _bytes: &[u8],
        _base: u64,
        _segments: &[crate::context::MemorySegment],
    ) -> crate::cfg::ResolvedTargets {
        crate::cfg::ResolvedTargets::default()
    }

    // --- per-instruction lift + state sync (delegate to inherent methods) ---
    fn lift_instruction(lifter: &mut LiftEngine<'_, Self>, i: &Self::Instr) -> crate::Result<()>;
    fn load_state_from_context(lifter: &mut LiftEngine<'_, Self>) -> crate::Result<()>;
    fn sync_to_context(
        lifter: &mut LiftEngine<'_, Self>,
        pc: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()>;
}

/// Lift a region of guest code straight to a Cranelift [`Function`].
///
/// This is the one-call entry point for callers that just want IR and do not
/// need to drive the [`FunctionBuilder`] themselves. It encapsulates the full
/// setup that every consumer would otherwise hand-roll (and which the crate's
/// own tests repeat verbatim): build a `Function` with the host calling
/// convention and the single `*mut RegisterContext` (`I64`) parameter, create a
/// [`FunctionBuilder`], construct the [`LiftEngine`], run [`LiftEngine::lift`],
/// and release the builder's mutable borrow before returning the IR.
///
/// `A` selects the guest architecture (e.g.
/// [`crate::arch::x86_64::X86_64`]). `ip` is the virtual address the first byte
/// maps to.
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

/// Like [`lift_to_function`], but also returns the analysis-mode call-target map
/// ([`LiftEngine::analysis_call_targets`]): index `i` is the callee virtual
/// address referenced by `ExternalName::user(ANALYSIS_CALL_NAMESPACE, i)`, i.e.
/// the `i`-th analysis `call fnN` in the IR. The vec is empty unless
/// `config.analysis_mode` is set (only that mode emits analysis `call`s).
pub fn lift_to_function_with_call_targets<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<(Function, Vec<u64>)> {
    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    // The lifted function takes a single argument: `*mut RegisterContext`.
    func.signature.params.push(AbiParam::new(types::I64));

    let mut builder_ctx = FunctionBuilderContext::new();
    let builder = FunctionBuilder::new(&mut func, &mut builder_ctx);

    let mut engine = LiftEngine::<A>::with_config(builder, config);
    engine.lift(bytes, ip)?;
    let call_targets = std::mem::take(&mut engine.analysis_call_targets);

    // Release the mutable borrow of `func` held by the builder/engine.
    drop(engine);

    Ok((func, call_targets))
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

/// Build a Cranelift `Signature` from a `DynamicSymbol`'s `arg_types` slice.
///
/// `arg_types[0]` is the return type; `arg_types[1..]` are parameter types.
/// Stops at the first `VariadicList`/`None` marker - the same cutoff
/// [`CallAbi::place_args`] uses, so the emitted call always carries exactly as
/// many arguments as this signature declares.
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
/// signature (and, in `RuntimeAssisted` mode, its external function and
/// `aot_imports` entry) on the way.
///
/// Returns `None` when the import cannot be fast-called at all - it is variadic,
/// or its registered ABI cannot place every argument - in which case the caller
/// falls back to the slow `ImportedFunctionCall` exit.
///
/// Shared by config-time prebuilding and the lazy path taken for imports the
/// [`ImportRegistry`] reports mid-lift, so both produce identical IR and both
/// respect the one-`FuncRef`-per-symbol-name invariant `aot_imports` depends on.
fn build_fast_call(
    builder: &mut FunctionBuilder,
    execution_mode: ExecutionMode,
    aot_imports: &mut Vec<(&'static str, *const u8, Signature)>,
    fast_call_by_name: &mut HashMap<&'static str, FastCall>,
    import: &FastImport,
) -> Option<FastCall> {
    let sym = &import.symbol;
    // Variadic functions, and functions with arguments the registered ABI
    // cannot place, are demoted to slow calls.
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

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    pub syscall_handler_call_conv: CallConv,
    pub segments: Vec<MemorySegment>,
    pub interpreter_fallback: bool,
    pub emit_resume_blocks: bool,
    /// Statically registered import addresses.
    ///
    /// Key: virtual address of the imported function as it appears in the code
    /// being lifted. Value: how that address is reached (see [`ImportKind`]).
    ///
    /// This map is a snapshot taken when the config is built. Addresses that
    /// only become imports *later* (a `dlopen`ed image, a lazily bound stub)
    /// belong in [`import_registry`](Self::import_registry) instead, which the
    /// engine queries while lifting.
    pub imports: HashMap<u64, ImportKind>,
    /// A live view of which addresses are imports, queried per call site at lift
    /// time rather than snapshotted here.
    ///
    /// Consulted only for addresses absent from [`imports`](Self::imports), so
    /// an explicit registration always wins. See [`ImportRegistry`] for why this
    /// is what makes `dlopen`/`dlsym` calls work without re-plumbing anything.
    pub import_registry: Option<Arc<dyn ImportRegistry>>,
    /// Optional bridge for indirect calls whose target is only known at run
    /// time (`CALL reg` / `CALL [mem]` to an unregistered address).
    ///
    /// When set, an indirect call becomes a call to
    /// `extern "C" fn(*mut RegisterContext, target: u64) -> u32` instead of an
    /// [`ExitReason::IndirectCall`] VM exit: the runtime does its own dispatch
    /// (imported function? another guest image? a `dlsym`ed pointer?) on the
    /// host side and returns 0 when it handled the call inline, or non-zero when
    /// the target is guest code and a real exit is required.
    ///
    /// On the non-zero path the context already describes a well-formed
    /// `IndirectCall` exit (callee in `imported_function_va`, PC at the
    /// instruction after the call, stack pointer unchanged); the bridge may
    /// overwrite `exit_reason`/`exit_code` to say something else.
    pub indirect_call_bridge: Option<BridgeImport>,
    /// When `true`, vector registers are stored as native `I64X2` SSA variables
    /// and whole-register handlers take a vector fast path. When `false` (the
    /// default) they are stored as scalar lo/hi 64-bit pairs, which avoids
    /// relying on Cranelift's vector lowering and is required for AOT targets
    /// that lack SIMD support (e.g. RISC-V64).
    pub use_vector_xmm: bool,
    /// Produce *inspection* IR for static analysis rather than runnable IR.
    ///
    /// When `true`, the engine omits the machinery that exists purely to support
    /// re-entrant JIT/AOT execution:
    /// - the entry PC-dispatch `Switch` is replaced by a direct jump to the
    ///   region's first block, and
    /// - context write-back (`sync_to_context`) is skipped, so guest registers
    ///   stay as pure SSA values and no per-exit stores are emitted, and
    /// - out-of-region exit stubs become bare `return`s.
    ///
    /// The resulting `Function` is a clean control-flow graph of the guest's own
    /// blocks - far more readable for analysis - but it does **not** write guest
    /// state back to the `RegisterContext`, so it is not suitable for execution.
    pub analysis_mode: bool,
}

impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            syscall_handler_call_conv: CallConv::SystemV,
            segments: Vec::new(),
            interpreter_fallback: false,
            emit_resume_blocks: false,
            imports: HashMap::new(),
            import_registry: None,
            indirect_call_bridge: None,
            use_vector_xmm: false,
            analysis_mode: false,
        }
    }
}

impl LifterConfig {
    /// Register an import with full fast-call metadata.
    ///
    /// `abi` describes how the *guest* passes arguments to this function - the
    /// convention the guest binary was compiled for, which is a per-function
    /// property and need not match the host's. Use a preset such as
    /// [`CallAbi::sysv_amd64`](crate::core::abi::CallAbi::sysv_amd64) or
    /// [`CallAbi::win64`](crate::core::abi::CallAbi::win64), or build your own.
    ///
    /// Imports are demoted to slow `ImportedFunctionCall` exits when they are
    /// variadic, or when `abi` cannot place all of their arguments.
    pub fn register_fast_import(&mut self, va: u64, symbol: DynamicSymbol, abi: CallAbi) {
        self.imports
            .insert(va, ImportKind::Fast(FastImport { symbol, abi }));
    }

    /// Register an import address with no fast-call metadata. Every call to this
    /// address produces an `ImportedFunctionCall` VM exit.
    pub fn register_slow_import(&mut self, va: u64) {
        self.imports.insert(va, ImportKind::Slow);
    }

    /// Register an import reached through a runtime-supplied bridge.
    ///
    /// Strictly more general than [`register_fast_import`](Self::register_fast_import):
    /// it drops the requirement that the callee's signature be expressible in
    /// [`FunctionArgumentType`]/[`CallAbi`], which is what forces
    /// struct-by-value, callback-takers and variadics onto the slow exit path
    /// today. It is also cheaper than the slow exit in the common case, since a
    /// bridge that returns 0 continues inline with no VM exit at all.
    ///
    /// `register_fast_import` remains the right thing for imports whose
    /// signature *is* describable - it emits a direct call with no context sync.
    ///
    /// See [`crate::core::bridge`] for the calling contract.
    ///
    /// Both bridge targets embed a host address in the emitted code (directly,
    /// or as the address of the slot to load from), so unlike a
    /// `RuntimeAssisted` fast import a bridge produces no named relocation.
    /// That is the intent - a bridge belongs to the running runtime - but it
    /// means AOT output containing one is only valid for a process that maps
    /// the bridge (or its slot) at the same address.
    pub fn register_bridge_import(&mut self, va: u64, bridge: BridgeImport) {
        self.imports.insert(va, ImportKind::Bridge(bridge));
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
    /// `next_ip` of the instruction currently being lifted - the value the PC
    /// would hold after that instruction retires.
    pub current_ip: u64,
    pub visited_blocks: HashSet<Block>,
    /// Whether the current block has been terminated (jmp, ret, etc.).
    pub block_terminated: bool,
    /// Designated address for controlled VM exit via RET.
    pub exit_trampoline_address: u64,
    pub syscall_handler_sig: Option<SigRef>,
    /// Fast-call data for registered non-variadic imports, keyed by import VA.
    /// The execution mode decides whether the target is absolute (`Jit`) or
    /// relocatable (`RuntimeAssisted`).
    ///
    /// Pre-built for [`LifterConfig::imports`]; filled in lazily (and cached
    /// here) for imports the [`ImportRegistry`] reports while lifting.
    pub fast_calls: HashMap<u64, FastCall>,
    /// The same imported symbol may be registered at several virtual addresses
    /// (e.g. its PLT stub and its GOT slot). In `RuntimeAssisted` mode every
    /// distinct symbol becomes one [`Self::aot_imports`] entry / object
    /// relocation target, so a single `FuncRef` is reused per symbol name;
    /// emitting the name twice would collide when the object backend
    /// deduplicates imports and break the `FuncId(i) == i` invariant. The whole
    /// `FastCall` is reused so the arguments marshaled at every site match the
    /// one signature that was declared.
    fast_call_by_name: HashMap<&'static str, FastCall>,
    /// Cached signature of the bridge ABI
    /// (`extern "C" fn(*mut RegisterContext, u64) -> u32`), created on first
    /// use.
    bridge_sig: Option<SigRef>,
    /// Ordered external functions for object/JIT backends. Index `i` corresponds
    /// to `ExternalName::user(0, i)`. Entry = `(symbol_name, fn_ptr, signature)`.
    pub aot_imports: Vec<(&'static str, *const u8, Signature)>,
    /// Virtual addresses registered in [`LifterConfig::imports`], whatever their
    /// kind. Addresses that only the [`ImportRegistry`] knows about are *not*
    /// here (a registry is not enumerable) - use
    /// [`is_import`](Self::is_import), which consults both.
    pub import_vas: HashSet<u64>,
    /// Records what kind of transfer targets each unvisited (out-of-region)
    /// block, so `finish()` can emit the correct `TransferKind`.
    pub block_transfer_kinds: HashMap<Block, TransferKind>,
    /// Analysis-mode only: callee virtual address for each analysis `call`
    /// emitted, indexed by the external-function index used for that call
    /// (`ExternalName::user(ANALYSIS_CALL_NAMESPACE, i)`). Indirect calls record
    /// [`u64::MAX`] since their target is not statically known. Lets a consumer
    /// map a printed `call fnN` back to the callee it represents.
    pub analysis_call_targets: Vec<u64>,
    /// Cached empty (no-arg, no-result) signature used by analysis-mode `call`s.
    analysis_call_sig: Option<SigRef>,
}

/// Namespace used for the external functions referenced by analysis-mode
/// `call`s. Distinct from namespace `0` (AOT imports) so the
/// [`LiftEngine::analysis_call_targets`] index space is unambiguous.
pub const ANALYSIS_CALL_NAMESPACE: u32 = 1;

impl<'a, A: GuestArch> LiftEngine<'a, A> {
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        Self::with_config(builder, LifterConfig::default())
    }

    pub fn with_config(builder: FunctionBuilder<'a>, config: LifterConfig) -> Self {
        let mut builder = builder;
        let state = A::new_state(&mut builder, &config);
        let syscall_handler_sig = match config.execution_mode {
            ExecutionMode::Jit => None,
            ExecutionMode::RuntimeAssisted => {
                let mut sig = Signature::new(config.syscall_handler_call_conv);
                sig.params.push(AbiParam::new(types::I64));
                Some(builder.import_signature(sig))
            }
        };

        // Build fast-call data for every statically registered non-variadic
        // import. Imports the `ImportRegistry` reports are built the same way,
        // lazily, when a call site referencing them is lifted.
        let mut fast_calls: HashMap<u64, FastCall> = HashMap::new();
        let mut aot_imports: Vec<(&'static str, *const u8, Signature)> = Vec::new();
        let mut import_vas: HashSet<u64> = HashSet::new();
        let mut fast_call_by_name: HashMap<&'static str, FastCall> = HashMap::new();

        for (&va, kind) in &config.imports {
            import_vas.insert(va);
            let ImportKind::Fast(import) = kind else {
                continue;
            };
            if let Some(fast_call) = build_fast_call(
                &mut builder,
                config.execution_mode,
                &mut aot_imports,
                &mut fast_call_by_name,
                import,
            ) {
                fast_calls.insert(va, fast_call);
            }
        }

        Self {
            builder,
            state,
            config,
            blocks: HashMap::new(),
            sealed_blocks: HashSet::new(),
            lift_base: 0,
            lift_end: 0,
            current_ip: 0,
            visited_blocks: HashSet::new(),
            block_terminated: false,
            exit_trampoline_address: 0xFFFF_FFFF_FFFF_FFFF,
            syscall_handler_sig,
            fast_calls,
            fast_call_by_name,
            bridge_sig: None,
            aot_imports,
            import_vas,
            block_transfer_kinds: HashMap::new(),
            analysis_call_targets: Vec::new(),
            analysis_call_sig: None,
        }
    }

    /// Declare (or reuse) an external function representing an analysis-mode call
    /// to `callee_va`, returning a `FuncRef` to a no-arg/no-result function and
    /// recording the callee in [`Self::analysis_call_targets`]. Only used when
    /// `config.analysis_mode` is set.
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
    /// [`ImportRegistry`](crate::core::bridge::ImportRegistry) if one is
    /// installed - so an explicit registration always wins over the live view.
    ///
    /// This is queried per call site *while lifting*, not snapshotted when the
    /// engine is constructed, which is what lets a region lifted after a
    /// `dlopen` see the newly bound stubs.
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
        // Not prebuilt: either not a fast import at all, or one the registry
        // only just reported. Only the latter is worth building.
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
    /// side of a branch that was just emitted.
    ///
    /// The emission loop only creates blocks for leaders, so straight-line
    /// code after a leader is tail-duplicated into every block that reaches it. When
    /// two such copies both end in a conditional branch, the second one finds its
    /// fallthrough block already filled by the first. There is nothing left to emit
    /// in that case - the `brif` edge already points at the filled block - so just
    /// terminate the current block instead of switching into it (which would trip
    /// cranelift's "switch to a block which is already filled" assertion).
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
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), val, ctx_ptr, A::exit_code_offset());
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
            let image_base = self.builder.use_var(A::image_base_var(&self.state));
            let offset = self.current_ip.wrapping_sub(self.lift_base);
            self.builder.ins().iadd_imm_s(image_base, offset as i64)
        };
        self.builder
            .ins()
            .store(MemFlagsData::trusted(), pc_val, ctx_ptr, A::pc_offset());

        let reason_val = self.builder.ins().iconst(types::I32, reason as i64);
        self.builder.ins().store(
            MemFlagsData::trusted(),
            reason_val,
            ctx_ptr,
            A::exit_reason_offset(),
        );
        Ok(())
    }

    pub fn lift(&mut self, bytes: &[u8], ip: u64) -> crate::Result<()> {
        self.lift_base = ip;
        self.lift_end = ip + bytes.len() as u64;

        // Phase 1: decode + block-leader discovery.
        //
        // We keep an address-keyed decode map so that the same bytes can be
        // decoded more than once at different starting offsets. This is what
        // makes *sub-encoded* / overlapping instructions work: an obfuscator can
        // branch into the middle of an instruction so those bytes are executed
        // as a different instruction. A pure linear sweep never sees the second
        // interpretation; here, decoding follows real control flow, so a branch
        // target that lands mid-instruction is decoded fresh at that offset.
        //
        // The map is primed with a linear sweep (covering every natural
        // instruction boundary, which preserves resume-block behavior and keeps
        // the leader set a superset of the old linear-sweep result), then a
        // worklist follows control flow from the entry and every discovered
        // leader, decoding any address not already present. The worklist is
        // seeded with every linearly-decoded instruction so leader discovery is
        // identical to the old sweep for non-overlapping code, and it iterates
        // to a fixpoint so overlapping instructions that themselves branch are
        // handled too.
        let mut decoded: HashMap<u64, A::Instr> = HashMap::new();
        for inst in A::decode_region(bytes, ip) {
            decoded.entry(A::instr_ip(&inst)).or_insert(inst);
        }

        let mut leaders: HashSet<u64> = HashSet::from([ip]);
        let mut work: Vec<u64> = decoded.keys().copied().collect();
        work.push(ip);
        let mut processed: HashSet<u64> = HashSet::new();

        while let Some(addr) = work.pop() {
            if addr < self.lift_base || addr >= self.lift_end {
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
                        if !self.is_import(target)
                            && target >= self.lift_base
                            && target < self.lift_end
                        {
                            // In-region call: the callee is a block leader.
                            leaders.insert(target);
                            work.push(target);
                        }
                    }
                    // A call returns to the following instruction, and that
                    // resume point is always a block leader - whoever completes
                    // the call routes back to it through a dispatch switch. For
                    // an import or out-of-region call that is the exit/resume
                    // path; for an in-region callee it is the `RET` dispatch,
                    // which can only reach addresses that have a block.
                    leaders.insert(next);
                    work.push(next);
                }
                ControlFlowKind::Ret | ControlFlowKind::IndirectBranch => {}
            }
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

        if self.config.emit_resume_blocks {
            for instruction in decoded.values() {
                leaders.insert(A::instr_ip(instruction));
                leaders.insert(A::instr_next_ip(instruction));
            }
        }

        for &leader in &leaders {
            self.get_block(leader);
        }

        let start_block = self.get_block(ip);

        if self.config.analysis_mode {
            // Analysis mode: skip the resumable PC dispatch entirely and just
            // enter at the region's first instruction.
            self.builder.ins().jump(start_block, &[]);
        } else {
            // Resumable dispatch: jump to the block matching the context PC.
            let pc_val = self.builder.ins().load(
                types::I64,
                MemFlagsData::trusted(),
                ctx_ptr_val,
                A::pc_offset(),
            );
            let image_base_val = self.builder.use_var(A::image_base_var(&self.state));
            let relative_pc = self.builder.ins().isub(pc_val, image_base_val);

            let mut switch = Switch::new();
            for (&offset, &block) in &self.blocks {
                let relative_offset = offset.wrapping_sub(self.lift_base);
                switch.set_entry(relative_offset.into(), block);
            }

            switch.emit(&mut self.builder, relative_pc, start_block);
        }

        // Phase 2: emit each block by walking instructions from its leader,
        // following `instr_next_ip` rather than linear list order. Overlapping
        // (sub-encoded) instructions make list order and fallthrough order
        // diverge, so a per-block walk is required for correctness.
        //
        // Repeatedly take the lowest-addressed in-region block that has not yet
        // been emitted. New in-region blocks can be created *during* lifting
        // (e.g. call resume points), so re-scan `self.blocks` each iteration
        // until every in-region block has been visited. Out-of-region target
        // blocks are left for `finish()` to turn into VM-exit stubs.
        loop {
            let next_leader = self
                .blocks
                .iter()
                .filter(|(off, b)| {
                    **off >= self.lift_base
                        && **off < self.lift_end
                        && !self.visited_blocks.contains(b)
                })
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
                if cur < self.lift_base || cur >= self.lift_end {
                    self.current_ip = cur;
                    A::sync_to_context(self, None, ExitReason::None)?;
                    self.builder.ins().return_(&[]);
                    self.block_terminated = true;
                    break;
                }
                // Reached another block's leader. Normally we emit an explicit
                // fallthrough jump and let that block be lifted on its own
                // iteration. The starting leader is exempt (that is the block we
                // are filling). Also exempt: when an instruction's semantics has
                // already switched the builder into this exact block (e.g. a
                // conditional branch positions us in its fallthrough block and
                // marks it visited, expecting the caller to keep emitting into
                // it) - jumping would target the block we are already in.
                if !first {
                    if let Some(&next_block) = self.blocks.get(&cur) {
                        if self.builder.current_block() != Some(next_block) {
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
                            A::sync_to_context(self, None, ExitReason::None)?;
                            self.builder.ins().return_(&[]);
                            self.block_terminated = true;
                            break;
                        }
                    }
                }
                let inst = &decoded[&cur];

                if A::is_invalid(inst) {
                    // Sync PC to point AT the invalid instruction.
                    self.current_ip = A::instr_ip(inst);
                    A::sync_to_context(self, None, ExitReason::InvalidInstruction)?;
                    self.builder.ins().return_(&[]);
                    self.block_terminated = true;
                    break;
                }

                // Advance current_ip before dispatch so a PC read inside the
                // handler sees the post-instruction value.
                let next_ip = A::instr_next_ip(inst);
                self.current_ip = next_ip;
                A::lift_instruction(self, inst)?;
                if self.block_terminated {
                    break;
                }
                cur = next_ip;
            }
        }

        self.finish()
    }

    /// Fill blocks that were created as jump targets but never visited - these
    /// are VM exits to addresses outside the lifted region.
    pub fn finish(&mut self) -> crate::Result<()> {
        let mut processed_blocks = HashSet::new();
        let all_offsets: Vec<u64> = self.blocks.keys().cloned().collect();

        for offset in all_offsets {
            let block = self.blocks[&offset];
            if !self.visited_blocks.contains(&block) && !processed_blocks.contains(&block) {
                self.builder.switch_to_block(block);
                if self.config.analysis_mode {
                    // Inspection IR: out-of-region targets are bare returns; no
                    // context sync or exit code.
                    self.builder.ins().return_(&[]);
                } else {
                    self.current_ip = offset;
                    A::sync_to_context(self, None, ExitReason::UnknownJump)?;
                    let kind = self
                        .block_transfer_kinds
                        .get(&block)
                        .copied()
                        .unwrap_or(TransferKind::Unknown);
                    self.write_exit_code(kind as u32);
                    self.builder.ins().return_(&[]);
                }
                processed_blocks.insert(block);
            }
        }

        self.builder.seal_all_blocks();
        Ok(())
    }
}
