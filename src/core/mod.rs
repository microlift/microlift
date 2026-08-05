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
pub mod srcloc;

use crate::arch::{ControlFlow, ControlFlowKind, InstructionEffects};
use crate::core::abi::CallAbi;
use crate::core::bridge::{BridgeImport, ImportKind, ImportRegistry};
use crate::core::exit::{ExitReason, MemorySegment, TransferKind};
use crate::core::srcloc::SourceLocMap;
use crate::function_traits::{DynamicSymbol, FunctionArgumentType};
use crate::os::get_host_calling_conv;
use cranelift::prelude::*;
use cranelift_codegen::ir::{
    ExtFuncData, ExternalName, FuncRef, Function, SigRef, SourceLoc, UserExternalName,
    UserExternalNameRef, UserFuncName,
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
    /// Byte offset of the `image_base` relocation field.
    fn image_base_offset() -> i32;

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

    /// **Experimental.** Recover the successors of indirect branches in `bytes`
    /// (mapped at `base`) by recognising compiler-emitted jump tables and
    /// reading them out of `segments`. Keyed by the address of the branch.
    ///
    /// Only consulted when
    /// [`LifterConfig::analyze_jump_tables`](LifterConfig::analyze_jump_tables)
    /// is set. Results are used purely to *add* block leaders, so a wrong answer
    /// costs dead code and never correctness - see
    /// [`crate::arch::x86_64::jumptable`], the one implementation.
    fn jump_table_targets(
        _bytes: &[u8],
        _base: u64,
        _segments: &[crate::context::MemorySegment],
    ) -> HashMap<u64, Vec<u64>> {
        HashMap::new()
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

/// A helper function emitted *alongside* the lifted region rather than inlined
/// into it - see [`LiftEngine::outlined_helper`].
///
/// It is a complete, self-contained Cranelift function that the lifted code
/// reaches with an ordinary `call`. Backends must define it with
/// [`Linkage::Local`](cranelift_module::Linkage::Local) *before* the lifted
/// function, in the order [`LiftedCode::helpers`] lists them, so its
/// `FuncId` matches the `ExternalName::user(0, i)` baked into the call.
#[derive(Debug, Clone)]
pub struct HelperFunction {
    /// Symbol the helper is declared under. Unique per helper body.
    pub name: &'static str,
    /// The helper's IR.
    pub func: Function,
}

/// Everything one lift produces: the region's own function plus the external
/// functions it references.
///
/// The namespace-0 index space `ExternalName::user(0, i)` covers **both**
/// lists, imports first: index `i < imports.len()` is `imports[i]`, and index
/// `imports.len() + j` is `helpers[j]`. Backends declare them in exactly that
/// order so `FuncId(i) == i`.
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
    /// [`func`](Self::func) back to guest instruction addresses.
    ///
    /// Meaningful only when the lift ran with
    /// [`LifterConfig::emit_source_locs`] on; with it off the IR carries no
    /// locations and this map has nothing to decode. Pass it to
    /// [`GuestCodeMap::from_compiled`](crate::core::srcloc::GuestCodeMap::from_compiled)
    /// to turn a compiled function's surviving locations into a host-offset ->
    /// guest-VA map.
    pub source_map: SourceLocMap,
}

/// Lift a region of guest code to a Cranelift [`Function`] plus the external
/// functions it references.
///
/// This is the complete form of [`lift_to_function`], and the one a backend
/// should use: a region containing x87 arithmetic calls into outlined
/// softfloat helpers, and those helper bodies come back in
/// [`LiftedCode::helpers`]. Compiling [`LiftedCode::func`] on its own would
/// leave those calls unresolved.
pub fn lift_to_code<A: GuestArch>(
    bytes: &[u8],
    ip: u64,
    config: LifterConfig,
) -> crate::Result<LiftedCode> {
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
    let analysis_call_targets = std::mem::take(&mut engine.analysis_call_targets);
    let imports = std::mem::take(&mut engine.aot_imports);
    let source_map = engine.source_map;
    let helpers = engine.take_helpers();

    // Release the mutable borrow of `func` held by the builder/engine.
    drop(engine);

    Ok(LiftedCode {
        func,
        imports,
        helpers,
        analysis_call_targets,
        source_map,
    })
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

/// Reduce a [`LiftedCode`] to its main function, erroring if that would drop
/// helper bodies the function calls.
///
/// The single-`Function` entry points predate outlined helpers; rather than
/// silently hand back IR with unresolvable `call`s, they refuse and name the
/// two ways out.
fn require_no_helpers(code: LiftedCode) -> crate::Result<(Function, Vec<u64>)> {
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

/// How the lifted code forms guest virtual addresses (RIP-relative operands,
/// PC values written on a VM exit, pushed return addresses, and the dispatch
/// that maps a runtime address back to a block).
///
/// Every such address is a *lift-time* VA - one the decoder saw, somewhere in
/// `[lift_base, lift_end)` or a branch/call target outside it. This mode picks
/// how that VA is materialized at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressMode {
    /// Addresses are formed as `image_base + (va - lift_base)`, with
    /// `image_base` loaded from the guest context at region entry, and mapped
    /// back with the inverse. The lifted artifact therefore stays correct when
    /// the guest image is loaded at a different base than it was lifted at,
    /// which is what makes it safe to cache compiled code on disk and reuse it
    /// across runs (and what `RuntimeAssisted` output requires).
    #[default]
    ImageBaseRelative,
    /// Addresses are baked in as constants: a lift-time VA becomes an `iconst`
    /// and the reverse mapping is the identity. Cheaper - the per-address `add`
    /// and the entry load of `image_base` disappear, and constant addresses fold
    /// into the addressing modes of the host instructions - but the resulting
    /// code is only valid at the base it was lifted for.
    ///
    /// Appropriate for an in-process JIT that lifts an image already mapped at
    /// its final address and re-lifts on the next run. `RegisterContext::image_base`
    /// is then ignored: the PC the host reads on exit, and the PC it sets to
    /// resume, are plain lift-time VAs.
    Direct,
}

/// How a guest 128-bit compare-and-swap (x86-64 `CMPXCHG16B`) is realized.
///
/// Cranelift can express a true 128-bit CAS - `atomic_cas` on `I128` - but only
/// the x64 backend lowers it, and only when the ISA setting `has_cmpxchg16b` is
/// enabled (it is off in the baseline `x86-64` feature set, so AOT callers must
/// pass it through `isa_settings`; `cranelift-native` detects it for the JIT).
/// On every other backend there is no IR-expressible lock-free 128-bit RMW at
/// all, so the lifter has to be told what to do instead - hence this knob rather
/// than a fixed lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Atomic128Mode {
    /// Lower to a plain 128-bit read-modify-write: load both halves, compare,
    /// store back. Portable to every backend and correct for a single-threaded
    /// guest, but **not atomic** - another thread can observe or clobber the
    /// value between the load and the store.
    ///
    /// The default, on the same grounds as [`LifterConfig::use_vector_xmm`]
    /// defaulting off: the portable lowering is the one that compiles
    /// everywhere. Pick a different mode for a guest that actually shares the
    /// location across threads.
    #[default]
    Emulated,
    /// Emit `atomic_cas` on `I128`, which the x64 backend lowers to a real
    /// `lock cmpxchg16b`. Exact and fast, but the compile **fails** on any
    /// backend that cannot lower it (aarch64, riscv64, s390x) or on an x64 ISA
    /// built without `has_cmpxchg16b`.
    Native,
    /// Do not lower it at all: exit to the host with
    /// [`ExitReason::UnimplementedInstruction`] so the runtime can perform the
    /// operation itself (e.g. under a host-side lock). The only option that is
    /// both portable and correct for a multi-threaded guest, at the cost of a VM
    /// exit per execution.
    ///
    /// Follows the ordinary unimplemented-instruction contract: guest state is
    /// fully synced, `exit_code` holds the instruction's iced `Code`, and the PC
    /// is already past the `CMPXCHG16B`, so the host emulates it and resumes
    /// without adjusting the PC.
    ///
    /// [`ExitReason::UnimplementedInstruction`]: crate::context::ExitReason::UnimplementedInstruction
    Exit,
}

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    /// How guest virtual addresses are materialized. See [`AddressMode`];
    /// defaults to [`AddressMode::ImageBaseRelative`], which is required for
    /// relocatable/cacheable artifacts.
    pub address_mode: AddressMode,
    pub syscall_handler_call_conv: CallConv,
    /// The layout of immutable segments in memory at lift time.
    /// This helps with value-set analysis based optimizations,
    /// as we can promote memory values into constants at lift time.
    ///
    /// NOTE: Do NOT use this on writable data! This is assumed to be *immutable*
    /// memory. There is a `writable` property incase future optimizations permit
    /// mutable memory to be valid for hints.
    pub segments: Vec<MemorySegment>,
    pub interpreter_fallback: bool,
    pub emit_resume_blocks: bool,
    /// **Experimental.** Before emitting IR, try to recover the successors of
    /// each indirect branch from a compiler-emitted jump table, using
    /// [`segments`](Self::segments) as the backing store the table is read from
    /// (the lifted region itself is always readable, so an in-function table
    /// needs no segment at all).
    ///
    /// A `switch` compiled by Clang otherwise costs a VM exit on every
    /// dispatch: `JMP reg` contributes no static targets, so the case blocks
    /// never become leaders, never get lifted, and the runtime dispatch switch
    /// has nothing to match. With this on, the recovered targets are added to
    /// the leader set and the existing dispatch covers them - the
    /// [`ExitReason::UnknownJump`] stub survives only as the default arm, for
    /// targets the analysis did not predict.
    ///
    /// Recovered targets are only ever *added* as leaders, and dispatch still
    /// happens on the address the guest computes at run time, so a mispredicted
    /// entry lifts unreachable code rather than producing wrong code. Off by
    /// default because the pattern matching is heuristic and the extra blocks
    /// are not free. See [`crate::arch::x86_64::jumptable`].
    pub analyze_jump_tables: bool,
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
    /// How a guest 128-bit compare-and-swap (`CMPXCHG16B`) is lowered. See
    /// [`Atomic128Mode`]; defaults to the portable, non-atomic emulation.
    pub atomic128: Atomic128Mode,
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
    /// Emit large shared code sequences - currently the 80-bit x87 softfloat
    /// kernels - once as helper functions and `call` them, instead of inlining
    /// a copy at every site. On by default: the kernels run to hundreds of IR
    /// instructions each, so a region with a handful of x87 ops otherwise
    /// compiles to code that thrashes the instruction cache.
    ///
    /// Turning it **off** yields a single self-contained `Function` again,
    /// which is what [`lift_to_function`] (as opposed to [`lift_to_code`])
    /// requires. Doing so is worthwhile for a region with one x87 operation,
    /// where the call is pure overhead and inlining lets Cranelift fold the
    /// kernel against known operands.
    ///
    /// Ignored under [`analysis_mode`](Self::analysis_mode), which always
    /// inlines so the inspection IR stays self-contained.
    pub outline_helpers: bool,
    /// Tag every emitted IR instruction with the guest instruction it was lifted
    /// from, encoded as a Cranelift `SourceLoc` (see [`crate::core::srcloc`]).
    ///
    /// This is debug information and nothing else: Cranelift treats a
    /// `SourceLoc` as opaque metadata, so it changes neither what is emitted nor
    /// how it is optimized. What it buys is the ability to answer "which guest
    /// instruction is this?" of a line of printed IR, of a byte of compiled
    /// machine code, or of a host `RIP` captured in a crash - the last of which
    /// is otherwise near-impossible in a region whose blocks have been merged,
    /// duplicated and scheduled out of guest order.
    ///
    /// On by default, since the cost is a `u32` per IR instruction at lift time
    /// and a sorted range list per compiled function, neither of which the
    /// generated code pays for. Turn it off for the smallest possible in-memory
    /// footprint when nothing will ever inspect the mapping.
    ///
    /// The lift-time half of the mapping comes back as
    /// [`LiftedCode::source_map`]; the compiled half is
    /// [`GuestCodeMap`](crate::core::srcloc::GuestCodeMap).
    pub emit_source_locs: bool,
}

impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            address_mode: AddressMode::ImageBaseRelative,
            syscall_handler_call_conv: CallConv::SystemV,
            segments: Vec::new(),
            interpreter_fallback: false,
            emit_resume_blocks: false,
            analyze_jump_tables: false,
            imports: HashMap::new(),
            import_registry: None,
            indirect_call_bridge: None,
            use_vector_xmm: false,
            atomic128: Atomic128Mode::default(),
            analysis_mode: false,
            outline_helpers: true,
            emit_source_locs: true,
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
    pub jump_tables: HashMap<u64, Vec<u64>>,
}

/// A helper under construction: its body plus the `UserExternalName` slot whose
/// index [`LiftEngine::finish`] rewrites once the import count is final.
struct HelperEntry {
    name: &'static str,
    func: Function,
    name_ref: UserExternalNameRef,
}

/// Namespace-0 index helpers are declared with before [`LiftEngine::finish`]
/// renumbers them. Imports are assigned indices densely from 0 as they are
/// discovered, and a helper's final index depends on how many imports the rest
/// of the lift turns up - so helpers park up here, far above any plausible
/// import count, where they cannot collide with a real index (which would make
/// `declare_imported_user_function` de-duplicate two distinct callees onto one
/// name).
const PROVISIONAL_HELPER_INDEX: u32 = 0x8000_0000;

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
            helpers: Vec::new(),
            helper_refs: HashMap::new(),
            source_map: SourceLocMap::default(),
            jump_tables: HashMap::new(),
        }
    }

    /// Attribute the IR emitted from here on to guest instruction `va`.
    ///
    /// The builder holds one current `SourceLoc` and stamps it onto every
    /// instruction it creates, so this is a mode switch, not a per-instruction
    /// call: the emission loop sets it once before handing an instruction to
    /// [`GuestArch::lift_instruction`], and everything that handler emits -
    /// however many IR instructions, across however many blocks, including
    /// anything Cranelift's SSA construction inserts on its behalf - inherits it.
    ///
    /// A no-op when [`LifterConfig::emit_source_locs`] is off, and a
    /// [`clear_source_loc`](Self::clear_source_loc) when `va` is too far from the
    /// region to encode.
    pub fn set_source_loc(&mut self, va: u64) {
        if !self.config.emit_source_locs {
            return;
        }
        let loc = self.source_map.encode(va);
        self.builder.set_srcloc(loc);
    }

    /// Stop attributing emitted IR to any guest instruction.
    ///
    /// Used for the region's own scaffolding - the entry prologue, the PC
    /// dispatch switch - which implements no guest instruction, and where a
    /// stale location would be actively misleading.
    pub fn clear_source_loc(&mut self) {
        self.builder.set_srcloc(SourceLoc::default());
    }

    /// Declare (or reuse) an outlined helper function named `name`, returning a
    /// `FuncRef` the caller can `call`.
    ///
    /// The first call for a given `name` builds the body: a fresh `Function`
    /// with `signature` is created and handed to `build`, which must fill it
    /// completely (entry block, body, `return`). Later calls with the same name
    /// reuse that one definition and its one `FuncRef` - that reuse is the
    /// whole point, so `name` must identify the body, not the call site.
    ///
    /// The helper is *locally defined*, not imported: it comes back in
    /// [`LiftedCode::helpers`] for the backend to compile. Its namespace-0
    /// index is provisional until [`Self::finish`], because it has to land
    /// after however many imports the rest of the lift discovers.
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
        // `finalize` only consults the frontend config to lay out stack maps,
        // which helper bodies never request; the pointer width is nonetheless
        // the one every backend this crate targets uses.
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
    /// This runs at the end of the lift because the import count is only final
    /// then: [`ImportRegistry`]-supplied imports append to `aot_imports` as
    /// their call sites are lifted, which can happen after a helper has already
    /// been declared.
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

    /// The value the `image_base` variable is defined to at region entry.
    ///
    /// Under [`AddressMode::ImageBaseRelative`] that is the relocation base the
    /// host put in the context. Under [`AddressMode::Direct`] it is the constant
    /// `lift_base`, which makes the identity `image_base + (va - lift_base) == va`
    /// hold: the load disappears, and anything still written in terms of
    /// `image_base` (rather than [`Self::runtime_addr`]) folds to the right
    /// constant instead of silently producing a relocated address.
    pub fn image_base_at_entry(&mut self, ctx_ptr: Value) -> Value {
        match self.config.address_mode {
            AddressMode::ImageBaseRelative => self.builder.ins().load(
                types::I64,
                MemFlagsData::trusted(),
                ctx_ptr,
                A::image_base_offset(),
            ),
            AddressMode::Direct => {
                let base = self.lift_base;
                self.builder.ins().iconst(types::I64, base as i64)
            }
        }
    }

    /// Materialize the run-time address of the lift-time virtual address `va`.
    ///
    /// This is the one place a guest VA becomes a Cranelift value: RIP-relative
    /// operands, PCs reported on a VM exit, pushed return addresses, and
    /// out-of-region callee addresses all go through it. See [`AddressMode`] for
    /// what the two representations mean; the inverse is [`Self::to_lift_va`].
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

    /// Map a run-time address value back into lift-time VA space, so it can be
    /// compared against (or switched on) the addresses the decoder saw.
    ///
    /// The inverse of [`Self::runtime_addr`]: under
    /// [`AddressMode::ImageBaseRelative`] this is
    /// `addr - image_base + lift_base`; under [`AddressMode::Direct`] the
    /// run-time address already *is* the lift-time VA, so it is the identity.
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
        // Guest addresses are encoded relative to the region base, so the map
        // has to be established before any IR is emitted.
        self.source_map = SourceLocMap::new(ip);

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

        // Experimental: recover jump-table successors so an indirect branch
        // contributes leaders instead of dead-ending at a VM exit. Run over the
        // whole region up front - the dispatch sequences it matches are
        // straight-line, so the linear view it works from is enough, and one
        // pass is cheaper than re-matching per branch inside the worklist.
        if self.config.analyze_jump_tables {
            self.jump_tables = A::jump_table_targets(bytes, ip, &self.config.segments);
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
                ControlFlowKind::Ret => {}
                ControlFlowKind::IndirectBranch => {
                    // Normally a dead end for discovery. If the jump-table pass
                    // resolved this branch, its in-region targets are leaders
                    // like any other branch target; out-of-region ones stay the
                    // host's problem and keep the exit path they already had.
                    if let Some(targets) = self.jump_tables.get(&addr) {
                        for &t in targets {
                            if t >= self.lift_base && t < self.lift_end {
                                leaders.insert(t);
                                work.push(t);
                            }
                        }
                    }
                }
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
            let entry_va = self.to_lift_va(pc_val);

            let mut switch = Switch::new();
            for (&offset, &block) in &self.blocks {
                switch.set_entry(offset.into(), block);
            }

            switch.emit(&mut self.builder, entry_va, start_block);
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
                    self.set_source_loc(cur);
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
                    self.set_source_loc(self.current_ip);
                    A::sync_to_context(self, None, ExitReason::InvalidInstruction)?;
                    self.builder.ins().return_(&[]);
                    self.block_terminated = true;
                    break;
                }

                // Advance current_ip before dispatch so a PC read inside the
                // handler sees the post-instruction value. The source location,
                // by contrast, names the instruction *being* lifted, so it is
                // the pre-advance address.
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
                // The stub implements the transfer *to* `offset`, which is the
                // only guest address it can honestly be attributed to (the
                // instruction it belongs to may be any of several branches
                // targeting it).
                self.set_source_loc(offset);
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

        self.clear_source_loc();
        self.builder.seal_all_blocks();
        self.finalize_helper_indices();
        Ok(())
    }
}
