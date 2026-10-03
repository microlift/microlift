//! Shared integration-test / fuzz / bench harness.
//! This module centralizes that pattern behind a small builder API. It is a
//! thin wrapper over the library's own [`lift_to_code`] entry point plus
//! [`CraneliftBackend`] - it does not re-implement lifting.
//!
//! ```no_run
//! use microlift::testkit::prelude::*;
//!
//! // mov rax, 42 ; ret
//! let bytes = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];
//! let out = LiftTest::new(&bytes).run();
//! assert_eq!(out.ctx.rax, 42);
//! ```
//!
//! Enabled by the `testkit` cargo feature so test-only helpers never ship in a
//! production build.

use crate::arch::x86_64::X86_64;
use crate::backend::CraneliftBackend;
use crate::context::{ExitReason, MemorySegment, RegisterContext};
use crate::core::optimize::Optimizations;
use crate::core::specialize::Specialization;
use crate::core::srcloc::GuestCodeMap;
use crate::core::{
    AddressMode, Atomic128Mode, ExecutionMode, GeneratedNan, LifterConfig, VmResumptionPoints,
    lift_to_code,
};
use cranelift_codegen::settings::{self, Configurable};

pub mod oracle;
pub mod rng;
pub mod trampoline;

pub const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

pub const DEFAULT_LIFT_BASE: u64 = 0x1000;

pub const DEFAULT_STACK_WORDS: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Opt {
    #[default]
    None,
    Speed,
    SpeedAndSize,
}

impl Opt {
    fn as_str(self) -> &'static str {
        match self {
            Opt::None => "none",
            Opt::Speed => "speed",
            Opt::SpeedAndSize => "speed_and_size",
        }
    }

    /// Assemble the `settings::Flags` the test backend runs with. Always sets
    /// `use_colocated_libcalls=false` and `is_pic=false` (as every prior
    /// harness intended to) plus the selected `opt_level`.
    pub fn flags(self) -> settings::Flags {
        let mut b = settings::builder();
        b.set("use_colocated_libcalls", "false").unwrap();
        b.set("is_pic", "false").unwrap();
        b.set("opt_level", self.as_str()).unwrap();
        settings::Flags::new(b)
    }
}

/// What to write at `[rsp]` before entering the lifted code.
#[derive(Clone, Copy, Debug)]
pub enum RetAddr {
    /// Push [`EXIT_TRAMPOLINE`] (the default) so the first `RET` exits cleanly.
    Trampoline,
    /// Push a caller-chosen sentinel.
    Sentinel(u64),
    /// Push nothing - the caller has arranged the stack itself.
    None,
}

/// Error from the lift-and-compile pipeline, tagged with the stage that failed.
#[derive(Debug)]
pub enum TestError {
    /// Lifting the guest bytes to IR failed.
    Lift(crate::MicroliftError),
    /// Compiling the lifted IR failed.
    Compile(crate::MicroliftError),
}

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestError::Lift(e) => write!(f, "lift failed: {e}"),
            TestError::Compile(e) => write!(f, "compile failed: {e}"),
        }
    }
}

impl std::error::Error for TestError {}

/// Builder describing a single lift-compile-run. Nothing is compiled until a
/// terminal ([`run`](Self::run) / [`try_run`](Self::try_run) /
/// [`compile`](Self::compile)) is called.
pub struct LiftTest<'a> {
    bytes: &'a [u8],
    ip: u64,
    image_base: u64,
    config: LifterConfig,
    opt: Opt,
    stack_words: usize,
    ret_addr: RetAddr,
    init: RegisterContext,
    /// When the `unicorn-oracle` feature is on, `run`/`try_run` automatically
    /// re-run the same program on Unicorn and assert the two engines agree
    /// (see [`LiftTest::no_unicorn`]). Default `true`.
    #[cfg(feature = "unicorn-oracle")]
    unicorn_check: bool,
}

impl<'a> LiftTest<'a> {
    /// Start describing a test that lifts `bytes` at [`DEFAULT_LIFT_BASE`].
    pub fn new(bytes: &'a [u8]) -> Self {
        // The harness's whole exit protocol is the sentinel at `[rsp]`, so the
        // test config declares it even though the library default is `None`.
        let config = LifterConfig {
            exit_trampoline_address: Some(EXIT_TRAMPOLINE),
            ..LifterConfig::default()
        };
        Self {
            bytes,
            ip: DEFAULT_LIFT_BASE,
            image_base: DEFAULT_LIFT_BASE,
            config,
            opt: Opt::None,
            stack_words: DEFAULT_STACK_WORDS,
            ret_addr: RetAddr::Trampoline,
            init: RegisterContext::default(),
            #[cfg(feature = "unicorn-oracle")]
            unicorn_check: true,
        }
    }

    /// Set the virtual address the first byte maps to (the lift base).
    pub fn at(mut self, ip: u64) -> Self {
        self.ip = ip;
        self
    }

    /// Set the runtime `image_base` used for ASLR reconciliation. Only applied
    /// when the seeded context leaves `image_base == 0` (the default), so an
    /// explicit relocation base set via [`ctx`](Self::ctx)/[`reg`](Self::reg)
    /// wins.
    pub fn image_base(mut self, base: u64) -> Self {
        self.image_base = base;
        self
    }

    /// Replace the whole [`LifterConfig`].
    ///
    /// This also drops the harness's default
    /// [`LifterConfig::exit_trampoline_address`], since a supplied config is taken
    /// as-is - set it there, or call [`exit_trampoline`](Self::exit_trampoline)
    /// afterwards, for a test whose program ends in a `RET` to the sentinel.
    pub fn config(mut self, config: LifterConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the sentinel return address that makes a `RET` exit with
    /// [`ExitReason::Return`] (`config.exit_trampoline_address`). Defaults to
    /// `Some(EXIT_TRAMPOLINE)`; pass `None` to lift the way the library does by
    /// default, where such a `RET` is an `UnknownJump` instead.
    pub fn exit_trampoline(mut self, addr: Option<u64>) -> Self {
        self.config.exit_trampoline_address = addr;
        self
    }

    /// Mutate the [`LifterConfig`] in place.
    pub fn with_config(mut self, f: impl FnOnce(&mut LifterConfig)) -> Self {
        f(&mut self.config);
        self
    }

    /// Select the backend optimization level.
    pub fn opt(mut self, opt: Opt) -> Self {
        self.opt = opt;
        self
    }

    /// Toggle native `I64X2` XMM variables (`config.use_vector_xmm`).
    pub fn vector_xmm(mut self, on: bool) -> Self {
        self.config.use_vector_xmm = on;
        self
    }

    /// Enable target-directed instruction specialization.
    ///
    /// This does **not** change what the JIT compiles for - always the host - so a
    /// `.specialize(Specialization::for_target(Aarch64))` run on an x86 host
    /// produces IR the x86 backend still has to lower. That is what the equivalence
    /// suite wants, but it is not a performance measurement.
    pub fn specialize(mut self, spec: Specialization) -> Self {
        self.config.specialization = spec;
        self
    }

    /// Select how guest virtual addresses are materialized
    /// (`config.address_mode`). Under [`AddressMode::Direct`] the seeded
    /// `image_base` is ignored by the lifted code, and the PC in the exit state
    /// is a plain lift-time VA.
    pub fn address_mode(mut self, mode: AddressMode) -> Self {
        self.config.address_mode = mode;
        self
    }

    /// Select how a 128-bit guest CAS (`CMPXCHG16B`) is lowered
    /// (`config.atomic128`). [`Atomic128Mode::Native`] only compiles on an x64
    /// host with `has_cmpxchg16b`.
    pub fn atomic128(mut self, mode: Atomic128Mode) -> Self {
        self.config.atomic128 = mode;
        self
    }

    /// Select what bit pattern an SSE instruction's *generated* NaN gets
    /// (`config.generated_nan`). [`GeneratedNan::HostDefault`] diverges from
    /// hardware only on a host whose default NaN is not x86's floating
    /// indefinite, so a test asserting on it has to say which host it means.
    pub fn generated_nan(mut self, mode: GeneratedNan) -> Self {
        self.config.generated_nan = mode;
        self
    }

    /// Select which accuracy-trading [`Optimization`](crate::core::optimize::Optimization)s
    /// are enabled (`config.optimizations`). Defaults to
    /// [`Optimizations::none`] - nothing enabled.
    pub fn optimizations(mut self, optimizations: Optimizations) -> Self {
        self.config.optimizations = optimizations;
        self
    }

    /// Set the execution mode (`Jit` vs `RuntimeAssisted`).
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.config.execution_mode = mode;
        self
    }

    /// Configure the lift for resumable VM exits: `RuntimeAssisted` execution with a
    /// resume point at every instruction boundary ([`VmResumptionPoints::All`]) and
    /// interpreter fallback. Use with [`RunOutcome::resume_with`].
    ///
    /// `All` rather than the default so a test can resume at any address without
    /// declaring it first.
    pub fn resumable(mut self) -> Self {
        self.config.execution_mode = ExecutionMode::RuntimeAssisted;
        self.config.resumption = VmResumptionPoints::All;
        self.config.interpreter_fallback = true;
        self
    }

    /// Set the resumption mode ([`LifterConfig::resumption`]).
    pub fn resumption(mut self, mode: VmResumptionPoints) -> Self {
        self.config.resumption = mode;
        self
    }

    /// Provide static memory segments (rodata hints etc.) to the lifter.
    pub fn segments(mut self, segments: Vec<MemorySegment>) -> Self {
        self.config.set_segments(segments);
        self
    }

    /// Enable the experimental jump-table recovery pass
    /// ([`LifterConfig::analyze_jump_tables`]).
    pub fn analyze_jump_tables(mut self) -> Self {
        self.config.analyze_jump_tables = true;
        self
    }

    /// Skip the IR for condition-flag writes nothing reads
    /// ([`LifterConfig::eliminate_dead_flags`]).
    ///
    /// A metamorphic switch: turning it on may not change a single bit of
    /// architectural state, which is what `tests/dead_flag_tests.rs` and the
    /// cross-config fuzz suite check.
    pub fn eliminate_dead_flags(mut self) -> Self {
        self.config.eliminate_dead_flags = true;
        self
    }

    /// Set the guest stack size in 64-bit words.
    pub fn stack_words(mut self, n: usize) -> Self {
        self.stack_words = n;
        self
    }

    /// Choose what is written at `[rsp]` before entry.
    pub fn ret_addr(mut self, ret: RetAddr) -> Self {
        self.ret_addr = ret;
        self
    }

    /// Opt this test out of the automatic Unicorn differential cross-check that
    /// `run`/`try_run` perform under the `unicorn-oracle` feature. Use it for
    /// programs whose architectural result legitimately cannot be diffed
    /// against Unicorn - e.g. state Microlift models but [`oracle::CpuState`]
    /// does not (the x87 stack), or host-specific results. A no-op when the
    /// feature is off, so it is always safe to call.
    #[cfg(feature = "unicorn-oracle")]
    pub fn no_unicorn(mut self) -> Self {
        self.unicorn_check = false;
        self
    }

    /// No-op stand-in for [`no_unicorn`](Self::no_unicorn) when the
    /// `unicorn-oracle` feature is disabled, so tests can call it
    /// unconditionally.
    #[cfg(not(feature = "unicorn-oracle"))]
    pub fn no_unicorn(self) -> Self {
        self
    }

    /// Seed the entire initial register context at once (eases migration from
    /// the old `run_test(bytes, ctx)` helpers).
    pub fn ctx(mut self, ctx: RegisterContext) -> Self {
        self.init = ctx;
        self
    }

    /// Mutate the initial register context (e.g. seed `rax`, set a flag).
    pub fn reg(mut self, f: impl FnOnce(&mut RegisterContext)) -> Self {
        f(&mut self.init);
        self
    }

    /// Whether the run relocates the region: the lifted code forms guest addresses
    /// as `image_base + (va - lift_base)` with an `image_base` that is not the lift
    /// base, so a lift-time VA and the address actually touched differ.
    ///
    /// That is what makes such a run non-diffable against Unicorn, which has no
    /// relocation step. Under [`AddressMode::Direct`] the lifted code ignores
    /// `image_base` entirely, so there is no relocation to speak of.
    #[cfg(feature = "unicorn-oracle")]
    fn image_is_relocated(&self) -> bool {
        // Mirrors `compile`: an `image_base` seeded through `ctx`/`reg` wins,
        // and only a zero there falls back to the builder's value.
        let effective = if self.init.image_base == 0 {
            self.image_base
        } else {
            self.init.image_base
        };
        self.config.address_mode == AddressMode::ImageBaseRelative && effective != self.ip
    }

    /// Lift and compile, but do not run. Returns a ready-to-enter [`Harness`].
    /// Lift and compile into a [`Translation`] for the
    /// [`runtime`](crate::runtime) module.
    ///
    /// The counterpart to [`compile`](Self::compile): that one produces a
    /// `Harness`, which owns a guest stack and is entered directly, while this
    /// produces something a [`CodeCache`](crate::runtime::CodeCache) can hold
    /// and a [`ThreadRuntime`](crate::runtime::ThreadRuntime) can dispatch to.
    /// The stack is the runtime's to provide.
    #[cfg(feature = "runtime")]
    pub fn translation(self) -> Result<crate::runtime::Translation, TestError> {
        let image_base = self.image_base;
        let flags = self.opt.flags();
        let code =
            lift_to_code::<X86_64>(self.bytes, self.ip, self.config).map_err(TestError::Lift)?;
        let backend = CraneliftBackend::with_flags(flags, &[]).map_err(TestError::Compile)?;
        crate::runtime::Translation::compile_with(code, image_base, backend)
            .map_err(TestError::Compile)
    }

    pub fn compile(self) -> Result<Harness, TestError> {
        let code =
            lift_to_code::<X86_64>(self.bytes, self.ip, self.config).map_err(TestError::Lift)?;

        let mut backend =
            CraneliftBackend::with_flags(self.opt.flags(), &[]).map_err(TestError::Compile)?;
        let (region, code_map) = backend
            .compile_lifted_region(code)
            .map_err(TestError::Compile)?;
        let code_ptr = region.entry;

        // The guest stack is a heap `Vec`; its backing allocation address is
        // stable across moves (only the handle moves), so `rsp` remains valid
        // once the `Harness` owns it - including across resumes.
        let stack = vec![0u64; self.stack_words];
        let stack_top = stack.as_ptr() as u64 + (self.stack_words as u64) * 8;

        let mut ctx = self.init;
        if ctx.rsp == 0 {
            ctx.rsp = stack_top - 8;
        }
        if ctx.image_base == 0 {
            ctx.image_base = self.image_base;
        }
        match self.ret_addr {
            RetAddr::Trampoline => unsafe { *(ctx.rsp as *mut u64) = EXIT_TRAMPOLINE },
            RetAddr::Sentinel(v) => unsafe { *(ctx.rsp as *mut u64) = v },
            RetAddr::None => {}
        }

        let initial_rsp = ctx.rsp;

        // `get_host_calling_conv()` returns the platform C ABI (SystemV /
        // WindowsFastcall / AppleAarch64), all of which Rust's `extern "C"`
        // matches for a single pointer argument.
        let entry: unsafe extern "C" fn(*mut RegisterContext) =
            unsafe { std::mem::transmute(code_ptr) };

        Ok(Harness {
            ctx,
            initial_rsp,
            code_ptr,
            link_target: region.link_target,
            code_map,
            _stack: stack,
            _backend: backend,
            entry,
        })
    }

    /// Lift, compile, and run once. Panics if lifting or compilation fails -
    /// convenient for tests where those are never expected to fail.
    pub fn run(self) -> RunOutcome {
        self.try_run().expect("testkit: lift/compile failed")
    }

    /// Like [`run`](Self::run), but returns lift/compile errors instead of
    /// panicking.
    pub fn try_run(self) -> Result<RunOutcome, TestError> {
        // Capture everything the Unicorn cross-check needs *before* `compile`
        // consumes `self`. Only when the feature is on, the check is enabled,
        // and the config is a plain executable JIT (Unicorn models real
        // execution - analysis IR and RuntimeAssisted resume flows are not
        // comparable).
        #[cfg(feature = "unicorn-oracle")]
        let xcheck = (self.unicorn_check
            && !self.config.analysis_mode
            && self.config.execution_mode == ExecutionMode::Jit
            && !self.image_is_relocated())
        .then(|| {
            (
                self.bytes.to_vec(),
                self.ip,
                oracle::CpuState::from_ctx(&self.init),
            )
        });

        let mut harness = self.compile()?;
        let exit = harness.enter();
        let outcome = RunOutcome {
            ctx: harness.ctx,
            initial_rsp: harness.initial_rsp,
            exit,
            harness,
        };

        // The harness (and thus the shared guest stack) is alive inside
        // `outcome`, so Unicorn can be pointed at the exact same stack pages.
        #[cfg(feature = "unicorn-oracle")]
        if let Some((bytes, ip, init)) = xcheck {
            cross_check_unicorn(&bytes, ip, &init, &outcome);
        }

        Ok(outcome)
    }
}

/// Re-run a just-executed program on Unicorn from the same initial state and
/// assert the two engines' architectural state agrees.
///
/// Soundness guards, all derived from a static decode so no test needs
/// hand-annotation:
///
/// - **Unrelocated image only.** Unicorn executes at the lift-time VAs, so a
///   relocated run touches different memory for the same guest address. Checked
///   by the caller ([`LiftTest::image_is_relocated`]), since it is a property of
///   the configuration. A *hard* guard, not an accuracy one: Unicorn's
///   `MEM_UNMAPPED` hook identity-maps a guest page onto the same *host*
///   address, so a read of an unrelocated lift-time VA faults the host process.
/// - **Clean halt only.** Any VM exit is non-diffable and skipped, as is a
///   Unicorn fault.
/// - **Shared stack.** The harness's `initial_rsp` is handed to Unicorn, which
///   identity-maps those pages, so `rsp`-derived values and the pushed exit
///   trampoline line up. Both runs re-push the same bytes, so the stack is
///   self-healing and stack accesses stay diffable.
/// - **No persistent data-memory writes.** Microlift runs first and mutates the
///   shared memory, so a program writing to non-stack memory would feed Unicorn
///   already-mutated inputs. Stack writes are exempt.
/// - **No syscalls.** Unicorn does not run Microlift's host syscall callback.
/// - **x86-undefined flags masked.** `MUL`/`IMUL` (SF/ZF) and variable-count
///   shifts/rotates/`BT*` (OF) leave flags architecturally undefined, so the two
///   engines may pick different bits without either being wrong.
#[cfg(feature = "unicorn-oracle")]
fn cross_check_unicorn(bytes: &[u8], ip: u64, init: &oracle::CpuState, outcome: &RunOutcome) {
    use oracle::{CpuOracle, CpuState, DiffOptions, OracleOutcome, diff_states};

    // Only a clean `RET` into the exit trampoline is diffable. `ExitReason::None`
    // means Microlift stopped at the *region boundary*; Unicorn has no region
    // concept and would keep executing whatever follows, where the zero-padding
    // decodes as `add [rax], al` and the identity hook maps an arbitrary guest
    // address to that *host* address, faulting the process.
    if outcome.exit != ExitReason::Return {
        return;
    }

    let ProgramFacts {
        undefined_rflags,
        writes_data_mem,
        has_syscall,
    } = analyze_program(bytes, ip);

    if has_syscall {
        return;
    }

    // A program that persists writes to non-stack memory cannot be diffed by
    // sequential re-execution on shared host memory (Microlift already ran and
    // mutated it). Skip rather than report a spurious divergence.
    if writes_data_mem {
        return;
    }

    // Share Microlift's stack so the two engines see identical stack memory and
    // stack-pointer values (rsp itself stays out of the diff by default).
    let mut uinit = *init;
    uinit.gpr[oracle::gpr::RSP] = outcome.initial_rsp;

    let mut uc = oracle::unicorn::UnicornOracle::new();
    let out = uc.run(bytes, ip, &uinit, &[], 1 << 20);
    let OracleOutcome::Halted(ustate) = out else {
        return;
    };

    // Drop architecturally-undefined flag bits before comparing.
    let keep = !undefined_rflags;
    let mut ml = CpuState::from_ctx(&outcome.ctx);
    let mut uc_state = ustate;
    ml.rflags &= keep;
    uc_state.rflags &= keep;

    let diff = diff_states(&ml, &uc_state, DiffOptions::default());
    assert!(
        diff.is_empty(),
        "unicorn differential mismatch at ip={ip:#x}\n  code = {bytes:02x?}\n  \
         diff (microlift vs unicorn) = {diff:?}",
    );
}

/// Static facts about a lifted program used to keep the Unicorn cross-check
/// sound (see [`cross_check_unicorn`]).
#[cfg(feature = "unicorn-oracle")]
struct ProgramFacts {
    /// Packed-EFLAGS mask of flags left architecturally undefined at program
    /// exit (last-writer-wins), to exclude from the state diff.
    undefined_rflags: u64,
    /// Whether any instruction writes to memory that is not `RSP`-relative.
    writes_data_mem: bool,
    has_syscall: bool,
}

/// Decode `bytes` (linear sweep from `ip`) and derive the [`ProgramFacts`] the
/// cross-check needs. Decoding stops at the first invalid byte; that is fine as
/// the facts only need to cover the instructions that actually executed.
#[cfg(feature = "unicorn-oracle")]
fn analyze_program(bytes: &[u8], ip: u64) -> ProgramFacts {
    use iced_x86::{Decoder, DecoderOptions, InstructionInfoFactory, Mnemonic, OpAccess, Register};

    // iced's `RflagsBits` uses its own bit layout; translate to the packed
    // EFLAGS positions `CpuState::rflags` uses.
    fn iced_to_eflags(bits: u32) -> u64 {
        use iced_x86::RflagsBits as R;
        let mut out = 0u64;
        if bits & R::CF != 0 {
            out |= 1 << 0;
        }
        if bits & R::PF != 0 {
            out |= 1 << 2;
        }
        if bits & R::AF != 0 {
            out |= 1 << 4;
        }
        if bits & R::ZF != 0 {
            out |= 1 << 6;
        }
        if bits & R::SF != 0 {
            out |= 1 << 7;
        }
        if bits & R::DF != 0 {
            out |= 1 << 10;
        }
        if bits & R::OF != 0 {
            out |= 1 << 11;
        }
        out
    }

    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let mut factory = InstructionInfoFactory::new();
    let mut undefined = 0u32;
    let mut writes_data_mem = false;
    let mut has_syscall = false;

    while decoder.can_decode() {
        let insn = decoder.decode();
        if insn.is_invalid() {
            break;
        }
        has_syscall |= insn.mnemonic() == Mnemonic::Syscall;

        // Last-writer-wins: flags this instruction gives a defined value become
        // defined again; flags it leaves undefined become undefined.
        undefined = (undefined & !insn.rflags_modified()) | insn.rflags_undefined();

        let info = factory.info(&insn);
        for m in info.used_memory() {
            let writes = matches!(
                m.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            );
            // Stack writes self-heal across the two runs; only non-`RSP` writes
            // corrupt the second engine's inputs.
            if writes && m.base() != Register::RSP {
                writes_data_mem = true;
            }
        }
    }

    ProgramFacts {
        undefined_rflags: iced_to_eflags(undefined),
        writes_data_mem,
        has_syscall,
    }
}

/// A compiled, runnable lifted function plus the host-backed guest stack and
/// executable memory that must outlive every entry. Dropping it frees both.
pub struct Harness {
    /// The live guest register context. Mutated in place on every entry.
    pub ctx: RegisterContext,
    /// The value of `rsp` at the point the lifted function is first entered
    /// (useful for stack-balance / alignment assertions).
    pub initial_rsp: u64,
    /// Host address the lifted function was compiled to. Base for
    /// [`GuestCodeMap::lookup_host_addr`].
    pub code_ptr: *const u8,
    /// The address a block-link row must name to reach this region -
    /// [`code_ptr`](Self::code_ptr) itself, except under a pinned ABI, where a
    /// link enters the tail-convention body in front of which `code_ptr` is a
    /// thunk. See [`CompiledRegion`](crate::backend::CompiledRegion).
    pub link_target: *const u8,
    /// Which guest instruction each byte of the compiled region came from; see
    /// [`crate::core::srcloc`]. Empty if the test disabled
    /// [`LifterConfig::emit_source_locs`].
    pub code_map: GuestCodeMap,
    // Keeps the guest stack allocation alive for the duration of execution.
    _stack: Vec<u64>,
    // Keeps the JIT module (and its executable pages) mapped.
    _backend: CraneliftBackend,
    entry: unsafe extern "C" fn(*mut RegisterContext),
}

impl Harness {
    /// Enter (or re-enter) the lifted code, returning the decoded exit reason.
    pub fn enter(&mut self) -> ExitReason {
        unsafe {
            (self.entry)(&mut self.ctx as *mut RegisterContext);
        }
        ExitReason::try_from(self.ctx.exit_reason).unwrap_or(ExitReason::None)
    }
}

/// The result of a single (re-)entry into a [`Harness`].
pub struct RunOutcome {
    /// Snapshot of the register context after the entry returned.
    ///
    /// `ctx.rsp` borrows from this outcome: it is a raw host address into the guest
    /// stack owned by the retained [`Harness`], so copying `ctx` out and dropping
    /// the outcome leaves `rsp` dangling, and feeding such a `ctx` back into
    /// [`LiftTest::ctx`] is a use-after-free. Keep the `RunOutcome` alive as long as
    /// the `rsp` is used, or clear `rsp` to `0`.
    pub ctx: RegisterContext,
    /// `rsp` at the point the function was first entered (see
    /// [`Harness::initial_rsp`]).
    pub initial_rsp: u64,
    /// Why the region exited, decoded from `ctx.exit_reason`.
    pub exit: ExitReason,
    harness: Harness,
}

impl RunOutcome {
    /// Inspect the final context (chainable).
    pub fn reg(&self, f: impl FnOnce(&RegisterContext)) -> &Self {
        f(&self.ctx);
        self
    }

    /// Mutate the retained context (e.g. advance `rip` past a trapping
    /// instruction, or fill `rax`/`rdx` for a `Cpuid` resume) and re-enter the
    /// same compiled function.
    pub fn resume_with(mut self, f: impl FnOnce(&mut RegisterContext)) -> RunOutcome {
        f(&mut self.harness.ctx);
        let exit = self.harness.enter();
        RunOutcome {
            ctx: self.harness.ctx,
            initial_rsp: self.harness.initial_rsp,
            exit,
            harness: self.harness,
        }
    }

    /// Re-enter without mutating the context.
    pub fn resume(self) -> RunOutcome {
        self.resume_with(|_| {})
    }

    /// Which guest instruction each byte of the compiled region came from - the
    /// map built from the IR's source locations. See [`crate::core::srcloc`].
    pub fn code_map(&self) -> &GuestCodeMap {
        &self.harness.code_map
    }

    /// Host address the compiled region starts at, for
    /// [`GuestCodeMap::lookup_host_addr`].
    pub fn code_ptr(&self) -> *const u8 {
        self.harness.code_ptr
    }
}

/// Differential-oracle wiring: run the described program on both the Microlift
/// JIT and the Unicorn engine from an identical initial state and compare.
/// Available only under the `unicorn-oracle` feature.
#[cfg(feature = "unicorn-oracle")]
impl<'a> LiftTest<'a> {
    /// Run this program on the Microlift JIT and on Unicorn from the same seeded
    /// state, returning their architectural [`StateDiff`]. `None` means one engine
    /// hit a non-diffable event, so there is nothing sound to compare.
    ///
    /// Only the register/flag/xmm seed and the lift base are carried over, not the
    /// full [`LifterConfig`]. Feed it code from the diffable subset only.
    pub fn differential_unicorn(&self, opts: oracle::DiffOptions) -> Option<oracle::StateDiff> {
        let state = oracle::CpuState::from_ctx(&self.init);
        oracle::unicorn::differential_unicorn(self.bytes, self.ip, &state, &[], 1 << 20, opts)
    }

    /// Like [`differential_unicorn`](Self::differential_unicorn) but asserts the
    /// two engines agree, panicking with the offending diff otherwise. A
    /// non-diffable outcome (returns `None`) is treated as "nothing to check"
    /// and passes. This is a one-line hook tests use to gain Unicorn
    /// cross-checking on top of their existing Microlift assertions.
    pub fn assert_matches_unicorn(&self, opts: oracle::DiffOptions) {
        if let Some(diff) = self.differential_unicorn(opts) {
            assert!(diff.is_empty(), "microlift and unicorn disagree: {diff:?}");
        }
    }
}

/// Convenience for the common case: lift `bytes` at [`DEFAULT_LIFT_BASE`] with
/// the default config and `Opt::None`, run against `ctx`, and return the
/// mutated context. Mirrors the old per-file `run_test(bytes, ctx)` helpers so
/// migration is a one-line import.
pub fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

/// Re-exports for `use microlift::testkit::prelude::*;`.
pub mod prelude {
    pub use super::{Harness, LiftTest, Opt, RetAddr, RunOutcome, TestError, run_test};
    pub use crate::context::{ExitReason, RegisterContext};
    pub use crate::core::specialize::{Spec, SpecTarget, Specialization};
    pub use crate::core::{AddressMode, Atomic128Mode, GeneratedNan, VmResumptionPoints};
}
