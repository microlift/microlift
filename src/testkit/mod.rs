//! Shared integration-test / fuzz / bench harness: a builder over [`lift_to_code`] + [`CraneliftBackend`].
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
//! Behind the `testkit` feature.

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

    /// Backend flags: `use_colocated_libcalls=false`, `is_pic=false`, plus `opt_level`.
    pub fn flags(self) -> settings::Flags {
        let mut b = settings::builder();
        b.set("use_colocated_libcalls", "false").unwrap();
        b.set("is_pic", "false").unwrap();
        b.set("opt_level", self.as_str()).unwrap();
        settings::Flags::new(b)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RetAddr {
    /// Push [`EXIT_TRAMPOLINE`] (default) so the first `RET` exits cleanly.
    Trampoline,
    Sentinel(u64),
    /// Push nothing; the caller arranged the stack.
    None,
}

#[derive(Debug)]
pub enum TestError {
    Lift(crate::MicroliftError),
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

/// Describes one lift-compile-run; nothing compiles until a terminal ([`run`](Self::run) /
/// [`try_run`](Self::try_run) / [`compile`](Self::compile)) is called.
pub struct LiftTest<'a> {
    bytes: &'a [u8],
    ip: u64,
    image_base: u64,
    config: LifterConfig,
    opt: Opt,
    stack_words: usize,
    ret_addr: RetAddr,
    init: RegisterContext,
    /// Under `unicorn-oracle`, `run` re-runs on Unicorn and asserts agreement (see [`LiftTest::no_unicorn`]). Default `true`.
    #[cfg(feature = "unicorn-oracle")]
    unicorn_check: bool,
}

impl<'a> LiftTest<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        // The exit protocol is the sentinel at `[rsp]`, so declare it despite the library default `None`.
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

    pub fn at(mut self, ip: u64) -> Self {
        self.ip = ip;
        self
    }

    /// Runtime `image_base`; applied only when the seeded context leaves it 0.
    pub fn image_base(mut self, base: u64) -> Self {
        self.image_base = base;
        self
    }

    /// Replaces the whole [`LifterConfig`], dropping the harness's default
    /// `exit_trampoline_address`; set it again (or call [`exit_trampoline`](Self::exit_trampoline)) if the program ends in `RET`.
    pub fn config(mut self, config: LifterConfig) -> Self {
        self.config = config;
        self
    }

    /// Sentinel return address that makes `RET` exit with [`ExitReason::Return`]; default
    /// `Some(EXIT_TRAMPOLINE)`, `None` gives the library behaviour (`UnknownJump`).
    pub fn exit_trampoline(mut self, addr: Option<u64>) -> Self {
        self.config.exit_trampoline_address = addr;
        self
    }

    pub fn with_config(mut self, f: impl FnOnce(&mut LifterConfig)) -> Self {
        f(&mut self.config);
        self
    }

    pub fn opt(mut self, opt: Opt) -> Self {
        self.opt = opt;
        self
    }

    pub fn vector_xmm(mut self, on: bool) -> Self {
        self.config.use_vector_xmm = on;
        self
    }

    /// Enables target specialization. The JIT still compiles for the host, so a foreign
    /// target yields IR the host must lower: for equivalence tests, not performance.
    pub fn specialize(mut self, spec: Specialization) -> Self {
        self.config.specialization = spec;
        self
    }

    /// `config.address_mode`; under [`AddressMode::Direct`] the seeded `image_base` is ignored.
    pub fn address_mode(mut self, mode: AddressMode) -> Self {
        self.config.address_mode = mode;
        self
    }

    /// `config.atomic128`; [`Atomic128Mode::Native`] needs an x64 host with `has_cmpxchg16b`.
    pub fn atomic128(mut self, mode: Atomic128Mode) -> Self {
        self.config.atomic128 = mode;
        self
    }

    /// `config.generated_nan`; [`GeneratedNan::HostDefault`] diverges from hardware only on hosts whose default NaN is not x86's.
    pub fn generated_nan(mut self, mode: GeneratedNan) -> Self {
        self.config.generated_nan = mode;
        self
    }

    /// `config.optimizations`; default [`Optimizations::none`].
    pub fn optimizations(mut self, optimizations: Optimizations) -> Self {
        self.config.optimizations = optimizations;
        self
    }

    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.config.execution_mode = mode;
        self
    }

    /// Resumable VM exits: `RuntimeAssisted` with a resume point at every instruction
    /// ([`VmResumptionPoints::All`]) and interpreter fallback; use with [`RunOutcome::resume_with`].
    pub fn resumable(mut self) -> Self {
        self.config.execution_mode = ExecutionMode::RuntimeAssisted;
        self.config.resumption = VmResumptionPoints::All;
        self.config.interpreter_fallback = true;
        self
    }

    pub fn resumption(mut self, mode: VmResumptionPoints) -> Self {
        self.config.resumption = mode;
        self
    }

    pub fn segments(mut self, segments: Vec<MemorySegment>) -> Self {
        self.config.set_segments(segments);
        self
    }

    /// Experimental jump-table recovery ([`LifterConfig::analyze_jump_tables`]).
    pub fn analyze_jump_tables(mut self) -> Self {
        self.config.analyze_jump_tables = true;
        self
    }

    /// [`LifterConfig::eliminate_dead_flags`]; must not change any architectural state.
    pub fn eliminate_dead_flags(mut self) -> Self {
        self.config.eliminate_dead_flags = true;
        self
    }

    pub fn stack_words(mut self, n: usize) -> Self {
        self.stack_words = n;
        self
    }

    pub fn ret_addr(mut self, ret: RetAddr) -> Self {
        self.ret_addr = ret;
        self
    }

    /// Opts out of the automatic Unicorn cross-check, for results it cannot diff (x87 stack,
    /// host-specific values). No-op without `unicorn-oracle`.
    #[cfg(feature = "unicorn-oracle")]
    pub fn no_unicorn(mut self) -> Self {
        self.unicorn_check = false;
        self
    }

    /// No-op stand-in for [`no_unicorn`](Self::no_unicorn) without `unicorn-oracle`.
    #[cfg(not(feature = "unicorn-oracle"))]
    pub fn no_unicorn(self) -> Self {
        self
    }

    /// Seeds the whole initial register context.
    pub fn ctx(mut self, ctx: RegisterContext) -> Self {
        self.init = ctx;
        self
    }

    pub fn reg(mut self, f: impl FnOnce(&mut RegisterContext)) -> Self {
        f(&mut self.init);
        self
    }

    /// Whether lifted code forms addresses as `image_base + (va - lift_base)` with `image_base`
    /// not the lift base; such runs are non-diffable against Unicorn. Never under [`AddressMode::Direct`].
    #[cfg(feature = "unicorn-oracle")]
    fn image_is_relocated(&self) -> bool {
        // Mirrors `compile`: a seeded `image_base` wins over the builder's.
        let effective = if self.init.image_base == 0 {
            self.image_base
        } else {
            self.init.image_base
        };
        self.config.address_mode == AddressMode::ImageBaseRelative && effective != self.ip
    }

    /// Lifts and compiles into a [`Translation`] for the [`runtime`](crate::runtime) module;
    /// unlike [`compile`](Self::compile), no guest stack is owned.
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

        // The Vec's allocation address is stable across moves, so `rsp` stays valid in the `Harness`.
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

        // The host C ABI matches `extern "C"` for a single pointer argument.
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

    /// Lifts, compiles and runs once; panics on failure.
    pub fn run(self) -> RunOutcome {
        self.try_run().expect("testkit: lift/compile failed")
    }

    /// Like [`run`](Self::run) but returns lift/compile errors.
    pub fn try_run(self) -> Result<RunOutcome, TestError> {
        // Captured before `compile` consumes `self`; only for plain executable JIT configs.
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

        // The harness is alive inside `outcome`, so Unicorn shares the same stack pages.
        #[cfg(feature = "unicorn-oracle")]
        if let Some((bytes, ip, init)) = xcheck {
            cross_check_unicorn(&bytes, ip, &init, &outcome);
        }

        Ok(outcome)
    }
}

/// Re-runs a just-executed program on Unicorn from the same initial state and asserts agreement.
///
/// Soundness guards, all from static decode:
/// - **Unrelocated image only** ([`LiftTest::image_is_relocated`]): Unicorn's `MEM_UNMAPPED` hook
///   identity-maps pages, so a lift-time VA read in a relocated run faults the host.
/// - **Clean halt only**: any VM exit or Unicorn fault is skipped.
/// - **Shared stack**: `initial_rsp` is handed to Unicorn so `rsp`-derived values line up.
/// - **No non-stack memory writes**: Microlift runs first and would feed Unicorn mutated inputs.
/// - **No syscalls**: Unicorn does not run the host syscall callback.
/// - **Undefined flags masked**: `MUL`/`IMUL` (SF/ZF) and variable shifts/rotates/`BT*` (OF).
/// - **No undefined flag reads**: a flag copied out (`LAHF`, `SETcc`, `PUSHF`) escapes the mask.
#[cfg(feature = "unicorn-oracle")]
fn cross_check_unicorn(bytes: &[u8], ip: u64, init: &oracle::CpuState, outcome: &RunOutcome) {
    use oracle::{CpuOracle, CpuState, DiffOptions, OracleOutcome, diff_states};

    // Only a clean `RET` to the exit trampoline is diffable: at a region boundary Unicorn would
    // keep executing zero padding and fault the host via the identity hook.
    if outcome.exit != ExitReason::Return {
        return;
    }

    let ProgramFacts {
        undefined_rflags,
        writes_data_mem,
        has_syscall,
        reads_undefined_flags,
    } = analyze_program(bytes, ip);

    if has_syscall || reads_undefined_flags {
        return;
    }

    // Persisted non-stack writes cannot be diffed on shared memory.
    if writes_data_mem {
        return;
    }

    // Share Microlift's stack so stack contents and pointers match.
    let mut uinit = *init;
    uinit.gpr[oracle::gpr::RSP] = outcome.initial_rsp;

    let mut uc = oracle::unicorn::UnicornOracle::new();
    let out = uc.run(bytes, ip, &uinit, &[], 1 << 20);
    let OracleOutcome::Halted(ustate) = out else {
        return;
    };

    // Drop architecturally-undefined flag bits.
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

/// Static facts about a lifted program that keep the Unicorn cross-check sound.
#[cfg(feature = "unicorn-oracle")]
struct ProgramFacts {
    /// Packed-EFLAGS mask of flags left undefined at exit (last writer wins).
    undefined_rflags: u64,
    /// Whether any instruction writes memory that is not `RSP`-relative.
    writes_data_mem: bool,
    has_syscall: bool,
    /// Whether any instruction reads a flag an earlier instruction left undefined.
    reads_undefined_flags: bool,
}

/// Linear-sweep decode from `ip` to derive [`ProgramFacts`]; stops at the first invalid byte.
#[cfg(feature = "unicorn-oracle")]
fn analyze_program(bytes: &[u8], ip: u64) -> ProgramFacts {
    use iced_x86::{Decoder, DecoderOptions, InstructionInfoFactory, Mnemonic, OpAccess, Register};

    // Translate iced's `RflagsBits` to packed EFLAGS positions.
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
    let mut reads_undefined_flags = false;

    while decoder.can_decode() {
        let insn = decoder.decode();
        if insn.is_invalid() {
            break;
        }
        has_syscall |= insn.mnemonic() == Mnemonic::Syscall;

        reads_undefined_flags |= undefined & insn.rflags_read() != 0;

        // Last writer wins: defined values re-define a flag, undefined ones undefine it.
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
            // Stack writes self-heal across runs; only non-`RSP` writes corrupt the second engine.
            if writes && m.base() != Register::RSP {
                writes_data_mem = true;
            }
        }
    }

    ProgramFacts {
        undefined_rflags: iced_to_eflags(undefined),
        writes_data_mem,
        has_syscall,
        reads_undefined_flags,
    }
}

/// A compiled, runnable function plus the guest stack and executable memory that must outlive every entry.
pub struct Harness {
    /// Live register context, mutated on every entry.
    pub ctx: RegisterContext,
    /// `rsp` at first entry.
    pub initial_rsp: u64,
    /// Host address of the compiled code; base for [`GuestCodeMap::lookup_host_addr`].
    pub code_ptr: *const u8,
    /// Address a link row must name: [`code_ptr`](Self::code_ptr), except under a pinned ABI where
    /// it is the tail-convention body behind the thunk ([`CompiledRegion`](crate::backend::CompiledRegion)).
    pub link_target: *const u8,
    /// Guest instruction of each compiled byte ([`crate::core::srcloc`]); empty without [`LifterConfig::emit_source_locs`].
    pub code_map: GuestCodeMap,
    // Keeps the guest stack alive.
    _stack: Vec<u64>,
    // Keeps the JIT module mapped.
    _backend: CraneliftBackend,
    entry: unsafe extern "C" fn(*mut RegisterContext),
}

impl Harness {
    pub fn enter(&mut self) -> ExitReason {
        unsafe {
            (self.entry)(&mut self.ctx as *mut RegisterContext);
        }
        ExitReason::try_from(self.ctx.exit_reason).unwrap_or(ExitReason::None)
    }
}

pub struct RunOutcome {
    /// Context snapshot after the entry. `ctx.rsp` points into the stack owned by the retained
    /// [`Harness`]: keep this outcome alive while using it, or zero `rsp`.
    pub ctx: RegisterContext,
    /// `rsp` at first entry ([`Harness::initial_rsp`]).
    pub initial_rsp: u64,
    pub exit: ExitReason,
    harness: Harness,
}

impl RunOutcome {
    pub fn reg(&self, f: impl FnOnce(&RegisterContext)) -> &Self {
        f(&self.ctx);
        self
    }

    /// Mutates the retained context (e.g. advance `rip`, fill `rax`/`rdx`) and re-enters.
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

    pub fn resume(self) -> RunOutcome {
        self.resume_with(|_| {})
    }

    /// Guest instruction of each compiled byte ([`crate::core::srcloc`]).
    pub fn code_map(&self) -> &GuestCodeMap {
        &self.harness.code_map
    }

    /// Host address of the compiled region ([`GuestCodeMap::lookup_host_addr`]).
    pub fn code_ptr(&self) -> *const u8 {
        self.harness.code_ptr
    }
}

/// Differential-oracle wiring against Unicorn; `unicorn-oracle` feature only.
#[cfg(feature = "unicorn-oracle")]
impl<'a> LiftTest<'a> {
    /// Runs on the Microlift JIT and Unicorn from the same seed, returning the [`StateDiff`];
    /// `None` when an engine hit a non-diffable event. Only registers/flags/xmm and lift base carry over.
    pub fn differential_unicorn(&self, opts: oracle::DiffOptions) -> Option<oracle::StateDiff> {
        let state = oracle::CpuState::from_ctx(&self.init);
        oracle::unicorn::differential_unicorn(self.bytes, self.ip, &state, &[], 1 << 20, opts)
    }

    /// Like [`differential_unicorn`](Self::differential_unicorn) but panics on a diff; `None` passes.
    pub fn assert_matches_unicorn(&self, opts: oracle::DiffOptions) {
        if let Some(diff) = self.differential_unicorn(opts) {
            assert!(diff.is_empty(), "microlift and unicorn disagree: {diff:?}");
        }
    }
}

/// Lifts `bytes` at [`DEFAULT_LIFT_BASE`] with defaults and `Opt::None`, runs against `ctx`, returns the mutated context.
pub fn run_test(bytes: &[u8], ctx: RegisterContext) -> RegisterContext {
    LiftTest::new(bytes).ctx(ctx).run().ctx
}

pub mod prelude {
    pub use super::{Harness, LiftTest, Opt, RetAddr, RunOutcome, TestError, run_test};
    pub use crate::context::{ExitReason, RegisterContext};
    pub use crate::core::specialize::{Spec, SpecTarget, Specialization};
    pub use crate::core::{AddressMode, Atomic128Mode, GeneratedNan, VmResumptionPoints};
}
