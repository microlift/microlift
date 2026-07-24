//! Shared integration-test / fuzz / bench harness.
//! This module centralizes that pattern behind a small builder API. It is a
//! thin wrapper over the library's own [`lift_to_function`] entry point plus
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
use crate::core::{lift_to_function, ExecutionMode, LifterConfig};
use cranelift_codegen::settings::{self, Configurable};

pub mod oracle;
pub mod rng;

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
}

impl<'a> LiftTest<'a> {
    /// Start describing a test that lifts `bytes` at [`DEFAULT_LIFT_BASE`].
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            ip: DEFAULT_LIFT_BASE,
            image_base: DEFAULT_LIFT_BASE,
            config: LifterConfig::default(),
            opt: Opt::None,
            stack_words: DEFAULT_STACK_WORDS,
            ret_addr: RetAddr::Trampoline,
            init: RegisterContext::default(),
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
    pub fn config(mut self, config: LifterConfig) -> Self {
        self.config = config;
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

    /// Set the execution mode (`Jit` vs `RuntimeAssisted`).
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.config.execution_mode = mode;
        self
    }

    /// Configure the lift for resumable VM exits: `RuntimeAssisted` execution
    /// with resume blocks and interpreter fallback. Use with
    /// [`RunOutcome::resume_with`] to drive exit → mutate → resume cycles.
    pub fn resumable(mut self) -> Self {
        self.config.execution_mode = ExecutionMode::RuntimeAssisted;
        self.config.emit_resume_blocks = true;
        self.config.interpreter_fallback = true;
        self
    }

    /// Provide static memory segments (rodata hints etc.) to the lifter.
    pub fn segments(mut self, segments: Vec<MemorySegment>) -> Self {
        self.config.segments = segments;
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

    /// Lift and compile, but do not run. Returns a ready-to-enter [`Harness`].
    pub fn compile(self) -> Result<Harness, TestError> {
        let func = lift_to_function::<X86_64>(self.bytes, self.ip, self.config)
            .map_err(TestError::Lift)?;

        let mut backend =
            CraneliftBackend::with_flags(self.opt.flags()).map_err(TestError::Compile)?;
        let code_ptr = backend.compile(func).map_err(TestError::Compile)?;

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
        let mut harness = self.compile()?;
        let exit = harness.enter();
        Ok(RunOutcome {
            ctx: harness.ctx,
            initial_rsp: harness.initial_rsp,
            exit,
            harness,
        })
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
    pub use super::{run_test, Harness, LiftTest, Opt, RetAddr, RunOutcome, TestError};
    pub use crate::context::{ExitReason, RegisterContext};
}
