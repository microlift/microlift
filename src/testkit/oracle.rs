//! A pluggable CPU-oracle abstraction for differential testing.
//! Only deterministic, self-contained instructions: reg/reg and reg/imm
//! arithmetic, logic, shifts, `setcc`, `cmov`, `lea`, `mov`, and loads/stores
//! confined to the shared arena. Anything that produces a Microlift *VM exit*
//! (port I/O, cpuid, xgetbv, rdtsc, imported/indirect call, ud2, int3)
//! or is host-specific (rdtsc/rdrand) is **not** diffable and surfaces as
//! [`OracleOutcome::Exit`] rather than a comparable [`CpuOracle`] halt.

use crate::context::{ExitReason, RegisterContext};
use crate::core::specialize::Specialization;
use crate::testkit::{LiftTest, RetAddr};

/// x86 general-purpose register indices, in hardware encoding order, for
/// [`CpuState::gpr`].
pub mod gpr {
    pub const RAX: usize = 0;
    pub const RCX: usize = 1;
    pub const RDX: usize = 2;
    pub const RBX: usize = 3;
    pub const RSP: usize = 4;
    pub const RBP: usize = 5;
    pub const RSI: usize = 6;
    pub const RDI: usize = 7;
}

/// An architectural machine state at an oracle boundary. Flat and decoupled
/// from [`RegisterContext`] so a non-Microlift oracle need not depend on our
/// struct layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuState {
    /// GPRs in hardware encoding order: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
    /// r8..r15 (see [`gpr`]).
    pub gpr: [u64; 16],
    pub rip: u64,
    /// Packed RFLAGS (only the arithmetic flags Microlift models: CF, PF, AF,
    /// ZF, SF, OF, DF).
    pub rflags: u64,
    /// XMM0..15, each `[lo64, hi64]`.
    pub xmm: [[u64; 2]; 16],
    /// FS segment base (for `fs:`-prefixed / `rd/wrfsbase` accesses). An
    /// *input* to a run, not part of the diff - see [`diff_states`].
    pub fs_base: u64,
    /// GS segment base (for `gs:`-prefixed / `rd/wrgsbase` accesses). An input,
    /// not diffed.
    pub gs_base: u64,
}

// RFLAGS bit positions.
const CF: u64 = 1 << 0;
const PF: u64 = 1 << 2;
const AF: u64 = 1 << 4;
const ZF: u64 = 1 << 6;
const SF: u64 = 1 << 7;
const DF: u64 = 1 << 10;
const OF: u64 = 1 << 11;

impl CpuState {
    /// All zero (a clean reset state).
    pub fn zeroed() -> Self {
        CpuState {
            gpr: [0; 16],
            rip: 0,
            rflags: 0,
            xmm: [[0; 2]; 16],
            fs_base: 0,
            gs_base: 0,
        }
    }

    /// Snapshot the diffable subset of a [`RegisterContext`].
    pub fn from_ctx(c: &RegisterContext) -> Self {
        let mut rflags = 0u64;
        if c.cf != 0 {
            rflags |= CF;
        }
        if c.pf() {
            rflags |= PF;
        }
        if c.af() {
            rflags |= AF;
        }
        if c.zf != 0 {
            rflags |= ZF;
        }
        if c.sf != 0 {
            rflags |= SF;
        }
        if c.of != 0 {
            rflags |= OF;
        }
        if c.df != 0 {
            rflags |= DF;
        }
        CpuState {
            gpr: [
                c.rax, c.rcx, c.rdx, c.rbx, c.rsp, c.rbp, c.rsi, c.rdi, c.r8, c.r9, c.r10, c.r11,
                c.r12, c.r13, c.r14, c.r15,
            ],
            rip: c.rip,
            rflags,
            xmm: *c.xmm,
            fs_base: c.fs_base,
            gs_base: c.gs_base,
        }
    }

    /// Write this state into a [`RegisterContext`] (registers, rip, flags, xmm).
    pub fn apply_to_ctx(&self, c: &mut RegisterContext) {
        let g = &self.gpr;
        c.rax = g[0];
        c.rcx = g[1];
        c.rdx = g[2];
        c.rbx = g[3];
        c.rsp = g[4];
        c.rbp = g[5];
        c.rsi = g[6];
        c.rdi = g[7];
        c.r8 = g[8];
        c.r9 = g[9];
        c.r10 = g[10];
        c.r11 = g[11];
        c.r12 = g[12];
        c.r13 = g[13];
        c.r14 = g[14];
        c.r15 = g[15];
        c.rip = self.rip;
        c.cf = (self.rflags & CF != 0) as u8;
        c.set_pf(self.rflags & PF != 0);
        c.set_af(self.rflags & AF != 0);
        c.zf = (self.rflags & ZF != 0) as u8;
        c.sf = (self.rflags & SF != 0) as u8;
        c.of = (self.rflags & OF != 0) as u8;
        c.df = (self.rflags & DF != 0) as u8;
        c.xmm = self.xmm.into();
        c.fs_base = self.fs_base;
        c.gs_base = self.gs_base;
    }
}

/// A block of guest memory to be present (identity-mapped) before a run: `bytes`
/// placed at guest virtual address `base`.
#[derive(Clone, Debug)]
pub struct MemImage {
    pub base: u64,
    pub bytes: Vec<u8>,
}

/// Why a run stopped, from an oracle's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleExit {
    /// A Microlift VM exit (or an emulator event) that is not part of the
    /// diffable instruction set - the two engines cannot be meaningfully
    /// compared past this point.
    NonDiffable(ExitReason),
    /// The external emulator (Unicorn) stopped with an error/exception rather
    /// than reaching the exit trampoline: an invalid opcode, an unmapped
    /// fetch/access it could not satisfy, an `int` with no handler,
    /// etc. The payload is the raw `uc_error` discriminant, kept only for
    /// diagnostics. Like [`OracleExit::NonDiffable`], nothing past this point
    /// is comparable, so [`differential`] declines the comparison.
    Emulator(i32),
}

/// The result of running an oracle.
#[derive(Clone, Debug)]
pub enum OracleOutcome {
    /// The code ran to a clean natural stop; `CpuState` is comparable.
    Halted(CpuState),
    /// The code hit a non-diffable event; state is reported but should be
    /// excluded from a value comparison.
    Exit { state: CpuState, reason: OracleExit },
}

impl OracleOutcome {
    /// The final state, whether the run halted cleanly or exited.
    pub fn state(&self) -> &CpuState {
        match self {
            OracleOutcome::Halted(s) => s,
            OracleOutcome::Exit { state, .. } => state,
        }
    }

    /// Whether the run reached a clean, comparable halt.
    pub fn is_halted(&self) -> bool {
        matches!(self, OracleOutcome::Halted(_))
    }
}

/// A machine-state execution engine that can be differentially tested.
pub trait CpuOracle {
    /// Load `code` such that its first byte is at guest VA `code_va`, seed
    /// registers from `init`, ensure each [`MemImage`] is present, run until a
    /// natural stop / non-diffable event / the `max_insns` budget, and report
    /// the outcome. (`max_insns` is advisory; the in-process JIT has no
    /// instruction budget and ignores it.)
    fn run(
        &mut self,
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        mem: &[MemImage],
        max_insns: u64,
    ) -> OracleOutcome;

    /// A short human-readable name (used in diff messages).
    fn name(&self) -> &'static str;
}

/// The Microlift JIT as an oracle: lift `code`, JIT it, and run it against a
/// [`RegisterContext`] seeded from `init`.
///
/// Because Microlift executes with *guest VA == host VA*, any [`MemImage`] the
/// guest dereferences must already be host-backed at its `base` (place it in an
/// [`IdentityArena`]); this oracle asserts that rather than copying.
#[derive(Default)]
pub struct MicroliftOracle {
    /// Backend optimization level to compile with.
    pub opt: crate::testkit::Opt,
    /// Whether to use native `I64X2` XMM variables and the vector fast paths.
    ///
    /// Defaults off, matching [`LifterConfig`](crate::core::LifterConfig).
    pub vector_xmm: bool,
    /// Target-directed instruction specializations to apply.
    ///
    /// Defaults to [`Specialization::generic`]. Setting it points the oracle at the
    /// specialized lowerings, which is how those get differential coverage against
    /// real hardware rather than only against microlift's own portable path. The
    /// vector ones additionally need [`vector_xmm`](Self::vector_xmm), and
    /// [`Spec::IntegerOverflowFlags`](crate::core::specialize::Spec::IntegerOverflowFlags)
    /// needs [`eliminate_dead_flags`](Self::eliminate_dead_flags);
    /// [`specialized`](Self::specialized) sets both.
    pub specialization: Specialization,
    /// Whether to run the dead-flag liveness pass (`config.eliminate_dead_flags`).
    ///
    /// Defaults off, matching [`LifterConfig`](crate::core::LifterConfig).
    pub eliminate_dead_flags: bool,
}

impl MicroliftOracle {
    pub fn new(opt: crate::testkit::Opt) -> Self {
        Self {
            opt,
            ..Default::default()
        }
    }

    /// Lift with `spec` applied, and with everything the specializations need to be
    /// reachable: vector-backed XMM registers and the dead-flag liveness pass.
    ///
    /// Both are turned on together rather than per-`Spec` so a caller enabling a
    /// whole target's set gets a lift where every member can fire. Neither changes
    /// an architectural result, and both are separately proven not to.
    pub fn specialized(mut self, spec: Specialization) -> Self {
        self.vector_xmm = true;
        self.eliminate_dead_flags = true;
        self.specialization = spec;
        self
    }
}

impl CpuOracle for MicroliftOracle {
    fn run(
        &mut self,
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        _mem: &[MemImage],
        _max_insns: u64,
    ) -> OracleOutcome {
        let mut ctx = RegisterContext::default();
        init.apply_to_ctx(&mut ctx);

        // If the caller placed a stack VA in the state (e.g. inside an
        // `IdentityArena`), the harness honors it; otherwise it allocates one.
        // Either way the exit trampoline is pushed at `[rsp]` so a guest `ret`
        // stops cleanly.
        let out = LiftTest::new(code)
            .at(code_va)
            .image_base(code_va)
            .opt(self.opt)
            .vector_xmm(self.vector_xmm)
            .specialize(self.specialization)
            .ctx(ctx)
            .ret_addr(RetAddr::Trampoline);
        let out = if self.eliminate_dead_flags {
            out.eliminate_dead_flags()
        } else {
            out
        }
        .run();

        let state = CpuState::from_ctx(&out.ctx);
        match out.exit {
            ExitReason::Return | ExitReason::None => OracleOutcome::Halted(state),
            other => OracleOutcome::Exit {
                state,
                reason: OracleExit::NonDiffable(other),
            },
        }
    }

    fn name(&self) -> &'static str {
        "microlift-jit"
    }
}

/// A field-by-field difference between two [`CpuState`]s. Empty means equal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateDiff {
    /// GPR index -> (a, b) for each differing register (rsp is skipped by
    /// [`differential`] unless a shared stack VA is used - see its docs).
    pub gpr: Vec<(usize, u64, u64)>,
    pub rflags: Option<(u64, u64)>,
    pub rip: Option<(u64, u64)>,
    pub xmm: Vec<(usize, [u64; 2], [u64; 2])>,
}

impl StateDiff {
    pub fn is_empty(&self) -> bool {
        self.gpr.is_empty() && self.rflags.is_none() && self.rip.is_none() && self.xmm.is_empty()
    }
}

/// Options controlling which fields a differential comparison considers.
///
/// Both fields default to `false`: `rsp` is only meaningful when both oracles
/// share a stack VA (an [`IdentityArena`]) - the in-process JIT otherwise
/// allocates its own stack, so `rsp` legitimately differs - and `rip`
/// trampoline/return bookkeeping can differ across engines.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffOptions {
    /// Compare `rsp` (requires a shared stack VA).
    pub compare_rsp: bool,
    /// Compare `rip`.
    pub compare_rip: bool,
}

/// Compute the difference between two states under `opts`.
pub fn diff_states(a: &CpuState, b: &CpuState, opts: DiffOptions) -> StateDiff {
    let mut d = StateDiff::default();
    for i in 0..16 {
        if i == gpr::RSP && !opts.compare_rsp {
            continue;
        }
        if a.gpr[i] != b.gpr[i] {
            d.gpr.push((i, a.gpr[i], b.gpr[i]));
        }
    }
    if a.rflags != b.rflags {
        d.rflags = Some((a.rflags, b.rflags));
    }
    if opts.compare_rip && a.rip != b.rip {
        d.rip = Some((a.rip, b.rip));
    }
    for i in 0..16 {
        if a.xmm[i] != b.xmm[i] {
            d.xmm.push((i, a.xmm[i], b.xmm[i]));
        }
    }
    d
}

/// Run the same program on two oracles and return their state difference.
///
/// Each oracle is given the identical `init`/`mem`; the caller is responsible
/// for providing fresh (uncorrupted) memory images if an oracle mutates shared
/// backing. When either run reports a non-diffable exit the comparison is
/// skipped and `None` is returned (there is nothing sound to compare).
pub fn differential<A: CpuOracle, B: CpuOracle>(
    a: &mut A,
    b: &mut B,
    code: &[u8],
    code_va: u64,
    init: &CpuState,
    mem: &[MemImage],
    max_insns: u64,
    opts: DiffOptions,
) -> Option<StateDiff> {
    let oa = a.run(code, code_va, init, mem, max_insns);
    let ob = b.run(code, code_va, init, mem, max_insns);
    if !oa.is_halted() || !ob.is_halted() {
        return None;
    }
    Some(diff_states(oa.state(), ob.state(), opts))
}

/// A host memory buffer mapped at a fixed virtual address so that *guest VA ==
/// host VA* - the reconciliation primitive that lets microlift's
/// pointer-is-the-host-address JIT be diffed against an external emulator.
///
/// Unix only, and it never displaces an existing mapping - see
/// [`IdentityArena::map`].
#[cfg(unix)]
pub struct IdentityArena {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
impl IdentityArena {
    /// Map `len` bytes read/write at exactly `va`. `va` and `len` should be page
    /// aligned and in a normally-free range. `None` if the mapping could not be
    /// established there, which callers treat as "skip this test".
    ///
    /// **Never `MAP_FIXED`.** It does not fail on a collision, it *replaces* the
    /// mapping that was there - which in a test binary is another thread's JIT code
    /// or heap, placed there precisely because the range was free a moment ago. That
    /// corrupts the other thread and survives being noticed: the owner's `munmap`
    /// then tears down what is now *this* arena, and the fault lands here, at
    /// exactly `va`, in a test that did nothing wrong. It reproduced as an
    /// intermittent SIGSEGV in `unicorn_oracle_tests` under `--release`.
    ///
    /// Linux has `MAP_FIXED_NOREPLACE`: same exact placement, `EEXIST` instead of a
    /// silent replacement. Everywhere else `va` is a plain **hint** with no fixed
    /// flag, and the exact-address check below turns "somewhere else" into `None`.
    /// Both spellings are atomic with respect to other threads.
    pub fn map(va: u64, len: usize) -> Option<Self> {
        #[cfg(target_os = "linux")]
        let fixed = libc_flags::MAP_FIXED_NOREPLACE;
        #[cfg(not(target_os = "linux"))]
        let fixed = 0;

        let prot = libc_flags::PROT_READ | libc_flags::PROT_WRITE;
        let flags = libc_flags::MAP_PRIVATE | libc_flags::MAP_ANON | fixed;
        // SAFETY: standard anonymous mapping; we own the returned range.
        let p = unsafe { mmap_raw(va as *mut core::ffi::c_void, len, prot, flags) };
        if p.is_null() || p as isize == -1 || p as u64 != va {
            if !p.is_null() && p as isize != -1 {
                unsafe { munmap_raw(p, len) };
            }
            return None;
        }
        Some(IdentityArena {
            ptr: p as *mut u8,
            len,
        })
    }

    /// Base virtual address of the arena (equal to the requested `va`).
    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    /// The whole arena as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: we own `len` bytes at `ptr` for the arena's lifetime.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Write `bytes` at offset `off` within the arena.
    pub fn write_at(&mut self, off: usize, bytes: &[u8]) {
        self.as_mut_slice()[off..off + bytes.len()].copy_from_slice(bytes);
    }
}

#[cfg(unix)]
impl Drop for IdentityArena {
    fn drop(&mut self) {
        // SAFETY: unmapping the exact range we mapped.
        unsafe { munmap_raw(self.ptr as *mut core::ffi::c_void, self.len) };
    }
}

// Minimal libc bindings for the arena so `testkit` need not pull in the `libc`
// crate. These match the System V / BSD `mmap` ABI on the supported hosts.
#[cfg(unix)]
mod libc_flags {
    pub const PROT_READ: i32 = 0x1;
    pub const PROT_WRITE: i32 = 0x2;
    pub const MAP_PRIVATE: i32 = 0x2;
    #[cfg(target_os = "linux")]
    pub const MAP_ANON: i32 = 0x20;
    #[cfg(target_os = "linux")]
    pub const MAP_FIXED_NOREPLACE: i32 = 0x100000;
    #[cfg(not(target_os = "linux"))]
    pub const MAP_ANON: i32 = 0x1000;
}

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        addr: *mut core::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
}

// Wrapper supplying the fixed `fd = -1, offset = 0` for anonymous maps.
#[cfg(unix)]
unsafe fn mmap_raw(
    addr: *mut core::ffi::c_void,
    len: usize,
    prot: i32,
    flags: i32,
) -> *mut core::ffi::c_void {
    // SAFETY: forwarding to the libc `mmap` with the fixed anonymous fd/offset.
    unsafe { mmap(addr, len, prot, flags, -1, 0) }
}

#[cfg(unix)]
unsafe fn munmap_raw(addr: *mut core::ffi::c_void, len: usize) -> i32 {
    // SAFETY: forwarding to libc `munmap` for a range we mapped.
    unsafe { munmap(addr, len) }
}

#[cfg(feature = "unicorn-oracle")]
pub mod unicorn {
    use super::{CpuOracle, CpuState, MemImage, OracleExit, OracleOutcome, gpr};
    use crate::testkit::trampoline::{ImportOracle, ImportTable};
    use crate::testkit::{DEFAULT_STACK_WORDS, EXIT_TRAMPOLINE};
    use core::cell::Cell;
    use core::ffi::c_void;
    use std::rc::Rc;
    use unicorn_engine::{Arch, HookType, Mode, Prot, RegisterX86, Unicorn};

    const PAGE: u64 = 0x1000;

    /// GPR-index (see [`gpr`]) -> Unicorn register id, in [`CpuState::gpr`] order.
    const GPR_REG: [RegisterX86; 16] = [
        RegisterX86::RAX,
        RegisterX86::RCX,
        RegisterX86::RDX,
        RegisterX86::RBX,
        RegisterX86::RSP,
        RegisterX86::RBP,
        RegisterX86::RSI,
        RegisterX86::RDI,
        RegisterX86::R8,
        RegisterX86::R9,
        RegisterX86::R10,
        RegisterX86::R11,
        RegisterX86::R12,
        RegisterX86::R13,
        RegisterX86::R14,
        RegisterX86::R15,
    ];

    const XMM_REG: [RegisterX86; 16] = [
        RegisterX86::XMM0,
        RegisterX86::XMM1,
        RegisterX86::XMM2,
        RegisterX86::XMM3,
        RegisterX86::XMM4,
        RegisterX86::XMM5,
        RegisterX86::XMM6,
        RegisterX86::XMM7,
        RegisterX86::XMM8,
        RegisterX86::XMM9,
        RegisterX86::XMM10,
        RegisterX86::XMM11,
        RegisterX86::XMM12,
        RegisterX86::XMM13,
        RegisterX86::XMM14,
        RegisterX86::XMM15,
    ];

    const RFLAGS_MASK: u64 = {
        const CF: u64 = 1 << 0;
        const PF: u64 = 1 << 2;
        const AF: u64 = 1 << 4;
        const ZF: u64 = 1 << 6;
        const SF: u64 = 1 << 7;
        const DF: u64 = 1 << 10;
        const OF: u64 = 1 << 11;
        CF | PF | AF | ZF | SF | DF | OF
    };
    // Bit 1 of EFLAGS is reserved and reads as 1 on real hardware; set it so the
    // seeded value is architecturally well-formed.
    const RFLAGS_RESERVED: u64 = 1 << 1;

    fn page_base(addr: u64) -> u64 {
        addr & !(PAGE - 1)
    }

    pub struct UnicornOracle {
        pub stack_words: usize,
    }

    impl Default for UnicornOracle {
        fn default() -> Self {
            Self {
                stack_words: DEFAULT_STACK_WORDS,
            }
        }
    }

    impl UnicornOracle {
        pub fn new() -> Self {
            Self::default()
        }

        /// Seed the engine's architectural registers from `st`.
        fn write_state(
            uc: &mut Unicorn<'_, ()>,
            st: &CpuState,
        ) -> Result<(), unicorn_engine::uc_error> {
            for (i, &reg) in GPR_REG.iter().enumerate() {
                uc.reg_write(reg, st.gpr[i])?;
            }
            uc.reg_write(RegisterX86::RIP, st.rip)?;
            uc.reg_write(
                RegisterX86::EFLAGS,
                (st.rflags & RFLAGS_MASK) | RFLAGS_RESERVED,
            )?;
            // Segment bases, so `fs:`/`gs:`-prefixed accesses hit the same
            // host-backed TLS buffer Microlift addresses.
            uc.reg_write(RegisterX86::FS_BASE, st.fs_base)?;
            uc.reg_write(RegisterX86::GS_BASE, st.gs_base)?;
            // `MXCSR` is not part of `CpuState`, so both engines have to be put
            // in the same *implicit* starting state. Microlift's is the
            // architectural reset value (`RegisterContext::default`); Unicorn
            // starts it at zero, which unmasks every SSE exception and is not a
            // state hardware powers up in. Seed it so an `STMXCSR` reads the
            // same value on both sides.
            uc.reg_write(RegisterX86::MXCSR, crate::context::MXCSR_RESET as u64)?;
            // Same story for the x87 environment, which `CpuState` also does
            // not carry: Microlift starts with the `FNINIT` control word and
            // every register tagged empty, Unicorn with a zero control word and
            // every register tagged valid.
            uc.reg_write(RegisterX86::FPCW, crate::context::X87_CW_RESET as u64)?;
            uc.reg_write(RegisterX86::FPTAG, 0xFFFF)?;
            for (i, &reg) in XMM_REG.iter().enumerate() {
                let mut bytes = [0u8; 16];
                bytes[0..8].copy_from_slice(&st.xmm[i][0].to_le_bytes());
                bytes[8..16].copy_from_slice(&st.xmm[i][1].to_le_bytes());
                uc.reg_write_long(reg, &bytes)?;
            }
            Ok(())
        }

        /// Read the diffable architectural state back out of the engine.
        fn snapshot(uc: &Unicorn<'_, ()>) -> CpuState {
            let mut st = CpuState::zeroed();
            for (i, &reg) in GPR_REG.iter().enumerate() {
                st.gpr[i] = uc.reg_read(reg).unwrap_or(0);
            }
            st.rip = uc.reg_read(RegisterX86::RIP).unwrap_or(0);
            st.rflags = uc.reg_read(RegisterX86::EFLAGS).unwrap_or(0) & RFLAGS_MASK;
            for (i, &reg) in XMM_REG.iter().enumerate() {
                if let Ok(bytes) = uc.reg_read_long(reg) {
                    if bytes.len() >= 16 {
                        let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                        let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
                        st.xmm[i] = [lo, hi];
                    }
                }
            }
            st
        }
    }

    impl UnicornOracle {
        /// The engine-side half of the [`ImportTable`] trampoline: for each trapped VA,
        /// make the address fetchable and break out of emulation the moment control
        /// reaches it.
        ///
        /// The returned cell carries the VA the run stopped at. Breaking out - rather
        /// than redirecting `RIP` from inside the hook - keeps the trampoline
        /// independent of how Unicorn sequences a code hook against the instruction it
        /// guards: the service loop simply restarts emulation at the address it chose.
        fn install_import_stubs(
            uc: &mut Unicorn<'_, ()>,
            vas: &[u64],
        ) -> Result<Rc<Cell<Option<u64>>>, unicorn_engine::uc_error> {
            let pending: Rc<Cell<Option<u64>>> = Rc::new(Cell::new(None));
            for &va in vas {
                // A PLT entry usually shares a page with the lifted code, which
                // is already mapped; only map (and fill) a page we don't have.
                let mut probe = [0u8; 1];
                if uc.mem_read(va, &mut probe).is_err() {
                    uc.mem_map(page_base(va), PAGE, Prot::ALL)?;
                    // `hlt`. The hook stops emulation before this ever runs; if
                    // that assumption were to break we want a loud emulator
                    // fault rather than a silent state divergence.
                    uc.mem_write(va, &[0xF4])?;
                }
                let pending = Rc::clone(&pending);
                uc.add_code_hook(va, va, move |uc, _addr, _size| {
                    pending.set(Some(va));
                    let _ = uc.emu_stop();
                })?;
            }
            Ok(pending)
        }

        fn run_impl(
            &mut self,
            code: &[u8],
            code_va: u64,
            init: &CpuState,
            _mem: &[MemImage],
            max_insns: u64,
            mut imports: Option<&mut ImportTable<'_>>,
        ) -> OracleOutcome {
            // A construction/mapping failure is reported as an emulator exit
            // (the state is meaningless) rather than panicking the fuzzer.
            let mut uc = match Unicorn::new(Arch::X86, Mode::MODE_64) {
                Ok(uc) => uc,
                Err(e) => {
                    return OracleOutcome::Exit {
                        state: CpuState::zeroed(),
                        reason: OracleExit::Emulator(e as i32),
                    };
                }
            };

            macro_rules! bail {
                ($e:expr) => {
                    return OracleOutcome::Exit {
                        state: Self::snapshot(&uc),
                        reason: OracleExit::Emulator($e as i32),
                    }
                };
            }

            let code_page = page_base(code_va);
            let end = code_va + code.len() as u64;
            let code_len = (page_base(end - 1) + PAGE) - code_page;
            if let Err(e) = uc.mem_map(code_page, code_len, Prot::ALL) {
                bail!(e);
            }
            if let Err(e) = uc.mem_write(code_va, code) {
                bail!(e);
            }

            // on first touch of any other page, map it backed by its own address so
            // guest VA == host VA
            if let Err(e) =
                uc.add_mem_hook(HookType::MEM_UNMAPPED, 1, 0, |uc, _ty, addr, _sz, _v| {
                    let base = page_base(addr);
                    // SAFETY: The host owns this page for every address
                    // the diffable program legitimately touches (stack / arena).
                    // A bad access faults here exactly as it would under
                    // Microlift's JIT. Ignore the result: an `Err` (e.g. a race
                    // that mapped it first) simply means the retry will succeed or
                    // fault on its own.
                    let _ = unsafe { uc.mem_map_ptr(base, PAGE, Prot::ALL, base as *mut c_void) };
                    true
                })
            {
                bail!(e);
            }

            let import_vas: Vec<u64> = imports
                .as_deref()
                .map(|t| t.vas().to_vec())
                .unwrap_or_default();
            let pending = match Self::install_import_stubs(&mut uc, &import_vas) {
                Ok(p) => p,
                Err(e) => bail!(e),
            };

            let mut rsp = init.gpr[gpr::RSP];
            let mut scratch_stack: Option<Vec<u64>> = None;
            if rsp == 0 {
                let stack = vec![0u64; self.stack_words];
                let top = stack.as_ptr() as u64 + (self.stack_words as u64) * 8;
                rsp = top - 8;
                scratch_stack = Some(stack);
            }
            // Push the exit trampoline at [rsp] directly into host memory (the
            // identity hook backs the same bytes for Unicorn's own reads).
            // SAFETY: `rsp` is a live host stack address (scratch `Vec` above,
            // or a caller-provided host-backed VA per the oracle contract).
            unsafe {
                *(rsp as *mut u64) = EXIT_TRAMPOLINE;
            }

            let mut seed = *init;
            seed.gpr[gpr::RSP] = rsp;
            seed.rip = code_va;
            if let Err(e) = Self::write_state(&mut uc, &seed) {
                let _ = scratch_stack;
                bail!(e);
            }

            let count = max_insns as usize;
            let mut pc = code_va;
            let result = loop {
                pending.set(None);
                let result = uc.emu_start(pc, EXIT_TRAMPOLINE, 0, count);
                if result.is_err() {
                    break result;
                }
                // No pending trampoline: the run reached the exit trampoline (or
                // exhausted its instruction budget) on its own.
                let Some(va) = pending.get() else {
                    break Ok(());
                };
                let Some(table) = imports.as_deref_mut() else {
                    break Ok(());
                };

                // Present the handler with the same view Microlift's
                // `ImportedFunctionCall` exit gives it: arguments in place and
                // `rsp` already past the return address the `CALL` pushed
                // (Microlift never pushes one).
                let mut state = Self::snapshot(&uc);
                let sp = state.gpr[gpr::RSP];
                let mut ret_bytes = [0u8; 8];
                if let Err(e) = uc.mem_read(sp, &mut ret_bytes) {
                    bail!(e);
                }
                let ret = u64::from_le_bytes(ret_bytes);
                state.gpr[gpr::RSP] = sp.wrapping_add(8);
                state.rip = ret;

                if !table.service(va, &mut state) {
                    return OracleOutcome::Exit {
                        state,
                        reason: OracleExit::NonDiffable(
                            crate::context::ExitReason::ImportedFunctionCall,
                        ),
                    };
                }
                if let Err(e) = Self::write_state(&mut uc, &state) {
                    bail!(e);
                }
                // An import invoked as the outermost frame returns straight to
                // the trampoline; there is nothing left to emulate.
                if ret == EXIT_TRAMPOLINE {
                    break Ok(());
                }
                pc = ret;
            };

            let state = Self::snapshot(&uc);
            // Keep the scratch stack mapped through the read-back.
            drop(scratch_stack);
            match result {
                Ok(()) => OracleOutcome::Halted(state),
                Err(e) => OracleOutcome::Exit {
                    state,
                    reason: OracleExit::Emulator(e as i32),
                },
            }
        }
    }

    impl CpuOracle for UnicornOracle {
        fn run(
            &mut self,
            code: &[u8],
            code_va: u64,
            init: &CpuState,
            mem: &[MemImage],
            max_insns: u64,
        ) -> OracleOutcome {
            self.run_impl(code, code_va, init, mem, max_insns, None)
        }

        fn name(&self) -> &'static str {
            "unicorn"
        }
    }

    impl ImportOracle for UnicornOracle {
        fn run_with_imports(
            &mut self,
            code: &[u8],
            code_va: u64,
            init: &CpuState,
            mem: &[MemImage],
            max_insns: u64,
            imports: &mut ImportTable<'_>,
        ) -> OracleOutcome {
            self.run_impl(code, code_va, init, mem, max_insns, Some(imports))
        }
    }

    pub fn differential_unicorn(
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        mem: &[MemImage],
        max_insns: u64,
        opts: super::DiffOptions,
    ) -> Option<super::StateDiff> {
        let mut a = super::MicroliftOracle::default();
        let mut b = UnicornOracle::new();
        super::differential(&mut a, &mut b, code, code_va, init, mem, max_insns, opts)
    }

    /// [`differential_unicorn`] with microlift lifting under `spec`.
    ///
    /// The check that the target-specialized lowerings agree with the *hardware*,
    /// not merely with microlift's own portable expansion. `tests/specialize_tests.rs`
    /// proves specialized == portable; this proves portable == hardware, and it is
    /// the second that would catch both of microlift's paths being wrong the same
    /// way.
    pub fn differential_unicorn_spec(
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        max_insns: u64,
        spec: crate::core::specialize::Specialization,
    ) -> Option<super::StateDiff> {
        let mut a = super::MicroliftOracle::default().specialized(spec);
        let mut b = UnicornOracle::new();
        super::differential(
            &mut a,
            &mut b,
            code,
            code_va,
            init,
            &[],
            max_insns,
            super::DiffOptions::default(),
        )
    }
}
