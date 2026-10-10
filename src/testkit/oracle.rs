//! Pluggable CPU oracles for differential testing.
//!
//! Only deterministic, self-contained instructions are diffable; VM exits and host-specific
//! instructions (cpuid, rdtsc, port I/O, ...) surface as [`OracleOutcome::Exit`].

use crate::context::{ExitReason, RegisterContext};
use crate::core::specialize::Specialization;
use crate::testkit::{LiftTest, RetAddr};

/// x86 GPR indices in hardware encoding order, for [`CpuState::gpr`].
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

/// Architectural state at an oracle boundary; flat so non-Microlift oracles need not know our context layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuState {
    /// GPRs in encoding order (see [`gpr`]).
    pub gpr: [u64; 16],
    pub rip: u64,
    /// Packed RFLAGS: CF, PF, AF, ZF, SF, OF, DF.
    pub rflags: u64,
    pub xmm: [[u64; 2]; 16],
    /// FS base; an input to a run, not diffed ([`diff_states`]).
    pub fs_base: u64,
    /// GS base; an input, not diffed.
    pub gs_base: u64,
}

const CF: u64 = 1 << 0;
const PF: u64 = 1 << 2;
const AF: u64 = 1 << 4;
const ZF: u64 = 1 << 6;
const SF: u64 = 1 << 7;
const DF: u64 = 1 << 10;
const OF: u64 = 1 << 11;

impl CpuState {
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

/// Guest memory to be present (identity-mapped) at `base` before a run.
#[derive(Clone, Debug)]
pub struct MemImage {
    pub base: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleExit {
    /// A VM exit or emulator event outside the diffable set; nothing past it is comparable.
    NonDiffable(ExitReason),
    /// The emulator stopped with an error (payload: raw `uc_error`) before the exit trampoline;
    /// [`differential`] declines the comparison.
    Emulator(i32),
}

#[derive(Clone, Debug)]
pub enum OracleOutcome {
    Halted(CpuState),
    /// Non-diffable event; state is reported but excluded from comparison.
    Exit { state: CpuState, reason: OracleExit },
}

impl OracleOutcome {
    pub fn state(&self) -> &CpuState {
        match self {
            OracleOutcome::Halted(s) => s,
            OracleOutcome::Exit { state, .. } => state,
        }
    }

    pub fn is_halted(&self) -> bool {
        matches!(self, OracleOutcome::Halted(_))
    }
}

pub trait CpuOracle {
    /// Loads `code` at `code_va`, seeds registers from `init`, ensures each [`MemImage`] is present,
    /// and runs to a natural stop, non-diffable event or `max_insns` (advisory; the JIT ignores it).
    fn run(
        &mut self,
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        mem: &[MemImage],
        max_insns: u64,
    ) -> OracleOutcome;

    fn name(&self) -> &'static str;
}

/// The Microlift JIT as an oracle. Guest VA == host VA, so any [`MemImage`] the guest touches must be
/// host-backed at its `base` (an [`IdentityArena`]); asserted, not copied.
#[derive(Default)]
pub struct MicroliftOracle {
    pub opt: crate::testkit::Opt,
    /// Native `I64X2` XMM variables and vector fast paths. Default off.
    pub vector_xmm: bool,
    /// Target specializations; default [`Specialization::generic`]. Gives specialized lowerings
    /// coverage against hardware. Vector ones need [`vector_xmm`](Self::vector_xmm),
    /// [`Spec::IntegerOverflowFlags`](crate::core::specialize::Spec::IntegerOverflowFlags) needs
    /// [`eliminate_dead_flags`](Self::eliminate_dead_flags); [`specialized`](Self::specialized) sets both.
    pub specialization: Specialization,
    /// Dead-flag liveness pass (`config.eliminate_dead_flags`). Default off.
    pub eliminate_dead_flags: bool,
}

impl MicroliftOracle {
    pub fn new(opt: crate::testkit::Opt) -> Self {
        Self {
            opt,
            ..Default::default()
        }
    }

    /// Lifts with `spec` plus the vector XMM and dead-flag passes it needs to fire; neither changes results.
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

        // A caller-placed stack VA is honored, else one is allocated; the exit trampoline is pushed at `[rsp]`.
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateDiff {
    /// GPR index -> (a, b) per differing register; `rsp` is skipped by [`differential`] without a shared stack.
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

/// Which fields a differential comparison considers. Both default `false`: `rsp` is only
/// meaningful with a shared stack VA ([`IdentityArena`]), and `rip` bookkeeping can differ.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffOptions {
    pub compare_rsp: bool,
    pub compare_rip: bool,
}

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

/// Runs the same program on two oracles and diffs the states; `None` on a non-diffable exit.
/// Each oracle gets identical `init`/`mem`; the caller supplies fresh images if one mutates them.
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

/// Host buffer mapped at a fixed VA so guest VA == host VA; Unix only, never displaces a mapping
/// ([`IdentityArena::map`]).
#[cfg(unix)]
pub struct IdentityArena {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
impl IdentityArena {
    /// Maps `len` bytes read/write at exactly `va`; `None` if it cannot be placed there (callers skip).
    ///
    /// **Never `MAP_FIXED`**: it silently replaces another thread's mapping and corrupts it
    /// (an intermittent SIGSEGV under `--release`). Linux uses `MAP_FIXED_NOREPLACE`; elsewhere
    /// `va` is a hint and the exact-address check turns "elsewhere" into `None`.
    pub fn map(va: u64, len: usize) -> Option<Self> {
        #[cfg(target_os = "linux")]
        let fixed = libc_flags::MAP_FIXED_NOREPLACE;
        #[cfg(not(target_os = "linux"))]
        let fixed = 0;

        let prot = libc_flags::PROT_READ | libc_flags::PROT_WRITE;
        let flags = libc_flags::MAP_PRIVATE | libc_flags::MAP_ANON | fixed;
        // SAFETY: anonymous mapping we own.
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

    pub fn base(&self) -> u64 {
        self.ptr as u64
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: we own `len` bytes at `ptr`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn write_at(&mut self, off: usize, bytes: &[u8]) {
        self.as_mut_slice()[off..off + bytes.len()].copy_from_slice(bytes);
    }
}

#[cfg(unix)]
impl Drop for IdentityArena {
    fn drop(&mut self) {
        // SAFETY: unmapping the range we mapped.
        unsafe { munmap_raw(self.ptr as *mut core::ffi::c_void, self.len) };
    }
}

// Minimal libc bindings so `testkit` need not depend on the `libc` crate (System V / BSD `mmap` ABI).
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

// Fixes `fd = -1, offset = 0` for anonymous maps.
#[cfg(unix)]
unsafe fn mmap_raw(
    addr: *mut core::ffi::c_void,
    len: usize,
    prot: i32,
    flags: i32,
) -> *mut core::ffi::c_void {
    // SAFETY: forwards to libc `mmap` with fixed anonymous args.
    unsafe { mmap(addr, len, prot, flags, -1, 0) }
}

#[cfg(unix)]
unsafe fn munmap_raw(addr: *mut core::ffi::c_void, len: usize) -> i32 {
    // SAFETY: unmaps a range we mapped.
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
    // Bit 1 of EFLAGS is reserved and reads 1.
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
            // Segment bases, so `fs:`/`gs:` accesses hit the same TLS buffer as Microlift.
            uc.reg_write(RegisterX86::FS_BASE, st.fs_base)?;
            uc.reg_write(RegisterX86::GS_BASE, st.gs_base)?;
            // MXCSR is not in `CpuState`: seed the reset value (Unicorn starts at zero, all exceptions unmasked).
            uc.reg_write(RegisterX86::MXCSR, crate::context::MXCSR_RESET as u64)?;
            // Same for the x87 environment: Microlift starts at the `FNINIT` word with all registers empty.
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
        /// Engine half of the [`ImportTable`] trampoline: makes each trapped VA fetchable and stops
        /// emulation when control reaches it. The returned cell carries the stop VA; breaking out
        /// (not redirecting `RIP` in the hook) keeps this independent of Unicorn's hook sequencing.
        fn install_import_stubs(
            uc: &mut Unicorn<'_, ()>,
            vas: &[u64],
        ) -> Result<Rc<Cell<Option<u64>>>, unicorn_engine::uc_error> {
            let pending: Rc<Cell<Option<u64>>> = Rc::new(Cell::new(None));
            for &va in vas {
                // A PLT entry usually shares an already-mapped page; map only missing pages.
                let mut probe = [0u8; 1];
                if uc.mem_read(va, &mut probe).is_err() {
                    uc.mem_map(page_base(va), PAGE, Prot::ALL)?;
                    // `hlt`; never executed, loud fault if the hook assumption breaks.
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
            // Mapping failure is reported as an emulator exit, not a panic.
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

            // Map any other page on first touch, backed by its own address (guest VA == host VA).
            if let Err(e) =
                uc.add_mem_hook(HookType::MEM_UNMAPPED, 1, 0, |uc, _ty, addr, _sz, _v| {
                    let base = page_base(addr);
                    // SAFETY: the host owns these pages; a bad access faults as under the JIT, and an
                    // `Err` (page mapped by a race) just means the retry succeeds or faults.
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
            // Push the exit trampoline at [rsp] in host memory (the identity hook backs Unicorn's reads).
            // SAFETY: `rsp` is a live host stack address.
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
                // Reached the exit trampoline or exhausted the budget.
                let Some(va) = pending.get() else {
                    break Ok(());
                };
                let Some(table) = imports.as_deref_mut() else {
                    break Ok(());
                };

                // Match Microlift's view: arguments in place, `rsp` past the pushed return address
                // (Microlift pushes none).
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
                // Outermost import returns straight to the trampoline; nothing left to emulate.
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

    /// [`differential_unicorn`] with microlift lifting under `spec`: checks specialized lowerings
    /// against hardware (`tests/specialize_tests.rs` proves specialized == portable).
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
