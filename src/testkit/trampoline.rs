use super::oracle::{
    CpuOracle, CpuState, DiffOptions, MemImage, MicroliftOracle, OracleExit, OracleOutcome,
    StateDiff,
};
use super::{LiftTest, RetAddr};
use crate::context::{ExitReason, RegisterContext};

/// Default cap on how many import calls a single run may service before it is
/// declared runaway. Guards against a lifting bug that loops on the import exit
/// instead of making progress.
pub const DEFAULT_IMPORT_CALL_BUDGET: usize = 4096;

/// The set of guest addresses that trap to the host, plus the handler that
/// services them.
///
/// The handler receives the callee VA and the guest state *as an ordinary
/// function would see it at its entry point*: arguments in the ABI registers,
/// the stack pointer already past the pushed return address. It writes results
/// back into the same [`CpuState`] (`rax`, and any host memory it wants to
/// touch - guest VA == host VA under both oracles). `rip` is managed by the
/// engine; a handler must not set it.
///
/// One table drives one run: it borrows the handler mutably and counts calls,
/// so [`differential_with_imports`] builds a fresh handler per engine.
pub struct ImportTable<'h> {
    vas: Vec<u64>,
    handler: &'h mut dyn FnMut(u64, &mut CpuState),
    calls: usize,
    budget: usize,
}

impl<'h> ImportTable<'h> {
    /// Trap every VA in `vas` to `handler`.
    pub fn new(
        vas: impl IntoIterator<Item = u64>,
        handler: &'h mut dyn FnMut(u64, &mut CpuState),
    ) -> Self {
        Self {
            vas: vas.into_iter().collect(),
            handler,
            calls: 0,
            budget: DEFAULT_IMPORT_CALL_BUDGET,
        }
    }

    /// Override [`DEFAULT_IMPORT_CALL_BUDGET`].
    pub fn budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// The trapped addresses.
    pub fn vas(&self) -> &[u64] {
        &self.vas
    }

    /// Whether `va` is trapped.
    pub fn contains(&self, va: u64) -> bool {
        self.vas.contains(&va)
    }

    /// How many calls have been serviced so far in this run.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Run the handler for `va` against `state`. Returns `false` once the call
    /// budget is exhausted, at which point the engine abandons the run.
    pub fn service(&mut self, va: u64, state: &mut CpuState) -> bool {
        if self.calls >= self.budget {
            return false;
        }
        self.calls += 1;
        (self.handler)(va, state);
        true
    }
}

/// A host-allocated guest stack at a fixed address, so two oracles can be run
/// against *the same* stack memory and their `rsp` values compared directly.
///
/// The allocation lives in the `HostStack`; keep it alive for the whole
/// comparison. Both oracles push the exit trampoline at `[rsp]` themselves.
pub struct HostStack {
    mem: Vec<u64>,
}

impl HostStack {
    /// Allocate a `words * 8` byte guest stack.
    pub fn new(words: usize) -> Self {
        Self {
            mem: vec![0u64; words],
        }
    }

    /// The initial `rsp`: one slot below the top, where the exit trampoline
    /// return address is written.
    pub fn rsp(&self) -> u64 {
        let top = self.mem.as_ptr() as u64 + (self.mem.len() as u64) * 8;
        top - 8
    }
}

/// A [`CpuOracle`] that can additionally service host calls through an
/// [`ImportTable`].
pub trait ImportOracle: CpuOracle {
    /// Like [`CpuOracle::run`], but calls to any VA in `imports` are trapped and
    /// serviced by its handler instead of being executed as guest code.
    fn run_with_imports(
        &mut self,
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        mem: &[MemImage],
        max_insns: u64,
        imports: &mut ImportTable<'_>,
    ) -> OracleOutcome;
}

impl ImportOracle for MicroliftOracle {
    fn run_with_imports(
        &mut self,
        code: &[u8],
        code_va: u64,
        init: &CpuState,
        _mem: &[MemImage],
        _max_insns: u64,
        imports: &mut ImportTable<'_>,
    ) -> OracleOutcome {
        let mut ctx = RegisterContext::default();
        init.apply_to_ctx(&mut ctx);

        let vas = imports.vas().to_vec();
        let mut test = LiftTest::new(code)
            .at(code_va)
            .image_base(code_va)
            .opt(self.opt)
            .ctx(ctx)
            .ret_addr(RetAddr::Trampoline)
            // The automatic cross-check compares a *single* entry; an import
            // run is a multi-entry loop and is diffed by the caller instead.
            .no_unicorn();
        for va in vas {
            test = test.with_config(move |c| c.register_slow_import(va));
        }

        let mut harness = test.compile().expect("testkit: lift/compile failed");
        loop {
            let exit = harness.enter();
            let mut state = CpuState::from_ctx(&harness.ctx);
            match exit {
                ExitReason::Return | ExitReason::None => return OracleOutcome::Halted(state),
                ExitReason::ImportedFunctionCall => {
                    // Microlift leaves `rsp` untouched and parks the resume PC
                    // in `rip`, so `state` is already the handler's view.
                    let va = harness.ctx.imported_function_va;
                    if !imports.service(va, &mut state) {
                        return OracleOutcome::Exit {
                            state,
                            reason: OracleExit::NonDiffable(ExitReason::ImportedFunctionCall),
                        };
                    }
                    state.apply_to_ctx(&mut harness.ctx);
                }
                other => {
                    return OracleOutcome::Exit {
                        state,
                        reason: OracleExit::NonDiffable(other),
                    };
                }
            }
        }
    }
}

/// Run `code` on both oracles with the same import trampolines and diff the
/// resulting states. `None` means one engine hit a non-diffable event (a VM
/// exit Microlift cannot service, an emulator fault), so there is nothing sound
/// to compare - the same contract as [`super::oracle::differential`].
///
/// `handler_factory` is invoked once per engine: the runs are sequential and
/// each gets its own handler, so a test can count calls or record arguments per
/// engine without the two interleaving.
pub fn differential_oracles_with_imports<A, B, H, F>(
    a: &mut A,
    b: &mut B,
    code: &[u8],
    code_va: u64,
    init: &CpuState,
    mem: &[MemImage],
    max_insns: u64,
    opts: DiffOptions,
    import_vas: &[u64],
    mut handler_factory: F,
) -> Option<StateDiff>
where
    A: ImportOracle,
    B: ImportOracle,
    F: FnMut() -> H,
    H: FnMut(u64, &mut CpuState),
{
    let oa = {
        let mut h = handler_factory();
        let mut table = ImportTable::new(import_vas.iter().copied(), &mut h);
        a.run_with_imports(code, code_va, init, mem, max_insns, &mut table)
    };
    let ob = {
        let mut h = handler_factory();
        let mut table = ImportTable::new(import_vas.iter().copied(), &mut h);
        b.run_with_imports(code, code_va, init, mem, max_insns, &mut table)
    };
    if !oa.is_halted() || !ob.is_halted() {
        return None;
    }
    Some(super::oracle::diff_states(oa.state(), ob.state(), opts))
}

/// [`differential_oracles_with_imports`] against the default pair: the Microlift
/// JIT versus Unicorn.
#[cfg(feature = "unicorn-oracle")]
pub fn differential_with_imports<H, F>(
    code: &[u8],
    code_va: u64,
    init: &CpuState,
    max_insns: u64,
    opts: DiffOptions,
    import_vas: &[u64],
    handler_factory: F,
) -> Option<StateDiff>
where
    F: FnMut() -> H,
    H: FnMut(u64, &mut CpuState),
{
    let mut ml = MicroliftOracle::default();
    let mut uc = super::oracle::unicorn::UnicornOracle::new();
    differential_oracles_with_imports(
        &mut ml,
        &mut uc,
        code,
        code_va,
        init,
        &[],
        max_insns,
        opts,
        import_vas,
        handler_factory,
    )
}
