use super::oracle::{
    CpuOracle, CpuState, DiffOptions, MemImage, MicroliftOracle, OracleExit, OracleOutcome,
    StateDiff,
};
use super::{LiftTest, RetAddr};
use crate::context::{ExitReason, RegisterContext};

/// Cap on import calls per run before it is declared runaway.
pub const DEFAULT_IMPORT_CALL_BUDGET: usize = 4096;

/// Guest addresses that trap to the host, plus the handler servicing them.
///
/// The handler sees the callee VA and the state as at function entry (args in ABI registers, `rsp`
/// past the return address) and writes results back; it must not set `rip`. One table drives one
/// run: [`differential_with_imports`] builds a fresh handler per engine.
pub struct ImportTable<'h> {
    vas: Vec<u64>,
    handler: &'h mut dyn FnMut(u64, &mut CpuState),
    calls: usize,
    budget: usize,
}

impl<'h> ImportTable<'h> {
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

    pub fn budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    pub fn vas(&self) -> &[u64] {
        &self.vas
    }

    pub fn contains(&self, va: u64) -> bool {
        self.vas.contains(&va)
    }

    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Runs the handler for `va`; `false` once the call budget is exhausted.
    pub fn service(&mut self, va: u64, state: &mut CpuState) -> bool {
        if self.calls >= self.budget {
            return false;
        }
        self.calls += 1;
        (self.handler)(va, state);
        true
    }
}

/// Host-allocated guest stack at a fixed address, so two oracles share stack memory and `rsp` is
/// comparable. Keep it alive for the whole comparison.
pub struct HostStack {
    mem: Vec<u64>,
}

impl HostStack {
    pub fn new(words: usize) -> Self {
        Self {
            mem: vec![0u64; words],
        }
    }

    /// Initial `rsp`: one slot below the top, where the exit trampoline address goes.
    pub fn rsp(&self) -> u64 {
        let top = self.mem.as_ptr() as u64 + (self.mem.len() as u64) * 8;
        top - 8
    }
}

/// A [`CpuOracle`] that can also service host calls through an [`ImportTable`].
pub trait ImportOracle: CpuOracle {
    /// Like [`CpuOracle::run`], but calls to `imports` are serviced by its handler, not executed.
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
            // The automatic cross-check compares a single entry; import runs are diffed by the caller.
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
                    // Microlift leaves `rsp` untouched and parks the resume PC in `rip`.
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

/// Runs `code` on both oracles with the same import trampolines and diffs the states; `None` when
/// an engine hit a non-diffable event. `handler_factory` is called once per engine.
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

/// [`differential_oracles_with_imports`] on the default pair: Microlift JIT versus Unicorn.
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
