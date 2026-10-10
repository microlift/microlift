use std::sync::{Arc, RwLock};

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::ExitReason;
use microlift::arch::x86_64::pin;
use microlift::core::link::{BlockLinkConfig, LinkSiteKind};
use microlift::core::pin::PinnedAbi;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::runtime::{CodeCache, Step, ThreadRuntime, Translation};
use microlift::testkit::prelude::*;
use microlift::testkit::DEFAULT_LIFT_BASE;

const TARGET: u64 = 0x200C;
const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// `mov rax, 1 ; jmp 0x200c`: a lift-time-known destination, so it gets a cell.
#[rustfmt::skip]
const STATIC_JUMP: &[u8] = &[
    0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
    0xE9, 0x00, 0x10, 0x00, 0x00,              // jmp 0x200c
];

const CALLEE: &[u8] = &[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3];

fn config(cells: bool) -> impl Fn(&mut LifterConfig) {
    move |c: &mut LifterConfig| {
        c.block_link = Some(link_config());
        c.link_cells = cells;
    }
}

fn caller(cells: bool) -> Translation {
    LiftTest::new(STATIC_JUMP)
        .with_config(config(cells))
        .translation()
        .expect("compiles")
}

fn callee(cells: bool) -> Translation {
    LiftTest::new(CALLEE)
        .at(TARGET)
        .image_base(TARGET)
        .with_config(config(cells))
        .translation()
        .expect("compiles")
}

struct Guest {
    runtime: ThreadRuntime,
    _stack: Vec<u64>,
}

impl Guest {
    fn new(cache: &Arc<RwLock<CodeCache>>) -> Self {
        let stack = vec![0u64; 1024];
        let top = stack.as_ptr() as u64 + (stack.len() as u64) * 8;
        let mut runtime = ThreadRuntime::new(link_config(), Arc::clone(cache));
        runtime.ctx.rsp = top - 8;
        // SAFETY: inside `stack`, which this struct owns.
        unsafe { *(runtime.ctx.rsp as *mut u64) = microlift::testkit::EXIT_TRAMPOLINE };
        runtime.ctx.rip = DEFAULT_LIFT_BASE;
        Self {
            runtime,
            _stack: stack,
        }
    }
}

#[test]
fn a_direct_jump_out_of_the_region_gets_a_cell() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(STATIC_JUMP, DEFAULT_LIFT_BASE, config).unwrap();

    assert_eq!(code.link_sites.len(), 1);
    assert_eq!(code.link_sites[0].target, TARGET);
    assert_eq!(code.link_sites[0].kind, LinkSiteKind::StaticEdge);
}

#[test]
fn the_cell_array_is_reached_by_an_absolute_address() {
    use cranelift_codegen::ir::GlobalValueData;

    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(STATIC_JUMP, DEFAULT_LIFT_BASE, config).unwrap();

    let symbols: Vec<bool> = code
        .func
        .global_values
        .values()
        .filter_map(|gv| match gv {
            GlobalValueData::Symbol { colocated, .. } => Some(*colocated),
            _ => None,
        })
        .collect();
    assert_eq!(symbols.len(), 1, "expected one cell-array symbol");
    assert!(
        !symbols[0],
        "the cell array is colocated, so it is reached by an `adrp` that a JIT \
         cannot promise is in range"
    );
}

#[test]
fn a_dynamic_target_gets_no_cell() {
    // Control: a RET to a popped address has no lift-time name, so it must get no cell.
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        exit_trampoline_address: None,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLEE, TARGET, config).unwrap();

    assert!(
        code.link_sites.is_empty(),
        "a `ret` must stay on the table probe: {:?}",
        code.link_sites
    );
}

#[test]
fn the_switch_being_off_emits_nothing() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: false,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(STATIC_JUMP, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(code.link_sites.is_empty());
}

#[test]
fn cells_need_block_linking_to_mean_anything() {
    // Cells without block linking are inert, following `block_link_offsets`.
    let config = LifterConfig {
        block_link: None,
        link_cells: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(STATIC_JUMP, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(code.link_sites.is_empty());
}

#[test]
fn two_branches_to_one_address_share_a_cell() {
    // `jz 0x200c ; jmp 0x200c`: edges to one destination share one cell and link together.
    #[rustfmt::skip]
    let bytes: &[u8] = &[
        0x0F, 0x84, 0x06, 0x10, 0x00, 0x00,  // jz  0x200c
        0xE9, 0x01, 0x10, 0x00, 0x00,        // jmp 0x200c
    ];
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).unwrap();

    assert_eq!(
        code.link_sites.len(),
        1,
        "one cell for one destination: {:?}",
        code.link_sites
    );
    assert_eq!(code.link_sites[0].target, TARGET);
}

/// With `link_table` null, only a cell can run the callee.
#[test]
fn a_static_edge_links_with_no_table_at_all() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    // Callee first so the caller's cell is filled at insert time.
    cache.write().unwrap().insert(callee(true));
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rax, 1, "the caller ran");
    assert_eq!(guest.runtime.ctx.rbx, 7, "and the cell reached the callee");
    assert_eq!(cache.read().unwrap().stats().cells_installed, 1);
}

#[test]
fn without_cells_the_same_program_needs_the_table() {
    // Cells off: the callee is unreachable with a null table.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    cache.write().unwrap().insert(callee(false));
    let caller = cache.write().unwrap().insert(caller(false));
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rax, 1);
    assert_eq!(guest.runtime.ctx.rbx, 0, "the callee did not run");
    assert_eq!(guest.runtime.ctx.rip, TARGET);
}

#[test]
fn an_uninstalled_cell_falls_through_to_the_probe() {
    // A cell is empty until its target compiles; the probe serves the edge until then.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    assert_eq!(cache.read().unwrap().cell_counts(), (0, 1), "one pending");

    let callee = cache.write().unwrap().insert(callee(true));
    guest.runtime.publish(&callee);
    // Inserting the target fills the waiting cell.
    assert_eq!(cache.read().unwrap().cell_counts(), (1, 0));

    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);
}

#[test]
fn a_cell_names_the_link_target_and_the_image_base() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let callee = cache.write().unwrap().insert(callee(true));
    let caller = cache.write().unwrap().insert(caller(true));

    let cell = caller.read_cell(0);
    assert_eq!(cell.guest_pc, TARGET);
    assert_eq!(cell.code, callee.link_target as u64);
    assert_eq!(
        cell.image_base, TARGET,
        "the cell carries `image_base` because on the pinned path there is no \
         host to set it"
    );
}

#[test]
fn invalidating_the_target_clears_the_cell_that_named_it() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    cache.write().unwrap().insert(callee(true));
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);

    guest.runtime.invalidate_range(TARGET, 8);
    assert_eq!(cache.read().unwrap().stats().cells_retracted, 1);
    assert_eq!(caller.read_cell(0).code, 0, "the cell was cleared");

    guest.runtime.ctx.rip = DEFAULT_LIFT_BASE;
    guest.runtime.ctx.rbx = 0;
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::UnknownJump)
    );
    assert_eq!(guest.runtime.ctx.rbx, 0, "the freed translation did not run");
    assert_eq!(guest.runtime.ctx.rip, TARGET);
}

#[test]
fn a_retracted_cell_relinks_when_the_address_is_translated_again() {
    // Retraction returns the cell to pending, so no re-lift is needed to link again.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    cache.write().unwrap().insert(callee(true));
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);

    guest.runtime.invalidate_range(TARGET, 8);
    assert_eq!(cache.read().unwrap().cell_counts(), (0, 1));

    cache.write().unwrap().insert(callee(true));
    assert_eq!(cache.read().unwrap().cell_counts(), (1, 0));
    assert_ne!(caller.read_cell(0).code, 0, "installed again");

    guest.runtime.detach_table();
    // SAFETY: as above.
    assert_eq!(
        unsafe { guest.runtime.step() },
        Step::Exit(ExitReason::Return)
    );
    assert_eq!(guest.runtime.ctx.rbx, 7);
}

#[test]
fn a_dying_region_stops_owning_cells() {
    // Cells owned by an invalidated translation must be forgotten, or an install writes into unmapped memory.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    cache.write().unwrap().insert(callee(true));
    cache.write().unwrap().insert(caller(true));
    assert_eq!(cache.read().unwrap().cell_counts(), (1, 0));

    cache
        .write()
        .unwrap()
        .invalidate_range(DEFAULT_LIFT_BASE, STATIC_JUMP.len() as u64);

    assert_eq!(
        cache.read().unwrap().cell_counts(),
        (0, 0),
        "the dying region's cell is gone from both maps"
    );
    assert_eq!(cache.write().unwrap().reclaim(), 1, "and it can be freed");
}

/// Pinned arm: a hit loads `image_base` and `code` from the cell and tail-calls; no host is there to set `image_base`.
#[test]
fn a_pinned_cell_tail_calls_its_successor() {
    fn abi() -> PinnedAbi {
        PinnedAbi::new()
            .with(pin::RSP)
            .with(pin::RAX)
            .with(pin::RBX)
    }
    let pinned = move |c: &mut LifterConfig| {
        c.block_link = Some(BlockLinkConfig::new(ROWS).pinned(abi()));
        c.link_cells = true;
    };

    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);

    let callee = LiftTest::new(CALLEE)
        .at(TARGET)
        .image_base(TARGET)
        .with_config(pinned)
        .translation()
        .expect("compiles");
    let caller = LiftTest::new(STATIC_JUMP)
        .with_config(pinned)
        .translation()
        .expect("compiles");

    // Under a pinned ABI the call address differs from the link address; the cell names the latter.
    assert_ne!(
        callee.entry, callee.link_target,
        "the fixture must actually be pinned"
    );

    let callee = cache.write().unwrap().insert(callee);
    let caller = cache.write().unwrap().insert(caller);
    assert_eq!(caller.read_cell(0).code, callee.link_target as u64);

    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rax, 1);
    assert_eq!(guest.runtime.ctx.rbx, 7, "the tail call arrived");
    assert_eq!(
        guest.runtime.ctx.link_hit, 0,
        "a pinned link is invisible to the host"
    );
    assert_eq!(
        cache.read().unwrap().stats().links_followed,
        0,
        "and so the dispatch loop had nothing to follow"
    );
}


fn text_len(bytes: &[u8], cells: bool, triple: &str) -> usize {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: cells,
        ..LifterConfig::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    // Override the baked host convention so this measures lowering.
    code.func.signature.call_conv = cranelift_codegen::isa::CallConv::SystemV;
    microlift::backend::ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
        .expect("compile")
        .len()
}

fn n_static_exits(n: usize) -> Vec<u8> {
    // Conditional, not `jmp`: unconditional runs are deleted as unreachable.
    let mut bytes = Vec::new();
    for i in 0..n {
        bytes.extend_from_slice(&[0x0F, 0x84]);
        bytes.extend_from_slice(&(0x10000 + (i as i32) * 0x100).to_le_bytes());
    }
    bytes.push(0xC3);
    bytes
}

/// Cell size budget: cells are a fast path bolted in front of the probe, so this bounds the cost rather than showing a win.
#[test]
fn cells_cost_a_bounded_amount_of_code_and_data_per_exit() {
    // Ceiling with room for codegen drift.
    const CEILING_PER_EXIT: i64 = 64;

    for (triple, observed) in [
        ("x86_64-unknown-linux-gnu", 42),
        ("aarch64-unknown-linux-gnu", 28),
    ] {
        for n in [1usize, 4, 16] {
            let program = n_static_exits(n);
            let off = text_len(&program, false, triple) as i64;
            let on = text_len(&program, true, triple) as i64;

            let per_exit = (on - off) / n as i64;
            assert!(
                per_exit > 0,
                "{triple}/{n}: cells are expected to *add* code ({on} vs {off}); if this \
                 ever goes negative the fallback probe has been dropped, which would be a \
                 correctness bug and not an optimisation"
            );
            assert!(
                per_exit <= CEILING_PER_EXIT,
                "{triple}/{n}: {per_exit} bytes per exit exceeds the {CEILING_PER_EXIT}-byte \
                 budget (measured at ~{observed})"
            );
        }
    }
}

#[test]
fn the_cell_array_is_thirty_two_bytes_per_distinct_target() {
    // Data cost is per distinct target, not per edge.
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(&n_static_exits(16), DEFAULT_LIFT_BASE, config).unwrap();

    assert_eq!(code.link_sites.len(), 16);
    assert_eq!(
        code.link_sites.len() * microlift::core::link::ENTRY_SIZE as usize,
        512
    );
}
