//! Link cells: exits whose destination is known at lift time linking through a
//! dedicated row instead of hashing a PC into the shared table.
//!
//! Microlift's answer to FEX's backpatched `b`/`bl`. The control that carries
//! this whole file is [`a_static_edge_links_with_no_table_at_all`]: it removes
//! the *only* other mechanism, so a region that reaches its successor there
//! reached it through a cell and nothing else.

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

/// Where the static edge lands.
const TARGET: u64 = 0x200C;
const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// `mov rax, 1 ; jmp 0x200c` - a direct jump out of the region, which is a
/// destination the lifter knows at lift time and so gets a cell.
#[rustfmt::skip]
const STATIC_JUMP: &[u8] = &[
    0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
    0xE9, 0x00, 0x10, 0x00, 0x00,              // jmp 0x200c
];

/// `mov rbx, 7 ; ret` at [`TARGET`].
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
        // SAFETY: inside `stack`, which this struct owns. The callee `ret`s to
        // the trampoline and leaves for the host.
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

/// The cell array is referenced as a **far** symbol, and that is not a tuning
/// choice.
///
/// `colocated` reads like "is it in the same object", and the array is - but what
/// the aarch64 backend does with it is pick `adrp`/`add` over an absolute literal,
/// and `adrp` reaches ±4 GiB. A JIT does not honour that: `cranelift-jit` serves
/// executable and writable allocations from two separate `mmap`s, and in a
/// long-running host those drift apart without bound. Rosetta 3 found this the
/// expensive way - half a minute into a real game, with the two allocations 6 GiB
/// apart and `compiled_blob.rs` panicking on the relocation rather than veneering
/// it the way it would a far `bl`.
///
/// So the property is asserted on the IR rather than on the emitted bytes: this is
/// the field whose value decides it, and it regressed silently once.
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
    // The control for the case above: `ret` leaves for an address popped off the
    // guest stack, which no amount of lift-time knowledge can name. Without this,
    // an implementation that handed out a cell per exit would pass.
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
    // A cell *is* a `BlockLinkEntry`; asking for cells without the rest of block
    // linking is inert rather than an error, following `block_link_offsets`.
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
    // `jz 0x200c ; jmp 0x200c` - two edges, one destination. De-duplication is
    // what keeps a `switch` whose arms all leave the region from costing a cell
    // per edge, and means the second edge links the instant the first does.
    // Both displacements land on TARGET: `jz` is six bytes from 0x1000 and
    // `jmp` is five from 0x1006, so they are not the same immediate.
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

/// **The control that carries this file.** With `link_table` null there is no
/// table to hash into, so the only way the callee can run is a cell.
#[test]
fn a_static_edge_links_with_no_table_at_all() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    // Insert the callee first so the caller's cell is filled at insert time.
    cache.write().unwrap().insert(callee(true));
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    // Everything the table could have done is now impossible.
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
    // The other half of the control above: identical fixture, cells off. The
    // callee is unreachable with a null table, which is what proves the previous
    // test measured the cell rather than something else.
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
    // The fallback is not decorative: a cell is empty until its target is
    // compiled, so the probe is what serves the edge until then.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    // Caller first, so its cell has nothing to point at.
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    assert_eq!(cache.read().unwrap().cell_counts(), (0, 1), "one pending");

    let callee = cache.write().unwrap().insert(callee(true));
    guest.runtime.publish(&callee);
    // Inserting the target filled the waiting cell - the other direction of the
    // installer.
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
    // FEX's `DirectBlockDelinker`, and the reason invalidation needs no epoch on
    // the hot path - behavioural, because asserting the row went to zero would
    // pass against an implementation that zeroed everything.
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

    // And the edge now behaves as if it had never been installed.
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
    // A retraction moves the cell back to pending rather than forgetting it, so
    // the region does not have to be re-lifted to link again.
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
    // The other direction: cells *belonging to* an invalidated translation must
    // be forgotten, or a later install writes into an array about to be unmapped
    // - and the maps would pin the `Arc` so `reclaim` could never free anything.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    cache.write().unwrap().insert(callee(true));
    cache.write().unwrap().insert(caller(true));
    assert_eq!(cache.read().unwrap().cell_counts(), (1, 0));

    // Invalidate the *caller*, which owns the cell.
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

/// The pinned arm of the cell probe: instead of naming the cell for the host, a
/// hit loads `image_base` and `code` out of it and **tail-calls**, carrying the
/// guest register file in machine registers.
///
/// The same code path as the table probe's pinned arm, reached through a cell -
/// and worth its own test because it is the branch where a cell must supply
/// `image_base` itself, there being no host on that path to set it.
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

    // Under a pinned ABI the address to call and the address to link to are
    // different functions, and a cell must name the latter.
    assert_ne!(
        callee.entry, callee.link_target,
        "the fixture must actually be pinned"
    );

    let callee = cache.write().unwrap().insert(callee);
    let caller = cache.write().unwrap().insert(caller);
    assert_eq!(caller.read_cell(0).code, callee.link_target as u64);

    guest.runtime.publish(&caller);
    // No table: the chain can only run itself through the cell.
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

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

/// Compiled `.text` size of `bytes` under `cells`, cross-compiled for `triple`.
fn text_len(bytes: &[u8], cells: bool, triple: &str) -> usize {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: cells,
        ..LifterConfig::default()
    };
    let mut code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).expect("lift");
    // As in `specialize_tests`: the lifter bakes in the *host* calling
    // convention, so override it to measure lowering rather than trip the
    // target's ABI code on a convention it does not accept.
    code.func.signature.call_conv = cranelift_codegen::isa::CallConv::SystemV;
    microlift::backend::ArtifactBackend::compile_lifted_to_raw(code, Some(triple), None)
        .expect("compile")
        .len()
}

/// `n` reachable out-of-region conditional branches, each to its own address.
fn n_static_exits(n: usize) -> Vec<u8> {
    // Conditional, not `jmp`: a run of unconditional jumps makes all but the
    // first unreachable, Cranelift deletes them, and the measurement is of one
    // exit however many were written.
    let mut bytes = Vec::new();
    for i in 0..n {
        bytes.extend_from_slice(&[0x0F, 0x84]);
        bytes.extend_from_slice(&(0x10000 + (i as i32) * 0x100).to_le_bytes());
    }
    bytes.push(0xC3);
    bytes
}

/// **Cells make the compiled region bigger, and this is the budget.**
///
/// Unlike a `Spec`, this cannot prove itself by shrinking: the table probe stays
/// as the fallback - a cell is empty until its target is compiled and empty
/// again after an invalidation - so a cell is a fast path bolted in front of the
/// slow one. What it buys is what the exit *executes* (a load and a branch
/// instead of six instructions, and a lookup that cannot collide), and that is
/// what the behavioural tests above measure by reaching a successor with the
/// table detached.
///
/// So this test exists to keep the cost *bounded and known*, not to show a win.
/// The bound matters: on a page-bounded region of branch-dense code, cells can
/// rival the code itself, and a host budgeting for them needs a number rather
/// than an adjective.
#[test]
fn cells_cost_a_bounded_amount_of_code_and_data_per_exit() {
    // Measured; the assertion is a ceiling with room for codegen drift, not the
    // observation itself.
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
    // The data half of the cost, and the reason it is per *distinct* target: a
    // switch whose arms all leave the region pays once per destination, not once
    // per edge.
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
