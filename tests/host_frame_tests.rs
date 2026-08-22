//! Host call frames: a guest `CALL`/`RET` pair as a real host call and return.
//!
//! See [`microlift::core::hostframe`] for why this has no shadow stack and how a
//! return that nobody expected unwinds. What is checked here is that it *happens*
//! and that the unwinding is correct, and the control that carries the file is the
//! same one the cell tests use: with `ctx.link_table` null and the ring off, a
//! guest that gets all the way back to `main` can only have done it through host
//! frames.

use std::sync::{Arc, RwLock};

use microlift::arch::x86_64::X86_64;
use microlift::core::exit::ExitReason;
use microlift::core::hostframe::HostFrameConfig;
use microlift::core::link::BlockLinkConfig;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::runtime::{CodeCache, Step, ThreadRuntime, Translation};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

const CALLEE: u64 = 0x2000;
/// The address the call comes back to: `DEFAULT_LIFT_BASE` plus the `call`'s five
/// bytes.
const RETURN_TO: u64 = DEFAULT_LIFT_BASE + 5;
const ROWS: u32 = 256;

/// `call 0x2000 ; mov rax, 1 ; ret`
#[rustfmt::skip]
const CALLER: &[u8] = &[
    0xE8, 0xFB, 0x0F, 0x00, 0x00,              // call 0x2000
    0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
    0xC3,                                      // ret
];

/// `mov rbx, 7 ; ret`
const CALLEE_CODE: &[u8] = &[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3];

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

fn config(frames: bool) -> impl Fn(&mut LifterConfig) {
    move |c: &mut LifterConfig| {
        c.block_link = Some(link_config());
        c.link_cells = true;
        c.host_call_frames = frames.then(|| HostFrameConfig::new(64));
    }
}

fn caller(frames: bool) -> Translation {
    LiftTest::new(CALLER)
        .with_config(config(frames))
        .translation()
        .expect("compiles")
}

fn callee(frames: bool) -> Translation {
    LiftTest::new(CALLEE_CODE)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(config(frames))
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
        // The guest `call` pushes below this, so leave it a word of headroom.
        runtime.ctx.rsp = top - 8;
        // SAFETY: inside `stack`, which this struct owns.
        unsafe { *(runtime.ctx.rsp as *mut u64) = EXIT_TRAMPOLINE };
        runtime.ctx.rip = DEFAULT_LIFT_BASE;
        Self {
            runtime,
            _stack: stack,
        }
    }
}

/// Both regions in the cache, the caller published, the callee's cell installed.
fn seeded(frames: bool) -> (Arc<RwLock<CodeCache>>, Guest) {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    cache.write().unwrap().insert(callee(frames));
    let caller = cache.write().unwrap().insert(caller(frames));
    guest.runtime.publish(&caller);
    (cache, guest)
}

#[test]
fn an_out_of_region_call_becomes_a_host_call() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        host_call_frames: Some(HostFrameConfig::new(64)),
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();
    let ir = code.func.display().to_string();

    assert!(
        ir.contains("call_indirect"),
        "no nested call was emitted:\n{ir}"
    );
    // The address it comes back to is still a resume point: the fallback path,
    // taken until the callee is compiled, re-enters this region there.
    assert!(code.resume_points.contains(&RETURN_TO));
}

#[test]
fn the_switch_being_off_emits_no_nested_call() {
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        host_call_frames: None,
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(!code.func.display().to_string().contains("call_indirect"));
}

#[test]
fn nesting_needs_cells_to_have_an_address_to_call() {
    // The callee's address comes from a cell, so this is inert without them -
    // the same silent-fallback shape as every other gate here.
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: false,
        host_call_frames: Some(HostFrameConfig::new(64)),
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(CALLER, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(!code.func.display().to_string().contains("call_indirect"));
    assert!(code.link_sites.is_empty());
}

/// **The control that carries this file.** With no table and no ring, a `RET`
/// from the callee can only get back into the caller through the host frame the
/// call left on the stack.
#[test]
fn a_guest_call_and_return_run_on_the_host_stack() {
    let (_cache, mut guest) = seeded(true);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee ran");
    assert_eq!(
        guest.runtime.ctx.rax, 1,
        "and its `ret` resumed the caller after the call"
    );
    assert_eq!(
        guest.runtime.ctx.host_frame_depth, 0,
        "the depth is restored however the call ended"
    );
}

#[test]
fn without_nesting_the_same_return_needs_a_link() {
    // The other half of the control: identical program, nesting off, no table.
    // The callee's `RET` has nothing to return to and must exit.
    let (_cache, mut guest) = seeded(false);
    guest.runtime.detach_table();

    // SAFETY: as above.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee still ran");
    assert_eq!(guest.runtime.ctx.rax, 0, "but the caller did not resume");
    assert_eq!(guest.runtime.ctx.rip, RETURN_TO);
}

#[test]
fn an_uninstalled_cell_falls_back_to_the_ordinary_exit() {
    // A call whose target has not been translated cannot be a host call, and the
    // fallback arm is what serves it until it has.
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: only the caller is entered here.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    // The ordinary out-of-region-call exit, unchanged: `rip` is the *return*
    // address and the callee is in `imported_function_va`, which is the shape a
    // host already services by swapping the two.
    assert_eq!(guest.runtime.ctx.rip, RETURN_TO);
    assert_eq!(guest.runtime.ctx.imported_function_va, CALLEE);
    assert_eq!(guest.runtime.ctx.host_frame_depth, 0);
}

/// A return to an address no frame is waiting for unwinds instead of resuming
/// the wrong place - the property that makes "always return, let the caller
/// decide" correct rather than merely cheap.
#[test]
fn a_return_the_caller_did_not_expect_unwinds() {
    const ELSEWHERE: u64 = 0x3000;
    // `mov rbx, 7 ; mov qword ptr [rsp], 0x3000 ; ret` - returns somewhere the
    // caller is not waiting for.
    #[rustfmt::skip]
    let rogue: &[u8] = &[
        0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0x04, 0x24, 0x00, 0x30, 0x00, 0x00,
        0xC3,
    ];

    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let rogue = LiftTest::new(rogue)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(config(true))
        .translation()
        .expect("compiles");
    cache.write().unwrap().insert(rogue);
    let caller = cache.write().unwrap().insert(caller(true));
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(guest.runtime.ctx.rbx, 7, "the callee ran");
    assert_eq!(
        guest.runtime.ctx.rax, 0,
        "and the caller did NOT resume: the address was not the one it left"
    );
    assert_eq!(step, Step::Exit(ExitReason::UnknownJump));
    assert_eq!(guest.runtime.ctx.rip, ELSEWHERE, "handed to the host as-is");
    assert_eq!(
        guest.runtime.ctx.host_frame_depth, 0,
        "the unwinding frame still restored the depth"
    );
}

// ---------------------------------------------------------------------------
// Depth: the part that is not one call
// ---------------------------------------------------------------------------

/// `DEPTH` regions, each calling the next and adding one on the way back, with
/// the innermost just adding one. A run that gets `rax == DEPTH` executed every
/// level exactly once, in order, and unwound through every one of them.
const DEPTH: u64 = 5;

fn chain(frames: bool, max_depth: u32) -> (Arc<RwLock<CodeCache>>, Guest) {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    // `Guest::new` puts the trampoline *at* `rsp`, so the guest's own pushes go
    // below it and the outermost `ret` finds it last. The stack has ample room
    // beneath for DEPTH return addresses.
    let mut guest = Guest::new(&cache);

    for i in 0..DEPTH {
        let at = if i == 0 { DEFAULT_LIFT_BASE } else { (i + 1) * 0x1000 };
        let bytes: Vec<u8> = if i + 1 == DEPTH {
            // `add rax, 1 ; ret`
            vec![0x48, 0x83, 0xC0, 0x01, 0xC3]
        } else {
            // `call next ; add rax, 1 ; ret`
            let next = (i + 2) * 0x1000;
            let disp = (next as i64 - (at as i64 + 5)) as i32;
            let mut v = vec![0xE8];
            v.extend_from_slice(&disp.to_le_bytes());
            v.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01, 0xC3]);
            v
        };
        let t = LiftTest::new(Box::leak(bytes.into_boxed_slice()))
            .at(at)
            .image_base(at)
            .with_config(move |c: &mut LifterConfig| {
                c.block_link = Some(link_config());
                c.link_cells = true;
                c.host_call_frames = frames.then(|| HostFrameConfig::new(max_depth));
            })
            .translation()
            .expect("compiles");
        let t = cache.write().unwrap().insert(t);
        guest.runtime.publish(&t);
    }
    (cache, guest)
}

/// Drive the chain to completion, however many host dispatches it takes.
///
/// # Safety
///
/// Every translation reachable was compiled by this test for this context.
unsafe fn drive(guest: &mut Guest) -> (ExitReason, usize) {
    for steps in 1..64 {
        // SAFETY: the caller's obligation.
        match unsafe { guest.runtime.step() } {
            Step::Exit(ExitReason::Return) => return (ExitReason::Return, steps),
            Step::Exit(_) => continue,
            Step::NeedsTranslation(pc) => panic!("no translation for {pc:#x}"),
        }
    }
    panic!("the chain did not terminate")
}

#[test]
fn a_chain_of_nested_calls_unwinds_through_every_level() {
    let (_cache, mut guest) = chain(true, 64);
    guest.runtime.detach_table();

    // SAFETY: as `drive`.
    let (exit, steps) = unsafe { drive(&mut guest) };

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(guest.runtime.ctx.rax, DEPTH, "every level ran exactly once");
    assert_eq!(guest.runtime.ctx.host_frame_depth, 0);
    assert_eq!(
        steps, 1,
        "with no table and no host dispatch to fall back on, the whole chain \
         must have run inside one entry"
    );
}

/// **The bound on the host stack, and the thing it must not cost.** A guest
/// nesting deeper than the ceiling gets the same answer; the calls past it take
/// the ordinary linked exit, which is a host dispatch and nothing else.
#[test]
fn a_guest_deeper_than_the_ceiling_still_gets_the_right_answer() {
    let (_cache, mut guest) = chain(true, 2);

    // SAFETY: as `drive`.
    let (exit, steps) = unsafe { drive(&mut guest) };

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(
        guest.runtime.ctx.rax, DEPTH,
        "the ceiling must change how a call is made, never what it computes"
    );
    assert_eq!(guest.runtime.ctx.host_frame_depth, 0);
    assert!(
        steps > 1,
        "a ceiling of 2 under {DEPTH} levels should have forced the host to \
         dispatch at least once; it took {steps} step(s)"
    );
}

#[test]
fn the_answer_is_the_same_with_nesting_off() {
    // The control for both of the above: the same chain, no host frames at all.
    let (_cache, mut guest) = chain(false, 64);

    // SAFETY: as `drive`.
    let (exit, _) = unsafe { drive(&mut guest) };

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(guest.runtime.ctx.rax, DEPTH);
    assert_eq!(
        guest.runtime.ctx.host_frame_depth, 0,
        "and nothing touched the depth, since nothing nested"
    );
}

/// **An exit from a nested frame.** A syscall four calls deep cannot return to
/// the host directly - there are four host frames between it and the thunk - so
/// it unwinds through all of them, each restoring its own depth on the way past.
///
/// This is the case that makes "always return, let the caller decide" load-bearing
/// rather than merely tidy: the intermediate frames are not told what happened and
/// do not need to be. They compare, fail, and return.
#[test]
fn an_exit_from_a_nested_frame_unwinds_to_the_host_and_resumes() {
    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);

    for i in 0..DEPTH {
        let at = if i == 0 { DEFAULT_LIFT_BASE } else { (i + 1) * 0x1000 };
        let bytes: Vec<u8> = if i + 1 == DEPTH {
            // `syscall ; add rax, 1 ; ret` - the exit happens at the bottom.
            vec![0x0F, 0x05, 0x48, 0x83, 0xC0, 0x01, 0xC3]
        } else {
            let next = (i + 2) * 0x1000;
            let disp = (next as i64 - (at as i64 + 5)) as i32;
            let mut v = vec![0xE8];
            v.extend_from_slice(&disp.to_le_bytes());
            v.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01, 0xC3]);
            v
        };
        let t = LiftTest::new(Box::leak(bytes.into_boxed_slice()))
            .at(at)
            .image_base(at)
            .with_config(|c: &mut LifterConfig| {
                c.block_link = Some(link_config());
                c.link_cells = true;
                c.host_call_frames = Some(HostFrameConfig::new(64));
            })
            .translation()
            .expect("compiles");
        let t = cache.write().unwrap().insert(t);
        guest.runtime.publish(&t);
    }

    // SAFETY: every translation was compiled by this test for this context.
    let first = unsafe { guest.runtime.step() };

    assert_eq!(
        first,
        Step::Exit(ExitReason::Syscall),
        "the syscall must reach the host, four frames or not"
    );
    assert_eq!(
        guest.runtime.ctx.host_frame_depth, 0,
        "every frame it unwound through restored its own depth"
    );
    assert_eq!(
        guest.runtime.ctx.rip,
        DEPTH * 0x1000 + 2,
        "and it resumes after the syscall, inside the innermost region"
    );

    // Servicing it is the host's business; here it is a no-op. Resuming from the
    // reported PC has to finish the program, which is what says the unwinding
    // left the guest's own stack intact.
    // SAFETY: as above.
    let (exit, _) = unsafe { drive(&mut guest) };
    assert_eq!(exit, ExitReason::Return);
    assert_eq!(guest.runtime.ctx.rax, DEPTH, "every level still ran once");
    assert_eq!(guest.runtime.ctx.host_frame_depth, 0);
}

/// Nesting under a **pinned** register file, where a region's parameters and
/// results are half the guest's registers and a call has to forward them.
#[test]
fn a_nested_call_carries_a_pinned_register_file() {
    use microlift::arch::x86_64::pin;
    use microlift::core::pin::PinnedAbi;

    let abi = PinnedAbi::new()
        .with(pin::RSP)
        .with(pin::RAX)
        .with(pin::RBX);
    let pinned = move |c: &mut LifterConfig| {
        c.block_link = Some(BlockLinkConfig::new(ROWS).pinned(abi));
        c.link_cells = true;
        c.host_call_frames = Some(HostFrameConfig::new(64));
    };

    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let callee = LiftTest::new(CALLEE_CODE)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(pinned)
        .translation()
        .expect("compiles");
    assert_ne!(
        callee.entry, callee.link_target,
        "the fixture must actually be pinned"
    );
    cache.write().unwrap().insert(callee);
    let caller = cache
        .write()
        .unwrap()
        .insert(LiftTest::new(CALLER).with_config(pinned).translation().unwrap());
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };

    assert_eq!(step, Step::Exit(ExitReason::Return));
    assert_eq!(
        guest.runtime.ctx.rbx, 7,
        "the callee's pinned write came back through the call's results"
    );
    assert_eq!(guest.runtime.ctx.rax, 1, "and the caller resumed after it");
}

/// **The lazy-sync prune survives a nested call, and this is what says so.**
///
/// A caller that writes none of these registers still has to publish what the
/// *callee* wrote to them. The prune deletes the sync store of a register nothing
/// in the region wrote; the claim is that a nested call does not invalidate that,
/// because after the post-call reload such a register holds exactly what the
/// context holds and its store is idempotent either way.
///
/// Being wrong about that is a stale register and not a crash, so it is checked
/// against every register a callee can plausibly write rather than the one or two
/// the other tests here happen to use.
#[test]
fn a_callee_s_writes_survive_a_caller_that_touches_nothing() {
    // Callee: `mov rbx,0xb ; mov rcx,0xc ; mov rdx,0xd ; mov rsi,0x6 ;
    //          mov rdi,0x7 ; mov r8,0x8 ; mov r9,0x9 ; ret`
    #[rustfmt::skip]
    let callee_bytes: &[u8] = &[
        0x48, 0xC7, 0xC3, 0x0B, 0x00, 0x00, 0x00, // rbx
        0x48, 0xC7, 0xC1, 0x0C, 0x00, 0x00, 0x00, // rcx
        0x48, 0xC7, 0xC2, 0x0D, 0x00, 0x00, 0x00, // rdx
        0x48, 0xC7, 0xC6, 0x06, 0x00, 0x00, 0x00, // rsi
        0x48, 0xC7, 0xC7, 0x07, 0x00, 0x00, 0x00, // rdi
        0x49, 0xC7, 0xC0, 0x08, 0x00, 0x00, 0x00, // r8
        0x49, 0xC7, 0xC1, 0x09, 0x00, 0x00, 0x00, // r9
        0xC3,
    ];
    // Caller: `call callee ; mov rax, 1 ; ret` - writes `rax` and nothing else,
    // so every register above is one the prune would drop the store of.
    let caller_bytes = CALLER;

    let cache = Arc::new(RwLock::new(CodeCache::new()));
    let mut guest = Guest::new(&cache);
    let callee = LiftTest::new(callee_bytes)
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(config(true))
        .translation()
        .expect("compiles");
    cache.write().unwrap().insert(callee);
    let caller = cache
        .write()
        .unwrap()
        .insert(LiftTest::new(caller_bytes).with_config(config(true)).translation().unwrap());
    guest.runtime.publish(&caller);
    guest.runtime.detach_table();

    // SAFETY: both translations were compiled by this test for this context.
    let step = unsafe { guest.runtime.step() };
    assert_eq!(step, Step::Exit(ExitReason::Return));

    let ctx = &guest.runtime.ctx;
    assert_eq!(ctx.rax, 1, "the caller's own write");
    for (name, got, want) in [
        ("rbx", ctx.rbx, 0xb),
        ("rcx", ctx.rcx, 0xc),
        ("rdx", ctx.rdx, 0xd),
        ("rsi", ctx.rsi, 0x6),
        ("rdi", ctx.rdi, 0x7),
        ("r8", ctx.r8, 0x8),
        ("r9", ctx.r9, 0x9),
    ] {
        assert_eq!(
            got, want,
            "{name} lost the callee's write: the caller pruned a store it needed"
        );
    }
}

/// A call whose **resume point is past the end of the region** does not nest.
///
/// There is nowhere in this region for such a call to come back to: `finish`
/// turns the address into an exit stub, and returning into one would sync the
/// caller's pre-call register file over everything the callee wrote. Nesting is
/// an optimisation, so it declines rather than working around it.
#[test]
fn a_call_at_the_end_of_a_region_does_not_nest() {
    // `call 0x9000` and nothing after it: the resume point is the byte past the
    // region.
    let bytes: &[u8] = &[0xE8, 0xFB, 0x7F, 0x00, 0x00];
    let config = LifterConfig {
        block_link: Some(link_config()),
        link_cells: true,
        host_call_frames: Some(HostFrameConfig::new(64)),
        ..LifterConfig::default()
    };
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config).unwrap();

    assert!(
        !code.func.display().to_string().contains("call_indirect"),
        "a call with no in-region resume point must take the ordinary exit"
    );
    // It is still a *static edge*, so it still gets a cell - only the nesting is
    // declined.
    assert!(code.link_sites.iter().any(|s| s.target == 0x9000));
}
