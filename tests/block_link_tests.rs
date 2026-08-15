//! Block linking: a region resolving its own successor out of the host's table.
//!
//! Every test here drives the same two-region program - one that `ret`s into
//! another - and differs only in what the host put in the table. That is the
//! whole mechanism: the *only* thing lifted code decides is whether a row
//! matches, and these are the cases it has to get right.

use microlift::arch::x86_64::context::RegisterContext;
use microlift::core::exit::{ExitReason, TransferKind};
use microlift::core::link::{BlockLinkConfig, BlockLinkEntry};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE, Harness};

/// Where the second region is lifted. Far enough from the first that no test
/// can confuse one for the other.
const CALLEE: u64 = 0x2000;

/// A table big enough to make a collision between two addresses impossible to
/// arrange by accident, and small enough to build in a test.
const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

/// An empty table, and the geometry lifted code was compiled against.
fn table() -> Vec<BlockLinkEntry> {
    vec![BlockLinkEntry::default(); ROWS as usize]
}

/// Publish `code`/`base` for `pc`, exactly as a host's own installer would.
fn publish(table: &mut [BlockLinkEntry], pc: u64, code: *const u8, base: u64) {
    let slot = link_config().slot(pc);
    table[slot] = BlockLinkEntry {
        guest_pc: pc,
        code: code as u64,
        image_base: base,
        reserved: 0,
    };
}

/// `mov rax, 1 ; ret` - the region that exits, at [`DEFAULT_LIFT_BASE`].
fn caller() -> Harness {
    LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .with_config(|c| {
            c.block_link = Some(link_config());
        })
        .compile()
        .expect("the caller region compiles")
}

/// `mov rbx, 7 ; ret` at [`CALLEE`] - the region a link lands in.
///
/// Compiled with a stack of its own that nothing uses: this harness exists to
/// own compiled code, and every entry into it goes through the caller's context.
fn callee() -> Harness {
    LiftTest::new(&[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3])
        .at(CALLEE)
        .image_base(CALLEE)
        .with_config(|c| {
            c.block_link = Some(link_config());
        })
        .compile()
        .expect("the callee region compiles")
}

/// One turn of the host dispatch loop: take the link the last exit resolved and
/// enter it. Returns the exit reason of the region it entered.
///
/// This is deliberately the whole of what a host has to do on a hit - set
/// `image_base`, clear `link_hit`, call - because that is the claim block
/// linking makes.
fn follow(ctx: &mut RegisterContext) -> ExitReason {
    assert_ne!(ctx.link_hit, 0, "expected a link to follow");
    // SAFETY: `link_hit` is an address lifted code took out of the table this
    // test built and still owns.
    let entry = unsafe { *(ctx.link_hit as *const BlockLinkEntry) };
    ctx.image_base = entry.image_base;
    ctx.link_hit = 0;
    ctx.exit_reason = ExitReason::None as u32;
    // SAFETY: `code` is the entry point of a region this test compiled, whose
    // signature is `extern "C" fn(*mut RegisterContext)`.
    let code: unsafe extern "C" fn(*mut RegisterContext) =
        unsafe { std::mem::transmute(entry.code as *const u8) };
    unsafe { code(ctx as *mut RegisterContext) };
    ExitReason::try_from(ctx.exit_reason).unwrap_or(ExitReason::None)
}

/// Arrange the caller's stack so its `RET` goes to `ret_to`, and enter.
fn run_caller(harness: &mut Harness, ret_to: u64, table: &mut [BlockLinkEntry]) -> ExitReason {
    // Two words, not one: the caller's `RET` pops `ret_to`, and whatever the
    // link lands in then finds the trampoline under it and can `ret` out to the
    // host. The harness seeds `rsp` at the *top* word of its stack, so the
    // second word has to be made by moving `rsp` down rather than by writing
    // above it - which is off the end of the allocation.
    harness.ctx.rsp -= 8;
    // SAFETY: both words are inside the harness's own stack allocation, which
    // it owns for as long as it is alive.
    unsafe {
        *(harness.ctx.rsp as *mut u64) = ret_to;
        *((harness.ctx.rsp + 8) as *mut u64) = EXIT_TRAMPOLINE;
    }
    harness.ctx.link_table = table.as_mut_ptr();
    harness.ctx.link_hit = 0;
    harness.enter()
}

/// The base case: a `RET` out of the region finds its successor and the host
/// does no lookup at all.
#[test]
fn a_ret_out_of_the_region_links_to_the_region_that_covers_it() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

    // The exit is the same one the host has always seen - linking adds an
    // answer, it does not change the question.
    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(caller.ctx.exit_code),
        Ok(TransferKind::Return)
    );
    assert_eq!(caller.ctx.rip, CALLEE);
    assert_eq!(caller.ctx.rax, 1);
    assert_eq!(
        caller.ctx.link_hit,
        &table[link_config().slot(CALLEE)] as *const _ as u64,
        "the probe should name the row it matched"
    );

    // And the row is enterable: following it runs the callee against the same
    // context, which is the entire point.
    let exit = follow(&mut caller.ctx);
    assert_eq!(caller.ctx.rbx, 7);
    assert_eq!(caller.ctx.rax, 1, "the callee inherits the caller's state");
    assert_eq!(exit, ExitReason::Return, "the callee ret'd to the sentinel");
}

/// A host that has published nothing for the address gets the behaviour it had
/// before linking existed.
#[test]
fn an_address_with_no_row_does_not_link() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    // A row for a *different* address, in a different slot: a populated table
    // is not the same as a matching one.
    publish(&mut table, CALLEE + 0x1000, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(caller.ctx.rip, CALLEE);
    assert_eq!(caller.ctx.link_hit, 0);
}

/// A collision - the right slot holding the wrong address - is a miss and not a
/// wrong answer. This is what makes a direct-mapped table with no eviction
/// policy sound.
#[test]
fn a_colliding_row_is_a_miss_rather_than_a_wrong_jump() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    // Same slot (the table is indexed on `pc >> 4`, masked to 256 rows), a
    // different address.
    let colliding = CALLEE + (ROWS as u64) * 16;
    assert_eq!(link_config().slot(colliding), link_config().slot(CALLEE));
    publish(&mut table, colliding, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(caller.ctx.link_hit, 0, "a colliding row must not be taken");
    assert_eq!(caller.ctx.rbx, 0, "and the callee must not have run");
}

/// A thread with no table published runs exactly as it did before.
#[test]
fn a_null_table_is_not_a_special_case() {
    let mut untabled = caller();
    // SAFETY: as in `run_caller`.
    unsafe { *(untabled.ctx.rsp as *mut u64) = CALLEE };
    // No table: the field is what a `RegisterContext::new` leaves it as.
    untabled.ctx.link_table = std::ptr::null_mut();
    untabled.ctx.link_hit = 0;
    assert_eq!(untabled.enter(), ExitReason::UnknownJump);
    assert_eq!(untabled.ctx.rip, CALLEE);
    assert_eq!(untabled.ctx.link_hit, 0);
}

/// **The exclusion that matters.** A `RET` to the host's exit trampoline means
/// "the call you made is finished", and the trampoline address is not a guest
/// address at all. Even with a row published for it - which a host would never
/// do, but which is exactly the mistake that would be silent - the region must
/// return to the host.
#[test]
fn a_return_to_the_exit_trampoline_never_links() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    publish(&mut table, EXIT_TRAMPOLINE, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, EXIT_TRAMPOLINE, &mut table);

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(caller.ctx.link_hit, 0);
    assert_eq!(caller.ctx.rbx, 0, "the callee must not have run");
}

/// A direct `CALL` to an address outside the region leaves `rip` at the *return*
/// address and the callee in `imported_function_va` - so a link there has to
/// commit the callee as the PC, or the host would resume past the call it never
/// made.
#[test]
fn a_direct_call_out_of_region_links_on_the_callee_and_commits_it() {
    // 0x1000: e8 fb 0f 00 00   call 0x2000
    // 0x1005: 90               nop - the return address, never reached here
    let mut caller = LiftTest::new(&[0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x90])
        .with_config(|c| {
            c.block_link = Some(link_config());
        })
        .compile()
        .expect("the calling region compiles");
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0;
    let exit = caller.enter();

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(caller.ctx.exit_code),
        Ok(TransferKind::DirectCall)
    );
    assert_eq!(caller.ctx.imported_function_va, CALLEE);
    assert_ne!(caller.ctx.link_hit, 0, "a direct call out of region links");
    assert_eq!(
        caller.ctx.rip, CALLEE,
        "the link must commit the callee as the PC, not the return address"
    );

    // The return address the `CALL` pushed is still on the stack, so the
    // callee's own `ret` goes back to the instruction after it.
    let return_address = DEFAULT_LIFT_BASE + 5;
    // SAFETY: `rsp` points into the caller harness's own stack.
    assert_eq!(unsafe { *(caller.ctx.rsp as *const u64) }, return_address);
    follow(&mut caller.ctx);
    assert_eq!(caller.ctx.rbx, 7);
    assert_eq!(caller.ctx.rip, return_address);
}

/// Linking is off by default, and a lift that did not ask for it emits nothing:
/// the same program leaves `link_hit` alone even with a matching table.
#[test]
fn without_the_config_nothing_probes() {
    let mut caller = LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .compile()
        .expect("compiles");
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    // SAFETY: `rsp` is inside the harness's own stack allocation.
    unsafe { *(caller.ctx.rsp as *mut u64) = CALLEE };
    caller.ctx.link_table = table.as_mut_ptr();
    // A sentinel a host would never write, to prove nothing wrote over it.
    caller.ctx.link_hit = 0xDEAD;

    assert_eq!(caller.enter(), ExitReason::UnknownJump);
    assert_eq!(caller.ctx.link_hit, 0xDEAD);
}
