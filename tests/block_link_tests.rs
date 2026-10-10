//! Block linking: the only thing lifted code decides is whether a table row matches.
//! Each test drives the same two-region program and varies what the host published.

use microlift::arch::x86_64::context::RegisterContext;
use microlift::core::exit::{ExitReason, TransferKind};
use microlift::core::link::{BlockLinkConfig, BlockLinkEntry};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE, Harness};

const CALLEE: u64 = 0x2000;

const ROWS: u32 = 256;

fn link_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS)
}

fn table() -> Vec<BlockLinkEntry> {
    vec![BlockLinkEntry::default(); ROWS as usize]
}

fn publish(table: &mut [BlockLinkEntry], pc: u64, code: *const u8, base: u64) {
    let slot = link_config().slot(pc);
    table[slot] = BlockLinkEntry {
        guest_pc: pc,
        code: code as u64,
        image_base: base,
        reserved: 0,
    };
}

fn caller() -> Harness {
    LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .with_config(|c| {
            c.block_link = Some(link_config());
        })
        .compile()
        .expect("the caller region compiles")
}

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

/// One host dispatch turn: set `image_base`, clear `link_hit`, enter the linked row.
fn follow(ctx: &mut RegisterContext) -> ExitReason {
    assert_ne!(ctx.link_hit, 0, "expected a link to follow");
    // SAFETY: `link_hit` points into the table this test owns.
    let entry = unsafe { *(ctx.link_hit as *const BlockLinkEntry) };
    ctx.image_base = entry.image_base;
    ctx.link_hit = 0;
    ctx.exit_reason = ExitReason::None as u32;
    // SAFETY: `code` is a compiled region entry with the context-only signature.
    let code: unsafe extern "C" fn(*mut RegisterContext) =
        unsafe { std::mem::transmute(entry.code as *const u8) };
    unsafe { code(ctx as *mut RegisterContext) };
    ExitReason::try_from(ctx.exit_reason).unwrap_or(ExitReason::None)
}

fn run_caller(harness: &mut Harness, ret_to: u64, table: &mut [BlockLinkEntry]) -> ExitReason {
    // Two stack words: the caller's RET pops `ret_to`, then the callee's RET pops the trampoline.
    harness.ctx.rsp -= 8;
    // SAFETY: both words are inside the harness stack.
    unsafe {
        *(harness.ctx.rsp as *mut u64) = ret_to;
        *((harness.ctx.rsp + 8) as *mut u64) = EXIT_TRAMPOLINE;
    }
    harness.ctx.link_table = table.as_mut_ptr();
    harness.ctx.link_hit = 0;
    harness.enter()
}

#[test]
fn a_ret_out_of_the_region_links_to_the_region_that_covers_it() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

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

    let exit = follow(&mut caller.ctx);
    assert_eq!(caller.ctx.rbx, 7);
    assert_eq!(caller.ctx.rax, 1, "the callee inherits the caller's state");
    assert_eq!(exit, ExitReason::Return, "the callee ret'd to the sentinel");
}

#[test]
fn an_address_with_no_row_does_not_link() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE + 0x1000, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(caller.ctx.rip, CALLEE);
    assert_eq!(caller.ctx.link_hit, 0);
}

/// A collision is a miss, never a wrong answer (what makes the direct-mapped table sound).
#[test]
fn a_colliding_row_is_a_miss_rather_than_a_wrong_jump() {
    let mut caller = caller();
    let callee = callee();
    let mut table = table();
    // Table is indexed on `pc >> 4`; this shares the slot.
    let colliding = CALLEE + (ROWS as u64) * 16;
    assert_eq!(link_config().slot(colliding), link_config().slot(CALLEE));
    publish(&mut table, colliding, callee.code_ptr, CALLEE);

    let exit = run_caller(&mut caller, CALLEE, &mut table);

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(caller.ctx.link_hit, 0, "a colliding row must not be taken");
    assert_eq!(caller.ctx.rbx, 0, "and the callee must not have run");
}

#[test]
fn a_null_table_is_not_a_special_case() {
    let mut untabled = caller();
    // SAFETY: as in `run_caller`.
    unsafe { *(untabled.ctx.rsp as *mut u64) = CALLEE };
    untabled.ctx.link_table = std::ptr::null_mut();
    untabled.ctx.link_hit = 0;
    assert_eq!(untabled.enter(), ExitReason::UnknownJump);
    assert_eq!(untabled.ctx.rip, CALLEE);
    assert_eq!(untabled.ctx.link_hit, 0);
}

/// A RET to the exit trampoline must return to the host even if a row is published for it.
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

/// A direct out-of-region CALL leaves `rip` at the return address; a link must commit the callee as PC.
#[test]
fn a_direct_call_out_of_region_links_on_the_callee_and_commits_it() {
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

    let return_address = DEFAULT_LIFT_BASE + 5;
    // SAFETY: `rsp` is inside the caller harness stack.
    assert_eq!(unsafe { *(caller.ctx.rsp as *const u64) }, return_address);
    follow(&mut caller.ctx);
    assert_eq!(caller.ctx.rbx, 7);
    assert_eq!(caller.ctx.rip, return_address);
}

/// Linking off by default: no probe is emitted.
#[test]
fn without_the_config_nothing_probes() {
    let mut caller = LiftTest::new(&[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3])
        .compile()
        .expect("compiles");
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    // SAFETY: `rsp` is inside the harness stack.
    unsafe { *(caller.ctx.rsp as *mut u64) = CALLEE };
    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0xDEAD;

    assert_eq!(caller.enter(), ExitReason::UnknownJump);
    assert_eq!(caller.ctx.link_hit, 0xDEAD);
}

fn through_rax(insn: &[u8]) -> Harness {
    let mut bytes = vec![0x48, 0xB8];
    bytes.extend_from_slice(&CALLEE.to_le_bytes());
    bytes.extend_from_slice(insn);
    bytes.push(0x90);
    LiftTest::new(&bytes)
        .with_config(|c| {
            c.block_link = Some(link_config());
        })
        .compile()
        .expect("the region compiles")
}

const CALL_RAX: &[u8] = &[0xFF, 0xD0];
const JMP_RAX: &[u8] = &[0xFF, 0xE0];

const AFTER_CALL: u64 = DEFAULT_LIFT_BASE + 12;

/// An indirect CALL is a host decision, so the return-address push belongs only in the hit arm.
#[test]
fn an_indirect_call_links_on_the_resolved_callee_and_pushes_the_return_address() {
    let mut caller = through_rax(CALL_RAX);
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0;
    let entry_rsp = caller.ctx.rsp;
    let exit = caller.enter();

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(caller.ctx.exit_code),
        Ok(TransferKind::IndirectCall)
    );
    assert_ne!(caller.ctx.link_hit, 0, "a resolved callee links");
    assert_eq!(
        caller.ctx.rip, CALLEE,
        "the link must commit the callee as the PC, not the return address"
    );
    assert_eq!(caller.ctx.imported_function_va, CALLEE);
    assert_eq!(caller.ctx.rsp, entry_rsp - 8, "the link performed the push");
    // SAFETY: `rsp` is inside the caller harness stack.
    assert_eq!(unsafe { *(caller.ctx.rsp as *const u64) }, AFTER_CALL);

    follow(&mut caller.ctx);
    assert_eq!(caller.ctx.rbx, 7);
    assert_eq!(
        caller.ctx.rip, AFTER_CALL,
        "the callee's own `ret` finds the address the link pushed"
    );
    assert_eq!(caller.ctx.rsp, entry_rsp, "and pops it");
}

/// On a miss nothing is pushed and the exit means what it always did.
#[test]
fn an_indirect_call_that_does_not_link_leaves_the_stack_alone() {
    let mut caller = through_rax(CALL_RAX);
    let mut table = table();

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0;
    let entry_rsp = caller.ctx.rsp;
    let exit = caller.enter();

    assert_eq!(exit, ExitReason::IndirectCall);
    assert_eq!(caller.ctx.imported_function_va, CALLEE);
    assert_eq!(caller.ctx.rip, AFTER_CALL, "the PC is the resume point");
    assert_eq!(caller.ctx.rsp, entry_rsp, "the stack pointer is unchanged");
    assert_eq!(caller.ctx.link_hit, 0);
}

#[test]
fn an_indirect_call_without_linking_is_unchanged() {
    let mut bytes = vec![0x48, 0xB8];
    bytes.extend_from_slice(&CALLEE.to_le_bytes());
    bytes.extend_from_slice(CALL_RAX);
    bytes.push(0x90);
    let mut caller = LiftTest::new(&bytes).compile().expect("compiles");
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0xDEAD;
    let entry_rsp = caller.ctx.rsp;

    assert_eq!(caller.enter(), ExitReason::IndirectCall);
    assert_eq!(caller.ctx.link_hit, 0xDEAD);
    assert_eq!(caller.ctx.rsp, entry_rsp);
}

#[test]
fn an_indirect_jump_links_to_the_region_that_covers_it() {
    let mut caller = through_rax(JMP_RAX);
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE, callee.code_ptr, CALLEE);

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0;
    let exit = caller.enter();

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(caller.ctx.exit_code),
        Ok(TransferKind::IndirectJump)
    );
    assert_ne!(caller.ctx.link_hit, 0, "an indirect jump links");
    assert_eq!(caller.ctx.rip, CALLEE);

    assert_eq!(follow(&mut caller.ctx), ExitReason::Return);
    assert_eq!(caller.ctx.rbx, 7);
}

#[test]
fn an_indirect_call_does_not_take_a_row_about_another_address() {
    let mut caller = through_rax(CALL_RAX);
    let callee = callee();
    let mut table = table();
    publish(&mut table, CALLEE + 1, callee.code_ptr, CALLEE);

    caller.ctx.link_table = table.as_mut_ptr();
    caller.ctx.link_hit = 0;
    let entry_rsp = caller.ctx.rsp;

    assert_eq!(caller.enter(), ExitReason::IndirectCall);
    assert_eq!(caller.ctx.link_hit, 0);
    assert_eq!(caller.ctx.rsp, entry_rsp);
}
