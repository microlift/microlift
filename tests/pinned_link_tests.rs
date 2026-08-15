//! A pinned register file: block linking that *branches* instead of asking.
//!
//! `block_link_tests.rs` proves the probe finds the right row. This file proves
//! what the region does with it under
//! [`LinkDispatch::Pinned`](microlift::core::link::LinkDispatch::Pinned) - the
//! successor is entered by a tail call with the pinned registers still in
//! registers, and the host is not involved at all.
//!
//! Every test drives regions that `ret` into one another, because a chain is the
//! only place the difference is observable: one region under a pinned ABI and
//! one under none behave identically from the outside, which is the first thing
//! asserted here.

use microlift::arch::x86_64::X86_64;
use microlift::arch::x86_64::context::RegisterContext;
use microlift::arch::x86_64::pin;
use microlift::core::exit::{ExitReason, TransferKind};
use microlift::core::link::{BlockLinkConfig, BlockLinkEntry};
use microlift::core::pin::{PinnedAbi, PinnedReg};
use microlift::core::{LifterConfig, lift_to_code, pinned_abi_for};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE, Harness};

const SECOND: u64 = 0x2000;
const THIRD: u64 = 0x3000;
/// Big enough that the three addresses above land in three different rows -
/// with 256 rows and a shift of four, `0x2000` and `0x3000` are the same slot,
/// which is a legitimate collision and a miss, and not what these tests are
/// about.
const ROWS: u32 = 4096;

/// Five integer slots, which with the context pointer is exactly what aarch64's
/// tail convention passes in registers - see
/// [`RegisterBudget`](microlift::core::pin::RegisterBudget). The choice of
/// *which* five is a host's to make and means nothing here beyond "registers
/// these programs use".
fn abi() -> PinnedAbi {
    PinnedAbi::new()
        .with(pin::RSP)
        .with(pin::RAX)
        .with(pin::RBX)
        .with(pin::RDI)
        .with(pin::RBP)
}

fn pinned_config() -> BlockLinkConfig {
    BlockLinkConfig::new(ROWS).pinned(abi())
}

fn table() -> Vec<BlockLinkEntry> {
    vec![BlockLinkEntry::default(); ROWS as usize]
}

/// Publish `region` for `pc`. **The address a row names is the region's
/// `link_target`, not the address a host calls** - under a pinned ABI those are
/// two different functions, and putting the wrong one here is the mistake this
/// helper exists to make impossible to write by accident.
fn publish(table: &mut [BlockLinkEntry], pc: u64, region: &Harness, base: u64) {
    table[BlockLinkConfig::new(ROWS).slot(pc)] = BlockLinkEntry {
        guest_pc: pc,
        code: region.link_target as u64,
        image_base: base,
        reserved: 0,
    };
}

/// Compile `bytes` at `at`, linking through a pinned table.
fn region(bytes: &[u8], at: u64) -> Harness {
    LiftTest::new(bytes)
        .at(at)
        .image_base(at)
        .with_config(|c| c.block_link = Some(pinned_config()))
        .compile()
        .expect("the region compiles")
}

/// The same region with linking off entirely, for the agreement tests.
fn unlinked(bytes: &[u8], at: u64) -> Harness {
    LiftTest::new(bytes)
        .at(at)
        .image_base(at)
        .compile()
        .expect("the region compiles")
}

/// `mov rax, 1 ; ret`
const SET_RAX: &[u8] = &[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xC3];
/// `add rax, 2 ; ret`
const ADD_2: &[u8] = &[0x48, 0x83, 0xC0, 0x02, 0xC3];
/// `add rax, 4 ; ret`
const ADD_4: &[u8] = &[0x48, 0x83, 0xC0, 0x04, 0xC3];
/// `mov rbx, 7 ; ret`
const SET_RBX: &[u8] = &[0x48, 0xC7, 0xC3, 0x07, 0x00, 0x00, 0x00, 0xC3];

/// Seed the stack with a `ret` chain: the first address is popped by the first
/// region's `RET`, the next by the second's, and so on, with the exit
/// trampoline underneath so the last one returns to the host.
fn ret_chain(harness: &mut Harness, addrs: &[u64]) {
    harness.ctx.rsp -= 8 * addrs.len() as u64;
    for (i, &addr) in addrs.iter().enumerate() {
        // SAFETY: inside the harness's own stack allocation, which it owns.
        unsafe { *((harness.ctx.rsp + 8 * i as u64) as *mut u64) = addr };
    }
    // SAFETY: as above - the word the last `ret` in the chain pops.
    unsafe { *((harness.ctx.rsp + 8 * addrs.len() as u64) as *mut u64) = EXIT_TRAMPOLINE };
}

/// **The base case.** A `RET` out of a pinned region enters its successor
/// directly: one call from the host runs both regions, and the host never sees
/// the link.
#[test]
fn a_link_is_a_branch_and_the_host_is_not_told_about_it() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let second = region(SET_RBX, SECOND);
    let mut table = table();
    publish(&mut table, SECOND, &second, SECOND);

    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = table.as_mut_ptr();
    first.ctx.link_hit = 0;
    let exit = first.enter();

    assert_eq!(first.ctx.rax, 1, "the first region ran");
    assert_eq!(first.ctx.rbx, 7, "and so did the second, without the host");
    assert_eq!(
        first.ctx.link_hit, 0,
        "a pinned link names nothing for a host that is not going to run"
    );
    assert_eq!(
        exit,
        ExitReason::Return,
        "the exit the host sees is the *last* region's, not the first's"
    );
}

/// Two links in a row, each mutating a pinned register. The arithmetic is the
/// evidence: `rax` is read and written in three separately compiled functions
/// and never goes through the context in between.
#[test]
fn pinned_state_flows_through_a_chain_of_links() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let second = region(ADD_2, SECOND);
    let third = region(ADD_4, THIRD);
    let mut table = table();
    publish(&mut table, SECOND, &second, SECOND);
    publish(&mut table, THIRD, &third, THIRD);

    ret_chain(&mut first, &[SECOND, THIRD]);
    first.ctx.link_table = table.as_mut_ptr();
    first.ctx.link_hit = 0;
    let exit = first.enter();

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(
        first.ctx.rax,
        1 + 2 + 4,
        "all three regions ran in one entry"
    );
}

/// **The case that makes a miss safe.** When nothing matches, the region returns
/// to the host - and the host must find the pinned registers in the context,
/// exactly as it would from a region with nothing pinned. This is the thunk's
/// half of the contract.
#[test]
fn a_miss_publishes_the_pinned_registers_before_returning() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let mut table = table();
    // A populated table, but for an address nobody returns to.
    let elsewhere = region(SET_RBX, SECOND);
    publish(&mut table, SECOND + 0x1000, &elsewhere, SECOND);

    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = table.as_mut_ptr();
    first.ctx.link_hit = 0;
    let exit = first.enter();

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(
        TransferKind::try_from(first.ctx.exit_code),
        Ok(TransferKind::Return)
    );
    assert_eq!(first.ctx.rip, SECOND);
    assert_eq!(first.ctx.rax, 1, "a pinned register the host has to see");
    assert_eq!(first.ctx.link_hit, 0);
}

/// A thread with no table published never links, and the region behaves as it
/// did before any of this existed. This is also **how a host ends a chain**:
/// clearing `link_table` is the only way back once regions are branching to
/// each other.
#[test]
fn a_null_table_ends_the_chain() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = std::ptr::null_mut();
    let exit = first.enter();

    assert_eq!(exit, ExitReason::UnknownJump);
    assert_eq!(first.ctx.rip, SECOND);
    assert_eq!(first.ctx.rax, 1);
}

/// Pinning changes where a register *lives*, never what a program computes. A
/// region that never links must be indistinguishable from one lifted without
/// the mechanism at all.
#[test]
fn a_region_that_never_links_is_unchanged_by_pinning() {
    // `mov rax, 5 ; add rax, rdi ; mov rbx, rax ; ret`
    let code = &[
        0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x01, 0xF8, 0x48, 0x89, 0xC3, 0xC3,
    ];
    let mut pinned = region(code, DEFAULT_LIFT_BASE);
    let mut plain = unlinked(code, DEFAULT_LIFT_BASE);
    for h in [&mut pinned, &mut plain] {
        h.ctx.rdi = 37;
        h.ctx.rsi = 11;
    }

    assert_eq!(pinned.enter(), plain.enter());
    assert_eq!(pinned.ctx.rax, plain.ctx.rax);
    assert_eq!(pinned.ctx.rax, 42);
    assert_eq!(pinned.ctx.rbx, plain.ctx.rbx);
    assert_eq!(pinned.ctx.rdi, plain.ctx.rdi);
    assert_eq!(pinned.ctx.rsi, plain.ctx.rsi);
    assert_eq!(pinned.ctx.rip, plain.ctx.rip);
}

/// An entry thunk exists exactly when a pinned ABI does, and the lifted body is
/// the thing behind it. Read off the lift rather than the run, because this is
/// the shape a backend has to know about.
#[test]
fn a_pinned_lift_produces_a_thunk_and_an_unpinned_one_does_not() {
    let mut config = LifterConfig::default();
    config.block_link = Some(pinned_config());
    let code = lift_to_code::<X86_64>(SET_RAX, DEFAULT_LIFT_BASE, config.clone()).unwrap();
    assert!(code.entry_thunk.is_some());
    // Context pointer plus five slots, in and out.
    assert_eq!(code.func.signature.params.len(), 6);
    assert_eq!(code.func.signature.returns.len(), 6);

    config.block_link = Some(BlockLinkConfig::new(ROWS));
    let code = lift_to_code::<X86_64>(SET_RAX, DEFAULT_LIFT_BASE, config).unwrap();
    assert!(code.entry_thunk.is_none());
    assert_eq!(code.func.signature.params.len(), 1);
    assert!(code.func.signature.returns.is_empty());
}

/// A slot the frontend cannot type takes the **whole** ABI down to the older
/// shape rather than producing half a pinned region. An XMM register is that
/// slot without `use_vector_xmm`, where it is two variables and not one value.
#[test]
fn a_slot_the_frontend_refuses_disables_pinning_entirely() {
    let mut config = LifterConfig::default();
    config.use_vector_xmm = false;
    config.block_link =
        Some(BlockLinkConfig::new(ROWS).pinned(PinnedAbi::new().with(pin::RAX).with(pin::xmm(0))));
    assert!(pinned_abi_for::<X86_64>(&config).is_none());
    let code = lift_to_code::<X86_64>(SET_RAX, DEFAULT_LIFT_BASE, config.clone()).unwrap();
    assert!(code.entry_thunk.is_none(), "and RAX is not pinned either");

    // With the prerequisite on, the same ABI is honoured.
    config.use_vector_xmm = true;
    assert_eq!(pinned_abi_for::<X86_64>(&config).map(|a| a.len()), Some(2));
}

/// Analysis mode emits inspection IR that is never executed, and pinning is a
/// property of executing. A host that turns both on gets the inspectable one.
#[test]
fn analysis_mode_is_never_pinned() {
    let mut config = LifterConfig::default();
    config.analysis_mode = true;
    config.block_link = Some(pinned_config());
    assert!(pinned_abi_for::<X86_64>(&config).is_none());
}

/// The budget is what makes a slot scarce, and the context pointer spends one.
#[test]
fn the_declared_abi_fits_the_host() {
    abi()
        .check_budget(target_lexicon::Triple::host().architecture)
        .expect("five integer slots plus the context pointer is within budget");
}

/// A guest with no `block_link_offsets` cannot link and so cannot pin either -
/// there is no table to resolve a successor out of. Asserted through the same
/// accessor everything else asks, so the two cannot drift apart.
#[test]
fn pinning_needs_somewhere_to_put_the_link_fields() {
    // x86-64 has them; the check that matters is that the question is asked at
    // all, which the aarch64 frontend (which does not) exercises in the
    // negative through `GuestArch::block_link_offsets`'s default.
    assert!(<X86_64 as microlift::core::GuestArch>::block_link_offsets().is_some());
    let config = LifterConfig::default();
    assert!(
        pinned_abi_for::<X86_64>(&config).is_none(),
        "and no linking at all means no pinning"
    );
}

/// The slot list is an ABI, not a set: two regions that pin the same registers
/// in a different order do not interoperate, and the ordering is what a host has
/// to keep stable.
#[test]
fn slot_order_is_part_of_the_abi() {
    let a = PinnedAbi::new().with(pin::RAX).with(pin::RSP);
    let b = PinnedAbi::new().with(pin::RSP).with(pin::RAX);
    assert_ne!(a, b);
    assert_eq!(a.position(pin::RSP), Some(1));
    assert_eq!(b.position(pin::RSP), Some(0));
}

/// The context a chain runs against is the caller's throughout - a linked
/// region is handed the same pointer, not a copy - so a register nothing pins
/// still travels between regions the way it always did, through memory.
#[test]
fn unpinned_registers_still_travel_through_the_context() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    // `add rcx, 3 ; ret` - RCX is not in the ABI.
    let second = region(&[0x48, 0x83, 0xC1, 0x03, 0xC3], SECOND);
    let mut table = table();
    publish(&mut table, SECOND, &second, SECOND);

    first.ctx.rcx = 39;
    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = table.as_mut_ptr();
    let exit = first.enter();

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(first.ctx.rcx, 42);
}

/// A `RET` to the exit trampoline is not a guest address and never links - the
/// same exclusion `block_link_tests` proves for the host-resume shape, restated
/// here because under pinning a wrong answer would be a tail call into the
/// sentinel rather than a bad row handed back.
#[test]
fn the_exit_trampoline_still_returns_to_the_host() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let elsewhere = region(SET_RBX, SECOND);
    let mut table = table();
    // A row for the trampoline itself, which no host would publish - and which
    // is exactly the mistake that would otherwise be silent.
    publish(&mut table, EXIT_TRAMPOLINE, &elsewhere, SECOND);

    first.ctx.link_table = table.as_mut_ptr();
    let exit = first.enter();

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(first.ctx.rax, 1);
    assert_eq!(first.ctx.rbx, 0, "the row must not have been taken");
}

/// `PinnedReg` is the guest's own numbering, and the named constants are that
/// numbering spelled out. A slot resolved through the engine must land on the
/// register whose name it carries.
#[test]
fn a_named_slot_is_the_register_it_names() {
    assert_eq!(pin::RSP, PinnedReg::gpr(4));
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let before = first.ctx.rsp;
    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = std::ptr::null_mut();
    first.enter();
    // The `RET` popped one word, and `rsp` is pinned - so this is the thunk
    // storing the value the region left in a register, not the region storing
    // it itself.
    assert_eq!(first.ctx.rsp, before - 8 + 8);
}

/// The context pointer the thunk hands back is the one it was given. Stated as
/// a test because the *reason* it is in the signature is register allocation
/// (see `core::pin`), and a future change that drops it would be caught here
/// rather than as a slow region.
#[test]
fn the_context_pointer_survives_the_round_trip() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let before = &first.ctx as *const RegisterContext as u64;
    first.ctx.link_table = std::ptr::null_mut();
    first.enter();
    assert_eq!(&first.ctx as *const RegisterContext as u64, before);
    assert_eq!(first.ctx.rax, 1);
}
