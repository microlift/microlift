//! Pinned dispatch: linked successors are entered by tail call with pinned registers kept in registers.
//! Chains of `ret`-linked regions are the only place this is observable; `block_link_tests.rs` covers the probe.

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
/// Large enough that the three addresses land in different rows (256 rows, shift 4: `0x2000` and `0x3000` collide).
const ROWS: u32 = 4096;

/// Five integer slots plus the context pointer fill aarch64's tail convention ([`RegisterBudget`](microlift::core::pin::RegisterBudget)).
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

/// Publish `region` for `pc`. Rows must name `link_target`, not the host entry.
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

/// Same region with linking off, for the agreement tests.
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

/// Seed the stack with a `ret` chain ending in the exit trampoline.
fn ret_chain(harness: &mut Harness, addrs: &[u64]) {
    harness.ctx.rsp -= 8 * addrs.len() as u64;
    for (i, &addr) in addrs.iter().enumerate() {
        // SAFETY: inside the harness's own stack allocation, which it owns.
        unsafe { *((harness.ctx.rsp + 8 * i as u64) as *mut u64) = addr };
    }
    // SAFETY: the word the last `ret` pops.
    unsafe { *((harness.ctx.rsp + 8 * addrs.len() as u64) as *mut u64) = EXIT_TRAMPOLINE };
}

/// Base case: a `RET` out of a pinned region enters its successor directly, without the host.
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

/// Two links each mutating a pinned register; `rax` never goes through the context.
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

/// A miss returns to the host, which must find the pinned registers in the context (the thunk's half).
#[test]
fn a_miss_publishes_the_pinned_registers_before_returning() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let mut table = table();
    // Populated table, but not for the returned-to address.
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

/// No published table means no linking; clearing `link_table` is how a host ends a chain.
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

/// Pinning changes where a register lives, not what is computed.
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

/// An entry thunk exists exactly when a pinned ABI does; read off the lift.
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

/// An untypable slot (XMM without `use_vector_xmm`) demotes the whole ABI.
#[test]
fn a_slot_the_frontend_refuses_disables_pinning_entirely() {
    let mut config = LifterConfig::default();
    config.use_vector_xmm = false;
    config.block_link =
        Some(BlockLinkConfig::new(ROWS).pinned(PinnedAbi::new().with(pin::RAX).with(pin::xmm(0))));
    assert!(pinned_abi_for::<X86_64>(&config).is_none());
    let code = lift_to_code::<X86_64>(SET_RAX, DEFAULT_LIFT_BASE, config.clone()).unwrap();
    assert!(code.entry_thunk.is_none(), "and RAX is not pinned either");

    // With the prerequisite on, the ABI is honoured.
    config.use_vector_xmm = true;
    assert_eq!(pinned_abi_for::<X86_64>(&config).map(|a| a.len()), Some(2));
}

/// Analysis mode wins over pinning.
#[test]
fn analysis_mode_is_never_pinned() {
    let mut config = LifterConfig::default();
    config.analysis_mode = true;
    config.block_link = Some(pinned_config());
    assert!(pinned_abi_for::<X86_64>(&config).is_none());
}

/// The context pointer spends one slot of the budget.
#[test]
fn the_declared_abi_fits_the_host() {
    abi()
        .check_budget(target_lexicon::Triple::host().architecture)
        .expect("five integer slots plus the context pointer is within budget");
}

/// No `block_link_offsets` means no linking and no pinning.
#[test]
fn pinning_needs_somewhere_to_put_the_link_fields() {
    // x86-64 has them; aarch64 exercises the negative via the trait default.
    assert!(<X86_64 as microlift::core::GuestArch>::block_link_offsets().is_some());
    let config = LifterConfig::default();
    assert!(
        pinned_abi_for::<X86_64>(&config).is_none(),
        "and no linking at all means no pinning"
    );
}

/// Slot order is part of the ABI: regions pinning the same registers in a different order do not interoperate.
#[test]
fn slot_order_is_part_of_the_abi() {
    let a = PinnedAbi::new().with(pin::RAX).with(pin::RSP);
    let b = PinnedAbi::new().with(pin::RSP).with(pin::RAX);
    assert_ne!(a, b);
    assert_eq!(a.position(pin::RSP), Some(1));
    assert_eq!(b.position(pin::RSP), Some(0));
}

/// A linked region gets the caller's context pointer, so unpinned registers still travel through memory.
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

/// A `RET` to the exit trampoline never links; under pinning a wrong answer would tail-call the sentinel.
#[test]
fn the_exit_trampoline_still_returns_to_the_host() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let elsewhere = region(SET_RBX, SECOND);
    let mut table = table();
    // A row for the trampoline, which no host would publish.
    publish(&mut table, EXIT_TRAMPOLINE, &elsewhere, SECOND);

    first.ctx.link_table = table.as_mut_ptr();
    let exit = first.enter();

    assert_eq!(exit, ExitReason::Return);
    assert_eq!(first.ctx.rax, 1);
    assert_eq!(first.ctx.rbx, 0, "the row must not have been taken");
}

/// Named `PinnedReg` constants must resolve to the register they name.
#[test]
fn a_named_slot_is_the_register_it_names() {
    assert_eq!(pin::RSP, PinnedReg::gpr(4));
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let before = first.ctx.rsp;
    ret_chain(&mut first, &[SECOND]);
    first.ctx.link_table = std::ptr::null_mut();
    first.enter();
    // `rsp` is pinned, so this is the thunk storing the value, not the region.
    assert_eq!(first.ctx.rsp, before - 8 + 8);
}

/// The thunk returns the context pointer it was given (kept for register allocation, see `core::pin`).
#[test]
fn the_context_pointer_survives_the_round_trip() {
    let mut first = region(SET_RAX, DEFAULT_LIFT_BASE);
    let before = &first.ctx as *const RegisterContext as u64;
    first.ctx.link_table = std::ptr::null_mut();
    first.enter();
    assert_eq!(&first.ctx as *const RegisterContext as u64, before);
    assert_eq!(first.ctx.rax, 1);
}

/// Indirect `CALL` under pinning: push, branch and return all stay in registers.
/// `mov rax, SECOND ; call rax ; ret` - pinned `rax` survives only if the callee's `ret` links back.
#[test]
fn an_indirect_call_links_both_ways_without_the_host() {
    let mut bytes = vec![0x48, 0xB8];
    bytes.extend_from_slice(&SECOND.to_le_bytes());
    bytes.extend_from_slice(&[0xFF, 0xD0, 0xC3]);
    let mut first = region(&bytes, DEFAULT_LIFT_BASE);
    let second = region(SET_RBX, SECOND);
    let after_call = DEFAULT_LIFT_BASE + 12;

    let mut table = table();
    publish(&mut table, SECOND, &second, SECOND);
    publish(&mut table, after_call, &first, DEFAULT_LIFT_BASE);

    first.ctx.link_table = table.as_mut_ptr();
    first.ctx.link_hit = 0;
    let entry_rsp = first.ctx.rsp;
    let exit = first.enter();

    assert_eq!(exit, ExitReason::Return, "the whole chain ran in one entry");
    assert_eq!(first.ctx.rax, SECOND, "the pinned file survived the call");
    assert_eq!(first.ctx.rbx, 7, "the callee ran");
    assert_eq!(first.ctx.link_hit, 0);
    // Down 8 for the call, up 8 for the callee's ret, up 8 for the final ret.
    assert_eq!(
        first.ctx.rsp,
        entry_rsp + 8,
        "the return address the link pushed is the one the callee popped"
    );
}
