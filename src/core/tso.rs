//! x86 total store order on a weakly ordered host, by fencing guest memory accesses.
//!
//! x86 lets a store be passed by a later load and nothing else. AArch64 reorders every
//! pair, so lock-free guest code that publishes with a plain `MOV` (Mono's write barrier
//! is a compiler barrier, Unity's job queues hand nodes over with plain stores) sees
//! another thread's writes out of order. A fence goes before a guest access whenever an
//! earlier one since the last fence forms a pair x86 keeps ordered: load-load,
//! load-store or store-store. Store-load is the one TSO does not promise, so a load
//! after only stores gets none.
//!
//! Only accesses in the guest-memory alias region count; the register context and the
//! host's own tables are this thread's alone. Block entries and calls are treated as
//! having both kinds pending, since what ran before them is not visible here.

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{AliasRegion, Function, InstBuilder};

#[derive(Clone, Copy, Default)]
struct Pending {
    load: bool,
    store: bool,
}

impl Pending {
    const UNKNOWN: Self = Self {
        load: true,
        store: true,
    };
}

/// Insert the fences TSO needs around every access to `guest` in `func`.
pub fn insert_fences(func: &mut Function, guest: Option<AliasRegion>) {
    let Some(guest) = guest else { return };
    let mut cursor = FuncCursor::new(func);
    while let Some(_block) = cursor.next_block() {
        let mut pending = Pending::UNKNOWN;
        while let Some(inst) = cursor.next_inst() {
            let data = &cursor.func.dfg.insts[inst];
            let opcode = data.opcode();
            if opcode.is_call() {
                pending = Pending::UNKNOWN;
                continue;
            }
            if opcode == cranelift_codegen::ir::Opcode::Fence {
                pending = Pending::default();
                continue;
            }
            let (loads, stores) = (opcode.can_load(), opcode.can_store());
            if !(loads || stores) || data.alias_region(&cursor.func.dfg) != Some(guest) {
                continue;
            }
            // Atomics are already sequentially consistent on every host Cranelift has.
            if loads && stores {
                pending = Pending::default();
                continue;
            }
            let needs = match stores {
                true => pending.load || pending.store,
                false => pending.load,
            };
            if needs {
                cursor.ins().fence();
                pending = Pending::default();
            }
            match stores {
                true => pending.store = true,
                false => pending.load = true,
            }
        }
    }
}
