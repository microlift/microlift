//! Alias regions for the `RegisterContext`.

use cranelift::prelude::*;
use cranelift_codegen::ir::{AliasRegionData, Function};

pub const GUEST_MEMORY_ID: u32 = 0x8000_0000;

pub const X87_STACK_ID: u32 = 0x8000_0001;

pub const LINK_TABLE_ID: u32 = 0x8000_0003;

pub const HOST_SLOT_ID: u32 = 0x8000_0002;

/// A region's own link-cell array ([`crate::core::link::LinkSite`]).
///
/// Distinct from [`LINK_TABLE_ID`]: the table is the host's, shared by every
/// region on the thread, while this array belongs to one compiled artifact. They
/// hold the same row type but are never the same memory, and telling Cranelift
/// so is what lets a store to one leave the other's loads alone.
pub const LINK_CELL_ID: u32 = 0x8000_0004;

/// The host's call-return ring ([`crate::core::callret::CallRetEntry`]).
///
/// Its own region: a push writes here and nothing else, so tagging it is what
/// keeps a call from looking like a fence to everything downstream.
pub const CALLRET_ID: u32 = 0x8000_0005;

pub fn region_flags(func: &mut Function, user_id: u32, description: &'static str) -> MemFlagsData {
    let region = func.dfg.alias_regions.insert(AliasRegionData {
        user_id,
        description: description.into(),
    });
    MemFlagsData::trusted()
        .with_can_move()
        .with_alias_region(Some(region))
}

/// Memflags for a guest memory access: trapping and unaligned, like any guest
/// address, but tagged as disjoint from the context.
pub fn guest_memory_flags(func: &mut Function) -> MemFlagsData {
    let region = func.dfg.alias_regions.insert(AliasRegionData {
        user_id: GUEST_MEMORY_ID,
        description: "guest memory".into(),
    });
    MemFlagsData::new().with_alias_region(Some(region))
}
