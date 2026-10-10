//! Alias regions for the `RegisterContext`.

use cranelift::prelude::*;
use cranelift_codegen::ir::{AliasRegionData, Function};

pub const GUEST_MEMORY_ID: u32 = 0x8000_0000;

pub const X87_STACK_ID: u32 = 0x8000_0001;

pub const LINK_TABLE_ID: u32 = 0x8000_0003;

pub const HOST_SLOT_ID: u32 = 0x8000_0002;

/// A region's own link-cell array ([`crate::core::link::LinkSite`]); never the same memory as [`LINK_TABLE_ID`].
pub const LINK_CELL_ID: u32 = 0x8000_0004;

/// The host's call-return ring ([`crate::core::callret::CallRetEntry`]).
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

/// Trapping, unaligned memflags tagged as disjoint from the context.
pub fn guest_memory_flags(func: &mut Function) -> MemFlagsData {
    let region = func.dfg.alias_regions.insert(AliasRegionData {
        user_id: GUEST_MEMORY_ID,
        description: "guest memory".into(),
    });
    MemFlagsData::new().with_alias_region(Some(region))
}

/// Read-only data ([`crate::core::DataObject`]); loads may float past any store.
pub const CONST_DATA_ID: u32 = 0x8000_0006;
