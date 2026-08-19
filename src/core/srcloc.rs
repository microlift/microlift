//! Guest source locations: which guest instruction a piece of lifted IR - and,
//! after compilation, a piece of host machine code - came from.
//!
//! [`SourceLocMap`] converts between a guest VA and Cranelift's [`SourceLoc`];
//! [`GuestCodeMap`] is the post-compilation product, sorted host code ranges
//! tagged with the guest VA they implement.
//!
//! A `SourceLoc` is a `u32` and `u32::MAX` is reserved as "no location", so a VA
//! is stored as a *signed* displacement from the lift base, biased into the
//! unsigned range: within ±2 GiB it round-trips exactly, further away it encodes
//! as no location rather than a wrong one. Signed because out-of-region branch
//! targets go backwards as often as forwards.
//!
//! The VA is a *lift-time* address, so under
//! [`AddressMode::ImageBaseRelative`](crate::core::AddressMode::ImageBaseRelative)
//! a consumer comparing against a live PC must map it as
//! [`LiftEngine::to_lift_va`](crate::core::LiftEngine::to_lift_va) does.

use cranelift_codegen::CompiledCode;
use cranelift_codegen::ir::SourceLoc;

/// The bias applied to the signed displacement so it fills the `u32` range with
/// its ordering intact. `delta == i32::MIN` maps to `0`, `delta == 0` maps to
/// `0x8000_0000`, and `delta == i32::MAX` maps to `u32::MAX` - which collides
/// with Cranelift's reserved "no location" value and is therefore rejected.
const SIGN_BIAS: u32 = 0x8000_0000;

/// Converts between guest virtual addresses and Cranelift [`SourceLoc`]s for
/// one lifted region. Cheap to copy: the whole mapping is the lift base plus
/// the fixed encoding described in the [module docs](self).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceLocMap {
    /// The VA the region's first byte was lifted at
    /// ([`LiftEngine::lift_base`](crate::core::LiftEngine::lift_base)).
    /// Displacements are measured from here.
    pub lift_base: u64,
}

impl SourceLocMap {
    pub fn new(lift_base: u64) -> Self {
        Self { lift_base }
    }

    /// Encode guest VA `va` as a `SourceLoc`.
    ///
    /// [`SourceLoc::default`] ("no location") when `va` is more than ~2 GiB from
    /// [`lift_base`](Self::lift_base). Never returns a *wrong* location.
    pub fn encode(&self, va: u64) -> SourceLoc {
        self.try_encode(va).unwrap_or_default()
    }

    /// [`encode`](Self::encode), distinguishing "out of range" from a real
    /// location.
    pub fn try_encode(&self, va: u64) -> Option<SourceLoc> {
        let delta = va.wrapping_sub(self.lift_base) as i64;
        let delta = i32::try_from(delta).ok()?;
        let bits = (delta as u32) ^ SIGN_BIAS;
        // `i32::MAX` biases to the reserved "no location" encoding.
        if bits == u32::MAX {
            return None;
        }
        Some(SourceLoc::new(bits))
    }

    /// Recover the guest VA a `SourceLoc` stands for, or `None` if it carries no
    /// location.
    pub fn decode(&self, loc: SourceLoc) -> Option<u64> {
        if loc.is_default() {
            return None;
        }
        let delta = (loc.bits() ^ SIGN_BIAS) as i32;
        Some(self.lift_base.wrapping_add(delta as i64 as u64))
    }
}

/// One contiguous run of host machine code, and the guest instruction it was
/// lifted from. Offsets are relative to the start of the compiled function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GuestCodeRange {
    /// First byte of the run, relative to the function's entry.
    pub start: u32,
    /// One past the last byte of the run, relative to the function's entry.
    pub end: u32,
    /// Guest VA of the instruction this code implements.
    pub guest_va: u64,
}

impl GuestCodeRange {
    pub fn contains(&self, offset: u32) -> bool {
        offset >= self.start && offset < self.end
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Host code offset -> guest instruction map for one compiled function.
///
/// Built from the [`SourceLoc`]s that survived lowering, so it covers only code
/// Cranelift attributed to a guest instruction - the prologue and the entry
/// dispatch deliberately carry none. Sorted by `start`, non-overlapping.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GuestCodeMap {
    ranges: Vec<GuestCodeRange>,
}

impl GuestCodeMap {
    /// Extract the map from a compiled function.
    ///
    /// `map` must be the [`SourceLocMap`] of the lift that produced the IR -
    /// [`LiftedCode::source_map`](crate::core::LiftedCode::source_map).
    pub fn from_compiled(compiled: &CompiledCode, map: SourceLocMap) -> Self {
        let ranges = compiled
            .buffer
            .get_srclocs_sorted()
            .iter()
            .filter_map(|srcloc| {
                Some(GuestCodeRange {
                    start: srcloc.start,
                    end: srcloc.end,
                    guest_va: map.decode(srcloc.loc)?,
                })
            })
            .collect();
        Self { ranges }
    }

    /// Build a map directly from sorted, non-overlapping ranges. Mostly useful
    /// for tests and for callers that persist a map and read it back.
    pub fn from_ranges(mut ranges: Vec<GuestCodeRange>) -> Self {
        ranges.sort_by_key(|r| r.start);
        Self { ranges }
    }

    /// Every range, ordered by host code offset.
    pub fn ranges(&self) -> &[GuestCodeRange] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The guest instruction whose code covers host `offset` (relative to the
    /// function entry), or `None` if that offset belongs to code with no guest
    /// origin - a prologue, the dispatch switch, or codegen filler.
    pub fn lookup(&self, offset: u32) -> Option<u64> {
        // Ranges are sorted and disjoint, so the candidate is the last one
        // starting at or before `offset`.
        let idx = self.ranges.partition_point(|r| r.start <= offset);
        let range = self.ranges.get(idx.checked_sub(1)?)?;
        range.contains(offset).then_some(range.guest_va)
    }

    /// [`lookup`](Self::lookup) for an absolute host address, given the address
    /// the function was compiled to (what
    /// [`CraneliftBackend::compile_lifted`](crate::backend::CraneliftBackend::compile_lifted)
    /// returns). Answers "which guest instruction is this `RIP` in?".
    pub fn lookup_host_addr(&self, func_ptr: *const u8, addr: u64) -> Option<u64> {
        let base = func_ptr as u64;
        let offset = addr.checked_sub(base)?;
        self.lookup(u32::try_from(offset).ok()?)
    }

    /// Every host code range lifted from guest instruction `va`, in code order.
    /// There can be several: a guest instruction's IR may be split, duplicated
    /// across blocks, or interleaved with its neighbours by scheduling.
    pub fn ranges_for(&self, va: u64) -> impl Iterator<Item = &GuestCodeRange> {
        self.ranges.iter().filter(move |r| r.guest_va == va)
    }

    /// Total host bytes attributed to guest instruction `va`.
    pub fn code_size_for(&self, va: u64) -> u32 {
        self.ranges_for(va).map(|r| r.len()).sum()
    }

    /// The distinct guest instruction addresses that survived into machine code,
    /// ascending.
    ///
    /// An absent instruction was eliminated outright - its result was dead - which
    /// makes this a direct read on what eager flag computation costs.
    pub fn guest_addresses(&self) -> Vec<u64> {
        let mut vas: Vec<u64> = self.ranges.iter().map(|r| r.guest_va).collect();
        vas.sort_unstable();
        vas.dedup();
        vas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_in_and_out_of_region() {
        let map = SourceLocMap::new(0x1000);
        for va in [0x1000u64, 0x1001, 0x2000, 0x0, 0xFFFF_FFFF_FFFF_FFFF] {
            assert_eq!(map.decode(map.encode(va)), Some(va), "va {va:#x}");
        }
    }

    #[test]
    fn out_of_range_yields_no_location() {
        let map = SourceLocMap::new(0x1000);
        assert!(map.try_encode(0x1000 + (1 << 31)).is_none());
        assert!(
            map.try_encode(0x1000u64.wrapping_sub(1 << 31) - 1)
                .is_none()
        );
        // `i32::MAX` collides with the reserved encoding and is rejected.
        assert!(map.try_encode(0x1000 + i32::MAX as u64).is_none());
        assert!(map.try_encode(0x1000 + i32::MAX as u64 - 1).is_some());
        assert_eq!(map.decode(SourceLoc::default()), None);
    }

    #[test]
    fn lookup_finds_the_covering_range() {
        let map = GuestCodeMap::from_ranges(vec![
            GuestCodeRange {
                start: 16,
                end: 24,
                guest_va: 0x2000,
            },
            GuestCodeRange {
                start: 4,
                end: 12,
                guest_va: 0x1000,
            },
        ]);
        assert_eq!(map.lookup(0), None); // prologue
        assert_eq!(map.lookup(4), Some(0x1000));
        assert_eq!(map.lookup(11), Some(0x1000));
        assert_eq!(map.lookup(12), None); // gap
        assert_eq!(map.lookup(23), Some(0x2000));
        assert_eq!(map.lookup(24), None); // past the end
        assert_eq!(map.guest_addresses(), vec![0x1000, 0x2000]);
        assert_eq!(map.code_size_for(0x1000), 8);
    }
}
