//! Guest source locations: [`SourceLocMap`] maps guest VA <-> Cranelift [`SourceLoc`];
//! [`GuestCodeMap`] maps compiled host code ranges back to guest VAs.
//!
//! A VA is a signed displacement from the lift base, biased into 31 bits: exact within ~1 GiB,
//! "no location" beyond (never a wrong one). It is a lift-time address, so compare live PCs via
//! [`LiftEngine::to_lift_va`](crate::core::LiftEngine::to_lift_va).

use cranelift_codegen::CompiledCode;
use cranelift_codegen::ir::SourceLoc;

/// Bias placing the signed displacement in 31 bits; `2^30 - 1` would hit the reserved "no location" and is rejected.
const SIGN_BIAS: i64 = 0x4000_0000;

/// Guest VA <-> [`SourceLoc`] for one lifted region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceLocMap {
    /// [`LiftEngine::lift_base`](crate::core::LiftEngine::lift_base); displacements are measured from here.
    pub lift_base: u64,
}

impl SourceLocMap {
    pub fn new(lift_base: u64) -> Self {
        Self { lift_base }
    }

    /// [`SourceLoc::default`] ("no location") when `va` is more than ~1 GiB from the base.
    pub fn encode(&self, va: u64) -> SourceLoc {
        self.try_encode(va).unwrap_or_default()
    }

    /// [`encode`](Self::encode) returning `None` when out of range.
    pub fn try_encode(&self, va: u64) -> Option<SourceLoc> {
        let delta = va.wrapping_sub(self.lift_base) as i64;
        let bits = u32::try_from(delta + SIGN_BIAS).ok()?;
        if bits >= SourceLoc::default().bits() {
            return None;
        }
        Some(SourceLoc::new(bits))
    }

    pub fn decode(&self, loc: SourceLoc) -> Option<u64> {
        if loc.is_default() {
            return None;
        }
        let delta = i64::from(loc.bits()) - SIGN_BIAS;
        Some(self.lift_base.wrapping_add(delta as u64))
    }
}

/// A host code run and the guest instruction it came from; offsets are function-relative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GuestCodeRange {
    pub start: u32,
    pub end: u32,
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

/// Host code offset -> guest instruction for one compiled function.
///
/// Covers only code with a surviving [`SourceLoc`] (not the prologue or entry dispatch); sorted, non-overlapping.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GuestCodeMap {
    ranges: Vec<GuestCodeRange>,
}

impl GuestCodeMap {
    /// `map` must be the lift's [`LiftedCode::source_map`](crate::core::LiftedCode::source_map).
    pub fn from_compiled(compiled: &CompiledCode, map: SourceLocMap) -> Self {
        let ranges = compiled
            .buffer
            .get_srclocs_sorted()
            .iter()
            .filter_map(|srcloc| {
                Some(GuestCodeRange {
                    start: srcloc.start,
                    end: srcloc.end,
                    guest_va: map.decode(srcloc.loc.as_abs())?,
                })
            })
            .collect();
        Self { ranges }
    }

    /// From sorted, non-overlapping ranges.
    pub fn from_ranges(mut ranges: Vec<GuestCodeRange>) -> Self {
        ranges.sort_by_key(|r| r.start);
        Self { ranges }
    }

    pub fn ranges(&self) -> &[GuestCodeRange] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Guest instruction covering function-relative `offset`; `None` for prologue, dispatch or filler.
    pub fn lookup(&self, offset: u32) -> Option<u64> {
        let idx = self.ranges.partition_point(|r| r.start <= offset);
        let range = self.ranges.get(idx.checked_sub(1)?)?;
        range.contains(offset).then_some(range.guest_va)
    }

    /// [`lookup`](Self::lookup) for an absolute host address, given the function's compiled address.
    pub fn lookup_host_addr(&self, func_ptr: *const u8, addr: u64) -> Option<u64> {
        let base = func_ptr as u64;
        let offset = addr.checked_sub(base)?;
        self.lookup(u32::try_from(offset).ok()?)
    }

    /// Every host range lifted from `va`, in code order (IR may be split or duplicated).
    pub fn ranges_for(&self, va: u64) -> impl Iterator<Item = &GuestCodeRange> {
        self.ranges.iter().filter(move |r| r.guest_va == va)
    }

    pub fn code_size_for(&self, va: u64) -> u32 {
        self.ranges_for(va).map(|r| r.len()).sum()
    }

    /// Distinct guest addresses that survived into machine code, ascending.
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
        assert!(map.try_encode(0x1000 + (1 << 30)).is_none());
        assert!(map.try_encode(0x1000u64.wrapping_sub(1 << 30) - 1).is_none());
        assert!(map.try_encode(0x1000 + (1 << 30) - 1).is_none());
        assert!(map.try_encode(0x1000 + (1 << 30) - 2).is_some());
        assert!(map.try_encode(0x1000u64.wrapping_sub(1 << 30)).is_some());
        assert_eq!(map.decode(SourceLoc::default()), None);
    }

    #[test]
    fn encodings_stay_inside_cranelifts_31_bit_range() {
        let map = SourceLocMap::new(0x1000);
        for delta in [-(1i64 << 30), -1, 0, 1, (1 << 30) - 2] {
            let va = 0x1000u64.wrapping_add(delta as u64);
            let loc = map.try_encode(va).unwrap();
            assert!(loc.bits() < 0x8000_0000, "delta {delta}");
            assert!(!loc.is_default(), "delta {delta}");
            assert_eq!(map.decode(loc), Some(va), "delta {delta}");
        }
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
