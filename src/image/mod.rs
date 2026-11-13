//! Position-independent compiled regions, and the file that holds many of them.
//!
//! Every host reference a lift makes is loaded from a GOT entry ([`GotTarget`]), so
//! compiled code holds no address at all: the JIT ([`crate::runtime::jit`]) places a
//! region beside its own few entries, and an image file ([`file`]) is mapped as it is,
//! its entries filled in at load. Both run the code [`compile::PicCompiler`] emits.

pub mod compile;
pub mod file;
pub mod sign;
mod varint;

use std::ops::Range;
use std::sync::Arc;

use crate::core::DataObject;
use crate::core::link::LinkSite;
use crate::core::srcloc::GuestCodeMap;
use crate::core::symbol::{HostSymbol, ImportDecision};

/// Alignment of each function in `.text`.
pub const TEXT_ALIGN: u64 = 16;

/// What one GOT entry holds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GotTarget {
    Host(HostSymbol),
    /// See [`DataTarget::GuestSlide`](crate::core::symbol::DataTarget::GuestSlide).
    GuestSlide,
    /// The region's own link-cell array.
    Cells,
    /// The region's own body, which a pinned entry thunk tail-calls.
    Body,
    /// An outlined helper, compiled once per image whatever number of regions call it.
    Helper(&'static str),
    /// A read-only constant table, placed once per image.
    Table(&'static str),
}

/// How a GOT reference is encoded in the instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GotRefKind {
    /// `adrp xN, entry@PAGE`.
    Aarch64Page21,
    /// `ldr xN, [xN, entry@PAGEOFF]`.
    Aarch64Lo12Ld64,
    /// `mov reg, [rip + entry]`: a 32-bit PC-relative displacement.
    X86PcRel32,
    /// `auipc rd, %hi(entry); ld rd, %lo(entry)(rd)`, both patched at once.
    Riscv64HiLo,
}

/// One reference from code to a GOT entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GotRef {
    /// Offset of the instruction in its [`CodeObject`].
    pub offset: u32,
    pub kind: GotRefKind,
    /// Index into [`CodeObject::targets`].
    pub target: u32,
    pub addend: i32,
}

/// Compiled functions laid out together, with the GOT entries they load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeObject {
    pub text: Vec<u8>,
    pub refs: Vec<GotRef>,
    /// Each entry the code needs, once.
    pub targets: Vec<GotTarget>,
}

impl CodeObject {
    fn target(&mut self, target: GotTarget) -> u32 {
        match self.targets.iter().position(|t| *t == target) {
            Some(index) => index as u32,
            None => {
                self.targets.push(target);
                self.targets.len() as u32 - 1
            }
        }
    }
}

/// One compiled region: its code, what it shares with other regions, and what a
/// host needs to publish it.
#[derive(Debug)]
pub struct RegionObject {
    /// The body at offset 0, then the entry thunk, if pinned.
    pub code: CodeObject,
    /// Offset of the host entry ([`CompiledRegion::entry`](crate::backend::CompiledRegion::entry)).
    pub entry: u32,
    /// Offset a link row names.
    pub link_target: u32,
    /// The helpers its [`GotTarget::Helper`] entries name.
    pub helpers: Vec<(&'static str, Arc<CodeObject>)>,
    /// The tables its [`GotTarget::Table`] entries name.
    pub tables: Vec<DataObject>,
    pub meta: RegionMeta,
}

/// What a region was lifted from and how it may be entered, in lift-time VAs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionMeta {
    pub lift_base: u64,
    pub resume_points: Vec<u64>,
    pub covered: Vec<Range<u64>>,
    pub data_dependencies: Vec<Range<u64>>,
    pub link_sites: Vec<LinkSite>,
    pub import_decisions: Vec<(u64, ImportDecision)>,
    /// Host offsets relative to the body.
    pub map: GuestCodeMap,
}

/// Points every reference in `text`, placed at `text_va`, at its GOT entry, which
/// `slot` places for each target index.
pub fn link(
    text: &mut [u8],
    refs: &[GotRef],
    text_va: u64,
    slot: impl Fn(u32) -> u64,
) -> crate::Result<()> {
    for reference in refs {
        let at = reference.offset as usize;
        let site = text_va + u64::from(reference.offset);
        let entry = slot(reference.target);
        let word = |text: &[u8], at: usize| {
            u32::from_le_bytes(text[at..at + 4].try_into().expect("four bytes"))
        };
        let put = |text: &mut [u8], at: usize, value: u32| {
            text[at..at + 4].copy_from_slice(&value.to_le_bytes())
        };
        match reference.kind {
            GotRefKind::Aarch64Page21 => {
                const IMMLO: u32 = 0x3 << 29;
                const IMMHI: u32 = 0x7_ffff << 5;
                let pages = (entry as i64 >> 12) - (site as i64 >> 12);
                if !(-(1 << 20)..(1 << 20)).contains(&pages) {
                    return Err(out_of_range(site, entry));
                }
                let imm = pages as u32;
                let insn = word(text, at) & !(IMMLO | IMMHI);
                put(
                    text,
                    at,
                    insn | ((imm & 0x3) << 29) | (((imm >> 2) & 0x7_ffff) << 5),
                );
            }
            GotRefKind::Aarch64Lo12Ld64 => {
                const IMM12: u32 = 0xfff << 10;
                if !entry.is_multiple_of(8) {
                    return Err(malformed("a GOT entry is not 8-byte aligned"));
                }
                let imm = ((entry & 0xfff) >> 3) as u32;
                let insn = word(text, at) & !IMM12;
                put(text, at, insn | (imm << 10));
            }
            GotRefKind::X86PcRel32 => {
                let delta = (entry as i64)
                    .wrapping_add(i64::from(reference.addend))
                    .wrapping_sub(site as i64);
                let delta = i32::try_from(delta).map_err(|_| out_of_range(site, entry))?;
                text[at..at + 4].copy_from_slice(&delta.to_le_bytes());
            }
            GotRefKind::Riscv64HiLo => {
                let delta = (entry as i64).wrapping_sub(site as i64);
                if i32::try_from(delta).is_err() {
                    return Err(out_of_range(site, entry));
                }
                let hi = (delta + 0x800) >> 12;
                let lo = delta - (hi << 12);
                let auipc = word(text, at) & 0xfff;
                put(text, at, auipc | ((hi as u32 & 0xf_ffff) << 12));
                let load = word(text, at + 4) & 0x000f_ffff;
                put(text, at + 4, load | ((lo as u32 & 0xfff) << 20));
            }
        }
    }
    Ok(())
}

fn out_of_range(site: u64, entry: u64) -> crate::MicroliftError {
    malformed(&format!(
        "the GOT entry at {entry:#x} is out of range of the code at {site:#x}"
    ))
}

pub(crate) fn malformed(why: &str) -> crate::MicroliftError {
    crate::MicroliftError::CraneliftError(format!("malformed image: {why}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linked(kind: GotRefKind, insns: &[u32], site: u64, entry: u64) -> Vec<u32> {
        let mut text: Vec<u8> = insns.iter().flat_map(|i| i.to_le_bytes()).collect();
        let refs = [GotRef {
            offset: 0,
            kind,
            target: 0,
            addend: 0,
        }];
        link(&mut text, &refs, site, |_| entry).unwrap();
        text.chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn adrp_reaches_the_entry_page() {
        // adrp x1, #0
        let [insn] = linked(GotRefKind::Aarch64Page21, &[0x9000_0001], 0x1000, 0x5000)[..] else {
            unreachable!()
        };
        // Four pages forward: immlo = 0, immhi = 1.
        assert_eq!(insn, 0x9000_0001 | (1 << 5));
    }

    #[test]
    fn a_page_offset_load_takes_the_low_bits_in_eights() {
        // ldr x1, [x1]
        let [insn] = linked(GotRefKind::Aarch64Lo12Ld64, &[0xf940_0021], 0x1000, 0x5018)[..] else {
            unreachable!()
        };
        assert_eq!(insn, 0xf940_0021 | (3 << 10));
    }

    #[test]
    fn a_riscv_pair_splits_the_delta_with_a_signed_low_half() {
        // auipc a0, 0; ld a0, 0(a0)
        let out = linked(
            GotRefKind::Riscv64HiLo,
            &[0x0000_0517, 0x0005_3503],
            0x1000,
            0x1ff0,
        );
        let hi = out[0] >> 12;
        let lo = ((out[1] as i32) >> 20) as i64;
        assert_eq!((i64::from(hi as i32) << 12) + lo, 0xff0);
        assert_eq!(out[0] & 0xfff, 0x517);
        assert_eq!(out[1] & 0x000f_ffff, 0x0005_3503 & 0x000f_ffff);
    }

    #[test]
    fn an_entry_beyond_adrp_range_is_refused() {
        let mut text = 0x9000_0001u32.to_le_bytes().to_vec();
        let refs = [GotRef {
            offset: 0,
            kind: GotRefKind::Aarch64Page21,
            target: 0,
            addend: 0,
        }];
        assert!(link(&mut text, &refs, 0, |_| 1 << 33).is_err());
    }
}
