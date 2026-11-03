//! The container [`ImageBuilder`](crate::backend::ImageBuilder) writes: many compiled regions,
//! their relocations and everything a host needs to publish them, without an object file.
//!
//! Little-endian throughout. Every relocation is an absolute 8-byte store into `.text`,
//! since every external a lift names is far ([`crate::core::symbol`]); anything else is
//! rejected when the image is built.

use std::ops::Range;

use crate::core::link::{LinkSite, LinkSiteKind};
use crate::core::srcloc::{GuestCodeMap, GuestCodeRange};
use crate::core::symbol::{HostSymbol, HostSymbolKind, ImportDecision};

const MAGIC: [u8; 8] = *b"MLIMAGE\0";

/// Bumped whenever the encoding or the meaning of a field changes.
pub const FORMAT_VERSION: u32 = 1;

/// A name a relocation resolves through the loader's resolver.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImageSymbol {
    Host(HostSymbol),
    /// See [`DataTarget::GuestSlide`](crate::core::symbol::DataTarget::GuestSlide).
    GuestSlide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Section {
    Text,
    Rodata,
    /// Zeroed, writable: link cells.
    Bss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelocTarget {
    Local {
        section: Section,
        offset: u64,
    },
    /// Index into [`Image::symbols`].
    Symbol(u32),
}

/// `*(text + offset) as u64 = target + addend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageReloc {
    pub offset: u64,
    pub target: RelocTarget,
    pub addend: i64,
}

/// One compiled region. Addresses are lift-time VAs; offsets index the image's sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRegion {
    pub lift_base: u64,
    /// `.text` offset of the host entry ([`CompiledRegion::entry`](crate::backend::CompiledRegion::entry)).
    pub entry: u64,
    /// `.text` offset a link row names.
    pub link_target: u64,
    /// `.bss` offset of the cell array, present when `link_sites` is non-empty.
    pub cells: Option<u64>,
    pub resume_points: Vec<u64>,
    pub covered: Vec<Range<u64>>,
    pub data_dependencies: Vec<Range<u64>>,
    pub link_sites: Vec<LinkSite>,
    pub import_decisions: Vec<(u64, ImportDecision)>,
    /// Offsets relative to `link_target`.
    pub map: GuestCodeMap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// [`crate::backend::isa_fingerprint`] of the ISA the code was compiled for.
    pub isa_fingerprint: u64,
    /// [`LifterConfig::codegen_fingerprint`](crate::core::LifterConfig::codegen_fingerprint).
    pub config_fingerprint: u64,
    pub text: Vec<u8>,
    pub rodata: Vec<u8>,
    pub rodata_align: u64,
    pub bss_len: u64,
    pub symbols: Vec<ImageSymbol>,
    pub relocs: Vec<ImageReloc>,
    pub regions: Vec<ImageRegion>,
}

/// Alignment of `.text` and of each function in it.
pub const TEXT_ALIGN: u64 = 16;

impl Image {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.bytes_raw(&MAGIC);
        w.u32(FORMAT_VERSION);
        w.u64(self.isa_fingerprint);
        w.u64(self.config_fingerprint);
        w.bytes(&self.text);
        w.bytes(&self.rodata);
        w.u64(self.rodata_align);
        w.u64(self.bss_len);
        w.len(self.symbols.len());
        for symbol in &self.symbols {
            w.symbol(symbol);
        }
        w.len(self.relocs.len());
        for reloc in &self.relocs {
            w.u64(reloc.offset);
            w.reloc_target(reloc.target);
            w.i64(reloc.addend);
        }
        w.len(self.regions.len());
        for region in &self.regions {
            w.region(region);
        }
        w.buf
    }

    pub fn decode(bytes: &[u8]) -> crate::Result<Self> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(MAGIC.len())? != MAGIC {
            return Err(malformed("not a microlift image"));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(malformed(&format!(
                "format version {version}, this build reads {FORMAT_VERSION}"
            )));
        }
        let isa_fingerprint = r.u64()?;
        let config_fingerprint = r.u64()?;
        let text = r.bytes()?;
        let rodata = r.bytes()?;
        let rodata_align = r.u64()?;
        let bss_len = r.u64()?;
        let symbols = r.list(Reader::symbol)?;
        let relocs = r.list(|r| {
            Ok(ImageReloc {
                offset: r.u64()?,
                target: r.reloc_target()?,
                addend: r.i64()?,
            })
        })?;
        let regions = r.list(Reader::region)?;
        if r.at != bytes.len() {
            return Err(malformed("trailing bytes"));
        }
        let image = Image {
            isa_fingerprint,
            config_fingerprint,
            text,
            rodata,
            rodata_align,
            bss_len,
            symbols,
            relocs,
            regions,
        };
        image.validate()?;
        Ok(image)
    }

    /// Every offset in bounds, so a loader can trust them.
    fn validate(&self) -> crate::Result<()> {
        let section_len = |section| match section {
            Section::Text => self.text.len() as u64,
            Section::Rodata => self.rodata.len() as u64,
            Section::Bss => self.bss_len,
        };
        let text = self.text.len() as u64;
        for reloc in &self.relocs {
            if reloc.offset.checked_add(8).is_none_or(|end| end > text) {
                return Err(malformed("relocation outside .text"));
            }
            match reloc.target {
                RelocTarget::Local { section, offset } if offset > section_len(section) => {
                    return Err(malformed("relocation target outside its section"));
                }
                RelocTarget::Symbol(index) if index as usize >= self.symbols.len() => {
                    return Err(malformed("relocation names no symbol"));
                }
                _ => {}
            }
        }
        if !self.rodata_align.is_power_of_two() {
            return Err(malformed(".rodata alignment is not a power of two"));
        }
        for region in &self.regions {
            if region.entry >= text || region.link_target >= text {
                return Err(malformed("region entry outside .text"));
            }
            let cells_end = region
                .cells
                .map(|at| at + region.link_sites.len() as u64 * crate::core::link::ENTRY_SIZE);
            if cells_end.is_some_and(|end| end > self.bss_len)
                || region.cells.is_none() != region.link_sites.is_empty()
            {
                return Err(malformed("region cells outside .bss"));
            }
        }
        Ok(())
    }
}

fn malformed(why: &str) -> crate::MicroliftError {
    crate::MicroliftError::CraneliftError(format!("malformed image: {why}"))
}

#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn bytes_raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.bytes_raw(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.bytes_raw(&v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.bytes_raw(&v.to_le_bytes());
    }

    fn len(&mut self, n: usize) {
        self.u64(n as u64);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.len(bytes.len());
        self.bytes_raw(bytes);
    }

    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    fn ranges(&mut self, ranges: &[Range<u64>]) {
        self.len(ranges.len());
        for r in ranges {
            self.u64(r.start);
            self.u64(r.end);
        }
    }

    fn symbol(&mut self, symbol: &ImageSymbol) {
        match symbol {
            ImageSymbol::GuestSlide => self.u8(0),
            ImageSymbol::Host(host) => {
                match &host.kind {
                    HostSymbolKind::Function(sig) => {
                        self.u8(1);
                        self.str(sig);
                    }
                    HostSymbolKind::Data => self.u8(2),
                    HostSymbolKind::LibCall => self.u8(3),
                }
                self.str(&host.provider);
                self.str(&host.name);
            }
        }
    }

    fn reloc_target(&mut self, target: RelocTarget) {
        match target {
            RelocTarget::Symbol(index) => {
                self.u8(0);
                self.u32(index);
            }
            RelocTarget::Local { section, offset } => {
                self.u8(match section {
                    Section::Text => 1,
                    Section::Rodata => 2,
                    Section::Bss => 3,
                });
                self.u64(offset);
            }
        }
    }

    fn decision(&mut self, decision: &ImportDecision) {
        match decision {
            ImportDecision::NotImport => self.u8(0),
            ImportDecision::Slow => self.u8(1),
            ImportDecision::Fast {
                provider,
                name,
                abi,
            } => {
                self.u8(2);
                self.str(provider);
                self.str(name);
                self.str(abi);
            }
            ImportDecision::Bridge { name } => {
                self.u8(3);
                self.str(name);
            }
        }
    }

    fn region(&mut self, region: &ImageRegion) {
        self.u64(region.lift_base);
        self.u64(region.entry);
        self.u64(region.link_target);
        match region.cells {
            None => self.u8(0),
            Some(at) => {
                self.u8(1);
                self.u64(at);
            }
        }
        self.len(region.resume_points.len());
        for &pc in &region.resume_points {
            self.u64(pc);
        }
        self.ranges(&region.covered);
        self.ranges(&region.data_dependencies);
        self.len(region.link_sites.len());
        for site in &region.link_sites {
            self.u64(site.target);
            self.u8(match site.kind {
                LinkSiteKind::StaticEdge => 0,
                LinkSiteKind::CallReturn => 1,
            });
        }
        self.len(region.import_decisions.len());
        for (va, decision) in &region.import_decisions {
            self.u64(*va);
            self.decision(decision);
        }
        let ranges = region.map.ranges();
        self.len(ranges.len());
        for range in ranges {
            self.u32(range.start);
            self.u32(range.end);
            self.u64(range.guest_va);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| malformed("truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> crate::Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("took N bytes"))
    }

    fn u8(&mut self) -> crate::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> crate::Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> crate::Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn i64(&mut self) -> crate::Result<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }

    /// A count, bounded by the bytes left so a corrupt length cannot allocate wildly.
    fn len(&mut self) -> crate::Result<usize> {
        let n = self.u64()?;
        usize::try_from(n)
            .ok()
            .filter(|&n| n <= self.bytes.len() - self.at)
            .ok_or_else(|| malformed("length exceeds the file"))
    }

    fn bytes(&mut self) -> crate::Result<Vec<u8>> {
        let n = self.len()?;
        Ok(self.take(n)?.to_vec())
    }

    fn str(&mut self) -> crate::Result<String> {
        String::from_utf8(self.bytes()?).map_err(|_| malformed("string is not UTF-8"))
    }

    fn list<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> crate::Result<T>,
    ) -> crate::Result<Vec<T>> {
        let n = self.len()?;
        (0..n).map(|_| item(self)).collect()
    }

    fn ranges(&mut self) -> crate::Result<Vec<Range<u64>>> {
        self.list(|r| Ok(r.u64()?..r.u64()?))
    }

    fn symbol(&mut self) -> crate::Result<ImageSymbol> {
        let tag = self.u8()?;
        let kind = match tag {
            0 => return Ok(ImageSymbol::GuestSlide),
            1 => HostSymbolKind::Function(self.str()?.into()),
            2 => HostSymbolKind::Data,
            3 => HostSymbolKind::LibCall,
            _ => return Err(malformed("unknown symbol kind")),
        };
        Ok(ImageSymbol::Host(HostSymbol {
            provider: self.str()?.into(),
            name: self.str()?.into(),
            kind,
        }))
    }

    fn reloc_target(&mut self) -> crate::Result<RelocTarget> {
        let section = match self.u8()? {
            0 => return Ok(RelocTarget::Symbol(self.u32()?)),
            1 => Section::Text,
            2 => Section::Rodata,
            3 => Section::Bss,
            _ => return Err(malformed("unknown relocation target")),
        };
        Ok(RelocTarget::Local {
            section,
            offset: self.u64()?,
        })
    }

    fn decision(&mut self) -> crate::Result<ImportDecision> {
        Ok(match self.u8()? {
            0 => ImportDecision::NotImport,
            1 => ImportDecision::Slow,
            2 => ImportDecision::Fast {
                provider: self.str()?,
                name: self.str()?,
                abi: self.str()?,
            },
            3 => ImportDecision::Bridge { name: self.str()? },
            _ => return Err(malformed("unknown import decision")),
        })
    }

    fn region(&mut self) -> crate::Result<ImageRegion> {
        let lift_base = self.u64()?;
        let entry = self.u64()?;
        let link_target = self.u64()?;
        let cells = match self.u8()? {
            0 => None,
            1 => Some(self.u64()?),
            _ => return Err(malformed("bad cell tag")),
        };
        let resume_points = self.list(Reader::u64)?;
        let covered = self.ranges()?;
        let data_dependencies = self.ranges()?;
        let link_sites = self.list(|r| {
            let target = r.u64()?;
            let kind = match r.u8()? {
                0 => LinkSiteKind::StaticEdge,
                1 => LinkSiteKind::CallReturn,
                _ => return Err(malformed("unknown link site kind")),
            };
            Ok(LinkSite { target, kind })
        })?;
        let import_decisions = self.list(|r| Ok((r.u64()?, r.decision()?)))?;
        let ranges = self.list(|r| {
            Ok(GuestCodeRange {
                start: r.u32()?,
                end: r.u32()?,
                guest_va: r.u64()?,
            })
        })?;
        Ok(ImageRegion {
            lift_base,
            entry,
            link_target,
            cells,
            resume_points,
            covered,
            data_dependencies,
            link_sites,
            import_decisions,
            map: GuestCodeMap::from_ranges(ranges),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Image {
        Image {
            isa_fingerprint: 1,
            config_fingerprint: 2,
            text: vec![0xcc; 32],
            rodata: vec![1, 2, 3],
            rodata_align: 16,
            bss_len: 32,
            symbols: vec![
                ImageSymbol::GuestSlide,
                ImageSymbol::Host(HostSymbol::data("", "slot")),
            ],
            relocs: vec![
                ImageReloc {
                    offset: 8,
                    target: RelocTarget::Symbol(1),
                    addend: -4,
                },
                ImageReloc {
                    offset: 16,
                    target: RelocTarget::Local {
                        section: Section::Bss,
                        offset: 0,
                    },
                    addend: 0,
                },
            ],
            regions: vec![ImageRegion {
                lift_base: 0x1000,
                entry: 0,
                link_target: 0,
                cells: Some(0),
                resume_points: vec![0x1000, 0x1004],
                covered: vec![0x1000..0x1008, 0x100c..0x1010],
                data_dependencies: vec![],
                link_sites: vec![LinkSite {
                    target: 0x2000,
                    kind: LinkSiteKind::StaticEdge,
                }],
                import_decisions: vec![(0x3000, ImportDecision::Slow)],
                map: GuestCodeMap::from_ranges(vec![GuestCodeRange {
                    start: 0,
                    end: 4,
                    guest_va: 0x1000,
                }]),
            }],
        }
    }

    #[test]
    fn an_image_survives_encoding() {
        let image = sample();
        assert_eq!(Image::decode(&image.encode()).unwrap(), image);
    }

    #[test]
    fn a_truncated_image_is_rejected() {
        let bytes = sample().encode();
        for cut in [0, 8, bytes.len() / 2, bytes.len() - 1] {
            assert!(Image::decode(&bytes[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn a_relocation_outside_text_is_rejected() {
        let mut image = sample();
        image.relocs[0].offset = 28;
        assert!(Image::decode(&image.encode()).is_err());
    }
}
