//! The image file: many regions' position-independent code, laid out to be mapped as
//! it is and published one region at a time.
//!
//! ```text
//! [header] [code, from a page boundary] [shared entries] [import sites] [regions]
//! [entry index] [region records] [signature] [embedder data]
//! ```
//!
//! At load the code is mapped where its GOT entries are [`GOT_SPAN`] below it: the
//! entries regions share (host symbols, the slide, helpers, tables) at the bottom of
//! that span, and each region's own two (its cells and its body) counted down from
//! the code. Nothing in the code is written at load, so on hosts that allow it the
//! pages stay clean and shared with every process mapping the file. Everything a
//! region needs to be published is decoded only when it is first needed.
//!
//! Little-endian throughout.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;

use super::sign::PageHasher;
use super::varint::{self, Reader};
use super::{GotTarget, RegionMeta, RegionObject, TEXT_ALIGN, link, malformed};
use crate::core::link::{ENTRY_SIZE, LinkSite, LinkSiteKind};
use crate::core::srcloc::{GuestCodeMap, GuestCodeRange};
use crate::core::symbol::{HostSymbol, HostSymbolKind, ImportDecision};

const MAGIC: [u8; 8] = *b"MLIMAGE2";

/// Bumped whenever the encoding or the meaning of a field changes.
pub const FORMAT_VERSION: u32 = 2;

/// Mapping granularity: the largest host page size.
pub const PAGE: u64 = 16 * 1024;

/// Distance from the shared GOT entries to the code.
pub const GOT_SPAN: u64 = 256 << 20;

/// The most code one file holds: every reference reaches its entry by `adrp`, within
/// 4 GiB, from anywhere in the code.
pub const MAX_TEXT: u64 = 2 << 30;

/// GOT bytes of each region's own entries.
pub const REGION_SLOTS: u64 = 16;

const HEADER_LEN: u64 = 512;
const RECORD_LEN: usize = 40;
const ENTRY_LEN: usize = 8;
const NO_CELLS: u32 = u32::MAX;

/// What the header says about the rest of the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Header {
    pub isa_fingerprint: u64,
    pub config_fingerprint: u64,
    pub text_offset: u64,
    pub text_len: u64,
    pub bss_len: u64,
    pub shared_count: u64,
    pub region_count: u64,
    pub entry_count: u64,
    /// The address entry `pc`s are counted from.
    pub entry_base: u64,
    pub shared: Section,
    pub sites: Section,
    pub regions: Section,
    pub entries: Section,
    pub meta: Section,
    pub signature: Section,
    pub user: Section,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Section {
    pub offset: u64,
    pub len: u64,
}

impl Section {
    fn range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.len) as usize
    }
}

impl Header {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN as usize);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for value in [
            self.isa_fingerprint,
            self.config_fingerprint,
            self.text_offset,
            self.text_len,
            self.bss_len,
            self.shared_count,
            self.region_count,
            self.entry_count,
            self.entry_base,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for section in self.sections() {
            out.extend_from_slice(&section.offset.to_le_bytes());
            out.extend_from_slice(&section.len.to_le_bytes());
        }
        out.resize(HEADER_LEN as usize, 0);
        out
    }

    fn sections(&self) -> [Section; 7] {
        [
            self.shared,
            self.sites,
            self.regions,
            self.entries,
            self.meta,
            self.signature,
            self.user,
        ]
    }

    fn decode(bytes: &[u8]) -> crate::Result<Self> {
        let header = bytes
            .get(..HEADER_LEN as usize)
            .ok_or_else(|| malformed("truncated header"))?;
        if header[..8] != MAGIC {
            return Err(malformed("not a microlift image"));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().expect("four bytes"));
        if version != FORMAT_VERSION {
            return Err(malformed(&format!(
                "format version {version}, this build reads {FORMAT_VERSION}"
            )));
        }
        let mut words = header[16..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w));
        let mut next = || words.next().expect("the header holds every field");
        let mut header = Header {
            isa_fingerprint: next(),
            config_fingerprint: next(),
            text_offset: next(),
            text_len: next(),
            bss_len: next(),
            shared_count: next(),
            region_count: next(),
            entry_count: next(),
            entry_base: next(),
            ..Header::default()
        };
        let mut section = || Section {
            offset: next(),
            len: next(),
        };
        header.shared = section();
        header.sites = section();
        header.regions = section();
        header.entries = section();
        header.meta = section();
        header.signature = section();
        header.user = section();
        Ok(header)
    }

    fn validate(&self, file_len: u64) -> crate::Result<()> {
        let inside =
            |offset: u64, len: u64| offset.checked_add(len).is_some_and(|end| end <= file_len);
        if !self.text_offset.is_multiple_of(PAGE) || !inside(self.text_offset, self.text_len) {
            return Err(malformed("code outside the file"));
        }
        if self.text_len > MAX_TEXT {
            return Err(malformed("more code than one file may hold"));
        }
        if self.sections().iter().any(|s| !inside(s.offset, s.len)) {
            return Err(malformed("a table outside the file"));
        }
        if self.regions.len != self.region_count * RECORD_LEN as u64
            || self.entries.len != self.entry_count * ENTRY_LEN as u64
        {
            return Err(malformed("table lengths disagree with their counts"));
        }
        if self.shared_count * 8 + self.region_count * REGION_SLOTS > GOT_SPAN {
            return Err(malformed("more GOT entries than the span holds"));
        }
        Ok(())
    }
}

/// A GOT entry regions share, as the file records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedEntry {
    Host(HostSymbol),
    GuestSlide,
    /// An offset into the code: a helper or a table.
    Text(u64),
}

/// One region's fixed-size record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionRecord {
    pub lift_base: u64,
    /// Offset of the region's code in the file's code.
    pub code: u32,
    pub code_len: u32,
    /// Relative to `code`.
    pub entry: u32,
    /// Relative to `code`.
    pub link_target: u32,
    /// Offset of its cell array in the zeroed data after the code.
    pub cells: Option<u32>,
    meta_offset: u64,
    meta_len: u32,
}

impl RegionRecord {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.lift_base.to_le_bytes());
        out.extend_from_slice(&self.meta_offset.to_le_bytes());
        for value in [
            self.code,
            self.code_len,
            self.entry,
            self.link_target,
            self.cells.unwrap_or(NO_CELLS),
            self.meta_len,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn decode(bytes: &[u8]) -> Self {
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("eight"));
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("four"));
        RegionRecord {
            lift_base: u64_at(0),
            meta_offset: u64_at(8),
            code: u32_at(16),
            code_len: u32_at(20),
            entry: u32_at(24),
            link_target: u32_at(28),
            cells: Some(u32_at(32)).filter(|&c| c != NO_CELLS),
            meta_len: u32_at(36),
        }
    }
}

/// Writes an image a region at a time, so a build holds none of the code it has
/// written. Regions keep the order they were added in.
pub struct Writer {
    out: std::io::BufWriter<std::fs::File>,
    /// Region metadata, which follows the code and so waits in a file of its own.
    meta: std::io::BufWriter<std::fs::File>,
    meta_len: u64,
    hasher: PageHasher,
    text_len: u64,
    bss_len: u64,
    isa_fingerprint: u64,
    config_fingerprint: u64,
    shared: HashMap<GotTarget, u32>,
    shared_table: Vec<u8>,
    shared_count: u32,
    sites: HashMap<(u64, ImportDecision), u32>,
    sites_table: Vec<u8>,
    records: Vec<u8>,
    region_count: u32,
    entries: Vec<(u64, u32)>,
    /// Text offsets of the helpers and tables already written.
    placed: HashMap<GotTarget, u64>,
    scratch: Vec<u8>,
}

impl Writer {
    /// Writes into `out` from its start; `meta` is scratch space, read back by
    /// [`finish`](Self::finish).
    pub fn new(
        mut out: std::fs::File,
        meta: std::fs::File,
        isa_fingerprint: u64,
        config_fingerprint: u64,
    ) -> std::io::Result<Self> {
        out.seek(SeekFrom::Start(0))?;
        out.write_all(&vec![0u8; PAGE as usize])?;
        Ok(Writer {
            out: std::io::BufWriter::with_capacity(1 << 20, out),
            meta: std::io::BufWriter::with_capacity(1 << 20, meta),
            meta_len: 0,
            hasher: PageHasher::default(),
            text_len: 0,
            bss_len: 0,
            isa_fingerprint,
            config_fingerprint,
            shared: HashMap::new(),
            shared_table: Vec::new(),
            shared_count: 0,
            sites: HashMap::new(),
            sites_table: Vec::new(),
            records: Vec::new(),
            region_count: 0,
            entries: Vec::new(),
            placed: HashMap::new(),
            scratch: Vec::new(),
        })
    }

    pub fn text_len(&self) -> u64 {
        self.text_len
    }

    pub fn region_count(&self) -> u32 {
        self.region_count
    }

    /// Appends `object`, with whatever helpers and tables it needs that are not
    /// written yet.
    pub fn add(&mut self, object: &RegionObject) -> crate::Result<()> {
        for (name, helper) in &object.helpers {
            let key = GotTarget::Helper(name);
            if self.placed.contains_key(&key) {
                continue;
            }
            let at = self.pad_to(TEXT_ALIGN)?;
            let mut text = helper.text.clone();
            let slots = helper
                .targets
                .iter()
                .map(|target| self.shared_slot(target))
                .collect::<crate::Result<Vec<_>>>()?;
            link(&mut text, &helper.refs, GOT_SPAN + at, |i| {
                slots[i as usize]
            })?;
            self.write_text(&text)?;
            self.placed.insert(key, at);
        }
        for table in &object.tables {
            let key = GotTarget::Table(table.name);
            if self.placed.contains_key(&key) {
                continue;
            }
            let at = self.pad_to(table.align.max(1))?;
            self.write_text(&table.bytes)?;
            self.placed.insert(key, at);
        }

        let index = u64::from(self.region_count);
        let own = GOT_SPAN - REGION_SLOTS * (index + 1);
        if u64::from(self.shared_count) * 8 > own {
            return Err(malformed("the GOT span is full"));
        }
        let code = self.pad_to(TEXT_ALIGN)?;
        let mut text = object.code.text.clone();
        let slots = object
            .code
            .targets
            .iter()
            .map(|target| match target {
                GotTarget::Cells => Ok(own),
                GotTarget::Body => Ok(own + 8),
                shared => self.shared_slot(shared),
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let shared: Vec<u64> = slots.iter().copied().filter(|&s| s < own).collect();
        link(&mut text, &object.code.refs, GOT_SPAN + code, |i| {
            slots[i as usize]
        })?;
        if object.code.targets.contains(&GotTarget::Body) && object.link_target != 0 {
            return Err(malformed(
                "a thunk calls a body that is not the link target",
            ));
        }
        self.write_text(&text)?;

        let cells = (!object.meta.link_sites.is_empty()).then(|| {
            let at = self.bss_len.next_multiple_of(ENTRY_SIZE);
            self.bss_len = at + object.meta.link_sites.len() as u64 * ENTRY_SIZE;
            at
        });

        self.scratch.clear();
        let mut meta = std::mem::take(&mut self.scratch);
        varint::put(&mut meta, shared.len() as u64);
        for slot in shared {
            varint::put(&mut meta, slot / 8);
        }
        self.encode_meta(&object.meta, &mut meta);
        self.meta.write_all(&meta).map_err(io)?;
        let record = RegionRecord {
            lift_base: object.meta.lift_base,
            code: code as u32,
            code_len: text.len() as u32,
            entry: object.entry,
            link_target: object.link_target,
            cells: cells
                .map(|c| u32::try_from(c).map_err(|_| malformed("too many cells")))
                .transpose()?,
            meta_offset: self.meta_len,
            meta_len: meta.len() as u32,
        };
        self.meta_len += meta.len() as u64;
        self.scratch = meta;
        record.encode(&mut self.records);
        for &pc in &object.meta.resume_points {
            self.entries.push((pc, self.region_count));
        }
        self.region_count += 1;
        Ok(())
    }

    /// Writes the tables and the header, and returns the file.
    pub fn finish(mut self, user: &[u8]) -> crate::Result<std::fs::File> {
        let text_end = self.pad_to(PAGE)?;
        debug_assert_eq!(text_end, self.text_len);
        let signature = std::mem::take(&mut self.hasher).finish();
        self.entries.sort_unstable();
        let entry_base = self.entries.first().map_or(0, |&(pc, _)| pc);
        let mut entries = Vec::with_capacity(self.entries.len() * ENTRY_LEN);
        for &(pc, region) in &self.entries {
            let delta = u32::try_from(pc - entry_base)
                .map_err(|_| malformed("entries span more than 4 GiB"))?;
            entries.extend_from_slice(&delta.to_le_bytes());
            entries.extend_from_slice(&region.to_le_bytes());
        }

        let mut header = Header {
            isa_fingerprint: self.isa_fingerprint,
            config_fingerprint: self.config_fingerprint,
            text_offset: PAGE,
            text_len: self.text_len,
            bss_len: self.bss_len,
            shared_count: self.shared_count.into(),
            region_count: self.region_count.into(),
            entry_count: self.entries.len() as u64,
            entry_base,
            ..Header::default()
        };
        let mut at = PAGE + self.text_len;
        let mut place = |len: u64| {
            let section = Section { offset: at, len };
            at += len;
            section
        };
        header.shared = place(self.shared_table.len() as u64);
        header.sites = place(self.sites_table.len() as u64);
        header.regions = place(self.records.len() as u64);
        header.entries = place(entries.len() as u64);
        header.meta = place(self.meta_len);
        header.signature = place(signature.len() as u64);
        header.user = place(user.len() as u64);

        for table in [
            &self.shared_table,
            &self.sites_table,
            &self.records,
            &entries,
        ] {
            self.out.write_all(table).map_err(io)?;
        }
        let mut meta = self.meta.into_inner().map_err(|e| io(e.into_error()))?;
        meta.seek(SeekFrom::Start(0)).map_err(io)?;
        let copied = std::io::copy(&mut meta.take(self.meta_len), &mut self.out).map_err(io)?;
        if copied != self.meta_len {
            return Err(malformed("region metadata went missing while building"));
        }
        self.out.write_all(&signature).map_err(io)?;
        self.out.write_all(user).map_err(io)?;
        let mut out = self.out.into_inner().map_err(|e| io(e.into_error()))?;
        out.seek(SeekFrom::Start(0)).map_err(io)?;
        out.write_all(&header.encode()).map_err(io)?;
        out.seek(SeekFrom::End(0)).map_err(io)?;
        Ok(out)
    }

    /// Pads the code to `align`, returning the offset reached.
    fn pad_to(&mut self, align: u64) -> crate::Result<u64> {
        let at = self.text_len.next_multiple_of(align);
        let pad = vec![0u8; (at - self.text_len) as usize];
        self.write_text(&pad)?;
        Ok(at)
    }

    fn write_text(&mut self, bytes: &[u8]) -> crate::Result<()> {
        if self.text_len + bytes.len() as u64 > MAX_TEXT {
            return Err(malformed("the file holds as much code as it may"));
        }
        self.out.write_all(bytes).map_err(io)?;
        self.hasher.update(bytes);
        self.text_len += bytes.len() as u64;
        Ok(())
    }

    /// The span offset of a shared entry, written to the table the first time.
    fn shared_slot(&mut self, target: &GotTarget) -> crate::Result<u64> {
        if let Some(&slot) = self.shared.get(target) {
            return Ok(u64::from(slot) * 8);
        }
        let entry = match target {
            GotTarget::Host(symbol) => SharedEntry::Host(symbol.clone()),
            GotTarget::GuestSlide => SharedEntry::GuestSlide,
            GotTarget::Helper(_) | GotTarget::Table(_) => {
                SharedEntry::Text(*self.placed.get(target).ok_or_else(|| {
                    malformed("a helper or table is referenced before it is written")
                })?)
            }
            GotTarget::Cells | GotTarget::Body => {
                return Err(malformed("a region's own entry is not shared"));
            }
        };
        encode_shared(&entry, &mut self.shared_table);
        let slot = self.shared_count;
        self.shared_count += 1;
        self.shared.insert(target.clone(), slot);
        Ok(u64::from(slot) * 8)
    }

    fn site(&mut self, va: u64, decision: &ImportDecision) -> u32 {
        if let Some(&index) = self.sites.get(&(va, decision.clone())) {
            return index;
        }
        let index = self.sites.len() as u32;
        varint::put(&mut self.sites_table, va);
        encode_decision(decision, &mut self.sites_table);
        self.sites.insert((va, decision.clone()), index);
        index
    }

    fn encode_meta(&mut self, meta: &RegionMeta, out: &mut Vec<u8>) {
        let base = meta.lift_base;
        let delta = |a: u64, b: u64| a.wrapping_sub(b) as i64;
        varint::put(out, meta.resume_points.len() as u64);
        let mut previous = base;
        for &pc in &meta.resume_points {
            varint::put_signed(out, delta(pc, previous));
            previous = pc;
        }
        for ranges in [&meta.covered, &meta.data_dependencies] {
            varint::put(out, ranges.len() as u64);
            let mut previous = base;
            for range in ranges {
                varint::put_signed(out, delta(range.start, previous));
                varint::put(out, range.end - range.start);
                previous = range.end;
            }
        }
        varint::put(out, meta.link_sites.len() as u64);
        for site in &meta.link_sites {
            varint::put_signed(out, delta(site.target, base));
            out.push(match site.kind {
                LinkSiteKind::StaticEdge => 0,
                LinkSiteKind::CallReturn => 1,
            });
        }
        varint::put(out, meta.import_decisions.len() as u64);
        for (va, decision) in &meta.import_decisions {
            let index = self.site(*va, decision);
            varint::put(out, index.into());
        }
        let ranges = meta.map.ranges();
        varint::put(out, ranges.len() as u64);
        let mut end = 0u32;
        for range in ranges {
            varint::put_signed(out, i64::from(range.start) - i64::from(end));
            varint::put(out, (range.end - range.start).into());
            varint::put_signed(out, delta(range.guest_va, base));
            end = range.end;
        }
    }
}

fn io(e: std::io::Error) -> crate::MicroliftError {
    crate::MicroliftError::CraneliftError(format!("writing an image: {e}"))
}

fn encode_symbol(symbol: &HostSymbol, out: &mut Vec<u8>) {
    match &symbol.kind {
        HostSymbolKind::Function(signature) => {
            out.push(1);
            varint::put_bytes(out, signature.as_bytes());
        }
        HostSymbolKind::Data => out.push(2),
        HostSymbolKind::LibCall => out.push(3),
    }
    varint::put_bytes(out, symbol.provider.as_bytes());
    varint::put_bytes(out, symbol.name.as_bytes());
}

fn encode_shared(entry: &SharedEntry, out: &mut Vec<u8>) {
    match entry {
        SharedEntry::GuestSlide => out.push(0),
        SharedEntry::Host(symbol) => encode_symbol(symbol, out),
        SharedEntry::Text(offset) => {
            out.push(4);
            varint::put(out, *offset);
        }
    }
}

fn encode_decision(decision: &ImportDecision, out: &mut Vec<u8>) {
    match decision {
        ImportDecision::NotImport => out.push(0),
        ImportDecision::Slow => out.push(1),
        ImportDecision::Fast {
            provider,
            name,
            abi,
        } => {
            out.push(2);
            for s in [provider, name, abi] {
                varint::put_bytes(out, s.as_bytes());
            }
        }
        ImportDecision::Bridge { name } => {
            out.push(3);
            varint::put_bytes(out, name.as_bytes());
        }
    }
}

/// An image file's bytes, read in place.
pub struct View<'a> {
    bytes: &'a [u8],
    header: Header,
}

impl<'a> View<'a> {
    pub fn parse(bytes: &'a [u8]) -> crate::Result<Self> {
        let header = Header::decode(bytes)?;
        header.validate(bytes.len() as u64)?;
        Ok(View { bytes, header })
    }

    /// `bytes` with the header [`parse`](Self::parse) already read from them.
    pub(crate) fn with_header(bytes: &'a [u8], header: Header) -> Self {
        View { bytes, header }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn text(&self) -> &'a [u8] {
        let start = self.header.text_offset as usize;
        &self.bytes[start..start + self.header.text_len as usize]
    }

    pub fn signature(&self) -> &'a [u8] {
        &self.bytes[self.header.signature.range()]
    }

    pub fn user(&self) -> &'a [u8] {
        &self.bytes[self.header.user.range()]
    }

    /// Every shared GOT entry, in slot order.
    pub fn shared(&self) -> crate::Result<Vec<SharedEntry>> {
        let mut r = Reader::new(&self.bytes[self.header.shared.range()]);
        let mut entries = Vec::with_capacity(self.header.shared_count as usize);
        for _ in 0..self.header.shared_count {
            let tag = r.byte()?;
            let symbol = |r: &mut Reader<'_>, kind| -> crate::Result<SharedEntry> {
                Ok(SharedEntry::Host(HostSymbol {
                    kind,
                    provider: r.str()?.into(),
                    name: r.str()?.into(),
                }))
            };
            entries.push(match tag {
                0 => SharedEntry::GuestSlide,
                1 => {
                    let signature = r.str()?.into();
                    symbol(&mut r, HostSymbolKind::Function(signature))?
                }
                2 => symbol(&mut r, HostSymbolKind::Data)?,
                3 => symbol(&mut r, HostSymbolKind::LibCall)?,
                4 => {
                    let offset = r.get()?;
                    if offset >= self.header.text_len {
                        return Err(malformed("a shared entry outside the code"));
                    }
                    SharedEntry::Text(offset)
                }
                _ => return Err(malformed("unknown shared entry")),
            });
        }
        Ok(entries)
    }

    /// Every import site any region consulted, unslid, in the order regions name them.
    pub fn sites(&self) -> crate::Result<Vec<(u64, ImportDecision)>> {
        let mut r = Reader::new(&self.bytes[self.header.sites.range()]);
        let mut sites = Vec::new();
        while !r.is_empty() {
            let va = r.get()?;
            let decision = match r.byte()? {
                0 => ImportDecision::NotImport,
                1 => ImportDecision::Slow,
                2 => ImportDecision::Fast {
                    provider: r.str()?.into(),
                    name: r.str()?.into(),
                    abi: r.str()?.into(),
                },
                3 => ImportDecision::Bridge {
                    name: r.str()?.into(),
                },
                _ => return Err(malformed("unknown import decision")),
            };
            sites.push((va, decision));
        }
        Ok(sites)
    }

    pub fn region_count(&self) -> u32 {
        self.header.region_count as u32
    }

    /// Region `index`'s record, checked against the file.
    pub fn record(&self, index: u32) -> crate::Result<RegionRecord> {
        if u64::from(index) >= self.header.region_count {
            return Err(malformed("no such region"));
        }
        let at = self.header.regions.offset as usize + index as usize * RECORD_LEN;
        let record = RegionRecord::decode(&self.bytes[at..at + RECORD_LEN]);
        let code_end = u64::from(record.code) + u64::from(record.code_len);
        let cells_end = record.cells.map(u64::from);
        if code_end > self.header.text_len
            || record.entry >= record.code_len
            || record.link_target >= record.code_len
            || cells_end.is_some_and(|c| c > self.header.bss_len)
            || record.meta_offset + u64::from(record.meta_len) > self.header.meta.len
        {
            return Err(malformed("a region record outside the file"));
        }
        Ok(record)
    }

    /// A region's metadata, its import decisions named through `sites`, and the
    /// shared GOT entries its code loads.
    pub fn meta(
        &self,
        record: &RegionRecord,
        sites: &[(u64, ImportDecision)],
    ) -> crate::Result<(RegionMeta, Vec<u32>)> {
        let start = (self.header.meta.offset + record.meta_offset) as usize;
        let mut r = Reader::new(&self.bytes[start..start + record.meta_len as usize]);
        let mut shared = Vec::new();
        for _ in 0..r.count()? {
            let slot = r.get_u32()?;
            if u64::from(slot) >= self.header.shared_count {
                return Err(malformed("a region names no shared entry"));
            }
            shared.push(slot);
        }
        let base = record.lift_base;
        let mut meta = RegionMeta {
            lift_base: base,
            ..RegionMeta::default()
        };
        let mut previous = base;
        for _ in 0..r.count()? {
            previous = previous.wrapping_add_signed(r.get_signed()?);
            meta.resume_points.push(previous);
        }
        for ranges in [&mut meta.covered, &mut meta.data_dependencies] {
            let mut previous = base;
            for _ in 0..r.count()? {
                let start = previous.wrapping_add_signed(r.get_signed()?);
                let end = start.wrapping_add(r.get()?);
                ranges.push(start..end);
                previous = end;
            }
        }
        for _ in 0..r.count()? {
            let target = base.wrapping_add_signed(r.get_signed()?);
            let kind = match r.byte()? {
                0 => LinkSiteKind::StaticEdge,
                1 => LinkSiteKind::CallReturn,
                _ => return Err(malformed("unknown link site kind")),
            };
            meta.link_sites.push(LinkSite { target, kind });
        }
        for _ in 0..r.count()? {
            let site = sites
                .get(r.get()? as usize)
                .ok_or_else(|| malformed("a region names no import site"))?;
            meta.import_decisions.push(site.clone());
        }
        let mut ranges = Vec::new();
        let mut end = 0i64;
        for _ in 0..r.count()? {
            let start = end + r.get_signed()?;
            let len = i64::from(r.get_u32()?);
            let guest_va = base.wrapping_add_signed(r.get_signed()?);
            let range = GuestCodeRange {
                start: u32::try_from(start).map_err(|_| malformed("bad map range"))?,
                end: u32::try_from(start + len).map_err(|_| malformed("bad map range"))?,
                guest_va,
            };
            end = i64::from(range.end);
            ranges.push(range);
        }
        meta.map = GuestCodeMap::from_ranges(ranges);
        if !r.is_empty() {
            return Err(malformed("trailing bytes in a region's metadata"));
        }
        let cells = record.cells.is_some();
        if cells == meta.link_sites.is_empty() {
            return Err(malformed("a region's cells disagree with its link sites"));
        }
        Ok((meta, shared))
    }

    /// The first region that can be entered at `pc`, a lift-time address.
    pub fn lookup(&self, pc: u64) -> Option<u32> {
        let delta = u32::try_from(pc.checked_sub(self.header.entry_base)?).ok()?;
        let entries = &self.bytes[self.header.entries.range()];
        let at = |i: usize| {
            let e = &entries[i * ENTRY_LEN..(i + 1) * ENTRY_LEN];
            (
                u32::from_le_bytes(e[..4].try_into().expect("four")),
                u32::from_le_bytes(e[4..].try_into().expect("four")),
            )
        };
        let (mut lo, mut hi) = (0usize, self.header.entry_count as usize);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match at(mid).0 < delta {
                true => lo = mid + 1,
                false => hi = mid,
            }
        }
        (lo < self.header.entry_count as usize && at(lo).0 == delta).then(|| at(lo).1)
    }
}
