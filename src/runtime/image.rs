//! Loading an [`Image`]: map it, apply its relocations, and hand out translations.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider, SystemMemoryProvider};

use crate::backend::CompiledRegion;
use crate::core::link::{BlockLinkEntry, ENTRY_SIZE, LinkSite};
use crate::core::srcloc::GuestCodeMap;
use crate::core::symbol::{HostSymbol, ImportDecision};
use crate::image::{Image, ImageRegion, ImageSymbol, RelocTarget, Section, TEXT_ALIGN};
use crate::runtime::{CodeOwner, Translation};

/// What an image's symbols mean in this process.
pub trait SymbolResolver {
    /// The address of `symbol`, or `None` if nothing here provides it.
    fn resolve(&self, symbol: &HostSymbol) -> Option<usize>;
    /// Run-time minus lift-time address of the image's guest code.
    fn guest_slide(&self) -> u64;
}

/// A loaded region, in run-time addresses.
pub struct LoadedRegion {
    pub translation: Translation,
    /// Host offsets relative to [`Translation::link_target`].
    pub map: GuestCodeMap,
    pub import_decisions: Vec<(u64, ImportDecision)>,
    pub lift_base: u64,
}

/// The pages of one loaded image; every translation from it holds this.
struct Mapping {
    memory: Mutex<SystemMemoryProvider>,
}

impl CodeOwner for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        let memory = self.memory.get_mut().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the last translation holding these pages is being dropped.
        unsafe { memory.free_memory() };
    }
}

/// Decodes, maps and relocates `bytes`, yielding its regions in image order.
///
/// Every symbol must resolve; an image that names something this process lacks is
/// refused whole rather than loaded with a hole in it.
pub fn load(bytes: &[u8], resolver: &dyn SymbolResolver) -> crate::Result<Vec<LoadedRegion>> {
    load_image(&Image::decode(bytes)?, resolver)
}

/// [`load`] for an already decoded image.
pub fn load_image(
    image: &Image,
    resolver: &dyn SymbolResolver,
) -> crate::Result<Vec<LoadedRegion>> {
    let slide = resolver.guest_slide();
    let symbols: Vec<u64> = image
        .symbols
        .iter()
        .map(|symbol| match symbol {
            ImageSymbol::GuestSlide => Ok(slide),
            ImageSymbol::Host(host) => resolver.resolve(host).map(|a| a as u64).ok_or_else(|| {
                crate::MicroliftError::CraneliftError(format!(
                    "image needs {host}, which nothing provides"
                ))
            }),
        })
        .collect::<crate::Result<_>>()?;

    let mut memory = SystemMemoryProvider::new();
    let allocate = |memory: &mut SystemMemoryProvider, len: usize, align: u64, kind| {
        memory
            .allocate(len.max(1), align, kind)
            .map_err(|e| crate::MicroliftError::CraneliftError(format!("mapping an image: {e}")))
    };
    let text = allocate(
        &mut memory,
        image.text.len(),
        TEXT_ALIGN,
        JITMemoryKind::Executable,
    )?;
    let rodata = allocate(
        &mut memory,
        image.rodata.len(),
        image.rodata_align,
        JITMemoryKind::ReadOnly,
    )?;
    let bss_len = usize::try_from(image.bss_len).expect("validated against the file length");
    let bss = allocate(&mut memory, bss_len, ENTRY_SIZE, JITMemoryKind::Writable)?;
    // SAFETY: each destination was just allocated at least this long, writable until `finalize`.
    unsafe {
        std::ptr::copy_nonoverlapping(image.text.as_ptr(), text, image.text.len());
        std::ptr::copy_nonoverlapping(image.rodata.as_ptr(), rodata, image.rodata.len());
        std::ptr::write_bytes(bss, 0, bss_len);
    }

    let base = |section| match section {
        Section::Text => text,
        Section::Rodata => rodata,
        Section::Bss => bss,
    } as u64;
    for reloc in &image.relocs {
        let target = match reloc.target {
            RelocTarget::Local { section, offset } => base(section) + offset,
            RelocTarget::Symbol(index) => symbols[index as usize],
        };
        let value = target.wrapping_add_signed(reloc.addend);
        // SAFETY: `Image::decode` checked the 8 bytes lie inside `.text`.
        unsafe {
            text.add(reloc.offset as usize)
                .cast::<u64>()
                .write_unaligned(value)
        };
    }
    memory
        .finalize(BranchProtection::None)
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

    let mapping: Arc<dyn CodeOwner> = Arc::new(Mapping {
        memory: Mutex::new(memory),
    });
    Ok(image
        .regions
        .iter()
        .map(|region| loaded_region(region, text, bss, slide, Arc::clone(&mapping)))
        .collect())
}

fn loaded_region(
    region: &ImageRegion,
    text: *mut u8,
    bss: *mut u8,
    slide: u64,
    owner: Arc<dyn CodeOwner>,
) -> LoadedRegion {
    let slid = |va: u64| va.wrapping_add(slide);
    let slid_ranges =
        |ranges: &[Range<u64>]| ranges.iter().map(|r| slid(r.start)..slid(r.end)).collect();
    // SAFETY: `Image::decode` bounds-checked every offset against its section.
    let compiled = unsafe {
        CompiledRegion {
            entry: text.add(region.entry as usize),
            link_target: text.add(region.link_target as usize),
            link_cells: region.cells.map_or(std::ptr::null_mut(), |at| {
                bss.add(at as usize).cast::<BlockLinkEntry>()
            }),
        }
    };
    let translation = Translation::from_owned(
        compiled,
        slid(region.lift_base),
        region.resume_points.iter().map(|&pc| slid(pc)).collect(),
        slid_ranges(&region.covered),
        slid_ranges(&region.data_dependencies),
        region
            .link_sites
            .iter()
            .map(|site| LinkSite {
                target: slid(site.target),
                kind: site.kind,
            })
            .collect(),
        owner,
    );
    LoadedRegion {
        translation,
        map: region.map.clone(),
        import_decisions: region
            .import_decisions
            .iter()
            .map(|(va, decision)| (slid(*va), decision.clone()))
            .collect(),
        lift_base: slid(region.lift_base),
    }
}
