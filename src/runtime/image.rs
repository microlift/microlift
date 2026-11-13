//! Loading an image file: map its code where it can run as it is, and publish its
//! regions one at a time, as they are first needed.
//!
//! Darwin maps a file's pages executable once a code signature covers them, so there
//! the code is mapped straight from the file: clean pages, shared with every process
//! running the same file and never copied. Elsewhere it is mapped too, unless the
//! file system forbids it, in which case it is copied once into anonymous memory.

use std::ops::Range;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::CompiledRegion;
use crate::core::link::{BlockLinkEntry, LinkSite};
use crate::core::srcloc::GuestCodeMap;
use crate::core::symbol::{HostSymbol, ImportDecision};
use crate::image::file::{GOT_SPAN, Header, PAGE, REGION_SLOTS, SharedEntry, View};
use crate::runtime::{CodeOwner, Translation};

/// What an image's symbols mean in this process.
pub trait SymbolResolver {
    /// The address of `symbol`, or `None` if nothing here provides it.
    fn resolve(&self, symbol: &HostSymbol) -> Option<usize>;
    /// Run-time minus lift-time address of the image's guest code.
    fn guest_slide(&self) -> u64;
}

/// A region published from an image, in run-time addresses.
pub struct LoadedRegion {
    pub translation: Translation,
    /// Host offsets relative to [`Translation::link_target`].
    pub map: GuestCodeMap,
    pub import_decisions: Vec<(u64, ImportDecision)>,
    pub lift_base: u64,
    /// Bytes of code.
    pub code_len: u32,
}

/// The memory an image is mapped into; every translation from it holds this.
struct Mapping {
    file: *mut u8,
    file_len: usize,
    reservation: *mut u8,
    reservation_len: usize,
}

// SAFETY: the mappings are only read, except GOT entries, which are written before the
// code that loads them is published.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl CodeOwner for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the last translation that could run this code is being dropped.
        unsafe {
            libc::munmap(self.file.cast(), self.file_len);
            if !self.reservation.is_null() {
                libc::munmap(self.reservation.cast(), self.reservation_len);
            }
        }
    }
}

/// An image file mapped for execution.
pub struct MappedImage {
    mapping: Arc<Mapping>,
    header: Header,
    sites: Vec<(u64, ImportDecision)>,
    text: *mut u8,
    bss: *mut u8,
    /// Shared GOT entries that resolved; a region loading any other is refused.
    resolved: Vec<bool>,
    slide: u64,
    /// Kept to map the code once asked to.
    file: std::fs::File,
    /// Whether the code is mapped, and if so whether in place.
    code: Mutex<Option<bool>>,
    ready: AtomicBool,
}

// SAFETY: see `Mapping`; the raw pointers name its memory.
unsafe impl Send for MappedImage {}
unsafe impl Sync for MappedImage {}

fn os_error(what: &str) -> crate::MicroliftError {
    crate::MicroliftError::CraneliftError(format!("{what}: {}", std::io::Error::last_os_error()))
}

fn map_anonymous(at: *mut u8, len: usize, prot: libc::c_int) -> crate::Result<()> {
    // SAFETY: `at..at + len` lies inside a reservation this module owns.
    let mapped = unsafe {
        libc::mmap(
            at.cast(),
            len,
            prot,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    match mapped == libc::MAP_FAILED {
        true => Err(os_error("mapping an image")),
        false => Ok(()),
    }
}

fn page_round(len: u64) -> usize {
    len.next_multiple_of(PAGE) as usize
}

impl MappedImage {
    /// Maps `file` for reading and reserves the space its code, GOT and cells go in.
    /// Its GOT is empty until [`bind`](Self::bind), and its code unmapped until
    /// [`map_code`](Self::map_code).
    pub fn open(file: &std::fs::File) -> crate::Result<Self> {
        let fd = file.as_raw_fd();
        let file_len = file
            .metadata()
            .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?
            .len() as usize;
        // SAFETY: a fresh read-only mapping of the whole file.
        let bytes = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                file_len.max(1),
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        if bytes == libc::MAP_FAILED {
            return Err(os_error("mapping an image"));
        }
        let bytes = bytes.cast::<u8>();
        // Owned from here, so an error below unmaps what was mapped.
        let mut mapping = Mapping {
            file: bytes,
            file_len: file_len.max(1),
            reservation: std::ptr::null_mut(),
            reservation_len: 0,
        };
        // SAFETY: the mapping above is `file_len` bytes and lives as long as `mapping`.
        let view = View::parse(unsafe { std::slice::from_raw_parts(bytes, file_len) })?;
        let header = *view.header();
        let text_len = page_round(header.text_len);
        let bss_len = page_round(header.bss_len);
        let reservation_len = GOT_SPAN as usize + text_len + bss_len;
        // SAFETY: a fresh inaccessible reservation.
        let reservation = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reservation_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if reservation == libc::MAP_FAILED {
            return Err(os_error("reserving an image"));
        }
        mapping.reservation = reservation.cast();
        mapping.reservation_len = reservation_len;
        let base = mapping.reservation;
        // SAFETY: every offset below is within the reservation just made.
        let text = unsafe { base.add(GOT_SPAN as usize) };
        let shared_len = page_round(header.shared_count * 8);
        let own_len = page_round(header.region_count * REGION_SLOTS);
        if shared_len > 0 {
            map_anonymous(base, shared_len, libc::PROT_READ | libc::PROT_WRITE)?;
        }
        if own_len > 0 {
            // SAFETY: within the reservation, below the code.
            let own = unsafe { text.sub(own_len) };
            map_anonymous(own, own_len, libc::PROT_READ | libc::PROT_WRITE)?;
        }
        // SAFETY: within the reservation, after the code.
        let bss = unsafe { text.add(text_len) };
        if bss_len > 0 {
            map_anonymous(bss, bss_len, libc::PROT_READ | libc::PROT_WRITE)?;
        }
        let sites = view.sites()?;
        Ok(MappedImage {
            mapping: Arc::new(mapping),
            header,
            sites,
            text,
            bss,
            resolved: Vec::new(),
            slide: 0,
            file: file
                .try_clone()
                .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?,
            code: Mutex::new(None),
            ready: AtomicBool::new(header.text_len == 0),
        })
    }

    /// Maps the code, executable, where the GOT expects it; returns whether it runs
    /// from the file's own pages rather than a copy. Until this is done no region can
    /// be published.
    ///
    /// Not part of [`open`](Self::open) because it can be slow: the first time a
    /// file's pages are made executable, Darwin validates the whole file, about a
    /// second for every 200 MiB. The validation stays with the file, so a process that
    /// maps it later, or a mapping made after it, costs nothing.
    pub fn map_code(&self) -> crate::Result<bool> {
        let mut code = self.code.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(in_place) = *code {
            return Ok(in_place);
        }
        let text_len = page_round(self.header.text_len);
        let view = self.view();
        let in_place = text_len > 0 && map_code(self.file.as_raw_fd(), &view, self.text);
        if !in_place && text_len > 0 {
            map_anonymous(self.text, text_len, libc::PROT_READ | libc::PROT_WRITE)?;
            // SAFETY: the anonymous pages just mapped hold `text_len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(view.text().as_ptr(), self.text, view.text().len())
            };
            // SAFETY: as above.
            let protected = unsafe {
                libc::mprotect(
                    self.text.cast(),
                    text_len,
                    libc::PROT_READ | libc::PROT_EXEC,
                )
            };
            if protected != 0 {
                return Err(os_error("protecting an image's code"));
            }
            // SAFETY: the range just written.
            unsafe {
                wasmtime_internal_jit_icache_coherence::clear_cache(
                    self.text as *const _,
                    text_len,
                )
                .expect("clearing the instruction cache");
            }
        }
        *code = Some(in_place);
        self.ready.store(true, Ordering::Release);
        Ok(in_place)
    }

    /// Whether [`map_code`](Self::map_code) has been done.
    pub fn code_mapped(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn view(&self) -> View<'_> {
        // SAFETY: the file mapping lives as long as `self.mapping`.
        let bytes = unsafe { std::slice::from_raw_parts(self.mapping.file, self.mapping.file_len) };
        View::with_header(bytes, self.header)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// What the embedder stored beside the code.
    pub fn user(&self) -> &[u8] {
        let range = self.header.user;
        // SAFETY: `View::parse` checked the section lies inside the file mapping.
        unsafe {
            std::slice::from_raw_parts(
                self.mapping.file.add(range.offset as usize),
                range.len as usize,
            )
        }
    }

    /// Whether the code runs from the file's own pages rather than a copy; `false`
    /// until it is mapped.
    pub fn in_place(&self) -> bool {
        self.code
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .unwrap_or(false)
    }

    /// Every import site a region consulted, unslid.
    pub fn sites(&self) -> &[(u64, ImportDecision)] {
        &self.sites
    }

    pub fn region_count(&self) -> u32 {
        self.header.region_count as u32
    }

    /// Fills the shared GOT entries. A symbol `resolver` does not know only refuses
    /// the regions that load it.
    pub fn bind(&mut self, resolver: &dyn SymbolResolver) -> crate::Result<()> {
        self.slide = resolver.guest_slide();
        let shared = self.view().shared()?;
        let got = self.mapping.reservation.cast::<u64>();
        self.resolved = Vec::with_capacity(shared.len());
        for (slot, entry) in shared.iter().enumerate() {
            let value = match entry {
                SharedEntry::GuestSlide => Some(self.slide),
                SharedEntry::Text(offset) => Some(self.text as u64 + offset),
                SharedEntry::Host(symbol) => resolver.resolve(symbol).map(|a| a as u64),
            };
            self.resolved.push(value.is_some());
            // SAFETY: slot `slot` lies in the shared entries mapped by `open`.
            unsafe { got.add(slot).write(value.unwrap_or(0)) };
        }
        Ok(())
    }

    /// The first region that can be entered at `pc`, a run-time address.
    pub fn lookup(&self, pc: u64) -> Option<u32> {
        self.view().lookup(pc.wrapping_sub(self.slide))
    }

    /// Region `index`, in run-time addresses, with its own GOT entries filled, or
    /// `Ok(None)` if it loads a shared entry this process could not resolve.
    pub fn region(&self, index: u32) -> crate::Result<Option<LoadedRegion>> {
        if !self.code_mapped() {
            return Err(crate::MicroliftError::CraneliftError(
                "an image's region was asked for before its code was mapped".into(),
            ));
        }
        let view = self.view();
        let record = view.record(index)?;
        let (meta, shared) = view.meta(&record, &self.sites)?;
        if shared
            .iter()
            .any(|&slot| !self.resolved.get(slot as usize).copied().unwrap_or(false))
        {
            return Ok(None);
        }
        // SAFETY: `View::record` checked the code and cells lie inside the image.
        let (code, cells) = unsafe {
            let code = self.text.add(record.code as usize);
            let cells = record.cells.map_or(std::ptr::null_mut(), |at| {
                self.bss.add(at as usize).cast::<BlockLinkEntry>()
            });
            (code, cells)
        };
        let own = self.text as u64 - REGION_SLOTS * (u64::from(index) + 1);
        // SAFETY: the region's own entries, mapped by `open`; the code loading them is
        // not published until after this returns.
        unsafe {
            (own as *mut u64).write(cells as u64);
            ((own + 8) as *mut u64).write(code as u64 + u64::from(record.link_target));
        }

        let slide = self.slide;
        let slid = |va: u64| va.wrapping_add(slide);
        let slid_ranges = |ranges: &[Range<u64>]| -> Vec<Range<u64>> {
            ranges.iter().map(|r| slid(r.start)..slid(r.end)).collect()
        };
        // SAFETY: `View::record` checked both offsets are inside the region's code.
        let region = unsafe {
            CompiledRegion {
                entry: code.add(record.entry as usize),
                link_target: code.add(record.link_target as usize),
                link_cells: cells,
            }
        };
        let owner: Arc<dyn CodeOwner> = Arc::clone(&self.mapping) as Arc<dyn CodeOwner>;
        let translation = Translation::from_owned(
            region,
            slid(meta.lift_base),
            meta.resume_points.iter().map(|&pc| slid(pc)).collect(),
            slid_ranges(&meta.covered),
            slid_ranges(&meta.data_dependencies),
            meta.link_sites
                .iter()
                .map(|site| LinkSite {
                    target: slid(site.target),
                    kind: site.kind,
                })
                .collect(),
            owner,
        );
        Ok(Some(LoadedRegion {
            translation,
            map: meta.map,
            import_decisions: meta
                .import_decisions
                .into_iter()
                .map(|(va, decision)| (slid(va), decision))
                .collect(),
            lift_base: slid(meta.lift_base),
            code_len: record.code_len,
        }))
    }
}

/// Maps the code from the file at `at`; `false` where the host will not run it so.
fn map_code(fd: libc::c_int, view: &View<'_>, at: *mut u8) -> bool {
    let header = view.header();
    #[cfg(target_os = "macos")]
    if !attach_signature(fd, view) {
        return false;
    }
    // SAFETY: `at` begins the code's part of a reservation this module owns.
    let mapped = unsafe {
        libc::mmap(
            at.cast(),
            header.text_len as usize,
            libc::PROT_READ | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_FIXED,
            fd,
            header.text_offset as libc::off_t,
        )
    };
    mapped != libc::MAP_FAILED
}

/// Gives the kernel the file's signature over its code, without which Darwin will
/// not map it executable.
#[cfg(target_os = "macos")]
fn attach_signature(fd: libc::c_int, view: &View<'_>) -> bool {
    const F_ADDSIGS: libc::c_int = 59;
    #[repr(C)]
    struct FSignatures {
        file_start: libc::off_t,
        blob_start: *mut libc::c_void,
        blob_size: libc::size_t,
        fsignatures_size: libc::size_t,
        cdhash: [u8; 20],
        hash_type: libc::c_int,
    }
    let signature = view.signature();
    if signature.is_empty() {
        return false;
    }
    let mut request = FSignatures {
        file_start: view.header().text_offset as libc::off_t,
        blob_start: signature.as_ptr().cast_mut().cast(),
        blob_size: signature.len(),
        fsignatures_size: 0,
        cdhash: [0; 20],
        hash_type: 0,
    };
    // SAFETY: `request` names a blob that stays mapped for the call.
    unsafe { libc::fcntl(fd, F_ADDSIGS, &mut request) == 0 }
}
