//! Shared translation cache: guest address -> compiled region, with a page index for invalidation.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::{CompiledRegion, CraneliftBackend};
use crate::core::LiftedCode;
use crate::core::link::{BlockLinkEntry, LinkSite};

/// Index granularity only; need not match the guest page size (a mismatch widens or narrows sweeps).
const PAGE_SHIFT: u32 = 12;

/// One compiled region and what a host needs to publish, enter and retract it.
///
/// Behind an [`Arc`]: an unpublished translation may still be executing on another thread.
pub struct Translation {
    /// `extern "C" fn(*mut RegisterContext)`.
    pub entry: *const u8,
    /// What a link row must name; **not** `entry` under a pinned ABI ([`CompiledRegion`]).
    pub link_target: *const u8,
    pub image_base: u64,
    /// Addresses this region can be entered at: the set it may be published under.
    ///
    /// **In `ctx.rip`'s space.** Under `AddressMode::Direct` this is
    /// [`LiftedCode::resume_points`](crate::core::LiftedCode::resume_points); under
    /// [`ImageBaseRelative`](crate::core::AddressMode::ImageBaseRelative) the host must relocate them
    /// (unchecked; wrong values publish rows that never match).
    pub resume_points: Vec<u64>,
    /// Guest bytes derived from: the invalidation key, in **lift-time** VAs (unlike `resume_points`).
    pub covered: Vec<Range<u64>>,
    /// Bytes outside `covered` read at lift time and baked in.
    pub data_dependencies: Vec<Range<u64>>,
    pub link_sites: Vec<LinkSite>,
    /// The zeroed-until-installed cell array; null when empty.
    cells: *mut BlockLinkEntry,
    /// Keeps the JIT pages mapped; dropping the last `Arc` unmaps them.
    _module: CraneliftBackend,
}

// SAFETY: the raw pointers name pages owned by `_module`, and nothing is mutated after construction.
unsafe impl Send for Translation {}
unsafe impl Sync for Translation {}

impl Translation {
    /// Compiles `code` into a module of its own. `image_base` is what the region was lifted for.
    pub fn compile(code: LiftedCode, image_base: u64) -> crate::Result<Self> {
        Self::compile_with(code, image_base, CraneliftBackend::new()?)
    }

    /// [`compile`](Self::compile) into a caller-supplied **fresh** backend (helper symbols are named per body).
    pub fn compile_with(
        code: LiftedCode,
        image_base: u64,
        mut module: CraneliftBackend,
    ) -> crate::Result<Self> {
        let resume_points = code.resume_points.clone();
        let covered = code.covered.clone();
        let data_dependencies = code.data_dependencies.clone();
        let link_sites = code.link_sites.clone();
        let (region, _map) = module.compile_lifted_region(code)?;
        Ok(Self::from_compiled(
            region,
            image_base,
            resume_points,
            covered,
            data_dependencies,
            link_sites,
            module,
        ))
    }

    /// `module` must own the pages `region` points into.
    #[allow(clippy::too_many_arguments)]
    pub fn from_compiled(
        region: CompiledRegion,
        image_base: u64,
        resume_points: Vec<u64>,
        covered: Vec<Range<u64>>,
        data_dependencies: Vec<Range<u64>>,
        link_sites: Vec<LinkSite>,
        module: CraneliftBackend,
    ) -> Self {
        Self {
            entry: region.entry,
            link_target: region.link_target,
            image_base,
            resume_points,
            covered,
            data_dependencies,
            link_sites,
            cells: region.link_cells,
            _module: module,
        }
    }

    /// The address of cell `index`.
    ///
    /// # Panics
    ///
    /// If `index` is not a cell of this region.
    fn cell(&self, index: usize) -> *mut BlockLinkEntry {
        assert!(
            index < self.link_sites.len(),
            "cell {index} is out of range for a region with {} of them",
            self.link_sites.len()
        );
        // SAFETY: `cells` is non-null whenever `link_sites` is non-empty and `index` is in range.
        unsafe { self.cells.add(index) }
    }

    /// Reads cell `index` for tests/diagnostics; `code` is read `Acquire`, pairing with `write_cell`.
    pub fn read_cell(&self, index: usize) -> BlockLinkEntry {
        let cell = self.cell(index);
        // SAFETY: `cell` bounds-checks and the array is this translation's own.
        unsafe {
            let code = AtomicU64::from_ptr(&raw mut (*cell).code).load(Ordering::Acquire);
            BlockLinkEntry {
                guest_pc: (*cell).guest_pc,
                code,
                image_base: (*cell).image_base,
                reserved: (*cell).reserved,
            }
        }
    }

    /// The row to publish so a link to `pc` lands here (where `link_target` is chosen over `entry`).
    ///
    /// `pc` must be a resume point; [`CodeCache::insert`] ensures it.
    pub fn row(&self, pc: u64) -> BlockLinkEntry {
        BlockLinkEntry {
            guest_pc: pc,
            code: self.link_target as u64,
            image_base: self.image_base,
            reserved: 0,
        }
    }

    /// Whether `pc` is a resume point (binary search), not merely inside `covered`: entering a
    /// non-resume address runs the region from the top or exits without progress.
    pub fn covers(&self, pc: u64) -> bool {
        self.resume_points.binary_search(&pc).is_ok()
    }

    /// Whether a write to `[start, start + len)` may have changed this translation.
    fn depends_on(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        let hit = |r: &Range<u64>| r.start < end && start < r.end;
        self.covered.iter().any(hit) || self.data_dependencies.iter().any(hit)
    }

    fn pages(&self) -> impl Iterator<Item = u64> + '_ {
        self.covered
            .iter()
            .chain(self.data_dependencies.iter())
            .flat_map(|r| (r.start >> PAGE_SHIFT)..=((r.end - 1) >> PAGE_SHIFT))
    }
}

/// Cache usage snapshot ([`CodeCache::stats`]); [`miss_rate`](Self::miss_rate) says whether linking works.
///
/// Live counters are atomic so [`CodeCache::lookup`] needs only `&self`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Host lookups, i.e. exits that did not resolve their own successor. **The cost linking removes.**
    pub lookups: u64,
    /// Lookups that found a translation.
    pub hits: u64,
    pub inserts: u64,
    pub retractions: u64,
    pub cells_installed: u64,
    pub cells_retracted: u64,
    /// Successors resolved without the host.
    pub links_followed: u64,
}

impl CacheStats {
    /// Fraction of lookups that missed; `None` with no lookups.
    pub fn miss_rate(&self) -> Option<f64> {
        (self.lookups > 0).then(|| (self.lookups - self.hits) as f64 / self.lookups as f64)
    }
}

/// Live counters behind [`CacheStats`]: `Relaxed` (counted for humans); atomic only for `&self` lookups.
#[derive(Default)]
struct Counters {
    lookups: AtomicU64,
    hits: AtomicU64,
    inserts: AtomicU64,
    retractions: AtomicU64,
    cells_installed: AtomicU64,
    cells_retracted: AtomicU64,
    links_followed: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> CacheStats {
        let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
        CacheStats {
            lookups: read(&self.lookups),
            hits: read(&self.hits),
            inserts: read(&self.inserts),
            retractions: read(&self.retractions),
            cells_installed: read(&self.cells_installed),
            cells_retracted: read(&self.cells_retracted),
            links_followed: read(&self.links_followed),
        }
    }
}

fn bump(counter: &AtomicU64, by: u64) {
    counter.fetch_add(by, Ordering::Relaxed);
}

#[derive(Default)]
pub struct CodeCache {
    /// Every entry address; one translation appears once per resume point.
    by_resume_point: HashMap<u64, Arc<Translation>>,
    /// Page index -> entry addresses touching that page (`BTreeMap`: invalidation sweeps ranges).
    pages: BTreeMap<u64, Vec<u64>>,
    /// Installed cells by target address.
    links: HashMap<u64, Vec<CellRef>>,
    /// Cells whose target was not yet translated.
    pending: HashMap<u64, Vec<CellRef>>,
    /// Unpublished, not yet freed (safepoint rule).
    retired: Vec<Arc<Translation>>,
    stats: Counters,
}

/// Publishes `row` into `cell`, or retracts it when `row` is `None`.
///
/// A cell is the one part of linking written while another thread may be executing the code
/// that reads it. `code` is written **last with `Release`**; retraction writes only `code`.
///
/// Residual: lifted code has no acquire, so on weakly-ordered targets `image_base` may be read
/// stale. Only `LinkDispatch::Pinned` under `ImageBaseRelative` is exposed.
///
/// # Safety
///
/// `cell` must be a live cell of a mapped artifact (what a [`CellRef`] guarantees).
unsafe fn write_cell(cell: *mut BlockLinkEntry, row: Option<BlockLinkEntry>) {
    // SAFETY: the caller's obligation; `&raw mut` so no `&mut` exists while lifted code reads.
    let code = unsafe { AtomicU64::from_ptr(&raw mut (*cell).code) };
    match row {
        Some(row) => {
            // SAFETY: as above.
            unsafe {
                (*cell).guest_pc = row.guest_pc;
                (*cell).image_base = row.image_base;
                (*cell).reserved = row.reserved;
            }
            code.store(row.code, Ordering::Release);
        }
        // `guest_pc`/`image_base` are left: only read after `code` tests non-zero.
        None => code.store(0, Ordering::Release),
    }
}

/// An installable link cell: a raw pointer plus the owner `Arc` that keeps it mapped (so retraction is safe in either order).
struct CellRef {
    cell: *mut BlockLinkEntry,
    owner: Arc<Translation>,
}

// SAFETY: `cell` points into `owner`'s array; access is serialised by the host's lock, and racing
// lifted code is the documented model (aligned 8-byte store).
unsafe impl Send for CellRef {}
unsafe impl Sync for CellRef {}

impl CodeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    /// Distinct published translations (not `inserts`, not resume points).
    pub fn len(&self) -> usize {
        let mut seen: Vec<*const Translation> = self
            .by_resume_point
            .values()
            .map(Arc::as_ptr)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_resume_point.is_empty()
    }

    /// Records `translation` and makes it findable at every resume point (none: never found by address).
    pub fn insert(&mut self, translation: Translation) -> Arc<Translation> {
        let translation = Arc::new(translation);
        for &pc in &translation.resume_points {
            self.by_resume_point.insert(pc, Arc::clone(&translation));
        }
        // Every resume point, so the sweep finds this after earlier invalidations removed some.
        for page in translation.pages() {
            let slot = self.pages.entry(page).or_default();
            slot.extend_from_slice(&translation.resume_points);
        }
        self.install_cells(&translation);
        bump(&self.stats.inserts, 1);
        translation
    }

    /// Fills the cell graph both ways: this translation's cells with compiled targets, and cells
    /// elsewhere waiting for it. Eager, so a static edge's first execution is already a branch.
    fn install_cells(&mut self, translation: &Arc<Translation>) {
        for (index, site) in translation.link_sites.iter().enumerate() {
            let cell = CellRef {
                cell: translation.cell(index),
                owner: Arc::clone(translation),
            };
            match self.by_resume_point.get(&site.target) {
                Some(target) => {
                    let row = target.row(site.target);
            // SAFETY: the cell is in `translation`'s own bounds-checked array.
                    unsafe { write_cell(cell.cell, Some(row)) };
                    self.links.entry(site.target).or_default().push(cell);
                    bump(&self.stats.cells_installed, 1);
                }
            // Not compiled yet: the exit falls through to the table probe.
                None => self.pending.entry(site.target).or_default().push(cell),
            }
        }

        // Cells elsewhere waiting for this address.
        for &pc in &translation.resume_points {
            let Some(waiting) = self.pending.remove(&pc) else {
                continue;
            };
            let row = translation.row(pc);
            for cell in waiting {
            // SAFETY: `owner` is alive via the `CellRef`.
                unsafe { write_cell(cell.cell, Some(row)) };
                bump(&self.stats.cells_installed, 1);
                self.links.entry(pc).or_default().push(cell);
            }
        }
    }

    /// Clears every cell naming `pc` (whole row, indistinguishable from never installed).
    fn retract_cells(&mut self, pc: u64) -> usize {
        let Some(cells) = self.links.remove(&pc) else {
            return 0;
        };
        let n = cells.len();
        for cell in &cells {
            // SAFETY: `owner` is alive; the target being retracted says nothing about the cell's region.
            unsafe { write_cell(cell.cell, None) };
        }
        // Back to pending, so a re-translation relinks.
        self.pending.entry(pc).or_default().extend(cells);
        n
    }

    /// The translation enterable at `pc`, counting the lookup.
    ///
    /// The only lookup, so [`CacheStats::miss_rate`] stays honest. `&self`: a host takes the **read** lock.
    pub fn lookup(&self, pc: u64) -> Option<Arc<Translation>> {
        bump(&self.stats.lookups, 1);
        let found = self.by_resume_point.get(&pc).cloned();
        if found.is_some() {
            bump(&self.stats.hits, 1);
        }
        found
    }

    /// Looks up `pc` without counting.
    pub fn peek(&self, pc: u64) -> Option<&Arc<Translation>> {
        self.by_resume_point.get(&pc)
    }

    /// Records `n` successors resolved without a lookup (public: [`enter`](super::enter) returns the
    /// count so the lock is not held across guest code).
    pub fn note_links_followed(&self, n: usize) {
        if n > 0 {
            bump(&self.stats.links_followed, n as u64);
        }
    }

    /// Retires translations a write to `[start, start + len)` may have changed; returns the
    /// unpublished addresses for each thread to retract
    /// ([`ThreadRuntime::invalidate_range`](super::ThreadRuntime::invalidate_range)).
    pub fn invalidate_range(&mut self, start: u64, len: u64) -> Vec<u64> {
        let end = start.saturating_add(len);
        let (lo, hi) = (start >> PAGE_SHIFT, (end.saturating_sub(1)) >> PAGE_SHIFT);

        // The page index narrows the sweep; `depends_on` decides.
        let candidates: Vec<u64> = self
            .pages
            .range(lo..=hi)
            .flat_map(|(_, entries)| entries.iter().copied())
            .collect();

        let mut unpublished = Vec::new();
        let mut doomed: Vec<Arc<Translation>> = Vec::new();
        for pc in candidates {
            let Some(translation) = self.by_resume_point.get(&pc) else {
                continue;
            };
            if !translation.depends_on(start, len) {
                continue;
            }
            let translation = Arc::clone(translation);
            // Remove from every resume point, not just the one the page index named.
            for &rp in &translation.resume_points {
                if self
                    .by_resume_point
                    .get(&rp)
                    .is_some_and(|t| Arc::ptr_eq(t, &translation))
                {
                    self.by_resume_point.remove(&rp);
                    unpublished.push(rp);
                }
            }
            doomed.push(translation);
        }

        for translation in &doomed {
            for page in translation.pages() {
                if let Some(entries) = self.pages.get_mut(&page) {
                    entries.retain(|pc| !translation.resume_points.contains(pc));
                }
            }
        }
        self.pages.retain(|_, entries| !entries.is_empty());

        // Clear cells pointing at the dying translation and forget cells belonging to it.
        for pc in &unpublished {
            let cleared = self.retract_cells(*pc) as u64;
            bump(&self.stats.cells_retracted, cleared);
        }
        for translation in &doomed {
            self.forget_cells_of(translation);
        }

        bump(&self.stats.retractions, doomed.len() as u64);
        // Held, not dropped: another thread may be executing it until `reclaim`.
        self.retired.extend(doomed);
        unpublished
    }

    /// Drops every cell reference owned by `translation` (else its `Arc` lives forever and installs write into it).
    fn forget_cells_of(&mut self, translation: &Arc<Translation>) {
        let owned = |cells: &mut Vec<CellRef>| {
            cells.retain(|c| !Arc::ptr_eq(&c.owner, translation));
        };
        self.links.values_mut().for_each(owned);
        self.pending.values_mut().for_each(owned);
        self.links.retain(|_, cells| !cells.is_empty());
        self.pending.retain(|_, cells| !cells.is_empty());
    }

    /// Installed and pending cell counts.
    pub fn cell_counts(&self) -> (usize, usize) {
        (
            self.links.values().map(Vec::len).sum(),
            self.pending.values().map(Vec::len).sum(),
        )
    }

    /// Frees retired translations; a translation still held elsewhere unmaps when that clone drops.
    ///
    /// **Only safe once every thread has been observed outside lifted code since the invalidation.**
    /// Returns how many were freed.
    pub fn reclaim(&mut self) -> usize {
        let n = self.retired.len();
        self.retired.clear();
        n
    }

    pub fn retired_count(&self) -> usize {
        self.retired.len()
    }
}
