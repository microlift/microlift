//! The shared translation cache: guest address -> compiled region, plus the
//! page index that makes invalidation possible.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::{CompiledRegion, CraneliftBackend};
use crate::core::LiftedCode;
use crate::core::link::{BlockLinkEntry, LinkSite};

/// Guest page size the cache indexes translations by. Only an index
/// granularity: it does not have to match the guest's real page size, and a
/// mismatch costs a wider or narrower invalidation sweep, never a wrong answer.
const PAGE_SHIFT: u32 = 12;

/// One compiled region and everything a host must know to publish, enter and
/// retract it.
///
/// Held behind an [`Arc`] because a translation that has been *unpublished* may
/// still be executing on another thread; see the safepoint rule in the
/// [module doc](super).
pub struct Translation {
    /// Entry point for a host call - `extern "C" fn(*mut RegisterContext)`.
    pub entry: *const u8,
    /// The address a link row must name. **Not** [`entry`](Self::entry) under a
    /// pinned ABI. See [`CompiledRegion`].
    pub link_target: *const u8,
    /// What `ctx.image_base` must hold before entering.
    pub image_base: u64,
    /// Every guest address this region can be entered at - the exact set of
    /// addresses it may be published under.
    ///
    /// **In the same address space as `ctx.rip`**, which is what a link row's
    /// `guest_pc` is compared against and what [`CodeCache`] keys by. Under
    /// [`AddressMode::Direct`](crate::core::AddressMode) - where a region runs at
    /// the address it was lifted for - that is
    /// [`LiftedCode::resume_points`](crate::core::LiftedCode::resume_points)
    /// unchanged. Under
    /// [`ImageBaseRelative`](crate::core::AddressMode::ImageBaseRelative) it is
    /// **not**: those are lift-time VAs, and a host that relocates its artifacts
    /// must relocate them here too, the way
    /// [`LiftEngine::runtime_addr`](crate::core::LiftEngine::runtime_addr) would.
    /// Nothing checks it, and getting it wrong publishes rows that can never
    /// match.
    pub resume_points: Vec<u64>,
    /// The guest bytes it was derived from. The invalidation key.
    ///
    /// **Lift-time** VAs, unlike [`resume_points`](Self::resume_points): this is
    /// matched against the range a host says was written, which it names in
    /// whatever space it tracks its guest mappings in.
    pub covered: Vec<Range<u64>>,
    /// Guest bytes outside `covered` that were read at lift time and baked in.
    pub data_dependencies: Vec<Range<u64>>,
    /// This region's link cells, in cell order.
    pub link_sites: Vec<LinkSite>,
    /// The cell array itself - `link_sites.len()` rows, zeroed until installed.
    /// Null when there are none.
    cells: *mut BlockLinkEntry,
    /// Keeps the JIT module - and therefore the executable pages `entry` and
    /// `link_target` point into - mapped. Dropping the last `Arc` unmaps them,
    /// which is why retraction and reclamation are separate steps.
    _module: CraneliftBackend,
}

// SAFETY: the two raw pointers name executable pages owned by `_module`, which
// is moved in with them and outlives them. Nothing in this struct is mutated
// after construction, so sharing a `&Translation` across threads exposes only
// reads of immutable data. Entering the code it names is `unsafe` and is the
// caller's obligation, as it is for any function pointer.
unsafe impl Send for Translation {}
unsafe impl Sync for Translation {}

impl Translation {
    /// Compile `code` into a module of its own and record everything the cache
    /// needs about it.
    ///
    /// `image_base` is what the region was lifted for; a host using
    /// [`AddressMode::Direct`](crate::core::AddressMode) may pass the lift base.
    pub fn compile(code: LiftedCode, image_base: u64) -> crate::Result<Self> {
        Self::compile_with(code, image_base, CraneliftBackend::new()?)
    }

    /// [`compile`](Self::compile) into a caller-supplied backend, for a host
    /// that needs particular flags or ISA settings.
    ///
    /// The backend must be **fresh**: helper symbols are named per body, so a
    /// module that already holds a region would have them redefined.
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

    /// Record an already-compiled region.
    ///
    /// `module` must be whatever owns the pages `region` points into; the
    /// translation keeps it alive.
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
    /// If `index` is not a cell of this region - which would otherwise be a
    /// write past the end of the array.
    fn cell(&self, index: usize) -> *mut BlockLinkEntry {
        assert!(
            index < self.link_sites.len(),
            "cell {index} is out of range for a region with {} of them",
            self.link_sites.len()
        );
        // SAFETY: `cells` is non-null whenever `link_sites` is non-empty (the
        // backend allocates one row per site) and `index` is in range.
        unsafe { self.cells.add(index) }
    }

    /// Read cell `index`. For tests and diagnostics.
    ///
    /// `code` is read with `Acquire` to pair with the `Release` in `write_cell`,
    /// so a caller that sees an installed cell sees the whole of it. That is a
    /// guarantee *lifted code* does not have - see `write_cell` - and this is the
    /// one reader that can have it, so it does.
    pub fn read_cell(&self, index: usize) -> BlockLinkEntry {
        let cell = self.cell(index);
        // SAFETY: `cell` bounds-checks; the array is this translation's own and
        // is alive as long as it is.
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

    /// The row to publish so that a link to `pc` lands here.
    ///
    /// **The one place `link_target` is chosen over `entry`.** `pc` must be one
    /// of [`resume_points`](Self::resume_points); nothing checks it here, and
    /// [`CodeCache::insert`] is what makes sure it is.
    pub fn row(&self, pc: u64) -> BlockLinkEntry {
        BlockLinkEntry {
            guest_pc: pc,
            code: self.link_target as u64,
            image_base: self.image_base,
            reserved: 0,
        }
    }

    /// Whether this translation can be entered at `pc`.
    ///
    /// Exactly "is `pc` a resume point", by binary search - the set is ascending.
    /// A host publishing rows or answering a lookup wants this and not "is `pc`
    /// inside `covered`": the two differ at every address the region contains but
    /// cannot be resumed at, and entering one of those either runs the region
    /// from the top or exits having made no progress.
    pub fn covers(&self, pc: u64) -> bool {
        self.resume_points.binary_search(&pc).is_ok()
    }

    /// Whether a write to `[start, start + len)` may have changed what this
    /// translation means.
    fn depends_on(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        let hit = |r: &Range<u64>| r.start < end && start < r.end;
        self.covered.iter().any(hit) || self.data_dependencies.iter().any(hit)
    }

    /// Every page index this translation's code and data occupy.
    fn pages(&self) -> impl Iterator<Item = u64> + '_ {
        self.covered
            .iter()
            .chain(self.data_dependencies.iter())
            .flat_map(|r| (r.start >> PAGE_SHIFT)..=((r.end - 1) >> PAGE_SHIFT))
    }
}

/// How the cache is being used. Not diagnostics for their own sake: the whole
/// claim of block linking is that the host stops being asked, and
/// [`miss_rate`](Self::miss_rate) is the number that says whether it did.
///
/// A **snapshot**, taken by [`CodeCache::stats`]. The live counters behind it are
/// atomic so that [`CodeCache::lookup`] needs only `&self` - a host that shares
/// one cache behind an `RwLock` would otherwise have to take the *write* lock on
/// every miss, which is exactly the path this type exists to prove is cold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Times a host dispatch had to look an address up - i.e. lifted code
    /// exited without resolving its own successor. **The cost linking exists to
    /// remove.**
    pub lookups: u64,
    /// Of those, how many found a translation already compiled.
    pub hits: u64,
    /// Translations inserted.
    pub inserts: u64,
    /// Translations unpublished by an invalidation.
    pub retractions: u64,
    /// Link cells filled in with a compiled target.
    pub cells_installed: u64,
    /// Link cells cleared by an invalidation.
    pub cells_retracted: u64,
    /// Times a region resolved its own successor and the host was not consulted.
    pub links_followed: u64,
}

impl CacheStats {
    /// Fraction of host lookups that found nothing compiled, or `None` if there
    /// have been no lookups - which for a steady-state hot loop is the answer
    /// you want.
    pub fn miss_rate(&self) -> Option<f64> {
        (self.lookups > 0).then(|| (self.lookups - self.hits) as f64 / self.lookups as f64)
    }
}

/// The live counters [`CacheStats`] is a snapshot of.
///
/// Atomic, and `Relaxed` throughout: these order nothing and guard nothing, they
/// are counted for a human. What they buy by being atomic is `&self` on
/// [`CodeCache::lookup`].
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

/// The shared guest-address -> translation map.
#[derive(Default)]
pub struct CodeCache {
    /// Every address any translation can be entered at. One translation appears
    /// once per resume point, which is why the value is an `Arc`.
    by_resume_point: HashMap<u64, Arc<Translation>>,
    /// Page index -> the entry addresses of translations touching that page.
    /// `BTreeMap` because invalidation asks for a *range* of pages.
    pages: BTreeMap<u64, Vec<u64>>,
    /// Every installed link cell, keyed by the guest address it names.
    links: HashMap<u64, Vec<CellRef>>,
    /// Cells whose target had no translation when the region was inserted, so
    /// they can be filled the moment one appears.
    pending: HashMap<u64, Vec<CellRef>>,
    /// Translations unpublished but not yet freed. See the safepoint rule.
    retired: Vec<Arc<Translation>>,
    stats: Counters,
}

/// Publish `row` into the cell at `cell`, or retract it when `row` is `None`.
///
/// **A cell is the one part of block linking written while another thread may be
/// executing the code that reads it.** A table row is not: the table is
/// per-thread, and its owner writes it only while outside lifted code. A cell is
/// in a shared artifact and the cache fills it the moment its target compiles,
/// which may be in the middle of a different thread's exit through that very
/// cell.
///
/// So `code` is written **last and with `Release`**, and everything else before
/// it: `code` is what the probe tests, and a reader that observes the new one has
/// the rest of the row behind it. Retraction writes only `code`, leaving the other
/// fields alone - a zero `code` means the row is dead and nothing reads further.
///
/// The residual, stated rather than hidden: the reader is *lifted code* doing
/// plain loads, so it has no acquire to pair with this. On a weakly-ordered
/// target a load of `image_base` in the taken arm may in principle be speculated
/// ahead of the load of `code` that selected it, yielding the pre-install value.
/// Only [`LinkDispatch::Pinned`](crate::core::link::LinkDispatch::Pinned) reads
/// `image_base` from a cell at all, and only a host using
/// [`ImageBaseRelative`](crate::core::AddressMode::ImageBaseRelative) has a
/// non-constant one to get wrong - under
/// [`Direct`](crate::core::AddressMode::Direct) lifted code ignores the field
/// entirely. Closing it needs an acquire the lifter cannot currently emit.
///
/// # Safety
///
/// `cell` must be a live cell of a mapped artifact - which is what a [`CellRef`]
/// holding its owner guarantees.
unsafe fn write_cell(cell: *mut BlockLinkEntry, row: Option<BlockLinkEntry>) {
    // SAFETY: the caller's obligation. `&raw mut` rather than a reference, so no
    // `&mut` to the row ever exists while lifted code may be reading it.
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
        // `guest_pc` and `image_base` are left behind deliberately: they are only
        // ever read after `code` tested non-zero, and leaving them costs nothing
        // while writing them would be two more stores racing the same reader.
        None => code.store(0, Ordering::Release),
    }
}

/// One installable link cell, as the cache tracks it.
///
/// A raw pointer plus the `Arc` that keeps it mapped: a cell belongs to a
/// compiled artifact, and clearing one after that artifact was freed is a
/// use-after-free. Holding the owner is what makes retraction safe in either
/// order.
struct CellRef {
    cell: *mut BlockLinkEntry,
    owner: Arc<Translation>,
}

// SAFETY: `cell` points into `owner`'s cell array, which `owner` keeps mapped.
// Every access goes through `&mut CodeCache`, so the host's own lock serialises
// them against each other; racing them against *lifted code* reading the same
// cell is the documented model - an 8-byte aligned store, seen whole or not at
// all. See the module doc.
unsafe impl Send for CellRef {}
unsafe impl Sync for CellRef {}

impl CodeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    /// How many **distinct** translations are published.
    ///
    /// Not `CacheStats::inserts`, which counts calls: two threads that missed on
    /// one address both compile it, and the loser is displaced here rather than
    /// counted twice. Not the number of resume points either - one translation
    /// answers at every address it can be entered at.
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

    /// Whether nothing is published.
    pub fn is_empty(&self) -> bool {
        self.by_resume_point.is_empty()
    }

    /// Record `translation` and make it findable at every address it can be
    /// entered at.
    ///
    /// A translation with no resume points is still inserted and simply never
    /// found by address - that is what
    /// [`VmResumptionPoints::None`](crate::core::VmResumptionPoints) asks for.
    pub fn insert(&mut self, translation: Translation) -> Arc<Translation> {
        let translation = Arc::new(translation);
        for &pc in &translation.resume_points {
            self.by_resume_point.insert(pc, Arc::clone(&translation));
        }
        // The index names resume points rather than translations because that is
        // what `by_resume_point` is keyed by, and a page entry that cannot be
        // turned back into a translation is useless. Every resume point, not
        // just the first: the sweep must still find this translation after an
        // earlier invalidation removed some of them.
        for page in translation.pages() {
            let slot = self.pages.entry(page).or_default();
            slot.extend_from_slice(&translation.resume_points);
        }
        self.install_cells(&translation);
        bump(&self.stats.inserts, 1);
        translation
    }

    /// Fill in both directions of the cell graph for a newly inserted
    /// translation: its own cells whose targets are already compiled, and every
    /// cell elsewhere that was waiting for *it*.
    ///
    /// **Eager, unlike FEX.** FEX must resolve a static edge on its first
    /// execution, because a stub is the only way it can learn the host address.
    /// Here the host knows at insert time, so a static edge's first execution is
    /// already a load and a branch rather than a trip through the runtime.
    fn install_cells(&mut self, translation: &Arc<Translation>) {
        for (index, site) in translation.link_sites.iter().enumerate() {
            let cell = CellRef {
                cell: translation.cell(index),
                owner: Arc::clone(translation),
            };
            match self.by_resume_point.get(&site.target) {
                Some(target) => {
                    let row = target.row(site.target);
                    // SAFETY: the cell is in `translation`'s own array, which
                    // `cell` bounds-checked and `owner` keeps mapped.
                    unsafe { write_cell(cell.cell, Some(row)) };
                    self.links.entry(site.target).or_default().push(cell);
                    bump(&self.stats.cells_installed, 1);
                }
                // Nothing compiled there yet. The exit falls through to the
                // table probe until something is, which is what makes the
                // fallback in `emit_static_exit` load-bearing rather than
                // decorative.
                None => self.pending.entry(site.target).or_default().push(cell),
            }
        }

        // The other direction: somebody has been waiting for this address.
        for &pc in &translation.resume_points {
            let Some(waiting) = self.pending.remove(&pc) else {
                continue;
            };
            let row = translation.row(pc);
            for cell in waiting {
                // SAFETY: as above - `owner` is alive because the `CellRef`
                // holds it.
                unsafe { write_cell(cell.cell, Some(row)) };
                bump(&self.stats.cells_installed, 1);
                self.links.entry(pc).or_default().push(cell);
            }
        }
    }

    /// Clear every cell naming `pc`, so the exits that used it fall back to the
    /// table probe.
    ///
    /// FEX's `DirectBlockDelinker`, minus the instruction encoding: zeroing
    /// `code` is exactly the store the probe's `if code == 0` exists to observe.
    /// The whole row goes, not just `code`, so a retracted cell is
    /// indistinguishable from one that was never installed.
    fn retract_cells(&mut self, pc: u64) -> usize {
        let Some(cells) = self.links.remove(&pc) else {
            return 0;
        };
        let n = cells.len();
        for cell in &cells {
            // SAFETY: `owner` is still alive, which is the entire reason a
            // `CellRef` holds it - the target being retracted says nothing about
            // whether the region *containing* the cell was freed.
            unsafe { write_cell(cell.cell, None) };
        }
        // Back to pending: if the address is translated again the cell should
        // link again, without the region having to be re-lifted.
        self.pending.entry(pc).or_default().extend(cells);
        n
    }

    /// The translation enterable at `pc`, counting the lookup.
    ///
    /// Every call is a host dispatch that block linking failed to avoid, so this
    /// is deliberately the *only* way to look one up: making the counter
    /// unavoidable is what keeps [`CacheStats::miss_rate`] honest.
    ///
    /// `&self`, so a host sharing one cache behind an `RwLock` looks up under the
    /// **read** lock. That is the whole point of the counters being atomic: a
    /// miss is on its way to a lift, and two threads missing on different
    /// addresses should compile at once rather than queue behind a counter.
    pub fn lookup(&self, pc: u64) -> Option<Arc<Translation>> {
        bump(&self.stats.lookups, 1);
        let found = self.by_resume_point.get(&pc).cloned();
        if found.is_some() {
            bump(&self.stats.hits, 1);
        }
        found
    }

    /// Look `pc` up without counting it - for tests and for a host inspecting
    /// its own cache.
    pub fn peek(&self, pc: u64) -> Option<&Arc<Translation>> {
        self.by_resume_point.get(&pc)
    }

    /// Record that `n` successors were resolved without a host lookup.
    ///
    /// Public because the count comes back from
    /// [`enter`](super::enter) rather than being taken inside it: the chain runs
    /// guest code, and counting from in there would hold this cache's lock for
    /// its duration.
    pub fn note_links_followed(&self, n: usize) {
        if n > 0 {
            bump(&self.stats.links_followed, n as u64);
        }
    }

    /// Retire every translation that may have been changed by a write to
    /// `[start, start + len)`, returning the guest addresses that were
    /// unpublished so each thread can retract them from its own table.
    ///
    /// This is the half of invalidation the cache can do alone. The tables are
    /// per-thread and this does not know about them, which is why the addresses
    /// come back rather than being cleared here - see
    /// [`ThreadRuntime::invalidate_range`](super::ThreadRuntime::invalidate_range).
    pub fn invalidate_range(&mut self, start: u64, len: u64) -> Vec<u64> {
        let end = start.saturating_add(len);
        let (lo, hi) = (start >> PAGE_SHIFT, (end.saturating_sub(1)) >> PAGE_SHIFT);

        // The page index narrows the sweep; `depends_on` is what decides, so a
        // page shared by an unrelated translation costs a check and not a
        // spurious retraction.
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
            // A translation is published at *every* resume point it has, and
            // must be removed from all of them - not just the one the page
            // index happened to name.
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

        // Two directions again, and both are needed. Cells *pointing at* a dying
        // translation must be cleared or they enter freed code; cells *belonging
        // to* one must be forgotten or a later install writes into an array that
        // is about to be unmapped.
        for pc in &unpublished {
            let cleared = self.retract_cells(*pc) as u64;
            bump(&self.stats.cells_retracted, cleared);
        }
        for translation in &doomed {
            self.forget_cells_of(translation);
        }

        bump(&self.stats.retractions, doomed.len() as u64);
        // Held, not dropped: another thread may be executing this code right
        // now. `reclaim` is what frees it.
        self.retired.extend(doomed);
        unpublished
    }

    /// Drop every reference to cells owned by `translation`, from both the
    /// installed and the pending map.
    ///
    /// Without this the maps would keep the dying region's `Arc` alive forever -
    /// so this is what makes [`reclaim`](Self::reclaim) able to actually unmap
    /// anything - and a later install would write into its array.
    fn forget_cells_of(&mut self, translation: &Arc<Translation>) {
        let owned = |cells: &mut Vec<CellRef>| {
            cells.retain(|c| !Arc::ptr_eq(&c.owner, translation));
        };
        self.links.values_mut().for_each(owned);
        self.pending.values_mut().for_each(owned);
        self.links.retain(|_, cells| !cells.is_empty());
        self.pending.retain(|_, cells| !cells.is_empty());
    }

    /// How many cells are installed, and how many are waiting for a target.
    /// For tests and diagnostics.
    pub fn cell_counts(&self) -> (usize, usize) {
        (
            self.links.values().map(Vec::len).sum(),
            self.pending.values().map(Vec::len).sum(),
        )
    }

    /// Free every retired translation.
    ///
    /// **Only safe once every guest thread has passed a safepoint since the
    /// invalidation that retired them** - i.e. has been observed outside lifted
    /// code. A single-threaded host may call this immediately after
    /// [`invalidate_range`](Self::invalidate_range).
    ///
    /// Returns how many were freed. A translation still held elsewhere (an
    /// `Arc` a caller kept) is released here but not unmapped until that clone
    /// drops.
    pub fn reclaim(&mut self) -> usize {
        let n = self.retired.len();
        self.retired.clear();
        n
    }

    /// How many translations are retired and awaiting reclamation.
    pub fn retired_count(&self) -> usize {
        self.retired.len()
    }
}
