// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Storage tiers behind a trait, plus the promotion/demotion policy that joins them.
//!
//! Three tiers, exactly the ones the telemetry names:
//!
//! * Tier::Uma is the coherent, GPU-visible pool (Tier 1);
//! * Tier::Dram is the staging and promotion tier;
//! * Tier::Nvme is the persistent 72-hour cache (Tier 2).
//!
//! The NVMe tier mirrors the alignment contract the Milestone 2 Engram store uses rather
//! than hardcoding a sector size: it *asks the device*. [probe_direct_alignment] opens a
//! scratch file with O_DIRECT and finds the smallest power-of-two alignment the filesystem
//! actually accepts, seeding the candidates from the directory's preferred I/O size. The
//! file-backed tier then pads every page into an aligned slot, so an O_DIRECT transfer is
//! always an aligned window sliced to the logical page boundary, never a bare page-sized
//! write that a 4Kn device would reject.
//!
//! The Engram store lives in a sibling crate that was still under construction when this
//! module was written, so the code could not depend on it; the contract (device-queried
//! alignment, aligned windows sliced to records, a batched submission seam) is mirrored
//! exactly so the two can be unified later without changing callers.
//!
//! # Submission is synchronous today
//!
//! There is no io_uring here, and no feature flag that pretends otherwise. Every NVMe
//! transfer is one blocking pread/pwrite per page ([FileBackedTier::read_page] and
//! [FileBackedTier::write_page] go through `FileExt::read_exact_at` / `write_all_at`), issued
//! on the calling thread. Correctness, the device-queried alignment and the byte-exact
//! promotion path are all testable this way, but queue depth and batched submission are not
//! implemented, and no latency claim may rest on this tier.
//!
//! The seam for batching is the [TierBackend] trait, not a flag: its methods take one page at
//! a time, so a ring-backed backend adds a multi-page submit/complete pair behind the same
//! trait and the callers in [TieredPageStore] keep working unchanged. Spec 5.5 does not ask
//! for io_uring on the KV tier - spec 5.3 does, for the Engram store - so this module stays
//! on the portable path until Milestone 2's store exists to reuse.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::thread::ThreadId;

use parking_lot::Mutex;

use crate::error::KvError;
use crate::layout::{PageId, align_up};
use crate::telemetry::KvStoreTelemetry;

/// A storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Tier 1: coherent unified memory, directly visible to the GPU.
    Uma,
    /// Staging tier: unified DRAM used for promotion and demotion.
    Dram,
    /// Tier 2: the persistent NVMe cache.
    Nvme,
}

impl Tier {
    /// Canonical lower-case name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uma => "uma",
            Self::Dram => "dram",
            Self::Nvme => "nvme",
        }
    }

    /// The next tier down, or None for the terminal tier.
    #[must_use]
    pub const fn next_down(self) -> Option<Self> {
        match self {
            Self::Uma => Some(Self::Dram),
            Self::Dram => Some(Self::Nvme),
            Self::Nvme => None,
        }
    }
}

/// A whole-page storage backend.
pub trait TierBackend: Send + Sync {
    /// Which tier this backend is.
    fn tier(&self) -> Tier;
    /// Human-readable backend name, used in diagnostics.
    fn name(&self) -> &'static str;
    /// Logical page size in bytes.
    fn page_bytes(&self) -> usize;
    /// Pages this backend can hold.
    fn capacity_pages(&self) -> u64;
    /// Pages this backend currently holds.
    fn resident_pages(&self) -> u64;
    /// Alignment this backend needs from callers, in bytes.
    fn alignment(&self) -> u64;
    /// True when this backend currently holds the page.
    fn contains(&self, page: PageId) -> bool;
    /// Writes a whole page.
    ///
    /// # Errors
    /// Returns an error when the payload has the wrong length or the write fails.
    fn write_page(&self, page: PageId, data: &[u8]) -> Result<(), KvError>;
    /// Reads a whole page.
    ///
    /// # Errors
    /// Returns an error when the output has the wrong length, the page is absent, or the
    /// read fails.
    fn read_page(&self, page: PageId, out: &mut [u8]) -> Result<(), KvError>;
    /// Drops a page. Dropping an absent page is not an error.
    ///
    /// # Errors
    /// Returns an error when the backend cannot complete the removal.
    fn remove_page(&self, page: PageId) -> Result<(), KvError>;
    /// Flushes any persistent index. In-memory tiers have nothing to flush.
    ///
    /// # Errors
    /// Returns an error when the flush fails.
    fn flush(&self) -> Result<(), KvError> {
        Ok(())
    }

    /// Installs the guard this backend refuses replay-time writes through.
    ///
    /// [TieredPageStore] calls this on every backend it owns, so a write attempted from a
    /// thread inside a bounded replay fails on every path, including a direct
    /// [TieredPageStore::backend] call that bypasses the store's own publish check. A backend
    /// that does not override this stays unguarded on its own: that reaches a caller who drives
    /// the type without a store, and also a store caller who used the raw
    /// [TieredPageStore::backend] handle to install a backend that skipped this call.
    fn set_replay_guard(&self, _guard: ReplayGuard) {}
}

/// Per-thread bounded-replay scopes, shared by a store, its backends and its cache.
///
/// A scope marks the calling thread as replaying, which is what stops a bounded replay from
/// publishing global KV: the store's publish path and every guarded backend refuse while the
/// scope is open. Scopes nest, each one is released by dropping the guard that
/// [ReplayState::enter] returns, and the thread stays in scope until its last guard is gone.
/// Because the guard is the only way out, a replay that panics cannot leave the thread marked
/// as replaying for the life of the cache.
#[derive(Debug)]
pub struct ReplayState {
    /// Nesting depth per thread; a thread absent from the map is not replaying.
    scopes: Mutex<HashMap<ThreadId, u32>>,
    /// Counter every scope is recorded in, so opening one is visible wherever it happens.
    telemetry: Arc<KvStoreTelemetry>,
}

impl ReplayState {
    /// Creates an empty replay state that records its scopes in a telemetry sink.
    #[must_use]
    pub fn new(telemetry: Arc<KvStoreTelemetry>) -> Self {
        Self {
            scopes: Mutex::new(HashMap::new()),
            telemetry,
        }
    }

    /// Arms a scope on the calling thread and returns the guard that releases it.
    #[must_use]
    pub fn enter(self: &Arc<Self>) -> ReplayScope {
        let thread = std::thread::current().id();
        *self.scopes.lock().entry(thread).or_insert(0) += 1;
        self.telemetry
            .replay_scopes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ReplayScope {
            state: Arc::clone(self),
            thread,
        }
    }

    /// True when the calling thread holds at least one scope.
    #[must_use]
    pub fn in_replay(&self) -> bool {
        self.scopes
            .lock()
            .contains_key(&std::thread::current().id())
    }

    /// Number of scopes the calling thread holds, zero when it is not replaying.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.scopes
            .lock()
            .get(&std::thread::current().id())
            .copied()
            .unwrap_or(0)
    }

    /// Releases one scope held by one thread.
    fn leave(&self, thread: ThreadId) {
        let mut scopes = self.scopes.lock();
        if let Some(depth) = scopes.get_mut(&thread) {
            *depth = depth.saturating_sub(1);
            if *depth == 0 {
                scopes.remove(&thread);
            }
        }
    }
}

/// RAII handle for one bounded replay scope; dropping it leaves the scope.
#[derive(Debug)]
pub struct ReplayScope {
    state: Arc<ReplayState>,
    thread: ThreadId,
}

impl ReplayScope {
    /// The thread this scope belongs to.
    #[must_use]
    pub fn thread(&self) -> ThreadId {
        self.thread
    }
}

impl Drop for ReplayScope {
    fn drop(&mut self) {
        self.state.leave(self.thread);
    }
}

/// Shared handle a backend uses to refuse writes made from inside a bounded replay.
#[derive(Clone, Debug)]
pub struct ReplayGuard {
    state: Arc<ReplayState>,
    telemetry: Arc<KvStoreTelemetry>,
}

impl ReplayGuard {
    /// Builds a guard over a replay state and the telemetry refusals are counted in.
    #[must_use]
    pub fn new(state: Arc<ReplayState>, telemetry: Arc<KvStoreTelemetry>) -> Self {
        Self { state, telemetry }
    }

    /// Refuses one write made from a replaying thread, counting the refusal.
    ///
    /// # Errors
    /// Returns KvError::ReplayWroteGlobalKv when the calling thread is inside a replay scope.
    pub fn refuse(&self, page: PageId) -> Result<(), KvError> {
        if !self.state.in_replay() {
            return Ok(());
        }
        self.telemetry
            .replay_global_kv_writes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(KvError::ReplayWroteGlobalKv(page))
    }
}

fn check_page_len(context: &'static str, expected: usize, actual: usize) -> Result<(), KvError> {
    if expected == actual {
        Ok(())
    } else {
        Err(KvError::InvalidLength {
            context,
            expected,
            actual,
        })
    }
}

/// A page backend backed by a hash map; the test and single-node fast path.
#[derive(Debug)]
pub struct InMemoryTier {
    tier: Tier,
    page_bytes: usize,
    capacity_pages: u64,
    alignment: u64,
    pages: Mutex<HashMap<PageId, Box<[u8]>>>,
    /// Replay guard installed by [TieredPageStore::new], if this tier belongs to a store.
    guard: OnceLock<ReplayGuard>,
}

impl InMemoryTier {
    /// Creates an in-memory tier of a fixed capacity.
    pub fn new(tier: Tier, page_bytes: usize, capacity_pages: u64) -> Self {
        Self {
            tier,
            page_bytes,
            capacity_pages,
            alignment: 1,
            pages: Mutex::new(HashMap::new()),
            guard: OnceLock::new(),
        }
    }

    /// Overrides the reported alignment, used to test aligned callers.
    #[must_use]
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment.max(1);
        self
    }
}

impl TierBackend for InMemoryTier {
    fn tier(&self) -> Tier {
        self.tier
    }

    fn name(&self) -> &'static str {
        "in-memory"
    }

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn capacity_pages(&self) -> u64 {
        self.capacity_pages
    }

    fn resident_pages(&self) -> u64 {
        self.pages.lock().len() as u64
    }

    fn alignment(&self) -> u64 {
        self.alignment
    }

    fn contains(&self, page: PageId) -> bool {
        self.pages.lock().contains_key(&page)
    }

    fn write_page(&self, page: PageId, data: &[u8]) -> Result<(), KvError> {
        check_page_len("in-memory page", self.page_bytes, data.len())?;
        if let Some(guard) = self.guard.get() {
            guard.refuse(page)?;
        }
        self.pages
            .lock()
            .insert(page, data.to_vec().into_boxed_slice());
        Ok(())
    }

    fn read_page(&self, page: PageId, out: &mut [u8]) -> Result<(), KvError> {
        check_page_len("in-memory page", self.page_bytes, out.len())?;
        let pages = self.pages.lock();
        let stored = pages.get(&page).ok_or(KvError::PageNotResident(page))?;
        out.copy_from_slice(stored);
        Ok(())
    }

    fn remove_page(&self, page: PageId) -> Result<(), KvError> {
        self.pages.lock().remove(&page);
        Ok(())
    }

    fn set_replay_guard(&self, guard: ReplayGuard) {
        let _ = self.guard.set(guard);
    }
}

/// Magic at the head of the persistent roster file.
const ROSTER_MAGIC: u64 = 0x5350_4b56_524f_5354;
/// Bytes of roster header: magic plus count.
const ROSTER_HEADER_BYTES: usize = 16;
/// Bytes of one roster entry: the page id and the slot holding it.
const ROSTER_ENTRY_BYTES: usize = 16;

/// A page backend that stores aligned slots in one file.
#[derive(Debug)]
pub struct FileBackedTier {
    tier: Tier,
    path: PathBuf,
    file: File,
    page_bytes: usize,
    slot_bytes: u64,
    alignment: u64,
    direct: bool,
    capacity_pages: u64,
    /// Which slot holds each resident page, plus the free list that keeps the file bounded by
    /// its capacity instead of by the highest page id ever written.
    slots: Mutex<Slots>,
    /// Replay guard installed by [TieredPageStore::new], if this tier belongs to a store.
    guard: OnceLock<ReplayGuard>,
}

/// Slot bookkeeping for one file-backed tier.
///
/// The file is sized for exactly capacity_pages slots at open time, so a page id that no
/// longer fits the pool must not decide the file's length: page ids are allocated globally and
/// grow monotonically, while slots are reused as soon as a page leaves the tier.
#[derive(Debug, Default)]
struct Slots {
    /// Slot of every resident page.
    by_page: HashMap<PageId, u64>,
    /// Slots released by removal, reused before the tier grows into unused space.
    free: Vec<u64>,
    /// Next never-used slot.
    next: u64,
}

impl Slots {
    /// Slot of a resident page.
    fn slot_of(&self, page: PageId) -> Option<u64> {
        self.by_page.get(&page).copied()
    }

    /// Claims a slot for a page that is not resident yet.
    ///
    /// # Errors
    /// Returns KvError::CapacityExceeded when every slot is taken.
    fn claim(&mut self, page: PageId, capacity: u64) -> Result<u64, KvError> {
        let slot = if let Some(slot) = self.free.pop() {
            slot
        } else if self.next < capacity {
            let slot = self.next;
            self.next += 1;
            slot
        } else {
            return Err(KvError::CapacityExceeded {
                requested: self.by_page.len() as u64 + 1,
                available: capacity,
            });
        };
        self.by_page.insert(page, slot);
        Ok(slot)
    }

    /// Returns a page's slot to the free list.
    fn release(&mut self, page: PageId) {
        if let Some(slot) = self.by_page.remove(&page) {
            self.free.push(slot);
        }
    }

    /// Rebuilds the slot state from a persisted roster.
    ///
    /// # Errors
    /// Returns KvError::Corrupt when a slot is outside the capacity, or when the roster names
    /// the same page or the same slot twice.
    fn from_pairs(pairs: Vec<(PageId, u64)>, capacity: u64, path: &Path) -> Result<Self, KvError> {
        let roster = roster_path(path);
        let mut slots = Self::default();
        let mut used = HashSet::new();
        for (page, slot) in pairs {
            if slot >= capacity {
                return Err(KvError::Corrupt(format!(
                    "roster {} maps page {} to slot {slot}, beyond the {capacity}-page capacity",
                    roster.display(),
                    page.0
                )));
            }
            if !used.insert(slot) {
                return Err(KvError::Corrupt(format!(
                    "roster {} lists slot {slot} twice",
                    roster.display()
                )));
            }
            if slots.by_page.insert(page, slot).is_some() {
                return Err(KvError::Corrupt(format!(
                    "roster {} lists page {} twice",
                    roster.display(),
                    page.0
                )));
            }
            slots.next = slots.next.max(slot + 1);
        }
        // Holes below next were freed before the roster was written; they are reusable now.
        slots.free = (0..slots.next)
            .filter(|slot| !used.contains(slot))
            .collect();
        Ok(slots)
    }
}

impl FileBackedTier {
    /// Opens or creates a file-backed tier, asking the device whether O_DIRECT works.
    ///
    /// The alignment is probed on the target directory, because whether O_DIRECT is accepted
    /// depends on the filesystem mounted there and not on its type: measured on this host,
    /// tmpfs answers the probe with 512 and opens with O_DIRECT successfully, ext4 with 4096,
    /// and ZFS with 512. When the probe reports that O_DIRECT is unsupported, or when the
    /// O_DIRECT open fails after a successful probe, the tier degrades to buffered I/O and
    /// reports that through [FileBackedTier::is_direct].
    /// [FileBackedTier::open_buffered] forces the buffered path without probing, which is how
    /// the tests exercise it; the second fallback (a successful probe followed by a failed
    /// open) is a safety net that no test can force on a host that accepts O_DIRECT.
    ///
    /// The recover flag selects what happens to the page roster on disk. True adopts it, so
    /// the tier holds the pages a previous run wrote there and maps each one to the slot it
    /// actually occupies. False is a clean start: the tier holds nothing and the roster is
    /// deleted, so a later recovery cannot resurrect a page id whose slot this run has reused
    /// for different bytes. Recovery is only meaningful for a caller that also persists the
    /// index those pages belong to, which is why [TieredPageStore] requires false: its
    /// location map is not persisted, so a recovered page would be unreachable.
    ///
    /// # Errors
    /// Returns an I/O error when the file cannot be created or sized, and
    /// KvError::Corrupt when a recovered roster is malformed or names a slot the capacity
    /// cannot hold.
    pub fn open(
        tier: Tier,
        path: &Path,
        page_bytes: usize,
        capacity_pages: u64,
        recover: bool,
    ) -> Result<Self, KvError> {
        let directory = path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let probed = probe_direct_alignment(&directory).map_err(|source| KvError::Io {
            path: directory.clone(),
            source,
        })?;
        Self::open_with_alignment(tier, path, page_bytes, capacity_pages, recover, probed)
    }

    /// Opens a file-backed tier in buffered mode, without asking the device.
    ///
    /// Slots are then exactly one page with no alignment padding. A caller that already knows
    /// its filesystem refuses O_DIRECT should come here rather than paying for a probe that
    /// will report None. The recover flag means what it means in [FileBackedTier::open].
    ///
    /// # Errors
    /// The same errors as [FileBackedTier::open].
    pub fn open_buffered(
        tier: Tier,
        path: &Path,
        page_bytes: usize,
        capacity_pages: u64,
        recover: bool,
    ) -> Result<Self, KvError> {
        Self::open_with_alignment(tier, path, page_bytes, capacity_pages, recover, None)
    }

    /// Shared constructor: the probed alignment, or None to force buffered I/O.
    fn open_with_alignment(
        tier: Tier,
        path: &Path,
        page_bytes: usize,
        capacity_pages: u64,
        recover: bool,
        probed: Option<u64>,
    ) -> Result<Self, KvError> {
        let (file, alignment, direct) = match probed {
            Some(alignment) => {
                let direct = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .custom_flags(libc::O_DIRECT)
                    .open(path);
                match direct {
                    Ok(file) => (file, alignment, true),
                    Err(_) => (open_buffered(path)?, 1, false),
                }
            }
            None => (open_buffered(path)?, 1, false),
        };
        let slot_bytes = if direct {
            align_up(page_bytes as u64, alignment)
        } else {
            page_bytes as u64
        };
        let required =
            slot_bytes
                .checked_mul(capacity_pages)
                .ok_or_else(|| KvError::CapacityExceeded {
                    requested: capacity_pages,
                    available: u64::MAX / slot_bytes.max(1),
                })?;
        let metadata = file.metadata().map_err(|source| KvError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.len() < required {
            file.set_len(required).map_err(|source| KvError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        }
        let slots = if recover {
            Slots::from_pairs(read_roster_pairs(path)?, capacity_pages, path)?
        } else {
            // A clean start discards the roster on disk as well as in memory: it describes
            // pages this process never indexed, and leaving it behind would let a later
            // recovery resurrect a page id whose slot this tier has since reused for
            // different bytes.
            discard_roster(path)?;
            Slots::default()
        };
        Ok(Self {
            tier,
            path: path.to_path_buf(),
            file,
            page_bytes,
            slot_bytes,
            alignment,
            direct,
            capacity_pages,
            slots: Mutex::new(slots),
            guard: OnceLock::new(),
        })
    }

    /// True when the file is opened with O_DIRECT.
    #[must_use]
    pub const fn is_direct(&self) -> bool {
        self.direct
    }

    /// Aligned on-disk stride of one page.
    #[must_use]
    pub const fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Path of the persistent roster sidecar.
    #[must_use]
    pub fn roster_path(&self) -> PathBuf {
        roster_path(&self.path)
    }

    /// Writes the roster sidecar and syncs the data file.
    ///
    /// # Errors
    /// Returns an I/O error when the sync or the roster write fails.
    pub fn persist(&self) -> Result<(), KvError> {
        self.file.sync_all().map_err(|source| KvError::Io {
            path: self.path.clone(),
            source,
        })?;
        let slots = self.slots.lock();
        let mut bytes =
            Vec::with_capacity(ROSTER_HEADER_BYTES + slots.by_page.len() * ROSTER_ENTRY_BYTES);
        bytes.extend_from_slice(&ROSTER_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(slots.by_page.len() as u64).to_le_bytes());
        for (page, slot) in &slots.by_page {
            bytes.extend_from_slice(&page.0.to_le_bytes());
            bytes.extend_from_slice(&slot.to_le_bytes());
        }
        std::fs::write(self.roster_path(), bytes).map_err(|source| KvError::Io {
            path: self.roster_path(),
            source,
        })
    }
}

impl TierBackend for FileBackedTier {
    fn tier(&self) -> Tier {
        self.tier
    }

    fn name(&self) -> &'static str {
        if self.direct {
            "file-direct"
        } else {
            "file-buffered"
        }
    }

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn capacity_pages(&self) -> u64 {
        self.capacity_pages
    }

    fn resident_pages(&self) -> u64 {
        self.slots.lock().by_page.len() as u64
    }

    fn alignment(&self) -> u64 {
        self.alignment
    }

    fn contains(&self, page: PageId) -> bool {
        self.slots.lock().by_page.contains_key(&page)
    }

    fn write_page(&self, page: PageId, data: &[u8]) -> Result<(), KvError> {
        check_page_len("file page", self.page_bytes, data.len())?;
        if let Some(guard) = self.guard.get() {
            guard.refuse(page)?;
        }
        // Held across the transfer: a slot must not be released and handed to another page
        // while this write is still in flight.
        let mut slots = self.slots.lock();
        let fresh = slots.slot_of(page).is_none();
        let slot = match slots.slot_of(page) {
            Some(slot) => slot,
            None => slots.claim(page, self.capacity_pages)?,
        };
        let offset = slot * self.slot_bytes;
        let written = if self.direct {
            let (mut buffer, start) = aligned_buffer(self.slot_bytes, self.alignment);
            let end = start + self.slot_bytes as usize;
            buffer[start..start + data.len()].copy_from_slice(data);
            self.file.write_all_at(&buffer[start..end], offset)
        } else {
            self.file.write_all_at(data, offset)
        };
        if let Err(source) = written {
            // The slot was claimed for a page that never landed, so hand it back rather than
            // leaking capacity on a failed write.
            if fresh {
                slots.release(page);
            }
            return Err(KvError::Io {
                path: self.path.clone(),
                source,
            });
        }
        Ok(())
    }

    fn read_page(&self, page: PageId, out: &mut [u8]) -> Result<(), KvError> {
        check_page_len("file page", self.page_bytes, out.len())?;
        let slot = self
            .slots
            .lock()
            .slot_of(page)
            .ok_or(KvError::PageNotResident(page))?;
        let offset = slot * self.slot_bytes;
        if self.direct {
            let (mut buffer, start) = aligned_buffer(self.slot_bytes, self.alignment);
            let end = start + self.slot_bytes as usize;
            self.file
                .read_exact_at(&mut buffer[start..end], offset)
                .map_err(|source| KvError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            out.copy_from_slice(&buffer[start..start + out.len()]);
        } else {
            self.file
                .read_exact_at(out, offset)
                .map_err(|source| KvError::Io {
                    path: self.path.clone(),
                    source,
                })?;
        }
        Ok(())
    }

    fn remove_page(&self, page: PageId) -> Result<(), KvError> {
        self.slots.lock().release(page);
        Ok(())
    }

    fn flush(&self) -> Result<(), KvError> {
        self.persist()
    }

    fn set_replay_guard(&self, guard: ReplayGuard) {
        let _ = self.guard.set(guard);
    }
}

fn open_buffered(path: &Path) -> Result<File, KvError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|source| KvError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn roster_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".roster");
    PathBuf::from(name)
}

/// Removes the roster sidecar, tolerating its absence.
fn discard_roster(path: &Path) -> Result<(), KvError> {
    let roster = roster_path(path);
    match std::fs::remove_file(&roster) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(KvError::Io {
            path: roster,
            source,
        }),
    }
}

fn read_roster_pairs(path: &Path) -> Result<Vec<(PageId, u64)>, KvError> {
    let roster = roster_path(path);
    let bytes = match std::fs::read(&roster) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(KvError::Io {
                path: roster,
                source,
            });
        }
    };
    if bytes.len() < ROSTER_HEADER_BYTES {
        return Err(KvError::Corrupt(format!(
            "roster {} is {} bytes, shorter than its header",
            roster.display(),
            bytes.len()
        )));
    }
    let magic = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .map_err(|_| KvError::Corrupt("roster magic is not eight bytes".to_owned()))?,
    );
    if magic != ROSTER_MAGIC {
        return Err(KvError::Corrupt(format!(
            "roster {} has magic {magic:#x}",
            roster.display()
        )));
    }
    let count = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| KvError::Corrupt("roster count is not eight bytes".to_owned()))?,
    );
    // Checked arithmetic: the count comes off the disk, so a hostile value must come back as
    // Corrupt rather than panicking on overflow. The length check below is what makes the
    // allocation safe, so the arithmetic that drives it has to survive any count.
    let Some(expected) = count
        .checked_mul(ROSTER_ENTRY_BYTES as u64)
        .and_then(|entries| entries.checked_add(ROSTER_HEADER_BYTES as u64))
    else {
        return Err(KvError::Corrupt(format!(
            "roster {} claims {count} entries, more than a roster can address",
            roster.display()
        )));
    };
    if bytes.len() as u64 != expected {
        return Err(KvError::Corrupt(format!(
            "roster {} is {} bytes, expected {expected}",
            roster.display(),
            bytes.len()
        )));
    }
    let mut pairs = Vec::with_capacity(count as usize);
    // The length was checked above, so the tail is empty.
    let (entries, _) = bytes[ROSTER_HEADER_BYTES..].as_chunks::<ROSTER_ENTRY_BYTES>();
    for entry in entries {
        let mut page = [0u8; 8];
        page.copy_from_slice(&entry[0..8]);
        let mut slot = [0u8; 8];
        slot.copy_from_slice(&entry[8..16]);
        pairs.push((PageId(u64::from_le_bytes(page)), u64::from_le_bytes(slot)));
    }
    Ok(pairs)
}

/// Allocates a zeroed buffer whose base pointer satisfies the alignment.
fn aligned_buffer(length: u64, alignment: u64) -> (Vec<u8>, usize) {
    let align = alignment.max(1) as usize;
    let buffer = vec![0u8; length as usize + align];
    let base = buffer.as_ptr() as usize;
    let start = (align - (base % align)) % align;
    (buffer, start)
}

/// Finds the smallest power-of-two O_DIRECT alignment a directory's filesystem accepts.
///
/// The candidate list starts at 512 bytes, the smallest logical sector the Linux block layer
/// defines, and doubles up to the directory's preferred I/O size. Returning None means
/// O_DIRECT itself was refused at open time, not that 512 failed. The answer is a property of
/// the mounted filesystem, not of the filesystem type: measured on this host, tmpfs returns
/// Some(512), ext4 returns Some(4096) and ZFS returns Some(512).
///
/// # Errors
/// Propagates an I/O error that is not an O_DIRECT capability failure.
pub fn probe_direct_alignment(directory: &Path) -> std::io::Result<Option<u64>> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let probe = directory.join(format!(
        ".sparkflash-kv-probe-{}-{nanos}",
        std::process::id()
    ));
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_DIRECT)
        .open(&probe)
    {
        Ok(file) => file,
        Err(error) => {
            let _ = std::fs::remove_file(&probe);
            let unsupported = error.raw_os_error().is_some_and(|code| {
                code == libc::EINVAL || code == libc::EOPNOTSUPP || code == libc::EPERM
            });
            return if unsupported { Ok(None) } else { Err(error) };
        }
    };
    let preferred = std::fs::metadata(directory)
        .map_or(4096, |meta| meta.blksize())
        .max(512);
    let mut candidates = Vec::new();
    let mut candidate = 512u64;
    while candidate <= 1 << 20 {
        candidates.push(candidate);
        if candidate >= preferred {
            break;
        }
        candidate <<= 1;
    }
    let mut found = None;
    for candidate in candidates {
        let (mut buffer, start) = aligned_buffer(candidate, candidate);
        let end = start + candidate as usize;
        let pattern = (candidate as u8).wrapping_mul(37).wrapping_add(11);
        buffer[start..end].fill(pattern);
        if file.write_all_at(&buffer[start..end], 0).is_err() {
            continue;
        }
        buffer[start..end].fill(0);
        if file.read_exact_at(&mut buffer[start..end], 0).is_err() {
            continue;
        }
        if buffer[start..end].iter().all(|byte| *byte == pattern) {
            found = Some(candidate);
            break;
        }
    }
    let _ = std::fs::remove_file(&probe);
    Ok(found)
}

/// Where a page currently lives, as the policy sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLocation {
    /// Tier holding the page.
    pub tier: Tier,
    /// Monotone recency stamp used by the eviction policy.
    pub recency: u64,
}

/// The three-tier page store.
pub struct TieredPageStore {
    uma: Box<dyn TierBackend>,
    dram: Box<dyn TierBackend>,
    nvme: Box<dyn TierBackend>,
    location: HashMap<PageId, PageLocation>,
    clock: u64,
    telemetry: Arc<KvStoreTelemetry>,
    /// Replay scopes, shared with the cache and installed on every backend this store owns, so
    /// a replay is refused on every write path and only on the thread that opened it.
    replay: Arc<ReplayState>,
}

impl std::fmt::Debug for TieredPageStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredPageStore")
            .field("uma", &self.uma.name())
            .field("dram", &self.dram.name())
            .field("nvme", &self.nvme.name())
            .field("pages", &self.location.len())
            .finish()
    }
}

impl TieredPageStore {
    /// Assembles a store from three backends.
    ///
    /// Every backend must start empty. The store's location map is not persisted, so a page a
    /// backend already holds at construction would be untracked and unreachable, and the store
    /// would fail its own invariant before doing any work. Open file-backed tiers for a store
    /// with recover = false, which is a clean start.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the backends disagree about the page size, and
    /// KvError::Corrupt when a backend already holds pages.
    pub fn new(
        uma: Box<dyn TierBackend>,
        dram: Box<dyn TierBackend>,
        nvme: Box<dyn TierBackend>,
        telemetry: Arc<KvStoreTelemetry>,
    ) -> Result<Self, KvError> {
        let page_bytes = uma.page_bytes();
        for backend in [&dram, &nvme] {
            if backend.page_bytes() != page_bytes {
                return Err(KvError::InvalidLength {
                    context: "tier page size",
                    expected: page_bytes,
                    actual: backend.page_bytes(),
                });
            }
        }
        for backend in [&uma, &dram, &nvme] {
            let resident = backend.resident_pages();
            if resident > 0 {
                return Err(KvError::Corrupt(format!(
                    "the {} backend already holds {resident} pages at construction, and a store                      starts empty because its location map is not persisted (open a file-backed                      tier with recover = false for a clean start)",
                    backend.name()
                )));
            }
        }
        let replay = Arc::new(ReplayState::new(Arc::clone(&telemetry)));
        let guard = ReplayGuard::new(Arc::clone(&replay), Arc::clone(&telemetry));
        for backend in [&uma, &dram, &nvme] {
            backend.set_replay_guard(guard.clone());
        }
        Ok(Self {
            uma,
            dram,
            nvme,
            location: HashMap::new(),
            clock: 0,
            telemetry,
            replay,
        })
    }

    /// Backend of a tier.
    ///
    /// This is the store's own backend, guard included: a write attempted from a thread inside
    /// a bounded replay is refused here as well, so the accessor is not a way around the
    /// replay check. It is still a raw handle - a write through it does not update the store's
    /// location map - so the published path stays [TieredPageStore::insert], and the accessor
    /// is for diagnostics and for a caller that keeps its own bookkeeping.
    #[must_use]
    pub fn backend(&self, tier: Tier) -> &dyn TierBackend {
        match tier {
            Tier::Uma => self.uma.as_ref(),
            Tier::Dram => self.dram.as_ref(),
            Tier::Nvme => self.nvme.as_ref(),
        }
    }

    /// Logical page size.
    #[must_use]
    pub fn page_bytes(&self) -> usize {
        self.uma.page_bytes()
    }

    /// Capacity of a tier in pages.
    #[must_use]
    pub fn capacity(&self, tier: Tier) -> u64 {
        self.backend(tier).capacity_pages()
    }

    /// Resident pages in a tier.
    #[must_use]
    pub fn resident(&self, tier: Tier) -> u64 {
        self.backend(tier).resident_pages()
    }

    /// Pages the store tracks across every tier.
    #[must_use]
    pub fn tracked_pages(&self) -> usize {
        self.location.len()
    }

    /// Where a page lives, if the store tracks it.
    #[must_use]
    pub fn location(&self, page: PageId) -> Option<PageLocation> {
        self.location.get(&page).copied()
    }

    /// Every page the store currently tracks, in unspecified order.
    #[must_use]
    pub fn tracked_page_ids(&self) -> Vec<PageId> {
        self.location.keys().copied().collect()
    }

    /// Checks that every tracked page lives in exactly one tier, and that no tier holds an
    /// untracked page.
    ///
    /// This is the store's own consistency contract. A promotion that leaves a copy behind in
    /// the source tier, or a move that writes a page without recording it, is a silent capacity
    /// leak until someone asks; this makes it loud. Callers that mutate the store under
    /// contention should call it once the mutating threads have joined.
    ///
    /// # Errors
    /// Returns KvError::Corrupt describing the first inconsistency found.
    pub fn check_invariants(&self) -> Result<(), KvError> {
        let mut counted = 0u64;
        for tier in [Tier::Uma, Tier::Dram, Tier::Nvme] {
            let backend = self.backend(tier);
            let tracked = self
                .location
                .values()
                .filter(|entry| entry.tier == tier)
                .count() as u64;
            let resident = backend.resident_pages();
            if resident > backend.capacity_pages() {
                return Err(KvError::Corrupt(format!(
                    "tier {tier:?} holds {resident} pages but its capacity is {}",
                    backend.capacity_pages()
                )));
            }
            if tracked != resident {
                return Err(KvError::Corrupt(format!(
                    "tier {tier:?} holds {resident} pages but {tracked} are tracked there"
                )));
            }
            counted += resident;
        }
        if counted != self.location.len() as u64 {
            return Err(KvError::Corrupt(format!(
                "{counted} pages are resident across the tiers but {} are tracked",
                self.location.len()
            )));
        }
        for (page, entry) in &self.location {
            if !self.backend(entry.tier).contains(*page) {
                return Err(KvError::Corrupt(format!(
                    "page {} is tracked in {:?} but not resident there",
                    page.0, entry.tier
                )));
            }
        }
        Ok(())
    }

    /// The telemetry sink this store writes to.
    #[must_use]
    pub fn telemetry(&self) -> &Arc<KvStoreTelemetry> {
        &self.telemetry
    }

    /// Enters a bounded replay scope on the calling thread.
    ///
    /// While the scope is open, every guarded write path refuses to publish global KV and
    /// counts the attempt, because a bounded replay may regenerate SWA KV only. Scopes are per
    /// thread, so a concurrent publish on another thread is unaffected.
    ///
    /// The scope is released by dropping the returned guard, and only by dropping it: a replay
    /// that returns early or panics cannot leave the thread marked as replaying. Scopes nest,
    /// and the thread stays in scope until its last guard is dropped.
    #[must_use]
    pub fn begin_replay(&self) -> ReplayScope {
        self.replay.enter()
    }

    /// The replay-scope state shared with the cache and with every backend of this store.
    #[must_use]
    pub fn replay_state(&self) -> Arc<ReplayState> {
        Arc::clone(&self.replay)
    }

    /// Whether the calling thread is inside a bounded replay scope.
    #[must_use]
    pub fn in_replay(&self) -> bool {
        self.replay.in_replay()
    }

    /// Nesting depth of the calling thread's replay scopes, zero when it is not replaying.
    #[must_use]
    pub fn replay_depth(&self) -> u32 {
        self.replay.depth()
    }

    /// Guards the global-KV publish path against a replay reaching it.
    ///
    /// Counting and refusing are deliberately both done here: the counter alone would let a
    /// future replay path corrupt the cached global KV and only report it afterwards.
    fn note_global_publish(&mut self, page: PageId) -> Result<(), KvError> {
        if self.replay.in_replay() {
            self.telemetry
                .replay_global_kv_writes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(KvError::ReplayWroteGlobalKv(page));
        }
        Ok(())
    }

    /// Inserts a brand new page into UMA, making room by demoting unpinned pages.
    ///
    /// # Errors
    /// Returns KvError::DuplicatePage when the page is already tracked,
    /// KvError::ReplayWroteGlobalKv when the calling thread is inside a bounded replay scope
    /// (a replay must not publish global KV), and KvError::CapacityExceeded when no unpinned
    /// victim exists.
    pub fn insert(
        &mut self,
        page: PageId,
        data: &[u8],
        protected: &HashSet<PageId>,
    ) -> Result<(), KvError> {
        if self.location.contains_key(&page) {
            return Err(KvError::DuplicatePage(page));
        }
        self.note_global_publish(page)?;
        self.ensure_room(Tier::Uma, protected, page)?;
        self.uma.write_page(page, data)?;
        self.record(page, Tier::Uma);
        self.telemetry
            .page_inserts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.telemetry
            .bytes_written
            .fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Brings a page up to UMA, staging through DRAM when it started on NVMe.
    ///
    /// Returns the tier the payload was read from, which is what separates a Tier-1 hit from
    /// a Tier-2 restore in the request plan.
    ///
    /// # Errors
    /// Returns KvError::ReplayWroteGlobalKv when the calling thread is inside a bounded replay
    /// scope, KvError::PageNotResident when the store does not track the page, and
    /// KvError::CapacityExceeded when no unpinned victim exists.
    pub fn promote(&mut self, page: PageId, protected: &HashSet<PageId>) -> Result<Tier, KvError> {
        // Checked before anything else, because a promotion is a compound mutation: it lifts
        // the payload, releases the source copy and only then writes through a guarded backend.
        // A refusal discovered at that write would arrive after the store had already given up
        // its only copy of the page, losing the id, the bytes and the location entry while every
        // invariant still held. Refusing up front also counts the attempt exactly once instead
        // of once per backend write the refusal would have reached.
        self.note_global_publish(page)?;
        let Some(entry) = self.location.get(&page).copied() else {
            return Err(KvError::PageNotResident(page));
        };
        match entry.tier {
            Tier::Uma => {
                let recency = self.tick();
                if let Some(current) = self.location.get_mut(&page) {
                    current.recency = recency;
                }
                Ok(Tier::Uma)
            }
            Tier::Dram => {
                // The payload is lifted into a bounce buffer before the cascade runs, so the
                // page's own DRAM slot can be handed to the demotion below it. Asking UMA for
                // room while the page still occupied DRAM would deadlock on a full DRAM tier.
                let buffer = self.read_into_buffer(page, Tier::Dram)?;
                self.unplace(page, Tier::Dram)?;
                match self
                    .ensure_room(Tier::Uma, protected, page)
                    .and_then(|()| self.place(page, Tier::Uma, &buffer, Some(Tier::Dram)))
                {
                    Ok(()) => Ok(Tier::Dram),
                    Err(error) => {
                        self.restore(page, Tier::Dram, &buffer)?;
                        Err(error)
                    }
                }
            }
            Tier::Nvme => {
                // Hop through DRAM. A real deployment stages into pinned host memory so the
                // NVMe read and the UMA write never contend for one bounce buffer; here the
                // hop is a residency hop with its own telemetry and counter.
                let buffer = self.read_into_buffer(page, Tier::Nvme)?;
                self.ensure_room(Tier::Dram, protected, page)?;
                // Staging hop, not a promotion: uma_promotions counts pages that arrive in
                // coherent UMA, and a Tier-2 restore arrives once, at the end of the hop.
                self.place(page, Tier::Dram, &buffer, None)?;
                // Release the source copy, then the staging copy: both are redundant now that
                // the payload is in the bounce buffer, and releasing the staging slot is what
                // lets the cascade triggered by the UMA request reclaim it. Leaving the NVMe
                // copy behind would count the page twice, forever.
                self.unplace(page, Tier::Nvme)?;
                self.unplace(page, Tier::Dram)?;
                match self
                    .ensure_room(Tier::Uma, protected, page)
                    .and_then(|()| self.place(page, Tier::Uma, &buffer, Some(Tier::Dram)))
                {
                    Ok(()) => Ok(Tier::Nvme),
                    Err(error) => {
                        // A refusal must put the payload back rather than lose it.
                        self.restore(page, Tier::Nvme, &buffer)?;
                        Err(error)
                    }
                }
            }
        }
    }

    /// Reads a page into a bounce buffer without changing where it lives.
    fn read_into_buffer(&mut self, page: PageId, tier: Tier) -> Result<Vec<u8>, KvError> {
        let mut buffer = vec![0u8; self.page_bytes()];
        self.backend(tier).read_page(page, &mut buffer)?;
        let order = std::sync::atomic::Ordering::Relaxed;
        self.telemetry
            .bytes_read
            .fetch_add(buffer.len() as u64, order);
        let counter = match tier {
            Tier::Uma => &self.telemetry.uma_reads,
            Tier::Dram => &self.telemetry.dram_reads,
            Tier::Nvme => &self.telemetry.nvme_reads,
        };
        counter.fetch_add(1, order);
        Ok(buffer)
    }

    /// Stops tracking a page in a tier, leaving the caller's buffer as the only copy.
    fn unplace(&mut self, page: PageId, tier: Tier) -> Result<(), KvError> {
        self.backend(tier).remove_page(page)?;
        self.location.remove(&page);
        Ok(())
    }

    /// Writes a bounce buffer into a tier and records the page there.
    fn place(
        &mut self,
        page: PageId,
        tier: Tier,
        data: &[u8],
        from: Option<Tier>,
    ) -> Result<(), KvError> {
        self.backend(tier).write_page(page, data)?;
        let order = std::sync::atomic::Ordering::Relaxed;
        self.telemetry
            .bytes_written
            .fetch_add(data.len() as u64, order);
        self.record(page, tier);
        match from {
            Some(previous) if to_is_promotion(previous, tier) => {
                self.telemetry.uma_promotions.fetch_add(1, order);
            }
            Some(_) => {
                self.telemetry.demotions.fetch_add(1, order);
            }
            None => {}
        }
        Ok(())
    }

    /// Puts a page back where it came from after a refused promotion.
    fn restore(&mut self, page: PageId, tier: Tier, data: &[u8]) -> Result<(), KvError> {
        self.backend(tier).write_page(page, data)?;
        self.record(page, tier);
        Ok(())
    }

    /// Reads a page from whichever tier currently holds it, without promoting it.
    ///
    /// # Errors
    /// Returns KvError::PageNotResident when untracked, or a backend error.
    pub fn read(&self, page: PageId, out: &mut [u8]) -> Result<Tier, KvError> {
        let Some(entry) = self.location.get(&page) else {
            return Err(KvError::PageNotResident(page));
        };
        self.backend(entry.tier).read_page(page, out)?;
        let bytes = out.len() as u64;
        self.telemetry
            .bytes_read
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        let counter = match entry.tier {
            Tier::Uma => &self.telemetry.uma_reads,
            Tier::Dram => &self.telemetry.dram_reads,
            Tier::Nvme => &self.telemetry.nvme_reads,
        };
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(entry.tier)
    }

    /// Drops a page from every tier.
    ///
    /// This is the one mutator that is not refused inside a bounded replay, and deliberately
    /// so: it removes a copy rather than publishing one, and the drop is the caller's explicit
    /// request. The mutators that write through a backend, [TieredPageStore::insert] and
    /// [TieredPageStore::promote], both check the replay scope before their first mutation, so
    /// a refusal can never arrive after a page has been given up.
    ///
    /// # Errors
    /// Returns an error when the backend cannot complete the removal.
    pub fn drop_page(&mut self, page: PageId) -> Result<(), KvError> {
        let Some(entry) = self.location.remove(&page) else {
            return Ok(());
        };
        self.backend(entry.tier).remove_page(page)?;
        self.telemetry
            .page_drops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Whether a page is tracked at all.
    #[must_use]
    pub fn contains(&self, page: PageId) -> bool {
        self.location.contains_key(&page)
    }

    /// Flushes every tier's persistent state.
    ///
    /// # Errors
    /// Returns the first backend flush error.
    pub fn flush(&self) -> Result<(), KvError> {
        self.uma.flush()?;
        self.dram.flush()?;
        self.nvme.flush()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn record(&mut self, page: PageId, tier: Tier) {
        let recency = self.tick();
        self.location.insert(page, PageLocation { tier, recency });
    }

    fn ensure_room(
        &mut self,
        tier: Tier,
        protected: &HashSet<PageId>,
        keep: PageId,
    ) -> Result<(), KvError> {
        self.ensure_room_keeping(tier, protected, &[keep])
    }

    /// Makes room in a tier while keeping a set of pages in place at *every* level of the
    /// cascade.
    ///
    /// Demoting a victim into the tier below can cascade, and a page the caller is currently
    /// moving must not be chosen as a victim by a nested call, or the caller's payload leaves
    /// the tier it is being read from. The set is accumulated for that reason and pinned by
    /// a_cascade_keeps_every_page_the_caller_is_moving_not_just_the_first_victim below, which
    /// fails if only the immediate victim is carried down.
    ///
    /// No caller reaches that configuration today: insert passes a page that is not resident
    /// yet, and promote unplaces the page before asking for room, so the nested tier never
    /// holds it. The accumulation is therefore defensive - kept, documented and tested rather
    /// than claimed as covered - because the cascade is the only place the invariant can break
    /// and the next caller that keeps a resident payload in flight would need it.
    fn ensure_room_keeping(
        &mut self,
        tier: Tier,
        protected: &HashSet<PageId>,
        keeps: &[PageId],
    ) -> Result<(), KvError> {
        loop {
            if self.resident(tier) < self.capacity(tier) {
                return Ok(());
            }
            let Some(next) = tier.next_down() else {
                self.telemetry
                    .capacity_refusals
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(KvError::CapacityExceeded {
                    requested: 1,
                    available: 0,
                });
            };
            let Some(victim) = self.lru_victim(tier, protected, keeps) else {
                self.telemetry
                    .capacity_refusals
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(KvError::CapacityExceeded {
                    requested: 1,
                    available: self.resident(tier),
                });
            };
            let mut extended = keeps.to_vec();
            extended.push(victim);
            self.ensure_room_keeping(next, protected, &extended)?;
            self.move_page(victim, tier, next)?;
        }
    }

    fn lru_victim(
        &self,
        tier: Tier,
        protected: &HashSet<PageId>,
        keeps: &[PageId],
    ) -> Option<PageId> {
        self.location
            .iter()
            .filter(|(page, entry)| {
                entry.tier == tier && !keeps.contains(page) && !protected.contains(page)
            })
            .min_by_key(|(page, entry)| (entry.recency, page.0))
            .map(|(page, _)| *page)
    }

    fn move_page(&mut self, page: PageId, from: Tier, to: Tier) -> Result<(), KvError> {
        let mut buffer = vec![0u8; self.page_bytes()];
        self.backend(from).read_page(page, &mut buffer)?;
        self.backend(to).write_page(page, &buffer)?;
        self.backend(from).remove_page(page)?;
        let recency = self.tick();
        if let Some(entry) = self.location.get_mut(&page) {
            entry.tier = to;
            entry.recency = recency;
        }
        let order = std::sync::atomic::Ordering::Relaxed;
        self.telemetry
            .bytes_read
            .fetch_add(buffer.len() as u64, order);
        self.telemetry
            .bytes_written
            .fetch_add(buffer.len() as u64, order);
        if to_is_promotion(from, to) {
            self.telemetry.uma_promotions.fetch_add(1, order);
        } else {
            self.telemetry.demotions.fetch_add(1, order);
        }
        Ok(())
    }
}

/// True when moving a page from one tier to another brings it closer to the compute.
fn to_is_promotion(from: Tier, to: Tier) -> bool {
    to < from
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store with 64-byte pages, so a unit test can state the tier layout exactly.
    fn small_store(uma: u64, dram: u64, nvme: u64) -> TieredPageStore {
        TieredPageStore::new(
            Box::new(InMemoryTier::new(Tier::Uma, 64, uma)),
            Box::new(InMemoryTier::new(Tier::Dram, 64, dram)),
            Box::new(InMemoryTier::new(Tier::Nvme, 64, nvme)),
            Arc::new(KvStoreTelemetry::default()),
        )
        .expect("store")
    }

    /// One page per tier with uma=1 and dram=1: page 0 lands in DRAM, page 1 in UMA.
    fn one_page_per_tier() -> TieredPageStore {
        let mut store = small_store(1, 1, 2);
        let protected = HashSet::new();
        let data = [7u8; 64];
        store.insert(PageId(0), &data, &protected).expect("page 0");
        store.insert(PageId(1), &data, &protected).expect("page 1");
        assert_eq!(
            store.location(PageId(0)).map(|entry| entry.tier),
            Some(Tier::Dram)
        );
        assert_eq!(
            store.location(PageId(1)).map(|entry| entry.tier),
            Some(Tier::Uma)
        );
        store
    }

    #[test]
    fn a_cascade_keeps_every_page_the_caller_is_moving_not_just_the_first_victim() {
        // The caller holds page 0's payload and asks UMA for room. The cascade evicts page 1
        // out of UMA and must then find page 0 protected in DRAM: with only the immediate
        // victim carried into the nested call, page 0 is chosen as the DRAM victim and the
        // request succeeds on top of the payload the caller is moving.
        let mut store = one_page_per_tier();
        let error = store
            .ensure_room_keeping(Tier::Uma, &HashSet::new(), &[PageId(0)])
            .expect_err("page 0 must not be evicted while the caller holds its payload");
        assert!(matches!(error, KvError::CapacityExceeded { .. }), "{error}");
        assert_eq!(
            store.location(PageId(0)).map(|entry| entry.tier),
            Some(Tier::Dram),
            "the kept page left the tier the caller is reading it from"
        );
        assert_eq!(store.resident(Tier::Nvme), 0);
        assert_eq!(store.tracked_pages(), 2);
        store.check_invariants().expect("store invariants");
    }

    #[test]
    fn the_same_cascade_moves_the_page_when_the_caller_is_not_keeping_it() {
        // The same configuration with no keep set: page 1 leaves UMA and page 0 is demoted to
        // the terminal tier. This is what shows the refusal above comes from the keep set and
        // not from a capacity limit.
        let mut store = one_page_per_tier();
        store
            .ensure_room_keeping(Tier::Uma, &HashSet::new(), &[])
            .expect("with no kept pages the cascade may move either victim");
        assert_eq!(
            store.location(PageId(0)).map(|entry| entry.tier),
            Some(Tier::Nvme)
        );
        assert_eq!(
            store.location(PageId(1)).map(|entry| entry.tier),
            Some(Tier::Dram)
        );
        assert_eq!(store.resident(Tier::Nvme), 1);
        store.check_invariants().expect("store invariants");
    }
}
