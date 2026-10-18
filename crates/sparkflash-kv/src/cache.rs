//! The three-tier paged KV cache: radix index, page store and SWA ring wired together.
//!
//! The cache owns exactly two locks:
//!
//! * one over the radix tree, the tiered page store and the page-id allocator, which is the
//!   global-KV path;
//! * one over the SWA tail ring.
//!
//! No code path ever takes the SWA lock and then the global-KV lock, and the bounded replay
//! functions take only the SWA lock. That is the structural half of the "an encoder replay
//! must never overwrite cached global KV" invariant: a replay has no way to reach the page
//! store at all. Three things back it up, so a future refactor cannot quietly drop it:
//!
//! * the store carries a per-thread replay scope, armed by [KvCache::bounded_replay] around the
//!   producer's work and released only by dropping its guard, so a producer that panics cannot
//!   wedge the thread; the global-KV publish path and every backend the store owns refuse to
//!   write inside one and count the attempt in [KvStoreTelemetry::replay_global_kv_writes];
//! * a producer callback that re-enters the cache panics with a message naming
//!   [SwaProducer::fill_layer] instead of deadlocking on the lock order inversion;
//! * a test fingerprints every resident page before and after a replay and rereads them.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use crate::config::{KV_PAGE_TOKENS, SWA_WINDOW_SIZE};
use crate::error::KvError;
use crate::layout::{
    DEFAULT_GLOBAL_KV_POOL_BYTES, DEFAULT_SWA_TAIL_SLOTS, GLOBAL_PAGE_BYTES, MemoryBudget, PageId,
    SWA_DECODER_LAYERS, SWA_ENCODER_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES,
};
use crate::plan::{KvCacheStatsSnapshot, KvCacheTelemetry, PrefixMatchPlan};
use crate::radix::{NodeId, RadixTree};
use crate::swa::{ReplayKind, ReplayOutcome, SwaProducer, SwaTailRing, TailId};
use crate::telemetry::{KvStoreSnapshot, KvStoreTelemetry};
use crate::tier::{FileBackedTier, InMemoryTier, ReplayState, Tier, TierBackend, TieredPageStore};

/// Local NVMe the default NVMe tier models: the 3.7 TB drive memory note #66 measured.
pub const DEFAULT_NVME_CAPACITY_BYTES: u64 = 3_700_000_000_000;

thread_local! {
    /// Depth of nested producer callbacks on this thread.
    ///
    /// A SwaProducer callback runs while the SWA lock is held and may not re-enter the cache:
    /// taking the global-KV lock there inverts the inner-then-SWA order every multi-lock path
    /// uses, and taking the SWA lock again would deadlock because parking_lot's mutexes are not
    /// reentrant. This depth turns either mistake into a panic at the point of the call instead
    /// of a hang with no trace.
    static PRODUCER_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Marks one producer callback window on the current thread.
struct ProducerWindow;

impl ProducerWindow {
    /// Enters a callback window, recording it for the re-entrancy guard.
    fn enter() -> Self {
        PRODUCER_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for ProducerWindow {
    fn drop(&mut self) {
        PRODUCER_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Whether the current thread is inside a SwaProducer callback.
fn in_producer_callback() -> bool {
    PRODUCER_DEPTH.with(std::cell::Cell::get) > 0
}

/// Static configuration of a cache instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Per-node memory ceiling and floor guard.
    pub budget: MemoryBudget,
    /// Pages in the Tier-1 coherent UMA pool.
    pub uma_pages: u64,
    /// Pages in the DRAM staging tier.
    pub dram_pages: u64,
    /// Pages the NVMe tier may hold.
    pub nvme_pages: u64,
    /// SWA tail slots in the ring.
    pub swa_tail_slots: usize,
    /// When set, the NVMe tier is file-backed at this path; otherwise it is in memory.
    pub nvme_path: Option<PathBuf>,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        let budget = MemoryBudget::default();
        let page = GLOBAL_PAGE_BYTES as u64;
        let swa_bytes = DEFAULT_SWA_TAIL_SLOTS as u64 * SWA_TAIL_BYTES as u64;
        let after_swa = budget.usable_bytes().saturating_sub(swa_bytes);
        let uma_bytes = DEFAULT_GLOBAL_KV_POOL_BYTES.min(after_swa);
        let uma_pages = uma_bytes / page;
        let dram_pages = after_swa.saturating_sub(uma_pages * page) / page;
        Self {
            budget,
            uma_pages,
            dram_pages,
            nvme_pages: DEFAULT_NVME_CAPACITY_BYTES / page,
            swa_tail_slots: DEFAULT_SWA_TAIL_SLOTS,
            nvme_path: None,
        }
    }
}

impl KvCacheConfig {
    /// Bytes this configuration reserves from the node memory budget.
    #[must_use]
    pub fn required_bytes(&self) -> u64 {
        let page = GLOBAL_PAGE_BYTES as u64;
        self.uma_pages * page
            + self.dram_pages * page
            + self.swa_tail_slots as u64 * SWA_TAIL_BYTES as u64
    }

    /// Checks the configuration against the budget.
    ///
    /// # Errors
    /// Returns KvError::BudgetExceeded when the pools plus the floor guard do not fit.
    pub fn validate(&self) -> Result<(), KvError> {
        let required = self.required_bytes();
        if required > self.budget.usable_bytes() {
            return Err(KvError::BudgetExceeded {
                required,
                budget: self.budget.usable_bytes(),
            });
        }
        Ok(())
    }
}

/// A live attachment of a prompt to the radix tree.
///
/// Dropping the sequence releases the reference counts it took, so a pinned page can never
/// leak because a caller forgot to unpin it.
pub struct Sequence<'a> {
    cache: &'a KvCache,
    path: Vec<NodeId>,
    /// The plan that was computed when this sequence was admitted.
    pub plan: PrefixMatchPlan,
}

impl<'a> Sequence<'a> {
    /// Pinned nodes backing the matched prefix, root-most first.
    #[must_use]
    pub fn path(&self) -> &[NodeId] {
        &self.path
    }
}

impl Drop for Sequence<'_> {
    fn drop(&mut self) {
        self.cache.release_path(&self.path);
    }
}

/// What cache_prefix stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheOutcome {
    /// Tokens now covered by the tree, matched plus newly written.
    pub cached_tokens: usize,
    /// New pages written.
    pub pages_added: usize,
    /// Blocks that were already cached.
    pub pages_reused: usize,
    /// Pages evicted from the tree to make room.
    pub pages_evicted: usize,
}

/// A consistent-enough view of both telemetry families plus residency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheSnapshot {
    /// Request-planning counters.
    pub cache: KvCacheStatsSnapshot,
    /// Storage-tier counters.
    pub store: KvStoreSnapshot,
    /// Pages resident in Tier-1 UMA.
    pub uma_pages: u64,
    /// Pages resident in the DRAM staging tier.
    pub dram_pages: u64,
    /// Pages resident in the NVMe tier.
    pub nvme_pages: u64,
    /// Pages the store tracks across every tier.
    pub tracked_pages: usize,
    /// SWA tails in the ring.
    pub swa_tails: usize,
    /// SWA tails that have all 40 layers and can be served.
    pub swa_complete_tails: usize,
    /// Bytes of SWA tail payload.
    pub swa_bytes: u64,
}

struct Inner {
    tree: RadixTree,
    store: TieredPageStore,
    next_page: u64,
}

/// The paged three-tier KV cache.
pub struct KvCache {
    inner: Mutex<Inner>,
    swa: Mutex<SwaTailRing>,
    telemetry: KvCacheTelemetry,
    store_telemetry: Arc<KvStoreTelemetry>,
    /// Replay scopes, shared with the store and its backends: arming one here refuses every
    /// global-KV write from this thread, wherever it is attempted.
    replay: Arc<ReplayState>,
    config: KvCacheConfig,
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCache")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl KvCache {
    /// Builds a cache from a configuration, probing the NVMe device when a path is set.
    ///
    /// The NVMe tier is always opened for a clean start, with recovery off. The radix index is
    /// not persisted, so a roster recovered from an earlier run could never be reached by a
    /// lookup, and adopting it would leave the store's location map disagreeing with the tier
    /// that holds the pages. A cache therefore starts empty every time, and a page id it hands
    /// out cannot alias a page left in the file by a previous run: the roster on disk is
    /// discarded when the tier is opened without recovery. Recovery across restarts is a
    /// raw-tier facility, [FileBackedTier::open] with recover = true, for a caller that also
    /// persists and restores the index the pages belong to.
    ///
    /// # Errors
    /// Returns KvError::BudgetExceeded for an over-budget configuration and any I/O error
    /// from opening the NVMe tier.
    pub fn new(config: KvCacheConfig) -> Result<Self, KvError> {
        config.validate()?;
        let page_bytes = GLOBAL_PAGE_BYTES;
        let uma: Box<dyn TierBackend> = Box::new(
            InMemoryTier::new(Tier::Uma, page_bytes, config.uma_pages).with_alignment(256),
        );
        let dram: Box<dyn TierBackend> =
            Box::new(InMemoryTier::new(Tier::Dram, page_bytes, config.dram_pages));
        let nvme: Box<dyn TierBackend> = match &config.nvme_path {
            // recover = false: see the note on KvCache::new. The index that would make a
            // recovered page reachable is not persisted, so a cache starts clean.
            Some(path) => Box::new(FileBackedTier::open(
                Tier::Nvme,
                path,
                page_bytes,
                config.nvme_pages,
                false,
            )?),
            None => Box::new(InMemoryTier::new(Tier::Nvme, page_bytes, config.nvme_pages)),
        };
        Self::with_backends(config, uma, dram, nvme)
    }

    /// Builds a cache over injected backends, which is how the tests get file-backed tiers
    /// on every level.
    ///
    /// # Errors
    /// Returns KvError::BudgetExceeded for an over-budget configuration and
    /// KvError::InvalidLength when the backends disagree about the page size.
    pub fn with_backends(
        config: KvCacheConfig,
        uma: Box<dyn TierBackend>,
        dram: Box<dyn TierBackend>,
        nvme: Box<dyn TierBackend>,
    ) -> Result<Self, KvError> {
        config.validate()?;
        let store_telemetry = Arc::new(KvStoreTelemetry::default());
        let store = TieredPageStore::new(uma, dram, nvme, Arc::clone(&store_telemetry))?;
        let replay = store.replay_state();
        let telemetry = KvCacheTelemetry::default();
        telemetry.set_total_pages(config.uma_pages);
        let swa = SwaTailRing::new(config.swa_tail_slots, Arc::clone(&store_telemetry));
        Ok(Self {
            inner: Mutex::new(Inner {
                tree: RadixTree::new(),
                store,
                next_page: 0,
            }),
            swa: Mutex::new(swa),
            telemetry,
            store_telemetry,
            replay,
            config,
        })
    }

    /// Configuration this cache was built with.
    #[must_use]
    pub const fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Plan counters.
    #[must_use]
    pub const fn telemetry(&self) -> &KvCacheTelemetry {
        &self.telemetry
    }

    /// Store counters.
    #[must_use]
    pub fn store_telemetry(&self) -> &KvStoreTelemetry {
        &self.store_telemetry
    }

    /// Matches a prompt, pins the usable prefix, promotes its pages into UMA and records the
    /// exact SWA Bounded Replay plan.
    ///
    /// The returned plan is exactly what [KvCacheTelemetry::plan_request] computes from the
    /// two tier matches and the SWA tail probe, so the counters it records and the plan it
    /// returns can never disagree.
    ///
    /// # Errors
    /// Returns an error only when the tier store cannot read a page it claims to hold.
    pub fn begin_sequence(&self, tokens: &[u32]) -> Result<Sequence<'_>, KvError> {
        let (uma_hit, nvme_hit, path) = {
            let mut inner = self.lock_inner();
            let matched = inner.tree.match_prefix(tokens);
            let cached = usable_cached_tokens(tokens.len(), matched.matched_tokens);
            let blocks = cached / KV_PAGE_TOKENS;
            let path: Vec<NodeId> = matched.path[..blocks].to_vec();
            // Promotion happens before pinning, and the protected set is the *other* live
            // sequences' pins. Pinning first would let this request protect pages it cannot
            // serve yet and deadlock its own promotion when UMA is full of its own path.
            let protected: HashSet<PageId> = inner.tree.pinned_pages().into_iter().collect();
            let mut uma_hit = 0usize;
            let mut nvme_hit = 0usize;
            let mut served = 0usize;
            for block in 0..blocks {
                match inner.store.promote(matched.pages[block], &protected) {
                    Ok(Tier::Uma) => {
                        uma_hit += KV_PAGE_TOKENS;
                        served += 1;
                    }
                    Ok(_) => {
                        nvme_hit += KV_PAGE_TOKENS;
                        served += 1;
                    }
                    Err(_) => break,
                }
            }
            let path = path[..served].to_vec();
            inner.tree.attach(&path)?;
            inner.tree.touch(&path);
            (uma_hit, nvme_hit, path)
        };
        let cached_tokens = uma_hit + nvme_hit;
        let swa_tail_hit = if cached_tokens >= SWA_WINDOW_SIZE {
            let ring = self.lock_swa();
            ring.probe_prefix(&tokens[..cached_tokens]).is_some()
        } else {
            false
        };
        let plan = self
            .telemetry
            .plan_request(tokens.len(), uma_hit, nvme_hit, swa_tail_hit);
        Ok(Sequence {
            cache: self,
            path,
            plan,
        })
    }

    /// Caches every whole 64-token block of a prompt.
    ///
    /// A trailing partial page is not stored. If the pool refuses capacity the cache evicts
    /// least-recently-used unpinned pages until the write fits; when nothing is evictable the
    /// error propagates and the blocks written so far stay cached, which is harmless because
    /// prefix caching is advisory.
    ///
    /// # Errors
    /// Returns the first error from the filler or from the tier store.
    pub fn cache_prefix<F>(&self, tokens: &[u32], mut fill: F) -> Result<CacheOutcome, KvError>
    where
        F: FnMut(usize, &mut [u8]) -> Result<(), KvError>,
    {
        let mut inner = self.lock_inner();
        let matched = inner.tree.match_prefix(tokens);
        let mut node = matched
            .path
            .last()
            .copied()
            .unwrap_or_else(|| inner.tree.root());
        let mut offset = matched.matched_tokens;
        let mut block_index = offset / KV_PAGE_TOKENS;
        let mut outcome = CacheOutcome {
            cached_tokens: offset,
            ..CacheOutcome::default()
        };
        // Two different protections, for two different hazards:
        //
        // * `pinned` keeps the pages another live sequence is reading in place. Demoting them
        //   would be safe but slow, so the store treats them as untouchable.
        // * `building` is the prefix this call is still assembling. Demoting those pages is
        //   fine (the payload survives, only the tier changes), but letting the *tree* evict
        //   one would free the node this walk is descending through and silently truncate the
        //   prefix.
        let pinned: HashSet<PageId> = inner.tree.pinned_pages().into_iter().collect();
        let mut building: HashSet<PageId> = matched.pages.iter().copied().collect();
        while offset + KV_PAGE_TOKENS <= tokens.len() {
            let block = tokens[offset..offset + KV_PAGE_TOKENS].to_vec();
            if let Some(existing) = inner.tree.child_for(node, &block) {
                node = existing;
                outcome.pages_reused += 1;
            } else {
                let mut buffer = vec![0u8; GLOBAL_PAGE_BYTES];
                fill(block_index, &mut buffer)?;
                let page = PageId(inner.next_page);
                loop {
                    match inner.store.insert(page, &buffer, &pinned) {
                        Ok(()) => break,
                        Err(KvError::CapacityExceeded { .. }) => {
                            let freed = inner.tree.evict_lru_avoiding(1, &building);
                            if freed.is_empty() {
                                return Err(KvError::CapacityExceeded {
                                    requested: 1,
                                    available: 0,
                                });
                            }
                            for victim in freed {
                                inner.store.drop_page(victim)?;
                                self.telemetry.note_page_freed();
                                outcome.pages_evicted += 1;
                            }
                        }
                        Err(other) => return Err(other),
                    }
                }
                inner.next_page += 1;
                self.telemetry.note_page_allocated();
                node = inner.tree.insert_page(node, &block, page);
                building.insert(page);
                outcome.pages_added += 1;
            }
            offset += KV_PAGE_TOKENS;
            block_index += 1;
            outcome.cached_tokens = offset;
        }
        Ok(outcome)
    }

    /// Takes the global-KV lock, refusing to do so from inside a producer callback.
    ///
    /// # Panics
    /// Panics when the calling thread is inside [SwaProducer::fill_layer]. A producer that
    /// reaches the global-KV path takes this lock while the replay holds the SWA lock, which is
    /// the inverse of the inner-then-SWA order used everywhere else; without the guard that is
    /// a deadlock with no diagnostic.
    fn lock_inner(&self) -> MutexGuard<'_, Inner> {
        assert!(
            !in_producer_callback(),
            "KvCache cannot be re-entered from SwaProducer::fill_layer: the replay holds the \
             SWA lock, so taking the global-KV lock here inverts the lock order and deadlocks"
        );
        self.inner.lock()
    }

    /// Takes the SWA lock, refusing to do so from inside a producer callback.
    ///
    /// # Panics
    /// Panics when the calling thread is inside [SwaProducer::fill_layer], where taking the SWA
    /// lock again would deadlock on a non-reentrant mutex.
    fn lock_swa(&self) -> MutexGuard<'_, SwaTailRing> {
        assert!(
            !in_producer_callback(),
            "KvCache cannot be re-entered from SwaProducer::fill_layer: the replay already \
             holds the SWA lock, and parking_lot mutexes are not reentrant"
        );
        self.swa.lock()
    }

    /// Runs one CED half of a Bounded Replay over the tail of a token window.
    ///
    /// The producer fills only the layers of this half (D7: encoder 0..=20, 21 layers; decoder
    /// 21..=39, 19 layers); the other half keeps whatever was already in the ring. Only the SWA
    /// lock is held across the producer's work, and the store's replay scope is armed around it,
    /// so a producer that nonetheless reached the global-KV publish path would be refused and
    /// counted instead of silently overwriting cached KV (see [TieredPageStore::begin_replay]).
    ///
    /// # Panics
    /// Panics if the producer re-enters this cache: see [SwaProducer::fill_layer].
    ///
    /// # Errors
    /// Returns the producer's error, or an error from the ring.
    pub fn bounded_replay<P: SwaProducer>(
        &self,
        kind: ReplayKind,
        tokens: &[u32],
        end_position: usize,
        producer: &mut P,
    ) -> Result<ReplayOutcome, KvError> {
        let start = tokens.len().saturating_sub(SWA_SLOTS_PER_TAIL);
        let window = &tokens[start..];
        if window.is_empty() {
            return Ok(ReplayOutcome::default());
        }
        // The scope is armed before the SWA lock is taken and released only by dropping its
        // guard, so it closes on the error path and while unwinding: a producer that panics
        // cannot leave this thread marked as replaying for the life of the cache.
        let before = self
            .store_telemetry
            .replay_global_kv_writes
            .load(std::sync::atomic::Ordering::Relaxed);
        let scope = self.replay.enter();
        let result = self.replay_window(kind, window, end_position, producer);
        drop(scope);
        let written = self
            .store_telemetry
            .replay_global_kv_writes
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(before);
        result.map(|mut outcome| {
            outcome.global_kv_pages_written = written;
            outcome
        })
    }

    /// The SWA-lock half of [KvCache::bounded_replay].
    fn replay_window<P: SwaProducer>(
        &self,
        kind: ReplayKind,
        window: &[u32],
        end_position: usize,
        producer: &mut P,
    ) -> Result<ReplayOutcome, KvError> {
        let full_window = window.len() == SWA_SLOTS_PER_TAIL;
        let mut ring = self.lock_swa();
        let tail = if full_window {
            Some(ring.ensure_tail(window, end_position)?)
        } else {
            None
        };
        let mut layer_buffer = vec![0u8; SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES];
        let mut slots = 0u64;
        for layer in kind.layers() {
            layer_buffer.fill(0);
            {
                // Held for the callback only: the guard must not fire for the ring writes
                // this function makes around it.
                let _callback = ProducerWindow::enter();
                producer.fill_layer(layer, window, &mut layer_buffer)?;
            }
            if let Some(id) = tail {
                ring.write_layer(id, layer, &layer_buffer)?;
            }
            slots += window.len() as u64;
        }
        let tail_completed = if let Some(id) = tail {
            ring.mark_layers(id, kind)?;
            ring.get(id).is_some_and(|entry| entry.is_complete())
        } else {
            false
        };
        ring.note_replay(slots);
        Ok(ReplayOutcome {
            replayed_tokens: window.len(),
            swa_slots_written: slots,
            global_kv_pages_written: 0,
            tail_completed,
        })
    }

    /// Fills all 40 layers of a tail; this is the prefill path, as opposed to the two
    /// half-backbone replays.
    ///
    /// # Errors
    /// Propagates the producer's error and any ring error.
    pub fn cache_swa_tail<P: SwaProducer>(
        &self,
        window: &[u32],
        end_position: usize,
        producer: &mut P,
    ) -> Result<ReplayOutcome, KvError> {
        let encoder = self.bounded_replay(ReplayKind::Encoder, window, end_position, producer)?;
        let decoder = self.bounded_replay(ReplayKind::Decoder, window, end_position, producer)?;
        Ok(merge_replay_outcomes(&encoder, &decoder))
    }

    /// Looks for a complete tail at the end of a cached prefix.
    #[must_use]
    pub fn probe_swa_tail(&self, prefix: &[u32]) -> Option<TailId> {
        self.lock_swa().probe_prefix(prefix)
    }

    /// Reads one SWA slot of a resident tail.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail was evicted and KvError::InvalidLength
    /// when the output buffer is not one slot.
    pub fn read_swa_slot(
        &self,
        id: TailId,
        layer: usize,
        slot: usize,
        out: &mut [u8],
    ) -> Result<(), KvError> {
        self.lock_swa().read_slot(id, layer, slot, out)
    }

    /// Reads one whole SWA layer of a resident tail.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail was evicted and KvError::InvalidLength
    /// when the output buffer is not one layer.
    pub fn read_swa_layer(&self, id: TailId, layer: usize, out: &mut [u8]) -> Result<(), KvError> {
        self.lock_swa().read_layer(id, layer, out)
    }

    /// Number of tail slots in the ring.
    #[must_use]
    pub fn swa_capacity(&self) -> usize {
        self.lock_swa().capacity()
    }

    /// Reads one page from whichever tier holds it.
    ///
    /// # Errors
    /// Returns KvError::PageNotResident when the store does not track the page.
    pub fn read_page(&self, page: PageId, out: &mut [u8]) -> Result<Tier, KvError> {
        let inner = self.lock_inner();
        inner.store.read(page, out)
    }

    /// Evicts up to a number of least-recently-used unpinned pages.
    ///
    /// # Errors
    /// Returns an error when a tier cannot complete the removal.
    pub fn evict(&self, count: usize) -> Result<usize, KvError> {
        let mut inner = self.lock_inner();
        let freed = inner.tree.evict_lru(count);
        for page in &freed {
            inner.store.drop_page(*page)?;
            self.telemetry.note_page_freed();
        }
        Ok(freed.len())
    }

    /// Pages the radix tree currently indexes.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.lock_inner().tree.page_count()
    }

    /// Resident pages in a tier.
    #[must_use]
    pub fn resident_pages(&self, tier: Tier) -> u64 {
        self.lock_inner().store.resident(tier)
    }

    /// Checks that the radix tree, its reference counts and the three tiers agree about which
    /// pages exist, how many live sequences hold each one, and where each one lives.
    ///
    /// Trees and tiers are mutated together under one lock, so a disagreement means a bug in
    /// one of the two-phase mutations rather than a race. Tests that hammer the cache from many
    /// threads call this once the threads have joined.
    ///
    /// # Errors
    /// Returns KvError::Corrupt describing the first inconsistency found.
    pub fn check_invariants(&self) -> Result<(), KvError> {
        let inner = self.lock_inner();
        inner.tree.check_invariants()?;
        inner.store.check_invariants()?;
        let tree: HashSet<PageId> = inner.tree.all_pages().into_iter().collect();
        let store: HashSet<PageId> = inner.store.tracked_page_ids().into_iter().collect();
        if tree != store {
            return Err(KvError::Corrupt(format!(
                "the radix tree holds {} pages and the store tracks {}",
                tree.len(),
                store.len()
            )));
        }
        Ok(())
    }

    /// A content fingerprint of every resident page.
    ///
    /// This is a verification aid: the invariant tests hash the global KV before and after a
    /// bounded replay and require the digest to be unchanged. Checksumming every page is far
    /// too slow for a serving path, hence the documentation.
    ///
    /// # Errors
    /// Returns an error when a tracked page cannot be read back.
    pub fn resident_page_fingerprint(&self) -> Result<u64, KvError> {
        let inner = self.lock_inner();
        let mut pages = inner.store.tracked_page_ids();
        pages.sort_unstable();
        let mut buffer = vec![0u8; GLOBAL_PAGE_BYTES];
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for page in pages {
            inner.store.read(page, &mut buffer)?;
            hash ^= page.0;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            for byte in &buffer {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Ok(hash)
    }

    /// Flushes every tier's persistent state.
    ///
    /// # Errors
    /// Returns the first tier flush error.
    pub fn flush(&self) -> Result<(), KvError> {
        self.lock_inner().store.flush()
    }

    /// Snapshots both telemetry families and current residency.
    #[must_use]
    pub fn snapshot(&self) -> KvCacheSnapshot {
        let inner = self.lock_inner();
        let swa = self.lock_swa();
        KvCacheSnapshot {
            cache: self.telemetry.snapshot(),
            store: self.store_telemetry.snapshot(),
            uma_pages: inner.store.resident(Tier::Uma),
            dram_pages: inner.store.resident(Tier::Dram),
            nvme_pages: inner.store.resident(Tier::Nvme),
            tracked_pages: inner.store.tracked_pages(),
            swa_tails: swa.len(),
            swa_complete_tails: swa.complete_len(),
            swa_bytes: swa.byte_len(),
        }
    }

    fn release_path(&self, path: &[NodeId]) {
        if path.is_empty() {
            return;
        }
        let mut inner = self.lock_inner();
        let _ = inner.tree.release(path);
    }
}

/// Number of tokens a plan may serve from cached global KV, aligned to whole pages.
#[must_use]
pub fn usable_cached_tokens(total_prompt_tokens: usize, matched_tokens: usize) -> usize {
    let raw = matched_tokens.min(total_prompt_tokens.saturating_sub(1));
    raw / KV_PAGE_TOKENS * KV_PAGE_TOKENS
}

/// Combines the two halves of a full-tail replay into the one outcome the caller sees.
///
/// The global-KV counter is carried through from both halves rather than hardcoded: a literal
/// zero here would discard the only evidence a replay has of a refused publish, and the sum is
/// what makes the field falsifiable when a half is tested on its own.
#[must_use]
fn merge_replay_outcomes(encoder: &ReplayOutcome, decoder: &ReplayOutcome) -> ReplayOutcome {
    ReplayOutcome {
        replayed_tokens: encoder.replayed_tokens,
        swa_slots_written: encoder.swa_slots_written + decoder.swa_slots_written,
        global_kv_pages_written: encoder.global_kv_pages_written + decoder.global_kv_pages_written,
        tail_completed: decoder.tail_completed,
    }
}

/// Number of layers a bounded replay covers, per the D7 split: 21 encoder, 19 decoder.
#[must_use]
pub const fn replay_layer_count(kind: ReplayKind) -> usize {
    match kind {
        ReplayKind::Encoder => SWA_ENCODER_LAYERS,
        ReplayKind::Decoder => SWA_DECODER_LAYERS,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SWA_LAYERS;
    use crate::tier::ReplayGuard;

    fn test_cache(uma: u64, dram: u64, nvme: u64) -> KvCache {
        KvCache::new(KvCacheConfig {
            budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
            uma_pages: uma,
            dram_pages: dram,
            nvme_pages: nvme,
            swa_tail_slots: 2,
            nvme_path: None,
        })
        .expect("cache")
    }

    #[test]
    fn a_full_tail_outcome_carries_both_halves_global_write_counts() {
        // The merge is the only place that could still hardcode a zero, so it is tested with
        // counts that a hardcoded zero would not reproduce.
        let encoder = ReplayOutcome {
            replayed_tokens: SWA_SLOTS_PER_TAIL,
            swa_slots_written: SWA_ENCODER_LAYERS as u64 * SWA_SLOTS_PER_TAIL as u64,
            global_kv_pages_written: 1,
            tail_completed: false,
        };
        let decoder = ReplayOutcome {
            replayed_tokens: SWA_SLOTS_PER_TAIL,
            swa_slots_written: SWA_DECODER_LAYERS as u64 * SWA_SLOTS_PER_TAIL as u64,
            global_kv_pages_written: 2,
            tail_completed: true,
        };
        let merged = merge_replay_outcomes(&encoder, &decoder);
        assert_eq!(merged.replayed_tokens, SWA_SLOTS_PER_TAIL);
        assert_eq!(
            merged.swa_slots_written,
            SWA_LAYERS as u64 * SWA_SLOTS_PER_TAIL as u64
        );
        assert_eq!(
            merged.global_kv_pages_written, 3,
            "a hardcoded zero would hide a refused publish from both halves"
        );
        assert!(merged.tail_completed);
        assert!(!merged.global_kv_untouched());
    }

    #[test]
    fn the_replay_outcome_reports_the_global_write_counter_delta() {
        // ReplayOutcome::global_kv_pages_written is the difference this function returns, so
        // this test is what stops that field from being a literal zero: the counter is moved
        // here exactly the way a replay that reached the publish path would move it.
        let cache = test_cache(2, 2, 2);
        let before = cache
            .store_telemetry()
            .replay_global_kv_writes
            .load(std::sync::atomic::Ordering::Relaxed);
        let scope = cache.lock_inner().store.begin_replay();
        assert_eq!(cache.replay.depth(), 1);
        let refused = cache.lock_inner().store.insert(
            PageId(0),
            &vec![0u8; GLOBAL_PAGE_BYTES],
            &HashSet::new(),
        );
        assert!(
            matches!(refused, Err(KvError::ReplayWroteGlobalKv(_))),
            "{refused:?}"
        );
        drop(scope);
        let written = cache
            .store_telemetry()
            .replay_global_kv_writes
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(before);
        assert_eq!(
            written, 1,
            "the delta a replay would report must include the refused publish"
        );
        let snapshot = cache.store_telemetry().snapshot();
        assert_eq!(snapshot.replay_global_kv_writes, 1);
        assert_eq!(snapshot.replay_scopes, 1);
        assert_eq!(cache.replay.depth(), 0);
        assert!(
            !cache.lock_inner().store.in_replay(),
            "the scope stayed open"
        );
    }

    /// A producer that reaches the same guard the store installs on its backends, once per layer.
    ///
    /// The cache's producer window deliberately has no handle on the store, so a real replay
    /// reports a zero delta by construction and two halves can only be compared as 0 == 0 + 0.
    /// This producer moves the counter the way a replay path that reached a guarded backend
    /// would, which makes the end-to-end value non-zero and the merge falsifiable.
    struct GuardPokingProducer {
        guard: ReplayGuard,
        refusals: u64,
    }

    impl GuardPokingProducer {
        fn new(cache: &KvCache) -> Self {
            Self {
                guard: ReplayGuard::new(
                    Arc::clone(&cache.replay),
                    Arc::clone(&cache.store_telemetry),
                ),
                refusals: 0,
            }
        }
    }

    impl SwaProducer for GuardPokingProducer {
        fn fill_layer(
            &mut self,
            layer: usize,
            window: &[u32],
            out: &mut [u8],
        ) -> Result<(), KvError> {
            // One refused publish attempt per layer. The refusal is not a failure of this
            // producer - it is the event under test - so the error is consumed and the layer is
            // still filled.
            let _ = self.guard.refuse(PageId(self.refusals + 1));
            self.refusals += 1;
            for (slot, _) in window.iter().enumerate() {
                let base = slot * SWA_SLOT_BYTES;
                out[base..base + SWA_SLOT_BYTES].fill((layer % 251) as u8);
            }
            Ok(())
        }
    }

    #[test]
    fn a_full_tail_outcome_sums_a_non_zero_counter_from_both_halves() {
        // The end-to-end falsifier for the merge: a full tail fills both halves, one layer at a
        // time, so the counter ends at the ring's layer count. A hardcoded zero (0), a single
        // half (21 or 19) or a doubled merge (80) all fail here, which the all-zero public-path
        // comparison cannot show.
        let cache = test_cache(2, 2, 2);
        let tokens: Vec<u32> = (0..320u32).collect();
        let mut producer = GuardPokingProducer::new(&cache);
        let outcome = cache
            .cache_swa_tail(&tokens, 320, &mut producer)
            .expect("full tail");
        assert_eq!(
            producer.refusals, SWA_LAYERS as u64,
            "one refused publish per layer of the full tail"
        );
        assert_eq!(
            outcome.global_kv_pages_written, SWA_LAYERS as u64,
            "the merged outcome must carry the sum of both halves"
        );
        assert!(!outcome.global_kv_untouched());
        let snapshot = cache.store_telemetry().snapshot();
        assert_eq!(snapshot.replay_global_kv_writes, SWA_LAYERS as u64);
        assert_eq!(
            snapshot.replay_scopes, 2,
            "a full tail is two bounded replays"
        );
    }
}
