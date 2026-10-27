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

//! The 16-way set-associative UMA + DRAM row cache for Engram lookups.
//!
//! The specification asks for a two-level resident cache in front of the NVMe tables:
//! a small, fast UMA tier (`tier 2`) and a larger DRAM tier (`tier 1`, the specification's
//! "DRM" carveout). Both are 16-way set-associative and keyed by `(layer, row)`, and both
//! use the same eviction policy, so the two tiers differ only in capacity and in the order
//! they are consulted.
//!
//! Correctness under concurrency uses one `parking_lot::RwLock` per set plus atomics for
//! the replacement metadata:
//!
//! * readers take a shared lock, copy the 264-byte row out and bump the way's clock
//!   reference bit; many readers of the same set proceed in parallel;
//! * the single writer takes the exclusive lock only when it inserts, so a row can never
//!   be overwritten while a reader is copying it;
//! * hit/miss/eviction counters are relaxed atomics, cheap enough for the hot path.
//!
//! Eviction is CLOCK by default: each way has a reference bit set on access, and the
//! per-set hand advances until it finds a way whose bit is clear, clearing bits as it goes
//! (second chance). Strict LRU is available as an alternative and shares the same code
//! path with a per-way stamp instead of a bit, which the tests use for deterministic
//! expectations.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::layout::ENGRAM_ROW_BYTES;
use crate::store::RowTable;
use crate::telemetry::EngramLayerTelemetry;

/// Fixed associativity from the specification.
pub const WAYS: usize = 16;

/// Highest row index a cache key can hold.
pub const MAX_ROW: u64 = (1 << 40) - 1;

/// Highest layer index a cache key can hold.
pub const MAX_LAYER: usize = 255;

/// Bit marking a populated way; `0` means empty.
const VALID_BIT: u64 = 1 << 63;

/// Default UMA capacity: 256 MiB, ~1.0 million rows.
pub const DEFAULT_UMA_BYTES: usize = 256 * 1024 * 1024;

/// Default DRAM capacity: 8 GiB, ~32.5 million rows.
pub const DEFAULT_DRAM_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Largest number of sets the constructor will allocate, as a guard against a capacity
/// typo turning into an enormous allocation.
const MAX_SETS: usize = 1 << 22;

/// Tier a lookup resolved in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTier {
    /// Tier 2: the small, fast UMA tier.
    Uma,
    /// Tier 1: the larger DRAM tier.
    Dram,
    /// Neither tier held the row; the caller must fetch it from the store.
    Nvme,
}

impl CacheTier {
    /// Telemetry field this tier increments: `1` is the DRAM/resident tier, `2` the UMA
    /// tier and `3` an NVMe miss, matching `[EngramLayerTelemetry::record_lookups]`.
    #[must_use]
    pub fn telemetry_index(&self) -> u8 {
        match self {
            Self::Dram => 1,
            Self::Uma => 2,
            Self::Nvme => 3,
        }
    }
}

/// Replacement policy used by every set of both tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Second-chance CLOCK with one hand per set. The default.
    #[default]
    Clock,
    /// Exact least-recently-used, selected by comparing per-way stamps.
    Lru,
}

/// Cache geometry and capacities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    /// UMA tier capacity in bytes.
    pub uma_bytes: usize,
    /// DRAM tier capacity in bytes.
    pub dram_bytes: usize,
    /// Associativity of both tiers; the specification fixes it at `[WAYS]`.
    pub ways: usize,
    /// Replacement policy.
    pub policy: EvictionPolicy,
    /// Number of layers the cache is keyed by, so per-layer telemetry has a home.
    pub layer_count: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            uma_bytes: DEFAULT_UMA_BYTES,
            dram_bytes: DEFAULT_DRAM_BYTES,
            ways: WAYS,
            policy: EvictionPolicy::Clock,
            layer_count: crate::layout::ENGRAM_LAYERS.len(),
        }
    }
}

impl CacheConfig {
    /// Number of sets a tier with `bytes` capacity gets.
    ///
    /// # Errors
    /// When the capacity cannot hold even one set of `ways` rows.
    pub fn sets_for(bytes: usize, ways: usize) -> anyhow::Result<usize> {
        anyhow::ensure!(
            ways > 0 && ways <= 64,
            "associativity {ways} is out of 1..=64"
        );
        let set_bytes = ways * ENGRAM_ROW_BYTES;
        anyhow::ensure!(
            bytes >= set_bytes,
            "capacity {bytes} bytes cannot hold one {ways}-way set ({set_bytes} bytes)"
        );
        Ok((bytes / set_bytes).min(MAX_SETS))
    }

    /// Validates the configuration and returns the set counts of both tiers.
    ///
    /// # Errors
    /// When either capacity is too small, or the associativity is invalid.
    pub fn validate(&self) -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(
            self.layer_count >= 1 && self.layer_count <= MAX_LAYER + 1,
            "layer count {} is out of 1..={}",
            self.layer_count,
            MAX_LAYER + 1
        );
        Ok((
            Self::sets_for(self.uma_bytes, self.ways)?,
            Self::sets_for(self.dram_bytes, self.ways)?,
        ))
    }
}

/// Packs `(layer, row)` into a tag, or `None` when it cannot be represented.
#[must_use]
pub fn pack_key(layer: usize, row: u64) -> Option<u64> {
    if layer > MAX_LAYER || row > MAX_ROW {
        return None;
    }
    Some(VALID_BIT | ((layer as u64) << 40) | row)
}

/// Unpacks a tag into `(layer, row)`.
#[must_use]
pub fn unpack_key(key: u64) -> (usize, u64) {
    (((key >> 40) & 0xff) as usize, key & MAX_ROW)
}

/// One set-associative set, holding `ways` rows of `[ENGRAM_ROW_BYTES]` bytes.
struct Set16 {
    /// Way tags; `0` marks an empty way.
    tags: Box<[AtomicU64]>,
    /// CLOCK reference bits.
    refbits: Box<[AtomicBool]>,
    /// LRU stamps.
    stamps: Box<[AtomicU64]>,
    /// Monotonic stamp source for `[EvictionPolicy::Lru]`.
    tick: AtomicU64,
    /// CLOCK hand; only touched by the single writer.
    hand: usize,
    /// Packed row storage, `ways * ROW_BYTES` bytes.
    data: Box<[u8]>,
    /// Replacement policy.
    policy: EvictionPolicy,
}

impl Set16 {
    fn new(ways: usize, policy: EvictionPolicy) -> Self {
        let tags = (0..ways).map(|_| AtomicU64::new(0)).collect();
        let refbits = (0..ways).map(|_| AtomicBool::new(false)).collect();
        let stamps = (0..ways).map(|_| AtomicU64::new(0)).collect();
        Self {
            tags,
            refbits,
            stamps,
            tick: AtomicU64::new(1),
            hand: 0,
            data: vec![0u8; ways * ENGRAM_ROW_BYTES].into_boxed_slice(),
            policy,
        }
    }

    fn ways(&self) -> usize {
        self.tags.len()
    }

    /// Way holding `key`, if any.
    fn find(&self, key: u64) -> Option<usize> {
        self.tags
            .iter()
            .position(|tag| tag.load(Ordering::Acquire) == key)
    }

    /// Row stored in `way`.
    fn row(&self, way: usize) -> &[u8] {
        let start = way * ENGRAM_ROW_BYTES;
        &self.data[start..start + ENGRAM_ROW_BYTES]
    }

    /// Marks a way as recently used. Safe to call while other readers hold the set: the
    /// replacement metadata is atomic and an inconsistent bit cannot corrupt a row.
    fn touch(&self, way: usize) {
        match self.policy {
            EvictionPolicy::Clock => self.refbits[way].store(true, Ordering::Relaxed),
            EvictionPolicy::Lru => {
                let stamp = self.tick.fetch_add(1, Ordering::Relaxed);
                self.stamps[way].store(stamp, Ordering::Relaxed);
            }
        }
    }

    /// Chooses the way to replace, preferring an empty way.
    fn victim(&mut self) -> usize {
        if let Some(way) = self
            .tags
            .iter()
            .position(|tag| tag.load(Ordering::Relaxed) == 0)
        {
            return way;
        }
        match self.policy {
            EvictionPolicy::Clock => {
                // Classic second chance: advance the hand past referenced ways, clearing
                // each reference bit, and take the first way whose bit is clear. After at
                // most a full turn every bit is clear, so this always terminates.
                let ways = self.ways();
                loop {
                    let way = self.hand;
                    self.hand = (self.hand + 1) % ways;
                    if !self.refbits[way].load(Ordering::Relaxed) {
                        return way;
                    }
                    self.refbits[way].store(false, Ordering::Relaxed);
                }
            }
            EvictionPolicy::Lru => {
                let mut victim = 0;
                let mut oldest = u64::MAX;
                for way in 0..self.ways() {
                    let stamp = self.stamps[way].load(Ordering::Relaxed);
                    if stamp < oldest {
                        oldest = stamp;
                        victim = way;
                    }
                }
                victim
            }
        }
    }

    /// Invalidates `key`, returning true when it was resident.
    fn remove(&self, key: u64) -> bool {
        match self.find(key) {
            Some(way) => {
                self.tags[way].store(0, Ordering::Release);
                self.refbits[way].store(false, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// Writes `row` under `key`, returning the evicted entry when a way had to be reused.
    fn store(
        &mut self,
        key: u64,
        row: &[u8; ENGRAM_ROW_BYTES],
    ) -> Option<(u64, [u8; ENGRAM_ROW_BYTES])> {
        if let Some(way) = self.find(key) {
            self.write_row(way, row);
            self.touch(way);
            return None;
        }
        let way = self.victim();
        let previous = self.tags[way].load(Ordering::Relaxed);
        let evicted = (previous != 0).then(|| {
            let mut copy = [0u8; ENGRAM_ROW_BYTES];
            copy.copy_from_slice(self.row(way));
            (previous, copy)
        });
        self.write_row(way, row);
        self.tags[way].store(key, Ordering::Release);
        self.touch(way);
        evicted
    }

    fn write_row(&mut self, way: usize, row: &[u8; ENGRAM_ROW_BYTES]) {
        let start = way * ENGRAM_ROW_BYTES;
        self.data[start..start + ENGRAM_ROW_BYTES].copy_from_slice(row);
    }
}

/// One cache tier: a fixed array of sets plus its own capacity counters.
struct Tier {
    sets: Vec<RwLock<Set16>>,
    capacity_bytes: usize,
    insertions: AtomicU64,
    evictions: AtomicU64,
}

impl Tier {
    fn new(sets: usize, ways: usize, policy: EvictionPolicy) -> Self {
        let mut storage = Vec::with_capacity(sets);
        for _ in 0..sets {
            storage.push(RwLock::new(Set16::new(ways, policy)));
        }
        Self {
            sets: storage,
            capacity_bytes: sets * ways * ENGRAM_ROW_BYTES,
            insertions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn set_count(&self) -> usize {
        self.sets.len()
    }

    /// Invalidates a resident row, returning true when it was there.
    fn remove(&self, key: u64) -> bool {
        self.sets[self.set_index(key)].write().remove(key)
    }

    fn set_index(&self, key: u64) -> usize {
        (mix(key) as usize) % self.sets.len()
    }

    /// Looks a row up, marking it used on a hit.
    fn get(&self, key: u64, out: &mut [u8; ENGRAM_ROW_BYTES]) -> bool {
        let set = self.sets[self.set_index(key)].read();
        match set.find(key) {
            Some(way) => {
                out.copy_from_slice(set.row(way));
                set.touch(way);
                true
            }
            None => false,
        }
    }

    /// True when the row is resident, without marking it used.
    fn contains(&self, key: u64) -> bool {
        self.sets[self.set_index(key)].read().find(key).is_some()
    }

    /// Inserts a row, returning the entry that had to be demoted to make room.
    fn insert(
        &self,
        key: u64,
        row: &[u8; ENGRAM_ROW_BYTES],
    ) -> Option<(u64, [u8; ENGRAM_ROW_BYTES])> {
        let mut set = self.sets[self.set_index(key)].write();
        self.insertions.fetch_add(1, Ordering::Relaxed);
        let evicted = set.store(key, row);
        if evicted.is_some() {
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
        evicted
    }
}

/// SplitMix64 finaliser: spreads sequential rows across sets.
fn mix(key: u64) -> u64 {
    let mut x = key;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// Lookup counters, shared by the whole cache and by each layer's telemetry slot.
#[derive(Debug, Default)]
pub struct CacheCounters {
    /// Lookups served.
    pub lookups: AtomicU64,
    /// Lookups answered by the UMA tier.
    pub uma_hits: AtomicU64,
    /// Lookups answered by the DRAM tier (before promotion).
    pub dram_hits: AtomicU64,
    /// Lookups that had to reach the store.
    pub misses: AtomicU64,
    /// Row bytes fetched from the store.
    pub bytes_read_nvme: AtomicU64,
    /// Row bytes moved between the tiers by promotions and demotions: one row for a DRAM to UMA
    /// promotion that found the row resident, and one row for each UMA to DRAM move, including
    /// the victim a promotion displaces.
    pub bytes_moved_between_tiers: AtomicU64,
}

impl CacheCounters {
    fn record(&self, tier: CacheTier) {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        match tier {
            CacheTier::Uma => self.uma_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::Dram => self.dram_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::Nvme => self.misses.fetch_add(1, Ordering::Relaxed),
        };
    }

    fn add_bytes_read(&self, bytes: u64) {
        self.bytes_read_nvme.fetch_add(bytes, Ordering::Relaxed);
    }

    fn add_bytes_moved(&self, bytes: u64) {
        self.bytes_moved_between_tiers
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Reads the counters without disturbing them.
    #[must_use]
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            lookups: self.lookups.load(Ordering::Relaxed),
            uma_hits: self.uma_hits.load(Ordering::Relaxed),
            dram_hits: self.dram_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_read_nvme: self.bytes_read_nvme.load(Ordering::Relaxed),
            bytes_moved_between_tiers: self.bytes_moved_between_tiers.load(Ordering::Relaxed),
        }
    }

    /// Reads the counters and resets them, for delta export into telemetry.
    #[must_use]
    pub fn take(&self) -> CounterSnapshot {
        CounterSnapshot {
            lookups: self.lookups.swap(0, Ordering::Relaxed),
            uma_hits: self.uma_hits.swap(0, Ordering::Relaxed),
            dram_hits: self.dram_hits.swap(0, Ordering::Relaxed),
            misses: self.misses.swap(0, Ordering::Relaxed),
            bytes_read_nvme: self.bytes_read_nvme.swap(0, Ordering::Relaxed),
            bytes_moved_between_tiers: self.bytes_moved_between_tiers.swap(0, Ordering::Relaxed),
        }
    }
}

/// Plain-value view of `[CacheCounters]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterSnapshot {
    /// Lookups served.
    pub lookups: u64,
    /// UMA hits.
    pub uma_hits: u64,
    /// DRAM hits.
    pub dram_hits: u64,
    /// Store misses.
    pub misses: u64,
    /// Row bytes fetched from the store.
    pub bytes_read_nvme: u64,
    /// Row bytes moved between the tiers by promotions and demotions: one row for a DRAM to UMA
    /// promotion that found the row resident, and one row for each UMA to DRAM move, including
    /// the victim a promotion displaces. It is the same counter as
    /// [`CacheCounters::bytes_moved_between_tiers`], and the two descriptions are one
    /// description (review A2, round 4).
    pub bytes_moved_between_tiers: u64,
}

impl CounterSnapshot {
    /// Fraction of lookups that did not reach the store, or `0.0` with no lookups.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            (self.uma_hits + self.dram_hits) as f64 / self.lookups as f64
        }
    }
}

/// Per-layer counters inside a cache snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerCacheSnapshot {
    /// Position of the layer in the Engram layer table.
    pub layer_index: usize,
    /// That layer's counters.
    pub counters: CounterSnapshot,
    /// That layer's hit rate.
    pub hit_rate: f64,
}

/// Serializable view of a cache's geometry and counters, for the cluster stats endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheSnapshot {
    /// Aggregate counters.
    pub counters: CounterSnapshot,
    /// Aggregate hit rate.
    pub hit_rate: f64,
    /// UMA hit rate, relative to all lookups.
    pub uma_hit_rate: f64,
    /// DRAM hit rate, relative to all lookups.
    pub dram_hit_rate: f64,
    /// Usable UMA capacity; a whole number of sets.
    pub uma_capacity_bytes: usize,
    /// Usable DRAM capacity; a whole number of sets.
    pub dram_capacity_bytes: usize,
    /// Number of UMA sets.
    pub uma_sets: usize,
    /// Number of DRAM sets.
    pub dram_sets: usize,
    /// Associativity.
    pub ways: usize,
    /// Replacement policy.
    pub policy: EvictionPolicy,
    /// Ways the UMA tier had to reuse.
    pub uma_evictions: u64,
    /// Ways the DRAM tier had to reuse.
    pub dram_evictions: u64,
    /// Rows stored in the UMA tier.
    pub uma_insertions: u64,
    /// Rows stored in the DRAM tier.
    pub dram_insertions: u64,
    /// Per-layer counters, one slot per configured layer.
    pub per_layer: Vec<LayerCacheSnapshot>,
}

/// The UMA + DRAM Engram row cache.
pub struct EngramRowCache {
    config: CacheConfig,
    uma: Tier,
    dram: Tier,
    counters: CacheCounters,
    per_layer: Vec<CacheCounters>,
}

impl Tier {
    /// Drops every resident row, leaving both tiers empty.
    fn clear(&self) {
        for set in &self.sets {
            let mut set = set.write();
            for tag in set.tags.iter() {
                tag.store(0, Ordering::Relaxed);
            }
            for bit in set.refbits.iter() {
                bit.store(false, Ordering::Relaxed);
            }
            for stamp in set.stamps.iter() {
                stamp.store(0, Ordering::Relaxed);
            }
            set.hand = 0;
        }
    }
}

impl EngramRowCache {
    /// Builds a cache from a configuration.
    ///
    /// # Errors
    /// When the configuration is invalid; see `[CacheConfig::validate]`.
    pub fn new(config: CacheConfig) -> anyhow::Result<Self> {
        let (uma_sets, dram_sets) = config.validate()?;
        let per_layer = (0..config.layer_count)
            .map(|_| CacheCounters::default())
            .collect();
        Ok(Self {
            uma: Tier::new(uma_sets, config.ways, config.policy),
            dram: Tier::new(dram_sets, config.ways, config.policy),
            config,
            counters: CacheCounters::default(),
            per_layer,
        })
    }

    /// Builds a CLOCK cache with explicit capacities.
    ///
    /// # Errors
    /// When a capacity cannot hold one set.
    pub fn with_capacities(
        uma_bytes: usize,
        dram_bytes: usize,
        layer_count: usize,
    ) -> anyhow::Result<Self> {
        Self::new(CacheConfig {
            uma_bytes,
            dram_bytes,
            ways: WAYS,
            policy: EvictionPolicy::Clock,
            layer_count,
        })
    }

    /// The configuration this cache was built from.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Usable `(uma, dram)` capacity in bytes; a whole number of sets, so it can be
    /// slightly below the configured capacity.
    #[must_use]
    pub fn capacity_bytes(&self) -> (usize, usize) {
        (self.uma.capacity_bytes, self.dram.capacity_bytes)
    }

    /// Telemetry slot for a layer id.
    ///
    /// Engram layer ids (`1` and `14`) map to their position in `[crate::layout::ENGRAM_LAYERS]`,
    /// which is the index the per-layer telemetry arrays use; other ids index directly, and
    /// a slot outside the configured layer count is simply skipped.
    fn layer_slot(&self, layer: usize) -> usize {
        crate::layout::ENGRAM_LAYERS
            .iter()
            .position(|id| *id == layer)
            .unwrap_or(layer)
    }

    fn record(&self, layer: usize, tier: CacheTier) {
        self.counters.record(tier);
        if let Some(counters) = self.per_layer.get(self.layer_slot(layer)) {
            counters.record(tier);
        }
    }

    /// Looks a row up, UMA tier first and then DRAM, promoting a DRAM hit into UMA.
    ///
    /// The tiers are exclusive, so a resident row has exactly one tier and a promotion moves
    /// it rather than copying it. The returned tier says where the row was found;
    /// `CacheTier::Nvme` means the caller must fetch it from the store. A row index too large
    /// to key is reported as a miss rather than an error.
    pub fn get(&self, layer: usize, row: u64, out: &mut [u8; ENGRAM_ROW_BYTES]) -> CacheTier {
        let Some(key) = pack_key(layer, row) else {
            self.record(layer, CacheTier::Nvme);
            return CacheTier::Nvme;
        };
        if self.uma.get(key, out) {
            self.record(layer, CacheTier::Uma);
            return CacheTier::Uma;
        }
        if self.dram.get(key, out) {
            self.record(layer, CacheTier::Dram);
            self.promote(key, out);
            return CacheTier::Dram;
        }
        self.record(layer, CacheTier::Nvme);
        CacheTier::Nvme
    }

    /// Whether a row is resident and in which tier, without touching the replacement state
    /// or the counters.
    #[must_use]
    pub fn contains(&self, layer: usize, row: u64) -> Option<CacheTier> {
        let key = pack_key(layer, row)?;
        if self.uma.contains(key) {
            Some(CacheTier::Uma)
        } else if self.dram.contains(key) {
            Some(CacheTier::Dram)
        } else {
            None
        }
    }

    /// Inserts or refreshes a row in the UMA tier, demoting whatever it displaces.
    pub fn insert(&self, layer: usize, row: u64, data: &[u8; ENGRAM_ROW_BYTES]) {
        if let Some(key) = pack_key(layer, row) {
            let _ = self.promote(key, data);
        }
    }

    /// Drops every resident row and leaves the counters alone.
    pub fn clear(&self) {
        self.uma.clear();
        self.dram.clear();
    }

    /// Moves a row into the UMA tier, demoting the displaced row into DRAM.
    ///
    /// The tiers are exclusive: a row that was resident in DRAM is invalidated there, so a
    /// resident row is stored and accounted for exactly once. An inclusive design would keep
    /// a second, unreachable copy of every promoted row in DRAM and count those 264 bytes
    /// against the DRAM capacity forever.
    fn promote(&self, key: u64, row: &[u8; ENGRAM_ROW_BYTES]) -> bool {
        if self.dram.remove(key) {
            self.counters.add_bytes_moved(ENGRAM_ROW_BYTES as u64);
        }
        match self.uma.insert(key, row) {
            Some((victim_key, victim_row)) => self.demote(victim_key, &victim_row),
            None => false,
        }
    }

    /// Pushes a row out of UMA and into DRAM.
    fn demote(&self, key: u64, row: &[u8; ENGRAM_ROW_BYTES]) -> bool {
        self.counters.add_bytes_moved(ENGRAM_ROW_BYTES as u64);
        self.dram.insert(key, row).is_some()
    }
}

impl EngramRowCache {
    /// Fetches a row through the cache, reading it from the store only on a miss.
    ///
    /// # Errors
    /// When `table_index` is out of range or the store read fails.
    pub fn get_or_fill(
        &self,
        table: &dyn RowTable,
        table_index: usize,
        row: u64,
        out: &mut [u8; ENGRAM_ROW_BYTES],
    ) -> anyhow::Result<CacheTier> {
        let layer = table
            .spec(table_index)
            .map(|spec| spec.layer_id)
            .ok_or_else(|| anyhow::anyhow!("table {table_index} is out of range"))?;
        let tier = self.get(layer, row, out);
        if tier != CacheTier::Nvme {
            return Ok(tier);
        }
        *out = table.read_row(table_index, row)?;
        self.counters.add_bytes_read(ENGRAM_ROW_BYTES as u64);
        self.insert(layer, row, out);
        Ok(CacheTier::Nvme)
    }

    /// Batched fetch: every row that misses is read from the store in one merged request.
    ///
    /// # Errors
    /// When the slice lengths differ, `table_index` is out of range, or a store read fails.
    pub fn get_or_fill_rows(
        &self,
        table: &dyn RowTable,
        table_index: usize,
        rows: &[u32],
        out: &mut [[u8; ENGRAM_ROW_BYTES]],
    ) -> anyhow::Result<Vec<CacheTier>> {
        anyhow::ensure!(
            rows.len() == out.len(),
            "{} rows requested but {} output slots given",
            rows.len(),
            out.len()
        );
        let layer = table
            .spec(table_index)
            .map(|spec| spec.layer_id)
            .ok_or_else(|| anyhow::anyhow!("table {table_index} is out of range"))?;
        let mut tiers = Vec::with_capacity(rows.len());
        let mut missing = Vec::new();
        for (slot, row) in rows.iter().enumerate() {
            let tier = self.get(layer, *row as u64, &mut out[slot]);
            tiers.push(tier);
            if tier == CacheTier::Nvme {
                missing.push(slot);
            }
        }
        if missing.is_empty() {
            return Ok(tiers);
        }
        let wanted: Vec<u32> = missing.iter().map(|slot| rows[*slot]).collect();
        let mut fetched = vec![[0u8; ENGRAM_ROW_BYTES]; wanted.len()];
        table.read_rows_into(table_index, &wanted, &mut fetched)?;
        self.counters
            .add_bytes_read((wanted.len() * ENGRAM_ROW_BYTES) as u64);
        for (slot, data) in missing.into_iter().zip(fetched) {
            out[slot] = data;
            self.insert(layer, rows[slot] as u64, &out[slot]);
        }
        Ok(tiers)
    }

    /// Snapshot of the geometry and every counter, ready to serialise.
    #[must_use]
    pub fn snapshot(&self) -> CacheSnapshot {
        let counters = self.counters.snapshot();
        let (uma_capacity_bytes, dram_capacity_bytes) = self.capacity_bytes();
        let per_layer = self
            .per_layer
            .iter()
            .enumerate()
            .map(|(layer_index, layer)| {
                let counters = layer.snapshot();
                LayerCacheSnapshot {
                    layer_index,
                    counters,
                    hit_rate: counters.hit_rate(),
                }
            })
            .collect();
        CacheSnapshot {
            counters,
            hit_rate: counters.hit_rate(),
            uma_hit_rate: rate(counters.uma_hits, counters.lookups),
            dram_hit_rate: rate(counters.dram_hits, counters.lookups),
            uma_capacity_bytes,
            dram_capacity_bytes,
            uma_sets: self.uma.set_count(),
            dram_sets: self.dram.set_count(),
            ways: self.config.ways,
            policy: self.config.policy,
            uma_evictions: self.uma.evictions.load(Ordering::Relaxed),
            dram_evictions: self.dram.evictions.load(Ordering::Relaxed),
            uma_insertions: self.uma.insertions.load(Ordering::Relaxed),
            dram_insertions: self.dram.insertions.load(Ordering::Relaxed),
            per_layer,
        }
    }

    /// Adds one layer's counters, as a delta, to that layer's telemetry.
    ///
    /// Returns `false` when `layer_index` is outside the configured layer count. The
    /// counters are consumed, so a metrics poller can call this repeatedly without double
    /// counting.
    pub fn record_telemetry(&self, layer_index: usize, telemetry: &EngramLayerTelemetry) -> bool {
        let Some(counters) = self.per_layer.get(layer_index) else {
            return false;
        };
        let delta = counters.take();
        if delta.lookups == 0 {
            return true;
        }
        telemetry.record_lookups(
            delta.lookups,
            delta.dram_hits,
            delta.uma_hits,
            delta.misses,
            0,
        );
        true
    }
}

/// `hits / lookups`, or `0.0` when there were no lookups.
fn rate(hits: u64, lookups: u64) -> f64 {
    if lookups == 0 {
        0.0
    } else {
        hits as f64 / lookups as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryTable, TableSpec, TailFill};
    use std::path::PathBuf;

    fn cfg(ways: usize, uma_rows: usize, dram_rows: usize, policy: EvictionPolicy) -> CacheConfig {
        CacheConfig {
            uma_bytes: ways * uma_rows * ENGRAM_ROW_BYTES,
            dram_bytes: ways * dram_rows * ENGRAM_ROW_BYTES,
            ways,
            policy,
            layer_count: 2,
        }
    }

    fn row(value: u8) -> [u8; ENGRAM_ROW_BYTES] {
        [value; ENGRAM_ROW_BYTES]
    }

    #[test]
    fn capacity_and_associativity_are_validated() {
        assert_eq!(CacheConfig::sets_for(16 * ENGRAM_ROW_BYTES, 16).unwrap(), 1);
        assert_eq!(CacheConfig::sets_for(48 * ENGRAM_ROW_BYTES, 16).unwrap(), 3);
        assert_eq!(
            CacheConfig::sets_for(64 * ENGRAM_ROW_BYTES - 1, 16).unwrap(),
            3
        );
        assert!(CacheConfig::sets_for(16 * ENGRAM_ROW_BYTES - 1, 16).is_err());
        assert!(CacheConfig::sets_for(16 * ENGRAM_ROW_BYTES, 0).is_err());
        assert!(CacheConfig::sets_for(16 * ENGRAM_ROW_BYTES, 65).is_err());
        assert!(
            EngramRowCache::new(CacheConfig {
                uma_bytes: 100,
                ..CacheConfig::default()
            })
            .is_err()
        );
        assert!(
            EngramRowCache::new(CacheConfig {
                layer_count: 0,
                ..CacheConfig::default()
            })
            .is_err()
        );
        let (uma, dram) = EngramRowCache::new(cfg(4, 1, 2, EvictionPolicy::Clock))
            .unwrap()
            .capacity_bytes();
        assert_eq!(uma, 4 * ENGRAM_ROW_BYTES);
        assert_eq!(dram, 8 * ENGRAM_ROW_BYTES);
    }

    #[test]
    fn clock_gives_a_recently_used_way_a_second_chance() {
        let cache = EngramRowCache::new(cfg(4, 1, 2, EvictionPolicy::Clock)).unwrap();
        for value in 0..4u8 {
            cache.insert(1, value as u64, &row(value));
        }
        // Filling the last way makes the next insert sweep the hand: every reference bit
        // is set on insert, so the hand makes a full turn and takes way 0.
        cache.insert(1, 4, &row(4));
        assert_eq!(cache.contains(1, 0), Some(CacheTier::Dram));
        cache.insert(1, 5, &row(5));
        // Reference way 1, which is exactly where the hand now points.
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        assert_eq!(cache.get(1, 5, &mut out), CacheTier::Uma);
        assert_eq!(out[0], 5);
        assert_eq!(cache.snapshot().uma_evictions, 2);
        // The hand must skip way 1 (referenced) and take way 2 instead.
        cache.insert(1, 6, &row(6));
        assert_eq!(cache.get(1, 5, &mut out), CacheTier::Uma);
        assert_eq!(out[0], 5);
        // Row 2 was the victim and now answers from the DRAM tier.
        assert_eq!(cache.get(1, 2, &mut out), CacheTier::Dram);
        assert_eq!(out[0], 2);
    }

    #[test]
    fn lru_evicts_the_least_recently_used_way() {
        let cache = EngramRowCache::new(cfg(4, 1, 2, EvictionPolicy::Lru)).unwrap();
        for value in 0..4u8 {
            cache.insert(1, value as u64, &row(value));
        }
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        assert_eq!(cache.get(1, 0, &mut out), CacheTier::Uma);
        cache.insert(1, 9, &row(9));
        // Row 0 was refreshed, so row 1 is the oldest and was evicted.
        assert_eq!(cache.get(1, 0, &mut out), CacheTier::Uma);
        assert_eq!(out[0], 0);
        assert_eq!(cache.get(1, 1, &mut out), CacheTier::Dram);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn dram_hits_promote_and_uma_victims_demote() {
        let cache = EngramRowCache::new(cfg(4, 1, 4, EvictionPolicy::Clock)).unwrap();
        for value in 0..4u8 {
            cache.insert(1, value as u64, &row(value));
        }
        cache.insert(1, 4, &row(4));
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        // Row 0 left UMA and was demoted into DRAM.
        assert_eq!(cache.contains(1, 0), Some(CacheTier::Dram));
        assert_eq!(cache.get(1, 0, &mut out), CacheTier::Dram);
        assert_eq!(out[0], 0);
        // The promotion put it back in UMA, and the tiers are exclusive: the DRAM copy is
        // gone, so a resident row is stored once.
        assert_eq!(cache.get(1, 0, &mut out), CacheTier::Uma);
        let key = pack_key(1, 0).expect("key");
        assert!(cache.uma.contains(key));
        assert!(
            !cache.dram.contains(key),
            "promotion must not leave a copy of the row in DRAM"
        );
        // One demotion while filling UMA, the DRAM copy removed by the promotion, and the
        // demotion of the row the promotion displaced.
        assert_eq!(
            cache.snapshot().counters.bytes_moved_between_tiers,
            (3 * ENGRAM_ROW_BYTES) as u64
        );
    }

    #[test]
    fn snapshot_reports_hit_rates_and_serialises() {
        let cache = EngramRowCache::new(cfg(16, 1, 1, EvictionPolicy::Clock)).unwrap();
        for value in 0..16u8 {
            cache.insert(1, value as u64, &row(value));
        }
        // Filling the set then adding one more demotes row 0 into DRAM.
        cache.insert(1, 16, &row(16));
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        assert_eq!(cache.get(1, 0, &mut out), CacheTier::Dram);
        assert_eq!(cache.get(1, 5, &mut out), CacheTier::Uma);
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.counters.lookups, 2);
        assert_eq!(snapshot.counters.uma_hits, 1);
        assert_eq!(snapshot.counters.dram_hits, 1);
        assert_eq!(snapshot.counters.misses, 0);
        assert_eq!(snapshot.hit_rate, 1.0);
        assert_eq!(snapshot.uma_hit_rate, 0.5);
        assert_eq!(snapshot.dram_hit_rate, 0.5);
        assert_eq!(snapshot.ways, 16);
        assert_eq!(snapshot.uma_sets, 1);
        // One eviction from the insert, one from promoting row 0 back into UMA.
        assert_eq!(snapshot.uma_evictions, 2);
        assert_eq!(snapshot.dram_insertions, 2);
        assert_eq!(snapshot.per_layer.len(), 2);
        // Layer 1 maps to telemetry slot 0; layer 14 was never touched.
        assert_eq!(snapshot.per_layer[0].counters.lookups, 2);
        assert_eq!(snapshot.per_layer[0].hit_rate, 1.0);
        assert_eq!(snapshot.per_layer[1].counters.lookups, 0);
        let json = serde_json::to_value(&snapshot).unwrap();
        for key in [
            "counters",
            "hit_rate",
            "uma_sets",
            "dram_sets",
            "ways",
            "policy",
            "per_layer",
            "uma_evictions",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["uma_sets"], 1);
    }

    #[test]
    fn record_telemetry_exports_layer_deltas() {
        let cache = EngramRowCache::new(cfg(16, 1, 1, EvictionPolicy::Clock)).unwrap();
        let telemetry = EngramLayerTelemetry::default();
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        assert_eq!(cache.get(1, 7, &mut out), CacheTier::Nvme);
        cache.insert(1, 7, &row(7));
        assert_eq!(cache.get(1, 7, &mut out), CacheTier::Uma);
        assert!(cache.record_telemetry(0, &telemetry));
        let snapshot = telemetry.snapshot(1);
        assert_eq!(snapshot.lookups_total, 2);
        assert_eq!(snapshot.tier2_uma_hits, 1);
        assert_eq!(snapshot.tier3_nvme_misses, 1);
        assert_eq!(snapshot.tier1_drm_hits, 0);
        // The export consumed the delta, so polling again adds nothing.
        assert!(cache.record_telemetry(0, &telemetry));
        assert_eq!(telemetry.snapshot(1).lookups_total, 2);
        assert!(!cache.record_telemetry(9, &telemetry));
    }

    fn small_table() -> MemoryTable {
        let spec = TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: PathBuf::from("/tmp/never-opened.bin"),
            row_start: 0,
            physical_rows: 8,
            logical_rows: 8,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        };
        let mut table = MemoryTable::new().with_layer(spec);
        table.fill_pattern(0).unwrap();
        table
    }

    #[test]
    fn batched_fill_reads_only_the_missing_rows() {
        let table = small_table();
        let cache = EngramRowCache::new(cfg(16, 1, 1, EvictionPolicy::Clock)).unwrap();
        let rows = [0u32, 1, 2, 3];
        let mut out = vec![[0u8; ENGRAM_ROW_BYTES]; 4];
        let tiers = cache.get_or_fill_rows(&table, 0, &rows, &mut out).unwrap();
        assert!(tiers.iter().all(|tier| *tier == CacheTier::Nvme));
        for (slot, data) in out.iter().enumerate() {
            assert_eq!(
                u64::from_le_bytes(data[..8].try_into().unwrap()),
                slot as u64
            );
        }
        let read = (4 * ENGRAM_ROW_BYTES) as u64;
        assert_eq!(cache.snapshot().counters.bytes_read_nvme, read);
        // A second pass is answered from UMA and reads nothing at all.
        let mut again = vec![[0u8; ENGRAM_ROW_BYTES]; 4];
        let tiers = cache
            .get_or_fill_rows(&table, 0, &rows, &mut again)
            .unwrap();
        assert!(tiers.iter().all(|tier| *tier == CacheTier::Uma));
        assert_eq!(cache.snapshot().counters.bytes_read_nvme, read);
        assert_eq!(out, again);
        // A single row still goes through get_or_fill.
        let mut one = [0u8; ENGRAM_ROW_BYTES];
        assert_eq!(
            cache.get_or_fill(&table, 0, 5, &mut one).unwrap(),
            CacheTier::Nvme
        );
        assert_eq!(u64::from_le_bytes(one[..8].try_into().unwrap()), 5);
        assert!(cache.get_or_fill(&table, 9, 0, &mut one).is_err());
        let mut short = vec![[0u8; ENGRAM_ROW_BYTES]; 1];
        assert!(
            cache
                .get_or_fill_rows(&table, 0, &[0, 1], &mut short)
                .is_err()
        );
    }

    #[test]
    fn concurrent_access_is_consistent() {
        let cache =
            std::sync::Arc::new(EngramRowCache::new(cfg(16, 4, 4, EvictionPolicy::Clock)).unwrap());
        let threads = 8u64;
        let iterations = 300u64;
        std::thread::scope(|scope| {
            for thread in 0..threads {
                let cache = std::sync::Arc::clone(&cache);
                scope.spawn(move || {
                    let mut out = [0u8; ENGRAM_ROW_BYTES];
                    for step in 0..iterations {
                        let index = (step * 7 + thread) % 24;
                        let value = (index % 251) as u8;
                        cache.insert(1, index, &row(value));
                        if cache.get(1, index, &mut out) != CacheTier::Nvme {
                            // Every writer of a key writes the same bytes, so a hit can
                            // never observe a torn or stale row.
                            assert_eq!(out[0], value);
                        }
                    }
                });
            }
        });
        assert_eq!(
            cache.snapshot().counters.lookups,
            threads * iterations,
            "every iteration records exactly one lookup"
        );
    }
}
