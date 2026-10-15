//! Storage-side KV telemetry: cumulative atomic counters, snapshots and deltas.
//!
//! # Consistency guarantee
//!
//! Every counter is an independent AtomicU64 updated with Ordering::Relaxed. A snapshot reads
//! the counters one at a time in a fixed order and is therefore **not** a linearizable
//! point-in-time view: a writer that runs between two loads can make one counter include an
//! operation that a counter read earlier does not. Two properties do hold, and are tested:
//!
//! * each individual counter is exact and monotone, because it is only ever mutated by
//!   fetch_add and every load observes some prefix of those additions;
//! * a delta between two snapshots taken in program order by the same thread is non-negative
//!   for every counter, because both snapshots see a prefix of the same monotone sequence.
//!
//! If a caller needs a consistent multi-counter view it must take it from a single thread
//! while no other thread is writing, which is what the single-threaded tests do.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// Cumulative counters for the paged KV store.
#[derive(Debug, Default)]
pub struct KvStoreTelemetry {
    /// Pages inserted into the store for the first time.
    pub page_inserts: AtomicU64,
    /// Pages removed from every tier and returned to the pool.
    pub page_drops: AtomicU64,
    /// Pages moved up a tier into coherent UMA.
    pub uma_promotions: AtomicU64,
    /// Pages moved down a tier to make room.
    pub demotions: AtomicU64,
    /// Page reads served from coherent UMA.
    pub uma_reads: AtomicU64,
    /// Page reads served from the DRAM staging tier.
    pub dram_reads: AtomicU64,
    /// Page reads served from the persistent NVMe tier.
    pub nvme_reads: AtomicU64,
    /// Payload bytes read from any tier.
    pub bytes_read: AtomicU64,
    /// Payload bytes written to any tier.
    pub bytes_written: AtomicU64,
    /// Requests that needed capacity the store could not free without dropping a referenced
    /// page. Each one makes the caller evict radix nodes and retry.
    pub capacity_refusals: AtomicU64,
    /// SWA tail slots written by a bounded replay.
    pub replay_swa_slots_written: AtomicU64,
    /// Global KV pages written while a bounded replay was in scope on the writing thread.
    /// Must stay zero: a replay may produce SWA KV only. The counter is incremented by
    /// TieredPageStore::insert, which is the global-KV publish path, whenever the calling
    /// thread is inside a replay scope opened by TieredPageStore::begin_replay, so a future
    /// replay path that reaches the store is caught by a counter rather than by review.
    pub replay_global_kv_writes: AtomicU64,
    /// Bounded replay scopes entered on the store, whether they wrote global KV or not.
    ///
    /// This is what makes replay_global_kv_writes falsifiable: it proves the real replay path
    /// arms the scope, so a global write during a replay cannot hide behind a flag that is
    /// never set.
    pub replay_scopes: AtomicU64,
}

/// Point-in-time copy of every store counter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStoreSnapshot {
    /// See KvStoreTelemetry::page_inserts.
    pub page_inserts: u64,
    /// See KvStoreTelemetry::page_drops.
    pub page_drops: u64,
    /// See KvStoreTelemetry::uma_promotions.
    pub uma_promotions: u64,
    /// See KvStoreTelemetry::demotions.
    pub demotions: u64,
    /// See KvStoreTelemetry::uma_reads.
    pub uma_reads: u64,
    /// See KvStoreTelemetry::dram_reads.
    pub dram_reads: u64,
    /// See KvStoreTelemetry::nvme_reads.
    pub nvme_reads: u64,
    /// See KvStoreTelemetry::bytes_read.
    pub bytes_read: u64,
    /// See KvStoreTelemetry::bytes_written.
    pub bytes_written: u64,
    /// See KvStoreTelemetry::capacity_refusals.
    pub capacity_refusals: u64,
    /// See KvStoreTelemetry::replay_swa_slots_written.
    pub replay_swa_slots_written: u64,
    /// See KvStoreTelemetry::replay_global_kv_writes.
    pub replay_global_kv_writes: u64,
    /// See KvStoreTelemetry::replay_scopes.
    pub replay_scopes: u64,
}

/// The difference between two snapshots.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStoreDelta {
    /// Operations between the two snapshots, per counter.
    pub page_inserts: u64,
    /// See KvStoreTelemetry::page_drops.
    pub page_drops: u64,
    /// See KvStoreTelemetry::uma_promotions.
    pub uma_promotions: u64,
    /// See KvStoreTelemetry::demotions.
    pub demotions: u64,
    /// See KvStoreTelemetry::uma_reads.
    pub uma_reads: u64,
    /// See KvStoreTelemetry::dram_reads.
    pub dram_reads: u64,
    /// See KvStoreTelemetry::nvme_reads.
    pub nvme_reads: u64,
    /// See KvStoreTelemetry::bytes_read.
    pub bytes_read: u64,
    /// See KvStoreTelemetry::bytes_written.
    pub bytes_written: u64,
    /// See KvStoreTelemetry::capacity_refusals.
    pub capacity_refusals: u64,
    /// See KvStoreTelemetry::replay_swa_slots_written.
    pub replay_swa_slots_written: u64,
    /// See KvStoreTelemetry::replay_global_kv_writes.
    pub replay_global_kv_writes: u64,
    /// See KvStoreTelemetry::replay_scopes.
    pub replay_scopes: u64,
}

impl KvStoreTelemetry {
    /// Reads every counter once, in field order.
    #[must_use]
    pub fn snapshot(&self) -> KvStoreSnapshot {
        KvStoreSnapshot {
            page_inserts: self.page_inserts.load(Ordering::Relaxed),
            page_drops: self.page_drops.load(Ordering::Relaxed),
            uma_promotions: self.uma_promotions.load(Ordering::Relaxed),
            demotions: self.demotions.load(Ordering::Relaxed),
            uma_reads: self.uma_reads.load(Ordering::Relaxed),
            dram_reads: self.dram_reads.load(Ordering::Relaxed),
            nvme_reads: self.nvme_reads.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            capacity_refusals: self.capacity_refusals.load(Ordering::Relaxed),
            replay_swa_slots_written: self.replay_swa_slots_written.load(Ordering::Relaxed),
            replay_global_kv_writes: self.replay_global_kv_writes.load(Ordering::Relaxed),
            replay_scopes: self.replay_scopes.load(Ordering::Relaxed),
        }
    }

    /// Difference between the current counters and an earlier snapshot.
    ///
    /// Uses saturating subtraction so a snapshot taken by a different thread, or a future
    /// counter wrap, degrades to zero rather than to a huge bogus delta.
    #[must_use]
    pub fn delta_since(&self, previous: &KvStoreSnapshot) -> KvStoreDelta {
        let now = self.snapshot();
        let sub = |a: u64, b: u64| a.saturating_sub(b);
        KvStoreDelta {
            page_inserts: sub(now.page_inserts, previous.page_inserts),
            page_drops: sub(now.page_drops, previous.page_drops),
            uma_promotions: sub(now.uma_promotions, previous.uma_promotions),
            demotions: sub(now.demotions, previous.demotions),
            uma_reads: sub(now.uma_reads, previous.uma_reads),
            dram_reads: sub(now.dram_reads, previous.dram_reads),
            nvme_reads: sub(now.nvme_reads, previous.nvme_reads),
            bytes_read: sub(now.bytes_read, previous.bytes_read),
            bytes_written: sub(now.bytes_written, previous.bytes_written),
            capacity_refusals: sub(now.capacity_refusals, previous.capacity_refusals),
            replay_swa_slots_written: sub(
                now.replay_swa_slots_written,
                previous.replay_swa_slots_written,
            ),
            replay_global_kv_writes: sub(
                now.replay_global_kv_writes,
                previous.replay_global_kv_writes,
            ),
            replay_scopes: sub(now.replay_scopes, previous.replay_scopes),
        }
    }
}

impl KvStoreDelta {
    /// True when no counter moved.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}
