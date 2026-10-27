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

//! Concurrency: many threads sharing one radix tree, one cache and one telemetry block.

use std::collections::HashSet;
use std::sync::Arc;

use sparkflash_kv::layout::{GLOBAL_PAGE_BYTES, MemoryBudget};
use sparkflash_kv::radix::RadixTree;
use sparkflash_kv::tier::{InMemoryTier, Tier, TieredPageStore};
use sparkflash_kv::{KvCache, KvCacheConfig, KvCacheTelemetry, KvError, KvStoreTelemetry};

fn cache_with(uma: u64, dram: u64, nvme: u64, swa: usize) -> KvCache {
    KvCache::new(KvCacheConfig {
        budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
        uma_pages: uma,
        dram_pages: dram,
        nvme_pages: nvme,
        swa_tail_slots: swa,
        nvme_path: None,
    })
    .expect("cache")
}

fn fill(block: usize, out: &mut [u8]) -> Result<(), KvError> {
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = ((block * 13 + index * 3) % 251) as u8;
    }
    Ok(())
}

fn shared_prefix() -> Vec<u32> {
    (0..64u32).map(|index| index % 7).collect()
}

#[test]
fn contention_on_one_radix_tree_leaves_no_leaked_reference() {
    // Repeated because the interleaving is scheduling-dependent.
    for attempt in 0..4 {
        let tree = Arc::new(parking_lot::Mutex::new(RadixTree::new()));
        let mut handles = Vec::new();
        for thread in 0..8u64 {
            let tree = Arc::clone(&tree);
            handles.push(std::thread::spawn(move || {
                let mut rng = thread.wrapping_mul(0x9E37_79B9) | 1;
                for _ in 0..60 {
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let blocks = 1 + (rng >> 33) % 4;
                    let mut tokens = shared_prefix();
                    for index in 0..(blocks as usize * 64) {
                        tokens.push(((rng >> (index % 17)) as u32) ^ (index as u32));
                    }
                    let matched = {
                        let mut guard = tree.lock();
                        let outcome = guard.insert(&tokens);
                        guard.attach(&outcome.path).expect("attach");
                        outcome.path
                    };
                    {
                        let mut guard = tree.lock();
                        guard.touch(&matched);
                    }
                    {
                        let mut guard = tree.lock();
                        guard.release(&matched).expect("release");
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread panicked");
        }
        let mut guard = tree.lock();
        let pages = guard.page_count();
        assert!(pages > 0, "attempt {attempt}: nothing was cached");
        assert!(
            guard.pinned_pages().is_empty(),
            "attempt {attempt}: a page is still pinned after every release"
        );
        // Only leaves are candidates up front: an interior node becomes evictable once the
        // last page below it is gone, which is what makes the full drain below possible.
        let leaves = guard.evictable_pages().len();
        assert!(
            leaves > 0 && leaves <= pages,
            "attempt {attempt}: {leaves} leaves for {pages} pages"
        );
        assert_eq!(guard.evict_lru(usize::MAX).len(), pages);
        assert!(guard.is_empty());
    }
}

#[test]
fn many_threads_sharing_one_cache_keep_the_accounting_consistent() {
    // Repeated because the interleaving is scheduling-dependent.
    for attempt in 0..3 {
        let cache = Arc::new(cache_with(8, 8, 64, 8));
        let mut handles = Vec::new();
        for thread in 0..8u64 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for step in 0..40u64 {
                    let seed = thread * 1_000 + step;
                    let mut tokens = shared_prefix();
                    let extra = 64 * (1 + (seed % 3) as usize);
                    for index in 0..extra {
                        tokens.push(((seed.wrapping_mul(31) + index as u64) % 61) as u32);
                    }
                    match cache.cache_prefix(&tokens, fill) {
                        Ok(outcome) => assert_eq!(outcome.cached_tokens % 64, 0),
                        Err(KvError::CapacityExceeded { .. }) => {}
                        Err(other) => panic!("unexpected cache error: {other}"),
                    }
                    let sequence = cache.begin_sequence(&tokens).expect("plan");
                    assert!(sequence.plan.global_kv_cached_tokens < tokens.len());
                    assert_eq!(
                        sequence.plan.uma_hit_tokens
                            + sequence.plan.nvme_restored_tokens
                            + sequence.plan.uncached_suffix_tokens,
                        tokens.len()
                    );
                    drop(sequence);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread panicked");
        }
        cache.check_invariants().expect("cache invariants");
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.tracked_pages, cache.page_count());
        assert_eq!(
            snapshot.uma_pages + snapshot.dram_pages + snapshot.nvme_pages,
            snapshot.tracked_pages as u64,
            "attempt {attempt}: a page is resident in no tier or in two"
        );
        assert_eq!(
            snapshot.store.page_inserts - snapshot.store.page_drops,
            snapshot.tracked_pages as u64,
            "attempt {attempt}: insert/drop accounting drifted"
        );
        assert_eq!(snapshot.cache.requests_total, 8 * 40);
        assert_eq!(snapshot.cache.active_kv_pages as usize, cache.page_count());
        assert_eq!(snapshot.cache.total_kv_pages, 8);
        assert_eq!(snapshot.cache.prompt_tokens_total, {
            let mut expected = 0u64;
            for thread in 0..8u64 {
                for step in 0..40u64 {
                    let seed = thread * 1_000 + step;
                    expected += 64 + 64 * (1 + (seed % 3));
                }
            }
            expected
        });
    }
}

#[test]
fn concurrent_promotion_keeps_every_page_in_exactly_one_tier() {
    // Repeated because the interleaving is scheduling-dependent.
    for attempt in 0..3 {
        let store = Arc::new(parking_lot::Mutex::new(
            TieredPageStore::new(
                Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 2)),
                Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 2)),
                Box::new(InMemoryTier::new(Tier::Nvme, GLOBAL_PAGE_BYTES, 32)),
                Arc::new(KvStoreTelemetry::default()),
            )
            .expect("store"),
        ));
        {
            let protected = HashSet::new();
            let mut guard = store.lock();
            for page in 0..12u64 {
                let mut buffer = vec![0u8; GLOBAL_PAGE_BYTES];
                fill(page as usize, &mut buffer).expect("fill");
                guard
                    .insert(sparkflash_kv::PageId(page), &buffer, &protected)
                    .expect("insert");
            }
            guard.check_invariants().expect("prefill invariants");
            assert!(
                guard.resident(Tier::Nvme) > 0,
                "attempt {attempt}: the prefill must put pages in the terminal tier"
            );
        }
        let mut handles = Vec::new();
        for thread in 0..8u64 {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let protected = HashSet::new();
                for step in 0..60u64 {
                    let page = sparkflash_kv::PageId((thread * 60 + step * 7) % 12);
                    let mut guard = store.lock();
                    match guard.promote(page, &protected) {
                        Ok(_) | Err(KvError::CapacityExceeded { .. }) => {}
                        Err(other) => panic!("unexpected promote error: {other}"),
                    }
                    // Whether the promotion succeeded or was refused, the payload must still be
                    // where the bookkeeping says it is, byte for byte.
                    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
                    guard.read(page, &mut out).expect("read back");
                    let mut expected = vec![0u8; GLOBAL_PAGE_BYTES];
                    fill(page.0 as usize, &mut expected).expect("fill");
                    assert_eq!(out, expected, "page {} changed during promotion", page.0);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread panicked");
        }
        let mut guard = store.lock();
        guard
            .check_invariants()
            .unwrap_or_else(|error| panic!("attempt {attempt}: {error}"));
        for page in guard.tracked_page_ids() {
            guard.drop_page(page).expect("drop");
        }
        guard.check_invariants().expect("drained invariants");
        assert_eq!(guard.tracked_pages(), 0);
    }
}

#[test]
fn concurrent_page_allocation_is_exactly_counted() {
    let telemetry = Arc::new(KvCacheTelemetry::default());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let telemetry = Arc::clone(&telemetry);
        handles.push(std::thread::spawn(move || {
            for _ in 0..1_000 {
                telemetry.note_page_allocated();
            }
        }));
    }
    for handle in handles {
        handle.join().expect("thread panicked");
    }
    assert_eq!(telemetry.snapshot().active_kv_pages, 8_000);
    for _ in 0..500 {
        telemetry.note_page_freed();
    }
    assert_eq!(telemetry.snapshot().active_kv_pages, 7_500);
}

#[test]
fn concurrent_store_traffic_keeps_three_tiers_consistent() {
    let telemetry = Arc::new(KvStoreTelemetry::default());
    let store = Arc::new(parking_lot::Mutex::new(
        TieredPageStore::new(
            Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 4)),
            Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 4)),
            Box::new(InMemoryTier::new(Tier::Nvme, GLOBAL_PAGE_BYTES, 64)),
            Arc::clone(&telemetry),
        )
        .expect("store"),
    ));
    let mut handles = Vec::new();
    for thread in 0..8u64 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            let protected = HashSet::new();
            for step in 0..200u64 {
                let page = sparkflash_kv::PageId(thread * 1_000 + step);
                let mut buffer = vec![0u8; GLOBAL_PAGE_BYTES];
                fill(step as usize, &mut buffer).expect("fill");
                let mut guard = store.lock();
                match guard.insert(page, &buffer, &protected) {
                    Ok(()) => {}
                    Err(KvError::CapacityExceeded { .. }) => {}
                    Err(other) => panic!("unexpected store error: {other}"),
                }
                let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
                guard.read(page, &mut out).expect("read back");
                assert_eq!(out, buffer, "thread {thread} step {step}");
                guard.drop_page(page).expect("drop");
            }
        }));
    }
    for handle in handles {
        handle.join().expect("thread panicked");
    }
    let guard = store.lock();
    guard.check_invariants().expect("store invariants");
    assert_eq!(guard.tracked_pages(), 0);
    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.page_inserts, snapshot.page_drops);
    assert_eq!(snapshot.page_inserts, 8 * 200);
    assert_eq!(
        snapshot.uma_reads + snapshot.dram_reads + snapshot.nvme_reads,
        8 * 200,
        "every read must be attributed to exactly one tier"
    );
    assert_eq!(snapshot.bytes_read, 8 * 200 * GLOBAL_PAGE_BYTES as u64);
}
