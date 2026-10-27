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

//! Capacity, eviction and memory-budget behaviour under pressure.

use sparkflash_kv::layout::{
    DEFAULT_GLOBAL_KV_POOL_BYTES, DEFAULT_SWA_TAIL_SLOTS, GIB, GLOBAL_PAGE_BYTES, MemoryBudget,
    NODE_MEMORY_BUDGET_BYTES, PageId, SWA_TAIL_BYTES,
};
use sparkflash_kv::{KvCache, KvCacheConfig, KvError};

fn config(uma: u64, dram: u64, nvme: u64, swa: usize) -> KvCacheConfig {
    KvCacheConfig {
        budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
        uma_pages: uma,
        dram_pages: dram,
        nvme_pages: nvme,
        swa_tail_slots: swa,
        nvme_path: None,
    }
}

fn fill(block: usize, out: &mut [u8]) -> Result<(), KvError> {
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = ((block * 29 + index * 11) % 249) as u8;
    }
    Ok(())
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| seed.wrapping_add((index as u32).wrapping_mul(40_503)))
        .collect()
}

#[test]
fn the_pool_evicts_the_oldest_prefix_and_keeps_serving() {
    let cache = KvCache::new(config(2, 0, 0, 2)).expect("cache");
    let first = prompt(1, 128);
    let second = prompt(9_999, 128);
    cache.cache_prefix(&first, fill).expect("first prefix");
    assert_eq!(cache.page_count(), 2);
    let outcome = cache.cache_prefix(&second, fill).expect("second prefix");
    assert_eq!(outcome.pages_added, 2);
    assert_eq!(
        outcome.pages_evicted, 2,
        "the older prefix had to make room"
    );
    assert_eq!(cache.page_count(), 2);

    let cold = cache.begin_sequence(&first).expect("plan");
    assert_eq!(cold.plan.global_kv_cached_tokens, 0);
    drop(cold);
    let warm = cache.begin_sequence(&second).expect("plan");
    assert_eq!(warm.plan.global_kv_cached_tokens, 64);
    assert_eq!(warm.plan.uma_hit_tokens, 64);
    assert_eq!(warm.plan.uncached_suffix_tokens, 64);
}

#[test]
fn a_pinned_page_blocks_eviction_and_the_refusal_is_reported() {
    let cache = KvCache::new(config(1, 0, 0, 2)).expect("cache");
    let hot = prompt(5, 65);
    cache.cache_prefix(&hot, fill).expect("cache one page");
    let sequence = cache.begin_sequence(&hot).expect("plan");
    assert_eq!(sequence.plan.global_kv_cached_tokens, 64);

    let other = prompt(777, 64);
    let error = cache
        .cache_prefix(&other, fill)
        .expect_err("nothing is evictable");
    assert!(matches!(error, KvError::CapacityExceeded { .. }));
    assert_eq!(cache.page_count(), 1);
    assert_eq!(cache.evict(usize::MAX).expect("evict"), 0);
    assert_eq!(cache.page_count(), 1);

    drop(sequence);
    assert_eq!(cache.evict(usize::MAX).expect("evict"), 1);
    assert_eq!(cache.page_count(), 0);
}

#[test]
fn capacity_exhaustion_is_reported_rather_than_silently_dropping() {
    let cache = KvCache::new(config(1, 0, 0, 0)).expect("cache");
    let tokens = prompt(3, 192);
    let error = cache
        .cache_prefix(&tokens, fill)
        .expect_err("only one page can fit");
    assert!(matches!(error, KvError::CapacityExceeded { .. }));
    assert_eq!(cache.page_count(), 1);
    assert!(cache.store_telemetry().snapshot().capacity_refusals >= 1);
}

#[test]
fn a_sequence_keeps_its_pages_readable_while_it_is_alive() {
    let cache = KvCache::new(config(2, 2, 8, 2)).expect("cache");
    let tokens = prompt(11, 256);
    cache.cache_prefix(&tokens, fill).expect("cache");
    let sequence = cache.begin_sequence(&tokens).expect("plan");
    assert!(sequence.plan.global_kv_cached_tokens >= 128);
    for block in 0..sequence.path().len() {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        cache
            .read_page(PageId(block as u64), &mut out)
            .expect("a pinned page must stay readable");
        let mut reference = vec![0u8; GLOBAL_PAGE_BYTES];
        fill(block, &mut reference).expect("fill");
        assert_eq!(out, reference, "block {block}");
    }
}

#[test]
fn the_memory_budget_rejects_an_oversized_configuration() {
    let budget = MemoryBudget::new(1 << 20, 1 << 19).expect("budget");
    let oversized = KvCacheConfig {
        budget,
        uma_pages: 100,
        dram_pages: 0,
        nvme_pages: 0,
        swa_tail_slots: 0,
        nvme_path: None,
    };
    let error = KvCache::new(oversized).expect_err("over budget");
    assert!(matches!(error, KvError::BudgetExceeded { .. }));
    let exact = KvCacheConfig {
        budget,
        uma_pages: (1 << 19) / GLOBAL_PAGE_BYTES as u64,
        dram_pages: 0,
        nvme_pages: 0,
        swa_tail_slots: 0,
        nvme_path: None,
    };
    assert!(KvCache::new(exact).is_ok());
}

#[test]
fn a_floor_larger_than_the_node_budget_is_rejected() {
    assert!(MemoryBudget::new(1 << 20, 1 << 21).is_err());
    assert!(MemoryBudget::new(1 << 20, 1 << 20).is_ok());
}

#[test]
fn the_default_configuration_fits_the_node_and_matches_the_specification_rows() {
    let default = KvCacheConfig::default();
    default.validate().expect("default config fits");
    assert_eq!(default.swa_tail_slots, DEFAULT_SWA_TAIL_SLOTS);
    assert_eq!(default.swa_tail_slots, 128);
    assert_eq!(
        default.uma_pages * GLOBAL_PAGE_BYTES as u64 / (1 << 20),
        DEFAULT_GLOBAL_KV_POOL_BYTES / (1 << 20)
    );
    // The whole reservation must leave the documented floor untouched.
    let reserved = default.budget.floor_bytes + default.required_bytes();
    assert!(reserved <= NODE_MEMORY_BUDGET_BYTES);
    let headroom = NODE_MEMORY_BUDGET_BYTES - reserved;
    assert!(headroom > 0);
    // 121.7 GiB ceiling at 890 B/token is the specification's global KV bound.
    let pool_gib = default.uma_pages * GLOBAL_PAGE_BYTES as u64 / GIB;
    assert_eq!(pool_gib, 13, "the 13.4 GiB pool rounds to 13 GiB");
    let tokens = default.uma_pages * 64;
    assert!(
        tokens > 16_000_000 && tokens < 16_300_000,
        "pool holds {tokens} tokens"
    );
}

#[test]
fn the_swa_ring_reservation_is_128_tails() {
    let default = KvCacheConfig::default();
    assert_eq!(
        default.swa_tail_slots as u64 * SWA_TAIL_BYTES as u64,
        346_030_080
    );
    let cache = KvCache::new(default).expect("default cache builds");
    assert_eq!(cache.swa_capacity(), 128);
    assert_eq!(cache.resident_pages(sparkflash_kv::Tier::Uma), 0);
    assert_eq!(cache.page_count(), 0);
}

#[test]
fn the_page_counters_follow_the_pool_and_the_tree() {
    let cache = KvCache::new(config(4, 2, 2, 2)).expect("cache");
    assert_eq!(cache.snapshot().cache.total_kv_pages, 4);
    assert_eq!(cache.snapshot().cache.active_kv_pages, 0);
    let tokens = prompt(21, 256);
    cache.cache_prefix(&tokens, fill).expect("cache");
    let snapshot = cache.snapshot();
    assert_eq!(snapshot.cache.active_kv_pages, 4);
    assert_eq!(snapshot.cache.total_kv_pages, 4);
    assert_eq!(
        snapshot.uma_pages + snapshot.dram_pages + snapshot.nvme_pages,
        4
    );
    assert_eq!(cache.evict(2).expect("evict"), 2);
    let snapshot = cache.snapshot();
    assert_eq!(snapshot.cache.active_kv_pages, 2);
    assert_eq!(snapshot.store.page_drops, 2);
}
