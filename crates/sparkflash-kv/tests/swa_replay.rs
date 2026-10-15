//! End-to-end SWA tail probing, bounded replay accounting and the global-KV invariant.

use std::collections::HashSet;

use sparkflash_kv::config::{KV_PAGE_TOKENS, SWA_WINDOW_SIZE};
use sparkflash_kv::layout::{
    GLOBAL_PAGE_BYTES, MemoryBudget, PageId, SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL,
};
use sparkflash_kv::swa::{ReplayKind, SwaProducer};
use sparkflash_kv::tier::{InMemoryTier, Tier, TieredPageStore};
use sparkflash_kv::{KvCache, KvCacheConfig, KvError, KvStoreTelemetry};

fn small_cache(uma: u64, dram: u64, nvme: u64, swa_slots: usize) -> KvCache {
    let config = KvCacheConfig {
        budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
        uma_pages: uma,
        dram_pages: dram,
        nvme_pages: nvme,
        swa_tail_slots: swa_slots,
        nvme_path: None,
    };
    KvCache::new(config).expect("cache")
}

/// Fills one page deterministically from its block index.
fn page_fill(block_index: usize, out: &mut [u8]) -> Result<(), KvError> {
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = ((block_index * 17 + index * 5) % 253) as u8;
    }
    Ok(())
}

fn prompt(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(12_345)
                % 100_003
        })
        .collect()
}

/// A weight-free producer whose output is checkable from the layer, the token and the byte.
struct DeterministicProducer;

impl SwaProducer for DeterministicProducer {
    fn fill_layer(&mut self, layer: usize, window: &[u32], out: &mut [u8]) -> Result<(), KvError> {
        assert!(out.len() >= window.len() * SWA_SLOT_BYTES);
        for (slot, token) in window.iter().enumerate() {
            let base = slot * SWA_SLOT_BYTES;
            for offset in 0..SWA_SLOT_BYTES {
                out[base + offset] = ((layer * 31 + (*token as usize) * 7 + offset) % 251) as u8;
            }
        }
        Ok(())
    }
}

fn expected_byte(layer: usize, token: u32, offset: usize) -> u8 {
    ((layer * 31 + (token as usize) * 7 + offset) % 251) as u8
}

#[test]
fn a_cold_prompt_plans_a_full_encoder_run_and_a_bounded_decoder_replay() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    let sequence = cache.begin_sequence(&tokens).expect("plan");
    assert_eq!(sequence.plan.global_kv_cached_tokens, 0);
    assert_eq!(sequence.plan.uma_hit_tokens, 0);
    assert_eq!(sequence.plan.nvme_restored_tokens, 0);
    assert_eq!(sequence.plan.uncached_suffix_tokens, 320);
    assert!(!sequence.plan.swa_tail_hit);
    assert_eq!(sequence.plan.encoder_swa_replay_tokens, 0);
    assert_eq!(
        sequence.plan.decoder_swa_replay_tokens,
        SWA_WINDOW_SIZE.min(320)
    );
}

#[test]
fn a_freshly_cached_prefix_plans_the_two_bounded_replays() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    let outcome = cache.cache_prefix(&tokens, page_fill).expect("cache");
    assert_eq!(outcome.pages_added, 5);
    assert_eq!(outcome.cached_tokens, 320);

    let sequence = cache.begin_sequence(&tokens).expect("plan");
    // plan_request clamps the raw match to total - 1 before aligning down.
    assert_eq!(sequence.plan.global_kv_cached_tokens, 256);
    assert_eq!(sequence.plan.uma_hit_tokens, 256);
    assert_eq!(sequence.plan.nvme_restored_tokens, 0);
    assert_eq!(sequence.plan.uncached_suffix_tokens, 64);
    assert!(!sequence.plan.swa_tail_hit);
    assert_eq!(sequence.plan.encoder_swa_replay_tokens, 128);
    assert_eq!(sequence.plan.decoder_swa_replay_tokens, 128);
    // Token conservation, the property the whole plan rests on.
    assert_eq!(
        sequence.plan.uma_hit_tokens
            + sequence.plan.nvme_restored_tokens
            + sequence.plan.uncached_suffix_tokens,
        sequence.plan.total_prompt_tokens
    );
    // And the SWA window never exceeds n_win.
    assert!(sequence.plan.encoder_swa_replay_tokens <= SWA_WINDOW_SIZE);
    assert!(sequence.plan.decoder_swa_replay_tokens <= SWA_WINDOW_SIZE);
}

#[test]
fn cached_pages_that_are_read_back_from_a_lower_tier_count_as_restored_tokens() {
    // One page per tier: three cached blocks land on all three tiers, so the plan has to
    // split the hit across the two counters.
    let cache = small_cache(1, 1, 8, 4);
    let tokens = prompt(192);
    let outcome = cache.cache_prefix(&tokens, page_fill).expect("cache");
    assert_eq!(outcome.pages_added, 3);
    let before = cache.snapshot();
    assert_eq!(before.tracked_pages, 3);
    assert_eq!(before.uma_pages, 1);
    assert_eq!(before.dram_pages, 1);
    assert_eq!(before.nvme_pages, 1, "the oldest page cascaded to NVMe");

    let sequence = cache.begin_sequence(&tokens).expect("plan");
    assert_eq!(sequence.plan.global_kv_cached_tokens, 128);
    assert_eq!(
        sequence.plan.uma_hit_tokens + sequence.plan.nvme_restored_tokens,
        128
    );
    assert_eq!(
        sequence.plan.nvme_restored_tokens, 128,
        "both blocks came from Tier 2"
    );
    assert_eq!(sequence.plan.uma_hit_tokens, 0);
    assert_eq!(sequence.plan.uncached_suffix_tokens, 64);
    let recorded = cache.telemetry().snapshot();
    assert_eq!(recorded.global_kv_nvme_hit_tokens, 128);
    assert_eq!(recorded.global_kv_uma_hit_tokens, 0);
    assert_eq!(recorded.global_kv_miss_tokens, 64);
    cache.check_invariants().expect("cache invariants");
}

#[test]
fn pages_can_be_read_back_byte_for_byte_through_the_cache() {
    let cache = small_cache(1, 2, 8, 2);
    let tokens = prompt(256);
    cache.cache_prefix(&tokens, page_fill).expect("cache");
    for block in 0..4usize {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        let tier = cache
            .read_page(PageId(block as u64), &mut out)
            .expect("read");
        let expected = match block {
            0 => Tier::Nvme,
            1 | 2 => Tier::Dram,
            _ => Tier::Uma,
        };
        assert_eq!(tier, expected, "block {block} tier");
        let mut reference = vec![0u8; GLOBAL_PAGE_BYTES];
        page_fill(block, &mut reference).expect("fill");
        assert_eq!(out, reference, "block {block} payload");
    }
}

#[test]
fn a_bounded_replay_never_rewrites_a_cached_global_kv_page() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    cache.cache_prefix(&tokens, page_fill).expect("cache");
    let fingerprint_before = cache.resident_page_fingerprint().expect("fingerprint");
    let inserts_before = cache.store_telemetry().snapshot().page_inserts;

    let mut producer = DeterministicProducer;
    let encoder = cache
        .bounded_replay(ReplayKind::Encoder, &tokens[..256], 256, &mut producer)
        .expect("encoder replay");
    let decoder = cache
        .bounded_replay(ReplayKind::Decoder, &tokens, 320, &mut producer)
        .expect("decoder replay");

    assert_eq!(encoder.replayed_tokens, SWA_WINDOW_SIZE);
    assert_eq!(encoder.swa_slots_written, 20 * SWA_SLOTS_PER_TAIL as u64);
    assert_eq!(decoder.replayed_tokens, SWA_WINDOW_SIZE);
    assert_eq!(decoder.swa_slots_written, 20 * SWA_SLOTS_PER_TAIL as u64);
    assert!(encoder.global_kv_untouched());
    assert!(decoder.global_kv_untouched());
    assert_eq!(encoder.global_kv_pages_written, 0);
    assert_eq!(decoder.global_kv_pages_written, 0);

    let fingerprint_after = cache.resident_page_fingerprint().expect("fingerprint");
    assert_eq!(fingerprint_before, fingerprint_after, "a page byte changed");
    let snapshot = cache.store_telemetry().snapshot();
    assert_eq!(snapshot.page_inserts, inserts_before, "a page was written");
    assert_eq!(snapshot.replay_global_kv_writes, 0);
    // The zero above is only meaningful because the scope was really armed: a counter that is
    // never set cannot move, and this asserts one scope per replay call.
    assert_eq!(
        snapshot.replay_scopes, 2,
        "each bounded replay must arm the store's replay scope"
    );
    assert_eq!(
        snapshot.replay_swa_slots_written,
        2 * 20 * SWA_SLOTS_PER_TAIL as u64
    );

    // And every cached page still decodes to exactly the filler's bytes.
    for block in 0..5usize {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        cache
            .read_page(PageId(block as u64), &mut out)
            .expect("read");
        let mut reference = vec![0u8; GLOBAL_PAGE_BYTES];
        page_fill(block, &mut reference).expect("fill");
        assert_eq!(out, reference, "block {block} after replay");
    }
}

#[test]
fn one_replay_half_does_not_make_a_tail_servable() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    cache.cache_prefix(&tokens, page_fill).expect("cache");
    let window = tokens[128..256].to_vec();
    let mut producer = DeterministicProducer;
    let outcome = cache
        .bounded_replay(ReplayKind::Encoder, &window, 256, &mut producer)
        .expect("encoder replay");
    assert!(!outcome.tail_completed);
    assert_eq!(cache.probe_swa_tail(&tokens[..256]), None);
    assert!(
        !cache
            .begin_sequence(&tokens)
            .expect("plan")
            .plan
            .swa_tail_hit
    );

    // The decoder half of the same window completes it.
    let outcome = cache
        .bounded_replay(ReplayKind::Decoder, &window, 256, &mut producer)
        .expect("decoder replay");
    assert!(outcome.tail_completed);
    assert_eq!(cache.snapshot().swa_complete_tails, 1);
}

#[test]
fn a_complete_tail_turns_the_encoder_replay_off_and_serves_its_slots() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    cache.cache_prefix(&tokens, page_fill).expect("cache");
    let window = tokens[128..256].to_vec();
    let mut producer = DeterministicProducer;
    let outcome = cache
        .cache_swa_tail(&window, 256, &mut producer)
        .expect("prefill tail");
    assert_eq!(
        outcome.swa_slots_written,
        SWA_LAYERS as u64 * SWA_SLOTS_PER_TAIL as u64
    );
    assert!(outcome.tail_completed);

    let tail = cache.probe_swa_tail(&tokens[..256]).expect("tail resident");
    let sequence = cache.begin_sequence(&tokens).expect("plan");
    assert!(sequence.plan.swa_tail_hit);
    assert_eq!(sequence.plan.encoder_swa_replay_tokens, 0);
    assert_eq!(sequence.plan.decoder_swa_replay_tokens, 128);
    assert_eq!(cache.telemetry().snapshot().swa_tail_cache_hits, 1);

    // Serve a slot and check it is the producer's bytes, not a placeholder.
    for layer in [0usize, 19, 20, 39] {
        for slot in [0usize, 1, 127] {
            let mut out = vec![0u8; SWA_SLOT_BYTES];
            cache
                .read_swa_slot(tail, layer, slot, &mut out)
                .expect("slot");
            let token = window[slot];
            for (offset, byte) in out.iter().enumerate() {
                assert_eq!(
                    *byte,
                    expected_byte(layer, token, offset),
                    "layer {layer} slot {slot} byte {offset}"
                );
            }
        }
    }
}

#[test]
fn a_short_window_is_replayed_but_is_not_cacheable_as_a_tail() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    let mut producer = DeterministicProducer;
    let outcome = cache
        .bounded_replay(ReplayKind::Decoder, &tokens[..64], 64, &mut producer)
        .expect("short replay");
    assert_eq!(outcome.replayed_tokens, 64);
    assert_eq!(outcome.swa_slots_written, 20 * 64);
    assert!(!outcome.tail_completed);
    assert_eq!(cache.snapshot().swa_tails, 0);
}

#[test]
fn an_empty_window_replays_nothing() {
    let cache = small_cache(8, 8, 8, 4);
    let mut producer = DeterministicProducer;
    let outcome = cache
        .bounded_replay(ReplayKind::Encoder, &[], 0, &mut producer)
        .expect("empty replay");
    assert_eq!(outcome, sparkflash_kv::ReplayOutcome::default());
    assert_eq!(
        cache.store_telemetry().snapshot().replay_swa_slots_written,
        0
    );
}

#[test]
fn the_tail_ring_evicts_least_recently_used_tails_under_pressure() {
    let cache = small_cache(8, 8, 8, 2);
    let mut producer = DeterministicProducer;
    let first = prompt(200);
    let second: Vec<u32> = first.iter().map(|token| token.wrapping_add(1)).collect();
    let third: Vec<u32> = first.iter().map(|token| token.wrapping_add(2)).collect();
    cache
        .cache_swa_tail(&first[72..200], 200, &mut producer)
        .expect("first");
    cache
        .cache_swa_tail(&second[72..200], 200, &mut producer)
        .expect("second");
    cache
        .cache_swa_tail(&third[72..200], 200, &mut producer)
        .expect("third");
    assert_eq!(cache.snapshot().swa_tails, 2);
    assert_eq!(
        cache.probe_swa_tail(&first[..200]),
        None,
        "LRU should be gone"
    );
    assert!(cache.probe_swa_tail(&third[..200]).is_some());
    assert_eq!(cache.swa_capacity(), 2);
}

#[test]
fn a_tail_must_match_the_window_exactly() {
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    let mut producer = DeterministicProducer;
    let window = tokens[128..256].to_vec();
    cache
        .cache_swa_tail(&window, 256, &mut producer)
        .expect("tail");
    assert!(cache.probe_swa_tail(&tokens[..256]).is_some());
    let mut changed = window.clone();
    changed[0] = changed[0].wrapping_add(9_999);
    assert_eq!(cache.probe_swa_tail(&changed), None);
    // A prefix that is not exactly the same window ending position also misses.
    assert_eq!(cache.probe_swa_tail(&tokens[..192]), None);
}

#[test]
fn the_plan_counters_agree_with_the_returned_plan_across_many_requests() {
    let cache = small_cache(4, 4, 32, 4);
    let mut expected_requests = 0u64;
    for round in 0..40usize {
        let tokens = prompt(128 + round * 3);
        cache.cache_prefix(&tokens, page_fill).expect("cache");
        let sequence = cache.begin_sequence(&tokens).expect("plan");
        let plan = sequence.plan.clone();
        assert_eq!(
            plan.uma_hit_tokens + plan.nvme_restored_tokens + plan.uncached_suffix_tokens,
            plan.total_prompt_tokens
        );
        assert_eq!(
            plan.uma_hit_tokens + plan.nvme_restored_tokens,
            plan.global_kv_cached_tokens
        );
        assert_eq!(plan.global_kv_cached_tokens % KV_PAGE_TOKENS, 0);
        assert!(plan.global_kv_cached_tokens < plan.total_prompt_tokens);
        expected_requests += 1;
    }
    let snapshot = cache.telemetry().snapshot();
    assert_eq!(snapshot.requests_total, expected_requests);
    cache.check_invariants().expect("cache invariants");
}

#[test]
fn the_sequence_guard_releases_its_pins_even_when_the_caller_forgets() {
    let cache = small_cache(2, 2, 8, 2);
    let tokens = prompt(256);
    cache.cache_prefix(&tokens, page_fill).expect("cache");
    {
        let sequence = cache.begin_sequence(&tokens).expect("plan");
        assert_eq!(sequence.path().len(), 3);
        assert_eq!(sequence.plan.global_kv_cached_tokens, 192);
        assert_eq!(
            sequence.plan.uma_hit_tokens + sequence.plan.nvme_restored_tokens,
            192
        );
        // Two of the three pages had been displaced below UMA, so they are restores.
        assert!(sequence.plan.nvme_restored_tokens > 0);
    }
    cache.check_invariants().expect("cache invariants");
    // Every page is now evictable, so a full drain succeeds.
    assert_eq!(cache.evict(usize::MAX).expect("evict"), 4);
    assert_eq!(cache.page_count(), 0);
    cache.check_invariants().expect("drained invariants");
}

#[test]
fn a_promotion_keeps_its_payload_byte_exact_and_releases_its_staging_copies() {
    // Regression for the duplicate-copy bug: a Tier-2 restore stages through DRAM, and leaving
    // either the NVMe source or the DRAM staging copy behind would keep the page resident in
    // two tiers at once. The cascading-keep-set semantics this test used to be named for is
    // pinned at the helper itself, in tier.rs::tests::a_cascade_keeps_every_page_the_caller_is_moving_not_just_the_first_victim,
    // because no promote path places a resident page in a victim-selecting tier any more.
    let telemetry = std::sync::Arc::new(KvStoreTelemetry::default());
    let page_bytes = GLOBAL_PAGE_BYTES;
    let mut store = TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, page_bytes, 1)),
        Box::new(InMemoryTier::new(Tier::Dram, page_bytes, 1)),
        Box::new(InMemoryTier::new(Tier::Nvme, page_bytes, 4)),
        telemetry,
    )
    .expect("store");
    let protected = HashSet::new();
    for page in 0..3u64 {
        let mut buffer = vec![0u8; page_bytes];
        page_fill(page as usize, &mut buffer).expect("fill");
        store
            .insert(PageId(page), &buffer, &protected)
            .expect("insert");
    }
    // Page 2 is in UMA, page 1 in DRAM, page 0 in NVMe. Promoting page 0 has to stage
    // through a full DRAM, whose own demotion cascade would otherwise choose page 0.
    let from = store.promote(PageId(0), &protected).expect("promote");
    assert_eq!(from, Tier::Nvme);
    assert_eq!(store.location(PageId(0)).map(|l| l.tier), Some(Tier::Uma));
    let mut out = vec![0u8; page_bytes];
    store.read(PageId(0), &mut out).expect("read");
    let mut reference = vec![0u8; page_bytes];
    page_fill(0, &mut reference).expect("fill");
    assert_eq!(out, reference, "the promoted payload changed in flight");
    // The store-level version of the same invariant: the promoted page's NVMe source copy is
    // gone, so the terminal tier holds only the page the cascade displaced, not two copies.
    assert_eq!(
        store.resident(Tier::Nvme),
        1,
        "only the displaced page may remain in the terminal tier"
    );
    assert_eq!(store.tracked_pages(), 3);
    store.check_invariants().expect("store invariants");
}

/// A producer that panics part-way through a replay, to prove the scope is released.
struct PanickingProducer {
    /// Layer the panic happens on.
    panic_at: usize,
}

impl SwaProducer for PanickingProducer {
    fn fill_layer(
        &mut self,
        layer: usize,
        _window: &[u32],
        _out: &mut [u8],
    ) -> Result<(), KvError> {
        assert_ne!(layer, self.panic_at, "the producer exploded");
        Ok(())
    }
}

#[test]
fn a_panicking_producer_leaves_no_replay_scope_behind() {
    // The scope is released only by dropping its guard, so a panic inside the producer window
    // must leave the cache able to publish again. Without the guard, one panicking replay used
    // to leave this thread marked as replaying for the life of the cache: every later publish
    // failed with ReplayWroteGlobalKv and page_inserts stayed at zero.
    let cache = small_cache(8, 8, 8, 4);
    let tokens = prompt(320);
    let mut producer = PanickingProducer { panic_at: 3 };
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.bounded_replay(ReplayKind::Encoder, &tokens, 320, &mut producer)
    }));
    assert!(panicked.is_err(), "the producer panic must propagate");
    assert_eq!(
        cache.store_telemetry().snapshot().replay_scopes,
        1,
        "the replay did arm a scope before it panicked"
    );
    // The scope has to be gone, so an ordinary publish works and is counted as an insert.
    let outcome = cache
        .cache_prefix(&tokens, page_fill)
        .expect("a publish after a panicking replay must not be refused");
    assert_eq!(outcome.pages_added, 5);
    let snapshot = cache.store_telemetry().snapshot();
    assert_eq!(
        snapshot.replay_global_kv_writes, 0,
        "the panic must not be counted as a replay writing global KV"
    );
    assert_eq!(snapshot.page_inserts, 5);
    cache.check_invariants().expect("cache invariants");
}

#[test]
fn a_full_tail_replay_agrees_with_its_two_halves() {
    // cache_swa_tail runs the encoder half and then the decoder half, and what it returns must
    // be the merge of those two outcomes rather than a fresh literal. Replaying the halves
    // separately on an identical cache gives the numbers to compare against. Both halves report
    // zero here, because a producer has no way to reach the store, so this is a shape check:
    // the falsifier for the sum is the in-crate test that pokes the same guard with non-zero
    // counts (cache::tests::a_full_tail_outcome_sums_a_non_zero_counter_from_both_halves).
    let tokens = prompt(320);
    let full = small_cache(8, 8, 8, 4);
    let mut producer = DeterministicProducer;
    let combined = full
        .cache_swa_tail(&tokens, 320, &mut producer)
        .expect("full tail");

    let halves = small_cache(8, 8, 8, 4);
    let mut producer = DeterministicProducer;
    let encoder = halves
        .bounded_replay(ReplayKind::Encoder, &tokens, 320, &mut producer)
        .expect("encoder half");
    let decoder = halves
        .bounded_replay(ReplayKind::Decoder, &tokens, 320, &mut producer)
        .expect("decoder half");

    assert_eq!(combined.replayed_tokens, encoder.replayed_tokens);
    assert_eq!(
        combined.swa_slots_written,
        encoder.swa_slots_written + decoder.swa_slots_written
    );
    assert_eq!(
        combined.global_kv_pages_written,
        encoder.global_kv_pages_written + decoder.global_kv_pages_written
    );
    assert_eq!(combined.tail_completed, decoder.tail_completed);
    assert!(combined.tail_completed);
    assert_eq!(
        full.store_telemetry().snapshot().replay_scopes,
        2,
        "a full tail is two bounded replays"
    );
}
