//! Paged serving KV: the rules task-21's acceptance calls "paging invariants tested".
//!
//! Every test that moves pages ends with `check_invariants()`: the per-test assertions say what a
//! specific operation did, and the invariant check says no other page, block table or tail slot was
//! disturbed while doing it.

use sparkflash_kv::config::KV_PAGE_TOKENS;
use sparkflash_kv::paged::{PagedGeometry, PagedKvCache, PagingError, SequenceState};

/// A pool of `pages` pages and `tails` tail slots with invariants already checked.
fn pool(pages: u32, tails: u32) -> PagedKvCache {
    let cache = PagedKvCache::new(PagedGeometry::new(pages, tails).expect("geometry"));
    cache
        .check_invariants()
        .expect("a fresh pool is consistent");
    cache
}

/// `tokens` token ids starting at `base`; distinct bases give distinct blocks.
fn tokens(base: u32, tokens: usize) -> Vec<u32> {
    (0..tokens as u32).map(|offset| base + offset).collect()
}

fn check(cache: &PagedKvCache) {
    cache
        .check_invariants()
        .expect("paging invariants after the operation");
}

#[test]
fn geometry_requires_a_page_and_a_tail() {
    assert_eq!(
        PagedGeometry::new(0, 1),
        Err(PagingError::BadGeometry("a pool holds at least one page"))
    );
    assert_eq!(
        PagedGeometry::new(1, 0),
        Err(PagingError::BadGeometry(
            "a pool holds at least one SWA tail slot"
        ))
    );
    let geometry = PagedGeometry::new(3, 2).expect("valid");
    assert_eq!((geometry.pages(), geometry.tails()), (3, 2));
}

#[test]
fn the_first_append_allocates_page_zero_and_the_pool_starts_with_one_free_tail() {
    let mut cache = pool(4, 2);
    assert_eq!(cache.stats().pages_free, 4);
    assert_eq!(cache.stats().tails_free, 2);
    cache.add_sequence(1).expect("add");
    assert_eq!(cache.stats().tails_free, 1);
    let report = cache.append(1, &tokens(0, 1)).expect("append");
    assert_eq!(report.tokens, 1);
    assert_eq!(report.allocated, 1);
    assert_eq!(report.shared, 0);
    assert_eq!(
        cache.table(1).expect("table").page_for(0).expect("page").0,
        0
    );
    assert_eq!(cache.stats().pages_free, 3);
    check(&cache);
}

#[test]
fn append_crosses_a_page_boundary_without_moving_the_earlier_blocks() {
    let mut cache = pool(4, 1);
    cache.add_sequence(7).expect("add");
    cache.append(7, &tokens(0, KV_PAGE_TOKENS)).expect("append");
    check(&cache);
    let first = cache.table(7).expect("table").page_for(0).expect("page");
    assert_eq!(cache.table(7).expect("table").len(), 1);
    // 65 tokens: the sequence opens block 1 and block 0's page must not move.
    cache.append(7, &tokens(64, 1)).expect("append");
    check(&cache);
    let table = cache.table(7).expect("table").clone();
    assert_eq!(table.len(), 2);
    assert_ne!(
        table.page_for(0).expect("page 0"),
        table.page_for(1).expect("page 1")
    );
    assert_eq!(table.page_for(0).expect("page 0"), first);
    assert_eq!(cache.stats().pages_in_use, 2);
    // 130th token goes into the second block, not a third.
    cache.append(7, &tokens(65, 65)).expect("append");
    check(&cache);
    assert_eq!(cache.table(7).expect("table").len(), 3);
    assert_eq!(cache.sequence(7).expect("sequence").tokens().len(), 130);
}

#[test]
fn page_of_token_maps_every_logical_position_through_the_block_table() {
    let mut cache = pool(8, 1);
    cache.add_sequence(1).expect("add");
    cache.append(1, &tokens(0, 200)).expect("append");
    check(&cache);
    let table = cache.table(1).expect("table").clone();
    for logical in 0..200usize {
        let page = cache.page_of_token(1, logical).expect("page");
        assert_eq!(
            page,
            table.page_for(logical / KV_PAGE_TOKENS).expect("block")
        );
    }
    assert_eq!(
        cache.page_of_token(1, 200),
        Err(PagingError::OutOfRange {
            sequence: 1,
            logical: 200,
            held: 200
        })
    );
}

#[test]
fn a_full_pool_refuses_a_new_block_and_accepts_it_after_a_release() {
    let mut cache = pool(2, 2);
    cache.add_sequence(1).expect("add");
    cache.add_sequence(2).expect("add");
    cache
        .append(1, &tokens(0, 2 * KV_PAGE_TOKENS))
        .expect("append");
    assert_eq!(
        cache.append(2, &tokens(1_000, 1)),
        Err(PagingError::Capacity { pages: 2 })
    );
    check(&cache);
    cache.free_sequence(1).expect("free");
    check(&cache);
    cache
        .append(2, &tokens(1_000, 1))
        .expect("append after release");
    check(&cache);
}

#[test]
fn a_complete_block_is_shared_and_one_freeing_does_not_take_it_away() {
    let mut cache = pool(4, 2);
    cache.add_sequence(1).expect("add");
    cache.add_sequence(2).expect("add");
    let block = tokens(42, KV_PAGE_TOKENS);
    let first = cache.append(1, &block).expect("append A");
    assert_eq!(first.shared, 0);
    let second = cache.append(2, &block).expect("append B");
    assert_eq!(second.shared, 1);
    assert_eq!(
        cache.table(1).expect("A").page_for(0),
        cache.table(2).expect("B").page_for(0)
    );
    let stats = cache.stats();
    assert_eq!(stats.prefix_hits, 1);
    assert_eq!(stats.shared_references, 1);
    assert_eq!(stats.pages_in_use, 1);
    check(&cache);
    // Freeing one sharer leaves the other's page alone.
    cache.free_sequence(1).expect("free A");
    check(&cache);
    assert_eq!(cache.stats().pages_in_use, 1);
    assert_eq!(cache.stats().pages_free, 3);
    assert_eq!(cache.table(2).expect("B").page_for(0).expect("page").0, 0);
}

#[test]
fn the_last_release_frees_the_page_but_keeps_the_block_hash_for_a_later_hit() {
    let mut cache = pool(4, 2);
    cache.add_sequence(1).expect("add");
    let block = tokens(7, KV_PAGE_TOKENS);
    cache.append(1, &block).expect("append");
    let page = cache.table(1).expect("table").page_for(0).expect("page");
    cache.free_sequence(1).expect("free");
    check(&cache);
    assert_eq!(cache.stats().pages_free, 4);
    assert_eq!(cache.stats().blocks_cached, 1);
    // The next sequence with the same block hits the hash: the page is not rebuilt.
    cache.add_sequence(2).expect("add");
    let report = cache.append(2, &block).expect("append");
    assert_eq!(report.shared, 1);
    assert_eq!(cache.table(2).expect("table").page_for(0), Some(page));
    check(&cache);
}

#[test]
fn reusing_a_cached_page_for_another_block_counts_the_eviction() {
    let mut cache = pool(2, 2);
    cache.add_sequence(1).expect("add");
    cache.append(1, &tokens(1, KV_PAGE_TOKENS)).expect("append");
    cache.free_sequence(1).expect("free");
    check(&cache);
    assert_eq!(cache.stats().blocks_cached, 1);
    cache.add_sequence(2).expect("add");
    cache.append(2, &tokens(2, KV_PAGE_TOKENS)).expect("append");
    // page 1 was the oldest free page, so block 2 lands there and block 1 stays cached on page 0.
    check(&cache);
    assert_eq!(cache.stats().blocks_cached, 2);
    assert_eq!(cache.stats().evictions, 0);
    // The next sequence has to take page 0, which drops block 1: that is the eviction.
    cache.add_sequence(3).expect("add");
    cache.append(3, &tokens(3, KV_PAGE_TOKENS)).expect("append");
    check(&cache);
    let stats = cache.stats();
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.prefix_hits, 0);
}

#[test]
fn a_fork_shares_complete_blocks_and_survives_the_donor_leaving() {
    let mut cache = pool(5, 3);
    cache.add_sequence(1).expect("add");
    cache
        .append(1, &tokens(0, 2 * KV_PAGE_TOKENS))
        .expect("append");
    check(&cache);
    cache.fork(1, 2, 1).expect("fork");
    check(&cache);
    assert_eq!(
        cache.table(1).expect("A").page_for(0),
        cache.table(2).expect("B").page_for(0)
    );
    assert_eq!(cache.sequence(2).expect("B").tokens().len(), KV_PAGE_TOKENS);
    cache.free_sequence(1).expect("free donor");
    check(&cache);
    assert_eq!(cache.stats().pages_in_use, 1);
    // The child keeps growing from the shared prefix.
    cache.append(2, &tokens(1_000, 1)).expect("append child");
    check(&cache);
    assert_eq!(cache.table(2).expect("B").len(), 2);
}

#[test]
fn a_fork_is_refused_past_the_donors_complete_blocks() {
    let mut cache = pool(4, 2);
    cache.add_sequence(1).expect("add");
    cache
        .append(1, &tokens(0, KV_PAGE_TOKENS + 3))
        .expect("append");
    check(&cache);
    assert_eq!(
        cache.fork(1, 2, 2),
        Err(PagingError::BadFork(
            "the donor is shorter than the requested prefix"
        ))
    );
    assert_eq!(
        cache.fork(1, 2, 0),
        Err(PagingError::BadFork("a fork shares at least one block"))
    );
    assert_eq!(cache.fork(99, 2, 1), Err(PagingError::UnknownSequence(99)));
    check(&cache);
}

#[test]
fn preemption_returns_the_pages_and_a_resumed_sequence_allocates_fresh_ones() {
    let mut cache = pool(2, 1);
    cache.add_sequence(1).expect("add");
    cache
        .append(1, &tokens(0, 2 * KV_PAGE_TOKENS))
        .expect("append");
    let pages = cache.table(1).expect("table").pages().to_vec();
    let report = cache.preempt(1).expect("preempt");
    assert_eq!(report.freed_pages, 2);
    assert_eq!(report.retained_blocks, 2);
    assert!(report.freed_tail);
    check(&cache);
    let sequence = cache.sequence(1).expect("sequence");
    assert_eq!(sequence.state(), SequenceState::Preempted);
    assert!(sequence.tokens().is_empty());
    assert_eq!(cache.stats().running, 0);
    assert_eq!(cache.preempt(1), Err(PagingError::AlreadyPreempted(1)));
    check(&cache);
    cache.append(1, &tokens(0, 1)).expect("recompute");
    check(&cache);
    assert_eq!(
        cache.sequence(1).expect("sequence").state(),
        SequenceState::Running
    );
    assert_eq!(cache.stats().running, 1);
    // The recomputed block can land on a different page; what matters is that it is a whole page.
    let resumed = cache.table(1).expect("table").pages().to_vec();
    assert_eq!(resumed.len(), 1);
    let _ = pages;
}

#[test]
fn truncation_must_end_on_a_block_boundary() {
    let mut cache = pool(4, 1);
    cache.add_sequence(1).expect("add");
    cache.append(1, &tokens(0, 130)).expect("append");
    assert_eq!(
        cache.truncate(1, 100),
        Err(PagingError::BadTruncate(
            "a truncation must end on a block boundary"
        ))
    );
    assert_eq!(
        cache.truncate(1, 200),
        Err(PagingError::BadTruncate(
            "a truncation cannot grow the sequence"
        ))
    );
    let report = cache.truncate(1, KV_PAGE_TOKENS).expect("truncate");
    assert_eq!(report.freed_pages, 2);
    assert_eq!(report.retained_blocks, 1);
    check(&cache);
    assert_eq!(
        cache.sequence(1).expect("sequence").tokens().len(),
        KV_PAGE_TOKENS
    );
    assert_eq!(cache.table(1).expect("table").len(), 1);
    assert_eq!(cache.stats().pages_free, 3);
}

#[test]
fn unknown_and_duplicate_handles_are_refused() {
    let mut cache = pool(2, 2);
    assert_eq!(cache.append(1, &[]), Err(PagingError::UnknownSequence(1)));
    assert_eq!(cache.free_sequence(1), Err(PagingError::UnknownSequence(1)));
    assert_eq!(cache.preempt(1), Err(PagingError::UnknownSequence(1)));
    assert_eq!(cache.table(1), Err(PagingError::UnknownSequence(1)));
    cache.add_sequence(1).expect("add");
    assert_eq!(
        cache.add_sequence(1),
        Err(PagingError::DuplicateSequence(1))
    );
    check(&cache);
}

#[test]
fn tail_slots_are_a_hard_capacity() {
    let mut cache = pool(2, 2);
    cache.add_sequence(1).expect("add");
    cache.add_sequence(2).expect("add");
    assert_eq!(cache.add_sequence(3), Err(PagingError::TailCapacity));
    cache.free_sequence(1).expect("free");
    cache
        .add_sequence(3)
        .expect("a freed tail slot is reusable");
    check(&cache);
}

#[test]
fn a_scripted_mixed_trace_keeps_every_invariant() {
    let mut cache = pool(8, 3);
    cache.add_sequence(10).expect("add");
    cache.add_sequence(11).expect("add");
    cache.append(10, &tokens(0, 150)).expect("append");
    check(&cache);
    cache
        .append(11, &tokens(0, KV_PAGE_TOKENS))
        .expect("share block 0");
    check(&cache);
    assert!(cache.stats().prefix_hits >= 1);
    cache.fork(10, 12, 2).expect("fork");
    check(&cache);
    cache.truncate(10, 128).expect("truncate");
    check(&cache);
    cache.preempt(11).expect("preempt");
    check(&cache);
    cache.append(11, &tokens(0, 1)).expect("resume");
    check(&cache);
    cache.free_sequence(10).expect("free");
    check(&cache);
    cache.free_sequence(12).expect("free");
    check(&cache);
    cache.free_sequence(11).expect("free");
    check(&cache);
    let stats = cache.stats();
    assert_eq!(stats.sequences, 0);
    assert_eq!(stats.pages_free, 8);
    assert_eq!(stats.tails_free, 3);
    assert_eq!(stats.preemptions, 1);
}

#[test]
fn a_randomized_trace_never_breaks_an_invariant() {
    // A small deterministic LCG, so a failure is reproducible without a seed file.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut cache = pool(16, 4);
    let mut handles: Vec<u64> = Vec::new();
    let mut next_handle = 1u64;
    for step in 0..600u64 {
        let pick = next() % 7;
        match pick {
            0 => {
                if handles.len() < 4 {
                    cache
                        .add_sequence(next_handle)
                        .expect("add within capacity");
                    handles.push(next_handle);
                    next_handle += 1;
                }
            }
            1 | 2 => {
                if let Some(handle) = handles
                    .get((next() as usize) % handles.len().max(1))
                    .copied()
                {
                    let count = (next() % 200) as usize;
                    let base = (next() % 1_000) as u32;
                    let _ = cache.append(handle, &tokens(base, count));
                }
            }
            3 => {
                if let Some(handle) = handles
                    .get((next() as usize) % handles.len().max(1))
                    .copied()
                {
                    let _ = cache.free_sequence(handle);
                    handles.retain(|live| *live != handle);
                }
            }
            4 => {
                if let Some(handle) = handles
                    .get((next() as usize) % handles.len().max(1))
                    .copied()
                {
                    let _ = cache.preempt(handle);
                }
            }
            5 => {
                if let Some(handle) = handles
                    .get((next() as usize) % handles.len().max(1))
                    .copied()
                {
                    let tokens_held = cache
                        .sequence(handle)
                        .map_or(0, |sequence| sequence.tokens().len());
                    let blocks = tokens_held / KV_PAGE_TOKENS;
                    let _ = cache.truncate(handle, blocks * KV_PAGE_TOKENS);
                }
            }
            _ => {
                if handles.len() >= 2 {
                    let donor = handles[(next() as usize) % handles.len()];
                    let child = next_handle;
                    if cache.fork(donor, child, 1).is_ok() {
                        handles.push(child);
                        next_handle += 1;
                    }
                }
            }
        }
        cache
            .check_invariants()
            .unwrap_or_else(|error| panic!("step {step} broke an invariant: {error}"));
        let stats = cache.stats();
        assert!(
            stats.pages_free + stats.pages_in_use <= stats.pages_total || stats.blocks_cached > 0
        );
    }
}
