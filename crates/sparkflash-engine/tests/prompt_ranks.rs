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

//! The TP=4 prompt-logprob rank contract against the serial host path.
//!
//! Every shard here is a contiguous slice of a row the serial host engine produced, so the merge
//! is compared with the exact serial log_softmax row: the same tiny plan, the same prompt, the same
//! rows. The tests pin the provable-order rule, the exact owning-token rank, and the typed
//! refusals (certified bounds, ties at the boundary, a missing owning token).

use std::collections::HashSet;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::prompt_ranks::{
    CANDIDATES_PER_RANK_MAX, Candidate, LocalCandidates, MergeRefusal, MergedPosition,
    RankPosition, candidate_budget, decode_phase1, encode_phase1, join_scaled_sum,
    local_candidates, local_greater_count, local_scaled_sum, logprob_of, merge, phase1_stride,
};
use sparkflash_engine::tensor::TensorPlan;

mod common;

use common::FiniteSource;

fn engine(config: &ModelConfig, seed: u64) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), 1 << 22).expect("tiny engine")
}

/// The serial host path's formula, copied exactly from crates/sparkflash-cli/src/serve.rs
/// log_softmax so the parity claim is checked against the reference arithmetic, not a paraphrase.
fn serial_log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let sum: f64 = logits
        .iter()
        .map(|&value| ((value - max) as f64).exp())
        .sum();
    if !(sum.is_finite() && sum > 0.0) {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let log_sum = sum.ln();
    logits
        .iter()
        .map(|&value| ((value - max) as f64 - log_sum) as f32)
        .collect()
}

fn serial_top(row: &[f32], count: usize) -> Vec<(u32, f32)> {
    let values = serial_log_softmax(row);
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order
        .into_iter()
        .take(count)
        .map(|token| (token as u32, values[token]))
        .collect()
}

fn serial_own(row: &[f32], token: u32) -> (f32, u32) {
    let values = serial_log_softmax(row);
    let own = values[token as usize];
    let greater = values.iter().filter(|&&value| value > own).count();
    (own, u32::try_from(greater).unwrap() + 1)
}

/// Splits one full row into the world's contiguous rank shards.
fn shard_ranges(vocab: usize, world: usize) -> Vec<(usize, usize)> {
    (0..world)
        .map(|rank| {
            let per = vocab.div_ceil(world);
            let start = (rank * per).min(vocab);
            let end = (start + per).min(vocab);
            (start, end)
        })
        .collect()
}

/// Simulates the three-pass exchange on already computed shard logits and merges.
fn exchange(
    world: usize,
    rows: &[Vec<f32>],
    own_tokens: &[u32],
    selection: Option<usize>,
    budget: usize,
) -> Result<Vec<MergedPosition>, MergeRefusal> {
    let mut per_rank: Vec<Vec<RankPosition>> = vec![Vec::with_capacity(rows.len()); world];
    for row in rows {
        let ranges = shard_ranges(row.len(), world);
        let locals: Vec<LocalCandidates> = ranges
            .iter()
            .enumerate()
            .map(|(rank, (start, end))| {
                local_candidates(
                    rank,
                    *start,
                    &row[*start..*end],
                    None,
                    own_tokens[per_rank[0].len()],
                    budget,
                )
                .unwrap_or_else(|error| panic!("local candidates: {error}"))
            })
            .collect();
        let global_max = locals
            .iter()
            .fold(f32::NEG_INFINITY, |best, local| best.max(local.local_max));
        let own_token = own_tokens[per_rank[0].len()];
        let own_value = locals
            .iter()
            .find_map(|local| local.own_value)
            .unwrap_or_else(|| panic!("owning token {own_token} is not in any shard"));
        let mut total = 0.0f64;
        for (rank, &(start, end)) in ranges.iter().enumerate() {
            let (high, low) = local_scaled_sum(&row[start..end], None, global_max);
            total += join_scaled_sum(high, low);
            per_rank[rank].push(RankPosition {
                rank,
                offset: locals[rank].offset,
                local_vocab: locals[rank].local_vocab,
                candidates: locals[rank].candidates.clone(),
                own_value: locals[rank].own_value,
                local_max: locals[rank].local_max,
                omitted_exact_max: locals[rank].omitted_exact_max,
                certified_count: locals[rank].certified_count,
                certified_max_bound: locals[rank].certified_max_bound,
                scaled_sum_hi: high,
                scaled_sum_lo: low,
                greater_than_own: local_greater_count(
                    &row[start..end],
                    None,
                    global_max,
                    total.ln(),
                    logprob_of(own_value, global_max, total.ln()),
                ),
            });
        }
    }
    merge(world, selection, own_tokens, &per_rank)
}

fn assert_parity(
    got: &[MergedPosition],
    rows: &[Vec<f32>],
    own_tokens: &[u32],
    selection: Option<usize>,
) {
    assert_eq!(got.len(), rows.len());
    for (index, position) in got.iter().enumerate() {
        assert_eq!(position.position, index);
        assert_eq!(position.own_token_id, own_tokens[index]);
        let (expected_lp, expected_rank) = serial_own(&rows[index], own_tokens[index]);
        assert_eq!(position.own_rank, expected_rank, "own rank at {index}");
        let difference = (position.own_logprob - expected_lp).abs();
        assert!(
            difference <= f32::EPSILON * expected_lp.abs().max(1.0),
            "own logprob at {index}: {} vs {}",
            position.own_logprob,
            expected_lp
        );
        if let Some(count) = selection {
            let expected = serial_top(&rows[index], count);
            let got_ids: Vec<u32> = position.alternatives.iter().map(|a| a.token_id).collect();
            let expected_ids: Vec<u32> = expected.iter().map(|(token, _)| *token).collect();
            assert_eq!(got_ids, expected_ids, "top order at {index}");
            for (alternative, (_, expected_lp)) in position.alternatives.iter().zip(&expected) {
                let difference = (alternative.logprob - expected_lp).abs();
                assert!(
                    difference <= f32::EPSILON * expected_lp.abs().max(1.0),
                    "alternative logprob at {index}: {} vs {}",
                    alternative.logprob,
                    expected_lp
                );
            }
            // The contract's order proof is the same order serial top_k_indices produces.
            let mut unique = HashSet::new();
            for alternative in &position.alternatives {
                assert!(unique.insert(alternative.token_id));
            }
        }
    }
}

#[test]
fn merged_prompt_logprobs_match_the_serial_host_path() {
    let config = ModelConfig::tiny();
    let prompt = [3u32, 7, 11, 19, 5, 23];
    let mut state = SequenceState::new(&config);
    let positions: Vec<usize> = (0..prompt.len() - 1).collect();
    let produced = engine(&config, 11)
        .prefill_with_position_logits(&mut state, &prompt, &positions)
        .expect("selected rows");
    let rows = produced.rows;
    let own_tokens = &prompt[1..];
    for world in [1usize, 2, 4] {
        for selection in [None, Some(1usize), Some(3), Some(5)] {
            let budget = candidate_budget(selection).expect("budget");
            let got = exchange(world, &rows, own_tokens, selection, budget)
                .unwrap_or_else(|error| panic!("world {world} selection {selection:?}: {error}"));
            assert_parity(&got, &rows, own_tokens, selection);
        }
    }
}

#[test]
fn the_phase_one_wire_round_trips_a_shard_view() {
    let row: Vec<f32> = (0..32).map(|index| (index as f32 * 0.37).sin()).collect();
    let budget = 4;
    for rank in 0..4usize {
        let start = rank * 8;
        let local = local_candidates(rank, start, &row[start..start + 8], None, 3, budget)
            .expect("local view");
        let encoded = encode_phase1(&local, budget);
        assert_eq!(encoded.len(), phase1_stride(budget));
        let decoded = decode_phase1(rank, &encoded, budget).expect("decode");
        assert_eq!(decoded, local);
    }
}

#[test]
fn certified_bounds_are_refused_not_approximated() {
    let config = ModelConfig::tiny();
    let prompt = [1u32, 2, 3];
    let mut state = SequenceState::new(&config);
    let positions: Vec<usize> = (0..2).collect();
    let produced = engine(&config, 5)
        .prefill_with_position_logits(&mut state, &prompt, &positions)
        .expect("rows");
    let rows = produced.rows;
    let own_tokens = &prompt[1..];
    let world = 2;
    let budget = candidate_budget(Some(2)).expect("budget");
    let mut per_rank: Vec<Vec<RankPosition>> = vec![Vec::new(), Vec::new()];
    for row in &rows {
        let ranges = shard_ranges(row.len(), world);
        let locals: Vec<LocalCandidates> = ranges
            .iter()
            .enumerate()
            .map(|(rank, (start, end))| {
                local_candidates(
                    rank,
                    *start,
                    &row[*start..*end],
                    None,
                    own_tokens[0],
                    budget,
                )
                .expect("local")
            })
            .collect();
        let global_max = locals
            .iter()
            .fold(f32::NEG_INFINITY, |best, local| best.max(local.local_max));
        let mut total = 0.0f64;
        let sums: Vec<(f32, f32)> = ranges
            .iter()
            .map(|(start, end)| local_scaled_sum(&row[*start..*end], None, global_max))
            .collect();
        for (high, low) in &sums {
            total += join_scaled_sum(*high, *low);
        }
        let own_logprob = logprob_of(
            locals
                .iter()
                .find_map(|local| local.own_value)
                .expect("own"),
            global_max,
            total.ln(),
        );
        for (rank, &(start, end)) in ranges.iter().enumerate() {
            per_rank[rank].push(RankPosition {
                rank,
                offset: locals[rank].offset,
                local_vocab: locals[rank].local_vocab,
                candidates: locals[rank].candidates.clone(),
                own_value: locals[rank].own_value,
                local_max: locals[rank].local_max,
                omitted_exact_max: locals[rank].omitted_exact_max,
                certified_count: 0,
                certified_max_bound: f32::NEG_INFINITY,
                scaled_sum_hi: sums[rank].0,
                scaled_sum_lo: sums[rank].1,
                greater_than_own: local_greater_count(
                    &row[start..end],
                    None,
                    global_max,
                    total.ln(),
                    own_logprob,
                ),
            });
        }
    }
    // One certified bound entry is enough: the distribution is no longer exact.
    per_rank[1][0].certified_count = 1;
    per_rank[1][0].certified_max_bound = 100.0;
    let refusal = merge(world, Some(1), own_tokens, &per_rank).expect_err("certified refusal");
    match refusal {
        MergeRefusal::CertifiedEntries {
            position,
            rank,
            certified_count,
        } => {
            assert_eq!((position, rank, certified_count), (0, 1, 1));
        }
        other => panic!("expected the certified refusal, got {other}"),
    }
}

#[test]
fn a_boundary_tie_is_refused_as_unproven() {
    // A deliberately tied row: two entries share the second-best value, so the top-2 boundary
    // cannot be proven from the candidate union at f32 precision.
    let row: Vec<f32> = vec![10.0, 9.0, 9.0, 1.0, 0.5, -1.0, -2.0, -3.0];
    let own_token = 5u32;
    let world = 2;
    let budget = candidate_budget(Some(2)).expect("budget");
    let ranges = shard_ranges(row.len(), world);
    let locals: Vec<LocalCandidates> = ranges
        .iter()
        .enumerate()
        .map(|(rank, (start, end))| {
            local_candidates(rank, *start, &row[*start..*end], None, own_token, budget)
                .expect("local")
        })
        .collect();
    let global_max = locals
        .iter()
        .fold(f32::NEG_INFINITY, |best, local| best.max(local.local_max));
    let own_value = locals
        .iter()
        .find_map(|local| local.own_value)
        .expect("owning token");
    let mut total = 0.0f64;
    let sums: Vec<(f32, f32)> = ranges
        .iter()
        .map(|(start, end)| local_scaled_sum(&row[*start..*end], None, global_max))
        .collect();
    for (high, low) in &sums {
        total += join_scaled_sum(*high, *low);
    }
    let own_logprob = logprob_of(own_value, global_max, total.ln());
    let mut per_rank: Vec<Vec<RankPosition>> = vec![Vec::new(), Vec::new()];
    for (rank, &(start, end)) in ranges.iter().enumerate() {
        per_rank[rank].push(RankPosition {
            rank,
            offset: locals[rank].offset,
            local_vocab: locals[rank].local_vocab,
            candidates: locals[rank].candidates.clone(),
            own_value: locals[rank].own_value,
            local_max: locals[rank].local_max,
            omitted_exact_max: locals[rank].omitted_exact_max,
            certified_count: 0,
            certified_max_bound: f32::NEG_INFINITY,
            scaled_sum_hi: sums[rank].0,
            scaled_sum_lo: sums[rank].1,
            greater_than_own: local_greater_count(
                &row[start..end],
                None,
                global_max,
                total.ln(),
                own_logprob,
            ),
        });
    }
    let refusal = merge(world, Some(2), &[own_token], &per_rank).expect_err("unproven order");
    match refusal {
        MergeRefusal::UnprovenOrder {
            position,
            requested,
            ..
        } => assert_eq!((position, requested), (0, 2)),
        other => panic!("expected the unproven-order refusal, got {other}"),
    }
    // The owning token's logprob and rank are still exactly computable without the alternatives.
    let merged = merge(world, None, &[own_token], &per_rank).expect("own-only is provable");
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].own_token_id, own_token);
}

#[test]
fn a_missing_owning_token_is_a_layout_refusal() {
    let row: Vec<f32> = (0..8).map(|index| index as f32).collect();
    let budget = 4;
    let locals: Vec<LocalCandidates> = (0..2)
        .map(|rank| {
            local_candidates(
                rank,
                rank * 4,
                &row[rank * 4..rank * 4 + 4],
                None,
                6,
                budget,
            )
            .expect("local")
        })
        .collect();
    let mut per_rank: Vec<Vec<RankPosition>> = vec![Vec::new(), Vec::new()];
    for (rank, local) in locals.iter().enumerate() {
        per_rank[rank].push(RankPosition {
            rank,
            offset: local.offset,
            local_vocab: local.local_vocab,
            candidates: local.candidates.clone(),
            own_value: None,
            local_max: local.local_max,
            omitted_exact_max: local.omitted_exact_max,
            certified_count: 0,
            certified_max_bound: f32::NEG_INFINITY,
            scaled_sum_hi: 1.0,
            scaled_sum_lo: 0.0,
            greater_than_own: 0,
        });
    }
    let refusal = merge(2, None, &[6], &per_rank).expect_err("missing own token");
    assert!(matches!(
        refusal,
        MergeRefusal::OwnTokenMissing { token_id: 6, .. }
    ));
}

#[test]
fn a_non_finite_exact_entry_is_refused() {
    let logits = [1.0f32, f32::NAN, 2.0];
    let refusal = local_candidates(0, 0, &logits, None, 0, 4).expect_err("non-finite exact entry");
    assert!(matches!(
        refusal,
        MergeRefusal::NonFiniteExact { token_id: 1, .. }
    ));
}

#[test]
fn the_candidate_budget_is_bounded() {
    assert_eq!(candidate_budget(None).expect("own only"), 16);
    assert_eq!(candidate_budget(Some(1)).expect("top 1"), 16);
    assert_eq!(
        candidate_budget(Some(CANDIDATES_PER_RANK_MAX - 1)).expect("cap"),
        CANDIDATES_PER_RANK_MAX
    );
    assert!(matches!(
        candidate_budget(Some(CANDIDATES_PER_RANK_MAX)).expect_err("over cap"),
        MergeRefusal::RequestTooWide { .. }
    ));
}

#[test]
fn the_shard_layout_must_be_contiguous() {
    let row: Vec<f32> = (0..8).map(|index| index as f32).collect();
    let budget = 4;
    let locals: Vec<LocalCandidates> = vec![
        local_candidates(0, 0, &row[0..4], None, 1, budget).expect("local"),
        local_candidates(1, 4, &row[4..8], None, 1, budget).expect("local"),
    ];
    let mut per_rank: Vec<Vec<RankPosition>> = vec![Vec::new(), Vec::new()];
    for (rank, local) in locals.iter().enumerate() {
        per_rank[rank].push(RankPosition {
            rank,
            offset: local.offset,
            local_vocab: local.local_vocab,
            candidates: local.candidates.clone(),
            own_value: local.own_value,
            local_max: local.local_max,
            omitted_exact_max: local.omitted_exact_max,
            certified_count: 0,
            certified_max_bound: f32::NEG_INFINITY,
            scaled_sum_hi: 1.0,
            scaled_sum_lo: 0.0,
            greater_than_own: 0,
        });
    }
    // Shift the second shard so the partition has a hole; the merge must refuse the layout.
    per_rank[1][0].offset = 5;
    per_rank[1][0].candidates = vec![Candidate {
        token_id: 5,
        value: 1.0,
    }];
    let refusal = merge(2, None, &[1], &per_rank).expect_err("layout refusal");
    assert!(matches!(refusal, MergeRefusal::ShardLayout { .. }));
}
