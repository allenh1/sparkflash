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

//! The two top-k selections more than one model makes: which pooled key blocks a sparse-attention
//! query attends over, and which routed experts a token goes to.
//!
//! * [top_pools] is the pooled-block indexer's selection. Qwen3.8's QSA indexer and
//!   GLM-5.3-Flash's DSA k-pool indexer both score whole blocks of `pool` consecutive tokens, keep
//!   the best `budget` complete blocks a query can see, and attend over their tokens plus the
//!   incomplete block the query sits in. How a block is pooled and scored differs per model and
//!   stays with the pack.
//! * [noaux_top_k] is DeepSeek's `noaux_tc` router selection, which V4.1 and GLM-5.3-Flash share:
//!   the selection bias decides which experts, the unbiased scores weight them, and the weights
//!   are normalised and scaled. The score function before it (V4.1's `sqrtsoftplus`, GLM's
//!   sigmoid) stays with the pack.
//!
//! Both break ties toward the lower index, so a selection is a function of the scores rather than
//! of the order the candidates were visited in.
//!
//! # Attribution
//!
//! [noaux_top_k] follows the gate of DeepSeek's released V4.1-Flash inference reference
//! (`model.py`; MIT License, Copyright (c) DeepSeek), whose `noaux_tc` rule GLM-5.3-Flash's router
//! shares. [top_pools] is the block selection of Qwen Sparse Attention's indexer
//! (Qwen3.8-Flash-Next) and of DeepSeek Sparse Attention's k-pool indexer as GLM-5.3-Flash runs it
//! (DeepSeek-V3.2, arXiv:2512.02556). Read for the arithmetic; no source is copied. See the root
//! NOTICE file.

/// The tokens a query at `position` attends over, ascending, into `selected` (cleared first).
///
/// `scores` holds one score per complete block the query can see: block `b` covers tokens
/// `b * pool..(b + 1) * pool`, and there are `(position + 1) / pool` of them. The best
/// `budget` blocks by score (descending, then block index ascending) contribute all their tokens.
/// With `tail`, the tokens after the last complete block, up to and including `position`, follow.
///
/// The tie-break is not decoration: a block whose head scores are all non-positive scores
/// `relu(0) = 0`, so equal scores are common, and an unstable sort alone would let two runs (or a
/// host and a device arm) keep different blocks and attend over different tokens. `torch.topk`,
/// which the references use, breaks ties by index order.
///
/// # Panics
///
/// When `pool` is zero, or `scores` is not one score per complete block before `position`.
pub fn top_pools(
    scores: &[f32],
    pool: usize,
    budget: usize,
    position: usize,
    tail: bool,
    selected: &mut Vec<usize>,
) {
    assert!(pool > 0, "a block of zero tokens");
    let blocks = (position + 1) / pool;
    assert_eq!(
        scores.len(),
        blocks,
        "a query at {position} sees {blocks} complete {pool}-token blocks"
    );
    selected.clear();
    let mut ranked: Vec<usize> = (0..blocks).collect();
    ranked.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b)));
    ranked.truncate(budget.min(blocks));
    ranked.sort_unstable();
    for block in ranked {
        selected.extend(block * pool..(block + 1) * pool);
    }
    if tail {
        selected.extend(blocks * pool..=position);
    }
}

/// The reduction epsilon of the weight normalisation: the literal `1e-20` both references add to
/// the sum, not a norm epsilon.
pub const ROUTE_NORM_EPSILON: f32 = 1e-20;

/// `noaux_tc` expert selection for one token, into `selected` (cleared first) as
/// `(expert, weight)` pairs in descending selection-signal order.
///
/// `scores` are the token's unbiased router scores, after the model's score function. The
/// selection signal is `scores + bias`, so the bias decides only *which* experts: each chosen
/// expert is weighted by its plain score, and an expert can be chosen by the bias and still carry
/// a small weight. With `norm_topk_prob` and more than one expert, the weights are divided by
/// their sum plus [ROUTE_NORM_EPSILON], summed in selection order; then every weight is multiplied
/// by `route_scale`.
///
/// Ties go to the lower expert index. A NaN signal is never preferred over a number, but the first
/// candidate is accepted while nothing is chosen yet, so a token whose signals are all NaN still
/// gets `topk` experts.
///
/// # Panics
///
/// When `bias` is shorter than `scores`.
pub fn noaux_top_k(
    scores: &[f32],
    bias: &[f32],
    topk: usize,
    norm_topk_prob: bool,
    route_scale: f32,
    selected: &mut Vec<(usize, f32)>,
) {
    let n_routed = scores.len();
    assert!(
        bias.len() >= n_routed,
        "{n_routed} scores against {} biases",
        bias.len()
    );
    selected.clear();
    for _ in 0..topk {
        let mut best: Option<usize> = None;
        let mut best_signal = f32::NEG_INFINITY;
        for expert in 0..n_routed {
            if selected.iter().any(|(taken, _)| *taken == expert) {
                continue;
            }
            let signal = scores[expert] + bias[expert];
            if best.is_none() || signal > best_signal {
                best = Some(expert);
                best_signal = signal;
            }
        }
        match best {
            Some(expert) => selected.push((expert, scores[expert])),
            None => break,
        }
    }
    if norm_topk_prob && topk > 1 {
        let sum: f32 = selected.iter().map(|(_, weight)| *weight).sum();
        let denominator = sum + ROUTE_NORM_EPSILON;
        for (_, weight) in selected.iter_mut() {
            *weight /= denominator;
        }
    }
    for (_, weight) in selected.iter_mut() {
        *weight *= route_scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_best_blocks_and_the_tail_are_attended_in_order() {
        let mut selected = Vec::new();
        // Position 6 with 2-token blocks: blocks 0..3 are complete, token 6 is the tail.
        top_pools(&[0.5, 2.0, 1.0], 2, 2, 6, true, &mut selected);
        assert_eq!(selected, vec![2, 3, 4, 5, 6]);
        top_pools(&[0.5, 2.0, 1.0], 2, 2, 6, false, &mut selected);
        assert_eq!(selected, vec![2, 3, 4, 5]);
    }

    #[test]
    fn a_block_tie_keeps_the_lower_block() {
        let mut selected = Vec::new();
        top_pools(&[0.0, 0.0, 0.0, 0.0], 1, 2, 3, true, &mut selected);
        assert_eq!(selected, vec![0, 1]);
    }

    #[test]
    fn a_query_before_the_first_complete_block_sees_only_its_tail() {
        let mut selected = vec![9];
        top_pools(&[], 4, 2, 2, true, &mut selected);
        assert_eq!(selected, vec![0, 1, 2]);
        // A query that completes a block has no tail.
        top_pools(&[1.0], 4, 2, 3, true, &mut selected);
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_bias_selects_and_the_plain_score_weights() {
        let mut selected = Vec::new();
        // Expert 2 is chosen for its bias but weighted by its plain score.
        noaux_top_k(
            &[0.6, 0.3, 0.1, 0.5],
            &[0.0, 0.0, 1.0, 0.0],
            2,
            true,
            2.0,
            &mut selected,
        );
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 2);
        assert_eq!(selected[1].0, 0);
        let denominator = 0.1f32 + 0.6 + ROUTE_NORM_EPSILON;
        assert_eq!(selected[0].1, 0.1 / denominator * 2.0);
        assert_eq!(selected[1].1, 0.6 / denominator * 2.0);
    }

    #[test]
    fn an_expert_tie_keeps_the_lower_index_and_one_expert_is_not_normalised() {
        let mut selected = Vec::new();
        noaux_top_k(&[0.4, 0.4], &[0.0, 0.0], 1, true, 1.5, &mut selected);
        assert_eq!(selected, vec![(0, 0.4 * 1.5)]);
    }
}
