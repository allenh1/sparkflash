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

//! CPU twin of `kernels/ref_select_sm121.cu`: the reference candidate select.
//!
//! Every rule here is a transcription of the engine's, with the citing line, and the kernel's body
//! is the same transcription in CUDA. The twin is the comparison the kernel's oracle test uses: a
//! launch's ids must equal [select]'s ids **as an index vector**, not as a score set - a score may
//! differ inside a tie, a chosen index may not.
//!
//! Rules (crates/sparkflash-engine/src/forward/attention.rs):
//! * score: index_scores, :1476-1506 - heads ascending, ascending f32 dot per head,
//!   raw = bf16_round(dot), sum += bf16_round(raw * weight) when raw > 0, final bf16_round.
//! * reach: :1610-1616 - position >= (start_pos + query + 1) / ratio becomes -inf.
//! * block mask: select_candidate_blocks, :1704-1734 - per block the best score, the block holding
//!   the newest reachable position pinned to +inf, the best topk_blocks blocks by (score desc,
//!   block asc) kept, a kept -inf block dropped, kept positions' bits set.
//! * top-k: :1660-1686 - total order (score desc, position asc), first min(topk, n_pos),
//!   re-sorted ascending, then position < reach maps to position + offset, else no_index.

use sparkflash_engine::quant::bf16_round;

/// The most compressed positions one launch can hold: the kernel's shared score table.
pub const MAX_POSITIONS: usize = 4096;

/// The mask is always this many words: `ceil(MAX_POSITIONS / 32)`, the kernel writes all of them.
pub const MASK_WORDS: usize = MAX_POSITIONS / 32;

/// One thread block, as the kernel declares it.
pub const BLOCK: usize = 256;

/// How the block (HSI) mask is used on this launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    /// Window-only selection: no mask, no published mask.
    None,
    /// This layer reads a block mask an earlier layer published.
    Consume,
    /// This layer is the candidate source: computes and publishes the mask, and does not apply it
    /// to its own scores (the engine publishes without masking its own selection).
    Produce,
}

impl MaskMode {
    /// The kernel's `mask_mode` word.
    #[must_use]
    pub fn word(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Consume => 1,
            Self::Produce => 2,
        }
    }
}

/// One query's select request; the launcher's argument words in named form.
#[derive(Debug, Clone, Copy)]
pub struct SelectRequest<'a> {
    /// The indexer query row, `heads * head_dim` values (already rotated and fp4 round-tripped).
    pub q_index: &'a [f32],
    /// The index keys, `n_pos * head_dim` values.
    pub keys: &'a [f32],
    /// The per-query weights, `heads` values.
    pub weights: &'a [f32],
    /// How many compressed positions this query can see.
    pub n_pos: usize,
    /// Index heads.
    pub heads: usize,
    /// One index head's width.
    pub head_dim: usize,
    /// The query's reachable count: `(start_pos + 1) / ratio` for a decode step.
    pub reach: usize,
    /// `config.index_topk`.
    pub topk: usize,
    /// The window rows the shared index space is shifted by (`window_rows`).
    pub offset: u32,
    /// How the block mask is used.
    pub mask_mode: MaskMode,
    /// `config.candidate_topk_blocks`.
    pub topk_blocks: usize,
    /// `config.candidate_block_size`.
    pub block_size: usize,
    /// The shared index space's "nothing here" value.
    pub no_index: u32,
    /// The mask this launch consumes; `MASK_WORDS` words when [MaskMode::Consume].
    pub mask_in: &'a [u32],
}

/// One launch's outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectOutput {
    /// The chosen ids in the shared index space, ascending by compressing position.
    pub ids: Vec<u32>,
    /// Every position's score after the masks, by position.
    pub scores: Vec<f32>,
    /// The published block mask, when [MaskMode::Produce].
    pub mask: Option<Vec<u32>>,
}

impl SelectOutput {
    /// The number of ids written: `min(topk, n_pos)`.
    #[must_use]
    pub fn id_count(n_pos: usize, topk: usize) -> usize {
        topk.min(n_pos)
    }
}

fn reasons(request: &SelectRequest<'_>) -> Result<(), String> {
    if request.n_pos > MAX_POSITIONS {
        return Err(format!(
            "ref select: {} compressed positions, the kernel's score table holds {MAX_POSITIONS}",
            request.n_pos
        ));
    }
    if request.heads == 0 || request.head_dim == 0 {
        return Err("ref select: zero heads or zero head width".to_string());
    }
    if request.q_index.len() != request.heads * request.head_dim {
        return Err(format!(
            "ref select: the query row is {} values, {heads} heads of {width} need {}",
            request.q_index.len(),
            request.heads * request.head_dim,
            heads = request.heads,
            width = request.head_dim
        ));
    }
    if request.keys.len() < request.n_pos * request.head_dim {
        return Err(format!(
            "ref select: the keys hold {} values, {n} positions of {width} need {}",
            request.keys.len(),
            request.n_pos * request.head_dim,
            n = request.n_pos,
            width = request.head_dim
        ));
    }
    if request.weights.len() != request.heads {
        return Err(format!(
            "ref select: {} query weights for {} index heads",
            request.weights.len(),
            request.heads
        ));
    }
    if request.mask_mode == MaskMode::Consume && request.mask_in.len() != MASK_WORDS {
        return Err(format!(
            "ref select: {} mask words for a consumed mask of {MASK_WORDS}",
            request.mask_in.len()
        ));
    }
    if request.mask_mode == MaskMode::Produce && request.block_size == 0 {
        return Err("ref select: a produced block mask needs a non-zero block size".to_string());
    }
    Ok(())
}

/// The engine's `index_scores` (:1476-1506) for one query: `n_pos` scores.
#[must_use]
pub fn scores(request: &SelectRequest<'_>) -> Vec<f32> {
    let mut score = vec![0.0f32; request.n_pos];
    for head in 0..request.heads {
        let q_head = &request.q_index[head * request.head_dim..(head + 1) * request.head_dim];
        let weight = request.weights[head];
        for (position, sum) in score.iter_mut().enumerate() {
            let key = &request.keys[position * request.head_dim..(position + 1) * request.head_dim];
            let mut acc = 0.0f32;
            for (q, k) in q_head.iter().zip(key) {
                acc += *q * *k;
            }
            let raw = bf16_round(acc);
            if raw > 0.0 {
                *sum += bf16_round(raw * weight);
            }
        }
    }
    for value in score.iter_mut() {
        *value = bf16_round(*value);
    }
    score
}

/// Applies the reach and consumed-mask rules to a score row (:1610-1616, :1640-1647).
pub fn mask_scores(score: &mut [f32], reach: usize, mask: Option<&[u32]>) {
    for (position, value) in score.iter_mut().enumerate() {
        if position >= reach
            || mask.is_some_and(|bits| bits[position >> 5] & (1 << (position & 31)) == 0)
        {
            *value = f32::NEG_INFINITY;
        }
    }
}

/// `select_candidate_blocks` (:1704-1734): the HSI block mask one query publishes.
#[must_use]
pub fn candidate_mask(
    score: &[f32],
    reach: usize,
    topk_blocks: usize,
    block_size: usize,
) -> Vec<u32> {
    let mut mask = vec![0u32; MASK_WORDS];
    let n_pos = score.len();
    if block_size == 0 || n_pos == 0 {
        return mask;
    }
    let num_blocks = n_pos.div_ceil(block_size);
    let keep = topk_blocks.min(num_blocks);
    let mut block_scores = vec![f32::NEG_INFINITY; num_blocks];
    for (block, best) in block_scores.iter_mut().enumerate() {
        let start = block * block_size;
        let end = (start + block_size).min(n_pos);
        *best = score[start..end]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
    }
    if reach > 0 {
        let last = (reach - 1) / block_size;
        if let Some(best) = block_scores.get_mut(last) {
            *best = f32::INFINITY;
        }
    }
    let mut order: Vec<usize> = (0..num_blocks).collect();
    order.sort_by(|left, right| {
        block_scores[*right]
            .partial_cmp(&block_scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    order.truncate(keep);
    for &block in &order {
        if block_scores[block] > f32::NEG_INFINITY {
            let start = block * block_size;
            let end = (start + block_size).min(n_pos);
            for position in start..end {
                mask[position >> 5] |= 1 << (position & 31);
            }
        }
    }
    mask
}

/// The top-k and the shared-index-space mapping over already-masked scores.
#[must_use]
pub fn select_ids(
    score: &[f32],
    reach: usize,
    topk: usize,
    offset: u32,
    no_index: u32,
) -> Vec<u32> {
    let n_pos = score.len();
    let keep = topk.min(n_pos);
    let mut order: Vec<usize> = (0..n_pos).collect();
    order.sort_by(|left, right| {
        score[*right]
            .partial_cmp(&score[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    order.truncate(keep);
    order.sort_unstable();
    order
        .into_iter()
        .map(|position| {
            if position < reach {
                position as u32 + offset
            } else {
                no_index
            }
        })
        .collect()
}

/// One launch: scores, masks, selection. The twin the kernel's ids are compared with.
///
/// # Errors
/// Shape and limit refusals, before any arithmetic: a launch that cannot be faithful does not run.
pub fn select(request: &SelectRequest<'_>) -> Result<SelectOutput, String> {
    reasons(request)?;
    let mut score = scores(request);
    let (published, consumed) = match request.mask_mode {
        MaskMode::None => (None, None),
        MaskMode::Consume => (None, Some(request.mask_in)),
        MaskMode::Produce => (
            Some(candidate_mask(
                &score,
                request.reach,
                request.topk_blocks,
                request.block_size,
            )),
            None,
        ),
    };
    mask_scores(&mut score, request.reach, consumed);
    let ids = select_ids(
        &score,
        request.reach,
        request.topk,
        request.offset,
        request.no_index,
    );
    Ok(SelectOutput {
        ids,
        scores: score,
        mask: published,
    })
}
