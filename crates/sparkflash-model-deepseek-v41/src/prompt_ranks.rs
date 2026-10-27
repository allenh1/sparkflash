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

//! The TP=4 all-position prompt-logprob rank contract.
//!
//! The Phase-1 TP=4 host path spreads the LM head over the ranks: rank r serves the contiguous
//! vocabulary rows [offset, offset + local_vocab) and no rank holds the whole distribution. A
//! prompt-logprob request needs, for every scored prompt position, the *whole* distribution's
//! log-softmax: the position's own token logprob and rank, and the requested top-n alternatives in
//! order. This module is the host-side contract that turns rank-local shard views into that
//! payload, and it states exactly when the answer is provable.
//!
//! # Certificate semantics (m16/m27)
//!
//! The device screen stores one certification byte per vocabulary entry. The measured semantics
//! (docs/archive/build/phase3-lm-head.md, docs/archive/build/device-forward-chain.md, and the host twin in
//! crates/sparkflash-cli/src/tp4_host.rs) are:
//!
//! * certified == 0 entries store the exact f32 logit;
//! * certified == 1 entries store an upper certificate bound (>= the exact logit within tolerance),
//!   not the exact value.
//!
//! That is the opposite direction from "masked entries are exact": a certified entry is a bound.
//! This contract therefore refuses to report prompt logprobs from a shard that contains any
//! certified entry: the log-sum-exp over the vocabulary, and therefore every logprob value, cannot
//! be computed exactly from bounds. The refusal is typed ([MergeRefusal::CertifiedEntries]) and
//! says which rank and position caused it. Order-only reasoning from bounds is deliberately not
//! served: a serving value needs the distribution, not just the ordering.
//!
//! # What a rank exchanges
//!
//! Phase 1: per scored position, the budget best *exact* candidates (value, token id) sorted by
//! (value descending, token id ascending), the rank's own-token value when that token lives in
//! this shard, the local maximum, the best omitted exact value, and the certified count/bound.
//! Phase 2: after every rank knows the global maximum, the rank's f64 partial log-sum-exp, split
//! into a double-f32 pair so the host can join it without losing the f64 precision. Phase 3: after
//! every rank knows the log-sum, the rank's count of entries whose f32 logprob is strictly greater
//! than the owning token's logprob. The host merge is deterministic: every rank computes the same
//! payload, and rank 0 is the only one that builds wire entries.
//!
//! # Provable order
//!
//! The union of the rank-local candidate lists is a superset of the global top budget when every
//! rank reports its exact best entries and the shards partition the vocabulary. The requested
//! top-n order is reported only when the n-th union value's f32 logprob is strictly greater than
//! the best omitted value's logprob, and strictly separated from the (n+1)-th union value, by the
//! documented f32 safety margin [order_margin]. Otherwise the merge returns
//! [MergeRefusal::UnprovenOrder]: the bound is not provable at f32 precision, so the contract
//! refuses instead of reporting a different order.
//!
//! # Numeric bar
//!
//! Every value uses the same formula as the serial host path (log_softmax in
//! crates/sparkflash-cli/src/serve.rs): max over the vocabulary, then
//! ((value - max) as f64).exp() summed in f64, then ((value - max) as f64 - ln(sum)) as f32.
//! The distributed sum adds one f64 partial per rank instead of the serial path's single f64
//! accumulation in token-id order. Regrouping f64 additions changes only the last bits: the
//! relative difference is bounded by about (vocab_size * f64::EPSILON), which is about 1.5e-11 for
//! the released 129,280-entry vocabulary and about 1.5e-14 for the 128-entry tiny plan. That is
//! several orders of magnitude below one f32 ULP of a logprob at the magnitudes this model
//! produces, so the served f32 bits are expected to match the serial path; the tests assert the
//! exact order and a one-ULP value tolerance and record where the bits differ.

use std::collections::HashMap;
use std::fmt;

/// The smallest per-rank candidate list the exchange uses.
pub const CANDIDATES_PER_RANK_MIN: usize = 16;

/// The largest per-rank candidate list the exchange uses.
pub const CANDIDATES_PER_RANK_MAX: usize = 64;

/// Token ids must fit an f32 exactly on the candidate wire (2^24 is the f32 integer limit).
pub const TOKEN_ID_LIMIT: usize = 1 << 24;

/// One exact (value, token id) pair in a rank-local candidate list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The global vocabulary id.
    pub token_id: u32,
    /// The exact f32 logit.
    pub value: f32,
}

/// One rank's phase-1 view of one scored prompt position.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalCandidates {
    /// The rank that produced this view.
    pub rank: usize,
    /// The first global token id of the rank's shard.
    pub offset: usize,
    /// The rank's shard width.
    pub local_vocab: usize,
    /// The exact best candidates, sorted by (value descending, token id ascending).
    pub candidates: Vec<Candidate>,
    /// The owning prompt token's exact value, when that token lives in this rank's shard.
    pub own_value: Option<f32>,
    /// The best exact value in the shard, or -inf when the shard has no exact entry.
    pub local_max: f32,
    /// The best exact value that did not fit the candidate budget, or -inf when nothing was left
    /// out.
    pub omitted_exact_max: f32,
    /// How many entries are certified bounds rather than exact logits (m16/m27 semantics).
    pub certified_count: usize,
    /// The best certified bound, or -inf when the shard has no certified entry.
    pub certified_max_bound: f32,
}

/// One rank's complete contribution to one scored prompt position, including the phase-2 and
/// phase-3 scans that depend on the global maximum and the log-sum.
#[derive(Debug, Clone, PartialEq)]
pub struct RankPosition {
    /// The rank that produced this view.
    pub rank: usize,
    /// The first global token id of the rank's shard.
    pub offset: usize,
    /// The rank's shard width.
    pub local_vocab: usize,
    /// The exact best candidates, sorted by (value descending, token id ascending).
    pub candidates: Vec<Candidate>,
    /// The owning prompt token's exact value, when that token lives in this rank's shard.
    pub own_value: Option<f32>,
    /// The best exact value in the shard.
    pub local_max: f32,
    /// The best exact value that did not fit the candidate budget.
    pub omitted_exact_max: f32,
    /// How many entries are certified bounds rather than exact logits.
    pub certified_count: usize,
    /// The best certified bound.
    pub certified_max_bound: f32,
    /// The high f32 of the rank's f64 partial log-sum-exp against the global maximum.
    pub scaled_sum_hi: f32,
    /// The low f32 of the rank's f64 partial log-sum-exp against the global maximum.
    pub scaled_sum_lo: f32,
    /// How many of the rank's entries have a strictly greater f32 logprob than the owning token.
    pub greater_than_own: u32,
}

/// One alternative in a merged prompt-logprob position.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedAlternative {
    /// The global vocabulary id.
    pub token_id: u32,
    /// The f32 log probability.
    pub logprob: f32,
}

/// One merged scored prompt position, before tokenizer decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedPosition {
    /// The zero-based index in the scored-position list (prompt index minus one for the TP=4 path).
    pub position: usize,
    /// The prompt token this position scores.
    pub own_token_id: u32,
    /// Its log probability.
    pub own_logprob: f32,
    /// Its exact 1-based competition rank (strictly greater count plus one).
    pub own_rank: u32,
    /// The requested alternatives, in serial top-k order.
    pub alternatives: Vec<MergedAlternative>,
}

/// Why the contract refused to report a prompt-logprob payload.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeRefusal {
    /// The registry cannot report the whole vocabulary from bounded candidates.
    AllRequiresFullVocabulary,
    /// The requested top count exceeds the bounded candidate exchange.
    RequestTooWide {
        /// The requested count.
        requested: usize,
        /// The largest count the exchange can prove.
        budget: usize,
    },
    /// An "exact" shard entry was not finite, so no distribution can be formed from it.
    NonFiniteExact {
        /// The rank that reported it.
        rank: usize,
        /// The global token id.
        token_id: u32,
    },
    /// A shard carried certified bounds (m16/m27): values cannot be exact, so serving refuses.
    CertifiedEntries {
        /// The scored position index.
        position: usize,
        /// The rank that reported certified entries.
        rank: usize,
        /// How many entries were certified bounds.
        certified_count: usize,
    },
    /// No rank's shard held a finite exact value: the row has no distribution.
    NoDistribution {
        /// The scored position index.
        position: usize,
    },
    /// The owning prompt token was not found in any rank shard.
    OwnTokenMissing {
        /// The scored position index.
        position: usize,
        /// The owning prompt token id.
        token_id: u32,
    },
    /// The owning prompt token's value was not finite.
    OwnTokenNonFinite {
        /// The scored position index.
        position: usize,
        /// The owning prompt token id.
        token_id: u32,
    },
    /// The per-position shard layout was not a contiguous exact partition.
    ShardLayout {
        /// The scored position index.
        position: usize,
        /// A human-readable detail.
        detail: String,
    },
    /// A group-level layout or protocol problem.
    RankLayout {
        /// A human-readable detail.
        detail: String,
    },
    /// The candidate union did not prove the requested top-n order at f32 precision.
    UnprovenOrder {
        /// The scored position index.
        position: usize,
        /// The requested count.
        requested: usize,
        /// The n-th union value's f32 logprob, when it exists.
        boundary_logprob: Option<f32>,
        /// The best omitted value's f32 logprob, when it exists.
        omitted_logprob: Option<f32>,
    },
}

impl fmt::Display for MergeRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllRequiresFullVocabulary => write!(
                formatter,
                "prompt_logprobs -1 (the whole vocabulary) needs the full-vocabulary gather; the TP=4 bounded candidate exchange only serves a top-n count"
            ),
            Self::RequestTooWide { requested, budget } => write!(
                formatter,
                "prompt_logprobs {requested} exceeds the TP=4 bounded candidate budget of {budget} alternatives that can be proven from rank-local candidate lists"
            ),
            Self::NonFiniteExact { rank, token_id } => write!(
                formatter,
                "rank {rank} reports the exact entry {token_id} as non-finite; the TP=4 prompt-logprob contract needs finite exact logits"
            ),
            Self::CertifiedEntries {
                position,
                rank,
                certified_count,
            } => write!(
                formatter,
                "rank {rank} holds {certified_count} certified bound entries at scored position {position}; m16/m27 certified entries are upper bounds, not exact logits, so the distribution (and therefore every logprob value) cannot be reported exactly"
            ),
            Self::NoDistribution { position } => write!(
                formatter,
                "scored position {position} has no finite exact logit in any rank shard; there is no distribution to report"
            ),
            Self::OwnTokenMissing { position, token_id } => write!(
                formatter,
                "the owning token {token_id} at scored position {position} was not present in any rank shard, so the shard layout does not partition the vocabulary"
            ),
            Self::OwnTokenNonFinite { position, token_id } => write!(
                formatter,
                "the owning token {token_id} at scored position {position} has a non-finite logit; the TP=4 contract cannot report its logprob exactly"
            ),
            Self::ShardLayout { position, detail } => write!(
                formatter,
                "the rank shards at scored position {position} are not a contiguous exact partition: {detail}"
            ),
            Self::RankLayout { detail } => {
                write!(formatter, "the rank exchange is malformed: {detail}")
            }
            Self::UnprovenOrder {
                position,
                requested,
                boundary_logprob,
                omitted_logprob,
            } => write!(
                formatter,
                "the requested top-{requested} order at scored position {position} cannot be proven at f32 precision: the n-th candidate logprob {boundary_logprob:?} does not strictly dominate the best omitted logprob {omitted_logprob:?} by the f32 safety margin"
            ),
        }
    }
}

impl std::error::Error for MergeRefusal {}

/// The f32 safety margin an order proof needs.
///
/// The distributed log-sum-exp can differ from the serial one by a relative regrouping term that
/// is around one part in 1e11, but the reported values are f32. When two candidate logprobs are
/// closer than a few f32 ULPs, the serial path's own rounding can order them differently from the
/// distributed formula, so the merge refuses instead of claiming an order it cannot prove. The
/// margin scales with the magnitude of the boundary logprob, which is where the ULP lives.
#[must_use]
pub fn order_margin(boundary_logprob: f32) -> f32 {
    4.0 * f32::EPSILON * boundary_logprob.abs().max(1.0)
}

/// The per-rank candidate budget for a requested top count.
///
/// The budget covers the requested alternatives plus one spare so the boundary and its successor
/// are both visible to the merge, with a floor that keeps small requests cheap and a cap that keeps
/// the exchange bounded.
///
/// # Errors
///
/// [MergeRefusal::RequestTooWide] when the requested count is above the bounded exchange.
pub fn candidate_budget(selection: Option<usize>) -> Result<usize, MergeRefusal> {
    let requested = selection.unwrap_or(0);
    let maximum = CANDIDATES_PER_RANK_MAX - 1;
    if requested > maximum {
        return Err(MergeRefusal::RequestTooWide {
            requested,
            budget: maximum,
        });
    }
    Ok(requested
        .saturating_add(1)
        .clamp(CANDIDATES_PER_RANK_MIN, CANDIDATES_PER_RANK_MAX))
}

/// The value of the logprob formula the serial host path uses, for one logit.
#[must_use]
pub fn logprob_of(value: f32, global_max: f32, log_sum: f64) -> f32 {
    ((value - global_max) as f64 - log_sum) as f32
}

/// Splits one f64 partial sum into a double-f32 pair for the exchange.
#[must_use]
pub fn split_scaled_sum(sum: f64) -> (f32, f32) {
    let high = sum as f32;
    let low = (sum - f64::from(high)) as f32;
    (high, low)
}

/// Joins a double-f32 pair back into the f64 partial sum.
#[must_use]
pub fn join_scaled_sum(high: f32, low: f32) -> f64 {
    f64::from(high) + f64::from(low)
}

/// Builds one rank's exact candidate view of one shard.
///
/// The shard is the contiguous global token-id range [offset, offset + logits.len()). Entries whose
/// certification byte is nonzero are bounds and are counted, never treated as exact. Among exact
/// entries the best budget candidates are returned sorted by (value descending, token id
/// ascending), exactly the serial top_k_indices tie-break.
///
/// # Errors
///
/// [MergeRefusal::ShardLayout] for a malformed shard, and [MergeRefusal::NonFiniteExact] for an
/// exact entry that is not finite.
pub fn local_candidates(
    rank: usize,
    offset: usize,
    logits: &[f32],
    certified: Option<&[u8]>,
    own_token: u32,
    budget: usize,
) -> Result<LocalCandidates, MergeRefusal> {
    if budget == 0 || budget > CANDIDATES_PER_RANK_MAX {
        return Err(MergeRefusal::RankLayout {
            detail: format!("candidate budget {budget} is outside 1..={CANDIDATES_PER_RANK_MAX}"),
        });
    }
    if let Some(certified) = certified
        && certified.len() != logits.len()
    {
        return Err(MergeRefusal::RankLayout {
            detail: format!(
                "{} certification bytes for {} shard entries",
                certified.len(),
                logits.len()
            ),
        });
    }
    let end = offset
        .checked_add(logits.len())
        .ok_or_else(|| MergeRefusal::ShardLayout {
            position: 0,
            detail: format!("rank {rank} shard at offset {offset} overflows"),
        })?;
    if end > TOKEN_ID_LIMIT {
        return Err(MergeRefusal::ShardLayout {
            position: 0,
            detail: format!(
                "rank {rank} shard ends at token {end}, above the {TOKEN_ID_LIMIT} f32-exact limit"
            ),
        });
    }
    let mut exact: Vec<(f32, u32)> = Vec::with_capacity(logits.len());
    let mut certified_count = 0usize;
    let mut certified_max_bound = f32::NEG_INFINITY;
    let mut own_value = None;
    for (index, &value) in logits.iter().enumerate() {
        let token_id = u32::try_from(offset + index).map_err(|_| MergeRefusal::ShardLayout {
            position: 0,
            detail: format!("token id {} does not fit u32", offset + index),
        })?;
        let is_certified = certified.is_some_and(|flags| flags[index] != 0);
        if is_certified {
            certified_count += 1;
            if value.is_finite() && value > certified_max_bound {
                certified_max_bound = value;
            }
            continue;
        }
        if !value.is_finite() {
            return Err(MergeRefusal::NonFiniteExact { rank, token_id });
        }
        if token_id == own_token {
            own_value = Some(value);
        }
        exact.push((value, token_id));
    }
    exact.sort_unstable_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let local_max = exact.first().map_or(f32::NEG_INFINITY, |pair| pair.0);
    let omitted_exact_max = exact.get(budget).map_or(f32::NEG_INFINITY, |pair| pair.0);
    let candidates = exact
        .into_iter()
        .take(budget)
        .map(|(value, token_id)| Candidate { token_id, value })
        .collect();
    Ok(LocalCandidates {
        rank,
        offset,
        local_vocab: logits.len(),
        candidates,
        own_value,
        local_max,
        omitted_exact_max,
        certified_count,
        certified_max_bound,
    })
}

/// The phase-2 scan: the rank's f64 partial sum of exp((value - global_max)) over its exact
/// entries, split into a double-f32 pair.
#[must_use]
pub fn local_scaled_sum(logits: &[f32], certified: Option<&[u8]>, global_max: f32) -> (f32, f32) {
    let mut sum = 0.0f64;
    for (index, &value) in logits.iter().enumerate() {
        if certified.is_some_and(|flags| flags[index] != 0) {
            continue;
        }
        sum += f64::from(value - global_max).exp();
    }
    split_scaled_sum(sum)
}

/// The phase-3 scan: how many of the rank's exact entries have a strictly greater f32 logprob
/// than the owning token, computed with the serial formula so the count matches the serial
/// competition_rank definition on the same values.
#[must_use]
pub fn local_greater_count(
    logits: &[f32],
    certified: Option<&[u8]>,
    global_max: f32,
    log_sum: f64,
    own_logprob: f32,
) -> u32 {
    let mut greater = 0u32;
    for (index, &value) in logits.iter().enumerate() {
        if certified.is_some_and(|flags| flags[index] != 0) {
            continue;
        }
        let logprob = logprob_of(value, global_max, log_sum);
        if logprob > own_logprob {
            greater = greater.saturating_add(1);
        }
    }
    greater
}

/// The per-position stride of the phase-1 exchange in f32 slots.
#[must_use]
pub fn phase1_stride(budget: usize) -> usize {
    // budget (value, token id) pairs, then the own-token value, then the six aggregates:
    // offset, local_vocab, local_max, omitted_exact_max, certified_count, certified_max_bound.
    // The owning token id is not carried: the request names it, and the owning value plus the
    // requested token is enough to place it.
    budget * 2 + 7
}

/// The token-id sentinel used by the phase-1 wire for an absent slot.
const NO_TOKEN: f32 = -1.0;

/// Encodes one rank's phase-1 view into its fixed-stride f32 block.
#[must_use]
pub fn encode_phase1(local: &LocalCandidates, budget: usize) -> Vec<f32> {
    let mut wire = Vec::with_capacity(phase1_stride(budget));
    for slot in 0..budget {
        match local.candidates.get(slot) {
            Some(candidate) => {
                wire.push(candidate.value);
                wire.push(candidate.token_id as f32);
            }
            None => {
                wire.push(f32::NEG_INFINITY);
                wire.push(NO_TOKEN);
            }
        }
    }
    wire.push(local.own_value.unwrap_or(f32::NEG_INFINITY));
    wire.push(local.offset as f32);
    wire.push(local.local_vocab as f32);
    wire.push(local.local_max);
    wire.push(local.omitted_exact_max);
    wire.push(local.certified_count as f32);
    wire.push(local.certified_max_bound);
    wire
}

/// Decodes one rank's phase-1 block.
///
/// The owning token id is not carried by the block: [LocalCandidates] keeps only the owning value,
/// and the merge pairs it with the request's prompt token. The decode is here so tests and the
/// rank exchange agree on the layout.
///
/// # Errors
///
/// [MergeRefusal::ShardLayout] for a malformed block.
pub fn decode_phase1(
    rank: usize,
    block: &[f32],
    budget: usize,
) -> Result<LocalCandidates, MergeRefusal> {
    let stride = phase1_stride(budget);
    if block.len() != stride {
        return Err(MergeRefusal::RankLayout {
            detail: format!(
                "rank {rank} phase-1 block has {} slots, expected {stride}",
                block.len()
            ),
        });
    }
    let mut candidates = Vec::new();
    for slot in 0..budget {
        let value = block[slot * 2];
        let token = block[slot * 2 + 1];
        if token == NO_TOKEN {
            if value != f32::NEG_INFINITY {
                return Err(MergeRefusal::ShardLayout {
                    position: 0,
                    detail: format!(
                        "rank {rank} candidate slot {slot} has value {value} but no token id"
                    ),
                });
            }
            continue;
        }
        if !(token >= 0.0 && token < TOKEN_ID_LIMIT as f32) || token.fract() != 0.0 {
            return Err(MergeRefusal::ShardLayout {
                position: 0,
                detail: format!("rank {rank} candidate slot {slot} has token id {token}"),
            });
        }
        candidates.push(Candidate {
            token_id: token as u32,
            value,
        });
    }
    let own_value = block[budget * 2];
    let own_value = if own_value == f32::NEG_INFINITY {
        None
    } else {
        Some(own_value)
    };
    let offset = block[budget * 2 + 1];
    let local_vocab = block[budget * 2 + 2];
    let count = block[budget * 2 + 5];
    let offset = checked_usize(offset, "offset")?;
    let local_vocab = checked_usize(local_vocab, "local_vocab")?;
    let certified_count = checked_usize(count, "certified_count")?;
    Ok(LocalCandidates {
        rank,
        offset,
        local_vocab,
        candidates,
        own_value,
        local_max: block[budget * 2 + 3],
        omitted_exact_max: block[budget * 2 + 4],
        certified_count,
        certified_max_bound: block[budget * 2 + 6],
    })
}

fn checked_usize(value: f32, what: &str) -> Result<usize, MergeRefusal> {
    if value < 0.0 || value > TOKEN_ID_LIMIT as f32 || value.fract() != 0.0 {
        return Err(MergeRefusal::ShardLayout {
            position: 0,
            detail: format!("{what} {value} is not a non-negative f32-exact integer"),
        });
    }
    Ok(value as usize)
}

/// Merges every rank's contribution into the scored positions' logprobs.
///
/// Selection is the OpenAI-style top count: None asks only for the owning token's logprob and
/// rank; Some(n) asks for the n alternatives as well.
///
/// # Errors
///
/// A [MergeRefusal] when the payload is not provable: certified bounds, a malformed shard layout,
/// a non-finite exact value, no distribution, a missing owning token, or an order the candidate
/// union cannot prove at f32 precision.
pub fn merge(
    world: usize,
    selection: Option<usize>,
    own_tokens: &[u32],
    per_rank: &[Vec<RankPosition>],
) -> Result<Vec<MergedPosition>, MergeRefusal> {
    if per_rank.len() != world || world == 0 {
        return Err(MergeRefusal::RankLayout {
            detail: format!(
                "merge given {} rank rows for a world of {world}",
                per_rank.len()
            ),
        });
    }
    for (rank, rows) in per_rank.iter().enumerate() {
        if rows.len() != own_tokens.len() {
            return Err(MergeRefusal::RankLayout {
                detail: format!(
                    "rank {rank} reports {} positions for a {} token prompt",
                    rows.len(),
                    own_tokens.len()
                ),
            });
        }
    }
    let requested = selection.unwrap_or(0);
    let mut merged = Vec::with_capacity(own_tokens.len());
    for position in 0..own_tokens.len() {
        let rows: Vec<&RankPosition> = per_rank
            .iter()
            .map(|rank_rows| &rank_rows[position])
            .collect();
        let own_token = own_tokens[position];
        // 1. Certified bounds make the distribution unprovable: every logprob value would be an
        //    approximation of an unknown log-sum-exp, not the serial value.
        for row in &rows {
            if row.certified_count > 0 {
                return Err(MergeRefusal::CertifiedEntries {
                    position,
                    rank: row.rank,
                    certified_count: row.certified_count,
                });
            }
        }
        // 2. The shards must be one contiguous exact partition of the vocabulary.
        let mut covered = 0usize;
        let mut seen: HashMap<u32, f32> = HashMap::new();
        for (rank_index, row) in rows.iter().enumerate() {
            if row.rank != rank_index || row.local_vocab > TOKEN_ID_LIMIT {
                return Err(MergeRefusal::RankLayout {
                    detail: format!(
                        "rank row {rank_index} is labeled rank {} with {} shard rows",
                        row.rank, row.local_vocab
                    ),
                });
            }
            if row.offset != covered {
                return Err(MergeRefusal::ShardLayout {
                    position,
                    detail: format!(
                        "rank {} starts at token {} but the previous ranks cover up to {covered}",
                        row.rank, row.offset
                    ),
                });
            }
            covered = covered.saturating_add(row.local_vocab);
            if covered > TOKEN_ID_LIMIT {
                return Err(MergeRefusal::ShardLayout {
                    position,
                    detail: format!("the rank shards cover more than {TOKEN_ID_LIMIT} tokens"),
                });
            }
            for candidate in &row.candidates {
                if !candidate.value.is_finite() {
                    return Err(MergeRefusal::RankLayout {
                        detail: format!(
                            "rank {} reports a non-finite candidate {}",
                            row.rank, candidate.token_id
                        ),
                    });
                }
                let token = candidate.token_id as usize;
                if token < row.offset || token >= row.offset + row.local_vocab {
                    return Err(MergeRefusal::ShardLayout {
                        position,
                        detail: format!(
                            "rank {} candidate {} is outside its shard [{}, {})",
                            row.rank,
                            candidate.token_id,
                            row.offset,
                            row.offset + row.local_vocab
                        ),
                    });
                }
                if let Some(previous) = seen.insert(candidate.token_id, candidate.value) {
                    let detail = if previous.to_bits() == candidate.value.to_bits() {
                        format!("token {} appears in two rank shards", candidate.token_id)
                    } else {
                        format!(
                            "token {} appears twice with different values {previous} and {}",
                            candidate.token_id, candidate.value
                        )
                    };
                    return Err(MergeRefusal::ShardLayout { position, detail });
                }
            }
        }
        // 3. The owning token must have exactly one exact value in exactly one shard.
        let mut own_value = None;
        for row in &rows {
            if let Some(value) = row.own_value {
                if !value.is_finite() {
                    return Err(MergeRefusal::OwnTokenNonFinite {
                        position,
                        token_id: own_token,
                    });
                }
                if !(row.offset..row.offset + row.local_vocab).contains(&(own_token as usize)) {
                    return Err(MergeRefusal::ShardLayout {
                        position,
                        detail: format!(
                            "rank {} claims owning token {own_token} outside its shard",
                            row.rank
                        ),
                    });
                }
                if own_value.is_some() {
                    return Err(MergeRefusal::ShardLayout {
                        position,
                        detail: format!(
                            "owning token {own_token} appears in more than one rank shard"
                        ),
                    });
                }
                own_value = Some(value);
            }
        }
        let Some(own_value) = own_value else {
            return Err(MergeRefusal::OwnTokenMissing {
                position,
                token_id: own_token,
            });
        };
        // 4. Global maximum and the distributed log-sum-exp.
        let global_max = rows.iter().fold(f32::NEG_INFINITY, |best, row| {
            if row.local_max.is_finite() {
                best.max(row.local_max)
            } else {
                best
            }
        });
        if !global_max.is_finite() {
            return Err(MergeRefusal::NoDistribution { position });
        }
        let mut total = 0.0f64;
        for row in &rows {
            total += join_scaled_sum(row.scaled_sum_hi, row.scaled_sum_lo);
        }
        if !(total.is_finite() && total > 0.0) {
            return Err(MergeRefusal::NoDistribution { position });
        }
        let log_sum = total.ln();
        let own_logprob = logprob_of(own_value, global_max, log_sum);
        let greater: u64 = rows.iter().map(|row| u64::from(row.greater_than_own)).sum();
        let own_rank = u32::try_from(greater).unwrap_or(u32::MAX).saturating_add(1);
        // 5. The candidate union, ordered by the reported f32 logprob then token id.
        let mut union: Vec<(f32, u32)> = Vec::new();
        for row in &rows {
            for candidate in &row.candidates {
                union.push((
                    logprob_of(candidate.value, global_max, log_sum),
                    candidate.token_id,
                ));
            }
            if row.own_value.is_some() {
                union.push((own_logprob, own_token));
            }
        }
        union.sort_unstable_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
        union.dedup_by_key(|entry| entry.1);
        // 6. Prove the requested order at f32 precision before reporting it.
        if requested > 0 {
            let Some(boundary) = union.get(requested - 1) else {
                return Err(MergeRefusal::UnprovenOrder {
                    position,
                    requested,
                    boundary_logprob: None,
                    omitted_logprob: None,
                });
            };
            let mut omitted_raw = f32::NEG_INFINITY;
            for row in &rows {
                omitted_raw = omitted_raw
                    .max(row.omitted_exact_max)
                    .max(row.certified_max_bound);
            }
            let omitted_logprob = if omitted_raw.is_finite() {
                order_probe(omitted_raw, global_max, log_sum)
            } else {
                f32::NEG_INFINITY
            };
            let margin = order_margin(boundary.0);
            let next = union.get(requested).map(|entry| entry.0);
            let dominated = boundary.0 > omitted_logprob && boundary.0 - omitted_logprob > margin;
            let separated = next.is_none_or(|next| boundary.0 - next > margin);
            if !(dominated && separated) {
                return Err(MergeRefusal::UnprovenOrder {
                    position,
                    requested,
                    boundary_logprob: Some(boundary.0),
                    omitted_logprob: Some(omitted_logprob),
                });
            }
        }
        let alternatives = union
            .iter()
            .take(requested)
            .map(|(logprob, token_id)| MergedAlternative {
                token_id: *token_id,
                logprob: *logprob,
            })
            .collect();
        merged.push(MergedPosition {
            position,
            own_token_id: own_token,
            own_logprob,
            own_rank,
            alternatives,
        });
    }
    Ok(merged)
}

/// The omitted-value probe uses the same formula as every other value in the payload.
fn order_probe(value: f32, global_max: f32, log_sum: f64) -> f32 {
    logprob_of(value, global_max, log_sum)
}
