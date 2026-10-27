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

//! Host reference for the certified head's device reduction (kernels/lm_head_reduce_sm121.cu).
//!
//! This is the host assembly the c4c chain performs today, written as a pure function so the
//! device reduction can be compared with it entry for entry:
//!
//!   * the 128 blocks are the m16 layout, rank-major: global block r * 32 + b covers rank r's
//!     entries [b * 1024, min((b + 1) * 1024, 32,320)), so the last block of each rank has 576
//!     entries;
//!   * each block's answer is the maximum stored value and its global entry, with ties going to the
//!     lowest global entry index (the certified screen's own rule);
//!   * the 129,280-way answer folds the 128 block answers in ascending global block order under the
//!     same rule, which is the same sequence as the chain's consider_head_block over the four rank
//!     shards;
//!   * the top-5 is sorted by value descending and then by ascending global index, over the stored
//!     values (a certified entry stores its bound, so the set is only CLAIMED when no entry was
//!     certified, mirroring m16's claims_top5 rule without the host reference condition);
//!   * a certified argmax entry makes the token not claimable and is reported by
//!     STATUS_WINNER_CERTIFIED instead of being asserted away.
//!
//! No arithmetic is performed on a logit: the reference compares and moves f32 values exactly as
//! the kernel does, so the comparison is bit-for-bit, not a tolerance.

use crate::lm_head_reduce_launch::{
    BLOCKS, CHUNK, CHUNKS_PER_RANK, RANKS, STATUS_ANY_CERTIFIED, STATUS_TOP5_CLAIMED,
    STATUS_WINNER_CERTIFIED, TOP, VOCAB_PER_RANK,
};

/// The running maximum a scan starts from: -FLT_MAX, the value the certified screen's per-thread
/// scan uses, so an all--FLT_MAX input resolves to global entry 0 exactly as the host does.
const MIN: f32 = -f32::MAX;

/// The device reduction's report, in the same order and encoding as the kernel's output buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    /// The global argmax entry; diagnostic when STATUS_WINNER_CERTIFIED is set.
    pub token: u32,
    /// The stored value at that entry (a certificate bound when the entry is certified).
    pub value: f32,
    /// The 128 block maxima, rank-major.
    pub block_max: Vec<f32>,
    /// The 128 block argmax entries, global indices.
    pub block_argmax: Vec<u32>,
    /// The five largest stored values, descending.
    pub top5_value: [f32; TOP],
    /// The five global entries, descending by value then ascending by index.
    pub top5_index: [u32; TOP],
    /// The pinned status bits.
    pub status: u32,
    /// The total number of certified flags over the four shards.
    pub certified_count: u32,
}

/// Whether (value, index) ranks before (held, at): greater value wins, equal value goes to the
/// lower global entry.
#[must_use]
pub fn better(value: f32, index: u32, held: f32, at: u32) -> bool {
    value > held || (value == held && index < at)
}

/// Inserts (value, index) into a descending top-5, value then ascending index.
pub fn top5_insert(value: f32, index: u32, top_value: &mut [f32; TOP], top_index: &mut [u32; TOP]) {
    if top_index.contains(&index) {
        return;
    }
    for slot in 0..TOP {
        if better(value, index, top_value[slot], top_index[slot]) {
            for move_slot in (slot + 1..TOP).rev() {
                top_value[move_slot] = top_value[move_slot - 1];
                top_index[move_slot] = top_index[move_slot - 1];
            }
            top_value[slot] = value;
            top_index[slot] = index;
            return;
        }
    }
}

/// The 128 block maxima and argmax entries, rank-major, over the four shard buffers.
///
/// # Errors
/// Refuses a shard whose length is not the released per-rank vocabulary.
pub fn blocks(logits: &[Vec<f32>], certified: &[Vec<u8>]) -> Result<(Vec<f32>, Vec<u32>), String> {
    validate(logits, certified)?;
    let mut block_max = vec![0.0f32; BLOCKS];
    let mut block_argmax = vec![0u32; BLOCKS];
    for (rank, shard) in logits.iter().enumerate() {
        for block in 0..CHUNKS_PER_RANK {
            let global_block = rank * CHUNKS_PER_RANK + block;
            let base = block * CHUNK;
            let count = CHUNK.min(VOCAB_PER_RANK - base);
            let mut value = MIN;
            let mut entry = u32::MAX;
            for local in 0..count {
                let index = (rank * VOCAB_PER_RANK + base + local) as u32;
                let candidate = shard[base + local];
                if better(candidate, index, value, entry) {
                    value = candidate;
                    entry = index;
                }
            }
            block_max[global_block] = value;
            block_argmax[global_block] = entry;
        }
    }
    Ok((block_max, block_argmax))
}

/// Assembles the token, the block arrays, the top-5 and the status word from the four shard
/// buffers, exactly as the kernel does.
///
/// # Errors
/// Refuses a wrong shard count, a wrong shard length, or a flag that is not 0 or 1.
pub fn assemble(logits: &[Vec<f32>], certified: &[Vec<u8>]) -> Result<Reduction, String> {
    validate(logits, certified)?;
    let (block_max, block_argmax) = blocks(logits, certified)?;
    let mut value = MIN;
    let mut token = u32::MAX;
    for block in 0..BLOCKS {
        if better(block_max[block], block_argmax[block], value, token) {
            value = block_max[block];
            token = block_argmax[block];
        }
    }
    let certified_count = certified
        .iter()
        .map(|flags| flags.iter().filter(|flag| **flag == 1).count())
        .sum::<usize>() as u32;
    let winner_certified = certified
        .get(token as usize / VOCAB_PER_RANK)
        .and_then(|flags| flags.get(token as usize % VOCAB_PER_RANK))
        .is_some_and(|flag| *flag == 1);

    let mut top5_value = [MIN; TOP];
    let mut top5_index = [u32::MAX; TOP];
    for (rank, shard) in logits.iter().enumerate() {
        for (local, candidate) in shard.iter().enumerate() {
            let index = (rank * VOCAB_PER_RANK + local) as u32;
            top5_insert(*candidate, index, &mut top5_value, &mut top5_index);
        }
    }

    let mut status = 0u32;
    if certified_count != 0 {
        status |= STATUS_ANY_CERTIFIED;
    }
    if winner_certified {
        status |= STATUS_WINNER_CERTIFIED;
    } else if certified_count == 0 {
        status |= STATUS_TOP5_CLAIMED;
    }
    Ok(Reduction {
        token,
        value,
        block_max,
        block_argmax,
        top5_value,
        top5_index,
        status,
        certified_count,
    })
}

/// The token the chain's block-based host assembly produces from the same buffers: the block arrays
/// are folded with consider_head_block, i.e. greater value, then lower global entry.
///
/// # Errors
/// Propagates blocks' validation.
pub fn host_assembly_token(
    logits: &[Vec<f32>],
    certified: &[Vec<u8>],
) -> Result<(u32, f32), String> {
    let (block_max, block_argmax) = blocks(logits, certified)?;
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0u32;
    for (block, value) in block_max.iter().enumerate() {
        let global = block_argmax[block];
        if *value > best_value || (*value == best_value && global < best_index) {
            best_value = *value;
            best_index = global;
        }
    }
    Ok((best_index, best_value))
}

fn validate(logits: &[Vec<f32>], certified: &[Vec<u8>]) -> Result<(), String> {
    if logits.len() != RANKS || certified.len() != RANKS {
        return Err(format!(
            "the certified head writes {} shards, got {} logits and {} certified buffers",
            RANKS,
            logits.len(),
            certified.len()
        ));
    }
    for rank in 0..RANKS {
        if logits[rank].len() != VOCAB_PER_RANK {
            return Err(format!(
                "rank {rank} has {} logits, the released per-rank vocabulary is {VOCAB_PER_RANK}",
                logits[rank].len()
            ));
        }
        if certified[rank].len() != VOCAB_PER_RANK {
            return Err(format!(
                "rank {rank} has {} certified flags, expected {VOCAB_PER_RANK}",
                certified[rank].len()
            ));
        }
        if let Some((entry, flag)) = certified[rank]
            .iter()
            .enumerate()
            .find(|(_, flag)| **flag > 1)
        {
            return Err(format!(
                "rank {rank} entry {entry} has certified flag {flag}: the screen writes 0 or 1"
            ));
        }
    }
    Ok(())
}

/// A deterministic pseudo-random f32 buffer with a mild distribution, for the tests.
#[cfg(test)]
fn lcg_values(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
        values.push(unit - 0.5);
    }
    values
}

/// The tests' brute-force argmax over all 129,280 entries, lowest index on ties.
#[cfg(test)]
fn brute_token(logits: &[Vec<f32>]) -> (u32, f32) {
    let mut value = MIN;
    let mut token = u32::MAX;
    for (rank, shard) in logits.iter().enumerate() {
        for (local, candidate) in shard.iter().enumerate() {
            let index = (rank * VOCAB_PER_RANK + local) as u32;
            if better(*candidate, index, value, token) {
                value = *candidate;
                token = index;
            }
        }
    }
    (token, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm_head_reduce_launch::VOCAB;

    fn buffers(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
        let logits = (0..RANKS)
            .map(|rank| lcg_values(seed + rank as u64, VOCAB_PER_RANK))
            .collect::<Vec<_>>();
        let certified = (0..RANKS).map(|_| vec![0u8; VOCAB_PER_RANK]).collect();
        (logits, certified)
    }

    #[test]
    fn the_block_arrays_are_the_m16_layout() {
        let (logits, certified) = buffers(7);
        let (block_max, block_argmax) = blocks(&logits, &certified).expect("blocks");
        assert_eq!(block_max.len(), BLOCKS);
        assert_eq!(block_argmax.len(), BLOCKS);
        for (rank, shard) in logits.iter().enumerate() {
            for block in 0..CHUNKS_PER_RANK {
                let global = rank * CHUNKS_PER_RANK + block;
                let base = block * CHUNK;
                let count = CHUNK.min(VOCAB_PER_RANK - base);
                let mut expected_value = MIN;
                let mut expected_entry = u32::MAX;
                for local in 0..count {
                    let index = (rank * VOCAB_PER_RANK + base + local) as u32;
                    let candidate = shard[base + local];
                    if better(candidate, index, expected_value, expected_entry) {
                        expected_value = candidate;
                        expected_entry = index;
                    }
                }
                assert_eq!(block_max[global].to_bits(), expected_value.to_bits());
                assert_eq!(block_argmax[global], expected_entry);
                // The partial last block is the only one with 576 entries.
                if block == CHUNKS_PER_RANK - 1 {
                    assert_eq!(count, 576);
                }
            }
        }
    }

    #[test]
    fn the_token_is_the_brute_force_argmax_and_the_lowest_index_on_a_tie() {
        let (mut logits, certified) = buffers(11);
        let brute = brute_token(&logits);
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(
            (reduction.token, reduction.value.to_bits()),
            (brute.0, brute.1.to_bits())
        );
        assert_eq!(
            host_assembly_token(&logits, &certified).expect("host assembly"),
            (brute.0, brute.1)
        );

        // A tie across ranks: entry 100 in rank 0 and 100 in rank 3, same value.
        for values in &mut logits {
            values.fill(-100.0);
        }
        logits[0][100] = 5.0;
        logits[3][100] = 5.0;
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(reduction.token, 100);
        assert_eq!(reduction.value.to_bits(), 5.0f32.to_bits());
        let block = 100 / CHUNK;
        assert_eq!(reduction.block_argmax[block], 100);
    }

    #[test]
    fn the_top5_is_value_descending_then_index_ascending() {
        let (mut logits, certified) = buffers(13);
        for values in &mut logits {
            values.fill(-100.0);
        }
        let ties = [(0usize, 1000usize), (0, 2000), (0, 500), (1, 1000), (3, 90)];
        for (rank, local) in ties {
            logits[rank][local] = 5.0;
        }
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(
            reduction.top5_index,
            [500, 1_000, 2_000, 32_320 + 1_000, 3 * 32_320 + 90]
        );
        for value in reduction.top5_value {
            assert_eq!(value.to_bits(), 5.0f32.to_bits());
        }
        assert_eq!(reduction.status, STATUS_TOP5_CLAIMED);
    }

    #[test]
    fn the_top5_insertion_stores_an_entry_once_and_orders_ties_by_index() {
        let mut value = [MIN; TOP];
        let mut index = [u32::MAX; TOP];
        top5_insert(5.0, 7, &mut value, &mut index);
        top5_insert(5.0, 7, &mut value, &mut index);
        top5_insert(5.0, 32_325, &mut value, &mut index);
        assert_eq!(index, [7, 32_325, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(value[0].to_bits(), 5.0f32.to_bits());
        assert_eq!(value[1].to_bits(), 5.0f32.to_bits());
        // An equal value at a lower global index sorts before the held entry, once.
        top5_insert(5.0, 3, &mut value, &mut index);
        assert_eq!(index, [3, 7, 32_325, u32::MAX, u32::MAX]);
        top5_insert(5.0, 3, &mut value, &mut index);
        assert_eq!(index, [3, 7, 32_325, u32::MAX, u32::MAX]);
    }

    #[test]
    fn the_certified_count_sums_four_different_rank_counts() {
        let (mut logits, mut certified) = buffers(29);
        for values in &mut logits {
            values.fill(-100.0);
        }
        logits[0][1_000] = 50.0;
        let per_rank = [3usize, 5, 7, 11];
        for (rank, count) in per_rank.iter().enumerate() {
            for flag in &mut certified[rank][..*count] {
                *flag = 1;
            }
        }
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(reduction.certified_count, 26);
        assert_eq!(reduction.status, STATUS_ANY_CERTIFIED);
        assert_eq!(reduction.status & STATUS_WINNER_CERTIFIED, 0);
        assert_eq!(reduction.token, 1_000);
    }

    #[test]
    fn a_certified_bound_tying_the_exact_max_at_a_lower_index_refuses() {
        let (mut logits, mut certified) = buffers(37);
        for values in &mut logits {
            values.fill(-100.0);
        }
        logits[0][100] = 5.0;
        logits[0][200] = 5.0;
        certified[0][100] = 1;
        let reduction = assemble(&logits, &certified).expect("assemble");
        // The stored-value order is lowest index on a tie; entry 100 is a bound, so the token is
        // diagnostic and the reduce must not claim it as exact.
        assert_eq!(reduction.token, 100);
        assert_eq!(reduction.value.to_bits(), 5.0f32.to_bits());
        assert_eq!(reduction.certified_count, 1);
        assert_eq!(
            reduction.status,
            STATUS_WINNER_CERTIFIED | STATUS_ANY_CERTIFIED
        );
        assert_eq!(reduction.status & STATUS_TOP5_CLAIMED, 0);
        assert_eq!(
            host_assembly_token(&logits, &certified).expect("host assembly"),
            (100, 5.0)
        );
    }

    #[test]
    fn a_certified_winner_refuses_the_token_and_the_top5() {
        let (mut logits, mut certified) = buffers(17);
        for values in &mut logits {
            values.fill(-100.0);
        }
        logits[0][4_242] = 9.0;
        certified[0][4_242] = 1;
        logits[0][777] = 1.0;
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(reduction.token, 4_242);
        assert_eq!(reduction.certified_count, 1);
        assert_eq!(
            reduction.status,
            STATUS_WINNER_CERTIFIED | STATUS_ANY_CERTIFIED
        );
        assert_eq!(reduction.status & STATUS_TOP5_CLAIMED, 0);
    }

    #[test]
    fn a_certified_field_with_an_exact_winner_claims_the_token_but_not_the_top5() {
        let (mut logits, mut certified) = buffers(19);
        for values in &mut logits {
            values.fill(-100.0);
        }
        let winner = 33_865u32;
        let winner_local = winner as usize % VOCAB_PER_RANK;
        let winner_rank = winner as usize / VOCAB_PER_RANK;
        logits[winner_rank][winner_local] = 20.0;
        // The concentrated case's per-rank flag counts: 1,062 + 1,205 + 1,215 + 1,417 = 4,899.
        let per_rank = [1_062usize, 1_205, 1_215, 1_417];
        let mut total = 0usize;
        for (rank, count) in per_rank.iter().enumerate() {
            for flag in &mut certified[rank][..*count] {
                *flag = 1;
                total += 1;
            }
        }
        assert_eq!(total, 4_899);
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(reduction.token, winner);
        assert_eq!(reduction.value.to_bits(), 20.0f32.to_bits());
        assert_eq!(reduction.certified_count, 4_899);
        assert_eq!(reduction.status, STATUS_ANY_CERTIFIED);
        assert_eq!(reduction.status & STATUS_TOP5_CLAIMED, 0);
        assert_eq!(reduction.status & STATUS_WINNER_CERTIFIED, 0);
        assert_eq!(reduction.top5_index[0], winner);
        assert_eq!(
            host_assembly_token(&logits, &certified).expect("host assembly"),
            (winner, 20.0)
        );
    }

    #[test]
    fn the_all_exact_control_claims_the_top5() {
        let (logits, certified) = buffers(23);
        let reduction = assemble(&logits, &certified).expect("assemble");
        assert_eq!(reduction.certified_count, 0);
        assert_eq!(reduction.status, STATUS_TOP5_CLAIMED);
        assert_eq!(reduction.top5_index[0], brute_token(&logits).0);
        assert_eq!(
            reduction.block_max.len(),
            BLOCKS,
            "the report carries the m16 block arrays"
        );
    }

    #[test]
    fn the_reference_refuses_a_wrong_geometry() {
        let mut logits = vec![vec![0.0f32; VOCAB_PER_RANK]; 3];
        let certified = vec![vec![0u8; VOCAB_PER_RANK]; 3];
        assert!(assemble(&logits, &certified).is_err());
        logits.push(vec![0.0f32; VOCAB_PER_RANK]);
        let mut certified = certified;
        certified.push(vec![0u8; VOCAB_PER_RANK]);
        assert!(assemble(&logits, &certified).is_ok());
        certified[0][0] = 2;
        assert!(assemble(&logits, &certified).is_err());
        logits[0].push(0.0);
        logits[0].pop();
        certified[0][0] = 0;
        logits[1].pop();
        assert!(assemble(&logits, &certified).is_err());
        assert_eq!(VOCAB, RANKS * VOCAB_PER_RANK);
    }

    #[test]
    fn the_host_assembly_token_is_the_full_scan_on_pseudo_random_buffers() {
        for seed in [31u64, 32, 33, 34] {
            let (logits, certified) = buffers(seed);
            let brute = brute_token(&logits);
            assert_eq!(
                host_assembly_token(&logits, &certified).expect("host assembly"),
                (brute.0, brute.1)
            );
            assert_eq!(
                assemble(&logits, &certified).expect("assemble").token,
                brute.0
            );
        }
    }
}
