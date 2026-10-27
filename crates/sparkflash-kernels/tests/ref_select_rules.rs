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

//! The ref_select rules against the engine's host rules, on the CPU twin.
//!
//! The twin (crates/sparkflash-kernels/src/ref_select_ref.rs) is the kernel's arithmetic model; these
//! tests pin the rules the kernel must not drift from, with the engine's citing lines. The ids are
//! compared as an index vector, which is the comparison the device oracle test uses.

use sparkflash_engine::quant::bf16_round;
use sparkflash_kernels::ref_select_launch::{
    ARG_WORDS, BLOCK, MASK_WORDS, MAX_POSITIONS, declaration_matches, params,
};
use sparkflash_kernels::ref_select_ref::{
    MaskMode, SelectRequest, candidate_mask, mask_scores, scores, select, select_ids,
};

const NO_INDEX: u32 = 0xffff_ffff;
const OFFSET: u32 = 129;

#[allow(clippy::too_many_arguments)]
fn request<'a>(
    q: &'a [f32],
    keys: &'a [f32],
    weights: &'a [f32],
    n_pos: usize,
    heads: usize,
    head_dim: usize,
    reach: usize,
    topk: usize,
) -> SelectRequest<'a> {
    SelectRequest {
        q_index: q,
        keys,
        weights,
        n_pos,
        heads,
        head_dim,
        reach,
        topk,
        offset: OFFSET,
        mask_mode: MaskMode::None,
        topk_blocks: 1,
        block_size: 4,
        no_index: NO_INDEX,
        mask_in: &[],
    }
}

/// The engine's index_scores (attention.rs:1476-1506), transcribed independently of the twin.
fn engine_scores(
    q_index: &[f32],
    keys: &[f32],
    weights: &[f32],
    n_pos: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut score = vec![0.0f32; n_pos];
    for head in 0..heads {
        let q_head = &q_index[head * head_dim..(head + 1) * head_dim];
        for position in 0..n_pos {
            let key = &keys[position * head_dim..(position + 1) * head_dim];
            let mut acc = 0.0f32;
            for (q, k) in q_head.iter().zip(key) {
                acc += *q * *k;
            }
            let raw = bf16_round(acc);
            if raw > 0.0 {
                score[position] += bf16_round(raw * weights[head]);
            }
        }
    }
    score.iter().map(|value| bf16_round(*value)).collect()
}

fn stream(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn the_scores_are_the_engine_rule() {
    let (heads, head_dim, n_pos) = (4usize, 8usize, 13usize);
    let q = stream(heads * head_dim, 0x51ed, 1.0);
    let keys = stream(n_pos * head_dim, 0x2701, 1.0);
    let weights = stream(heads, 0x3333, 1.0);
    let request = request(&q, &keys, &weights, n_pos, heads, head_dim, n_pos, 3);
    let twin = scores(&request);
    let engine = engine_scores(&q, &keys, &weights, n_pos, heads, head_dim);
    for (position, (twin, engine)) in twin.iter().zip(&engine).enumerate() {
        assert_eq!(
            twin.to_bits(),
            engine.to_bits(),
            "score {position}: {twin} vs {engine}"
        );
    }
}

#[test]
fn a_tie_goes_to_the_lower_position() {
    let score = [1.0f32, 1.0, 1.0];
    let ids = select_ids(&score, 3, 2, OFFSET, NO_INDEX);
    assert_eq!(ids, vec![OFFSET, OFFSET + 1]);
}

#[test]
fn unreachable_positions_map_to_no_index() {
    let mut score = [5.0f32, 4.0, 3.0];
    mask_scores(&mut score, 2, None);
    assert_eq!(score[2], f32::NEG_INFINITY);
    let ids = select_ids(&score, 2, 3, OFFSET, NO_INDEX);
    assert_eq!(ids, vec![OFFSET, OFFSET + 1, NO_INDEX]);
}

#[test]
fn the_consumed_block_mask_hides_its_positions() {
    let mut mask = vec![0u32; MASK_WORDS];
    mask[0] = (1 << 0) | (1 << 3);
    let mut score = [5.0f32, 4.0, 3.0, 2.0];
    mask_scores(&mut score, 4, Some(&mask));
    assert!(score[1].is_infinite() && score[1].is_sign_negative());
    assert!(score[2].is_infinite() && score[2].is_sign_negative());
    let ids = select_ids(&score, 4, 2, OFFSET, NO_INDEX);
    assert_eq!(ids, vec![OFFSET, OFFSET + 3]);
}

#[test]
fn the_produced_block_mask_pins_the_newest_block_and_keeps_the_best() {
    // Two blocks of four; the newest reachable position (reach 6) is in block 1, which is pinned even
    // though its scores are the worst. topk_blocks = 1 keeps only that block.
    let score = [8.0f32, 7.0, 6.0, 5.0, 1.0, 0.0, -1.0, -2.0];
    let mask = candidate_mask(&score, 6, 1, 4);
    assert_eq!(
        mask[0], 0b1111_0000,
        "only the pinned newest block survives"
    );
}

#[test]
fn an_unreachable_block_that_is_kept_sets_no_bits() {
    // reach 3 -> the newest block is block 0. Keeping both blocks: block 1's best score is finite
    // here, so it is kept and its bits are set; make block 1 all -inf and it must be dropped.
    let mut score = [
        8.0f32,
        7.0,
        6.0,
        5.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let mask = candidate_mask(&score, 3, 2, 4);
    assert_eq!(mask[0], 0b0000_1111);
    score = [
        1.0f32,
        2.0,
        3.0,
        4.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let pinned = candidate_mask(&score, 3, 2, 4);
    assert_eq!(pinned[0], 0b0000_1111);
}

#[test]
fn the_top_k_is_truncated_to_n_pos() {
    let score = [3.0f32, 2.0, 1.0];
    let ids = select_ids(&score, 3, 10, OFFSET, NO_INDEX);
    assert_eq!(ids.len(), 3);
}

#[test]
fn one_whole_select_is_the_host_rule() {
    let (heads, head_dim, n_pos, reach, topk) = (2usize, 4usize, 8usize, 5usize, 3usize);
    let q = stream(heads * head_dim, 0x9c4f, 1.0);
    let keys = stream(n_pos * head_dim, 0x1a27, 1.0);
    let weights = stream(heads, 0x77b1, 1.0);
    let request = request(&q, &keys, &weights, n_pos, heads, head_dim, reach, topk);
    let output = select(&request).expect("a whole select");
    let mut engine = engine_scores(&q, &keys, &weights, n_pos, heads, head_dim);
    for value in engine.iter_mut().skip(reach) {
        *value = f32::NEG_INFINITY;
    }
    let mut order: Vec<usize> = (0..n_pos).collect();
    order.sort_by(|left, right| {
        engine[*right]
            .partial_cmp(&engine[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    order.truncate(topk);
    order.sort_unstable();
    let expected: Vec<u32> = order
        .into_iter()
        .map(|position| {
            if position < reach {
                position as u32 + OFFSET
            } else {
                NO_INDEX
            }
        })
        .collect();
    assert_eq!(
        output.ids, expected,
        "the ids are compared as an index vector"
    );
}

#[test]
fn the_request_refuses_a_position_past_the_score_table() {
    let q = vec![0.0f32; 4];
    let keys = vec![0.0f32; 4];
    let weights = vec![1.0f32];
    let request = request(&q, &keys, &weights, MAX_POSITIONS + 1, 1, 4, 1, 1);
    let error = select(&request).expect_err("a position past the table must refuse");
    assert!(error.contains("4096"), "{error}");
}

#[test]
fn the_declaration_and_launch_words_are_the_pinned_ones() {
    assert!(
        declaration_matches(),
        "the kernel declaration drifted from the launcher's pin"
    );
    assert_eq!(params().len(), ARG_WORDS);
    assert_eq!(BLOCK, (256, 1, 1));
    assert_eq!(MASK_WORDS, 128);
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/baseline/kernel_baseline.txt"
    ))
    .expect("the baseline manifest is in the crate");
    let line = manifest
        .lines()
        .find(|line| {
            line.starts_with(&format!(
                "kernel {} ",
                sparkflash_kernels::ref_select_launch::KERNEL
            ))
        })
        .expect("the manifest pins the reference select");
    assert!(
        line.contains(&format!(
            "cubin_bytes={}",
            sparkflash_kernels::ref_select_launch::CUBIN_BYTES
        )),
        "the launcher byte count is the manifest: {line}"
    );
    assert!(
        line.contains(&format!(
            "cubin_sha256={}",
            sparkflash_kernels::ref_select_launch::CUBIN_SHA256
        )),
        "the launcher digest is the manifest: {line}"
    );
}

#[test]
fn the_id_space_is_the_window_offset_plus_the_compressed_row() {
    // The device-select path's contract: ref_select applies the window_rows offset, so every id is
    // in [window_rows, window_rows + compressed_positions) and maps below the 512-row store. The
    // fill kernel assumes this and the host path's two refusals are reference-path checks only.
    let window_rows = 128usize;
    let ratio = 4usize;
    let compressed_positions = 512 / ratio;
    let score = vec![1.0f32; compressed_positions];
    let ids = select_ids(
        &score,
        compressed_positions,
        compressed_positions,
        window_rows as u32,
        NO_INDEX,
    );
    assert_eq!(ids.len(), compressed_positions);
    for (compressed_row, id) in ids.iter().enumerate() {
        assert_eq!(*id, window_rows as u32 + compressed_row as u32);
        assert!(*id >= window_rows as u32, "never below the window offset");
        assert!(
            ((*id - window_rows as u32) as usize) * ratio < 512,
            "expanded row {id} must stay inside the store"
        );
    }
}
