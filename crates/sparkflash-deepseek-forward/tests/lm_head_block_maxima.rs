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

//! The certified head's block maxima are derived at the plan's hidden width.
//!
//! The defect this test pins: `lm_head_ref::block_maxima_bf16` divided a rank's head shard by the
//! released model's HIDDEN (5,120) instead of the plan's width. The synthetic second shape is
//! 4,096 wide, so every rank's shard ended 2,048 / 1,024 / 3,072 values past a 5,120-wide row
//! boundary at worlds 1 / 2 / 4 and the derivation panicked at tail open, before any kernel
//! launched. The width is a plan fact; the derivation now states it and the tail passes it.

use std::path::PathBuf;

use sparkflash_deepseek_forward::{plan, tp4_tail};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::bf16_bits_to_f32;
use sparkflash_model_deepseek_v41::config::ModelConfig;

fn second_shape() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/second-shape")
}

/// A deterministic BF16 byte image, `hidden` values a row, with a spread of mantissas and signs so
/// a block's maximum is not an accident of the fixture.
fn image(rows: usize, hidden: usize, seed: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * hidden * 2);
    let mut state = seed;
    for index in 0..rows * hidden {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Near 1.0: exponent 0x7f, mantissa from the state, sign from another bit.
        let mut bits = 0x3f80u16 | ((state >> 45) as u16 & 0x007f);
        if (state >> 60) & 1 == 1 {
            bits |= 0x8000;
        }
        // Enough variation inside a block that a max taken of the wrong width would differ.
        bits ^= ((index as u16) & 0x0007) << 4;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// The derivation's rule stated independently: one mask-and-maximum scan per 128-value block.
fn expected_maxima(image: &[u8], rows: usize, hidden: usize) -> Vec<u8> {
    let blocks = hidden / lm_head_ref::SCREEN_TILE;
    let mut expected = vec![0u8; rows * blocks * 2];
    for row in 0..rows {
        for block in 0..blocks {
            let mut best_bits = 0u16;
            let mut best = 0.0f32;
            for i in 0..lm_head_ref::SCREEN_TILE {
                let at = (row * hidden + block * lm_head_ref::SCREEN_TILE + i) * 2;
                let bits = u16::from_le_bytes([image[at], image[at + 1]]);
                let magnitude = bf16_bits_to_f32(bits).abs();
                if magnitude > best {
                    best = magnitude;
                    best_bits = bits & 0x7fff;
                }
            }
            let at = (row * blocks + block) * 2;
            expected[at..at + 2].copy_from_slice(&best_bits.to_le_bytes());
        }
    }
    expected
}

#[test]
fn the_second_shapes_block_maxima_are_derived_at_the_plans_hidden_width() {
    let dir = second_shape();
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the fixture config");
    let config = ModelConfig::from_json(&text).expect("the fixture parses");
    for world in [1usize, 2, 4] {
        let plan = plan::plan_from_path(&dir, world, 0)
            .unwrap_or_else(|error| panic!("world {world}: {error}"));
        plan::install(plan).unwrap_or_else(|error| panic!("world {world}: {error}"));
        let hidden = plan::hidden();
        assert_eq!(hidden, config.dim, "world {world}: the plan's hidden width");
        assert_ne!(
            hidden,
            lm_head_ref::HIDDEN,
            "world {world}: the fixture is a second shape"
        );
        let per_rank = plan::vocab_per_rank();
        assert_eq!(per_rank, config.vocab_size / world, "world {world}: shard");

        // TailStage::open reads one rank's shard as vocabulary rows at the plan's hidden width,
        // two BF16 bytes a value, and expects the derivation to fill exactly the buffer it
        // allocated. Both lengths are plan facts, not released constants.
        let shard_bytes = per_rank * hidden * 2;
        assert_eq!(
            shard_bytes / (hidden * 2),
            per_rank,
            "world {world}: the shard is a whole number of plan-wide rows"
        );
        assert_eq!(
            per_rank * lm_head_ref::blocks_per_row(hidden) * 2,
            tp4_tail::w_block_max_bytes(),
            "world {world}: the derivation's output is the tail's buffer"
        );

        // Run the derivation over the same row-length rule. A width that reverted to the
        // released constant would not divide this shard into rows, and would panic here.
        let rows = 8;
        let shard = image(rows, hidden, 11 + world as u64);
        assert_eq!(shard.len(), rows * hidden * 2);
        let maxima = lm_head_ref::block_maxima_bf16(&shard, hidden);
        assert_eq!(
            maxima.len(),
            rows * (hidden / lm_head_ref::SCREEN_TILE) * 2,
            "world {world}: one maximum per screen block and row"
        );
        assert_eq!(
            maxima,
            expected_maxima(&shard, rows, hidden),
            "world {world}: the maxima are the block maxima of the plan-wide rows"
        );
    }
}

#[test]
fn the_released_width_still_derives_the_layout_it_always_did() {
    assert_eq!(
        lm_head_ref::blocks_per_row(lm_head_ref::HIDDEN),
        lm_head_ref::LM_BLOCKS,
        "the released width is still 40 blocks a row"
    );
    let rows = 8;
    let shard = image(rows, lm_head_ref::HIDDEN, 23);
    let maxima = lm_head_ref::block_maxima_bf16(&shard, lm_head_ref::HIDDEN);
    assert_eq!(maxima.len(), rows * lm_head_ref::LM_BLOCKS * 2);
    assert_eq!(maxima, expected_maxima(&shard, rows, lm_head_ref::HIDDEN));
}

#[test]
#[should_panic(expected = "whole number of")]
fn a_width_that_does_not_tile_by_the_screen_tile_is_refused() {
    // The certified kernel static-asserts the same rule: a width that is not a whole number of
    // 128-value blocks has no CERT_BLOCKS to lay the maxima out by.
    let _ = lm_head_ref::block_maxima_bf16(&[], 5130);
}
