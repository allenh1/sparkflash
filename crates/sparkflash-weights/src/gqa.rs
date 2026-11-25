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

//! Grouped-query attention on the host: Hugging Face's `rotate_half` RoPE and causal softmax
//! attention of one row's query heads over a key/value cache.
//!
//! The host paths that run a grouped-query block (the dynamic host serve path and the DeepSeek
//! reference's grouped-query layers) share this arithmetic; each keeps its own projections, norms
//! and output gate.

use crate::linalg::softmax;

/// Hugging Face's `rotate_half` RoPE over `heads` heads of `head_dim` floats at `position`:
/// dimension `i` pairs with `i + rotated / 2` inside each head, and exactly the first `rotated`
/// dimensions turn. Pair `i` turns at `1 / theta^(2i / rotated) / scaling` a position.
///
/// `rotated` below 2 rotates nothing; the caller refuses an odd `rotated` or one wider than a
/// head before it gets here.
pub fn rotate_half(
    values: &mut [f32],
    (heads, head_dim): (usize, usize),
    rotated: usize,
    (theta, scaling): (f32, f32),
    position: usize,
) {
    if rotated < 2 {
        return;
    }
    let half = rotated / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..half {
            let frequency = 1.0 / theta.powf((2 * pair) as f32 / rotated as f32) / scaling;
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let left = values[base + pair];
            let right = values[base + pair + half];
            values[base + pair] = left * cos - right * sin;
            values[base + pair + half] = left * sin + right * cos;
        }
    }
}

/// One row's attention: each of `heads` query heads reads its group's key/value head over every
/// cached position (`keys`/`values` are `[positions][kv_heads * head_dim]`), softmax at
/// `1 / sqrt(head_dim)`. Returns the attended heads, `[heads * head_dim]`, ungated.
#[must_use]
pub fn attend(
    queries: &[f32],
    (keys, values): (&[f32], &[f32]),
    (heads, kv_heads, head_dim): (usize, usize, usize),
) -> Vec<f32> {
    let kv_width = kv_heads * head_dim;
    let positions = keys.len().checked_div(kv_width).unwrap_or(0);
    let groups = heads / kv_heads.max(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; heads * head_dim];
    for head in 0..heads {
        let kv_head = head / groups;
        let query = &queries[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(positions);
        for position in 0..positions {
            let key = &keys[position * kv_width + kv_head * head_dim
                ..position * kv_width + (kv_head + 1) * head_dim];
            scores.push(
                query
                    .iter()
                    .zip(key)
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale,
            );
        }
        softmax(&mut scores);
        let target = &mut out[head * head_dim..(head + 1) * head_dim];
        for (position, probability) in scores.iter().enumerate() {
            let value = &values[position * kv_width + kv_head * head_dim
                ..position * kv_width + (kv_head + 1) * head_dim];
            for (slot, value) in target.iter_mut().zip(value) {
                *slot += probability * value;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_turns_nothing_and_one_cached_row_is_its_value() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        rotate_half(&mut row, (1, 4), 4, (10_000.0, 1.0), 0);
        assert_eq!(row, vec![1.0, 2.0, 3.0, 4.0]);
        // Two query heads share one key/value head: one cached row attends to itself whole.
        let out = attend(
            &[1.0, 0.0, 0.0, 1.0],
            (&[0.5, 0.5], &[3.0, -2.0]),
            (2, 1, 2),
        );
        assert_eq!(out, vec![3.0, -2.0, 3.0, -2.0]);
    }

    #[test]
    fn a_turn_keeps_each_pair_length() {
        let mut row = vec![3.0, 0.0, 4.0, 1.0];
        rotate_half(&mut row, (1, 4), 4, (10_000.0, 1.0), 7);
        let first = (row[0] * row[0] + row[2] * row[2]).sqrt();
        let second = (row[1] * row[1] + row[3] * row[3]).sqrt();
        assert!((first - 5.0).abs() < 1e-5 && (second - 1.0).abs() < 1e-5);
    }
}
