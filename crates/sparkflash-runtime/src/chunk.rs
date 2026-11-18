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

//! How many prompt rows one pass takes.
//!
//! [`ChunkPolicy`] is a service, not a method on an engine: the tick engine and the fused engine
//! ask it the same question, and a later engine changes only who calls it.
//!
//! The size is chosen from [`Limits`], the lane's bytes a row and the free memory reported at
//! boot, but the answer is quantised to [`CHUNK_SIZES`]. That quantisation is deliberate: a
//! chunk boundary can move the last bits of a pass (DeepSeek records exactly that dependence),
//! so the boundary must not move with the host's load. A policy never returns an arbitrary
//! number of rows; it steps down the documented list when memory is tight.
//!
//! The list below is the whole vocabulary. It covers the chunk sizes the released arms state
//! today: dense's 256, GLM's 4096, Inkling's 4096, 8192 and 65,535. The small sizes exist for
//! CPU fixtures and for models whose pass holds few rows; the top end is Inkling's whole-context
//! pass. An arm whose limit is below an entry gets the largest entry that fits, so its
//! conformance run uses the same documented value like any other.

use crate::layout::Limits;

/// The chunk sizes a [`ChunkPolicy`] may return, ascending. This is the documented set.
pub const CHUNK_SIZES: [usize; 12] = [
    1, 2, 4, 8, 64, 256, 1_024, 4_096, 8_192, 16_384, 32_768, 65_535,
];

/// The prompt chunk size one engine uses, chosen once from [`Limits`] and memory.
///
/// The policy is `Copy` and has no interior state: every engine that chooses the same inputs
/// gets the same number of rows, so a chunk boundary is reproducible offline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPolicy {
    rows: usize,
}

impl ChunkPolicy {
    /// The policy for `limits`, where one positional row occupies `bytes_per_row` and
    /// `free_bytes` are reported free. A zero `free_bytes` means "not measured": only the
    /// limits apply. A zero `bytes_per_row` also removes the memory term.
    ///
    /// The context does not lower the pass width: a prompt is at most its own length and the
    /// runtime clamps a chunk to it, so a short context still runs a prompt that fits as one
    /// whole pass (dense's 256 rows at a 128-position context, for example).
    #[must_use]
    pub fn new(limits: Limits, free_bytes: u64, bytes_per_row: usize) -> Self {
        let by_limits = limits.rows.max(1);
        let by_memory = if free_bytes > 0 && bytes_per_row > 0 {
            usize::try_from(free_bytes / bytes_per_row as u64)
                .unwrap_or(usize::MAX)
                .max(1)
        } else {
            usize::MAX
        };
        Self {
            rows: quantise(by_limits.min(by_memory)),
        }
    }

    /// The policy from the limits alone, as an arm with no memory figure states it.
    #[must_use]
    pub fn for_limits(limits: Limits) -> Self {
        Self::new(limits, 0, 0)
    }

    /// The rows one prompt pass takes.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// The documented sizes this policy chooses between.
    #[must_use]
    pub const fn sizes() -> &'static [usize] {
        &CHUNK_SIZES
    }
}

/// The largest documented chunk size at most `rows`, or `rows` itself below the smallest.
///
/// This is the whole "do not move with load" rule: [`CHUNK_SIZES`] is the only source of
/// chunk sizes, and the fallback is a limit, not a measurement.
#[must_use]
pub const fn quantise(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let mut chosen = rows;
    let mut index = 0;
    while index < CHUNK_SIZES.len() {
        let size = CHUNK_SIZES[index];
        if size <= rows {
            chosen = size;
        } else {
            break;
        }
        index += 1;
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_answer_is_a_documented_size_or_an_arms_own_limit() {
        for rows in 1..=70_000 {
            let chosen = quantise(rows);
            assert!(
                CHUNK_SIZES.contains(&chosen) || chosen == rows,
                "quantise({rows}) = {chosen}"
            );
            assert!(chosen <= rows);
        }
    }

    #[test]
    fn the_released_chunk_sizes_are_in_the_set() {
        for size in [256usize, 4_096, 8_192, 65_535] {
            assert_eq!(quantise(size), size, "{size} is a released chunk size");
        }
        assert_eq!(quantise(5_000), 4_096);
        assert_eq!(quantise(300), 256);
    }

    #[test]
    fn memory_steps_down_the_list_and_a_busy_host_cannot_pick_between_sizes() {
        let limits = Limits::new(65_535, 65_535);
        let all = ChunkPolicy::new(limits, u64::MAX, 1);
        assert_eq!(all.rows(), 65_535);
        // 4,000 bytes at 1 byte a row is between 1,024 and 4,096: the policy takes 1,024.
        assert_eq!(ChunkPolicy::new(limits, 4_000, 1).rows(), 1_024);
        // Free memory moving within a band does not move the size.
        assert_eq!(
            ChunkPolicy::new(limits, 4_000, 1).rows(),
            ChunkPolicy::new(limits, 4_095, 1).rows()
        );
        // Across a band it does.
        assert_eq!(ChunkPolicy::new(limits, 4_096, 1).rows(), 4_096);
    }

    #[test]
    fn a_context_below_the_chunk_size_does_not_split_a_prompt() {
        let dense = Limits::new(256, 128);
        assert_eq!(ChunkPolicy::for_limits(dense).rows(), 256);
    }

    #[test]
    fn limits_below_the_smallest_entry_use_the_limit() {
        let limits = Limits::new(4, 1_000);
        assert_eq!(ChunkPolicy::for_limits(limits).rows(), 4);
        assert_eq!(ChunkPolicy::new(limits, 1, 1).rows(), 1);
    }

    #[test]
    fn dense_keeps_its_256_rows() {
        let dense = Limits::new(256, 32_768);
        assert_eq!(ChunkPolicy::for_limits(dense).rows(), 256);
    }
}
