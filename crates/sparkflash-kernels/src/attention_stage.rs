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

//! stage_attention as a pure function: the operands the body takes, validated, nothing copied.
//!
//! CONTRACT DECISION (lead, this slice): the stage stays and becomes validation-only. It launches
//! nothing and copies nothing. With the store native (compressed_append_sm121) and the ring
//! allocated to max_positions, the sequence entry reads the window contiguously from a
//! base-pointer offset, so the old window copy has nothing left to do; and since the ring rotation
//! is the only thing that could bring a copy back, the stage is the seam where the D49 134-slot
//! ring would become real body again.
//!
//! What the stage does own is the operand validation, because that is where a plausible wrong
//! answer would otherwise hide: a ring larger than its allocation, a position past the rows the
//! ring holds, a cursor word that disagrees with the store, and the two selected-id refusals (an
//! id below the window offset, an id that expands past the store). The five checks are named in
//! [StageError] and each has a test; the assembly that follows them is
//! window_rows = min(position + 1, WINDOW_ROWS), window_start = position + 1 - window_rows,
//! store_zero = (cursor == 0), and the expanded-id mapping of
//! [compressed_cache_ref::selected_rows]. The window INCLUDES the current token (the released
//! engine rule is filled = min(position + 1, window); m20's window_count counts the current row),
//! so the append must precede the stage: at position 0 the union is exactly the current row, and a
//! stage run before its append is refused instead of assembled sink-only.
//!
//! Reading of the five conditions, stated because the contract row is prose: a ring that reports
//! more rows than max_positions is refused (ring_rows == max_positions is the full ring, the normal
//! case), a stage run before its append is refused because the window includes the current row (the
//! normal state is ring_rows == position + 1, position being the 0-based token index p), the
//! selected ids are mapped by selected_rows with its two refusals, and store_zero is asserted
//! against the cursor word the caller read, not assumed.
//!
//! CONVENTION, stated because the shared word is where the silent failure lives: [assemble]'s
//! position is the 0-based token index p, the same word the body launch takes
//! (FlashMlaSeqParams.position), NOT the token count p + 1. Both window words are derived from p
//! inside the module through [window_span], so the body's position and window_rows words cannot
//! disagree. A caller that passes the count p + 1 is refused (p + 1 > ring_rows when the ring holds
//! p + 1 rows) instead of silently getting window_rows = p; [window_span] is pinned against the
//! engine's window_indices (attention.rs:945), not against its own arithmetic. Append-before-read is
//! a requirement for this reason: the released window includes the current row, so the ring must
//! hold it before the body reads.

use core::fmt;

use crate::compressed_cache_ref as cache;

/// The window the engine publishes: window_rows = min(position + 1, 128), current token included.
pub const WINDOW_ROWS: usize = 128;

/// The operands attention_body takes, after validation. No buffer, no offset into a copy: the body
/// reads the ring at window_start and the store at the expanded ids in selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageOperands {
    /// Rows the ring holds at this position.
    pub ring_rows: usize,
    /// The 0-based token index p, the body launch's position word.
    pub position: usize,
    /// First ring row the window reads (the base-pointer offset). This is NOT the selected-id base:
    /// the id space is offset by the window count, see the selected field.
    pub window_start: usize,
    /// Window rows: min(position + 1, [WINDOW_ROWS]), the current token included. This count is also
    /// the selected-id base: ids are in the compressed-position space offset by this, not by the
    /// ring row offset.
    pub window_rows: usize,
    /// Compressed rows the store has published.
    pub store_rows: usize,
    /// The body store_zero skip rule: true while no compressed row exists.
    pub store_zero: bool,
    /// Compressed rows the store can hold at this ratio.
    pub compressed_rows: usize,
    /// The layer compress ratio.
    pub ratio: usize,
    /// Expanded store row ids, in the select order, NO_INDEX skipped. The ids are in the
    /// compressed-position space offset by the window COUNT: id = window_rows + j, where j is the
    /// compressed row (tp4_prefill stages id - seam.window_rows to the store row, and
    /// compressed_selected_fill_sm121's base argument is the same count). window_start is the ring
    /// row offset the body reads, not this base; switching the base to window_start would shift
    /// every id by up to 128 as the window slides, so it is refused instead of absorbed.
    pub selected: Vec<u32>,
}

/// Why the stage refused to assemble operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    /// The ring reports more rows than the allocation bound.
    RingPastCapacity {
        /// Rows the ring reports.
        ring_rows: usize,
        /// The allocation bound.
        max_positions: usize,
    },
    /// The position has no row yet: the append must precede the stage.
    PositionPastRing {
        /// The position the stage runs at.
        position: usize,
        /// Rows the ring holds.
        ring_rows: usize,
    },
    /// The cursor word and the store disagree: store_zero would be a guess.
    CursorMismatch {
        /// The cursor word the caller read.
        cursor: u32,
        /// Rows the store says it has published.
        rows_written: usize,
    },
    /// A selected id or a store row was refused by the mapping.
    Cache(cache::CacheError),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RingPastCapacity {
                ring_rows,
                max_positions,
            } => {
                write!(
                    f,
                    "ring reports {ring_rows} rows, the allocation is {max_positions}"
                )
            }
            Self::PositionPastRing {
                position,
                ring_rows,
            } => {
                write!(
                    f,
                    "position {position} needs its row appended first: the ring holds {ring_rows} rows"
                )
            }
            Self::CursorMismatch {
                cursor,
                rows_written,
            } => write!(
                f,
                "cursor word {cursor} disagrees with the store {rows_written} published rows",
            ),
            Self::Cache(error) => write!(f, "{error}"),
        }
    }
}

impl From<cache::CacheError> for StageError {
    fn from(error: cache::CacheError) -> Self {
        Self::Cache(error)
    }
}

/// The window span for the 0-based token index p: (window_start, window_rows), where
/// window_rows = min(p + 1, window) includes the current row and window_start = p + 1 -
/// window_rows is the first row (or ring slot) the window reads.
#[must_use]
pub fn window_span(position: usize, window: usize) -> (usize, usize) {
    let rows = (position + 1).min(window);
    (position + 1 - rows, rows)
}

/// Assembles and validates the operands the body takes.
///
/// Pure: it reads the cache and the ring state, launches nothing, allocates nothing but the
/// selected vector, and returns the numbers the body needs. The checks run before any derived
/// operand is returned, so a caller cannot act on a half-valid set.
///
/// # Errors
/// [StageError::RingPastCapacity], [StageError::PositionPastRing],
/// [StageError::CursorMismatch], or the selected-id refusals of
/// [cache::selected_rows] wrapped as [StageError::Cache].
pub fn assemble(
    plan: &cache::Plan,
    store: &cache::Store,
    ring_rows: usize,
    max_positions: usize,
    position: usize,
    cursor: u32,
    ids: &[u32],
) -> Result<StageOperands, StageError> {
    if ring_rows > max_positions {
        return Err(StageError::RingPastCapacity {
            ring_rows,
            max_positions,
        });
    }
    if position >= ring_rows {
        return Err(StageError::PositionPastRing {
            position,
            ring_rows,
        });
    }
    let rows_written = store.rows_written();
    if cursor as usize != rows_written {
        return Err(StageError::CursorMismatch {
            cursor,
            rows_written,
        });
    }
    let (window_start, window_rows) = window_span(position, WINDOW_ROWS);
    let selected = cache::selected_rows(ids, window_rows, plan.ratio())?;
    Ok(StageOperands {
        ring_rows,
        position,
        window_start,
        window_rows,
        store_rows: rows_written,
        store_zero: store.zero(),
        compressed_rows: plan.capacity_rows(),
        ratio: plan.ratio(),
        selected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan4() -> cache::Plan {
        cache::Plan::new(4).expect("ratio 4")
    }

    fn latent(rows: usize) -> Vec<f32> {
        (0..rows * cache::DIM)
            .map(|index| (index % 17) as f32 * 0.01 - 0.08)
            .collect()
    }

    #[test]
    fn a_stage_run_before_its_append_is_refused() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        let error = assemble(&plan, &store, 0, cache::EXPANDED_ROWS, 0, 0, &[])
            .expect_err("no row to attend");
        assert_eq!(
            error,
            StageError::PositionPastRing {
                position: 0,
                ring_rows: 0
            }
        );
        assert!(error.to_string().contains("append"));
    }

    #[test]
    fn position_zero_attends_exactly_its_own_row() {
        let plan = plan4();
        let mut store = cache::Store::zeroed();
        let operands = assemble(&plan, &store, 1, cache::EXPANDED_ROWS, 0, 0, &[]).expect("row 0");
        assert_eq!(
            operands.window_rows, 1,
            "the current token is in the window"
        );
        assert_eq!(operands.window_start, 0);
        assert!(operands.store_zero, "an empty store is the skip rule");
        assert_eq!(operands.compressed_rows, cache::EXPANDED_ROWS / 4);
        assert_eq!(operands.ratio, 4);
        assert!(operands.selected.is_empty());
        store.append(&plan, 0, &latent(4)).expect("append");
        let published = assemble(&plan, &store, 1, cache::EXPANDED_ROWS, 0, 4, &[]).expect("row 0");
        assert_eq!((published.window_rows, published.store_rows), (1, 4));
    }

    #[test]
    fn the_window_is_the_last_128_rows_from_a_base_offset() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        let short = assemble(&plan, &store, 41, cache::EXPANDED_ROWS, 40, 0, &[]).expect("short");
        assert_eq!((short.window_rows, short.window_start), (41, 0));
        let full = assemble(&plan, &store, 301, cache::EXPANDED_ROWS, 300, 0, &[]).expect("full");
        assert_eq!(
            (full.window_rows, full.window_start),
            (WINDOW_ROWS, 301 - WINDOW_ROWS)
        );
    }

    #[test]
    fn a_ring_past_its_allocation_is_refused() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        let error = assemble(
            &plan,
            &store,
            cache::EXPANDED_ROWS + 1,
            cache::EXPANDED_ROWS,
            0,
            0,
            &[],
        )
        .expect_err("ring past capacity");
        assert_eq!(
            error,
            StageError::RingPastCapacity {
                ring_rows: cache::EXPANDED_ROWS + 1,
                max_positions: cache::EXPANDED_ROWS,
            }
        );
        assert_eq!(
            error.to_string(),
            "ring reports 513 rows, the allocation is 512"
        );
    }

    #[test]
    fn a_position_past_the_ring_is_refused() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        let error =
            assemble(&plan, &store, 4, cache::EXPANDED_ROWS, 5, 0, &[]).expect_err("past ring");
        // The row for the position must have been appended: 4 rows hold positions 0..=3.
        assert_eq!(
            error,
            StageError::PositionPastRing {
                position: 5,
                ring_rows: 4
            }
        );
    }

    #[test]
    fn a_cursor_that_disagrees_with_the_store_is_refused() {
        let plan = plan4();
        let mut store = cache::Store::zeroed();
        store.append(&plan, 0, &latent(2)).expect("append");
        let error =
            assemble(&plan, &store, 3, cache::EXPANDED_ROWS, 2, 0, &[]).expect_err("stale cursor");
        assert_eq!(
            error,
            StageError::CursorMismatch {
                cursor: 0,
                rows_written: 2
            }
        );
        let ok =
            assemble(&plan, &store, 3, cache::EXPANDED_ROWS, 2, 2, &[]).expect("cursor agrees");
        assert_eq!(ok.store_rows, 2);
        assert!(!ok.store_zero, "a published row clears the skip rule");
    }

    #[test]
    fn the_selected_ids_are_the_expanded_rows_and_the_two_refusals_hold() {
        let plan = plan4();
        let mut store = cache::Store::zeroed();
        store.append(&plan, 0, &latent(4)).expect("append");
        let operands = assemble(
            &plan,
            &store,
            141,
            cache::EXPANDED_ROWS,
            140,
            4,
            &[140, 141, u32::MAX],
        )
        .expect("selected");
        assert_eq!(operands.window_rows, WINDOW_ROWS);
        assert_eq!(operands.window_start, 141 - WINDOW_ROWS);
        assert_eq!(operands.selected, vec![(140 - 128) * 4, (141 - 128) * 4]);
        let error = assemble(&plan, &store, 141, cache::EXPANDED_ROWS, 140, 4, &[127])
            .expect_err("below the window");
        assert!(
            matches!(error, StageError::Cache(cache::CacheError::Refused { .. })),
            "{error:?}"
        );
        let error = assemble(
            &plan,
            &store,
            141,
            cache::EXPANDED_ROWS,
            140,
            4,
            &[128 + 200],
        )
        .expect_err("past the store");
        assert!(
            matches!(error, StageError::Cache(cache::CacheError::Refused { .. })),
            "{error:?}"
        );
    }

    #[test]
    fn the_window_span_is_the_engines_window_indices() {
        // Pinned against engine window_indices (attention.rs:945), not against our own arithmetic.
        let config = sparkflash_engine::ModelConfig::tiny();
        let window = config.window_size;
        assert_eq!(
            window, 8,
            "tiny's window, the case the engine test documents"
        );
        let decode = sparkflash_engine::forward::attention::window_indices(&config, 10, 1);
        assert_eq!(decode, vec![3, 4, 5, 6, 7, 0, 1, 2]);
        let (start, rows) = window_span(10, window);
        assert_eq!((start, rows), (3, 8));
        for (index, slot) in decode.iter().enumerate() {
            if index >= window - rows {
                let expected = (start + (index - (window - rows))) % window;
                assert_eq!(*slot, expected, "slot {index} of the engine row");
            }
        }
        // Position 0: exactly the current row, not the sink-only empty union.
        let first = sparkflash_engine::forward::attention::window_indices(&config, 0, 1);
        assert_eq!(first, vec![0]);
        assert_eq!(window_span(0, window), (0, 1));
        // A short sequence: the leading slots the sequence has not reached stay NO_INDEX and the
        // data slots start at window_start.
        let early = sparkflash_engine::forward::attention::window_indices(&config, 2, 1);
        assert_eq!(
            early,
            vec![
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                0,
                1,
                2
            ]
        );
        assert_eq!(window_span(2, window), (0, 3));
        for (start_pos, expected) in [
            (0usize, (0usize, 1usize)),
            (1, (0, 2)),
            (7, (0, 8)),
            (8, (1, 8)),
            (300, (293, 8)),
        ] {
            assert_eq!(
                window_span(start_pos, window),
                expected,
                "position {start_pos}"
            );
        }
    }

    #[test]
    fn a_caller_passing_the_count_instead_of_the_index_is_refused_not_short() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        // p = 40 with its row appended: the correct call.
        let ok = assemble(&plan, &store, 41, cache::EXPANDED_ROWS, 40, 0, &[]).expect("index p");
        assert_eq!(ok.window_rows, 41);
        // The same call with the count 41: refused, so the one-row-short window cannot happen.
        assert_eq!(
            assemble(&plan, &store, 41, cache::EXPANDED_ROWS, 41, 0, &[]).expect_err("count"),
            StageError::PositionPastRing {
                position: 41,
                ring_rows: 41
            }
        );
    }

    #[test]
    fn the_body_position_and_window_rows_words_cannot_disagree() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        for position in [0usize, 1, 7, 8, 40, 127, 128, 255, 300] {
            let operands = assemble(
                &plan,
                &store,
                position + 1,
                cache::EXPANDED_ROWS,
                position,
                0,
                &[],
            )
            .expect("valid at p");
            assert_eq!(
                operands.window_start + operands.window_rows,
                position + 1,
                "p {position}"
            );
            assert_eq!(operands.window_rows, (position + 1).min(WINDOW_ROWS));
        }
    }

    #[test]
    fn the_selected_ids_are_offset_by_the_window_count_not_the_ring_start() {
        let plan = plan4();
        let store = cache::Store::zeroed();
        // p = 140: window_start = 13 and window_rows = 128, so the two candidate bases differ by 115.
        let operands = assemble(
            &plan,
            &store,
            141,
            cache::EXPANDED_ROWS,
            140,
            0,
            &[128, 129, u32::MAX],
        )
        .expect("ids in the window-count space");
        assert_eq!(operands.window_start, 13);
        assert_eq!(operands.window_rows, 128);
        assert_eq!(operands.window_start, 141 - 128, "not equal to the count");
        assert_eq!(
            operands.selected,
            vec![0, 4],
            "id - window_rows times the ratio"
        );
        // id = window_start + j is the wrong base: refused by selected_rows, not absorbed.
        let error = assemble(&plan, &store, 141, cache::EXPANDED_ROWS, 140, 0, &[13])
            .expect_err("ring-start base");
        assert!(
            matches!(error, StageError::Cache(cache::CacheError::Refused { .. })),
            "{error:?}"
        );
    }
}
