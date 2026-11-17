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

//! The routed experts' shared launch pieces: the expert plan, the grouped MMA's tile and the
//! combine.
//!
//! Every arm with routed experts sorts its slots by expert with one shared kernel,
//! `kernels/moe_plan_sm121.cu`, and the Qwen image's grouped W4A16 MMA (`q4p_moe_*`) is the body
//! Qwen and Inkling run a chunk's experts through. Their argument layout, tile constants, item
//! counts and grids are one rule, here. Which body a pass runs (the token kernels, the rows form
//! or the grouped tile) stays each arm's: the bodies round differently, so moving that choice
//! changes bits.

/// A kernel's grid and block.
pub type Launch = ((u32, u32, u32), (u32, u32, u32));

/// The most experts the plan's counting pass and the grouped MMA's tables hold
/// (`MOE_PLAN_MAX_EXPERTS`, `Q4P_MAX_EXPERTS`).
pub const MAX_EXPERTS: usize = 512;

/// What the plan does with an id outside `[0, experts)` (`MOE_PLAN_INVALID_FIRST`,
/// `MOE_PLAN_CLAMP`, `MOE_PLAN_DROP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanInvalid {
    /// The ids no expert owns sort first as one run, which the token path's expert kernel zeroes.
    First = 0,
    /// The id is clamped into range.
    Clamp = 1,
    /// The slot is left out of the counts, the items and the order; a consumer that sums every
    /// slot's row also names rows to zero ([MoePlanCall::zeroing]).
    Drop = 2,
}

/// One launch of the shared expert plan (`moe_plan`): its arguments in the kernel's order, and
/// its one-block launch.
///
/// Pointers are device addresses; a zero pointer is an output the caller does not want.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoePlanCall {
    /// The routed expert id of every slot, `[slots]`.
    pub ids: u64,
    /// The slots in expert order (the expert kernels' permutation), `[slots]`.
    pub sorted: u64,
    /// The item list a grouped body walks, or zero for a run-only plan.
    pub items: u64,
    /// The items written, or zero for a run-only plan.
    pub item_count: u64,
    /// Each slot's run in `sorted`, for the token path's run walk; zero otherwise.
    pub run_bounds: u64,
    /// The routed slots (rows times the top-k).
    pub slots: usize,
    /// The experts the ids range over.
    pub experts: usize,
    /// The experts a token routes to.
    pub top_k: usize,
    /// The slots one item holds (zero when no items are cut).
    pub item_rows: usize,
    /// The most items the item list holds.
    pub max_items: usize,
    /// How an id outside the experts is treated.
    pub invalid: PlanInvalid,
    /// Rows to zero for every dropped slot, `[slots][zero_stride]`, or zero for none.
    pub zero_rows: u64,
    /// The floats a zeroed row holds.
    pub zero_stride: usize,
}

impl MoePlanCall {
    /// A plan that cuts the slots into items of `item_rows` for a grouped body, dropping a slot
    /// no expert owns.
    #[must_use]
    pub const fn items(
        ids: u64,
        sorted: u64,
        (items, item_count): (u64, u64),
        (slots, experts, top_k): (usize, usize, usize),
        item_rows: usize,
        max_items: usize,
    ) -> Self {
        Self {
            ids,
            sorted,
            items,
            item_count,
            run_bounds: 0,
            slots,
            experts,
            top_k,
            item_rows,
            max_items,
            invalid: PlanInvalid::Drop,
            zero_rows: 0,
            zero_stride: 0,
        }
    }

    /// A plan that writes only the order and each slot's run, for the token path; an id no
    /// expert owns sorts first.
    #[must_use]
    pub const fn runs(
        ids: u64,
        sorted: u64,
        run_bounds: u64,
        (slots, experts, top_k): (usize, usize, usize),
    ) -> Self {
        Self {
            ids,
            sorted,
            items: 0,
            item_count: 0,
            run_bounds,
            slots,
            experts,
            top_k,
            item_rows: 0,
            max_items: 0,
            invalid: PlanInvalid::First,
            zero_rows: 0,
            zero_stride: 0,
        }
    }

    /// The same plan, zeroing a dropped slot's row of `rows` (`stride` floats a row), for a
    /// consumer that sums every slot's row.
    #[must_use]
    pub const fn zeroing(self, rows: u64, stride: usize) -> Self {
        Self {
            zero_rows: rows,
            zero_stride: if rows == 0 { 0 } else { stride },
            ..self
        }
    }

    /// One block of the profile's plan threads.
    #[must_use]
    pub fn launch() -> Launch {
        (
            (1, 1, 1),
            (
                sparkflash_model::active_profile().plan.block_threads(),
                1,
                1,
            ),
        )
    }

    /// The kernel's arguments. No segments are named, so every slot is live.
    #[must_use]
    pub const fn args(&self) -> [u64; 18] {
        [
            self.ids,
            self.sorted,
            self.items,
            self.item_count,
            self.run_bounds,
            self.slots as u64,
            self.experts as u64,
            self.top_k as u64,
            self.item_rows as u64,
            self.max_items as u64,
            0,
            self.invalid as u64,
            0,
            0,
            0,
            0,
            self.zero_rows,
            self.zero_stride as u64,
        ]
    }
}

/// The grouped W4A16 MMA's tile (`q4p_moe_gate_up`/`q4p_moe_down` and their MX8, FP8 and capped
/// forms, which Qwen and Inkling launch): an item's slots, a stage's K and a CTA's columns.
pub mod grouped {
    use super::{Launch, MAX_EXPERTS};

    /// The (token, slot) rows one item holds (`Q4P_MOE_ROWS`).
    pub const ITEM_ROWS: usize = 32;
    /// The K a pipeline stage reads (`Q4P_BK`): four 16-column E4M3 scale groups.
    pub const K_STAGE: usize = 64;
    /// The intermediate columns a gate_up CTA covers: 64 gate rows and the matching 64 up rows.
    pub const GATE_UP_COLUMNS: usize = 64;
    /// The output columns a down CTA covers (`Q4P_BN`).
    pub const DOWN_COLUMNS: usize = 128;
    /// The threads of a grouped CTA.
    pub const THREADS: u32 = 256;
    /// The rows form's (`q4p_moe_rows_*`) gate_up and down columns a CTA: four warps of 8 and 16.
    pub const ROWS_GATE_UP_COLUMNS: usize = 32;
    /// See [ROWS_GATE_UP_COLUMNS].
    pub const ROWS_DOWN_COLUMNS: usize = 64;
    /// The K a rows-form lane reads at once: two stages, 16 bytes of a packed row.
    pub const ROWS_K: usize = 2 * K_STAGE;
    /// The threads of a rows-form CTA.
    pub const ROWS_THREADS: u32 = 128;

    /// The most items the plan can cut from `slots` slots over `experts` experts: a run per
    /// expert and every [ITEM_ROWS] rows of one. The grid is this tall; past the count the plan
    /// wrote, a CTA returns before touching anything.
    #[must_use]
    pub const fn items(slots: usize, experts: usize) -> usize {
        slots.div_ceil(ITEM_ROWS) + if experts < slots { experts } else { slots }
    }

    /// Whether the tile covers an expert: `hidden` the gate_up K, `intermediate` its columns,
    /// and a down projection of `down_rows` outputs over `down_columns` K, at most [MAX_EXPERTS]
    /// a rank.
    #[must_use]
    pub const fn fits(
        hidden: usize,
        intermediate: usize,
        (down_rows, down_columns): (usize, usize),
        experts: usize,
    ) -> bool {
        down_rows.is_multiple_of(DOWN_COLUMNS)
            && hidden.is_multiple_of(K_STAGE)
            && intermediate.is_multiple_of(GATE_UP_COLUMNS)
            && intermediate.is_multiple_of(K_STAGE)
            && down_columns.is_multiple_of(K_STAGE)
            && experts <= MAX_EXPERTS
    }

    /// Whether the rows form's lanes cover both K dimensions (whole stage pairs).
    #[must_use]
    pub const fn rows_fit(hidden: usize, intermediate: usize) -> bool {
        hidden.is_multiple_of(ROWS_K) && intermediate.is_multiple_of(ROWS_K)
    }

    /// The body's two launches over `items` grid rows: gate_up over `intermediate` columns and
    /// down over `down_rows` outputs, in the grouped tile or (`rows`) the rows form.
    #[must_use]
    pub const fn grids(
        intermediate: usize,
        down_rows: usize,
        items: u32,
        rows: bool,
    ) -> (Launch, Launch) {
        let (gate_up, down, threads) = if rows {
            (ROWS_GATE_UP_COLUMNS, ROWS_DOWN_COLUMNS, ROWS_THREADS)
        } else {
            (GATE_UP_COLUMNS, DOWN_COLUMNS, THREADS)
        };
        (
            (((intermediate / gate_up) as u32, items, 1), (threads, 1, 1)),
            (((down_rows / down) as u32, items, 1), (threads, 1, 1)),
        )
    }
}

/// The combine (`q4e_moe_combine`, which GLM and Qwen launch): a 256-thread block a 256-column
/// slice of a row.
#[must_use]
pub const fn combine_grid(hidden: usize, rows: usize) -> Launch {
    let blocks = hidden.div_ceil(256);
    (
        (if blocks == 0 { 1 } else { blocks } as u32, rows as u32, 1),
        (256, 1, 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_arguments_are_the_arms_old_lists() {
        // Qwen's grouped prefill (prefill.rs) and GLM's bulk plan (device.rs), verbatim.
        let call = MoePlanCall::items(1, 2, (3, 4), (40, 512, 10), 32, 17).zeroing(5, 2560);
        assert_eq!(
            call.args(),
            [
                1, 2, 3, 4, 0, 40, 512, 10, 32, 17, 0, 2, 0, 0, 0, 0, 5, 2560
            ]
        );
        // Qwen's token run walk (blocks.rs).
        let runs = MoePlanCall::runs(1, 2, 6, (20, 512, 10));
        assert_eq!(
            runs.args(),
            [1, 2, 0, 0, 6, 20, 512, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        // A plan zeroing nothing names no stride: the kernel reads it only beside a row pointer.
        assert_eq!(
            MoePlanCall::items(1, 2, (3, 4), (8, 4, 2), 0, 9)
                .zeroing(0, 64)
                .args()[17],
            0
        );
    }

    #[test]
    fn the_grouped_rules_are_the_arms_old_rules() {
        for (slots, experts) in [(1, 512), (40, 512), (4096, 512), (100, 8)] {
            assert_eq!(
                grouped::items(slots, experts),
                slots.div_ceil(32) + experts.min(slots)
            );
        }
        assert!(grouped::fits(2560, 1024, (2560, 1024), 512));
        assert!(!grouped::fits(2560, 1024, (2560, 1024), 513));
        assert!(!grouped::fits(2560, 96, (2560, 96), 8));
        assert!(grouped::rows_fit(2560, 1024));
        assert!(!grouped::rows_fit(2560, 1088));
        assert_eq!(
            grouped::grids(1024, 2560, 7, false),
            (((16, 7, 1), (256, 1, 1)), ((20, 7, 1), (256, 1, 1)))
        );
        assert_eq!(
            grouped::grids(1024, 2560, 7, true),
            (((32, 7, 1), (128, 1, 1)), ((40, 7, 1), (128, 1, 1)))
        );
        assert_eq!(combine_grid(4096, 3), ((16, 3, 1), (256, 1, 1)));
        assert_eq!(combine_grid(2560, 1), ((10, 1, 1), (256, 1, 1)));
    }
}
