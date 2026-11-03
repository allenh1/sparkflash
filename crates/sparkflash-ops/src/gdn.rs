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

//! The Gated DeltaNet and Kimi Delta Attention scan constants and launch shape.

/// The key head width the scan is compiled for.
pub const GDN_KEY: usize = 128;

/// The value head width the scan is compiled for.
pub const GDN_VALUE: usize = 128;

/// Floats per staged (token, key head): q then k, padded.
pub const GDN_QK_ROW: usize = 288;

/// The conv history the scan carries in registers.
pub const CONV_HISTORY: usize = 8;

/// Tokens a Qwen sparse-attention query selects at most.
pub const MAX_SELECTED_TOKENS: usize = 2304;

/// The indexer blocks the selection kernel folds into.
pub const MAX_BUDGET_BLOCKS: usize = 1024;

/// Blocks the scan aims for: one an SM on GB10's forty-eight.
const GDN_SCAN_TARGET_BLOCKS: usize = 48;

/// The most a head's value columns split.
const GDN_SCAN_MAX_SPLITS: usize = 16;

/// The scan grid and block width for heads heads.
#[must_use]
pub fn gdn_scan_shape(heads: usize) -> ((u32, u32, u32), u32) {
    let mut splits = 1;
    while heads * splits < GDN_SCAN_TARGET_BLOCKS && splits < GDN_SCAN_MAX_SPLITS {
        splits *= 2;
    }
    (
        (heads as u32, splits as u32, 1),
        (4 * GDN_VALUE / splits) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_four_rank_share_still_fills_the_part() {
        let ((heads, splits, _), threads) = gdn_scan_shape(8);
        assert_eq!(heads, 8);
        assert!(8 * splits as usize >= 48);
        assert_eq!(threads as usize * splits as usize, 4 * GDN_VALUE);
    }
}
