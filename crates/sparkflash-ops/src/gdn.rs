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

//! The Gated DeltaNet and Kimi Delta Attention scan constants, launch shape and typed calls.
//!
//! The chunk kernels of the scan image (`kernels/qwen4_scan_sm121.cu`) are one Gated DeltaNet
//! layer's recurrence for any number of rows: the causal conv over every row at once
//! ([conv_rows]), the history advanced past them ([conv_history]), the normalised queries and keys
//! and each row's decay and beta ([GdnGeometry::prep]), the scan with the state in registers
//! ([GdnGeometry::scan_args], on the grid [crate::scan::ScanPlan] resolves) and the gated norm
//! ([gated_norm]). Qwen's, GLM's and the shared dense arm's launches are built here, so each
//! spells the kernels' argument order once.

use crate::calls::{BLOCK, Call, flat_blocks};

/// The key head width the scan is compiled for.
pub const GDN_KEY: usize = 128;

/// The value head width the scan is compiled for.
pub const GDN_VALUE: usize = 128;

/// Floats per staged (token, key head): q then k, padded.
pub const GDN_QK_ROW: usize = 288;

/// The conv history the scan carries in registers.
pub const CONV_HISTORY: usize = 8;

/// Blocks the scan aims for: one an SM on GB10's forty-eight.
const GDN_SCAN_TARGET_BLOCKS: usize = 48;

/// The most a head's value columns split.
const GDN_SCAN_MAX_SPLITS: usize = 16;

/// The scan grid and block width for heads heads.
#[must_use]
pub fn gdn_scan_shape(heads: usize) -> ((u32, u32, u32), u32) {
    gdn_scan_shape_for(heads, GDN_VALUE)
}

/// [gdn_scan_shape] for one value-head width.
///
/// The block is `4 * value_dim / splits` threads, a multiple of 32 and at most the entry's
/// launch bound; `V_DIM` 64 caps the splits at 8 (32 threads) and `V_DIM` 128/256 at 16. The
/// lower split bound is the launch bound: with every head accounted for, a `V_DIM` 256 entry
/// still needs two splits to stay inside its 512 threads.
#[must_use]
pub fn gdn_scan_shape_for(heads: usize, value_dim: usize) -> ((u32, u32, u32), u32) {
    // q4p_gdn_scan_64_64 is launched with a 256-thread bound and the two 128-value entries
    // with 512; a block past its entry's bound fails the launch.
    let bound = if value_dim == 64 { 256 } else { 512 };
    let max_splits = (4 * value_dim / 32).max(1);
    let mut splits = 1;
    while 4 * value_dim / splits.max(1) > bound && splits < max_splits {
        splits *= 2;
    }
    while heads * splits < GDN_SCAN_TARGET_BLOCKS
        && splits < GDN_SCAN_MAX_SPLITS
        && splits < max_splits
    {
        splits *= 2;
    }
    (
        (heads as u32, splits as u32, 1),
        (4 * value_dim / splits) as u32,
    )
}

/// The floats `q4p_gdn_prep` stages for one (token, key head): q then k, each `key_dim` wide with
/// four padding floats after every 32 (`Q4P_GDN_ROW`).
#[must_use]
pub const fn gdn_qk_row(key_dim: usize) -> usize {
    2 * (key_dim + key_dim / 8)
}

/// One rank's Gated DeltaNet geometry, as the chunk kernels take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnGeometry {
    /// Key heads.
    pub key_heads: usize,
    /// Value heads: the scan's heads.
    pub heads: usize,
    /// Key head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// The causal convolution's taps.
    pub taps: usize,
}

impl GdnGeometry {
    /// Every query and key head's floats.
    #[must_use]
    pub const fn key_width(&self) -> usize {
        self.key_heads * self.key_dim
    }

    /// Every value head's floats.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.heads * self.value_dim
    }

    /// A convolved row: q, k and v.
    #[must_use]
    pub const fn conv_width(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }

    /// The value heads that share a key head (repeat-interleave order).
    #[must_use]
    pub const fn groups(&self) -> usize {
        match self.heads.checked_div(self.key_heads) {
            Some(groups) => groups,
            None => 0,
        }
    }

    /// The conv history's bytes: `max(taps - 1, 1)` f32 rows of [Self::conv_width], oldest first.
    #[must_use]
    pub const fn conv_bytes(&self) -> usize {
        let rows = if self.taps > 1 { self.taps - 1 } else { 1 };
        rows * self.conv_width() * 4
    }

    /// The delta-rule state's bytes: f32 `[heads][key_dim][value_dim]`.
    #[must_use]
    pub const fn state_bytes(&self) -> usize {
        self.heads * self.key_dim * self.value_dim * 4
    }

    /// The staged queries and keys of `rows` rows, floats.
    #[must_use]
    pub const fn qk_floats(&self, rows: usize) -> usize {
        rows * self.key_heads * gdn_qk_row(self.key_dim)
    }

    /// `q4p_gdn_prep` over `rows` convolved rows of `mixed` and their `a` and `b` logits: the
    /// normalised q (times `1 / sqrt(key_dim)`) and k into `qk`, and each row's (decay, beta) per
    /// value head into `decay_beta`. Grid (key heads, rows), a thread a key.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn prep(
        &self,
        mixed: u64,
        a: u64,
        b: u64,
        a_log: u64,
        dt_bias: u64,
        qk: u64,
        decay_beta: u64,
        rows: usize,
    ) -> Call<12> {
        Call {
            grid: (self.key_heads as u32, rows as u32, 1),
            block: (self.key_dim as u32, 1, 1),
            args: [
                mixed,
                a,
                b,
                a_log,
                dt_bias,
                qk,
                decay_beta,
                self.key_width() as u64,
                self.key_heads as u64,
                self.heads as u64,
                self.value_dim as u64,
                self.groups() as u64,
            ],
        }
    }

    /// The scan's arguments over `rows` rows (`q4p_gdn_scan` and its other geometries, launched on
    /// the [crate::scan::ScanPlan]'s grid): from the state in `from` to the state in `to` (which
    /// may be `from`), the raw head outputs into `out`, and no per-row snapshot.
    #[must_use]
    pub const fn scan_args(
        &self,
        (qk, decay_beta, mixed): (u64, u64, u64),
        (from, to): (u64, u64),
        out: u64,
        rows: usize,
    ) -> [u64; 15] {
        [
            qk,
            decay_beta,
            mixed,
            from,
            to,
            out,
            self.key_width() as u64,
            self.key_heads as u64,
            self.heads as u64,
            self.value_dim as u64,
            self.groups() as u64,
            rows as u64,
            0,
            0,
            0,
        ]
    }
}

/// `q4p_gdn_conv`: the causal depthwise conv with SiLU over `rows` raw rows of `width` floats,
/// the rows before them read from `history`, into `out`. Grid (width / 256, rows).
#[must_use]
pub const fn conv_rows(
    raw: u64,
    history: u64,
    weight: u64,
    out: u64,
    (width, taps): (usize, usize),
    rows: usize,
) -> Call<6> {
    Call {
        grid: (flat_blocks(width), rows as u32, 1),
        block: (BLOCK, 1, 1),
        args: [raw, history, weight, out, width as u64, taps as u64],
    }
}

/// `q4p_gdn_conv_state`: `history` advanced past `rows` raw rows of `width` floats. A verify's
/// `snapshot` is (windows, rows, region stride): region r receives the history after row r, for
/// the first `rows` rows; (0, 0, 0) keeps none.
#[must_use]
pub const fn conv_history(
    raw: u64,
    history: u64,
    (width, taps): (usize, usize),
    rows: usize,
    (snapshot, snapshot_rows, region_stride): (u64, usize, u64),
) -> Call<8> {
    Call {
        grid: (flat_blocks(width), 1, 1),
        block: (BLOCK, 1, 1),
        args: [
            raw,
            history,
            width as u64,
            taps as u64,
            rows as u64,
            snapshot,
            snapshot_rows as u64,
            region_stride,
        ],
    }
}

/// `q4p_gdn_gate`: the gated RMSNorm of `rows` rows of `heads` raw head outputs `value_dim` wide,
/// `raw * rsqrt(mean(raw²) + eps) * norm * gate(z)` with a sigmoid or SiLU gate, into `out` (which
/// may be `raw`). Grid (heads, rows), a thread a value.
#[must_use]
pub fn gated_norm(
    (raw, z, norm, out): (u64, u64, u64, u64),
    (heads, value_dim): (usize, usize),
    eps: f32,
    sigmoid: bool,
    rows: usize,
) -> Call<8> {
    Call {
        grid: (heads as u32, rows as u32, 1),
        block: (value_dim as u32, 1, 1),
        args: [
            raw,
            z,
            norm,
            out,
            heads as u64,
            value_dim as u64,
            u64::from(eps.to_bits()),
            u64::from(sigmoid),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The calls keep the kernels' argument order, as Qwen's and GLM's launches spelled it.
    #[test]
    fn the_chunk_calls_keep_the_kernels_argument_order() {
        let geometry = GdnGeometry {
            key_heads: 2,
            heads: 4,
            key_dim: 128,
            value_dim: 128,
            taps: 4,
        };
        assert_eq!(geometry.conv_width(), 1024);
        assert_eq!(geometry.groups(), 2);
        assert_eq!(geometry.conv_bytes(), 3 * 1024 * 4);
        assert_eq!(gdn_qk_row(GDN_KEY), GDN_QK_ROW);
        let conv = conv_rows(1, 2, 3, 4, (1024, 4), 7);
        assert_eq!((conv.grid, conv.args), ((4, 7, 1), [1, 2, 3, 4, 1024, 4]));
        let history = conv_history(1, 2, (1000, 4), 7, (0, 0, 0));
        assert_eq!(
            (history.grid, history.args),
            ((4, 1, 1), [1, 2, 1000, 4, 7, 0, 0, 0])
        );
        let prep = geometry.prep(1, 2, 3, 4, 5, 6, 7, 9);
        assert_eq!((prep.grid, prep.block), ((2, 9, 1), (128, 1, 1)));
        assert_eq!(prep.args, [1, 2, 3, 4, 5, 6, 7, 256, 2, 4, 128, 2]);
        assert_eq!(
            geometry.scan_args((1, 2, 3), (4, 5), 6, 9),
            [1, 2, 3, 4, 5, 6, 256, 2, 4, 128, 2, 9, 0, 0, 0]
        );
        let gate = gated_norm((1, 2, 3, 1), (4, 128), 1e-6, true, 9);
        assert_eq!((gate.grid, gate.block), ((4, 9, 1), (128, 1, 1)));
        assert_eq!(
            gate.args,
            [1, 2, 3, 1, 4, 128, u64::from(1e-6f32.to_bits()), 1]
        );
    }

    #[test]
    fn a_four_rank_share_still_fills_the_part() {
        let ((heads, splits, _), threads) = gdn_scan_shape(8);
        assert_eq!(heads, 8);
        assert!(8 * splits as usize >= 48);
        assert_eq!(threads as usize * splits as usize, 4 * GDN_VALUE);
    }
}
