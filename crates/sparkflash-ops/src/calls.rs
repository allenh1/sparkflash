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

//! Typed calls of the token image's shared kernels: norm, rotary, residual add, SwiGLU, copy,
//! zero, embedding, the BF16 cache append, the row GEMV, the head deinterleave and the split full
//! attention.
//!
//! Every device arm launches these bodies from the Qwen token image (`kernels/qwen4_exp_sm121.cu`
//! and `kernels/qwen4_attn_sm121.cu`), the block drafter ([crate::dflash]) included, and each
//! spelled their argument lists and grids by hand. A [Call] is one launch's grid, block and
//! arguments in the kernel's order, built from what the launch means; the epilogue plans
//! ([crate::epilogue::NormPlan], [crate::epilogue::RopePlan], [crate::epilogue::ActivationPlan])
//! resolve to these bodies (`q4e_rmsnorm`, `q4e_rope`, `q4e_swiglu`) among their candidates. The
//! grids are the arms' own: the elementwise bodies walk a grid stride, so any grid computes the
//! same values, and the row bodies take one block a row or group.

/// The threads of every token body here but the rotary (`Q4E_BLOCK`).
pub const BLOCK: u32 = 256;

/// The threads of a rotary block: a head of one token.
pub const ROPE_BLOCK: u32 = 128;

/// One launch: its grid, its block and its arguments in the kernel's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Call<const N: usize> {
    /// The grid.
    pub grid: (u32, u32, u32),
    /// The block.
    pub block: (u32, u32, u32),
    /// The arguments, pointers as device addresses.
    pub args: [u64; N],
}

/// The blocks an elementwise body takes over `count` values: one a [BLOCK] of them, at least
/// one.
#[must_use]
pub const fn flat_blocks(count: usize) -> u32 {
    let blocks = count.div_ceil(BLOCK as usize);
    (if blocks == 0 { 1 } else { blocks }) as u32
}

const fn flat<const N: usize>(count: usize, args: [u64; N]) -> Call<N> {
    Call {
        grid: (flat_blocks(count), 1, 1),
        block: (BLOCK, 1, 1),
        args,
    }
}

/// `q4e_add`: `into[i] += from[i]` over `count` floats.
#[must_use]
pub const fn add(into: u64, from: u64, count: usize) -> Call<3> {
    flat(count, [into, from, count as u64])
}

/// `q4e_swiglu`: `out[i] = silu(gate[i]) * up[i]` over `count` floats (`out` may be `up`).
#[must_use]
pub const fn swiglu(gate: u64, up: u64, out: u64, count: usize) -> Call<4> {
    flat(count, [gate, up, out, count as u64])
}

/// `q4e_swiglu_clamped`: [swiglu] with the gate clamped to at most `limit` and up to
/// `[-limit, limit]` first (a release's `swiglu_limit`).
#[must_use]
pub const fn swiglu_clamped(gate: u64, up: u64, out: u64, count: usize, limit: f32) -> Call<5> {
    flat(count, [gate, up, out, count as u64, limit.to_bits() as u64])
}

/// `q4e_copy_flat`: `count` floats from `src` to `dst`.
#[must_use]
pub const fn copy(dst: u64, src: u64, count: usize) -> Call<3> {
    flat(count, [dst, src, count as u64])
}

/// `q4e_narrow_bf16`: `count` floats from `src` narrowed to BF16 words at `dst`.
#[must_use]
pub const fn narrow(dst: u64, src: u64, count: usize) -> Call<3> {
    flat(count, [dst, src, count as u64])
}

/// `q4e_zero`: `count` floats of `dst` set to zero.
#[must_use]
pub const fn zero(dst: u64, count: usize) -> Call<2> {
    flat(count, [dst, count as u64])
}

/// `q4e_sum_slots`: this rank's `count` floats at `own` summed with every peer's at
/// `slots + peer * count`, in ascending rank order from 0.0, into `own` (the rank's own slot is
/// not read). `count` is a multiple of four: a thread sums a float4.
#[must_use]
pub const fn sum_slots(own: u64, slots: u64, count: usize, rank: usize, world: usize) -> Call<5> {
    flat(
        count / 4,
        [own, slots, count as u64, rank as u64, world as u64],
    )
}

/// `q4e_cache_bf16`: `count` f32 values of `rows` narrowed to BF16 into `cache`, one contiguous
/// run (a chunk's keys or values, appended at once).
#[must_use]
pub const fn cache_bf16(cache: u64, rows: u64, count: usize) -> Call<3> {
    flat(count, [cache, rows, count as u64])
}

/// `q4e_cache_rows_bf16`: `rows` rows of `width` f32 values, `src_stride` floats apart in
/// `src`, narrowed to BF16 rows `dst_stride` apart in `cache`.
#[must_use]
pub const fn cache_rows_bf16(
    (cache, dst_stride): (u64, usize),
    (src, src_stride): (u64, usize),
    width: usize,
    rows: usize,
) -> Call<5> {
    Call {
        grid: (flat_blocks(width), rows as u32, 1),
        block: (BLOCK, 1, 1),
        args: [
            cache,
            src,
            width as u64,
            dst_stride as u64,
            src_stride as u64,
        ],
    }
}

/// `q4e_embed` for one row: `token`'s BF16 row of `table` (`hidden` wide) widened to f32 into
/// `out`, its `width` floats (the hidden, or the hidden times the residual streams).
#[must_use]
pub const fn embed(table: u64, out: u64, token: u32, (hidden, width): (usize, usize)) -> Call<6> {
    Call {
        grid: (flat_blocks(width), 1, 1),
        block: (BLOCK, 1, 1),
        args: [table, out, token as u64, hidden as u64, width as u64, 0],
    }
}

/// `q4e_embed` for `rows` rows, `width` floats apart in `out`, their ids read on the device from
/// `tokens` (`[rows]` u32).
#[must_use]
pub const fn embed_rows(
    table: u64,
    out: u64,
    tokens: u64,
    (hidden, width): (usize, usize),
    rows: usize,
) -> Call<6> {
    Call {
        grid: (flat_blocks(width), rows as u32, 1),
        block: (BLOCK, 1, 1),
        args: [table, out, 0, hidden as u64, width as u64, tokens],
    }
}

/// `q4e_rmsnorm` and `q4e_rmsnorm_rows`: RMSNorm of `width`-wide rows from `input` into
/// `output` (which may be `input`), by `group` floats at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNorm {
    /// The rows normed, f32.
    pub input: u64,
    /// The gains, BF16.
    pub weight: u64,
    /// The normed rows, f32.
    pub output: u64,
    /// A row's floats.
    pub width: usize,
    /// The floats one norm covers: the row, or a head of it.
    pub group: usize,
    /// The epsilon.
    pub eps: f32,
    /// Whether the gains are zero-centred (`1 + w`) rather than multiplied directly.
    pub zero_centred: bool,
    /// Whether each group takes the same `group` gains (a per-head norm) rather than its
    /// columns of a row's.
    pub weight_per_group: bool,
}

impl RmsNorm {
    /// One norm a row, its gains multiplied directly.
    #[must_use]
    pub const fn new(input: u64, weight: u64, output: u64, width: usize, eps: f32) -> Self {
        Self {
            input,
            weight,
            output,
            width,
            group: width,
            eps,
            zero_centred: false,
            weight_per_group: false,
        }
    }

    /// The same norm, its gains zero-centred or not.
    #[must_use]
    pub const fn zero_centred(self, zero_centred: bool) -> Self {
        Self {
            zero_centred,
            ..self
        }
    }

    /// The same norm by `group` floats a norm, each group's gains its own columns or (with
    /// `weight_per_group`) the same `group` gains.
    #[must_use]
    pub const fn groups(self, group: usize, weight_per_group: bool) -> Self {
        Self {
            group,
            weight_per_group,
            ..self
        }
    }

    const fn head(&self) -> [u64; 8] {
        [
            self.input,
            self.weight,
            self.output,
            self.width as u64,
            self.group as u64,
            self.eps.to_bits() as u64,
            self.zero_centred as u64,
            self.weight_per_group as u64,
        ]
    }

    /// `q4e_rmsnorm` over `rows` contiguous rows: a block a group.
    #[must_use]
    pub const fn call(&self, rows: usize) -> Call<8> {
        Call {
            grid: ((rows * self.width / self.group) as u32, 1, 1),
            block: (BLOCK, 1, 1),
            args: self.head(),
        }
    }

    /// `q4e_rmsnorm_rows` over `rows` rows, `src_stride` floats apart in the input and
    /// `dst_stride` in the output: a block a row's group.
    #[must_use]
    pub const fn rows_call(&self, rows: usize, src_stride: usize, dst_stride: usize) -> Call<10> {
        let head = self.head();
        // A row whole (or no group at all) is one block; a per-head norm a block a head.
        let groups = if self.group > 0 && self.width / self.group > 1 {
            self.width / self.group
        } else {
            1
        };
        Call {
            grid: (groups as u32, rows as u32, 1),
            block: (BLOCK, 1, 1),
            args: [
                head[0],
                head[1],
                head[2],
                head[3],
                head[4],
                head[5],
                head[6],
                head[7],
                src_stride as u64,
                dst_stride as u64,
            ],
        }
    }
}

/// `q4e_rope`: NeoX rotary over `tokens` rows of `heads` heads, in place, `row_stride` floats a
/// row, row t at `position + t`; the first `rotary` floats of each `head_dim` rotate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
    /// The rows, f32.
    pub rows: u64,
    /// The heads a row holds.
    pub heads: usize,
    /// A head's floats.
    pub head_dim: usize,
    /// The floats of a head that rotate.
    pub rotary: usize,
    /// The first row's position.
    pub position: usize,
    /// The rotary base.
    pub theta: f32,
    /// A row's floats.
    pub row_stride: usize,
    /// The rows.
    pub tokens: usize,
}

impl Rope {
    /// A block a (head, row).
    #[must_use]
    pub const fn call(&self) -> Call<7> {
        Call {
            grid: (self.heads as u32, self.tokens as u32, 1),
            block: (ROPE_BLOCK, 1, 1),
            args: [
                self.rows,
                self.head_dim as u64,
                self.rotary as u64,
                self.position as u64,
                self.theta.to_bits() as u64,
                self.row_stride as u64,
                self.tokens as u64,
            ],
        }
    }
}

/// The warps of a row GEMV block (`q4e_gemv_bf16`): [BLOCK] threads.
pub const GEMV_WARPS: usize = BLOCK as usize / 32;

/// `q4e_gemv_bf16`: `output[t][r] = weight[r] . input[t]` for `batch` rows (at most
/// [crate::gemv::GEMV_MAX_BATCH]) over a row-major `rows x columns` BF16 weight.
///
/// One output row a warp for a single row, two for a batch: the grid follows the kernel's
/// row-to-warp mapping, or half the blocks exit unused.
#[must_use]
pub const fn gemv_bf16(
    weight: u64,
    input: u64,
    output: u64,
    (rows, columns): (usize, usize),
    batch: usize,
) -> Call<6> {
    let per_block = if batch > 1 {
        GEMV_WARPS * 2
    } else {
        GEMV_WARPS
    };
    Call {
        grid: (rows.div_ceil(per_block) as u32, 1, 1),
        block: (BLOCK, 1, 1),
        args: [
            weight,
            input,
            output,
            rows as u64,
            columns as u64,
            batch as u64,
        ],
    }
}

/// `q4e_deinterleave`: a gated query projection's `[head0 query | head0 gate | ...]` rows split
/// into `[rows][heads * head_dim]` queries and gates.
#[must_use]
pub const fn deinterleave(
    interleaved: u64,
    (query, gate): (u64, u64),
    (heads, head_dim): (usize, usize),
    rows: usize,
) -> Call<5> {
    Call {
        grid: (flat_blocks(heads * head_dim), rows as u32, 1),
        block: (BLOCK, 1, 1),
        args: [interleaved, query, gate, heads as u64, head_dim as u64],
    }
}

/// Keys a `q4e_attn_full` block takes a pass (`Q4E_FULL_TILE`).
pub const FULL_ATTENTION_TILE: usize = 256;

/// The most blocks `q4e_attn_full` splits one row's context over (`Q4E_FULL_SPLITS`).
pub const FULL_ATTENTION_SPLITS: usize = 16;

/// The splits a row that sees `visible` positions takes (`q4e_full_splits`).
#[must_use]
pub const fn full_attention_splits(visible: usize) -> usize {
    let splits = visible.div_ceil(FULL_ATTENTION_TILE);
    if splits == 0 {
        1
    } else if splits > FULL_ATTENTION_SPLITS {
        FULL_ATTENTION_SPLITS
    } else {
        splits
    }
}

/// `q4e_attn_full` and `q4e_attn_full_combine`: causal grouped-query attention of `rows` query
/// rows at absolute `position` over a BF16 `[position][kv_heads * head_dim]` key and value
/// cache, each row's context split over up to [FULL_ATTENTION_SPLITS] blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullAttention {
    /// The queries, `[rows][heads][head_dim]` f32.
    pub query: u64,
    /// The cached keys, BF16.
    pub keys: u64,
    /// The cached values, BF16.
    pub values: u64,
    /// The split partials, `[rows][heads][FULL_ATTENTION_SPLITS][head_dim + 2]` f32
    /// ([Self::partial_floats]).
    pub partials: u64,
    /// The first row's absolute position: row `t` sees `position + t + 1` keys.
    pub position: usize,
    /// The query rows.
    pub rows: usize,
    /// The query heads.
    pub heads: usize,
    /// The key/value heads.
    pub kv_heads: usize,
    /// A head's floats (at most [BLOCK]).
    pub head_dim: usize,
}

impl FullAttention {
    /// The partial floats `rows` rows of `heads` heads of `head_dim` take.
    #[must_use]
    pub const fn partial_floats(rows: usize, heads: usize, head_dim: usize) -> usize {
        rows * heads * FULL_ATTENTION_SPLITS * (head_dim + 2)
    }

    /// The split pass: a block a (head, row, split).
    #[must_use]
    pub const fn call(&self) -> Call<9> {
        let groups = match self.heads.checked_div(self.kv_heads) {
            Some(groups) => groups,
            None => 1,
        };
        Call {
            grid: (
                self.heads as u32,
                self.rows as u32,
                full_attention_splits(self.position + self.rows) as u32,
            ),
            block: (BLOCK, 1, 1),
            args: [
                self.query,
                self.keys,
                self.values,
                self.partials,
                self.position as u64,
                self.heads as u64,
                self.kv_heads as u64,
                self.head_dim as u64,
                groups as u64,
            ],
        }
    }

    /// The combine into `out` (`[rows][heads][head_dim]`), times `sigmoid(gate)` when `gate`
    /// is not null: a block a (head, row).
    #[must_use]
    pub const fn combine(&self, gate: u64, out: u64) -> Call<6> {
        Call {
            grid: (self.heads as u32, self.rows as u32, 1),
            block: (BLOCK, 1, 1),
            args: [
                self.partials,
                gate,
                out,
                self.position as u64,
                self.heads as u64,
                self.head_dim as u64,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_attention_and_projection_calls_are_the_arms_old_lists() {
        // Qwen's row GEMV (device.rs), one row and a batch.
        assert_eq!(
            gemv_bf16(1, 2, 3, (100, 64), 1),
            Call {
                grid: (13, 1, 1),
                block: (256, 1, 1),
                args: [1, 2, 3, 100, 64, 1],
            }
        );
        assert_eq!(gemv_bf16(1, 2, 3, (100, 64), 4).grid, (7, 1, 1));
        // Qwen's head deinterleave (blocks.rs).
        assert_eq!(
            deinterleave(1, (2, 3), (16, 256), 5),
            Call {
                grid: (16, 5, 1),
                block: (256, 1, 1),
                args: [1, 2, 3, 16, 256],
            }
        );
        // Qwen's full attention (blocks.rs) at a position past one split.
        let attention = FullAttention {
            query: 1,
            keys: 2,
            values: 3,
            partials: 4,
            position: 300,
            rows: 7,
            heads: 16,
            kv_heads: 2,
            head_dim: 256,
        };
        assert_eq!(
            attention.call(),
            Call {
                grid: (16, 7, 2),
                block: (256, 1, 1),
                args: [1, 2, 3, 4, 300, 16, 2, 256, 8],
            }
        );
        assert_eq!(
            attention.combine(5, 6),
            Call {
                grid: (16, 7, 1),
                block: (256, 1, 1),
                args: [4, 5, 6, 300, 16, 256],
            }
        );
        assert_eq!(full_attention_splits(0), 1);
        assert_eq!(full_attention_splits(256), 1);
        assert_eq!(full_attention_splits(257), 2);
        assert_eq!(full_attention_splits(1 << 20), 16);
        assert_eq!(FullAttention::partial_floats(2, 3, 4), 2 * 3 * 16 * 6);
    }

    #[test]
    fn the_calls_are_the_arms_old_lists() {
        let eps = 1e-6f32;
        let bits = u64::from(eps.to_bits());
        // Qwen's `norm` (device.rs): a per-head, zero-centred norm of four rows.
        let norm = RmsNorm::new(1, 2, 3, 512, eps)
            .zero_centred(true)
            .groups(128, true);
        assert_eq!(
            norm.call(4),
            Call {
                grid: (16, 1, 1),
                block: (256, 1, 1),
                args: [1, 2, 3, 512, 128, bits, 1, 1],
            }
        );
        // GLM's and Inkling's strided rows (`norm_rows`).
        assert_eq!(
            RmsNorm::new(1, 2, 3, 64, eps).rows_call(5, 96, 128),
            Call {
                grid: (1, 5, 1),
                block: (256, 1, 1),
                args: [1, 2, 3, 64, 64, bits, 0, 0, 96, 128],
            }
        );
        // Qwen's query rotary (blocks.rs).
        let rope = Rope {
            rows: 7,
            heads: 16,
            head_dim: 256,
            rotary: 64,
            position: 40,
            theta: 1e7,
            row_stride: 4096,
            tokens: 3,
        };
        assert_eq!(
            rope.call(),
            Call {
                grid: (16, 3, 1),
                block: (128, 1, 1),
                args: [7, 256, 64, 40, u64::from(1e7f32.to_bits()), 4096, 3],
            }
        );
        assert_eq!(add(1, 2, 513).grid, (3, 1, 1));
        assert_eq!(add(1, 2, 0).grid, (1, 1, 1), "an empty launch is one block");
        assert_eq!(swiglu(1, 2, 2, 300).args, [1, 2, 2, 300]);
        assert_eq!(cache_bf16(1, 2, 256).grid, (1, 1, 1));
        assert_eq!(
            cache_rows_bf16((1, 128), (2, 640), 128, 9),
            Call {
                grid: (1, 9, 1),
                block: (256, 1, 1),
                args: [1, 2, 128, 128, 640],
            }
        );
        assert_eq!(
            embed(1, 2, 42, (2048, 8192)),
            Call {
                grid: (32, 1, 1),
                block: (256, 1, 1),
                args: [1, 2, 42, 2048, 8192, 0],
            }
        );
        assert_eq!(
            embed_rows(1, 2, 9, (2048, 2048), 4),
            Call {
                grid: (8, 4, 1),
                block: (256, 1, 1),
                args: [1, 2, 0, 2048, 2048, 9],
            }
        );
    }
}
