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

//! Single-pass mHC: the coefficient mixes, the split sinkhorn and the two reductions.
//!
//! Manifold-constrained hyper-connections keep the hidden state as an `hc_mult`-copy stream and put
//! three coefficient sets around each sublayer. DeepSeek V4.1 and GLM-5.3-Flash run the same
//! schedule and differ only in the geometry and the epsilons an [HcGeometry] carries: V4.1's input
//! norm uses `norm_eps = 1e-20`, GLM-5.3-Flash's its `rms_norm_eps`. The line references below are
//! to V4.1's reference (`model.py`, `kernel.py`), and `docs/engine/01-forward-order.md` section 3.2
//! is the specification of every function here. `Block.forward` (model.py:968-994) is:
//!
//! ```text
//! residual = x
//! (attn_pre, attn_post, attn_comb) = hc_mixes(x, hc_attn_fn, hc_attn_scale, hc_attn_base)  # 982
//! x = hc_pre(x, pre_mix)   # the previous sublayer's pre_mix; an identity one-hot before it
//! x = attn_norm(x); x = attn(x, ...)                                                      # 984-985
//! x = hc_post(x, residual, attn_post, attn_comb)                                          # 986
//! ```
//!
//! A token's stream is flattened copy by copy to `[hc_mult * dim]` and its coefficient sets are
//! `hc_mult + hc_mult + hc_mult^2` values - exactly [HcGeometry::hc_mix_count]. These functions
//! take a whole chunk: `n` tokens are `n * hc_mult * dim` stream values, every coefficient buffer
//! holds one set per token laid out token by token - [Mixes::for_tokens] sizes them,
//! [Mixes::for_config] is one token's - and the reductions write one row per token. Every length is
//! checked against the geometry and against the other buffers rather than trusted, so a caller that
//! sizes a buffer for the wrong number of tokens is refused with [EngineError::Shape] instead of
//! being silently half-processed.
//!
//! This is an `f32` reference over decoded weights: the released kernels quantise the activation of
//! the `hc_*_fn` GEMM (section 6 point 1 of the specification), and no quantised kernel runs on
//! this path.
//!
//! # Attribution
//!
//! Manifold-constrained hyper-connections are the method of Zhenda Xie, Yixuan Wei, Huanqi Cao et
//! al. ("mHC: Manifold-Constrained Hyper-Connections", arXiv:2512.24880). The schedule here follows
//! DeepSeek's released V4.1-Flash inference reference (`model.py`, `kernel.py`; MIT License,
//! Copyright (c) DeepSeek), which was read for the arithmetic; no source is copied. See the root
//! NOTICE file.

use crate::linalg::{self, EngineError};

/// The mHC parameters of one model: the copy count, the width of one copy, the Sinkhorn iterations
/// and the two epsilons. A model pack converts its configuration into one (`From<&Config>`), so
/// every function here takes either.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcGeometry {
    /// The number of stream copies (`hc_mult`).
    pub hc_mult: usize,
    /// The width of one copy: the model's hidden size.
    pub dim: usize,
    /// `hc_sinkhorn_iters`: the split sinkhorn runs twice as many passes.
    pub hc_sinkhorn_iters: usize,
    /// The sinkhorn's and the input mixes' epsilon.
    pub hc_eps: f32,
    /// The epsilon of the weightless RMS norm over a token's flattened stream.
    pub norm_eps: f32,
}

impl HcGeometry {
    /// The coefficient count of one token: `hc_mult` pre, `hc_mult` post and `hc_mult^2` comb.
    #[must_use]
    pub const fn hc_mix_count(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// The width of one token's flattened stream: `hc_mult * dim`.
    #[must_use]
    pub const fn hc_width(&self) -> usize {
        self.hc_mult * self.dim
    }
}

/// The mHC coefficient sets of a chunk: `(pre, post, comb)` per token, the output of [hc_mixes].
#[derive(Debug, Clone, PartialEq)]
pub struct Mixes {
    /// The input mixes, `hc_mult` per token: `sigmoid(mixes[0:hc] * scale[0] + base[0:hc]) +
    /// hc_eps` (kernel.py:427).
    pub pre: Vec<f32>,
    /// The output mixes, `hc_mult` per token: `2 * sigmoid(mixes[hc:2hc] * scale[1] +
    /// base[hc:2hc])`, with **no** epsilon (kernel.py:429).
    pub post: Vec<f32>,
    /// The residual mixes, `hc_mult * hc_mult` per token, row-major: `mixes[2hc:] * scale[2] +
    /// base[2hc:]` reshaped `[hc, hc]` (kernel.py:430-431) and then normalised by the split
    /// sinkhorn (kernel.py:436-458). Row `j` is source copy `j` and column `k` the destination
    /// copy - the released `comb[j][k]` orientation that [hc_post] applies (`model.py:965`;
    /// `docs/architecture/02-mhc.md` section 4; `crates/sparkflash-kernels/src/mega_mhc_ref.rs:58-67`).
    pub comb: Vec<f32>,
}

impl Mixes {
    /// One token's zeroed coefficient sets: `pre` and `post` are `hc_mult` long and `comb` is
    /// `hc_mult * hc_mult`. This is [Self::for_tokens] with one token.
    #[must_use]
    pub fn for_config(cfg: impl Into<HcGeometry>) -> Self {
        let cfg: HcGeometry = cfg.into();
        Mixes::for_tokens(cfg, 1)
    }

    /// `positions` tokens' zeroed coefficient sets, token by token: `pre` and `post` are
    /// `positions * hc_mult` long and `comb` is `positions * hc_mult * hc_mult`. Token `t`'s sets
    /// are `pre[t * hc..]`, `post[t * hc..]` and `comb[t * hc * hc..]` - the layout [hc_mixes]
    /// writes and [hc_pre] and [hc_post] read.
    ///
    /// Sizing here rather than at each call site is what keeps a caller from growing a buffer per
    /// token, and `positions` is what a chunk's scratch has to be built with.
    #[must_use]
    pub fn for_tokens(cfg: impl Into<HcGeometry>, positions: usize) -> Self {
        let cfg: HcGeometry = cfg.into();
        let hc = cfg.hc_mult;
        Mixes {
            pre: vec![0.0; positions * hc],
            post: vec![0.0; positions * hc],
            comb: vec![0.0; positions * hc * hc],
        }
    }
}

/// The identity input mix: a one-hot on copy 0 of every token (model.py:1159-1163, 1260).
///
/// The model starts with this because no sublayer has produced a `pre_mix` yet, so layer 0's
/// attention sees copy 0 of the stream (section 3.2 of the specification). `out` holds one
/// `hc_mult`-long mix per token, so its length is a whole number of tokens.
///
/// # Errors
///
/// [EngineError::Shape] when `out` is empty or not a whole number of `hc_mult`-long mixes.
pub fn identity_pre_mix(cfg: impl Into<HcGeometry>, out: &mut [f32]) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let hc = cfg.hc_mult;
    if hc == 0 || out.is_empty() || !out.len().is_multiple_of(hc) {
        return Err(shape_error(
            "identity pre-mix",
            format!(
                "{hc} mHC copies need a pre-mix of whole {hc}s, got {}",
                out.len()
            ),
        ));
    }
    out.fill(0.0);
    for token in 0..out.len() / hc {
        out[token * hc] = 1.0;
    }
    Ok(())
}

/// `hc_mixes` (model.py:948-955): the three coefficient sets of every token in the chunk.
///
/// `x_hc` is the flattened stream, `n` tokens of `hc_mult` copies of `dim` laid out copy by copy,
/// and `rsqrt` is per token: `1 / sqrt(mean(x^2) + norm_eps)` over that token's `hc_mult * dim`
/// values, with V4.1's `norm_eps = 1e-20` - **not** `hc_eps = 1e-6`, which is the epsilon of the
/// sinkhorn below (model.py:949-950). `fn_w` is the `[mix_hc, hc * dim]` `hc_*_fn` weight,
/// `mixes = fn_w * x_hc * rsqrt` (model.py:951), and `scale`/`base` are the `[3]` and `[mix_hc]`
/// tensors that `hc_split_sinkhorn` applies, to every token:
///
/// - `pre = sigmoid(mix[0:hc] * scale[0] + base[0:hc]) + hc_eps` (kernel.py:427),
/// - `post = 2 * sigmoid(mix[hc:2hc] * scale[1] + base[hc:2hc])`, no epsilon (kernel.py:429),
/// - `comb = mix[2hc:] * scale[2] + base[2hc:]`, reshaped `[hc, hc]` (row: source copy, column:
///   destination copy - the released orientation), then normalised (kernel.py:430-431, 436-458).
///
/// The normalisation runs `2 * hc_sinkhorn_iters` passes per token: one row softmax (plus `hc_eps`),
/// then a column normalisation, then `hc_sinkhorn_iters - 1` pairs of a row and a column
/// normalisation, with `hc_eps` in every denominator. The last pass normalises columns and each
/// destination column sums to 1 across its sources, which is the 40 passes the released
/// `hc_sinkhorn_iters = 20` runs (kernel.py:436-458; specification section 3.2).
///
/// `mixes` must hold one set per token ([Mixes::for_tokens]): the token count is
/// `x_hc.len() / hc_width()` and every buffer must agree with it.
///
/// # Errors
///
/// [EngineError::Shape] when any slice's length disagrees with the configuration or with another
/// slice.
pub fn hc_mixes(
    cfg: impl Into<HcGeometry>,
    x_hc: &[f32],
    fn_w: &[f32],
    scale: &[f32],
    base: &[f32],
    mixes: &mut Mixes,
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let hc = cfg.hc_mult;
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    if hc == 0 || width == 0 {
        return Err(shape_error(
            "hc mixes",
            "the configuration has no mHC copies".to_owned(),
        ));
    }
    if !x_hc.len().is_multiple_of(width) {
        return Err(shape_error(
            "hc mixes",
            format!(
                "{} stream values are not a whole number of {width}-wide tokens",
                x_hc.len()
            ),
        ));
    }
    let tokens = x_hc.len() / width;
    if fn_w.len() != mix_hc * width {
        return Err(shape_error(
            "hc mixes",
            format!(
                "the hc fn weight is [{}] = {}, got {}",
                mix_hc,
                mix_hc * width,
                fn_w.len()
            ),
        ));
    }
    if scale.len() != 3 {
        return Err(shape_error(
            "hc mixes",
            format!(
                "scale holds pre, post and comb, so it is 3 long, got {}",
                scale.len()
            ),
        ));
    }
    if base.len() != mix_hc {
        return Err(shape_error(
            "hc mixes",
            format!("base is {mix_hc} long, got {}", base.len()),
        ));
    }
    if mixes.pre.len() != tokens * hc
        || mixes.post.len() != tokens * hc
        || mixes.comb.len() != tokens * hc * hc
    {
        return Err(shape_error(
            "hc mixes",
            format!(
                "{tokens} tokens need {} + {} + {} coefficient values, got {} + {} + {}",
                tokens * hc,
                tokens * hc,
                tokens * hc * hc,
                mixes.pre.len(),
                mixes.post.len(),
                mixes.comb.len()
            ),
        ));
    }

    // The projection, then the coefficient tail over it. [hc_mixes] is exactly these two, so a
    // caller that already has the projection can skip the first half and get the same mixes.
    let projection = hc_projection(cfg, x_hc, fn_w)?;
    hc_mixes_from_projection(cfg, &projection, scale, base, mixes)
}

/// One token's `x_hc` against row `row` of the `hc_*_fn` weight: the `fn_w * x_hc` of model.py:951,
/// before the `rsqrt` the caller multiplies in.
///
/// One coefficient at a time, so no `[mix_hc]` buffer is needed between the projection and the
/// split.
///
/// # Errors
///
/// [EngineError::Shape] if [linalg::matmul] refuses the geometry.
fn project(fn_w: &[f32], row: usize, width: usize, token_x: &[f32]) -> Result<f32, EngineError> {
    let mut raw = 0.0f32;
    linalg::matmul(
        std::slice::from_mut(&mut raw),
        token_x,
        &fn_w[row * width..(row + 1) * width],
        1,
        width,
        1,
    )?;
    Ok(raw)
}

/// The largest coefficient-row count the interleaved projection keeps in one stack array.
///
/// The released model has `2 * hc_mult + hc_mult^2` = 24 with `hc_mult = 4`; the bound exists so
/// a malformed configuration is refused instead of overflowing a fixed-size accumulator array.
const MAX_MIX_HC: usize = 32;

/// The `hc_*_fn` projection of a chunk: the raw `[mix_hc]` dot products per token and the sum of
/// squares each token's `rsqrt` factor is built from.
///
/// This is the first half of [hc_mixes] - the dot products of every `hc_*_fn` row - separated from
/// the coefficient arithmetic so that it can be evaluated in the loop order the host likes
/// ([hc_projection_transposed]) without duplicating the tail. [hc_projection] is the same function
/// in [hc_mixes]'s own loop order.
#[derive(Debug, Clone, PartialEq)]
pub struct HcProjection {
    /// `mix_hc` raw dot products per token, token by token, before the `rsqrt`.
    pub raw: Vec<f32>,
    /// `sum(x^2)` over each token's flattened `hc_mult * dim` stream, in stream order.
    pub sum_squares: Vec<f32>,
}

/// [hc_mixes]'s projection in its own order: every coefficient row one at a time, ascending `k`,
/// one accumulator.
///
/// # Errors
///
/// [EngineError::Shape] when the stream or the weight disagrees with the configuration.
pub fn hc_projection(
    cfg: impl Into<HcGeometry>,
    x_hc: &[f32],
    fn_w: &[f32],
) -> Result<HcProjection, EngineError> {
    let cfg: HcGeometry = cfg.into();
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    let tokens = check_projection_geometry(cfg, x_hc, fn_w.len(), "hc projection")?;
    let mut raw = vec![0.0f32; tokens * mix_hc];
    let mut sum_squares = vec![0.0f32; tokens];
    for token in 0..tokens {
        let token_x = &x_hc[token * width..(token + 1) * width];
        // mean(x^2)'s numerator: the specification's own x.pow(2).mean(-1) (model.py:949-950).
        sum_squares[token] = token_x.iter().map(|value| value * value).sum::<f32>();
        for coefficient in 0..mix_hc {
            raw[token * mix_hc + coefficient] = project(fn_w, coefficient, width, token_x)?;
        }
    }
    Ok(HcProjection { raw, sum_squares })
}

/// [hc_projection] with the loop order swapped: `k` outer, coefficient rows inner.
///
/// Every row's accumulation is still one accumulator in ascending `k` - the inner loop adds at most
/// one term to any given row per step - so the value of each dot product is the same expression
/// sequence as [hc_projection]'s and the bytes are identical by construction. What changes is that
/// all `mix_hc` rows are in flight at once, which is what lets an out-of-order core overlap their
/// dependency chains, and the weight is read in its transposed `[width][mix_hc]` layout
/// ([hc_transpose_predictor]), so each `k` reads `mix_hc` contiguous f32 values instead of one value
/// from each of `mix_hc` rows 80 KB apart.
///
/// # Errors
///
/// [EngineError::Shape] when the stream, the transposed weight or the configuration disagree, or
/// when the configuration has more than [MAX_MIX_HC] coefficient rows.
pub fn hc_projection_transposed(
    cfg: impl Into<HcGeometry>,
    x_hc: &[f32],
    fn_w_transposed: &[f32],
) -> Result<HcProjection, EngineError> {
    let cfg: HcGeometry = cfg.into();
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    let tokens = check_projection_geometry(cfg, x_hc, fn_w_transposed.len(), "hc projection")?;
    if mix_hc > MAX_MIX_HC {
        return Err(shape_error(
            "hc projection",
            format!(
                "{mix_hc} coefficient rows are more than the {MAX_MIX_HC} the interleaved pass holds"
            ),
        ));
    }
    let mut raw = vec![0.0f32; tokens * mix_hc];
    let mut sum_squares = vec![0.0f32; tokens];
    let mut acc = [0.0f32; MAX_MIX_HC];
    for token in 0..tokens {
        let token_x = &x_hc[token * width..(token + 1) * width];
        // The same iterator sum [hc_projection] runs: one accumulator, ascending, no reassociation.
        sum_squares[token] = token_x.iter().map(|value| value * value).sum::<f32>();
        let raw_token = &mut raw[token * mix_hc..(token + 1) * mix_hc];
        if mix_hc == RELEASED_MIX_HC {
            project_transposed_released(width, token_x, fn_w_transposed, raw_token);
        } else {
            let acc = &mut acc[..mix_hc];
            acc.fill(0.0);
            for k in 0..width {
                let a = token_x[k];
                let column = &fn_w_transposed[k * mix_hc..(k + 1) * mix_hc];
                for (slot, weight) in acc.iter_mut().zip(column) {
                    *slot += a * *weight;
                }
            }
            raw_token.copy_from_slice(acc);
        }
    }
    Ok(HcProjection { raw, sum_squares })
}

/// The released model's coefficient-row count: `2 * hc_mult + hc_mult^2` with `hc_mult = 4`.
///
/// It gets its own projection loop because a compile-time row count lets the accumulators stay in
/// registers: the aarch64 backend keeps the six four-lane groups in vector registers and the inner
/// body is six `fmul v.4s` and six `fadd v.4s` per `k` - the multiply and the add stay separate, so
/// no part of the chain is fused into an `fmla` - with no spill and no bounds check in the loop.
/// Every lane is one coefficient row, accumulating in ascending `k` with one accumulator, which is
/// [hc_projection]'s own chain.
const RELEASED_MIX_HC: usize = 24;

/// One token's projection with the released [RELEASED_MIX_HC] rows: four coefficient rows per lane
/// group, the same multiply-then-add sequence per row as [hc_projection].
#[inline]
fn project_transposed_released(
    width: usize,
    token_x: &[f32],
    fn_w_transposed: &[f32],
    raw: &mut [f32],
) {
    let mut acc = [0.0f32; RELEASED_MIX_HC];
    for k in 0..width {
        let a = token_x[k];
        let column = &fn_w_transposed[k * RELEASED_MIX_HC..(k + 1) * RELEASED_MIX_HC];
        let (columns, _) = column.as_chunks::<4>();
        for (slots, weights) in acc.as_chunks_mut::<4>().0.iter_mut().zip(columns) {
            slots[0] += a * weights[0];
            slots[1] += a * weights[1];
            slots[2] += a * weights[2];
            slots[3] += a * weights[3];
        }
    }
    raw.copy_from_slice(&acc);
}

/// [hc_mixes]'s coefficient arithmetic over an already-computed [HcProjection]: the `rsqrt`, the
/// three scaled-and-biased groups, the split sinkhorn, and every buffer check.
///
/// The operations are [hc_mixes]'s own, in its order: `rsqrt` is `1 / sqrt(sum_squares / width +
/// norm_eps)` and each mix is `raw * rsqrt`.
///
/// # Errors
///
/// [EngineError::Shape] when `scale`, `base`, `mixes` or the projection disagree with the
/// configuration.
pub fn hc_mixes_from_projection(
    cfg: impl Into<HcGeometry>,
    projection: &HcProjection,
    scale: &[f32],
    base: &[f32],
    mixes: &mut Mixes,
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let hc = cfg.hc_mult;
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    if hc == 0 || width == 0 {
        return Err(shape_error(
            "hc mixes",
            "the configuration has no mHC copies".to_owned(),
        ));
    }
    let tokens = projection.sum_squares.len();
    if projection.raw.len() != tokens * mix_hc {
        return Err(shape_error(
            "hc mixes",
            format!(
                "{tokens} tokens need {} raw projection values, got {}",
                tokens * mix_hc,
                projection.raw.len()
            ),
        ));
    }
    if scale.len() != 3 {
        return Err(shape_error(
            "hc mixes",
            format!(
                "scale holds pre, post and comb, so it is 3 long, got {}",
                scale.len()
            ),
        ));
    }
    if base.len() != mix_hc {
        return Err(shape_error(
            "hc mixes",
            format!("base is {mix_hc} long, got {}", base.len()),
        ));
    }
    if mixes.pre.len() != tokens * hc
        || mixes.post.len() != tokens * hc
        || mixes.comb.len() != tokens * hc * hc
    {
        return Err(shape_error(
            "hc mixes",
            format!(
                "{tokens} tokens need {} + {} + {} coefficient values, got {} + {} + {}",
                tokens * hc,
                tokens * hc,
                tokens * hc * hc,
                mixes.pre.len(),
                mixes.post.len(),
                mixes.comb.len()
            ),
        ));
    }

    let hc_eps = cfg.hc_eps;
    let norm_eps = cfg.norm_eps;
    for token in 0..tokens {
        let mean_square = projection.sum_squares[token] / width as f32;
        let rsqrt = 1.0 / (mean_square + norm_eps).sqrt();
        let raw = &projection.raw[token * mix_hc..(token + 1) * mix_hc];

        for (coefficient, bias) in base.iter().enumerate().take(hc) {
            let mix = raw[coefficient] * rsqrt;
            mixes.pre[token * hc + coefficient] = linalg::sigmoid(mix * scale[0] + *bias) + hc_eps;
        }
        for coefficient in 0..hc {
            let row = hc + coefficient;
            let mix = raw[row] * rsqrt;
            mixes.post[token * hc + coefficient] =
                2.0 * linalg::sigmoid(mix * scale[1] + base[row]);
        }
        for row in 0..hc {
            for col in 0..hc {
                let coefficient = 2 * hc + row * hc + col;
                let mix = raw[coefficient] * rsqrt;
                mixes.comb[token * hc * hc + row * hc + col] = mix * scale[2] + base[coefficient];
            }
        }
        split_sinkhorn(
            &mut mixes.comb[token * hc * hc..(token + 1) * hc * hc],
            hc,
            cfg.hc_sinkhorn_iters,
            hc_eps,
        );
    }
    Ok(())
}

/// [hc_mixes] with the projection evaluated in [hc_projection_transposed]'s loop order.
///
/// `fn_w_transposed` is the `[width][mix_hc]` transpose of the `[mix_hc][width]` `hc_*_fn` weight
/// ([hc_transpose_predictor]); the mixes are the values [hc_mixes] produces, because both evaluate
/// each row's dot product with the same operations in the same order.
///
/// # Errors
///
/// [hc_projection_transposed]'s and [hc_mixes_from_projection]'s refusals.
pub fn hc_mixes_transposed(
    cfg: impl Into<HcGeometry>,
    x_hc: &[f32],
    fn_w_transposed: &[f32],
    scale: &[f32],
    base: &[f32],
    mixes: &mut Mixes,
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let projection = hc_projection_transposed(cfg, x_hc, fn_w_transposed)?;
    hc_mixes_from_projection(cfg, &projection, scale, base, mixes)
}

/// Transposes a `[mix_hc][hc_mult * dim]` `hc_*_fn` weight into [hc_projection_transposed]'s
/// `[hc_mult * dim][mix_hc]` layout: value `(row, k)` lands at `out[k * mix_hc + row]`.
///
/// This moves every value; it does not re-encode one. The transposed copy is the same f32 bits, and
/// a caller does it once per layer when the weights are staged rather than once per call.
///
/// # Errors
///
/// [EngineError::Shape] when the weight or `out` is not `mix_hc * width` long.
pub fn hc_transpose_predictor(
    cfg: impl Into<HcGeometry>,
    fn_w: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    if width == 0 || mix_hc == 0 || fn_w.len() != mix_hc * width || out.len() != fn_w.len() {
        return Err(shape_error(
            "hc transpose",
            format!(
                "the hc fn weight is [{}] = {} values and the transposed copy the same, got {} and {}",
                mix_hc,
                mix_hc * width,
                fn_w.len(),
                out.len()
            ),
        ));
    }
    for row in 0..mix_hc {
        for k in 0..width {
            out[k * mix_hc + row] = fn_w[row * width + k];
        }
    }
    Ok(())
}

/// Validates a projection's stream and weight lengths and returns the token count.
///
/// # Errors
///
/// [EngineError::Shape] when the configuration has no copies, the stream is not whole tokens, or the
/// weight is not `mix_hc * width` long.
fn check_projection_geometry(
    cfg: HcGeometry,
    x_hc: &[f32],
    weight_len: usize,
    what: &str,
) -> Result<usize, EngineError> {
    let hc = cfg.hc_mult;
    let width = cfg.hc_width();
    let mix_hc = cfg.hc_mix_count();
    if hc == 0 || width == 0 {
        return Err(shape_error(
            what,
            "the configuration has no mHC copies".to_owned(),
        ));
    }
    if !x_hc.len().is_multiple_of(width) {
        return Err(shape_error(
            what,
            format!(
                "{} stream values are not a whole number of {width}-wide tokens",
                x_hc.len()
            ),
        ));
    }
    if weight_len != mix_hc * width {
        return Err(shape_error(
            what,
            format!(
                "the hc fn weight is [{}] = {}, got {}",
                mix_hc,
                mix_hc * width,
                weight_len
            ),
        ));
    }
    Ok(x_hc.len() / width)
}

/// One column normalisation of the `[hc, hc]` `comb`, with `hc_eps` in the denominator
/// (kernel.py:445-448, the unconditional pass, and 455-458, the one inside the loop; 441-443 is the
/// ROW pass, which is not this one).
fn normalise_columns(comb: &mut [f32], hc: usize, hc_eps: f32) {
    for col in 0..hc {
        let column_sum: f32 = (0..hc).map(|row| comb[row * hc + col]).sum();
        let denominator = column_sum + hc_eps;
        for row in 0..hc {
            comb[row * hc + col] /= denominator;
        }
    }
}

/// `hc_split_sinkhorn` (kernel.py:436-458) on one token's `comb`.
///
/// The schedule is the kernel's: one row softmax plus `hc_eps`, then a column normalisation that runs
/// unconditionally, then `iters - 1` pairs of a row normalisation and a column one. That is
/// `1 + 1 + 2 * (iters - 1)`, i.e. `2 * iters` passes: 40 at the released `hc_sinkhorn_iters = 20`.
/// The last pass is a column pass, so every column of the result sums to 1. Every denominator carries
/// `hc_eps`, which is `1e-6`, not the `norm_eps` of the `rsqrt` above.
pub fn split_sinkhorn(comb: &mut [f32], hc: usize, iters: usize, hc_eps: f32) {
    for row in comb.chunks_mut(hc) {
        linalg::softmax(row);
        for value in row.iter_mut() {
            *value += hc_eps;
        }
    }
    normalise_columns(comb, hc, hc_eps);
    for _ in 1..iters {
        for row in comb.chunks_mut(hc) {
            let row_sum: f32 = row.iter().sum();
            let denominator = row_sum + hc_eps;
            for value in row.iter_mut() {
                *value /= denominator;
            }
        }
        normalise_columns(comb, hc, hc_eps);
    }
}

/// `hc_pre` (model.py:957-960): collapse the `hc_mult` copies of every chunk token into `dim`
/// values.
///
/// `out[t * dim + j] = sum_c pre_mix[t * hc + c] * x_hc[t * hc * dim + c * dim + j]`, with
/// `pre_mix` holding one `[hc_mult]` mix per token - the previous sublayer's `pre_mix` in
/// `Block.forward` (model.py:983, 990), or [identity_pre_mix] for the first sublayer. The token
/// count is `x_hc.len() / (hc_mult * dim)` and every buffer must agree with it.
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not describe a whole number of tokens of this
/// configuration.
pub fn hc_pre(
    cfg: impl Into<HcGeometry>,
    pre_mix: &[f32],
    x_hc: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let token_width = hc * dim;
    if hc == 0 || dim == 0 || !x_hc.len().is_multiple_of(token_width) {
        return Err(shape_error(
            "hc pre",
            format!(
                "one token's stream is {token_width} wide, got {}",
                x_hc.len()
            ),
        ));
    }
    let tokens = x_hc.len() / token_width;
    if pre_mix.len() != tokens * hc {
        return Err(shape_error(
            "hc pre",
            format!(
                "{tokens} tokens need {} pre-mix values, got {}",
                tokens * hc,
                pre_mix.len()
            ),
        ));
    }
    if out.len() != tokens * dim {
        return Err(shape_error(
            "hc pre",
            format!(
                "{tokens} collapsed tokens are {} wide, got {}",
                tokens * dim,
                out.len()
            ),
        ));
    }
    for token in 0..tokens {
        for (j, value) in out[token * dim..(token + 1) * dim].iter_mut().enumerate() {
            let mut accumulated = 0.0f32;
            for copy in 0..hc {
                accumulated +=
                    pre_mix[token * hc + copy] * x_hc[token * token_width + copy * dim + j];
            }
            *value = accumulated;
        }
    }
    Ok(())
}

/// `hc_post` (model.py:962-966): mix one sublayer's output back onto the mHC stream of every chunk
/// token.
///
/// `out_hc[t, k, d] = post_mix[t, k] * x[t, d] + sum_j comb[t, j, k] * residual_hc[t, j, d]`, the
/// `post (.) x + comb (.) residual` of the specification. `comb` is row-major `[hc, hc]` per
/// token with row `j` the **source** copy and column `k` the **destination** copy - the released
/// `comb[j][k]` orientation: the product `C.unsqueeze(-1) * X.unsqueeze(-2)` is
/// `[b, s, hc, hc, dim]` and is summed over dim 2, the source copy (model.py:965;
/// `docs/architecture/02-mhc.md:91-99`). Each destination column sums to 1 after the sinkhorn, so
/// every output copy is a convex combination of the old copies plus its own post-scaled sublayer
/// output. The token count is `x.len() / dim` and every buffer must agree with it.
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not describe a whole number of tokens of this
/// configuration.
pub fn hc_post(
    cfg: impl Into<HcGeometry>,
    x: &[f32],
    residual_hc: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    out_hc: &mut [f32],
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    if hc == 0 || dim == 0 || !x.len().is_multiple_of(dim) {
        return Err(shape_error(
            "hc post",
            format!("{dim} values per token, got {}", x.len()),
        ));
    }
    let tokens = x.len() / dim;
    if residual_hc.len() != tokens * hc * dim {
        return Err(shape_error(
            "hc post",
            format!(
                "{tokens} residual tokens are {} wide, got {}",
                tokens * hc * dim,
                residual_hc.len()
            ),
        ));
    }
    if post_mix.len() != tokens * hc {
        return Err(shape_error(
            "hc post",
            format!(
                "{tokens} tokens need {} post-mix values, got {}",
                tokens * hc,
                post_mix.len()
            ),
        ));
    }
    if comb.len() != tokens * hc * hc {
        return Err(shape_error(
            "hc post",
            format!(
                "{tokens} tokens need {} comb values, got {}",
                tokens * hc * hc,
                comb.len()
            ),
        ));
    }
    if out_hc.len() != tokens * hc * dim {
        return Err(shape_error(
            "hc post",
            format!(
                "{tokens} mixed tokens are {} wide, got {}",
                tokens * hc * dim,
                out_hc.len()
            ),
        ));
    }
    for token in 0..tokens {
        for copy in 0..hc {
            for j in 0..dim {
                let mut accumulated = post_mix[token * hc + copy] * x[token * dim + j];
                for source in 0..hc {
                    accumulated += comb[token * hc * hc + source * hc + copy]
                        * residual_hc[token * hc * dim + source * dim + j];
                }
                out_hc[token * hc * dim + copy * dim + j] = accumulated;
            }
        }
    }
    Ok(())
}

/// Rounds every value of one row to the release's BF16 dtype boundary, in place.
///
/// Native mHC's reductions end in a cast to the stream dtype: `hc_pre` returns `y.to(x.dtype)`
/// (model.py:954-957), `hc_post` returns `y.type_as(x)` (model.py:962-966), and the block
/// stream is BF16 (model.py:976-994). The released RMSNorm also returns `(weight * x).to(dtype)`
/// (model.py:288-293). [crate::quant::bf16_round] is that cast, round-to-nearest-even included.
pub fn bf16_round_row(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = crate::quant::bf16_round(*value);
    }
}

/// `hc_pre` with the release's BF16 boundary: the collapsed row is rounded once, because native
/// returns `y.to(x.dtype)` and `x` is the BF16 residual stream (model.py:954-957).
///
/// The reduction order is [hc_pre]'s; the boundary is the term native has and the f32 reference
/// does not.
///
/// # Errors
/// [hc_pre]'s shape refusals.
pub fn hc_pre_bf16(
    cfg: impl Into<HcGeometry>,
    pre_mix: &[f32],
    x_hc: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    hc_pre(cfg, pre_mix, x_hc, out)?;
    bf16_round_row(out);
    Ok(())
}

/// `rms_norm` with the release's BF16 boundary: the released RMSNorm computes in f32 and returns
/// `(weight * x).to(dtype)` (model.py:288-293), so the norm output native hands to the sublayer
/// (and its activation quantiser) is BF16, not f32.
///
/// # Errors
/// [linalg::rms_norm]'s shape refusals.
pub fn rms_norm_bf16(
    x: &mut [f32],
    weight: &[f32],
    cols: usize,
    eps: f32,
) -> Result<(), EngineError> {
    linalg::rms_norm(x, weight, cols, eps)?;
    bf16_round_row(x);
    Ok(())
}

/// `hc_post` with the release's BF16 boundary AND its expression structure.
///
/// Native evaluates
/// `post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`
/// (model.py:962-966): the comb sum is its own f32 reduction over the source copies, the post term
/// is one f32 product added to it once, and the row is then rounded to BF16 by `y.type_as(x)`.
/// [hc_post] instead seeds the accumulator with the post term and adds each comb product into it;
/// f32 addition is not associative, so the two can disagree in the last bits before the BF16
/// round. This function keeps the release order, then rounds.
///
/// # Errors
/// [hc_post]'s shape refusals (it is run first to own the length contract, then overwritten in the
/// release order).
pub fn hc_post_bf16(
    cfg: impl Into<HcGeometry>,
    x: &[f32],
    residual_hc: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    out_hc: &mut [f32],
) -> Result<(), EngineError> {
    let cfg: HcGeometry = cfg.into();
    // [hc_post] owns the shape contract; run it so a malformed buffer is refused with the same
    // error, then recompute in the release's order. The two passes are hc * dim values each.
    hc_post(cfg, x, residual_hc, post_mix, comb, out_hc)?;
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let token_width = hc * dim;
    for token in 0..out_hc.len() / token_width {
        for copy in 0..hc {
            for j in 0..dim {
                let mut comb_sum = 0.0f32;
                for source in 0..hc {
                    comb_sum += comb[token * hc * hc + source * hc + copy]
                        * residual_hc[token * token_width + source * dim + j];
                }
                let post_term = post_mix[token * hc + copy] * x[token * dim + j];
                out_hc[token * token_width + copy * dim + j] = post_term + comb_sum;
            }
        }
    }
    bf16_round_row(out_hc);
    Ok(())
}

/// A shape refusal naming the operation that could not be applied.
fn shape_error(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}
