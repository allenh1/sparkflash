//! Single-pass mHC: the coefficient mixes, the split sinkhorn and the two reductions.
//!
//! `docs/engine/01-forward-order.md` section 3.2 is the specification of every function here.
//! `Block.forward` (model.py:968-994) keeps the hidden state as an `hc_mult`-copy stream and puts
//! three coefficient sets around each sublayer:
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
//! `hc_mult + hc_mult + hc_mult^2` values - exactly [crate::config::ModelConfig::hc_mix_count].
//! These functions take a whole chunk: `n` tokens are `n * hc_mult * dim` stream values, every
//! coefficient buffer holds one set per token laid out token by token - [Mixes::for_tokens] sizes
//! them, [Mixes::for_config] is one token's - and the reductions write one row per token. Every
//! length is checked against the configuration and against the other buffers rather than trusted,
//! so a caller that sizes a buffer for the wrong number of tokens is refused with
//! [crate::linalg::EngineError::Shape] instead of being silently half-processed.
//!
//! This is an `f32` reference over decoded weights: the released kernels quantise the activation of
//! the `hc_*_fn` GEMM (section 6 point 1 of the specification), and no quantised kernel runs on
//! this path (`forward/mod.rs`).

use crate::config::ModelConfig;
use crate::linalg::{self, EngineError};

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
    /// sinkhorn (kernel.py:436-458). Column `k` mixes input copy `k`, row `r` is output copy `r`.
    pub comb: Vec<f32>,
}

impl Mixes {
    /// One token's zeroed coefficient sets: `pre` and `post` are `hc_mult` long and `comb` is
    /// `hc_mult * hc_mult`. This is [Self::for_tokens] with one token.
    #[must_use]
    pub fn for_config(cfg: &ModelConfig) -> Self {
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
    pub fn for_tokens(cfg: &ModelConfig, positions: usize) -> Self {
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
pub fn identity_pre_mix(cfg: &ModelConfig, out: &mut [f32]) -> Result<(), EngineError> {
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
/// values, with `norm_eps = 1e-20` - **not** `hc_eps = 1e-6`, which is the epsilon of the sinkhorn
/// below (model.py:949-950). `fn_w` is the `[mix_hc, hc * dim]` `hc_*_fn` weight,
/// `mixes = fn_w * x_hc * rsqrt` (model.py:951), and `scale`/`base` are the `[3]` and `[mix_hc]`
/// tensors that `hc_split_sinkhorn` applies, to every token:
///
/// - `pre = sigmoid(mix[0:hc] * scale[0] + base[0:hc]) + hc_eps` (kernel.py:427),
/// - `post = 2 * sigmoid(mix[hc:2hc] * scale[1] + base[hc:2hc])`, no epsilon (kernel.py:429),
/// - `comb = mix[2hc:] * scale[2] + base[2hc:]`, reshaped `[hc, hc]`, then normalised
///   (kernel.py:430-431, 436-458).
///
/// The normalisation runs `2 * hc_sinkhorn_iters` passes per token: one row softmax (plus `hc_eps`),
/// then a column normalisation, then `hc_sinkhorn_iters - 1` pairs of a row and a column
/// normalisation, with `hc_eps` in every denominator. The last pass normalises columns and each
/// column of `comb` sums to 1, which is the 40 passes the released `hc_sinkhorn_iters = 20` runs
/// (kernel.py:436-458; specification section 3.2).
///
/// `mixes` must hold one set per token ([Mixes::for_tokens]): the token count is
/// `x_hc.len() / hc_width()` and every buffer must agree with it.
///
/// # Errors
///
/// [EngineError::Shape] when any slice's length disagrees with the configuration or with another
/// slice.
pub fn hc_mixes(
    cfg: &ModelConfig,
    x_hc: &[f32],
    fn_w: &[f32],
    scale: &[f32],
    base: &[f32],
    mixes: &mut Mixes,
) -> Result<(), EngineError> {
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

    let hc_eps = cfg.hc_eps as f32;
    let norm_eps = cfg.norm_eps as f32;
    for token in 0..tokens {
        let token_x = &x_hc[token * width..(token + 1) * width];
        // mean(x^2) over this token's flattened stream. This is the specification's own
        // x.pow(2).mean(-1) (model.py:949-950): [linalg::mean] averages a slice, and materialising
        // x^2 into a buffer to hand it one would allocate inside a per-token loop.
        let mean_square = token_x.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let rsqrt = 1.0 / (mean_square + norm_eps).sqrt();

        for (coefficient, bias) in base.iter().enumerate().take(hc) {
            let mix = project(fn_w, coefficient, width, token_x)? * rsqrt;
            mixes.pre[token * hc + coefficient] = linalg::sigmoid(mix * scale[0] + *bias) + hc_eps;
        }
        for coefficient in 0..hc {
            let row = hc + coefficient;
            let mix = project(fn_w, row, width, token_x)? * rsqrt;
            mixes.post[token * hc + coefficient] =
                2.0 * linalg::sigmoid(mix * scale[1] + base[row]);
        }
        for row in 0..hc {
            for col in 0..hc {
                let coefficient = 2 * hc + row * hc + col;
                let mix = project(fn_w, coefficient, width, token_x)? * rsqrt;
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
fn split_sinkhorn(comb: &mut [f32], hc: usize, iters: usize, hc_eps: f32) {
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
    cfg: &ModelConfig,
    pre_mix: &[f32],
    x_hc: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
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
/// `out_hc[t * hc * dim + c * dim + j] = post_mix[t * hc + c] * x[t * dim + j] +
/// sum_k comb[t * hc * hc + c * hc + k] * residual_hc[t * hc * dim + k * dim + j]`, the
/// `post (.) x + comb (.) residual` of the specification. `comb` is row-major, `[hc, hc]` per
/// token, with row `c` the output copy and column `k` the input copy, which is why column `k` sums
/// to 1 after the sinkhorn: copy `k` of the residual is distributed over the outputs. The token
/// count is `x.len() / dim` and every buffer must agree with it.
///
/// # Errors
///
/// [EngineError::Shape] when the slices do not describe a whole number of tokens of this
/// configuration.
pub fn hc_post(
    cfg: &ModelConfig,
    x: &[f32],
    residual_hc: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    out_hc: &mut [f32],
) -> Result<(), EngineError> {
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
                for input_copy in 0..hc {
                    accumulated += comb[token * hc * hc + copy * hc + input_copy]
                        * residual_hc[token * hc * dim + input_copy * dim + j];
                }
                out_hc[token * hc * dim + copy * dim + j] = accumulated;
            }
        }
    }
    Ok(())
}

/// A shape refusal naming the operation that could not be applied.
fn shape_error(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorPlan;

    /// A `[mix_hc, width]` `hc_*_fn` weight whose listed rows read `x_hc[0]` and whose other rows
    /// are zero.
    fn fn_weight(mix_hc: usize, width: usize, rows: &[(usize, f32)]) -> Vec<f32> {
        let mut weight = vec![0.0; mix_hc * width];
        for (row, value) in rows {
            assert!(*row < mix_hc, "row {row} is outside {mix_hc} coefficients");
            weight[row * width] = *value;
        }
        weight
    }

    /// `pre` and `post` are one mix per token and `comb` is one matrix per token: `hc_mult`,
    /// `hc_mult` and `hc_mult * hc_mult`. The `hc_*_fn` weight is the wider `hc_mix_count()` rows,
    /// so the two sizes have to be told apart: 2/2/4 and 8 rows at the tiny configuration's
    /// `hc_mult = 2`, 4/4/16 and 24 rows at the released 4. The plan carries the same geometry, and
    /// that is what is asserted here rather than a literal.
    #[test]
    fn the_coefficient_sets_are_sized_from_the_configuration() {
        let cfg = ModelConfig::tiny();
        let mixes = Mixes::for_config(&cfg);
        assert_eq!(
            (mixes.pre.len(), mixes.post.len(), mixes.comb.len()),
            (cfg.hc_mult, cfg.hc_mult, cfg.hc_mult * cfg.hc_mult)
        );
        assert_eq!((mixes.pre.len(), mixes.comb.len()), (2, 4));
        assert_eq!(cfg.hc_mix_count(), 8);

        let released = ModelConfig::v41_flash_cached();
        let mixes = Mixes::for_config(released);
        assert_eq!(
            (mixes.pre.len(), mixes.post.len(), mixes.comb.len()),
            (
                released.hc_mult,
                released.hc_mult,
                released.hc_mult * released.hc_mult
            )
        );
        assert_eq!((mixes.pre.len(), mixes.comb.len()), (4, 16));
        assert_eq!(released.hc_mix_count(), 24);

        let plan = TensorPlan::language_model(&cfg);
        for (name, shape) in [
            (
                "layers.0.hc_attn_fn",
                vec![cfg.hc_mix_count(), cfg.hc_width()],
            ),
            ("layers.0.hc_attn_base", vec![cfg.hc_mix_count()]),
            ("layers.0.hc_attn_scale", vec![3]),
        ] {
            let spec = plan
                .specs()
                .iter()
                .find(|spec| spec.name == name)
                .unwrap_or_else(|| panic!("the plan names {name}"));
            assert_eq!(spec.shape, shape, "{name}");
        }
    }

    /// `identity_pre_mix` puts 1 on copy 0 and 0 on every other copy, and refuses a buffer that is
    /// not `hc_mult` long.
    #[test]
    fn the_identity_pre_mix_is_a_one_hot_on_copy_zero() {
        let cfg = ModelConfig::tiny();
        let mut out = vec![9.0; cfg.hc_mult];
        identity_pre_mix(&cfg, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);

        let released = ModelConfig::v41_flash_cached();
        let mut out = vec![9.0; released.hc_mult];
        identity_pre_mix(released, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        assert!(out[1..].iter().all(|value| *value == 0.0));

        assert!(matches!(
            identity_pre_mix(&cfg, &mut [0.0; 3]),
            Err(EngineError::Shape { .. })
        ));
    }

    /// The mix is `fn_w * x * rsqrt(mean(x^2) + norm_eps)` with `norm_eps = 1e-20` under the root
    /// and the mean taken over the whole flattened token. Two mutants are pinned by two tokens: the
    /// deliberately tiny one makes `hc_eps` (1e-6) as the epsilon visible, and the one whose second
    /// copy dominates the mean but not the projection makes a mean over `dim` instead of over
    /// `hc_mult * dim` visible, because the two denominators differ by `sqrt(hc_mult)`.
    #[test]
    fn the_mixes_are_the_rms_scaled_projection_with_norm_eps() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let dim = cfg.dim;
        let mix_hc = cfg.hc_mix_count();
        let base = vec![0.0; mix_hc];
        let fn_w = fn_weight(mix_hc, width, &[(0, 1.0)]);

        // A token so small that the epsilon under the root decides the value.
        let mut tiny = vec![0.0; width];
        tiny[0] = 1e-4;
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &tiny, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        let mean_square = 1e-4f32 * 1e-4 / width as f32;
        let expected = linalg::sigmoid(1e-4 * (1.0 / (mean_square + 1e-20f32).sqrt())) + 1e-6;
        assert!(
            (mixes.pre[0] - expected).abs() < 1e-6,
            "pre[0] is {} but the specification's rsqrt gives {expected}",
            mixes.pre[0]
        );
        let with_hc_eps = linalg::sigmoid(1e-4 * (1.0 / (mean_square + 1e-6f32).sqrt())) + 1e-6;
        assert!(
            (expected - with_hc_eps).abs() > 1e-3,
            "the two epsilons must be distinguishable at this token"
        );

        // A token whose copy 1 dominates the mean but not the row 0 projection.
        let mut spread = vec![0.0; width];
        spread[0] = 1.0;
        spread[dim] = 3.0;
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &spread, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        let sum_squares = 1.0f32 + 3.0 * 3.0;
        let flattened =
            linalg::sigmoid(1.0 / (sum_squares / width as f32 + 1e-20f32).sqrt()) + 1e-6;
        let per_dim = linalg::sigmoid(1.0 / (sum_squares / dim as f32 + 1e-20f32).sqrt()) + 1e-6;
        assert!(
            (mixes.pre[0] - flattened).abs() < 1e-6,
            "pre[0] is {} but the flattened mean gives {flattened}",
            mixes.pre[0]
        );
        assert!(
            (flattened - per_dim).abs() > 1e-3,
            "the mean must be over the flattened token: {flattened} against {per_dim}"
        );
    }

    /// `pre` is `sigmoid(...) + hc_eps` and `post` is `2 * sigmoid(...)` with no epsilon
    /// (kernel.py:427, 429). A zero weight makes both mixes their base value, so the epsilon is
    /// the whole difference: dropping it from `pre` or adding it to `post` fails here.
    #[test]
    fn pre_carries_an_epsilon_and_post_does_not() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let x_hc = vec![0.0; width];
        let fn_w = vec![0.0; mix_hc * width];
        let base = vec![0.0; mix_hc];
        let mut mixes = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
        for value in &mixes.pre {
            assert_eq!(*value, 0.5 + 1e-6, "pre adds hc_eps to sigmoid(0)");
        }
        for value in &mixes.post {
            assert_eq!(*value, 1.0, "post is 2 * sigmoid(0) with no epsilon");
        }
    }

    /// The split sinkhorn (kernel.py:436-458) runs `2 * iters` passes: a row softmax plus `hc_eps`,
    /// then ONE column normalisation outside the loop, then `iters - 1` pairs of a row and a column
    /// normalisation. The release's `hc_sinkhorn_iters = 20` is therefore 20 row and 20 column passes,
    /// and the last one is a column pass, so the result is column-stochastic and not row-stochastic:
    /// at `iters = 1` the single unconditional column pass already makes the columns sum to 1 (to
    /// within the `hc_eps` in every denominator). The test
    /// writes that schedule out from the kernel lines and compares the two entry for entry at iters 1,
    /// 2, 3 and 20, then asserts the power that comparison actually has: the softmax alone (the
    /// unconditional column pass folded into the loop) is 0.5 away at 1, and at 20 a schedule one pass
    /// short is 0.026 away and the `iters`-pass schedule is 0.025 away. All of those are far above the
    /// 1e-6 comparison, so neither a count nor a final-axis error can hide in convergence.
    #[test]
    fn the_split_sinkhorn_runs_two_passes_per_iteration() {
        let hc = 4usize;
        let hc_eps = 1e-6f32;
        // A wide dynamic range, so every normalisation keeps moving. A comb whose rows are
        // proportional - an affine ramp, say - is at the fixed point after one column pass and cannot
        // tell the schedules apart at all.
        let raw: Vec<f32> = (0..hc * hc)
            .map(|index| (0.4 * ((index * 3) % (hc * hc)) as f32).exp())
            .collect();
        let max_diff = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max)
        };
        let row_pass = |comb: &mut [f32]| {
            for row in comb.chunks_mut(hc) {
                let sum: f32 = row.iter().sum();
                for value in row.iter_mut() {
                    *value /= sum + hc_eps;
                }
            }
        };
        let column_pass = |comb: &mut [f32]| {
            for col in 0..hc {
                let sum: f32 = (0..hc).map(|row| comb[row * hc + col]).sum();
                for row in 0..hc {
                    comb[row * hc + col] /= sum + hc_eps;
                }
            }
        };
        // The reference ends on a column pass, so any schedule that ends on a row pass, or that
        // leaves the unconditional column pass out, is a different matrix here. The column SUMS are
        // checked end to end below, on the engine's own comb: with this comb's range the `hc_eps` in
        // every denominator is not negligible against a vanishing column sum, so the reference only
        // makes the columns sum to 1 to within `hc_eps`, and the tight version of that property belongs
        // on a comb whose columns are O(1).
        for iters in [1usize, 2, 3, 20] {
            let mut mine = raw.clone();
            split_sinkhorn(&mut mine, hc, iters, hc_eps);
            let mut expected = raw.clone();
            for row in expected.chunks_mut(hc) {
                linalg::softmax(row);
                for value in row.iter_mut() {
                    *value += hc_eps;
                }
            }
            column_pass(&mut expected);
            for _ in 1..iters {
                row_pass(&mut expected);
                column_pass(&mut expected);
            }
            for (index, (got, want)) in mine.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "iters={iters} entry {index}: {got} against {want}"
                );
            }
        }

        // The power the comparison above has, asserted rather than assumed. First the count: the
        // `iters`-pass schedule the module used to run, which is the softmax and then `iters - 1`
        // passes alternating column and row.
        let mut full = raw.clone();
        split_sinkhorn(&mut full, hc, 20, hc_eps);
        let mut alternating = raw.clone();
        for row in alternating.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        for pass in 1..20 {
            if pass % 2 == 1 {
                column_pass(&mut alternating);
            } else {
                row_pass(&mut alternating);
            }
        }
        let count = max_diff(&full, &alternating);
        assert!(
            count > 1e-3,
            "the 20-pass schedule must be distinguishable from the kernel's 40, got {count}"
        );

        // Then a schedule one pass short at 20: the kernel's, without its last column pass. This is
        // what an off-by-one loop gives, and it must be visible for the comparison to mean anything.
        let mut one_short = raw.clone();
        for row in one_short.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        column_pass(&mut one_short);
        for _ in 1..19 {
            row_pass(&mut one_short);
            column_pass(&mut one_short);
        }
        row_pass(&mut one_short);
        let short_at_twenty = max_diff(&full, &one_short);
        assert!(
            short_at_twenty > 1e-3,
            "a schedule one pass short must be visible at 20, got {short_at_twenty}"
        );

        // And the unconditional column pass folded into the loop, which at iters = 1 leaves the softmax
        // alone: that is what the module used to do, and what the old test asserted the columns of.
        let mut softmax_only = raw.clone();
        for row in softmax_only.chunks_mut(hc) {
            linalg::softmax(row);
            for value in row.iter_mut() {
                *value += hc_eps;
            }
        }
        let mut one_iter = raw.clone();
        split_sinkhorn(&mut one_iter, hc, 1, hc_eps);
        let folded = max_diff(&one_iter, &softmax_only);
        assert!(
            folded > 1e-2,
            "the unconditional column pass must move the iters = 1 result, got {folded}"
        );

        // End to end through hc_mixes: the columns already sum to 1 at iters = 1, because the column
        // pass after the softmax is unconditional. The module used to leave them at 0.25 and 1.75.
        let rows = [(4usize, 0.1f32), (5, 0.5), (6, 0.2), (7, 0.3)];
        let mut cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut x_hc = vec![0.0; width];
        x_hc[0] = 1.0;
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base = vec![0.0; mix_hc];
        for iters in [1usize, 2, 20] {
            cfg.hc_sinkhorn_iters = iters;
            let mut mixes = Mixes::for_config(&cfg);
            hc_mixes(&cfg, &x_hc, &fn_w, &[1.0, 1.0, 1.0], &base, &mut mixes).unwrap();
            let hc_mult = cfg.hc_mult;
            for col in 0..hc_mult {
                let sum: f32 = (0..hc_mult)
                    .map(|row| mixes.comb[row * hc_mult + col])
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "iters={iters} column {col} sums to {sum}"
                );
            }
            if iters == 1 {
                // At iters = 1 the single column pass is the last pass, so the columns are 1 and the
                // rows are not: the module used to run the softmax alone, which leaves the rows at 1 and
                // the columns at 0.25 and 1.75. After 40 passes the matrix is converged and its rows are
                // 1 as well, so this contrast belongs here; the final axis at 20 is pinned by the direct
                // comparison above, where a row-ending schedule is 0.026 away.
                let row_sums: Vec<f32> = (0..hc_mult)
                    .map(|row| mixes.comb[row * hc_mult..(row + 1) * hc_mult].iter().sum())
                    .collect();
                assert!(
                    row_sums.iter().any(|sum| (sum - 1.0).abs() > 1e-3),
                    "iters=1 rows {row_sums:?} must not all sum to 1: the last pass normalised columns"
                );
            }
        }
    }

    /// `hc_pre` collapses the copies with the pre-mix: `sum_c pre[c] * x_hc[c * dim + j]`.
    #[test]
    fn hc_pre_collapses_the_copies_with_the_pre_mix() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let mut x_hc = vec![0.0; hc * dim];
        for j in 0..dim {
            x_hc[j] = j as f32;
            x_hc[dim + j] = 2.0 * j as f32;
        }
        let mut out = vec![0.0; dim];
        hc_pre(&cfg, &[0.25, 0.75], &x_hc, &mut out).unwrap();
        for (j, value) in out.iter().enumerate() {
            assert_eq!(*value, 1.75 * j as f32);
        }
    }

    /// `hc_post` is `post[c] * x + sum_k comb[c][k] * residual[k]` with the comb row-major and its
    /// second index the input copy. The comb here is asymmetric: transposing it would give 1.5 for
    /// the first copy instead of 1.9, so a flipped index is caught.
    #[test]
    fn hc_post_is_post_times_x_plus_the_comb_mix() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let x = vec![1.0; dim];
        let mut residual = vec![0.0; hc * dim];
        for j in 0..dim {
            residual[j] = 1.0;
            residual[dim + j] = 2.0;
        }
        let comb = [0.6, 0.4, 0.2, 0.8];
        let mut out = vec![0.0; hc * dim];
        hc_post(&cfg, &x, &residual, &[0.5, 0.25], &comb, &mut out).unwrap();
        for (j, value) in out.iter().take(dim).enumerate() {
            assert!(
                (*value - 1.9).abs() < 1e-6,
                "copy 0 value at {j} is {value}"
            );
        }
        for (j, value) in out.iter().skip(dim).enumerate() {
            assert!(
                (*value - 2.05).abs() < 1e-6,
                "copy 1 value at {j} is {value}"
            );
        }
    }

    /// The same token and the same weights give bit-identical mixes, and a different token does
    /// not: the second assertion is what fails if a term of the projection is dropped.
    #[test]
    fn the_mixes_are_deterministic_and_depend_on_the_token() {
        let cfg = ModelConfig::tiny();
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut x_hc = vec![0.0; width];
        x_hc[0] = 2.0;
        x_hc[width - 1] = -1.5;
        let rows: Vec<(usize, f32)> = (0..mix_hc)
            .map(|row| (row, 0.5 + row as f32 * 0.25))
            .collect();
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base: Vec<f32> = (0..mix_hc).map(|index| index as f32 * 0.01).collect();
        let scale = [0.5, 1.5, 2.0];

        let mut first = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut first).unwrap();
        let mut second = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut second).unwrap();
        assert_eq!(first, second, "the same token must give the same mixes");

        let mut other_token = x_hc.clone();
        other_token[0] = 3.0;
        let mut third = Mixes::for_config(&cfg);
        hc_mixes(&cfg, &other_token, &fn_w, &scale, &base, &mut third).unwrap();
        assert_ne!(first.pre, third.pre, "the mixes must depend on the token");
    }

    /// Every function refuses a geometry the configuration does not describe rather than reading
    /// past a slice.
    #[test]
    fn the_mhc_reductions_refuse_geometry_that_does_not_fit() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let x_hc = vec![0.0; width];
        let fn_w = vec![0.0; mix_hc * width];
        let base = vec![0.0; mix_hc];
        let scale = [1.0, 1.0, 1.0];
        let mut mixes = Mixes::for_config(&cfg);

        assert!(hc_mixes(&cfg, &x_hc[1..], &fn_w, &scale, &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w[1..], &scale, &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale[..2], &base, &mut mixes).is_err());
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base[1..], &mut mixes).is_err());
        let mut wrong = Mixes {
            pre: vec![0.0; hc - 1],
            post: vec![0.0; hc],
            comb: vec![0.0; hc * hc],
        };
        assert!(hc_mixes(&cfg, &x_hc, &fn_w, &scale, &base, &mut wrong).is_err());

        assert!(hc_pre(&cfg, &[1.0], &x_hc, &mut vec![0.0; dim]).is_err());
        assert!(hc_pre(&cfg, &[0.5, 0.5], &x_hc[1..], &mut vec![0.0; dim]).is_err());
        assert!(hc_pre(&cfg, &[0.5, 0.5], &x_hc, &mut vec![0.0; dim - 1]).is_err());

        let residual = vec![0.0; hc * dim];
        let comb = vec![0.0; hc * hc];
        let post = vec![0.0; hc];
        assert!(
            hc_post(
                &cfg,
                &[0.0; 1],
                &residual,
                &post,
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual[1..],
                &post,
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post[1..],
                &comb,
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post,
                &comb[1..],
                &mut vec![0.0; hc * dim]
            )
            .is_err()
        );
        assert!(
            hc_post(
                &cfg,
                &[0.0; 64],
                &residual,
                &post,
                &comb,
                &mut vec![0.0; hc * dim - 1]
            )
            .is_err()
        );
    }

    /// A chunk gets one coefficient set per token, laid out token by token: each slot equals what a
    /// one-token call on that token gives, and a `Mixes` sized for one token is refused for a
    /// two-token chunk rather than half-filled.
    #[test]
    fn a_chunk_gets_one_coefficient_set_per_token() {
        let cfg = ModelConfig::tiny();
        let hc = cfg.hc_mult;
        let width = cfg.hc_width();
        let mix_hc = cfg.hc_mix_count();
        let mut first = vec![0.0f32; width];
        first[0] = 2.0;
        let mut second = vec![0.0f32; width];
        second[1] = -3.0;
        second[width - 1] = 4.0;
        let rows: Vec<(usize, f32)> = (0..mix_hc)
            .map(|row| (row, 0.25 + row as f32 * 0.5))
            .collect();
        let fn_w = fn_weight(mix_hc, width, &rows);
        let base: Vec<f32> = (0..mix_hc).map(|index| index as f32 * 0.02).collect();
        let scale = [1.0, 1.5, 0.5];

        let mut chunk = vec![0.0f32; 2 * width];
        chunk[..width].copy_from_slice(&first);
        chunk[width..].copy_from_slice(&second);
        let mut mixes = Mixes::for_tokens(&cfg, 2);
        hc_mixes(&cfg, &chunk, &fn_w, &scale, &base, &mut mixes).unwrap();

        for (slot, token) in [&first, &second].into_iter().enumerate() {
            let mut alone = Mixes::for_config(&cfg);
            hc_mixes(&cfg, token, &fn_w, &scale, &base, &mut alone).unwrap();
            assert_eq!(&mixes.pre[slot * hc..(slot + 1) * hc], &alone.pre[..]);
            assert_eq!(&mixes.post[slot * hc..(slot + 1) * hc], &alone.post[..]);
            assert_eq!(
                &mixes.comb[slot * hc * hc..(slot + 1) * hc * hc],
                &alone.comb[..]
            );
        }

        let mut too_small = Mixes::for_config(&cfg);
        assert!(
            hc_mixes(&cfg, &chunk, &fn_w, &scale, &base, &mut too_small).is_err(),
            "a one-token Mixes cannot hold a two-token chunk"
        );
    }

    /// The reductions work a whole chunk: every token's row is the collapse of its own copies and
    /// the mix of its own residual, so a loop that reused token 0's buffers fails here.
    #[test]
    fn the_reductions_work_a_whole_chunk() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let token_width = hc * dim;
        let mut x_hc = vec![0.0f32; 2 * token_width];
        for token in 0..2 {
            for copy in 0..hc {
                for j in 0..dim {
                    x_hc[token * token_width + copy * dim + j] =
                        (token + 1) as f32 + copy as f32 * 10.0 + j as f32 * 0.5;
                }
            }
        }
        let mut pre = vec![0.0f32; 2 * hc];
        for token in 0..2 {
            pre[token * hc] = 0.25;
            pre[token * hc + 1] = 0.75;
        }
        let mut out = vec![0.0f32; 2 * dim];
        hc_pre(&cfg, &pre, &x_hc, &mut out).unwrap();
        for token in 0..2 {
            for j in 0..dim {
                let expected = 0.25 * x_hc[token * token_width + j]
                    + 0.75 * x_hc[token * token_width + dim + j];
                assert!(
                    (out[token * dim + j] - expected).abs() < 1e-4,
                    "token {token} dim {j}"
                );
            }
        }

        let x = vec![1.0f32; 2 * dim];
        let post_mix = [0.5f32, 0.25];
        let comb = [0.6f32, 0.4, 0.2, 0.8];
        let mut post = vec![0.0f32; 2 * hc];
        let mut combs = vec![0.0f32; 2 * hc * hc];
        for token in 0..2 {
            post[token * hc..(token + 1) * hc].copy_from_slice(&post_mix);
            combs[token * hc * hc..(token + 1) * hc * hc].copy_from_slice(&comb);
        }
        let mut out_hc = vec![0.0f32; 2 * token_width];
        hc_post(&cfg, &x, &x_hc, &post, &combs, &mut out_hc).unwrap();
        for token in 0..2 {
            for j in 0..dim {
                let expected = 0.5 * x[token * dim + j]
                    + 0.6 * x_hc[token * token_width + j]
                    + 0.4 * x_hc[token * token_width + dim + j];
                assert!(
                    (out_hc[token * token_width + j] - expected).abs() < 1e-4,
                    "token {token} dim {j}"
                );
            }
        }
    }
}
