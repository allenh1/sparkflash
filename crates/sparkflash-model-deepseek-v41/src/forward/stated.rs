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

//! The value types a description states, on the host's shared helpers.
//!
//! The base's host forward holds its values in f32 and rounds to BF16 where DeepSeek's reference
//! does. A description whose layout states a BF16 stream ([StatedWiring::bf16_stream]) holds each
//! module output in BF16, as its producer's modules return it, and quantises an FP8 linear's
//! input on the stated tiles ([StatedWiring::activation_tile]). Each helper here is the release
//! arithmetic when nothing is stated, so a DeepSeek call site keeps its bits.
//!
//! [StatedWiring::bf16_stream]: crate::config::StatedWiring::bf16_stream
//! [StatedWiring::activation_tile]: crate::config::StatedWiring::activation_tile

use super::weight_view::WeightView;
use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::quant;
use crate::weights::WeightSource;

/// Every value narrowed to BF16 in place, round-to-nearest-even.
pub(crate) fn round_bf16(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = quant::bf16_round(*value);
    }
}

/// Round to BF16 when the stream is stated BF16; the f32 values otherwise.
pub(crate) fn hold(config: &ModelConfig, values: &mut [f32]) {
    if config.stated().bf16_stream {
        round_bf16(values);
    }
}

/// The input an FP8 linear multiplies: each `inputs`-wide row quantised on the stated tiles
/// (`amax / 448` an f32 scale a tile, E4M3 codes) and decoded back, as the release's dynamic
/// scheme hands its GEMM; `None` when nothing is stated or the weight is not FP8.
///
/// # Errors
/// A row that is not whole tiles, or a value the quantiser refuses.
fn quantised_input(
    config: &ModelConfig,
    source: &dyn WeightSource,
    weight: &str,
    input: &[f32],
    inputs: usize,
) -> Result<Option<Vec<f32>>, EngineError> {
    let tile = config.stated().activation_tile;
    if tile == 0 || source.spec(weight)?.dtype != DType::F8E4M3 {
        return Ok(None);
    }
    if inputs == 0 || !inputs.is_multiple_of(tile) {
        return Err(EngineError::Shape {
            what: format!("{weight}'s input"),
            detail: format!("a {inputs}-wide row is not whole {tile}-wide activation tiles"),
        });
    }
    let mut out = Vec::with_capacity(input.len());
    for row in input.chunks(inputs) {
        out.extend(
            quant::dequantized_activation_fp8_tiles(row, tile).map_err(|error| {
                EngineError::Shape {
                    what: format!("{weight}'s input"),
                    detail: error.to_string(),
                }
            })?,
        );
    }
    Ok(Some(out))
}

/// `{base}.weight` (with its scale when the source has one) times every `inputs`-wide row of
/// `input`: the input quantised first when the description states an activation tile and the
/// weight is FP8, the output held in the stated stream type.
///
/// # Errors
/// A weight that is not `[outputs, inputs]`, a read error, or the quantiser's refusal.
#[allow(clippy::too_many_arguments)] // one linear's operands; bundling them would hide the shape
pub(crate) fn project(
    config: &ModelConfig,
    source: &dyn WeightSource,
    base: &str,
    outputs: usize,
    inputs: usize,
    input: &[f32],
    grid: ScaleGrid,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let view = WeightView::read_optional_scale(source, base, outputs, inputs, grid, limit)?;
    let quantised = quantised_input(config, source, &format!("{base}.weight"), input, inputs)?;
    let operand = quantised.as_deref().unwrap_or(input);
    let count = operand.len() / inputs.max(1);
    let mut projected = vec![0.0f32; count * outputs];
    view.matmul(&mut projected, operand, outputs, inputs, count)?;
    hold(config, &mut projected);
    Ok(projected)
}

/// [project] through a view the caller already holds (an expert body's), with the weight's
/// stored name for the activation rule.
///
/// # Errors
/// As [project].
pub(crate) fn project_view(
    config: &ModelConfig,
    source: &dyn WeightSource,
    view: &WeightView<'_>,
    weight: &str,
    (outputs, inputs): (usize, usize),
    input: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    let quantised = quantised_input(config, source, weight, input, inputs)?;
    let operand = quantised.as_deref().unwrap_or(input);
    let count = operand.len() / inputs.max(1);
    view.matmul(out, operand, outputs, inputs, count)?;
    hold(config, out);
    Ok(())
}

/// A sublayer's RMSNorm in the stated type: on a BF16 stream the producer's, which normalises in
/// f32, rounds to BF16, then multiplies by the BF16 weight and rounds again
/// (`modeling_glm5_next.py` 74-79); the release's f32 norm otherwise.
///
/// # Errors
/// [linalg::rms_norm]'s shape refusals.
pub(crate) fn rms_norm(
    config: &ModelConfig,
    rows: &mut [f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), EngineError> {
    let dim = weight.len();
    if !config.stated().bf16_stream {
        return linalg::rms_norm(rows, weight, dim, eps);
    }
    if dim == 0 || !rows.len().is_multiple_of(dim) {
        return Err(EngineError::Shape {
            what: "the stated RMSNorm".to_owned(),
            detail: format!("{} values are not whole {dim}-wide rows", rows.len()),
        });
    }
    for row in rows.chunks_mut(dim) {
        let mean = row.iter().map(|value| value * value).sum::<f32>() / dim as f32;
        let inverse = 1.0 / (mean + eps).sqrt();
        for (value, scale) in row.iter_mut().zip(weight) {
            *value = quant::bf16_round(*scale * quant::bf16_round(*value * inverse));
        }
    }
    Ok(())
}

/// The stream update after a sublayer in the stated type. On a BF16 stream it is the producer's
/// expression (`modeling_glm5_next.py` 1317-1328): `post` and `comb` cast to BF16, the post term
/// `post * out` one BF16 product, the comb term one BF16 matmul over the source copies (f32
/// accumulation, one round), and their sum rounded. Otherwise the shared [super::mhc::hc_post].
///
/// # Errors
/// [super::mhc::hc_post]'s shape refusals.
pub(crate) fn hc_post(
    config: &ModelConfig,
    x: &[f32],
    residual_hc: &[f32],
    post_mix: &[f32],
    comb: &[f32],
    out_hc: &mut [f32],
) -> Result<(), EngineError> {
    super::mhc::hc_post(config, x, residual_hc, post_mix, comb, out_hc)?;
    if !config.stated().bf16_stream {
        return Ok(());
    }
    let (hc, dim) = (config.hc_mult, config.dim);
    let width = hc * dim;
    let bf16 = quant::bf16_round;
    for token in 0..out_hc.len() / width {
        for copy in 0..hc {
            let post = bf16(post_mix[token * hc + copy]);
            for column in 0..dim {
                let mut mixed = 0.0f32;
                for source in 0..hc {
                    mixed += bf16(comb[token * hc * hc + source * hc + copy])
                        * residual_hc[token * width + source * dim + column];
                }
                let expanded = bf16(post * x[token * dim + column]);
                out_hc[token * width + copy * dim + column] = bf16(expanded + bf16(mixed));
            }
        }
    }
    Ok(())
}

/// The collapse a sublayer reads, in the stated type: `(pre * streams).sum` over the copies in
/// f32, narrowed to the stream type (`modeling_glm5_next.py` 295).
///
/// # Errors
/// [super::mhc::hc_pre]'s shape refusals.
pub(crate) fn hc_pre(
    config: &ModelConfig,
    pre_mix: &[f32],
    x_hc: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    super::mhc::hc_pre(config, pre_mix, x_hc, out)?;
    hold(config, out);
    Ok(())
}

/// The head's fold of the streams: the mean over the copies when the description states it
/// (`modeling_glm5_next.py` 332-336, accumulated in f32 and narrowed once), else the base's
/// collapse through the last pre-mix.
///
/// # Errors
/// [super::mhc::hc_pre]'s shape refusals.
pub(crate) fn head_fold(
    config: &ModelConfig,
    pre_mix: &[f32],
    x_hc: &[f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    if !config.stated().mean_fold {
        return super::mhc::hc_pre(config, pre_mix, x_hc, out);
    }
    let (hc, dim) = (config.hc_mult, config.dim);
    if hc == 0 || x_hc.len() != out.len() * hc {
        return Err(EngineError::Shape {
            what: "the head fold".to_owned(),
            detail: format!(
                "{} stream values do not fold {hc} copies into {}",
                x_hc.len(),
                out.len()
            ),
        });
    }
    for (token, row) in out.chunks_mut(dim).enumerate() {
        for (column, value) in row.iter_mut().enumerate() {
            let sum: f32 = (0..hc)
                .map(|copy| x_hc[(token * hc + copy) * dim + column])
                .sum();
            *value = sum / hc as f32;
        }
    }
    hold(config, out);
    Ok(())
}
