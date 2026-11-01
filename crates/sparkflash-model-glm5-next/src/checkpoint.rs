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

//! A synthetic checkpoint for any configuration, in the released storage formats.
//!
//! [write_synthetic_checkpoint] writes every tensor the served plan (and the MTP layer) names, with
//! the released dtype: block-scaled FP8 codes with their F32 tile scales, BF16 and F32 elsewhere. The
//! host reference loads it through the same [sparkflash_weights::safetensors::SafeTensorsSource] a
//! released checkpoint goes through, so the tiny fixture exercises the whole load: names, dtypes,
//! tile scales.
//!
//! The values are chosen for arithmetic coverage, not fidelity, and so that nothing degenerates:
//! a matrix entry is uniform with a width of `1 / sqrt(columns)`, so a product keeps the scale of
//! its input; norm weights sit near one; FP8 codes span the whole E4M3 range and their tile scales
//! bring the product back to the same width; `A_log`, `dt_bias`, the router bias and the mHC base
//! and scale are small numbers of their own.

use std::path::Path;

use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::write_checkpoint;
use sparkflash_weights::{DType, TensorSpec, quant};

use crate::config::Glm5NextConfig;
use crate::tensor::Glm5NextTensorPlan;

/// The widest a synthetic FP8 code is drawn: the E4M3 maximum.
const CODE_RANGE: f32 = quant::FP8_E4M3_MAX;

/// The f32 value of one element of a tensor that is not FP8 coded.
fn value_at(spec: &TensorSpec, rng: &mut TensorRng) -> f32 {
    let name = spec.name.as_str();
    if name.ends_with("A_log") {
        // exp(A_log) between e^-0.5 and e^1.5: the forget gate's sigmoid is neither flat nor
        // saturated, so the decay spans most of (e^lb, 1).
        return 0.5 + rng.signed();
    }
    if name.ends_with("dt_bias") || name.ends_with("k_norm.bias") {
        return 0.1 * rng.signed();
    }
    if name.ends_with("e_score_correction_bias") {
        return 0.05 * rng.signed();
    }
    if name.ends_with("_base") {
        return 0.5 * rng.signed();
    }
    if name.ends_with("_scale") {
        return 0.5 + 0.5 * rng.unit();
    }
    if name.contains("norm") {
        return 1.0 + 0.1 * rng.signed();
    }
    if name.ends_with("embed_tokens.weight") {
        return rng.signed();
    }
    if name.ends_with("index_kpool_compress_ape") {
        return 0.5 * rng.signed();
    }
    rng.signed() * matrix_width(spec.shape.last().copied().unwrap_or(1))
}

/// The entry width of a matrix with `columns` inputs.
fn matrix_width(columns: usize) -> f32 {
    1.0 / (columns.max(1) as f32).sqrt()
}

/// The bytes of one planned tensor.
///
/// # Errors
/// When the dtype is one this plan never emits, or a tile scale arrives without an FP8 tile side.
fn synthetic_bytes(
    spec: &TensorSpec,
    seed: u64,
    fp8_block: Option<usize>,
) -> Result<Vec<u8>, String> {
    let elements = usize::try_from(spec.elements())
        .map_err(|_| format!("{} is too large for this machine", spec.name))?;
    let mut rng = TensorRng::new(seed, &spec.name);
    let mut out = Vec::with_capacity(spec.dtype.storage_bytes(elements));
    match spec.dtype {
        DType::F8E4M3 => {
            for _ in 0..elements {
                out.push(quant::e4m3_encode(rng.signed() * CODE_RANGE));
            }
        }
        DType::F32 if spec.name.ends_with("weight_scale_inv") => {
            // A tile scale: a code of `CODE_RANGE` lands at the matrix width, give or take half.
            let block =
                fp8_block.ok_or_else(|| format!("{} scales FP8 tiles of no width", spec.name))?;
            let columns = spec.shape.last().copied().unwrap_or(1) * block;
            let base = matrix_width(columns) / CODE_RANGE;
            for _ in 0..elements {
                let scale = base * (0.5 + rng.unit());
                out.extend_from_slice(&scale.to_le_bytes());
            }
        }
        DType::F32 => {
            for _ in 0..elements {
                out.extend_from_slice(&value_at(spec, &mut rng).to_le_bytes());
            }
        }
        DType::BF16 => {
            for _ in 0..elements {
                let bits = quant::f32_to_bf16_bits(value_at(spec, &mut rng));
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        other => {
            return Err(format!(
                "the synthetic fixture cannot encode a {} tensor ({})",
                other.name(),
                spec.name
            ));
        }
    }
    Ok(out)
}

/// Write a synthetic checkpoint for `config` into `directory`, dealt over `shards` files.
///
/// The plan is the served text model and its MTP layer; the vision tower is not written.
///
/// # Errors
/// When a spec has a dtype the fixture cannot encode or the checkpoint cannot be written.
pub fn write_synthetic_checkpoint(
    directory: &Path,
    config: &Glm5NextConfig,
    seed: u64,
    shards: usize,
) -> Result<(), String> {
    let plan = Glm5NextTensorPlan::served_and_mtp(config);
    let tensors = plan
        .specs()
        .iter()
        .map(|spec| Ok((spec.clone(), synthetic_bytes(spec, seed, config.fp8_block)?)))
        .collect::<Result<Vec<_>, String>>()?;
    write_checkpoint(directory, &tensors, shards).map_err(|error| error.to_string())
}
