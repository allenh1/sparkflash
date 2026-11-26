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
//! [write_synthetic_checkpoint] writes every tensor [crate::TensorPlan::language_model] names for
//! `config`, so a model whose shapes differ from the released one can be opened by the same
//! loader the served checkpoint goes through. The values are a deterministic function of the
//! seed and the tensor name (the stream [sparkflash_weights::rng::TensorRng] hands out), chosen
//! to stay finite and non-degenerate: matrix entries have the width of their input axis, norm
//! weights sit near one, FP8 codes stay inside the E4M3 range, and FP4 codes are quantised from
//! small values under a unit E8M0 block scale.

use std::collections::BTreeMap;
use std::path::Path;

use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::write_checkpoint;
use sparkflash_weights::{DType, TensorSpec, quant};

use crate::config::ModelConfig;
use crate::tensor::TensorPlan;

/// The entry width of a matrix with `columns` inputs.
fn matrix_width(columns: usize) -> f32 {
    1.0 / (columns.max(1) as f32).sqrt()
}

/// One stored inverse scale, as a block-scale tensor holds it.
///
/// The values vary from block to block and are exactly representable in f32 and bf16, so a decode
/// that reads the wrong grid entry changes the number it multiplies by: a shape-only test would
/// not notice, and an all-ones grid would hide the index entirely.
fn scale_at(index: usize, shape: &[usize]) -> f32 {
    let cols = shape.get(1).copied().unwrap_or(1).max(1);
    let (row, col) = (index / cols, index % cols);
    0.5 + 0.5 * (((row * 13 + col * 7) % 7) as f32) / 7.0
}

/// The producer-capture fixture's embedding width (doc 08, "the fixture is made decisive").
///
/// A trained model's residual stream is dominated by one large component that both sides read bit
/// for bit, so each layer adds a small fraction of the stream and a 1-ulp difference inside that
/// layer stays a 1-ulp difference in the stream. Drawn at unit scale the fixture's stream is a sum
/// of comparable random vectors instead, and each layer's 1-ulp differences compound instead of
/// diluting: on `glm-tiny` the last row of the fixture sat at 1.15e-2 relative L2 against the
/// producer, which is the bar rather than under it. Three times wider keeps every arithmetic the
/// same and every sublayer input normalised, and puts the stream back under a component both
/// sides hold bit for bit.
const EMBEDDING_SPREAD: f32 = 3.0;

/// The producer-capture fixture's router step (doc 08, "the fixture is made decisive").
///
/// A router whose experts are drawn independently has a near-tie at the reorder's own scale, and no
/// draw width can avoid one: the signals spread over about one sigma, the smallest of the 28 gaps
/// between eight of them is about `sigma / 28`, and the absorbed form perturbs a logit by about
/// `4e-3 sigma` because its rounding is *relative*. Wider logits widen the gaps and the
/// perturbation together. Drawn as `0.05 * signed()` the bias therefore met the absorbed form at
/// position 5 of layer 6 with a margin of `3.3e-4`, and an expert flipped. A determined ramp with
/// a step wider than the score function's whole range (`sigmoid` spans 0..1) decides the same two
/// experts on both sides, at a margin of roughly a tenth of the score.
const ROUTER_STEP: f32 = 1.2;

/// The producer-capture fixture's head rows (doc 08, "the fixture is made decisive").
///
/// The top-1 bar asks that both sides pick the same token at every position. A trained model's
/// next-token distribution is peaked, so its top logit stands clear of the second and a 7e-3
/// difference in the row does not move the argmax. The fixture's head draws every row at one width,
/// which puts the best two of 512 logits about 0.4% of the row's norm apart — under the row's own
/// error — so the bar measured a near-tie between two arbitrary tokens (position 13 of `glm-tiny`
/// at `6.0e-4` against a row error of `5.9e-3`). The first [HEAD_WIDE_ROWS] rows are drawn
/// [HEAD_WIDE] times wider instead: a handful of logits carry most of the row, the argmax stays a
/// real one over 512 candidates, and its margin is about a tenth of the row rather than a thousandth.
const HEAD_WIDE_ROWS: usize = 8;
const HEAD_WIDE: f32 = 8.0;

/// The f32 value of one element that is not FP8/FP4 coded.
///
/// `index` is the element's offset in the tensor, which a recipe that needs a per-row value reads.
/// When `decisive` is false ([write_synthetic_checkpoint]), the original DeepSeek draw is kept
/// bit-for-bit unchanged; when `decisive` is true ([write_described_checkpoint]), the producer
/// capture's conditioning recipes (embedding width, per-layer rotated router ramp, peaked head rows)
/// apply so the fixture measures the stack rather than random near-ties.
fn value_at(spec: &TensorSpec, index: usize, rng: &mut TensorRng, decisive: bool) -> f32 {
    let name = spec.name.as_str();
    if name.contains("norm") || name.ends_with("embed.weight") {
        return if name.contains("norm") {
            1.0 + 0.1 * rng.signed()
        } else if decisive {
            EMBEDDING_SPREAD * rng.signed()
        } else {
            rng.signed()
        };
    }
    if name.ends_with("attn_sink") {
        return 0.1 * rng.signed();
    }
    if name.contains("e_score_correction_bias") {
        if decisive {
            let experts = spec.shape.first().copied().unwrap_or(1).max(1) as f32;
            return ROUTER_STEP * (index as f32 - (experts - 1.0) / 2.0) + 0.05 * rng.signed();
        }
        return 0.05 * rng.signed();
    }
    // The indexer's queries and keys are deliberately left alone here: scaling them moves no sign,
    // so it cannot break an all-negative pool. Their recipe is [mirror_indexer_heads], which
    // changes no magnitude at all.
    let columns = spec.shape.last().copied().unwrap_or(1);
    let wide = if decisive
        && name.ends_with("head.weight")
        && columns > 0
        && index / columns < HEAD_WIDE_ROWS
    {
        HEAD_WIDE
    } else {
        1.0
    };
    rng.signed() * matrix_width(columns) * wide
}

/// The producer-capture fixture's indexer heads, drawn as mirrored pairs (doc 08, "the fixture is
/// made decisive").
///
/// A pool's selection score is `sum_h w_h * relu(q_h . k_pool * scale)`. On a near-uniform draw
/// each head's dot is negative about as often as not, so a pool whose heads all drew negative
/// scores exactly `0.0`, several pools then share the largest score, and a selection at that edge
/// is decided by whichever `topk` the producer happens to run: PyTorch's CPU `topk` keeps a
/// different pool than [sparkflash_weights::topk::top_pools] does, and the producer's own CUDA
/// build differs from its CPU one, so a capture that lands there measures an unspecified order
/// rather than the stack.
///
/// Drawing head `h + heads/2` of `wq_b` as head `h` negated, with `weights_proj`'s row for that
/// head bit-identical to head `h`'s row, makes the pair contribute
/// `w_h * (relu(d) + relu(-d)) = w_h * |d|`: never zero for a pool whose dot is not zero, and
/// ordered by that dot's magnitude. No magnitude changes and no arithmetic changes; the `relu`
/// still takes its zero branch on one head of every pair, on both the host and the producer. The
/// recipe keys on the indexer's stored names, so a model that spells its indexer this way gets the
/// same conditioned fixture.
///
/// A head count that is odd, or anything that is not a two-dimensional matrix, is left alone.
fn mirror_indexer_heads(name: &str, shape: &[usize], element: usize, out: &mut [u8]) {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let queries = base.ends_with("attn.indexer.wq_b");
    if !(queries || base.ends_with("attn.indexer.weights_proj")) || element == 0 {
        return;
    }
    let (Some(&rows), Some(&columns)) = (shape.first(), shape.get(1)) else {
        return;
    };
    if columns == 0 || !rows.is_multiple_of(2) || out.len() != rows * columns * element {
        return;
    }
    let (half, row_bytes) = (rows / 2, columns * element);
    let (first, rest) = out.split_at_mut(half * row_bytes);
    for row in 0..half {
        rest[row * row_bytes..(row + 1) * row_bytes]
            .copy_from_slice(&first[row * row_bytes..(row + 1) * row_bytes]);
    }
    if queries {
        // The sign bit is the top bit of each element's most significant byte, which little-endian
        // storage puts last: BF16 and F32 in their high half, an E4M3 code in its only byte.
        for at in (0..half * row_bytes).step_by(element) {
            rest[at + element - 1] ^= 0x80;
        }
    }
}

/// The bytes of one planned tensor.
///
/// # Errors
/// A dtype this plan does not emit, or a packed FP4 tensor whose element count is not a whole
/// number of 32-value blocks.
fn synthetic_bytes(spec: &TensorSpec, seed: u64, decisive: bool) -> Result<Vec<u8>, String> {
    let elements = usize::try_from(spec.elements())
        .map_err(|_| format!("{} is too large for this machine", spec.name))?;
    let mut rng = TensorRng::new(seed, &spec.name);
    let mut out = Vec::with_capacity(spec.dtype.storage_bytes(elements));
    match spec.dtype {
        DType::F8E4M3 => {
            for index in 0..elements {
                // NVFP4's per-16 group scale is a positive E4M3 multiplier; an E4M3 weight code
                // is the same encoder's signed value.
                let value = if spec.name.ends_with(".scale") {
                    scale_at(index, &spec.shape)
                } else {
                    value_at(spec, index, &mut rng, decisive)
                };
                out.push(quant::e4m3_encode(value));
            }
        }
        DType::F32 => {
            for index in 0..elements {
                let value = if spec.name.ends_with("weight_scale_inv") {
                    matrix_width(spec.shape.last().copied().unwrap_or(1)) * (0.5 + rng.unit())
                } else if spec.name.ends_with("weight_scale_2") {
                    // Qwen's weight global: one positive f32 for the whole tensor.
                    scale_at(index, &spec.shape)
                } else if spec.name.ends_with("input_scale") {
                    // Qwen's static activation global: a second f32 carried beside the weight's.
                    0.75 + 0.5 * scale_at(index, &spec.shape)
                } else if spec.name.ends_with("weight_amax") {
                    // Inkling's per-expert amax: one f32 for this expert's tensor.
                    1.0 + scale_at(index, &spec.shape)
                } else if spec.name.ends_with("weight_outer_scale_denominator") {
                    // Inkling's denominator already carries the 6 * 448 block rescale.
                    2688.0
                } else if spec.name.ends_with(".scale") {
                    // A tile or row inverse scale: one positive value a block, varying block to
                    // block so a body that indexes the grid wrongly reads a different number.
                    scale_at(index, &spec.shape)
                } else {
                    value_at(spec, index, &mut rng, decisive)
                };
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::BF16 => {
            for index in 0..elements {
                let value = if spec.name.ends_with(".scale") {
                    scale_at(index, &spec.shape)
                } else {
                    value_at(spec, index, &mut rng, decisive)
                };
                let bits = quant::f32_to_bf16_bits(value);
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        DType::F8E8M0 => {
            // A unit block scale: E8M0's bias is 127, so 127 is 2^0.
            out.resize(elements, 127);
        }
        DType::F4E2M1 => {
            if !elements.is_multiple_of(2) {
                return Err(format!(
                    "{}: {elements} FP4 elements do not pack into bytes",
                    spec.name
                ));
            }
            let width = matrix_width(spec.shape.last().copied().unwrap_or(1));
            let values: Vec<f32> = (0..elements).map(|_| rng.signed() * width).collect();
            for pair in values.chunks(2) {
                let low = quant::f4e2m1_encode(pair[0]);
                let high = quant::f4e2m1_encode(pair[1]);
                out.push(low | (high << 4));
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
    if decisive {
        mirror_indexer_heads(
            &spec.name,
            &spec.shape,
            spec.dtype.storage_bytes(1),
            &mut out,
        );
    }
    Ok(out)
}

/// Write a synthetic checkpoint for `config` into `directory`, dealt over `shards` files.
///
/// The plan is the served text model and its MTP stages; a vision tower is not written.
///
/// # Errors
/// A spec with a dtype the fixture cannot encode, or a checkpoint that cannot be written.
pub fn write_synthetic_checkpoint(
    directory: &Path,
    config: &ModelConfig,
    seed: u64,
    shards: usize,
) -> Result<(), String> {
    let plan = TensorPlan::language_model(config);
    let mut tensors: Vec<(TensorSpec, Vec<u8>)> = Vec::with_capacity(plan.specs().len());
    for spec in plan.specs() {
        let bytes = synthetic_bytes(spec, seed, false)?;
        // The packed word form is an arrangement of the BF16 rows, not another arithmetic: the
        // writer packs the bytes it just made, and writes the escape list the loader needs beside
        // the packed rows. A row-major BF16 tensor is written as it is.
        if plan
            .weight_format(&spec.name)
            .is_some_and(|format| format.layout() == sparkflash_model::format::FormatLayout::Packed)
        {
            let rows = spec.shape.first().copied().unwrap_or(0);
            let columns = spec.shape.get(1).copied().unwrap_or(0);
            let (packed, escapes) = sparkflash_model::packed_bf16::pack(rows, columns, &bytes)
                .map_err(|error| format!("{}: {error}", spec.name))?;
            let stride = sparkflash_model::packed_bf16::row_bytes(columns);
            let escape_name = spec.name.trim_end_matches(".weight").to_owned() + ".escape";
            let mut escape_bytes = Vec::with_capacity(escapes.len() * 4);
            for entry in &escapes {
                escape_bytes.extend_from_slice(&entry.to_le_bytes());
            }
            tensors.push((
                TensorSpec {
                    name: spec.name.clone(),
                    dtype: DType::U8,
                    shape: vec![rows, stride],
                },
                packed,
            ));
            tensors.push((
                TensorSpec {
                    name: escape_name,
                    dtype: DType::U8,
                    shape: vec![escape_bytes.len()],
                },
                escape_bytes,
            ));
        } else {
            tensors.push((spec.clone(), bytes));
        }
    }
    write_checkpoint(directory, &tensors, shards).map_err(|error| error.to_string())
}

/// One FP8 tile class as written: its E4M3 codes, its scale grid `[rows, columns]` in tiles,
/// and one inverse scale a tile.
type Fp8Tiles = (Vec<u8>, Vec<usize>, Vec<f32>);

/// The bytes of one FP8 tile class: E4M3 codes over the format's whole range and one inverse
/// scale a `block_m x block_n` tile, so a decoded weight is about `signed / sqrt(columns)`.
fn fp8_tile_bytes(
    name: &str,
    shape: &[usize],
    (block_m, block_n): (usize, usize),
    seed: u64,
) -> Result<Fp8Tiles, String> {
    let (rows, columns) = match shape {
        [rows, columns] => (*rows, *columns),
        other => {
            return Err(format!(
                "{name}: an FP8 tile class is a matrix, not {other:?}"
            ));
        }
    };
    let mut rng = TensorRng::new(seed, name);
    let codes = (0..rows * columns)
        .map(|_| quant::e4m3_encode(rng.signed() * 448.0))
        .collect();
    let grid = vec![rows.div_ceil(block_m), columns.div_ceil(block_n)];
    let mut scales_rng = TensorRng::new(seed, &format!("{name}.weight_scale_inv"));
    let scales = (0..grid[0] * grid[1])
        .map(|_| matrix_width(columns) / 448.0 * (0.5 + scales_rng.unit()))
        .collect();
    Ok((codes, grid, scales))
}

/// Write a synthetic checkpoint in a description's own stored spelling: every tensor its layout
/// plans (`sparkflash_model::layout::described_tensors`), each class in the architecture's
/// stored format, the values from [write_synthetic_checkpoint]'s recipe read by the tensor's base
/// name (its slot). An FP8 tile class writes its codes and its `weight_scale_inv` sibling.
///
/// # Errors
/// No layout describes the architecture, a class whose stored format this writer has no recipe
/// for (named), or a checkpoint that cannot be written.
pub fn write_described_checkpoint(
    directory: &Path,
    architecture: &sparkflash_model::Architecture,
    seed: u64,
    shards: usize,
) -> Result<(), String> {
    use sparkflash_model::architecture::{ScaleDType, WeightFormat};
    use sparkflash_model::layout::LayoutDtype;
    let planned = sparkflash_model::layout::described_tensors(architecture)
        .map_err(|error| error.to_string())?
        .ok_or("no layout describes this architecture")?;
    let quantization = &architecture.quantization;
    let mut tensors: Vec<(TensorSpec, Vec<u8>)> = Vec::with_capacity(planned.len());
    for tensor in &planned {
        let format = match tensor.dtype {
            LayoutDtype::Bf16 => WeightFormat::Bf16,
            LayoutDtype::F32 => WeightFormat::F32,
            LayoutDtype::Dense => quantization.dense_format,
            LayoutDtype::Expert => match tensor.stack {
                sparkflash_model::StackKind::Draft => quantization
                    .mtp_expert_format
                    .unwrap_or(quantization.expert_format),
                _ => quantization.expert_format,
            },
            other => {
                return Err(format!(
                    "{}: the described writer has no recipe for the {other:?} class",
                    tensor.name
                ));
            }
        };
        // The recipe reads the base name, so a described tensor takes the value its slot takes
        // in the base's own spelling; the routing bias keeps its stored name, which the recipe
        // knows.
        let base = match &tensor.slot {
            Some(_) if tensor.name.contains("e_score_correction_bias") => tensor.name.clone(),
            Some(slot) => slot.clone(),
            None => tensor.name.clone(),
        };
        match format {
            WeightFormat::Bf16 | WeightFormat::F32 => {
                let dtype = if format == WeightFormat::Bf16 {
                    DType::BF16
                } else {
                    DType::F32
                };
                let recipe = TensorSpec {
                    name: if base.ends_with(".bias") && base.contains("norm") {
                        // A LayerNorm's bias sits near zero, as the weight sits near one.
                        base.replace("norm", "nrm")
                    } else {
                        base.clone()
                    },
                    dtype,
                    shape: tensor.shape.clone(),
                };
                let bytes = synthetic_bytes(&recipe, seed, true)?;
                tensors.push((
                    TensorSpec {
                        name: tensor.name.clone(),
                        dtype,
                        shape: tensor.shape.clone(),
                    },
                    bytes,
                ));
            }
            WeightFormat::Fp8Tile {
                block_m,
                block_n,
                scale_dtype,
                ..
            } => {
                let (codes, grid, scales) =
                    fp8_tile_bytes(&base, &tensor.shape, (block_m, block_n), seed)?;
                tensors.push((
                    TensorSpec {
                        name: tensor.name.clone(),
                        dtype: DType::F8E4M3,
                        shape: tensor.shape.clone(),
                    },
                    codes,
                ));
                let stem = tensor.name.strip_suffix(".weight").ok_or_else(|| {
                    format!("{}: an FP8 tile class names a .weight tensor", tensor.name)
                })?;
                let (dtype, bytes) = match scale_dtype {
                    ScaleDType::F32 => (
                        DType::F32,
                        scales
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect(),
                    ),
                    ScaleDType::Bf16 => (
                        DType::BF16,
                        scales
                            .iter()
                            .flat_map(|value| quant::f32_to_bf16_bits(*value).to_le_bytes())
                            .collect(),
                    ),
                    other => {
                        return Err(format!(
                            "{}: the described writer has no recipe for {other:?} tile scales",
                            tensor.name
                        ));
                    }
                };
                tensors.push((
                    TensorSpec {
                        name: format!("{stem}.weight_scale_inv"),
                        dtype,
                        shape: grid,
                    },
                    bytes,
                ));
            }
            other => {
                return Err(format!(
                    "{}: the described writer has no recipe for the stored format {other:?}",
                    tensor.name
                ));
            }
        }
    }
    write_checkpoint(directory, &tensors, shards).map_err(|error| error.to_string())
}

/// The summary of one written twin fixture.
#[derive(Debug, Clone)]
pub struct FixtureSummary {
    /// The number of tensors written.
    pub tensors: usize,
    /// The bytes of the shard file.
    pub bytes: u64,
    /// The FNV-1a hash of the shard file.
    pub hash: u64,
}

/// Write the deterministic twin fixture for `config` and its index.
///
/// This is the host twin the device-chain harness and its tests open: one `model.safetensors`
/// file and a `model.safetensors.index.json` that maps every planned tensor to it. The values
/// are the engine's own fixture recipe, not [write_synthetic_checkpoint]'s: the twin's evidence
/// records its FNV-1a hash, so the bytes stay where the evidence put them.
///
/// # Errors
/// A directory or file that cannot be created, or a tensor whose header cannot be serialised.
pub fn write_twin_fixture(
    directory: &Path,
    config: &ModelConfig,
) -> Result<FixtureSummary, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    let plan = TensorPlan::language_model(config);
    let mut entries: Vec<String> = Vec::with_capacity(plan.specs().len());
    let mut data: Vec<u8> = Vec::new();
    let mut tensors = 0usize;
    for spec in plan.specs() {
        let (dtype_word, shape, bytes) = fixture_tensor_bytes(spec);
        entries.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            serde_json::to_string(&spec.name).map_err(|error| error.to_string())?,
            dtype_word,
            serde_json::to_string(&shape).map_err(|error| error.to_string())?,
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend_from_slice(&bytes);
        tensors += 1;
    }
    let header = format!(
        "{{{},\"__metadata__\":{{\"format\":\"pt\"}}}}",
        entries.join(",")
    );
    let header_len = header.len() as u64;
    let mut file = Vec::with_capacity(8 + header.len() + data.len());
    file.extend_from_slice(&header_len.to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    let shard_path = directory.join("model.safetensors");
    std::fs::write(&shard_path, &file)
        .map_err(|error| format!("write {}: {error}", shard_path.display()))?;
    let index = serde_json::json!({
        "metadata": {"total_size": data.len()},
        "weight_map": plan.specs().iter().map(|spec| (spec.name.clone(), "model.safetensors".to_owned())).collect::<BTreeMap<_, _>>(),
    });
    let index_path = directory.join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        serde_json::to_vec_pretty(&index).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", index_path.display()))?;
    Ok(FixtureSummary {
        tensors,
        bytes: data.len() as u64,
        hash: crate::synthetic::fnv1a(&file),
    })
}

/// The dtype word, stored shape and bytes for one twin tensor.
fn fixture_tensor_bytes(spec: &TensorSpec) -> (&'static str, Vec<usize>, Vec<u8>) {
    let engram = spec.name.contains(".engram.");
    if engram && spec.dtype == DType::F8E4M3 {
        let salt = name_salt(&spec.name);
        let elements = spec.elements() as usize;
        let bytes = (0..elements)
            .map(|index| quant::e4m3_encode(fixture_value(index, salt, 0x5eed)))
            .collect();
        ("F8_E4M3", spec.shape.clone(), bytes)
    } else if engram && spec.dtype == DType::F8E8M0 {
        let elements = spec.elements() as usize;
        ("F8_E8M0", spec.shape.clone(), vec![127u8; elements])
    } else {
        let salt = name_salt(&spec.name);
        let elements = spec.elements() as usize;
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            bytes.extend_from_slice(&fixture_value(index, salt, 0x5eed).to_le_bytes());
        }
        ("F32", spec.shape.clone(), bytes)
    }
}

/// The finite value the twin fixture writes, the same recipe as the engine host tests.
fn fixture_value(index: usize, salt: u64, seed: u64) -> f32 {
    let mut hashed = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hashed ^= index as u64;
    for _ in 0..3 {
        hashed = hashed
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .rotate_left(17)
            .wrapping_add(0x94d0_49bb_1331_11eb);
    }
    0.01 + 0.0001 * ((hashed >> 56) as f32)
}

fn name_salt(name: &str) -> u64 {
    let mut salt = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        salt ^= u64::from(*byte);
        salt = salt.wrapping_mul(0x0000_0100_0000_01b3);
    }
    salt
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_weights::WeightSource;
    use sparkflash_weights::safetensors::SafeTensorsSource;

    #[test]
    fn a_nvfp4_configuration_writes_the_companions_its_convention_names() {
        for (declared, weight_global, activation_global, amax, denominator) in [
            ("nvfp4-qwen", true, true, false, false),
            ("nvfp4-inkling", false, false, true, true),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "deepseek-nvfp4-{}-{}-{}",
                declared,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("the clock")
                    .as_nanos()
            ));
            let mut config = ModelConfig::tiny();
            config.expert_dtype = declared.to_owned();
            write_synthetic_checkpoint(&dir, &config, 7, 1).expect("the checkpoint writes");
            let source = SafeTensorsSource::open(&dir, &dir.join("model.safetensors.index.json"))
                .expect("the index opens");
            let prefix = "layers.0.ffn.experts.0.w1";
            assert!(source.contains(&format!("{prefix}.scale")), "{declared}");
            assert_eq!(
                source.contains(&format!("{prefix}.weight_scale_2")),
                weight_global,
                "{declared}"
            );
            assert_eq!(
                source.contains(&format!("{prefix}.input_scale")),
                activation_global,
                "{declared}"
            );
            assert_eq!(
                source.contains(&format!("{prefix}.weight_amax")),
                amax,
                "{declared}"
            );
            assert_eq!(
                source.contains(&format!("{prefix}.weight_outer_scale_denominator")),
                denominator,
                "{declared}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    fn bf16_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect()
    }

    #[test]
    fn the_indexer_is_written_with_its_heads_mirrored_in_pairs() {
        let queries = TensorSpec {
            name: "layers.3.attn.indexer.wq_b.weight".to_owned(),
            dtype: DType::BF16,
            shape: vec![4, 3],
        };
        let written = synthetic_bytes(&queries, 7, true).expect("the tensor writes");
        let values = bf16_values(&written);
        for row in 0..2 {
            for column in 0..3 {
                let head = values[row * 3 + column];
                let mirror = values[(row + 2) * 3 + column];
                assert_eq!(mirror, -head, "row {row} column {column}");
            }
        }
        // The plain DeepSeek recipe leaves indexer heads unmirrored.
        let plain = bf16_values(&synthetic_bytes(&queries, 7, false).expect("plain writes"));
        assert_ne!(plain[6], -plain[0], "plain DeepSeek recipe is untouched");

        let weights = TensorSpec {
            name: "mtp.0.attn.indexer.weights_proj.weight".to_owned(),
            dtype: DType::BF16,
            shape: vec![2, 3],
        };
        let written = synthetic_bytes(&weights, 7, true).expect("the tensor writes");
        let values = bf16_values(&written);
        assert_eq!(values[..3], values[3..], "the pair's rows are one row");

        // A tensor whose name the recipe does not know, and an odd head count, are left alone.
        for name in [
            "layers.3.attn.indexer.wk.weight",
            "layers.3.attn.indexer.wq_b.weight",
        ] {
            let spec = TensorSpec {
                name: name.to_owned(),
                dtype: DType::BF16,
                shape: if name.contains("wk") {
                    vec![4, 3]
                } else {
                    vec![3, 3]
                },
            };
            let written = synthetic_bytes(&spec, 7, true).expect("the tensor writes");
            let values = bf16_values(&written);
            if name.contains("wk") {
                assert_ne!(values[..3], values[6..9], "{name} is not an indexer query");
            } else {
                assert_ne!(values[..3], values[3..6], "{name} has an odd head count");
            }
        }
    }

    #[test]
    fn a_tiny_configuration_writes_a_checkpoint_its_plan_reads() {
        let dir = std::env::temp_dir().join(format!(
            "deepseek-synthetic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("the clock")
                .as_nanos()
        ));
        let config = ModelConfig::tiny();
        write_synthetic_checkpoint(&dir, &config, 7, 2).expect("the checkpoint writes");
        let source = SafeTensorsSource::open(&dir, &dir.join("model.safetensors.index.json"))
            .expect("the index opens");
        for spec in TensorPlan::language_model(&config).specs() {
            let bytes = source
                .read(&spec.name)
                .expect("every planned tensor is stored");
            assert_eq!(bytes.len() as u128, spec.storage_bytes(), "{}", spec.name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn bf16_tiny(dense: &str) -> ModelConfig {
        let mut config = ModelConfig::tiny();
        // The packed row walk takes whole 256-column rounds, so the test's geometry has one.
        config.dim = 256;
        config.q_lora_rank = 256;
        config.n_heads = 16;
        config.head_dim = 16;
        config.o_groups = 1;
        config.o_lora_rank = 256;
        config.moe_inter_dim = 256;
        config.dtype = dense.to_owned();
        config.expert_dtype = "fp4".to_owned();
        config.weight_block = 32;
        config.weight_scale = "ue8m0".to_owned();
        config
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "deepseek-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("the clock")
                .as_nanos()
        ))
    }

    #[test]
    fn a_bf16_dense_class_writes_no_scale_plane() {
        let dir = temp_dir("bf16");
        let config = bf16_tiny("bf16");
        write_synthetic_checkpoint(&dir, &config, 7, 2).expect("the checkpoint writes");
        let source = SafeTensorsSource::open(&dir, &dir.join("model.safetensors.index.json"))
            .expect("the index opens");
        let spec = source
            .spec("layers.0.attn.wq_a.weight")
            .expect("the dense projection is planned");
        assert_eq!(spec.dtype, DType::BF16, "the dense class is unquantized");
        assert!(
            !source.contains("layers.0.attn.wq_a.scale"),
            "an unscaled weight has no scale tensor"
        );
        assert_eq!(
            source
                .read("layers.0.attn.wq_a.weight")
                .expect("read")
                .len(),
            spec.shape[0] * spec.shape[1] * 2
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_packed_dense_class_writes_the_rows_and_the_escape_list_it_stands_for() {
        let dir = temp_dir("packed");
        let config = bf16_tiny("packed-bf16");
        let plan = TensorPlan::language_model(&config);
        let weight = plan
            .specs()
            .iter()
            .find(|spec| spec.name == "layers.0.attn.wq_a.weight")
            .expect("the dense projection is planned");
        let original = synthetic_bytes(weight, 7, false).expect("the fixture's bf16 rows");
        write_synthetic_checkpoint(&dir, &config, 7, 2).expect("the checkpoint writes");
        let source = SafeTensorsSource::open(&dir, &dir.join("model.safetensors.index.json"))
            .expect("the index opens");
        let stored = source
            .spec("layers.0.attn.wq_a.weight")
            .expect("the packed weight is stored");
        assert_eq!(stored.dtype, DType::U8, "the packed rows are bytes");
        let escape = source
            .read("layers.0.attn.wq_a.escape")
            .expect("the escape list travels beside the rows");
        let escapes: Vec<u32> = escape
            .chunks(4)
            .map(|entry| u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]))
            .collect();
        let packed = source.read("layers.0.attn.wq_a.weight").expect("read");
        let rows = weight.shape[0];
        let columns = weight.shape[1];
        let unpacked = sparkflash_model::packed_bf16::unpack(rows, columns, &packed, &escapes)
            .expect("the packed rows and their escape list are one matrix");
        assert_eq!(
            unpacked, original,
            "the body reads the rows the writer held"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
