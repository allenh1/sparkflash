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

/// The f32 value of one element that is not FP8/FP4 coded.
fn value_at(spec: &TensorSpec, rng: &mut TensorRng) -> f32 {
    let name = spec.name.as_str();
    if name.contains("norm") || name.ends_with("embed.weight") {
        return if name.contains("norm") {
            1.0 + 0.1 * rng.signed()
        } else {
            rng.signed()
        };
    }
    if name.ends_with("attn_sink") {
        return 0.1 * rng.signed();
    }
    if name.contains("e_score_correction_bias") {
        return 0.05 * rng.signed();
    }
    rng.signed() * matrix_width(spec.shape.last().copied().unwrap_or(1))
}

/// The bytes of one planned tensor.
///
/// # Errors
/// A dtype this plan does not emit, or a packed FP4 tensor whose element count is not a whole
/// number of 32-value blocks.
fn synthetic_bytes(spec: &TensorSpec, seed: u64) -> Result<Vec<u8>, String> {
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
                    value_at(spec, &mut rng)
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
                    value_at(spec, &mut rng)
                };
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::BF16 => {
            for index in 0..elements {
                let value = if spec.name.ends_with(".scale") {
                    scale_at(index, &spec.shape)
                } else {
                    value_at(spec, &mut rng)
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
    let tensors = plan
        .specs()
        .iter()
        .map(|spec| Ok((spec.clone(), synthetic_bytes(spec, seed)?)))
        .collect::<Result<Vec<_>, String>>()?;
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
}
