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

//! The offline F1 comparison for the wo_a seam: one captured activation operand through both
//! paths, on the same bytes, with released weights.
//!
//! The device chain records each layer's first-position wo_a operand (f32, before the native BF16
//! round) and the device output through --dump-wo-a-operand, and the chain also prints the
//! per-layer comparison in the run log. This example exists so the same comparison can be re-run
//! without a device: give it an operand file (f32 little-endian, heads x head_dim), the released
//! layer-0 checkpoint for the wo_a codes and scales, and it prints the F1 magnitude - the native
//! path against the current MXFP8 path on the SAME operand.
//!
//! It also reproduces the operand the chain actually feeds wo_a when the operand was captured
//! before the inverse RoPE (the engine reference records the pre-projection output before the
//! rotation): with --rope inverse it applies the layer's inverse rotation exactly as the reference
//! does, using the engine's own table and rotation routines, so the comparison sees the operand
//! wo_a is called with. --rope none takes an operand that is already rotated (a window dump).
//!
//! It is CPU-only. No device, no cubin and no driver is touched.

use std::path::PathBuf;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::rope_tables_for_layer;
use sparkflash_engine::linalg::rope_rotate_inverse;
use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::bf16a_mxfp8w_ref;
use sparkflash_kernels::m4_quant::bf16_value;
use sparkflash_kernels::mxfp8_ref::b12x_weight_index;

fn checksum_f32(values: &[f32]) -> u64 {
    values.iter().fold(0u64, |acc, value| {
        acc.rotate_left(5) ^ u64::from(value.to_bits())
    })
}

fn read_operand(path: &PathBuf, expected: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != expected * 4 {
        return Err(format!(
            "{}: expected {} f32 values ({} bytes), got {} bytes",
            path.display(),
            expected,
            expected * 4,
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn main() -> Result<(), String> {
    let mut config_path: Option<PathBuf> = None;
    let mut layer = 0usize;
    let mut position: Option<usize> = None;
    let mut operand_path: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut weight_path: Option<PathBuf> = None;
    let mut scales_path: Option<PathBuf> = None;
    let mut rope = "inverse".to_string();
    let mut write_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--config" => config_path = Some(PathBuf::from(value)),
            "--layer" => layer = value.parse().map_err(|_| format!("--layer: {value}"))?,
            "--position" => {
                position = Some(value.parse().map_err(|_| format!("--position: {value}"))?)
            }
            "--operand" => operand_path = Some(PathBuf::from(value)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--weight" => weight_path = Some(PathBuf::from(value)),
            "--scales" => scales_path = Some(PathBuf::from(value)),
            "--rope" => rope = value,
            "--write" => write_dir = Some(PathBuf::from(value)),
            other => return Err(format!("unknown option {other}")),
        }
    }
    let config_path = config_path.ok_or_else(|| "--config PATH is required".to_string())?;
    let text = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config = ModelConfig::from_json(&text).map_err(|error| error.to_string())?;
    let operand_path = operand_path.ok_or_else(|| "--operand PATH is required".to_string())?;
    let heads = config.n_heads;
    let head_dim = config.head_dim;
    let groups = config.o_groups;
    let lo_rank = config.o_lora_rank;
    let wide_row = heads * head_dim;
    if wide_row % groups != 0 {
        return Err(format!("{wide_row} does not divide into {groups} groups"));
    }
    let inner = wide_row / groups;
    let mut wide = read_operand(&operand_path, wide_row)?;
    match rope.as_str() {
        "inverse" => {
            let position = position.unwrap_or(0);
            let positions: Vec<usize> = std::iter::repeat_n(position, heads).collect();
            let tables = rope_tables_for_layer(&config, layer, &positions);
            let pairs = config.rope_head_dim / 2;
            for row in 0..heads {
                let start = row * head_dim + head_dim - config.rope_head_dim;
                let slice = &mut wide[start..start + config.rope_head_dim];
                rope_rotate_inverse(
                    slice,
                    &tables.cos[row * pairs..(row + 1) * pairs],
                    &tables.sin[row * pairs..(row + 1) * pairs],
                )
                .map_err(|error| error.to_string())?;
            }
        }
        "none" => {}
        other => return Err(format!("--rope: {other} is not inverse or none")),
    }
    let name = format!("layers.{layer}.attn.wo_a.weight");
    let (raw_weight, raw_scales) = match (weight_path, scales_path, checkpoint) {
        (Some(weight), Some(scales), None) => (
            std::fs::read(&weight).map_err(|error| format!("{}: {error}", weight.display()))?,
            std::fs::read(&scales).map_err(|error| format!("{}: {error}", scales.display()))?,
        ),
        (None, None, Some(checkpoint)) => {
            let source = SafeTensorsSource::open(
                &checkpoint,
                &checkpoint.join("model.safetensors.index.json"),
            )
            .map_err(|error| format!("checkpoint {}: {error}", checkpoint.display()))?;
            let weight = source.read(&name).map_err(|error| error.to_string())?;
            let scales = source
                .read(&name.replace(".weight", ".scale"))
                .map_err(|error| error.to_string())?;
            (weight, scales)
        }
        _ => return Err("give either --weight FILE --scales FILE or --checkpoint DIR".to_string()),
    };
    if raw_weight.len() != groups * lo_rank * inner {
        return Err(format!(
            "{name}: expected {} E4M3 bytes, got {}",
            groups * lo_rank * inner,
            raw_weight.len()
        ));
    }
    let expected_scales = (groups * lo_rank).div_ceil(32) * inner.div_ceil(32);
    if raw_scales.len() != expected_scales {
        return Err(format!(
            "{name}.scale: expected {expected_scales} E8M0 bytes, got {}",
            raw_scales.len()
        ));
    }
    let cols = groups * lo_rank;
    let mut packed = vec![0u8; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            packed[b12x_weight_index(col, k, inner)] = raw_weight[col * inner + k];
        }
    }
    let comparison =
        bf16a_mxfp8w_ref::compare_paths(&wide, &packed, &raw_scales, groups, inner, lo_rank)?;
    let exact_fraction = comparison.codes_equal as f64 / comparison.total as f64;
    let relative = comparison.max_abs / comparison.output_max_abs.max(f32::MIN_POSITIVE);
    let mut delta_sum = 0.0f64;
    for (native, mxfp8) in comparison.native.iter().zip(comparison.mxfp8.iter()) {
        delta_sum += f64::from((bf16_value(*native) - bf16_value(*mxfp8)).abs());
    }
    let mean_abs = delta_sum / f64::from(comparison.total as u32);
    println!(
        "wo_a native compare: layer={} rope={} groups={} inner={} rank={} operand_checksum={:#018x} outputs={} native_max_abs={:.9e} mxfp8_codes_equal={}/{} exact_fraction={:.6} max_abs={:.9e} max_abs_relative={:.9e} mean_abs={:.9e}",
        layer,
        rope,
        groups,
        inner,
        lo_rank,
        checksum_f32(&wide),
        comparison.total,
        comparison.output_max_abs,
        comparison.codes_equal,
        comparison.total,
        exact_fraction,
        comparison.max_abs,
        relative,
        mean_abs
    );
    if let Some(dir) = write_dir {
        std::fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        let mut native_bytes = Vec::with_capacity(comparison.native.len() * 2);
        let mut mxfp8_bytes = Vec::with_capacity(comparison.mxfp8.len() * 2);
        for code in &comparison.native {
            native_bytes.extend_from_slice(&code.to_le_bytes());
        }
        for code in &comparison.mxfp8 {
            mxfp8_bytes.extend_from_slice(&code.to_le_bytes());
        }
        let native_path = dir.join(format!("wo_a_layer_{layer:02}_native.bf16"));
        let mxfp8_path = dir.join(format!("wo_a_layer_{layer:02}_mxfp8.bf16"));
        std::fs::write(&native_path, native_bytes)
            .map_err(|error| format!("{}: {error}", native_path.display()))?;
        std::fs::write(&mxfp8_path, mxfp8_bytes)
            .map_err(|error| format!("{}: {error}", mxfp8_path.display()))?;
        let operand_path = dir.join(format!("wo_a_layer_{layer:02}_operand_roped.f32"));
        let mut operand_bytes = Vec::with_capacity(wide.len() * 4);
        for value in &wide {
            operand_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&operand_path, operand_bytes)
            .map_err(|error| format!("{}: {error}", operand_path.display()))?;
    }
    Ok(())
}
