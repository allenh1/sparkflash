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

//! `sparkflash convert`: offline checkpoint converter into per-rank device-layout files.
//!
//! Converts the released safetensors checkpoint once, offline, into one contiguous file per rank
//! (`rank-{rank}-of-{world}.bin`) already in the exact device arena layout:
//! - Dense FP8 weights are B12x-tiled (`TensorLayout::B12x`).
//! - Routed MoE down-projection (`w2`) weights and scales are padded to the 640-wide K layout
//!   (`320` packed weight bytes and `20` E8M0 scale bytes per row, `TensorLayout::MoePadded640`).
//! - Scales are stored immediately next to their weights.
//! - A 4096-byte-aligned header block at the start of the file holds the tensor index
//!   (`name -> offset, length`), `header_hash`, and `file_digest`.
//!
//! A checkpoint whose `config.json` names GLM-5.3's model type is converted by the GLM pack
//! instead, into the same container ([sparkflash_model_glm5_next::convert]); a Qwen4-Exp
//! checkpoint is converted by its pack's TP=1 converter
//! ([sparkflash_model_qwen4_exp::convert]). `--inspect` tells the three apart by the model a
//! file's index names.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::Value;
use sparkflash_device::converted::{
    ConvertOptions, ConvertedPreamble, convert_checkpoint, inspect_converted_rank, read_index,
    verify_converted_file,
};
use sparkflash_engine::parallel_tp::TpLayout;
use sparkflash_engine::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_model_glm5_next::Glm5NextConfig;
use sparkflash_model_glm5_next::config::MODEL_TYPE;
use sparkflash_model_glm5_next::convert::{CONVERTED_MODEL, convert_rank, verify_converted};
use sparkflash_model_qwen4_exp::Qwen4ExpConfig;
use sparkflash_model_qwen4_exp::convert::{
    CONVERTED_MODEL as QWEN_CONVERTED_MODEL, convert_rank as qwen_convert_rank,
    read_index as qwen_read_index, verify_converted as qwen_verify_converted,
};

/// Prints one job-progress line for the web control plane and flushes it so the reader sees it
/// while the conversion is still running.
fn emit_progress(done: u64, total: u64, detail: &str) {
    let line = serde_json::json!({"done": done, "total": total, "detail": detail});
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "@progress {line}");
    let _ = stdout.flush();
}

/// Usage string for `sparkflash convert`.
pub const USAGE: &str = "sparkflash convert --checkpoint DIR [--output-dir DIR] [--tp-world-size 4] [--tp-rank R | --ranks 0,1,2,3] [--no-pad-moe-640] [--no-verify] | sparkflash convert --inspect FILE [--verify-weights]";

/// Entry point called by `main.rs`.
pub fn run(args: impl Iterator<Item = String>) -> ExitCode {
    let collected: Vec<String> = args.collect();
    let refs: Vec<&str> = collected.iter().map(String::as_str).collect();
    match run_lines(&refs) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sparkflash convert: {error}");
            ExitCode::from(2)
        }
    }
}

/// Runs `sparkflash convert` and returns output lines.
pub fn run_lines(args: &[&str]) -> Result<Vec<String>, String> {
    let mut checkpoint: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut world_size: usize = 4;
    let mut ranks: Vec<usize> = Vec::new();
    let mut pad_moe_640 = true;
    let mut verify = true;
    let mut inspect: Option<PathBuf> = None;
    let mut verify_weights = false;

    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index];
        let mut take = |name: &str| -> Result<String, String> {
            index += 1;
            args.get(index)
                .map(|v| (*v).to_string())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag {
            "--checkpoint" => checkpoint = Some(PathBuf::from(take("--checkpoint")?)),
            "--output-dir" | "-o" => output_dir = Some(PathBuf::from(take(flag)?)),
            "--tp-world-size" | "--world-size" => {
                let raw = take(flag)?;
                world_size = raw
                    .parse::<usize>()
                    .map_err(|_| format!("{flag}: {raw} is not a positive integer"))?;
            }
            "--tp-rank" => {
                let raw = take("--tp-rank")?;
                let rank = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--tp-rank: {raw} is not a rank index"))?;
                ranks.push(rank);
            }
            "--ranks" => {
                let raw = take("--ranks")?;
                for item in raw.split(',') {
                    let trimmed = item.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let rank = trimmed
                        .parse::<usize>()
                        .map_err(|_| format!("--ranks: {trimmed} is not a rank index"))?;
                    ranks.push(rank);
                }
            }
            "--no-pad-moe-640" => pad_moe_640 = false,
            "--no-verify" => verify = false,
            "--verify-weights" | "--verify" => {
                verify = true;
                verify_weights = true;
            }
            "--inspect" => inspect = Some(PathBuf::from(take("--inspect")?)),
            "--help" | "-h" => return Ok(vec![USAGE.to_string()]),
            other => return Err(format!("{other}: unknown flag (usage: {USAGE})")),
        }
        index += 1;
    }

    if let Some(path) = inspect {
        return inspect_file(&path, verify_weights);
    }

    let checkpoint_dir =
        checkpoint.ok_or_else(|| format!("--checkpoint DIR is required (usage: {USAGE})"))?;
    let out_dir = output_dir.unwrap_or_else(|| checkpoint_dir.clone());
    if let Some(config) = qwen_config(&checkpoint_dir)? {
        return convert_qwen(&checkpoint_dir, &out_dir, &config, &ranks, verify);
    }
    if let Some(config) = glm_config(&checkpoint_dir)? {
        if !pad_moe_640 {
            return Err(
                "--no-pad-moe-640 is DeepSeek's routed-expert layout, and this checkpoint is GLM-5.3"
                    .to_string(),
            );
        }
        return convert_glm(
            &checkpoint_dir,
            &out_dir,
            &config,
            world_size,
            &ranks,
            verify,
        );
    }
    let options = ConvertOptions {
        checkpoint_dir,
        output_dir: out_dir.clone(),
        world_size,
        ranks,
        pad_moe_640,
        verify,
        progress: Some(emit_progress),
    };
    let manifest = convert_checkpoint(&options)?;
    let mut lines = Vec::new();
    for summary in &manifest.ranks {
        lines.push(format!(
            "sparkflash convert: rank {}/{}: wrote {} ({} bytes, {} tensors, {} B12x, {} 640-padded MoE, header_hash={}, file_digest={}, {} ms)",
            summary.rank,
            summary.world,
            out_dir.join(&summary.file_name).display(),
            summary.file_bytes,
            summary.tensors,
            summary.tiled_tensors,
            summary.padded_moe_tensors,
            summary.header_hash,
            summary.file_digest,
            summary.convert_ms,
        ));
    }
    lines.push(format!(
        "sparkflash convert: manifest: {}",
        out_dir
            .join(sparkflash_device::converted::CONVERTED_MANIFEST_NAME)
            .display()
    ));
    Ok(lines)
}

/// The GLM-5.3 configuration of the checkpoint in `dir`, when its `config.json` names GLM's model
/// type. A checkpoint with no `config.json`, or another model's, is DeepSeek's to convert.
fn glm_config(dir: &Path) -> Result<Option<Glm5NextConfig>, String> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let root: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    if root.get("model_type").and_then(Value::as_str) != Some(MODEL_TYPE) {
        return Ok(None);
    }
    Glm5NextConfig::from_json(&text)
        .map(Some)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// The Qwen4-Exp configuration of the checkpoint in `dir`, when its `config.json` names the
/// pack's model type.
fn qwen_config(dir: &Path) -> Result<Option<Qwen4ExpConfig>, String> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let root: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let model_type = root.get("model_type").and_then(Value::as_str);
    if model_type != Some("qwen4_exp") && model_type != Some("qwen4_exp_text") {
        return Ok(None);
    }
    Qwen4ExpConfig::from_json(&text)
        .map(Some)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Converts the TP=1 rank file of a Qwen4-Exp checkpoint and re-reads its digests when `verify`
/// asks. The pack is TP=1; a rank other than 0 is refused rather than named a second file.
fn convert_qwen(
    dir: &Path,
    out_dir: &Path,
    config: &Qwen4ExpConfig,
    ranks: &[usize],
    verify: bool,
) -> Result<Vec<String>, String> {
    if let Some(rank) = ranks.iter().find(|rank| **rank != 0) {
        return Err(format!(
            "Qwen4-Exp is TP=1 and has no rank {rank}; convert rank 0 only"
        ));
    }
    let source = SafeTensorsSource::open(dir, &dir.join(INDEX_NAME))
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    let label = dir.file_name().map_or_else(
        || dir.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    let started = Instant::now();
    let converted = qwen_convert_rank(&source, config, dir, out_dir, &label)
        .map_err(|error| format!("Qwen4-Exp rank 0/1: {error}"))?;
    if verify {
        qwen_verify_converted(&converted.path)
            .map_err(|error| format!("Qwen4-Exp rank 0/1: {error}"))?;
    }
    Ok(vec![format!(
        "sparkflash convert: {QWEN_CONVERTED_MODEL} rank 0/1: wrote {} ({} bytes, {} weights, {} NVFP4 scale pairs, header_hash={:#018x}, file_digest={:#018x}, verified={verify}, {} ms)",
        converted.path.display(),
        converted.file_bytes,
        converted.slots,
        qwen_read_index(&converted.path).map_or(0, |(_, index)| index.nvfp4_scalars.len()),
        converted.header_hash,
        converted.file_digest,
        started.elapsed().as_millis(),
    )])
}

/// Converts `ranks` (every rank when empty) of a GLM-5.3 checkpoint one at a time, so one rank's
/// slices are all the converter holds, and re-reads each file's digests when `verify` asks.
fn convert_glm(
    dir: &Path,
    out_dir: &Path,
    config: &Glm5NextConfig,
    world_size: usize,
    ranks: &[usize],
    verify: bool,
) -> Result<Vec<String>, String> {
    let source = SafeTensorsSource::open(dir, &dir.join(INDEX_NAME))
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    let label = dir.file_name().map_or_else(
        || dir.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    let ranks: Vec<usize> = if ranks.is_empty() {
        (0..world_size).collect()
    } else {
        ranks.to_vec()
    };
    let mut lines = Vec::new();
    for rank in ranks {
        let layout = TpLayout { rank, world_size };
        let started = Instant::now();
        let converted = convert_rank(&source, config, layout, out_dir, &label)
            .map_err(|error| format!("rank {rank}/{world_size}: {error}"))?;
        if verify {
            verify_converted(&converted.path, layout)
                .map_err(|error| format!("rank {rank}/{world_size}: {error}"))?;
        }
        lines.push(format!(
            "sparkflash convert: {CONVERTED_MODEL} rank {rank}/{world_size}: wrote {} ({} bytes, {} weights, header_hash={:#018x}, file_digest={:#018x}, verified={verify}, {} ms)",
            converted.path.display(),
            converted.file_bytes,
            converted.slots,
            converted.header_hash,
            converted.file_digest,
            started.elapsed().as_millis(),
        ));
    }
    Ok(lines)
}

/// One GLM rank file's index, checked as the rank and world its preamble names, and with
/// `verify_weights` every slot's digest.
fn inspect_glm(
    path: &Path,
    preamble: ConvertedPreamble,
    verify_weights: bool,
) -> Result<Vec<String>, String> {
    let layout = TpLayout {
        rank: preamble.rank,
        world_size: preamble.world,
    };
    let (preamble, index) = if verify_weights {
        verify_converted(path, layout).map(|index| (preamble, index))?
    } else {
        sparkflash_model_glm5_next::convert::read_index(path, layout)?
    };
    Ok(vec![format!(
        "sparkflash convert inspect {}: {} rank {}/{}, {} weights from {}, data_offset={}, padded_data_bytes={}, header_hash={:#018x}, file_digest={:#018x}, weights_verified={verify_weights}",
        path.display(),
        index.model,
        index.rank,
        index.world,
        index.slots.len(),
        index.source,
        preamble.data_offset,
        preamble.padded_data_bytes,
        preamble.header_hash,
        index.file_digest,
    )])
}

/// One Qwen4-Exp rank file's index, with `verify_weights` every slot's digest.
fn inspect_qwen(
    path: &Path,
    preamble: ConvertedPreamble,
    verify_weights: bool,
) -> Result<Vec<String>, String> {
    let index = if verify_weights {
        qwen_verify_converted(path)?
    } else {
        qwen_read_index(path)?.1
    };
    Ok(vec![format!(
        "sparkflash convert inspect {}: {} rank {}/{}, {} weights, {} NVFP4 scale pairs from {}, data_offset={}, padded_data_bytes={}, header_hash={:#018x}, file_digest={:#018x}, weights_verified={verify_weights}",
        path.display(),
        index.model,
        index.rank,
        index.world,
        index.slots.len(),
        index.nvfp4_scalars.len(),
        index.source,
        preamble.data_offset,
        preamble.padded_data_bytes,
        preamble.header_hash,
        index.file_digest,
    )])
}

fn inspect_file(path: &Path, verify_weights: bool) -> Result<Vec<String>, String> {
    let (preamble, index): (_, Value) = read_index(path)?;
    if index.get("model").and_then(Value::as_str) == Some(CONVERTED_MODEL) {
        return inspect_glm(path, preamble, verify_weights);
    }
    if index.get("model").and_then(Value::as_str) == Some(QWEN_CONVERTED_MODEL) {
        return inspect_qwen(path, preamble, verify_weights);
    }
    let (preamble, header) = inspect_converted_rank(path)?;
    if verify_weights {
        verify_converted_file(path)?;
    }
    Ok(vec![format!(
        "sparkflash convert inspect {}: rank {}/{}, {} tensors ({} B12x, {} 640-padded MoE), data_offset={}, allocated_bytes={}, padded_data_bytes={}, header_hash={:#018x}, file_digest={:#018x}, weights_verified={verify_weights}",
        path.display(),
        header.rank,
        header.world,
        header.tensors.len(),
        header.tiled_tensors,
        header.padded_moe_tensors,
        preamble.data_offset,
        header.allocated_bytes,
        header.padded_data_bytes,
        preamble.header_hash,
        header.file_digest,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_tiny_safetensors(dir: &Path) {
        let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
            (
                "layers.0.attn.wq_a.weight",
                "F8_E4M3",
                vec![32, 32],
                vec![3u8; 1024],
            ),
            (
                "layers.0.attn.wq_a.scale",
                "F8_E8M0",
                vec![1, 1],
                vec![0x7f],
            ),
            ("norm.weight", "BF16", vec![32], vec![1u8; 64]),
        ];
        let shard_name = "model-00001-of-00001.safetensors";
        let mut header = String::from("{");
        let mut offset = 0u64;
        for (idx, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if idx > 0 {
                header.push(',');
            }
            let begin = offset;
            offset += bytes.len() as u64;
            let shape_text = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\"data_offsets\":[{begin},{offset}]}}"
            ));
        }
        header.push('}');
        let mut header_bytes = header.into_bytes();
        while (8 + header_bytes.len()) % 8 != 0 {
            header_bytes.push(b' ');
        }
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        for (_, _, _, bytes) in &tensors {
            file_bytes.extend_from_slice(bytes);
        }
        fs::write(dir.join(shard_name), file_bytes).expect("write shard");
        let mut weight_map = String::new();
        for (idx, (name, _, _, _)) in tensors.iter().enumerate() {
            if idx > 0 {
                weight_map.push(',');
            }
            weight_map.push_str(&format!("\"{name}\":\"{shard_name}\""));
        }
        let index_text = format!("{{\"metadata\":{{}},\"weight_map\":{{{weight_map}}}}}");
        fs::write(dir.join("model.safetensors.index.json"), index_text).expect("write index");
    }

    #[test]
    fn convert_cli_writes_and_inspects_rank_file() {
        let ckpt = tempfile::tempdir().expect("ckpt");
        let out = tempfile::tempdir().expect("out");
        write_tiny_safetensors(ckpt.path());

        let lines = run_lines(&[
            "--checkpoint",
            ckpt.path().to_str().unwrap(),
            "--output-dir",
            out.path().to_str().unwrap(),
            "--tp-world-size",
            "4",
            "--tp-rank",
            "0",
        ])
        .expect("convert rank 0");
        assert!(lines.iter().any(|l| l.contains("rank 0/4")));

        let rank0 = out.path().join("rank-0-of-4.bin");
        let inspect_lines = run_lines(&["--inspect", rank0.to_str().unwrap(), "--verify-weights"])
            .expect("inspect rank 0");
        assert!(inspect_lines[0].contains("weights_verified=true"));
    }

    /// A checkpoint whose `config.json` names GLM-5.3 is the GLM pack's to convert, so DeepSeek's
    /// padding flag is refused for it, and `--inspect` reads a GLM rank file by its index's model.
    #[test]
    fn a_qwen_checkpoint_and_rank_file_go_to_the_qwen_pack() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let config = sparkflash_model_qwen4_exp::Qwen4ExpConfig::tiny();
        sparkflash_model_qwen4_exp::write_synthetic_checkpoint(directory.path(), &config, 7)
            .expect("the fixture writes");
        let output = tempfile::tempdir().expect("an output directory");
        let lines = run_lines(&[
            "--checkpoint",
            directory.path().to_str().unwrap(),
            "--output-dir",
            output.path().to_str().unwrap(),
        ])
        .expect("the Qwen fixture converts");
        assert!(lines[0].contains("qwen4-exp rank 0/1"), "{}", lines[0]);
        let file = output.path().join("rank-0-of-1.bin");
        assert!(file.is_file());
        let inspected = run_lines(&["--inspect", file.to_str().unwrap(), "--verify-weights"])
            .expect("the rank file inspects");
        assert!(inspected[0].contains("qwen4-exp"), "{}", inspected[0]);
        assert!(
            inspected[0].contains("weights_verified=true"),
            "{}",
            inspected[0]
        );
    }

    #[test]
    fn a_glm_checkpoint_and_rank_file_go_to_the_glm_pack() {
        let ckpt = tempfile::tempdir().expect("ckpt");
        fs::write(
            ckpt.path().join("config.json"),
            include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json"),
        )
        .expect("write config");
        let error = run_lines(&[
            "--checkpoint",
            ckpt.path().to_str().unwrap(),
            "--no-pad-moe-640",
        ])
        .expect_err("GLM has no 640-wide padding");
        assert!(error.contains("GLM-5.3"), "{error}");

        let config = Glm5NextConfig::small();
        let fixture = tempfile::tempdir().expect("fixture");
        sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint(
            fixture.path(),
            &config,
            7,
            2,
        )
        .expect("the fixture writes");
        let source = SafeTensorsSource::open(fixture.path(), &fixture.path().join(INDEX_NAME))
            .expect("the fixture opens");
        let out = tempfile::tempdir().expect("out");
        let layout = TpLayout {
            rank: 1,
            world_size: 2,
        };
        let converted =
            convert_rank(&source, &config, layout, out.path(), "fixture").expect("rank 1 converts");
        let lines = run_lines(&[
            "--inspect",
            converted.path.to_str().unwrap(),
            "--verify-weights",
        ])
        .expect("inspect rank 1");
        assert!(lines[0].contains("glm5-next rank 1/2"), "{}", lines[0]);
        assert!(lines[0].contains("weights_verified=true"), "{}", lines[0]);
    }
}
