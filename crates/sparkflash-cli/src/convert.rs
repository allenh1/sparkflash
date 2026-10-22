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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sparkflash_device::converted::{
    ConvertOptions, convert_checkpoint, inspect_converted_rank, verify_converted_file,
};

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
    let options = ConvertOptions {
        checkpoint_dir,
        output_dir: out_dir.clone(),
        world_size,
        ranks,
        pad_moe_640,
        verify,
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

fn inspect_file(path: &Path, verify_weights: bool) -> Result<Vec<String>, String> {
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
}
