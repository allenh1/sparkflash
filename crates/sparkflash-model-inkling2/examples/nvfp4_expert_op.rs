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

//! One NVFP4 expert projection on real Inkling-2-Small tensors, serving arm against the shared
//! tile.
//!
//! It reads one released expert's `w13_weight` and `w2_weight` codes, per-16 E4M3 scales, the
//! expert's `amax` and the outer-scale denominator, and one real activation row from
//! `model.llm.embed.weight`. The serving arm is `inkling_nvfp4_expert_matvec`; the candidate is
//! the rows form of `shared_prefill_mma_sm121.cuh` through
//! `nvfp4_rows_inkling_{gate_up,down}_sm121`.
//!
//! Both operations have the same shape (4608 outputs x 2304 or 4608 contraction), so the w13 and
//! w2 projections are each one launch and each is compared separately. The factor is the
//! release's `amax / outer_scale_denominator`, applied to the output by both bodies.
//!
//! The harness prints max absolute/relative difference, max ULP distance and bit-equal count for
//! each projection, then median launch milliseconds over `--repeat`.
//!
//! Usage, after `scripts/build-cubins.sh`:
//! ```
//! cargo run -p sparkflash-model-inkling2 --example nvfp4_expert_op --release -- \
//!   --checkpoint <Inkling-2-Small snapshot> --cubin-dir target/cubins \
//!   [--layer 0] [--expert 0] [--repeat 50]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_weights::WeightSource;
use sparkflash_weights::quant::bf16_to_f32;
use sparkflash_weights::safetensors::SafeTensorsSource;

const HIDDEN: usize = 4608;
const INTER: usize = 2304;
const EXPERTS: usize = 384;

fn main() {
    if let Err(error) = run() {
        eprintln!("nvfp4_expert_op: {error}");
        std::process::exit(1);
    }
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    layer: usize,
    expert: usize,
    repeat: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("target/cubins"),
            layer: 0,
            expert: 0,
            repeat: 50,
        };
        let mut argv = std::env::args().skip(1);
        while let Some(flag) = argv.next() {
            let mut value = || argv.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--checkpoint" => args.checkpoint = PathBuf::from(value()?),
                "--cubin-dir" => args.cubin_dir = PathBuf::from(value()?),
                "--layer" => args.layer = value()?.parse().map_err(|e| format!("layer: {e}"))?,
                "--expert" => args.expert = value()?.parse().map_err(|e| format!("expert: {e}"))?,
                "--repeat" => args.repeat = value()?.parse().map_err(|e| format!("repeat: {e}"))?,
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required".to_owned());
        }
        Ok(args)
    }
}

fn upload(cuda: &Cuda, bytes: &[u8], total: &mut usize) -> Result<Deviceptr, String> {
    let ptr = cuda.alloc(bytes.len(), total)?;
    cuda.copy_htod(&ptr, bytes)?;
    Ok(ptr)
}

fn upload_f32(cuda: &Cuda, values: &[f32], total: &mut usize) -> Result<Deviceptr, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload(cuda, &bytes, total)
}

fn download_f32(cuda: &Cuda, ptr: Deviceptr, count: usize) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; count * 4];
    cuda.copy_dtoh(&mut bytes, &ptr)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn compare(name: &str, serving: &[f32], candidate: &[f32]) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_ulp = 0u32;
    let mut equal = 0usize;
    for (&a, &b) in serving.iter().zip(candidate) {
        if a.to_bits() == b.to_bits() {
            equal += 1;
        }
        if !a.is_finite() || !b.is_finite() {
            if a.to_bits() != b.to_bits() {
                println!("{name}: NON-FINITE MISMATCH {a} vs {b}");
            }
            continue;
        }
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / a.abs().max(1e-12));
        let ulp = if a.to_bits() >= b.to_bits() {
            a.to_bits() - b.to_bits()
        } else {
            b.to_bits() - a.to_bits()
        };
        max_ulp = max_ulp.max(ulp);
    }
    println!(
        "{name}: values={} bit_equal={equal} max_abs={max_abs:.6e} max_rel={max_rel:.6e} max_ulp={max_ulp}",
        serving.len()
    );
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    samples[samples.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn run_projection(
    cuda: &Cuda,
    name: &str,
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    factor: f32,
    x: &[f32],
    serving: sparkflash_kernels::cuda_driver::Kernel,
    tile: sparkflash_kernels::cuda_driver::Kernel,
    repeat: usize,
) -> Result<(), String> {
    let mut total = 0usize;
    let codes_d = upload(cuda, codes, &mut total)?;
    let scales_d = upload(cuda, scales, &mut total)?;
    let x_d = upload_f32(cuda, x, &mut total)?;
    let serving_out = cuda.alloc(rows * 4, &mut total)?;
    let tile_out = cuda.alloc(rows * 4, &mut total)?;
    cuda.launch(
        serving,
        (rows as u32, 1, 1),
        (256, 1, 1),
        &[
            codes_d.0,
            scales_d.0,
            x_d.0,
            serving_out.0,
            rows as u64,
            columns as u64,
            u64::from(factor.to_bits()),
            0,
        ],
    )?;
    cuda.launch(
        tile,
        ((rows as u32).div_ceil(32), 1, 1),
        (128, 1, 1),
        &[
            codes_d.0,
            scales_d.0,
            u64::from(factor.to_bits()),
            x_d.0,
            tile_out.0,
            rows as u64,
            columns as u64,
            1,
        ],
    )?;
    cuda.synchronize()?;
    let serving_values = download_f32(cuda, serving_out, rows)?;
    let candidate_values = download_f32(cuda, tile_out, rows)?;
    compare(name, &serving_values, &candidate_values);

    let mut serving_samples = Vec::new();
    let mut tile_samples = Vec::new();
    for _ in 0..repeat {
        let start = Instant::now();
        cuda.launch(
            serving,
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                codes_d.0,
                scales_d.0,
                x_d.0,
                serving_out.0,
                rows as u64,
                columns as u64,
                u64::from(factor.to_bits()),
                0,
            ],
        )?;
        cuda.synchronize()?;
        serving_samples.push(start.elapsed().as_secs_f64() * 1e3);
        let start = Instant::now();
        cuda.launch(
            tile,
            ((rows as u32).div_ceil(32), 1, 1),
            (128, 1, 1),
            &[
                codes_d.0,
                scales_d.0,
                u64::from(factor.to_bits()),
                x_d.0,
                tile_out.0,
                rows as u64,
                columns as u64,
                1,
            ],
        )?;
        cuda.synchronize()?;
        tile_samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{name} latency: serving_arm={:.3} ms, shared_tile={:.3} ms (median of {repeat})",
        median_ms(serving_samples),
        median_ms(tile_samples)
    );
    cuda.free_buffers(&[&codes_d, &scales_d, &x_d, &serving_out, &tile_out]);
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let index = args.checkpoint.join("model.safetensors.index.json");
    let source = SafeTensorsSource::open(&args.checkpoint, &index)
        .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?
        .with_limit(u128::MAX);
    let prefix = format!("model.llm.layers.{}.mlp.experts", args.layer);
    let w13 = source
        .read(&format!("{prefix}.w13_weight"))
        .map_err(|e| format!("w13: {e}"))?;
    let w13_s = source
        .read(&format!("{prefix}.w13_weight_scale"))
        .map_err(|e| format!("w13 scale: {e}"))?;
    let w2 = source
        .read(&format!("{prefix}.w2_weight"))
        .map_err(|e| format!("w2: {e}"))?;
    let w2_s = source
        .read(&format!("{prefix}.w2_weight_scale"))
        .map_err(|e| format!("w2 scale: {e}"))?;
    let amax13 = source
        .read(&format!("{prefix}.w13_weight_amax"))
        .map_err(|e| format!("w13 amax: {e}"))?;
    let amax2 = source
        .read(&format!("{prefix}.w2_weight_amax"))
        .map_err(|e| format!("w2 amax: {e}"))?;
    let denom13 = source
        .read(&format!("{prefix}.w13_weight_outer_scale_denominator"))
        .map_err(|e| format!("w13 denominator: {e}"))?;
    let denom2 = source
        .read(&format!("{prefix}.w2_weight_outer_scale_denominator"))
        .map_err(|e| format!("w2 denominator: {e}"))?;
    let read_scalar = |bytes: &[u8], name: &str| -> Result<f32, String> {
        let word: [u8; 4] = bytes
            .get(..4)
            .ok_or_else(|| format!("{name}: {} bytes", bytes.len()))?
            .try_into()
            .expect("four bytes");
        Ok(f32::from_le_bytes(word))
    };
    let expert = args.expert;
    if expert >= EXPERTS {
        return Err(format!("expert {expert} is outside {EXPERTS}"));
    }
    let w13_row = 2 * INTER * (HIDDEN / 2);
    let w13_scale_row = 2 * INTER * (HIDDEN / 16);
    let w2_row = HIDDEN * (INTER / 2);
    let w2_scale_row = HIDDEN * (INTER / 16);
    let slice = |bytes: &[u8], offset: usize, len: usize| -> Vec<u8> {
        bytes[offset..offset + len].to_vec()
    };
    let w13 = slice(&w13, expert * w13_row, w13_row);
    let w13_s = slice(&w13_s, expert * w13_scale_row, w13_scale_row);
    let w2 = slice(&w2, expert * w2_row, w2_row);
    let w2_s = slice(&w2_s, expert * w2_scale_row, w2_scale_row);

    let embed = source
        .read("model.llm.embed.weight")
        .map_err(|e| format!("embed: {e}"))?;
    let x: Vec<f32> = embed
        .as_chunks::<2>()
        .0
        .iter()
        .take(HIDDEN)
        .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
        .collect();

    let factor13 =
        read_scalar(&amax13[expert * 4..], "w13 amax")? / read_scalar(&denom13, "w13 denominator")?;
    let factor2 =
        read_scalar(&amax2[expert * 4..], "w2 amax")? / read_scalar(&denom2, "w2 denominator")?;

    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join("nvfp4_prefill_sm121_inkling.cubin");
    let cubin = std::fs::read(&cubin_path).map_err(|e| format!("{}: {e}", cubin_path.display()))?;
    let (_serving_module, serving) = cuda.load_named(
        sparkflash_kernels::images::inkling_cubin(),
        "inkling_nvfp4_expert_matvec",
    )?;
    let (_tile_gate_up_module, tile_gate_up) =
        cuda.load_named(&cubin, "nvfp4_rows_inkling_gate_up_sm121")?;
    let (_tile_down_module, tile_down) =
        cuda.load_named(&cubin, "nvfp4_rows_inkling_down_sm121")?;

    run_projection(
        &cuda,
        "w13",
        &w13,
        &w13_s,
        2 * INTER,
        HIDDEN,
        factor13,
        &x,
        serving,
        tile_gate_up,
        args.repeat,
    )?;
    run_projection(
        &cuda,
        "w2",
        &w2,
        &w2_s,
        HIDDEN,
        INTER,
        factor2,
        &x,
        serving,
        tile_down,
        args.repeat,
    )?;
    Ok(())
}
