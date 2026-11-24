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

//! One NVFP4 routed-expert operation on real Qwen3.8 tensors, serving arm against the shared
//! tile.
//!
//! It reads one released expert's gate/up/down codes, per-16 E4M3 scales and `weight_scale_2`
//! from the checkpoint, and one real activation row from `embed_tokens.weight`. The serving arm
//! is the embedded `q4e_expert_gate_up`/`q4e_expert_down`; the candidate is
//! `nvfp4_prefill_qwen_{gate_up,down}_sm121` from the shared bulk-prefill tile.
//!
//! Gate/up: the serving arm writes the activated (`silu(gate) * up`) rows itself; the shared tile
//! is launched twice (gate and up) and the same host `silu(gate) * up` is applied. Down: both
//! read the serving arm's activated rows, so only the NVFP4 decode and GEMM are compared.
//!
//! It prints, for each operation, the maximum absolute and relative difference, the maximum ULP
//! distance, the count of bit-identical values, and median launch milliseconds over
//! `--repeat` runs. The bar is printed before the numbers: no NaN, max relative difference below
//! 2 %, and the gate/up/down timings.
//!
//! Usage, after `scripts/build-cubins.sh`:
//! ```
//! cargo run -p sparkflash-model-qwen4-exp --example nvfp4_expert_op --release -- \
//!   --checkpoint <Qwen3.8-Flash-Next-NVFP4 snapshot> --cubin-dir target/cubins \
//!   [--layer 0] [--expert 0] [--repeat 50]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_weights::WeightSource;
use sparkflash_weights::quant::bf16_to_f32;
use sparkflash_weights::safetensors::SafeTensorsSource;

const HIDDEN: usize = 2560;
const INTER: usize = 640;
/// The serving arm's slots one batch row carries (Qwen3.8's top-10).
const SLOTS: usize = 10;

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

fn read_f32(source: &SafeTensorsSource, name: &str) -> Result<f32, String> {
    let bytes = source.read(name).map_err(|e| format!("{name}: {e}"))?;
    let word: [u8; 4] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| format!("{name}: {} bytes, expected one f32", bytes.len()))?;
    Ok(f32::from_le_bytes(word))
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

fn compare(name: &str, serving: &[f32], candidate: &[f32]) -> Result<(), String> {
    assert_eq!(serving.len(), candidate.len(), "{name}: lengths");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_ulp = 0u32;
    let mut equal = 0usize;
    for (&a, &b) in serving.iter().zip(candidate) {
        if !a.is_finite() || !b.is_finite() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("{name}: non-finite mismatch {a} vs {b}"));
            }
            continue;
        }
        if a == b {
            equal += 1;
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
    if max_rel > 0.02 {
        return Err(format!("{name}: max_rel {max_rel:.6e} exceeds 2%"));
    }
    Ok(())
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let index = args.checkpoint.join("model.safetensors.index.json");
    let source = SafeTensorsSource::open(&args.checkpoint, &index)
        .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?;
    let prefix = format!(
        "model.language_model.layers.{}.mlp.experts.{}",
        args.layer, args.expert
    );
    let gate = source
        .read(&format!("{prefix}.gate_proj.weight"))
        .map_err(|e| format!("gate weight: {e}"))?;
    let gate_s = source
        .read(&format!("{prefix}.gate_proj.weight_scale"))
        .map_err(|e| format!("gate scale: {e}"))?;
    let up = source
        .read(&format!("{prefix}.up_proj.weight"))
        .map_err(|e| format!("up weight: {e}"))?;
    let up_s = source
        .read(&format!("{prefix}.up_proj.weight_scale"))
        .map_err(|e| format!("up scale: {e}"))?;
    let down = source
        .read(&format!("{prefix}.down_proj.weight"))
        .map_err(|e| format!("down weight: {e}"))?;
    let down_s = source
        .read(&format!("{prefix}.down_proj.weight_scale"))
        .map_err(|e| format!("down scale: {e}"))?;
    let gate_wide = read_f32(&source, &format!("{prefix}.gate_proj.weight_scale_2"))?;
    let up_wide = read_f32(&source, &format!("{prefix}.up_proj.weight_scale_2"))?;
    let down_wide = read_f32(&source, &format!("{prefix}.down_proj.weight_scale_2"))?;
    let _input_scale = read_f32(&source, &format!("{prefix}.gate_proj.input_scale"))?;

    // One real activation row: the first released embedding row, as f32.
    let embed = source
        .read("model.language_model.embed_tokens.weight")
        .map_err(|e| format!("embed: {e}"))?;
    let embed: Vec<f32> = embed
        .as_chunks::<2>()
        .0
        .iter()
        .take(HIDDEN)
        .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
        .collect();
    let mut x = Vec::with_capacity(SLOTS * HIDDEN);
    for _ in 0..SLOTS {
        x.extend_from_slice(&embed);
    }

    let cuda = Cuda::load()?;
    let mut total = 0usize;
    let gate_d = upload(&cuda, &gate, &mut total)?;
    let gate_s_d = upload(&cuda, &gate_s, &mut total)?;
    let up_d = upload(&cuda, &up, &mut total)?;
    let up_s_d = upload(&cuda, &up_s, &mut total)?;
    let down_d = upload(&cuda, &down, &mut total)?;
    let down_s_d = upload(&cuda, &down_s, &mut total)?;
    let table = [gate_d.0, gate_s_d.0, up_d.0, up_s_d.0, down_d.0, down_s_d.0];
    let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let table_d = upload(&cuda, &table_bytes, &mut total)?;
    let scales_d = upload_f32(&cuda, &[gate_wide, up_wide, down_wide], &mut total)?;
    let indices: Vec<i32> = vec![0; SLOTS];
    let perm: Vec<i32> = (0..SLOTS as i32).collect();
    let bounds: [i32; 2] = [0, SLOTS as i32 - 1];
    let indices_d = upload(
        &cuda,
        &indices
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        &mut total,
    )?;
    let perm_d = upload(
        &cuda,
        &perm
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        &mut total,
    )?;
    let bounds_d = upload(
        &cuda,
        &bounds
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        &mut total,
    )?;
    let x_d = upload_f32(&cuda, &x, &mut total)?;
    let activated = cuda.alloc(SLOTS * INTER * 4, &mut total)?;
    let partials = cuda.alloc(SLOTS * HIDDEN * 4, &mut total)?;

    let shared = (2 * HIDDEN.max(INTER).div_ceil(32) * 33 * std::mem::size_of::<f32>()) as u32;
    let (_serving_gate_up_module, serving_gate_up) = cuda.load_named_shared(
        sparkflash_model_qwen4_exp::device::embedded_moe_cubin(),
        "q4e_expert_gate_up",
        shared,
    )?;
    let (_serving_down_module, serving_down) = cuda.load_named_shared(
        sparkflash_model_qwen4_exp::device::embedded_moe_cubin(),
        "q4e_expert_down",
        shared,
    )?;

    let cubin_path = args.cubin_dir.join("nvfp4_prefill_sm121_qwen.cubin");
    let cubin = std::fs::read(&cubin_path).map_err(|e| format!("{}: {e}", cubin_path.display()))?;
    let (_tile_gate_up_module, tile_gate_up) =
        cuda.load_named(&cubin, "nvfp4_prefill_qwen_gate_up_sm121")?;
    let (_tile_down_module, tile_down) =
        cuda.load_named(&cubin, "nvfp4_prefill_qwen_down_sm121")?;

    let gate_grid = (((INTER as u32).div_ceil(16)), 1, 1);
    let down_grid = (((HIDDEN as u32).div_ceil(32)), 1, 1);
    let args_serving = [
        table_d.0,
        scales_d.0,
        indices_d.0,
        perm_d.0,
        bounds_d.0,
        x_d.0,
        activated.0,
        SLOTS as u64,
        HIDDEN as u64,
        INTER as u64,
        0,
    ];
    cuda.launch(serving_gate_up, gate_grid, (256, 1, 1), &args_serving)?;
    cuda.launch(
        serving_down,
        down_grid,
        (256, 1, 1),
        &[
            table_d.0,
            scales_d.0,
            indices_d.0,
            perm_d.0,
            bounds_d.0,
            activated.0,
            partials.0,
            SLOTS as u64,
            HIDDEN as u64,
            INTER as u64,
            INTER as u64,
        ],
    )?;
    cuda.synchronize()?;
    let serving_activated = download_f32(&cuda, activated, SLOTS * INTER)?;
    let serving_down = download_f32(&cuda, partials, SLOTS * HIDDEN)?;

    // The shared tile: gate and up separately, the same host SwiGLU, then down on the serving
    // arm's activated rows.
    let gate_out = cuda.alloc(SLOTS * INTER * 4, &mut total)?;
    let up_out = cuda.alloc(SLOTS * INTER * 4, &mut total)?;
    let words = |w: Deviceptr,
                 s: Deviceptr,
                 wide: f32,
                 a: Deviceptr,
                 out: Deviceptr,
                 cols: usize,
                 inner: usize| {
        [
            w.0,
            s.0,
            u64::from(wide.to_bits()),
            a.0,
            out.0,
            cols as u64,
            inner as u64,
            SLOTS as u64,
        ]
    };
    cuda.launch(
        tile_gate_up,
        (
            (INTER as u32).div_ceil(128),
            (SLOTS as u32).div_ceil(128),
            1,
        ),
        (256, 1, 1),
        &words(gate_d, gate_s_d, gate_wide, x_d, gate_out, INTER, HIDDEN),
    )?;
    cuda.launch(
        tile_gate_up,
        (
            (INTER as u32).div_ceil(128),
            (SLOTS as u32).div_ceil(128),
            1,
        ),
        (256, 1, 1),
        &words(up_d, up_s_d, up_wide, x_d, up_out, INTER, HIDDEN),
    )?;
    cuda.synchronize()?;
    let gates = download_f32(&cuda, gate_out, SLOTS * INTER)?;
    let ups = download_f32(&cuda, up_out, SLOTS * INTER)?;
    let candidate_activated: Vec<f32> = gates
        .iter()
        .zip(&ups)
        .map(|(&g, &u)| sparkflash_weights::linalg::silu(g) * u)
        .collect();
    compare(
        "gate_up(activated)",
        &serving_activated,
        &candidate_activated,
    )?;

    let act_d = upload_f32(&cuda, &serving_activated, &mut total)?;
    let down_out = cuda.alloc(SLOTS * HIDDEN * 4, &mut total)?;
    cuda.launch(
        tile_down,
        (
            (HIDDEN as u32).div_ceil(128),
            (SLOTS as u32).div_ceil(128),
            1,
        ),
        (256, 1, 1),
        &words(down_d, down_s_d, down_wide, act_d, down_out, HIDDEN, INTER),
    )?;
    cuda.synchronize()?;
    let candidate_down = download_f32(&cuda, down_out, SLOTS * HIDDEN)?;
    compare("down", &serving_down, &candidate_down)?;

    let mut serving_samples = Vec::new();
    let mut tile_samples = Vec::new();
    for _ in 0..args.repeat {
        let start = Instant::now();
        cuda.launch(serving_gate_up, gate_grid, (256, 1, 1), &args_serving)?;
        cuda.synchronize()?;
        serving_samples.push(start.elapsed().as_secs_f64() * 1e3);
        let start = Instant::now();
        cuda.launch(
            tile_gate_up,
            (
                (INTER as u32).div_ceil(128),
                (SLOTS as u32).div_ceil(128),
                1,
            ),
            (256, 1, 1),
            &words(gate_d, gate_s_d, gate_wide, x_d, gate_out, INTER, HIDDEN),
        )?;
        cuda.launch(
            tile_gate_up,
            (
                (INTER as u32).div_ceil(128),
                (SLOTS as u32).div_ceil(128),
                1,
            ),
            (256, 1, 1),
            &words(up_d, up_s_d, up_wide, x_d, up_out, INTER, HIDDEN),
        )?;
        cuda.synchronize()?;
        tile_samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "gate_up latency: serving_arm={:.3} ms, shared_tile={:.3} ms ({} slots, median of {})",
        median_ms(serving_samples),
        median_ms(tile_samples),
        SLOTS,
        args.repeat
    );

    Ok(())
}
