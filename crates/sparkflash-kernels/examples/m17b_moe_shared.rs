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

//! m17b - the shared expert at real layer-0 geometry (the second half of m17).
//!
//! The released layer-0 FFN is a routed MoE block plus one shared expert. m17a measured the routed
//! branch; this harness measures the shared expert and declares the shared-add epilogue. It runs the
//! released shared-expert bytes through the pinned MXFP8 device path in the order the layer computes
//! them:
//!
//! 1. w1 and w3 dense GEMMs: `b12x_mxfp8_dense_sm121`, A = the activation stand-in quantised
//!    per row (E4M3 + ue8m0), W = the released shared w1/w3 repacked to B12x order, out bf16
//!    [tokens][2304].
//! 2. SwiGLU + MXFP8 quantise: `swiglu_mxfp8_quant_sm121` at the released 2304-wide
//!    intermediate, codes [tokens][2304] + ue8m0 scales [tokens][72].
//! 3. w2 dense GEMM: `b12x_mxfp8_dense_sm121`, A = stage 2's device codes/scales, W = the
//!    released shared w2 repacked, out bf16 [tokens][5120].
//! 4. shared-add epilogue: HOST, because no pinned kernel owns the elementwise add. The reference's
//!    own combination (routed bf16 + shared bf16 -> f32) is applied to the device's stage-3 bf16
//!    values and a declared stand-in routed vector. This stage is NOT a device measurement.
//!
//! The shared expert runs whole and replicated at the released 2304 width (D30); there is no rank
//! slicing here and no routed expert is touched. The routed branch is m17a and its window evidence
//! is not this harness's input.
//!
//! Pins. Two cubins are refused by size and sha256 before any allocation, and their constants are
//! cross-checked against the embedded baseline manifest (the crate's single source of pins):
//!
//! - `b12x_mxfp8_dense_sm121`: 190,400 B,
//!   `d751f2b65b076659ff0c4cc37542f8ae4899ec61b7f40c3d8223ba3181f9d350`
//! - `swiglu_mxfp8_quant_sm121`: 21,912 B,
//!   `17183ac9f1652850f319b2c21860016fac90db7cea631c991649415885b31008`
//!
//! Build lines (nvcc 13.0.88 / CUDA 13.0.3):
//! `nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/<kernel>.cu -o <dir>/<kernel>.cubin`.
//!
//! Comparison classes are printed before the numbers they judge. Stages 1 and 3 (dense): every output
//! bf16 bit must equal `mxfp8_ref::dense_gemm` on the SAME activation bytes, the SAME repacked
//! weights and the SAME scales (the M7 stage-7 and m12 class; the scalar path has no
//! reduction-order bar), and two launches must be byte-identical. Stage 2 (SwiGLU quantise): E4M3
//! codes and ue8m0 scales bit-exact against `swiglu_ref::quantise_row` over the same bf16
//! values (the stage-8 B1 class), and two launches byte-identical. Stage 4 (shared-add): HOST, exact
//! by construction - the reference's own combination function is applied to the same bytes; the
//! check is that an independent f32 loop in this harness reproduces it bit-for-bit, and the operand
//! provenance is printed. No device claim.
//!
//! Activation stand-in: released `embed.weight` rows 0..tokens, quantised by
//! `act_quant_fp8_row`. This tree does not carry layer 0's post-attention hidden into an
//! isolated harness, so every claim is about the shared expert's arithmetic on released bytes, not
//! about layer text.
//!
//! Usage (aarch64 node, checkpoint present):
//!
//! ```text
//! m17b_moe_shared --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 8   [--layer 0]
//! m17b_moe_shared --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 130 [--layer 0]
//! ```

#![forbid(unsafe_code)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, combine_bf16_routed_with_shared};
use sparkflash_engine::config::ModelConfig;
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::{Cuda, Kernel, KernelId};
use sparkflash_kernels::m4_quant::{QUANT_BLOCK, act_quant_fp8_row, bf16_value};
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref;
use sparkflash_kernels::swiglu_ref::{self, SWIGLU_LIMIT};

/// Hidden width (config.json text_config.hidden_size).
const HIDDEN: usize = 5120;
/// Shared-expert intermediate width (moe_intermediate_size); the released config asserts one shared
/// expert at this same width (D30: it stays whole and replicated).
const INTER: usize = 2304;
/// ue8m0 scales per hidden row.
const HIDDEN_SCALES: usize = HIDDEN / QUANT_BLOCK;
/// ue8m0 scales per intermediate row.
const INTER_SCALES: usize = INTER / QUANT_BLOCK;
/// The stored shared w1/w3 scale grid rows (2304 / 32).
const INTER_SCALE_ROWS: usize = INTER / QUANT_BLOCK;
/// The stored shared w1/w3 scale grid columns (5120 / 32).
const HIDDEN_SCALE_COLS: usize = HIDDEN / QUANT_BLOCK;

/// The dense MXFP8 kernel shipped name / pin.
const DENSE_KERNEL: &str = "b12x_mxfp8_dense_sm121";
const DENSE_PIN_BYTES: usize = 209_424;
const DENSE_PIN_SHA256: &str = "d751f2b65b076659ff0c4cc37542f8ae4899ec61b7f40c3d8223ba3181f9d350";
/// The SwiGLU + quantiser kernel shipped name / pin.
const SWIGLU_KERNEL: &str = "swiglu_mxfp8_quant_sm121";
const SWIGLU_PIN_BYTES: usize = 22_416;
const SWIGLU_PIN_SHA256: &str = "17183ac9f1652850f319b2c21860016fac90db7cea631c991649415885b31008";

/// One named pass/fail verdict, printed with its detail.
struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

impl Verdict {
    fn push(verdicts: &mut Vec<Self>, name: &str, ok: bool, detail: String) {
        println!(
            "  check {name}: {} ({detail})",
            if ok { "ok" } else { "FAIL" }
        );
        verdicts.push(Self {
            name: name.to_owned(),
            ok,
            detail,
        });
    }
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    tokens: usize,
    layer: usize,
    floor_gib: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1).collect())
    }

    fn parse_from(values: Vec<String>) -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            tokens: 8,
            layer: 0,
            floor_gib: 32,
        };
        let mut values = values;
        while let Some(flag) = values.first().cloned() {
            values.remove(0);
            let mut next = || {
                values
                    .first()
                    .cloned()
                    .inspect(|_| {
                        values.remove(0);
                    })
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--checkpoint" => args.checkpoint = PathBuf::from(next()?),
                "--cubin-dir" => args.cubin_dir = PathBuf::from(next()?),
                "--tokens" => {
                    args.tokens = next()?
                        .parse()
                        .map_err(|error| format!("--tokens: {error}"))?;
                }
                "--layer" => {
                    args.layer = next()?
                        .parse()
                        .map_err(|error| format!("--layer: {error}"))?;
                }
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|error| format!("--floor-gib: {error}"))?;
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required".to_owned());
        }
        if args.tokens == 0 {
            return Err("--tokens must be at least 1".to_owned());
        }
        Ok(args)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m17b_moe_shared: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m17b: shared expert at real layer-0 geometry (w1/w3 dense, 2304-wide SwiGLU+quantise, w2 dense, host shared-add) ==="
    );
    println!(
        "geometry: hidden {HIDDEN}, shared intermediate {INTER}, one shared expert (whole, replicated; D30); tokens {}",
        args.tokens
    );
    println!(
        "devices: two pinned cubins, each verified by size+sha256 AND the embedded baseline manifest before any allocation; each stage's comparison class is printed before the numbers it judges"
    );
    println!(
        "what this is not: not the served forward pass, not a chain/token/per-layer claim, not a timing claim, not the routed block (m17a), and not a device measurement of the shared-add (host seam). The activation rows and the routed stand-in are released embed.weight rows, a declared stand-in for layer 0's post-attention hidden and routed output."
    );
    check_floor(args.floor_gib, "m17b start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The released configuration, read from the checkpoint's own config.json.
    let config_path = args.checkpoint.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config = ModelConfig::from_released_json(&config_text)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let geometry_ok = config.dim == HIDDEN
        && config.moe_inter_dim == INTER
        && config.n_shared_experts == 1
        && config.expert_dtype == "fp4";
    println!(
        "released config.json: dim {}, moe_inter_dim {}, shared experts {}, routed experts {}, expert_dtype {}",
        config.dim,
        config.moe_inter_dim,
        config.n_shared_experts,
        config.n_routed_experts,
        config.expert_dtype
    );
    Verdict::push(
        &mut verdicts,
        "geometry matches the released config.json",
        geometry_ok,
        "hidden 5120, shared intermediate 2304, exactly 1 shared expert, fp4 experts".to_owned(),
    );

    let prefix = format!("layers.{}", args.layer);

    // ---------------------------------------------------------------------------------------
    // Released shared-expert operands: stored row-major FP8 weights and their ue8m0 scale grids.
    // ---------------------------------------------------------------------------------------
    let w1_raw = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.weight"),
        "F8_E4M3",
        &[INTER, HIDDEN],
    )?;
    let s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[INTER_SCALE_ROWS, HIDDEN_SCALE_COLS],
    )?;
    let w3_raw = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.weight"),
        "F8_E4M3",
        &[INTER, HIDDEN],
    )?;
    let s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[INTER_SCALE_ROWS, HIDDEN_SCALE_COLS],
    )?;
    let w2_raw = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.weight"),
        "F8_E4M3",
        &[HIDDEN, INTER],
    )?;
    let s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN_SCALE_COLS, INTER_SCALE_ROWS],
    )?;
    println!(
        "released shared operands: w1/w3 [{}x{}] F8_E4M3 ({} B each) + scales [{}x{}] F8_E8M0 ({} B each), w2 [{}x{}] F8_E4M3 ({} B) + scales [{}x{}] ({} B)",
        INTER,
        HIDDEN,
        w1_raw.len(),
        INTER_SCALE_ROWS,
        HIDDEN_SCALE_COLS,
        s1.len(),
        HIDDEN,
        INTER,
        w2_raw.len(),
        HIDDEN_SCALE_COLS,
        INTER_SCALE_ROWS,
        s2.len()
    );

    // B12x repack: a value-preserving index permutation on the host (no arithmetic), the same
    // declaration m12 and the M7 stage-7 chain make. The scale grids need no repack.
    let w1_packed = pack_b12x(&w1_raw, INTER, HIDDEN);
    let w3_packed = pack_b12x(&w3_raw, INTER, HIDDEN);
    let w2_packed = pack_b12x(&w2_raw, HIDDEN, INTER);
    let pack_ok = w1_packed.len() == mxfp8_ref::b12x_buffer_bytes(INTER, HIDDEN)
        && w3_packed.len() == mxfp8_ref::b12x_buffer_bytes(INTER, HIDDEN)
        && w2_packed.len() == mxfp8_ref::b12x_buffer_bytes(HIDDEN, INTER);
    println!(
        "host seam: B12x repack of the released shared weights - value-preserving index permutation, no arithmetic; packed w1/w3 {} B each, w2 {} B ({} B total read from the checkpoint)",
        w1_packed.len(),
        w2_packed.len(),
        w1_raw.len() + w3_raw.len() + w2_raw.len() + s1.len() + s3.len() + s2.len()
    );
    if !pack_ok {
        return Err("the B12x repack sizes do not match mxfp8_ref::b12x_buffer_bytes".to_owned());
    }

    // Activation stand-in: released embed rows, quantised per row by the D8 quantiser.
    let embed_bytes = read_embed_rows(&mut snap, args.tokens)?;
    let embed_f32 = bf16_bytes_to_f32(&embed_bytes);
    let mut act_codes = vec![0u8; args.tokens * HIDDEN];
    let mut act_scales = vec![0u8; args.tokens * HIDDEN_SCALES];
    for token in 0..args.tokens {
        let row = &embed_f32[token * HIDDEN..(token + 1) * HIDDEN];
        let (codes, scales) = act_quant_fp8_row(row);
        act_codes[token * HIDDEN..(token + 1) * HIDDEN].copy_from_slice(&codes);
        act_scales[token * HIDDEN_SCALES..(token + 1) * HIDDEN_SCALES].copy_from_slice(&scales);
    }
    println!(
        "activation stand-in: {} embed rows x {HIDDEN} bf16 values, D8-quantised to {} E4M3 bytes + {} ue8m0 bytes (released embed.weight rows, declared stand-in)",
        args.tokens,
        act_codes.len(),
        act_scales.len()
    );

    // ---------------------------------------------------------------------------------------
    // Pinned cubins: size + sha256 + the embedded baseline manifest, before any allocation.
    // ---------------------------------------------------------------------------------------
    let dense_cubin = pinned_cubin(
        &args.cubin_dir,
        DENSE_KERNEL,
        DENSE_PIN_BYTES,
        DENSE_PIN_SHA256,
    )?;
    let swiglu_cubin = pinned_cubin(
        &args.cubin_dir,
        SWIGLU_KERNEL,
        SWIGLU_PIN_BYTES,
        SWIGLU_PIN_SHA256,
    )?;
    compare_to_baseline(DENSE_KERNEL, DENSE_PIN_BYTES, DENSE_PIN_SHA256)?;
    compare_to_baseline(SWIGLU_KERNEL, SWIGLU_PIN_BYTES, SWIGLU_PIN_SHA256)?;

    check_floor(args.floor_gib, "m17b before context")?;
    let cuda = Cuda::load()?;
    let (_dense_module, dense_kernel) = cuda.load_kernel(&dense_cubin, KernelId::Mxfp8Dense)?;
    let (_swiglu_module, swiglu_kernel) =
        cuda.load_kernel(&swiglu_cubin, KernelId::SwigluMxfp8Quant)?;

    // ---------------------------------------------------------------------------------------
    // Stage 1: the two dense gate/up GEMMs at [tokens, 2304] over inner 5120.
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 1: b12x_mxfp8_dense_sm121 shared w1 and w3 ([tokens][2304], inner 5120) ---"
    );
    println!(
        "comparison class (declared before any number): every output bf16 bit equals mxfp8_ref::dense_gemm on the same activation codes/scales, the same repacked weights and the same weight scales - the M7 stage-7/m12 bit-identical class (the scalar path has one f32 accumulation in K-ascending 32-value blocks and one bf16 rounding, exactly the reference's order, so there is no reduction-order bar); two launches byte-identical per projection. A difference is a finding."
    );
    let gate = dense_apply(
        &cuda,
        dense_kernel,
        "stage 1 w1",
        args.layer,
        &act_codes,
        &act_scales,
        &w1_packed,
        &s1,
        args.tokens,
        INTER,
        HIDDEN,
        &mut verdicts,
    )?;
    let up = dense_apply(
        &cuda,
        dense_kernel,
        "stage 1 w3",
        args.layer,
        &act_codes,
        &act_scales,
        &w3_packed,
        &s3,
        args.tokens,
        INTER,
        HIDDEN,
        &mut verdicts,
    )?;
    println!(
        "  stage 1 output: {tokens} x {INTER} bf16 per projection",
        tokens = args.tokens
    );
    // The two outputs stay as bf16 bits; stage 2 consumes exactly these bytes.

    // ---------------------------------------------------------------------------------------
    // Stage 2: SwiGLU + MXFP8 quantise at the 2304-wide intermediate, on the device gate/up bits.
    // ---------------------------------------------------------------------------------------
    println!("--- stage 2: swiglu_mxfp8_quant_sm121 at the released 2304-wide intermediate ---");
    println!(
        "comparison class (declared before any number): E4M3 codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row over the same bf16 values (the stage-8 B1 class); two launches byte-identical. The launch is unweighted - the shared expert is not multiplied by a routing weight."
    );
    let gate_bytes = bf16_bits_to_bytes(&gate);
    let up_bytes = bf16_bits_to_bytes(&up);
    let mut expected_codes = vec![0u8; args.tokens * INTER];
    let mut expected_scales = vec![0u8; args.tokens * INTER_SCALES];
    let gate_f32 = bf16_bits_slice_to_f32(&gate, 0, gate.len());
    let up_f32 = bf16_bits_slice_to_f32(&up, 0, up.len());
    for row in 0..args.tokens {
        let (codes, scales) = swiglu_ref::quantise_row(
            &gate_f32[row * INTER..(row + 1) * INTER],
            &up_f32[row * INTER..(row + 1) * INTER],
            SWIGLU_LIMIT,
        );
        expected_codes[row * INTER..(row + 1) * INTER].copy_from_slice(&codes);
        expected_scales[row * INTER_SCALES..(row + 1) * INTER_SCALES].copy_from_slice(&scales);
    }
    let mut allocated = 0usize;
    let gate_d = cuda.alloc(gate_bytes.len(), &mut allocated)?;
    let up_d = cuda.alloc(up_bytes.len(), &mut allocated)?;
    let codes_d = cuda.alloc(expected_codes.len(), &mut allocated)?;
    let scales_d = cuda.alloc(expected_scales.len(), &mut allocated)?;
    println!(
        "  stage 2 device bytes admitted {allocated} for rows {} x cols {INTER}",
        args.tokens
    );
    cuda.copy_htod(&gate_d, &gate_bytes)?;
    cuda.copy_htod(&up_d, &up_bytes)?;
    let params: [u64; 10] = [
        gate_d.0,
        up_d.0,
        codes_d.0,
        scales_d.0,
        args.tokens as u64,
        INTER as u64,
        u64::from(SWIGLU_LIMIT.to_bits()),
        0,
        args.layer as u64,
        0,
    ];
    let total_blocks = (args.tokens * INTER / QUANT_BLOCK) as u32;
    let grid = (total_blocks.div_ceil(8), 1u32, 1u32);
    check_floor(args.floor_gib, "m17b stage 2 pre-launch")?;
    cuda.launch(swiglu_kernel, grid, (256, 1, 1), &params)?;
    let mut codes_first = vec![0u8; expected_codes.len()];
    let mut scales_first = vec![0u8; expected_scales.len()];
    cuda.copy_dtoh(&mut codes_first, &codes_d)?;
    cuda.copy_dtoh(&mut scales_first, &scales_d)?;
    cuda.launch(swiglu_kernel, grid, (256, 1, 1), &params)?;
    let mut codes_second = vec![0u8; expected_codes.len()];
    let mut scales_second = vec![0u8; expected_scales.len()];
    cuda.copy_dtoh(&mut codes_second, &codes_d)?;
    cuda.copy_dtoh(&mut scales_second, &scales_d)?;
    check_floor(args.floor_gib, "m17b stage 2 post-launch")?;
    cuda.free_buffers(&[&gate_d, &up_d, &codes_d, &scales_d]);
    let code_mismatches = codes_first
        .iter()
        .zip(&expected_codes)
        .filter(|(a, b)| a != b)
        .count();
    let scale_mismatches = scales_first
        .iter()
        .zip(&expected_scales)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "stage 2: E4M3 codes bit-exact",
        code_mismatches == 0,
        format!("{code_mismatches} mismatched of {}", codes_first.len()),
    );
    Verdict::push(
        &mut verdicts,
        "stage 2: ue8m0 scales bit-exact",
        scale_mismatches == 0,
        format!("{scale_mismatches} mismatched of {}", scales_first.len()),
    );
    Verdict::push(
        &mut verdicts,
        "stage 2: two launches byte-identical",
        codes_first == codes_second && scales_first == scales_second,
        format!(
            "{} code + {} scale bytes compared",
            codes_first.len(),
            scales_first.len()
        ),
    );

    // ---------------------------------------------------------------------------------------
    // Stage 3: the w2 dense GEMM at [tokens, 5120] over inner 2304, on the device codes/scales.
    // ---------------------------------------------------------------------------------------
    println!("--- stage 3: b12x_mxfp8_dense_sm121 shared w2 ([tokens][5120], inner 2304) ---");
    println!(
        "comparison class (declared before any number): every output bf16 bit equals mxfp8_ref::dense_gemm on the stage-2 device codes/scales, the same repacked w2 and the same weight scales; two launches byte-identical. This is the same bit-identical class as stage 1, at the second released shape."
    );
    let shared_bits = dense_apply(
        &cuda,
        dense_kernel,
        "stage 3 w2",
        args.layer,
        &codes_first,
        &scales_first,
        &w2_packed,
        &s2,
        args.tokens,
        HIDDEN,
        INTER,
        &mut verdicts,
    )?;
    println!(
        "  stage 3 output: {} x {HIDDEN} bf16 (the shared expert's contribution)",
        args.tokens
    );

    // ---------------------------------------------------------------------------------------
    // Stage 4: the shared-add epilogue, HOST (no pinned kernel owns the elementwise add).
    // ---------------------------------------------------------------------------------------
    println!(
        "--- stage 4: shared-add epilogue, host seam (declared, not a device measurement) ---"
    );
    println!(
        "comparison class (declared before any number): HOST, exact by construction - the reference's own combination (routed bf16 widened + shared bf16 widened, f32 sum) applied to the same bytes; the check is that an independent f32 loop in this harness reproduces it bit-for-bit. The routed operand is a declared stand-in (released embed rows), NOT the routed block's output; the routed block is m17a."
    );
    let routed_bits = bytes_bf16_bits(&embed_bytes);
    let shared_f32 = bf16_bits_slice_to_f32(&shared_bits, 0, shared_bits.len());
    let reference_ffn_out = combine_bf16_routed_with_shared(&routed_bits, &shared_f32)?;
    let independent_ffn_out: Vec<f32> = routed_bits
        .iter()
        .zip(&shared_f32)
        .map(|(bits, shared)| bf16_value(*bits) + shared)
        .collect();
    let seam_mismatches = reference_ffn_out
        .iter()
        .zip(&independent_ffn_out)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "  shared-add seam crossing: routed bf16 [{}][{HIDDEN}] (released embed rows, stand-in) + shared bf16 [{}][{HIDDEN}] (stage-3 device output) -> f32 [{}][{HIDDEN}] (reference's own combination order, host)",
        args.tokens, args.tokens, args.tokens
    );
    Verdict::push(
        &mut verdicts,
        "stage 4 host seam: the harness reproduces the reference's combination bit-for-bit",
        seam_mismatches == 0,
        format!(
            "{seam_mismatches} mismatched f32 of {} (host seam, not a device check)",
            reference_ffn_out.len()
        ),
    );

    // ---------------------------------------------------------------------------------------
    // Summary.
    // ---------------------------------------------------------------------------------------
    let failed: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    println!("--- NOT claimed by m17b ---");
    println!(
        "- not a device shared-add: the epilogue is host arithmetic and this run does not measure a kernel for it"
    );
    println!(
        "- not a chain, token, per-layer or end-to-end FFN claim: one layer's shared expert, one declared stand-in activation and a stand-in routed operand; the routed block is m17a and is not re-measured here"
    );
    println!("- not layer text: the activation and routed operand are released embed.weight rows");
    println!(
        "- not the shared expert at any other layer or a routed-expert claim; not engine integration; not a timing claim"
    );
    println!("- not a performance claim and not a host-time saving: no engine seam is changed");
    if failed.is_empty() {
        println!(
            "m17b: PASS ({} checks; 3 device stages + host shared-add; tokens {}; two pinned cubins)",
            verdicts.len(),
            args.tokens
        );
        Ok(())
    } else {
        for verdict in &failed {
            eprintln!("  FAIL {}: {}", verdict.name, verdict.detail);
        }
        Err(format!(
            "m17b: FAIL ({} of {} checks failed)",
            failed.len(),
            verdicts.len()
        ))
    }
}

// -----------------------------------------------------------------------------------------------
// Device helpers.
// -----------------------------------------------------------------------------------------------

/// One dense MXFP8 launch, compared bit-for-bit to the in-crate reference on the same bytes.
#[allow(clippy::too_many_arguments)] // the launch takes the operand set and the shape as distinct arguments.
fn dense_apply(
    cuda: &Cuda,
    kernel: Kernel,
    label: &str,
    layer: usize,
    a_codes: &[u8],
    a_scales: &[u8],
    w_packed: &[u8],
    w_scales: &[u8],
    rows: usize,
    cols: usize,
    inner: usize,
    verdicts: &mut Vec<Verdict>,
) -> Result<Vec<u16>, String> {
    let k_blocks = inner / QUANT_BLOCK;
    if a_codes.len() != rows * inner
        || a_scales.len() != rows * k_blocks
        || w_packed.len() != mxfp8_ref::b12x_buffer_bytes(cols, inner)
        || w_scales.len() != mxfp8_ref::b12x_scale_bytes(cols, inner)
    {
        return Err(format!(
            "{label}: operands are {} codes / {} scales for A [{} x {}], {} packed / {} scale bytes for W [{} x {}]",
            a_codes.len(),
            a_scales.len(),
            rows,
            inner,
            w_packed.len(),
            w_scales.len(),
            cols,
            inner
        ));
    }
    let expected_bits =
        mxfp8_ref::dense_gemm(a_codes, a_scales, w_packed, w_scales, rows, cols, inner);
    let mut allocated = 0usize;
    let a_d = cuda.alloc(a_codes.len(), &mut allocated)?;
    let as_d = cuda.alloc(a_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_packed.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(rows * cols * 2, &mut allocated)?;
    cuda.copy_htod(&a_d, a_codes)?;
    cuda.copy_htod(&as_d, a_scales)?;
    cuda.copy_htod(&w_d, w_packed)?;
    cuda.copy_htod(&ws_d, w_scales)?;
    let params = Mxfp8DenseParams::new(
        a_d,
        as_d,
        w_d,
        ws_d,
        out_d,
        rows as u32,
        cols as u32,
        inner as u32,
        0,
        layer as u32,
        0,
    );
    let grid = mxfp8_dense_launch::grid(rows as u32, cols as u32)?;
    if params.as_slice().len() != mxfp8_dense_launch::ARG_WORDS {
        return Err(format!(
            "{label}: the typed params carry {} words, the pinned declaration has {}",
            params.as_slice().len(),
            mxfp8_dense_launch::ARG_WORDS
        ));
    }
    println!(
        "  {label}: {rows} rows x {cols} cols, inner {inner}; grid {grid:?}, block {:?}, {} argument words through Mxfp8DenseParams",
        mxfp8_dense_launch::BLOCK,
        params.as_slice().len()
    );
    mxfp8_dense_launch::launch(cuda, kernel, &params)?;
    let mut first = vec![0u8; rows * cols * 2];
    cuda.copy_dtoh(&mut first, &out_d)?;
    mxfp8_dense_launch::launch(cuda, kernel, &params)?;
    let mut second = vec![0u8; rows * cols * 2];
    cuda.copy_dtoh(&mut second, &out_d)?;
    cuda.free_buffers(&[&a_d, &as_d, &w_d, &ws_d, &out_d]);
    let bits = bytes_bf16_bits(&first);
    let mismatches = bits
        .iter()
        .zip(&expected_bits)
        .filter(|(device, reference)| device != reference)
        .count();
    Verdict::push(
        verdicts,
        &format!("{label}: bit-identical to mxfp8_ref::dense_gemm"),
        mismatches == 0,
        format!("{mismatches} mismatched bf16 values of {}", bits.len()),
    );
    Verdict::push(
        verdicts,
        &format!("{label}: two launches byte-identical"),
        first == second,
        format!("{} bf16 bytes compared", first.len()),
    );
    Ok(bits)
}

// -----------------------------------------------------------------------------------------------
// Host helpers.
// -----------------------------------------------------------------------------------------------

/// Repack a checkpoint FP8 weight from row-major [cols, inner] into the B12x micro-tile order the
/// kernel and mxfp8_ref::dense_gemm walk. The scale grid needs no repack.
fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        weights.len(),
        cols * inner,
        "the checkpoint stores one E4M3 byte per FP8 value"
    );
    let mut tiled = vec![0u8; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            tiled[mxfp8_ref::b12x_weight_index(col, k, inner)] = weights[col * inner + k];
        }
    }
    tiled
}

fn check_floor(floor_gib: u64, point: &str) -> Result<(), String> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("cannot read /proc/meminfo: {error}"))?;
    let kb = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "cannot parse MemAvailable".to_owned())?;
    let bytes = kb.saturating_mul(1024);
    let floor = floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes < floor {
        return Err(format!(
            "MemAvailable at {point} is {kb} kB ({:.2} GiB), below the {floor_gib} GiB floor: refusing to run",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }
    println!(
        "MemAvailable at {point}: {kb} kB ({:.1} GiB), floor {floor_gib} GiB - ok",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn pinned_cubin(
    dir: &std::path::Path,
    name: &str,
    bytes: usize,
    sha: &str,
) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{name}.cubin"));
    let cubin = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/{name}.cu -o {})",
            path.display(),
            path.display()
        )
    })?;
    let digest = Sha256::digest(&cubin);
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "cubin: {} ({} B, sha256 {hex})",
        path.display(),
        cubin.len()
    );
    if cubin.len() != bytes || hex != sha {
        return Err(format!(
            "{name}: pinned {bytes} B / {sha}, got {} B / {hex}; refusing to measure a different kernel",
            cubin.len()
        ));
    }
    Ok(cubin)
}

/// Cross-check a harness pin against the crate's embedded baseline manifest (one source of pins).
fn compare_to_baseline(name: &str, bytes: usize, sha: &str) -> Result<(), String> {
    let pin = baseline::pin(name)?
        .ok_or_else(|| format!("{name}: the embedded baseline manifest has no pin"))?;
    if pin.cubin_bytes != bytes || pin.cubin_sha256 != sha {
        return Err(format!(
            "{name}: the harness pins {bytes} B / {sha}, the baseline manifest says {} B / {}",
            pin.cubin_bytes, pin.cubin_sha256
        ));
    }
    println!(
        "baseline manifest: {name} {} B / {sha} ({} PTX membar lines) - the harness pin is the manifest pin",
        pin.cubin_bytes, pin.ptx_membar_lines
    );
    Ok(())
}

fn read_embed_rows(snap: &mut Snapshot, tokens: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(tokens * HIDDEN * 2);
    for token in 0..tokens {
        let mut row = vec![0u8; HIDDEN * 2];
        snap.read_row_bf16_into("embed.weight", token, HIDDEN, &mut row)?;
        bytes.extend_from_slice(&row);
    }
    Ok(bytes)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_value(u16::from_le_bytes(*pair)))
        .collect()
}

fn bytes_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len() * 2);
    for value in bits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bf16_bits_slice_to_f32(bits: &[u16], start: usize, len: usize) -> Vec<f32> {
    bits[start..start + len]
        .iter()
        .map(|value| bf16_value(*value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_documented_window_command_parses() {
        let args = Args::parse_from(
            [
                "--checkpoint",
                "/snap",
                "--cubin-dir",
                "/cubins",
                "--tokens",
                "130",
                "--layer",
                "0",
                "--floor-gib",
                "32",
            ]
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        )
        .expect("the documented command parses");
        assert_eq!(args.tokens, 130);
        assert_eq!(args.layer, 0);
        assert_eq!(args.floor_gib, 32);
        assert_eq!(args.cubin_dir, PathBuf::from("/cubins"));
    }

    #[test]
    fn the_b12x_repack_is_a_value_preserving_permutation() {
        let cols = 64usize;
        let inner = 64usize;
        let weights: Vec<u8> = (0..cols * inner).map(|value| value as u8).collect();
        let packed = pack_b12x(&weights, cols, inner);
        assert_eq!(packed.len(), mxfp8_ref::b12x_buffer_bytes(cols, inner));
        for col in 0..cols {
            for k in 0..inner {
                assert_eq!(
                    packed[mxfp8_ref::b12x_weight_index(col, k, inner)],
                    weights[col * inner + k],
                    "the repack must only move bytes"
                );
            }
        }
    }

    #[test]
    fn the_shared_add_is_the_reference_combination_in_the_reference_order() {
        let routed_bits: Vec<u16> = vec![0x3f80, 0x4000, 0xbe80]; // 1.0, 2.0, -0.25
        let shared = vec![0.5f32, -1.0, 0.25];
        let reference = combine_bf16_routed_with_shared(&routed_bits, &shared).unwrap();
        let independent: Vec<f32> = routed_bits
            .iter()
            .zip(&shared)
            .map(|(bits, value)| bf16_value(*bits) + value)
            .collect();
        assert_eq!(reference, independent);
    }

    #[test]
    fn the_pin_constants_are_the_baseline_manifest_lines() {
        let manifest = baseline::BASELINE_TEXT;
        for (name, bytes, sha) in [
            (DENSE_KERNEL, DENSE_PIN_BYTES, DENSE_PIN_SHA256),
            (SWIGLU_KERNEL, SWIGLU_PIN_BYTES, SWIGLU_PIN_SHA256),
        ] {
            let line = format!("kernel {name} cubin_bytes={bytes} cubin_sha256={sha} ");
            assert!(
                manifest.contains(&line),
                "{name}: the harness pin is not the baseline line: {line}"
            );
        }
    }
}
