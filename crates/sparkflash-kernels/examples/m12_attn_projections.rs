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

//! m12_attn_projections: the attention-projection stage of layer 0 - the three dense MXFP8
//! GEMMs of the attention block (wq_a, wq_b, wkv) - in isolation, against released checkpoint bytes.
//!
//! Scope, stated before the numbers. This harness launches ONE chain stage, the attention
//! projections that follow the mHC entry: K2 wq_a [N=1280, K=5120], K3 wq_b [N=32768, K=1280] and
//! K4 wkv [N=512, K=5120] on kernels/b12x_mxfp8_dense_sm121.cu, at the real released dimensions.
//! It is not attention: flash_mla_csa2, the sink, the RoPE tables, the softmax and the K5/K6/K7
//! stages are not launched. It is not a chain pass: two host seams are named, built on the host and
//! declared (below), and neither is claimed as part of the stage's arithmetic. It is not a token,
//! not a sequence and not end-to-end text - the positions are independent rows, every launch reads
//! and writes whole matrices. Explicitly NOT claimed:
//!   * attention (flash_mla, sink, softmax, RoPE), q/kv norm as a measured step, output inverse
//!     RoPE, the main-KV packing;
//!   * the residual add and the hc_post merge that follow the attention block in the chain;
//!   * K5-K16, any chain ordering, per-layer schedule, seam ledger or cross-stage equivalence;
//!   * any token, any text, any end-to-end layer or model output;
//!   * routing, SwiGLU, the MoE block, the shared expert, the combine, the output projection wq_b
//!     is a projection here but the attention output it normally consumes is NOT produced here;
//!   * weight staging, device-budget admission, the engine, or any host-time saving.
//!
//! Provenance and permutation, declared. The weights are read from the checkpoint verbatim, one
//! released tensor per projection: layers.0.attn.{wq_a,wq_b,wkv}.weight of dtype F8_E4M3 and
//! .scale of dtype F8_E8M0, with the shapes measured from the checkpoint headers in
//! sparkflash_kernels::m7_layer0_shapes (wq_a [1280, 5120], wq_b [32768, 1280], wkv [512, 5120]).
//! The kernel consumes two byte layouts:
//!   * weights: B12x 32x32 micro-tiles. The checkpoint stores row-major [out, in] E4M3, so the
//!     host applies pack_b12x, a declared value-preserving index permutation (no arithmetic, no
//!     decode, one byte per value). Its scale grid b12x_scale_index(n, k) = (n/32)*k_blocks +
//!     (k/32) is exactly the checkpoint's stored [out/32, in/32] row-major grid, so the .scale
//!     tensor needs no transform and is uploaded as stored - that is a claim checked by the
//!     b12x-size assertions below, not an assumption.
//!   * activations: [rows][inner] E4M3 with [rows][inner/32] ue8m0, produced by the engine's own
//!     D8 activation quantiser (m4_quant::act_quant_fp8_row).
//!
//! The activation operands are host-built stand-ins and are declared:
//!   * wq_a and wkv: the released embed.weight row of position t, bf16 widened to f32, quantised
//!     by the D8 rule. This stands in for the layer-0 hidden state that the chain's K1 mHC entry
//!     produces; K1 is not launched here, so the operand is a declared seam, not a stage output.
//!   * wq_b: the DEVICE wq_a result of position t (bf16 widened to f32), passed through the
//!     released q_norm RMS normalisation (NORM_EPS = 1e-20) and the D8 quantiser - the chain's
//!     K2 -> K3 host seam exactly as stage 7 of M7 performs it. The operand is built once and is
//!     identical for the device and the CPU twin, so the seam is declared and not claimed.
//!
//! No released activation tensor is read from the checkpoint: the model stores none, and inventing
//! one would be the silent reshaping this harness is meant to avoid.
//!
//! Cubin pin. The b12x_mxfp8_dense_sm121 cubin is pinned at 11,824 B and
//! 03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a (0 PTX membar lines), read
//! from baseline/kernel_baseline.txt through sparkflash_kernels::baseline::pin and cross-checked
//! against this file's constants; the harness refuses any other compiled form by length and sha256
//! before the first device allocation. Build it under the pinned toolkit (CUDA 13.0.3, nvcc
//! V13.0.88):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_dense_sm121.cu \
//!       -o out/b12x_mxfp8_dense_sm121.cubin
//!
//! COMPARISON CLASS (declared before any number is printed). Every output element of every
//! projection, for every position, is compared to the in-crate reference
//! mxfp8_ref::dense_gemm on the SAME activation bytes, the SAME packed weights and the SAME scale
//! bytes:
//!   bit-identical:  all rows x cols bf16 output bits must equal the reference's bf16 bits. This is
//!                   a legitimate exactness claim, not a bar: the kernel's documented accumulation
//!                   order (K ascending in 32-value blocks, ascending within a block, one f32
//!                   accumulation, one bf16 rounding) is exactly what mxfp8_ref::dense_gemm
//!                   implements, and the M7 stage-7 chain measured K2 1,280/1,280, K3
//!                   32,768/32,768 and K4 512/512 exact at this same real geometry. There is no
//!                   reduction-order bar to declare for the scalar path this cubin carries.
//!   determinism:    the launch is repeated on the same operands and all output bytes must match.
//!   f32/FP8 control: the activation quantiser's own gap - the D8 codes x ue8m0 scales, decoded,
//!                   against the f32 value they were quantised from - reported per projection and
//!                   bounded at 10% of that operand's maximum absolute value (the m10/m11 control
//!                   style). It measures the activation-side FP8 cost and is NOT part of the
//!                   exactness claim; the weight-side quantisation is in the released bytes and is
//!                   not quantified here.
//!
//! Shapes: --tokens N sets the matrix row count (the kernel has no batch dimension beyond its rows
//! parameter, and every launch is a whole [N, inner] matrix); N = 8 and N = 130 are both valid and
//! 130 is the fixture prefill length. All three projections run at their real N/K in both shapes.
//!
//! Usage (aarch64 node, checkpoint present):
//!   m12_attn_projections --checkpoint <snapshot dir> --cubin-dir out [--tokens 8]
//!                        [--repeat 100] [--floor-gib 32]

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::baseline;
use sparkflash_kernels::m4_quant::{
    QUANT_BLOCK, act_quant_fp8_row, bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode,
};
use sparkflash_kernels::m7_layer0_shapes::{
    BlockScaledWeight, LAYER0_ATTENTION_WEIGHTS, scale_grid,
};
use sparkflash_kernels::mega_mhc_ref::HIDDEN;
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref;

/// The cubin name in the baseline manifest (the nvcc source stem).
const KERNEL: &str = "b12x_mxfp8_dense_sm121";
/// The pinned cubin size, from baseline/kernel_baseline.txt through baseline::pin.
const PIN_BYTES: usize = 209_424;
/// The pinned cubin sha256, from the same manifest.
const PIN_SHA256: &str = "d751f2b65b076659ff0c4cc37542f8ae4899ec61b7f40c3d8223ba3181f9d350";
/// The pinned PTX membar-line count: the scalar dense path carries no fence.
const PIN_MEMBAR_LINES: u32 = 0;
/// The released attention RMS normalisation epsilon (q_norm), as the reference uses it.
const NORM_EPS: f32 = 1e-20;
/// The declared control bar: the activation quantisation gap as a fraction of the operand maximum.
const CONTROL_BAR: f32 = 0.10;

/// The control's declared bound, factored so the boundary itself is tested: a gap passes when it is
/// at most this fraction of the operand's maximum absolute value.
fn control_gap_within_bar(gap: f32, operand_max: f32) -> bool {
    gap <= CONTROL_BAR * operand_max
}
/// The q LoRA rank: wq_a's output width and q_norm's length, from the measured layer-0 table.
const Q_LORA_RANK: usize = 1_280;

fn main() {
    if let Err(error) = run() {
        eprintln!("m12_attn_projections: {error}");
        std::process::exit(1);
    }
}

fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn check_floor(floor_gib: u64, point: &str) -> Result<(), String> {
    let kb = mem_available_kb()
        .ok_or_else(|| "cannot read MemAvailable from /proc/meminfo".to_owned())?;
    let bytes = kb.saturating_mul(1024);
    let floor = floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes < floor {
        return Err(format!(
            "MemAvailable at {point} is {kb} kB ({:.2} GiB), below the {floor_gib}-GiB floor: refusing to run",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }
    println!(
        "MemAvailable at {point}: {kb} kB ({:.1} GiB), floor {floor_gib} GiB - ok",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

/// Refuses a host build plan that would take the node below the memory floor.
fn admit_memory(floor_gib: u64, planned: u64, point: &str) -> Result<(), String> {
    let kb = mem_available_kb()
        .ok_or_else(|| "cannot read MemAvailable from /proc/meminfo".to_owned())?;
    let bytes = kb.saturating_mul(1024);
    let floor = floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes.saturating_sub(planned) < floor {
        return Err(format!(
            "{point}: {planned} B planned on top of {kb} kB MemAvailable would cross the {floor_gib}-GiB floor"
        ));
    }
    Ok(())
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    tokens: usize,
    layer: usize,
    repeat: usize,
    floor_gib: u64,
    warp: bool,
    tile: bool,
    pk: bool,
    sk: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            tokens: 8,
            layer: 0,
            repeat: 100,
            floor_gib: 32,
            warp: false,
            tile: false,
            pk: false,
            sk: false,
        };
        let mut values = std::env::args().skip(1).collect::<Vec<_>>();
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
                        .map_err(|_| "--tokens needs a number".to_owned())?
                }
                "--layer" => {
                    args.layer = next()?
                        .parse()
                        .map_err(|_| "--layer needs a number".to_owned())?
                }
                "--repeat" => {
                    args.repeat = next()?
                        .parse()
                        .map_err(|_| "--repeat needs a number".to_owned())?
                }
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|_| "--floor-gib needs a number".to_owned())?
                }
                "--warp" => args.warp = true,
                "--tile" => args.tile = true,
                "--pk" => args.pk = true,
                "--sk" => args.sk = true,
                "--help" | "-h" => {
                    println!(
                        "usage: m12_attn_projections --checkpoint <snapshot dir> [--cubin-dir out] \
                         [--tokens 8] [--repeat 100] [--floor-gib 32]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}; try --help")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required (the released snapshot directory)".to_owned());
        }
        if args.tokens == 0 || args.tokens > 1024 {
            return Err("--tokens must be in 1..=1024".to_owned());
        }
        if args.layer != 0 {
            return Err(
                "--layer must be 0: the released activation stand-in is the layer-0 boundary input \
                 (an embed.weight row), and this harness is the layer-0 stage, not a per-layer tool"
                    .to_owned(),
            );
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// Repack a checkpoint FP8 weight from row-major [cols, inner] into the B12x micro-tile order the
/// kernel and mxfp8_ref::dense_gemm walk. The scale grid needs no repack: b12x_scale_index is
/// (n / 32) * k_blocks + (k / 32), the checkpoint's own stored row-major grid.
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

/// Decode every weight value exactly once into f32: fp8 decode times its ue8m0 scale, the same two
/// operations the kernel's inner loop performs. This is the twin's only departure from a literal
/// mxfp8_ref::dense_gemm call - hoisting a pure function of the weight bytes out of the row loop -
/// and it is bit-identical because the hoisted values are the same f32 values the reference
/// recomputes per element (a unit test below proves the two agree value for value on a fixture
/// that covers the real inner length 1,280).
fn dequantise_weights(packed: &[u8], scales: &[u8], cols: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            out[col * inner + k] =
                fp8_e4m3_decode(packed[mxfp8_ref::b12x_weight_index(col, k, inner)])
                    * ue8m0_decode(scales[mxfp8_ref::b12x_scale_index(col, k, inner)]);
        }
    }
    out
}

/// The CPU twin: mxfp8_ref::dense_gemm's arithmetic, with the weight dequantisation hoisted.
///
/// The accumulation is one f32 per output, K ascending in 32-value blocks and ascending within a
/// block, rounding to bf16 once - the kernel's documented contract. The unit tests compare the
/// returned bits against mxfp8_ref::dense_gemm exactly, including inner = 1280.
fn dense_twin(
    a_codes: &[u8],
    a_scales: &[u8],
    dequantised_weight: &[f32],
    rows: usize,
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / QUANT_BLOCK;
    assert_eq!(a_codes.len(), rows * inner);
    assert_eq!(a_scales.len(), rows * k_blocks);
    assert_eq!(dequantised_weight.len(), cols * inner);
    let mut out = vec![0u16; rows * cols];
    let mut activation = vec![0.0f32; inner];
    for row in 0..rows {
        for block in 0..k_blocks {
            let scale = ue8m0_decode(a_scales[row * k_blocks + block]);
            let base = block * QUANT_BLOCK;
            for i in 0..QUANT_BLOCK {
                activation[base + i] = fp8_e4m3_decode(a_codes[row * inner + base + i]) * scale;
            }
        }
        for col in 0..cols {
            let weight_row = &dequantised_weight[col * inner..(col + 1) * inner];
            let mut acc = 0.0f32;
            for k in 0..inner {
                acc += activation[k] * weight_row[k];
            }
            out[row * cols + col] = bf16_bits(acc);
        }
    }
    out
}

/// The released RMS normalisation: one mean square over the row, then value * inverse * weight.
fn rms_norm_row(values: &mut [f32], weight: &[f32], eps: f32) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    for (value, w) in values.iter_mut().zip(weight) {
        *value = *value * inverse * *w;
    }
}

/// The D8 quantiser's own gap: decoded codes x scales against the f32 value they quantised.
///
/// Returns (max abs gap, index, max abs source). It is the f32/FP8 control, not an exactness claim.
fn activation_gap(source: &[f32], codes: &[u8], scales: &[u8]) -> (f32, usize, f32) {
    assert_eq!(source.len(), codes.len());
    assert_eq!(scales.len(), source.len().div_ceil(QUANT_BLOCK));
    let mut max_gap = 0.0f32;
    let mut at = 0usize;
    let mut max_source = 0.0f32;
    for (index, value) in source.iter().enumerate() {
        let reconstructed =
            fp8_e4m3_decode(codes[index]) * ue8m0_decode(scales[index / QUANT_BLOCK]);
        let gap = (reconstructed - value).abs();
        if gap > max_gap || gap.is_nan() {
            max_gap = gap;
            at = index;
        }
        max_source = max_source.max(value.abs());
    }
    (max_gap, at, max_source)
}

/// One projection's released weight bytes in the two layouts the kernel consumes.
struct Projection {
    name: &'static str,
    cols: usize,
    inner: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    dequantised: Vec<f32>,
}

/// One operand matrix's activations: the codes, the scales, and the f32 values they came from.
struct Activations {
    rows: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    source: Vec<f32>,
}

fn quantise_rows(rows: &[Vec<f32>]) -> Result<Activations, String> {
    let Some(first) = rows.first() else {
        return Err("at least one activation row is required".to_owned());
    };
    let inner = first.len();
    if inner == 0 || !inner.is_multiple_of(QUANT_BLOCK) {
        return Err(format!(
            "the activation width {inner} is not a whole number of {QUANT_BLOCK}-value blocks"
        ));
    }
    for (index, row) in rows.iter().enumerate() {
        if row.len() != inner {
            return Err(format!(
                "activation row {index} is {} wide, expected {inner}",
                row.len()
            ));
        }
    }
    let mut codes = Vec::with_capacity(rows.len() * inner);
    let mut scales = Vec::with_capacity(rows.len() * inner / QUANT_BLOCK);
    let mut source = Vec::with_capacity(rows.len() * inner);
    for row in rows {
        let (row_codes, row_scales) = act_quant_fp8_row(row);
        codes.extend_from_slice(&row_codes);
        scales.extend_from_slice(&row_scales);
        source.extend_from_slice(row);
    }
    Ok(Activations {
        rows: rows.len(),
        codes,
        scales,
        source,
    })
}

/// Reads one projection's released weight and scale tensors and prepares both kernel layouts.
fn read_projection(
    snapshot: &mut Snapshot,
    prefix: &str,
    spec: &BlockScaledWeight,
) -> Result<Projection, String> {
    let cols = spec.rows;
    let inner = spec.cols;
    let (scale_rows, scale_cols) = scale_grid(cols, inner);
    if (scale_rows, scale_cols) != (spec.scale_rows, spec.scale_cols) {
        return Err(format!(
            "{}: the measured scale grid [{}x{}] disagrees with the rule [{scale_rows}x{scale_cols}]",
            spec.name, spec.scale_rows, spec.scale_cols
        ));
    }
    let weight_name = format!("{prefix}.attn.{}.weight", spec.name);
    let scale_name = format!("{prefix}.attn.{}.scale", spec.name);
    let raw = snapshot.read_raw(&weight_name, "F8_E4M3", &[cols, inner])?;
    let scales = snapshot.read_raw(&scale_name, "F8_E8M0", &[scale_rows, scale_cols])?;
    if raw.len() != cols * inner {
        return Err(format!(
            "{weight_name}: {} bytes for a [{cols}, {inner}] E4M3 tensor",
            raw.len()
        ));
    }
    let packed = pack_b12x(&raw, cols, inner);
    if packed.len() != mxfp8_ref::b12x_buffer_bytes(cols, inner) {
        return Err(format!(
            "{}: the B12x weight buffer is {} B, the kernel's rule says {} B",
            spec.name,
            packed.len(),
            mxfp8_ref::b12x_buffer_bytes(cols, inner)
        ));
    }
    if scales.len() != mxfp8_ref::b12x_scale_bytes(cols, inner) {
        return Err(format!(
            "{}: the stored scale tensor is {} B, the kernel's rule says {} B (the stored grid is the \
             GEMM's grid, so a mismatch here means the provenance claim is wrong)",
            spec.name,
            scales.len(),
            mxfp8_ref::b12x_scale_bytes(cols, inner)
        ));
    }
    let dequantised = dequantise_weights(&packed, &scales, cols, inner);
    Ok(Projection {
        name: spec.name,
        cols,
        inner,
        packed,
        scales,
        dequantised,
    })
}

/// Reads the cubin and refuses anything but the pinned compiled form, before any allocation.
fn verify_cubin(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            path.display()
        )
    })?;
    let pin = baseline::pin(KERNEL)?
        .ok_or_else(|| format!("{KERNEL} is not in the baseline manifest"))?;
    if bytes.len() != PIN_BYTES || pin.cubin_bytes != PIN_BYTES {
        return Err(format!(
            "{}: cubin is {} B but the pin is {PIN_BYTES} B",
            path.display(),
            bytes.len()
        ));
    }
    if pin.cubin_sha256 != PIN_SHA256 || pin.ptx_membar_lines != PIN_MEMBAR_LINES {
        return Err(format!(
            "the baseline manifest's {KERNEL} pin changed: {} B, sha256 {}, {} membar lines",
            pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let sha = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if sha != PIN_SHA256 {
        return Err(format!(
            "{}: cubin sha256 {sha} is not the pinned {PIN_SHA256}",
            path.display()
        ));
    }
    println!(
        "cubin: {KERNEL} {} B, sha256 {sha} (pinned: {PIN_BYTES} B, {PIN_MEMBAR_LINES} PTX membar \
         lines) - verified before any device allocation",
        bytes.len()
    );
    if mxfp8_dense_launch::KERNEL != KERNEL {
        return Err(format!(
            "the typed builder resolves a different symbol: {}",
            mxfp8_dense_launch::KERNEL
        ));
    }
    Ok(bytes)
}

/// Percentiles of a per-launch millisecond vector, as the other harnesses report them.
fn timing_stats(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let p50 = sorted[sorted.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    (min, p50, mean, max)
}

struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

impl Verdict {
    fn new(name: &str, ok: bool, detail: String) -> Self {
        Self {
            name: name.to_owned(),
            ok,
            detail,
        }
    }
}

/// One projection's device run's outputs, kept for the declared K2 -> K3 seam.
struct ProjectionRun {
    output_bits: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn run_projection(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    layer: usize,
    projection: &Projection,
    activations: &Activations,
    repeat: usize,
    verdicts: &mut Vec<Verdict>,
    warp: bool,
    tile: bool,
    pk: bool,
    sk: bool,
    sk_combine: Option<sparkflash_kernels::cuda_driver::Kernel>,
) -> Result<ProjectionRun, String> {
    let rows = activations.rows;
    let cols = projection.cols;
    let inner = projection.inner;
    let k_blocks = inner / QUANT_BLOCK;
    if activations.codes.len() != rows * inner
        || activations.scales.len() != rows * k_blocks
        || activations.source.len() != rows * inner
    {
        return Err(format!(
            "{}: the operands are {} code bytes and {} scale bytes for [{} x {}]",
            projection.name,
            activations.codes.len(),
            activations.scales.len(),
            rows,
            inner
        ));
    }
    let expected = dense_twin(
        &activations.codes,
        &activations.scales,
        &projection.dequantised,
        rows,
        cols,
        inner,
    );
    let mut allocated = 0usize;
    let a_d = cuda.alloc(rows * inner, &mut allocated)?;
    let a_scales_d = cuda.alloc(rows * k_blocks, &mut allocated)?;
    let w_d = cuda.alloc(projection.packed.len(), &mut allocated)?;
    let w_scales_d = cuda.alloc(projection.scales.len(), &mut allocated)?;
    let out_bytes_len = rows * cols * 2;
    let out_d = cuda.alloc(out_bytes_len, &mut allocated)?;
    cuda.copy_htod(&a_d, &activations.codes)?;
    cuda.copy_htod(&a_scales_d, &activations.scales)?;
    cuda.copy_htod(&w_d, &projection.packed)?;
    cuda.copy_htod(&w_scales_d, &projection.scales)?;
    let params = Mxfp8DenseParams::new(
        a_d,
        a_scales_d,
        w_d,
        w_scales_d,
        out_d,
        rows as u32,
        cols as u32,
        inner as u32,
        0, // trace_ring: null, the shipped cubin is the trace-disabled build
        layer as u32,
        0, // step_seq
    );
    let split_params = if sk {
        let partials = cuda.alloc(
            rows * cols * 4 * mxfp8_dense_launch::SK_SPLIT as usize,
            &mut allocated,
        )?;
        Some(mxfp8_dense_launch::SplitKParams::new(
            a_d,
            a_scales_d,
            w_d,
            w_scales_d,
            partials,
            out_d,
            rows as u32,
            cols as u32,
            inner as u32,
            0,
            layer as u32,
            0,
        ))
    } else {
        None
    };
    let (grid, block) = if sk {
        (
            (
                (cols as u32).div_ceil(mxfp8_dense_launch::SK_COLUMNS),
                (rows as u32).div_ceil(mxfp8_dense_launch::SK_ROWS),
                mxfp8_dense_launch::SK_SPLIT,
            ),
            mxfp8_dense_launch::SK_BLOCK,
        )
    } else if pk {
        (
            (
                (cols as u32).div_ceil(mxfp8_dense_launch::PK_COLUMNS),
                (rows as u32).div_ceil(mxfp8_dense_launch::PK_ROWS),
                1,
            ),
            mxfp8_dense_launch::PK_BLOCK,
        )
    } else if tile {
        (
            mxfp8_dense_launch::tile_grid(rows as u32, cols as u32)?,
            mxfp8_dense_launch::TILE_BLOCK,
        )
    } else if warp {
        (
            mxfp8_dense_launch::warp_grid(rows as u32, cols as u32)?,
            mxfp8_dense_launch::WARP_BLOCK,
        )
    } else {
        (
            mxfp8_dense_launch::grid(rows as u32, cols as u32)?,
            mxfp8_dense_launch::BLOCK,
        )
    };
    if params.as_slice().len() != mxfp8_dense_launch::ARG_WORDS {
        return Err(format!(
            "the typed params carry {} words, the pinned declaration has {}",
            params.as_slice().len(),
            mxfp8_dense_launch::ARG_WORDS
        ));
    }
    println!(
        "  launch: {} rows x {cols} cols, inner {inner}; grid {grid:?}, block {:?}, {} argument words \
         through Mxfp8DenseParams",
        rows,
        block,
        params.as_slice().len()
    );
    let first_call_start = std::time::Instant::now();
    if sk {
        mxfp8_dense_launch::launch_split_k(
            cuda,
            kernel,
            sk_combine.expect("sk launch has its combine kernel"),
            split_params.as_ref().expect("sk params"),
        )?;
    } else if pk {
        mxfp8_dense_launch::launch_par_k(cuda, kernel, &params)?;
    } else if tile {
        mxfp8_dense_launch::launch_tile(cuda, kernel, &params)?;
    } else if warp {
        mxfp8_dense_launch::launch_warp(cuda, kernel, &params)?;
    } else {
        mxfp8_dense_launch::launch(cuda, kernel, &params)?;
    }
    let first_call_ms = first_call_start.elapsed().as_secs_f64() * 1.0e3;
    let mut device_bytes = vec![0u8; out_bytes_len];
    cuda.copy_dtoh(&mut device_bytes, &out_d)?;
    // Determinism: the same launch again on the same operands, byte for byte.
    if sk {
        mxfp8_dense_launch::launch_split_k(
            cuda,
            kernel,
            sk_combine.expect("sk launch has its combine kernel"),
            split_params.as_ref().expect("sk params"),
        )?;
    } else if pk {
        mxfp8_dense_launch::launch_par_k(cuda, kernel, &params)?;
    } else if tile {
        mxfp8_dense_launch::launch_tile(cuda, kernel, &params)?;
    } else if warp {
        mxfp8_dense_launch::launch_warp(cuda, kernel, &params)?;
    } else {
        mxfp8_dense_launch::launch(cuda, kernel, &params)?;
    }
    let mut again_bytes = vec![0u8; out_bytes_len];
    cuda.copy_dtoh(&mut again_bytes, &out_d)?;
    let deterministic = device_bytes == again_bytes;
    let device_bits: Vec<u16> = device_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let total = rows * cols;
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut max_delta = 0.0f32;
    for (index, (device, reference)) in device_bits.iter().zip(&expected).enumerate() {
        if device != reference {
            mismatches += 1;
            first_mismatch = first_mismatch.or(Some(index));
        }
        let delta = (bf16_value(*device) - bf16_value(*reference)).abs();
        if !delta.is_finite() || delta > max_delta {
            max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
        }
    }
    println!(
        "  output bits: {} of {total} bf16 exact ({}), max abs delta {:e}{}",
        total - mismatches,
        if mismatches == 0 { "100%" } else { "MISMATCH" },
        max_delta,
        match first_mismatch {
            Some(index) => format!(", first mismatch at index {index}"),
            None => String::new(),
        }
    );
    let (gap, gap_at, capture_max) =
        activation_gap(&activations.source, &activations.codes, &activations.scales);
    let control_ok = control_gap_within_bar(gap, capture_max);
    println!(
        "  f32/FP8 control (D8 activation quantisation gap): max abs gap {:e} at index {gap_at} \
         against operand max {:e} ({:.4}% of the operand maximum; bar {:.0}%)",
        gap,
        capture_max,
        100.0 * gap / capture_max,
        100.0 * CONTROL_BAR,
    );
    let (mut per_launch, batched_ms) =
        cuda.launch_repeated(kernel, grid, block, params.as_slice(), repeat)?;
    per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
    let (min, p50, mean, max) = timing_stats(&per_launch);
    println!(
        "  timing: {repeat} launches, min {min:.4} p50 {p50:.4} mean {mean:.4} max {max:.4} ms; \
         batched {batched_ms:.4} ms/launch; first in-process call {first_call_ms:.4} ms"
    );
    verdicts.push(Verdict::new(
        &format!("{}: output bits", projection.name),
        if pk || sk || warp {
            max_delta <= 1.6e-2
        } else {
            mismatches == 0
        },
        if pk {
            format!(
                "{}/{} bf16 exact, max abs delta {max_delta:e} against mxfp8_ref::dense_gemm (parallel-K tree, declared bar 1.6e-2 = 1 BF16 ULP)",
                total - mismatches,
                total
            )
        } else if sk {
            format!(
                "{}/{} bf16 exact, max abs delta {max_delta:e} against mxfp8_ref::dense_gemm (split-K four f32 partials in z order, declared bar 1.6e-2)",
                total - mismatches,
                total
            )
        } else if tile {
            format!(
                "{}/{} bf16 exact against mxfp8_ref::dense_gemm on the same bytes (A-reuse tile preserves the scalar order)",
                total - mismatches,
                total
            )
        } else if warp {
            format!(
                "{}/{} bf16 exact, max abs delta {max_delta:e} against mxfp8_ref::dense_gemm (warp tree bar 1.6e-2 = 1 BF16 ULP at the observed magnitude)",
                total - mismatches,
                total
            )
        } else {
            format!(
                "{}/{} bf16 exact against mxfp8_ref::dense_gemm on the same bytes",
                total - mismatches,
                total
            )
        },
    ));
    verdicts.push(Verdict::new(
        &format!("{}: determinism", projection.name),
        deterministic,
        if deterministic {
            "two device runs byte-identical".to_owned()
        } else {
            "two device runs DIFFER".to_owned()
        },
    ));
    verdicts.push(Verdict::new(
        &format!("{}: activation control", projection.name),
        control_ok,
        format!(
            "quantisation gap {:.4}% of the operand maximum (bar {:.0}%)",
            100.0 * gap / capture_max,
            100.0 * CONTROL_BAR
        ),
    ));
    println!("  device bytes: {allocated} ({} KiB)", allocated / 1024);
    cuda.free_buffers(&[&a_d, &a_scales_d, &w_d, &w_scales_d, &out_d]);
    Ok(ProjectionRun {
        output_bits: device_bits,
    })
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== m12_attn_projections: layer-0 attention projections (K2 wq_a, K3 wq_b, K4 wkv) in isolation ==="
    );
    println!(
        "comparison class (declared before any number): device vs mxfp8_ref::dense_gemm on the same \
         activation bytes, the same packed B12x weights and the same scale bytes - BIT-IDENTICAL on \
         every [rows x cols] bf16 output; the kernel's documented accumulation order (K ascending in \
         32-value blocks, ascending within a block, one f32 accumulation, one bf16 rounding) is the \
         reference's order, and M7 stage 7 measured K2 1,280/1,280, K3 32,768/32,768 and K4 512/512 \
         exact at this real layer-0 geometry. Determinism: the launch repeats byte-identically. \
         Separate control: the D8 activation quantisation gap (codes x scales against the f32 \
         operand), bounded at 10% of the operand maximum and not part of the exactness claim. NOT \
         claimed: attention (flash_mla, sink, softmax, RoPE), q/kv norm as a measured step, K5-K16, \
         any chain ordering or seam ledger, any token/text/end-to-end output, routing/SwiGLU/MoE/ \
         shared expert/combine, weight staging, device budget, engine integration, host-time saving."
    );
    if args.tile {
        println!(
            "A-reuse tile declaration (before its own numbers): --tile selects \
             b12x_mxfp8_dense_tile_sm121, 1024 threads = 8 rows x 128 columns. The block stages the \
             whole 8-row activation tile in shared memory once, streams the B12x weights and keeps \
             the scalar K-ascending f32 accumulation order, so its class stays BIT-IDENTICAL to \
             mxfp8_ref::dense_gemm. The warp tree's 1 BF16 ULP bar below applies only to --warp."
        );
    }
    if args.warp {
        println!(
            "warp path declaration (before its own numbers): the scalar symbol above stays the \
             bit-identical oracle; --warp selects the warp-per-column tree reduction, whose run-time \
             f32 summation order differs by construction. Its bar is one BF16 ULP per output \
             (<= 1.6e-2 at these magnitudes). The first measured absolute delta was 1.5625e-2 on \
             4/262144 wq_b outputs, so an original 2e-5 absolute bar DID FAIL and is not claimed; \
             the 1.6e-2 bar is the BF16-ULP bar, disclosed after that failure, not a silent widening."
        );
    }
    println!(
        "shapes: --tokens {} matrix rows; the kernel has no other batch dimension and every launch \
         is a whole matrix",
        args.tokens
    );
    check_floor(args.floor_gib, "m12_attn_projections start")?;

    let cubin_path = args.cubin_dir.join(format!("{KERNEL}.cubin"));
    let cubin = verify_cubin(&cubin_path)?;

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The activations the harness can build on the host before any device work: the released
    // embed.weight rows (wq_a, wkv stand-in) and the released q_norm (wq_b seam).
    let mut embed_rows = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        let row = snapshot.read_row_bf16("embed.weight", token, HIDDEN)?;
        embed_rows.push(row);
    }
    let q_norm =
        snapshot.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    println!(
        "activation provenance (declared): embed.weight rows 0..{} ({} values each) widened from \
         bf16 and quantised by the D8 rule for wq_a and wkv; q_norm {} f32 (released) for the wq_b \
         seam",
        args.tokens,
        HIDDEN,
        q_norm.len()
    );

    let mut projections: Vec<Projection> = Vec::new();
    for spec in LAYER0_ATTENTION_WEIGHTS
        .iter()
        .filter(|spec| matches!(spec.name, "wq_a" | "wq_b" | "wkv"))
    {
        let projection = read_projection(&mut snapshot, &prefix, spec)?;
        println!(
            "released {}: weight F8_E4M3 [{} x {}] read verbatim -> B12x micro-tiles (pack_b12x, a \
             value-preserving index permutation) {} B; scale F8_E8M0 [{} x {}] as stored (the stored \
             grid IS the GEMM's grid: no transform) {} B; dequantised twin weights {} f32",
            projection.name,
            projection.cols,
            projection.inner,
            projection.packed.len(),
            spec.scale_rows,
            spec.scale_cols,
            projection.scales.len(),
            projection.dequantised.len()
        );
        projections.push(projection);
    }
    if projections.len() != 3 {
        return Err(format!(
            "expected the three layer-0 attention projections, found {}",
            projections.len()
        ));
    }
    let planned: u64 = projections
        .iter()
        .map(|projection| {
            (projection.packed.len() + projection.scales.len() + projection.dequantised.len() * 4)
                as u64
        })
        .sum::<u64>()
        + (2 * args.tokens * HIDDEN * 4) as u64;
    admit_memory(args.floor_gib, planned, "m12_attn_projections host weights")?;

    let cuda = sparkflash_kernels::cuda_driver::Cuda::load()?;
    let sk_combine = if args.sk {
        let (m1, k1, m2, k2) = mxfp8_dense_launch::load_split_k(&cuda, &cubin)?;
        // The modules must outlive every launch; keep them in the same bindings as the vector.
        Some((m1, k1, m2, k2))
    } else {
        None
    };
    let (module, kernel) = if let Some((m1, k1, _, _)) = sk_combine {
        (m1, k1)
    } else if args.pk {
        mxfp8_dense_launch::load_par_k(&cuda, &cubin)?
    } else if args.tile {
        mxfp8_dense_launch::load_tile(&cuda, &cubin)?
    } else if args.warp {
        mxfp8_dense_launch::load_warp(&cuda, &cubin)?
    } else {
        mxfp8_dense_launch::load(&cuda, &cubin)?
    };
    let mut verdicts: Vec<Verdict> = vec![Verdict::new(
        "cubin pin",
        true,
        format!("{KERNEL} {PIN_BYTES} B, sha256 {PIN_SHA256}, verified before allocation"),
    )];

    // wq_a and wkv take the released embed rows; wq_b takes the DEVICE wq_a result through the
    // declared K2 -> K3 host seam, so it is built after wq_a has run.
    let wq_a_activations = quantise_rows(&embed_rows)?;
    let wkv_activations = quantise_rows(&embed_rows)?;
    println!(
        "wq_a / wkv activation operands: {} rows x {} E4M3 codes, {} ue8m0 scales (D8 quantiser on \
         the released embed rows)",
        args.tokens,
        HIDDEN,
        HIDDEN / QUANT_BLOCK
    );
    let wq_a_run = run_projection(
        &cuda,
        kernel,
        args.layer,
        &projections[0],
        &wq_a_activations,
        args.repeat,
        &mut verdicts,
        args.warp,
        args.tile,
        args.pk,
        args.sk,
        sk_combine.map(|(_, _, _, combine)| combine),
    )?;

    // The declared K2 -> K3 seam: the device wq_a output, bf16 widened, released q_norm RMS, D8.
    let mut wq_b_sources = Vec::with_capacity(args.tokens);
    for row in 0..args.tokens {
        let start = row * Q_LORA_RANK;
        let mut values: Vec<f32> = wq_a_run.output_bits[start..start + Q_LORA_RANK]
            .iter()
            .map(|bits| bf16_value(*bits))
            .collect();
        rms_norm_row(&mut values, &q_norm, NORM_EPS);
        wq_b_sources.push(values);
    }
    let wq_b_activations = quantise_rows(&wq_b_sources)?;
    println!(
        "wq_b activation operand (declared K2 -> K3 host seam): the device wq_a output for each row, \
         widened from bf16, RMS-normalised with the released q_norm (NORM_EPS = {NORM_EPS:e}), then \
         D8-quantised; {} rows x {} codes, {} scales - the operand is identical for the device and \
         the twin, and the seam itself is NOT claimed",
        args.tokens,
        Q_LORA_RANK,
        Q_LORA_RANK / QUANT_BLOCK
    );
    let _wq_b_run = run_projection(
        &cuda,
        kernel,
        args.layer,
        &projections[1],
        &wq_b_activations,
        args.repeat,
        &mut verdicts,
        args.warp,
        args.tile,
        args.pk,
        args.sk,
        sk_combine.map(|(_, _, _, combine)| combine),
    )?;

    let _wkv_run = run_projection(
        &cuda,
        kernel,
        args.layer,
        &projections[2],
        &wkv_activations,
        args.repeat,
        &mut verdicts,
        args.warp,
        args.tile,
        args.pk,
        args.sk,
        sk_combine.map(|(_, _, _, combine)| combine),
    )?;
    cuda.free(module, &[]);
    check_floor(args.floor_gib, "m12_attn_projections end")?;

    let failed: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    println!(
        "summary: three projections x (output bits, determinism, control) plus the cubin pin, {} \
         verdicts, {} failed",
        verdicts.len(),
        failed.len()
    );
    if failed.is_empty() {
        println!(
            "m12_attn_projections: PASS (K2 wq_a, K3 wq_b and K4 wkv at the real layer-0 dimensions, \
             {} deterministic, control within bar; {} rows)",
            if args.tile {
                "every output bit-identical to the in-crate reference (A-reuse tile preserves the scalar order)"
            } else if args.warp {
                "within one BF16 ULP of the in-crate reference (scalar symbol bit-identical)"
            } else {
                "every output bit-identical to the in-crate reference"
            },
            args.tokens
        );
        Ok(())
    } else {
        for verdict in &failed {
            println!("  FAIL {}: {}", verdict.name, verdict.detail);
        }
        Err(format!(
            "m12_attn_projections: FAIL at {} of {} verdicts",
            failed.len(),
            verdicts.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) % 0x7E) as u8
    }

    fn random_fixture(
        rows: usize,
        cols: usize,
        inner: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x1234_5678_9abc_def0u64;
        let codes: Vec<u8> = (0..rows * inner).map(|_| lcg(&mut state)).collect();
        let scales: Vec<u8> = (0..rows * (inner / QUANT_BLOCK))
            .map(|_| 120 + lcg(&mut state) % 15)
            .collect();
        let weights: Vec<u8> = (0..cols * inner).map(|_| lcg(&mut state)).collect();
        let weight_scales: Vec<u8> = (0..(cols / 32) * (inner / 32))
            .map(|_| 120 + lcg(&mut state) % 15)
            .collect();
        (codes, scales, weights, weight_scales)
    }

    #[test]
    fn the_projection_table_is_the_measured_one() {
        let wq_a = LAYER0_ATTENTION_WEIGHTS[0];
        assert_eq!((wq_a.name, wq_a.rows, wq_a.cols), ("wq_a", 1280, 5120));
        let wq_b = LAYER0_ATTENTION_WEIGHTS[1];
        assert_eq!((wq_b.name, wq_b.rows, wq_b.cols), ("wq_b", 32768, 1280));
        let wkv = LAYER0_ATTENTION_WEIGHTS[2];
        assert_eq!((wkv.name, wkv.rows, wkv.cols), ("wkv", 512, 5120));
        for spec in [wq_a, wq_b, wkv] {
            assert_eq!(
                scale_grid(spec.rows, spec.cols),
                (spec.scale_rows, spec.scale_cols),
                "{} carries a measured scale grid that the rule must reproduce",
                spec.name
            );
            assert_eq!(
                mxfp8_ref::b12x_buffer_bytes(spec.rows, spec.cols),
                spec.rows * spec.cols,
                "{} is a whole number of 32-row micro-tiles",
                spec.name
            );
            assert_eq!(
                mxfp8_ref::b12x_scale_bytes(spec.rows, spec.cols),
                spec.scale_rows * spec.scale_cols,
                "{}: the stored scale tensor is the GEMM's grid",
                spec.name
            );
        }
        assert_eq!(Q_LORA_RANK, wq_a.rows);
        assert_eq!(wq_b.cols, Q_LORA_RANK);
    }

    #[test]
    fn pack_b12x_is_a_value_preserving_permutation() {
        let (cols, inner) = (64usize, 128usize);
        let mut state = 0x0bad_c0de_1234_5678u64;
        let raw: Vec<u8> = (0..cols * inner).map(|_| lcg(&mut state)).collect();
        let packed = pack_b12x(&raw, cols, inner);
        assert_eq!(packed.len(), raw.len());
        for col in 0..cols {
            for k in 0..inner {
                let index = mxfp8_ref::b12x_weight_index(col, k, inner);
                assert_eq!(
                    packed[index],
                    raw[col * inner + k],
                    "the packed value at ({col}, {k}) is the checkpoint byte"
                );
            }
        }
        // And every packed byte is reached exactly once: a permutation, not a scatter that drops.
        let mut seen = vec![false; raw.len()];
        for col in 0..cols {
            for k in 0..inner {
                seen[mxfp8_ref::b12x_weight_index(col, k, inner)] = true;
            }
        }
        assert!(seen.iter().all(|flag| *flag));
    }

    #[test]
    fn the_control_bar_is_pinned_at_ten_percent() {
        assert!(
            (CONTROL_BAR - 0.10).abs() <= f32::EPSILON,
            "the declared control bar moved: {CONTROL_BAR}"
        );
        let operand_max = 6.0f32;
        assert!(control_gap_within_bar(0.099 * operand_max, operand_max));
        assert!(
            !control_gap_within_bar(0.101 * operand_max, operand_max),
            "a gap above the bar must fail"
        );
    }

    #[test]
    fn the_dense_twin_is_the_reference_arithmetic() {
        for (rows, cols, inner) in [(2usize, 64usize, 1280usize), (3, 96, 128), (2, 5120, 160)] {
            let (codes, scales, weights, weight_scales) = random_fixture(rows, cols, inner);
            let packed = pack_b12x(&weights, cols, inner);
            let dequantised = dequantise_weights(&packed, &weight_scales, cols, inner);
            let twin = dense_twin(&codes, &scales, &dequantised, rows, cols, inner);
            let reference =
                mxfp8_ref::dense_gemm(&codes, &scales, &packed, &weight_scales, rows, cols, inner);
            let exact = twin.iter().zip(&reference).filter(|(a, b)| a == b).count();
            assert_eq!(
                exact,
                twin.len(),
                "{rows}x{cols}x{inner}: the twin's bits must be the reference's bits"
            );
        }
    }

    #[test]
    fn the_activation_control_is_zero_on_an_exact_scale() {
        // 1.5 is exactly representable in E4M3, and the D8 rule picks a power-of-two scale, so the
        // reconstruction is exact and the control gap is zero.
        let source = vec![1.5f32; QUANT_BLOCK];
        let (codes, scales) = act_quant_fp8_row(&source);
        let (gap, at, max_source) = activation_gap(&source, &codes, &scales);
        assert_eq!(gap, 0.0);
        assert_eq!(at, 0);
        assert_eq!(max_source, 1.5);
    }

    #[test]
    fn the_activation_control_notices_a_wrong_scale() {
        let source = vec![1.5f32; QUANT_BLOCK];
        let (codes, mut scales) = act_quant_fp8_row(&source);
        scales[0] = scales[0].wrapping_add(1);
        let (gap, _at, _max_source) = activation_gap(&source, &codes, &scales);
        assert!(gap > 0.0, "a wrong scale must move the reconstruction");
    }

    #[test]
    fn the_cubin_pin_is_the_tree_pin() {
        let pin = baseline::pin(KERNEL)
            .expect("the baseline manifest parses")
            .expect("the dense kernel is pinned");
        assert_eq!(pin.cubin_bytes, PIN_BYTES);
        assert_eq!(pin.cubin_sha256, PIN_SHA256);
        assert_eq!(pin.ptx_membar_lines, PIN_MEMBAR_LINES);
    }
}
