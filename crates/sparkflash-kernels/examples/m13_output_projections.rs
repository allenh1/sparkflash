//! m13_output_projections: the output-projection pair of layer 0 - wo_a (grouped MXFP8) and
//! wo_b (dense MXFP8) - in isolation, against released checkpoint bytes, in one harness.
//!
//! Scope, stated before the numbers. This harness launches TWO chain stages of the attention
//! sublayer's output path, in their chain order:
//!   K6 wo_a: kernels/b12x_mxfp8_wo_a_sm121.cu, 8 groups of 1024 output channels over 4096
//!            reduction values each (8 heads x 512-dimensional head output per group), one
//!            position per launch, grid (8, 4, 1), 256 threads;
//!   K7 wo_b: kernels/b12x_mxfp8_dense_sm121.cu, N=5120, K=8192, rows = positions, one whole-matrix
//!            launch.
//! It is not attention: flash_mla_csa2 (K5, the window/sink/FLSA stage) is the NEXT stage and is
//! not launched here; the attention output that wo_a normally consumes is a declared host-built
//! stand-in (below). It is not a chain pass: the K6 -> K7 host seam (widen the lo rank from bf16,
//! D8-quantise it, exactly as M7 stage 7 does it) is declared and not claimed, and neither is the
//! residual add/hc_post that follows this pair. It is not a token, not a sequence and not end-to-end
//! text: the positions are independent rows. Explicitly NOT claimed:
//!   * attention, the sink, softmax, RoPE, the window-KV stage, the HSI candidate merge, the
//!     attention-output inverse RoPE;
//!   * the residual add / hc_post / any chain ordering, seam ledger or per-layer schedule;
//!   * any token, any text, any end-to-end layer or model output;
//!   * routing, SwiGLU, the MoE block, the shared expert, the combine;
//!   * weight staging, device-budget admission, the engine, or any host-time saving;
//!   * any per-decode-token timing: the measured launch times are whole-stage measurements at the
//!     stated row counts, and no per-token cost is derived from them.
//!
//! Provenance and permutation, declared. Every weight is read verbatim from the released checkpoint:
//!   K6 wo_a: layers.0.attn.wo_a.weight F8_E4M3 [8192, 4096] and .scale F8_E8M0 [256, 128];
//!   K7 wo_b: layers.0.attn.wo_b.weight F8_E4M3 [5120, 8192] and .scale F8_E8M0 [160, 256].
//! The shapes are the headers measured in sparkflash_kernels::m7_layer0_shapes, not assumptions.
//! Layouts:
//!   * wo_b (dense): the checkpoint stores row-major [out, in] E4M3; the kernel walks B12x 32x32
//!     micro-tiles, so the host applies pack_b12x, a declared value-preserving index permutation
//!     (no arithmetic, no decode, one byte per value; a unit test proves the bijection). The stored
//!     .scale grid [out/32, in/32] is exactly the dense GEMM's scale grid, so it is uploaded as
//!     stored with no transform, and the harness asserts the byte counts equal the GEMM's rule.
//!   * wo_a (grouped): the checkpoint stores [8192, 4096] row-major, which is 8 groups x 1024 output
//!     channels x 4096 reduction width. The kernel takes EIGHT separate B12x buffers of
//!     [1024, 4096], so the host slices the stored rows by group and applies pack_b12x per group -
//!     a declared, value-preserving per-group repack. The stored .scale [256, 128] splits into 8
//!     contiguous per-group [32, 128] grids, which is the grouped GEMM's per-group scale grid, so
//!     it is uploaded as stored. The harness asserts both the per-group and the whole-tensor byte
//!     counts against the B12x rules, so a wrong group stride fails closed.
//!
//! Activation operands are host-built and declared, because the checkpoint stores no activation for
//! either stage:
//!   * wo_a: the released K4 wkv latent of position t (computed by the CPU twin from the released
//!     wkv weight + scales on that position's D8-quantised embed row), broadcast across all 64
//!     heads' 512-wide slices: for each group, its 4096-wide slice is the 512-wide latent repeated
//!     8 times. That is the single-candidate attention output up to the sink-weighted scalar this
//!     harness deliberately does not model (no attention, sink or softmax is claimed). The wkv
//!     bytes and the D8 quantiser are real; the broadcast is declared.
//!   * wo_b: the DEVICE wo_a lo_out of position t (bf16), widened to f32 and D8-quantised - the
//!     chain's K6 -> K7 host seam as M7 stage 7 performs it. The operand is built once and is
//!     identical for the device and the CPU twin, and the seam itself is NOT claimed.
//!
//! Cubins. Two pins, both verified by length and sha256 from baseline/kernel_baseline.txt through
//! baseline::pin BEFORE the first device allocation, and cross-checked against this file's constants:
//!   b12x_mxfp8_wo_a_sm121  9,056 B  sha256 84833781a77d094945d9f8e33168ff75cae4d9bf81b2a24b531b404984160eed  0 membar lines
//!   b12x_mxfp8_dense_sm121 11,824 B sha256 03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a  0 membar lines
//! Build both under the pinned toolkit (CUDA 13.0.3, nvcc V13.0.88):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_wo_a_sm121.cu -o out/b12x_mxfp8_wo_a_sm121.cubin
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_mxfp8_dense_sm121.cu -o out/b12x_mxfp8_dense_sm121.cubin
//!
//! COMPARISON CLASS (declared before any number is printed). Every output element of both stages, for
//! every position, is compared to the in-crate reference on the SAME activation bytes, the SAME
//! packed weights and the SAME scale bytes:
//!   bit-identical:  all K6 outputs (positions x 8,192 bf16) against mxfp8_ref::grouped_wo_a, and
//!                   all K7 outputs (positions x 5,120 bf16) against mxfp8_ref::dense_gemm. This is
//!                   a legitimate exactness claim, not a bar: both kernels' documented scalar
//!                   accumulation order is the reference's contract (K ascending in 32-value blocks,
//!                   ascending within a block, one f32 accumulation, one bf16 rounding), the
//!                   grouped kernel is explicitly the per-group dense GEMM, and M7 stage 7 measured
//!                   K6 8,192/8,192 and K7 5,120/5,120 exact at this same real geometry.
//!   determinism:    every launch is repeated on the same operands and all output bytes must match.
//!   f32/FP8 control: each stage's activation quantiser gap - the D8 codes x ue8m0 scales, decoded,
//!                   against the f32 operand - reported per stage and bounded at 10% of that
//!                   operand's maximum absolute value (the m10/m11/m12 control style). It measures
//!                   the activation-side FP8 cost and is NOT part of the exactness claim.
//! Tokens: --tokens N positions. wo_a is a single-position kernel (no rows argument), so it launches
//! once per position; wo_b takes rows = N and launches once. Both row counts 8 and 130 are valid and
//! 130 is the fixture prefill length. The two stages are measured at the same row counts, and the
//! per-launch times are reported as measured without any per-token derivation.
//!
//! The next chain stage after this pair is attention (K5 flash_mla_csa2: window/sink/FLSA), which is
//! m14 and is NOT part of this harness.
//!
//! Usage (aarch64 node, checkpoint present):
//!   m13_output_projections --checkpoint <snapshot dir> --cubin-dir out [--tokens 8]
//!                          [--repeat 100] [--floor-gib 32]

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
use sparkflash_kernels::mla_ref::{MLA_HEADS, MLA_LATENT};
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref;
use sparkflash_kernels::mxfp8_wo_a_launch::{self, Mxfp8WoAParams};

/// The dense cubin name in the baseline manifest.
const DENSE_KERNEL: &str = "b12x_mxfp8_dense_sm121";
/// The grouped wo_a cubin name in the baseline manifest.
const WO_A_KERNEL: &str = "b12x_mxfp8_wo_a_sm121";
/// Pinned dense cubin size.
const DENSE_PIN_BYTES: usize = 11_824;
/// Pinned dense cubin sha256.
const DENSE_PIN_SHA256: &str = "03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a";
/// Pinned wi_a cubin size.
const WO_A_PIN_BYTES: usize = 9_056;
/// Pinned wo_a cubin sha256.
const WO_A_PIN_SHA256: &str = "84833781a77d094945d9f8e33168ff75cae4d9bf81b2a24b531b404984160eed";
/// Both pinned kernels carry no fence.
const PIN_MEMBAR_LINES: u32 = 0;
/// The grouped output projection's geometry, from the released kernel and the measured layer-0 table.
const O_GROUPS: usize = 8;
const O_LORA_RANK: usize = 1_024;
const GROUP_INNER: usize = 4_096;
/// The grouped output width: 8 groups x 1024 channels.
const WO_A_OUT: usize = O_GROUPS * O_LORA_RANK;
/// The attention-output width: 64 heads x 512 latent.
const ATTENTION_WIDTH: usize = MLA_HEADS * MLA_LATENT;
/// The declared control bar: quantisation gap as a fraction of the operand maximum.
const CONTROL_BAR: f32 = 0.10;

/// The control's declared bound, factored so the boundary itself is tested: a gap passes when it is
/// at most this fraction of the operand's maximum absolute value.
fn control_gap_within_bar(gap: f32, operand_max: f32) -> bool {
    gap <= CONTROL_BAR * operand_max
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m13_output_projections: {error}");
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
                "--help" | "-h" => {
                    println!(
                        "usage: m13_output_projections --checkpoint <snapshot dir> [--cubin-dir out] \
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
                "--layer must be 0: the activation stand-ins are the layer-0 boundary inputs and \
                 this harness is the layer-0 stage, not a per-layer tool"
                    .to_owned(),
            );
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// Repack a checkpoint FP8 weight from row-major [cols, inner] into B12x micro-tiles.
fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(weights.len(), cols * inner, "one E4M3 byte per FP8 value");
    let mut tiled = vec![0u8; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            tiled[mxfp8_ref::b12x_weight_index(col, k, inner)] = weights[col * inner + k];
        }
    }
    tiled
}

/// The per-group repack of wo_a: the stored [8*1024, 4096] rows, each group's 1024-row slice packed
/// into its own B12x buffer and concatenated in group order, exactly the layout the kernel walks.
/// The stored scale tensor is already the concatenation of the per-group [32, 128] grids.
fn pack_wo_a(weight: &[u8], scales: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    pack_wo_a_with(O_GROUPS, O_LORA_RANK, GROUP_INNER, weight, scales)
}

/// The same repack with the geometry as arguments, so the test can exercise the stride and the
/// per-group permutation on a small stand-in instead of re-deriving the loop it is meant to check.
fn pack_wo_a_with(
    groups: usize,
    lo_rank: usize,
    group_inner: usize,
    weight: &[u8],
    scales: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if weight.len() != groups * lo_rank * group_inner {
        return Err(format!(
            "wo_a weight: {} bytes, expected {}",
            weight.len(),
            groups * lo_rank * group_inner
        ));
    }
    let group_scale_bytes = (lo_rank / 32) * (group_inner / 32);
    if scales.len() != groups * group_scale_bytes {
        return Err(format!(
            "wo_a scales: {} bytes, expected {}",
            scales.len(),
            groups * group_scale_bytes
        ));
    }
    let mut packed = Vec::with_capacity(weight.len());
    for group in 0..groups {
        let start = group * lo_rank * group_inner;
        packed.extend_from_slice(&pack_b12x(
            &weight[start..start + lo_rank * group_inner],
            lo_rank,
            group_inner,
        ));
    }
    Ok((packed, scales.to_vec()))
}

/// Decode every dense weight value once into f32: fp8 decode times its ue8m0 scale - the same two
/// f32 operations the kernel performs per element. Hoisting them out of the row loop is the twin's
/// only departure from a literal mxfp8_ref call and is bit-identical (the hoisted values are the
/// f32 values the reference recomputes per element; the unit tests prove it on real inner lengths).
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

/// Decode wo_a's per-group weights: eight [1024, 4096] f32 matrices, group order.
fn dequantise_group_weights(
    packed: &[u8],
    scales: &[u8],
    groups: usize,
    lo_rank: usize,
    group_inner: usize,
) -> Vec<Vec<f32>> {
    let weight_bytes = mxfp8_ref::b12x_buffer_bytes(lo_rank, group_inner);
    let scale_bytes = mxfp8_ref::b12x_scale_bytes(lo_rank, group_inner);
    (0..groups)
        .map(|group| {
            dequantise_weights(
                &packed[group * weight_bytes..(group + 1) * weight_bytes],
                &scales[group * scale_bytes..(group + 1) * scale_bytes],
                lo_rank,
                group_inner,
            )
        })
        .collect()
}

/// The dense CPU twin: mxfp8_ref::dense_gemm's arithmetic with the weight dequantisation hoisted.
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

/// The grouped CPU twin for one position: each group is the dense GEMM over that group's 4096-wide
/// attention slice, accumulated in the kernel's order, then the eight 1024-wide results are
/// concatenated - exactly mxfp8_ref::grouped_wo_a's arithmetic with the weights hoisted.
fn grouped_twin(
    attn_codes: &[u8],
    attn_scales: &[u8],
    group_weights: &[Vec<f32>],
    groups: usize,
    group_inner: usize,
    lo_rank: usize,
) -> Vec<u16> {
    let k_blocks = group_inner / QUANT_BLOCK;
    assert_eq!(attn_codes.len(), groups * group_inner);
    assert_eq!(attn_scales.len(), groups * k_blocks);
    assert_eq!(group_weights.len(), groups);
    let mut out = vec![0u16; groups * lo_rank];
    let mut activation = vec![0.0f32; group_inner];
    for group in 0..groups {
        for block in 0..k_blocks {
            let scale = ue8m0_decode(attn_scales[group * k_blocks + block]);
            let base = block * QUANT_BLOCK;
            for i in 0..QUANT_BLOCK {
                activation[base + i] =
                    fp8_e4m3_decode(attn_codes[group * group_inner + base + i]) * scale;
            }
        }
        let weights = &group_weights[group];
        for col in 0..lo_rank {
            let weight_row = &weights[col * group_inner..(col + 1) * group_inner];
            let mut acc = 0.0f32;
            for k in 0..group_inner {
                acc += activation[k] * weight_row[k];
            }
            out[group * lo_rank + col] = bf16_bits(acc);
        }
    }
    out
}

/// The D8 quantiser's own gap: decoded codes x scales against the f32 value they quantised.
///
/// Returns (max abs gap, index, max abs source). The f32/FP8 control, not an exactness claim.
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

/// One stage's operand matrix: codes, scales and the f32 source values.
struct Activations {
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
        codes,
        scales,
        source,
    })
}

/// Reads a block-scaled tensor pair verbatim, returning (weight bytes, scale bytes).
fn read_weight_and_scales(
    snapshot: &mut Snapshot,
    prefix: &str,
    spec: &BlockScaledWeight,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let out = spec.rows;
    let inner = spec.cols;
    let weight = snapshot.read_raw(
        &format!("{prefix}.attn.{}.weight", spec.name),
        "F8_E4M3",
        &[out, inner],
    )?;
    let (scale_rows, scale_cols) = scale_grid(out, inner);
    if (scale_rows, scale_cols) != (spec.scale_rows, spec.scale_cols) {
        return Err(format!(
            "{}: the released scale grid ({scale_rows}, {scale_cols}) disagrees with the measured \
             table's ({}, {})",
            spec.name, spec.scale_rows, spec.scale_cols
        ));
    }
    let scales = snapshot.read_raw(
        &format!("{prefix}.attn.{}.scale", spec.name),
        "F8_E8M0",
        &[scale_rows, scale_cols],
    )?;
    if weight.len() != out * inner || scales.len() != scale_rows * scale_cols {
        return Err(format!(
            "{}: the released tensor sizes disagree with the measured table",
            spec.name
        ));
    }
    Ok((weight, scales))
}

/// Reads the cubin and refuses anything but the pinned compiled form, before any allocation.
fn verify_cubin(
    path: &Path,
    kernel: &str,
    pin_bytes: usize,
    pin_sha256: &str,
) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            path.display()
        )
    })?;
    let pin = baseline::pin(kernel)?
        .ok_or_else(|| format!("{kernel} is not in the baseline manifest"))?;
    if bytes.len() != pin_bytes || pin.cubin_bytes != pin_bytes {
        return Err(format!(
            "{}: cubin is {} B but the pin is {pin_bytes} B",
            path.display(),
            bytes.len()
        ));
    }
    if pin.cubin_sha256 != pin_sha256 || pin.ptx_membar_lines != PIN_MEMBAR_LINES {
        return Err(format!(
            "the baseline manifest's {kernel} pin changed: {} B, sha256 {}, {} membar lines",
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
    if sha != pin_sha256 {
        return Err(format!(
            "{}: cubin sha256 {sha} is not the pinned {pin_sha256}",
            path.display()
        ));
    }
    println!(
        "cubin: {kernel} {} B, sha256 {sha} (pinned: {pin_bytes} B, {PIN_MEMBAR_LINES} PTX membar \
         lines) - verified before any device allocation",
        bytes.len()
    );
    Ok(bytes)
}

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

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== m13_output_projections: layer-0 output projections (K6 wo_a grouped, K7 wo_b dense) in isolation ==="
    );
    println!(
        "comparison class (declared before any number): device vs the in-crate references on the \
         same activation bytes, packed B12x weights and scale bytes - BIT-IDENTICAL on every K6 \
         output (rows x 8,192 bf16 vs mxfp8_ref::grouped_wo_a) and every K7 output (rows x 5,120 \
         bf16 vs mxfp8_ref::dense_gemm); the kernels' scalar accumulation order is the references' \
         contract (K ascending in 32-value blocks, ascending within a block, one f32 accumulation, \
         one bf16 rounding) and M7 stage 7 measured K6 8,192/8,192 and K7 5,120/5,120 exact at this \
         real geometry. Determinism: every launch repeats byte-identically. Separate controls: each \
         stage's D8 activation quantisation gap (codes x scales against the f32 operand), bounded at \
         10% of the operand maximum and not part of the exactness claim. NOT claimed: attention / \
         sink / softmax / RoPE / window-KV (K5 is the NEXT stage, m14), the residual add/hc_post, any \
         chain order or seam ledger, any token/text/end-to-end output, routing/SwiGLU/MoE/shared \
         expert/combine, weight staging, device budget, engine integration, host-time saving, any \
         per-decode-token timing derived from these numbers."
    );
    println!(
        "tokens: {} positions. K6 wo_a is a single-position kernel (one launch per position); K7 wo_b \
         takes rows = positions (one launch). Both row counts are measured at the same N.",
        args.tokens
    );
    check_floor(args.floor_gib, "m13_output_projections start")?;

    let wo_a_cubin_path = args.cubin_dir.join(format!("{WO_A_KERNEL}.cubin"));
    let dense_cubin_path = args.cubin_dir.join(format!("{DENSE_KERNEL}.cubin"));
    let wo_a_cubin = verify_cubin(
        &wo_a_cubin_path,
        WO_A_KERNEL,
        WO_A_PIN_BYTES,
        WO_A_PIN_SHA256,
    )?;
    let dense_cubin = verify_cubin(
        &dense_cubin_path,
        DENSE_KERNEL,
        DENSE_PIN_BYTES,
        DENSE_PIN_SHA256,
    )?;
    if mxfp8_wo_a_launch::KERNEL != WO_A_KERNEL || mxfp8_dense_launch::KERNEL != DENSE_KERNEL {
        return Err("a typed builder resolves a different kernel symbol".to_owned());
    }

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The measured layer-0 table: the shapes are headers, not assumptions.
    let wo_a_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wo_a")
        .ok_or_else(|| "wo_a is not in the measured layer-0 table".to_owned())?;
    let wo_b_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wo_b")
        .ok_or_else(|| "wo_b is not in the measured layer-0 table".to_owned())?;
    let wkv_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wkv")
        .ok_or_else(|| "wkv is not in the measured layer-0 table".to_owned())?;
    if (wo_a_spec.rows, wo_a_spec.cols) != (WO_A_OUT, GROUP_INNER)
        || (wo_b_spec.rows, wo_b_spec.cols) != (5_120, WO_A_OUT)
        || (wkv_spec.rows, wkv_spec.cols) != (MLA_LATENT, 5_120)
        || ATTENTION_WIDTH != O_GROUPS * GROUP_INNER
        || MLA_LATENT * 8 != GROUP_INNER
    {
        return Err(
            "the measured layer-0 geometry disagrees with this harness's constants".to_owned(),
        );
    }

    // wo_b's dense operand: verbatim released bytes, B12x-packed by the host, scales as stored.
    let (wo_b_raw, wo_b_scales) = read_weight_and_scales(&mut snapshot, &prefix, &wo_b_spec)?;
    let wo_b_packed = pack_b12x(&wo_b_raw, wo_b_spec.rows, wo_b_spec.cols);
    if wo_b_packed.len() != mxfp8_ref::b12x_buffer_bytes(wo_b_spec.rows, wo_b_spec.cols)
        || wo_b_scales.len() != mxfp8_ref::b12x_scale_bytes(wo_b_spec.rows, wo_b_spec.cols)
    {
        return Err("wo_b: the packed buffers disagree with the B12x rules".to_owned());
    }
    let wo_b_weights =
        dequantise_weights(&wo_b_packed, &wo_b_scales, wo_b_spec.rows, wo_b_spec.cols);
    println!(
        "released wo_b: F8_E4M3 [{} x {}] read verbatim -> B12x {} B (pack_b12x, value-preserving \
         permutation); scale F8_E8M0 [{} x {}] as stored (the GEMM's own grid) {} B; dequantised \
         twin weights {} f32",
        wo_b_spec.rows,
        wo_b_spec.cols,
        wo_b_packed.len(),
        wo_b_spec.scale_rows,
        wo_b_spec.scale_cols,
        wo_b_scales.len(),
        wo_b_weights.len()
    );

    // wo_a's grouped operand: verbatim bytes, per-group B12x repack, scales as stored.
    let (wo_a_raw, wo_a_scales) = read_weight_and_scales(&mut snapshot, &prefix, &wo_a_spec)?;
    let (wo_a_packed, wo_a_scales) = pack_wo_a(&wo_a_raw, &wo_a_scales)?;
    let wo_a_weights = dequantise_group_weights(
        &wo_a_packed,
        &wo_a_scales,
        O_GROUPS,
        O_LORA_RANK,
        GROUP_INNER,
    );
    println!(
        "released wo_a: F8_E4M3 [{} x {}] read verbatim -> 8 per-group B12x buffers of [{} x {}] \
         ({} B total, declared per-group repack); scale F8_E8M0 [{} x {}] as stored = 8 contiguous \
         per-group [32 x 128] grids ({} B); dequantised per-group twin weights {} x {} f32",
        wo_a_spec.rows,
        wo_a_spec.cols,
        O_LORA_RANK,
        GROUP_INNER,
        wo_a_packed.len(),
        wo_a_spec.scale_rows,
        wo_a_spec.scale_cols,
        wo_a_scales.len(),
        O_GROUPS,
        O_LORA_RANK * GROUP_INNER
    );

    // The K4 wkv latent that stands in for the attention output: real released bytes, CPU twin.
    let (wkv_raw, wkv_scales) = read_weight_and_scales(&mut snapshot, &prefix, &wkv_spec)?;
    let wkv_packed = pack_b12x(&wkv_raw, wkv_spec.rows, wkv_spec.cols);
    let wkv_weights = dequantise_weights(&wkv_packed, &wkv_scales, wkv_spec.rows, wkv_spec.cols);
    println!(
        "released wkv (used ONLY to build the declared attention-output stand-in): F8_E4M3 [{} x {}] \
         + scale [{} x {}]",
        wkv_spec.rows, wkv_spec.cols, wkv_spec.scale_rows, wkv_spec.scale_cols
    );

    let mut embed_rows = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        embed_rows.push(snapshot.read_row_bf16("embed.weight", token, 5_120)?);
    }
    let embed_activations = quantise_rows(&embed_rows)?;
    let kv_latents = dense_twin(
        &embed_activations.codes,
        &embed_activations.scales,
        &wkv_weights,
        args.tokens,
        wkv_spec.rows,
        wkv_spec.cols,
    );
    println!(
        "declared attention-output stand-in: the released K4 wkv latent of each position (CPU twin on \
         the D8-quantised embed row), broadcast across the 64 heads' 512-wide slices (each group's \
         4096-wide slice = the latent repeated 8 times); this is the single-candidate attention \
         output up to the sink-weighted scalar, and no attention/sink/softmax is claimed",
    );

    let mut attention_operands: Vec<Vec<f32>> = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        let latent_start = token * MLA_LATENT;
        let latent = &kv_latents[latent_start..latent_start + MLA_LATENT];
        let mut operand = Vec::with_capacity(ATTENTION_WIDTH);
        for _head in 0..MLA_HEADS {
            operand.extend(latent.iter().map(|bits| bf16_value(*bits)));
        }
        attention_operands.push(operand);
    }
    let wo_a_activations = quantise_rows(&attention_operands)?;
    println!(
        "wo_a activation operands: {} positions x {} E4M3 codes, {} ue8m0 scales (D8 quantiser)",
        args.tokens,
        ATTENTION_WIDTH,
        ATTENTION_WIDTH / QUANT_BLOCK
    );

    let planned: u64 = (wo_b_packed.len()
        + wo_b_scales.len()
        + wo_b_weights.len() * 4
        + wo_a_packed.len()
        + wo_a_scales.len()
        + wo_a_weights
            .iter()
            .map(|weights| weights.len() * 4)
            .sum::<usize>()
        + wkv_packed.len()
        + wkv_weights.len() * 4
        + wo_a_activations.codes.len()
        + wo_a_activations.scales.len()
        + wo_a_activations.source.len() * 4) as u64;
    admit_memory(
        args.floor_gib,
        planned,
        "m13_output_projections host weights",
    )?;

    let cuda = sparkflash_kernels::cuda_driver::Cuda::load()?;
    let (wo_a_module, wo_a_kernel) = mxfp8_wo_a_launch::load(&cuda, &wo_a_cubin)?;
    let (dense_module, dense_kernel) = mxfp8_dense_launch::load(&cuda, &dense_cubin)?;
    let mut verdicts: Vec<Verdict> = vec![
        Verdict::new(
            "cubin pin wo_a",
            true,
            format!("{WO_A_KERNEL} {WO_A_PIN_BYTES} B sha256 {WO_A_PIN_SHA256}"),
        ),
        Verdict::new(
            "cubin pin dense",
            true,
            format!("{DENSE_KERNEL} {DENSE_PIN_BYTES} B sha256 {DENSE_PIN_SHA256}"),
        ),
    ];

    // --- K6 wo_a: one launch per position, compared per position --------------------------------
    let mut allocated = 0usize;
    let attn_d = cuda.alloc(ATTENTION_WIDTH, &mut allocated)?;
    let attn_scales_d = cuda.alloc(ATTENTION_WIDTH / QUANT_BLOCK, &mut allocated)?;
    let wo_a_w_d = cuda.alloc(wo_a_packed.len(), &mut allocated)?;
    let wo_a_s_d = cuda.alloc(wo_a_scales.len(), &mut allocated)?;
    let lo_out_d = cuda.alloc(WO_A_OUT * 2, &mut allocated)?;
    cuda.copy_htod(&wo_a_w_d, &wo_a_packed)?;
    cuda.copy_htod(&wo_a_s_d, &wo_a_scales)?;
    let wo_a_params = Mxfp8WoAParams::new(
        attn_d,
        attn_scales_d,
        wo_a_w_d,
        wo_a_s_d,
        lo_out_d,
        O_GROUPS as u32,
        0, // trace_ring
        args.layer as u32,
        0, // step_seq
    );
    let wo_a_grid = mxfp8_wo_a_launch::grid(O_GROUPS as u32)?;
    println!(
        "K6 wo_a launch shape: one position, grid {wo_a_grid:?}, block {:?}, {} argument words \
         through Mxfp8WoAParams",
        mxfp8_wo_a_launch::BLOCK,
        wo_a_params.as_slice().len()
    );
    let mut lo_out_bits: Vec<u16> = Vec::with_capacity(args.tokens * WO_A_OUT);
    let mut wo_a_exact = 0usize;
    let mut wo_a_total = 0usize;
    let mut wo_a_mismatches = 0usize;
    let mut wo_a_first_mismatch: Option<String> = None;
    let mut wo_a_deterministic = true;
    let mut wo_a_control_max = 0.0f32;
    let mut wo_a_control_max_source = 0.0f32;
    for token in 0..args.tokens {
        let codes = &wo_a_activations.codes[token * ATTENTION_WIDTH..(token + 1) * ATTENTION_WIDTH];
        let scales = &wo_a_activations.scales[token * (ATTENTION_WIDTH / QUANT_BLOCK)
            ..(token + 1) * (ATTENTION_WIDTH / QUANT_BLOCK)];
        let source =
            &wo_a_activations.source[token * ATTENTION_WIDTH..(token + 1) * ATTENTION_WIDTH];
        let expected = grouped_twin(
            codes,
            scales,
            &wo_a_weights,
            O_GROUPS,
            GROUP_INNER,
            O_LORA_RANK,
        );
        cuda.copy_htod(&attn_d, codes)?;
        cuda.copy_htod(&attn_scales_d, scales)?;
        mxfp8_wo_a_launch::launch(&cuda, wo_a_kernel, &wo_a_params)?;
        let mut device_bytes = vec![0u8; WO_A_OUT * 2];
        cuda.copy_dtoh(&mut device_bytes, &lo_out_d)?;
        mxfp8_wo_a_launch::launch(&cuda, wo_a_kernel, &wo_a_params)?;
        let mut again_bytes = vec![0u8; WO_A_OUT * 2];
        cuda.copy_dtoh(&mut again_bytes, &lo_out_d)?;
        if device_bytes != again_bytes {
            wo_a_deterministic = false;
        }
        let device_bits: Vec<u16> = device_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        for (index, (device, reference)) in device_bits.iter().zip(&expected).enumerate() {
            wo_a_total += 1;
            if device == reference {
                wo_a_exact += 1;
            } else {
                wo_a_mismatches += 1;
                if wo_a_first_mismatch.is_none() {
                    wo_a_first_mismatch = Some(format!("position {token} index {index}"));
                }
            }
        }
        let (gap, _at, max_source) = activation_gap(source, codes, scales);
        wo_a_control_max = wo_a_control_max.max(gap);
        wo_a_control_max_source = wo_a_control_max_source.max(max_source);
        lo_out_bits.extend_from_slice(&device_bits);
    }
    println!(
        "K6 wo_a output bits: {wo_a_exact} of {wo_a_total} bf16 exact ({}), determinism {}",
        if wo_a_mismatches == 0 {
            "100%"
        } else {
            "MISMATCH"
        },
        if wo_a_deterministic {
            "byte-identical"
        } else {
            "DIFFERS"
        }
    );
    let wo_a_control_ok = control_gap_within_bar(wo_a_control_max, wo_a_control_max_source);
    println!(
        "K6 wo_a f32/FP8 control: worst D8 activation gap {:e} against operand max {:e} ({:.4}%; bar \
         {:.0}%)",
        wo_a_control_max,
        wo_a_control_max_source,
        100.0 * wo_a_control_max / wo_a_control_max_source,
        100.0 * CONTROL_BAR,
    );
    let (mut wo_a_samples, wo_a_batched) = cuda.launch_repeated(
        wo_a_kernel,
        wo_a_grid,
        mxfp8_wo_a_launch::BLOCK,
        wo_a_params.as_slice(),
        args.repeat,
    )?;
    wo_a_samples.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
    let (min, p50, mean, max) = timing_stats(&wo_a_samples);
    println!(
        "K6 wo_a timing: {repeat} launches of one position, min {min:.4} p50 {p50:.4} mean {mean:.4} \
         max {max:.4} ms; batched {wo_a_batched:.4} ms/launch",
        repeat = args.repeat
    );
    verdicts.push(Verdict::new(
        "K6 wo_a output bits",
        wo_a_mismatches == 0,
        format!("{wo_a_exact}/{wo_a_total} exact vs mxfp8_ref::grouped_wo_a"),
    ));
    verdicts.push(Verdict::new(
        "K6 wo_a determinism",
        wo_a_deterministic,
        if wo_a_deterministic {
            "every position's repeat launch byte-identical".to_owned()
        } else {
            "a repeat launch differs".to_owned()
        },
    ));
    verdicts.push(Verdict::new(
        "K6 wo_a control",
        wo_a_control_ok,
        format!(
            "worst gap {:.4}% of operand max (bar {:.0}%)",
            100.0 * wo_a_control_max / wo_a_control_max_source,
            100.0 * CONTROL_BAR
        ),
    ));

    // --- the declared K6 -> K7 host seam: device lo_out widened and D8-quantised ---------------
    let mut wo_b_sources: Vec<Vec<f32>> = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        let start = token * WO_A_OUT;
        wo_b_sources.push(
            lo_out_bits[start..start + WO_A_OUT]
                .iter()
                .map(|bits| bf16_value(*bits))
                .collect(),
        );
    }
    let wo_b_activations = quantise_rows(&wo_b_sources)?;
    println!(
        "K7 wo_b activation operand (declared K6 -> K7 host seam): the device wo_a lo_out widened \
         from bf16 and D8-quantised, {} positions x {} codes, {} scales - identical for device and \
         twin, and the seam itself is NOT claimed",
        args.tokens,
        WO_A_OUT,
        WO_A_OUT / QUANT_BLOCK
    );
    let expected = dense_twin(
        &wo_b_activations.codes,
        &wo_b_activations.scales,
        &wo_b_weights,
        args.tokens,
        wo_b_spec.rows,
        wo_b_spec.cols,
    );
    let wo_b_rows = args.tokens;
    let wo_b_cols = wo_b_spec.rows;
    let wo_b_inner = wo_b_spec.cols;
    let wo_b_a_d = cuda.alloc(wo_b_rows * wo_b_inner, &mut allocated)?;
    let wo_b_a_scales_d = cuda.alloc(wo_b_rows * (wo_b_inner / QUANT_BLOCK), &mut allocated)?;
    let wo_b_w_d = cuda.alloc(wo_b_packed.len(), &mut allocated)?;
    let wo_b_s_d = cuda.alloc(wo_b_scales.len(), &mut allocated)?;
    let wo_b_out_d = cuda.alloc(wo_b_rows * wo_b_cols * 2, &mut allocated)?;
    cuda.copy_htod(&wo_b_a_d, &wo_b_activations.codes)?;
    cuda.copy_htod(&wo_b_a_scales_d, &wo_b_activations.scales)?;
    cuda.copy_htod(&wo_b_w_d, &wo_b_packed)?;
    cuda.copy_htod(&wo_b_s_d, &wo_b_scales)?;
    let wo_b_params = Mxfp8DenseParams::new(
        wo_b_a_d,
        wo_b_a_scales_d,
        wo_b_w_d,
        wo_b_s_d,
        wo_b_out_d,
        wo_b_rows as u32,
        wo_b_cols as u32,
        wo_b_inner as u32,
        0,
        args.layer as u32,
        0,
    );
    let wo_b_grid = mxfp8_dense_launch::grid(wo_b_rows as u32, wo_b_cols as u32)?;
    println!(
        "K7 wo_b launch shape: {} rows x {wo_b_cols} cols, inner {wo_b_inner}; grid {wo_b_grid:?}, \
         block {:?}, {} argument words through Mxfp8DenseParams",
        wo_b_rows,
        mxfp8_dense_launch::BLOCK,
        wo_b_params.as_slice().len()
    );
    mxfp8_dense_launch::launch(&cuda, dense_kernel, &wo_b_params)?;
    let mut wo_b_bytes = vec![0u8; wo_b_rows * wo_b_cols * 2];
    cuda.copy_dtoh(&mut wo_b_bytes, &wo_b_out_d)?;
    mxfp8_dense_launch::launch(&cuda, dense_kernel, &wo_b_params)?;
    let mut wo_b_again = vec![0u8; wo_b_rows * wo_b_cols * 2];
    cuda.copy_dtoh(&mut wo_b_again, &wo_b_out_d)?;
    let wo_b_deterministic = wo_b_bytes == wo_b_again;
    let wo_b_bits: Vec<u16> = wo_b_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let mut wo_b_exact = 0usize;
    let mut wo_b_mismatches = 0usize;
    let mut wo_b_first_mismatch: Option<usize> = None;
    for (index, (device, reference)) in wo_b_bits.iter().zip(&expected).enumerate() {
        if device == reference {
            wo_b_exact += 1;
        } else {
            wo_b_mismatches += 1;
            wo_b_first_mismatch = wo_b_first_mismatch.or(Some(index));
        }
    }
    let wo_b_total = wo_b_bits.len();
    println!(
        "K7 wo_b output bits: {wo_b_exact} of {wo_b_total} bf16 exact ({}), determinism {}{}",
        if wo_b_mismatches == 0 {
            "100%"
        } else {
            "MISMATCH"
        },
        if wo_b_deterministic {
            "byte-identical"
        } else {
            "DIFFERS"
        },
        match wo_b_first_mismatch {
            Some(index) => format!(", first mismatch at index {index}"),
            None => String::new(),
        }
    );
    let (wo_b_gap, _at, wo_b_max_source) = activation_gap(
        &wo_b_activations.source,
        &wo_b_activations.codes,
        &wo_b_activations.scales,
    );
    let wo_b_control_ok = control_gap_within_bar(wo_b_gap, wo_b_max_source);
    println!(
        "K7 wo_b f32/FP8 control: max D8 activation gap {wo_b_gap:e} against operand max \
         {wo_b_max_source:e} ({:.4}%; bar {:.0}%)",
        100.0 * wo_b_gap / wo_b_max_source,
        100.0 * CONTROL_BAR,
    );
    let (mut wo_b_samples, wo_b_batched) = cuda.launch_repeated(
        dense_kernel,
        wo_b_grid,
        mxfp8_dense_launch::BLOCK,
        wo_b_params.as_slice(),
        args.repeat,
    )?;
    wo_b_samples.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
    let (min, p50, mean, max) = timing_stats(&wo_b_samples);
    println!(
        "K7 wo_b timing: {repeat} launches, min {min:.4} p50 {p50:.4} mean {mean:.4} max {max:.4} ms; \
         batched {wo_b_batched:.4} ms/launch",
        repeat = args.repeat
    );
    verdicts.push(Verdict::new(
        "K7 wo_b output bits",
        wo_b_mismatches == 0,
        format!("{wo_b_exact}/{wo_b_total} exact vs mxfp8_ref::dense_gemm"),
    ));
    verdicts.push(Verdict::new(
        "K7 wo_b determinism",
        wo_b_deterministic,
        if wo_b_deterministic {
            "the repeat launch is byte-identical".to_owned()
        } else {
            "the repeat launch differs".to_owned()
        },
    ));
    verdicts.push(Verdict::new(
        "K7 wo_b control",
        wo_b_control_ok,
        format!(
            "gap {:.4}% of operand max (bar {:.0}%)",
            100.0 * wo_b_gap / wo_b_max_source,
            100.0 * CONTROL_BAR
        ),
    ));
    println!(
        "device bytes: {allocated} ({} KiB) held across both stages",
        allocated / 1024
    );

    cuda.free_buffers(&[
        &attn_d,
        &attn_scales_d,
        &wo_a_w_d,
        &wo_a_s_d,
        &lo_out_d,
        &wo_b_a_d,
        &wo_b_a_scales_d,
        &wo_b_w_d,
        &wo_b_s_d,
        &wo_b_out_d,
    ]);
    cuda.free(wo_a_module, &[]);
    cuda.free(dense_module, &[]);
    check_floor(args.floor_gib, "m13_output_projections end")?;

    let failed: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    println!(
        "summary: two stages x (output bits, determinism, control) plus two cubin pins, {} verdicts, \
         {} failed",
        verdicts.len(),
        failed.len()
    );
    if failed.is_empty() {
        println!(
            "m13_output_projections: PASS (K6 wo_a and K7 wo_b at the real layer-0 dimensions, every \
             output bit-identical to the in-crate references, deterministic, controls within bar; \
             {} positions; K5 attention is the next stage and is NOT part of this harness)",
            args.tokens
        );
        Ok(())
    } else {
        for verdict in &failed {
            println!("  FAIL {}: {}", verdict.name, verdict.detail);
        }
        Err(format!(
            "m13_output_projections: FAIL at {} of {} verdicts",
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
    fn the_output_projection_table_is_the_measured_one() {
        let wo_a = LAYER0_ATTENTION_WEIGHTS[3];
        assert_eq!((wo_a.name, wo_a.rows, wo_a.cols), ("wo_a", 8192, 4096));
        let wo_b = LAYER0_ATTENTION_WEIGHTS[4];
        assert_eq!((wo_b.name, wo_b.rows, wo_b.cols), ("wo_b", 5120, 8192));
        assert_eq!(wo_b.cols, wo_a.rows, "wo_b consumes wo_a's widened rank");
        assert_eq!(WO_A_OUT, O_GROUPS * O_LORA_RANK);
        assert_eq!(WO_A_OUT, wo_a.rows);
        assert_eq!(GROUP_INNER, wo_a.cols);
        assert_eq!(ATTENTION_WIDTH, O_GROUPS * GROUP_INNER);
        assert_eq!(ATTENTION_WIDTH, MLA_HEADS * MLA_LATENT);
        for spec in [wo_a, wo_b] {
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
        // The grouped decomposition of the stored wo_a tensor: 8 groups x 1024 x 4096, and its
        // per-group scale strides.
        assert_eq!(wo_a.rows * wo_a.cols, O_GROUPS * O_LORA_RANK * GROUP_INNER);
        assert_eq!(
            wo_a.scale_rows * wo_a.scale_cols,
            O_GROUPS * (O_LORA_RANK / 32) * (GROUP_INNER / 32),
            "the stored scale tensor is 8 contiguous per-group grids"
        );
    }

    #[test]
    fn pack_wo_a_with_rejects_a_wrong_group_stride() {
        // The repack's only structural choice is the per-group stride: group * lo_rank * group_inner.
        // The test calls the SAME function with a small geometry and compares it to an independently
        // written per-group expectation, then to a deliberately mis-strided copy: a wrong stride
        // cannot reproduce the declared bytes, so a mutation of the stride is caught here.
        let (groups, lo_rank, group_inner) = (2usize, 64usize, 128usize);
        let mut state = 0x0bad_c0de_1234_5678u64;
        let raw: Vec<u8> = (0..groups * lo_rank * group_inner)
            .map(|_| lcg(&mut state))
            .collect();
        let scales: Vec<u8> = (0..groups * (lo_rank / 32) * (group_inner / 32))
            .map(|_| 120 + lcg(&mut state) % 15)
            .collect();
        let (packed, returned_scales) = pack_wo_a_with(groups, lo_rank, group_inner, &raw, &scales)
            .expect("the small geometry is well-formed");
        assert_eq!(packed.len(), raw.len());
        assert_eq!(returned_scales, scales, "the scales are uploaded as stored");
        let mut expected = Vec::with_capacity(packed.len());
        for group in 0..groups {
            let start = group * lo_rank * group_inner;
            let mut tiled = vec![0u8; lo_rank * group_inner];
            for col in 0..lo_rank {
                for k in 0..group_inner {
                    tiled[mxfp8_ref::b12x_weight_index(col, k, group_inner)] =
                        raw[start + col * group_inner + k];
                }
            }
            expected.extend(tiled);
        }
        assert_eq!(
            packed, expected,
            "every byte is the checkpoint byte, permuted"
        );
        // The same construction with the stride off by one group row: it must differ, so the
        // equality above is not stride-invariant.
        let mut mis_strided = Vec::with_capacity(packed.len());
        for group in 0..groups {
            let start = group * (lo_rank - 1) * group_inner;
            let mut tiled = vec![0u8; lo_rank * group_inner];
            for col in 0..lo_rank {
                for k in 0..group_inner {
                    tiled[mxfp8_ref::b12x_weight_index(col, k, group_inner)] =
                        raw[start + col * group_inner + k];
                }
            }
            mis_strided.extend(tiled);
        }
        assert_ne!(
            packed, mis_strided,
            "a wrong group stride must not reproduce the declared repack"
        );
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
    fn the_grouped_twin_is_the_reference_arithmetic() {
        let (groups, lo_rank, group_inner) = (2usize, 32usize, 128usize);
        let mut state = 0xfeed_face_dead_beefu64;
        let attn: Vec<u8> = (0..groups * group_inner).map(|_| lcg(&mut state)).collect();
        let attn_scales: Vec<u8> = (0..groups * (group_inner / QUANT_BLOCK))
            .map(|_| 120 + lcg(&mut state) % 15)
            .collect();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut weights = Vec::new();
        for _group in 0..groups {
            let raw: Vec<u8> = (0..lo_rank * group_inner)
                .map(|_| lcg(&mut state))
                .collect();
            let raw_scales: Vec<u8> = (0..(lo_rank / 32) * (group_inner / 32))
                .map(|_| 120 + lcg(&mut state) % 15)
                .collect();
            let group_packed = pack_b12x(&raw, lo_rank, group_inner);
            weights.push(dequantise_weights(
                &group_packed,
                &raw_scales,
                lo_rank,
                group_inner,
            ));
            packed.extend_from_slice(&group_packed);
            scales.extend_from_slice(&raw_scales);
        }
        let twin = grouped_twin(&attn, &attn_scales, &weights, groups, group_inner, lo_rank);
        let reference = mxfp8_ref::grouped_wo_a(
            &attn,
            &attn_scales,
            &packed,
            &scales,
            groups,
            group_inner,
            lo_rank,
        );
        let exact = twin.iter().zip(&reference).filter(|(a, b)| a == b).count();
        assert_eq!(
            exact,
            twin.len(),
            "the grouped twin's bits must be the grouped reference's bits"
        );
    }

    #[test]
    fn the_dense_twin_is_the_reference_arithmetic_at_wo_b_dimensions() {
        // wo_b's real released geometry: [out 5120, in 8192], one position.
        let (rows, cols, inner) = (1usize, 5120usize, 8192usize);
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
            "the dense twin's bits must be the reference's bits"
        );
    }

    #[test]
    fn the_grouped_twin_is_the_reference_arithmetic_at_wo_a_dimensions() {
        // wo_a's real released geometry: 8 groups x [out 1024, in 4096], one position.
        let (groups, lo_rank, group_inner) = (O_GROUPS, O_LORA_RANK, GROUP_INNER);
        let mut state = 0x5eed_1234_abcd_0001u64;
        let attn: Vec<u8> = (0..groups * group_inner).map(|_| lcg(&mut state)).collect();
        let attn_scales: Vec<u8> = (0..groups * (group_inner / QUANT_BLOCK))
            .map(|_| 120 + lcg(&mut state) % 15)
            .collect();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut weights = Vec::new();
        for _group in 0..groups {
            let raw: Vec<u8> = (0..lo_rank * group_inner)
                .map(|_| lcg(&mut state))
                .collect();
            let raw_scales: Vec<u8> = (0..(lo_rank / 32) * (group_inner / 32))
                .map(|_| 120 + lcg(&mut state) % 15)
                .collect();
            let group_packed = pack_b12x(&raw, lo_rank, group_inner);
            weights.push(dequantise_weights(
                &group_packed,
                &raw_scales,
                lo_rank,
                group_inner,
            ));
            packed.extend_from_slice(&group_packed);
            scales.extend_from_slice(&raw_scales);
        }
        let twin = grouped_twin(&attn, &attn_scales, &weights, groups, group_inner, lo_rank);
        let reference = mxfp8_ref::grouped_wo_a(
            &attn,
            &attn_scales,
            &packed,
            &scales,
            groups,
            group_inner,
            lo_rank,
        );
        let exact = twin.iter().zip(&reference).filter(|(a, b)| a == b).count();
        assert_eq!(
            exact,
            twin.len(),
            "the grouped twin's bits must be the grouped reference's bits at the real geometry"
        );
    }

    #[test]
    fn the_activation_control_is_zero_on_an_exact_scale() {
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
        assert!(gap > 0.0);
    }

    #[test]
    fn both_cubin_pins_are_the_tree_pins() {
        let wo_a = baseline::pin(WO_A_KERNEL)
            .expect("the baseline manifest parses")
            .expect("wo_a is pinned");
        assert_eq!(wo_a.cubin_bytes, WO_A_PIN_BYTES);
        assert_eq!(wo_a.cubin_sha256, WO_A_PIN_SHA256);
        assert_eq!(wo_a.ptx_membar_lines, PIN_MEMBAR_LINES);
        let dense = baseline::pin(DENSE_KERNEL)
            .expect("the baseline manifest parses")
            .expect("the dense kernel is pinned");
        assert_eq!(dense.cubin_bytes, DENSE_PIN_BYTES);
        assert_eq!(dense.cubin_sha256, DENSE_PIN_SHA256);
        assert_eq!(dense.ptx_membar_lines, PIN_MEMBAR_LINES);
    }
}
