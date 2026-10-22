//! m14_flash_mla: the attention stage (K5 flash_mla_csa2) in isolation, format 2 - the released
//! CSA union of the 128-row FP8 window and the compressed candidates, softmaxed with the per-head
//! sink - at the real layer-0 geometry, in the m10-m13 window-harness style.
//!
//! Scope, stated before the numbers. One launch group per query position, on the device kernel
//! flash_mla_csa2_sm121: 64 head blocks of 256 threads, grid (64, 1, 1), the pinned 19-word
//! argument list through cuda_harness::flash_mla_params. The subject is the format-2 attention
//! arithmetic: the supplied engine-format window rows are all attended oldest first, then the
//! selected compressed candidates are attended when candidate <= position, and ONE f32 softmax runs
//! over that union with the per-head checkpoint sink in the denominator after the maximum. Window
//! b11 was the first device execution of format 2 in a Lead window; window b12 (the corrected
//! bf16-vs-bf16 measurement) is the standing record in docs/archive/build/phase3-flash-mla.md. Before b11
//! the kernel's own header recorded formats 0 and 1 as the measured columns (node0,
//! 2026-10-17) and format 2 as not yet executed on a device.
//!
//! What it is not, explicitly NOT claimed:
//!   * the HSI indexer scoring stage: the index side channel (index_key, index_query, block_scores,
//!     block_best) is uploaded as zeros plus sentinels and its writes are ignored, never compared;
//!   * the CSA candidate selection: the selected list here is a DECLARED host list of compressed
//!     absolute positions (every position older than the query's window, capped at index_topk), not
//!     the indexer's top-512 merge;
//!   * the K1-K4 device stages: q and the KV rows are CPU twins over the released weights, with the
//!     seams declared below;
//!   * RoPE as a claimed step: the kernel contract is that the caller supplies rope'd q and k, and
//!     this harness applies the engine reference's rope as a declared host seam;
//!   * the attention output's inverse rope, wo_a, wo_b, the residual add/hc_post, any chain order,
//!     any token/text/end-to-end output, staging/device budget, engine integration or host-time
//!     saving, and any per-decode-token timing derived from the measured launch times;
//!   * the historical formats 0 (FP4) and 1 (engine FP8): they are measured in M7 stage 2 and stage
//!     7 and are deliberately not re-run here;
//!   * long candidate lists: at the fixture lengths the union is at most 128 window rows plus the
//!     positions older than the window (0 at 8 positions, 1 at 129 positions - position 128 is the
//!     first with one - and 2 at 130), so the 512-candidate /
//!     640-slot regime is NOT exercised by this device run - it is covered by mla_ref's union
//!     tests and the m4_abi guards, not here.
//!
//! COMPARISON CLASS (declared before any number). The kernel's union attention is compared to
//! mla_ref::attention_with_sink_union - the reference union path - on the same q bytes, the same
//! window rows, the same compressed pool bytes, the same candidate list, position and sink. Both
//! sides are three-pass union paths; this is NOT a bit-identity claim, because they still differ in
//! expf against libm exp, the tree reduction of the normalising sum against the serial one, the
//! cached against recomputed weight handling, and the bf16 output write - the degrees of freedom the
//! M7 stage-2 record names. The declared bars are the stage-2 tolerance class the merged chain meets
//! for formats 0 and 1 on real checkpoint bytes:
//!   exact fraction >= 99.5% of the bf16 output elements equal to the bf16 rounding of the
//!   reference, and max abs delta (bf16 device output against the bf16-rounded reference) <= 1.6e-2
//!   (the stage-2 DELTA_BAR), with the measured max ULP distance reported. Both the aggregate exact
//!   fraction AND every position's exact fraction are judged against the bar, so one sub-bar
//!   position fails the run instead of being averaged away.
//! Determinism: every position's launch is repeated and all output bytes must match. Separate
//! control: the engine window format's quantisation gap (bf16-rounded f32 row against its decoded
//! E4M3 x E8M0 values) bounded at 10% of the operand's maximum absolute value - not part of the
//! exactness class.
//!
//! Provenance. Released checkpoint bytes, read verbatim through the snapshot reader:
//!   layers.0.attn.wq_a.weight/.scale, wq_b.weight/.scale, wkv.weight/.scale (FP8 weights,
//!   block-scaled, packed to B12x by the declared pack_b12x permutation),
//!   layers.0.attn.q_norm.weight [1280], kv_norm.weight [512] (F32 or BF16, widened as stored),
//!   layers.0.attn.attn_sink [64] F32 - consumed on BOTH sides: passed to the kernel as a [64]
//!   argument and to mla_ref::attention_with_sink_union as the per-head additive logit.
//! Declared seams (host-built, identical for device and reference):
//!   * q: the released embed.weight row t is the declared stand-in for the K1 block input (no mHC is
//!     run here); wq_a twin -> q_norm (NORM_EPS = 1e-20) -> D8 quantiser -> wq_b twin -> rope tail
//!     (last 64 of 512, theta 10000) at position t -> bf16, the dtype the kernel consumes;
//!   * main-KV pool: wkv twin on the same input -> kv_norm -> rope tail at its own position ->
//!     sparkflash-kv's FP4 main-KV quantiser (E2M1 codes, one E4M3 scale per 16 channels), the
//!     format format-2 candidate reads require (candidate ids index the per-position view);
//!   * window rows: the same kv latent rounded to bf16 (the engine's storage dtype) and quantised
//!     with the engine's per-32-channel E4M3/ue8m0 window quantiser (m4_quant::act_quant_fp8_row,
//!     which M7 stage 2 asserted byte-identical to the engine's quantiser), oldest first.
//!
//! Window construction for position p: the rows for positions max(0, p+1-128)..=p, oldest first.
//! Declared selected list: the compressed absolute positions strictly older than the window,
//! (max(0, p+1-128-512))..(p+1-128), i.e. every position behind the window capped at index_topk -
//! NOT the indexer's top-512.
//!
//! Cubin: flash_mla_csa2_sm121, 60,792 B, sha256
//! 64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b, 0 PTX membar lines, verified
//! by size and sha256 against baseline/kernel_baseline.txt BEFORE the first device allocation.
//! Build it under the pinned toolkit (CUDA 13.0.3, nvcc V13.0.88):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_sm121.cu -o out/flash_mla_csa2_sm121.cubin
//!
//! Shapes: --tokens 8 and --tokens 130 are the fixture lengths (independent query positions 0..N-1,
//! not a batch or decode step). Both are measured; 130 is the prefill length that first has
//! out-of-window candidates. The measured launch times are whole-launch attention timings at those
//! position counts, with vLLM stopped; no per-token cost is derived from them.
//!
//! The next chain stage after attention is the output projection pair (wo_a/wo_b), which m13 covers
//! in isolation; the chain wiring itself is M3's, not this harness's.
//!
//! Usage (aarch64 node, checkpoint present):
//!   m14_flash_mla --checkpoint <snapshot dir> --cubin-dir out [--tokens 8] [--repeat 100]
//!                 [--floor-gib 32]

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::cuda_harness::{
    flash_mla_params, launch_flash_mla, launch_flash_mla_repeated, load_flash_mla,
};
use sparkflash_kernels::m4_quant::{
    act_quant_fp8_row, bf16_bits, bf16_value, fp4_e2m1_encode, fp8_e4m3_decode, fp8_e4m3_encode,
};
use sparkflash_kernels::m7_layer0_shapes::{
    BlockScaledWeight, LAYER0_ATTENTION_WEIGHTS, scale_grid,
};
use sparkflash_kernels::mla_ref::{
    MAX_SELECTED, MLA_HEADS, MLA_LATENT, SWA_WINDOW, WINDOW_SCALE_BYTES, attention_with_sink_union,
};
use sparkflash_kernels::mxfp8_ref;

/// The kernel name in the baseline manifest.
const KERNEL: &str = "flash_mla_csa2_sm121";
/// Pinned cubin size.
const PIN_BYTES: usize = 69_064;
/// Pinned cubin sha256.
const PIN_SHA256: &str = "64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b";
/// The pinned kernel carries no fence.
const PIN_MEMBAR_LINES: u32 = 0;
/// One head per block, 256 threads.
const THREADS: u32 = 256;
/// The format-2 kernel value (CSA union): #define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u.
const FORMAT_CSA_UNION: u32 = 2;
/// The HSI pool geometry the kernel's indexer stage reads (its writes are ignored here).
const HSI_BLOCKS: usize = 2_048;
const HSI_BLOCK_TOKENS: usize = 8;
const INDEX_HEAD_DIM: usize = 128;
/// q_lora_rank: wq_a's width and q_norm's length.
const Q_LORA_RANK: usize = 1_280;
/// qk_rope_head_dim: rope rotates the last 64 of a head's 512.
const ROPE_HEAD_DIM: usize = 64;
/// text_config.rope_theta.
const ROPE_THETA: f32 = 10_000.0;
/// text_config.rms_norm_eps.
const NORM_EPS: f32 = 1e-20;
/// The FP4 main-KV row: 512 values pack into 256 bytes with one E4M3 scale per 16 channels.
const KV_PACKED_BYTES: usize = MLA_LATENT / 2;
const KV_SCALE_CHANNELS: usize = 16;
const KV_SCALE_BYTES: usize = MLA_LATENT / KV_SCALE_CHANNELS;
/// 6 * 2^-9: the floor sparkflash-kv's quantize_main_kv puts under a group's amax.
const KV_AMAX_FLOOR: f32 = 6.0 * 0.001_953_125;
/// E2M1's largest finite magnitude.
const E2M1_MAX: f32 = 6.0;
/// The stage-2 pass bars, printed before the numbers they judge. Each has a factored predicate so
/// the boundary itself is unit-tested: mutating a bar or inverting its comparison fails the tests.
const EXACT_BAR_PERCENT: f64 = 99.5;
const DELTA_BAR: f32 = 1.6e-2;
/// The declared control bar.
const CONTROL_BAR: f32 = 0.10;

/// The exact-fraction bar, applied to the aggregate AND to each position's own fraction.
fn passes_exact_fraction(pct: f64) -> bool {
    pct >= EXACT_BAR_PERCENT
}

/// The max-abs-delta bar: bf16 device output against the bf16 rounding of the f32 reference.
fn passes_max_abs_delta(delta: f32) -> bool {
    delta <= DELTA_BAR
}

/// The exact verdict: the aggregate AND the worst single position must clear the bar. An aggregate
/// alone would let a sub-bar position hide among passing ones.
fn exact_verdict(aggregate_pct: f64, worst_pct: f64) -> bool {
    passes_exact_fraction(aggregate_pct) && passes_exact_fraction(worst_pct)
}

/// The window control bar, as a fraction of the operand maximum.
fn window_gap_within_bar(gap: f32, operand_max: f32) -> bool {
    gap <= CONTROL_BAR * operand_max
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m14_flash_mla: {error}");
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
                        "usage: m14_flash_mla --checkpoint <snapshot dir> [--cubin-dir out] \
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
        if args.tokens == 0 || args.tokens > 1_024 {
            return Err("--tokens must be in 1..=1024".to_owned());
        }
        if args.layer != 0 {
            return Err(
                "--layer must be 0: the q/KV stand-ins are the layer-0 boundary inputs and this \
                 harness is the layer-0 stage, not a per-layer tool"
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

/// Decode every dense weight value once into f32: fp8 decode times its ue8m0 scale.
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

/// The dense CPU twin: mxfp8_ref::dense_gemm's arithmetic with the weight dequantisation hoisted.
fn dense_twin(
    a_codes: &[u8],
    a_scales: &[u8],
    dequantised_weight: &[f32],
    rows: usize,
    cols: usize,
    inner: usize,
) -> Vec<u16> {
    let k_blocks = inner / 32;
    assert_eq!(a_codes.len(), rows * inner);
    assert_eq!(a_scales.len(), rows * k_blocks);
    assert_eq!(dequantised_weight.len(), cols * inner);
    let mut out = vec![0u16; rows * cols];
    let mut activation = vec![0.0f32; inner];
    for row in 0..rows {
        for block in 0..k_blocks {
            let scale = ue8m0_decode(a_scales[row * k_blocks + block]);
            let base = block * 32;
            for i in 0..32 {
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

fn ue8m0_decode(byte: u8) -> f32 {
    sparkflash_kernels::m4_quant::ue8m0_decode(byte)
}

/// The monotone integer key of a bf16 bit pattern, so two codes one ULP apart differ by one.
fn bf16_order(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7FFF);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// RMSNorm over one row, the engine reference's formula.
fn rms_norm_row(values: &mut [f32], weight: &[f32], eps: f32) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    for (value, w) in values.iter_mut().zip(weight) {
        *value = *value * inverse * *w;
    }
}

/// The engine reference's rope tables for one position.
fn rope_tables_for(position: u32, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let pairs = dim / 2;
    let angles: Vec<f32> = (0..pairs)
        .map(|k| position as f32 / theta.powf(2.0 * k as f32 / dim as f32))
        .collect();
    (
        angles.iter().map(|angle| angle.cos()).collect(),
        angles.iter().map(|angle| angle.sin()).collect(),
    )
}

/// Rotates the LAST rope-long tail of each width-wide row in place, as the engine's rotate_rows.
fn rotate_rope_tail(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    theta: f32,
    position: u32,
) {
    let (cos, sin) = rope_tables_for(position, rope, theta);
    for row in 0..rows {
        let start = row * width + width - rope;
        for pair in 0..rope / 2 {
            let a = values[start + 2 * pair];
            let b = values[start + 2 * pair + 1];
            values[start + 2 * pair] = a * cos[pair] - b * sin[pair];
            values[start + 2 * pair + 1] = a * sin[pair] + b * cos[pair];
        }
    }
}

/// Quantises one 512-wide latent to the FP4 main-KV format, by sparkflash-kv's quantize_main_kv
/// rule: per-16-channel amax floored at 6 * 2^-9, an E4M3 scale of amax / 6, E2M1 codes clamped to
/// +-6, low nibble first.
fn quantize_kv_row(latent: &[f32]) -> ([u8; KV_PACKED_BYTES], [u8; KV_SCALE_BYTES]) {
    let mut packed = [0u8; KV_PACKED_BYTES];
    let mut scales = [0u8; KV_SCALE_BYTES];
    for group in 0..KV_SCALE_BYTES {
        let block = &latent[group * KV_SCALE_CHANNELS..(group + 1) * KV_SCALE_CHANNELS];
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale_byte = fp8_e4m3_encode(amax.max(KV_AMAX_FLOOR) / E2M1_MAX);
        let scale = fp8_e4m3_decode(scale_byte);
        scales[group] = scale_byte;
        for (offset, value) in block.iter().enumerate() {
            let index = group * KV_SCALE_CHANNELS + offset;
            let code = fp4_e2m1_encode((value / scale).clamp(-E2M1_MAX, E2M1_MAX));
            if index.is_multiple_of(2) {
                packed[index / 2] = code & 0x0F;
            } else {
                packed[index / 2] |= (code & 0x0F) << 4;
            }
        }
    }
    (packed, scales)
}

/// The bf16 storage rounding the window format applies exactly once before quantising.
fn bf16_row(source: &[f32]) -> Vec<f32> {
    source
        .iter()
        .map(|value| bf16_value(bf16_bits(*value)))
        .collect()
}

/// The window format's own quantisation gap: the decoded value against the bf16-STORED row it came
/// from. The source is rounded first, so the gap is the E4M3 x E8M0 quantisation error and does not
/// include the bf16 storage round the kernel's operand has already taken.
fn window_gap(source: &[f32], codes: &[u8], scales: &[u8]) -> (f32, f32) {
    let mut gap = 0.0f32;
    let mut max_source = 0.0f32;
    for index in 0..source.len() {
        let reconstructed = bf16_value(bf16_bits(
            fp8_e4m3_decode(codes[index])
                * ue8m0_decode(scales[index / sparkflash_kernels::mla_ref::WINDOW_QUANT_BLOCK]),
        ));
        gap = gap.max((reconstructed - source[index]).abs());
        max_source = max_source.max(source[index].abs());
    }
    (gap, max_source)
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

/// Reads the cubin and refuses anything but the pinned compiled form, before any allocation.
fn verify_cubin(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            path.display()
        )
    })?;
    let pin = baseline::pin(KERNEL)?.ok_or_else(|| format!("{KERNEL} is not in the manifest"))?;
    if bytes.len() != PIN_BYTES || pin.cubin_bytes != PIN_BYTES {
        return Err(format!(
            "{}: cubin is {} B but the pin is {PIN_BYTES} B",
            path.display(),
            bytes.len()
        ));
    }
    if pin.cubin_sha256 != PIN_SHA256 || pin.ptx_membar_lines != PIN_MEMBAR_LINES {
        return Err(format!(
            "the manifest's {KERNEL} pin changed: {} B, sha256 {}, {} membar lines",
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
    Ok(bytes)
}

fn read_scaled_weight(
    snapshot: &mut Snapshot,
    prefix: &str,
    spec: &BlockScaledWeight,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (scale_rows, scale_cols) = scale_grid(spec.rows, spec.cols);
    if (scale_rows, scale_cols) != (spec.scale_rows, spec.scale_cols) {
        return Err(format!(
            "{}: the measured scale grid [{}x{}] disagrees with the rule [{scale_rows}x{scale_cols}]",
            spec.name, spec.scale_rows, spec.scale_cols
        ));
    }
    let weight = snapshot.read_raw(
        &format!("{prefix}.attn.{}.weight", spec.name),
        "F8_E4M3",
        &[spec.rows, spec.cols],
    )?;
    let scales = snapshot.read_raw(
        &format!("{prefix}.attn.{}.scale", spec.name),
        "F8_E8M0",
        &[scale_rows, scale_cols],
    )?;
    if weight.len() != spec.rows * spec.cols || scales.len() != scale_rows * scale_cols {
        return Err(format!(
            "{}: the released tensor sizes disagree with the measured table",
            spec.name
        ));
    }
    Ok((weight, scales))
}

struct Position {
    q: Vec<f32>,
    window_codes: Vec<u8>,
    window_scales: Vec<u8>,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== m14_flash_mla: layer-0 attention (K5 flash_mla_csa2) format 2, the CSA union of the \
         128-row FP8 window and the compressed candidates, with the per-head sink ==="
    );
    println!(
        "comparison class (declared before any number): the kernel's union attention against \
         mla_ref::attention_with_sink_union on the same q bytes, window rows, FP4 pool, candidate \
         list, position and sink - NOT bit-identity, although both sides are three-pass union paths: \
         expf against libm exp, the tree reduction of the normalising sum against the serial one, the \
         cached against recomputed weight handling, and the bf16 output write still differ (the \
         stage-2 degrees of freedom). Bars: aggregate AND per-position exact fraction >= {:.1}% of \
         bf16 output elements equal to the bf16 rounding of the reference, and max abs delta (bf16 \
         device vs bf16-rounded reference) <= {:e} - the stage-2 tolerance class the merged chain \
         meets (format 0 FP4 163,838/163,840 at 2.4414063e-4; format 1 engine FP8 163,834/163,840 at \
         9.765625e-4) - with max ULP reported. \
         Determinism: every position's launch repeats byte-identically. Separate control: the \
         engine window format's quantisation gap against the bf16-stored row, bar {:.0}% of the \
         operand max. NOT claimed: the \
         HSI indexer scoring (its side channel is zeros plus sentinels, writes ignored), the CSA \
         candidate selection (declared list, not top-512), K1-K4 device stages (q/KV are CPU twins \
         with declared seams), RoPE as a claimed step, the output inverse rope/wo_a/wo_b, the \
         residual/hc_post, any chain, any token/text, staging/budget/engine/host-time, formats 0/1 \
         (measured in M7), and long candidate lists - at these fixture lengths the union tops out at \
         128 window rows plus 2 older positions, so the 512/640 regime is NOT device-exercised here.",
        EXACT_BAR_PERCENT,
        DELTA_BAR,
        100.0 * CONTROL_BAR
    );
    println!(
        "tokens: {} query positions (positions 0..{}); each is one launch group of ({MLA_HEADS}, 1, \
         1) blocks of ({THREADS}, 1, 1) threads; window = {SWA_WINDOW} rows oldest first.",
        args.tokens, args.tokens
    );
    check_floor(args.floor_gib, "m14_flash_mla start")?;

    let cubin_path = args.cubin_dir.join(format!("{KERNEL}.cubin"));
    let cubin = verify_cubin(&cubin_path)?;
    if SWA_WINDOW != 128 || MLA_HEADS != 64 || MLA_LATENT != 512 || MAX_SELECTED != 512 {
        return Err("the reference constants drifted from the measured geometry".to_owned());
    }

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let wq_a_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wq_a")
        .ok_or_else(|| "wq_a is not in the measured layer-0 table".to_owned())?;
    let wq_b_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wq_b")
        .ok_or_else(|| "wq_b is not in the measured layer-0 table".to_owned())?;
    let wkv_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wkv")
        .ok_or_else(|| "wkv is not in the measured layer-0 table".to_owned())?;
    if (wq_a_spec.rows, wq_a_spec.cols) != (Q_LORA_RANK, 5_120)
        || (wq_b_spec.rows, wq_b_spec.cols) != (MLA_HEADS * MLA_LATENT, Q_LORA_RANK)
        || (wkv_spec.rows, wkv_spec.cols) != (MLA_LATENT, 5_120)
    {
        return Err(
            "the measured layer-0 geometry disagrees with this harness's constants".to_owned(),
        );
    }
    let (wq_a_raw, wq_a_scales) = read_scaled_weight(&mut snapshot, &prefix, &wq_a_spec)?;
    let (wq_b_raw, wq_b_scales) = read_scaled_weight(&mut snapshot, &prefix, &wq_b_spec)?;
    let (wkv_raw, wkv_scales) = read_scaled_weight(&mut snapshot, &prefix, &wkv_spec)?;
    let wq_a_packed = pack_b12x(&wq_a_raw, wq_a_spec.rows, wq_a_spec.cols);
    let wq_b_packed = pack_b12x(&wq_b_raw, wq_b_spec.rows, wq_b_spec.cols);
    let wkv_packed = pack_b12x(&wkv_raw, wkv_spec.rows, wkv_spec.cols);
    for (name, packed, scales, spec) in [
        ("wq_a", &wq_a_packed, &wq_a_scales, &wq_a_spec),
        ("wq_b", &wq_b_packed, &wq_b_scales, &wq_b_spec),
        ("wkv", &wkv_packed, &wkv_scales, &wkv_spec),
    ] {
        if packed.len() != mxfp8_ref::b12x_buffer_bytes(spec.rows, spec.cols)
            || scales.len() != mxfp8_ref::b12x_scale_bytes(spec.rows, spec.cols)
        {
            return Err(format!(
                "{name}: the packed buffers disagree with the B12x rules"
            ));
        }
    }
    let wq_a_weights =
        dequantise_weights(&wq_a_packed, &wq_a_scales, wq_a_spec.rows, wq_a_spec.cols);
    let wq_b_weights =
        dequantise_weights(&wq_b_packed, &wq_b_scales, wq_b_spec.rows, wq_b_spec.cols);
    let wkv_weights = dequantise_weights(&wkv_packed, &wkv_scales, wkv_spec.rows, wkv_spec.cols);
    let q_norm =
        snapshot.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm =
        snapshot.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    let sink = snapshot.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let sink_min = sink.iter().copied().fold(f32::INFINITY, f32::min);
    let sink_max = sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "released weights: wq_a [{}x{}] + scale [{}x{}], wq_b [{}x{}] + scale [{}x{}], wkv [{}x{}] + \
         scale [{}x{}], all pack_b12x'd (value-preserving permutation), scales as stored",
        wq_a_spec.rows,
        wq_a_spec.cols,
        wq_a_spec.scale_rows,
        wq_a_spec.scale_cols,
        wq_b_spec.rows,
        wq_b_spec.cols,
        wq_b_spec.scale_rows,
        wq_b_spec.scale_cols,
        wkv_spec.rows,
        wkv_spec.cols,
        wkv_spec.scale_rows,
        wkv_spec.scale_cols
    );
    println!(
        "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] - consumed on BOTH sides: a [64] kernel \
         argument added to the softmax denominator after the maximum, and the same per-head logit in \
         mla_ref::attention_with_sink_union",
        sink.len()
    );

    let mut embed_rows: Vec<Vec<f32>> = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        embed_rows.push(snapshot.read_row_bf16("embed.weight", token, 5_120)?);
    }
    let mut xhat_codes: Vec<Vec<u8>> = Vec::with_capacity(args.tokens);
    let mut xhat_scales: Vec<Vec<u8>> = Vec::with_capacity(args.tokens);
    for row in &embed_rows {
        let (codes, scales) = act_quant_fp8_row(row);
        xhat_codes.push(codes);
        xhat_scales.push(scales);
    }
    println!(
        "declared stand-in for the K1 block input: the released embed.weight row directly (D8 \
         quantised); no mHC pass runs here - the K1 seam is declared, not claimed"
    );

    // Per-position q and main-KV: CPU twins over the released weights, seams declared.
    let mut positions: Vec<Position> = Vec::with_capacity(args.tokens);
    let mut window_gap_max = 0.0f32;
    let mut window_gap_source_max = 0.0f32;
    let mut pool_fp4 = vec![0u8; args.tokens * KV_PACKED_BYTES];
    let mut pool_scales = vec![0u8; args.tokens * KV_SCALE_BYTES];
    for token in 0..args.tokens {
        let q_latent_bits = dense_twin(
            &xhat_codes[token],
            &xhat_scales[token],
            &wq_a_weights,
            1,
            Q_LORA_RANK,
            5_120,
        );
        let mut q_latent: Vec<f32> = q_latent_bits.iter().map(|bits| bf16_value(*bits)).collect();
        rms_norm_row(&mut q_latent, &q_norm, NORM_EPS);
        let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
        let q_bits = dense_twin(
            &q_codes,
            &q_scales,
            &wq_b_weights,
            1,
            MLA_HEADS * MLA_LATENT,
            Q_LORA_RANK,
        );
        let mut q: Vec<f32> = q_bits.iter().map(|bits| bf16_value(*bits)).collect();
        rotate_rope_tail(
            &mut q,
            MLA_HEADS,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            token as u32,
        );
        let q_bf16: Vec<f32> = q
            .iter()
            .map(|value| bf16_value(bf16_bits(*value)))
            .collect();

        let kv_bits = dense_twin(
            &xhat_codes[token],
            &xhat_scales[token],
            &wkv_weights,
            1,
            MLA_LATENT,
            5_120,
        );
        let mut kv_latent: Vec<f32> = kv_bits.iter().map(|bits| bf16_value(*bits)).collect();
        rms_norm_row(&mut kv_latent, &kv_norm, NORM_EPS);
        rotate_rope_tail(
            &mut kv_latent,
            1,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            token as u32,
        );
        let (packed, row_scales) = quantize_kv_row(&kv_latent);
        pool_fp4[token * KV_PACKED_BYTES..(token + 1) * KV_PACKED_BYTES].copy_from_slice(&packed);
        pool_scales[token * KV_SCALE_BYTES..(token + 1) * KV_SCALE_BYTES]
            .copy_from_slice(&row_scales);
        // The one library construction of an engine-format window row: bf16 storage round
        // exactly once, then the engine quantiser, asserted byte-identical to act_quant_fp8_row.
        let engine_row = sparkflash_kernels::cuda_harness::engine_window_row(&kv_latent)?;
        let (window_codes, window_scales) = (engine_row.codes, engine_row.scales);
        let (gap, max_source) = window_gap(&bf16_row(&kv_latent), &window_codes, &window_scales);
        window_gap_max = window_gap_max.max(gap);
        window_gap_source_max = window_gap_source_max.max(max_source);
        positions.push(Position {
            q: q_bf16,
            window_codes,
            window_scales,
        });
    }
    println!(
        "declared seams: q = wq_a twin -> q_norm -> D8 -> wq_b twin -> rope(64 of 512, theta 10000) \
         -> bf16; main-KV pool = wkv twin -> kv_norm -> rope at its position -> FP4 (E2M1, E4M3 per \
         16 channels); window rows = the same latent rounded to bf16 and quantised per 32 channels \
         to E4M3 x ue8m0 (the engine's released window format)"
    );

    let cuda = Cuda::load()?;
    let (module, kernel) = load_flash_mla(&cuda, &cubin)?;
    let mut allocated = 0usize;
    let index_key_bytes = HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * INDEX_HEAD_DIM * 2;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(pool_fp4.len(), &mut allocated)?;
    let kv_scales_d = cuda.alloc(pool_scales.len(), &mut allocated)?;
    let window_codes_d = cuda.alloc(SWA_WINDOW as usize * MLA_LATENT, &mut allocated)?;
    let window_scales_d = cuda.alloc(SWA_WINDOW as usize * WINDOW_SCALE_BYTES, &mut allocated)?;
    let selected_d = cuda.alloc(MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let block_scores_d = cuda.alloc(HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(HSI_BLOCKS * 4, &mut allocated)?;
    let out_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    cuda.copy_htod(&kv_fp4_d, &pool_fp4)?;
    cuda.copy_htod(&kv_scales_d, &pool_scales)?;
    cuda.copy_htod(&index_key_d, &vec![0u8; index_key_bytes])?;
    cuda.copy_htod(&index_query_d, &vec![0u8; index_query_bytes])?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&sink))?;
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; HSI_BLOCKS]),
    )?;
    cuda.copy_htod(&block_best_d, &u32s_to_bytes(&vec![u32::MAX; HSI_BLOCKS]))?;
    println!(
        "index side channel: zeros plus sentinels, writes ignored and never compared (the indexer \
         scoring stage is NOT claimed); device bytes {allocated}"
    );

    let mut exact = 0usize;
    let mut total = 0usize;
    let mut max_delta = 0.0f32;
    let mut max_delta_at = String::new();
    let mut max_ulp = 0i32;
    let mut first_mismatch: Option<String> = None;
    let mut deterministic = true;
    let mut window_checks = 0usize;
    let mut selected_checks = 0usize;
    let mut worst_exact_pct = 100.0f64;
    let mut worst_position = 0usize;
    let mut union_max = 0usize;
    let mut timings: Vec<f64> = Vec::new();
    let mut batched = 0.0f64;
    for position_index in 0..args.tokens {
        let position = position_index as u32;
        let w0 = position.saturating_add(1).saturating_sub(SWA_WINDOW);
        let window_rows = (position - w0 + 1) as usize;
        let selected: Vec<u32> = if w0 == 0 {
            Vec::new()
        } else {
            let start = w0.saturating_sub(MAX_SELECTED as u32);
            (start..w0).collect()
        };
        union_max = union_max.max(window_rows + selected.len());
        if window_rows > 1 {
            window_checks += 1;
        }
        if !selected.is_empty() {
            selected_checks += 1;
        }
        // Window rows for this query: positions w0..=position, oldest first.
        let mut window_codes = Vec::with_capacity(window_rows * MLA_LATENT);
        let mut window_scales = Vec::with_capacity(window_rows * WINDOW_SCALE_BYTES);
        for row in w0..=position {
            let source = &positions[row as usize];
            window_codes.extend_from_slice(&source.window_codes);
            window_scales.extend_from_slice(&source.window_scales);
        }
        cuda.copy_htod(&window_codes_d, &window_codes)?;
        cuda.copy_htod(&window_scales_d, &window_scales)?;
        cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
        cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&positions[position_index].q))?;
        let params = flash_mla_params(
            q_d.0,
            kv_fp4_d.0,
            kv_scales_d.0,
            window_codes_d.0,
            window_scales_d.0,
            u64::from(FORMAT_CSA_UNION),
            window_rows as u64,
            selected_d.0,
            index_key_d.0,
            index_query_d.0,
            sink_d.0,
            block_scores_d.0,
            block_best_d.0,
            out_d.0,
            u64::from(position),
            selected.len(),
            0, // trace_ring: the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
            args.layer,
            0, // step_seq
        );
        launch_flash_mla(
            &cuda,
            kernel,
            (MLA_HEADS as u32, 1, 1),
            (THREADS, 1, 1),
            &params,
        )?;
        let mut device_bytes = vec![0u8; MLA_HEADS * MLA_LATENT * 2];
        cuda.copy_dtoh(&mut device_bytes, &out_d)?;
        launch_flash_mla(
            &cuda,
            kernel,
            (MLA_HEADS as u32, 1, 1),
            (THREADS, 1, 1),
            &params,
        )?;
        let mut again_bytes = vec![0u8; MLA_HEADS * MLA_LATENT * 2];
        cuda.copy_dtoh(&mut again_bytes, &out_d)?;
        if device_bytes != again_bytes {
            deterministic = false;
        }
        let device_bits: Vec<u16> = device_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        let mut position_exact = 0usize;
        for (head, sink_value) in sink.iter().enumerate() {
            let q_head = &positions[position_index].q[head * MLA_LATENT..(head + 1) * MLA_LATENT];
            let reference = attention_with_sink_union(
                q_head,
                &window_codes,
                &window_scales,
                window_rows,
                &pool_fp4,
                &pool_scales,
                &selected,
                position,
                *sink_value,
            );
            let base = head * MLA_LATENT;
            for dim in 0..MLA_LATENT {
                let device = device_bits[base + dim];
                let expected = bf16_bits(reference[dim]);
                total += 1;
                if device == expected {
                    exact += 1;
                    position_exact += 1;
                } else if first_mismatch.is_none() {
                    first_mismatch = Some(format!("position {position} head {head} dim {dim}"));
                }
                // Same guard as stage2.rs's compare_bf16_ulp: a non-finite delta becomes INFINITY
                // rather than being silently dropped by a max, and its index is reported.
                let delta = (bf16_value(device) - bf16_value(expected)).abs();
                if !delta.is_finite() || delta > max_delta {
                    max_delta = if delta.is_finite() {
                        delta
                    } else {
                        f32::INFINITY
                    };
                    max_delta_at = format!("position {position} head {head} dim {dim}");
                }
                max_ulp = max_ulp.max((bf16_order(device) - bf16_order(expected)).abs());
            }
        }
        let position_pct = 100.0 * position_exact as f64 / (MLA_HEADS * MLA_LATENT) as f64;
        if position_pct < worst_exact_pct {
            worst_exact_pct = position_pct;
            worst_position = position_index;
        }
        if args.tokens <= 8 {
            println!(
                "  position {position}: window {window_rows} rows, selected {} ids, union {} slots - \
                 {position_exact}/{} bf16 exact ({position_pct:.4}%)",
                selected.len(),
                window_rows + selected.len(),
                MLA_HEADS * MLA_LATENT
            );
        }
        // Timing: repeats of the same launch on the last position.
        if position_index + 1 == args.tokens {
            let (mut samples, batch) = launch_flash_mla_repeated(
                &cuda,
                kernel,
                (MLA_HEADS as u32, 1, 1),
                (THREADS, 1, 1),
                &params,
                args.repeat,
            )?;
            samples.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
            timings = samples;
            batched = batch;
        }
    }
    let exact_pct = 100.0 * exact as f64 / total as f64;
    println!(
        "output class: {exact} of {total} bf16 elements exact against the reference ({exact_pct:.4}%), \
         max abs delta (bf16 vs bf16) {max_delta:e} at {max_delta_at}, max ULP {max_ulp}; bars \
         {EXACT_BAR_PERCENT:.1}% (aggregate and every position) and {DELTA_BAR:e} \
         (the stage-2 tolerance class)"
    );
    println!(
        "coverage: positions 0..{}, window rows 1..{}, out-of-window selected lists at {} position(s), \
         max union {} slots; worst position {worst_position} at {worst_exact_pct:.4}% exact; the \
         512-candidate/640-slot regime is NOT exercised at these lengths",
        args.tokens - 1,
        SWA_WINDOW.min(args.tokens as u32),
        selected_checks,
        union_max
    );
    let control_ok = window_gap_within_bar(window_gap_max, window_gap_source_max);
    println!(
        "window f32/FP8 control: max quantisation gap {:e} against operand max {:e} ({:.4}%; bar \
         {:.0}%) over {window_checks} multi-row windows",
        window_gap_max,
        window_gap_source_max,
        100.0 * window_gap_max / window_gap_source_max,
        100.0 * CONTROL_BAR
    );
    let (min, p50, mean, max) = timing_stats(&timings);
    println!(
        "timing: {repeat} launches of one position ({MLA_HEADS} blocks of {THREADS}), min {min:.4} p50 \
         {p50:.4} mean {mean:.4} max {max:.4} ms; batched {batched:.4} ms/launch",
        repeat = args.repeat
    );
    println!("device bytes: {allocated} ({} KiB)", allocated / 1024);
    cuda.free_buffers(&[
        &q_d,
        &kv_fp4_d,
        &kv_scales_d,
        &window_codes_d,
        &window_scales_d,
        &selected_d,
        &index_key_d,
        &index_query_d,
        &sink_d,
        &block_scores_d,
        &block_best_d,
        &out_d,
    ]);
    cuda.free(module, &[]);
    check_floor(args.floor_gib, "m14_flash_mla end")?;

    let mut failed: Vec<String> = Vec::new();
    fn verdict(failed: &mut Vec<String>, name: &str, ok: bool, detail: String) {
        println!(
            "verdict {name}: {} - {detail}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failed.push(name.to_owned());
        }
    }
    verdict(
        &mut failed,
        "cubin pin",
        true,
        format!("{KERNEL} {PIN_BYTES} B sha256 {PIN_SHA256}"),
    );
    verdict(
        &mut failed,
        "output exact fraction",
        exact_verdict(exact_pct, worst_exact_pct),
        format!(
            "{exact_pct:.4}% overall, worst position {worst_position} at \
             {worst_exact_pct:.4}% (bar {EXACT_BAR_PERCENT:.1}% on both)"
        ),
    );
    verdict(
        &mut failed,
        "output max abs delta",
        passes_max_abs_delta(max_delta),
        format!(
            "{max_delta:e} bf16-vs-bf16 at {max_delta_at} (bar {DELTA_BAR:e}), max ULP {max_ulp}"
        ),
    );
    verdict(
        &mut failed,
        "determinism",
        deterministic,
        if deterministic {
            "every position's repeat launch byte-identical".to_owned()
        } else {
            "a repeat launch differs".to_owned()
        },
    );
    verdict(
        &mut failed,
        "window control",
        control_ok,
        format!(
            "gap {:.4}% of operand max (bar {:.0}%)",
            100.0 * window_gap_max / window_gap_source_max,
            100.0 * CONTROL_BAR
        ),
    );
    if failed.is_empty() {
        println!(
            "m14_flash_mla: PASS (format-2 CSA union attention at the real layer-0 geometry, {} \
             positions, union attention within the declared stage-2 tolerance class, deterministic, \
             window control within bar; indexer scoring, candidate selection and the chain are NOT \
             claimed)",
            args.tokens
        );
        Ok(())
    } else {
        Err(format!(
            "m14_flash_mla: FAIL at {} verdict(s): {}",
            failed.len(),
            failed.join(", ")
        ))
    }
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_bits_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pinned_geometry_is_the_measured_one() {
        let wq_a = LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == "wq_a")
            .expect("wq_a is measured");
        let wq_b = LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == "wq_b")
            .expect("wq_b is measured");
        let wkv = LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == "wkv")
            .expect("wkv is measured");
        assert_eq!((wq_a.rows, wq_a.cols), (Q_LORA_RANK, 5_120));
        assert_eq!(
            (wq_b.rows, wq_b.cols),
            (MLA_HEADS * MLA_LATENT, Q_LORA_RANK)
        );
        assert_eq!((wkv.rows, wkv.cols), (MLA_LATENT, 5_120));
        assert_eq!(KV_PACKED_BYTES, 256);
        assert_eq!(KV_SCALE_BYTES, 32);
        assert_eq!(WINDOW_SCALE_BYTES, 16);
        assert_eq!(ROPE_HEAD_DIM, 64);
        assert_eq!(SWA_WINDOW, 128);
        assert_eq!(MAX_SELECTED, 512);
    }

    #[test]
    fn the_cubin_pin_is_the_tree_pin() {
        let pin = baseline::pin(KERNEL)
            .expect("the manifest parses")
            .expect("the kernel is pinned");
        assert_eq!(pin.cubin_bytes, PIN_BYTES);
        assert_eq!(pin.cubin_sha256, PIN_SHA256);
        assert_eq!(pin.ptx_membar_lines, PIN_MEMBAR_LINES);
    }

    #[test]
    fn the_dense_twin_is_the_reference_arithmetic() {
        let (rows, cols, inner) = (2usize, 64usize, 1280usize);
        let mut state = 0x1234_5678u64;
        let mut lcg = |bound: u32| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as u32) % bound
        };
        let codes: Vec<u8> = (0..rows * inner).map(|_| lcg(0x7E) as u8).collect();
        let scales: Vec<u8> = (0..rows * (inner / 32))
            .map(|_| 120 + lcg(15) as u8)
            .collect();
        let weights: Vec<u8> = (0..cols * inner).map(|_| lcg(0x7E) as u8).collect();
        let weight_scales: Vec<u8> = (0..(cols / 32) * (inner / 32))
            .map(|_| 120 + lcg(15) as u8)
            .collect();
        let packed = pack_b12x(&weights, cols, inner);
        let dequantised = dequantise_weights(&packed, &weight_scales, cols, inner);
        let twin = dense_twin(&codes, &scales, &dequantised, rows, cols, inner);
        let reference =
            mxfp8_ref::dense_gemm(&codes, &scales, &packed, &weight_scales, rows, cols, inner);
        let exact = twin.iter().zip(&reference).filter(|(a, b)| a == b).count();
        assert_eq!(exact, twin.len(), "the twin must be the reference's bits");
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let mut values: Vec<f32> = (0..MLA_LATENT).map(|i| i as f32 * 0.5).collect();
        let before = values.clone();
        rotate_rope_tail(&mut values, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
        assert_eq!(values, before);
    }

    #[test]
    fn rope_preserves_pair_norms() {
        let mut values: Vec<f32> = (0..MLA_LATENT).map(|i| (i % 7) as f32 - 3.0).collect();
        let start = MLA_LATENT - ROPE_HEAD_DIM;
        let untouched = values[..start].to_vec();
        let before: Vec<f32> = values[start..].to_vec();
        rotate_rope_tail(&mut values, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 7);
        assert_eq!(
            &values[..start],
            &untouched[..],
            "rope only touches the last 64"
        );
        for pair in 0..ROPE_HEAD_DIM / 2 {
            let a0 = before[2 * pair];
            let b0 = before[2 * pair + 1];
            let a1 = values[start + 2 * pair];
            let b1 = values[start + 2 * pair + 1];
            assert!((a0 * a0 + b0 * b0 - (a1 * a1 + b1 * b1)).abs() < 1e-4);
        }
    }

    #[test]
    fn the_fp4_quantiser_round_trips_through_the_reference_decoder() {
        let latent: Vec<f32> = (0..MLA_LATENT)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let (packed, scales) = quantize_kv_row(&latent);
        for index in 0..MLA_LATENT {
            let decoded = sparkflash_kernels::mla_ref::kv_latent(&packed, &scales, 0, index);
            assert!(decoded.is_finite());
            assert!((decoded - latent[index]).abs() <= 6.0 * 2.0f32.powi(-2));
        }
    }

    #[test]
    fn the_window_row_is_bf16_rounded_then_quantised() {
        let latent: Vec<f32> = (0..MLA_LATENT).map(|i| (i as f32).sin() * 3.0).collect();
        let row = sparkflash_kernels::cuda_harness::engine_window_row(&latent)
            .expect("a 512-wide row quantises");
        assert_eq!(row.codes.len(), MLA_LATENT);
        assert_eq!(row.scales.len(), WINDOW_SCALE_BYTES);
        let (gap, max_source) = window_gap(&bf16_row(&latent), &row.codes, &row.scales);
        assert!(gap > 0.0, "the FP8 window format is lossy");
        assert!(max_source > 0.0);
        assert!(
            window_gap_within_bar(gap, max_source),
            "the control bar must hold on this row"
        );
    }

    #[test]
    fn the_stage2_bars_are_pinned() {
        assert!(
            (EXACT_BAR_PERCENT - 99.5).abs() < f64::EPSILON,
            "the declared exact bar moved: {EXACT_BAR_PERCENT}"
        );
        assert!(
            (DELTA_BAR - 1.6e-2).abs() <= f32::EPSILON,
            "the declared delta bar moved: {DELTA_BAR}"
        );
        assert!(
            (CONTROL_BAR - 0.10).abs() <= f32::EPSILON,
            "the declared control bar moved: {CONTROL_BAR}"
        );
        assert!(passes_exact_fraction(99.5));
        assert!(!passes_exact_fraction(99.4999));
        assert!(passes_max_abs_delta(1.6e-2));
        assert!(!passes_max_abs_delta(1.6001e-2));
        let operand_max = 6.0f32;
        assert!(window_gap_within_bar(0.099 * operand_max, operand_max));
        assert!(!window_gap_within_bar(0.101 * operand_max, operand_max));
    }

    #[test]
    fn one_sub_bar_position_fails_the_exact_verdict() {
        assert!(exact_verdict(99.9, 99.9));
        assert!(exact_verdict(99.5, 99.5));
        assert!(
            !exact_verdict(99.9, 99.0),
            "an aggregate above the bar must not hide a sub-bar position"
        );
    }

    #[test]
    fn the_declared_selected_list_is_behind_the_window() {
        for position in 0..300u32 {
            let w0 = position.saturating_add(1).saturating_sub(SWA_WINDOW);
            let start = w0.saturating_sub(MAX_SELECTED as u32);
            let selected: Vec<u32> = if w0 == 0 {
                Vec::new()
            } else {
                (start..w0).collect()
            };
            assert!(selected.len() <= MAX_SELECTED);
            for id in &selected {
                assert!(*id < w0, "an id inside the window would be duplicated");
                assert!(*id <= position);
            }
            let window_rows = (position - w0 + 1) as usize;
            assert!(window_rows <= SWA_WINDOW as usize);
            assert!(window_rows + selected.len() <= SWA_WINDOW as usize + MAX_SELECTED);
        }
    }

    #[test]
    fn the_union_reference_prices_the_declared_list() {
        // A tiny fixture through the reference, so the call shape this harness uses is exercised
        // without a device: one window row and two compressed candidates, one behind the window and
        // one ahead of the query (excluded by the causal mask).
        let q = vec![0.5f32; MLA_LATENT];
        let mut window_codes = vec![0u8; MLA_LATENT];
        let mut window_scales = vec![0u8; WINDOW_SCALE_BYTES];
        window_codes[0] = 0x38;
        window_scales[0] = 127;
        let mut pool_fp4 = vec![0u8; 4 * KV_PACKED_BYTES];
        let mut pool_scales = vec![120u8; 4 * KV_SCALE_BYTES];
        pool_fp4[0] = 0x21;
        pool_scales[0] = 121;
        let out = attention_with_sink_union(
            &q,
            &window_codes,
            &window_scales,
            1,
            &pool_fp4,
            &pool_scales,
            &[0, 2],
            1,
            0.25,
        );
        assert_eq!(out.len(), MLA_LATENT);
        assert!(out.iter().all(|value| value.is_finite()));
    }
}
