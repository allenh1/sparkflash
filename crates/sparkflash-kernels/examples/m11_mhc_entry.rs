//! m11_mhc_entry: the first device stage of the chain - layer-0 Mega-mHC entry
//! (kernels/mega_mhc_sm121.cu) - in isolation, against released layer-0 checkpoint bytes.
//!
//! Scope, stated before the numbers. This harness launches ONE chain stage: the fused mHC entry of
//! layer 0, on real released tensor bytes (hc_attn_fn/hc_attn_base/hc_attn_scale/attn_norm.weight),
//! with the layer-0 boundary seed as the only host-built input. It is not attention, not a chain
//! pass, not a token, and not end-to-end text. The host seams BEFORE this stage are named and not
//! measured here: the embedding row (the X_-1 seed), the layer-0 boundary coefficients (B =
//! identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0 - the checkpoint stores no layer -1 coefficients),
//! and the staging of the mHC tensors themselves. The chain's seam ledger, weight staging and the
//! device budget are M3-b/c work and are NOT in this harness. Explicitly NOT claimed:
//!   * any attention work (flash_mla_csa2_sm121 is the next stage and is not launched);
//!   * any chain ordering, seam ledger or per-layer schedule;
//!   * any token, any text, any end-to-end layer or model output;
//!   * routing, SwiGLU, the MoE block, the shared expert, the combine;
//!   * weight staging / device-budget admission / engine integration;
//!   * a host-time saving of any kind.
//!
//! Kernel arguments use the typed builder (mega_mhc_launch::MegaMhcParams + launch), so this
//! harness also exercises the builder on device for the first time: 19 words in the pinned
//! declaration order, the fixed (4, 1, 1) grid of 128 threads, and the builder's load() resolving
//! the pinned symbol mega_mhc_fused_bf16_sm121.
//!
//! Cubin: the pinned mega_mhc_sm121 cubin is required - 141,632 B and
//! 7040411617954a17e89e6d9d8fcea1a872171de259a5515bde1e69a96c69af5c, read from the tree's baseline
//! manifest (sparkflash_kernels::baseline::pin), and the harness refuses anything else by length and
//! sha256 before the first allocation. Build it under the pinned toolkit (CUDA 13.0.3, nvcc
//! V13.0.88):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/mega_mhc_sm121.cu \
//!       -o out/mega_mhc_sm121.cubin
//!
//! COMPARISON CLASS (declared before any number is printed). The CPU twin is
//! mega_mhc_ref::mega_mhc_layer_with_scale on the SAME bytes the device gets - the released
//! full-projection arithmetic, both sides, no fold and no averaging. Per layer-0 position:
//!   bit-identical:  the residual-stream write X_l (after bf16 rounding), the 160 ue8m0 scale bytes,
//!                   and the f32 block-input capture xhat_normed (computed from the same inputs and
//!                   the device's own 1/rms, so it checks the capture arithmetic itself);
//!   declared bars:  the 5,120 E4M3 codes (>= 99.5% exact and every code within 1), the 24 mixed
//!                   coefficients B/A/C (max abs delta <= 1e-4), and 1/rms(X_hat) (<= 3e-4).
//! The bars are not looseness: the kernel reduces the two sums of squares in a fixed tile order and
//! rounds once, while the reference accumulates sequentially, so the two are not the same floating
//! point program. The class is the one m7 stage 1 declares for this kernel, unchanged.
//!   determinism:    two device runs of every position are byte-identical across all six outputs;
//!   f32 control:    the kernel's own E4M3 codes x ue8m0 scales, decoded, against the kernel's own
//!                   f32 block-input capture - the FP8 quantisation gap, bounded at 10% of that
//!                   capture's maximum absolute value (the m10 control style). It is reported
//!                   separately and is not part of the exactness claim.
//!
//! Tokens: this kernel has no batch dimension - it runs one position per launch - so --tokens N
//! launches N independent layer-0 positions (the released embed.weight rows 0..N as the X_-1 seed,
//! four copies, exactly the layer-0 boundary). N = 8 and N = 130 are both valid shapes; 130 is the
//! fixture prefill length, and nothing about the kernel changes with N (the grid and the arguments
//! are identical), so the two runs test the same stage over more positions, not a second shape.
//!
//! Usage (aarch64 node, checkpoint present):
//!   m11_mhc_entry --checkpoint <snapshot dir> --cubin-dir out [--tokens 8] [--repeat 100]
//!                 [--floor-gib 32]

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, bytes_to_f32};
use sparkflash_kernels::baseline;
use sparkflash_kernels::m4_quant::{
    QUANT_BLOCK, bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode,
};
use sparkflash_kernels::mega_mhc_launch::{self, MegaMhcParams};
use sparkflash_kernels::mega_mhc_ref::{
    HC_MULT, HIDDEN, NORM_EPS, PREDICTOR_ROWS, Streams, TrafficLedger, bias_into_kernel_rows,
    block_input_normed, mega_mhc_layer_with_scale, released_predictor_into_kernel_rows,
};

/// The cubin name in the baseline manifest (the nvcc source stem).
const KERNEL: &str = "mega_mhc_sm121";
/// The pinned cubin size, from baseline/kernel_baseline.txt through baseline::pin.
const PIN_BYTES: usize = 151_592;
/// The pinned cubin sha256, from the same manifest.
const PIN_SHA256: &str = "7040411617954a17e89e6d9d8fcea1a872171de259a5515bde1e69a96c69af5c";
/// The pinned PTX membar-line count: this kernel's grid rendezvous carries fences.
const PIN_MEMBAR_LINES: u32 = 7;
/// The hc_stats grid scratch the launcher must allocate: STATS x (tiles + 1).
const MHC_GRID_STATS: usize = 130;

fn main() {
    if let Err(error) = run() {
        eprintln!("m11_mhc_entry: {error}");
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
                        "usage: m11_mhc_entry --checkpoint <snapshot dir> [--cubin-dir out] \
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
                "--layer must be 0: the layer-0 boundary seed (B = identity, C = 0, A = [1,0,0,0], \
                 Y_-1 = 0) is what makes this position's inputs well defined, and a later layer \
                 needs the previous layer's streams, which this isolated harness does not have"
                    .to_owned(),
            );
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// The layer-0 boundary seed: B = identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0.
///
/// The checkpoint stores no layer -1 coefficients, so this is the released boundary state; it is
/// host-built and declared as such, not read from the checkpoint.
fn layer_zero_seed() -> (
    [[f32; HC_MULT]; HC_MULT],
    [f32; HC_MULT],
    [f32; HC_MULT],
    Vec<f32>,
) {
    let mut b = [[0.0f32; HC_MULT]; HC_MULT];
    for (index, row) in b.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let c = [0.0f32; HC_MULT];
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let y = vec![0.0f32; HIDDEN];
    (b, c, a, y)
}

/// The residual-stream update the kernel writes, X_l' = B X_l + C Y_-1, on the host.
fn expected_stream_update(
    x: &[f32],
    b: &[[f32; HC_MULT]; HC_MULT],
    c: &[f32; HC_MULT],
    y: &[f32],
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), HC_MULT * hidden);
    assert_eq!(y.len(), hidden);
    let mut expected = vec![0.0f32; HC_MULT * hidden];
    for stream in 0..HC_MULT {
        for column in 0..hidden {
            let mut value = 0.0f32;
            for source in 0..HC_MULT {
                value += b[stream][source] * x[source * hidden + column];
            }
            value += c[stream] * y[column];
            expected[stream * hidden + column] = value;
        }
    }
    expected
}

/// X_hat = A .* X (one factor per stream) and the released 1/rms(X_hat) statistic.
fn host_xhat(x: &[f32], a: &[f32; HC_MULT], hidden: usize) -> (Vec<f32>, f32) {
    assert_eq!(x.len(), HC_MULT * hidden);
    let mut xhat = vec![0.0f32; hidden];
    let mut squares = 0.0f32;
    for (column, slot) in xhat.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (stream, weight) in a.iter().enumerate() {
            value += weight * x[stream * hidden + column];
        }
        squares += value * value;
        *slot = value;
    }
    (xhat, 1.0f32 / (squares / hidden as f32 + NORM_EPS).sqrt())
}

/// The quantisation gap of the kernel's own FP8 block input: max |codes x scales - f32 capture|.
///
/// Returns (max abs gap, index, max abs capture). The scales are per QUANT_BLOCK values, as the
/// kernel writes them.
fn quantisation_gap(capture: &[f32], codes: &[u8], scales: &[u8]) -> (f32, usize, f32) {
    assert_eq!(capture.len(), codes.len());
    assert_eq!(scales.len(), capture.len().div_ceil(QUANT_BLOCK));
    let mut max_gap = 0.0f32;
    let mut at = 0usize;
    let mut max_capture = 0.0f32;
    for (index, value) in capture.iter().enumerate() {
        let reconstructed =
            fp8_e4m3_decode(codes[index]) * ue8m0_decode(scales[index / QUANT_BLOCK]);
        let gap = (reconstructed - value).abs();
        if gap > max_gap || gap.is_nan() {
            max_gap = gap;
            at = index;
        }
        max_capture = max_capture.max(value.abs());
    }
    (max_gap, at, max_capture)
}

/// One comparison's numbers, so a verdict is computed from what was printed.
struct Stats {
    len: usize,
    exact: usize,
    max_delta: f32,
    at: usize,
}

impl Stats {
    fn exact_pct(&self) -> f64 {
        100.0 * self.exact as f64 / self.len as f64
    }

    fn all_exact(&self) -> bool {
        self.exact == self.len
    }
}

fn compare_f32(label: &str, device: &[f32], expected: &[f32]) -> Result<Stats, String> {
    if device.len() != expected.len() {
        return Err(format!(
            "{label}: {} device values against {} expected values",
            device.len(),
            expected.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, e)) in device.iter().zip(expected).enumerate() {
        if d.to_bits() == e.to_bits() {
            stats.exact += 1;
        }
        let delta = (d - e).abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} f32 values, exact {}/{} ({:.4}%), max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
}

fn compare_u8(label: &str, device: &[u8], expected: &[u8]) -> Result<Stats, String> {
    if device.len() != expected.len() {
        return Err(format!(
            "{label}: {} device values against {} expected values",
            device.len(),
            expected.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, e)) in device.iter().zip(expected).enumerate() {
        if d == e {
            stats.exact += 1;
        }
        let delta = (f32::from(*d) - f32::from(*e)).abs();
        if delta > stats.max_delta {
            stats.max_delta = delta;
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} bytes, exact {}/{} ({:.4}%), max delta {} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
}

fn compare_bf16_write(label: &str, device_bits: &[u16], expected: &[f32]) -> Result<Stats, String> {
    let expected_bits: Vec<u16> = expected.iter().map(|value| bf16_bits(*value)).collect();
    if device_bits.len() != expected_bits.len() {
        return Err(format!(
            "{label}: {} device values against {} expected values",
            device_bits.len(),
            expected_bits.len()
        ));
    }
    let mut stats = Stats {
        len: device_bits.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, e)) in device_bits.iter().zip(&expected_bits).enumerate() {
        if d == e {
            stats.exact += 1;
        }
        let delta = (bf16_value(*d) - bf16_value(*e)).abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
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

/// The released per-sublayer mHC tensors in the form the kernel and the reference consume.
struct ReleasedMhcTensors {
    predictor: Vec<f32>,
    hc_base: [f32; PREDICTOR_ROWS],
    hc_base_kernel: [f32; PREDICTOR_ROWS],
    hc_scale: [f32; 3],
}

fn read_released_mhc_tensors(
    snap: &mut Snapshot,
    prefix: &str,
    sublayer: &str,
) -> Result<ReleasedMhcTensors, String> {
    let raw_fn = snap.read_f32(
        &format!("{prefix}.hc_{sublayer}_fn"),
        &[PREDICTOR_ROWS, HC_MULT * HIDDEN],
    )?;
    let base = snap.read_f32(&format!("{prefix}.hc_{sublayer}_base"), &[PREDICTOR_ROWS])?;
    let group_scale = snap.read_f32(&format!("{prefix}.hc_{sublayer}_scale"), &[3])?;
    let hc_base: [f32; PREDICTOR_ROWS] = base
        .as_slice()
        .try_into()
        .map_err(|_| format!("{prefix}.hc_{sublayer}_base must be {PREDICTOR_ROWS} values"))?;
    let hc_scale: [f32; 3] = group_scale
        .as_slice()
        .try_into()
        .map_err(|_| format!("{prefix}.hc_{sublayer}_scale must be three values"))?;
    // B0: permute the released rows into kernel order and keep all n copies. Nothing here may
    // average - the fold is retired and both sides consume the full released projection.
    let mut predictor = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN];
    released_predictor_into_kernel_rows(&raw_fn, &mut predictor);
    Ok(ReleasedMhcTensors {
        predictor,
        hc_base,
        hc_base_kernel: bias_into_kernel_rows(&hc_base),
        hc_scale,
    })
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
        "=== m11_mhc_entry: layer-0 Mega-mHC entry (first chain device stage) in isolation ==="
    );
    println!(
        "quantities compared per position: 4 x {HIDDEN} residual-stream f32 (written as bf16), \
         {HIDDEN} E4M3 codes, {HIDDEN} / {QUANT_BLOCK} ue8m0 scale bytes, 24 mixed coefficients, \
         1/rms(X_hat), {HIDDEN} f32 block-input capture"
    );
    println!(
        "comparison class (declared before any number): device vs the CPU twin on the same released \
         bytes, per layer-0 position - BIT-IDENTICAL on the residual-stream write X_l after bf16 \
         rounding, on the ue8m0 scale bytes, and on the f32 block-input capture computed from the \
         same inputs and the device's own 1/rms; DECLARED BARS on the E4M3 codes (>= 99.5% exact, \
         every code within 1), on B/A/C (max abs delta <= 1e-4) and on 1/rms (<= 3e-4) because the \
         kernel's fixed-order tile reduction and the reference's sequential accumulation are not \
         the same floating-point program. Determinism: two device runs of every position \
         byte-identical. Separate control: the FP8 quantisation gap of the kernel's own block \
         input, bounded at 10% of its capture maximum. NOT claimed here: attention, any chain/schedule \
         or seam ledger, any token or text, routing/SwiGLU/MoE/shared expert, weight staging, device \
         budget, engine integration, host-time saving. The layer-0 boundary seed (B = identity, C = 0, \
         A = [1,0,0,0], Y_-1 = 0) and the embed rows are host inputs, declared as such."
    );
    println!(
        "tokens: {0} independent layer-0 positions, one launch each (the kernel has no batch \
         dimension; the grid and arguments are identical for every position)",
        args.tokens
    );
    check_floor(args.floor_gib, "m11_mhc_entry start")?;

    let cubin_path = args.cubin_dir.join(format!("{KERNEL}.cubin"));
    let cubin = verify_cubin(&cubin_path)?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    println!(
        "released mHC tensors: hc_attn_fn {} f32 (released [24, 4*d] permuted into kernel B/A/C \
         order, no averaging), hc_attn_base {PREDICTOR_ROWS} f32, hc_attn_scale {:?}, \
         attn_norm.weight {HIDDEN} f32 - read verbatim from the checkpoint",
        mhc.predictor.len(),
        mhc.hc_scale
    );
    let (b_prev, c_prev, a_prev, y_prev) = layer_zero_seed();
    println!(
        "layer-0 boundary seed (host-built, declared): B = identity ({}), C = 0, A = {:?}, \
         Y_-1 = 0",
        HC_MULT * HC_MULT,
        a_prev
    );

    // Device buffers, admitted against the memory floor before anything is allocated.
    let planned = (HC_MULT * HIDDEN * 2
        + HIDDEN * 2
        + HC_MULT * HC_MULT * 4
        + HC_MULT * 4
        + HC_MULT * 4
        + mhc.predictor.len() * 4
        + mhc.hc_scale.len() * 4
        + mhc.hc_base_kernel.len() * 4
        + norm.len() * 4
        + HIDDEN
        + HIDDEN / QUANT_BLOCK
        + MHC_GRID_STATS * 4
        + 2 * 4
        + PREDICTOR_ROWS * 4
        + 4
        + HIDDEN * 4) as u64;
    let cuda = sparkflash_kernels::cuda_driver::Cuda::load()?;
    let (module, kernel) = mega_mhc_launch::load(&cuda, &cubin)?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?;
    let y_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let b_d = cuda.alloc(HC_MULT * HC_MULT * 4, &mut allocated)?;
    let c_d = cuda.alloc(HC_MULT * 4, &mut allocated)?;
    let a_d = cuda.alloc(HC_MULT * 4, &mut allocated)?;
    let predictor_d = cuda.alloc(mhc.predictor.len() * 4, &mut allocated)?;
    let scale_d = cuda.alloc(mhc.hc_scale.len() * 4, &mut allocated)?;
    let base_d = cuda.alloc(mhc.hc_base_kernel.len() * 4, &mut allocated)?;
    let norm_d = cuda.alloc(norm.len() * 4, &mut allocated)?;
    let fp8_d = cuda.alloc(HIDDEN, &mut allocated)?;
    let scales_d = cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?;
    let stats_d = cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?;
    let sync_d = cuda.alloc(2 * 4, &mut allocated)?;
    let coeffs_d = cuda.alloc(PREDICTOR_ROWS * 4, &mut allocated)?;
    let rms_d = cuda.alloc(4, &mut allocated)?;
    let normed_d = cuda.alloc(HIDDEN * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB); device buffers planned {planned} B",
        allocated / 1024
    );

    cuda.copy_htod(
        &y_d,
        &bf16_bits_to_bytes(&y_prev.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>()),
    )?;
    cuda.copy_htod(&b_d, &f32_to_bytes(&flatten_b(&b_prev)))?;
    cuda.copy_htod(&c_d, &f32_to_bytes(&c_prev))?;
    cuda.copy_htod(&a_d, &f32_to_bytes(&a_prev))?;
    cuda.copy_htod(&predictor_d, &f32_to_bytes(&mhc.predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&mhc.hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&mhc.hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;

    let params = MegaMhcParams::new(
        x_d,
        y_d,
        b_d,
        c_d,
        a_d,
        predictor_d,
        scale_d,
        base_d,
        norm_d,
        fp8_d,
        scales_d,
        stats_d,
        sync_d,
        coeffs_d,
        rms_d,
        normed_d,
        0, // trace_ring: null, the shipped cubin is the trace-disabled build
        args.layer as u32,
        0,
    );
    let mut verdicts: Vec<Verdict> = Vec::new();

    for token in 0..args.tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let x_bits: Vec<u16> = x.iter().map(|value| bf16_bits(*value)).collect();
        let x_bytes = bf16_bits_to_bytes(&x_bits);

        // The CPU twin on the same bytes the device gets.
        let mut streams = Streams::new(x.clone());
        let mut ledger = TrafficLedger::default();
        let (_next, coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &norm,
            &a_prev,
            &mhc.predictor,
            &mhc.hc_base,
            &mhc.hc_scale,
            &mut ledger,
        );
        let expected_x = expected_stream_update(&x, &b_prev, &c_prev, &y_prev, HIDDEN);
        let (host_xhat_values, expected_inv_rms) = host_xhat(&x, &a_prev, HIDDEN);

        let outputs = run_once(&cuda, kernel, &params, &x_d, &x_bytes, &stats_d, &sync_d)?;

        println!("--- position {token} (embed row {token}, four copies) ---");
        let b_ref: Vec<f32> = coefficients
            .b
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let s_b = compare_f32("hc_coeffs B (the mixer)", &outputs.coeffs[0..16], &b_ref)?;
        let s_a = compare_f32(
            "hc_coeffs A (the input mixes)",
            &outputs.coeffs[16..20],
            &coefficients.a,
        )?;
        let s_c = compare_f32(
            "hc_coeffs C (the block-output weights)",
            &outputs.coeffs[20..24],
            &coefficients.c,
        )?;
        let s_fp8 = compare_u8("xhat_fp8 (E4M3 codes)", &outputs.codes, codes.as_slice())?;
        let s_scales = compare_u8("xhat_scales (ue8m0)", &outputs.scales, scales.as_slice())?;
        let s_x = compare_bf16_write("residual streams X_l", &outputs.stream_bits, &expected_x)?;
        let s_rms = compare_f32("1/rms(X_hat)", &outputs.rms, &[expected_inv_rms])?;
        let expected_normed = block_input_normed(&host_xhat_values, &norm, outputs.rms[0]);
        let s_capture = compare_f32(
            "block input f32 capture (xhat_normed)",
            &outputs.normed,
            &expected_normed,
        )?;

        let coefficients_ok =
            s_b.max_delta <= 1e-4 && s_a.max_delta <= 1e-4 && s_c.max_delta <= 1e-4;
        let rms_ok = s_rms.max_delta <= 3e-4;
        let fp8_ok = s_fp8.exact_pct() >= 99.5 && s_fp8.max_delta <= 1.0;
        let capture_ok = s_capture.all_exact();
        let scales_ok = s_scales.all_exact();
        let streams_ok = s_x.all_exact();
        if s_fp8.exact != s_fp8.len {
            if scales_ok && s_fp8.max_delta <= 1.0 {
                println!(
                    "  finding: {} of {} E4M3 codes differ (first at index {}), each by one code \
                     with the scales bit-exact; the cause is upstream of the quantiser: the device's \
                     fixed-order tile reduction and the reference's sequential accumulation round \
                     1/rms differently (delta {:e}), moving a value across an E4M3 boundary",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_rms.max_delta,
                );
            } else {
                println!(
                    "  finding: {} of {} E4M3 codes differ (first at index {}), max code delta {}, \
                     with {} of {} ue8m0 scales differing: a QUANTISER-RULE difference, not the \
                     reduction-order residual",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_fp8.max_delta,
                    s_scales.len - s_scales.exact,
                    s_scales.len,
                );
            }
        }

        // Determinism: the same launch again on the same uploaded operands, all outputs byte-identical.
        let again = run_once(&cuda, kernel, &params, &x_d, &x_bytes, &stats_d, &sync_d)?;
        let deterministic = outputs.bytes_equal(&again);
        verdicts.push(Verdict::new(
            &format!("position {token}: determinism"),
            deterministic,
            if deterministic {
                "two device runs byte-identical".to_owned()
            } else {
                "two device runs DIFFER".to_owned()
            },
        ));

        // The FP8 control: the kernel's own quantised block input against its own f32 capture.
        let (gap, gap_at, capture_max) =
            quantisation_gap(&outputs.normed, &outputs.codes, &outputs.scales);
        let control_ok = gap <= 0.10 * capture_max;
        println!(
            "  f32 control (FP8 quantisation gap): max abs gap {:e} at index {} against capture \
             max {:e} ({:.4}% of the capture maximum; bar 10%)",
            gap,
            gap_at,
            capture_max,
            100.0 * gap / capture_max,
        );

        let position_ok = fp8_ok
            && scales_ok
            && streams_ok
            && coefficients_ok
            && rms_ok
            && capture_ok
            && deterministic
            && control_ok;
        verdicts.push(Verdict::new(
            &format!("position {token}: boundaries and control"),
            position_ok,
            format!(
                "fp8 {} scales {} X_l {} B/A/C {} 1/rms {} capture {} control {}",
                if fp8_ok { "within bar" } else { "DIFFER" },
                if scales_ok { "bit-exact" } else { "DIFFER" },
                if streams_ok { "bit-exact" } else { "DIFFER" },
                if coefficients_ok {
                    "<=1e-4"
                } else {
                    "OVER 1e-4"
                },
                if rms_ok { "<=3e-4" } else { "OVER 3e-4" },
                if capture_ok { "bit-exact" } else { "DIFFER" },
                if control_ok { "within bar" } else { "OVER bar" },
            ),
        ));
    }

    // Timing on the last uploaded operands (the kernel writes the streams in place, so the values
    // after the repeats are a different quantity from the compared ones and are not compared).
    let (mut per_launch, batched_ms) =
        mega_mhc_launch::launch_repeated(&cuda, kernel, &params, args.repeat)?;
    per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
    let (min, p50, mean, max) = timing_stats(&per_launch);
    println!(
        "timing: {} launches, one position each (values after the repeats are not compared)",
        args.repeat
    );
    println!(
        "per-launch ms: min {min:.4} p50 {p50:.4} mean {mean:.4} max {max:.4}; batched ms/launch \
         {batched_ms:.4}"
    );

    cuda.free(
        module,
        &[
            x_d,
            y_d,
            b_d,
            c_d,
            a_d,
            predictor_d,
            scale_d,
            base_d,
            norm_d,
            fp8_d,
            scales_d,
            stats_d,
            sync_d,
            coeffs_d,
            rms_d,
            normed_d,
        ],
    );
    check_floor(args.floor_gib, "m11_mhc_entry end")?;

    let failed: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    println!(
        "summary: {} positions x 2 verdicts, {} failed",
        args.tokens,
        failed.len()
    );
    if failed.is_empty() {
        println!(
            "m11_mhc_entry: PASS ({} of {} positions at every boundary; the mHC entry stage is \
             exact where its arithmetic is exact and within the declared bars where the reduction \
             order differs)",
            args.tokens, args.tokens
        );
        Ok(())
    } else {
        for verdict in &failed {
            println!("  FAIL {}: {}", verdict.name, verdict.detail);
        }
        Err(format!(
            "m11_mhc_entry: FAIL at {} of {} position verdicts",
            failed.len(),
            verdicts.len()
        ))
    }
}

/// One device run's outputs, kept byte-addressable so determinism is a byte comparison.
struct Outputs {
    stream_bits: Vec<u16>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    coeffs: Vec<f32>,
    rms: Vec<f32>,
    normed: Vec<f32>,
    raw: Vec<Vec<u8>>,
}

impl Outputs {
    fn bytes_equal(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

/// Uploads one position's streams, zeroes the per-launch rendezvous state, launches once, and reads
/// every output back.
fn run_once(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    params: &MegaMhcParams,
    x_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    x_bytes: &[u8],
    stats_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    sync_d: &sparkflash_kernels::cuda_driver::Deviceptr,
) -> Result<Outputs, String> {
    cuda.copy_htod(x_d, x_bytes)?;
    // The grid rendezvous counters are per-launch state: without re-zeroing them the second launch
    // starts with the first one's tile count and never publishes its coefficients.
    cuda.copy_htod(stats_d, &vec![0u8; MHC_GRID_STATS * 4])?;
    cuda.copy_htod(sync_d, &[0u8; 2 * 4])?;
    mega_mhc_launch::launch(cuda, kernel, params)?;

    let mut stream_bytes = vec![0u8; HC_MULT * HIDDEN * 2];
    let mut codes = vec![0u8; HIDDEN];
    let mut scales = vec![0u8; HIDDEN / QUANT_BLOCK];
    let mut coeff_bytes = vec![0u8; PREDICTOR_ROWS * 4];
    let mut rms_bytes = vec![0u8; 4];
    let mut normed_bytes = vec![0u8; HIDDEN * 4];
    // The params carry the addresses; read them back by position, not by re-deriving offsets.
    let words = params.as_slice();
    let (stream_d, codes_d, scales_d) = (
        sparkflash_kernels::cuda_driver::Deviceptr(words[0]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[9]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[10]),
    );
    let (coeffs_d, rms_d, normed_d) = (
        sparkflash_kernels::cuda_driver::Deviceptr(words[13]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[14]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[15]),
    );
    cuda.copy_dtoh(&mut stream_bytes, &stream_d)?;
    cuda.copy_dtoh(&mut codes, &codes_d)?;
    cuda.copy_dtoh(&mut scales, &scales_d)?;
    cuda.copy_dtoh(&mut coeff_bytes, &coeffs_d)?;
    cuda.copy_dtoh(&mut rms_bytes, &rms_d)?;
    cuda.copy_dtoh(&mut normed_bytes, &normed_d)?;
    let stream_bits: Vec<u16> = stream_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let raw = vec![
        stream_bytes,
        codes.clone(),
        scales.clone(),
        coeff_bytes.clone(),
        rms_bytes.clone(),
        normed_bytes.clone(),
    ];
    Ok(Outputs {
        stream_bits,
        codes,
        scales,
        coeffs: bytes_to_f32(&coeff_bytes),
        rms: bytes_to_f32(&rms_bytes),
        normed: bytes_to_f32(&normed_bytes),
        raw,
    })
}

fn flatten_b(b: &[[f32; HC_MULT]; HC_MULT]) -> Vec<f32> {
    b.iter().flat_map(|row| row.iter().copied()).collect()
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layer_zero_seed_is_the_documented_one() {
        let (b, c, a, y) = layer_zero_seed();
        for (row, entries) in b.iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                assert_eq!(*value, if row == column { 1.0 } else { 0.0 });
            }
        }
        assert_eq!(c, [0.0f32; HC_MULT]);
        assert_eq!(a, [1.0f32, 0.0, 0.0, 0.0]);
        assert_eq!(y.len(), HIDDEN);
        assert!(y.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn the_expected_stream_update_is_the_kernel_formula() {
        // Two columns, four streams: stream 0 is [1, 2], stream 1 is [3, 4], streams 2 and 3 are
        // zero; C = [0.5, -0.25, 0, 0], Y = [10, 20].
        let hidden = 2;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let b = [
            [1.0f32, 2.0, 0.0, 0.0],
            [3.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let c = [0.5f32, -0.25, 0.0, 0.0];
        let y = vec![10.0f32, 20.0];
        let expected = expected_stream_update(&x, &b, &c, &y, hidden);
        assert_eq!(expected.len(), 8);
        // stream 0, column 0: 1*1 + 2*3 + 0.5*10 = 12
        assert_eq!(expected[0], 12.0);
        // stream 0, column 1: 1*2 + 2*4 + 0.5*20 = 20
        assert_eq!(expected[1], 20.0);
        // stream 1, column 0: 3*1 + -1*3 + -0.25*10 = -2.5
        assert_eq!(expected[2], -2.5);
        // stream 1, column 1: 3*2 + -1*4 + -0.25*20 = -3
        assert_eq!(expected[3], -3.0);
        // streams 2 and 3 are the identity rows with no bias term: still zero.
        assert_eq!(&expected[4..], &[0.0f32; 4]);
    }

    #[test]
    fn the_quantisation_gap_is_zero_on_exact_reconstruction() {
        // 1.5 is exactly representable in E4M3 and 2^0 is an exact ue8m0 scale (code 127).
        let capture = vec![1.5f32; QUANT_BLOCK];
        let codes = vec![fp8_e4m3_encode_for_test(1.5); QUANT_BLOCK];
        let scales = vec![127u8];
        let (gap, at, max_capture) = quantisation_gap(&capture, &codes, &scales);
        assert_eq!(gap, 0.0);
        assert_eq!(at, 0);
        assert_eq!(max_capture, 1.5);
    }

    #[test]
    fn the_quantisation_gap_notices_a_wrong_scale() {
        // The same exact codes with a scale one too large must show the gap.
        let capture = vec![1.5f32; QUANT_BLOCK];
        let codes = vec![fp8_e4m3_encode_for_test(1.5); QUANT_BLOCK];
        let scales = vec![128u8];
        let (gap, _at, _max_capture) = quantisation_gap(&capture, &codes, &scales);
        assert!(gap > 0.0, "a wrong scale must move the reconstruction");
    }

    fn fp8_e4m3_encode_for_test(value: f32) -> u8 {
        // The tree's encoder, used only to build an exact test vector.
        sparkflash_kernels::m4_quant::fp8_e4m3_encode(value)
    }
}
