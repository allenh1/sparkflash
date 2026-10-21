//! m10_gate_up: the routed-expert W4A8 gate/up projection
//! (kernels/routed_gate_up_sm121.cu) against released layer-0 expert bytes.
//!
//! Scope, stated before the numbers: this harness runs ONE FP8-x-packed-FP4 GEMM - the routed gate
//! projection (one expert's w1 slice, 576 output rows per rank) and then the same kernel for the up
//! projection (w3) - on released layer-0 weights and released activation stand-ins. It is not the
//! MoE block, not SwiGLU, not the combine, and not layer text: the two gaps between this kernel and
//! the pinned chain are named in the report and are NOT measured here:
//!
//!   * the routing-weight multiply (it belongs before the bf16 cast inside the down projection's
//!     activation quantiser, model.py:849-851, which is why this kernel takes no weight operand);
//!   * the shared expert (dense FP8, already served by b12x_mxfp8_dense_sm121 at K = 5120) and the
//!     routed sum / shared-add epilogue (b12x_mxfp4_moe_sm121's job).
//!
//! Activation stand-in: the released embed.weight rows 0..tokens, bf16-rounded and quantised with
//! the D8 quantiser (m4_quant::act_quant_fp8_row), exactly as mega_mhc_sm121 leaves the ffn block
//! input. It is a stand-in for the post-attention hidden, reported as such, never as layer text.
//!
//! Weight slice: the checkpoint stores each routed expert's w1/w3 as I8 [2304, 2560] (packed FP4,
//! two values per byte) with F8_E8M0 scales [2304, 160]. This harness probes rank 0's slice, rows
//! 0..576 and their scale rows, read byte-for-byte from the checkpoint - no expert weight is decoded
//! and re-encoded on the way to the device.
//!
//! PASS bar (declared before any number is printed):
//!   1. the cubin is the pinned one: bytes and sha256 equal to routed_gate_up_abi;
//!   2. the CPU twin is bit-identical to the shared harness arithmetic (b2_harness::gate_up_rows)
//!      over the same dequantised activation, on all tokens * top_k * 576 outputs of both
//!      projections - two separately written host implementations of the same definition. The
//!      shared path reuses the tree's FP4 loader, so this cross-checks the arithmetic ordering, not
//!      the decoders (the twin's own test does the independent decode);
//!   3. the device gate output is bit-identical to the twin on every one of tokens * top_k * 640
//!      bf16 values (ULP 0), and the same for the up projection;
//!   4. the 576..639 padding columns of both projections are exactly zero on the device;
//!   5. two device runs are byte-identical;
//!   6. the engine-style f32 control (unquantised block input, same expert slice) is within a
//!      printed bar: max abs delta <= 10% of the control's max abs value, over the 576 real
//!      columns only (the twin's padding columns are not part of the control). This is the
//!      quantisation gap, not an exactness claim, and it is reported separately from the exact
//!      device comparison.
//!
//! Anything outside the bar exits non-zero with the failing line named.
//!
//! Usage (aarch64 node, checkpoint present):
//!
//! ```text
//! m10_gate_up --checkpoint <snapshot dir> --cubin-dir out [--tokens 8] [--layer 0] [--experts 8]
//!             [--rank 0] [--repeat 100] [--floor-gib 32]
//! ```
//!
//! The cubin must be the pinned one: `nvcc -std=c++17 -arch=sm_121a -cubin -I kernels
//! kernels/routed_gate_up_sm121.cu -o out/routed_gate_up_sm121.cubin` under CUDA 13.0.3
//! (V13.0.88); the harness refuses a different cubin by hash rather than measuring it.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, gate_up_rows};
use sparkflash_engine::config::ModelConfig;
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::m4_quant::{
    act_quant_fp8_row, bf16_bits, bf16_value, fp8_e4m3_decode, ue8m0_decode,
};
use sparkflash_kernels::routed_gate_up_abi as abi;
use sparkflash_kernels::routed_gate_up_launch::{self, GateUpLaunchParams};
use sparkflash_kernels::routed_gate_up_ref as twin;

/// Stored intermediate width of one routed expert (moe_intermediate_size).
const INTER: usize = 2304;
/// Packed FP4 bytes per stored expert row: 5120 / 2.
const PACKED_ROW: usize = 2560;
/// ue8m0 scale bytes per stored expert row: 5120 / 32.
const SCALE_ROW: usize = 160;

fn main() {
    if let Err(error) = run() {
        eprintln!("m10_gate_up: {error}");
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
    experts: usize,
    rank: usize,
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
            experts: 8,
            rank: 0,
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
                "--experts" => {
                    args.experts = next()?
                        .parse()
                        .map_err(|_| "--experts needs a number".to_owned())?
                }
                "--rank" => {
                    args.rank = next()?
                        .parse()
                        .map_err(|_| "--rank needs a number".to_owned())?
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
                        "usage: m10_gate_up --checkpoint <snapshot dir> [--cubin-dir out] \
                         [--tokens 8] [--layer 0] [--experts 8] [--rank 0] [--repeat 100] \
                         [--floor-gib 32]"
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
        if args.experts == 0 || args.experts > 384 {
            return Err("--experts must be in 1..=384".to_owned());
        }
        let ranks = INTER / twin::WIDTH;
        if args.rank >= ranks {
            return Err(format!("--rank must be in 0..{ranks}"));
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// Little-endian byte pairs to u16 (BF16 bit patterns, unconverted).
fn bf16_bits_of_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    let mut at = 0usize;
    for (index, (left, right)) in a.iter().zip(b).enumerate() {
        let delta = (left - right).abs();
        if delta > max || delta.is_nan() {
            max = delta;
            at = index;
        }
    }
    (max, at)
}

/// The real columns (0..WIDTH) of every routed row of a padded `[rows][WIDTH_PADDED]` buffer.
///
/// The twin returns the ABI's 640-wide rows, but the engine-style f32 control has no padding
/// columns, so a gap comparison must select the 576 real columns per row. Comparing the whole
/// padded row is the mistake that panicked in the first window run: at 8 tokens the control has
/// 48 x 576 = 27,648 values while the twin has 48 x 640 = 30,720.
fn real_columns(padded: &[f32], routed_rows: usize) -> Vec<f32> {
    assert_eq!(
        padded.len(),
        routed_rows * twin::WIDTH_PADDED,
        "the padded buffer is {} routed rows of {} f32 values",
        routed_rows,
        twin::WIDTH_PADDED
    );
    let mut real = Vec::with_capacity(routed_rows * twin::WIDTH);
    for row in 0..routed_rows {
        let start = row * twin::WIDTH_PADDED;
        real.extend_from_slice(&padded[start..start + twin::WIDTH]);
    }
    real
}

/// A control buffer already in the engine's real-column width (no ABI padding).
///
/// The distinct type is the regression guard for the first-window panic: `max_abs_delta` takes
/// `&[f32]`, so restoring the old call site `max_abs_delta(&twin_gate_f32, &control_gate)` is a
/// type error instead of a silent 640-wide-vs-576-wide comparison.
#[derive(Debug, Clone, Default)]
struct RealColumns(Vec<f32>);

impl RealColumns {
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    fn extend(&mut self, values: impl IntoIterator<Item = f32>) {
        self.0.extend(values);
    }

    fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// The engine-style f32 control is `[rows][WIDTH]` (no padding columns) while the twin returns the
/// ABI's padded `[rows][WIDTH_PADDED]` rows. The gap comparison therefore goes through this helper:
/// it selects the real columns first and only then takes the max abs delta, so a call site cannot
/// accidentally compare a padded row with the control again.
fn control_gap(twin_padded: &[f32], control: &RealColumns, routed_rows: usize) -> (f32, usize) {
    let real = real_columns(twin_padded, routed_rows);
    max_abs_delta(&real, control.as_slice())
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
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

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let cfg = ModelConfig::v41_flash_cached();
    if cfg.dim != twin::HIDDEN
        || cfg.moe_inter_dim != INTER
        || cfg.n_activated_experts != twin::TOP_K
    {
        return Err(format!(
            "the released configuration is not the GEMM's geometry: dim {}, moe_inter_dim {}, \
             n_activated_experts {}; the kernel pins {}, {}, {}",
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_activated_experts,
            twin::HIDDEN,
            INTER,
            twin::TOP_K
        ));
    }
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== m10_gate_up: routed W4A8 gate/up projection against released layer-0 expert bytes ==="
    );
    println!(
        "kernel: routed_gate_up_sm121, K = {}, N = {} per rank (padded {}), one projection per \
         launch (gate then up); {} experts resident, rank {} slice rows {}..{}",
        twin::HIDDEN,
        twin::WIDTH,
        twin::WIDTH_PADDED,
        args.experts,
        args.rank,
        args.rank * twin::WIDTH,
        (args.rank + 1) * twin::WIDTH
    );
    println!(
        "comparison class (declared before any number): device gate and up outputs bit-identical \
         to the CPU twin on every one of {} x 640 bf16 values; twin bit-identical to the shared \
         harness arithmetic on every one of {} x 576 values; padding columns 576..639 exactly \
         zero; two device runs byte-identical; engine-style f32 control gap reported separately, \
         over the {} real columns only (the twin's padding columns are excluded) and bounded at 10% \
         of the control maximum. What is NOT compared here: the routing weight, SwiGLU, the combine \
         and the shared expert - those stay outside this kernel by contract.",
        args.tokens * twin::TOP_K,
        args.tokens * twin::TOP_K,
        args.tokens * twin::TOP_K * twin::WIDTH
    );
    check_floor(args.floor_gib, "m10_gate_up start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // Activation stand-in: released embed rows, bf16-rounded, quantised with the D8 quantiser.
    let mut act_fp8 = Vec::with_capacity(args.tokens * twin::HIDDEN);
    let mut act_scales = Vec::with_capacity(args.tokens * twin::SCALE_ROW_BYTES);
    let mut block_activation: Vec<Vec<f32>> = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        let row_f32 = snap.read_row_bf16("embed.weight", token, twin::HIDDEN)?;
        let (codes, scales) = act_quant_fp8_row(&row_f32);
        act_fp8.extend_from_slice(&codes);
        act_scales.extend_from_slice(&scales);
        block_activation.push(row_f32);
    }
    println!(
        "activation stand-in: {} released embed rows x {} values, D8-quantised to E4M3 + {} ue8m0 \
         scales per row ({} + {} bytes)",
        args.tokens,
        twin::HIDDEN,
        twin::SCALE_ROW_BYTES,
        act_fp8.len(),
        act_scales.len()
    );

    // Expert weights: the rank's 576-row window of the stored [2304, 2560] I8 tensor and its
    // [2304, 160] F8_E8M0 scale tensor, read as stored for both projections.
    let mut w_gate = Vec::with_capacity(args.experts * twin::WIDTH * PACKED_ROW);
    let mut s_gate = Vec::with_capacity(args.experts * twin::WIDTH * SCALE_ROW);
    let mut w_up = Vec::with_capacity(args.experts * twin::WIDTH * PACKED_ROW);
    let mut s_up = Vec::with_capacity(args.experts * twin::WIDTH * SCALE_ROW);
    let win_packed = twin::WIDTH * PACKED_ROW;
    let win_scales = twin::WIDTH * SCALE_ROW;
    for expert in 0..args.experts {
        let base = format!("{prefix}.ffn.experts.{expert}");
        for (weight_name, scale_name, weight_out, scale_out) in [
            ("w1", "w1", &mut w_gate, &mut s_gate),
            ("w3", "w3", &mut w_up, &mut s_up),
        ] {
            let weight = snap.read_raw(
                &format!("{base}.{weight_name}.weight"),
                "I8",
                &[INTER, PACKED_ROW],
            )?;
            let scales = snap.read_raw(
                &format!("{base}.{scale_name}.scale"),
                "F8_E8M0",
                &[INTER, SCALE_ROW],
            )?;
            let row0 = args.rank * twin::WIDTH;
            let take = row0 * PACKED_ROW;
            weight_out.extend_from_slice(&weight[take..take + win_packed]);
            let take = row0 * SCALE_ROW;
            scale_out.extend_from_slice(&scales[take..take + win_scales]);
        }
    }
    let expert_bytes = w_gate.len() + s_gate.len() + w_up.len() + s_up.len();
    println!(
        "expert slices: {} experts x (w1 + w3) x {} stored rows, rank {} window {}..{}; \
         {expert_bytes} B read from the checkpoint verbatim (I8 packed FP4 + F8_E8M0)",
        args.experts,
        INTER,
        args.rank,
        args.rank * twin::WIDTH,
        (args.rank + 1) * twin::WIDTH
    );

    // Expert ids: a deterministic stand-in for the router selection. The GEMM takes ids as an
    // input; the routing decision itself is m9_router's claim, not this harness's.
    let ids: Vec<u32> = (0..args.tokens * twin::TOP_K)
        .map(|index| ((index * 7 + 3) % args.experts) as u32)
        .collect();

    // Host reference 1: the module's twin.
    let (twin_gate_bits, twin_gate_cost) = twin::gate_up(
        &act_fp8,
        &act_scales,
        &w_gate,
        &s_gate,
        &ids,
        args.tokens,
        args.experts,
    );
    let (twin_up_bits, twin_up_cost) = twin::gate_up(
        &act_fp8,
        &act_scales,
        &w_up,
        &s_up,
        &ids,
        args.tokens,
        args.experts,
    );
    println!(
        "twin cost (counted, per projection): {} MACs, weight {} B, activation {} B, scale {} B, \
         padding writes {}",
        twin_gate_cost.macs,
        twin_gate_cost.weight_bytes,
        twin_gate_cost.activation_bytes,
        twin_gate_cost.scale_bytes,
        twin_gate_cost.padding_writes
    );
    let mut verdicts: Vec<Verdict> = Vec::new();
    verdicts.push(Verdict::new(
        "twin cost is the geometry's arithmetic",
        twin_gate_cost.macs == (args.tokens * twin::TOP_K * twin::WIDTH * twin::HIDDEN) as u64
            && twin_up_cost.macs == twin_gate_cost.macs
            && twin_gate_cost.padding_writes
                == (args.tokens * twin::TOP_K * (twin::WIDTH_PADDED - twin::WIDTH)) as u64,
        format!(
            "{} MACs per projection, {} padding zero writes",
            twin_gate_cost.macs, twin_gate_cost.padding_writes
        ),
    ));

    // Host reference 2: the shared b2_harness arithmetic on the same dequantised activation. The two
    // host implementations must agree bit for bit, or the device comparison has no meaning. This is
    // an ordering cross-check: b2_harness shares the tree's FP4 loader, so it does not independently
    // check the decoders. Then the engine-style f32 control (unquantised activation) for the gap.
    let dequant_rows: Vec<Vec<f32>> = (0..args.tokens)
        .map(|token| {
            (0..twin::HIDDEN)
                .map(|index| {
                    fp8_e4m3_decode(act_fp8[token * twin::HIDDEN + index])
                        * ue8m0_decode(act_scales[token * twin::SCALE_ROW_BYTES + index / 32])
                })
                .collect()
        })
        .collect();
    let mut shared_gate = Vec::with_capacity(args.tokens * twin::TOP_K * twin::WIDTH);
    let mut shared_up = Vec::with_capacity(args.tokens * twin::TOP_K * twin::WIDTH);
    let mut control_gate = RealColumns::with_capacity(args.tokens * twin::TOP_K * twin::WIDTH);
    let mut control_up = RealColumns::with_capacity(args.tokens * twin::TOP_K * twin::WIDTH);
    for (routed_row, expert) in ids.iter().enumerate() {
        let token = routed_row / twin::TOP_K;
        let expert = *expert as usize;
        let weight_gate = &w_gate[expert * win_packed..(expert + 1) * win_packed];
        let scale_gate = &s_gate[expert * win_scales..(expert + 1) * win_scales];
        let weight_up = &w_up[expert * win_packed..(expert + 1) * win_packed];
        let scale_up = &s_up[expert * win_scales..(expert + 1) * win_scales];
        // The rank window is already the loaded slice: these helpers see the same 576 rows the
        // kernel will, so the comparison is over identical operands.
        shared_gate.extend(gate_up_rows(
            &dequant_rows[token],
            weight_gate,
            scale_gate,
            twin::WIDTH,
            twin::HIDDEN,
        ));
        shared_up.extend(gate_up_rows(
            &dequant_rows[token],
            weight_up,
            scale_up,
            twin::WIDTH,
            twin::HIDDEN,
        ));
        control_gate.extend(gate_up_rows(
            &block_activation[token],
            weight_gate,
            scale_gate,
            twin::WIDTH,
            twin::HIDDEN,
        ));
        control_up.extend(gate_up_rows(
            &block_activation[token],
            weight_up,
            scale_up,
            twin::WIDTH,
            twin::HIDDEN,
        ));
    }
    let shared_gate_bits: Vec<u16> = shared_gate.iter().map(|value| bf16_bits(*value)).collect();
    let shared_up_bits: Vec<u16> = shared_up.iter().map(|value| bf16_bits(*value)).collect();
    let twin_gate_f32: Vec<f32> = twin_gate_bits
        .iter()
        .map(|bits| bf16_value(*bits))
        .collect();
    let twin_up_f32: Vec<f32> = twin_up_bits.iter().map(|bits| bf16_value(*bits)).collect();
    let mut shared_mismatches = 0usize;
    for routed_row in 0..args.tokens * twin::TOP_K {
        for column in 0..twin::WIDTH {
            let at = routed_row * twin::WIDTH + column;
            if twin_gate_bits[routed_row * twin::WIDTH_PADDED + column] != shared_gate_bits[at] {
                shared_mismatches += 1;
            }
            if twin_up_bits[routed_row * twin::WIDTH_PADDED + column] != shared_up_bits[at] {
                shared_mismatches += 1;
            }
        }
    }
    // The control is engine-style f32 (no padding columns), the twin is the ABI's padded rows:
    // compare the 576 real columns only, never the padding tail.
    let compared_rows = args.tokens * twin::TOP_K;
    let (gate_gap, _) = control_gap(&twin_gate_f32, &control_gate, compared_rows);
    let (up_gap, _) = control_gap(&twin_up_f32, &control_up, compared_rows);
    let control_max = max_abs(control_gate.as_slice()).max(max_abs(control_up.as_slice()));
    println!(
        "host: twin vs shared-harness arithmetic {shared_mismatches} mismatched bf16 values of {} \
         real outputs per projection; engine-style f32 control gap over the {} real columns only: \
         gate {gate_gap:.6e}, up {up_gap:.6e} (control max {control_max:.6e})",
        compared_rows * twin::WIDTH,
        compared_rows * twin::WIDTH
    );
    verdicts.push(Verdict::new(
        "twin is the shared harness arithmetic",
        shared_mismatches == 0,
        format!("{shared_mismatches} mismatched values"),
    ));
    let gap_ok = gate_gap <= 0.10 * control_max && up_gap <= 0.10 * control_max;
    verdicts.push(Verdict::new(
        "quantisation gap within 10% of the control maximum",
        gap_ok,
        format!("gate {gate_gap:.6e}, up {up_gap:.6e}, control max {control_max:.6e}"),
    ));

    // The device.
    let cubin_path = args
        .cubin_dir
        .join(format!("{}.cubin", abi::GATE_UP_KERNEL));
    let cubin = std::fs::read(&cubin_path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels \
             kernels/routed_gate_up_sm121.cu -o out/routed_gate_up_sm121.cubin)",
            cubin_path.display()
        )
    })?;
    let digest = Sha256::digest(&cubin);
    let sha = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "cubin: {} ({} B, sha256 {sha})",
        cubin_path.display(),
        cubin.len()
    );
    let cubin_ok = cubin.len() == abi::CUBIN_BYTES && sha == abi::CUBIN_SHA256;
    verdicts.push(Verdict::new(
        "cubin is the pinned one",
        cubin_ok,
        format!(
            "expected {} B / {}, got {} B / {sha}",
            abi::CUBIN_BYTES,
            abi::CUBIN_SHA256,
            cubin.len()
        ),
    ));
    if !cubin_ok {
        return Err(
            "the cubin does not match the pinned sha256; refusing to measure it".to_owned(),
        );
    }

    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let act_d = cuda.alloc(act_fp8.len(), &mut allocated)?;
    let scales_d = cuda.alloc(act_scales.len(), &mut allocated)?;
    let w_gate_d = cuda.alloc(w_gate.len(), &mut allocated)?;
    let s_gate_d = cuda.alloc(s_gate.len(), &mut allocated)?;
    let w_up_d = cuda.alloc(w_up.len(), &mut allocated)?;
    let s_up_d = cuda.alloc(s_up.len(), &mut allocated)?;
    let ids_d = cuda.alloc(ids.len() * 4, &mut allocated)?;
    let out_bytes = args.tokens * twin::TOP_K * twin::WIDTH_PADDED * 2;
    let gate_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    let up_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    println!(
        "device buffers: {allocated} bytes ({} MiB) admitted",
        allocated / (1024 * 1024)
    );
    let mut ids_bytes = Vec::with_capacity(ids.len() * 4);
    for id in &ids {
        ids_bytes.extend_from_slice(&id.to_le_bytes());
    }
    cuda.copy_htod(&act_d, &act_fp8)?;
    cuda.copy_htod(&scales_d, &act_scales)?;
    cuda.copy_htod(&w_gate_d, &w_gate)?;
    cuda.copy_htod(&s_gate_d, &s_gate)?;
    cuda.copy_htod(&w_up_d, &w_up)?;
    cuda.copy_htod(&s_up_d, &s_up)?;
    cuda.copy_htod(&ids_d, &ids_bytes)?;
    let (_module, kernel) = routed_gate_up_launch::load(&cuda, &cubin)?;
    let gate_params = GateUpLaunchParams {
        act_fp8: act_d,
        act_scales: scales_d,
        w_fp4: w_gate_d,
        w_scales: s_gate_d,
        expert_ids: ids_d,
        out: gate_out_d,
        tokens: args.tokens as u32,
        trace_ring: 0, // the shipped cubin is the trace-disabled build
        layer_id: args.layer as u32,
        step_seq: 0,
    };
    let up_params = GateUpLaunchParams {
        w_fp4: w_up_d,
        w_scales: s_up_d,
        out: up_out_d,
        ..gate_params
    };
    println!(
        "launch grid: {:?} blocks of {} threads ({} routed rows x {} tiles)",
        routed_gate_up_launch::grid(args.tokens as u32)?,
        abi::THREADS,
        args.tokens * twin::TOP_K,
        abi::TILES
    );
    routed_gate_up_launch::launch(&cuda, kernel, &gate_params)?;
    routed_gate_up_launch::launch(&cuda, kernel, &up_params)?;

    let read_out = |cuda: &Cuda| -> Result<(Vec<u8>, Vec<u8>), String> {
        let mut gate = vec![0u8; out_bytes];
        let mut up = vec![0u8; out_bytes];
        cuda.copy_dtoh(&mut gate, &gate_out_d)?;
        cuda.copy_dtoh(&mut up, &up_out_d)?;
        Ok((gate, up))
    };
    let (gate_bytes_first, up_bytes_first) = read_out(&cuda)?;

    // Determinism: a second pair of launches, and the bytes must not move.
    routed_gate_up_launch::launch(&cuda, kernel, &gate_params)?;
    routed_gate_up_launch::launch(&cuda, kernel, &up_params)?;
    let (gate_bytes_second, up_bytes_second) = read_out(&cuda)?;
    let deterministic = gate_bytes_first == gate_bytes_second && up_bytes_first == up_bytes_second;

    let device_gate_bits = bf16_bits_of_bytes(&gate_bytes_first);
    let device_up_bits = bf16_bits_of_bytes(&up_bytes_first);
    let mut exact_gate = 0usize;
    let mut exact_up = 0usize;
    let mut padding_nonzero = 0usize;
    let rows = args.tokens * twin::TOP_K;
    for routed_row in 0..rows {
        for column in 0..twin::WIDTH_PADDED {
            let at = routed_row * twin::WIDTH_PADDED + column;
            if device_gate_bits[at] == twin_gate_bits[at] {
                exact_gate += 1;
            }
            if device_up_bits[at] == twin_up_bits[at] {
                exact_up += 1;
            }
            if column >= twin::WIDTH && (device_gate_bits[at] != 0 || device_up_bits[at] != 0) {
                padding_nonzero += 1;
            }
        }
    }
    let total = rows * twin::WIDTH_PADDED;
    println!(
        "device vs twin: gate {exact_gate}/{total} bf16 values exact, up {exact_up}/{total} exact; \
         padding nonzero {padding_nonzero}; two runs byte-identical: {deterministic}"
    );
    verdicts.push(Verdict::new(
        "device gate is bit-identical to the twin",
        exact_gate == total,
        format!("{exact_gate}/{total}"),
    ));
    verdicts.push(Verdict::new(
        "device up is bit-identical to the twin",
        exact_up == total,
        format!("{exact_up}/{total}"),
    ));
    verdicts.push(Verdict::new(
        "the padding columns are exactly zero",
        padding_nonzero == 0,
        format!("{padding_nonzero} nonzero padding values"),
    ));
    verdicts.push(Verdict::new(
        "two device runs are byte-identical",
        deterministic,
        if deterministic {
            "byte-identical".to_owned()
        } else {
            "the bytes moved between runs".to_owned()
        },
    ));

    // The launch's own declared traffic: this rank's gate (or up) slice and scales plus the
    // activation row and its scales. Quoted to the roofline line below, not to a claim.
    let launch_traffic_bytes =
        (w_gate.len() + s_gate.len() + act_fp8.len() + act_scales.len()) as u64;
    if args.repeat > 0 {
        let (gate_samples, gate_batched) =
            routed_gate_up_launch::launch_repeated(&cuda, kernel, &gate_params, args.repeat)?;
        let (up_samples, up_batched) =
            routed_gate_up_launch::launch_repeated(&cuda, kernel, &up_params, args.repeat)?;
        let (min, p50, mean, max) = timing_stats(&gate_samples);
        println!(
            "gate timing, {} launches: min {min:.4} p50 {p50:.4} mean {mean:.4} max {max:.4} ms; \
             batched {gate_batched:.4} ms/launch; {:.4} ms/token",
            args.repeat,
            p50 / args.tokens as f64
        );
        println!(
            "{}",
            sparkflash_kernels::roofline::line(
                "routed_gate_up_sm121/w1",
                launch_traffic_bytes,
                p50
            )
        );
        let (min, p50_up, mean, max) = timing_stats(&up_samples);
        println!(
            "up timing, {} launches: min {min:.4} p50 {p50_up:.4} mean {mean:.4} max {max:.4} ms; \
             batched {up_batched:.4} ms/launch; {:.4} ms/token",
            args.repeat,
            p50_up / args.tokens as f64
        );
        println!(
            "{}",
            sparkflash_kernels::roofline::line(
                "routed_gate_up_sm121/w3",
                launch_traffic_bytes,
                p50_up
            )
        );
    }

    println!("--- verdicts (bar first, then numbers) ---");
    let mut failed = 0usize;
    for verdict in &verdicts {
        let tag = if verdict.ok { "PASS" } else { "FAIL" };
        println!("{tag}  {}: {}", verdict.name, verdict.detail);
        if !verdict.ok {
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(format!("{failed} pass-bar check(s) failed"));
    }
    println!(
        "m10_gate_up: PASS. Class: device correctness of one routed gate/up projection against \
         released layer-0 expert bytes, exact against the CPU twin. NOT claimed: routing weight, \
         SwiGLU, combine, shared expert, end-to-end MoE, or any host-time saving."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control gap compares real columns, never the padded row: the first window run panicked
    /// because the twin's rows are 640 wide and the engine-style control is 576 wide.
    #[test]
    fn the_control_comparison_uses_only_the_real_columns() {
        for tokens in [1usize, 8, 130] {
            let rows = tokens * twin::TOP_K;
            let control = rows * twin::WIDTH;
            let padded = rows * twin::WIDTH_PADDED;
            assert_eq!(control, rows * 576);
            assert_eq!(padded, rows * 640);
            assert_ne!(control, padded);
            let buffer: Vec<f32> = (0..padded).map(|index| index as f32).collect();
            let real = real_columns(&buffer, rows);
            assert_eq!(
                real.len(),
                control,
                "the compared length is the real columns"
            );
            // Row r's real columns are buffer[r * 640 .. r * 640 + 576].
            for row in 0..rows {
                for column in 0..twin::WIDTH {
                    assert_eq!(
                        real[row * twin::WIDTH + column],
                        (row * twin::WIDTH_PADDED + column) as f32
                    );
                }
            }
        }
    }

    /// The exact numbers the first window run printed: 8 tokens gave 30,720 padded values against
    /// 27,648 real ones, and 130 tokens repeated the mismatch. Pinned so the lengths cannot drift.
    #[test]
    fn the_window_panic_lengths_are_pinned() {
        assert_eq!(8 * twin::TOP_K * twin::WIDTH_PADDED, 30_720);
        assert_eq!(8 * twin::TOP_K * twin::WIDTH, 27_648);
        assert_eq!(130 * twin::TOP_K * twin::WIDTH_PADDED, 499_200);
        assert_eq!(130 * twin::TOP_K * twin::WIDTH, 449_280);
    }

    /// The gap comparison must be length-checked before it iterates: a padded twin buffer against a
    /// real-column control is exactly the panic, so the helper strips the padding first.
    #[test]
    fn the_padded_and_real_lengths_are_distinct_for_every_window_batch() {
        for tokens in [1usize, 8, 130] {
            let rows = tokens * twin::TOP_K;
            let real = real_columns(&vec![0.0f32; rows * twin::WIDTH_PADDED], rows);
            assert_eq!(real.len(), rows * twin::WIDTH);
            assert_ne!(real.len(), rows * twin::WIDTH_PADDED);
        }
    }

    /// The function the run actually calls. A 640-wide padded twin against a 576-wide control must
    /// not panic, must ignore the padding tail even when the tail is enormous, and must report a real
    /// column that differs. The call site cannot regress to `max_abs_delta(&twin_padded, &control)`:
    /// `RealColumns` is not `&[f32]`, so that line no longer compiles.
    #[test]
    fn the_control_gap_slices_the_padded_twin_at_every_window_batch() {
        for tokens in [1usize, 8, 130] {
            let rows = tokens * twin::TOP_K;
            let mut twin_padded = vec![0.5f32; rows * twin::WIDTH_PADDED];
            let mut control = RealColumns(vec![0.5f32; rows * twin::WIDTH]);
            // A padding tail that would dominate the gap if it were compared.
            for row in 0..rows {
                for column in twin::WIDTH..twin::WIDTH_PADDED {
                    twin_padded[row * twin::WIDTH_PADDED + column] = 1.0e6;
                }
            }
            assert_eq!(control_gap(&twin_padded, &control, rows), (0.0, 0));
            // One real column (row 0, column 7) differs by 1.5: the gap reports it at index 7.
            twin_padded[7] = 2.5;
            control.0[7] = 1.0;
            let (gap, at) = control_gap(&twin_padded, &control, rows);
            assert!((gap - 1.5).abs() < f32::EPSILON, "gap was {gap}");
            assert_eq!(at, 7);
        }
    }
}
