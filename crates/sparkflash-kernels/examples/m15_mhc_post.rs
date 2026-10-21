//! m15_mhc_post: the layer-0 Mega-mHC POST-attention merge (hc_post) in isolation
//! (kernels/mega_mhc_sm121.cu) - the complement of m11's entry - against released layer-0
//! checkpoint bytes.
//!
//! Scope, stated before the numbers. This harness launches ONE fused kernel call at the post
//! seam of layer 0: the 19-word pinned declaration of mega_mhc_fused_bf16_sm121, with the entry
//! stage's 24 published coefficients (B = comb, A = pre, C = post) as b_prev/c_prev/a_prev, the
//! attention block output as y_prev, and the entry stage's residual streams as x. It measures the
//! call's four comparisons: the post-mix residual-stream write X_1 = B X_0 + C Y_0 (the hc_post
//! step), the block-input capture X_hat_1 = A X_1 and its 1/rms, the E4M3 codes and ue8m0 scales
//! of that capture, and the 24 coefficients the fused kernel publishes from the updated streams.
//! It is not attention, not the attention output projection, not a chain pass, not a token, and
//! not end-to-end text.
//!
//! The call shape is declared, not inferred: the kernel has ONE entry, so the post call uses the
//! same 16 non-null device pointers as m11's entry call. What differs is what the host puts in
//! them: x = X_0 (the entry stage's write), y_prev = Y_0 (the attention block output), and
//! b_prev/c_prev/a_prev = the entry stage's coefficient readback. The harness declares every
//! input as host-provided and does not assume the chain's transport is solved - the device
//! wiring refuses MhcPost until a device-to-device coefficient handoff exists
//! (crates/sparkflash-device/src/chain_device.rs), and this harness copies both seams through the
//! host precisely because that handoff does not exist:
//!   * X_0 is the host twin of the entry stage's output on released bytes: the released embed row
//!     t, four copies, through the released layer-0 boundary seed (B = identity, C = 0,
//!     A = [1,0,0,0], Y_-1 = 0). The kernel's step 1 (X_l = B X_{l-1} + C Y_{l-1}) makes that write
//!     the identity on this seed; the harness checks the declared bytes against that write in the
//!     kernel's own arithmetic, not against the reference's first return value (which is the
//!     reference's B X_l + C F(X_hat_l) traffic-accounting self-loop, a different quantity).
//!   * B/A/C are the host twin of the entry stage's readback: mega_mhc_ref on the same X_0 and the
//!     same released hc_attn_fn/base/scale. These are the bytes the post call consumes.
//!   * Y_0 is a DECLARED attention-output stand-in, not a measured attention activation: no stage
//!     in this harness computes attention, and the released checkpoint ships no attention output.
//!     It is built from released bytes, bf16(0.5 * embed row t - 0.25 * embed row (t+1) % tokens),
//!     so C_0 Y_0 is a nonzero released-bytes-derived operand; its provenance is a stand-in and
//!     the seam it occupies is the attention block output.
//!
//! Explicitly NOT claimed:
//!   * any attention work, inverse RoPE, wo_a/wo_b or attention output value (Y_0 is a stand-in);
//!   * any chain ordering, seam ledger, per-layer schedule or device-to-device coefficient handoff;
//!   * the 24 coefficients the kernel publishes here as released LAYER-1 coefficients: the kernel
//!     applies this layer's hc tensors to the updated streams, so they are the H arithmetic on X_1
//!     under layer-0's installed module, not another layer's checkpoint tensors;
//!   * any token, any text, any end-to-end layer or model output;
//!   * routing, SwiGLU, the MoE block, the shared expert, the combine;
//!   * weight staging / device-budget admission / engine integration;
//!   * a host-time saving of any kind.
//!
//! COMPARISON CLASS (declared before any number is printed). The CPU twin is
//! mega_mhc_ref::mega_mhc_layer_with_scale on the SAME f32 values the device updates from - the
//! post-mix streams formed at the KERNEL'S OWN INSTRUCTION LEVEL: per stream the first B product is
//! the one rounded (a plain mul.f32), the C Y term is fused into it, then the remaining B terms are
//! fma.rn.f32 in stream order - then the released H/projection/quantisation arithmetic. Window b16
//! measured the cost of getting this wrong: with the C Y term rounded and the first B term fused
//! (the source's literal order) the bf16 write stayed bit-exact while the f32 capture differed on
//! every position. Per position:
//!   bit-identical:  the post-mix residual write X_1 after bf16 rounding and the f32 block-input
//!                   capture xhat_normed (computed from the same inputs and the device's own
//!                   1/rms), because the twin runs the same instruction sequence as the device
//!                   (the pinned PTX has 241 fma.rn.f32 and no mul.rn/add.rn; the
//!                   arrangement above is the one ptxas saw). The 160 ue8m0 scale bytes are also
//!                   exact, but NOT because the rms program is identical: the twin's 1/rms is a
//!                   sequential reduction and the device's is tile-order, so the scale is exact
//!                   because the ue8m0 exponent is coarse enough that the last-bit difference does
//!                   not move the block's scale code;
//!   declared bars:  the 5,120 E4M3 codes (>= 99.5% exact and every code within 1) against the
//!                   twin's own quantisation of its fma X_hat and 1/rms (not the reference's
//!                   mul-then-add X_hat), the 24 mixed coefficients (max abs delta <= 1e-4, against
//!                   the reference's sequential projection), and 1/rms(X_hat_1) (<= 3e-4). The bars
//!                   are not looseness: the kernel reduces the two sums of squares and the 26
//!                   projection statistics in a fixed tile order (four tiles, tree reduction per
//!                   tile, binary64 combination in tile order), while the twin and the reference
//!                   accumulate sequentially, so those quantities are not the same floating-point
//!                   program.
//!   determinism:    two device runs of every position are byte-identical across all six outputs;
//!   f32 control:    the kernel's own E4M3 codes x ue8m0 scales, decoded, against the kernel's own
//!                   f32 block-input capture - the FP8 quantisation gap, bounded at 10% of that
//!                   capture's maximum absolute value (the m10 control style). Reported separately,
//!                   not part of the exactness claim.
//!
//! Cubin: the pinned mega_mhc_sm121 cubin is required - 141,632 B and
//! 7040411617954a17e89e6d9d8fcea1a872171de259a5515bde1e69a96c69af5c, read from the tree's
//! baseline manifest (sparkflash_kernels::baseline::pin), and the harness refuses anything else by
//! length and sha256 before the first DEVICE allocation. Build it under the pinned toolkit (CUDA 13.0.3,
//! nvcc V13.0.88):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/mega_mhc_sm121.cu \
//!       -o out/mega_mhc_sm121.cubin
//!
//! Tokens: this kernel has no batch dimension - one position per launch - so --tokens N launches
//! N independent layer-0 positions (released embed rows 0..N as the X_-1 seed of the entry twin,
//! each position's own coefficient readback as the handoff bytes). N = 8 and N = 130 are both
//! valid shapes; 130 is the fixture prefill length, and nothing about the kernel changes with N.
//! The harness refuses to run below 32 GiB MemAvailable (--floor-gib) at start and end.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, bytes_to_f32};
use sparkflash_kernels::baseline;
use sparkflash_kernels::m4_quant::{
    QUANT_BLOCK, act_quant_scale_code, bf16_bits, bf16_value, fp8_e4m3_decode, fp8_e4m3_encode,
    ue8m0_decode,
};
use sparkflash_kernels::mega_mhc_launch::{self, MegaMhcParams};
use sparkflash_kernels::mega_mhc_ref::{
    Coefficients, HC_MULT, HIDDEN, NORM_EPS, PREDICTOR_ROWS, Streams, TrafficLedger,
    bias_into_kernel_rows, block_input_normed, mega_mhc_layer_with_scale,
    released_predictor_into_kernel_rows,
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
        eprintln!("m15_mhc_post: {error}");
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
                        "usage: m15_mhc_post --checkpoint <snapshot dir> [--cubin-dir out] \
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
                "--layer must be 0: this harness consumes the layer-0 released tensors and the \
                 layer-0 entry twin's readback as its seam bytes; a later layer needs that \
                 layer's staged hc tensors and its own entry readback, which this isolated harness \
                 does not have (staging is not staged here)"
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

/// The post-mix residual update the kernel writes, X_1 = B X_0 + C Y_0, at the KERNEL's own
/// instruction level, which is what [f32::mul_add] reproduces exactly.
///
/// This is not the source-level expression one would write by hand. The source is
/// `acc = c[s] * y; for t: acc += b[s][t] * x[t]`, and nvcc contracts it into this sequence per
/// stream, confirmed in the PTX of the pinned kernel (241 `fma.rn.f32`, no `mul.rn.f32`, no
/// `add.rn.f32`; compiled with the pinned 13.0.3 toolkit):
///
///     q     = b[s][0] * x[0]                 // the ONE plain mul.f32: THIS product is rounded
///     acc   = fma(c[s], y, q)                // the C Y term is the FUSED one
///     acc   = fma(b[s][1], x[1], acc)
///     acc   = fma(b[s][2], x[2], acc)
///     acc   = fma(b[s][3], x[3], acc)
///
/// The arrangement matters at f32 precision: rounding `c[s] * y` first and fusing the first B
/// term instead (the source's literal order) is a different floating-point program, and window b16
/// measured exactly its shadow - the bf16 write X_1 was bit-exact (rounding to bf16 hides a last
/// f32 bit) while the f32 block-input capture differed on every position, because X_hat sums those
/// same f32 values. Two tests pin the arrangement: the_post_update_matches_the_kernel_instruction_order
/// separates it from the C-Y-first order and from a twin with no fusion at all, and
/// the_first_b_product_is_the_rounded_one pins the mirror-image mistake (fusing the first B
/// product instead of rounding it).
fn expected_update(
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
            // The one non-fused product, then the fused C Y term, then the remaining B terms.
            let mut value = b[stream][0] * x[column];
            value = c[stream].mul_add(y[column], value);
            for source in 1..HC_MULT {
                value = b[stream][source].mul_add(x[source * hidden + column], value);
            }
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
            value = weight.mul_add(x[stream * hidden + column], value);
        }
        squares = value.mul_add(value, squares);
        *slot = value;
    }
    (xhat, 1.0f32 / (squares / hidden as f32 + NORM_EPS).sqrt())
}

/// The released activation quantisation of the block input, in the KERNEL's order, built on the
/// twin's own fma X_hat and the twin's own 1/rms(X_hat): `normed = (xhat * w) * inv_rms` as two
/// plain multiplies, one block amax, the released act-quant scale code, then
/// `fp8_e4m3_encode(normed * (1 / scale))` - a MULTIPLY by the reciprocal, which is what the
/// kernel writes (kernel lines 328-343), not a division.
///
/// This exists because the reference function's X_hat is a mul-then-add chain (Rust never contracts
/// the `+= weight * element` in mega_mhc_ref), while the device's is fma.rn; comparing the device
/// codes against the reference's codes would therefore declare a twin the harness does not use for
/// the capture, and the "code-difference cause" line would cite the wrong 1/rms. Both quantities
/// now come from the same twin: [host_xhat] and its rms.
fn expected_quantisation(
    xhat: &[f32],
    inv_rms: f32,
    rms_weight: &[f32],
) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    assert_eq!(xhat.len(), HIDDEN);
    assert_eq!(rms_weight.len(), HIDDEN);
    let normed: Vec<f32> = xhat
        .iter()
        .zip(rms_weight.iter())
        .map(|(value, weight)| value * weight * inv_rms)
        .collect();
    let blocks = HIDDEN / QUANT_BLOCK;
    let mut scales = Vec::with_capacity(blocks);
    let mut codes = Vec::with_capacity(HIDDEN);
    for block in 0..blocks {
        let mut amax = 0.0f32;
        for value in &normed[block * QUANT_BLOCK..(block + 1) * QUANT_BLOCK] {
            let magnitude = value.abs();
            amax = if magnitude > amax { magnitude } else { amax };
        }
        let scale_code = act_quant_scale_code(amax);
        scales.push(scale_code);
        let inverse_scale = 1.0f32 / ue8m0_decode(scale_code);
        for value in &normed[block * QUANT_BLOCK..(block + 1) * QUANT_BLOCK] {
            codes.push(fp8_e4m3_encode(value * inverse_scale));
        }
    }
    (codes, scales, normed)
}

/// The expected E4M3 codes and ue8m0 scales for one position - the ONE producer of those expected
/// bytes in the harness, so the comparison cannot silently fall back to the reference's mul-then-add
/// X_hat: revert the call site and this function becomes dead code (clippy -D warnings) while
/// the_codes_twin_is_the_fma_twin_not_the_reference still proves the twin is not the reference's.
fn twin_quantisation(xhat: &[f32], inv_rms: f32, norm: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let (codes, scales, _normed) = expected_quantisation(xhat, inv_rms, norm);
    (codes, scales)
}

/// The DECLARED attention-output stand-in: Y_0 = 0.5 * embed row t - 0.25 * embed row t+1 (the
/// next row wraps at N). It is built from released embed bytes so the operand is real data, but it
/// is NOT a measured attention output and no attention arithmetic is claimed; the seam it stands
/// in for is the attention block output that hc_post consumes.
fn attention_output_stand_in(row: &[f32], next_row: &[f32], hidden: usize) -> Vec<f32> {
    assert_eq!(row.len(), hidden);
    assert_eq!(next_row.len(), hidden);
    row.iter()
        .zip(next_row)
        .map(|(current, next)| 0.5 * current - 0.25 * next)
        .collect()
}

/// One position's eight boundary checks, named so the conjunction can be tested on its own (a
/// deleted term must fail a test, not slip through the example suite).
struct PositionChecks {
    fp8_ok: bool,
    scales_ok: bool,
    streams_ok: bool,
    coefficients_ok: bool,
    rms_ok: bool,
    capture_ok: bool,
    deterministic: bool,
    control_ok: bool,
}

/// A position passes only if EVERY check passes; this is the pure conjunction the verdict uses.
fn post_position_ok(checks: PositionChecks) -> bool {
    checks.fp8_ok
        && checks.scales_ok
        && checks.streams_ok
        && checks.coefficients_ok
        && checks.rms_ok
        && checks.capture_ok
        && checks.deterministic
        && checks.control_ok
}

/// Whether the entry twin's 24 published coefficients are inside the released activation ranges:
/// B comes out of the split Sinkhorn (so each entry is in [0, 1]), A is sigmoid + hc_eps (so in
/// (0, 1 + eps]) and C is 2 * sigmoid (so in (0, 2]). A readback outside these ranges is not the
/// bytes the entry stage publishes, and the harness refuses rather than passing them to the kernel.
fn entry_coefficients_are_in_released_ranges(coefficients: &Coefficients) -> bool {
    let b_ok = coefficients
        .b
        .iter()
        .flatten()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value));
    let a_ok = coefficients
        .a
        .iter()
        .all(|value| value.is_finite() && *value > 0.0 && *value <= 1.001);
    let c_ok = coefficients
        .c
        .iter()
        .all(|value| value.is_finite() && *value > 0.0 && *value <= 2.001);
    b_ok && a_ok && c_ok
}

/// The E4M3 exact-fraction bar: the m7 stage-1 class for this kernel, unchanged.
const CODES_EXACT_PCT_BAR: f64 = 99.5;
/// The largest per-code distance a position may show and stay inside the class.
const CODE_MAX_DELTA: f32 = 1.0;
/// The 24 coefficients' max abs delta bar (the m7 stage-1 class).
const COEF_BAR: f32 = 1e-4;
/// The 1/rms(X_hat) max abs delta bar (the m7 stage-1 class).
const RMS_BAR: f32 = 3e-4;
/// The FP8 quantisation control: the gap may not exceed this fraction of the capture maximum.
const CONTROL_BAR: f32 = 0.10;

/// The control predicate, as a function so the bar is one comparison the test can pin.
fn control_gap_within_bar(gap: f32, capture_max: f32) -> bool {
    gap <= CONTROL_BAR * capture_max
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
    println!("=== m15_mhc_post: layer-0 Mega-mHC post-mix (hc_post) in isolation ===");
    println!(
        "quantities compared per position: 4 x {HIDDEN} residual-stream f32 (written as bf16), \
         {HIDDEN} E4M3 codes, {HIDDEN} / {QUANT_BLOCK} ue8m0 scale bytes, 24 mixed coefficients \
         (the kernel's H(X_1) byproduct under layer-0's installed tensors), 1/rms(X_hat_1), \
         {HIDDEN} f32 block-input capture"
    );
    println!(
        "comparison class (declared before any number): device vs the CPU twin on the same bytes \
         the device updates from, per position - BIT-IDENTICAL on the post-mix residual write X_1 \
         after bf16 rounding (both sides use the kernel's exact instruction order: the FIRST B \
         product rounded, the C Y term fused into it, then the remaining B terms fused in stream \
         order - not the source's literal C-Y-first order), on the 160 ue8m0 scale bytes (quantised \
         with the released act-quant rule from the twin's own fma X_hat and 1/rms), and on the f32 \
         block-input capture computed from the same inputs and the device's own 1/rms; \
         DECLARED BARS on the E4M3 codes (>= {CODES_EXACT_PCT_BAR}% exact, every code within \
         {CODE_MAX_DELTA}, against the twin's own quantisation of its fma X_hat and 1/rms - not \
         the reference's mul-then-add X_hat), on the 24 coefficients (max abs delta <= {COEF_BAR:e}) \
         and on 1/rms (<= {RMS_BAR:e}) because the kernel's fixed tile-order reduction (four tiles, \
         tree per tile, binary64 combination in tile order) and the twin/reference sequential \
         accumulation are not the same floating-point program. Determinism: two device runs of \
         every position \
         byte-identical. Separate control: the FP8 quantisation gap of the kernel's own block \
         input, bounded at 10% of its capture maximum. HOST-PROVIDED, DECLARED SEAMS (not \
         measured, and the chain does not provide them yet): X_0 = the entry twin's residual write \
         on the released embed row and the released layer-0 boundary seed; B/A/C = the entry \
         twin's coefficient readback on the same bytes (the future device-to-device handoff, which \
         chain_device.rs refuses until it exists); Y_0 = the declared attention-output stand-in \
         bf16(0.5 * embed row t - 0.25 * embed row (t+1) % tokens). NOT claimed here: attention, \
         inverse RoPE, wo_a/wo_b, any chain or per-layer schedule, any token or text, \
         routing/SwiGLU/MoE/shared expert, weight staging, device budget, engine integration, \
         host-time saving."
    );
    println!(
        "tokens: {0} independent layer-0 positions, one launch each (the kernel has no batch \
         dimension; the grid and the argument shape are identical for every position, only the \
         host-copied seam bytes change)",
        args.tokens
    );
    check_floor(args.floor_gib, "m15_mhc_post start")?;

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
    let (seed_b, seed_c, seed_a, seed_y) = layer_zero_seed();
    println!(
        "entry boundary seed (host-built, declared): B = identity ({}), C = 0, A = {:?}, Y_-1 = 0 \
         - it defines the X_0 and B/A/C seams this harness copies; the post call itself consumes \
         the entry twin's outputs, not the seed",
        HC_MULT * HC_MULT,
        seed_a
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

    cuda.copy_htod(&predictor_d, &f32_to_bytes(&mhc.predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&mhc.hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&mhc.hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;

    // The post call's argument shape: the same 16 non-null device pointers and 19 words as m11's
    // entry call (the kernel has one entry). b/c/a are re-uploaded per position because the entry
    // readback is per position; y is re-uploaded per position because Y_0 is per position.
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
        let next_row = snap.read_row_bf16("embed.weight", (token + 1) % args.tokens, HIDDEN)?;

        // The entry twin: X_0 and the 24 coefficients the entry stage publishes. Both are
        // host-provided seam bytes for the post call; the chain's handoff does not exist yet.
        let mut x_entry = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x_entry.extend_from_slice(&row);
        }
        let x_bits: Vec<u16> = x_entry.iter().map(|value| bf16_bits(*value)).collect();
        let x_bytes = bf16_bits_to_bytes(&x_bits);
        let x_values: Vec<f32> = x_bits.iter().map(|bits| bf16_value(*bits)).collect();

        // The entry twin's WRITE, in the kernel's own arithmetic: the kernel's step 1 is
        // X_l = B X_{l-1} + C Y_{l-1}, so on the released layer-0 boundary seed (B = identity,
        // C = 0, Y_-1 = 0) the entry call writes X_0 = X_-1, i.e. four copies of the embed row.
        // That is the X_0 this harness hands to the post call as x, and the guard below checks the
        // bytes actually passed equal that write in the kernel's arithmetic.
        //
        // It deliberately does NOT use mega_mhc_layer_with_scale's first return value: the
        // reference returns its own X_{l+1} = B X_l + C F(X_hat_l) self-loop (F the dequantised
        // codes it just wrote), built from the coefficients it just formed, which is a traffic-
        // accounting quantity and NOT the kernel's write for this call (the kernel applies the
        // PREVIOUS call's b_prev/c_prev). Window b15's guard compared that reference return to X_0
        // and refused, which is why the harness now checks the declared write directly.
        let entry_write = expected_update(&x_values, &seed_b, &seed_c, &seed_y, HIDDEN);
        let entry_write_bits: Vec<u16> =
            entry_write.iter().map(|value| bf16_bits(*value)).collect();
        if entry_write_bits != x_bits {
            return Err(format!(
                "position {token}: the entry twin's write on the released boundary seed is not the \
                 declared X_0 bytes; refusing to pass a different stream buffer as the post seam's x"
            ));
        }

        // The 24 coefficients the post call consumes are the entry stage's published readback. The
        // reference computes H(X_0) on the same bytes; the harness refuses a readback outside the
        // released activation ranges rather than passing it to the kernel.
        let mut entry_streams = Streams::new(x_values.clone());
        let mut entry_ledger = TrafficLedger::default();
        let (_entry_next, entry_coeffs, _entry_codes, _entry_scales) = mega_mhc_layer_with_scale(
            &mut entry_streams,
            &norm,
            &seed_a,
            &mhc.predictor,
            &mhc.hc_base,
            &mhc.hc_scale,
            &mut entry_ledger,
        );
        if !entry_coefficients_are_in_released_ranges(&entry_coeffs) {
            return Err(format!(
                "position {token}: the entry twin's coefficient readback is outside the released \
                 activation ranges (B in [0,1], A in (0,1], C in (0,2]); refusing to pass it as \
                 the post call's b_prev/c_prev/a_prev"
            ));
        }

        // The declared attention-output stand-in, rounded to bf16 exactly as the kernel reads it.
        let y_host = attention_output_stand_in(&row, &next_row, HIDDEN);
        let y_bits: Vec<u16> = y_host.iter().map(|value| bf16_bits(*value)).collect();
        let y_bytes = bf16_bits_to_bytes(&y_bits);
        let y_values: Vec<f32> = y_bits.iter().map(|bits| bf16_value(*bits)).collect();

        // The CPU twin: the post-mix in the kernel's order, then the released fused arithmetic on
        // the updated f32 values (the kernel takes X_hat from the registers before the bf16 write).
        let updated = expected_update(
            &x_values,
            &entry_coeffs.b,
            &entry_coeffs.c,
            &y_values,
            HIDDEN,
        );
        let mut post_streams = Streams::new(updated.clone());
        let mut post_ledger = TrafficLedger::default();
        // The reference function is used for the COEFFICIENT twin only: its X_hat is a mul-then-add
        // chain (see expected_quantisation), so its codes/scales are not the twin this harness
        // declares. The coefficient comparison is a bar quantity either way (the kernel reduces the
        // projection in tile order, the reference sequentially).
        let (_post_next, coeffs_next, _ref_codes, _ref_scales) = mega_mhc_layer_with_scale(
            &mut post_streams,
            &norm,
            &entry_coeffs.a,
            &mhc.predictor,
            &mhc.hc_base,
            &mhc.hc_scale,
            &mut post_ledger,
        );
        let (host_xhat_values, expected_inv_rms) = host_xhat(&updated, &entry_coeffs.a, HIDDEN);
        let (expected_codes, expected_scales) =
            twin_quantisation(&host_xhat_values, expected_inv_rms, &norm);

        let b_bytes = f32_to_bytes(&flatten_b(&entry_coeffs.b));
        let c_bytes = f32_to_bytes(&entry_coeffs.c);
        let a_bytes = f32_to_bytes(&entry_coeffs.a);
        let uploads = PostUploads {
            x_bytes: &x_bytes,
            y_bytes: &y_bytes,
            b_bytes: &b_bytes,
            c_bytes: &c_bytes,
            a_bytes: &a_bytes,
        };
        let outputs = run_once(&cuda, kernel, &params, &x_d, &uploads, &stats_d, &sync_d)?;

        println!(
            "--- position {token} (embed row {token}, four copies; entry readback B/A/C; Y_0 \
             declared stand-in) ---"
        );
        let b_ref: Vec<f32> = coeffs_next
            .b
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let s_b = compare_f32(
            "hc_coeffs B (the mixer, H(X_1))",
            &outputs.coeffs[0..16],
            &b_ref,
        )?;
        let s_a = compare_f32(
            "hc_coeffs A (the input mixes, H(X_1))",
            &outputs.coeffs[16..20],
            &coeffs_next.a,
        )?;
        let s_c = compare_f32(
            "hc_coeffs C (the block-output weights, H(X_1))",
            &outputs.coeffs[20..24],
            &coeffs_next.c,
        )?;
        let s_fp8 = compare_u8(
            "xhat_fp8 (E4M3 codes)",
            &outputs.codes,
            expected_codes.as_slice(),
        )?;
        let s_scales = compare_u8(
            "xhat_scales (ue8m0)",
            &outputs.scales,
            expected_scales.as_slice(),
        )?;
        let s_x = compare_bf16_write(
            "residual streams X_1 (the post-mix write)",
            &outputs.stream_bits,
            &updated,
        )?;
        let s_rms = compare_f32("1/rms(X_hat_1)", &outputs.rms, &[expected_inv_rms])?;
        let expected_normed = block_input_normed(&host_xhat_values, &norm, outputs.rms[0]);
        let s_capture = compare_f32(
            "block input f32 capture (xhat_normed)",
            &outputs.normed,
            &expected_normed,
        )?;

        let coefficients_ok =
            s_b.max_delta <= COEF_BAR && s_a.max_delta <= COEF_BAR && s_c.max_delta <= COEF_BAR;
        let rms_ok = s_rms.max_delta <= RMS_BAR;
        let fp8_ok = s_fp8.exact_pct() >= CODES_EXACT_PCT_BAR && s_fp8.max_delta <= CODE_MAX_DELTA;
        let capture_ok = s_capture.all_exact();
        let scales_ok = s_scales.all_exact();
        let streams_ok = s_x.all_exact();
        if s_fp8.exact != s_fp8.len {
            if scales_ok && s_fp8.max_delta <= CODE_MAX_DELTA {
                println!(
                    "  finding: {} of {} E4M3 codes differ (first at index {}), each by one code \
                     with the scales bit-exact; the cause is upstream of the quantiser: the twin's \
                     1/rms (expected {:e}, the rms this twin's codes and scales were built with) \
                     and the device's fixed-order tile reduction round 1/rms differently (delta \
                     {:e}), moving a value across an E4M3 boundary",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    expected_inv_rms,
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
        let again = run_once(&cuda, kernel, &params, &x_d, &uploads, &stats_d, &sync_d)?;
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
        let control_ok = control_gap_within_bar(gap, capture_max);
        println!(
            "  f32 control (FP8 quantisation gap): max abs gap {:e} at index {} against capture \
             max {:e} ({:.4}% of the capture maximum; bar {}%)",
            gap,
            gap_at,
            capture_max,
            100.0 * gap / capture_max,
            100.0 * CONTROL_BAR,
        );

        let position_ok = post_position_ok(PositionChecks {
            fp8_ok,
            scales_ok,
            streams_ok,
            coefficients_ok,
            rms_ok,
            capture_ok,
            deterministic,
            control_ok,
        });
        verdicts.push(Verdict::new(
            &format!("position {token}: boundaries and control"),
            position_ok,
            format!(
                "fp8 {} scales {} X_1 {} B/A/C {} 1/rms {} capture {} control {}",
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
    check_floor(args.floor_gib, "m15_mhc_post end")?;

    let failed: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    println!(
        "summary: {} positions x 2 verdicts, {} failed",
        args.tokens,
        failed.len()
    );
    if failed.is_empty() {
        println!(
            "m15_mhc_post: PASS ({} of {} positions at every boundary; the post-mix write and the \
             block-input capture are exact where their arithmetic is the same program, and the \
             coefficients/codes are within the declared bars where the reduction order differs)",
            args.tokens, args.tokens
        );
        Ok(())
    } else {
        for verdict in &failed {
            println!("  FAIL {}: {}", verdict.name, verdict.detail);
        }
        Err(format!(
            "m15_mhc_post: FAIL at {} of {} position verdicts",
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

/// The host-copied seam bytes for one post call: the entry write X_0, the declared attention
/// output Y_0, and the entry coefficient readback B/A/C. All five are host-provided, because the
/// chain's device-to-device handoff does not exist; the launch's argument shape does not change.
struct PostUploads<'a> {
    x_bytes: &'a [u8],
    y_bytes: &'a [u8],
    b_bytes: &'a [u8],
    c_bytes: &'a [u8],
    a_bytes: &'a [u8],
}

/// Uploads one position's seam bytes, zeroes the per-launch rendezvous state, launches once, and
/// reads every output back.
fn run_once(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    params: &MegaMhcParams,
    x_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    uploads: &PostUploads<'_>,
    stats_d: &sparkflash_kernels::cuda_driver::Deviceptr,
    sync_d: &sparkflash_kernels::cuda_driver::Deviceptr,
) -> Result<Outputs, String> {
    cuda.copy_htod(x_d, uploads.x_bytes)?;
    // The y/b/c/a addresses come from the pinned 19-word argument array by position, never from a
    // second copy of the layout.
    let words = params.as_slice();
    let (y_d, b_d, c_d, a_d) = (
        sparkflash_kernels::cuda_driver::Deviceptr(words[1]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[2]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[3]),
        sparkflash_kernels::cuda_driver::Deviceptr(words[4]),
    );
    cuda.copy_htod(&y_d, uploads.y_bytes)?;
    cuda.copy_htod(&b_d, uploads.b_bytes)?;
    cuda.copy_htod(&c_d, uploads.c_bytes)?;
    cuda.copy_htod(&a_d, uploads.a_bytes)?;
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
    fn the_post_update_does_not_accumulate_b_before_the_block_term() {
        // A COARSE sanity pin, not the instruction-order pin: it distinguishes the twin from a
        // B-sum-first arrangement (sum the B terms, then add the C Y term), which is a different
        // order again. One column, four streams: destination 0 mixes sources 0 (+1) and 1 (-1);
        // C_0 = 1; Y_0 = 1; X_0 column = [1e8, 1e8, 0, 0]; f32 spacing at 1e8 is 8.
        //   kernel (rounded first B product, C Y fused):  1e8 + 1 -> 1e8, then -1e8 -> 0
        //   B-sum-first:                                   1e8 - 1e8 = 0, then +1 -> 1
        // Both the kernel arrangement and the C-Y-first arrangement give 0 here, so the REAL
        // arrangement pins are the_post_update_matches_the_kernel_instruction_order and
        // the_first_b_product_is_the_rounded_one below; this test only catches B-sum-first.
        let x = vec![1.0e8f32, 1.0e8, 0.0, 0.0];
        let mut b = [[0.0f32; HC_MULT]; HC_MULT];
        b[0] = [1.0, -1.0, 0.0, 0.0];
        let c = [1.0f32, 0.0, 0.0, 0.0];
        let y = vec![1.0f32];
        let expected = expected_update(&x, &b, &c, &y, 1);
        assert_eq!(
            expected[0], 0.0,
            "the B terms must not be summed before the C Y term"
        );
    }

    #[test]
    fn the_post_update_matches_the_kernel_instruction_order() {
        // What the PTX actually does per stream (b16's capture failure was exactly this shadow):
        //   q = round(b[s][0] * x[0]);  acc = fma(c[s], y, q);  acc = fma(b[s][t], x[t], acc)...
        // i.e. the first B product is the one rounded, and the C Y term is the FUSED one. The
        // source's literal order (round c*y first, then fuse the first B term) is different, and a
        // twin with no fusion at all is different again. This case separates all three.
        //   b00 = -(1 + 2^-22), x0 = 1  => q = -(1 + 2^-22) exactly
        //   c0 = 1 + 2^-23, y = 1 + 2^-23 => exact c0*y = 1 + 2^-22 + 2^-46
        //   kernel:      fma(c0, y, q) = round(2^-46) = 2^-46
        //   C-Y-first:   round(c0*y) = 1 + 2^-22, then + q = 0
        //   no-fusion:   round(c0*y) + round(b00*x0) = 0
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut b = [[0.0f32; HC_MULT]; HC_MULT];
        b[0][0] = -(1.0f32 + 2f32.powi(-22));
        let mut c = [0.0f32; HC_MULT];
        c[0] = 1.0f32 + 2f32.powi(-23);
        let y = vec![1.0f32 + 2f32.powi(-23)];
        let expected = expected_update(&x, &b, &c, &y, 1);
        assert_eq!(
            expected[0],
            2f32.powi(-46),
            "the C Y term must be the fused one, over a rounded first B product"
        );
        assert_ne!(
            expected[0], 0.0,
            "a C-Y-first-rounded or mul-then-add twin collapses this to 0"
        );
    }

    #[test]
    fn host_xhat_fuses_the_input_mix() {
        // The device's X_hat is an fma chain (kernel line 190-193: xhat += s_a[s] * updated[s]);
        // host_xhat must fuse it too, or the capture twin would be built from a different X_hat
        // than the codes/scales twin. This case separates them:
        //   a0*x0 = -(1 + 2^-22) exactly, then fma(a1, x1, .) with a1*x1 = 1 + 2^-22 + 2^-46
        //   fused:      round(2^-46) = 2^-46
        //   mul-then-add: round(a1*x1) = 1 + 2^-22, then + a0*x0 = 0
        let x = vec![1.0f32, 1.0f32 + 2f32.powi(-23), 0.0, 0.0];
        let a = [
            -(1.0f32 + 2f32.powi(-22)),
            1.0f32 + 2f32.powi(-23),
            0.0,
            0.0,
        ];
        let (xhat, _inv_rms) = host_xhat(&x, &a, 1);
        assert_eq!(
            xhat[0],
            2f32.powi(-46),
            "host_xhat must use a fused multiply-add for the input mix"
        );
        assert_ne!(xhat[0], 0.0, "a mul-then-add X_hat collapses this to 0");
    }

    #[test]
    fn the_quantisation_known_answer_pins_the_scale_rule() {
        // A KNOWN ANSWER, independent of twin_quantisation: X_hat = e_0 (hand squares = 1.0), RMS
        // weight all ones, so the hand-computed 1/rms is 1/sqrt(1/5120 + NORM_EPS) and normed[0] is
        // that value with every other normed value zero. Block 0's amax ceils to 2^-2 (ue8m0 code
        // 125); every all-zero block floors amax to 1e-4 and ceils to 2^-22 (code 105). At 2^-2,
        // normed[0] * 4 = 286.2167 (NORM_EPS = 1e-20) rounds to the E4M3 value 288 (code 0x79);
        // zero encodes to 0x00.
        // This pins the released rule: a change to the scale code, the floor, the reciprocal or the
        // quantiser moves one of these constants.
        let mut xhat = vec![0.0f32; HIDDEN];
        xhat[0] = 1.0;
        let squares = 1.0f32; // hand: only xhat[0] is non-zero
        let inv_rms = 1.0f32 / (squares / HIDDEN as f32 + NORM_EPS).sqrt();
        let norm = vec![1.0f32; HIDDEN];
        let (codes, scales, normed) = expected_quantisation(&xhat, inv_rms, &norm);
        assert_eq!(scales.len(), HIDDEN / QUANT_BLOCK);
        assert_eq!(scales[0], 125, "block 0's amax ceils to 2^-2");
        assert_eq!(
            scales[1], 105,
            "an all-zero block floors to 1e-4 and ceils to 2^-22"
        );
        assert!(
            scales.iter().skip(1).all(|code| *code == 105),
            "every all-zero block uses the floored scale code"
        );
        assert_eq!(
            codes[0], 0x79,
            "286.2167 at 2^-2 rounds to the E4M3 value 288"
        );
        assert!(
            codes.iter().skip(1).all(|code| *code == 0),
            "zero encodes to zero"
        );
        assert_eq!(
            normed[0], inv_rms,
            "normed[0] is 1.0 * 1.0 * the hand 1/rms"
        );
        assert!(normed.iter().skip(1).all(|value| *value == 0.0));
    }

    #[test]
    fn the_codes_twin_is_wired_to_the_fma_twin() {
        // The expected codes/scales must come from THIS harness's fma X_hat and its own 1/rms, in
        // the kernel's order. twin_quantisation is the ONE producer of those expected bytes in a
        // non-test build: revert run()'s call site to the reference's codes/scales and this
        // function loses its only non-test use, so the bin build fails dead_code under
        // "clippy --all-targets -D warnings". This test pins its content (the released quantisation
        // rule applied to the twin's normed vector), and host_xhat_fuses_the_input_mix pins that the
        // X_hat it consumes is the fma one, not mega_mhc_ref's mul-then-add chain.
        let mut updated = vec![0.0f32; HC_MULT * HIDDEN];
        updated[0] = 1.0f32;
        updated[HIDDEN] = 1.0f32 + 2f32.powi(-23);
        let a = [
            -(1.0f32 + 2f32.powi(-22)),
            1.0f32 + 2f32.powi(-23),
            0.0,
            0.0,
        ];
        let norm = vec![1.0f32; HIDDEN];
        let (xhat, inv_rms) = host_xhat(&updated, &a, HIDDEN);
        assert_eq!(
            xhat[0],
            2f32.powi(-46),
            "the twin consumes the fma X_hat, not the reference's mul-then-add one"
        );
        let (codes, scales) = twin_quantisation(&xhat, inv_rms, &norm);
        let (want_codes, want_scales, normed) = expected_quantisation(&xhat, inv_rms, &norm);
        assert_eq!(codes, want_codes, "twin_quantisation is the released rule");
        assert_eq!(
            scales, want_scales,
            "twin_quantisation is the released rule"
        );
        assert!(
            normed.iter().any(|value| *value > 0.0),
            "the twin's normed vector is not empty"
        );
        // Deterministic by construction: a second build is the same bytes.
        let (again, _scales) = twin_quantisation(&xhat, inv_rms, &norm);
        assert_eq!(codes, again, "the codes twin is deterministic");
    }

    #[test]
    fn the_position_conjunction_requires_every_check() {
        let all_true = || PositionChecks {
            fp8_ok: true,
            scales_ok: true,
            streams_ok: true,
            coefficients_ok: true,
            rms_ok: true,
            capture_ok: true,
            deterministic: true,
            control_ok: true,
        };
        assert!(post_position_ok(all_true()), "all checks passing must pass");
        // Every single check must be able to fail the position on its own.
        let mut flip = all_true();
        flip.fp8_ok = false;
        assert!(!post_position_ok(flip), "fp8_ok is not required");
        let mut flip = all_true();
        flip.scales_ok = false;
        assert!(!post_position_ok(flip), "scales_ok is not required");
        let mut flip = all_true();
        flip.streams_ok = false;
        assert!(!post_position_ok(flip), "streams_ok is not required");
        let mut flip = all_true();
        flip.coefficients_ok = false;
        assert!(!post_position_ok(flip), "coefficients_ok is not required");
        let mut flip = all_true();
        flip.rms_ok = false;
        assert!(!post_position_ok(flip), "rms_ok is not required");
        let mut flip = all_true();
        flip.capture_ok = false;
        assert!(!post_position_ok(flip), "capture_ok is not required");
        let mut flip = all_true();
        flip.deterministic = false;
        assert!(!post_position_ok(flip), "deterministic is not required");
        let mut flip = all_true();
        flip.control_ok = false;
        assert!(!post_position_ok(flip), "control_ok is not required");
    }

    #[test]
    fn the_first_b_product_is_the_rounded_one() {
        // A companion case that separates the corrected arrangement from the mirror one (fuse the
        // first B product, round c*y) using the same idea with the roles of the rounded and fused
        // products exchanged.
        //   b00 = 1 + 2^-23, x0 = 1 + 2^-23 => exact 1 + 2^-22 + 2^-46
        //   c0 = -(1 + 2^-22), y = 1       => exact -(1 + 2^-22)
        //   kernel: round(b00*x0) = 1 + 2^-22, fma(c0, y, q) = 0
        //   C-Y-first: round(c0*y) = -(1+2^-22), fma(b00, x0, .) = 2^-46
        let x = vec![1.0f32 + 2f32.powi(-23), 0.0, 0.0, 0.0];
        let mut b = [[0.0f32; HC_MULT]; HC_MULT];
        b[0][0] = 1.0f32 + 2f32.powi(-23);
        let mut c = [0.0f32; HC_MULT];
        c[0] = -(1.0f32 + 2f32.powi(-22));
        let y = vec![1.0f32];
        let expected = expected_update(&x, &b, &c, &y, 1);
        assert_eq!(
            expected[0], 0.0,
            "the first B product must be the rounded one, not the fused one"
        );
    }

    #[test]
    fn the_post_update_matches_a_hand_computed_case() {
        // Two columns, four streams: stream 0 = [1, 2], stream 1 = [3, 4], streams 2,3 = 0;
        // B row 0 = [1, 2, 0, 0], row 1 = [3, -1, 0, 0], rows 2,3 identity; C = [0.5, -0.25, 0, 0];
        // Y = [10, 20].
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
        let expected = expected_update(&x, &b, &c, &y, hidden);
        assert_eq!(expected.len(), 8);
        assert_eq!(expected[0], 12.0);
        assert_eq!(expected[1], 20.0);
        assert_eq!(expected[2], -2.5);
        assert_eq!(expected[3], -3.0);
        assert_eq!(&expected[4..], &[0.0f32; 4]);
    }

    #[test]
    fn the_attention_output_stand_in_is_the_declared_formula() {
        let row = [2.0f32, -4.0, 0.0];
        let next = [4.0f32, 8.0, 1.0];
        let y = attention_output_stand_in(&row, &next, 3);
        assert_eq!(y, vec![0.0, -4.0, -0.25]);
    }

    #[test]
    fn the_control_bar_is_pinned_at_ten_percent() {
        assert!(
            (CONTROL_BAR - 0.10).abs() <= f32::EPSILON,
            "the declared control bar moved: {CONTROL_BAR}"
        );
        assert!(control_gap_within_bar(0.099, 1.0));
        assert!(!control_gap_within_bar(0.101, 1.0));
        // Scale-free: the same fraction of a different capture maximum is the same verdict.
        assert!(control_gap_within_bar(0.099 * 2.5, 2.5));
        assert!(!control_gap_within_bar(0.101 * 2.5, 2.5));
    }

    #[test]
    fn the_post_bars_are_pinned() {
        assert!(
            (CODES_EXACT_PCT_BAR - 99.5).abs() <= f64::EPSILON,
            "the declared E4M3 exact-fraction bar moved: {CODES_EXACT_PCT_BAR}"
        );
        assert!(
            (CODE_MAX_DELTA - 1.0).abs() <= f32::EPSILON,
            "the declared per-code distance moved: {CODE_MAX_DELTA}"
        );
        assert!(
            (COEF_BAR - 1e-4).abs() <= f32::EPSILON,
            "the declared coefficient bar moved: {COEF_BAR}"
        );
        assert!(
            (RMS_BAR - 3e-4).abs() <= f32::EPSILON,
            "the declared 1/rms bar moved: {RMS_BAR}"
        );
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
