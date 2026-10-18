//! The device MoE router against the released layer-0 bytes: kernel vs CPU twin vs engine host path.
//!
//! THIS EXAMPLE IS THE EVIDENCE THE KERNEL IS CORRECT, not the engine integration. The engine's host
//! router (crates/sparkflash-engine/src/forward/moe.rs) is untouched and still the path a served
//! token takes; replacing it is a separate MR so the change can be bisected. This harness loads the
//! released gate weight and bias, feeds the same operands to three implementations, and reports
//! which of the three agree and to what class:
//!
//! * the ENGINE host path, `moe::route`, which is the oracle the released arithmetic is defined by;
//! * the CPU TWIN in `sparkflash_kernels::router_ref`, which the crate's own tests compare to the
//!   engine bit for bit (weights included) on both the released and the tiny configuration;
//! * the DEVICE kernel `moe_router_sm121`, whose cubin is pinned in `router_abi` and whose ABI the
//!   same module parsed out of the kernel source.
//!
//! The pass bar is printed BEFORE the numbers it judges. It is a tolerance bar for the score
//! function and an exactness bar for ids and for the host path, because of what the implementations
//! are:
//!
//! * the GEMM is DESIGNED to match bit for bit: the kernel accumulates every score in ascending k
//!   with `__fmul_rn` then `__fadd_rn` (never an FMA), which is exactly the engine's serial
//!   `acc += x[col] * w[col]` loop for this geometry, over the same operands (BF16 widens to f32
//!   exactly, so no decode rounding differs). The harness does NOT separately assert the pre-softplus
//!   z: the device output starts at the score, so the measured bound on the device GEMM is the score
//!   bar (1e-6 and 8 ULP), and `scores_bit_exact` is printed as information, not used as a verdict;
//! * the score function CANNOT match bit for bit: the host uses libm's `exp`/`ln_1p`, the device
//!   uses CUDA's `expf`/`log1pf`, which are different implementations of the same function.
//!   `sqrtf` is correctly rounded, so it is not a source of difference;
//! * the selection is therefore expected to match exactly whenever the score difference does not
//!   change an ordering. It is pinned as exact, not probabilistic: on the released operands the
//!   top-6 gaps are far above 1e-7, and a tie that DID cross would be a finding, not a tolerance.
//!
//! The operand source: the released gate weight (BF16, [384, 5120]), the released noaux_tc bias
//! (F32, [384]) and the released embed rows 0..tokens as the activation rows. The embed row is a
//! stand-in: the real router consumes the post-attention hidden, and this harness does not have
//! layer 0's FFN-side residual stream. Every claim below is about the ROUTER ARITHMETIC over
//! released gate bytes and released activation bytes, not about end-to-end layer text.
//!
//! Usage (aarch64 node, checkpoint present):
//!
//! ```text
//! m9_router --checkpoint <snapshot dir> --cubin-dir out [--tokens 8] [--layer 0] [--repeat 100]
//! ```
//!
//! The cubin must be the pinned one: `nvcc -std=c++17 -arch=sm_121a -cubin -I kernels
//! kernels/moe_router_sm121.cu -o out/moe_router_sm121.cubin` under CUDA 13.0.3 (V13.0.88), 37,384 B,
//! sha256 `5992cef9...`; the harness refuses a different cubin by hash rather than measuring it.

// The shared CUDA driver is safe-only for examples: the typed API is the whole surface, and this
// pin keeps a future edit from opening an FFI path beside it (m4_abi checks it).
#![forbid(unsafe_code)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::router_abi as abi;
use sparkflash_kernels::router_launch::{self, RouterLaunchParams};
use sparkflash_kernels::router_ref::{self, RouterParams};

/// The routed experts in the released configuration.
const EXPERTS: usize = abi::N_ROUTED as usize;
/// The hidden width in the released configuration.
const HIDDEN: usize = abi::HIDDEN as usize;
/// The experts activated per token.
const TOP_K: usize = abi::TOP_K as usize;

/// The three routing outputs read back from the device - ids, weights, scores - and the raw bytes
/// they were decoded from, kept together so the determinism check compares the bytes it read.
type DeviceOutput = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u8>);

/// Print MemAvailable in kB, from /proc/meminfo.
fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

/// Refuse to run under the floor: this harness allocates a few MB, but the node is shared with vLLM
/// runs and the floor is the same discipline the other harnesses use.
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

#[derive(Debug)]
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
        Self::parse_from(std::env::args().skip(1).collect())
    }

    /// The parser proper, over an explicit argument list: `parse` is the process's argv.
    ///
    /// Kept separate so the CPU tests can drive it without a second process or a test-only argv.
    fn parse_from(values: Vec<String>) -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            tokens: 8,
            layer: 0,
            repeat: 100,
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
                        "usage: m9_router --checkpoint <snapshot dir> [--cubin-dir out] \
                         [--tokens 8] [--layer 0] [--repeat 100] [--floor-gib 32]"
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
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// Little-endian byte pairs to u16 (BF16 bit patterns, unconverted).
fn bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// BF16 bytes widened exactly to f32.
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bf16_bits(bytes)
        .into_iter()
        .map(router_ref::bf16_to_f32)
        .collect()
}

/// F32 bytes to f32 (the checkpoint stores little-endian).
fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
        .collect()
}

/// The engine host path: `Gate.forward` per token.
fn route_host(
    cfg: &sparkflash_engine::config::ModelConfig,
    prefix: &str,
    x: &[f32],
    gate: &[f32],
    bias: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let mut selected = Vec::new();
    sparkflash_engine::forward::moe::route(cfg, prefix, x, gate, bias, None, &mut selected)
        .expect("the engine route must accept the released geometry");
    let ids = selected.iter().map(|(expert, _)| *expert as u32).collect();
    let weights = selected.iter().map(|(_, weight)| *weight).collect();
    (ids, weights)
}

/// The largest absolute difference between two same-length f32 vectors.
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

/// ULP distance between two positive finite f32 values: the difference of their bit patterns, which
/// is monotone for positive values. Both scores and weights here are strictly positive.
fn max_ulp(a: &[f32], b: &[f32]) -> (u32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0u32;
    let mut at = 0usize;
    for (index, (left, right)) in a.iter().zip(b).enumerate() {
        assert!(
            *left >= 0.0 && *right >= 0.0 && left.is_finite() && right.is_finite(),
            "ULP distance is defined for the positive finite values here"
        );
        let distance = left.to_bits().abs_diff(right.to_bits());
        if distance > max {
            max = distance;
            at = index;
        }
    }
    (max, at)
}

fn bit_identical(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(left, right)| left.to_bits() == right.to_bits())
        .count()
}

/// One pass/fail line in the report, accumulated rather than asserted so every measured line is
/// printed even after a failure.
struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let cfg = sparkflash_engine::config::ModelConfig::v41_flash_cached();
    let prefix = format!("layers.{}", args.layer);
    println!("=== m9_router: the device MoE router against the released layer-0 bytes ===");
    println!(
        "gate: {}.ffn.gate.weight (BF16 [{EXPERTS}, {HIDDEN}]) and {}.ffn.gate.bias (F32); \
         activations: released embed.weight rows 0..{} (a stand-in for the post-attention hidden, \
         reported as such, never as layer text)",
        prefix, prefix, args.tokens
    );
    println!(
        "comparison class (declared before any number): ids EXACT against the engine host path and \
         the CPU twin, for every one of the {} selections; the host path's weights are bit-identical \
         to the twin (asserted below), and the device's weights are within 1e-6 and 16 ULP of the \
         twin - so within that bar of the host path too, but there is NO separate device-vs-host \
         weight comparison; device scores max abs delta <= 1e-6 and max ULP <= 8 against the twin; \
         two device runs byte-identical. The GEMM is not separately asserted (its output is the \
         score); the score bar is the measured bound. expf/log1pf on the device are not libm's \
         exp/ln_1p, which is the only known degree of freedom. sqrtf is correctly rounded.",
        args.tokens * TOP_K
    );
    println!(
        "what this is not: not an engine integration, not layer text, not a device window \
         measurement of host time saved - the host seam is still in place and removing it is a \
         separate MR."
    );
    check_floor(args.floor_gib, "m9_router start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let gate_bytes = snap.read_raw(
        &format!("{prefix}.ffn.gate.weight"),
        "BF16",
        &[EXPERTS, HIDDEN],
    )?;
    let bias_bytes = snap.read_raw(&format!("{prefix}.ffn.gate.bias"), "F32", &[EXPERTS])?;
    println!(
        "gate bytes: weight {} B (BF16), bias {} B (F32); no decode beyond the exact BF16 widening",
        gate_bytes.len(),
        bias_bytes.len()
    );

    // The activations, row by row, kept in their released BF16 bytes for the device and widened
    // exactly for the host references.
    let mut x_bytes = Vec::with_capacity(args.tokens * HIDDEN * 2);
    let mut x_f32 = Vec::with_capacity(args.tokens * HIDDEN);
    for token in 0..args.tokens {
        let mut row = vec![0u8; HIDDEN * 2];
        snap.read_row_bf16_into("embed.weight", token, HIDDEN, &mut row)?;
        x_f32.extend(bf16_bytes_to_f32(&row));
        x_bytes.extend_from_slice(&row);
    }
    let gate_f32 = bf16_bytes_to_f32(&gate_bytes);
    let bias_f32 = f32_bytes_to_vec(&bias_bytes);

    // Host references first, so the device numbers are compared to a reference already checked.
    let mut verdicts = Vec::new();
    let mut host_ids: Vec<u32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut host_weights: Vec<f32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_ids: Vec<u32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_weights: Vec<f32> = Vec::with_capacity(args.tokens * TOP_K);
    let mut twin_scores: Vec<f32> = Vec::with_capacity(args.tokens * EXPERTS);
    for token in 0..args.tokens {
        let x = &x_f32[token * HIDDEN..(token + 1) * HIDDEN];
        let (ids, weights) = route_host(cfg, &prefix, x, &gate_f32, &bias_f32);
        host_ids.extend_from_slice(&ids);
        host_weights.extend_from_slice(&weights);
        let twin = router_ref::route_f32(x, &gate_f32, &bias_f32, RouterParams::RELEASED)?;
        twin_ids.extend_from_slice(&twin.ids);
        twin_weights.extend_from_slice(&twin.weights);
        twin_scores.extend_from_slice(&twin.scores);
    }
    let ids_equal = host_ids == twin_ids;
    let weights_bit_equal = bit_identical(&host_weights, &twin_weights) == host_weights.len();
    println!(
        "host vs twin: ids {} of {} equal, weights {} of {} bit-identical",
        host_ids
            .iter()
            .zip(&twin_ids)
            .filter(|(a, b)| a == b)
            .count(),
        host_ids.len(),
        bit_identical(&host_weights, &twin_weights),
        host_weights.len()
    );
    verdicts.push(Verdict {
        name: "host engine path vs CPU twin: ids exact".to_owned(),
        ok: ids_equal,
        detail: if ids_equal {
            format!("{} selections identical", host_ids.len())
        } else {
            "the twin and the engine disagree: the device comparison would be meaningless"
                .to_owned()
        },
    });
    verdicts.push(Verdict {
        name: "host engine path vs CPU twin: weights bit-identical".to_owned(),
        ok: weights_bit_equal,
        detail: format!(
            "{} of {} weight bits identical",
            bit_identical(&host_weights, &twin_weights),
            host_weights.len()
        ),
    });

    // The device.
    let cubin_path = args.cubin_dir.join(format!("{}.cubin", abi::ROUTER_KERNEL));
    let cubin = std::fs::read(&cubin_path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels \
             kernels/moe_router_sm121.cu -o out/moe_router_sm121.cubin)",
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
    verdicts.push(Verdict {
        name: "cubin is the pinned one".to_owned(),
        ok: cubin_ok,
        detail: format!(
            "expected {} B / {}, got {} B / {sha}",
            abi::CUBIN_BYTES,
            abi::CUBIN_SHA256,
            cubin.len()
        ),
    });
    if !cubin_ok {
        // Hash mismatch means this is not the kernel the ABI and baseline pin: measuring it would
        // be measuring a different kernel. Stop before a device run.
        return Err(
            "the cubin does not match the pinned sha256; refusing to measure it".to_owned(),
        );
    }

    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(x_bytes.len(), &mut allocated)?;
    let gate_d = cuda.alloc(gate_bytes.len(), &mut allocated)?;
    let bias_d = cuda.alloc(bias_bytes.len(), &mut allocated)?;
    let ids_d = cuda.alloc(args.tokens * TOP_K * 4, &mut allocated)?;
    let weights_d = cuda.alloc(args.tokens * TOP_K * 4, &mut allocated)?;
    let scores_d = cuda.alloc(args.tokens * EXPERTS * 4, &mut allocated)?;
    println!(
        "device buffers: {allocated} bytes ({} KiB) admitted",
        allocated / 1024
    );
    cuda.copy_htod(&x_d, &x_bytes)?;
    cuda.copy_htod(&gate_d, &gate_bytes)?;
    cuda.copy_htod(&bias_d, &bias_bytes)?;
    let (module, kernel) = router_launch::load(&cuda, &cubin)?;
    let params = RouterLaunchParams {
        x: x_d,
        gate: gate_d,
        bias: bias_d,
        ids: ids_d,
        weights: weights_d,
        scores: scores_d,
        tokens: args.tokens as u32,
        trace_ring: 0, // the shipped cubin is the trace-disabled build
        layer_id: args.layer as u32,
        step_seq: 0,
    };
    println!(
        "launch grid: {:?} blocks of {} threads, {} tokens per block",
        router_launch::grid(params.tokens)?,
        abi::THREADS,
        abi::TOKENS_PER_BLOCK
    );

    let read_out = |cuda: &Cuda| -> Result<DeviceOutput, String> {
        let mut ids = vec![0u8; args.tokens * TOP_K * 4];
        let mut weights = vec![0u8; args.tokens * TOP_K * 4];
        let mut scores = vec![0u8; args.tokens * EXPERTS * 4];
        cuda.copy_dtoh(&mut ids, &ids_d)?;
        cuda.copy_dtoh(&mut weights, &weights_d)?;
        cuda.copy_dtoh(&mut scores, &scores_d)?;
        let ids: Vec<u32> = ids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let weights = f32_bytes_to_vec(&weights);
        let scores = f32_bytes_to_vec(&scores);
        let mut raw = Vec::with_capacity(ids.len() * 4 + weights.len() * 4 + scores.len() * 4);
        raw.extend_from_slice(
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        raw.extend_from_slice(
            &weights
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        raw.extend_from_slice(
            &scores
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        Ok((ids, weights, scores, raw))
    };

    check_floor(args.floor_gib, "m9_router pre-launch")?;
    router_launch::launch(&cuda, kernel, &params)?;
    let (device_ids, device_weights, device_scores, first_run) = read_out(&cuda)?;
    // A second launch on the same buffers: the kernel must be deterministic byte for byte.
    router_launch::launch(&cuda, kernel, &params)?;
    let (_, _, _, second_run) = read_out(&cuda)?;
    check_floor(args.floor_gib, "m9_router post-launch")?;

    let device_ids_equal = device_ids == twin_ids;
    println!(
        "device vs twin: ids {} of {} equal",
        device_ids
            .iter()
            .zip(&twin_ids)
            .filter(|(a, b)| a == b)
            .count(),
        device_ids.len()
    );
    verdicts.push(Verdict {
        name: "device vs twin: ids exact".to_owned(),
        ok: device_ids_equal,
        detail: if device_ids_equal {
            format!("{} selections identical", device_ids.len())
        } else {
            format!(
                "first difference at selection {}",
                device_ids
                    .iter()
                    .zip(&twin_ids)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0)
            )
        },
    });

    let (scores_delta, scores_at) = max_abs_delta(&device_scores, &twin_scores);
    let (scores_ulp, _) = max_ulp(&device_scores, &twin_scores);
    let scores_bit_exact = bit_identical(&device_scores, &twin_scores);
    let scores_ok = scores_delta <= 1e-6 && scores_ulp <= 8;
    println!(
        "device vs twin: scores max abs delta {scores_delta:e} (at {scores_at}), max ULP {scores_ulp}, \
         {scores_bit_exact}/{} bit-identical",
        device_scores.len()
    );
    verdicts.push(Verdict {
        name: "device vs twin: scores within the score-function bar".to_owned(),
        ok: scores_ok,
        detail: format!("max abs delta {scores_delta:e}, max ULP {scores_ulp} (bar 1e-6 and 8)"),
    });

    let (weights_delta, weights_at) = max_abs_delta(&device_weights, &twin_weights);
    let (weights_ulp, _) = max_ulp(&device_weights, &twin_weights);
    let weights_bit_exact = bit_identical(&device_weights, &twin_weights);
    let weights_ok = weights_delta <= 1e-6 && weights_ulp <= 16;
    println!(
        "device vs twin: weights max abs delta {weights_delta:e} (at {weights_at}), max ULP \
         {weights_ulp}, {weights_bit_exact}/{} bit-identical (twin is bit-identical to the host \
         path, so this is the host comparison with the twin's normalisation arithmetic)",
        device_weights.len()
    );
    verdicts.push(Verdict {
        name: "device vs twin: weights within the normalisation bar".to_owned(),
        ok: weights_ok,
        detail: format!("max abs delta {weights_delta:e}, max ULP {weights_ulp} (bar 1e-6 and 16)"),
    });

    let deterministic = first_run == second_run;
    verdicts.push(Verdict {
        name: "device determinism: two runs byte-identical".to_owned(),
        ok: deterministic,
        detail: format!("{} bytes compared", first_run.len()),
    });

    // A selection table for a few tokens, so a reviewer can see WHICH experts and HOW the ids moved.
    let shown = args.tokens.min(4);
    for token in 0..shown {
        let range = token * TOP_K..(token + 1) * TOP_K;
        println!(
            "token {token}: device ids {:?} weights {:?}",
            &device_ids[range.clone()],
            &device_weights[range]
        );
    }

    // Timing: repeat launches on the same buffers, the same discipline the stage-2 timing uses.
    let (per_launch, batched) =
        router_launch::launch_repeated(&cuda, kernel, &params, args.repeat)?;
    let mut sorted = per_launch.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    println!(
        "timing: {} launches of the {} tokens on the same buffers; per-launch ms min {min:.4} p50 \
         {p50:.4} mean {mean:.4} max {max:.4}; batched ms/launch {batched:.4}; p50 per token {:.4} \
         ms",
        args.repeat,
        args.tokens,
        p50 / args.tokens as f64
    );

    cuda.free(module, &[x_d, gate_d, bias_d, ids_d, weights_d, scores_d]);

    let failures = verdicts.iter().filter(|verdict| !verdict.ok).count();
    for verdict in &verdicts {
        println!(
            "  {} {}: {}",
            if verdict.ok { "PASS" } else { "FAIL" },
            verdict.name,
            verdict.detail
        );
    }
    if failures != 0 {
        return Err(format!("{failures} of {} checks failed", verdicts.len()));
    }
    println!(
        "m9_router: PASS ({} of {} checks within the declared class)",
        verdicts.len(),
        verdicts.len()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    //! The harness's own plumbing, on the box: the argument parser and the three byte/float
    //! helpers the comparison table depends on. None of this starts CUDA; the device path is the
    //! window's job. `cargo test --examples` is what runs these.

    use super::*;

    fn parse(values: &[&str]) -> Result<Args, String> {
        Args::parse_from(values.iter().map(|value| (*value).to_string()).collect())
    }

    #[test]
    fn the_argument_parser_takes_the_documented_invocation() {
        let args = parse(&[
            "--checkpoint",
            "/snap",
            "--cubin-dir",
            "/cubins",
            "--tokens",
            "130",
            "--layer",
            "3",
            "--repeat",
            "20",
            "--floor-gib",
            "16",
        ])
        .expect("the documented invocation must parse");
        assert_eq!(args.checkpoint, PathBuf::from("/snap"));
        assert_eq!(args.cubin_dir, PathBuf::from("/cubins"));
        assert_eq!(args.tokens, 130);
        assert_eq!(args.layer, 3);
        assert_eq!(args.repeat, 20);
        assert_eq!(args.floor_gib, 16);
    }

    #[test]
    fn the_argument_parser_defaults_match_the_usage_line() {
        let args = parse(&["--checkpoint", "/snap"]).expect("checkpoint is the one required flag");
        assert_eq!(args.cubin_dir, PathBuf::from("out"));
        assert_eq!(args.tokens, 8);
        assert_eq!(args.layer, 0);
        assert_eq!(args.repeat, 100);
        assert_eq!(args.floor_gib, 32);
    }

    #[test]
    fn the_argument_parser_refuses_what_the_harness_cannot_run() {
        assert_eq!(
            parse(&[]).unwrap_err(),
            "--checkpoint is required (the released snapshot directory)"
        );
        assert_eq!(
            parse(&["--wat"]).unwrap_err(),
            "unknown argument \"--wat\"; try --help"
        );
        assert_eq!(
            parse(&["--checkpoint"]).unwrap_err(),
            "--checkpoint needs a value"
        );
        assert_eq!(
            parse(&["--checkpoint", "/snap", "--tokens", "0"]).unwrap_err(),
            "--tokens must be in 1..=1024"
        );
        assert_eq!(
            parse(&["--checkpoint", "/snap", "--tokens", "1025"]).unwrap_err(),
            "--tokens must be in 1..=1024"
        );
        assert_eq!(
            parse(&["--checkpoint", "/snap", "--repeat", "0"]).unwrap_err(),
            "--repeat must be at least 1"
        );
        assert_eq!(
            parse(&["--checkpoint", "/snap", "--tokens", "x"]).unwrap_err(),
            "--tokens needs a number"
        );
    }

    #[test]
    fn the_byte_decoders_are_little_endian() {
        assert_eq!(bf16_bits(&[0x00, 0x3F, 0x00, 0xBF]), vec![0x3F00, 0xBF00]);
        // BF16 0x3F80 is 1.0; 0xBF80 is -1.0.
        assert_eq!(
            bf16_bytes_to_f32(&[0x80, 0x3F, 0x80, 0xBF]),
            vec![1.0, -1.0]
        );
        assert_eq!(
            f32_bytes_to_vec(&1.0f32.to_le_bytes()),
            vec![1.0],
            "f32 bytes are little-endian, like the checkpoint"
        );
    }

    #[test]
    fn the_delta_and_ulp_helpers_measure_what_the_bars_assert() {
        let (delta, at) = max_abs_delta(&[1.0, -2.0, 3.0], &[1.0, -2.5, 2.0]);
        assert_eq!(at, 2);
        assert!((delta - 1.0).abs() < f32::EPSILON);
        assert_eq!(bit_identical(&[1.0, 2.0], &[1.0, 3.0]), 1);
        let (ulp, at) = max_ulp(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!((ulp, at), (0, 0));
        // One step up the positive f32 bit pattern is one ULP.
        let one_ulp = f32::from_bits(1.0f32.to_bits() + 1);
        assert_eq!(max_ulp(&[1.0, one_ulp], &[1.0, 1.0]).0, 1);
    }
}
