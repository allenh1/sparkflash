//! m18: the serving tail's standalone final RMSNorm at the released geometry - device output
//! against the engine's own final-norm arithmetic, bit for bit.
//!
//! THIS EXAMPLE IS THE EVIDENCE THAT THE PINNED rmsnorm_sm121 KERNEL REPRODUCES THE ENGINE'S FINAL
//! NORM. It is not the served forward pass, not engine integration (c4c, engine-dev's wiring), not
//! the LM head, not a chain/token/per-layer claim and not a timing claim.
//!
//! The model's tail is: collapsed mHC residual [T, 5120] f32 -> final_norm (RMSNorm with the
//! released norm.weight) -> LM head. The engine currently runs that norm on the host at
//! crates/sparkflash-engine/src/forward/mod.rs:1092, calling
//! crates/sparkflash-engine/src/linalg.rs:318's \`rms_norm\`. The device kernel
//! (kernels/rmsnorm_sm121.cu) takes the same operation with the serving ABI the head already reads:
//! BF16 rows in, BF16 rows out, the released norm.weight bytes as BF16 [5120].
//!
//! REFERENCE. Not a second copy of the formula: rmsnorm_ref::final_norm_bf16 decodes the released
//! BF16 bytes to the exact f32 values the engine holds, calls the engine's own
//! \`sparkflash_engine::linalg::rms_norm\` (the function the forward pass calls), and applies the
//! kernel's output rounding, \`m4_quant::bf16_bits\` - the same round-to-nearest-even as CUDA's
//! \`__float2bfloat16_rn\`. A change in the engine's norm arithmetic therefore shows up here as a
//! mismatch, not as a twin that changed with the kernel.
//!
//! OPERANDS. \`norm.weight\` (released, BF16 [5120]) and the released \`config.json\` (hidden 5120,
//! \`rms_norm_eps\`; the released config's value is 1e-20, NOT 1e-6, and the epsilon used is the
//! config's, cast to f32 exactly as the engine casts it). The [T, 5120] activation rows are the
//! released \`embed.weight\` rows 0..T - a declared stand-in for the collapsed tail hidden, exactly
//! as m17a/m17b declare theirs - because the released checkpoint has no captured final residual.
//!
//! COMPARISON CLASS, printed before any number it judges:
//!
//! 1. REAL WIDTH, released bytes: device BF16 output must be BIT-EXACT (zero mismatched codes) to
//!    bf16(engine rms_norm) on the same released bytes at the configured epsilon. This is the
//!    arithmetic class, not a tolerance: the kernel walks each row in ascending column order with
//!    the engine's rounding order, so bit-exactness is the expected result and any difference is a
//!    finding. Two launches must be byte-identical.
//! 2. EPSILON INSIDE: the same launch at eps = 1.0 must still be bit-exact to the reference at
//!    eps = 1.0, and must differ from the released-epsilon output (proving the epsilon is applied
//!    inside the kernel and not ignored). Both are exact comparisons.
//! 3. ODD WIDTH: a synthetic [3, 513] BF16 case (513 is not a multiple of eight, so the kernel's
//!    scalar path runs) must be bit-exact to the reference. The operand is declared synthetic.
//!
//! PINS. The cubin is refused by size+sha256 before any allocation, and the constants are
//! cross-checked against the embedded baseline manifest (the crate's single source of pins) at
//! build/test time: rmsnorm_sm121, 34,512 B,
//! \`fb559d1bd3666fc249c4f5b5bf46fa236004b3068355963876db43245764e678\`, 0 PTX membar lines.
//!
//! Usage (aarch64 node, checkpoint present):
//!
//! \`\`\`text
//! m18_final_norm --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 8
//! m18_final_norm --checkpoint <snapshot dir> --cubin-dir <dir> --tokens 130
//! \`\`\`
//!
//! Build lines (nvcc 13.0.88 / CUDA 13.0.3):
//! \`nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/rmsnorm_sm121.cu -o <dir>/rmsnorm_sm121.cubin\`.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_engine::config::ModelConfig;
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::rmsnorm_launch::{self as norm_launch, RmsNormLaunchParams};
use sparkflash_kernels::rmsnorm_ref::final_norm_bf16;

/// Hidden width of the released configuration (config.json text_config.hidden_size).
const HIDDEN: usize = 5120;
/// The released \`rms_norm_eps\` (config.json text_config.rms_norm_eps): 1e-20, not 1e-6.
const RELEASED_EPS: f32 = 1e-20;
/// The pin of kernels/rmsnorm_sm121.cu under nvcc 13.0.88 / sm_121a.
const CUBIN_BYTES: usize = 34_512;
const CUBIN_SHA256: &str = "fb559d1bd3666fc249c4f5b5bf46fa236004b3068355963876db43245764e678";
/// The synthetic odd width: not a multiple of eight, so the kernel takes its scalar path.
const ODD_WIDTH: usize = 513;

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
    odd_width: usize,
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
            odd_width: ODD_WIDTH,
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
                "--odd-width" => {
                    args.odd_width = next()?
                        .parse()
                        .map_err(|error| format!("--odd-width: {error}"))?;
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
        if args.odd_width == 0 || args.odd_width.is_multiple_of(8) {
            return Err("--odd-width must be non-zero and not a multiple of eight".to_owned());
        }
        if u32::try_from(args.tokens).is_err() || u32::try_from(args.odd_width).is_err() {
            return Err(
                "the row and column counts must fit the kernel's u32 parameters".to_owned(),
            );
        }
        Ok(args)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m18_final_norm: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m18: serving tail's standalone final RMSNorm (rmsnorm_sm121) at the released geometry ==="
    );
    println!(
        "geometry: hidden {HIDDEN}, tokens {}, released epsilon {RELEASED_EPS:e}, odd-width case {} columns",
        args.tokens, args.odd_width
    );
    println!(
        "reference: sparkflash_engine::linalg::rms_norm over the bf16-decoded released rows, then bf16_bits; the device kernel is the same operation with the head's BF16 ABI"
    );
    println!(
        "what this is not: not the served forward pass, not c4c engine integration, not the LM head, not a chain/token/per-layer claim, not a timing claim. The activation rows are released embed.weight rows, a declared stand-in for the collapsed tail hidden."
    );
    check_floor(args.floor_gib, "m18 start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The released configuration, read from the checkpoint's own config.json.
    let config_path = args.checkpoint.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config = ModelConfig::from_released_json(&config_text)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let eps = config.norm_eps as f32;
    println!(
        "released config.json: dim {}, rms_norm_eps {:e} (f64) -> {eps:e} (f32), vocab {}",
        config.dim, config.norm_eps, config.vocab_size
    );
    Verdict::push(
        &mut verdicts,
        "released geometry is the kernel's geometry",
        config.dim == HIDDEN && eps == RELEASED_EPS,
        format!(
            "dim {} == {HIDDEN}, rms_norm_eps {eps:e} == {RELEASED_EPS:e} (the epsilon the engine casts to f32)",
            config.dim
        ),
    );

    let weight_bytes = snap.read_raw("norm.weight", "BF16", &[HIDDEN])?;
    Verdict::push(
        &mut verdicts,
        "released norm.weight is BF16 [5120]",
        weight_bytes.len() == HIDDEN * 2,
        format!("{} bytes, 2 per BF16 column", weight_bytes.len()),
    );

    // Released activation stand-in: embed.weight rows 0..tokens, BF16.
    let mut x_bytes = Vec::with_capacity(args.tokens * HIDDEN * 2);
    for token in 0..args.tokens {
        let mut row = vec![0u8; HIDDEN * 2];
        snap.read_row_bf16_into("embed.weight", token, HIDDEN, &mut row)?;
        x_bytes.extend_from_slice(&row);
    }
    println!(
        "released operands: norm.weight BF16 {} B; activation stand-in embed.weight rows 0..{} = {} B",
        weight_bytes.len(),
        args.tokens,
        x_bytes.len()
    );

    let reference = final_norm_bf16(&x_bytes, &weight_bytes, args.tokens, HIDDEN, eps)?;
    println!(
        "host reference: {} BF16 codes ({} bytes) from the engine's rms_norm",
        reference.len() / 2,
        reference.len()
    );

    let cubin = pinned_cubin(&args.cubin_dir, CUBIN_BYTES, CUBIN_SHA256)?;
    check_pin_matches_baseline()?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(x_bytes.len(), &mut allocated)?;
    let weight_d = cuda.alloc(weight_bytes.len(), &mut allocated)?;
    let out_d = cuda.alloc(args.tokens * HIDDEN * 2, &mut allocated)?;
    println!("device bytes admitted: {allocated}");
    cuda.copy_htod(&x_d, &x_bytes)?;
    cuda.copy_htod(&weight_d, &weight_bytes)?;
    let (module, kernel) = norm_launch::load(&cuda, &cubin)?;
    let params = RmsNormLaunchParams {
        x: x_d,
        weight: weight_d,
        out: out_d,
        rows: args.tokens as u32,
        cols: HIDDEN as u32,
        eps,
        trace_ring: 0,
        layer_id: 0,
        step_seq: 0,
    };
    norm_launch::launch(&cuda, kernel, &params)?;
    let first = read_bf16_bytes(&cuda, &out_d, args.tokens * HIDDEN)?;

    let mismatches = first.iter().zip(&reference).filter(|(a, b)| a != b).count();
    Verdict::push(
        &mut verdicts,
        "real width: device BF16 bit-exact vs the engine reference",
        mismatches == 0,
        format!(
            "{mismatches} mismatched bytes of {} ({} BF16 codes), tokens {}, cols {HIDDEN}, eps {eps:e}",
            reference.len(),
            reference.len() / 2,
            args.tokens
        ),
    );

    norm_launch::launch(&cuda, kernel, &params)?;
    let second = read_bf16_bytes(&cuda, &out_d, args.tokens * HIDDEN)?;
    Verdict::push(
        &mut verdicts,
        "two launches byte-identical",
        first == second,
        format!("{} BF16 codes compared", first.len() / 2),
    );

    // Epsilon control: the same row at eps = 1.0, still bit-exact against the reference at 1.0,
    // and different from the released-epsilon output (so the epsilon is applied inside).
    let eps_control = 1.0f32;
    let reference_control =
        final_norm_bf16(&x_bytes, &weight_bytes, args.tokens, HIDDEN, eps_control)?;
    let control_params = RmsNormLaunchParams {
        eps: eps_control,
        ..params
    };
    norm_launch::launch(&cuda, kernel, &control_params)?;
    let control = read_bf16_bytes(&cuda, &out_d, args.tokens * HIDDEN)?;
    let control_mismatches = control
        .iter()
        .zip(&reference_control)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "eps = 1.0: device bit-exact vs the reference at 1.0",
        control_mismatches == 0,
        format!(
            "{control_mismatches} mismatched bytes of {}",
            reference_control.len()
        ),
    );
    let differing = control.iter().zip(&first).filter(|(a, b)| a != b).count();
    Verdict::push(
        &mut verdicts,
        "eps = 1.0 vs eps = 1e-20 outputs differ (epsilon is inside the kernel)",
        differing > 0,
        format!("{differing} differing bytes of {}", first.len()),
    );

    // Odd width, declared synthetic: [3, odd_width] BF16, the kernel's scalar path.
    let odd_rows = 3usize;
    let odd_x = synthetic_rows(odd_rows, args.odd_width, 37, 17);
    let odd_weight = synthetic_weights(args.odd_width, 11, 13);
    let odd_reference = final_norm_bf16(&odd_x, &odd_weight, odd_rows, args.odd_width, eps)?;
    let odd_x_d = cuda.alloc(odd_x.len(), &mut allocated)?;
    let odd_weight_d = cuda.alloc(odd_weight.len(), &mut allocated)?;
    let odd_out_d = cuda.alloc(odd_rows * args.odd_width * 2, &mut allocated)?;
    println!("odd-width device bytes admitted: {allocated} total");
    cuda.copy_htod(&odd_x_d, &odd_x)?;
    cuda.copy_htod(&odd_weight_d, &odd_weight)?;
    let odd_params = RmsNormLaunchParams {
        x: odd_x_d,
        weight: odd_weight_d,
        out: odd_out_d,
        rows: odd_rows as u32,
        cols: args.odd_width as u32,
        eps,
        trace_ring: 0,
        layer_id: 0,
        step_seq: 0,
    };
    norm_launch::launch(&cuda, kernel, &odd_params)?;
    let odd_device = read_bf16_bytes(&cuda, &odd_out_d, odd_rows * args.odd_width)?;
    let odd_mismatches = odd_device
        .iter()
        .zip(&odd_reference)
        .filter(|(a, b)| a != b)
        .count();
    Verdict::push(
        &mut verdicts,
        "odd width: device BF16 bit-exact vs the engine reference",
        odd_mismatches == 0,
        format!(
            "{odd_mismatches} mismatched bytes of {} ({} rows x {} columns, synthetic operands)",
            odd_reference.len(),
            odd_rows,
            args.odd_width
        ),
    );

    cuda.free(module, &[]);
    check_floor(args.floor_gib, "m18 end")?;

    println!("--- NOT claimed by m18 ---");
    println!(
        "not c4c engine integration (engine-dev wires the launch and the fallback), not the head or a token/chain/per-layer claim, not host-time saved, not a performance or bandwidth claim. The input rows are released embed.weight rows, a declared stand-in for the collapsed tail hidden; the odd-width operands are declared synthetic. What is measured is the arithmetic class: device BF16 equals bf16(engine rms_norm) on the same released bytes at the configured epsilon."
    );

    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name.as_str())
        .collect();
    if failed.is_empty() {
        println!(
            "m18: PASS ({} checks; tokens {}; one pinned cubin)",
            verdicts.len(),
            args.tokens
        );
        Ok(())
    } else {
        let details: Vec<String> = verdicts
            .iter()
            .filter(|verdict| !verdict.ok)
            .map(|verdict| format!("{}: {}", verdict.name, verdict.detail))
            .collect();
        Err(format!(
            "m18: FAIL at {} of {} checks: {:?}",
            failed.len(),
            verdicts.len(),
            details
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------------

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

fn pinned_cubin(dir: &std::path::Path, bytes: usize, sha: &str) -> Result<Vec<u8>, String> {
    let path = dir.join("rmsnorm_sm121.cubin");
    let cubin = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/rmsnorm_sm121.cu -o {})",
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
            "rmsnorm_sm121: pinned {bytes} B / {sha}, got {} B / {hex}; refusing to measure a different kernel",
            cubin.len()
        ));
    }
    Ok(cubin)
}

/// Cross-checks the harness constants against the crate's embedded baseline manifest, so the two
/// records of the pin cannot drift apart silently.
fn check_pin_matches_baseline() -> Result<(), String> {
    let pin = baseline::pin("rmsnorm_sm121")?
        .ok_or_else(|| "the baseline manifest does not pin rmsnorm_sm121".to_owned())?;
    if pin.cubin_bytes != CUBIN_BYTES
        || pin.cubin_sha256 != CUBIN_SHA256
        || pin.ptx_membar_lines != 0
    {
        return Err(format!(
            "rmsnorm_sm121: baseline pins {} B / {} / membar {}, the harness declares {CUBIN_BYTES} B / {CUBIN_SHA256} / membar 0",
            pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
        ));
    }
    println!(
        "baseline manifest agrees: rmsnorm_sm121 {} B, {}, {} PTX membar lines",
        pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
    );
    Ok(())
}

fn read_bf16_bytes(
    cuda: &Cuda,
    ptr: &sparkflash_kernels::cuda_driver::Deviceptr,
    values: usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; values * 2];
    cuda.copy_dtoh(&mut bytes, ptr)?;
    Ok(bytes)
}

/// Deterministic synthetic BF16 rows: values in [-1, 1) from a small modular pattern, encoded with
/// the same round-to-nearest-even the kernel's output uses.
fn synthetic_rows(rows: usize, cols: usize, seed: usize, modulus: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for row in 0..rows {
        for column in 0..cols {
            let code = ((row * cols + column) * seed) % modulus;
            let value = code as f32 / 8.0 - 1.0;
            bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
        }
    }
    bytes
}

/// Deterministic synthetic BF16 weights in [-1.5, 1.5).
fn synthetic_weights(cols: usize, seed: usize, modulus: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(cols * 2);
    for column in 0..cols {
        let code = (column * seed) % modulus;
        let value = code as f32 / 4.0 - 1.5;
        bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_harness_pins_are_the_baselines_pins() {
        check_pin_matches_baseline()
            .expect("the harness constants and the baseline manifest agree");
    }

    #[test]
    fn the_default_geometry_is_the_released_one() {
        assert_eq!(HIDDEN, 5_120);
        assert_eq!(RELEASED_EPS.to_bits(), 1e-20f32.to_bits());
        assert_eq!(ODD_WIDTH % 8, 1, "the odd width is not a multiple of eight");
    }

    #[test]
    fn arguments_default_to_tokens_8_and_reject_bad_shapes() {
        let parsed = Args::parse_from(vec!["--checkpoint".to_owned(), "ckpt".to_owned()])
            .expect("checkpoint alone parses");
        assert_eq!(parsed.tokens, 8);
        assert_eq!(parsed.odd_width, ODD_WIDTH);
        assert!(
            Args::parse_from(vec![
                "--checkpoint".to_owned(),
                "ckpt".to_owned(),
                "--tokens".to_owned(),
                "0".to_owned()
            ])
            .is_err()
        );
        assert!(
            Args::parse_from(vec![
                "--checkpoint".to_owned(),
                "ckpt".to_owned(),
                "--odd-width".to_owned(),
                "512".to_owned()
            ])
            .is_err()
        );
    }

    #[test]
    fn the_synthetic_patterns_are_deterministic_and_finite() {
        let rows = synthetic_rows(2, 9, 37, 17);
        assert_eq!(rows.len(), 2 * 9 * 2);
        assert_eq!(rows, synthetic_rows(2, 9, 37, 17));
        let weights = synthetic_weights(9, 11, 13);
        assert_eq!(weights.len(), 9 * 2);
        assert_eq!(weights, synthetic_weights(9, 11, 13));
    }

    #[test]
    fn the_harness_calls_the_engine_reference_and_the_typed_launcher() {
        let source = include_str!("m18_final_norm.rs");
        let (production, _) = source
            .split_once("fn the_harness_calls_the_engine_reference_and_the_typed_launcher")
            .expect("the source contains this test");
        assert!(production.contains("final_norm_bf16"));
        assert!(production.contains("norm_launch::launch"));
        assert!(production.contains("check_pin_matches_baseline"));
        assert!(
            !production.contains("fn rms_norm("),
            "no second norm formula"
        );
    }
}
