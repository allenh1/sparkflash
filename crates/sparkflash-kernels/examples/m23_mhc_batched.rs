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

//! m23_mhc_batched: K8 mHC row-count invariance and six-row launch cost, synthetic operands.
//!
//! Shape of the claim (declared before any number is printed). This harness does NOT use released
//! checkpoint bytes and makes no token, layer or end-to-end claim. It builds one deterministic
//! random operand set at the kernel's fixed shape (n = 4 streams, d = 5,120 hidden, 24 released
//! predictor rows, six rows), then:
//!   * launches the SCALAR symbol once per row (with the device init kernel, rows = 1), reading the
//!     row's six outputs after each launch;
//!   * launches the BATCHED symbol once over the same six rows (init clears six rows);
//!   * byte-compares, for every row and every output, the scalar row against the batched row.
//!
//! Outputs compared byte-for-byte: x (the n*d residual write), xhat_fp8, xhat_scales,
//! hc_stats, hc_coeffs, hc_rms_out and xhat_normed.
//!
//! This is the per-row independence proof the Lead asked for. mHC has no L: the hidden width is
//! fixed at 5,120, so the {37,129,517,4097} row-count gate does not apply to it. The fixed six-row
//! FLOOR is the batched symbol's own contract: it reads six batch-major rows and cannot be called
//! with one row; a one-row caller uses the scalar symbol.
//!
//! Timing: one scalar launch and the first batched launch are reported separately (first-call vs
//! steady p50), then p50 over --repeat iterations. The scalar figure is six launches per six rows
//! including six init-kernel launches; the batched figure is one init + one main launch. No H2D is
//! in either timed region.
//!
//! Usage: m23_mhc_batched --cubin-dir out [--repeat 50]

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mega_mhc_launch::{self, MegaMhcParams};

const KERNEL: &str = "mega_mhc_sm121";
const PIN_BYTES: usize = 151_592;
const PIN_SHA256: &str = "7040411617954a17e89e6d9d8fcea1a872171de259a5515bde1e69a96c69af5c";
const PIN_MEMBAR_LINES: u32 = 7;
const ROWS: usize = 6;
const STREAMS: usize = 4;
const HIDDEN: usize = 5120;
const PREDICTOR_ROWS: usize = 24;
const STATS: usize = 26;
const TILES: usize = 4;
const GRID_STATS: usize = (TILES + 1) * STATS;
const SCALES: usize = HIDDEN / 32;

fn main() {
    if let Err(error) = run() {
        eprintln!("m23_mhc_batched: {error}");
        std::process::exit(1);
    }
}

struct Args {
    cubin_dir: PathBuf,
    repeat: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut cubin_dir = None;
    let mut repeat = 50usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--cubin-dir" => {
                cubin_dir = Some(PathBuf::from(
                    args.next().ok_or("missing --cubin-dir value")?,
                ))
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .ok_or("missing --repeat value")?
                    .parse()
                    .map_err(|_| "bad --repeat".to_owned())?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        cubin_dir: cubin_dir.ok_or("--cubin-dir is required")?,
        repeat,
    })
}

fn load_checked(cubin_dir: &Path) -> Result<Vec<u8>, String> {
    let path = cubin_dir.join(format!("{KERNEL}.cubin"));
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let pin = baseline::pin(KERNEL)?
        .ok_or_else(|| format!("{KERNEL} is not in the baseline manifest"))?;
    if bytes.len() != PIN_BYTES || pin.cubin_bytes != PIN_BYTES {
        return Err(format!(
            "{KERNEL}: cubin is {} B but the pin is {PIN_BYTES} B",
            bytes.len()
        ));
    }
    let sha = format!("{:x}", Sha256::digest(&bytes));
    if sha != PIN_SHA256
        || pin.cubin_sha256 != PIN_SHA256
        || pin.ptx_membar_lines != PIN_MEMBAR_LINES
    {
        return Err(format!(
            "{KERNEL}: cubin pin mismatch ({sha}, {} membar lines)",
            pin.ptx_membar_lines
        ));
    }
    Ok(bytes)
}

/// A tiny deterministic LCG, so the harness has no checkpoint dependency and the operands are
/// reproducible between runs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0
    }
    fn positive(&mut self) -> f32 {
        0.5 + 0.5 * self.f32().abs()
    }
}

fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn bytes_of_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn bytes_of_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect()
}

struct Buffers {
    x: Deviceptr,
    y_prev: Deviceptr,
    b_prev: Deviceptr,
    c_prev: Deviceptr,
    a_prev: Deviceptr,
    predictor: Deviceptr,
    hc_scale: Deviceptr,
    hc_base: Deviceptr,
    rms_weight: Deviceptr,
    xhat_fp8: Deviceptr,
    xhat_scales: Deviceptr,
    hc_stats: Deviceptr,
    hc_sync: Deviceptr,
    hc_coeffs: Deviceptr,
    hc_rms_out: Deviceptr,
    xhat_normed: Deviceptr,
}

fn params_for_row(buffers: &Buffers, row: usize) -> MegaMhcParams {
    let x_row = buffers.x.offset(row * STREAMS * HIDDEN * 2);
    let y_row = buffers.y_prev.offset(row * HIDDEN * 2);
    let b_row = buffers.b_prev.offset(row * STREAMS * STREAMS * 4);
    let c_row = buffers.c_prev.offset(row * STREAMS * 4);
    let a_row = buffers.a_prev.offset(row * STREAMS * 4);
    let base_row = buffers.hc_base.offset(row * PREDICTOR_ROWS * 4);
    let fp8_row = buffers.xhat_fp8.offset(row * HIDDEN);
    let scales_row = buffers.xhat_scales.offset(row * SCALES);
    let stats_row = buffers.hc_stats.offset(row * GRID_STATS * 4);
    let sync_row = buffers.hc_sync.offset(row * 2 * 4);
    let coeffs_row = buffers.hc_coeffs.offset(row * 24 * 4);
    let rms_row = buffers.hc_rms_out.offset(row * 4);
    let normed_row = buffers.xhat_normed.offset(row * HIDDEN * 4);
    MegaMhcParams::new(
        x_row,
        y_row,
        b_row,
        c_row,
        a_row,
        buffers.predictor,
        buffers.hc_scale,
        base_row,
        buffers.rms_weight,
        fp8_row,
        scales_row,
        stats_row,
        sync_row,
        coeffs_row,
        rms_row,
        normed_row,
        0,
        0,
        0,
    )
}

fn params_for_batch(buffers: &Buffers) -> MegaMhcParams {
    MegaMhcParams::new(
        buffers.x,
        buffers.y_prev,
        buffers.b_prev,
        buffers.c_prev,
        buffers.a_prev,
        buffers.predictor,
        buffers.hc_scale,
        buffers.hc_base,
        buffers.rms_weight,
        buffers.xhat_fp8,
        buffers.xhat_scales,
        buffers.hc_stats,
        buffers.hc_sync,
        buffers.hc_coeffs,
        buffers.hc_rms_out,
        buffers.xhat_normed,
        0,
        0,
        0,
    )
}

fn read(cuda: &Cuda, pointer: Deviceptr, length: usize) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; length];
    cuda.copy_dtoh(&mut bytes, &pointer)?;
    Ok(bytes)
}

struct RowOutputs {
    x: Vec<u8>,
    fp8: Vec<u8>,
    scales: Vec<u8>,
    stats: Vec<u8>,
    coeffs: Vec<u8>,
    rms: Vec<u8>,
    normed: Vec<u8>,
}

fn read_row(cuda: &Cuda, buffers: &Buffers, row: usize) -> Result<RowOutputs, String> {
    let x = read(
        cuda,
        buffers.x.offset(row * STREAMS * HIDDEN * 2),
        STREAMS * HIDDEN * 2,
    )?;
    let fp8 = read(cuda, buffers.xhat_fp8.offset(row * HIDDEN), HIDDEN)?;
    let scales = read(cuda, buffers.xhat_scales.offset(row * SCALES), SCALES)?;
    let stats = read(
        cuda,
        buffers.hc_stats.offset(row * GRID_STATS * 4),
        GRID_STATS * 4,
    )?;
    let coeffs = read(cuda, buffers.hc_coeffs.offset(row * 24 * 4), 24 * 4)?;
    let rms = read(cuda, buffers.hc_rms_out.offset(row * 4), 4)?;
    let normed = read(
        cuda,
        buffers.xhat_normed.offset(row * HIDDEN * 4),
        HIDDEN * 4,
    )?;
    Ok(RowOutputs {
        x,
        fp8,
        scales,
        stats,
        coeffs,
        rms,
        normed,
    })
}

fn read_batch(cuda: &Cuda, buffers: &Buffers) -> Result<Vec<RowOutputs>, String> {
    (0..ROWS).map(|row| read_row(cuda, buffers, row)).collect()
}

fn p50(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let bytes = load_checked(&args.cubin_dir)?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let buffers = Buffers {
        x: cuda.alloc(ROWS * STREAMS * HIDDEN * 2, &mut allocated)?,
        y_prev: cuda.alloc(ROWS * HIDDEN * 2, &mut allocated)?,
        b_prev: cuda.alloc(ROWS * STREAMS * STREAMS * 4, &mut allocated)?,
        c_prev: cuda.alloc(ROWS * STREAMS * 4, &mut allocated)?,
        a_prev: cuda.alloc(ROWS * STREAMS * 4, &mut allocated)?,
        predictor: cuda.alloc(PREDICTOR_ROWS * STREAMS * HIDDEN * 4, &mut allocated)?,
        hc_scale: cuda.alloc(3 * 4, &mut allocated)?,
        hc_base: cuda.alloc(ROWS * PREDICTOR_ROWS * 4, &mut allocated)?,
        rms_weight: cuda.alloc(HIDDEN * 4, &mut allocated)?,
        xhat_fp8: cuda.alloc(ROWS * HIDDEN, &mut allocated)?,
        xhat_scales: cuda.alloc(ROWS * SCALES, &mut allocated)?,
        hc_stats: cuda.alloc(ROWS * GRID_STATS * 4, &mut allocated)?,
        hc_sync: cuda.alloc(ROWS * 2 * 4, &mut allocated)?,
        hc_coeffs: cuda.alloc(ROWS * 24 * 4, &mut allocated)?,
        hc_rms_out: cuda.alloc(ROWS * 4, &mut allocated)?,
        xhat_normed: cuda.alloc(ROWS * HIDDEN * 4, &mut allocated)?,
    };
    println!(
        "m23_mhc_batched: synthetic fixed shape n={STREAMS} d={HIDDEN} rows={ROWS}; allocated {allocated} B"
    );

    // Operands: x/y are bf16-representable, predictor/rms/b/c/a f32, hc_base small, rms positive.
    let mut rng = Rng(0x5eed_1234_abcd_9876);
    let mut x = Vec::with_capacity(ROWS * STREAMS * HIDDEN);
    for _ in 0..ROWS * STREAMS * HIDDEN {
        x.push(rng.f32());
    }
    let mut y_prev = Vec::with_capacity(ROWS * HIDDEN);
    for _ in 0..ROWS * HIDDEN {
        y_prev.push(rng.f32());
    }
    let mut b_prev = Vec::with_capacity(ROWS * STREAMS * STREAMS);
    for _ in 0..ROWS * STREAMS * STREAMS {
        b_prev.push(0.25 * rng.f32());
    }
    let mut c_prev = Vec::with_capacity(ROWS * STREAMS);
    for _ in 0..ROWS * STREAMS {
        c_prev.push(0.25 * rng.f32());
    }
    let mut a_prev = Vec::with_capacity(ROWS * STREAMS);
    for _ in 0..ROWS * STREAMS {
        a_prev.push(0.25 * rng.f32());
    }
    let mut predictor = Vec::with_capacity(PREDICTOR_ROWS * STREAMS * HIDDEN);
    for _ in 0..PREDICTOR_ROWS * STREAMS * HIDDEN {
        predictor.push(0.01 * rng.f32());
    }
    let hc_scale = [1.0f32, 1.0, 1.0];
    let mut hc_base = Vec::with_capacity(ROWS * PREDICTOR_ROWS);
    for _ in 0..ROWS * PREDICTOR_ROWS {
        hc_base.push(0.1 * rng.f32());
    }
    let mut rms_weight = Vec::with_capacity(HIDDEN);
    for _ in 0..HIDDEN {
        rms_weight.push(rng.positive());
    }

    // x is the one in-place operand: both the scalar and the batched kernel update it. Keep the
    // original bytes so the batched launch can be compared against the same starting operand the
    // scalar launches saw; without the re-upload the batched run starts from the scalar-updated
    // x and every row (including row 0) differs for a reason that is not the kernel's arithmetic.
    let x_operand = bytes_of_bf16(&x);
    cuda.copy_htod(&buffers.x, &x_operand)?;
    cuda.copy_htod(&buffers.y_prev, &bytes_of_bf16(&y_prev))?;
    cuda.copy_htod(&buffers.b_prev, &bytes_of_f32(&b_prev))?;
    cuda.copy_htod(&buffers.c_prev, &bytes_of_f32(&c_prev))?;
    cuda.copy_htod(&buffers.a_prev, &bytes_of_f32(&a_prev))?;
    cuda.copy_htod(&buffers.predictor, &bytes_of_f32(&predictor))?;
    cuda.copy_htod(&buffers.hc_scale, &bytes_of_f32(&hc_scale))?;
    cuda.copy_htod(&buffers.hc_base, &bytes_of_f32(&hc_base))?;
    cuda.copy_htod(&buffers.rms_weight, &bytes_of_f32(&rms_weight))?;
    cuda.copy_htod(&buffers.hc_sync, &bytes_of_u32(&[0u32; ROWS * 2]))?;
    println!("operands uploaded: no checkpoint bytes, no H2D inside any timed region");

    let (_scalar_module, scalar_kernel): (_, Kernel) = mega_mhc_launch::load(&cuda, &bytes)?;
    let (_main_module, main_kernel, _init_module, init_kernel) =
        mega_mhc_launch::load_batched(&cuda, &bytes)?;

    let t = Instant::now();
    for row in 0..ROWS {
        let params = params_for_row(&buffers, row);
        mega_mhc_launch::launch_with_device_init(&cuda, init_kernel, scalar_kernel, &params)?;
    }
    let first_scalar_ms = t.elapsed().as_secs_f64() * 1e3;
    let scalar_rows: Vec<RowOutputs> = (0..ROWS)
        .map(|row| read_row(&cuda, &buffers, row))
        .collect::<Result<_, _>>()?;

    // The scalar launches above left x updated in place; restore the original operand before the
    // batched launch so both paths start from identical bytes.
    cuda.copy_htod(&buffers.x, &x_operand)?;
    let params = params_for_batch(&buffers);
    let t = Instant::now();
    mega_mhc_launch::launch_batched(&cuda, init_kernel, main_kernel, &params)?;
    let first_batched_ms = t.elapsed().as_secs_f64() * 1e3;
    let batched_rows = read_batch(&cuda, &buffers)?;

    let mut all_exact = true;
    for row in 0..ROWS {
        let scalar = &scalar_rows[row];
        let batched = &batched_rows[row];
        let checks = [
            ("x", &scalar.x, &batched.x),
            ("xhat_fp8", &scalar.fp8, &batched.fp8),
            ("xhat_scales", &scalar.scales, &batched.scales),
            ("hc_stats", &scalar.stats, &batched.stats),
            ("hc_coeffs", &scalar.coeffs, &batched.coeffs),
            ("hc_rms_out", &scalar.rms, &batched.rms),
            ("xhat_normed", &scalar.normed, &batched.normed),
        ];
        for (name, left, right) in checks {
            if left != right {
                all_exact = false;
                println!(
                    "row {row} {name}: MISMATCH ({} vs {} bytes)",
                    left.len(),
                    right.len()
                );
            }
        }
    }
    println!(
        "per-row byte equality (scalar row r vs batched row r): {}",
        if all_exact { "EXACT" } else { "FAILED" }
    );

    let mut scalar_samples = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let t = Instant::now();
        for row in 0..ROWS {
            let params = params_for_row(&buffers, row);
            mega_mhc_launch::launch_with_device_init(&cuda, init_kernel, scalar_kernel, &params)?;
        }
        scalar_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut batched_samples = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let t = Instant::now();
        mega_mhc_launch::launch_batched(&cuda, init_kernel, main_kernel, &params)?;
        batched_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let scalar_p50 = p50(scalar_samples);
    let batched_p50 = p50(batched_samples);
    println!(
        "six rows: scalar 6 launches+6 inits p50 {scalar_p50:.4} ms (first {first_scalar_ms:.4} ms)"
    );
    println!(
        "six rows: batched 1 init+1 main p50 {batched_p50:.4} ms (first {first_batched_ms:.4} ms)"
    );
    println!(
        "launch/init ratio: {:.2}x; fixed six-row floor: the batched symbol reads six rows and cannot be called with one",
        scalar_p50 / batched_p50
    );
    println!(
        "mHC has no L parameter (hidden fixed 5120): the row-count gate applies to flash_mla_csa2_seq, not this kernel"
    );

    cuda.free_buffers(&[
        &buffers.x,
        &buffers.y_prev,
        &buffers.b_prev,
        &buffers.c_prev,
        &buffers.a_prev,
        &buffers.predictor,
        &buffers.hc_scale,
        &buffers.hc_base,
        &buffers.rms_weight,
        &buffers.xhat_fp8,
        &buffers.xhat_scales,
        &buffers.hc_stats,
        &buffers.hc_sync,
        &buffers.hc_coeffs,
        &buffers.hc_rms_out,
        &buffers.xhat_normed,
    ]);
    if !all_exact {
        return Err("row independence is not byte-exact".to_owned());
    }
    Ok(())
}
