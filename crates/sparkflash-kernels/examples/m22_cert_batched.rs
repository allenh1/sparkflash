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

//! m22: K8 batched certified LM head (cert_lm_head_batched_sm121) against the scalar
//! cert_lm_head_sm121 on six distinct hidden rows at one rank's released head shard.
//!
//! Comparison class (declared before any number): for every one of the six rows, logits[entry],
//! certified[entry], certified_count, block_max[block] and block_argmax[block] are BYTE-IDENTICAL
//! between the batched launch and the scalar launch on that same row's hidden bytes. The six rows
//! are distinct deterministic hidden seams (sinusoids, declared here, not a token).
//!
//! NOT claimed: no token, no chain, no end-to-end latency; this is one rank's kernel stage.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Instant;

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::cert_lm_head_launch::CertLmHeadBatchParams;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, KernelId};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::bf16_bits;

const KERNEL: &str = "cert_lm_head_sm121";
const ROWS: usize = 6;
const HIDDEN: usize = 5_120;
const VOCAB: usize = 129_280;
const VOCAB_PER_RANK: usize = 32_320;
const GRID: usize = VOCAB_PER_RANK.div_ceil(1_024);
const BLOCKS: usize = HIDDEN / 128;
const THREADS: usize = 128;
const RANK: usize = 0;

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    repeat: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            repeat: 20,
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
                "--repeat" => {
                    args.repeat = next()?.parse().map_err(|_| "bad --repeat".to_owned())?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: m22_cert_batched --checkpoint <snapshot> --cubin-dir <dir> [--repeat 20]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}")),
            }
        }
        Ok(args)
    }
}

fn hidden_rows() -> Vec<Vec<u8>> {
    (0..ROWS)
        .map(|row| {
            let mut bytes = Vec::with_capacity(HIDDEN * 2);
            for i in 0..HIDDEN {
                let value = ((i as f32 * 0.013 + row as f32 * 0.71).sin()
                    + 0.25 * (i as f32 * 0.0007).cos())
                    * 0.5;
                bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            bytes
        })
        .collect()
}

fn read_bytes(cuda: &Cuda, ptr: Deviceptr, bytes: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; bytes];
    cuda.copy_dtoh(&mut out, &ptr)?;
    Ok(out)
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

thread_local! {
    static FIRST_SCALAR_MS: std::cell::RefCell<f64> = const { std::cell::RefCell::new(0.0) };
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m22_cert_batched: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!("=== m22_cert_batched: K8 six-row certified head vs the scalar head, rank 0 ===");
    println!(
        "comparison class (declared before any number): six DISTINCT deterministic hidden rows, one \
         rank's 32,320-row released head shard; batched vs scalar on the same row bytes must be \
         BYTE-IDENTICAL for logits, certified flags, certified_count and both block arrays. The six \
         rows are sinusoids declared here, not a token; no chain or token claim."
    );

    let mut snap = Snapshot::open(&args.checkpoint)?;
    let head = snap.read_raw("head.weight", "BF16", &[VOCAB, HIDDEN])?;
    if head.len() != VOCAB * HIDDEN * 2 {
        return Err("head.weight size mismatch".to_owned());
    }
    let block_maxima = lm_head_ref::block_maxima_bf16(&head);
    let rank_head = RANK * VOCAB_PER_RANK * HIDDEN * 2;
    // The head's bound geometry: BLOCKS hidden screen blocks per entry. GRID is the number of
    // 1,024-entry vocabulary chunks the kernel launches; the two must not be conflated.
    let rank_max = RANK * VOCAB_PER_RANK * BLOCKS * 2;
    let rank_maxima = &block_maxima[rank_max..rank_max + VOCAB_PER_RANK * BLOCKS * 2];
    // Transpose [vocab][40] -> [40][vocab] so a warp's bound reads are contiguous; declared
    // batched-only layout, computed from the same exact K4 precompute.
    let mut transposed = vec![0u8; VOCAB_PER_RANK * BLOCKS * 2];
    for entry in 0..VOCAB_PER_RANK {
        for block in 0..BLOCKS {
            let src = (entry * BLOCKS + block) * 2;
            let dst = (block * VOCAB_PER_RANK + entry) * 2;
            transposed[dst..dst + 2].copy_from_slice(&rank_maxima[src..src + 2]);
        }
    }
    let rows = hidden_rows();
    println!(
        "released head.weight: {} B, rank 0 shard {} B, block maxima {} B, {} hidden rows x {} B",
        head.len(),
        VOCAB_PER_RANK * HIDDEN * 2,
        VOCAB_PER_RANK * BLOCKS * 2,
        rows.len(),
        rows[0].len()
    );

    let cuda = Cuda::load()?;
    let cubin = std::fs::read(args.cubin_dir.join(format!("{KERNEL}.cubin")))
        .map_err(|e| format!("{KERNEL}.cubin: {e}"))?;
    let (_module, single) = cuda.load_kernel(&cubin, KernelId::CertLmHead)?;
    let (_batch_module, batched) = cuda.load_cert_lm_head_batched(&cubin)?;

    let mut allocated = 0usize;
    let weight = cuda.alloc(VOCAB_PER_RANK * HIDDEN * 2, &mut allocated)?;
    let block_max = cuda.alloc(VOCAB_PER_RANK * BLOCKS * 2, &mut allocated)?;
    let block_max_t = cuda.alloc(VOCAB_PER_RANK * BLOCKS * 2, &mut allocated)?;
    let hidden = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let logits = cuda.alloc(VOCAB_PER_RANK * 4, &mut allocated)?;
    let certified = cuda.alloc(VOCAB_PER_RANK, &mut allocated)?;
    let count = cuda.alloc(4, &mut allocated)?;
    let bmax = cuda.alloc(GRID * 4, &mut allocated)?;
    let bamax = cuda.alloc(GRID * 4, &mut allocated)?;
    let batch_hidden = cuda.alloc(ROWS * HIDDEN * 2, &mut allocated)?;
    let batch_logits = cuda.alloc(ROWS * VOCAB_PER_RANK * 4, &mut allocated)?;
    let batch_certified = cuda.alloc(ROWS * VOCAB_PER_RANK, &mut allocated)?;
    let batch_count = cuda.alloc(ROWS * 4, &mut allocated)?;
    let batch_bmax = cuda.alloc(ROWS * GRID * 4, &mut allocated)?;
    let batch_bamax = cuda.alloc(ROWS * GRID * 4, &mut allocated)?;
    cuda.copy_htod(
        &weight,
        &head[rank_head..rank_head + VOCAB_PER_RANK * HIDDEN * 2],
    )?;
    cuda.copy_htod(&block_max, rank_maxima)?;
    cuda.copy_htod(&block_max_t, &transposed)?;
    for (row, bytes) in rows.iter().enumerate() {
        cuda.copy_htod(&batch_hidden.offset(row * HIDDEN * 2), bytes)?;
    }
    println!("device buffers: {allocated} B");

    let single_params = |hidden: Deviceptr| {
        [
            hidden.0,
            weight.0,
            block_max.0,
            logits.0,
            certified.0,
            count.0,
            bmax.0,
            bamax.0,
            VOCAB_PER_RANK as u64,
            0,
            0,
            0,
        ]
    };
    let batch_params = CertLmHeadBatchParams::new(
        batch_hidden,
        weight,
        block_max_t,
        batch_logits,
        batch_certified,
        batch_count,
        batch_bmax,
        batch_bamax,
        VOCAB_PER_RANK as u32,
        0,
        0,
        0,
    );
    let grid = (GRID as u32, 1, 1);
    let block = (THREADS as u32, 1, 1);

    struct RowOut {
        logits: Vec<u8>,
        certified: Vec<u8>,
        count: Vec<u8>,
        bmax: Vec<u8>,
        bamax: Vec<u8>,
    }
    let mut scalar_rows = Vec::with_capacity(ROWS);
    for (row, bytes) in rows.iter().enumerate() {
        cuda.copy_htod(&hidden, bytes)?;
        cuda.copy_htod(&count, &[0u8; 4])?;
        let first_row_start = Instant::now();
        cuda.launch(single, grid, block, &single_params(hidden))?;
        if row == 0 {
            FIRST_SCALAR_MS
                .with(|slot| *slot.borrow_mut() = first_row_start.elapsed().as_secs_f64() * 1.0e3);
        }
        scalar_rows.push(RowOut {
            logits: read_bytes(&cuda, logits, VOCAB_PER_RANK * 4)?,
            certified: read_bytes(&cuda, certified, VOCAB_PER_RANK)?,
            count: read_bytes(&cuda, count, 4)?,
            bmax: read_bytes(&cuda, bmax, GRID * 4)?,
            bamax: read_bytes(&cuda, bamax, GRID * 4)?,
        });
        if row == 0 {
            let flags = scalar_rows[0]
                .certified
                .iter()
                .filter(|flag| **flag != 0)
                .count();
            println!("scalar row 0 certified flags: {flags} of {VOCAB_PER_RANK}");
        }
    }

    cuda.copy_htod(&batch_count, &[0u8; ROWS * 4])?;
    let first_batch_start = Instant::now();
    cuda.launch(batched, grid, block, batch_params.as_slice())?;
    let first_batch_ms = first_batch_start.elapsed().as_secs_f64() * 1.0e3;
    let batch_logits_bytes = read_bytes(&cuda, batch_logits, ROWS * VOCAB_PER_RANK * 4)?;
    let batch_certified_bytes = read_bytes(&cuda, batch_certified, ROWS * VOCAB_PER_RANK)?;
    let batch_count_bytes = read_bytes(&cuda, batch_count, ROWS * 4)?;
    let batch_bmax_bytes = read_bytes(&cuda, batch_bmax, ROWS * GRID * 4)?;
    let batch_bamax_bytes = read_bytes(&cuda, batch_bamax, ROWS * GRID * 4)?;

    let mut mismatches = 0usize;
    for (row, scalar) in scalar_rows.iter().enumerate() {
        let logits_row =
            &batch_logits_bytes[row * VOCAB_PER_RANK * 4..(row + 1) * VOCAB_PER_RANK * 4];
        let certified_row =
            &batch_certified_bytes[row * VOCAB_PER_RANK..(row + 1) * VOCAB_PER_RANK];
        let count_row = &batch_count_bytes[row * 4..(row + 1) * 4];
        let bmax_row = &batch_bmax_bytes[row * GRID * 4..(row + 1) * GRID * 4];
        let bamax_row = &batch_bamax_bytes[row * GRID * 4..(row + 1) * GRID * 4];
        if logits_row != scalar.logits.as_slice()
            || certified_row != scalar.certified.as_slice()
            || count_row != scalar.count.as_slice()
            || bmax_row != scalar.bmax.as_slice()
            || bamax_row != scalar.bamax.as_slice()
        {
            mismatches += 1;
        }
    }
    println!(
        "byte comparison: {}/{} rows identical across logits, certified, count, block_max and \
         block_argmax (mismatched rows: {mismatches})",
        ROWS - mismatches,
        ROWS
    );

    cuda.copy_htod(&hidden, &rows[0])?;
    let first_single_ms = FIRST_SCALAR_MS.with(|slot| *slot.borrow());
    let (mut single_samples, _) =
        cuda.launch_repeated(single, grid, block, &single_params(hidden), args.repeat)?;
    let (mut batch_samples, _) =
        cuda.launch_repeated(batched, grid, block, batch_params.as_slice(), args.repeat)?;
    single_samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    batch_samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let (smin, sp50, smean, smax) = timing_stats(&single_samples);
    let (bmin, bp50, bmean, bmax) = timing_stats(&batch_samples);
    let weight_bytes = VOCAB_PER_RANK * HIDDEN * 2
        + VOCAB_PER_RANK * BLOCKS * 2
        + ROWS * HIDDEN * 2
        + ROWS * VOCAB_PER_RANK * 4
        + ROWS * VOCAB_PER_RANK
        + 2 * ROWS * GRID * 4;
    let floor_ms = weight_bytes as f64 / 273.0e9 * 1.0e3;
    println!(
        "first-launch warm-up (same process, after module load and buffer upload): scalar one row \
         {first_single_ms:.4} ms vs its p50 {sp50:.4} ms; batched six rows {first_batch_ms:.4} ms \
         vs its p50 {bp50:.4} ms"
    );
    println!(
        "timing rank 0: scalar 1 row min {smin:.4} p50 {sp50:.4} mean {smean:.4} max {smax:.4} ms; \
         batched 6 rows min {bmin:.4} p50 {bp50:.4} mean {bmean:.4} max {bmax:.4} ms; ratio to one \
         scalar launch {:.3}x; 6 sequential scalar launches {:.4} ms; actual-traffic roofline \
         {:.4} ms ({:.1}% of it)",
        bp50 / sp50,
        sp50 * ROWS as f64,
        floor_ms,
        100.0 * floor_ms / bp50
    );
    if mismatches == 0 {
        println!(
            "m22_cert_batched: PASS (six distinct rows byte-identical to the scalar head; no token \
             or chain claimed)"
        );
        Ok(())
    } else {
        Err(format!("{mismatches} rows differ from the scalar head"))
    }
}
