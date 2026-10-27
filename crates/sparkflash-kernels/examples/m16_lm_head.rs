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

//! m16: the certified LM head (cert_lm_head_sm121, the head screen) in isolation, at the real
//! released head geometry.
//!
//! This is the Phase-3 stage harness for the last kernel between a full layer stack and a token.
//! The subject is one cert_lm_head_sm121 call per tensor-parallel rank over that rank's 32,320 rows
//! of the released head.weight (BF16 [129,280 x 5,120]), hidden width 5,120, grid 32 x 128 threads,
//! the pinned cubin 14,888 B / 7394b129... (D31: the count reduction is fixed and re-pinned;
//! windows b21/b23 measured the pre-fix a833a336... artifact and keep that sha in their record),
//! verified by size and sha256 before any device allocation. The four rank shards are all launched in this run and the 129,280-way argmax is
//! assembled from all four - a single rank is never presented as "the argmax".
//!
//! TWO DECLARED HIDDEN SEAMS (the head's input is not a released tensor):
//!
//!   * real-token control: --head-input holds the head input the phase-1 host chain produced in a
//!     Lead window on the released checkpoint (~/hostrun/phase1-host-token0.head.f32, f32 5,120,
//!     sha256 e6819e83...). It is sha256- and size-verified here and the harness REFUSES to run if it
//!     is missing or differs - there is no fabricated fallback. It was produced by the phase-1 chain
//!     in a Lead window, not by m16; it is a declared input, not a measured output of this harness.
//!     The head consumes it as the kernel's bf16 ABI, so the twin below is built from the same bf16
//!     codes: this compares the head's arithmetic, not the f32-to-bf16 conversion (measured in
//!     phase 1 as at most 2.4320781e-2 on a logit).
//!   * concentrated certificate case: a synthetic hidden, DECLARED here as bf16(1.0) at index 7 and
//!     zero elsewhere. It is not a released hidden state and no token is claimed from it. Its point
//!     is that the kernel's bound collides with the exact value in block 0 (bound_j = max over the
//!     first 128 weights of |w_ji|, and exact_j = w_j7 is one of those 128), so the certificate
//!     actually prunes on device - the phase-1 run certified 0 entries, so this is the first device
//!     execution of the pruning path if it prunes, and the real-token case is the non-pruning
//!     control.
//!
//! COMPARISON CLASS (declared before any number is printed). The kernel's PTX (pinned nvcc 13.0.3,
//! sm_121a) has 17 fma.rn.f32 and zero mul.rn/add.rn: the exact dot is one ascending fma chain per
//! entry, and the per-128-block weight maximum is an abs-max. Because both operands are bf16, every
//! product fits f32 exactly, so the fused chain and a mul-then-add chain are bit-identical - the
//! twin uses the fused chain the PTX shows and the class is not a tolerance. Per entry:
//!
//!   * exact entries (certified == 0): device logit BIT-IDENTICAL to the twin's bf16-matched
//!     ascending fma dot; the capture count is over all 129,280 entries.
//!   * certified entries (certified == 1): the stored value is the certificate bound, so the device
//!     value is checked as a BOUND, not as a logit: (i) it must be >= the twin's exact value within
//!     f32 accumulation eps, (ii) the kernel's own pruning rule is replayed in its per-thread order
//!     (chunk, lane, ascending steps; bound < running max, running max never updated on a skip) and
//!     every certified entry must satisfy it, so the pruning is proven sound on the measured values,
//!     and (iii) it is compared to the same formula computed on the host (absolute block sums times
//!     the rows per-block abs-max) within a declared bar, because the block-sum reduction order
//!     differs (the kernel tree-reduces, the host accumulates sequentially).
//!   * certified_count: the kernel's atomic count must equal the number of certified flags.
//!   * block reductions: each of the 32 blocks published block_max/block_argmax must equal a
//!     per-entry scan of that block's device logits with the lowest-entry tie-break.
//!   * the global 129,280-way argmax (lowest entry index on ties, across the four rank shards) must
//!     equal the twin's argmax; in the real-token control, with --reference-logits, it must also
//!     equal the phase-1 reference engine's argmax over its 129,280 f32 logits (the reference used
//!     the f32 head input, so this is a measured equality with the bf16-conversion caveat, not a
//!     proof). The reference logits are the real token's, so the synthetic concentrated case reports
//!     the reference argmax as not applicable instead of being gated on a comparison with no
//!     meaning for its hidden.
//!   * determinism: two device runs of every case are byte-identical across logits, certified flags,
//!     certified_count and the block arrays.
//!   * the top-5 SET is claimed against the reference only in the real-token control and only if
//!     that case certified nothing (then all 129,280 values are exact); the kernel's contract is the
//!     argmax, and a certified entry stores a bound rather than its value, so no top-5 value/set is
//!     claimed from the pruning case.
//!
//! NOT claimed: no token is claimed here (the head is a stage, not the chain), no throughput claim,
//! no latency claim, no statement about the f32-to-bf16 conversion (phase-1 measured), and no claim
//! that the concentrated case corresponds to any token.
//!
//! WINDOW COMMAND (from the node, with vLLM stopped; the cubin dir must hold the pinned
//! cert_lm_head_sm121.cubin, 14,888 B, 7394b129...; windows b21/b23 used the pre-fix
//! 14,712 B a833a336... artifact):
//!
//!     ~/phase2-evidence/window-2026-10-18/m16_lm_head_<sha8>_aarch64 \
//!         --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
//!         --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins \
//!         --head-input ~/hostrun/phase1-host-token0.head.f32 \
//!         --reference-logits ~/hostrun/phase1-host-token0.logits.f32 \
//!         --repeat 100
//!
//! The harness refuses to run below 32 GiB MemAvailable (--floor-gib) at start and end.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, bytes_to_f32};
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, KernelId};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::{bf16_bits, bf16_bits_to_f32};

/// The cubin name in the baseline manifest (the nvcc source stem).
const KERNEL: &str = "cert_lm_head_sm121";
/// The pinned cubin size, from baseline/kernel_baseline.txt through baseline::pin.
const PIN_BYTES: usize = 36_800;
/// The pinned cubin sha256, from the same manifest.
const PIN_SHA256: &str = "bbd30c8d6d358bd200b07cffa54d662c9144501b69310864933b68a0e53b6798";
/// The pinned PTX membar-line count.
const PIN_MEMBAR_LINES: u32 = 0;

const HIDDEN: usize = 5_120;
const VOCAB: usize = 129_280;
const VOCAB_PER_RANK: usize = 32_320;
const RANKS: usize = 4;
const SCREEN_TILE: usize = 128;
const LM_BLOCKS: usize = HIDDEN / SCREEN_TILE;
const THREADS: usize = 128;
const CHUNK: usize = 1_024;
const ENTRIES_PER_THREAD: usize = 8;
const GRID: usize = VOCAB_PER_RANK.div_ceil(CHUNK);

/// The declared hidden seam pins.
const HEAD_INPUT_BYTES: usize = HIDDEN * 4;
const HEAD_INPUT_SHA256: &str = "e6819e83a32c13fa6843ec576ba3959bbb9cb645047d59e445eaa235a7846c04";
const REFERENCE_LOGITS_BYTES: usize = VOCAB * 4;
/// The concentrated case: a single bf16(1.0) at this index and zeros elsewhere.
const SYNTHETIC_INDEX: usize = 7;

/// The declared bound bar: the block sums are tree-reduced on device and accumulated sequentially
/// on the host, so the two bounds differ by f32 accumulation, not by the formula.
const BOUND_BAR_ABS: f32 = 1.0e-3;
const BOUND_BAR_REL: f32 = 1.0e-5;
/// The certificate eps: a bound is a proof on exact reals; f32 accumulation may put it below the
/// measured exact value by a few ulps. Validity is checked with this eps and the margin reported.
const CERT_EPS_ABS: f32 = 1.0e-4;
const CERT_EPS_REL: f32 = 1.0e-5;

const _: () = assert!(
    VOCAB == VOCAB_PER_RANK * RANKS,
    "the head shards across ranks"
);
const _: () = assert!(
    GRID * CHUNK >= VOCAB_PER_RANK && (GRID - 1) * CHUNK < VOCAB_PER_RANK,
    "the grid covers the per-rank vocabulary in whole chunks"
);
const _: () = assert!(
    CHUNK == THREADS * ENTRIES_PER_THREAD,
    "one thread, eight entries"
);

fn main() {
    if let Err(error) = run() {
        eprintln!("m16_lm_head: {error}");
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    head_input: PathBuf,
    reference_logits: Option<PathBuf>,
    repeat: usize,
    floor_gib: u64,
    twin_only: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            head_input: PathBuf::new(),
            reference_logits: None,
            repeat: 100,
            floor_gib: 32,
            twin_only: false,
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
                "--head-input" => args.head_input = PathBuf::from(next()?),
                "--reference-logits" => args.reference_logits = Some(PathBuf::from(next()?)),
                "--repeat" => {
                    args.repeat = next()?
                        .parse()
                        .map_err(|_| "--repeat needs a number".to_owned())?
                }
                "--twin-only" => args.twin_only = true,
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|_| "--floor-gib needs a number".to_owned())?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: m16_lm_head --checkpoint <snapshot dir> --head-input <f32 5120> \
                         [--reference-logits <f32 129280>] [--cubin-dir out] [--repeat 100] \
                         [--floor-gib 32] [--twin-only]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}; try --help")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required (the released snapshot directory)".to_owned());
        }
        if args.head_input.as_os_str().is_empty() {
            return Err(
                "--head-input is required: the declared real-token hidden seam (see --help); \
                 this harness refuses to fabricate a hidden state"
                    .to_owned(),
            );
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// The declared hidden seam for one case, in the kernel's bf16 ABI and as bf16-matched f32 values.
struct HiddenSeam {
    name: &'static str,
    provenance: String,
    bytes: Vec<u8>,
    values: Vec<f32>,
}

/// The declared real-token seam: the phase-1 chain's head input, sha- and size-verified, refused if
/// it is missing or differs. No fallback exists on purpose.
fn real_token_hidden(path: &Path) -> Result<HiddenSeam, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "{}: {e} (the declared phase-1 chain head input)",
            path.display()
        )
    })?;
    if bytes.len() != HEAD_INPUT_BYTES {
        return Err(format!(
            "{}: the head input is {} B, the declared artifact is {HEAD_INPUT_BYTES} B (5120 f32)",
            path.display(),
            bytes.len()
        ));
    }
    let sha = sha256_hex(&bytes);
    if sha != HEAD_INPUT_SHA256 {
        return Err(format!(
            "{}: sha256 {sha} is not the declared phase-1 chain artifact {HEAD_INPUT_SHA256}: \
             refusing to substitute or fabricate a hidden state",
            path.display()
        ));
    }
    let f32_values = bytes_to_f32(&bytes);
    let (bytes, values) = bf16_codes(&f32_values, HIDDEN)?;
    Ok(HiddenSeam {
        name: "real-token control",
        provenance: format!(
            "phase-1 host chain head input from a Lead window, sha256-verified here (not produced \
             by m16): {} f32, sha256 {sha}",
            path.display()
        ),
        bytes,
        values,
    })
}

/// The declared concentrated case: bf16(1.0) at SYNTHETIC_INDEX, zeros elsewhere. Not a released
/// hidden state and no token is claimed from it.
fn concentrated_hidden() -> HiddenSeam {
    let mut values = vec![0.0f32; HIDDEN];
    values[SYNTHETIC_INDEX] = 1.0;
    let (bytes, values) = bf16_codes(&values, HIDDEN).expect("the declared values are 5120 wide");
    HiddenSeam {
        name: "concentrated certificate case",
        provenance: format!(
            "synthetic, declared formula: bf16(1.0) at index {SYNTHETIC_INDEX}, zero elsewhere; not \
             a released hidden state and not a token"
        ),
        bytes,
        values,
    }
}

/// The kernel's bf16 ABI for a hidden vector, plus the bf16-matched f32 values the twin consumes.
fn bf16_codes(values: &[f32], width: usize) -> Result<(Vec<u8>, Vec<f32>), String> {
    if values.len() != width {
        return Err(format!("the hidden state must be {width} wide"));
    }
    let mut bytes = vec![0u8; width * 2];
    let mut matched = vec![0.0f32; width];
    for (index, value) in values.iter().enumerate() {
        let code = bf16_bits(*value);
        bytes[2 * index..2 * index + 2].copy_from_slice(&code.to_le_bytes());
        matched[index] = bf16_bits_to_f32(code);
    }
    Ok((bytes, matched))
}

fn bf16_at(bytes: &[u8], index: usize) -> f32 {
    bf16_bits_to_f32(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]))
}

/// The twin's exact dot: one ascending fma chain over bf16-matched f32 values and bf16 weight codes,
/// the instruction sequence the pinned PTX contains.
fn dot_fma(hidden: &[f32], weights: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (value, weight) in hidden.iter().zip(weights.iter()) {
        acc = value.mul_add(*weight, acc);
    }
    acc
}

/// The absolute sum of one screen block.
fn block_sum_abs(hidden: &[f32], block: usize) -> f32 {
    let base = block * SCREEN_TILE;
    hidden[base..base + SCREEN_TILE]
        .iter()
        .map(|value| value.abs())
        .sum()
}

/// The certificate bound for one row: sum over blocks of (sum |h| in block) * (max |w| in block),
/// in the kernel's block order. The kernel tree-reduces each 128-wide block sum while this
/// accumulates sequentially, so the bound is compared with a declared bar rather than bit-for-bit.
fn row_bound(block_sums: &[f32], weights: &[f32]) -> f32 {
    let mut bound = 0.0f32;
    for (block, sum) in block_sums.iter().enumerate() {
        let base = block * SCREEN_TILE;
        let row_max = weights[base..base + SCREEN_TILE]
            .iter()
            .fold(0.0f32, |best, value| best.max(value.abs()));
        bound = sum.mul_add(row_max, bound);
    }
    bound
}

/// The twin for one rank shard: exact logits and certificate bounds.
struct Twin {
    logits: Vec<f32>,
    bounds: Vec<f32>,
}

/// The twin over a rank's rows of the released head. weights is this shard's bf16 rows.
fn twin_rank(hidden: &[f32], weights: &[u8]) -> Twin {
    let block_sums: Vec<f32> = (0..LM_BLOCKS)
        .map(|block| block_sum_abs(hidden, block))
        .collect();
    let mut row = vec![0.0f32; HIDDEN];
    let mut logits = vec![0.0f32; VOCAB_PER_RANK];
    let mut bounds = vec![0.0f32; VOCAB_PER_RANK];
    for entry in 0..VOCAB_PER_RANK {
        let base = entry * HIDDEN * 2;
        for (index, slot) in row.iter_mut().enumerate() {
            *slot = bf16_at(&weights[base..], index);
        }
        logits[entry] = dot_fma(hidden, &row);
        bounds[entry] = row_bound(&block_sums, &row);
    }
    Twin { logits, bounds }
}

/// The score of a logit slice with the lowest-index tie-break: strict greater-than only.
fn best_entry(logits: &[f32]) -> (f32, usize) {
    let mut best = f32::NEG_INFINITY;
    let mut at = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > best {
            best = *value;
            at = index;
        }
    }
    (best, at)
}

/// The global argmax across the rank shards, on global entry indices, lowest index on ties.
fn global_argmax(per_rank: &[Vec<f32>]) -> (f32, usize) {
    let mut best = f32::NEG_INFINITY;
    let mut at = 0usize;
    for (rank, logits) in per_rank.iter().enumerate() {
        for (entry, value) in logits.iter().enumerate() {
            if *value > best {
                best = *value;
                at = rank * VOCAB_PER_RANK + entry;
            }
        }
    }
    (best, at)
}

/// Replays the kernel's per-thread pruning rule over the measured values. Order is
/// (chunk, lane, ascending step) as the kernel visits entries; the running maximum starts at
/// -FLT_MAX, is not updated on a skip, and a certified entry must satisfy bound < running max
/// exactly as the kernel's strict test does.
fn replay_certificate(
    logits: &[f32],
    certified: &[u8],
    vocab: usize,
    chunk: usize,
    threads: usize,
    per_thread: usize,
) -> (usize, usize) {
    let mut violations = 0usize;
    let mut seen = 0usize;
    for block in 0..vocab.div_ceil(chunk) {
        let base = block * chunk;
        for lane in 0..threads {
            let mut running_max = f32::NEG_INFINITY;
            for step in 0..per_thread {
                let entry = base + step * threads + lane;
                if entry >= vocab {
                    continue;
                }
                if certified[entry] == 1 {
                    seen += 1;
                    if logits[entry] >= running_max {
                        violations += 1;
                    }
                } else if logits[entry] > running_max {
                    running_max = logits[entry];
                }
            }
        }
    }
    (violations, seen)
}

fn bound_within_bar(delta: f32, host_bound: f32) -> bool {
    delta <= BOUND_BAR_ABS + BOUND_BAR_REL * host_bound.abs()
}

fn certificate_holds(device_bound: f32, twin_exact: f32) -> bool {
    device_bound >= twin_exact - (CERT_EPS_ABS + CERT_EPS_REL * twin_exact.abs())
}

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
    let sha = sha256_hex(&bytes);
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

struct DeviceBuffers {
    hidden: Deviceptr,
    weight: Deviceptr,
    block_maxima: Deviceptr,
    logits: Deviceptr,
    certified: Deviceptr,
    count: Deviceptr,
    block_max: Deviceptr,
    block_argmax: Deviceptr,
}

impl DeviceBuffers {
    fn allocate(cuda: &Cuda) -> Result<(Self, usize), String> {
        let mut allocated = 0usize;
        let hidden = cuda.alloc(HIDDEN * 2, &mut allocated)?;
        let weight = cuda.alloc(VOCAB * HIDDEN * 2, &mut allocated)?;
        let block_maxima = cuda.alloc(VOCAB * LM_BLOCKS * 2, &mut allocated)?;
        let logits = cuda.alloc(VOCAB_PER_RANK * 4, &mut allocated)?;
        let certified = cuda.alloc(VOCAB_PER_RANK, &mut allocated)?;
        let count = cuda.alloc(4, &mut allocated)?;
        let block_max = cuda.alloc(GRID * 4, &mut allocated)?;
        let block_argmax = cuda.alloc(GRID * 4, &mut allocated)?;
        Ok((
            Self {
                hidden,
                weight,
                block_maxima,
                logits,
                certified,
                count,
                block_max,
                block_argmax,
            },
            allocated,
        ))
    }
}

/// One rank's measured outputs.
#[derive(Clone)]
struct RankOutputs {
    logits: Vec<f32>,
    certified: Vec<u8>,
    certified_count: usize,
    block_max: Vec<f32>,
    block_argmax: Vec<u32>,
}

impl RankOutputs {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in &self.logits {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.certified);
        bytes.extend_from_slice(&(self.certified_count as u32).to_le_bytes());
        for value in &self.block_max {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.block_argmax {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

fn zero_count(cuda: &Cuda, count: &Deviceptr) -> Result<(), String> {
    cuda.copy_htod(count, &[0u8; 4])
}

fn read_rank(cuda: &Cuda, buffers: &DeviceBuffers) -> Result<RankOutputs, String> {
    let mut logits_bytes = vec![0u8; VOCAB_PER_RANK * 4];
    let mut certified_bytes = vec![0u8; VOCAB_PER_RANK];
    let mut count_bytes = [0u8; 4];
    let mut block_max_bytes = vec![0u8; GRID * 4];
    let mut block_argmax_bytes = vec![0u8; GRID * 4];
    cuda.copy_dtoh(&mut logits_bytes, &buffers.logits)?;
    cuda.copy_dtoh(&mut certified_bytes, &buffers.certified)?;
    cuda.copy_dtoh(&mut count_bytes, &buffers.count)?;
    cuda.copy_dtoh(&mut block_max_bytes, &buffers.block_max)?;
    cuda.copy_dtoh(&mut block_argmax_bytes, &buffers.block_argmax)?;
    Ok(RankOutputs {
        logits: bytes_to_f32(&logits_bytes),
        certified: certified_bytes,
        certified_count: u32::from_le_bytes(count_bytes) as usize,
        block_max: bytes_to_f32(&block_max_bytes),
        block_argmax: block_argmax_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect(),
    })
}

/// Launches the one rank shard and returns its measured outputs.
fn launch_rank(
    cuda: &Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    buffers: &DeviceBuffers,
    rank: usize,
) -> Result<RankOutputs, String> {
    let rank_stride = (VOCAB_PER_RANK * HIDDEN * 2) as u64;
    let max_stride = (VOCAB_PER_RANK * LM_BLOCKS * 2) as u64;
    let params: [u64; 12] = [
        buffers.hidden.0,
        buffers.weight.0 + (rank as u64) * rank_stride,
        buffers.block_maxima.0 + (rank as u64) * max_stride,
        buffers.logits.0,
        buffers.certified.0,
        buffers.count.0,
        buffers.block_max.0,
        buffers.block_argmax.0,
        VOCAB_PER_RANK as u64,
        0,
        0,
        0,
    ];
    zero_count(cuda, &buffers.count)?;
    cuda.launch(kernel, (GRID as u32, 1, 1), (THREADS as u32, 1, 1), &params)?;
    read_rank(cuda, buffers)
}

fn launch_all(
    cuda: &Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    buffers: &DeviceBuffers,
) -> Result<Vec<RankOutputs>, String> {
    (0..RANKS)
        .map(|rank| launch_rank(cuda, kernel, buffers, rank))
        .collect()
}

/// The declared checks one case must satisfy. A separate value so the conjunction is testable and
/// no check can be dropped by an edit that only prints.
#[derive(Clone, Copy)]
struct CaseChecks {
    counts_ok: bool,
    blocks_ok: bool,
    exact_ok: bool,
    bounds_ok: bool,
    certificate_ok: bool,
    argmax_ok: bool,
    reference_ok: bool,
    determinism_ok: bool,
    top5_ok: bool,
}

fn case_ok(checks: CaseChecks) -> bool {
    checks.counts_ok
        && checks.blocks_ok
        && checks.exact_ok
        && checks.bounds_ok
        && checks.certificate_ok
        && checks.argmax_ok
        && checks.reference_ok
        && checks.determinism_ok
        && checks.top5_ok
}

/// The raw per-case measurements the verdict reads. A pure input, so the check assembly can be
/// tested field by field and no check can be silently dropped by an edit that only prints.
#[derive(Clone, Copy)]
struct CaseMeasurements {
    count_mismatches: usize,
    block_mismatches: usize,
    exact_mismatches: usize,
    bound_bar_violations: usize,
    certificate_violations: usize,
    replay_violations: usize,
    argmax_certified: bool,
    argmax: usize,
    twin_argmax: usize,
    argmax_value: f32,
    twin_value: f32,
    reference_argmax: Option<usize>,
    /// Whether the phase-1 reference argmax is meaningful for this case. The reference logits are
    /// the real token's, so only the real-token control has an input the reference can be compared
    /// with; the synthetic concentrated case reports it as not applicable instead of gating on it.
    reference_applicable: bool,
    deterministic: bool,
    top5_claimed: bool,
    top5_match: bool,
}

/// Turns the measurements into the declared per-case verdict. Every CaseChecks field is consumed by
/// case_ok, and a test below fails a synthetic measurement per field.
fn case_checks(measurements: &CaseMeasurements) -> CaseChecks {
    let reference_ok = if !measurements.reference_applicable {
        // The reference logits belong to the real token; the synthetic concentrated case has no
        // reference for its hidden, so a supplied reference must not gate that case.
        true
    } else {
        match measurements.reference_argmax {
            Some(at) => at == measurements.argmax,
            None => true,
        }
    };
    CaseChecks {
        counts_ok: measurements.count_mismatches == 0,
        blocks_ok: measurements.block_mismatches == 0,
        exact_ok: measurements.exact_mismatches == 0,
        bounds_ok: measurements.bound_bar_violations == 0,
        certificate_ok: measurements.certificate_violations == 0
            && measurements.replay_violations == 0
            && !measurements.argmax_certified,
        argmax_ok: measurements.argmax == measurements.twin_argmax
            && measurements.argmax_value.to_bits() == measurements.twin_value.to_bits(),
        reference_ok,
        determinism_ok: measurements.deterministic,
        top5_ok: !measurements.top5_claimed || measurements.top5_match,
    }
}

/// The top-5 set is claimed only for the all-exact control AND only when a reference exists to
/// compare it with; an absent reference is not an identical top-5 set.
fn claims_top5(is_control: bool, certified: usize, reference_supplied: bool) -> bool {
    is_control && certified == 0 && reference_supplied
}

/// One case's measured record.
struct CaseReport {
    name: &'static str,
    exact: usize,
    certified: usize,
    exact_mismatches: usize,
    count_mismatches: usize,
    /// Per rank: (certified flags counted from the published flags, certified_count the kernel
    /// published). Printed verbatim when they disagree so the record shows which side is which.
    counts: Vec<(usize, usize)>,
    block_mismatches: usize,
    bound_delta_max: f32,
    bound_bar_violations: usize,
    certificate_violations: usize,
    replay_violations: usize,
    argmax: usize,
    argmax_value: f32,
    twin_argmax: usize,
    twin_value: f32,
    /// How the reference argmax applies to this case, printed verbatim: the applicable comparison,
    /// "not supplied", or "not applicable - synthetic hidden".
    reference_note: String,
    /// Why the top-5 set is or is not claimed; printed verbatim so an absent reference cannot read
    /// as an identical top-5 set.
    top5_note: String,
    deterministic: bool,
    bound_below_max: usize,
    checks: CaseChecks,
}

/// Compares one case: two device runs for determinism, then the twin, the certificate replay, the
/// block reductions, the counts and the global argmax.
fn evaluate_case(
    cuda: &Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    buffers: &DeviceBuffers,
    hidden: &HiddenSeam,
    head: &[u8],
    reference_logits: Option<&[f32]>,
    is_control: bool,
) -> Result<CaseReport, String> {
    let first = launch_all(cuda, kernel, buffers)?;
    let second = launch_all(cuda, kernel, buffers)?;
    let deterministic = first
        .iter()
        .zip(second.iter())
        .all(|(a, b)| a.bytes() == b.bytes());

    let twins: Vec<Twin> = (0..RANKS)
        .map(|rank| {
            let base = rank * VOCAB_PER_RANK * HIDDEN * 2;
            twin_rank(&hidden.values, &head[base..])
        })
        .collect();

    let mut exact = 0usize;
    let mut certified = 0usize;
    let mut exact_mismatches = 0usize;
    let mut count_mismatches = 0usize;
    let mut counts = Vec::with_capacity(RANKS);
    let mut block_mismatches = 0usize;
    let mut bound_delta_max = 0.0f32;
    let mut bound_bar_violations = 0usize;
    let mut certificate_violations = 0usize;
    let mut replay_violations = 0usize;
    let mut bound_below_max = 0usize;
    for (rank, outputs) in first.iter().enumerate() {
        let twin = &twins[rank];
        let flagged = outputs.certified.iter().filter(|flag| **flag == 1).count();
        counts.push((flagged, outputs.certified_count));
        if flagged != outputs.certified_count {
            count_mismatches += 1;
        }
        for block in 0..GRID {
            let lo = block * CHUNK;
            let hi = (lo + CHUNK).min(VOCAB_PER_RANK);
            let (best, at) = best_entry(&outputs.logits[lo..hi]);
            let expect_at = at + lo;
            if outputs.block_max[block].to_bits() != best.to_bits()
                || outputs.block_argmax[block] as usize != expect_at
            {
                block_mismatches += 1;
            }
        }
        let (_, rank_best) = best_entry(&twin.logits);
        for entry in 0..VOCAB_PER_RANK {
            if outputs.certified[entry] == 1 {
                certified += 1;
                let delta = (outputs.logits[entry] - twin.bounds[entry]).abs();
                bound_delta_max = bound_delta_max.max(delta);
                if !bound_within_bar(delta, twin.bounds[entry]) {
                    bound_bar_violations += 1;
                }
                if !certificate_holds(outputs.logits[entry], twin.logits[entry]) {
                    certificate_violations += 1;
                }
                if outputs.logits[entry] < twin.logits[rank_best] {
                    bound_below_max += 1;
                }
            } else {
                exact += 1;
                if outputs.logits[entry].to_bits() != twin.logits[entry].to_bits() {
                    exact_mismatches += 1;
                }
            }
        }
        let (violations, seen) = replay_certificate(
            &outputs.logits,
            &outputs.certified,
            VOCAB_PER_RANK,
            CHUNK,
            THREADS,
            ENTRIES_PER_THREAD,
        );
        replay_violations += violations;
        if seen != flagged {
            replay_violations += seen.abs_diff(flagged);
        }
    }

    let device_logits: Vec<Vec<f32>> = first.iter().map(|outputs| outputs.logits.clone()).collect();
    let (argmax_value, argmax) = global_argmax(&device_logits);
    let twin_logits: Vec<Vec<f32>> = twins.iter().map(|twin| twin.logits.clone()).collect();
    let (twin_value, twin_argmax) = global_argmax(&twin_logits);
    let argmax_certified = first[argmax / VOCAB_PER_RANK].certified[argmax % VOCAB_PER_RANK] == 1;

    let (reference_argmax, reference_value) = match reference_logits {
        Some(reference) => {
            let (value, at) = best_entry(reference);
            (Some(at), Some(value))
        }
        None => (None, None),
    };
    let reference_supplied = reference_logits.is_some();
    let reference_note = if !is_control && reference_supplied {
        "not applicable - synthetic hidden".to_owned()
    } else {
        match (reference_argmax, reference_value) {
            (Some(at), Some(value)) => format!("argmax {at} value {value:.9}"),
            _ => "not supplied".to_owned(),
        }
    };
    let top5_claimed = claims_top5(is_control, certified, reference_supplied);
    let top5_match = match (top5_claimed, reference_logits) {
        (true, Some(reference)) => {
            let mut device_top = top_indices(&device_logits, 5);
            let mut reference_top = top_indices(&[reference.to_vec()], 5);
            device_top.sort_unstable();
            reference_top.sort_unstable();
            device_top == reference_top
        }
        _ => true,
    };
    let top5_note = if top5_claimed {
        format!(
            "top-5 set vs the reference {} (control, all values exact)",
            if top5_match { "identical" } else { "DIFFERS" }
        )
    } else if !reference_supplied {
        "top-5 set NOT claimed (reference not supplied)".to_owned()
    } else if !is_control {
        "top-5 set NOT claimed (not the control case)".to_owned()
    } else {
        format!(
            "top-5 set NOT claimed ({} certified entries store bounds)",
            certified
        )
    };

    let measurements = CaseMeasurements {
        count_mismatches,
        block_mismatches,
        exact_mismatches,
        bound_bar_violations,
        certificate_violations,
        replay_violations,
        argmax_certified,
        argmax,
        twin_argmax,
        argmax_value,
        twin_value,
        reference_argmax,
        reference_applicable: is_control,
        deterministic,
        top5_claimed,
        top5_match,
    };
    let checks = case_checks(&measurements);
    Ok(CaseReport {
        name: hidden.name,
        exact,
        certified,
        exact_mismatches,
        count_mismatches,
        counts,
        block_mismatches,
        bound_delta_max,
        bound_bar_violations,
        certificate_violations,
        replay_violations,
        argmax,
        argmax_value,
        twin_argmax,
        twin_value,
        reference_note,
        top5_note,
        deterministic,
        bound_below_max,
        checks,
    })
}

/// The highest indices of a rank concatenation, in descending value order, lowest index on ties.
fn top_indices(per_rank: &[Vec<f32>], count: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = Vec::new();
    for (rank, logits) in per_rank.iter().enumerate() {
        for (entry, value) in logits.iter().enumerate() {
            scored.push((*value, rank * VOCAB_PER_RANK + entry));
        }
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(count);
    scored.into_iter().map(|(_, index)| index).collect()
}

fn report_case(case: &CaseReport) -> bool {
    let ok = case_ok(case.checks);
    let status = if ok { "PASS" } else { "FAIL" };
    if ok {
        println!(
            "case {}: {status} - {} exact entries bit-identical, {} certified, counts {}, block \
             reductions {}, bound bar (max |device-host| {:.6e}) {}, certificate {} (replay \
             violations {}, validity violations {}), argmax {} value {:.9} (twin {} value {:.9}), \
             reference {}, determinism {}",
            case.name,
            case.exact,
            case.certified,
            if case.checks.counts_ok {
                "agree"
            } else {
                "DISAGREE"
            },
            if case.checks.blocks_ok {
                "agree"
            } else {
                "DISAGREE"
            },
            case.bound_delta_max,
            if case.checks.bounds_ok {
                "holds"
            } else {
                "EXCEEDED"
            },
            if case.checks.certificate_ok {
                "holds"
            } else {
                "BROKEN"
            },
            case.replay_violations,
            case.certificate_violations,
            case.argmax,
            case.argmax_value,
            case.twin_argmax,
            case.twin_value,
            case.reference_note,
            if case.deterministic {
                "byte-identical"
            } else {
                "DIFFERS"
            },
        );
        println!("case {}: {}", case.name, case.top5_note);
    } else {
        println!(
            "case {}: {status} - exact mismatches {}, count mismatches {}, block mismatches {}, \
             bound bar violations {}, certificate violations {}, replay violations {}, argmax \
             device {} vs twin {}, reference {}, determinism {}, {}",
            case.name,
            case.exact_mismatches,
            case.count_mismatches,
            case.block_mismatches,
            case.bound_bar_violations,
            case.certificate_violations,
            case.replay_violations,
            case.argmax,
            case.twin_argmax,
            case.reference_note,
            case.deterministic,
            case.top5_note
        );
        println!(
            "case {}: certified_count vs flags per rank (flagged, published) {:?}",
            case.name, case.counts
        );
    }
    ok
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

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m16_lm_head: the certified LM head (cert_lm_head_sm121, the head screen) in isolation ==="
    );
    println!(
        "release tensor: head.weight BF16 [{} x {}] ({} B) read from the snapshot; {} tensor-parallel \
         ranks of {} rows each; hidden width {}; kernel grid {} x {} threads; one launch per rank, all \
         {} ranks in this run and the 129,280-way argmax assembled from all four",
        VOCAB,
        HIDDEN,
        VOCAB * HIDDEN * 2,
        RANKS,
        VOCAB_PER_RANK,
        HIDDEN,
        GRID,
        THREADS,
        RANKS
    );
    println!(
        "comparison class (declared before any number): device vs the twin built from the SAME bf16 \
         codes the kernel consumes, per entry - BIT-IDENTICAL on every exact entry (certified == 0) \
         for the exact dot (the pinned PTX is one ascending fma chain, 17 fma.rn.f32 and no mul/add, \
         and bf16 x bf16 products are exact in f32); CERTIFIED entries (certified == 1) are bounds, \
         checked three ways: the kernel's own pruned-logit is still an upper bound on the twin's \
         exact value within f32 eps ({} + {} * |exact|), the kernel's per-thread pruning rule is \
         replayed from the measured values (chunk, lane, ascending steps; bound < running max) with \
         no violations, and the value matches the host's same bound formula within a declared bar \
         ({} + {} * |host|) because the kernel tree-reduces the block sums and the host accumulates \
         sequentially. Also checked: certified_count == flags; each of the {} blocks published max \
         and argmax equals a per-entry scan with the lowest-entry tie-break; the global argmax over \
         all 129,280 entries (lowest index on ties) equals the twin; in the real-token control and \
         only when its all-exact logits can be compared, it also equals the phase-1 reference \
         engine's argmax (the synthetic case reports the reference argmax as not applicable); two \
         device runs of every case are byte-identical. The top-5 set \
         is compared to the reference only in the real-token control and only if that case certified \
         nothing (all values exact); the kernel's contract is the argmax.",
        CERT_EPS_ABS, CERT_EPS_REL, BOUND_BAR_ABS, BOUND_BAR_REL, GRID
    );
    if !args.twin_only {
        check_floor(args.floor_gib, "m16_lm_head start")?;
    }

    // The cubin is verified by size and sha256 against the baseline pin BEFORE any device
    // allocation, exactly as m15 does.
    let cubin = if args.twin_only {
        Vec::new()
    } else {
        verify_cubin(&args.cubin_dir.join("cert_lm_head_sm121.cubin"))?
    };

    let control = real_token_hidden(&args.head_input)?;
    println!("hidden seam [{}]: {}", control.name, control.provenance);
    let concentrated = concentrated_hidden();
    println!(
        "hidden seam [{}]: {}",
        concentrated.name, concentrated.provenance
    );

    let reference = match &args.reference_logits {
        Some(path) => {
            let bytes = std::fs::read(path)
                .map_err(|e| format!("{}: {e} (the phase-1 reference logits)", path.display()))?;
            if bytes.len() != REFERENCE_LOGITS_BYTES {
                return Err(format!(
                    "{}: the reference logits are {} B, expected {REFERENCE_LOGITS_BYTES} B (129280 \
                     f32)",
                    path.display(),
                    bytes.len()
                ));
            }
            let values = bytes_to_f32(&bytes);
            let (value, at) = best_entry(&values);
            println!(
                "reference engine logits: {} f32 from {} (sha256 {}, argmax {at} value {value:.9}); \
                 declared as the phase-1 reference, not recomputed here",
                values.len(),
                path.display(),
                sha256_hex(&bytes)
            );
            Some(values)
        }
        None => {
            println!(
                "reference engine logits: not supplied (--reference-logits); the argmax is compared \
                 to the twin only and the reference comparison is not claimed"
            );
            None
        }
    };

    let mut snap = Snapshot::open(&args.checkpoint)?;
    let head = snap.read_raw("head.weight", "BF16", &[VOCAB, HIDDEN])?;
    if head.len() != VOCAB * HIDDEN * 2 {
        return Err(format!(
            "head.weight read {} B, the released head is {} B: refusing to upload a differently \
             sized tensor into the weight buffer",
            head.len(),
            VOCAB * HIDDEN * 2
        ));
    }
    println!(
        "released head.weight: BF16 [{} x {}] ({} B) read from {} ({} tensors read, {} B)",
        VOCAB,
        HIDDEN,
        head.len(),
        args.checkpoint.display(),
        snap.audit().0,
        snap.audit().1
    );

    if args.twin_only {
        println!(
            "twin-only dry run: no cubin check, no device allocation and no device claim; this              prints the host twin and a PREDICTED screen count only"
        );
        let reference_argmax = reference.as_ref().map(|values| best_entry(values).1);
        for hidden in [&control, &concentrated] {
            let start = Instant::now();
            let twins: Vec<Twin> = (0..RANKS)
                .map(|rank| {
                    let base = rank * VOCAB_PER_RANK * HIDDEN * 2;
                    twin_rank(&hidden.values, &head[base..])
                })
                .collect();
            let twin_logits: Vec<Vec<f32>> = twins.iter().map(|twin| twin.logits.clone()).collect();
            let (twin_value, twin_argmax) = global_argmax(&twin_logits);
            let mut predicted_certified = 0usize;
            let mut predicted_below_max = 0usize;
            let mut predicted_lane0 = 0usize;
            let mut predicted_lane0_per_rank = Vec::with_capacity(RANKS);
            for twin in &twins {
                let (_, rank_best) = best_entry(&twin.logits);
                for entry in 0..VOCAB_PER_RANK {
                    if twin.bounds[entry] < twin.logits[rank_best] {
                        predicted_below_max += 1;
                    }
                }
                let mut rank_lane0 = 0usize;
                for block in 0..GRID {
                    let base = block * CHUNK;
                    for lane in 0..THREADS {
                        let mut running_max = f32::NEG_INFINITY;
                        for step in 0..ENTRIES_PER_THREAD {
                            let entry = base + step * THREADS + lane;
                            if entry >= VOCAB_PER_RANK {
                                continue;
                            }
                            if twin.bounds[entry] < running_max {
                                predicted_certified += 1;
                                if lane == 0 {
                                    rank_lane0 += 1;
                                }
                            } else if twin.logits[entry] > running_max {
                                running_max = twin.logits[entry];
                            }
                        }
                    }
                }
                predicted_lane0 += rank_lane0;
                predicted_lane0_per_rank.push(rank_lane0);
            }
            println!(
                "twin-only [{}]: twin argmax {twin_argmax} value {twin_value:.9}, reference argmax                  {reference_argmax:?}, host twin {:.2} s, predicted screen {predicted_certified} of                  {VOCAB} certified, {predicted_below_max} entries with a bound below their shard max,                  predicted lane-0-only atomicAdd sum {predicted_lane0} (per rank                  {predicted_lane0_per_rank:?})",
                hidden.name,
                start.elapsed().as_secs_f64()
            );
        }
        println!("twin-only: done; this is a host dry run and no device number exists");
        return Ok(());
    }

    let cuda = Cuda::load()?;
    let (module, kernel) = cuda.load_kernel(&cubin, KernelId::CertLmHead)?;
    let (buffers, allocated) = DeviceBuffers::allocate(&cuda)?;
    println!("device buffers: {allocated} B on device 0");
    check_floor(args.floor_gib, "m16_lm_head pre-upload")?;
    cuda.copy_htod(&buffers.weight, &head)?;
    let block_maxima = lm_head_ref::block_maxima_bf16(&head);
    cuda.copy_htod(&buffers.block_maxima, &block_maxima)?;
    check_floor(args.floor_gib, "m16_lm_head post head upload")?;

    let mut failed = 0usize;
    let mut reports = Vec::new();
    for (hidden, is_control) in [(&control, true), (&concentrated, false)] {
        cuda.copy_htod(&buffers.hidden, &hidden.bytes)?;
        let report = evaluate_case(
            &cuda,
            kernel,
            &buffers,
            hidden,
            &head,
            reference.as_deref(),
            is_control,
        )?;
        if !report_case(&report) {
            failed += 1;
        }
        reports.push(report);
    }
    let control_report = &reports[0];
    let concentrated_report = &reports[1];
    for report in &reports {
        let flagged_total: usize = report.counts.iter().map(|(flagged, _)| flagged).sum();
        let published_total: usize = report.counts.iter().map(|(_, published)| published).sum();
        println!(
            "case {}: certified flags {} total, published certified_count {} total, per rank              (flagged, published) {:?}",
            report.name, flagged_total, published_total, report.counts
        );
    }
    println!(
        "control certified_count: {} of {} entries (phase 1 saw 0 on this token)",
        control_report.certified, VOCAB
    );
    if concentrated_report.certified > 0 {
        println!(
            "pruning path: FIRST device execution - the concentrated case certified {} of {} entries; \
             every certified bound passed the replay (violations {}), validity (violations {}) and the \
             host bound formula (max |device-host| {:.6e})",
            concentrated_report.certified,
            VOCAB,
            concentrated_report.replay_violations,
            concentrated_report.certificate_violations,
            concentrated_report.bound_delta_max
        );
    } else {
        println!(
            "pruning path: still certified 0 on the concentrated case - no entry's bound fell below \
             its thread's running maximum; {} entries had a bound below the shard maximum and the \
             tightest bound/weaker-is-safer gap is reported above rather than a claimed pruning",
            concentrated_report.bound_below_max
        );
    }

    // Timing, after the verdicts: repeated whole-rank-group launches on the same buffers. The
    // kernels do not write the hidden state or the weights, so the timed launches start from the
    // same bytes; their outputs are not compared and no number is claimed from a repeat.
    for hidden in [&control, &concentrated] {
        cuda.copy_htod(&buffers.hidden, &hidden.bytes)?;
        let mut samples = Vec::with_capacity(args.repeat);
        for _ in 0..args.repeat {
            let start = Instant::now();
            for rank in 0..RANKS {
                launch_rank(&cuda, kernel, &buffers, rank)?;
            }
            samples.push(start.elapsed().as_secs_f64() * 1.0e3);
        }
        let (min, p50, mean, max) = timing_stats(&samples);
        println!(
            "timing [{}, {} x 4-rank group]: per-token (group) ms min {:.4} p50 {:.4} mean {:.4} max \
             {:.4}; per-launch ms {:.4}",
            hidden.name,
            args.repeat,
            min,
            p50,
            mean,
            max,
            mean / RANKS as f64
        );
    }
    check_floor(args.floor_gib, "m16_lm_head end")?;

    cuda.free(
        module,
        &[
            buffers.hidden,
            buffers.weight,
            buffers.block_maxima,
            buffers.logits,
            buffers.certified,
            buffers.count,
            buffers.block_max,
            buffers.block_argmax,
        ],
    );

    println!("summary: 2 cases, {failed} failed");
    if failed == 0 {
        println!(
            "m16_lm_head: PASS (both hidden seams at the released head geometry: every exact entry is \
             bit-identical to the bf16-matched twin, every certified bound is a valid upper bound whose \
             per-thread pruning replay has no violations, certified counts and all {} block reductions \
             agree, the global 129,280-way argmax matches the twin{}, and two runs of every case are \
             byte-identical)",
            GRID,
            if reference.is_some() {
                " and the phase-1 reference engine"
            } else {
                ""
            }
        );
        Ok(())
    } else {
        Err(format!(
            "{failed} case(s) failed: the head's arithmetic, bound, count, block reduction, argmax or \
             determinism class did not hold; no PASS is claimed"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cubin_pin_is_the_baseline_pin() {
        let pin = baseline::pin(KERNEL)
            .unwrap()
            .expect("the head kernel is pinned");
        assert_eq!(pin.cubin_bytes, PIN_BYTES);
        assert_eq!(pin.cubin_sha256, PIN_SHA256);
        assert_eq!(pin.ptx_membar_lines, PIN_MEMBAR_LINES);
        // A cubin cannot be smaller than a few thousand bytes; the pin is not a placeholder.
        assert!(
            PIN_BYTES > 8_000,
            "the pinned cubin size is a real artifact"
        );
    }

    #[test]
    fn the_hidden_artifact_pin_is_declared_and_well_formed() {
        assert_eq!(HEAD_INPUT_BYTES, HIDDEN * 4, "the head input is 5120 f32");
        assert_eq!(HEAD_INPUT_SHA256.len(), 64, "a sha256 is 64 hex characters");
        assert!(
            HEAD_INPUT_SHA256.chars().all(|c| c.is_ascii_hexdigit()),
            "the pinned sha256 is hex"
        );
        assert!(
            SYNTHETIC_INDEX < SCREEN_TILE,
            "the single entry is in block 0"
        );
        assert_eq!(
            REFERENCE_LOGITS_BYTES,
            VOCAB * 4,
            "the reference is 129280 f32"
        );
    }

    #[test]
    fn the_concentrated_seam_is_the_declared_formula() {
        let seam = concentrated_hidden();
        assert_eq!(
            seam.bytes.len(),
            HIDDEN * 2,
            "the bf16 ABI is 2 bytes per value"
        );
        assert_eq!(seam.values.len(), HIDDEN);
        assert_eq!(seam.values[SYNTHETIC_INDEX], 1.0);
        for (index, value) in seam.values.iter().enumerate() {
            if index != SYNTHETIC_INDEX {
                assert_eq!(
                    *value, 0.0,
                    "the concentrated hidden is zero outside block 0"
                );
            }
        }
    }

    #[test]
    fn the_bf16_seam_round_trips_the_kernel_codes() {
        let values = vec![1.0f32, -0.5, 3.0, 0.0];
        let (bytes, matched) = bf16_codes(&values, 4).unwrap();
        assert_eq!(bytes.len(), 8);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                bf16_at(&bytes, index),
                *value,
                "the seam values are bf16 codes"
            );
            assert_eq!(matched[index].to_bits(), value.to_bits());
        }
        assert!(bf16_codes(&values, 5).is_err(), "a wrong width is refused");
    }

    #[test]
    fn the_exact_twin_is_the_ascending_fma_chain() {
        // bf16 x bf16 products are exact in f32, so the fused chain the PTX has is bit-identical to a
        // mul-then-add chain; this test pins that claim on a spread of bf16 codes.
        let mut codes: Vec<u16> = (0..64u16)
            .map(|index| 0x3F80u16.wrapping_add(index * 37))
            .collect();
        codes.push(0x0000);
        let codes_bits: Vec<f32> = codes.iter().map(|code| bf16_bits_to_f32(*code)).collect();
        let weights_bits: Vec<f32> = codes
            .iter()
            .rev()
            .map(|code| bf16_bits_to_f32(code.wrapping_add(11)))
            .collect();
        let mut plain = 0.0f32;
        for (value, weight) in codes_bits.iter().zip(weights_bits.iter()) {
            plain += *value * *weight;
        }
        assert_eq!(
            dot_fma(&codes_bits, &weights_bits).to_bits(),
            plain.to_bits(),
            "a bf16 fma chain and a mul-then-add chain are the same f32 program"
        );
    }

    #[test]
    fn the_block_sums_are_the_absolute_sums() {
        let mut hidden = vec![0.0f32; HIDDEN];
        hidden[0] = 2.0;
        hidden[1] = -3.0;
        hidden[SCREEN_TILE] = -1.5;
        assert_eq!(block_sum_abs(&hidden, 0), 5.0);
        assert_eq!(block_sum_abs(&hidden, 1), 1.5);
        assert_eq!(block_sum_abs(&hidden, 2), 0.0);
    }

    #[test]
    fn the_bound_is_an_upper_bound_and_the_signed_max_is_not() {
        // tile 2: weights [0.25, -2.0]; sums [1.0]; hidden has 1.0 where the weight is -2.0, so the
        // exact value is -2.0 and the abs-max bound is 2.0. A signed-max "bound" would be 0.25 and
        // would certify an entry whose exact value beats it.
        let block_sums = [1.0f32];
        let weights = [0.25f32, -2.0];
        let bound = block_sums[0] * weights[0].abs().max(weights[1].abs());
        assert!(bound >= 2.0, "the abs-max bound covers the exact value");
        let signed = block_sums[0] * weights[0].max(weights[1]);
        assert!(signed < 2.0, "the signed max is not a bound on |exact|");
    }

    #[test]
    fn the_argmax_tie_breaks_the_lowest_index() {
        let (value, at) = best_entry(&[1.5, 1.5, 1.5]);
        assert_eq!(value, 1.5);
        assert_eq!(at, 0, "a tie goes to the lowest entry index");
        let (value, at) = best_entry(&[1.0, 2.0, 2.0]);
        assert_eq!(value, 2.0);
        assert_eq!(at, 1, "a tie goes to the lowest entry index, not the last");
    }

    #[test]
    fn the_global_argmax_uses_the_rank_shards_and_global_indices() {
        // Four rank shards, because the released LM head has four: the maximum is only in rank 1, so
        // a per-rank-take(1) argmax (or a first-rank-only scan) returns the wrong entry. The second
        // entry is deliberately in rank 3, so a scan that takes only two of the four shards returns
        // rank 1's 2.0 instead of rank 3's 2.5 and fails the top-2 assertion.
        let per_rank = vec![
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            vec![0.5, 0.25],
            vec![1.5, 2.5],
        ];
        let (value, at) = global_argmax(&per_rank);
        assert_eq!(value, 3.0);
        assert_eq!(
            at,
            VOCAB_PER_RANK + 1,
            "the maximum 3.0 is only in rank 1, so the argmax must be its global entry index"
        );
        let top = top_indices(&per_rank, 2);
        assert_eq!(
            top,
            vec![VOCAB_PER_RANK + 1, 3 * VOCAB_PER_RANK + 1],
            "the top-2 is in descending value order across all four shards"
        );
        // The tie rule still goes to the lower global index across ranks.
        let tied = vec![
            vec![1.0, 3.0],
            vec![3.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let (value, at) = global_argmax(&tied);
        assert_eq!(value, 3.0);
        assert_eq!(
            at, 1,
            "the tie between rank 0 entry 1 and rank 1 entry 0 goes to the lower global index"
        );
    }

    #[test]
    fn the_certificate_replay_enforces_the_kernels_pruning_order() {
        // chunk 4, 2 lanes, 2 entries per thread: block 0 deals 0,2 to lane 0 and 1,3 to lane 1.
        let logits = [10.0f32, 1.0, 2.0, 0.5];
        let certified = [0u8, 0, 0, 1];
        let (violations, seen) = replay_certificate(&logits, &certified, 4, 4, 2, 2);
        assert_eq!(seen, 1);
        assert_eq!(violations, 0, "0.5 is below lane 1's running max of 1.0");
        // The same skip is a violation when the running maximum at that point is smaller.
        let logits = [10.0f32, 0.25, 2.0, 0.5];
        let (violations, seen) = replay_certificate(&logits, &certified, 4, 4, 2, 2);
        assert_eq!(seen, 1);
        assert_eq!(
            violations, 1,
            "0.5 is not below lane 1's running max of 0.25"
        );
        // A skip as the first entry of a lane cannot be certified: the running max starts at -inf.
        let first_skip = [0u8, 1, 0, 0];
        let logits = [10.0f32, 1.0, 2.0, 0.5];
        let (violations, _) = replay_certificate(&logits, &first_skip, 4, 4, 2, 2);
        assert_eq!(violations, 1, "the first entry of a lane cannot be pruned");
    }

    #[test]
    fn the_case_conjunction_requires_every_check() {
        let all_true = || CaseChecks {
            counts_ok: true,
            blocks_ok: true,
            exact_ok: true,
            bounds_ok: true,
            certificate_ok: true,
            argmax_ok: true,
            reference_ok: true,
            determinism_ok: true,
            top5_ok: true,
        };
        assert!(case_ok(all_true()), "all checks passing must pass");
        let flips: [fn(&mut CaseChecks); 9] = [
            |c| c.counts_ok = false,
            |c| c.blocks_ok = false,
            |c| c.exact_ok = false,
            |c| c.bounds_ok = false,
            |c| c.certificate_ok = false,
            |c| c.argmax_ok = false,
            |c| c.reference_ok = false,
            |c| c.determinism_ok = false,
            |c| c.top5_ok = false,
        ];
        for (index, flip) in flips.iter().enumerate() {
            let mut checks = all_true();
            flip(&mut checks);
            assert!(
                !case_ok(checks),
                "check {index} is not required by the case verdict"
            );
        }
    }

    #[test]
    fn the_case_checks_consume_every_measurement() {
        let passing = || CaseMeasurements {
            count_mismatches: 0,
            block_mismatches: 0,
            exact_mismatches: 0,
            bound_bar_violations: 0,
            certificate_violations: 0,
            replay_violations: 0,
            argmax_certified: false,
            argmax: 5,
            twin_argmax: 5,
            argmax_value: 13.0,
            twin_value: 13.0,
            reference_argmax: Some(5),
            reference_applicable: true,
            deterministic: true,
            top5_claimed: true,
            top5_match: true,
        };
        assert!(
            case_ok(case_checks(&passing())),
            "every passing measurement must pass the case"
        );
        // Each mutation must knock the case down, and the named check must be the one that fired.
        let cases: [(&str, fn(&mut CaseMeasurements), fn(&CaseChecks) -> bool); 12] = [
            ("counts", |m| m.count_mismatches = 1, |c| !c.counts_ok),
            ("blocks", |m| m.block_mismatches = 1, |c| !c.blocks_ok),
            ("exact", |m| m.exact_mismatches = 1, |c| !c.exact_ok),
            ("bounds", |m| m.bound_bar_violations = 1, |c| !c.bounds_ok),
            (
                "certificate",
                |m| m.certificate_violations = 1,
                |c| !c.certificate_ok,
            ),
            ("replay", |m| m.replay_violations = 1, |c| !c.certificate_ok),
            (
                "argmax_certified",
                |m| m.argmax_certified = true,
                |c| !c.certificate_ok,
            ),
            ("argmax index", |m| m.argmax = 6, |c| !c.argmax_ok),
            ("argmax value", |m| m.argmax_value = 14.0, |c| !c.argmax_ok),
            (
                "reference",
                |m| m.reference_argmax = Some(6),
                |c| !c.reference_ok,
            ),
            (
                "determinism",
                |m| m.deterministic = false,
                |c| !c.determinism_ok,
            ),
            ("top-5", |m| m.top5_match = false, |c| !c.top5_ok),
        ];
        for (name, mutate, assert_check) in cases {
            let mut measurements = passing();
            mutate(&mut measurements);
            let checks = case_checks(&measurements);
            assert!(
                assert_check(&checks),
                "the {name} measurement does not reach its check"
            );
            assert!(
                !case_ok(checks),
                "the {name} measurement does not fail the case"
            );
        }
        // A top-5 set that is not claimed cannot fail on its (meaningless) match flag.
        let mut unclaimed = passing();
        unclaimed.top5_claimed = false;
        unclaimed.top5_match = false;
        assert!(case_ok(case_checks(&unclaimed)));
    }

    #[test]
    fn the_synthetic_case_is_not_gated_on_the_reference_argmax() {
        let passing = || CaseMeasurements {
            count_mismatches: 0,
            block_mismatches: 0,
            exact_mismatches: 0,
            bound_bar_violations: 0,
            certificate_violations: 0,
            replay_violations: 0,
            argmax_certified: false,
            argmax: 33865,
            twin_argmax: 33865,
            argmax_value: 13.0,
            twin_value: 13.0,
            reference_argmax: Some(5),
            reference_applicable: false,
            deterministic: true,
            top5_claimed: false,
            top5_match: true,
        };
        // Synthetic hidden: the phase-1 reference argmax (5) is for the real token, so it must not
        // gate the concentrated case.
        let checks = case_checks(&passing());
        assert!(
            checks.reference_ok,
            "a synthetic case must not be gated on the reference argmax"
        );
        assert!(
            case_ok(checks),
            "a synthetic case with a supplied reference must not fail on it"
        );
        // Mutation proof A: gating the synthetic case (dropping the applicability condition) fails
        // it, so the test above cannot be satisfied by a gate that ignores applicability.
        let mut gated = passing();
        gated.reference_applicable = true;
        assert!(
            !case_ok(case_checks(&gated)),
            "the synthetic case must fail if the reference argmax gates it"
        );
        // Mutation proof B: the control must still fail when its reference argmax differs, so the
        // gate cannot be weakened to always-true.
        let mut control = passing();
        control.argmax = 5;
        control.twin_argmax = 5;
        control.reference_argmax = Some(6);
        control.reference_applicable = true;
        control.top5_claimed = true;
        assert!(
            !case_ok(case_checks(&control)),
            "the control must fail when the reference argmax differs"
        );
    }
    #[test]
    fn the_top5_set_is_not_claimed_without_a_reference() {
        assert!(
            !claims_top5(true, 0, false),
            "an absent reference must not read as an identical top-5 set"
        );
        assert!(claims_top5(true, 0, true));
        assert!(
            !claims_top5(true, 1, true),
            "a certifying control is not all-exact"
        );
        assert!(
            !claims_top5(false, 0, true),
            "the pruning case is not the control"
        );
    }

    #[test]
    fn the_certificate_eps_and_bound_bar_are_declared() {
        assert!(certificate_holds(1.0, 1.0));
        assert!(certificate_holds(1.0 - CERT_EPS_ABS / 2.0, 1.0));
        assert!(!certificate_holds(1.0 - 10.0 * CERT_EPS_ABS, 1.0));
        assert!(bound_within_bar(BOUND_BAR_ABS / 2.0, 10.0));
        assert!(!bound_within_bar(10.0 * BOUND_BAR_ABS, 10.0));
    }
}
