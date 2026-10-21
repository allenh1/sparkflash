//! m27_head_tail: the single-stream certified head tail - final norm, four rank head shards, the
//! exact f32 logit all-gather and the argmax - plus the batched-reduce drop for single-stream decode.
//!
//! What this harness pins (offline: the twin runs anywhere the released snapshot is readable; the
//! device run needs a window with a live GPU, as every kernel harness here does):
//!
//! 1. Host twin (--twin-only): the engine's final norm (rmsnorm_ref::final_norm_bf16), then for each
//!    rank shard the exact f32 dot (lm_head_ref::exact_logit) and the certificate bound
//!    (lm_head_ref::hidden_block_sums + entry_bound) for every entry, the exact f32 all-gather and
//!    the lowest-index argmax. This is the expected record a later window's device run is diffed
//!    against.
//! 2. Device run: rmsnorm_sm121 on the final hidden row, then cert_lm_head_sm121 per rank shard;
//!    every class is compared to the twin (normed BF16 bytes bit-exact, exact entries bit-identical,
//!    certified bounds within the m16 bar, the per-thread pruning replay, block reductions, counts,
//!    determinism), then the stored f32 logits are gathered word for word and the token is argued
//!    from the gathered vector with the lowest-index tie break.
//! 3. The batched-reduce drop. The certified single-node chain spends 5 launches per position on the
//!    head: 4 x cert_lm_head_sm121 rank shards plus 1 x lm_head_reduce_sm121. The six-row batched
//!    variants (cert_lm_head_batched_sm121, lm_head_reduce_batched_sm121, fixed BATCH_ROWS 6) are
//!    the verify-path kernels; for single-stream decode only the final position's logits matter, so
//!    the batched reduce is pure overhead and cannot even be called with one row. This harness
//!    records the inventory (5 launches -> 4, reduce dropped) and the expected per-position saving
//!    from the RECORDED m24 p50s; it makes no new device measurement and claims no device speedup.
//!
//! The exactness table is the certified head's own: exact entries are bit-identical to the twin,
//! certified entries store a bound that is within the m16 bound bar of the host formula and is a
//! valid upper bound on the twin's exact value, the gather is word movement (identical bytes, zero
//! ULP), and the argmax is exact with the lowest-index tie break and must win on an exact entry
//! (a certified winner is a refusal, not a token). No approximate row exists here.
//!
//! Usage:
//!   m27_head_tail --checkpoint <snapshot dir> --input <bundle dir or f32 file> [--cubin-dir out]
//!                 [--reference-logits <f32 129280>] [--repeat 20] [--floor-gib 32]
//!                 [--dump <dir>] [--evidence <file>] [--twin-only]
//!
//! The input file is the final hidden seam: 5120 little-endian f32 values (20,480 B). A bundle
//! directory must contain pre_head_hidden.f32 (the same bytes); the loop writes that file into its
//! evidence directory so the head stage is adjudicable from the record alone.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::baseline;
use sparkflash_kernels::cert_lm_head_launch;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::{bf16_bits, bf16_bits_to_f32};
use sparkflash_kernels::rmsnorm_launch::{self, RmsNormLaunchParams};
use sparkflash_kernels::rmsnorm_ref;

/// The released hidden width.
const HIDDEN: usize = 5_120;
/// The released vocabulary.
const VOCAB: usize = 129_280;
/// The per-rank head shard width (four tensor-parallel ranks).
const VOCAB_PER_RANK: usize = 32_320;
/// The tensor-parallel degree.
const RANKS: usize = 4;
/// The certified head's screen tile.
const SCREEN_TILE: usize = 128;
/// The head's screen blocks per entry.
const LM_BLOCKS: usize = HIDDEN / SCREEN_TILE;
/// The vocabulary entries one cert_lm_head_sm121 block screens.
const CHUNK: usize = 1_024;
/// The vocabulary chunks per rank shard (also the cert head grid x).
const GRID: usize = VOCAB_PER_RANK.div_ceil(CHUNK);
/// cert_lm_head_sm121's block threads.
const HEAD_THREADS: usize = 128;
/// The entries one head thread visits in a chunk.
const ENTRIES_PER_THREAD: usize = CHUNK / HEAD_THREADS;
/// The input seam is one hidden row of f32 values.
const INPUT_BYTES: usize = HIDDEN * 4;

const HEAD_KERNEL: &str = "cert_lm_head_sm121";
const HEAD_PIN_BYTES: usize = 36_800;
const HEAD_PIN_SHA256: &str = "bbd30c8d6d358bd200b07cffa54d662c9144501b69310864933b68a0e53b6798";
const HEAD_PIN_MEMBAR_LINES: u32 = 0;

const NORM_KERNEL: &str = "rmsnorm_sm121";
const NORM_PIN_BYTES: usize = 34_512;
const NORM_PIN_SHA256: &str = "fb559d1bd3666fc249c4f5b5bf46fa236004b3068355963876db43245764e678";
const NORM_PIN_MEMBAR_LINES: u32 = 0;

/// The m16 bound bar: the device tree-reduces the block sums, the host accumulates sequentially, so
/// the two bounds differ by f32 accumulation, not by the formula.
const BOUND_BAR_ABS: f32 = 1.0e-3;
const BOUND_BAR_REL: f32 = 1.0e-5;
/// The certificate eps: a bound is a proof on exact reals; f32 accumulation may put it a few ulps
/// below the measured exact value.
const CERT_EPS_ABS: f32 = 1.0e-4;
const CERT_EPS_REL: f32 = 1.0e-5;

/// The certified single-node chain's head launches per decode position: four cert_lm_head_sm121
/// rank shards plus the lm_head_reduce_sm121 cut-over. The 1,168-launch ledger calls the reduce the
/// 1,168th launch (kernel-occupancy-inventory.md sections 1, rows 15 and 16; the code is
/// crates/sparkflash-cli/src/device_chain.rs).
const CHAIN_HEAD_LAUNCHES: usize = 5;
/// Single-stream decode: the four rank shards, no reduce. The exact f32 gather is host word
/// movement, not a launch.
const SINGLE_STREAM_HEAD_LAUNCHES: usize = 4;
/// The batched reduce's fixed row count (lm_head_reduce_launch::BATCH_ROWS).
const BATCHED_REDUCE_ROWS: usize = 6;
/// Recorded m24 p50s, quoted not remeasured: one batched six-row reduce 2.4535 ms, six scalar
/// reduces 2.4709 ms total. These are the only timing numbers this page uses, and they are
/// host-side arithmetic from the b-window record, never a new device claim.
const RECORDED_BATCHED_SIX_ROW_REDUCE_P50_MS: f64 = 2.4535;
const RECORDED_SIX_SCALAR_REDUCE_P50_MS: f64 = 2.4709;
/// The recorded per-row share of six scalar launches: 2.4709 / 6 = 0.4118166666... ms.
const RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS: f64 = RECORDED_SIX_SCALAR_REDUCE_P50_MS / 6.0;

const SCHEMA: &str = "sparkflash-head-tail-evidence/1";

const _: () = assert!(VOCAB == VOCAB_PER_RANK * RANKS);
const _: () = assert!(HIDDEN.is_multiple_of(SCREEN_TILE));
const _: () = assert!(GRID * CHUNK >= VOCAB_PER_RANK && (GRID - 1) * CHUNK < VOCAB_PER_RANK);
const _: () = assert!(CHUNK == HEAD_THREADS * ENTRIES_PER_THREAD);
const _: () = assert!(SINGLE_STREAM_HEAD_LAUNCHES + 1 == CHAIN_HEAD_LAUNCHES);
const _: () = assert!(BATCHED_REDUCE_ROWS == 6);

fn main() {
    if let Err(error) = run() {
        eprintln!("m27_head_tail: {error}");
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

fn bytes_to_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The kernel's BF16 ABI for one f32 vector, plus the BF16-matched f32 values it means.
fn bf16_codes(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    let mut widened = Vec::with_capacity(values.len());
    for value in values {
        let bits = bf16_bits(*value);
        bytes.extend_from_slice(&bits.to_le_bytes());
        widened.push(bf16_bits_to_f32(bits));
    }
    (bytes, widened)
}

/// The declared hidden seam: the final hidden row the head stage consumes, as f32 bytes plus the
/// kernel's BF16 form.
struct HeadInput {
    path: PathBuf,
    bytes: Vec<u8>,
    sha256: String,
    bf16_codes: Vec<u8>,
}

fn load_head_input(path: &Path) -> Result<HeadInput, String> {
    let file = if path.is_dir() {
        path.join("pre_head_hidden.f32")
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(&file).map_err(|error| {
        format!(
            "{}: {error} (the declared final hidden seam: 5120 little-endian f32 values)",
            file.display()
        )
    })?;
    if bytes.len() != INPUT_BYTES {
        return Err(format!(
            "{}: the head input is {} B, the declared seam is {INPUT_BYTES} B (5120 f32)",
            file.display(),
            bytes.len()
        ));
    }
    let values = bytes_to_f32_le(&bytes);
    let (bf16_codes, _) = bf16_codes(&values);
    let sha256 = sha256_hex(&bytes);
    Ok(HeadInput {
        path: file,
        bytes,
        sha256,
        bf16_codes,
    })
}

/// The configured rms_norm_eps from the released config.json, cast to f32 exactly as the engine
/// does. The released DeepSeek-V4.1 snapshot nests the text-model fields under "text_config"
/// (`text_config.rms_norm_eps`); older and synthetic snapshots put it at the top level, so both
/// shapes are accepted. The value is always the released one, never a substituted default: missing
/// config, or a missing eps in either shape, is an error.
fn read_eps(checkpoint: &Path) -> Result<f32, String> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("{}: {error} (the released config.json)", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let eps = value
        .get("rms_norm_eps")
        .or_else(|| {
            value
                .get("text_config")
                .and_then(|text| text.get("rms_norm_eps"))
        })
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            format!(
                "{}: no numeric rms_norm_eps (top-level or text_config.rms_norm_eps)",
                path.display()
            )
        })?;
    println!(
        "config: rms_norm_eps {} (f32 {}) from {}",
        eps,
        eps as f32,
        path.display()
    );
    Ok(eps as f32)
}

struct Args {
    checkpoint: PathBuf,
    input: PathBuf,
    cubin_dir: PathBuf,
    reference_logits: Option<PathBuf>,
    repeat: usize,
    floor_gib: u64,
    dump: Option<PathBuf>,
    evidence: Option<PathBuf>,
    twin_only: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1))
    }

    fn parse_from(values: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            input: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            reference_logits: None,
            repeat: 20,
            floor_gib: 32,
            dump: None,
            evidence: None,
            twin_only: false,
        };
        let mut values = values.collect::<Vec<_>>();
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
                "--input" => args.input = PathBuf::from(next()?),
                "--cubin-dir" => args.cubin_dir = PathBuf::from(next()?),
                "--reference-logits" => args.reference_logits = Some(PathBuf::from(next()?)),
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
                "--dump" => args.dump = Some(PathBuf::from(next()?)),
                "--evidence" => args.evidence = Some(PathBuf::from(next()?)),
                "--twin-only" => args.twin_only = true,
                "--help" | "-h" => {
                    println!(
                        "usage: m27_head_tail --checkpoint <snapshot dir> --input <bundle dir or f32 file>                          [--reference-logits <f32 129280>] [--cubin-dir out] [--repeat 20]                          [--floor-gib 32] [--dump <dir>] [--evidence <file>] [--twin-only]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}; try --help")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required (the released snapshot directory)".to_owned());
        }
        if args.input.as_os_str().is_empty() {
            return Err(
                "--input is required: a bundle directory with pre_head_hidden.f32 or the f32 file                  itself (the declared final hidden seam, 5120 f32)"
                    .to_owned(),
            );
        }
        if args.repeat == 0 {
            return Err("--repeat must be at least 1".to_owned());
        }
        Ok(args)
    }
}

/// One rank shard's host twin: the exact f32 dot and the certificate bound for every entry.
struct TwinRank {
    logits: Vec<f32>,
    bounds: Vec<f32>,
}

/// The host twin of the shard's screening arithmetic. weight_rows is this shard's BF16 row image;
/// entries lets the tests use a small shard while production passes VOCAB_PER_RANK.
fn twin_rank(hidden: &[f32], weight_rows: &[u8], entries: usize) -> TwinRank {
    assert_eq!(weight_rows.len(), entries * HIDDEN * 2, "whole BF16 rows");
    let sums = lm_head_ref::hidden_block_sums(hidden);
    let mut logits = Vec::with_capacity(entries);
    let mut bounds = Vec::with_capacity(entries);
    let mut row = vec![0.0f32; HIDDEN];
    for entry in 0..entries {
        let base = entry * HIDDEN * 2;
        for (index, slot) in row.iter_mut().enumerate() {
            let at = base + index * 2;
            *slot = bf16_bits_to_f32(u16::from_le_bytes([weight_rows[at], weight_rows[at + 1]]));
        }
        logits.push(lm_head_ref::exact_logit(hidden, &row));
        bounds.push(lm_head_ref::entry_bound(&sums, &row));
    }
    TwinRank { logits, bounds }
}

/// The whole host twin: the normed BF16 row, the four rank shards and the lowest-index argmax over
/// the exact logits.
struct Twin {
    normed_bf16: Vec<u8>,
    ranks: Vec<TwinRank>,
    argmax_value: f32,
    argmax: usize,
}

fn twin_of(
    head_bytes: &[u8],
    norm_weight: &[u8],
    input_bf16: &[u8],
    eps: f32,
) -> Result<Twin, String> {
    if head_bytes.len() != VOCAB * HIDDEN * 2 {
        return Err(format!(
            "the head image is {} B, the released head is {} B",
            head_bytes.len(),
            VOCAB * HIDDEN * 2
        ));
    }
    let normed_bf16 = rmsnorm_ref::final_norm_bf16(input_bf16, norm_weight, 1, HIDDEN, eps)?;
    let normed_f32 = rmsnorm_ref::decode_bf16(&normed_bf16);
    let shard_bytes = VOCAB_PER_RANK * HIDDEN * 2;
    let mut ranks = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        let start = rank * shard_bytes;
        ranks.push(twin_rank(
            &normed_f32,
            &head_bytes[start..start + shard_bytes],
            VOCAB_PER_RANK,
        ));
    }
    let mut argmax_value = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (rank, shard) in ranks.iter().enumerate() {
        for (entry, value) in shard.logits.iter().enumerate() {
            if *value > argmax_value {
                argmax_value = *value;
                argmax = rank * VOCAB_PER_RANK + entry;
            }
        }
    }
    Ok(Twin {
        normed_bf16,
        ranks,
        argmax_value,
        argmax,
    })
}

/// The score of a slice with the lowest-index tie break: strict greater-than only, which is the
/// certified head's own best_entry rule.
fn best_entry(values: &[f32]) -> (f32, usize) {
    assert!(
        !values.is_empty(),
        "there is no best entry of an empty vector"
    );
    let mut best = f32::NEG_INFINITY;
    let mut at = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > best {
            best = *value;
            at = index;
        }
    }
    (best, at)
}

/// The number of bytes that differ between two same-length byte slices, or a length mismatch as a
/// large count (the caller reports it as a mismatched byte vector).
fn byte_mismatches(left: &[u8], right: &[u8]) -> usize {
    if left.len() != right.len() {
        return left.len().max(right.len());
    }
    left.iter().zip(right).filter(|(a, b)| a != b).count()
}

/// The pinned cubin reader: the baseline manifest pin is checked before the read bytes, and the read
/// bytes by size and sha256, before any device allocation.
fn verify_cubin(
    path: &Path,
    kernel: &str,
    pin_bytes: usize,
    pin_sha256: &str,
    pin_membar_lines: u32,
) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first; the module docs give the nvcc line)",
            path.display()
        )
    })?;
    let pin = baseline::pin(kernel)?
        .ok_or_else(|| format!("{kernel} is not in the baseline manifest"))?;
    if pin.cubin_bytes != pin_bytes || pin.cubin_sha256 != pin_sha256 {
        return Err(format!(
            "the baseline manifest's {kernel} pin changed: {} B, sha256 {}",
            pin.cubin_bytes, pin.cubin_sha256
        ));
    }
    if pin.ptx_membar_lines != pin_membar_lines {
        return Err(format!(
            "the baseline manifest's {kernel} membar count changed: {}",
            pin.ptx_membar_lines
        ));
    }
    if bytes.len() != pin_bytes {
        return Err(format!(
            "{}: cubin is {} B but the pin is {pin_bytes} B",
            path.display(),
            bytes.len()
        ));
    }
    let sha = sha256_hex(&bytes);
    if sha != pin_sha256 {
        return Err(format!(
            "{}: cubin sha256 {sha} is not the pinned {pin_sha256}",
            path.display()
        ));
    }
    println!(
        "cubin: {kernel} {} B, sha256 {sha} (pinned: {pin_bytes} B, {pin_membar_lines} PTX membar lines) - verified before any device allocation",
        bytes.len()
    );
    Ok(bytes)
}

/// One rank's measured device outputs.
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
        let mut bytes = f32_le_bytes(&self.logits);
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

    fn certified_flags(&self) -> usize {
        self.certified.iter().filter(|flag| **flag == 1).count()
    }
}

/// Replays the kernel's per-thread pruning rule over the measured values (the m16 rule, unchanged):
/// order is (chunk, lane, ascending step); the running maximum starts at -FLT_MAX, is not updated on
/// a skip, and a certified entry must satisfy bound < running max exactly as the kernel's strict
/// test does.
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

/// One rank's comparison against the twin: every class the certified head claims, measured.
struct RankComparison {
    exact_entries: usize,
    certified_entries: usize,
    exact_mismatches: usize,
    count_mismatches: usize,
    block_mismatches: usize,
    bound_delta_max: f32,
    bound_bar_violations: usize,
    certificate_violations: usize,
    replay_violations: usize,
}

fn compare_rank(outputs: &RankOutputs, twin: &TwinRank) -> RankComparison {
    let mut exact_entries = 0usize;
    let mut certified_entries = 0usize;
    let mut exact_mismatches = 0usize;
    let mut bound_delta_max = 0.0f32;
    let mut bound_bar_violations = 0usize;
    let mut certificate_violations = 0usize;
    for entry in 0..VOCAB_PER_RANK {
        if outputs.certified[entry] == 1 {
            certified_entries += 1;
            let delta = (outputs.logits[entry] - twin.bounds[entry]).abs();
            bound_delta_max = bound_delta_max.max(delta);
            if !bound_within_bar(delta, twin.bounds[entry]) {
                bound_bar_violations += 1;
            }
            if !certificate_holds(outputs.logits[entry], twin.logits[entry]) {
                certificate_violations += 1;
            }
        } else {
            exact_entries += 1;
            if outputs.logits[entry].to_bits() != twin.logits[entry].to_bits() {
                exact_mismatches += 1;
            }
        }
    }
    let flagged = outputs.certified_flags();
    let count_mismatches = (flagged != outputs.certified_count) as usize;
    let mut block_mismatches = 0usize;
    for block in 0..GRID {
        let base = block * CHUNK;
        let end = (base + CHUNK).min(VOCAB_PER_RANK);
        let (value, at) = best_entry(&outputs.logits[base..end]);
        let expected_argmax = (base + at) as u32;
        if outputs.block_max[block].to_bits() != value.to_bits()
            || outputs.block_argmax[block] != expected_argmax
        {
            block_mismatches += 1;
        }
    }
    let (replay_violations, _) = replay_certificate(
        &outputs.logits,
        &outputs.certified,
        VOCAB_PER_RANK,
        CHUNK,
        HEAD_THREADS,
        ENTRIES_PER_THREAD,
    );
    RankComparison {
        exact_entries,
        certified_entries,
        exact_mismatches,
        count_mismatches,
        block_mismatches,
        bound_delta_max,
        bound_bar_violations,
        certificate_violations,
        replay_violations,
    }
}

/// The exact f32 logit all-gather: the four shard vectors concatenated word for word, and the
/// lowest-index argmax over the gathered vector. The gather does no arithmetic, so its exactness is
/// byte identity, not a ULP bound.
struct Gathered {
    logits: Vec<f32>,
    bytes: Vec<u8>,
    sha256: String,
    argmax_value: f32,
    argmax: usize,
    argmax_certified: bool,
}

fn gather(outputs: &[RankOutputs]) -> Result<Gathered, String> {
    if outputs.len() != RANKS {
        return Err(format!(
            "the gather concatenates {RANKS} rank shards, got {}",
            outputs.len()
        ));
    }
    let mut logits = Vec::with_capacity(VOCAB);
    for (rank, shard) in outputs.iter().enumerate() {
        if shard.logits.len() != VOCAB_PER_RANK {
            return Err(format!(
                "rank {rank} published {} logits, the shard is {VOCAB_PER_RANK}",
                shard.logits.len()
            ));
        }
        logits.extend_from_slice(&shard.logits);
    }
    let bytes = f32_le_bytes(&logits);
    let sha256 = sha256_hex(&bytes);
    let (argmax_value, argmax) = best_entry(&logits);
    let argmax_certified = outputs[argmax / VOCAB_PER_RANK].certified[argmax % VOCAB_PER_RANK] == 1;
    Ok(Gathered {
        logits,
        bytes,
        sha256,
        argmax_value,
        argmax,
        argmax_certified,
    })
}

/// The single-stream head inventory and the batched-reduce drop arithmetic. Every number here is
/// launch-count arithmetic or a recorded m24 p50 from the b-window table; no new device measurement
/// is made and no device speedup is claimed.
fn inventory() -> Value {
    let saving_vs_batched =
        RECORDED_BATCHED_SIX_ROW_REDUCE_P50_MS - RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS;
    let saving_vs_scalar = RECORDED_SIX_SCALAR_REDUCE_P50_MS - RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS;
    json!({
        "chain_head_launches_per_position": CHAIN_HEAD_LAUNCHES,
        "chain_composition": "4 x cert_lm_head_sm121 rank shards + 1 x lm_head_reduce_sm121",
        "single_stream_head_launches_per_position": SINGLE_STREAM_HEAD_LAUNCHES,
        "single_stream_composition": "4 x cert_lm_head_sm121 rank shards + 0 reduce; the exact f32 all-gather is host word movement",
        "reduce_launches_dropped": CHAIN_HEAD_LAUNCHES - SINGLE_STREAM_HEAD_LAUNCHES,
        "reduce_mode": "dropped-single-stream",
        "batched_reduce": {
            "kernel": "lm_head_reduce_batched_sm121",
            "fixed_rows": BATCHED_REDUCE_ROWS,
            "callable_with_one_row": false,
            "role": "the DSpark verify path only; for one position it does one row of work at the six-row floor",
        },
        "recorded": {
            "batched_six_row_reduce_p50_ms": RECORDED_BATCHED_SIX_ROW_REDUCE_P50_MS,
            "six_scalar_reduce_p50_ms": RECORDED_SIX_SCALAR_REDUCE_P50_MS,
            "single_row_scalar_reduce_ms": RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS,
            "source": "m24 recorded p50s (batch19-plan table); host-side arithmetic from that record, not remeasured here",
        },
        "expected_per_position_saving": {
            "vs_batched_six_row_for_one_row_ms": saving_vs_batched,
            "vs_scalar_reduce_for_one_row_ms": saving_vs_scalar,
            "launches": CHAIN_HEAD_LAUNCHES - SINGLE_STREAM_HEAD_LAUNCHES,
            "note": "no new device measurement; this harness makes no device speedup claim",
        },
        "exact_gather": "host word movement, 0 device launches, identical bytes (0 ULP)",
    })
}

fn print_inventory() {
    println!(
        "head inventory: the certified single-node chain spends {CHAIN_HEAD_LAUNCHES} launches per position on the head (4 x cert_lm_head_sm121 rank shards + 1 x lm_head_reduce_sm121);"
    );
    println!(
        "single-stream decode needs only the final position: {SINGLE_STREAM_HEAD_LAUNCHES} head launches (4 rank shards, no reduce) plus the exact f32 all-gather, which is host word movement (0 device launches)."
    );
    println!(
        "batched-reduce drop: lm_head_reduce_batched_sm121 is a fixed {BATCHED_REDUCE_ROWS}-row kernel and cannot be called with one row; the verify path may keep it, single-stream decode must not."
    );
    println!(
        "expected per-position saving from the recorded m24 p50s: 1 launch; {RECORDED_BATCHED_SIX_ROW_REDUCE_P50_MS:.4} ms if the six-row batched reduce was used for one position, {:.4} ms against the scalar reduce (six scalar p50 {RECORDED_SIX_SCALAR_REDUCE_P50_MS:.4} ms / 6). Host-side arithmetic only: no new device measurement, no device speedup claimed.",
        RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS
    );
}

/// One check, printed as a stage line and exported in the evidence record.
struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

fn push(verdicts: &mut Vec<Verdict>, name: &str, ok: bool, detail: String) {
    verdicts.push(Verdict {
        name: name.to_owned(),
        ok,
        detail,
    });
}

fn write_evidence(path: &Path, evidence: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(evidence)
        .map_err(|error| format!("evidence serialisation: {error}"))?;
    std::fs::write(path, &text).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("evidence: wrote {} ({} B)", path.display(), text.len());
    Ok(())
}

/// Prints the verdicts, the declared NOT-claimed block and the final PASS/FAIL. Returns whether every
/// check passed; the caller decides the exit status after the evidence has been written.
fn finish(verdicts: &[Verdict], mode: &str) -> bool {
    if verdicts.is_empty() {
        println!("{mode}: no checks were run - refusing to call that a PASS");
        return false;
    }
    for verdict in verdicts {
        println!(
            "{} {} - {}",
            if verdict.ok { "PASS" } else { "FAIL" },
            verdict.name,
            verdict.detail
        );
    }
    let failed = verdicts.iter().filter(|verdict| !verdict.ok).count();
    println!("NOT claimed:");
    println!(
        "  - no device speedup: the batched-reduce saving above is host-side arithmetic from the recorded m24 p50s, not a new measurement;"
    );
    println!(
        "  - no token is claimed from the head: the argmax is a measured equality with the twin (and the optional reference), not a serving claim, and a certified winner is a refusal;"
    );
    println!(
        "  - no kernel or pin change is made: the harness uses the pinned rmsnorm_sm121 and cert_lm_head_sm121 from the twenty-two-entry baseline."
    );
    if failed == 0 {
        println!(
            "{mode}: PASS ({} checks; the head stage is pinned for the single-stream path)",
            verdicts.len()
        );
        true
    } else {
        println!(
            "{mode}: FAIL ({failed} of {} checks failed)",
            verdicts.len()
        );
        false
    }
}

/// The device buffers the head path needs. The whole head image and its block maxima are uploaded
/// once; each rank launch addresses its shard by pointer offset (the m16 layout).
struct DeviceBuffers {
    hidden: Deviceptr,
    norm_weight: Deviceptr,
    normed: Deviceptr,
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
        let norm_weight = cuda.alloc(HIDDEN * 2, &mut allocated)?;
        let normed = cuda.alloc(HIDDEN * 2, &mut allocated)?;
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
                norm_weight,
                normed,
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

    fn pointers(&self) -> [Deviceptr; 10] {
        [
            self.hidden,
            self.norm_weight,
            self.normed,
            self.weight,
            self.block_maxima,
            self.logits,
            self.certified,
            self.count,
            self.block_max,
            self.block_argmax,
        ]
    }
}

fn launch_norm(
    cuda: &Cuda,
    kernel: Kernel,
    buffers: &DeviceBuffers,
    eps: f32,
) -> Result<(), String> {
    let params = RmsNormLaunchParams {
        x: buffers.hidden,
        weight: buffers.norm_weight,
        out: buffers.normed,
        rows: 1,
        cols: HIDDEN as u32,
        eps,
        trace_ring: 0,
        layer_id: 0,
        step_seq: 0,
    };
    rmsnorm_launch::launch(cuda, kernel, &params)
}

fn read_normed(cuda: &Cuda, buffers: &DeviceBuffers) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; HIDDEN * 2];
    cuda.copy_dtoh(&mut bytes, &buffers.normed)?;
    Ok(bytes)
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
        logits: bytes_to_f32_le(&logits_bytes),
        certified: certified_bytes,
        certified_count: u32::from_le_bytes(count_bytes) as usize,
        block_max: bytes_to_f32_le(&block_max_bytes),
        block_argmax: block_argmax_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect(),
    })
}

/// Launches the certified head for one rank shard with the pinned 12-word ABI and returns the
/// measured outputs. The whole head image and block maxima are resident; the rank is a pointer
/// offset, exactly as the m16 launch does.
fn launch_rank(
    cuda: &Cuda,
    kernel: Kernel,
    buffers: &DeviceBuffers,
    rank: usize,
) -> Result<RankOutputs, String> {
    let rank_stride = (VOCAB_PER_RANK * HIDDEN * 2) as u64;
    let max_stride = (VOCAB_PER_RANK * LM_BLOCKS * 2) as u64;
    let params: [u64; cert_lm_head_launch::ARG_WORDS] = [
        buffers.normed.0,
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
    cuda.launch(
        kernel,
        (GRID as u32, 1, 1),
        (HEAD_THREADS as u32, 1, 1),
        &params,
    )?;
    read_rank(cuda, buffers)
}

/// Writes the expanded seam record a later window can inspect without re-running the head.
fn dump_seam(
    dir: &Path,
    input: &HeadInput,
    twin: &Twin,
    gathered: Option<&Gathered>,
    ranks: Option<&[RankOutputs]>,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    std::fs::write(dir.join("pre_head_hidden.f32"), &input.bytes)
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    std::fs::write(dir.join("normed.bf16"), &twin.normed_bf16)
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    for (rank, shard) in twin.ranks.iter().enumerate() {
        std::fs::write(
            dir.join(format!("twin-logits-r{rank}.f32")),
            f32_le_bytes(&shard.logits),
        )
        .map_err(|error| format!("{}: {error}", dir.display()))?;
        std::fs::write(
            dir.join(format!("twin-bounds-r{rank}.f32")),
            f32_le_bytes(&shard.bounds),
        )
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    }
    if let Some(gathered) = gathered {
        std::fs::write(dir.join("gathered-device-logits.f32"), &gathered.bytes)
            .map_err(|error| format!("{}: {error}", dir.display()))?;
    }
    if let Some(ranks) = ranks {
        for (rank, shard) in ranks.iter().enumerate() {
            std::fs::write(
                dir.join(format!("device-logits-r{rank}.f32")),
                f32_le_bytes(&shard.logits),
            )
            .map_err(|error| format!("{}: {error}", dir.display()))?;
            std::fs::write(
                dir.join(format!("device-certified-r{rank}.bin")),
                &shard.certified,
            )
            .map_err(|error| format!("{}: {error}", dir.display()))?;
        }
    }
    println!("dump: wrote seam files under {}", dir.display());
    Ok(())
}

/// The twin-side evidence block, shared by the twin-only and device records so a diff is mechanical.
fn twin_evidence(input: &HeadInput, twin: &Twin) -> Value {
    let ranks: Vec<Value> = twin
        .ranks
        .iter()
        .enumerate()
        .map(|(rank, shard)| {
            json!({
                "rank": rank,
                "logits_sha256": sha256_hex(&f32_le_bytes(&shard.logits)),
                "bounds_sha256": sha256_hex(&f32_le_bytes(&shard.bounds)),
            })
        })
        .collect();
    json!({
        "normed_bf16_sha256": sha256_hex(&twin.normed_bf16),
        "normed_bytes": twin.normed_bf16.len(),
        "argmax": twin.argmax,
        "argmax_value": twin.argmax_value,
        "ranks": ranks,
        "input_pre_head_sha256": input.sha256,
    })
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let start = Instant::now();
    println!(
        "m27_head_tail: schema {SCHEMA}; single-stream certified head tail (norm -> 4 rank head shards -> exact f32 all-gather -> argmax)"
    );
    println!(
        "geometry: hidden {HIDDEN}, vocab {VOCAB} over {RANKS} ranks of {VOCAB_PER_RANK}, screen tile {SCREEN_TILE}, {LM_BLOCKS} blocks, {GRID} chunks per rank, {HEAD_THREADS} threads x {ENTRIES_PER_THREAD} entries"
    );

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    let norm_weight = snapshot.read_raw("norm.weight", "BF16", &[HIDDEN])?;
    let head_bytes = snapshot.read_raw("head.weight", "BF16", &[VOCAB, HIDDEN])?;
    println!(
        "checkpoint: norm.weight {HIDDEN} BF16 values ({} B), head.weight [{VOCAB} x {HIDDEN}] BF16 ({} B) read from {}",
        norm_weight.len(),
        head_bytes.len(),
        args.checkpoint.display()
    );
    let eps = read_eps(&args.checkpoint)?;
    let input = load_head_input(&args.input)?;
    println!(
        "head input: {} B from {}; f32 sha256 {}, BF16-codes sha256 {}",
        input.bytes.len(),
        input.path.display(),
        input.sha256,
        sha256_hex(&input.bf16_codes)
    );

    let twin_start = Instant::now();
    let twin = twin_of(&head_bytes, &norm_weight, &input.bf16_codes, eps)?;
    let twin_ms = twin_start.elapsed().as_secs_f64() * 1.0e3;
    let normed_sha = sha256_hex(&twin.normed_bf16);
    println!(
        "twin: final normed BF16 sha256 {normed_sha}; argmax {} value {:.9}; twin time {twin_ms:.1} ms",
        twin.argmax, twin.argmax_value
    );
    for (rank, shard) in twin.ranks.iter().enumerate() {
        println!(
            "twin rank {rank}: logits sha256 {}, bounds sha256 {}",
            sha256_hex(&f32_le_bytes(&shard.logits)),
            sha256_hex(&f32_le_bytes(&shard.bounds))
        );
    }
    print_inventory();

    let mut verdicts: Vec<Verdict> = Vec::new();

    if args.twin_only {
        push(
            &mut verdicts,
            "twin: final norm produced the declared BF16 row",
            twin.normed_bf16.len() == HIDDEN * 2,
            format!(
                "{HIDDEN} BF16 bytes, sha256 {normed_sha}, engine arithmetic via rmsnorm_ref::final_norm_bf16"
            ),
        );
        for (rank, shard) in twin.ranks.iter().enumerate() {
            push(
                &mut verdicts,
                &format!("twin rank {rank}: shard screened"),
                shard.logits.len() == VOCAB_PER_RANK && shard.bounds.len() == VOCAB_PER_RANK,
                format!(
                    "{VOCAB_PER_RANK} exact logits and {VOCAB_PER_RANK} certificate bounds, logits sha256 {}",
                    sha256_hex(&f32_le_bytes(&shard.logits))
                ),
            );
        }
        push(
            &mut verdicts,
            "twin: argmax is a real best entry",
            twin.argmax < VOCAB && twin.argmax_value > f32::NEG_INFINITY,
            format!(
                "argmax {} value {:.9}, lowest-index tie break (strict greater-than), no token claimed",
                twin.argmax, twin.argmax_value
            ),
        );
        let evidence = json!({
            "schema": SCHEMA,
            "mode": "twin",
            "checkpoint": args.checkpoint.display().to_string(),
            "input": {"path": input.path.display().to_string(), "sha256": input.sha256, "bytes": input.bytes.len(), "bf16_codes_sha256": sha256_hex(&input.bf16_codes), "bf16_codes_bytes": input.bf16_codes.len()},
            "tokens": 1,
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "ranks": RANKS,
            "vocab_per_rank": VOCAB_PER_RANK,
            "norm": {"kernel": NORM_KERNEL, "eps": eps, "normed_bf16_sha256": normed_sha, "normed_bytes": twin.normed_bf16.len()},
            "head": {"kernel": HEAD_KERNEL},
            "twin": twin_evidence(&input, &twin),
            "reduce": inventory(),
            "checks": verdicts.iter().map(|verdict| json!({"name": verdict.name, "ok": verdict.ok, "detail": verdict.detail})).collect::<Vec<Value>>(),
        });
        if let Some(path) = &args.evidence {
            write_evidence(path, &evidence)?;
        }
        if let Some(dir) = &args.dump {
            dump_seam(dir, &input, &twin, None, None)?;
        }
        println!(
            "twin-only: expected head record written; the device run compares to these digests and bars"
        );
        let ok = finish(&verdicts, "m27_head_tail twin-only");
        return if ok {
            Ok(())
        } else {
            Err("twin-only checks failed".to_owned())
        };
    }

    check_floor(args.floor_gib, "m27 before device allocation")?;
    let head_cubin = verify_cubin(
        &args.cubin_dir.join(format!("{HEAD_KERNEL}.cubin")),
        HEAD_KERNEL,
        HEAD_PIN_BYTES,
        HEAD_PIN_SHA256,
        HEAD_PIN_MEMBAR_LINES,
    )?;
    let norm_cubin = verify_cubin(
        &args.cubin_dir.join(format!("{NORM_KERNEL}.cubin")),
        NORM_KERNEL,
        NORM_PIN_BYTES,
        NORM_PIN_SHA256,
        NORM_PIN_MEMBAR_LINES,
    )?;
    let cuda = Cuda::load()?;
    let (buffers, allocated) = DeviceBuffers::allocate(&cuda)?;
    println!("device bytes admitted: {allocated}");
    let (norm_module, norm_kernel) = cuda.load_kernel(&norm_cubin, KernelId::RmsNorm)?;
    let (head_module, head_kernel) = cuda.load_kernel(&head_cubin, KernelId::CertLmHead)?;
    check_floor(args.floor_gib, "m27 after device allocation")?;

    cuda.copy_htod(&buffers.hidden, &input.bf16_codes)?;
    cuda.copy_htod(&buffers.norm_weight, &norm_weight)?;
    let norm_start = Instant::now();
    launch_norm(&cuda, norm_kernel, &buffers, eps)?;
    let norm_ms = norm_start.elapsed().as_secs_f64() * 1.0e3;
    let normed_first = read_normed(&cuda, &buffers)?;
    launch_norm(&cuda, norm_kernel, &buffers, eps)?;
    let normed_second = read_normed(&cuda, &buffers)?;
    let norm_mismatches = byte_mismatches(&normed_first, &twin.normed_bf16);
    push(
        &mut verdicts,
        "stage 1: final norm BF16 bit-exact vs twin",
        norm_mismatches == 0,
        format!(
            "{norm_mismatches} mismatched bytes of {}",
            twin.normed_bf16.len()
        ),
    );
    push(
        &mut verdicts,
        "stage 1: two norm launches byte-identical",
        normed_first == normed_second,
        format!("{} bytes compared", normed_first.len()),
    );
    println!(
        "stage 1 device: normed sha256 {}, device launch {norm_ms:.4} ms",
        sha256_hex(&normed_first)
    );

    let maxima_start = Instant::now();
    let block_maxima = lm_head_ref::block_maxima_bf16(&head_bytes);
    let maxima_ms = maxima_start.elapsed().as_secs_f64() * 1.0e3;
    cuda.copy_htod(&buffers.weight, &head_bytes)?;
    cuda.copy_htod(&buffers.block_maxima, &block_maxima)?;
    println!(
        "device head staging: weight {} B and block maxima {} B (derived from the released rows in {maxima_ms:.1} ms), one H2D per tensor",
        head_bytes.len(),
        block_maxima.len()
    );

    let mut device_ranks: Vec<RankOutputs> = Vec::with_capacity(RANKS);
    let mut deterministic_flags: Vec<bool> = Vec::with_capacity(RANKS);
    let mut head_samples: Vec<f64> = Vec::with_capacity(RANKS);
    for rank in 0..RANKS {
        check_floor(args.floor_gib, "m27 rank launch")?;
        let launch_start = Instant::now();
        let first = launch_rank(&cuda, head_kernel, &buffers, rank)?;
        head_samples.push(launch_start.elapsed().as_secs_f64() * 1.0e3);
        let second = launch_rank(&cuda, head_kernel, &buffers, rank)?;
        let deterministic = first.bytes() == second.bytes();
        deterministic_flags.push(deterministic);
        let comparison = compare_rank(&first, &twin.ranks[rank]);
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: exact entries bit-identical"),
            comparison.exact_mismatches == 0,
            format!(
                "{} mismatched of {} exact entries ({} certified were bounds, not logits)",
                comparison.exact_mismatches, comparison.exact_entries, comparison.certified_entries
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: certified bounds within 1e-3 + 1e-5*|host|"),
            comparison.bound_bar_violations == 0,
            format!(
                "{} violations, max |device-host| {:.6e}",
                comparison.bound_bar_violations, comparison.bound_delta_max
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: certificate holds"),
            comparison.certificate_violations == 0,
            format!(
                "{} certified bounds below the twin exact value beyond the certificate eps",
                comparison.certificate_violations
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: pruning replay has no violations"),
            comparison.replay_violations == 0,
            format!(
                "{} per-thread bound/running-max violations",
                comparison.replay_violations
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: block reductions agree"),
            comparison.block_mismatches == 0,
            format!(
                "{} mismatched blocks of {GRID}",
                comparison.block_mismatches
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: certified_count agrees with the flags"),
            comparison.count_mismatches == 0,
            format!(
                "{} flagged, count {}",
                first.certified_flags(),
                first.certified_count
            ),
        );
        push(
            &mut verdicts,
            &format!("stage 2 rank {rank}: two head launches byte-identical"),
            deterministic,
            format!(
                "{} raw output bytes compared (logits, certified, count, block max/argmax)",
                first.bytes().len()
            ),
        );
        println!(
            "stage 2 rank {rank}: {} exact entries, {} certified bounds, count {}, logits sha256 {}, launch {:.4} ms",
            comparison.exact_entries,
            comparison.certified_entries,
            first.certified_count,
            sha256_hex(&f32_le_bytes(&first.logits)),
            head_samples.last().copied().unwrap_or_default()
        );
        device_ranks.push(first);
    }

    let gathered = gather(&device_ranks)?;
    let concatenated = {
        let mut bytes = Vec::with_capacity(VOCAB * 4);
        for shard in &device_ranks {
            bytes.extend_from_slice(&f32_le_bytes(&shard.logits));
        }
        bytes
    };
    push(
        &mut verdicts,
        "stage 3: exact f32 gather is the per-rank concatenation",
        gathered.bytes == concatenated && gathered.logits.len() == VOCAB,
        format!(
            "{} f32 values, {} B, sha256 {}; word movement, 0 ULP",
            gathered.logits.len(),
            gathered.bytes.len(),
            gathered.sha256
        ),
    );
    push(
        &mut verdicts,
        "stage 4: argmax is on an exact entry (not a certified bound)",
        !gathered.argmax_certified,
        format!(
            "argmax {} (rank {}, entry {}), certified in that rank: {}",
            gathered.argmax,
            gathered.argmax / VOCAB_PER_RANK,
            gathered.argmax % VOCAB_PER_RANK,
            gathered.argmax_certified
        ),
    );
    push(
        &mut verdicts,
        "stage 4: device argmax equals the twin argmax",
        gathered.argmax == twin.argmax
            && gathered.argmax_value.to_bits() == twin.argmax_value.to_bits(),
        format!(
            "device {} value {:.9}, twin {} value {:.9}",
            gathered.argmax, gathered.argmax_value, twin.argmax, twin.argmax_value
        ),
    );

    let reference = match &args.reference_logits {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|error| {
                format!("{}: {error} (the declared engine logits)", path.display())
            })?;
            if bytes.len() != VOCAB * 4 {
                return Err(format!(
                    "{}: the reference logits are {} B, the declared artifact is {} B (129280 f32)",
                    path.display(),
                    bytes.len(),
                    VOCAB * 4
                ));
            }
            let values = bytes_to_f32_le(&bytes);
            let (value, at) = best_entry(&values);
            println!(
                "reference: engine logits from {}, sha256 {}, argmax {} value {:.9}",
                path.display(),
                sha256_hex(&bytes),
                at,
                value
            );
            Some((at, value))
        }
        None => None,
    };
    if let Some((reference_argmax, reference_value)) = reference {
        push(
            &mut verdicts,
            "stage 4: reference engine argmax agrees",
            reference_argmax == gathered.argmax,
            format!(
                "reference {reference_argmax} value {reference_value:.9}, device {} value {:.9}",
                gathered.argmax, gathered.argmax_value
            ),
        );
    }

    push(
        &mut verdicts,
        "stage 5: single-stream head drops the batched reduce",
        CHAIN_HEAD_LAUNCHES - SINGLE_STREAM_HEAD_LAUNCHES == 1 && BATCHED_REDUCE_ROWS == 6,
        format!(
            "{} launches -> {} (1 reduce dropped); the six-row batched reduce cannot be called with one row; saving arithmetic recorded, no device claim",
            CHAIN_HEAD_LAUNCHES, SINGLE_STREAM_HEAD_LAUNCHES
        ),
    );

    let evidence = json!({
        "schema": SCHEMA,
        "mode": "device",
        "checkpoint": args.checkpoint.display().to_string(),
        "input": {"path": input.path.display().to_string(), "sha256": input.sha256, "bytes": input.bytes.len(), "bf16_codes_sha256": sha256_hex(&input.bf16_codes), "bf16_codes_bytes": input.bf16_codes.len()},
        "tokens": 1,
        "hidden": HIDDEN,
        "vocab": VOCAB,
        "ranks": RANKS,
        "vocab_per_rank": VOCAB_PER_RANK,
        "norm": {"kernel": NORM_KERNEL, "pin": {"cubin_bytes": NORM_PIN_BYTES, "cubin_sha256": NORM_PIN_SHA256}, "eps": eps, "normed_bf16_sha256": sha256_hex(&normed_first), "normed_bytes": normed_first.len()},
        "head": {"kernel": HEAD_KERNEL, "pin": {"cubin_bytes": HEAD_PIN_BYTES, "cubin_sha256": HEAD_PIN_SHA256}, "grid": [GRID, 1, 1], "block": [HEAD_THREADS, 1, 1], "chunks_per_rank": GRID, "deterministic": deterministic_flags, "ranks": device_ranks.iter().enumerate().map(|(rank, shard)| json!({"rank": rank, "logits_sha256": sha256_hex(&f32_le_bytes(&shard.logits)), "logits_f32_values": shard.logits.len(), "certified_count": shard.certified_count, "certified_sha256": sha256_hex(&shard.certified), "block_max_sha256": sha256_hex(&f32_le_bytes(&shard.block_max)), "block_argmax_sha256": sha256_hex(&shard.block_argmax.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>())})).collect::<Vec<Value>>()},
        "gather": {"route": "exact-f32-word-gather", "words_per_rank": VOCAB_PER_RANK * 2, "gathered_logits_f32_values": gathered.logits.len(), "gathered_logits_sha256": gathered.sha256, "exact": "per-rank f32 logits concatenated word for word; identical bytes, 0 ULP"},
        "argmax": {"token": gathered.argmax, "value": gathered.argmax_value, "certified": gathered.argmax_certified, "claimable": !gathered.argmax_certified, "twin_token": twin.argmax, "twin_value": twin.argmax_value},
        "reference": {"supplied": args.reference_logits.is_some()},
        "reduce": inventory(),
        "launches": {"loop_norm": RANKS, "loop_head": RANKS, "loop_reduce": 0, "harness_norm": 1, "harness_head": RANKS, "harness_reduce": 0, "gather": 1},
        "twin": twin_evidence(&input, &twin),
        "timing": {"norm_ms": norm_ms, "head_ms_per_rank": head_samples},
        "checks": verdicts.iter().map(|verdict| json!({"name": verdict.name, "ok": verdict.ok, "detail": verdict.detail})).collect::<Vec<Value>>(),
    });
    if let Some(path) = &args.evidence {
        write_evidence(path, &evidence)?;
    }
    if let Some(dir) = &args.dump {
        dump_seam(dir, &input, &twin, Some(&gathered), Some(&device_ranks))?;
    }
    check_floor(args.floor_gib, "m27 end")?;
    cuda.free(head_module, &buffers.pointers());
    cuda.free(norm_module, &[]);
    println!(
        "total harness time {:.1} ms",
        start.elapsed().as_secs_f64() * 1.0e3
    );
    let ok = finish(&verdicts, "m27_head_tail");
    if ok {
        Ok(())
    } else {
        Err("head tail device checks failed".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_rows(rows: &[&[f32]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            assert_eq!(row.len(), HIDDEN);
            for value in *row {
                bytes.extend_from_slice(&bf16_bits(*value).to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn the_constants_are_the_released_geometry() {
        assert_eq!(VOCAB, 129_280);
        assert_eq!(VOCAB_PER_RANK * RANKS, VOCAB);
        assert!(GRID * CHUNK >= VOCAB_PER_RANK && (GRID - 1) * CHUNK < VOCAB_PER_RANK);
        assert_eq!(CHUNK, HEAD_THREADS * ENTRIES_PER_THREAD);
        assert_eq!(LM_BLOCKS, 40);
        assert_eq!(CHAIN_HEAD_LAUNCHES, 5);
        assert_eq!(SINGLE_STREAM_HEAD_LAUNCHES, 4);
        assert_eq!(BATCHED_REDUCE_ROWS, 6);
    }

    #[test]
    fn bf16_codes_are_the_kernel_abi_and_bf16_values_are_exact() {
        let values = [0.0f32, 1.0, -2.5, 3.25];
        let (bytes, widened) = bf16_codes(&values);
        assert_eq!(bytes.len(), values.len() * 2);
        for (index, value) in widened.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                bf16_bits_to_f32(bf16_bits(values[index])).to_bits()
            );
        }
    }

    #[test]
    fn tiny_shard_twin_matches_the_exact_dot_and_the_bound() {
        let mut hidden = vec![0.0f32; HIDDEN];
        hidden[0] = 1.0;
        let mut hot = vec![0.0f32; HIDDEN];
        hot[0] = 1.0;
        let cold = vec![0.0f32; HIDDEN];
        let rows = tiny_rows(&[&hot, &cold]);
        let twin = twin_rank(&hidden, &rows, 2);
        assert_eq!(twin.logits.len(), 2);
        assert_eq!(twin.logits[0], 1.0);
        assert_eq!(twin.logits[1], 0.0);
        assert_eq!(twin.bounds[0], 1.0);
        assert_eq!(twin.bounds[1], 0.0);
    }

    #[test]
    fn best_entry_keeps_the_lowest_index_on_ties() {
        assert_eq!(best_entry(&[1.0, 1.0, 0.5]), (1.0, 0));
        assert_eq!(best_entry(&[0.5, 1.0, 1.0]), (1.0, 1));
        assert_eq!(best_entry(&[-3.0, -1.0]), (-1.0, 1));
    }

    #[test]
    fn the_gather_is_the_rank_concatenation_and_the_certified_winner_is_flagged() {
        let mut shards: Vec<RankOutputs> = (0..RANKS)
            .map(|rank| RankOutputs {
                logits: vec![rank as f32; VOCAB_PER_RANK],
                certified: vec![0u8; VOCAB_PER_RANK],
                certified_count: 0,
                block_max: vec![0.0f32; GRID],
                block_argmax: vec![0u32; GRID],
            })
            .collect();
        shards[2].logits[5] = 5.0;
        shards[2].certified[5] = 1;
        let gathered = gather(&shards).expect("the gather takes four shards");
        assert_eq!(gathered.logits.len(), VOCAB);
        assert_eq!(gathered.bytes.len(), VOCAB * 4);
        assert_eq!(gathered.argmax, 2 * VOCAB_PER_RANK + 5);
        assert_eq!(gathered.argmax_value, 5.0);
        assert!(gathered.argmax_certified);
        assert_eq!(gathered.sha256, sha256_hex(&gathered.bytes));
    }

    #[test]
    fn the_pruning_replay_flags_a_certified_bound_at_or_above_the_running_max() {
        let logits = [2.0f32, 1.0, 3.0, 0.5];
        let certified = [0u8, 0, 1, 0];
        let (violations, seen) = replay_certificate(&logits, &certified, 4, 4, 2, 2);
        assert_eq!(seen, 1);
        assert!(
            violations > 0,
            "a certified value at 3.0 above the 2.0 running max must be flagged"
        );
        let clean = [2.0f32, 1.0, 0.5, 0.25];
        let (violations, seen) = replay_certificate(&clean, &[0u8, 0, 1, 1], 4, 4, 2, 2);
        assert_eq!(seen, 2);
        assert_eq!(violations, 0);
    }

    #[test]
    fn the_inventory_arithmetic_is_the_recorded_one() {
        assert_eq!(CHAIN_HEAD_LAUNCHES - SINGLE_STREAM_HEAD_LAUNCHES, 1);
        let single = RECORDED_SINGLE_ROW_SCALAR_REDUCE_MS;
        assert!((single - 0.41181666666666666).abs() < 1.0e-12);
        assert!(RECORDED_BATCHED_SIX_ROW_REDUCE_P50_MS > single);
        assert!(RECORDED_SIX_SCALAR_REDUCE_P50_MS > single);
        let value = inventory();
        assert_eq!(value["reduce_launches_dropped"], 1);
        assert_eq!(value["batched_reduce"]["fixed_rows"], 6);
        assert_eq!(value["batched_reduce"]["callable_with_one_row"], false);
    }

    #[test]
    fn args_require_the_checkpoint_and_the_input() {
        let only = ["--checkpoint".to_owned(), "/tmp/snap".to_owned()];
        assert!(Args::parse_from(only.into_iter()).is_err());
        let both = [
            "--checkpoint".to_owned(),
            "/tmp/snap".to_owned(),
            "--input".to_owned(),
            "/tmp/in.f32".to_owned(),
        ];
        let args = Args::parse_from(both.into_iter()).expect("the declared pair parses");
        assert_eq!(args.cubin_dir, PathBuf::from("out"));
        assert_eq!(args.repeat, 20);
        assert_eq!(args.floor_gib, 32);
        assert!(!args.twin_only);
        let bad = [
            "--checkpoint".to_owned(),
            "/tmp/snap".to_owned(),
            "--input".to_owned(),
            "/tmp/in.f32".to_owned(),
            "--repeat".to_owned(),
            "0".to_owned(),
        ];
        assert!(Args::parse_from(bad.into_iter()).is_err());
    }

    #[test]
    fn the_evidence_schema_is_versioned() {
        assert!(SCHEMA.starts_with("sparkflash-head-tail-evidence/"));
        assert!(SCHEMA.ends_with("/1"));
    }

    #[test]
    fn read_eps_accepts_the_released_nested_text_config() {
        let dir = std::env::temp_dir().join(format!("m27-eps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp checkpoint dir");
        let config = dir.join("config.json");
        std::fs::write(&config, r#"{"text_config":{"rms_norm_eps":1e-20}}"#)
            .expect("write nested config");
        assert_eq!(
            read_eps(&dir)
                .expect("the released nesting parses")
                .to_bits(),
            1.0e-20f32.to_bits()
        );
        std::fs::write(&config, r#"{"rms_norm_eps":1.0e-20}"#).expect("write top-level config");
        assert_eq!(
            read_eps(&dir)
                .expect("the top level still parses")
                .to_bits(),
            1.0e-20f32.to_bits()
        );
        std::fs::write(&config, r#"{"text_config":{"hidden_size":5120}}"#)
            .expect("write eps-less config");
        assert!(read_eps(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
