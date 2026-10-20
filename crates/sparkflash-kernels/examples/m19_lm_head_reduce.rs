//! m19: the certified head's device reduction (lm_head_reduce_sm121) against the host assembly it
//! replaces, at the released 129,280-entry geometry.
//!
//! THIS EXAMPLE IS THE EVIDENCE THAT THE PINNED lm_head_reduce_sm121 REPRODUCES THE CHAIN'S HOST
//! BLOCK ASSEMBLY BIT FOR BIT, on the same four device shard buffers and with the same tie rules.
//! It is not the served forward pass, not the c4c wiring (engine-dev's cut-over), not the head
//! screen itself, not a token/chain/per-layer claim and not a timing claim.
//!
//! TODAY the c4c chain launches cert_lm_head_sm121 four times, copies each rank's 32,320 stored
//! logits, 32 block maxima and 32 block argmax entries to the host, folds the 128 block answers with
//! consider_head_block and takes the resulting token. m19 runs that same sequence and then the
//! reduce kernel on the same device buffers, and compares the two answers:
//!
//!   * the token and the stored argmax value, bit for bit;
//!   * all 128 block maxima and block argmax entries, bit for bit, and against the screen's own
//!     published 32-per-rank arrays remapped to global indices (so the reduce's block layout is the
//!     m16 one, not a redefinition);
//!   * the status word, the certified count, and the top-5 indices and values;
//!   * two reduce launches, byte for byte.
//!
//! CASES, the same two declared hidden seams m16 uses and no fabricated fallback:
//!
//!   1. real-token control: the phase-1 chain's 5,120-f32 head input, size- and sha256-verified;
//!      the published screen certifies nothing, so the reduce's top-5 is CLAIMED and compared with
//!      the phase-1 reference engine's top-5 set (when --reference-logits is supplied).
//!   2. concentrated certificate: the declared synthetic bf16(1.0)-at-index-7 hidden. The screen
//!      certifies 4,899 of 129,280 entries (per rank 1,062 / 1,205 / 1,215 / 1,417); the reduce must
//!      still find the exact winner (33,865), must NOT claim the top-5 (certified entries store
//!      bounds), and must report the count.
//!   3. synthetic full-width cases (declared synthetic operands, device buffers filled by the host):
//!      a tie at the maximum must resolve to the lowest global index, and a certified winner must
//!      set WINNER_CERTIFIED and not claim the token or the top-5 - refuse rather than assert.
//!
//! STATUS CONTRACT (pinned in the kernel, the launch module and the reference):
//!   WINNER_CERTIFIED 0x1: the argmax entry is certified, so its stored value is a bound; the token
//!                        is diagnostic and must not be claimed. A sound certificate makes this
//!                        impossible (bound < running maximum), so seeing it is a finding.
//!   TOP5_CLAIMED     0x2: set only when no entry was certified. The host still needs a reference
//!                        before it may claim the set (m16's claims_top5 rule).
//!   ANY_CERTIFIED    0x4: at least one entry is certified; informational.
//!
//! b42 (the first build, pin 37,152 B / 6cff3661...) FAILED at 7 of 14 checks: the kernel scanned
//! each block's local prefix instead of its base+local range (so every block of a rank re-read the
//! rank's first 1,024 entries), and the top-5 insertion could store the same entry twice. Both are
//! fixed and re-pinned; b43 then FAILED at 5 of 14 checks on the global-index write (the block
//! argmax and top-5 stored `local` instead of `base + local`, so block 1 reported block 0's entry
//! and a certified winner's flag was read at the wrong index). That is fixed too, the kernel is
//! re-pinned again, and the b42/b43 failure records stay in
//! docs/build/phase3-lm-head-reduce.md. Do not cite b42 or b43 as anything but FAILURES.
//!
//! PINS. Both cubins are refused by size and sha256 against the embedded baseline manifest before
//! any device allocation: cert_lm_head_sm121 38,088 B / 3b71c992... / membar 0, lm_head_reduce_sm121
//! 73,704 B / 2d18f68d... / membar 0. The m16 block layout is 128 blocks = four ranks x 32 chunks of 1,024
//! entries (the last chunk of each rank has 576).
//!
//! Usage (aarch64 node, checkpoint present):
//!
//!     m19_lm_head_reduce --checkpoint <snapshot dir> --cubin-dir <dir> \
//!         --head-input ~/hostrun/phase1-host-token0.head.f32 \
//!         [--reference-logits ~/hostrun/phase1-host-token0.logits.f32] [--floor-gib 32]
//!
//! Build lines (nvcc 13.0.88 / CUDA 13.0.3):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/cert_lm_head_sm121.cu -o <dir>/cert_lm_head_sm121.cubin
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/lm_head_reduce_sm121.cu -o <dir>/lm_head_reduce_sm121.cubin

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{Snapshot, bytes_to_f32};
use sparkflash_kernels::baseline;
use sparkflash_kernels::cert_lm_head_launch::{self, CertLmHeadParams};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
use sparkflash_kernels::lm_head_reduce_launch::{
    self as launch, BLOCK, BLOCKS, STATUS_ANY_CERTIFIED, STATUS_TOP5_CLAIMED,
    STATUS_WINNER_CERTIFIED, TOP, VOCAB, VOCAB_PER_RANK,
};
use sparkflash_kernels::lm_head_reduce_ref::{self, Reduction};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::bf16_bits;

const HIDDEN: usize = 5_120;
const RANKS: usize = 4;

/// The certified screen's pin (the c4c chain launches this artifact).
const SCREEN_KERNEL: &str = "cert_lm_head_sm121";
const SCREEN_BYTES: usize = 38_088;
const SCREEN_SHA256: &str = "3b71c99290ee91ca2431c0e232c644559f1c6e2bf340182fc103486789dad8f8";

/// This kernel's pin, from baseline/kernel_baseline.txt through baseline::pin.
const REDUCE_KERNEL: &str = "lm_head_reduce_sm121";
const REDUCE_BYTES: usize = 73_704;
const REDUCE_SHA256: &str = "2d18f68d71fbe657ed8e675f3b74c386678d51ff350a82368fff0df50c0032d3";

/// The declared hidden seam pins (the same artifacts m16 declares).
const HEAD_INPUT_BYTES: usize = HIDDEN * 4;
const HEAD_INPUT_SHA256: &str = "e6819e83a32c13fa6843ec576ba3959bbb9cb645047d59e445eaa235a7846c04";
const REFERENCE_LOGITS_BYTES: usize = VOCAB * 4;
/// The concentrated case: a single bf16(1.0) at this index and zeros elsewhere.
const SYNTHETIC_INDEX: usize = 7;
/// The concentrated case's certified flags measured by m16 in windows b21/b23/b25.
const CONCENTRATED_CERTIFIED: usize = 4_899;
const CONCENTRATED_PER_RANK: [usize; RANKS] = [1_062, 1_205, 1_215, 1_417];
/// The concentrated case's declared argmax, preserved by the certificate in m16's windows.
const CONCENTRATED_TOKEN: usize = 33_865;

fn main() {
    if let Err(error) = run() {
        eprintln!("m19_lm_head_reduce: {error}");
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
    floor_gib: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            head_input: PathBuf::new(),
            reference_logits: None,
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
                "--head-input" => args.head_input = PathBuf::from(next()?),
                "--reference-logits" => args.reference_logits = Some(PathBuf::from(next()?)),
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|_| "--floor-gib needs a number".to_owned())?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: m19_lm_head_reduce --checkpoint <snapshot dir> --head-input <f32 5120> \
                         [--reference-logits <f32 129280>] [--cubin-dir out] [--floor-gib 32]"
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
                "--head-input is required: the declared real-token hidden seam; this harness \
                 refuses to fabricate a hidden state"
                    .to_owned(),
            );
        }
        Ok(args)
    }
}

/// One verdict in the pass/fail list, printed on failure only (m18's shape).
struct Verdict {
    name: &'static str,
    ok: bool,
    detail: String,
}

impl Verdict {
    fn push(verdicts: &mut Vec<Self>, name: &'static str, ok: bool, detail: String) {
        println!(
            "  check {name}: {} ({detail})",
            if ok { "ok" } else { "FAILED" }
        );
        verdicts.push(Self { name, ok, detail });
    }
}

/// One launch's copied-back report, plus the raw bytes for the determinism comparison.
#[derive(Debug, Clone, PartialEq)]
struct ReduceReport {
    token: u32,
    value: f32,
    block_max: Vec<f32>,
    block_argmax: Vec<u32>,
    top5_value: [f32; TOP],
    top5_index: [u32; TOP],
    status: u32,
    certified_count: u32,
}

impl ReduceReport {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.token.to_le_bytes());
        bytes.extend_from_slice(&self.value.to_le_bytes());
        for value in &self.block_max {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for index in &self.block_argmax {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        for value in &self.top5_value {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for index in &self.top5_index {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        bytes.extend_from_slice(&self.status.to_le_bytes());
        bytes.extend_from_slice(&self.certified_count.to_le_bytes());
        bytes
    }

    fn bits_equal(&self, other: &Self) -> bool {
        self.value.to_bits() == other.value.to_bits()
            && self
                .block_max
                .iter()
                .zip(&other.block_max)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self
                .top5_value
                .iter()
                .zip(&other.top5_value)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

/// The certified screen's per-rank output, one entry per rank.
struct ScreenOutputs {
    logits: Vec<Vec<f32>>,
    certified: Vec<Vec<u8>>,
    block_max: Vec<Vec<f32>>,
    block_argmax: Vec<Vec<u32>>,
    certified_count: Vec<u32>,
}

struct ReduceBuffers {
    token: Deviceptr,
    value: Deviceptr,
    block_max: Deviceptr,
    block_argmax: Deviceptr,
    top5_value: Deviceptr,
    top5_index: Deviceptr,
    status: Deviceptr,
    certified_count: Deviceptr,
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m19: the certified head's device reduction (lm_head_reduce_sm121) against the host assembly ==="
    );
    println!(
        "geometry: vocabulary {VOCAB} over {RANKS} ranks of {VOCAB_PER_RANK}; {BLOCKS} m16 blocks = {RANKS} ranks x 32 chunks of {} entries (last chunk 576); reduce block {} threads, one launch; top-{TOP}",
        1024, BLOCK.0
    );
    println!(
        "comparison class (declared before any number): the reduce's token, stored value, 128 block maxima/argmax entries, top-5 indices/values, status and certified count must equal the host assembly's on the same four device shard buffers BIT FOR BIT; two reduce launches must be byte-identical. The host assembly is lm_head_reduce_ref::assemble, the same block fold the c4c chain performs with consider_head_block. Not compared with a tolerance."
    );
    check_floor(args.floor_gib, "m19 start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();

    // The pin checks happen before any device allocation, as in m16/m18.
    let screen_cubin = pinned_cubin(&args.cubin_dir, SCREEN_KERNEL, SCREEN_BYTES, SCREEN_SHA256)?;
    let reduce_cubin = pinned_cubin(&args.cubin_dir, REDUCE_KERNEL, REDUCE_BYTES, REDUCE_SHA256)?;
    check_pin(SCREEN_KERNEL, SCREEN_BYTES, SCREEN_SHA256)?;
    check_pin(REDUCE_KERNEL, REDUCE_BYTES, REDUCE_SHA256)?;
    Verdict::push(
        &mut verdicts,
        "released geometry and the m16 block layout",
        VOCAB == 129_280
            && VOCAB == RANKS * VOCAB_PER_RANK
            && BLOCKS == 128
            && TOP == 5
            && (VOCAB_PER_RANK - 31 * 1024) == 576,
        format!(
            "vocab {VOCAB} = {RANKS} x {VOCAB_PER_RANK}, {BLOCKS} blocks, last chunk 576 entries, top-{TOP}"
        ),
    );
    let geometry_ok = verdicts.last().is_some_and(|verdict| verdict.ok);
    if !geometry_ok {
        return Err("m19: the harness geometry is not the released head".to_owned());
    }

    let control = real_token_hidden(&args.head_input)?;
    println!("hidden seam [{}]: {}", control.0, control.2);
    let concentrated = concentrated_hidden();
    println!("hidden seam [{}]: {}", concentrated.0, concentrated.2);

    let reference = match &args.reference_logits {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|error| {
                format!("{}: {error} (the phase-1 reference logits)", path.display())
            })?;
            if bytes.len() != REFERENCE_LOGITS_BYTES {
                return Err(format!(
                    "{}: the reference logits are {} B, expected {REFERENCE_LOGITS_BYTES} B (129280 f32)",
                    path.display(),
                    bytes.len()
                ));
            }
            let values = bytes_to_f32(&bytes);
            let (value, at) = best_entry(&values);
            println!(
                "reference engine logits: {} f32 from {} (sha256 {}, argmax {at} value {value:.9}); declared as the phase-1 reference, not recomputed here",
                values.len(),
                path.display(),
                sha256_hex(&bytes)
            );
            Some(values)
        }
        None => {
            println!(
                "reference engine logits: not supplied (--reference-logits); the argmax is compared to the host assembly only and the reference comparison is not claimed"
            );
            None
        }
    };

    let mut snap = Snapshot::open(&args.checkpoint)?;
    let head = snap.read_raw("head.weight", "BF16", &[VOCAB, HIDDEN])?;
    if head.len() != VOCAB * HIDDEN * 2 {
        return Err(format!(
            "head.weight read {} B, the released head is {} B: refusing to upload a differently sized tensor",
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

    let cuda = Cuda::load()?;
    let (screen_module, screen_kernel) = preview_load(&cuda, &screen_cubin)?;
    let (reduce_module, reduce_kernel) = launch::load(&cuda, &reduce_cubin)?;
    let _modules = (screen_module, reduce_module);

    let mut allocated = 0usize;
    let weight_d = cuda.alloc(head.len(), &mut allocated)?;
    cuda.copy_htod(&weight_d, &head)?;
    let block_maxima = lm_head_ref::block_maxima_bf16(&head);
    let rank_max_bytes = VOCAB_PER_RANK * lm_head_ref::LM_BLOCKS * 2;
    let w_block_max_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(rank_max_bytes, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    for rank in 0..RANKS {
        cuda.copy_htod(
            &w_block_max_d[rank],
            &block_maxima[rank * rank_max_bytes..(rank + 1) * rank_max_bytes],
        )?;
    }
    let hidden_d = cuda.alloc(HEAD_INPUT_BYTES, &mut allocated)?;
    let logits_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(VOCAB_PER_RANK * 4, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    let certified_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(VOCAB_PER_RANK, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    let counts_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(4, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    let screen_block_max_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(32 * 4, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    let screen_block_argmax_d: Vec<Deviceptr> = (0..RANKS)
        .map(|_| cuda.alloc(32 * 4, &mut allocated))
        .collect::<Result<Vec<_>, _>>()?;
    let buffers = ReduceBuffers {
        token: cuda.alloc(4, &mut allocated)?,
        value: cuda.alloc(4, &mut allocated)?,
        block_max: cuda.alloc(BLOCKS * 4, &mut allocated)?,
        block_argmax: cuda.alloc(BLOCKS * 4, &mut allocated)?,
        top5_value: cuda.alloc(TOP * 4, &mut allocated)?,
        top5_index: cuda.alloc(TOP * 4, &mut allocated)?,
        status: cuda.alloc(4, &mut allocated)?,
        certified_count: cuda.alloc(4, &mut allocated)?,
    };
    println!("device bytes admitted: {allocated}");
    let rank_stride = VOCAB_PER_RANK * HIDDEN * 2;

    // Case 1: the real-token control.
    let control_bytes = &control.1;
    let mut control_outputs = None;
    for (case_name, hidden_bytes, is_control) in [
        ("real-token control", control_bytes, true),
        ("concentrated certificate", &concentrated.1, false),
    ] {
        println!("case {case_name}:");
        cuda.copy_htod(&hidden_d, hidden_bytes)?;
        let outputs = run_screen(
            &cuda,
            screen_kernel,
            hidden_d,
            weight_d,
            &w_block_max_d,
            rank_stride,
            &logits_d,
            &certified_d,
            &counts_d,
            &screen_block_max_d,
            &screen_block_argmax_d,
        )?;
        let first = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
        let second = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
        let expected = lm_head_reduce_ref::assemble(&outputs.logits, &outputs.certified)?;
        let (host_token, host_value) =
            lm_head_reduce_ref::host_assembly_token(&outputs.logits, &outputs.certified)?;
        let blocks_outcome =
            compare_blocks(&first, &expected, &outputs.block_max, &outputs.block_argmax);

        if is_control {
            Verdict::push(
                &mut verdicts,
                "control: reduce token and value are the host assembly's, bit for bit",
                first.token == expected.token
                    && first.token == host_token
                    && first.value.to_bits() == expected.value.to_bits()
                    && first.value.to_bits() == host_value.to_bits(),
                format!(
                    "reduce token {} value {:.9}, host assembly token {host_token} value {host_value:.9}, reference {}",
                    first.token,
                    first.value,
                    reference_note(reference.as_ref())
                ),
            );
            push_outcome(
                &mut verdicts,
                "control: all 128 block maxima and argmax entries match the screen's published arrays",
                blocks_outcome.clone(),
            );
            let count_flags: usize = outputs
                .certified
                .iter()
                .map(|flags| flags.iter().filter(|flag| **flag == 1).count())
                .sum();
            push_outcome(
                &mut verdicts,
                "control: certified count 0, top-5 claimed, no refusal",
                compare_count_and_status(
                    &first,
                    0,
                    count_flags,
                    &outputs.certified_count,
                    STATUS_TOP5_CLAIMED,
                ),
            );
            push_outcome(
                &mut verdicts,
                "control: top-5 set is the host's and the reference's",
                compare_top5(&first, &expected, reference.as_ref()),
            );
            let reference_ok = match &reference {
                Some(values) => best_entry(values).1 as u32 == first.token,
                None => true,
            };
            Verdict::push(
                &mut verdicts,
                "control: reduce token is the phase-1 reference engine's argmax",
                reference_ok,
                reference_note(reference.as_ref()),
            );
            Verdict::push(
                &mut verdicts,
                "control: two reduce launches are byte-identical",
                first.bytes() == second.bytes() && first.bits_equal(&second),
                format!("{} report bytes compared", first.bytes().len()),
            );
            control_outputs = Some((first, expected));
        } else {
            let flagged: Vec<usize> = outputs
                .certified
                .iter()
                .map(|flags| flags.iter().filter(|flag| **flag == 1).count())
                .collect();
            push_outcome(
                &mut verdicts,
                "concentrated: the screen certifies 4,899 entries (per rank 1,062/1,205/1,215/1,417)",
                compare_count_and_status(
                    &first,
                    CONCENTRATED_CERTIFIED as u32,
                    flagged.iter().sum::<usize>(),
                    &outputs.certified_count,
                    STATUS_ANY_CERTIFIED,
                )
                .and_then(|detail| {
                    if flagged == CONCENTRATED_PER_RANK {
                        Ok(detail)
                    } else {
                        Err(format!(
                            "per-rank flags {flagged:?} are not the declared {:?}",
                            CONCENTRATED_PER_RANK
                        ))
                    }
                }),
            );
            Verdict::push(
                &mut verdicts,
                "concentrated: reduce token is the host assembly's and the declared 33,865, bit for bit",
                first.token == expected.token
                    && first.token == host_token
                    && first.token as usize == CONCENTRATED_TOKEN
                    && first.value.to_bits() == expected.value.to_bits()
                    && first.value.to_bits() == host_value.to_bits(),
                format!(
                    "reduce token {} value {:.9}, host assembly token {host_token} value {host_value:.9}, declared {CONCENTRATED_TOKEN}",
                    first.token, first.value
                ),
            );
            push_outcome(
                &mut verdicts,
                "concentrated: all 128 block maxima and argmax entries match the screen's published arrays",
                blocks_outcome.clone(),
            );
            let status_outcome = if first.status == STATUS_ANY_CERTIFIED
                && first.status & STATUS_WINNER_CERTIFIED == 0
                && first.status & STATUS_TOP5_CLAIMED == 0
            {
                Ok(format!(
                    "status 0x{:x}; top-5 set NOT claimed ({} certified entries store bounds)",
                    first.status, first.certified_count
                ))
            } else {
                Err(format!(
                    "status 0x{:x}, expected exactly 0x{:x} (ANY_CERTIFIED only; no WINNER, no TOP5_CLAIMED)",
                    first.status, STATUS_ANY_CERTIFIED
                ))
            };
            push_outcome(
                &mut verdicts,
                "concentrated: exact winner, top-5 NOT claimed (certified entries store bounds)",
                status_outcome,
            );
            Verdict::push(
                &mut verdicts,
                "concentrated: two reduce launches are byte-identical",
                first.bytes() == second.bytes() && first.bits_equal(&second),
                format!("{} report bytes compared", first.bytes().len()),
            );
        }
    }
    let _ = control_outputs;

    // Case 3: synthetic full-width refusal and tie semantics.
    let mut synthetic_logits = vec![-100.0f32; VOCAB];
    synthetic_logits[7] = 5.0;
    synthetic_logits[VOCAB_PER_RANK + 100] = 5.0;
    upload_synthetic(&cuda, &logits_d, &certified_d, &synthetic_logits, None)?;
    let tie = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
    Verdict::push(
        &mut verdicts,
        "synthetic tie: the lowest global entry wins",
        tie.token == 7
            && tie.value.to_bits() == 5.0f32.to_bits()
            && tie.status == STATUS_TOP5_CLAIMED
            && tie.top5_index[0] == 7
            && tie.top5_index[1] == (VOCAB_PER_RANK + 100) as u32,
        format!(
            "token {} value {:.1}, top-5 starts {:?}, status 0x{:x}",
            tie.token,
            tie.value,
            &tie.top5_index[..2],
            tie.status
        ),
    );

    // Synthetic nonzero-block offsets: the block's published entry must be its base + local, in the
    // middle of a rank and in the 576-entry last block of the last rank.
    let mut offsets_logits = vec![-100.0f32; VOCAB];
    offsets_logits[1_024 + 100] = 9.0;
    offsets_logits[3 * VOCAB_PER_RANK + 31 * 1_024 + 500] = 8.0;
    upload_synthetic(&cuda, &logits_d, &certified_d, &offsets_logits, None)?;
    let offsets = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
    Verdict::push(
        &mut verdicts,
        "synthetic block offsets: a nonzero block's argmax is base + local",
        offsets.block_argmax[1] == 1_124
            && offsets.block_argmax[127] == (3 * VOCAB_PER_RANK + 31 * 1_024 + 500) as u32
            && offsets.token == 1_124
            && offsets.top5_index[0] == 1_124
            && offsets.top5_index[1] == (3 * VOCAB_PER_RANK + 31 * 1_024 + 500) as u32,
        format!(
            "block 1 argmax {} (want 1124), block 127 argmax {} (want 129204), token {}",
            offsets.block_argmax[1], offsets.block_argmax[127], offsets.token
        ),
    );

    // A certified entry is a bound: when its stored bound ties the exact maximum at a higher index,
    // the lower-index bound wins the stored-value order and the reduce must REFUSE the token rather
    // than assert the bound as an exact answer.
    let mut bound_tie_logits = vec![-100.0f32; VOCAB];
    bound_tie_logits[100] = 5.0;
    bound_tie_logits[200] = 5.0;
    let mut bound_tie_flags = vec![0u8; VOCAB];
    bound_tie_flags[100] = 1;
    upload_synthetic(
        &cuda,
        &logits_d,
        &certified_d,
        &bound_tie_logits,
        Some(&bound_tie_flags),
    )?;
    let bound_tie = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
    Verdict::push(
        &mut verdicts,
        "synthetic certified bound tying the exact max: refuses the token",
        bound_tie.token == 100
            && bound_tie.status & STATUS_WINNER_CERTIFIED != 0
            && bound_tie.status & STATUS_TOP5_CLAIMED == 0
            && bound_tie.certified_count == 1,
        format!(
            "token {} (diagnostic) status 0x{:x}; the certified bound must not be claimed as exact",
            bound_tie.token, bound_tie.status
        ),
    );

    // A certified winner in a rank other than 0: the flag lookup must use the global entry, not the
    // local one, and rank 2's flag array must be the one read.
    let mut rank2_logits = vec![-100.0f32; VOCAB];
    let rank2_winner = (2 * VOCAB_PER_RANK + 100) as u32;
    rank2_logits[rank2_winner as usize] = 9.0;
    let mut rank2_flags = vec![0u8; VOCAB];
    rank2_flags[rank2_winner as usize] = 1;
    upload_synthetic(
        &cuda,
        &logits_d,
        &certified_d,
        &rank2_logits,
        Some(&rank2_flags),
    )?;
    let rank2 = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
    Verdict::push(
        &mut verdicts,
        "synthetic certified winner in rank 2: WINNER_CERTIFIED is set",
        rank2.token == rank2_winner
            && rank2.status == (STATUS_WINNER_CERTIFIED | STATUS_ANY_CERTIFIED)
            && rank2.certified_count == 1,
        format!(
            "rank-2 token {} (diagnostic) status 0x{:x}, want 0x5",
            rank2.token, rank2.status
        ),
    );

    let mut refusal_logits = vec![-100.0f32; VOCAB];
    refusal_logits[4_242] = 2.0;
    refusal_logits[777] = 1.0;
    let mut refusal_flags = vec![0u8; VOCAB];
    refusal_flags[4_242] = 1;
    upload_synthetic(
        &cuda,
        &logits_d,
        &certified_d,
        &refusal_logits,
        Some(&refusal_flags),
    )?;
    let refused = run_reduce(&cuda, reduce_kernel, &logits_d, &certified_d, &buffers)?;
    Verdict::push(
        &mut verdicts,
        "synthetic certified winner: refuses the token and the top-5 instead of asserting",
        refused.token == 4_242
            && refused.status & STATUS_WINNER_CERTIFIED != 0
            && refused.status & STATUS_ANY_CERTIFIED != 0
            && refused.status & STATUS_TOP5_CLAIMED == 0
            && refused.certified_count == 1,
        format!(
            "token {} is diagnostic; status 0x{:x} names WINNER_CERTIFIED, no token/top-5 claim",
            refused.token, refused.status
        ),
    );

    check_floor(args.floor_gib, "m19 end")?;

    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name)
        .collect();
    if failed.is_empty() {
        println!(
            "m19: PASS ({} checks; real-token control, concentrated certificate, synthetic tie/refusal; one pinned cubin)",
            verdicts.len()
        );
        Ok(())
    } else {
        let details: Vec<String> = verdicts
            .iter()
            .filter(|verdict| !verdict.ok)
            .map(|verdict| format!("{}: {}", verdict.name, verdict.detail))
            .collect();
        Err(format!(
            "m19: FAIL at {} of {} checks: {:?}",
            failed.len(),
            verdicts.len(),
            details
        ))
    }
}

fn preview_load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cert_lm_head_launch::load(cuda, cubin)
}

fn reference_note(reference: Option<&Vec<f32>>) -> String {
    match reference {
        Some(values) => {
            let (value, at) = best_entry(values);
            format!("argmax {at} value {value:.9}")
        }
        None => "not supplied".to_owned(),
    }
}

fn best_entry(values: &[f32]) -> (f32, usize) {
    let mut best = f32::NEG_INFINITY;
    let mut at = 0usize;
    for (entry, value) in values.iter().enumerate() {
        if *value > best {
            best = *value;
            at = entry;
        }
    }
    (best, at)
}

fn top_indices(values: &[f32], count: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = values
        .iter()
        .enumerate()
        .map(|(index, value)| (*value, index as u32))
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(count);
    scored.into_iter().map(|(_, index)| index).collect()
}

/// The declared real-token seam: (name, bf16 bytes, provenance). Refused if missing or changed.
fn real_token_hidden(path: &Path) -> Result<(&'static str, Vec<u8>, String), String> {
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "{}: {error} (the declared phase-1 chain head input)",
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
            "{}: sha256 {sha} is not the declared phase-1 chain artifact {HEAD_INPUT_SHA256}: refusing to substitute or fabricate a hidden state",
            path.display()
        ));
    }
    let values = bytes_to_f32(&bytes);
    let bytes = values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect::<Vec<_>>();
    Ok((
        "real-token control",
        bytes,
        format!(
            "phase-1 host chain head input from a Lead window, sha256-verified here (not produced by m19): {} f32, sha256 {sha}",
            path.display()
        ),
    ))
}

/// The declared concentrated case: bf16(1.0) at SYNTHETIC_INDEX, zeros elsewhere.
fn concentrated_hidden() -> (&'static str, Vec<u8>, String) {
    let mut values = vec![0.0f32; HIDDEN];
    values[SYNTHETIC_INDEX] = 1.0;
    let bytes = values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect::<Vec<_>>();
    (
        "concentrated certificate",
        bytes,
        format!(
            "declared synthetic hidden: bf16(1.0) at index {SYNTHETIC_INDEX}, zero elsewhere; not a released hidden state and no token is claimed from it"
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_screen(
    cuda: &Cuda,
    kernel: Kernel,
    hidden_d: Deviceptr,
    weight_d: Deviceptr,
    w_block_max_d: &[Deviceptr],
    rank_stride: usize,
    logits_d: &[Deviceptr],
    certified_d: &[Deviceptr],
    counts_d: &[Deviceptr],
    block_max_d: &[Deviceptr],
    block_argmax_d: &[Deviceptr],
) -> Result<ScreenOutputs, String> {
    for rank in 0..RANKS {
        cuda.copy_htod(&counts_d[rank], &[0u8; 4])?;
        let params = CertLmHeadParams::new(
            hidden_d,
            Deviceptr(weight_d.0 + (rank * rank_stride) as u64),
            w_block_max_d[rank],
            logits_d[rank],
            certified_d[rank],
            counts_d[rank],
            block_max_d[rank],
            block_argmax_d[rank],
            VOCAB_PER_RANK as u32,
            0,
            0,
            0,
        );
        cert_lm_head_launch::launch(cuda, kernel, &params)?;
    }
    let mut outputs = ScreenOutputs {
        logits: Vec::with_capacity(RANKS),
        certified: Vec::with_capacity(RANKS),
        block_max: Vec::with_capacity(RANKS),
        block_argmax: Vec::with_capacity(RANKS),
        certified_count: Vec::with_capacity(RANKS),
    };
    for rank in 0..RANKS {
        let mut logits_bytes = vec![0u8; VOCAB_PER_RANK * 4];
        let mut certified_bytes = vec![0u8; VOCAB_PER_RANK];
        let mut count_bytes = [0u8; 4];
        let mut block_max_bytes = vec![0u8; 32 * 4];
        let mut block_argmax_bytes = vec![0u8; 32 * 4];
        cuda.copy_dtoh(&mut logits_bytes, &logits_d[rank])?;
        cuda.copy_dtoh(&mut certified_bytes, &certified_d[rank])?;
        cuda.copy_dtoh(&mut count_bytes, &counts_d[rank])?;
        cuda.copy_dtoh(&mut block_max_bytes, &block_max_d[rank])?;
        cuda.copy_dtoh(&mut block_argmax_bytes, &block_argmax_d[rank])?;
        outputs.logits.push(bytes_to_f32(&logits_bytes));
        outputs.certified.push(certified_bytes);
        outputs
            .certified_count
            .push(u32::from_le_bytes(count_bytes));
        outputs.block_max.push(f32s(&block_max_bytes));
        outputs.block_argmax.push(u32s(&block_argmax_bytes));
    }
    Ok(outputs)
}

fn run_reduce(
    cuda: &Cuda,
    kernel: Kernel,
    logits_d: &[Deviceptr],
    certified_d: &[Deviceptr],
    buffers: &ReduceBuffers,
) -> Result<ReduceReport, String> {
    let params = launch::LmHeadReduceParams::new(
        [logits_d[0], logits_d[1], logits_d[2], logits_d[3]],
        [
            certified_d[0],
            certified_d[1],
            certified_d[2],
            certified_d[3],
        ],
        buffers.token,
        buffers.value,
        buffers.block_max,
        buffers.block_argmax,
        buffers.top5_value,
        buffers.top5_index,
        buffers.status,
        buffers.certified_count,
        0,
        0,
        0,
    );
    launch::launch(cuda, kernel, &params)?;
    let mut token_bytes = [0u8; 4];
    let mut value_bytes = [0u8; 4];
    let mut block_max_bytes = vec![0u8; BLOCKS * 4];
    let mut block_argmax_bytes = vec![0u8; BLOCKS * 4];
    let mut top5_value_bytes = vec![0u8; TOP * 4];
    let mut top5_index_bytes = vec![0u8; TOP * 4];
    let mut status_bytes = [0u8; 4];
    let mut count_bytes = [0u8; 4];
    cuda.copy_dtoh(&mut token_bytes, &buffers.token)?;
    cuda.copy_dtoh(&mut value_bytes, &buffers.value)?;
    cuda.copy_dtoh(&mut block_max_bytes, &buffers.block_max)?;
    cuda.copy_dtoh(&mut block_argmax_bytes, &buffers.block_argmax)?;
    cuda.copy_dtoh(&mut top5_value_bytes, &buffers.top5_value)?;
    cuda.copy_dtoh(&mut top5_index_bytes, &buffers.top5_index)?;
    cuda.copy_dtoh(&mut status_bytes, &buffers.status)?;
    cuda.copy_dtoh(&mut count_bytes, &buffers.certified_count)?;
    let mut top5_value = [0.0f32; TOP];
    let mut top5_index = [0u32; TOP];
    top5_value.copy_from_slice(&f32s(&top5_value_bytes));
    top5_index.copy_from_slice(&u32s(&top5_index_bytes));
    Ok(ReduceReport {
        token: u32::from_le_bytes(token_bytes),
        value: f32::from_le_bytes(value_bytes),
        block_max: f32s(&block_max_bytes),
        block_argmax: u32s(&block_argmax_bytes),
        top5_value,
        top5_index,
        status: u32::from_le_bytes(status_bytes),
        certified_count: u32::from_le_bytes(count_bytes),
    })
}

fn upload_synthetic(
    cuda: &Cuda,
    logits_d: &[Deviceptr],
    certified_d: &[Deviceptr],
    logits: &[f32],
    certified: Option<&[u8]>,
) -> Result<(), String> {
    assert_eq!(logits.len(), VOCAB);
    let mut logits_bytes = Vec::with_capacity(VOCAB * 4);
    for value in logits {
        logits_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for (rank, ptr) in logits_d.iter().enumerate() {
        let start = rank * VOCAB_PER_RANK * 4;
        cuda.copy_htod(ptr, &logits_bytes[start..start + VOCAB_PER_RANK * 4])?;
    }
    let flags = certified.map_or_else(|| vec![0u8; VOCAB], |flags| flags.to_vec());
    assert_eq!(flags.len(), VOCAB);
    for (rank, ptr) in certified_d.iter().enumerate() {
        let start = rank * VOCAB_PER_RANK;
        cuda.copy_htod(ptr, &flags[start..start + VOCAB_PER_RANK])?;
    }
    Ok(())
}

fn push_outcome(verdicts: &mut Vec<Verdict>, name: &'static str, outcome: Result<String, String>) {
    match outcome {
        Ok(detail) => Verdict::push(verdicts, name, true, detail),
        Err(detail) => Verdict::push(verdicts, name, false, detail),
    }
}

/// Compares the reduce's 128 block answers with the host reference's and with the screen's published
/// arrays remapped rank-major. Returns the detail line on success and the exact first mismatch on
/// failure, so a FAILED check never prints success text (b42's lesson).
fn compare_blocks(
    report: &ReduceReport,
    expected: &Reduction,
    screen_block_max: &[Vec<f32>],
    screen_block_argmax: &[Vec<u32>],
) -> Result<String, String> {
    if report.block_max.len() != BLOCKS || report.block_argmax.len() != BLOCKS {
        return Err(format!(
            "the reduce report has {} maxima and {} entries, expected {BLOCKS} each",
            report.block_max.len(),
            report.block_argmax.len()
        ));
    }
    for block in 0..BLOCKS {
        if report.block_max[block].to_bits() != expected.block_max[block].to_bits() {
            return Err(format!(
                "block {block} max: reduce {} (0x{:08x}) vs host reference {} (0x{:08x})",
                report.block_max[block],
                report.block_max[block].to_bits(),
                expected.block_max[block],
                expected.block_max[block].to_bits()
            ));
        }
        if report.block_argmax[block] != expected.block_argmax[block] {
            return Err(format!(
                "block {block} argmax: reduce {} vs host reference {}",
                report.block_argmax[block], expected.block_argmax[block]
            ));
        }
        let (rank, local) = (block / 32, block % 32);
        let screen_value = screen_block_max[rank][local];
        let screen_index = (rank * VOCAB_PER_RANK) as u32 + screen_block_argmax[rank][local];
        if report.block_max[block].to_bits() != screen_value.to_bits()
            || report.block_argmax[block] != screen_index
        {
            return Err(format!(
                "block {block} vs screen remap: reduce ({} / {}), screen ({} / {})",
                report.block_max[block], report.block_argmax[block], screen_value, screen_index
            ));
        }
    }
    Ok(format!(
        "{BLOCKS} of {BLOCKS} blocks bit-identical to the screen remapped rank-major, and to the host reference"
    ))
}

/// Compares the top-5 arrays with the host reference and, when a reference logits buffer is
/// supplied and the set is claimed, the set with the reference engine's top-5.
fn compare_top5(
    report: &ReduceReport,
    expected: &Reduction,
    reference: Option<&Vec<f32>>,
) -> Result<String, String> {
    for slot in 0..TOP {
        if report.top5_index[slot] != expected.top5_index[slot] {
            return Err(format!(
                "top-5 slot {slot} entry: reduce {} vs host reference {}",
                report.top5_index[slot], expected.top5_index[slot]
            ));
        }
        if report.top5_value[slot].to_bits() != expected.top5_value[slot].to_bits() {
            return Err(format!(
                "top-5 slot {slot} value: reduce {} (0x{:08x}) vs host reference {} (0x{:08x})",
                report.top5_value[slot],
                report.top5_value[slot].to_bits(),
                expected.top5_value[slot],
                expected.top5_value[slot].to_bits()
            ));
        }
    }
    if report.status & STATUS_TOP5_CLAIMED == 0 {
        return Err(format!(
            "top-5 bit clear: status 0x{:x} does not claim the set, so it must not be compared as exact",
            report.status
        ));
    }
    match reference {
        Some(values) => {
            let mut reference_top = top_indices(values, TOP);
            reference_top.sort_unstable();
            let mut device_top = report.top5_index.to_vec();
            device_top.sort_unstable();
            if device_top != reference_top {
                return Err(format!(
                    "top-5 set: device sorted {device_top:?} vs reference sorted {reference_top:?}"
                ));
            }
            Ok("top-5 set vs the reference identical (control, all values exact)".to_owned())
        }
        None => Ok(
            "top-5 set NOT compared with a reference (not supplied); host reference only"
                .to_owned(),
        ),
    }
}

/// Compares the certified count with the expected total and the screen's published per-rank counts,
/// and the status word with the expected value.
fn compare_count_and_status(
    report: &ReduceReport,
    expected_count: u32,
    flagged_total: usize,
    published_counts: &[u32],
    expected_status: u32,
) -> Result<String, String> {
    let published_sum: u64 = published_counts.iter().map(|count| u64::from(*count)).sum();
    if report.certified_count != expected_count {
        return Err(format!(
            "reduce certified count {} != expected {expected_count}",
            report.certified_count
        ));
    }
    if published_sum != u64::from(expected_count) {
        return Err(format!(
            "published per-rank certified counts {published_counts:?} sum to {published_sum}, expected {expected_count}"
        ));
    }
    if flagged_total != expected_count as usize {
        return Err(format!(
            "the copied flags hold {flagged_total} entries, expected {expected_count}"
        ));
    }
    if report.status != expected_status {
        return Err(format!(
            "status 0x{:x} != expected 0x{expected_status:x}",
            report.status
        ));
    }
    Ok(format!(
        "{flagged_total} flagged entries = screen-published {published_sum} = reduce count {}, status 0x{:x}",
        report.certified_count, report.status
    ))
}

fn pinned_cubin(dir: &Path, kernel: &str, bytes: usize, sha: &str) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/{kernel}.cu -o {})",
            path.display(),
            path.display()
        )
    })?;
    let digest = sha256_hex(&cubin);
    println!(
        "cubin: {} ({} B, sha256 {digest})",
        path.display(),
        cubin.len()
    );
    if cubin.len() != bytes || digest != sha {
        return Err(format!(
            "{kernel}: pinned {bytes} B / {sha}, got {} B / {digest}; refusing to measure a different kernel",
            cubin.len()
        ));
    }
    Ok(cubin)
}

fn check_pin(kernel: &str, bytes: usize, sha: &str) -> Result<(), String> {
    let pin = baseline::pin(kernel)?
        .ok_or_else(|| format!("the baseline manifest does not pin {kernel}"))?;
    if pin.cubin_bytes != bytes || pin.cubin_sha256 != sha || pin.ptx_membar_lines != 0 {
        return Err(format!(
            "{kernel}: baseline pins {} B / {} / membar {}, the harness declares {bytes} B / {sha} / membar 0",
            pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
        ));
    }
    println!(
        "baseline manifest agrees: {kernel} {} B, {}, {} PTX membar lines",
        pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_kernels::lm_head_reduce_launch::STATUS_BITS;

    #[test]
    fn the_status_bits_are_the_pinned_contract() {
        assert_eq!(STATUS_BITS, [0x1, 0x2, 0x4]);
        assert_eq!(STATUS_WINNER_CERTIFIED, 1);
        assert_eq!(STATUS_TOP5_CLAIMED, 2);
        assert_eq!(STATUS_ANY_CERTIFIED, 4);
    }

    #[test]
    fn the_concentrated_counts_are_the_declared_ones() {
        assert_eq!(
            CONCENTRATED_PER_RANK.iter().sum::<usize>(),
            CONCENTRATED_CERTIFIED
        );
        assert_eq!(CONCENTRATED_CERTIFIED, 4_899);
        assert_eq!(CONCENTRATED_TOKEN, 33_865);
        assert_eq!(VOCAB, 129_280);
        assert_eq!(VOCAB_PER_RANK, 32_320);
        assert_eq!(BLOCKS, 128);
    }

    #[test]
    fn the_pins_are_the_baseline_ones() {
        check_pin(SCREEN_KERNEL, SCREEN_BYTES, SCREEN_SHA256).expect("screen pin");
        check_pin(REDUCE_KERNEL, REDUCE_BYTES, REDUCE_SHA256).expect("reduce pin");
    }

    fn fixture() -> (ReduceReport, Reduction, Vec<Vec<f32>>, Vec<Vec<u32>>) {
        let screen_block_max = vec![vec![7.0f32; 32]; RANKS];
        let screen_block_argmax = vec![vec![11u32; 32]; RANKS];
        let block_max = vec![7.0f32; BLOCKS];
        let block_argmax = (0..BLOCKS)
            .map(|block| ((block / 32) * VOCAB_PER_RANK + 11) as u32)
            .collect::<Vec<_>>();
        let top5_value = [4.0f32, 3.0, 2.0, 1.0, 0.0];
        let top5_index = [5u32, 6, 7, 8, 9];
        let expected = Reduction {
            token: 5,
            value: 4.0,
            block_max: block_max.clone(),
            block_argmax: block_argmax.clone(),
            top5_value,
            top5_index,
            status: STATUS_TOP5_CLAIMED,
            certified_count: 0,
        };
        let report = ReduceReport {
            token: 5,
            value: 4.0,
            block_max,
            block_argmax,
            top5_value,
            top5_index,
            status: STATUS_TOP5_CLAIMED,
            certified_count: 0,
        };
        (report, expected, screen_block_max, screen_block_argmax)
    }

    fn reference_for(top5: &[u32]) -> Vec<f32> {
        let mut values = vec![-1.0f32; VOCAB];
        for (slot, index) in top5.iter().enumerate() {
            values[*index as usize] = 10.0 - slot as f32;
        }
        values
    }

    #[test]
    fn the_block_comparison_accepts_equal_and_rejects_one_changed_entry() {
        let (mut report, expected, screen_block_max, screen_block_argmax) = fixture();
        assert!(
            compare_blocks(&report, &expected, &screen_block_max, &screen_block_argmax).is_ok(),
            "the equal fixture must pass"
        );

        report.block_max[5] = 7.5;
        let error = compare_blocks(&report, &expected, &screen_block_max, &screen_block_argmax)
            .expect_err("a changed block max must fail");
        assert!(error.starts_with("block 5 max"), "{error}");
        report.block_max[5] = 7.0;

        report.block_argmax[63] = 0;
        let error = compare_blocks(&report, &expected, &screen_block_max, &screen_block_argmax)
            .expect_err("a changed block argmax must fail");
        assert!(error.starts_with("block 63 argmax"), "{error}");
        report.block_argmax[63] = (1 * VOCAB_PER_RANK + 11) as u32;

        let (mut screen_block_max, mut screen_block_argmax) =
            (screen_block_max.clone(), screen_block_argmax.clone());
        screen_block_argmax[1][3] = 12;
        let error = compare_blocks(&report, &expected, &screen_block_max, &screen_block_argmax)
            .expect_err("a screen remap mismatch must fail");
        assert!(error.starts_with("block 35 vs screen remap"), "{error}");
        screen_block_argmax[1][3] = 11;
        assert!(
            compare_blocks(&report, &expected, &screen_block_max, &screen_block_argmax).is_ok(),
            "restoring the screen array must pass again"
        );

        let short = ReduceReport {
            block_max: vec![7.0; 3],
            ..report.clone()
        };
        assert!(
            compare_blocks(&short, &expected, &screen_block_max, &screen_block_argmax).is_err(),
            "a short report must fail"
        );
    }

    #[test]
    fn the_top5_comparison_rejects_a_duplicate_unclaimed_or_changed_set() {
        let (mut report, expected, _, _) = fixture();
        let reference = reference_for(&expected.top5_index);
        assert!(
            compare_top5(&report, &expected, Some(&reference)).is_ok(),
            "the equal fixture must pass"
        );

        report.top5_index[1] = report.top5_index[0];
        let error = compare_top5(&report, &expected, Some(&reference))
            .expect_err("a duplicate index must fail");
        assert!(error.contains("slot 1 entry"), "{error}");
        report.top5_index[1] = 6;

        report.status = STATUS_ANY_CERTIFIED;
        let error = compare_top5(&report, &expected, Some(&reference))
            .expect_err("an unclaimed set must fail");
        assert!(error.contains("top-5 bit clear"), "{error}");
        report.status = STATUS_TOP5_CLAIMED;

        let other_reference = reference_for(&[9, 8, 7, 6, 10]);
        let error = compare_top5(&report, &expected, Some(&other_reference))
            .expect_err("a different reference set must fail");
        assert!(error.contains("top-5 set"), "{error}");

        let mut changed = expected.clone();
        changed.top5_index[4] = 10;
        let error = compare_top5(&report, &changed, Some(&reference))
            .expect_err("a changed host-reference slot must fail");
        assert!(error.contains("slot 4 entry"), "{error}");
    }

    #[test]
    fn the_count_and_status_comparison_rejects_off_by_one_and_a_wrong_status() {
        let (mut report, _, _, _) = fixture();
        assert!(
            compare_count_and_status(&report, 0, 0, &[0u32; RANKS], STATUS_TOP5_CLAIMED).is_ok(),
            "the equal fixture must pass"
        );

        report.certified_count = 1;
        assert!(
            compare_count_and_status(&report, 0, 0, &[0u32; RANKS], STATUS_TOP5_CLAIMED).is_err(),
            "an off-by-one reduce count must fail"
        );
        report.certified_count = 0;

        assert!(
            compare_count_and_status(&report, 0, 0, &[1u32, 0, 0, 0], STATUS_TOP5_CLAIMED).is_err(),
            "a published-count mismatch must fail"
        );
        assert!(
            compare_count_and_status(&report, 0, 1, &[0u32; RANKS], STATUS_TOP5_CLAIMED).is_err(),
            "a flagged-total mismatch must fail"
        );
        assert!(
            compare_count_and_status(&report, 0, 0, &[0u32; RANKS], STATUS_ANY_CERTIFIED).is_err(),
            "a wrong status must fail"
        );
    }
}
