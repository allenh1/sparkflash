#![forbid(unsafe_code)]
//! Phase 2 milestone 2B: the device prefill/decode sequence harness.
//!
//! This example is the counterpart of the stage-9 chain in m7_layer0.rs: stage 9 chains forty
//! layers for ONE token and compares host/device boundaries; this harness runs the 130-token
//! sequence through the task-7 device KV store, with the CED split as a scheduling property, and
//! compares the device column against the 2A golden fixture on the compared set only.
//!
//! # Scope of this file, stated before any number
//!
//! The crate is built and reviewed without a GPU; CI cannot execute a kernel. The CLI has three
//! exclusive modes and every report's first line names the one that ran:
//!
//! * --dry-run (host only): builds the exact rotation and dispatch plans, bounds-checks every union
//!   call through the shared checked builder, and exercises the token-then-logits comparison
//!   plumbing against the golden fixture, with negative controls. No CUDA, no device allocation.
//! * --device: measures the window rotation with real device-to-device copies and runs the
//!   format-2/format-1 attention dispatch over harness-owned inputs. It is NOT the model forward:
//!   it produces no token and attempts no logits comparison, and its report says so.
//! * default --self-check: the CPU checks a host run prints; the unit tests beside this example run
//!   in CI through the test-examples job (`cargo test -p sparkflash-kernels --examples`).
//!
//! Every device number a run reports is measured in the window that produced it; this file contains
//! no assumed latency and no assumed byte count outside the store's own accounting.
//!
//! Two attention modes are named here because they must never be confused in a report:
//!
//! * AttentionMode::WindowOnly - the diagnostic column the Lead may sequence once: the existing
//!   kernel formats (0/1) attend the sliding-window rows only, and out-of-window compressed
//!   candidates are structurally excluded by the kernel predicate. It is labelled "window-only
//!   device column (compressed candidates structurally excluded), no equality claim" everywhere it
//!   appears, and its measured last-position token/logits delta is evidence about the CED window
//!   column, not a 2B pass.
//! * AttentionMode::CsaUnion - the real mode: the union of the window slots and the causal
//!   selected compressed candidates in one softmax with the sink in the denominator (task-11,
//!   window_kv_format 2). The contract, as agreed with the kernel owner: window rows 0..window_rows
//!   oldest-first all attended, then selected entries in supplied order; selected entries are
//!   ABSOLUTE TOKEN positions read from kv_fp4/kv_scales by row = id; an entry is attended exactly
//!   when id <= position; duplicates are attended once per entry. No 2B claim may be made until
//!   this path exists and matches token-for-token.
//!
//! # Ratio-2 harness precondition
//!
//! Format 2 takes no ratio argument. For a ratio-2 source (encoder layers 2/8/14) the harness must
//! pass a per-position expanded view (row p = store row p/ratio) and pass id = c*ratio for a
//! selected compressed row c; ratio-1 sources (the decoder reading layer 20) need no expansion. The
//! selected entries are therefore COMPRESSED ROW IDS mapped injectively to an absolute id, not a
//! generic token-position list: reachability is the caller's precondition, and the harness must
//! pass the engine's reachability-filtered compressed ids (compressed_ids emits c <
//! (start_pos + query + 1)/ratio and NO_INDEX otherwise). The kernel's id <= position guard is not
//! that rule for ratio 2; the negative control in the tests asserts an unfiltered list differs. The
//! expansion is a harness responsibility so the kernel stays 19-parameter. Its row strides are the
//! compressed store/kernel layout: 256 B FP4 codes (512 latent packed two per byte) + 32 B E4M3
//! scales (one per 16 channels), NOT the engine window buffers' 512 B codes + 16 B E8M0 scales.
//!
//! # Compared set (D7)
//!
//! The engine reference runs all forty layers over the whole prompt; the CED device path runs
//! layers 0..=20 over the prompt and 21..=39 over the last 128 tokens. Positions older than
//! prompt_tokens - window are therefore never compared. The 2B-i comparison below reports position
//! 129 (token first, then logits) and refuses any other position; 2B-ii extends it to the eight
//! decode positions through compare_full_sequence.
//!
//! # Rotation measurement
//!
//! The device window slot is not one 528-byte blob: each layer owns two arrays (128 slots x 512 B
//! E4M3 codes and 128 slots x 16 B E8M0 scales), and the kernel requires the supplied rows to be
//! contiguous and oldest-first. The skeleton's first accounting charged every execution the
//! forty-ring volume (2,703,360 B) and so over-reported by the layer factor; the plan sums each
//! execution's own view instead. The harness therefore rotates the ring into a contiguous view per
//! layer per step and measures that copy (bytes and copy time) with a real device-to-device copy.
//! A layer's view starts at its first executed position (0 for encoder layers, 2 for decoder layers
//! in the 130-token schedule), so rows are min(position - first + 1, 128), not min(position + 1,
//! 128); Rot 2B measured the latter and over-counted 4,807 slots. The clock is advanced only by the
//! rotate_window_ring calls themselves: the rotation line is copy time, and the loop wall time is
//! reported separately so dispatch and read-back work cannot be charged to the copy rate.
//! RotationPlan pins the byte arithmetic so a changed slot/stage layout cannot slip through a
//! timing report.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use sparkflash_engine::ModelConfig;
use sparkflash_engine::layer_plan::{LayerMode, LayerPlan};
use sparkflash_engine::sequence_fixture::{
    DeviceStep, LOGITS_ABS_BAR, SequenceFixture, compare_logits, compare_steps, compared_positions,
    exclusion_boundary,
};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
#[cfg(test)]
use sparkflash_kernels::cuda_harness::ring_view_segments;
use sparkflash_kernels::cuda_harness::{
    FlashMlaKernel, FlashMlaParams, WindowRingLayout, assert_engine_window_finite,
    engine_window_row, flash_mla_params, host_ring_view, launch_flash_mla, load_flash_mla,
    ring_view_position, rotate_window_ring,
};
use sparkflash_kernels::mla_ref;
use sparkflash_kernels::sequence_chain::{
    CSA_UNION_COMPRESSED_CODE_ROW_BYTES, CSA_UNION_COMPRESSED_SCALE_ROW_BYTES, Format2AttentionCall,
};
use sparkflash_kv::cache::replay_layer_count;
use sparkflash_kv::config::{
    CANDIDATE_SOURCE_LAYER, Csa2Mode, INDEX_SOURCE_LAYERS, KV_SOURCE_LAYERS, NUM_HIDDEN_LAYERS,
    SWA_WINDOW_SIZE, compress_ratio, csa2_mode, kv_source_for,
};
use sparkflash_kv::device_store::StoreGeometry;
use sparkflash_kv::schedule::{
    DECODER_FIRST_LAYER, ENCODER_LAST_LAYER, Publication, PublishedHandle, ced_schedule,
};
use sparkflash_kv::swa::ReplayKind;

/// The prompt length the golden fixture was generated with: over the 128-token window, so the
/// prefill crosses the CED boundary.
const PROMPT_TOKENS: usize = 130;

/// The golden fixture's decode-step count. 2B-i compares the prefill position only; 2B-ii adds
/// these positions.
const DECODE_STEPS: usize = 8;

/// The last prefill position: PROMPT_TOKENS - 1.
const PREFILL_POSITION: usize = PROMPT_TOKENS - 1;

/// The golden fixture's prefill next token, checked first and token-for-token.
const PREFILL_TOKEN: u32 = 491;

/// The sliding-window size. The store's window rings are 128 slots for this reason.
const WINDOW: usize = SWA_WINDOW_SIZE;

/// One engine-format window code row: one E4M3 code per latent channel, 512 B.
const WINDOW_CODE_ROW_BYTES: usize = 512;

/// One engine-format window scale row: one E8M0 scale per 32 channels, 16 B.
const WINDOW_SCALE_ROW_BYTES: usize = 16;

/// One layer's window codes: 128 slots x 512 B.
const WINDOW_CODES_BYTES: usize = WINDOW * WINDOW_CODE_ROW_BYTES;

/// One layer's window scales: 128 slots x 16 B.
const WINDOW_SCALES_BYTES: usize = WINDOW * WINDOW_SCALE_ROW_BYTES;

const _: () = assert!(WINDOW_CODE_ROW_BYTES == mla_ref::MLA_LATENT);
const _: () = assert!(WINDOW_SCALE_ROW_BYTES == mla_ref::WINDOW_SCALE_BYTES);

/// The compressed FP4 view's code row: 256 B (512 latent packed two per byte). This is the
/// store/global-KV layout, distinct from the 512-byte engine window code row.
const COMPRESSED_CODES_BYTES: usize = CSA_UNION_COMPRESSED_CODE_ROW_BYTES;

/// The compressed FP4 view's scale row: 32 B (one E4M3 scale per 16 channels), distinct from the
/// 16-byte engine window E8M0 scale row.
const COMPRESSED_SCALES_BYTES: usize = CSA_UNION_COMPRESSED_SCALE_ROW_BYTES;

const _: () = assert!(COMPRESSED_CODES_BYTES == 256 && COMPRESSED_SCALES_BYTES == 32);

/// The rotation copy for one layer: codes plus scales.
const ROTATION_BYTES_PER_LAYER: usize = WINDOW_CODES_BYTES + WINDOW_SCALES_BYTES;

/// One full ring for every layer: 2,703,360 B. This is the store's window geometry, NOT the total a
/// schedule rotates (a token-layer execution copies only the rows its oldest-first view contains);
/// the skeleton's first accounting multiplied this number by the execution count and over-reported.
const ROTATION_BYTES_PER_STEP: usize = ROTATION_BYTES_PER_LAYER * NUM_HIDDEN_LAYERS;

/// Byte length of the released golden fixture. A fixture with the same schema but different weights
/// is not the D25 truth, so the loader pins its size and hash before any comparison.
const EXPECTED_FIXTURE_BYTES: usize = 2_536_194;
/// sha256 of the released golden fixture (D25). Pinned by `load_fixture`.
const EXPECTED_FIXTURE_SHA256: &str =
    "e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308";

/// One attention head's latent width, the q/kv row the kernel takes.
const ATTENTION_LATENT: usize = mla_ref::MLA_LATENT;

/// The released MLA head count; the format-1/2 grid is (heads, 1, 1).
const ATTENTION_HEADS: usize = mla_ref::MLA_HEADS;

/// The flash_mla block size, a static_assert in kernels/flash_mla_csa2_sm121.cu.
const ATTENTION_THREADS: u32 = 256;

/// The last position the 130-prefill + 8-decode schedule reaches (137). Compressed views and the
/// dispatch plan are sized to it; nothing past it is ever selected.
const LAST_SEQUENCE_POSITION: usize = PROMPT_TOKENS + DECODE_STEPS - 1;

/// Device-vs-CPU-twin consistency bar for the dispatch checks. This is NOT the golden fixture's
/// logits bar ([LOGITS_ABS_BAR]): the twin is the same three-pass arithmetic over the same
/// device-format inputs, so a miss here is a dispatch/layout defect, and the bar is a reporting
/// threshold for the bf16 output round, not an equality claim about the model.
const ATTENTION_TWIN_BAR: f32 = 2e-2;

const _: () = assert!(ATTENTION_HEADS == 64 && ATTENTION_LATENT == 512);
const _: () = assert!(LAST_SEQUENCE_POSITION == 137);

/// The runtime union-attention mode agreed with task-11: window_kv_format = 2, ABI unchanged at
/// 19 parameters. The kernel's internal union capacity is 640 = 512 selected + 128 window rows;
/// this harness keeps selected_count <= 512 and window_rows <= 128. The value is the kernel's
/// SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION at kernels/flash_mla_csa2_sm121.cu:130, pinned here
/// against that source so a kernel-side renumbering cannot leave the launcher on a stale format.
const CSA_UNION_WINDOW_KV_FORMAT: u32 = 2;

const _: () = assert!(CSA_UNION_WINDOW_KV_FORMAT == 2);

/// The engine's released window format (E4M3 codes + E8M0 scales), used by the window-only
/// diagnostic and by the union mode's window half.
const ENGINE_WINDOW_KV_FORMAT: u32 = 1;

/// The per-tensor element limit the boundary-dump mode builds the engine with. A real checkpoint
/// needs the checkpoint limit: the released `embed.weight` holds 661,913,600 elements, which trips
/// the 32 Mi-element reference limit that the synthetic CPU fixture uses (the !58 class bug: a
/// checkpoint mode built with the reference limit aborts before any prefill). The dump is a
/// checkpoint-capable entry point, so it must use the same limit as the other checkpoint examples.
const BOUNDARY_DUMP_TENSOR_LIMIT: usize = sparkflash_engine::linalg::CHECKPOINT_TENSOR_LIMIT;

/// The historical FP4 window format, recorded only so a report cannot silently use it.
const HISTORICAL_FP4_WINDOW_KV_FORMAT: u32 = 0;

/// Errors from argument parsing, the CPU self-check, or a future device run.
type HarnessResult<T> = Result<T, String>;

/// Which attention contract the run asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionMode {
    /// Existing kernel formats only: window rows attended, out-of-window compressed candidates
    /// structurally excluded. Diagnostic only; never an equality claim.
    WindowOnly,
    /// Window rows plus causal compressed selected ids in one softmax with the sink (task-11).
    CsaUnion,
}

impl AttentionMode {
    /// Parses the --attention-mode value.
    fn parse(text: &str) -> HarnessResult<Self> {
        match text {
            "window-only" => Ok(Self::WindowOnly),
            "csa-union" => Ok(Self::CsaUnion),
            other => Err(format!(
                "--attention-mode must be window-only or csa-union, got {other:?}"
            )),
        }
    }

    /// The window_kv_format value the device launch passes.
    fn window_kv_format(self) -> u32 {
        match self {
            Self::WindowOnly => ENGINE_WINDOW_KV_FORMAT,
            Self::CsaUnion => CSA_UNION_WINDOW_KV_FORMAT,
        }
    }

    /// The label every report line carries. The window-only string is deliberately long: a reader
    /// must not be able to quote it as an equality result.
    fn scope_label(self) -> &'static str {
        match self {
            Self::WindowOnly => {
                "window-only device column (compressed candidates structurally excluded), no equality claim"
            }
            Self::CsaUnion => {
                "CSA union device column (window rows + causal compressed candidates)"
            }
        }
    }

    /// Whether this mode may be compared token-for-token with the fixture.
    fn is_equality_claim(self) -> bool {
        matches!(self, Self::CsaUnion)
    }
}

/// Fetches the value after a flag, or names the flag in the error.
fn next_value<I: Iterator<Item = String>>(iter: &mut I, flag: &str) -> HarnessResult<String> {
    iter.next().ok_or_else(|| format!("{flag} needs a value"))
}

/// The parsed command line. Manual parsing keeps the example dependency-free and makes the exact
/// accepted argv part of the evidence log.
#[derive(Debug, Clone)]
struct Args {
    checkpoint: Option<PathBuf>,
    index: Option<PathBuf>,
    golden: Option<PathBuf>,
    cubin_dir: Option<PathBuf>,
    out: Option<PathBuf>,
    boundary_dump: Option<PathBuf>,
    provenance: String,
    max_seq_len: usize,
    attention_mode: AttentionMode,
    no_ced: bool,
    prefill_only: bool,
    device: bool,
    dry_run: bool,
    self_check: bool,
    prompt_tokens: usize,
    decode_steps: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            checkpoint: None,
            index: None,
            golden: None,
            cubin_dir: None,
            out: None,
            boundary_dump: None,
            provenance: "devhost".to_string(),
            max_seq_len: 192,
            attention_mode: AttentionMode::CsaUnion,
            no_ced: false,
            prefill_only: false,
            device: false,
            dry_run: false,
            self_check: true,
            prompt_tokens: PROMPT_TOKENS,
            decode_steps: DECODE_STEPS,
        }
    }
}

impl Args {
    /// Parses std::env::args() (program name skipped).
    fn parse_from<I: IntoIterator<Item = String>>(argv: I) -> HarnessResult<Self> {
        let mut args = Self::default();
        let mut iter = argv.into_iter();
        while let Some(flag) = iter.next() {
            match flag.as_str() {
                "--checkpoint" => {
                    args.checkpoint = Some(PathBuf::from(next_value(&mut iter, "--checkpoint")?))
                }
                "--index" => args.index = Some(PathBuf::from(next_value(&mut iter, "--index")?)),
                "--golden" => args.golden = Some(PathBuf::from(next_value(&mut iter, "--golden")?)),
                "--cubin-dir" => {
                    args.cubin_dir = Some(PathBuf::from(next_value(&mut iter, "--cubin-dir")?))
                }
                "--out" => args.out = Some(PathBuf::from(next_value(&mut iter, "--out")?)),
                "--boundary-dump" => {
                    args.boundary_dump =
                        Some(PathBuf::from(next_value(&mut iter, "--boundary-dump")?))
                }
                "--provenance" => args.provenance = next_value(&mut iter, "--provenance")?,
                "--attention-mode" => {
                    args.attention_mode =
                        AttentionMode::parse(&next_value(&mut iter, "--attention-mode")?)?
                }
                "--max-seq-len" => {
                    let raw = next_value(&mut iter, "--max-seq-len")?;
                    args.max_seq_len = raw
                        .parse()
                        .map_err(|_| "--max-seq-len must be a positive integer".to_string())?;
                    if args.max_seq_len == 0 {
                        return Err("--max-seq-len must be positive".to_string());
                    }
                }
                "--prompt-tokens" => {
                    let raw = next_value(&mut iter, "--prompt-tokens")?;
                    args.prompt_tokens = raw
                        .parse()
                        .map_err(|_| "--prompt-tokens must be a positive integer".to_string())?;
                }
                "--decode-steps" => {
                    let raw = next_value(&mut iter, "--decode-steps")?;
                    args.decode_steps = raw
                        .parse()
                        .map_err(|_| "--decode-steps must be an integer".to_string())?;
                }
                "--device" => args.device = true,
                "--dry-run" => args.dry_run = true,
                "--self-check" => args.self_check = true,
                "--no-self-check" => args.self_check = false,
                "--no-ced" => args.no_ced = true,
                "--prefill-only" => args.prefill_only = true,
                "--help" | "-h" => return Err(Self::usage()),
                other => return Err(format!("unknown argument {other:?}\n\n{}", Self::usage())),
            }
        }
        if args.prompt_tokens != PROMPT_TOKENS {
            return Err(format!(
                "the golden fixture is for {PROMPT_TOKENS} prompt tokens; this harness pins that count"
            ));
        }
        if args.decode_steps != DECODE_STEPS {
            return Err(format!(
                "the golden fixture is for {DECODE_STEPS} decode steps; 2B-i/2B-ii pin that count"
            ));
        }
        if args.dry_run && args.device {
            return Err("--dry-run is host-only and cannot be combined with --device".to_string());
        }
        if args.dry_run && args.golden.is_none() {
            return Err(
                "--dry-run needs --golden: the comparison plumbing is exercised against it"
                    .to_string(),
            );
        }
        Ok(args)
    }

    /// The exact command line this file accepts, printed by --help and quoted in the run log.
    fn usage() -> String {
        [
            "usage: m8_sequence [options]",
            "",
            "  --golden PATH         2A golden fixture (SequenceFixture JSON)",
            "  --checkpoint DIR      real checkpoint snapshot (device runs)",
            "  --index PATH          safetensors index inside the checkpoint",
            "  --cubin-dir DIR       directory holding the built .cubin modules",
            "  --out PATH            write the report.txt here (dry-run and device modes)",
            "  --boundary-dump DIR   write the engine reference boundary v1 manifest+blob here (CPU only)",
            "  --provenance TEXT     free-form provenance label (default: devhost)",
            "  --attention-mode M    window-only | csa-union (default: csa-union)",
            "  --no-ced              control: layers 21..=39 over all prompt tokens",
            "  --prefill-only        2B-i: run/compare the prefill position only",
            "  --max-seq-len N       store geometry bound (page-aligned; default 192)",
            "  --dry-run             host only: build the dispatch/rotation plan and exercise the",
            "                        token-then-logits comparison plumbing; writes the report",
            "  --device              run the wired device half on the sequence geometry (requires a",
            "                        Lead-sequenced window; NOT the model forward yet - see the report)",
            "  --self-check          run the CPU checks (default on; the unit tests run in CI)",
            "  --no-self-check       skip the CPU checks",
        ]
        .join("\n")
    }
}

/// The 40-row mode table, written out rather than derived, so a changed assignment cannot pass by
/// agreeing with itself. The kv crate's pin test and the engine plan are checked against this list.
const EXPECTED_MODES: [Csa2Mode; NUM_HIDDEN_LAYERS] = [
    Csa2Mode::SwaOnly,
    Csa2Mode::SwaOnly,
    Csa2Mode::Full,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Full,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Full,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Full,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reindex,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reindex,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reindex,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reindex,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
    Csa2Mode::Reuse,
];

/// The publication order the device path must obey: main KV, index-K and Top-K at each Full layer,
/// the HSI pool right after layer 20's Top-K, and a republished Top-K at each Reindex layer.
const EXPECTED_PUBLICATIONS: [(usize, PublishedHandle); 17] = [
    (2, PublishedHandle::MainKv),
    (2, PublishedHandle::IndexerK),
    (2, PublishedHandle::TopK),
    (8, PublishedHandle::MainKv),
    (8, PublishedHandle::IndexerK),
    (8, PublishedHandle::TopK),
    (14, PublishedHandle::MainKv),
    (14, PublishedHandle::IndexerK),
    (14, PublishedHandle::TopK),
    (20, PublishedHandle::MainKv),
    (20, PublishedHandle::IndexerK),
    (20, PublishedHandle::TopK),
    (20, PublishedHandle::HsiPool),
    (24, PublishedHandle::TopK),
    (28, PublishedHandle::TopK),
    (32, PublishedHandle::TopK),
    (36, PublishedHandle::TopK),
];

/// The CPU-checked description of the schedule this run will execute.
#[derive(Debug, Clone)]
struct ScheduleCheck {
    prompt_tokens: usize,
    window: usize,
    encoder_tokens: usize,
    decoder_tokens: usize,
    exclusion_boundary: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    publications: Vec<Publication>,
    modes: Vec<Csa2Mode>,
    plan: sparkflash_kv::PrefixMatchPlan,
}

/// Builds the schedule and checks every value the device path depends on. no_ced changes only the
/// per-layer token extent (it is the control run), never the static modes or publication order.
fn checked_schedule(prompt_tokens: usize, no_ced: bool) -> HarnessResult<ScheduleCheck> {
    let schedule = ced_schedule(prompt_tokens)
        .map_err(|error| format!("ced_schedule({prompt_tokens}): {error}"))?;
    let expected_plan = sparkflash_kv::PrefixMatchPlan {
        total_prompt_tokens: prompt_tokens,
        global_kv_cached_tokens: 0,
        uma_hit_tokens: 0,
        nvme_restored_tokens: 0,
        swa_tail_hit: false,
        encoder_swa_replay_tokens: 0,
        uncached_suffix_tokens: prompt_tokens,
        decoder_swa_replay_tokens: prompt_tokens.min(WINDOW),
    };
    if schedule.plan != expected_plan {
        return Err(format!(
            "cold-prompt plan drifted: got {:?}, expected {expected_plan:?}",
            schedule.plan
        ));
    }
    let modes: Vec<Csa2Mode> = (0..NUM_HIDDEN_LAYERS).map(csa2_mode).collect();
    if modes.as_slice() != EXPECTED_MODES.as_slice() {
        return Err("the 40-row mode table drifted from the pinned list".to_string());
    }
    let expected_publications: Vec<Publication> = EXPECTED_PUBLICATIONS
        .iter()
        .map(|(layer, handle)| Publication {
            layer: *layer,
            handle: *handle,
        })
        .collect();
    if schedule.publications != expected_publications {
        return Err(format!(
            "publication order drifted: got {:?}",
            schedule.publications
        ));
    }
    // The kv crate's modes and the engine's layer plan are two derivations of one config; the
    // engine plan is what the host reference executes, so the device harness pins they agree.
    let config = ModelConfig::v41_flash().map_err(|error| format!("released config: {error}"))?;
    let engine_plan =
        LayerPlan::from_config(&config).map_err(|error| format!("engine layer plan: {error}"))?;
    for (layer, mode) in modes.iter().enumerate() {
        let spec = engine_plan
            .get(layer)
            .ok_or_else(|| format!("engine plan has no layer {layer}"))?;
        let want = match mode {
            Csa2Mode::SwaOnly => LayerMode::SwaOnly,
            Csa2Mode::Full => LayerMode::Full,
            Csa2Mode::Reindex => LayerMode::Reindex,
            Csa2Mode::Reuse => LayerMode::Reuse,
        };
        if spec.mode != want {
            return Err(format!(
                "layer {layer}: engine plan mode {:?} disagrees with the kv crate's {want:?}",
                spec.mode
            ));
        }
    }
    for row in &schedule.layers {
        if row.mode == Csa2Mode::Full && !KV_SOURCE_LAYERS.contains(&row.layer) {
            return Err(format!(
                "layer {} is Full but not in the released KV source list",
                row.layer
            ));
        }
        if row.owns_index && !INDEX_SOURCE_LAYERS.contains(&row.layer) {
            return Err(format!(
                "layer {} publishes Top-K but is not in the released index source list",
                row.layer
            ));
        }
        if row.layer == CANDIDATE_SOURCE_LAYER
            && row.candidate_source != Some(CANDIDATE_SOURCE_LAYER)
        {
            return Err("layer 20 must be the HSI candidate source".to_string());
        }
    }
    let decoder_tokens = if no_ced {
        prompt_tokens
    } else {
        schedule.decoder_tokens
    };
    if decoder_tokens > WINDOW && !no_ced {
        return Err("the CED decoder extent exceeds the window".to_string());
    }
    if schedule.encoder_tokens != prompt_tokens {
        return Err("the encoder half must run over the whole prompt".to_string());
    }
    let mut encoder_layers = 0usize;
    let mut decoder_layers = 0usize;
    for layer in 0..NUM_HIDDEN_LAYERS {
        if layer <= ENCODER_LAST_LAYER {
            encoder_layers += 1;
        } else {
            debug_assert!(layer >= DECODER_FIRST_LAYER);
            decoder_layers += 1;
        }
    }
    Ok(ScheduleCheck {
        prompt_tokens,
        window: WINDOW,
        encoder_tokens: prompt_tokens,
        decoder_tokens,
        exclusion_boundary: exclusion_boundary(prompt_tokens, WINDOW),
        encoder_layers,
        decoder_layers,
        publications: schedule.publications.clone(),
        modes,
        plan: schedule.plan,
    })
}

/// The absolute positions one layer consumes in the device path. Prefill position p is the prompt
/// token at p; decode positions follow the prefill.
fn layer_token_extent(layer: usize, prompt_tokens: usize, decoder_tokens: usize) -> Vec<usize> {
    if layer <= ENCODER_LAST_LAYER {
        (0..prompt_tokens).collect()
    } else {
        (prompt_tokens - decoder_tokens..prompt_tokens).collect()
    }
}

/// The engine's sentinel for a candidate that is not attended. It is private in the engine's
/// attention module (attention.rs NO_INDEX), so the harness restates the value it compares
/// against rather than importing a private constant. Only the test-side union cross-check uses it
/// now: the live dispatch takes its ids from harness_selected_ids and never sees a NO_INDEX value.
#[cfg(test)]
const ENGINE_NO_INDEX: usize = usize::MAX;

/// One KV value the attention softmax consumes. Window rows and compressed rows are different KV
/// values even when they describe the same token, which is why the reference concatenates
/// window_indices and compressed_ids instead of merging them.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum AttendedValue {
    /// A window slot, identified by the prompt position it holds.
    WindowToken(usize),
    /// A compressed-KV row, identified by its store row index (position / ratio).
    CompressedRow(usize),
}

/// A multiset of attended KV values: order is not meaningful for the union, duplicates are.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AttendedUnion {
    values: Vec<AttendedValue>,
}

#[cfg(test)]
impl AttendedUnion {
    fn push(&mut self, value: AttendedValue) {
        self.values.push(value);
    }

    /// The sorted value list, the canonical form for multiset equality.
    fn sorted(&self) -> Vec<AttendedValue> {
        let mut values = self.values.clone();
        values.sort_unstable();
        values
    }

    fn contains(&self, value: AttendedValue) -> bool {
        self.values.contains(&value)
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// The per-position expanded-view row a compressed store row is copied into for a ratio-2 source:
/// row c becomes row c*ratio. This is the approved injective compressed-row -> absolute-position
/// mapping for option A; ratio 1 is the identity.
#[cfg(test)]
#[must_use]
fn compressed_row_to_absolute_position(ratio: usize, row: usize) -> usize {
    assert!(ratio > 0, "a compressed source has a positive ratio");
    row * ratio
}

/// The store row an expanded absolute position reads from: position / ratio. Only positions
/// produced by compressed_row_to_absolute_position are selected, so the mapping is injective on
/// the rows actually used even though the expanded view repeats a row for ratio 2.
#[cfg(test)]
#[must_use]
fn expanded_store_row(ratio: usize, absolute_position: usize) -> usize {
    assert!(ratio > 0, "a compressed source has a positive ratio");
    absolute_position / ratio
}

/// The format-2 kernel's bounds, mirrored from kernels/flash_mla_csa2_sm121.cu:
/// 512 selected + 128 window rows, and the launcher must never exceed them.
const FORMAT2_MAX_SELECTED: usize = 512;
const FORMAT2_MAX_WINDOW_ROWS: usize = 128;
const FORMAT2_MAX_ATTENDED: usize = FORMAT2_MAX_SELECTED + FORMAT2_MAX_WINDOW_ROWS;

const _: () = assert!(FORMAT2_MAX_ATTENDED == 640);

/// What the format-2 launch passes for one query: the selected absolute ids the engine filtered, the
/// window row count, and the union size. This is the skeleton-era builder, kept under cfg(test) as
/// the cross-check of the engine-id mapping; the live dispatch type is AttentionDispatch and its
/// builder is dispatch_at_position.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Format2Selection {
    /// Absolute token positions c*ratio, in the engine's supplied order.
    selected: Vec<u32>,
    /// Window rows olders-first, at most 128.
    window_rows: u32,
    /// selected.len(), at most 512.
    selected_count: u32,
}

#[cfg(test)]
impl Format2Selection {
    /// window_rows + selected_count, the kernel's union capacity use; at most 640.
    fn union_rows(&self) -> usize {
        self.window_rows as usize + self.selected.len()
    }
}

/// Builds the format-2 launch selection from the engine's compressed ids (shared index space:
/// window_rows + compressed row) and the query position. NO_INDEX entries are dropped by the engine
/// rule; every remaining id must already satisfy row < (position + 1)/ratio, i.e. the engine's
/// reachability filter. A caller that has not applied that filter is refused rather than silently
/// repaired, because the kernel's id <= position guard is not the ratio-2 reach rule.
#[cfg(test)]
fn format2_selection_from_engine_ids(
    engine_compressed_ids: &[usize],
    engine_window_rows: usize,
    kernel_window_rows: usize,
    ratio: usize,
    position: usize,
) -> HarnessResult<Format2Selection> {
    if kernel_window_rows > FORMAT2_MAX_WINDOW_ROWS {
        return Err(format!(
            "format 2 kernel window_rows {kernel_window_rows} exceeds the kernel bound {FORMAT2_MAX_WINDOW_ROWS}"
        ));
    }
    if ratio == 0 {
        return Err("format 2 needs a positive compression ratio".to_string());
    }
    let mut selected = Vec::with_capacity(engine_compressed_ids.len());
    for id in engine_compressed_ids {
        if *id == ENGINE_NO_INDEX {
            continue;
        }
        if *id < engine_window_rows {
            return Err(format!(
                "engine id {id} is in window space (below {engine_window_rows}); format 2 selected entries are compressed-row-mapped absolute ids"
            ));
        }
        let row = id - engine_window_rows;
        let reach = (position + 1) / ratio;
        if row >= reach {
            return Err(format!(
                "compressed row {row} is at or past reach {reach} for query position {position}: the engine reachability filter was not applied (row must be < (position + 1)/{ratio})"
            ));
        }
        let absolute = row
            .checked_mul(ratio)
            .ok_or_else(|| format!("compressed row {row} overflows its ratio-{ratio} expansion"))?;
        selected.push(
            u32::try_from(absolute).map_err(|_| {
                format!("compressed row {row} expands past the kernel's u32 id space")
            })?,
        );
    }
    if selected.len() > FORMAT2_MAX_SELECTED {
        return Err(format!(
            "format 2 selected_count {} exceeds the kernel bound {FORMAT2_MAX_SELECTED}",
            selected.len()
        ));
    }
    let union_rows = kernel_window_rows + selected.len();
    if union_rows > FORMAT2_MAX_ATTENDED {
        return Err(format!(
            "format 2 union of {union_rows} entries exceeds the kernel bound {FORMAT2_MAX_ATTENDED}"
        ));
    }
    let selected_count = u32::try_from(selected.len())
        .map_err(|_| "selected_count does not fit a u32".to_string())?;
    Ok(Format2Selection {
        selected,
        window_rows: u32::try_from(kernel_window_rows)
            .map_err(|_| "window_rows does not fit a u32".to_string())?,
        selected_count,
    })
}

/// A per-position expanded view of a compressed store for a ratio source: row p holds store row
/// p/ratio. Ratio 1 is the identity and returns the input unchanged. The view is what lets format 2
/// index by absolute token position c*ratio without a ratio argument in the kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpandedView {
    /// Position rows of 256 compressed FP4 bytes (512 latent packed two per byte).
    codes: Vec<u8>,
    /// Position rows of 32 E4M3 scales (one per 16 channels).
    scales: Vec<u8>,
    /// Position rows, store_rows * ratio.
    rows: usize,
}

impl ExpandedView {
    fn codes_bytes(&self) -> usize {
        self.codes.len()
    }

    fn scales_bytes(&self) -> usize {
        self.scales.len()
    }
}

/// Builds the ratio expansion on the host from the compressed store arrays. Both the even and odd
/// position of a ratio-2 pair read the same store row; only the even positions are ever passed as
/// selected ids (c*ratio), so the expansion is injective on the rows actually read.
fn expand_compressed_view(
    codes: &[u8],
    scales: &[u8],
    store_rows: usize,
    ratio: usize,
) -> HarnessResult<ExpandedView> {
    if ratio == 0 {
        return Err("a compressed source has a positive ratio".to_string());
    }
    if codes.len() != store_rows * COMPRESSED_CODES_BYTES {
        return Err(format!(
            "compressed codes are {} B, expected {} rows x {} B",
            codes.len(),
            store_rows,
            COMPRESSED_CODES_BYTES
        ));
    }
    if scales.len() != store_rows * COMPRESSED_SCALES_BYTES {
        return Err(format!(
            "compressed scales are {} B, expected {} rows x {} B",
            scales.len(),
            store_rows,
            COMPRESSED_SCALES_BYTES
        ));
    }
    let rows = store_rows
        .checked_mul(ratio)
        .ok_or_else(|| "expanded row count overflow".to_string())?;
    let mut expanded = ExpandedView {
        codes: vec![0u8; rows * COMPRESSED_CODES_BYTES],
        scales: vec![0u8; rows * COMPRESSED_SCALES_BYTES],
        rows,
    };
    for position in 0..rows {
        let store_row = position / ratio;
        expanded.codes[position * COMPRESSED_CODES_BYTES..(position + 1) * COMPRESSED_CODES_BYTES]
            .copy_from_slice(
                &codes
                    [store_row * COMPRESSED_CODES_BYTES..(store_row + 1) * COMPRESSED_CODES_BYTES],
            );
        expanded.scales
            [position * COMPRESSED_SCALES_BYTES..(position + 1) * COMPRESSED_SCALES_BYTES]
            .copy_from_slice(
                &scales[store_row * COMPRESSED_SCALES_BYTES
                    ..(store_row + 1) * COMPRESSED_SCALES_BYTES],
            );
    }
    Ok(expanded)
}

/// Turns the engine's compressed ids (shared index space: a compressed row is window_rows + row)
/// into the format-2 selected list (absolute token positions c*ratio). The selected values are
/// compressed row ids, not a token-position list: this function rejects an entry that is still in
/// window space (id < window_rows) because that is not a compressed-row-mapped id at all. The
/// engine has already replaced an unreachable row with NO_INDEX, and that filtering is load-bearing
/// for ratio 2: the kernel's id <= position guard alone is weaker than the reference's reach rule
/// (see the causal-edge test). Window tokens and compressed rows share positions; the reference
/// counts them as distinct entries, so this function must not merge them.
#[cfg(test)]
fn kernel_selected_from_engine_compressed_ids(
    engine_ids: &[usize],
    window_rows: usize,
    ratio: usize,
) -> HarnessResult<Vec<u32>> {
    let mut selected = Vec::with_capacity(engine_ids.len());
    for id in engine_ids {
        if *id == ENGINE_NO_INDEX {
            continue;
        }
        if *id < window_rows {
            return Err(format!(
                "engine compressed id {id} is below the window offset {window_rows}: the caller passed a window id in the compressed list"
            ));
        }
        let row = id - window_rows;
        let position = compressed_row_to_absolute_position(ratio, row);
        let position = u32::try_from(position)
            .map_err(|_| format!("compressed row {row} expands past a u32 position"))?;
        selected.push(position);
    }
    Ok(selected)
}

/// The reference's attended multiset for one query: window ids first (chunk-relative positions in
/// prefill), then the engine's compressed ids shifted into the shared index space. NO_INDEX values
/// are dropped exactly as sparse_attention drops indices outside kv.
#[cfg(test)]
fn engine_attended_union(
    window_ids: &[usize],
    compressed_shared_ids: &[usize],
    window_rows: usize,
) -> HarnessResult<AttendedUnion> {
    let mut union = AttendedUnion::default();
    for id in window_ids {
        if *id == ENGINE_NO_INDEX {
            continue;
        }
        if *id >= window_rows {
            return Err(format!(
                "engine window id {id} is not below the window width {window_rows}"
            ));
        }
        union.push(AttendedValue::WindowToken(*id));
    }
    for id in compressed_shared_ids {
        if *id == ENGINE_NO_INDEX {
            continue;
        }
        if *id < window_rows {
            return Err(format!(
                "engine compressed id {id} is below the window offset {window_rows}"
            ));
        }
        union.push(AttendedValue::CompressedRow(id - window_rows));
    }
    Ok(union)
}

/// The format-2 kernel's attended multiset for the same query: every supplied window row (oldest
/// first, no predicate), then every selected id with id <= position, decoded from the expanded
/// per-position view as store row id / ratio.
#[cfg(test)]
fn format2_attended_union(
    window_positions: &[usize],
    selected_absolute_positions: &[u32],
    position: u32,
    ratio: usize,
    store_rows: usize,
) -> HarnessResult<AttendedUnion> {
    let mut union = AttendedUnion::default();
    for token in window_positions {
        union.push(AttendedValue::WindowToken(*token));
    }
    for id in selected_absolute_positions {
        if *id > position {
            continue;
        }
        let row = expanded_store_row(ratio, *id as usize);
        if row >= store_rows {
            return Err(format!(
                "selected id {id} expands to store row {row}, outside the {store_rows} stored rows"
            ));
        }
        union.push(AttendedValue::CompressedRow(row));
    }
    Ok(union)
}

/// The absolute positions one layer executes in the whole 130-prefill + 8-decode schedule: its CED
/// extent in the prefill, then every decode position. The 8 decode tokens run through all forty
/// layers; the CED split applies only to the prefill.
fn execution_positions(layer: usize, schedule: &ScheduleCheck) -> Vec<usize> {
    let mut positions = layer_token_extent(layer, schedule.encoder_tokens, schedule.decoder_tokens);
    positions.extend(PROMPT_TOKENS..PROMPT_TOKENS + DECODE_STEPS);
    positions
}

/// The absolute position where a layer's window ring starts filling: 0 for encoder layers and the
/// CED boundary for decoder layers. It is the anchor the oldest-first view must use; using 0 for a
/// decoder layer rotates slots that layer never wrote.
fn layer_first_position(layer: usize, schedule: &ScheduleCheck) -> usize {
    layer_token_extent(layer, schedule.encoder_tokens, schedule.decoder_tokens)[0]
}

/// One execution's oldest-first view: `(window_rows, oldest_slot)`, anchored at the layer's first
/// executed position. Both the plan and the device copy closure call this, so the byte arithmetic
/// and the copy cannot disagree.
fn window_view(
    layout: &WindowRingLayout,
    schedule: &ScheduleCheck,
    layer: usize,
    position: usize,
) -> HarnessResult<(usize, usize)> {
    let first = layer_first_position(layer, schedule);
    ring_view_position(layout, first, position)
        .map_err(|error| format!("window view layer {layer} position {position}: {error}"))
}

/// The exact byte arithmetic of the ring rotation. The reported byte count is derived from the rows
/// each execution actually copies, `min(position - first_position + 1, WINDOW)` slots, not from a
/// full-ring volume times an execution count: the skeleton multiplied the forty-ring volume
/// (2,703,360 B) by the execution count, charging all forty layers to every execution, so this type
/// replaced it. The critic's second round then showed the rows must be anchored at each layer's
/// first executed position: a decoder layer starts at position 2, so `min(position + 1, WINDOW)`
/// over-counted 253 slots per decoder layer (4,807 slots, 2,538,096 B). A test pins the sum, and
/// the device path binds the same plan to `cuda_harness::rotate_window_ring`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RotationPlan {
    layers: usize,
    exec_prefill: usize,
    exec_decode: usize,
    slots_copied: usize,
    slots_per_ring: usize,
}

impl RotationPlan {
    /// The plan for the full schedule: one rotation per token-layer execution, copying only the rows
    /// that execution's oldest-first view contains.
    fn for_schedule(schedule: &ScheduleCheck) -> Self {
        let mut slots_copied = 0usize;
        for layer in 0..NUM_HIDDEN_LAYERS {
            for position in execution_positions(layer, schedule) {
                let (rows, _) = window_view(&WindowRingLayout::ENGINE, schedule, layer, position)
                    .expect("executed positions always have a view");
                slots_copied += rows;
            }
        }
        let exec_prefill: usize = (0..NUM_HIDDEN_LAYERS)
            .map(|layer| {
                layer_token_extent(layer, schedule.encoder_tokens, schedule.decoder_tokens).len()
            })
            .sum();
        let exec_decode = DECODE_STEPS * NUM_HIDDEN_LAYERS;
        Self {
            layers: NUM_HIDDEN_LAYERS,
            exec_prefill,
            exec_decode,
            slots_copied,
            slots_per_ring: WINDOW,
        }
    }

    fn executions(&self) -> usize {
        self.exec_prefill + self.exec_decode
    }

    /// Bytes one full-ring layer rotation would move: 128 slots x 528 B = 67,584 B.
    fn bytes_per_layer(&self) -> usize {
        self.slots_per_ring * (WINDOW_CODE_ROW_BYTES + WINDOW_SCALE_ROW_BYTES)
    }

    /// Bytes one execution copies: the rows in its oldest-first view, never more than a full ring.
    fn bytes_per_execution_max(&self) -> usize {
        self.bytes_per_layer()
    }

    fn codes_bytes(&self) -> usize {
        self.slots_copied * WINDOW_CODE_ROW_BYTES
    }

    fn scales_bytes(&self) -> usize {
        self.slots_copied * WINDOW_SCALE_ROW_BYTES
    }

    fn bytes(&self) -> usize {
        self.codes_bytes() + self.scales_bytes()
    }

    /// The total volume of all forty rings once; the store's window geometry is this number.
    fn ring_bytes(&self) -> usize {
        self.layers * self.bytes_per_layer()
    }

    /// The most device-to-device copies a run can issue: two arrays x at most two wrap segments per
    /// execution.
    fn copy_calls_upper_bound(&self) -> usize {
        self.executions() * 4
    }

    /// The plan line the self-check and reports carry.
    fn line(&self) -> String {
        format!(
            "rotation plan: {} token-layer executions (prefill {}, decode {}), {} slots copied x {} B = {} B total (codes {} B + scales {} B), one full ring {} B, all rings once {} B, <= {} copy_dtod calls",
            self.executions(),
            self.exec_prefill,
            self.exec_decode,
            self.slots_copied,
            WINDOW_CODE_ROW_BYTES + WINDOW_SCALE_ROW_BYTES,
            self.bytes(),
            self.codes_bytes(),
            self.scales_bytes(),
            self.bytes_per_layer(),
            self.ring_bytes(),
            self.copy_calls_upper_bound()
        )
    }
}

/// The measured result of the rotation copies. Both clocks are measured, never assumed;
/// `bytes_moved` is the sum the copy closures reported, so a measurement cannot report bytes it did
/// not move. `copy_elapsed` is the sum of the durations the closures attribute to their copy calls
/// alone; `wall_elapsed` is the whole loop, which also verifies views and dispatches attention and
/// therefore must not be divided into the byte count.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RotationMeasurement {
    plan: RotationPlan,
    executions: usize,
    copies: usize,
    bytes_moved: usize,
    copy_elapsed: Duration,
    wall_elapsed: Duration,
}

impl RotationMeasurement {
    fn bytes_per_second(self) -> f64 {
        let seconds = self.copy_elapsed.as_secs_f64();
        if seconds == 0.0 {
            0.0
        } else {
            self.bytes_moved as f64 / seconds
        }
    }

    /// One line a report can carry without a reader needing the byte arithmetic.
    fn line(self) -> String {
        format!(
            "rotation: {} executions, {} copy_dtod calls, {} B moved (plan {} B) in {:.6} s of copy time ({:.3} MiB/s); copy time is the sum of the rotate_window_ring call durations only, the loop wall {:.6} s also covers view verification and attention dispatch",
            self.executions,
            self.copies,
            self.bytes_moved,
            self.plan.bytes(),
            self.copy_elapsed.as_secs_f64(),
            self.bytes_per_second() / (1024.0 * 1024.0),
            self.wall_elapsed.as_secs_f64()
        )
    }
}

/// Measures the rotation copies with a caller-supplied device-to-device copy. The closure performs
/// one (layer, position) rotation and returns its copy-call count, the bytes it moved, and the
/// duration of its copy calls alone; this function owns the accounting and the exclusion rule, so a
/// dispatch or a read-back that happens between copies cannot land in the copy rate. A closure whose
/// totals disagree with the plan is refused rather than reported.
fn measure_rotation<F>(
    schedule: &ScheduleCheck,
    plan: RotationPlan,
    mut copy_one: F,
) -> HarnessResult<RotationMeasurement>
where
    F: FnMut(usize, usize) -> HarnessResult<(usize, usize, Duration)>,
{
    let start = Instant::now();
    let mut executions = 0usize;
    let mut copies = 0usize;
    let mut bytes_moved = 0usize;
    let mut copy_elapsed = Duration::ZERO;
    for layer in 0..NUM_HIDDEN_LAYERS {
        for position in execution_positions(layer, schedule) {
            let (this_copies, this_bytes, this_copy_time) =
                copy_one(layer, position).map_err(|error| {
                    format!("rotation copy layer {layer} position {position}: {error}")
                })?;
            copies = copies
                .checked_add(this_copies)
                .ok_or_else(|| "rotation copy count overflow".to_string())?;
            bytes_moved = bytes_moved
                .checked_add(this_bytes)
                .ok_or_else(|| "rotation byte count overflow".to_string())?;
            copy_elapsed += this_copy_time;
            executions += 1;
        }
    }
    let wall_elapsed = start.elapsed();
    if executions != plan.executions() {
        return Err(format!(
            "rotation ran {executions} executions, the plan says {}",
            plan.executions()
        ));
    }
    if bytes_moved != plan.bytes() {
        return Err(format!(
            "rotation moved {bytes_moved} B, the plan says {} B",
            plan.bytes()
        ));
    }
    Ok(RotationMeasurement {
        plan,
        executions,
        copies,
        bytes_moved,
        copy_elapsed,
        wall_elapsed,
    })
}

/// Which attention contract one layer uses in the sequence dispatch. This is a property of the
/// checkpoint's topology (compress_ratio and the kv source layer), not of the run mode: the run mode
/// only decides whether the union layers are replaced by the window-only diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerAttentionKind {
    /// Format 1 over the engine-format window rows: the SWA-only layers (compress_ratio 0).
    WindowOnly,
    /// Format 2 over this layer's source store: window rows plus the reachable compressed rows.
    CsaUnion { store_layer: usize, ratio: usize },
}

/// The attention contract of one layer under the checkpoint's topology.
fn layer_attention_kind(layer: usize) -> HarnessResult<LayerAttentionKind> {
    match csa2_mode(layer) {
        Csa2Mode::SwaOnly => Ok(LayerAttentionKind::WindowOnly),
        Csa2Mode::Full | Csa2Mode::Reuse | Csa2Mode::Reindex => {
            let store_layer = if csa2_mode(layer).owns_main_kv() {
                layer
            } else {
                kv_source_for(layer)
            };
            let ratio = compress_ratio(store_layer);
            if ratio == 0 {
                return Err(format!(
                    "layer {layer} is a union layer but its store layer {store_layer} has ratio 0"
                ));
            }
            Ok(LayerAttentionKind::CsaUnion { store_layer, ratio })
        }
    }
}

/// The harness-supplied reachability seam for the CSA-union dispatch: the compressed rows a query at
/// `position` may attend, in ascending row order, mapped to absolute token ids `c * ratio`.
///
/// **Host seam.** The released engine fills this list from its HSI pool (the indexer's top-2048
/// blocks) plus the top-512 merge; neither the pool population nor the merge is a device kernel in
/// Phase 2, and the harness supplies the reachable rows directly. The rule implemented here is the
/// engine's reach mask, `reach = (position + 1) / ratio`, which is the rule the negative-control
/// test shows the kernel's `id <= position` guard does NOT reproduce for ratio 2. A top-512 HSI
/// selection would pass a subset (and possibly a permutation) of these ids; the union arithmetic is
/// order-sensitive only by float addition order.
fn harness_selected_ids(position: usize, ratio: usize) -> HarnessResult<Vec<u32>> {
    if ratio == 0 {
        return Err("a union dispatch needs a positive compression ratio".to_string());
    }
    let reach = (position + 1) / ratio;
    if reach > FORMAT2_MAX_SELECTED {
        return Err(format!(
            "query {position} reaches {reach} compressed rows, above the kernel bound {FORMAT2_MAX_SELECTED}"
        ));
    }
    let mut ids = Vec::with_capacity(reach);
    for row in 0..reach {
        let absolute = row
            .checked_mul(ratio)
            .ok_or_else(|| format!("compressed row {row} overflows its ratio-{ratio} expansion"))?;
        if absolute > position {
            // The kernel's causal guard would exclude it; supplying it would hide a reach bug.
            return Err(format!(
                "compressed row {row} expands to id {absolute}, past query position {position}"
            ));
        }
        ids.push(
            u32::try_from(absolute)
                .map_err(|_| format!("compressed row {row} does not fit a u32 id"))?,
        );
    }
    Ok(ids)
}

/// One attention dispatch at one position of the sequence: the layer, its format, the window row
/// count and the selected list the launch passes. The device path executes one of these per
/// (compared position, layer).
#[derive(Debug, Clone, PartialEq, Eq)]
struct AttentionDispatch {
    layer: usize,
    position: usize,
    window_rows: usize,
    window_kv_format: u32,
    selected: Vec<u32>,
    /// The store layer a union dispatch reads (None for the window-only format).
    store_layer: Option<usize>,
    /// The store's compression ratio (1 for window-only, which has no compressed store).
    ratio: usize,
}

impl AttentionDispatch {
    fn selected_count(&self) -> usize {
        self.selected.len()
    }

    fn union_rows(&self) -> usize {
        self.window_rows + self.selected.len()
    }
}

/// Builds the dispatch of every one of the forty layers at one query position, under the run mode.
fn dispatch_at_position(
    position: usize,
    attention_mode: AttentionMode,
) -> HarnessResult<Vec<AttentionDispatch>> {
    let window_rows = (position + 1).min(WINDOW);
    let mut dispatches = Vec::with_capacity(NUM_HIDDEN_LAYERS);
    for layer in 0..NUM_HIDDEN_LAYERS {
        let kind = layer_attention_kind(layer)?;
        let union = match (attention_mode, kind) {
            (AttentionMode::CsaUnion, LayerAttentionKind::CsaUnion { store_layer, ratio }) => {
                Some((store_layer, ratio))
            }
            _ => None,
        };
        let (window_kv_format, selected, store_layer, ratio) = match union {
            Some((store_layer, ratio)) => (
                CSA_UNION_WINDOW_KV_FORMAT,
                harness_selected_ids(position, ratio)?,
                Some(store_layer),
                ratio,
            ),
            None => (
                ENGINE_WINDOW_KV_FORMAT,
                (0..window_rows)
                    .map(|row| {
                        u32::try_from(row).map_err(|_| "window row does not fit a u32".to_string())
                    })
                    .collect::<HarnessResult<Vec<u32>>>()?,
                None,
                1,
            ),
        };
        let dispatch = AttentionDispatch {
            layer,
            position,
            window_rows,
            window_kv_format,
            selected,
            store_layer,
            ratio,
        };
        check_dispatch(&dispatch)?;
        dispatches.push(dispatch);
    }
    Ok(dispatches)
}

/// Refuses a dispatch the kernel's bounds or the engine's reach rule would not admit.
fn check_dispatch(dispatch: &AttentionDispatch) -> HarnessResult<()> {
    if dispatch.window_rows == 0 || dispatch.window_rows > FORMAT2_MAX_WINDOW_ROWS {
        return Err(format!(
            "layer {} position {}: window_rows {} is outside 1..={FORMAT2_MAX_WINDOW_ROWS}",
            dispatch.layer, dispatch.position, dispatch.window_rows
        ));
    }
    if dispatch.selected.len() > FORMAT2_MAX_SELECTED {
        return Err(format!(
            "layer {} position {}: selected_count {} exceeds {FORMAT2_MAX_SELECTED}",
            dispatch.layer,
            dispatch.position,
            dispatch.selected.len()
        ));
    }
    if dispatch.union_rows() > FORMAT2_MAX_ATTENDED {
        return Err(format!(
            "layer {} position {}: union of {} entries exceeds {FORMAT2_MAX_ATTENDED}",
            dispatch.layer,
            dispatch.position,
            dispatch.union_rows()
        ));
    }
    match dispatch.window_kv_format {
        ENGINE_WINDOW_KV_FORMAT => {
            if dispatch.selected.len() != dispatch.window_rows {
                return Err(format!(
                    "layer {} position {}: window-only format passes {} ids for {} window rows",
                    dispatch.layer,
                    dispatch.position,
                    dispatch.selected.len(),
                    dispatch.window_rows
                ));
            }
            for id in &dispatch.selected {
                if *id as usize >= dispatch.window_rows {
                    return Err(format!(
                        "layer {} position {}: window id {id} is outside {} rows",
                        dispatch.layer, dispatch.position, dispatch.window_rows
                    ));
                }
            }
        }
        CSA_UNION_WINDOW_KV_FORMAT => {
            let position = u32::try_from(dispatch.position)
                .map_err(|_| "position does not fit a u32".to_string())?;
            for id in &dispatch.selected {
                if *id > position {
                    return Err(format!(
                        "layer {} position {}: selected id {id} is in the future",
                        dispatch.layer, dispatch.position
                    ));
                }
                if dispatch.ratio > 0 && !(*id as usize).is_multiple_of(dispatch.ratio) {
                    return Err(format!(
                        "layer {}: selected id {id} is not a multiple of its ratio {}",
                        dispatch.layer, dispatch.ratio
                    ));
                }
            }
        }
        other => {
            return Err(format!(
                "layer {} position {}: unsupported window_kv_format {other}",
                dispatch.layer, dispatch.position
            ));
        }
    }
    Ok(())
}

/// The full dispatch plan of the compared positions (prefill last position plus the decode steps).
fn dispatch_plan(
    attention_mode: AttentionMode,
    positions: &[usize],
) -> HarnessResult<Vec<AttentionDispatch>> {
    if positions.is_empty() {
        return Err("a dispatch plan needs at least one position".to_string());
    }
    let mut plan = Vec::with_capacity(positions.len() * NUM_HIDDEN_LAYERS);
    for position in positions {
        plan.extend(dispatch_at_position(*position, attention_mode)?);
    }
    Ok(plan)
}

/// The positions a run compares. `--prefill-only` is 2B-i's narrowing knob: the same 130-token
/// prefill, but only its last position (the one with recorded logits) is compared and dispatched.
fn compared_set(prefill_only: bool) -> Vec<usize> {
    let compared = compared_positions(PROMPT_TOKENS, DECODE_STEPS);
    if prefill_only {
        vec![PREFILL_POSITION]
    } else {
        compared
    }
}

/// The compressed-view row counts for a source layer: the stored rows up to the last position, and
/// the expanded per-position rows (store_rows x ratio) the format-2 kernel indexes by absolute id.
fn compressed_view_rows(ratio: usize) -> HarnessResult<(usize, usize)> {
    if ratio == 0 {
        return Err("a compressed source has a positive ratio".to_string());
    }
    let store_rows = LAST_SEQUENCE_POSITION / ratio + 1;
    let view_rows = store_rows
        .checked_mul(ratio)
        .ok_or_else(|| "expanded row count overflow".to_string())?;
    Ok((store_rows, view_rows))
}

/// Reads the 2A golden fixture and validates the parts this harness relies on before any device
/// work: the schema invariants (via the engine's own validate) and the compared set.
fn load_fixture(path: &Path) -> HarnessResult<SequenceFixture> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let sha = sha256_hex(&bytes);
    if bytes.len() != EXPECTED_FIXTURE_BYTES || sha != EXPECTED_FIXTURE_SHA256 {
        return Err(format!(
            "{}: fixture is {} B sha256 {sha}, the pinned released fixture is {EXPECTED_FIXTURE_BYTES} B sha256 {EXPECTED_FIXTURE_SHA256}; refusing a look-alike",
            path.display(),
            bytes.len()
        ));
    }
    let fixture: SequenceFixture = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{}: not a SequenceFixture: {error}", path.display()))?;
    fixture
        .validate()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(fixture)
}

/// Checks the fixture against the pinned 2B facts. This is deliberately independent of the device
/// run: a fixture that no longer describes the 130-token sequence must fail before a launch.
fn check_fixture_shape(fixture: &SequenceFixture) -> HarnessResult<()> {
    if fixture.prompt_tokens.len() != PROMPT_TOKENS {
        return Err(format!(
            "fixture prompt has {} tokens, the harness pins {PROMPT_TOKENS}",
            fixture.prompt_tokens.len()
        ));
    }
    if fixture.decode_steps != DECODE_STEPS {
        return Err(format!(
            "fixture has {} decode steps, the harness pins {DECODE_STEPS}",
            fixture.decode_steps
        ));
    }
    if fixture.window != WINDOW {
        return Err(format!("fixture window {} is not {WINDOW}", fixture.window));
    }
    let expected_compared = compared_positions(PROMPT_TOKENS, DECODE_STEPS);
    if fixture.compared_positions != expected_compared {
        return Err(format!(
            "fixture compared positions {:?} are not the pinned set {expected_compared:?}",
            fixture.compared_positions
        ));
    }
    let boundary = exclusion_boundary(PROMPT_TOKENS, WINDOW);
    if boundary != 2 {
        return Err(format!("CED exclusion boundary is {boundary}, expected 2"));
    }
    let prefill = fixture
        .steps
        .iter()
        .find(|step| step.position == PREFILL_POSITION)
        .ok_or_else(|| format!("fixture has no prefill position {PREFILL_POSITION}"))?;
    if prefill.next_token != Some(PREFILL_TOKEN) {
        return Err(format!(
            "fixture prefill token is {:?}, the pinned token is {PREFILL_TOKEN}",
            prefill.next_token
        ));
    }
    Ok(())
}

/// The outcome of a 2B-i prefill comparison. Token agreement is reported first and separately: it
/// is the hard unit, and logits are only interpreted after it agrees.
#[derive(Debug, Clone, PartialEq)]
struct PrefillComparison {
    position: usize,
    expected_token: Option<u32>,
    device_token: Option<u32>,
    max_abs_logits_delta: Option<f32>,
    bar: f32,
    logits_within_bar: Option<bool>,
}

impl PrefillComparison {
    fn line(&self) -> String {
        let token = match (self.expected_token, self.device_token) {
            (Some(expected), Some(found)) if expected == found => {
                format!("token {found} (expected {expected})")
            }
            (expected, found) => format!("token {found:?} (expected {expected:?}) MISMATCH"),
        };
        match (self.max_abs_logits_delta, self.logits_within_bar) {
            (Some(delta), Some(within)) => format!(
                "position {}: {token}; logits max-abs delta {delta:.6e} vs bar {:.1e} ({})",
                self.position,
                self.bar,
                if within { "within bar" } else { "OUT OF BAR" }
            ),
            _ => format!(
                "position {}: {token}; no logits recorded in the fixture for this position",
                self.position
            ),
        }
    }
}

/// 2B-i: the prefill position only. The token is checked first and a mismatch is an error; a logits
/// delta is reported with the bar, and only within-bar is a pass.
fn compare_prefill(
    fixture: &SequenceFixture,
    device: &DeviceStep,
) -> HarnessResult<PrefillComparison> {
    let expected = fixture
        .steps
        .iter()
        .find(|step| step.position == PREFILL_POSITION)
        .ok_or_else(|| format!("fixture has no position {PREFILL_POSITION}"))?;
    if device.position != PREFILL_POSITION {
        return Err(format!(
            "2B-i compares position {PREFILL_POSITION} only, device reported {}",
            device.position
        ));
    }
    if device.next_token != expected.next_token {
        return Err(format!(
            "position {}: fixture token {:?}, device token {:?} (token comparison is first and hard)",
            PREFILL_POSITION, expected.next_token, device.next_token
        ));
    }
    let (max_abs_logits_delta, logits_within_bar) = match &expected.logits {
        Some(want) => {
            let got = device.logits.as_ref().ok_or_else(|| {
                format!(
                    "position {PREFILL_POSITION}: fixture records logits, the device step does not"
                )
            })?;
            if got.len() != want.len() {
                return Err(format!(
                    "position {PREFILL_POSITION}: {} logits on device, {} in fixture",
                    got.len(),
                    want.len()
                ));
            }
            let delta = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            (Some(delta), Some(delta <= LOGITS_ABS_BAR))
        }
        None => (None, None),
    };
    Ok(PrefillComparison {
        position: PREFILL_POSITION,
        expected_token: expected.next_token,
        device_token: device.next_token,
        max_abs_logits_delta,
        bar: LOGITS_ABS_BAR,
        logits_within_bar,
    })
}

/// 2B-ii: the full compared set (the last prefill position plus the eight decode positions), using
/// the engine's own comparators so the CED exclusion cannot be widened here: compare_steps rejects
/// a device position outside the fixture's compared set, and compare_logits applies the bar only to
/// positions the fixture recorded.
fn compare_full_sequence(fixture: &SequenceFixture, device: &[DeviceStep]) -> HarnessResult<()> {
    compare_steps(fixture, device).map_err(|error| format!("token comparison: {error}"))?;
    compare_logits(fixture, device, LOGITS_ABS_BAR)
        .map_err(|error| format!("logits comparison: {error}"))
}

/// The CPU self-check the default mode and --dry-run call. It never allocates device memory and
/// never launches a kernel. The example's unit tests are what run in CI (test-examples job); this
/// function is only executed by a host run.
fn run_self_check(args: &Args) -> HarnessResult<String> {
    let schedule = checked_schedule(args.prompt_tokens, args.no_ced)?;
    if args.attention_mode == AttentionMode::CsaUnion
        && CSA_UNION_WINDOW_KV_FORMAT == HISTORICAL_FP4_WINDOW_KV_FORMAT
    {
        return Err("the union mode cannot use the historical FP4 format".to_string());
    }
    if !args.attention_mode.is_equality_claim()
        && !args.attention_mode.scope_label().contains("equality claim")
    {
        return Err("the window-only label lost its no-equality-claim wording".to_string());
    }
    let rotation = RotationPlan::for_schedule(&schedule);
    if rotation.bytes_per_layer() != ROTATION_BYTES_PER_LAYER {
        return Err(format!(
            "rotation bytes per layer {} != {}",
            rotation.bytes_per_layer(),
            ROTATION_BYTES_PER_LAYER
        ));
    }
    if rotation.ring_bytes() != ROTATION_BYTES_PER_STEP {
        return Err(format!(
            "rotation ring bytes {} != {}",
            rotation.ring_bytes(),
            ROTATION_BYTES_PER_STEP
        ));
    }
    // The plan owns the sum of the rows each execution copies; this check pins the two numbers the
    // skeleton's first accounting conflated: the full-ring volume and the executed byte total.
    let expected_slots = {
        let mut slots = 0usize;
        for layer in 0..NUM_HIDDEN_LAYERS {
            for position in execution_positions(layer, &schedule) {
                let (rows, _) = window_view(&WindowRingLayout::ENGINE, &schedule, layer, position)?;
                slots += rows;
            }
        }
        slots
    };
    if rotation.slots_copied != expected_slots
        || rotation.bytes() != expected_slots * (WINDOW_CODE_ROW_BYTES + WINDOW_SCALE_ROW_BYTES)
    {
        return Err(format!(
            "rotation plan {} slots / {} B disagrees with the per-execution sum {expected_slots} slots / {} B",
            rotation.slots_copied,
            rotation.bytes(),
            expected_slots * (WINDOW_CODE_ROW_BYTES + WINDOW_SCALE_ROW_BYTES)
        ));
    }
    let geometry = StoreGeometry::for_max_seq_len(args.max_seq_len)
        .map_err(|error| format!("store geometry for {} tokens: {error}", args.max_seq_len))?;
    // The store's own accounting: forty rings, each 128 x 528 B, is one rotation step's bytes.
    if geometry.window_bytes() as usize != ROTATION_BYTES_PER_STEP {
        return Err(format!(
            "store window bytes {} != rotation bytes per step {}",
            geometry.window_bytes(),
            ROTATION_BYTES_PER_STEP
        ));
    }
    for layer in 0..NUM_HIDDEN_LAYERS {
        let expected_source = match schedule.modes[layer] {
            Csa2Mode::Full => Some(layer),
            Csa2Mode::Reuse | Csa2Mode::Reindex => {
                Some(sparkflash_kv::config::kv_source_for(layer))
            }
            Csa2Mode::SwaOnly => None,
        };
        let geometry_source = geometry.source_layer_for(layer);
        if geometry_source != expected_source {
            return Err(format!(
                "layer {layer}: store source {geometry_source:?} != schedule source {expected_source:?}"
            ));
        }
    }
    // The per-layer token extents are the work list the device path will execute; the counts here
    // also tie the rotation step count to the CED split instead of leaving it as an independent
    // arithmetic claim.
    let mut extents = Vec::with_capacity(NUM_HIDDEN_LAYERS);
    for layer in 0..NUM_HIDDEN_LAYERS {
        extents.push(layer_token_extent(
            layer,
            schedule.encoder_tokens,
            schedule.decoder_tokens,
        ));
    }
    let token_layer_executions: usize = extents.iter().map(Vec::len).sum();
    if token_layer_executions != rotation.exec_prefill {
        return Err(format!(
            "prefill token-layer executions {token_layer_executions} != rotation prefill executions {}",
            rotation.exec_prefill
        ));
    }
    if rotation.exec_decode != DECODE_STEPS * NUM_HIDDEN_LAYERS {
        return Err(format!(
            "rotation decode executions {} != {} decode steps x {} layers",
            rotation.exec_decode, DECODE_STEPS, NUM_HIDDEN_LAYERS
        ));
    }
    for (layer, extent) in extents.iter().enumerate() {
        let positions = execution_positions(layer, &schedule);
        let mut expected = extent.clone();
        expected.extend(PROMPT_TOKENS..PROMPT_TOKENS + DECODE_STEPS);
        if positions != expected {
            return Err(format!(
                "layer {layer}: execution positions do not follow its CED extent plus the decode steps"
            ));
        }
    }
    // Task-10 regression guard: !61 aligned sparkflash_kv's bounded-replay split with the CED
    // boundaries (encoder 21 / decoder 19). The schedule is the source of truth; this cross-check
    // fails the run if the kv crate and the schedule drift apart again.
    let kv_encoder = replay_layer_count(ReplayKind::Encoder);
    let kv_decoder = replay_layer_count(ReplayKind::Decoder);
    if kv_encoder != schedule.encoder_layers || kv_decoder != schedule.decoder_layers {
        return Err(format!(
            "SWA replay split drift: sparkflash_kv reports encoder {kv_encoder} / decoder {kv_decoder}, CED expects encoder {} / decoder {}",
            schedule.encoder_layers, schedule.decoder_layers
        ));
    }
    let mut lines = vec![
        format!(
            "schedule: prompt {} tokens, encoder {} tokens/layers {}, decoder {} tokens/layers {}, window {}, exclusion boundary {}",
            schedule.prompt_tokens,
            schedule.encoder_tokens,
            schedule.encoder_layers,
            schedule.decoder_tokens,
            schedule.decoder_layers,
            schedule.window,
            schedule.exclusion_boundary
        ),
        format!("plan: {:?}", schedule.plan),
        format!(
            "publications ({}): {:?}",
            schedule.publications.len(),
            schedule
                .publications
                .iter()
                .map(|publication| (publication.layer, format!("{:?}", publication.handle)))
                .collect::<Vec<_>>()
        ),
        format!(
            "store geometry: max_seq_len {}, window bytes {}, global KV bytes {}, total bytes {}, {:.3} B/token",
            geometry.max_seq_len(),
            geometry.window_bytes(),
            geometry.global_kv_bytes(),
            geometry.total_bytes(),
            geometry.kv_bytes_per_token()
        ),
        format!(
            "work list: {} layer extents, {} token-layer executions (encoder {}x{}, decoder {}x{})",
            extents.len(),
            token_layer_executions,
            schedule.encoder_layers,
            schedule.encoder_tokens,
            schedule.decoder_layers,
            schedule.decoder_tokens
        ),
        rotation.line(),
        format!(
            "store window geometry: {} rings x {} B = {} B (the full-ring volume, not the rotated total)",
            rotation.layers,
            rotation.bytes_per_layer(),
            rotation.ring_bytes()
        ),
        format!(
            "attention mode: {} (window_kv_format {})",
            args.attention_mode.scope_label(),
            args.attention_mode.window_kv_format()
        ),
        format!(
            "swa replay split: CED encoder 0..={} ({} layers), decoder {}..={} ({} layers); sparkflash_kv ReplayKind Encoder {} / Decoder {} (strict cross-check, task-10 regression guard)",
            ENCODER_LAST_LAYER,
            ENCODER_LAST_LAYER + 1,
            DECODER_FIRST_LAYER,
            NUM_HIDDEN_LAYERS - 1,
            NUM_HIDDEN_LAYERS - DECODER_FIRST_LAYER,
            replay_layer_count(ReplayKind::Encoder),
            replay_layer_count(ReplayKind::Decoder)
        ),
        "compressed-id expansion: ratio-2 c -> c*2 absolute position, expanded view row p reads store row p/2; selected ids are the engine's reachability-filtered list (kernel id<=position is a guard, not the reference reach rule)"
            .to_string(),
    ];
    // The dispatch plan is built here as well, so a host self-check refuses a mode
    // table, a reach rule or a bound that the device dispatch would have to refuse later.
    let compared = compared_set(args.prefill_only);
    let plan = dispatch_plan(args.attention_mode, &compared)?;
    let window_only = plan
        .iter()
        .filter(|dispatch| dispatch.window_kv_format == ENGINE_WINDOW_KV_FORMAT)
        .count();
    let union = plan.len() - window_only;
    lines.push(format!(
        "attention dispatch plan: {} positions x {} layers = {} calls ({} window-only format {}, {} union format {}); per-position selected counts (window rows excluded) {:?}",
        compared.len(),
        NUM_HIDDEN_LAYERS,
        plan.len(),
        window_only,
        ENGINE_WINDOW_KV_FORMAT,
        union,
        CSA_UNION_WINDOW_KV_FORMAT,
        compared
            .iter()
            .map(|position| dispatch_at_position(*position, args.attention_mode).map(|plan| {
                plan.iter()
                    .filter(|dispatch| dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT)
                    .map(AttentionDispatch::selected_count)
                    .sum::<usize>()
            }))
            .collect::<HarnessResult<Vec<_>>>()?
    ));
    lines.push(format!(
        "layer attention kinds: {} window-only (format {}), {} union (format {}); store layers {:?}; ratios {:?}",
        window_only / compared.len().max(1),
        ENGINE_WINDOW_KV_FORMAT,
        union / compared.len().max(1),
        CSA_UNION_WINDOW_KV_FORMAT,
        (0..NUM_HIDDEN_LAYERS)
            .filter_map(|layer| layer_attention_kind(layer)
                .ok()
                .and_then(|kind| match kind {
                    LayerAttentionKind::CsaUnion { store_layer, .. } => Some(store_layer),
                    LayerAttentionKind::WindowOnly => None,
                }))
            .collect::<std::collections::BTreeSet<_>>(),
        (0..NUM_HIDDEN_LAYERS)
            .filter_map(|layer| layer_attention_kind(layer)
                .ok()
                .and_then(|kind| match kind {
                    LayerAttentionKind::CsaUnion { ratio, .. } => Some(ratio),
                    LayerAttentionKind::WindowOnly => None,
                }))
            .collect::<std::collections::BTreeSet<_>>()
    ));
    if let Some(golden) = &args.golden {
        if golden.exists() {
            let fixture = load_fixture(golden)?;
            check_fixture_shape(&fixture)?;
            lines.push(format!(
                "fixture {}: prompt {}, decodes {}, window {}, compared {:?}, prefill token {:?}, prefill logits recorded: {}",
                golden.display(),
                fixture.prompt_tokens.len(),
                fixture.decode_steps,
                fixture.window,
                fixture.compared_positions,
                fixture
                    .steps
                    .iter()
                    .find(|step| step.position == PREFILL_POSITION)
                    .and_then(|step| step.next_token),
                fixture
                    .steps
                    .iter()
                    .find(|step| step.position == PREFILL_POSITION)
                    .and_then(|step| step.logits.as_ref())
                    .is_some()
            ));
            // The comparators are exercised on synthetic device steps whose tokens/logits are the
            // fixture's own, then perturbed once each: a comparator that cannot fail would make the
            // later device comparison meaningless.
            let mut device: Vec<DeviceStep> = fixture
                .steps
                .iter()
                .map(|step| DeviceStep {
                    position: step.position,
                    next_token: step.next_token,
                    logits: step.logits.clone(),
                    logits_sha256: None,
                })
                .collect();
            compare_full_sequence(&fixture, &device)?;
            let prefill = device
                .iter()
                .find(|step| step.position == PREFILL_POSITION)
                .ok_or_else(|| "synthetic device steps lost the prefill".to_string())?;
            let comparison = compare_prefill(&fixture, prefill)?;
            lines.push(format!(
                "fixture self-comparison (synthetic): {}",
                comparison.line()
            ));
            if let Some(step) = device
                .iter_mut()
                .find(|step| step.position == PREFILL_POSITION)
            {
                step.next_token = step.next_token.map(|token| token.wrapping_add(1));
            }
            if compare_full_sequence(&fixture, &device).is_ok() {
                return Err("the comparator accepted a wrong prefill token".to_string());
            }
            let mut device: Vec<DeviceStep> = fixture
                .steps
                .iter()
                .map(|step| DeviceStep {
                    position: step.position,
                    next_token: step.next_token,
                    logits: step.logits.clone(),
                    logits_sha256: None,
                })
                .collect();
            if let Some(step) = device
                .iter_mut()
                .find(|step| step.position == PREFILL_POSITION)
                && let Some(logits) = step.logits.as_mut()
            {
                logits[0] += 1.0;
            }
            if compare_full_sequence(&fixture, &device).is_ok() {
                return Err("the comparator accepted logits perturbed past the bar".to_string());
            }
        } else {
            lines.push(format!(
                "fixture {}: not present; fixture checks skipped (CPU schedule checks ran)",
                golden.display()
            ));
        }
    }
    Ok(lines.join("\n"))
}

/// SHA-256 of a byte slice, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// SHA-256 of the running binary, recorded so a report cannot be separated from the executable that
/// wrote it.
fn current_binary_sha256() -> HarnessResult<String> {
    let path = std::env::current_exe().map_err(|error| format!("current_exe: {error}"))?;
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// The host label a report carries, without making a report fail when the environment lacks one.
fn host_label() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// The report header every mode shares. The mode is the first line so a reader cannot quote a
/// dry-run line as device evidence.
fn report_header(mode: &str, args: &Args) -> HarnessResult<Vec<String>> {
    let argv: Vec<String> = std::env::args().collect();
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|error| format!("<unknown: {error}>"));
    Ok(vec![
        "m8_sequence report".to_string(),
        format!("mode: {mode}"),
        format!("generated unix: {}", now_unix()),
        format!("host: {}", host_label()),
        format!("cwd: {cwd}"),
        format!("binary sha256: {}", current_binary_sha256()?),
        format!("argv: {}", argv.join(" ")),
        format!(
            "attention mode: {} (window_kv_format {}, --attention-mode)",
            args.attention_mode.scope_label(),
            args.attention_mode.window_kv_format()
        ),
        format!("provenance: {}", args.provenance),
        format!("self-check requested: {}", args.self_check),
    ])
}

/// The honesty block every report ends with. Each line states a limit, not a result.
fn honesty_lines(compared: &[usize]) -> Vec<String> {
    vec![
        "honesty: the mode line is the only definition of what ran; no line below it upgrades a dry-run to device evidence"
            .to_string(),
        format!(
            "honesty: the compared set is exactly {compared:?} (the last prefill position plus the decode positions, or the prefill position alone under --prefill-only, D7); no older position is compared, and the fixture is never edited to make a run pass"
        ),
        format!(
            "honesty: tokens are compared first and hard; logits only at LOGITS_ABS_BAR = {LOGITS_ABS_BAR:.1e} and only where the fixture recorded them"
        ),
        "honesty: the HSI pool population and the top-512 candidate merge are a host seam in this iteration; the harness supplies the reachable compressed rows directly and labels the seam"
            .to_string(),
        "honesty: the per-layer forward chain (embedding -> 40 layers -> lm head -> greedy selection) is NOT wired in m8_sequence yet, so a device run measures the rotation and the attention dispatch, not tokens or logits against the fixture"
            .to_string(),
        "honesty: Phase 2 exit criterion 1 is OPEN; nothing in this report is an exit-criterion claim"
            .to_string(),
    ]
}

/// The host seams a report must name, so host work cannot read as device work.
fn seam_lines(plan: &[AttentionDispatch], compared: &[usize]) -> Vec<String> {
    let union_calls = plan
        .iter()
        .filter(|dispatch| dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT)
        .count();
    let window_calls = plan.len() - union_calls;
    vec![
        format!(
            "host seam HSI/indexer selection: {union_calls} union dispatch(es) carry a harness-supplied reachable compressed-row list; the pool population and the top-512 merge are NOT device kernels in this iteration and are not modelled here"
        ),
        "host seam compressed view: the per-position expanded view (row p = store row p/ratio; 256 B FP4 codes + 32 B E4M3 scales per row) is built on the host and uploaded; the ratio compressor that would produce the rows is not a device kernel in this iteration"
            .to_string(),
        format!(
            "host seam window format: the {window_calls} window-only dispatch(es) use the engine-format rows the harness supplies; the anchor/quantiser that fills them belongs to the layer chain, not to this dispatch"
        ),
        format!(
            "host seam forward chain: embedding, the 40-layer transformer chain, the lm head and greedy selection are absent; the {}-position token/logits comparison is therefore NOT attempted on device",
            compared.len()
        ),
        "format-2 builder: the shared sequence_chain::Format2AttentionCall now carries its four indexer buffers (index_key, index_query, block_scores, block_best) and refuses nulls, so params()/launch() are launchable; this harness validates every union dispatch through it and pins the pointer words with the format2_builder_carries_its_indexer_buffers test"
            .to_string(),
        "host seam launch assembly: format-2 dispatches are built through the shared Format2AttentionCall (which delegates to cuda_harness::flash_mla_params, the one constructor that owns the 19-word order); format-1 window-only calls take flash_mla_params directly because the shared builder is format-2 only"
            .to_string(),
    ]
}

/// One line per compared step: token first, logits second, source labelled. The step-plumbing report
/// is shared by the dry-run (where the steps are synthetic and say so) and the device run (where the
/// steps are not produced yet and say that instead).
#[derive(Debug, Clone)]
struct StepReport {
    position: usize,
    kind: &'static str,
    expected_token: Option<u32>,
    device_token: Option<u32>,
    logits_sha256: String,
    logits_delta: Option<f32>,
    device_source: &'static str,
}

impl StepReport {
    fn line(&self) -> String {
        let token = match (self.expected_token, self.device_token) {
            (Some(expected), Some(found)) if expected == found => {
                format!("token {found} (fixture {expected}) match=true")
            }
            (Some(expected), Some(found)) => {
                format!("token {found} (fixture {expected}) match=false MISMATCH")
            }
            (expected, None) => {
                format!("device token NOT PRODUCED (fixture {expected:?}) match=NOT ATTEMPTED")
            }
            (None, Some(found)) => {
                format!("token {found} (fixture records none) match=NOT ATTEMPTED")
            }
        };
        let logits = match self.logits_delta {
            Some(delta) => format!(
                "logits max-abs delta {delta:.6e} vs bar {LOGITS_ABS_BAR:.1e} ({}), fixture logits sha256 {}",
                if delta <= LOGITS_ABS_BAR {
                    "within bar"
                } else {
                    "OUT OF BAR"
                },
                self.logits_sha256
            ),
            None => format!(
                "no logits delta measured (fixture logits sha256 {}, recorded only where the fixture captured them)",
                self.logits_sha256
            ),
        };
        format!(
            "step position {} ({}): {token}; {logits} [{}]",
            self.position, self.kind, self.device_source
        )
    }
}

/// The dry-run's step table: every device step is a copy of the fixture step, so a token match and a
/// zero delta here prove the comparison plumbing and nothing about a device. The negative controls
/// below prove the same comparators reject a wrong token and out-of-bar logits.
fn dry_run_step_reports(
    fixture: &SequenceFixture,
    compared: &[usize],
) -> HarnessResult<(Vec<StepReport>, Vec<String>)> {
    let device: Vec<DeviceStep> = fixture
        .steps
        .iter()
        .map(|step| DeviceStep {
            position: step.position,
            next_token: step.next_token,
            logits: step.logits.clone(),
            logits_sha256: None,
        })
        .collect();
    compare_full_sequence(fixture, &device)?;
    // The prefill position comes from the fixture, not from the pinned 130-token constant: this
    // plumbing is also exercised against a small generated fixture in its own test.
    let prefill_position = fixture
        .steps
        .iter()
        .find(|step| step.kind == sparkflash_engine::sequence_fixture::StepKind::Prefill)
        .map(|step| step.position)
        .ok_or_else(|| "the fixture records no prefill step".to_string())?;
    let prefill = device
        .iter()
        .find(|step| step.position == prefill_position)
        .ok_or_else(|| "synthetic device steps lost the prefill".to_string())?;
    compare_prefill(fixture, prefill)?;
    let mut reports = Vec::with_capacity(compared.len());
    for step in fixture
        .steps
        .iter()
        .filter(|step| compared.contains(&step.position))
    {
        reports.push(StepReport {
            position: step.position,
            kind: match step.kind {
                sparkflash_engine::sequence_fixture::StepKind::Prefill => "prefill",
                sparkflash_engine::sequence_fixture::StepKind::Decode => "decode",
            },
            expected_token: step.next_token,
            device_token: step.next_token,
            logits_sha256: step.logits_sha256.clone(),
            logits_delta: step.logits.as_ref().map(|_| 0.0f32),
            device_source: "synthetic: device step copied from the fixture, not device output",
        });
    }
    let mut controls = Vec::new();
    let mut wrong_token = device.clone();
    if let Some(step) = wrong_token
        .iter_mut()
        .find(|step| step.position == prefill_position)
    {
        step.next_token = step.next_token.map(|token| token.wrapping_add(1));
    }
    match compare_full_sequence(fixture, &wrong_token) {
        Ok(()) => return Err("the comparator accepted a wrong prefill token".to_string()),
        Err(error) => controls.push(format!("wrong prefill token refused: {error}")),
    }
    if fixture
        .steps
        .iter()
        .any(|step| step.position == prefill_position && step.logits.is_some())
    {
        let mut wrong_logits = device.clone();
        if let Some(step) = wrong_logits
            .iter_mut()
            .find(|step| step.position == prefill_position)
            && let Some(logits) = step.logits.as_mut()
        {
            logits[0] += 1.0;
        }
        match compare_full_sequence(fixture, &wrong_logits) {
            Ok(()) => {
                return Err("the comparator accepted logits perturbed past the bar".to_string());
            }
            Err(error) => controls.push(format!("out-of-bar prefill logits refused: {error}")),
        }
    } else {
        controls.push(
            "out-of-bar logits control NOT exercised: this fixture records no prefill logits"
                .to_string(),
        );
    }
    let mut outside = device.clone();
    outside.push(DeviceStep {
        position: prefill_position.saturating_sub(1),
        next_token: None,
        logits: None,
        logits_sha256: None,
    });
    match compare_full_sequence(fixture, &outside) {
        Ok(()) => {
            return Err(
                "the comparator accepted an extra position outside the compared set".to_string(),
            );
        }
        Err(error) => controls.push(format!(
            "position outside the compared set refused: {error}"
        )),
    }
    Ok((reports, controls))
}

/// The device run's step table: the fixture is the expectation, the device half produced no token, and
/// the line says so instead of leaving a blank that could be read as a pass.
fn device_step_reports(fixture: &SequenceFixture, compared: &[usize]) -> Vec<StepReport> {
    fixture
        .steps
        .iter()
        .filter(|step| compared.contains(&step.position))
        .map(|step| StepReport {
            position: step.position,
            kind: match step.kind {
                sparkflash_engine::sequence_fixture::StepKind::Prefill => "prefill",
                sparkflash_engine::sequence_fixture::StepKind::Decode => "decode",
            },
            expected_token: step.next_token,
            device_token: None,
            logits_sha256: step.logits_sha256.clone(),
            logits_delta: None,
            device_source: "no device output: forward chain not wired in this iteration",
        })
        .collect()
}

/// Writes the report to --out (creating the parent directory) and prints its size and digest. The
/// file is the artifact a window copies back; the stdout copy is for the operator.
fn write_report_text(args: &Args, text: &str) -> HarnessResult<()> {
    let Some(path) = &args.out else {
        return Ok(());
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    let with_newline = format!("{text}\n");
    std::fs::write(path, with_newline.as_bytes())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    println!(
        "report: wrote {} ({} B, sha256 {})",
        path.display(),
        with_newline.len(),
        sha256_hex(with_newline.as_bytes())
    );
    Ok(())
}

/// The host dry-run: no CUDA, no cubin, no device. It builds the exact dispatch and rotation plans a
/// device run would execute, bounds-checks every union launch through the shared checked builder,
/// and exercises the token-then-logits comparison plumbing (including its negative controls)
/// against the golden fixture. That is what can be validated before a window; it is not evidence
/// about the device.
fn run_dry_run(args: &Args) -> HarnessResult<String> {
    let golden = args
        .golden
        .as_ref()
        .ok_or_else(|| "--dry-run needs --golden".to_string())?;
    let fixture_bytes =
        std::fs::read(golden).map_err(|error| format!("{}: {error}", golden.display()))?;
    let fixture_sha = sha256_hex(&fixture_bytes);
    let fixture = load_fixture(golden)?;
    check_fixture_shape(&fixture)?;
    let self_check = run_self_check(args)?;
    let compared = compared_set(args.prefill_only);
    let plan = dispatch_plan(args.attention_mode, &compared)?;
    // Every union dispatch goes through the shared checked format-2 builder with placeholder
    // pointers (that builder only bounds-checks counts and pointer nullness). This proves the
    // dispatch shape and the pinned ABI without a device allocation.
    let mut checked_format2 = 0usize;
    for dispatch in &plan {
        if dispatch.window_kv_format != CSA_UNION_WINDOW_KV_FORMAT {
            continue;
        }
        let call = Format2AttentionCall {
            q: Deviceptr(1),
            kv_fp4: Deviceptr(1),
            kv_scales: Deviceptr(1),
            window_codes: Deviceptr(1),
            window_scales: Deviceptr(1),
            sink: Deviceptr(1),
            selected: Deviceptr(1),
            out: Deviceptr(1),
            position: u32::try_from(dispatch.position)
                .map_err(|_| "dispatch position does not fit the kernel parameter".to_string())?,
            layer: dispatch.layer,
            step_seq: 0,
            window_rows: u32::try_from(dispatch.window_rows).map_err(|_| {
                "dispatch window_rows does not fit the kernel parameter".to_string()
            })?,
            selected_ids: &dispatch.selected,
            trace_ring: None,
            index_key: Deviceptr(1),
            index_query: Deviceptr(1),
            block_scores: Deviceptr(1),
            block_best: Deviceptr(1),
        };
        call.params()?;
        checked_format2 += 1;
    }
    let mut view_lines = Vec::new();
    for store_layer in KV_SOURCE_LAYERS {
        let ratio = compress_ratio(store_layer);
        let (store_rows, _) = compressed_view_rows(ratio)?;
        // Build the same deterministic store image and expansion the device run uploads, so the
        // dry-run exercises the builder and reports the bytes the device would allocate.
        let (codes, scales) = host_compressed_store(store_layer, store_rows)?;
        let expanded = expand_compressed_view(&codes, &scales, store_rows, ratio)?;
        view_lines.push(format!(
            "compressed store layer {store_layer} (ratio {ratio}): {store_rows} stored rows, expanded view {} rows = {} B codes + {} B scales",
            expanded.rows,
            expanded.codes_bytes(),
            expanded.scales_bytes()
        ));
    }
    let (step_reports, controls) = dry_run_step_reports(&fixture, &compared)?;
    let mut lines = report_header(
        "host dry-run (CPU only: no CUDA, no cubin, no device allocation)",
        args,
    )?;
    lines.push(format!(
        "fixture: {} ({} B, sha256 {})",
        golden.display(),
        fixture_bytes.len(),
        fixture_sha
    ));
    lines.push(format!(
        "fixture shape: prompt {}, decodes {}, window {}, compared {:?}, config label {}, source {:?}",
        fixture.prompt_tokens.len(),
        fixture.decode_steps,
        fixture.window,
        fixture.compared_positions,
        fixture.config_label,
        fixture.source.kind
    ));
    lines.push(String::new());
    lines.push("--- schedule / store / rotation (checked on the host) ---".to_string());
    lines.push(self_check);
    lines.push(String::new());
    lines
        .push("--- attention dispatch plan (the device calls this run would make) ---".to_string());
    let union_calls = plan
        .iter()
        .filter(|dispatch| dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT)
        .count();
    lines.push(format!(
        "dispatch plan: {} compared positions x {} layers = {} attention calls, {union_calls} union (format {CSA_UNION_WINDOW_KV_FORMAT}) + {} window-only (format {ENGINE_WINDOW_KV_FORMAT}); format-2 calls bounds-checked through the shared builder: {checked_format2}",
        compared.len(),
        NUM_HIDDEN_LAYERS,
        plan.len(),
        plan.len() - union_calls
    ));
    for dispatch in &plan {
        lines.push(format!(
            "  layer {:02} position {}: format {}, window_rows {}, selected_count {}, union {}, store_layer {:?}, ratio {}",
            dispatch.layer,
            dispatch.position,
            dispatch.window_kv_format,
            dispatch.window_rows,
            dispatch.selected_count(),
            dispatch.union_rows(),
            dispatch.store_layer,
            dispatch.ratio
        ));
    }
    lines.push(String::new());
    lines.push("--- compressed views ---".to_string());
    lines.extend(view_lines);
    lines.push(String::new());
    lines.push("--- host seams (labelled; not device work) ---".to_string());
    lines.extend(seam_lines(&plan, &compared));
    lines.push(String::new());
    lines.push(
        "--- per-step comparison plumbing (synthetic device steps; NOT device evidence) ---"
            .to_string(),
    );
    for report in &step_reports {
        lines.push(report.line());
    }
    lines.push(String::new());
    lines.push(
        "--- comparator negative controls (the plumbing must refuse what it should) ---"
            .to_string(),
    );
    lines.extend(controls);
    lines.push(String::new());
    lines.push("--- what this dry-run proves / does not prove ---".to_string());
    lines.push(
        "proves: the CLI and pinned schedule, the store geometry and the corrected rotation byte accounting, the per-layer attention-kind table (2 window-only + 38 union), the reachable-id lists, the kernel bounds for every union call, the expanded-view sizes, and that the token-then-logits comparators accept the fixture and reject a wrong token, out-of-bar logits and an out-of-set position"
            .to_string(),
    );
    lines.push(
        "does not prove: any CUDA execution, any device memory behaviour, any rotation timing, any device attention output, any token or logits of the device path, or anything about the reference model's numbers"
            .to_string(),
    );
    lines.push(format!(
        "device command this dry-run plans (Lead-sequenced window): the same binary with --device --checkpoint DIR --index PATH --golden {} --cubin-dir DIR --out report.txt",
        golden.display()
    ));
    lines.push(String::new());
    lines.extend(honesty_lines(&compared));
    lines.push(format!(
        "rotation plan note: the skeleton's first accounting multiplied {ROTATION_BYTES_PER_STEP} B (all forty rings once) by the execution count; the corrected total above sums the rows each execution actually copies"
    ));
    Ok(lines.join("\n"))
}

/// A deterministic engine-format window ring image for one layer: 128 slots, each built through the
/// one shared engine-format row construction (so the bytes are a real E4M3/E8M0 row, finite under
/// the guard, not random bytes). Harness-owned input, not model state.
fn host_ring_image(layer: usize) -> HarnessResult<(Vec<u8>, Vec<u8>)> {
    let layout = WindowRingLayout::ENGINE;
    let mut codes = vec![0u8; layout.slots * layout.code_row_bytes];
    let mut scales = vec![0u8; layout.slots * layout.scale_row_bytes];
    let mut latent = vec![0.0f32; ATTENTION_LATENT];
    for slot in 0..layout.slots {
        for (channel, value) in latent.iter_mut().enumerate() {
            let phase = layer as f32 * 0.37 + slot as f32 * 0.11 + channel as f32 * 0.013;
            *value = phase.sin() * 0.5 + (channel % 7) as f32 * 0.01;
        }
        let row = engine_window_row(&latent)?;
        codes[slot * layout.code_row_bytes..(slot + 1) * layout.code_row_bytes]
            .copy_from_slice(&row.codes);
        scales[slot * layout.scale_row_bytes..(slot + 1) * layout.scale_row_bytes]
            .copy_from_slice(&row.scales);
    }
    assert_engine_window_finite(
        &codes,
        &scales,
        layout.slots,
        &format!("harness ring image layer {layer}"),
    )?;
    Ok((codes, scales))
}

/// A deterministic compressed-store image for one source layer, in the store layout (256 B packed
/// FP4 codes + 32 B E4M3 scales per row). Harness-owned input, not model KV.
fn host_compressed_store(
    store_layer: usize,
    store_rows: usize,
) -> HarnessResult<(Vec<u8>, Vec<u8>)> {
    if compress_ratio(store_layer) == 0 {
        return Err(format!("store layer {store_layer} has ratio 0"));
    }
    let mut codes = vec![0u8; store_rows * COMPRESSED_CODES_BYTES];
    let mut scales = vec![0u8; store_rows * COMPRESSED_SCALES_BYTES];
    for row in 0..store_rows {
        for byte in 0..COMPRESSED_CODES_BYTES {
            let low = ((row * 13 + byte * 7) % 16) as u8;
            let high = ((row * 5 + byte * 3 + 8) % 16) as u8;
            codes[row * COMPRESSED_CODES_BYTES + byte] = low | (high << 4);
        }
        for scale in 0..COMPRESSED_SCALES_BYTES {
            // E4M3 0x38 is 1.0; every byte below 0x7F decodes finite, unlike 0x7F/0xFF.
            scales[row * COMPRESSED_SCALES_BYTES + scale] = 0x38;
        }
    }
    for row in [0usize, store_rows / 2, store_rows - 1] {
        for channel in [0usize, 1, 255, ATTENTION_LATENT - 1] {
            let value = mla_ref::kv_latent(&codes, &scales, row, channel);
            if !value.is_finite() {
                return Err(format!(
                    "harness compressed store row {row} channel {channel} decodes non-finite"
                ));
            }
        }
    }
    Ok((codes, scales))
}

/// The deterministic query rows the dispatch check launches with: 64 heads x 512 latent f32, later
/// rounded to the kernel's bf16 layout. Harness-owned input, not a model query.
fn host_query_values() -> Vec<f32> {
    (0..ATTENTION_HEADS * ATTENTION_LATENT)
        .map(|index| {
            let head = index / ATTENTION_LATENT;
            let channel = index % ATTENTION_LATENT;
            ((head as f32 * 0.07 + channel as f32 * 0.017).sin()) * 0.125
        })
        .collect()
}

fn host_query_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
    }
    bytes
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half::bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
        .collect()
}

/// The deterministic per-head sink the dispatch check launches with. Harness-owned input.
fn host_attention_sink() -> Vec<f32> {
    (0..ATTENTION_HEADS)
        .map(|head| (head as f32 * 0.05).sin() * 0.5)
        .collect()
}

/// The device store layer slot (0..4) for one of the four KV source layers.
fn store_layer_slot(layer: usize) -> HarnessResult<usize> {
    KV_SOURCE_LAYERS
        .iter()
        .position(|source| *source == layer)
        .ok_or_else(|| format!("layer {layer} is not one of the four KV source layers"))
}

/// The device allocations one sequence-plumbing run needs: forty window rings and forty oldest-first
/// views, four expanded compressed views, the query/sink/output buffers, the selected-id buffer and
/// the four indexer buffers the kernel's stage 1 still reads (it runs before the union branch, which
/// is why format 2 cannot launch with null index pointers - see Format2AttentionCall::params).
struct DeviceBuffers {
    ring_codes: Vec<Deviceptr>,
    ring_scales: Vec<Deviceptr>,
    view_codes: Vec<Deviceptr>,
    view_scales: Vec<Deviceptr>,
    kv_codes: [Deviceptr; 4],
    kv_scales: [Deviceptr; 4],
    q: Deviceptr,
    sink: Deviceptr,
    out: Deviceptr,
    selected: Deviceptr,
    index_key: Deviceptr,
    index_query: Deviceptr,
    block_scores: Deviceptr,
    block_best: Deviceptr,
    ring_codes_bytes: usize,
    ring_scales_bytes: usize,
    view_codes_bytes: usize,
    view_scales_bytes: usize,
    kv_codes_bytes: [usize; 4],
    kv_scales_bytes: [usize; 4],
    q_bytes: usize,
    sink_bytes: usize,
    out_bytes: usize,
    selected_bytes: usize,
    index_key_bytes: usize,
    index_query_bytes: usize,
    block_bytes: usize,
    allocated: usize,
}

impl DeviceBuffers {
    fn allocate(cuda: &Cuda) -> HarnessResult<Self> {
        let layout = WindowRingLayout::ENGINE;
        let ring_codes_bytes = layout.slots * layout.code_row_bytes;
        let ring_scales_bytes = layout.slots * layout.scale_row_bytes;
        let view_codes_bytes = ring_codes_bytes;
        let view_scales_bytes = ring_scales_bytes;
        let q_bytes = ATTENTION_HEADS * ATTENTION_LATENT * 2;
        let sink_bytes = ATTENTION_HEADS * 4;
        let out_bytes = q_bytes;
        let selected_bytes = FORMAT2_MAX_SELECTED * 4;
        let index_key_bytes =
            mla_ref::HSI_BLOCKS * mla_ref::HSI_BLOCK_TOKENS * mla_ref::INDEX_HEAD_DIM * 2;
        let index_query_bytes = mla_ref::MLA_HEADS * mla_ref::INDEX_HEAD_DIM * 2;
        let block_bytes = mla_ref::HSI_BLOCKS * 4;
        let mut allocated = 0usize;
        let mut ring_codes = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        let mut ring_scales = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        let mut view_codes = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        let mut view_scales = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for _ in 0..NUM_HIDDEN_LAYERS {
            ring_codes.push(cuda.alloc(ring_codes_bytes, &mut allocated)?);
            ring_scales.push(cuda.alloc(ring_scales_bytes, &mut allocated)?);
            view_codes.push(cuda.alloc(view_codes_bytes, &mut allocated)?);
            view_scales.push(cuda.alloc(view_scales_bytes, &mut allocated)?);
        }
        let mut kv_codes = [Deviceptr(0); 4];
        let mut kv_scales = [Deviceptr(0); 4];
        let mut kv_codes_bytes = [0usize; 4];
        let mut kv_scales_bytes = [0usize; 4];
        for (slot, store_layer) in KV_SOURCE_LAYERS.iter().enumerate() {
            let (_, view_rows) = compressed_view_rows(compress_ratio(*store_layer))?;
            kv_codes_bytes[slot] = view_rows * COMPRESSED_CODES_BYTES;
            kv_scales_bytes[slot] = view_rows * COMPRESSED_SCALES_BYTES;
            kv_codes[slot] = cuda.alloc(kv_codes_bytes[slot], &mut allocated)?;
            kv_scales[slot] = cuda.alloc(kv_scales_bytes[slot], &mut allocated)?;
        }
        let buffers = Self {
            ring_codes,
            ring_scales,
            view_codes,
            view_scales,
            kv_codes,
            kv_scales,
            q: cuda.alloc(q_bytes, &mut allocated)?,
            sink: cuda.alloc(sink_bytes, &mut allocated)?,
            out: cuda.alloc(out_bytes, &mut allocated)?,
            selected: cuda.alloc(selected_bytes, &mut allocated)?,
            index_key: cuda.alloc(index_key_bytes, &mut allocated)?,
            index_query: cuda.alloc(index_query_bytes, &mut allocated)?,
            block_scores: cuda.alloc(block_bytes, &mut allocated)?,
            block_best: cuda.alloc(block_bytes, &mut allocated)?,
            ring_codes_bytes,
            ring_scales_bytes,
            view_codes_bytes,
            view_scales_bytes,
            kv_codes_bytes,
            kv_scales_bytes,
            q_bytes,
            sink_bytes,
            out_bytes,
            selected_bytes,
            index_key_bytes,
            index_query_bytes,
            block_bytes,
            allocated,
        };
        if let Some(available) = sparkflash_kernels::cuda_driver::mem_available_kb()
            && buffers.allocated as u64 > available.saturating_mul(2)
        {
            return Err(format!(
                "the planned {} B of device buffers does not fit the {available} KB available",
                buffers.allocated
            ));
        }
        Ok(buffers)
    }

    fn all_ptrs(&self) -> Vec<Deviceptr> {
        let mut ptrs = self.ring_codes.clone();
        ptrs.extend_from_slice(&self.ring_scales);
        ptrs.extend_from_slice(&self.view_codes);
        ptrs.extend_from_slice(&self.view_scales);
        ptrs.extend_from_slice(&self.kv_codes);
        ptrs.extend_from_slice(&self.kv_scales);
        ptrs.extend([
            self.q,
            self.sink,
            self.out,
            self.selected,
            self.index_key,
            self.index_query,
            self.block_scores,
            self.block_best,
        ]);
        ptrs
    }
}

/// The host side of one dispatch check: everything the CPU twin needs to reproduce the device
/// output from the same device-format bytes.
struct TwinCase<'a> {
    q_values: &'a [f32],
    sink: &'a [f32],
    window_codes: &'a [u8],
    window_scales: &'a [u8],
    compressed: Option<(&'a [u8], &'a [u8])>,
}

/// The device-vs-twin comparison for one dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TwinStats {
    heads: usize,
    values: usize,
    max_abs: f32,
    max_abs_head: usize,
    max_abs_channel: usize,
    exact_bf16: usize,
    non_finite: usize,
}

impl TwinStats {
    fn within_bar(&self) -> bool {
        self.non_finite == 0 && self.max_abs <= ATTENTION_TWIN_BAR
    }

    fn line(&self) -> String {
        format!(
            "device-vs-twin: {} values over {} heads, max_abs {:.6e} at head {} channel {} (bar {ATTENTION_TWIN_BAR:.1e}), exact bf16 {}/{} , non-finite {}",
            self.values,
            self.heads,
            self.max_abs,
            self.max_abs_head,
            self.max_abs_channel,
            self.exact_bf16,
            self.values,
            self.non_finite
        )
    }
}

/// The CPU union/window twin over the exact device-format inputs. It is the same three-pass order
/// the kernel runs, so it localises a dispatch/layout defect; it is not the engine reference and no
/// golden-fixture bar is applied to it.
fn twin_attention(dispatch: &AttentionDispatch, case: &TwinCase<'_>) -> HarnessResult<Vec<f32>> {
    let mut out = vec![0.0f32; ATTENTION_HEADS * ATTENTION_LATENT];
    let position = u32::try_from(dispatch.position)
        .map_err(|_| "dispatch position does not fit a u32".to_string())?;
    for head in 0..ATTENTION_HEADS {
        let q = &case.q_values[head * ATTENTION_LATENT..(head + 1) * ATTENTION_LATENT];
        let sink = case.sink[head];
        let head_out = match dispatch.window_kv_format {
            CSA_UNION_WINDOW_KV_FORMAT => {
                let (kv_codes, kv_scales) = case.compressed.ok_or_else(|| {
                    format!(
                        "layer {} position {}: union dispatch has no compressed view",
                        dispatch.layer, dispatch.position
                    )
                })?;
                mla_ref::attention_with_sink_union(
                    q,
                    case.window_codes,
                    case.window_scales,
                    dispatch.window_rows,
                    kv_codes,
                    kv_scales,
                    &dispatch.selected,
                    position,
                    sink,
                )
            }
            ENGINE_WINDOW_KV_FORMAT => mla_ref::attention_with_sink_window(
                q,
                case.window_codes,
                case.window_scales,
                dispatch.window_rows,
                &dispatch.selected,
                sink,
            ),
            other => {
                return Err(format!("unsupported dispatch format {other}"));
            }
        };
        out[head * ATTENTION_LATENT..(head + 1) * ATTENTION_LATENT].copy_from_slice(&head_out);
    }
    Ok(out)
}

/// Compares the bf16 device output against the f32 twin: exact bf16 matches, worst absolute
/// difference and any non-finite value.
fn compare_device_out(device: &[u8], twin: &[f32]) -> HarnessResult<TwinStats> {
    if device.len() != twin.len() * 2 {
        return Err(format!(
            "device output is {} B, the twin has {} values",
            device.len(),
            twin.len()
        ));
    }
    let mut stats = TwinStats {
        heads: ATTENTION_HEADS,
        values: twin.len(),
        max_abs: 0.0,
        max_abs_head: 0,
        max_abs_channel: 0,
        exact_bf16: 0,
        non_finite: 0,
    };
    for (index, pair) in device.as_chunks::<2>().0.iter().enumerate() {
        let bits = u16::from_le_bytes(*pair);
        let value = half::bf16::from_bits(bits).to_f32();
        let expected = twin[index];
        if !value.is_finite() || !expected.is_finite() {
            stats.non_finite += 1;
            continue;
        }
        if half::bf16::from_f32(expected).to_bits() == bits {
            stats.exact_bf16 += 1;
        }
        let delta = (value - expected).abs();
        if delta > stats.max_abs {
            stats.max_abs = delta;
            stats.max_abs_head = index / ATTENTION_LATENT;
            stats.max_abs_channel = index % ATTENTION_LATENT;
        }
    }
    Ok(stats)
}

/// Builds the pinned 19-word launch parameters for one dispatch. Union (format 2) calls build the
/// shared [`Format2AttentionCall`] from the real device buffers and take its `params()`, so the
/// indexer pointers and the union bounds are checked by the same builder the isolated check uses.
/// Window-only (format 1) calls go through [`flash_mla_params`] directly, because the shared builder
/// is format-2 only; check_dispatch bounds both. Either way the 19-word order comes from
/// `flash_mla_params`, never from this file.
///
/// The four indexer buffers are real zeroed device allocations: the kernel's stage 1 (HSI block
/// scoring) runs before the union branch and dereferences them, so a null there would be an illegal
/// device access. The selection itself stays the harness-supplied list, which is the labelled host
/// seam.
///
/// The production cubin is built without tracing, so the trace ring word is zero.
fn dispatch_params(
    dispatch: &AttentionDispatch,
    buffers: &DeviceBuffers,
) -> HarnessResult<FlashMlaParams> {
    check_dispatch(dispatch)?;
    let (kv_codes, kv_scales) = match dispatch.store_layer {
        Some(store_layer) => {
            let slot = store_layer_slot(store_layer)?;
            (buffers.kv_codes[slot], buffers.kv_scales[slot])
        }
        // Format 1 never reads the compressed arrays; a zero word keeps that explicit.
        None => (Deviceptr(0), Deviceptr(0)),
    };
    if dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT {
        let call = Format2AttentionCall {
            q: buffers.q,
            kv_fp4: kv_codes,
            kv_scales,
            window_codes: buffers.view_codes[dispatch.layer],
            window_scales: buffers.view_scales[dispatch.layer],
            sink: buffers.sink,
            selected: buffers.selected,
            out: buffers.out,
            position: u32::try_from(dispatch.position)
                .map_err(|_| "dispatch position does not fit the kernel parameter".to_string())?,
            layer: dispatch.layer,
            step_seq: 0,
            window_rows: u32::try_from(dispatch.window_rows).map_err(|_| {
                "dispatch window_rows does not fit the kernel parameter".to_string()
            })?,
            selected_ids: &dispatch.selected,
            trace_ring: None,
            index_key: buffers.index_key,
            index_query: buffers.index_query,
            block_scores: buffers.block_scores,
            block_best: buffers.block_best,
        };
        return call.params();
    }
    Ok(flash_mla_params(
        buffers.q.0,
        kv_codes.0,
        kv_scales.0,
        buffers.view_codes[dispatch.layer].0,
        buffers.view_scales[dispatch.layer].0,
        u64::from(dispatch.window_kv_format),
        dispatch.window_rows as u64,
        buffers.selected.0,
        buffers.index_key.0,
        buffers.index_query.0,
        buffers.sink.0,
        buffers.block_scores.0,
        buffers.block_best.0,
        buffers.out.0,
        dispatch.position as u64,
        dispatch.selected_count(),
        0,
        dispatch.layer,
        0,
    ))
}

/// One dispatch's measured record: what ran, whether the device output matched the twin (prefill
/// position only) and whether a repeated launch was byte-identical (decode positions).
#[derive(Debug, Clone)]
struct DispatchRecord {
    layer: usize,
    position: usize,
    format: u32,
    window_rows: u32,
    selected_count: u32,
    twin: Option<TwinStats>,
    repeat_identical: Option<bool>,
    /// Launches this record issued: one, plus one for the determinism repeat.
    launches: usize,
    non_finite: usize,
    elapsed: Duration,
}

/// Runs one dispatch on the device: upload the selected ids, launch through the one typed path, read
/// the output, optionally repeat the launch to prove determinism, and optionally compare against the
/// CPU twin. No host arithmetic feeds the device; every byte it reads was uploaded earlier.
fn run_dispatch(
    cuda: &Cuda,
    kernel: FlashMlaKernel,
    buffers: &DeviceBuffers,
    dispatch: &AttentionDispatch,
    twin_case: Option<&TwinCase<'_>>,
    repeat: bool,
) -> HarnessResult<DispatchRecord> {
    let mut selected_bytes = Vec::with_capacity(dispatch.selected.len() * 4);
    for id in &dispatch.selected {
        selected_bytes.extend_from_slice(&id.to_le_bytes());
    }
    cuda.copy_htod(&buffers.selected, &selected_bytes)
        .map_err(|error| format!("upload selected ids: {error}"))?;
    let params = dispatch_params(dispatch, buffers)?;
    let start = Instant::now();
    launch_flash_mla(
        cuda,
        kernel,
        (ATTENTION_HEADS as u32, 1u32, 1u32),
        (ATTENTION_THREADS, 1u32, 1u32),
        &params,
    )?;
    let mut out = vec![0u8; buffers.out_bytes];
    cuda.copy_dtoh(&mut out, &buffers.out)
        .map_err(|error| format!("read attention output: {error}"))?;
    let elapsed = start.elapsed();
    let non_finite = out
        .as_chunks::<2>()
        .0
        .iter()
        .filter(|pair| {
            !half::bf16::from_bits(u16::from_le_bytes(**pair))
                .to_f32()
                .is_finite()
        })
        .count();
    let repeat_identical = if repeat {
        launch_flash_mla(
            cuda,
            kernel,
            (ATTENTION_HEADS as u32, 1u32, 1u32),
            (ATTENTION_THREADS, 1u32, 1u32),
            &params,
        )?;
        let mut again = vec![0u8; buffers.out_bytes];
        cuda.copy_dtoh(&mut again, &buffers.out)
            .map_err(|error| format!("read repeated attention output: {error}"))?;
        Some(again == out)
    } else {
        None
    };
    let twin = match twin_case {
        Some(case) => {
            let expected = twin_attention(dispatch, case)?;
            Some(compare_device_out(&out, &expected)?)
        }
        None => None,
    };
    Ok(DispatchRecord {
        launches: 1 + usize::from(repeat),
        layer: dispatch.layer,
        position: dispatch.position,
        format: dispatch.window_kv_format,
        window_rows: dispatch.window_rows as u32,
        selected_count: dispatch.selected_count() as u32,
        twin,
        repeat_identical,
        non_finite,
        elapsed,
    })
}

/// Reads one rotated view back and compares it byte-for-byte with the host's oldest-first
/// expectation. This is the rotation-order check: it proves the device view starts at the slot the
/// ring math says and wraps where it says, independently of the attention twin.
fn verify_rotated_view(
    cuda: &Cuda,
    view_codes: &Deviceptr,
    view_scales: &Deviceptr,
    expected_codes: &[u8],
    expected_scales: &[u8],
    context: &str,
) -> HarnessResult<usize> {
    let mut codes = vec![0u8; expected_codes.len()];
    let mut scales = vec![0u8; expected_scales.len()];
    cuda.copy_dtoh(&mut codes, view_codes)
        .map_err(|error| format!("{context}: read rotated codes: {error}"))?;
    cuda.copy_dtoh(&mut scales, view_scales)
        .map_err(|error| format!("{context}: read rotated scales: {error}"))?;
    if let Some(index) = codes
        .iter()
        .zip(expected_codes)
        .position(|(got, want)| got != want)
    {
        return Err(format!(
            "{context}: rotated code byte {index} is {}, expected {}",
            codes[index], expected_codes[index]
        ));
    }
    if let Some(index) = scales
        .iter()
        .zip(expected_scales)
        .position(|(got, want)| got != want)
    {
        return Err(format!(
            "{context}: rotated scale byte {index} is {}, expected {}",
            scales[index], expected_scales[index]
        ));
    }
    Ok(codes.len() + scales.len())
}

/// The device plumbing run: the full 130-prefill + 8-decode schedule's window rotation, measured with
/// real device-to-device copies, plus the format-2/format-1 attention dispatch at the compared
/// positions over harness-owned inputs. It is deliberately not a model forward: no weights are read,
/// no token is produced and no logits comparison is attempted. The report says exactly that.
fn run_device(args: &Args) -> HarnessResult<String> {
    let checkpoint = args
        .checkpoint
        .as_ref()
        .ok_or_else(|| "--device needs --checkpoint".to_string())?;
    let index = args
        .index
        .as_ref()
        .ok_or_else(|| "--device needs --index".to_string())?;
    let golden = args
        .golden
        .as_ref()
        .ok_or_else(|| "--device needs --golden".to_string())?;
    let cubin_dir = args
        .cubin_dir
        .as_ref()
        .ok_or_else(|| "--device needs --cubin-dir".to_string())?;
    if !checkpoint.is_dir() {
        return Err(format!(
            "{}: checkpoint is not a directory",
            checkpoint.display()
        ));
    }
    let config_path = checkpoint.join("config.json");
    if !config_path.is_file() {
        return Err(format!("{}: missing config.json", checkpoint.display()));
    }
    if !index.is_file() {
        return Err(format!("{}: index is not a file", index.display()));
    }
    let cubin_path = cubin_dir.join("flash_mla_csa2_sm121.cubin");
    let cubin =
        std::fs::read(&cubin_path).map_err(|error| format!("{}: {error}", cubin_path.display()))?;
    let cubin_sha = sha256_hex(&cubin);
    let fixture_bytes =
        std::fs::read(golden).map_err(|error| format!("{}: {error}", golden.display()))?;
    let fixture_sha = sha256_hex(&fixture_bytes);
    let fixture = load_fixture(golden)?;
    check_fixture_shape(&fixture)?;
    let schedule = checked_schedule(args.prompt_tokens, args.no_ced)?;
    let rotation = RotationPlan::for_schedule(&schedule);
    let compared = compared_set(args.prefill_only);
    let plan = dispatch_plan(args.attention_mode, &compared)?;
    let dispatch_index: HashMap<(usize, usize), usize> = plan
        .iter()
        .enumerate()
        .map(|(index, dispatch)| ((dispatch.position, dispatch.layer), index))
        .collect();
    let compared_set: BTreeSet<usize> = compared.iter().copied().collect();
    // Host-owned deterministic inputs and the expected oldest-first views they imply.
    let ring_images: Vec<(Vec<u8>, Vec<u8>)> = (0..NUM_HIDDEN_LAYERS)
        .map(host_ring_image)
        .collect::<HarnessResult<Vec<_>>>()?;
    let mut expanded_views: Vec<ExpandedView> = Vec::with_capacity(KV_SOURCE_LAYERS.len());
    for store_layer in KV_SOURCE_LAYERS {
        let ratio = compress_ratio(store_layer);
        let (store_rows, _) = compressed_view_rows(ratio)?;
        let (codes, scales) = host_compressed_store(store_layer, store_rows)?;
        expanded_views.push(expand_compressed_view(&codes, &scales, store_rows, ratio)?);
    }
    let q_values = host_query_values();
    let q_bytes = host_query_bf16_bytes(&q_values);
    // The twin must read the same bf16 query the kernel reads, not the pre-rounding f32 values.
    let q_device_values = bf16_bytes_to_f32(&q_bytes);
    let sink_values = host_attention_sink();
    let mut sink_bytes = Vec::with_capacity(sink_values.len() * 4);
    for value in &sink_values {
        sink_bytes.extend_from_slice(&value.to_le_bytes());
    }

    let cuda = Cuda::load().map_err(|error| format!("cuda driver: {error}"))?;
    let (module, kernel) =
        load_flash_mla(&cuda, &cubin).map_err(|error| format!("load flash_mla cubin: {error}"))?;
    let buffers = DeviceBuffers::allocate(&cuda)?;
    let run = (|| -> HarnessResult<String> {
        // Upload the harness-owned window rings and compressed views, the query and the sink; zero
        // the indexer buffers the kernel's stage 1 writes.
        for (layer, (codes, scales)) in ring_images.iter().enumerate() {
            cuda.copy_htod(&buffers.ring_codes[layer], codes)
                .map_err(|error| format!("upload ring codes layer {layer}: {error}"))?;
            cuda.copy_htod(&buffers.ring_scales[layer], scales)
                .map_err(|error| format!("upload ring scales layer {layer}: {error}"))?;
        }
        for (slot, store_layer) in KV_SOURCE_LAYERS.iter().enumerate() {
            cuda.copy_htod(&buffers.kv_codes[slot], &expanded_views[slot].codes)
                .map_err(|error| format!("upload compressed codes layer {store_layer}: {error}"))?;
            cuda.copy_htod(&buffers.kv_scales[slot], &expanded_views[slot].scales)
                .map_err(|error| {
                    format!("upload compressed scales layer {store_layer}: {error}")
                })?;
        }
        cuda.copy_htod(&buffers.q, &q_bytes)
            .map_err(|error| format!("upload q: {error}"))?;
        cuda.copy_htod(&buffers.sink, &sink_bytes)
            .map_err(|error| format!("upload sink: {error}"))?;
        let zeros = vec![0u8; buffers.index_key_bytes.max(buffers.index_query_bytes)];
        cuda.copy_htod(&buffers.index_key, &zeros[..buffers.index_key_bytes])
            .map_err(|error| format!("zero index_key: {error}"))?;
        cuda.copy_htod(&buffers.index_query, &zeros[..buffers.index_query_bytes])
            .map_err(|error| format!("zero index_query: {error}"))?;
        cuda.copy_htod(&buffers.block_scores, &zeros[..buffers.block_bytes])
            .map_err(|error| format!("zero block_scores: {error}"))?;
        cuda.copy_htod(&buffers.block_best, &zeros[..buffers.block_bytes])
            .map_err(|error| format!("zero block_best: {error}"))?;

        let layout = WindowRingLayout::ENGINE;
        let mut records: Vec<DispatchRecord> = Vec::with_capacity(plan.len());
        let mut view_checks = 0usize;
        let mut view_bytes_verified = 0usize;
        let measurement = measure_rotation(&schedule, rotation, |layer, position| {
            let (window_rows, oldest) = window_view(&layout, &schedule, layer, position)?;
            let copy_start = Instant::now();
            let copies = rotate_window_ring(
                &cuda,
                &buffers.view_codes[layer],
                &buffers.view_scales[layer],
                &buffers.ring_codes[layer],
                &buffers.ring_scales[layer],
                &layout,
                window_rows,
                oldest,
            )?;
            let copy_time = copy_start.elapsed();
            let bytes = window_rows * (WINDOW_CODE_ROW_BYTES + WINDOW_SCALE_ROW_BYTES);
            if compared_set.contains(&position) {
                let (expected_codes, expected_scales) = host_ring_view(
                    &ring_images[layer].0,
                    &ring_images[layer].1,
                    &layout,
                    window_rows,
                    oldest,
                )?;
                let context = format!("layer {layer} position {position}");
                view_bytes_verified += verify_rotated_view(
                    &cuda,
                    &buffers.view_codes[layer],
                    &buffers.view_scales[layer],
                    &expected_codes,
                    &expected_scales,
                    &context,
                )?;
                view_checks += 1;
                let index = *dispatch_index.get(&(position, layer)).ok_or_else(|| {
                    format!("no dispatch planned for layer {layer} position {position}")
                })?;
                let dispatch = &plan[index];
                let twin_owned;
                let twin_case = if position == PREFILL_POSITION {
                    let compressed = match dispatch.store_layer {
                        Some(store_layer) => {
                            let slot = store_layer_slot(store_layer)?;
                            Some((
                                expanded_views[slot].codes.as_slice(),
                                expanded_views[slot].scales.as_slice(),
                            ))
                        }
                        None => None,
                    };
                    twin_owned = TwinCase {
                        q_values: &q_device_values,
                        sink: &sink_values,
                        window_codes: &expected_codes,
                        window_scales: &expected_scales,
                        compressed,
                    };
                    Some(&twin_owned)
                } else {
                    None
                };
                records.push(run_dispatch(
                    &cuda,
                    kernel,
                    &buffers,
                    dispatch,
                    twin_case,
                    position != PREFILL_POSITION,
                )?);
            }
            Ok((copies, bytes, copy_time))
        })?;

        let mut lines = report_header(
            "device plumbing run (window rotation + format-2/1 attention dispatch; NOT a model forward)",
            args,
        )?;
        lines.push(format!(
            "fixture: {} ({} B, sha256 {})",
            golden.display(),
            fixture_bytes.len(),
            fixture_sha
        ));
        lines.push(format!(
            "cubin: {} ({} B, sha256 {})",
            cubin_path.display(),
            cubin.len(),
            cubin_sha
        ));
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("{}: {error}", config_path.display()))?;
        let config = ModelConfig::from_json(&config_text)
            .map_err(|error| format!("{}: {error}", config_path.display()))?;
        lines.push(format!(
            "checkpoint: {} (config: {} layers, dim {}, {} attention heads, window {})",
            checkpoint.display(),
            config.n_layers,
            config.dim,
            config.n_heads,
            config.window_size
        ));
        lines.push(format!(
            "cuda: driver loaded, flash_mla handle loaded from the cubin; allocated {} B across {} buffers",
            buffers.allocated,
            buffers.all_ptrs().len()
        ));
        lines.push(format!(
            "device buffers: per layer rings {} B codes + {} B scales and views {} B + {} B; compressed views {:?} B codes + {:?} B scales; q {} B, out {} B, sink {} B, selected {} B, index_key {} B, index_query {} B, block_scores/block_best {} B each",
            buffers.ring_codes_bytes,
            buffers.ring_scales_bytes,
            buffers.view_codes_bytes,
            buffers.view_scales_bytes,
            buffers.kv_codes_bytes,
            buffers.kv_scales_bytes,
            buffers.q_bytes,
            buffers.out_bytes,
            buffers.sink_bytes,
            buffers.selected_bytes,
            buffers.index_key_bytes,
            buffers.index_query_bytes,
            buffers.block_bytes
        ));
        lines.push(String::new());
        lines.push("--- schedule / store / rotation (checked on the host) ---".to_string());
        lines.push(format!(
            "schedule: prompt {} tokens, encoder {} tokens on {} layers, decoder {} tokens on {} layers, window {}, exclusion boundary {}",
            schedule.prompt_tokens,
            schedule.encoder_tokens,
            schedule.encoder_layers,
            schedule.decoder_tokens,
            schedule.decoder_layers,
            schedule.window,
            schedule.exclusion_boundary
        ));
        lines.push(rotation.line());
        lines.push(measurement.line());
        lines.push(format!(
            "rotation order check: {view_checks} oldest-first views read back and compared byte-for-byte with the host slot order ({} B verified), zero mismatches",
            view_bytes_verified
        ));
        lines.push(format!(
            "rotation per-execution bound: at most {} B (one full ring); the measured total above sums each view's rows",
            rotation.bytes_per_execution_max()
        ));
        lines.push(String::new());
        lines.push("--- attention dispatch (device; harness-owned inputs) ---".to_string());
        let union_calls = records
            .iter()
            .filter(|record| record.format == CSA_UNION_WINDOW_KV_FORMAT)
            .count();
        let twin_records: Vec<&DispatchRecord> = records
            .iter()
            .filter(|record| record.twin.is_some())
            .collect();
        let repeat_records: Vec<&DispatchRecord> = records
            .iter()
            .filter(|record| record.repeat_identical.is_some())
            .collect();
        let non_finite_total: usize = records.iter().map(|record| record.non_finite).sum();
        let twin_failures = twin_records
            .iter()
            .filter(|record| match record.twin {
                Some(stats) => !stats.within_bar(),
                None => true,
            })
            .count();
        let repeat_failures = repeat_records
            .iter()
            .filter(|record| record.repeat_identical == Some(false))
            .count();
        if non_finite_total != 0 {
            return Err(format!(
                "{non_finite_total} non-finite bf16 values in the device attention output; the dispatch is not usable"
            ));
        }
        if repeat_failures != 0 {
            return Err(format!(
                "{repeat_failures} repeated launches were not byte-identical; the dispatch is not deterministic"
            ));
        }
        if twin_failures != 0 {
            return Err(format!(
                "{twin_failures} dispatch(es) exceeded the device-vs-twin bar {ATTENTION_TWIN_BAR}"
            ));
        }
        lines.push(format!(
            "dispatches: {} calls ({} union format {CSA_UNION_WINDOW_KV_FORMAT}, {} window-only format {ENGINE_WINDOW_KV_FORMAT}) over {} compared positions x {} layers; grid ({} heads, 1, 1), block ({}, 1, 1)",
            records.len(),
            union_calls,
            records.len() - union_calls,
            compared.len(),
            NUM_HIDDEN_LAYERS,
            ATTENTION_HEADS,
            ATTENTION_THREADS
        ));
        lines.push(format!(
            "dispatch checks: {non_finite_total} non-finite outputs; {} device-vs-twin comparisons at position {PREFILL_POSITION} ({} within bar {ATTENTION_TWIN_BAR:.1e}); {} repeated-launch determinism checks ({} identical)",
            twin_records.len(),
            twin_records.len() - twin_failures,
            repeat_records.len(),
            repeat_records.len() - repeat_failures
        ));
        if let Some(worst) = twin_records
            .iter()
            .filter_map(|record| record.twin.map(|stats| (*record, stats)))
            .max_by(|left, right| left.1.max_abs.total_cmp(&right.1.max_abs))
        {
            lines.push(format!(
                "worst device-vs-twin: {} (layer {} position {})",
                worst.1.line(),
                worst.0.layer,
                worst.0.position
            ));
        }
        let dispatch_total: Duration = records.iter().map(|record| record.elapsed).sum();
        let launches: usize = records.iter().map(|record| record.launches).sum();
        let repeats = launches.saturating_sub(records.len());
        lines.push(format!(
            "dispatch wall time: {:.6} s over {} first-launch records (+{repeats} untimed repeat launches, {launches} launches issued in all; {:.3} us per record), measured around each first launch + output read",
            dispatch_total.as_secs_f64(),
            records.len(),
            if records.is_empty() {
                0.0
            } else {
                dispatch_total.as_secs_f64() * 1e6 / records.len() as f64
            }
        ));
        for record in &records {
            lines.push(format!(
                "  layer {:02} position {}: format {}, window_rows {}, selected_count {}, union entries (window + selected) {}, non-finite {}{}{}",
                record.layer,
                record.position,
                record.format,
                record.window_rows,
                record.selected_count,
                record.window_rows as usize + record.selected_count as usize,
                record.non_finite,
                record
                    .twin
                    .as_ref()
                    .map(|stats| format!(", twin max_abs {:.6e}", stats.max_abs))
                    .unwrap_or_default(),
                record
                    .repeat_identical
                    .map(|identical| format!(", repeated launch identical {identical}"))
                    .unwrap_or_default()
            ));
        }
        lines.push(String::new());
        lines.push("--- host seams (labelled; not device work) ---".to_string());
        lines.extend(seam_lines(&plan, &compared));
        lines.push(String::new());
        lines.push("--- per-step comparison (tokens first, then logits) ---".to_string());
        for report in device_step_reports(&fixture, &compared) {
            lines.push(report.line());
        }
        lines.push(String::new());
        lines.push("--- status ---".to_string());
        lines.push(format!(
            "STATUS: the window rotation was measured on the device for the full 130-prefill + {DECODE_STEPS}-decode schedule and the format-2/format-1 attention dispatch ran at the {} compared positions on harness-owned inputs. The model forward is NOT wired, so NO token and NO logits delta was produced: the {}-position comparison against the golden fixture was NOT ATTEMPTED.",
            compared.len(),
            compared.len()
        ));
        lines.push(
            "STATUS: Phase 2 exit criterion 1 remains OPEN. This report is not evidence about the reference model's numbers and must not be quoted as a 2B pass."
                .to_string(),
        );
        lines.push(String::new());
        lines.extend(honesty_lines(&compared));
        Ok(lines.join("\n"))
    })();
    cuda.free(module, &buffers.all_ptrs());
    run
}

/// Runs the engine reference prefill with boundary capture and writes the v1 manifest+blob.
///
/// This is the reference half of the incremental cross-check: the device chain compares each op
/// against this table, so the first divergence names a layer/op instead of only a final logits
/// delta. It is CPU-only and never runs under `--device`; the released-model dump is a node job.
fn run_boundary_dump(args: &Args) -> HarnessResult<String> {
    let golden = args
        .golden
        .as_ref()
        .ok_or_else(|| "--boundary-dump needs --golden".to_string())?;
    let fixture = load_fixture(golden)?;
    check_fixture_shape(&fixture)?;
    let checkpoint = args
        .checkpoint
        .as_ref()
        .ok_or_else(|| "--boundary-dump needs --checkpoint".to_string())?;
    let index = args
        .index
        .as_ref()
        .ok_or_else(|| "--boundary-dump needs --index".to_string())?;
    let config_path = checkpoint.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let config = ModelConfig::from_json(&config_text)
        .map_err(|error| format!("{}: {error}", config_path.display()))?;
    let source = sparkflash_engine::safetensors::SafeTensorsSource::open(checkpoint, index)
        .map_err(|error| format!("checkpoint source: {error}"))?;
    let engine = sparkflash_engine::forward::InferenceEngine::new(
        config.clone(),
        Box::new(source),
        BOUNDARY_DUMP_TENSOR_LIMIT,
    )
    .map_err(|error| format!("engine: {error}"))?;
    let mut state = sparkflash_engine::forward::SequenceState::new(&config);
    let observed = engine
        .prefill_observed(&mut state, &fixture.prompt_tokens)
        .map_err(|error| format!("boundary prefill: {error}"))?;
    let expected_token = fixture.steps[0].next_token;
    if observed.output.next_token != expected_token {
        return Err(format!(
            "reference boundary prefill selected {:?}, fixture says {expected_token:?}",
            observed.output.next_token
        ));
    }
    if observed.boundary.rows().len() != config.n_layers {
        return Err(format!(
            "reference boundary table has {} rows, expected {} backbone layers",
            observed.boundary.rows().len(),
            config.n_layers
        ));
    }
    let dir = args
        .boundary_dump
        .as_ref()
        .ok_or_else(|| "--boundary-dump path missing".to_string())?;
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let manifest_path = dir.join("boundary.manifest");
    let blob_path = dir.join("boundary.blob");
    let manifest = std::fs::File::create(&manifest_path)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    let blob = std::fs::File::create(&blob_path)
        .map_err(|error| format!("{}: {error}", blob_path.display()))?;
    let mut manifest_writer = std::io::BufWriter::new(manifest);
    let mut blob_writer = std::io::BufWriter::new(blob);
    observed
        .boundary
        .write_v1(config.n_layers, &mut manifest_writer, &mut blob_writer)
        .map_err(|error| format!("boundary write: {error}"))?;
    drop(manifest_writer);
    drop(blob_writer);
    let manifest_bytes = std::fs::metadata(&manifest_path)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?
        .len();
    let blob_bytes = std::fs::metadata(&blob_path)
        .map_err(|error| format!("{}: {error}", blob_path.display()))?
        .len();
    Ok(format!(
        "boundary dump: {} rows over {} prompt tokens, position {} token {:?} matches fixture, manifest {} B, blob {} B at {}",
        observed.boundary.rows().len(),
        fixture.prompt_tokens.len(),
        observed.boundary.start_position() + observed.boundary.tokens().len() - 1,
        observed.output.next_token,
        manifest_bytes,
        blob_bytes,
        dir.display()
    ))
}

/// The binary entry point. The three exclusive modes are checked in the CLI: the host dry-run
/// (CPU-only, writes its own report), the device plumbing run, and the default self-check plus the
/// optional boundary dump. The report is always written through one function so stdout and --out
/// carry the same bytes.
fn run(args: &Args) -> HarnessResult<Option<String>> {
    if args.dry_run {
        let report = run_dry_run(args)?;
        write_report_text(args, &report)?;
        return Ok(Some(report));
    }
    if args.device {
        let report = run_device(args)?;
        write_report_text(args, &report)?;
        return Ok(Some(report));
    }
    let mut report = Vec::new();
    if args.self_check {
        report.push(run_self_check(args)?);
    }
    if args.boundary_dump.is_some() {
        report.push(run_boundary_dump(args)?);
    }
    let report = report.join("\n");
    write_report_text(args, &report)?;
    Ok(Some(report))
}

fn main() {
    let args = match Args::parse_from(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(Some(report)) => println!("{report}"),
        Ok(None) => {}
        Err(message) => {
            eprintln!("m8_sequence error: {message}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_engine::TensorPlan;
    use sparkflash_engine::forward::attention::{
        AttnState, attention_forward, compressed_indices, window_indices,
    };
    use sparkflash_engine::synthetic::SyntheticWeights;

    fn parse(flags: &[&str]) -> HarnessResult<Args> {
        Args::parse_from(flags.iter().map(|flag| flag.to_string()))
    }

    #[test]
    fn cli_pins_the_prompt_and_decode_counts() {
        assert!(parse(&["--prompt-tokens", "129"]).is_err());
        assert!(parse(&["--decode-steps", "7"]).is_err());
        let args = parse(&["--attention-mode", "window-only", "--no-ced"]).expect("parses");
        assert_eq!(args.attention_mode, AttentionMode::WindowOnly);
        assert!(args.no_ced);
        assert!(!args.prefill_only);
        assert!(parse(&["--prefill-only"]).expect("parses").prefill_only);
        assert!(parse(&["--attention-mode", "both"]).is_err());
        assert!(parse(&["--max-seq-len", "0"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
    }

    #[test]
    fn schedule_and_mode_table_are_pinned() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        assert_eq!(schedule.encoder_tokens, 130);
        assert_eq!(schedule.decoder_tokens, 128);
        assert_eq!(schedule.encoder_layers, 21);
        assert_eq!(schedule.decoder_layers, 19);
        assert_eq!(schedule.exclusion_boundary, 2);
        assert_eq!(schedule.publications.len(), 17);
        let control = checked_schedule(PROMPT_TOKENS, true).expect("control schedule");
        assert_eq!(control.decoder_tokens, PROMPT_TOKENS);
        assert_eq!(control.modes, schedule.modes);
        assert_eq!(control.publications, schedule.publications);
    }

    #[test]
    fn layer_token_extents_match_the_ced_split() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        assert_eq!(
            layer_token_extent(20, 130, schedule.decoder_tokens).len(),
            130
        );
        assert_eq!(
            layer_token_extent(21, 130, schedule.decoder_tokens),
            (2..130).collect::<Vec<_>>()
        );
        assert_eq!(
            layer_token_extent(39, 130, schedule.decoder_tokens).len(),
            128
        );
        let control = checked_schedule(PROMPT_TOKENS, true).expect("control");
        assert_eq!(
            layer_token_extent(21, 130, control.decoder_tokens).len(),
            130
        );
    }

    #[test]
    fn rotation_plan_counts_the_rows_each_execution_copies() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let plan = RotationPlan::for_schedule(&schedule);
        assert_eq!(plan.layers, NUM_HIDDEN_LAYERS);
        assert_eq!(plan.slots_per_ring, WINDOW);
        assert_eq!(plan.exec_prefill, 21 * 130 + 19 * 128);
        assert_eq!(plan.exec_prefill, 5_162);
        assert_eq!(plan.exec_decode, DECODE_STEPS * NUM_HIDDEN_LAYERS);
        assert_eq!(plan.exec_decode, 320);
        assert_eq!(plan.executions(), 5_482);
        // One full ring is the store geometry, not the per-execution unit.
        assert_eq!(plan.bytes_per_layer(), WINDOW * 528);
        assert_eq!(plan.bytes_per_layer(), 67_584);
        assert_eq!(plan.ring_bytes(), ROTATION_BYTES_PER_STEP);
        assert_eq!(plan.ring_bytes(), 2_703_360);
        // The corrected total: every slot an oldest-first view holds, summed over executions.
        // Encoder layers start at 0, decoder layers at the CED boundary 2, so decoder views hold
        // 253 fewer slots each than a min(position + 1) formula says (4,807 slots in total).
        assert_eq!(plan.slots_copied, 376_576);
        assert_eq!(plan.codes_bytes(), 192_806_912);
        assert_eq!(plan.scales_bytes(), 6_025_216);
        assert_eq!(plan.bytes(), 198_832_128);
        assert_eq!(plan.slots_copied, 340_423 + 40_960 - 19 * 253);
        // A skeleton-style full-ring x executions total would be far larger; pin the difference so
        // the over-report cannot come back quietly.
        assert_ne!(plan.bytes(), plan.ring_bytes() * plan.executions());
        assert_eq!(plan.copy_calls_upper_bound(), plan.executions() * 4);
        let line = plan.line();
        assert!(line.contains("5482"), "line: {line}");
        assert!(line.contains("198832128"), "line: {line}");
    }

    #[test]
    fn rotation_slots_are_the_independent_closed_form_per_layer_split() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let plan = RotationPlan::for_schedule(&schedule);
        // Independent derivation, not the plan's loop: an encoder layer executes positions 0..=129
        // then 8 decodes, and its view at p is min(p + 1, 128); a decoder layer executes 2..=129
        // then 8 decodes, and its view at p is min(p - 1, 128).
        //   encoder: 1 + 2 + ... + 128 = 8256, then 128 at p=128 and 129, then 8 x 128
        //   decoder: 1 + 2 + ... + 126 = 8001, then 127 at p=128 and 128 at p=129, then 8 x 128
        let encoder_layer = 128 * 129 / 2 + 128 + 128 + DECODE_STEPS * WINDOW;
        let decoder_layer = 126 * 127 / 2 + 127 + 128 + DECODE_STEPS * WINDOW;
        assert_eq!(encoder_layer, 9_536);
        assert_eq!(decoder_layer, 9_280);
        let encoder_slots = schedule.encoder_layers * encoder_layer;
        let decoder_slots = schedule.decoder_layers * decoder_layer;
        assert_eq!(encoder_slots, 200_256);
        assert_eq!(decoder_slots, 176_320);
        assert_eq!(encoder_slots + decoder_slots, 376_576);
        assert_eq!(plan.slots_copied, encoder_slots + decoder_slots);
        assert_eq!(plan.bytes(), 376_576 * 528);
        assert_eq!(plan.bytes(), 198_832_128);
    }

    #[test]
    fn rotation_copy_calls_match_the_measured_window_number() {
        // The 2026-10-18 window counted 11,764 copy_dtod calls. That number is derivable from the
        // pinning geometry: one call per array per contiguous run, and a view wraps into two runs
        // only when oldest_slot + rows passes the ring.
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let layout = WindowRingLayout::ENGINE;
        let mut calls = 0usize;
        let mut one_run = 0usize;
        let mut wrapped = 0usize;
        for layer in 0..NUM_HIDDEN_LAYERS {
            for position in execution_positions(layer, &schedule) {
                let (rows, oldest) =
                    window_view(&layout, &schedule, layer, position).expect("view");
                let segments = ring_view_segments(&layout, rows, oldest).expect("segments");
                calls += 2 * segments.len();
                if segments.len() == 1 {
                    one_run += 1;
                } else {
                    wrapped += 1;
                }
            }
        }
        assert_eq!(calls, 11_764, "the measured window call count");
        assert_eq!((one_run, wrapped), (5_082, 400));
        assert_eq!(one_run + wrapped, 5_482);
        // The 360 compared views are a separate number: 9 positions x 40 layers.
        assert_eq!(
            compared_positions(PROMPT_TOKENS, DECODE_STEPS).len()
                * NUM_HIDDEN_LAYERS
                * WINDOW
                * 528,
            24_330_240
        );
    }

    #[test]
    fn rotation_measurement_owns_its_clock_and_counts_every_copy() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let plan = RotationPlan::for_schedule(&schedule);
        let mut called = 0usize;
        let measurement = measure_rotation(&schedule, plan, |layer, position| {
            called += 1;
            let (rows, _) =
                window_view(&WindowRingLayout::ENGINE, &schedule, layer, position).expect("view");
            assert!(layer < NUM_HIDDEN_LAYERS);
            // A rotation issues one copy for a contiguous view and two when it wraps.
            let (_, oldest) =
                window_view(&WindowRingLayout::ENGINE, &schedule, layer, position).expect("view");
            let copies = if oldest + rows <= WINDOW { 2 } else { 4 };
            Ok((copies, rows * 528, Duration::from_nanos(1)))
        })
        .expect("measurement");
        assert_eq!(called, plan.executions());
        assert_eq!(measurement.executions, plan.executions());
        assert_eq!(measurement.bytes_moved, plan.bytes());
        assert_eq!(measurement.bytes_moved, 198_832_128);
        assert_eq!(measurement.copies, 11_764);
        assert!(measurement.copies <= plan.copy_calls_upper_bound());
        assert!(measurement.wall_elapsed >= measurement.copy_elapsed);
        assert!(
            measurement.copy_elapsed <= Duration::from_secs(1),
            "the sum of one-nanosecond copies cannot dominate the loop"
        );
        assert!(measurement.line().contains("copy_dtod"));
        assert!(
            measure_rotation(&schedule, plan, |_layer, _position| {
                Err("driver says no".to_string())
            })
            .is_err(),
            "a failed copy must not become a measurement"
        );
        // A closure that moves the wrong byte count is refused, not reported.
        assert!(
            measure_rotation(&schedule, plan, |_layer, _position| {
                Ok((1, 0, Duration::ZERO))
            })
            .is_err(),
            "a wrong byte total must not become a measurement"
        );
    }

    #[test]
    fn rotation_time_excludes_work_that_is_not_a_copy() {
        // A deliberate 60 ms sleep between copies runs inside the measured loop. Copy time must not
        // see it; the loop wall time must, which is why both are reported separately.
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let plan = RotationPlan::for_schedule(&schedule);
        let mut slept = false;
        let measurement = measure_rotation(&schedule, plan, |layer, position| {
            let (rows, _) =
                window_view(&WindowRingLayout::ENGINE, &schedule, layer, position).expect("view");
            let copy_time = Duration::from_nanos(1);
            if !slept {
                slept = true;
                std::thread::sleep(Duration::from_millis(60));
            }
            Ok((1, rows * 528, copy_time))
        })
        .expect("measurement");
        assert!(slept, "the sleep has to run inside the loop");
        assert_eq!(measurement.bytes_moved, plan.bytes());
        assert!(
            measurement.wall_elapsed >= Duration::from_millis(50),
            "the wall clock must see the sleep, saw {:?}",
            measurement.wall_elapsed
        );
        assert!(
            measurement.copy_elapsed < Duration::from_millis(20),
            "copy time must exclude the sleep, saw {:?}",
            measurement.copy_elapsed
        );
        assert_eq!(
            measurement.copy_elapsed,
            Duration::from_nanos(plan.executions() as u64)
        );
    }

    #[test]
    fn prefill_only_narrows_the_compared_set_and_the_plan() {
        assert_eq!(
            compared_set(false),
            compared_positions(PROMPT_TOKENS, DECODE_STEPS)
        );
        assert_eq!(compared_set(true), vec![PREFILL_POSITION]);
        let plan = dispatch_plan(AttentionMode::CsaUnion, &compared_set(true)).expect("plan");
        assert_eq!(plan.len(), NUM_HIDDEN_LAYERS);
        assert!(
            plan.iter()
                .all(|dispatch| dispatch.position == PREFILL_POSITION)
        );
    }

    #[test]
    fn dispatch_plan_covers_the_compared_set_with_the_pinned_formats() {
        let compared = compared_positions(PROMPT_TOKENS, DECODE_STEPS);
        assert_eq!(compared.len(), 9);
        let plan = dispatch_plan(AttentionMode::CsaUnion, &compared).expect("plan");
        assert_eq!(plan.len(), 9 * NUM_HIDDEN_LAYERS);
        assert_eq!(
            plan.iter()
                .filter(|dispatch| dispatch.window_kv_format == ENGINE_WINDOW_KV_FORMAT)
                .count(),
            18
        );
        assert_eq!(
            plan.iter()
                .filter(|dispatch| dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT)
                .count(),
            342
        );
        for position in &compared {
            let at = dispatch_at_position(*position, AttentionMode::CsaUnion).expect("position");
            assert_eq!(at.len(), NUM_HIDDEN_LAYERS);
            let window_rows = (*position + 1).min(WINDOW);
            assert!(
                at.iter()
                    .all(|dispatch| dispatch.window_rows == window_rows)
            );
            assert!(at.iter().all(|dispatch| dispatch.position == *position));
            let mut ratio2 = 0usize;
            let mut ratio1 = 0usize;
            for dispatch in &at {
                for id in &dispatch.selected {
                    assert!(*id as usize <= *position, "a selected id must be causal");
                    if dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT {
                        assert_eq!(
                            *id as usize % dispatch.ratio,
                            0,
                            "a union id must be a multiple of its ratio"
                        );
                    }
                }
                if dispatch.window_kv_format == CSA_UNION_WINDOW_KV_FORMAT {
                    match dispatch.ratio {
                        2 => ratio2 += dispatch.selected_count(),
                        1 => ratio1 += dispatch.selected_count(),
                        other => panic!("unexpected ratio {other}"),
                    }
                }
            }
            // 18 ratio-2 union layers reach (position + 1) / 2 rows, 20 ratio-1 layers all rows.
            assert_eq!(ratio2, 18 * ((position + 1) / 2));
            assert_eq!(ratio1, 20 * (position + 1));
        }
        let window_plan = dispatch_plan(AttentionMode::WindowOnly, &compared).expect("window plan");
        assert_eq!(window_plan.len(), plan.len());
        assert!(
            window_plan
                .iter()
                .all(|dispatch| dispatch.window_kv_format == ENGINE_WINDOW_KV_FORMAT),
            "the window-only diagnostic must not smuggle a union call"
        );
        assert!(dispatch_plan(AttentionMode::CsaUnion, &[]).is_err());
    }

    #[test]
    fn compressed_view_rows_follow_the_store_layout() {
        assert_eq!(compressed_view_rows(2).expect("ratio 2"), (69, 138));
        assert_eq!(compressed_view_rows(1).expect("ratio 1"), (138, 138));
        assert!(compressed_view_rows(0).is_err());
        for store_layer in KV_SOURCE_LAYERS {
            let (store_rows, view_rows) =
                compressed_view_rows(compress_ratio(store_layer)).expect("store layout");
            assert_eq!(view_rows, store_rows * compress_ratio(store_layer));
            assert!(view_rows >= LAST_SEQUENCE_POSITION + 1);
        }
    }

    #[test]
    fn harness_images_are_the_layout_they_claim() {
        let layout = WindowRingLayout::ENGINE;
        let (codes, scales) = host_ring_image(20).expect("ring image");
        assert_eq!(codes.len(), layout.slots * layout.code_row_bytes);
        assert_eq!(scales.len(), layout.slots * layout.scale_row_bytes);
        assert_ne!(codes, vec![0u8; codes.len()]);
        assert_engine_window_finite(&codes, &scales, layout.slots, "test ring").expect("finite");
        let (store_codes, store_scales) = host_compressed_store(20, 69).expect("compressed store");
        assert_eq!(store_codes.len(), 69 * COMPRESSED_CODES_BYTES);
        assert_eq!(store_scales.len(), 69 * COMPRESSED_SCALES_BYTES);
        assert!(store_codes.iter().any(|byte| *byte != 0));
        assert!(host_compressed_store(0, 1).is_err(), "layer 0 has ratio 0");
        let q = host_query_values();
        assert_eq!(q.len(), ATTENTION_HEADS * ATTENTION_LATENT);
        assert!(q.iter().all(|value| value.is_finite()));
        let q_device = bf16_bytes_to_f32(&host_query_bf16_bytes(&q));
        assert_eq!(q_device.len(), q.len());
        assert!(q_device.iter().all(|value| value.is_finite()));
        assert_eq!(host_attention_sink().len(), ATTENTION_HEADS);
        assert_eq!(store_layer_slot(20).expect("source layer"), 3);
        assert!(store_layer_slot(21).is_err());
    }

    #[test]
    fn step_report_states_what_was_and_was_not_produced() {
        let produced = StepReport {
            position: 130,
            kind: "decode",
            expected_token: Some(491),
            device_token: Some(491),
            logits_sha256: "ab".to_string(),
            logits_delta: Some(0.0),
            device_source: "synthetic",
        };
        let line = produced.line();
        assert!(line.contains("match=true"), "line: {line}");
        assert!(line.contains("within bar"), "line: {line}");
        let mismatched = StepReport {
            device_token: Some(492),
            ..produced.clone()
        };
        assert!(mismatched.line().contains("MISMATCH"));
        let absent = StepReport {
            device_token: None,
            logits_delta: None,
            device_source: "no device output: forward chain not wired in this iteration",
            ..produced.clone()
        };
        let line = absent.line();
        assert!(line.contains("NOT PRODUCED"), "line: {line}");
        assert!(line.contains("NOT ATTEMPTED"), "line: {line}");
        assert!(line.contains("forward chain not wired"), "line: {line}");
        let out_of_bar = StepReport {
            logits_delta: Some(LOGITS_ABS_BAR * 2.0),
            ..produced
        };
        assert!(out_of_bar.line().contains("OUT OF BAR"));
    }

    /// The comparison plumbing is pinned to the 130-token milestone (position 129), so it can only
    /// be exercised against the released fixture. That fixture lives on the node (and on this dev
    /// host only as a copied artifact), so the test skips with an explicit line when it is absent
    /// instead of fabricating a smaller fixture the constants would refuse.
    #[test]
    fn comparison_plumbing_accepts_the_released_fixture_and_refuses_perturbations() {
        const CANDIDATES: [&str; 2] = [
            "/home/user/phase2-evidence/sequence-130-released.json",
            "/home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json",
        ];
        let Some(path) = CANDIDATES
            .iter()
            .map(Path::new)
            .find(|candidate| candidate.is_file())
        else {
            eprintln!("skipping released-fixture plumbing test: no fixture at {CANDIDATES:?}");
            return;
        };
        let fixture = load_fixture(path).expect("released fixture");
        check_fixture_shape(&fixture).expect("fixture shape");
        let compared = compared_positions(PROMPT_TOKENS, DECODE_STEPS);
        let (reports, controls) = dry_run_step_reports(&fixture, &compared).expect("plumbing");
        assert_eq!(reports.len(), fixture.steps.len());
        assert_eq!(reports.len(), 1 + fixture.decode_steps);
        // The released fixture records logits at the prefill position only; every other step has no
        // logits to compare, and the report says so instead of inventing a delta.
        for report in &reports {
            if report.position == PREFILL_POSITION {
                assert_eq!(report.logits_delta, Some(0.0));
            } else {
                assert_eq!(report.logits_delta, None);
            }
        }
        assert!(
            reports
                .iter()
                .all(|report| report.device_source.contains("synthetic"))
        );
        assert_eq!(controls.len(), 3);
        assert!(
            controls.iter().all(|control| control.contains("refused")),
            "controls: {controls:?}"
        );
    }

    /// Pins the shared format-2 builder's indexer pointers. The kernel runs indexer stage 1 before
    /// the union branch and dereferences those four words, so the builder must carry real buffers:
    /// a regression to nulls here is exactly the defect the critic found in round two.
    #[test]
    fn format2_builder_carries_its_indexer_buffers() {
        let selected = [0u32, 2];
        let mut call = Format2AttentionCall {
            q: Deviceptr(0x10),
            kv_fp4: Deviceptr(0x20),
            kv_scales: Deviceptr(0x30),
            window_codes: Deviceptr(0x40),
            window_scales: Deviceptr(0x50),
            sink: Deviceptr(0x60),
            selected: Deviceptr(0x70),
            out: Deviceptr(0x80),
            position: 2,
            layer: 20,
            step_seq: 0,
            window_rows: 2,
            selected_ids: &selected,
            trace_ring: None,
            index_key: Deviceptr(0x90),
            index_query: Deviceptr(0xa0),
            block_scores: Deviceptr(0xb0),
            block_best: Deviceptr(0xc0),
        };
        let params = call.params().expect("bounds");
        let words = params.as_slice();
        assert_eq!(words[5], CSA_UNION_WINDOW_KV_FORMAT as u64);
        assert_eq!(words[6], 2);
        // The pinned ABI order puts the four indexer pointers at 8, 9, 11 and 12, with the sink at
        // 10. They are the buffers the caller passed, so a launch through this builder can run.
        assert_eq!(words[8], 0x90, "index_key");
        assert_eq!(words[9], 0xa0, "index_query");
        assert_eq!(words[11], 0xb0, "block_scores");
        assert_eq!(words[12], 0xc0, "block_best");
        assert_eq!(words[10], 0x60, "sink");
        // A missing indexer buffer is refused instead of becoming a null device pointer.
        call.index_key = Deviceptr(0);
        assert!(call.params().is_err());
        call.index_key = Deviceptr(0x90);
        call.block_best = Deviceptr(0);
        assert!(call.params().is_err());
    }

    #[test]
    fn window_bytes_are_the_rotation_step() {
        let geometry = StoreGeometry::for_max_seq_len(192).expect("geometry");
        assert_eq!(geometry.window_bytes(), 2_703_360);
        assert_eq!(geometry.window_bytes() as usize, ROTATION_BYTES_PER_STEP);
    }

    #[test]
    fn source_handles_follow_the_pinned_sources() {
        let geometry = StoreGeometry::for_max_seq_len(192).expect("geometry");
        for layer in [2usize, 8, 14, 20] {
            assert_eq!(geometry.source_layer_for(layer), Some(layer));
        }
        for layer in [3usize, 24, 36, 39] {
            assert_eq!(
                geometry.source_layer_for(layer),
                Some(sparkflash_kv::config::kv_source_for(layer))
            );
        }
        assert_eq!(geometry.source_layer_for(0), None);
        assert_eq!(geometry.source_layer_for(1), None);
    }

    #[test]
    fn union_format_matches_the_kernel_macro() {
        let kernel = include_str!("../../../kernels/flash_mla_csa2_sm121.cu");
        let line = kernel
            .lines()
            .find(|line| line.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION"))
            .expect("the kernel defines the CSA union format");
        assert!(
            line.trim_end().ends_with("2u"),
            "kernel union macro must be 2, found: {line}"
        );
        assert_eq!(CSA_UNION_WINDOW_KV_FORMAT, 2);
        assert_eq!(
            AttentionMode::CsaUnion.window_kv_format(),
            CSA_UNION_WINDOW_KV_FORMAT
        );
    }

    #[test]
    fn attention_modes_dispatch_to_distinct_formats() {
        assert_eq!(AttentionMode::WindowOnly.window_kv_format(), 1);
        assert_eq!(
            AttentionMode::CsaUnion.window_kv_format(),
            CSA_UNION_WINDOW_KV_FORMAT
        );
        assert_ne!(
            AttentionMode::CsaUnion.window_kv_format(),
            AttentionMode::WindowOnly.window_kv_format()
        );
        assert!(AttentionMode::CsaUnion.is_equality_claim());
        assert!(!AttentionMode::WindowOnly.is_equality_claim());
        assert!(
            AttentionMode::WindowOnly
                .scope_label()
                .contains("no equality claim")
        );
    }

    #[test]
    fn boundary_dump_uses_the_checkpoint_tensor_limit() {
        // The !58 class bug: a checkpoint-capable mode built with the synthetic reference limit
        // aborts on the released embed.weight before any prefill. Pin both the constant the dump
        // passes to the engine and the released geometry that must pass it.
        assert_eq!(
            BOUNDARY_DUMP_TENSOR_LIMIT,
            sparkflash_engine::linalg::CHECKPOINT_TENSOR_LIMIT
        );
        assert!(BOUNDARY_DUMP_TENSOR_LIMIT > sparkflash_engine::linalg::REFERENCE_TENSOR_LIMIT);
        let config = ModelConfig::v41_flash().expect("released config");
        let embed_elements = config.vocab_size * config.dim;
        assert_eq!(embed_elements, 661_913_600);
        assert!(embed_elements > sparkflash_engine::linalg::REFERENCE_TENSOR_LIMIT);
        assert!(embed_elements <= BOUNDARY_DUMP_TENSOR_LIMIT);
    }

    fn synthetic_fixture() -> SequenceFixture {
        let config = ModelConfig::v41_flash().expect("released config");
        let compared = compared_positions(PROMPT_TOKENS, DECODE_STEPS);
        let steps = compared
            .iter()
            .enumerate()
            .map(|(index, position)| sparkflash_engine::FixtureStep {
                kind: if index == 0 {
                    sparkflash_engine::sequence_fixture::StepKind::Prefill
                } else {
                    sparkflash_engine::sequence_fixture::StepKind::Decode
                },
                position: *position,
                next_token: Some(if index == 0 {
                    PREFILL_TOKEN
                } else {
                    1_000 + index as u32
                }),
                logits_sha256: "00".repeat(32),
                logits: if index == 0 {
                    Some(vec![0.5f32; 16])
                } else {
                    None
                },
            })
            .collect::<Vec<_>>();
        SequenceFixture {
            schema_version: sparkflash_engine::sequence_fixture::FIXTURE_SCHEMA_VERSION,
            generator: "m8_sequence test".to_string(),
            source: sparkflash_engine::FixtureSource {
                kind: sparkflash_engine::FixtureWeightKind::FiniteFp32,
                seed: 7,
                checkpoint: None,
            },
            config_label: "tiny".to_string(),
            config,
            prompt_tokens: (0..PROMPT_TOKENS as u32).collect(),
            decode_steps: DECODE_STEPS,
            window: WINDOW,
            compared_positions: compared,
            steps,
            kv_rows: Vec::new(),
            provenance: sparkflash_engine::FixtureProvenance {
                command: "m8_sequence --self-check".to_string(),
                host: None,
                binary_sha256: None,
                notes: Vec::new(),
            },
        }
    }

    #[test]
    fn fixture_shape_check_names_the_pinned_units() {
        let fixture = synthetic_fixture();
        check_fixture_shape(&fixture).expect("synthetic fixture shape");
        let mut wrong = synthetic_fixture();
        wrong.steps[0].next_token = Some(490);
        assert!(check_fixture_shape(&wrong).is_err());
        let mut wrong = synthetic_fixture();
        wrong.compared_positions.pop();
        assert!(check_fixture_shape(&wrong).is_err());
        let mut wrong = synthetic_fixture();
        wrong.window = 256;
        assert!(check_fixture_shape(&wrong).is_err());
    }

    #[test]
    fn prefill_comparison_checks_token_first_then_logits() {
        let fixture = synthetic_fixture();
        let expected = fixture.steps[0].clone();
        let device = DeviceStep {
            position: PREFILL_POSITION,
            next_token: expected.next_token,
            logits: expected.logits.clone(),
            logits_sha256: None,
        };
        let comparison = compare_prefill(&fixture, &device).expect("comparison");
        assert_eq!(comparison.expected_token, Some(PREFILL_TOKEN));
        assert_eq!(comparison.device_token, Some(PREFILL_TOKEN));
        assert_eq!(comparison.max_abs_logits_delta, Some(0.0));
        assert_eq!(comparison.logits_within_bar, Some(true));
        let wrong = DeviceStep {
            next_token: Some(PREFILL_TOKEN + 1),
            ..device.clone()
        };
        let error = compare_prefill(&fixture, &wrong).expect_err("token must be first and hard");
        assert!(error.contains("token comparison is first and hard"));
        let wrong = DeviceStep {
            position: PREFILL_POSITION + 1,
            ..device
        };
        assert!(compare_prefill(&fixture, &wrong).is_err());
        let expected_logits = fixture.steps[0].logits.clone().expect("logits");
        let mut off = expected_logits;
        off[0] += 10.0 * LOGITS_ABS_BAR;
        let device = DeviceStep {
            position: PREFILL_POSITION,
            next_token: Some(PREFILL_TOKEN),
            logits: Some(off),
            logits_sha256: None,
        };
        let comparison = compare_prefill(&fixture, &device).expect("comparison");
        assert_eq!(comparison.logits_within_bar, Some(false));
        assert!(comparison.max_abs_logits_delta.unwrap() > LOGITS_ABS_BAR);
    }

    #[test]
    fn full_sequence_comparison_rejects_positions_outside_the_compared_set() {
        let fixture = synthetic_fixture();
        let mut device: Vec<DeviceStep> = fixture
            .steps
            .iter()
            .map(|step| DeviceStep {
                position: step.position,
                next_token: step.next_token,
                logits: step.logits.clone(),
                logits_sha256: None,
            })
            .collect();
        compare_full_sequence(&fixture, &device).expect("complete compared set");
        device.push(DeviceStep::tokens(1, Some(7)));
        assert!(compare_full_sequence(&fixture, &device).is_err());
        let short = device[..device.len() - 2].to_vec();
        assert!(compare_full_sequence(&fixture, &short).is_err());
        let empty: Vec<DeviceStep> = Vec::new();
        assert!(compare_full_sequence(&fixture, &empty).is_err());
    }

    #[test]
    fn self_check_runs_without_a_fixture() {
        let args = parse(&["--no-self-check"]).expect("parses");
        assert!(!args.self_check);
        let args = parse(&["--self-check", "--max-seq-len", "192"]).expect("parses");
        let line = run_self_check(&args).expect("self-check");
        assert!(line.contains("rotation plan"));
        assert!(line.contains("publications (17)"));
        assert!(!line.contains("fixture "));
        assert!(line.contains("swa replay split"));
        assert!(line.contains("compressed-id expansion"));
    }

    /// The engine's window ids for one prefill query, from the engine's own function: chunk-relative
    /// row positions, oldest first, exactly what the device path supplies to format 2 as window rows.
    fn engine_window_ids_for_query(config: &ModelConfig, query: usize) -> Vec<usize> {
        let width = config.window_size.min(PROMPT_TOKENS);
        let ids = window_indices(config, 0, PROMPT_TOKENS);
        assert_eq!(ids.len(), PROMPT_TOKENS * width);
        ids[query * width..(query + 1) * width].to_vec()
    }

    #[test]
    fn ratio2_expansion_attends_exactly_the_engine_union() {
        let config = ModelConfig::v41_flash_cached();
        let window_ids = engine_window_ids_for_query(config, PREFILL_POSITION);
        assert_eq!(window_ids.len(), 128);
        assert_eq!(window_ids[0], 2);
        assert_eq!(*window_ids.last().expect("last window id"), 129);
        // The engine's prefill kv layout puts the chunk's window rows first; compressed ids are
        // shifted by that width (attention.rs compressed_indices).
        let window_rows = PROMPT_TOKENS;
        let ratio = 2usize;
        let n_pos = PROMPT_TOKENS / ratio;
        let reach = (PREFILL_POSITION + 1) / ratio;
        assert_eq!(reach, n_pos);
        let compressed: Vec<usize> = (0..n_pos)
            .map(|row| {
                if row < reach {
                    row + window_rows
                } else {
                    ENGINE_NO_INDEX
                }
            })
            .collect();
        let engine =
            engine_attended_union(&window_ids, &compressed, window_rows).expect("engine union");
        let selected = kernel_selected_from_engine_compressed_ids(&compressed, window_rows, ratio)
            .expect("selected ids");
        let kernel = format2_attended_union(
            &window_ids,
            &selected,
            PREFILL_POSITION as u32,
            ratio,
            n_pos,
        )
        .expect("kernel union");
        assert_eq!(engine.sorted(), kernel.sorted());
        assert_eq!(engine.len(), 128 + 65);
        // Compressed row 1 is token position 2, which the window also holds: both KV values are
        // attended, because the reference concatenates the two lists rather than deduplicating.
        assert!(engine.contains(AttendedValue::WindowToken(2)));
        assert!(engine.contains(AttendedValue::CompressedRow(1)));
        // The mapping is injective and every selected id is inside the expanded view.
        assert_eq!(selected.len(), n_pos);
        let mut unique = selected.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), n_pos);
        for (row, position) in selected.iter().enumerate() {
            assert_eq!(
                *position as usize,
                compressed_row_to_absolute_position(ratio, row)
            );
            assert!(*position as usize <= PREFILL_POSITION);
            assert_eq!(expanded_store_row(ratio, *position as usize), row);
        }
    }

    #[test]
    fn ratio1_expansion_is_the_identity_and_keeps_both_views() {
        let config = ModelConfig::v41_flash_cached();
        let window_ids = engine_window_ids_for_query(config, PREFILL_POSITION);
        let window_rows = PROMPT_TOKENS;
        let ratio = 1usize;
        let n_pos = PROMPT_TOKENS;
        let compressed: Vec<usize> = (0..n_pos).map(|row| row + window_rows).collect();
        let engine =
            engine_attended_union(&window_ids, &compressed, window_rows).expect("engine union");
        let selected = kernel_selected_from_engine_compressed_ids(&compressed, window_rows, ratio)
            .expect("selected ids");
        assert_eq!(selected, (0..PROMPT_TOKENS as u32).collect::<Vec<_>>());
        let kernel = format2_attended_union(
            &window_ids,
            &selected,
            PREFILL_POSITION as u32,
            ratio,
            n_pos,
        )
        .expect("kernel union");
        assert_eq!(engine.sorted(), kernel.sorted());
        assert_eq!(engine.len(), 128 + 130);
        assert!(engine.contains(AttendedValue::WindowToken(2)));
        assert!(engine.contains(AttendedValue::CompressedRow(2)));
    }

    #[test]
    fn ratio2_causal_edge_needs_the_engine_reach_filter_not_the_kernel_guard() {
        let config = ModelConfig::v41_flash_cached();
        let query = 128usize;
        let window_ids = engine_window_ids_for_query(config, query);
        let window_rows = PROMPT_TOKENS;
        let ratio = 2usize;
        let n_pos = PROMPT_TOKENS / ratio;
        let reach = (query + 1) / ratio;
        assert_eq!(reach, 64);
        // The engine marks compressed row 64 unreachable for this query (NO_INDEX), even though
        // its expanded absolute position 128 is equal to the query position and the kernel's
        // id <= position guard alone would attend it.
        let filtered: Vec<usize> = (0..n_pos)
            .map(|row| {
                if row < reach {
                    row + window_rows
                } else {
                    ENGINE_NO_INDEX
                }
            })
            .collect();
        let engine =
            engine_attended_union(&window_ids, &filtered, window_rows).expect("engine union");
        let selected = kernel_selected_from_engine_compressed_ids(&filtered, window_rows, ratio)
            .expect("filtered selected ids");
        assert!(!selected.contains(&128));
        let kernel = format2_attended_union(&window_ids, &selected, query as u32, ratio, n_pos)
            .expect("filtered kernel union");
        assert_eq!(engine.sorted(), kernel.sorted());
        assert!(!engine.contains(AttendedValue::CompressedRow(64)));
        // The negative control: an unfiltered list maps row 64 to id 128, which the kernel guard
        // admits at position 128. That is why the harness must consume the engine's filtered ids;
        // the guard is not the reference reach rule.
        let unfiltered: Vec<usize> = (0..n_pos).map(|row| row + window_rows).collect();
        let naive = kernel_selected_from_engine_compressed_ids(&unfiltered, window_rows, ratio)
            .expect("naive selected ids");
        let naive_union = format2_attended_union(&window_ids, &naive, query as u32, ratio, n_pos)
            .expect("naive kernel union");
        assert!(naive_union.contains(AttendedValue::CompressedRow(64)));
        assert_ne!(engine.sorted(), naive_union.sorted());
    }

    /// Condition (a) of the Lead's approval: the harness filter must BE the engine's output, not
    /// a hand-rolled heuristic. This calls the engine's own compressed_indices for a synthetic
    /// 130-token prefill and checks, for every query position, that the filtered rows are exactly
    /// the non-NO_INDEX engine entries and that the boundary query 128 leaves row 64 out while 129
    /// admits it. The synthetic config selects every compressed row (index_topk >= n_pos and all
    /// candidate blocks visible), so this is the full rule and not a top-k-limited sample.
    #[test]
    fn harness_filter_matches_the_engine_compressed_ids_for_every_query() {
        let mut config = ModelConfig::tiny();
        config.window_size = 128;
        config.original_seq_len = 192;
        config.compress_ratios = vec![2, 2, 0];
        config.index_topk = 512;
        config.candidate_topk_blocks = 64;
        let plan = TensorPlan::language_model(&config);
        let weights = SyntheticWeights::from_plan(&plan, 7);
        let mut state = AttnState::for_config(&config);
        let seqlen = PROMPT_TOKENS;
        let x = vec![0.01f32; seqlen * config.dim];
        let wide = (seqlen * config.dim).max(seqlen * config.n_heads * config.head_dim);
        let mut out = vec![0.0f32; wide];
        attention_forward(
            &config,
            0,
            "layers.0",
            &x,
            0,
            &mut state,
            &weights,
            &mut out,
            1 << 28,
        )
        .expect("engine attention for the synthetic prefill");
        let q = vec![0.0f32; seqlen * config.q_lora_rank];
        let mut engine_ids = Vec::new();
        compressed_indices(
            &state,
            &config,
            0,
            &q,
            &weights,
            "layers.0",
            &mut engine_ids,
            1 << 28,
        )
        .expect("engine compressed ids");
        let n_pos = seqlen / 2;
        let topk = config.index_topk.min(n_pos);
        assert_eq!(topk, n_pos);
        assert_eq!(engine_ids.len(), seqlen * topk);
        let window_rows = seqlen;
        let mut boundary_128_seen = false;
        for query in 0..seqlen {
            let reach = (query + 1) / 2;
            let row_ids = &engine_ids[query * topk..(query + 1) * topk];
            for (row, &id) in row_ids.iter().enumerate() {
                if row < reach {
                    assert_eq!(
                        id,
                        row + window_rows,
                        "query {query}: row {row} is reachable and must be a shifted compressed id"
                    );
                } else {
                    assert_eq!(
                        id, ENGINE_NO_INDEX,
                        "query {query}: row {row} is beyond reach {reach} and must be NO_INDEX"
                    );
                }
            }
            let selected = kernel_selected_from_engine_compressed_ids(row_ids, window_rows, 2)
                .expect("harness filter");
            let expected: Vec<u32> = (0..reach).map(|row| (row * 2) as u32).collect();
            assert_eq!(selected, expected, "query {query}");
            if query == 128 {
                boundary_128_seen = true;
                assert_eq!(reach, 64);
                assert!(
                    !selected.contains(&128),
                    "row 64 is not reachable at query 128"
                );
            }
            if query == 129 {
                assert_eq!(reach, 65);
                assert_eq!(*selected.last().expect("selected"), 128);
            }
        }
        assert!(boundary_128_seen);
    }

    #[test]
    fn format2_selection_is_built_from_filtered_engine_ids_only() {
        let selection =
            format2_selection_from_engine_ids(&[130, 130, 132, ENGINE_NO_INDEX], 130, 128, 2, 129)
                .expect("selection");
        assert_eq!(selection.selected, vec![0, 0, 4]);
        assert_eq!(selection.window_rows, 128);
        assert_eq!(selection.selected_count, 3);
        assert_eq!(selection.union_rows(), 131);
        // A row the engine did not filter out because it is at the reach boundary is refused, not
        // weakened to the kernel's id <= position guard (row 64 maps to id 128 == position 128).
        let error = format2_selection_from_engine_ids(&[130 + 64], 130, 128, 2, 128)
            .expect_err("row 64 is unreachable at position 128");
        assert!(error.contains("reachability filter"));
        let error = format2_selection_from_engine_ids(&[129], 130, 128, 2, 129)
            .expect_err("window-space id");
        assert!(error.contains("window space"));
        let too_many: Vec<usize> = (0..=FORMAT2_MAX_SELECTED).map(|row| row + 128).collect();
        assert!(format2_selection_from_engine_ids(&too_many, 128, 128, 1, 1000).is_err());
    }

    #[test]
    fn ratio_expansion_duplicates_each_store_row_at_its_positions() {
        let store_rows = 3usize;
        let codes: Vec<u8> = (0..store_rows)
            .flat_map(|row| vec![row as u8; COMPRESSED_CODES_BYTES])
            .collect();
        let scales: Vec<u8> = (0..store_rows)
            .flat_map(|row| vec![row as u8; COMPRESSED_SCALES_BYTES])
            .collect();
        let expanded = expand_compressed_view(&codes, &scales, store_rows, 2).expect("expansion");
        assert_eq!(expanded.rows, 6);
        assert_eq!(expanded.codes_bytes(), 6 * COMPRESSED_CODES_BYTES);
        assert_eq!(expanded.scales_bytes(), 6 * COMPRESSED_SCALES_BYTES);
        assert_eq!(expanded.codes_bytes(), 6 * 256);
        assert_eq!(expanded.scales_bytes(), 6 * 32);
        for position in 0..6 {
            assert_eq!(
                expanded.codes[position * COMPRESSED_CODES_BYTES],
                (position / 2) as u8
            );
            assert_eq!(
                expanded.scales[position * COMPRESSED_SCALES_BYTES],
                (position / 2) as u8
            );
        }
        let identity = expand_compressed_view(&codes, &scales, store_rows, 1).expect("identity");
        assert_eq!(identity.codes, codes);
        assert_eq!(identity.scales, scales);
        assert!(expand_compressed_view(&codes, &scales, 2, 2).is_err());
    }

    #[test]
    fn compressed_ids_keep_duplicates_and_reject_window_ids() {
        let selected =
            kernel_selected_from_engine_compressed_ids(&[130, 130, 132, ENGINE_NO_INDEX], 130, 2)
                .expect("selected");
        assert_eq!(selected, vec![0, 0, 4]);
        assert!(
            kernel_selected_from_engine_compressed_ids(&[129], 130, 2).is_err(),
            "a window-space id in the compressed list is not a compressed-row-mapped id and must be rejected, not silently remapped"
        );
    }

    #[test]
    fn ced_split_strictly_agrees_with_the_kv_replay_split() {
        let schedule = checked_schedule(PROMPT_TOKENS, false).expect("schedule");
        let ced_encoder = ENCODER_LAST_LAYER + 1;
        let ced_decoder = NUM_HIDDEN_LAYERS - DECODER_FIRST_LAYER;
        assert_eq!((ced_encoder, ced_decoder), (21, 19));
        assert_eq!(ced_encoder + ced_decoder, NUM_HIDDEN_LAYERS);
        assert_eq!(schedule.encoder_layers, 21);
        assert_eq!(schedule.decoder_layers, 19);
        assert_eq!(replay_layer_count(ReplayKind::Encoder), 21);
        assert_eq!(replay_layer_count(ReplayKind::Decoder), 19);
        // The regression guard itself: the self-check must refuse a drift, so the run below fails
        // if either split changes without the other following.
        let args = parse(&["--self-check", "--max-seq-len", "192"]).expect("parses");
        let line = run_self_check(&args).expect("self-check after the task-10 fix");
        assert!(line.contains("ReplayKind Encoder 21 / Decoder 19"));
    }
}
