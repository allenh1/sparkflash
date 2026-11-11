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

//! Typed per-model runtime profiles.
//!
//! A model's behaviour is configuration, not a pile of environment switches read at the point of
//! use. A profile is parsed once, validated, printed at boot, and handed to the arm. The
//! environment remains an override, never the only way to reach a behaviour: every override
//! below names a typed field with a documented value domain, an unknown value is refused, and an
//! explicit TOML value wins over the environment.
//!
//! The table [ENV_OVERRIDES] is the complete environment surface this module owns. Arms with
//! switches not listed here keep them as deprecated aliases until their typed field lands; the
//! gate in docs/adding-a-model.md says a new switch must be added here, not read directly.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// A profile failure.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// The TOML does not parse, or a key or enum value is unknown.
    #[error("profile: {0}")]
    Parse(String),
    /// A typed field is out of range or internally inconsistent.
    #[error("profile: {0}")]
    Invalid(String),
    /// An environment override carries a value the field does not accept.
    #[error("{name}={value:?} is not a valid {field}: {expected}")]
    Env {
        /// The environment variable.
        name: &'static str,
        /// The value found.
        value: String,
        /// The typed field.
        field: &'static str,
        /// The accepted domain.
        expected: &'static str,
    },
    /// The file could not be read.
    #[error("profile {path}: {source}")]
    Io {
        /// The profile path.
        path: String,
        /// The read error.
        source: std::io::Error,
    },
}

/// The runtime section: sequence width and admission.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeProfile {
    /// The context window one sequence allocates.
    pub context_tokens: Option<usize>,
    /// Sequences one rank admits.
    pub concurrency: Option<usize>,
    /// The KV budget in gibibytes.
    pub kv_budget_gib: Option<usize>,
    /// Whether the device path runs instead of the host twin.
    pub device: Option<bool>,
    /// Per-lane device memory budget in MiB.
    pub lane_memory_mb: Option<usize>,
    /// Host-memory floor in GiB before the run refuses.
    pub mem_floor_gib: Option<usize>,
    /// Serve constrained output (`false`: report the capability and leave the request to the
    /// grammar the client enforces).
    pub constrained: Option<bool>,
    /// Map the resident arena with mmap registration.
    pub arena_mmap: Option<bool>,
    /// Allocate the arena as host-mapped unified memory.
    pub arena_hostmapped: Option<bool>,
    /// Verify every weight after loading.
    pub verify_weights: Option<bool>,
    /// Directory holding converted rank files.
    pub converted_dir: Option<String>,
    /// Harness multi-row verify width.
    pub rows: Option<usize>,
    /// First position of the multi-row verify.
    pub rows_from: Option<usize>,
}

/// Which projection path a bulk prefill takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationMode {
    /// Quantise activations to FP8.
    Fp8,
    /// Keep each activation as a bf16 head plus a bf16 remainder.
    Split,
    /// Keep activations in bf16.
    Bf16,
}

/// The prefill section.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrefillProfile {
    /// Whether the bulk prefill path is enabled.
    pub enabled: Option<bool>,
    /// Tokens per layer-major chunk.
    pub chunk: Option<usize>,
    /// Whether a wide chunk takes the tensor cores.
    pub tensor_cores: Option<bool>,
    /// Which activation form the tensor-core GEMMs take.
    pub activation: Option<ActivationMode>,
    /// Whether prefill is exact against the token path.
    pub exact: Option<bool>,
    /// Whether a request that drafts a block runs its prompt through the bulk chunk. The drafter
    /// keeps one step's target rows, so a chunk runs with it detached and its context starts at
    /// the generation; off, the drafter sees the prompt and the prompt costs a step a token.
    pub drafting_prompt: Option<bool>,
    /// Rows a bulk chunk takes while another sequence decodes.
    pub chunk_busy: Option<usize>,
    /// Steps whose cost the ledger reports apart from steady state.
    pub warmup_steps: Option<usize>,
    /// Exact steps per tick while another sequence decodes.
    pub exact_busy: Option<usize>,
    /// Reader threads for a bulk chunk's Engram reads.
    pub read_threads: Option<usize>,
    /// CTAs the bulk communicator opens.
    pub bulk_ctas: Option<usize>,
    /// Merge the routed rows of a bulk chunk's states.
    pub moe_merge: Option<bool>,
    /// Shortest uncached prompt the bulk prefill takes.
    pub bulk_min: Option<usize>,
    /// Concurrent bulk chunks.
    pub ways: Option<usize>,
    /// Most of the tick a chunk may take while decoding sequences wait (0.5 by default).
    pub decode_share: Option<f64>,
    /// The longest a prompt may wait for its first token, in milliseconds (200 s by default).
    pub deadline_ms: Option<f64>,
    /// Chunks paired under each other's launches.
    pub pair: Option<usize>,
    /// Warm the first request's kernels before serving.
    pub warmup: Option<bool>,
    /// Rank 0 decides each tick one ahead.
    pub tick_ahead: Option<bool>,
}

/// The decode section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecodeProfile {
    /// Whether a decode step and a speculative verify take the tensor cores, in the form that
    /// computes a row to the same bits at every row count.
    pub tensor_cores: Option<bool>,
}

/// The attention-cache representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentCacheMode {
    /// Pick from the model's release.
    Auto,
    /// Keep the latent in f32.
    F32,
    /// Keep the released E4M3 latent.
    Fp8,
}

/// The attention section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AttentionProfile {
    /// The latent-cache representation.
    pub latent_cache: Option<LatentCacheMode>,
    /// Selected tokens per sparse query.
    pub selected_tokens: Option<usize>,
}

/// How the speculator picks the number of drafts a step verifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthMode {
    /// Follow the drafts' probabilities.
    Adaptive,
    /// Verify the whole chain.
    Fixed,
}

/// The speculation section.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculationProfile {
    /// Whether speculation is enabled.
    pub enabled: Option<bool>,
    /// The fixed draft depth, when the request does not choose one.
    pub depth: Option<usize>,
    /// Drafts a greedy request verifies after each token.
    pub drafts: Option<usize>,
    /// How the depth policy picks a step's depth.
    pub depth_policy: Option<DepthMode>,
    /// The fewest drafts a step verifies.
    pub min_depth: Option<usize>,
    /// The MTP fold's scale.
    pub mtp_fold_scale: Option<f32>,
    /// DSpark drafts per step.
    pub dspark_k: Option<usize>,
    /// DSpark generated steps.
    pub dspark_steps: Option<usize>,
    /// Let several sequences share a speculative step.
    pub batch_spec: Option<bool>,
    /// Least calibrated chain probability a shared verify row needs.
    pub batch_spec_floor: Option<f64>,
    /// The drafter directory or repository id.
    pub drafter: Option<String>,
}

/// Which kernel the drafter's head logits take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftLogitsKernel {
    /// The CUDA-core form, one sequence's block per row.
    Simt,
    /// The tensor cores with a bf16 activation against the row-scaled FP8 head.
    Bf16,
    /// The tensor cores with the activations as E4M3 codes.
    Fp8,
}

/// The drafter section: which kernel variant each stage of the speculative drafter takes.
///
/// The variants compute the same function on different hardware paths, so a profile pins the one
/// the arm was qualified on; an operator's A/B run moves it through the alias in
/// [ENV_OVERRIDES].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DrafterProfile {
    /// The head-logits kernel.
    pub logits: Option<DraftLogitsKernel>,
    /// Keep the head in whole 128-entry tiles for the tiled copy (`false`: the plain copy).
    pub head_tiled: Option<bool>,
    /// The Markov biases on the tensor cores (`false`: a warp sum per sequence and entry).
    pub markov_mma: Option<bool>,
    /// Sequences one pass's stage kernels serve.
    pub seqs: Option<usize>,
    /// Dedup the routed rows before the shared experts (`false`: one row per occurrence).
    pub moe_dedup: Option<bool>,
    /// Keep the head slice as E4M3 codes and ue8m0 scales (`false`: the bf16 head).
    pub head_fp8: Option<bool>,
    /// The fewest head rows the tensor-core logits serve.
    pub logits_rows: Option<usize>,
    /// Split the seed's main_proj columns over the ranks (`false`: whole on every rank).
    pub seed_split: Option<bool>,
    /// Build the pass's row table (`false`: each block computes its own).
    pub row_table: Option<bool>,
}

impl DrafterProfile {
    /// The fewest head rows the tensor-core logits serve unless the profile says otherwise.
    pub const LOGITS_ROWS: usize = 16;

    /// The head-logits kernel, the tensor-core bf16 form unless the profile says otherwise.
    #[must_use]
    pub fn logits_kernel(&self) -> DraftLogitsKernel {
        self.logits.unwrap_or(DraftLogitsKernel::Bf16)
    }

    /// The fewest head rows the tensor-core logits serve.
    #[must_use]
    pub fn min_logits_rows(&self) -> usize {
        self.logits_rows.unwrap_or(Self::LOGITS_ROWS)
    }

    /// Whether the head copy is tiled. It is unless the profile says otherwise, and naming a
    /// head-logits kernel turns it off: the tensor-core logits path was qualified untiled.
    #[must_use]
    pub fn head_tiled(&self) -> bool {
        self.head_tiled.unwrap_or(self.logits.is_none())
    }

    /// Whether the Markov biases take the tensor cores.
    #[must_use]
    pub fn markov_mma(&self) -> bool {
        self.markov_mma.unwrap_or(true)
    }

    /// Whether the routed rows are deduped before the shared experts.
    #[must_use]
    pub fn moe_dedup(&self) -> bool {
        self.moe_dedup.unwrap_or(true)
    }

    /// Whether the head slice is kept as E4M3 codes and ue8m0 scales.
    #[must_use]
    pub fn head_fp8(&self) -> bool {
        self.head_fp8.unwrap_or(true)
    }

    /// Whether the seed's main_proj columns split over the ranks.
    #[must_use]
    pub fn seed_split(&self) -> bool {
        self.seed_split.unwrap_or(true)
    }

    /// Whether a pass builds its row table.
    #[must_use]
    pub fn build_row_table(&self) -> bool {
        self.row_table.unwrap_or(true)
    }
}

/// The shared routed-expert plan section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlanProfile {
    /// Threads one plan block runs.
    pub threads: Option<usize>,
}

impl PlanProfile {
    /// The block size one plan launch runs unless the profile says otherwise.
    pub const DEFAULT_THREADS: usize = 512;

    /// The plan's block size, held to a warp's granularity and one block's limit.
    ///
    /// The kernel derives every loop bound from `blockDim.x`, so this is a tuning number and
    /// not a fact about the layout: any value in range computes the same plan.
    #[must_use]
    pub fn block_threads(&self) -> u32 {
        self.threads
            .unwrap_or(Self::DEFAULT_THREADS)
            .clamp(32, 1024) as u32
    }
}

/// The sequence-parallel section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SequenceProfile {
    /// Whether a prefill chunk's hyper-connections run sequence-parallel.
    pub parallel: Option<bool>,
    /// Whether such a chunk overlaps its collectives with the other half.
    pub overlap: Option<bool>,
}

/// The probe the device path runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeMode {
    /// No probe.
    Off,
    /// Report non-finite buffers.
    Finite,
    /// Report per-buffer hashes for a line-by-line diff.
    Hash,
    /// Dump the final rows.
    Dump,
}

/// The diagnostics section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsProfile {
    /// Per-stage device timing.
    pub profile: Option<bool>,
    /// The device probe.
    pub probe: Option<ProbeMode>,
    /// The per-step trace.
    pub trace: Option<bool>,
    /// A trace of the speculative choices.
    pub spec_trace: Option<bool>,
    /// Keep the per-layer evidence chain.
    pub forward_evidence: Option<bool>,
    /// Keep the argmax/embedding gathers on the device exchange.
    pub device_gather: Option<bool>,
    /// Read Engram tables with O_DIRECT.
    pub engram_direct: Option<bool>,
    /// Run the host stages beside the device chain for a diff.
    pub device_diff: Option<bool>,
    /// Warm the mHC seams before the first request.
    pub seam_warm: Option<bool>,
    /// Read back only the last head position's logits.
    pub lean_head: Option<bool>,
    /// The layer arm selector (`device`, `host`, ...).
    pub layer_device: Option<String>,
    /// The CUDA graph mode (`off`, `on`, `report`).
    pub cuda_graph: Option<String>,
    /// Split bulk-prefill mHC seams over the world ranks.
    pub prefill_sp: Option<bool>,
    /// Tiles a segment select spreads over.
    pub seg_blocks: Option<usize>,
    /// Read Engram decode steps through io_uring.
    pub engram_uring: Option<bool>,
    /// Split the mHC seam over ranks.
    pub seam_split: Option<bool>,
    /// Split the rows attention into score/softmax/value kernels.
    pub attn_split: Option<bool>,
    /// Use the staged Engram wkv kernel.
    pub engram_staged: Option<bool>,
    /// Split the Engram output over ranks.
    pub engram_split: Option<bool>,
    /// Use the host-weight ring buffers.
    pub rows_ring: Option<bool>,
    /// Use the grouped rows kernels.
    pub rows_groups: Option<bool>,
    /// Screen the greedy head with the int8 table: score only the entries a bound cannot rule out.
    pub head_screen: Option<bool>,
    /// Run every screened head through the full head as well and refuse a step that differs.
    pub head_screen_check: Option<bool>,
    /// Replicate the embedding table on every rank.
    pub embed_replica: Option<bool>,
    /// Prefetch the wq_b weights.
    pub prefetch_wqb: Option<bool>,
    /// Use the rows select kernel.
    pub rows_select: Option<bool>,
    /// Select together in one kernel.
    pub select_together: Option<bool>,
    /// Use the rows indexer kernel.
    pub rows_indexer: Option<bool>,
    /// Use the row-index table.
    pub row_table: Option<bool>,
    /// Reuse a prefix cache.
    pub prefix_cache: Option<bool>,
    /// Read the next Engram rows ahead.
    pub engram_ahead: Option<bool>,
    /// Skip the decoder on exact prefill steps.
    pub prefill_decoder_skip: Option<bool>,
    /// Install the optional MoE image.
    pub opt_moe: Option<bool>,
    /// Install the optional GEMV image.
    pub opt_gemv: Option<bool>,
    /// Install the optional misc image.
    pub opt_misc: Option<bool>,
    /// Install the optional head image.
    pub opt_head: Option<bool>,
    /// Install the optional router image.
    pub opt_router: Option<bool>,
    /// Install the optional seam image.
    pub opt_seams: Option<bool>,
    /// Install the optional one-shot collective image.
    pub oneshot: Option<bool>,
    /// Install the optional attention image.
    pub opt_attn: Option<bool>,
    /// Keep the attention projections resident on the device.
    pub resident_attention: Option<bool>,
    /// Install the optional select image.
    pub opt_select: Option<bool>,
    /// Use the rank-form select kernel.
    pub select_rank: Option<bool>,
    /// Directory for the phase-1 boundary trace.
    pub boundary_trace: Option<String>,
    /// Directory for the TP4 boundary replay.
    pub tp4_boundary_dir: Option<String>,
    /// Comma-separated positions to trace.
    pub trace_positions: Option<String>,
    /// File for the per-position tap dump.
    pub tap_dump: Option<String>,
    /// File for the router route dump.
    pub route_dump: Option<String>,
    /// Directory for the reduce dump.
    pub reduce_dump: Option<String>,
    /// File for the device-diff dump.
    pub device_diff_dump: Option<String>,
    /// Generate measurement-only tokens.
    pub generate: Option<bool>,
    /// Print the per-position argmax.
    pub print_argmax: Option<bool>,
    /// Use the L2 prefetch side stream.
    pub l2_prefetch: Option<bool>,
    /// Cache Engram reads.
    pub engram_cache: Option<bool>,
    /// Keep the Engram reader alive across requests.
    pub engram_keepalive: Option<bool>,
    /// Shard the Engram table prefix.
    pub engram_shard: Option<String>,
    /// Share the TP collectives across layers.
    pub shared_tp: Option<bool>,
    /// Require io_uring rather than falling back.
    pub require_io_uring: Option<bool>,
    /// Enable the vision tower.
    pub vision: Option<bool>,
    /// Print the per-batch speculative trace.
    pub batch_trace: Option<bool>,
    /// Emit the device trace.
    pub device_trace: Option<bool>,
    /// The device-trace event selection.
    pub device_trace_events: Option<String>,
    /// Run the reference host seam beside the device chain.
    pub seam_ref_host: Option<bool>,
    /// Emit seam debug output.
    pub seam_debug: Option<bool>,
    /// Position the TP4 boundary replay writes.
    pub tp4_boundary_position: Option<usize>,
    /// The tolerance arm named in evidence.
    pub tolerance_arm: Option<String>,
    /// Run the router host twin.
    pub router_twin: Option<bool>,
    /// Force the host head assembly.
    pub head_host_assembly: Option<bool>,
}

/// One model's typed runtime profile.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelProfile {
    /// The registry model the profile configures (`numerics/registry.toml`). The declaration
    /// check (`scripts/numerics.py ci`) reads it to know whose references a changed setting needs
    /// recaptured, and a test holds it to the descriptor `weights` selects.
    pub model: Option<String>,
    /// The model family the profile belongs to, for a mismatch check.
    pub family: Option<String>,
    /// The weights selector.
    pub weights: Option<String>,
    /// Runtime width and admission.
    pub runtime: RuntimeProfile,
    /// Bulk prefill.
    pub prefill: PrefillProfile,
    /// Decode steps and speculative verifies.
    pub decode: DecodeProfile,
    /// Attention caches.
    pub attention: AttentionProfile,
    /// Speculation.
    pub speculation: SpeculationProfile,
    /// The speculative drafter's kernel variants.
    pub drafter: DrafterProfile,
    /// Sequence parallelism.
    pub sequence: SequenceProfile,
    /// The shared routed-expert plan.
    pub plan: PlanProfile,
    /// Diagnostics.
    pub diagnostics: DiagnosticsProfile,
}

/// The profile the process booted with.
///
/// A process serves one model, so its profile is process-wide state after boot. Device code that
/// used to read an environment variable at the point of use reads this typed field instead; the
/// environment is applied once when the profile is installed.
static ACTIVE: OnceLock<ModelProfile> = OnceLock::new();

/// Install the profile this process booted with. The first call wins; a later one is ignored.
pub fn install_active(profile: ModelProfile) {
    let _ = ACTIVE.set(profile);
}

/// The installed profile, or the environment-resolved default when no boot installed one.
#[must_use]
pub fn active() -> &'static ModelProfile {
    ACTIVE.get_or_init(|| {
        let mut profile = ModelProfile::default();
        let _ = profile.apply_process_env();
        profile
    })
}

/// The environment overrides this module owns, with the typed field each sets.
///
/// A value the field does not accept is refused rather than silently ignored. These names are
/// aliases: the TOML field is the source of truth and the override exists for an operator's A/B
/// run, not for behaviour a normal boot depends on.
pub const ENV_OVERRIDES: &[(&str, &str)] = &[
    ("SPARK_MAX_SEQ_LEN", "runtime.context_tokens"),
    ("SPARKFLASH_LANES", "runtime.concurrency"),
    ("SPARKFLASH_LANE_MEMORY_MB", "runtime.lane_memory_mb"),
    ("SPARKFLASH_MEM_FLOOR_GIB", "runtime.mem_floor_gib"),
    ("SPARKFLASH_CONSTRAINED", "runtime.constrained"),
    ("SPARKFLASH_ARENA_MMAP", "runtime.arena_mmap"),
    ("SPARKFLASH_ARENA_HOSTMAPPED", "runtime.arena_hostmapped"),
    ("SPARKFLASH_VERIFY_WEIGHTS", "runtime.verify_weights"),
    ("SPARKFLASH_CONVERTED_DIR", "runtime.converted_dir"),
    ("SPARKFLASH_ROWS", "runtime.rows"),
    ("SPARKFLASH_ROWS_FROM", "runtime.rows_from"),
    ("SPARKFLASH_PREFILL", "prefill.enabled"),
    ("SPARKFLASH_GLM_PREFILL", "prefill.enabled"),
    ("SPARKFLASH_PREFILL_CHUNK", "prefill.chunk"),
    ("SPARKFLASH_PREFILL_EXACT", "prefill.exact"),
    ("SPARKFLASH_PREFILL_CHUNK_BUSY", "prefill.chunk_busy"),
    ("SPARKFLASH_PREFILL_EXACT_BUSY", "prefill.exact_busy"),
    ("SPARKFLASH_PREFILL_READ_THREADS", "prefill.read_threads"),
    ("SPARKFLASH_BULK_CTAS", "prefill.bulk_ctas"),
    ("SPARKFLASH_PREFILL_MOE_MERGE", "prefill.moe_merge"),
    ("SPARKFLASH_PREFILL_BULK_MIN", "prefill.bulk_min"),
    ("SPARKFLASH_PREFILL_WAYS", "prefill.ways"),
    ("SPARKFLASH_PREFILL_PAIR", "prefill.pair"),
    ("SPARKFLASH_DECODE_SHARE", "prefill.decode_share"),
    ("SPARKFLASH_PREFILL_DEADLINE_MS", "prefill.deadline_ms"),
    ("SPARKFLASH_WARMUP", "prefill.warmup"),
    ("SPARKFLASH_WARMUP_STEPS", "prefill.warmup_steps"),
    ("SPARKFLASH_TICK_AHEAD", "prefill.tick_ahead"),
    ("SPARKFLASH_QWEN4_PREFILL_CHUNK", "prefill.chunk"),
    ("SPARKFLASH_GLM_PREFILL_CHUNK", "prefill.chunk"),
    ("SPARKFLASH_QWEN4_PREFILL_MMA", "prefill.tensor_cores"),
    ("SPARKFLASH_GLM_PREFILL_EXACT", "prefill.exact"),
    ("SPARKFLASH_PREFILL_DRAFTING", "prefill.drafting_prompt"),
    ("SPARKFLASH_GLM_PREFILL_DRAFTING", "prefill.drafting_prompt"),
    ("SPARKFLASH_GLM_PREFILL_ACTIVATION", "prefill.activation"),
    ("SPARKFLASH_DECODE_MMA", "decode.tensor_cores"),
    ("SPARKFLASH_GLM_LATENT_CACHE", "attention.latent_cache"),
    ("SPARKFLASH_DSPARK", "speculation.enabled"),
    ("SPARKFLASH_QWEN4_SPEC", "speculation.enabled"),
    ("SPARKFLASH_QWEN4_SPEC_DEPTH", "speculation.depth"),
    ("SPARKFLASH_DSPARK_K", "speculation.dspark_k"),
    ("SPARKFLASH_DSPARK_DEPTH", "speculation.depth_policy"),
    ("SPARKFLASH_DSPARK_MIN_DEPTH", "speculation.min_depth"),
    ("SPARKFLASH_QWEN4_DFLASH2_DEPTH", "speculation.depth_policy"),
    (
        "SPARKFLASH_QWEN4_DFLASH2_MIN_DEPTH",
        "speculation.min_depth",
    ),
    ("SPARKFLASH_DSPARK_STEPS", "speculation.dspark_steps"),
    ("SPARKFLASH_BATCH_SPEC", "speculation.batch_spec"),
    (
        "SPARKFLASH_BATCH_SPEC_FLOOR",
        "speculation.batch_spec_floor",
    ),
    ("SPARKFLASH_QWEN4_MTP_FOLD", "speculation.mtp_fold_scale"),
    ("SPARKFLASH_QWEN4_DFLASH2", "speculation.drafter"),
    ("SPARKFLASH_GLM_SPEC", "speculation.enabled"),
    ("SPARKFLASH_GLM_SPEC_DEPTH", "speculation.depth"),
    ("SPARKFLASH_GLM_DRAFTS", "speculation.drafts"),
    ("SPARKFLASH_GLM_DFLASH2", "speculation.drafter"),
    ("SPARKFLASH_GLM_PROF", "diagnostics.profile"),
    ("SPARKFLASH_SEQUENCE_PARALLEL", "sequence.parallel"),
    ("SPARKFLASH_PLAN_THREADS", "plan.threads"),
    ("SPARKFLASH_SEQUENCE_OVERLAP", "sequence.overlap"),
    ("SPARKFLASH_QWEN4_PROF", "diagnostics.profile"),
    ("SPARKFLASH_LAYER_DEVICE", "diagnostics.layer_device"),
    ("SPARKFLASH_CUDA_GRAPH", "diagnostics.cuda_graph"),
    ("SPARKFLASH_PREFILL_SP", "diagnostics.prefill_sp"),
    ("SPARKFLASH_SEG_BLOCKS", "diagnostics.seg_blocks"),
    ("SPARKFLASH_ENGRAM_URING", "diagnostics.engram_uring"),
    ("SPARKFLASH_SEAM_SPLIT", "diagnostics.seam_split"),
    ("SPARKFLASH_ATTN_SPLIT", "diagnostics.attn_split"),
    ("SPARKFLASH_ENGRAM_STAGED", "diagnostics.engram_staged"),
    ("SPARKFLASH_ENGRAM_SPLIT", "diagnostics.engram_split"),
    ("SPARKFLASH_ROWS_RING", "diagnostics.rows_ring"),
    ("SPARKFLASH_ROWS_GROUPS", "diagnostics.rows_groups"),
    ("SPARKFLASH_HEAD_SCREEN", "diagnostics.head_screen"),
    (
        "SPARKFLASH_HEAD_SCREEN_CHECK",
        "diagnostics.head_screen_check",
    ),
    ("SPARKFLASH_EMBED_REPLICA", "diagnostics.embed_replica"),
    ("SPARKFLASH_PREFETCH_WQB", "diagnostics.prefetch_wqb"),
    ("SPARKFLASH_ROWS_SELECT", "diagnostics.rows_select"),
    ("SPARKFLASH_SELECT_TOGETHER", "diagnostics.select_together"),
    ("SPARKFLASH_ROWS_INDEXER", "diagnostics.rows_indexer"),
    ("SPARKFLASH_ROW_TABLE", "diagnostics.row_table"),
    ("SPARKFLASH_PREFIX_CACHE", "diagnostics.prefix_cache"),
    ("SPARKFLASH_ENGRAM_AHEAD", "diagnostics.engram_ahead"),
    (
        "SPARKFLASH_PREFILL_DECODER_SKIP",
        "diagnostics.prefill_decoder_skip",
    ),
    ("SPARKFLASH_OPT_MOE", "diagnostics.opt_moe"),
    ("SPARKFLASH_OPT_GEMV", "diagnostics.opt_gemv"),
    ("SPARKFLASH_OPT_MISC", "diagnostics.opt_misc"),
    ("SPARKFLASH_OPT_HEAD", "diagnostics.opt_head"),
    ("SPARKFLASH_OPT_ROUTER", "diagnostics.opt_router"),
    ("SPARKFLASH_OPT_SEAMS", "diagnostics.opt_seams"),
    ("SPARKFLASH_ONESHOT", "diagnostics.oneshot"),
    ("SPARKFLASH_OPT_ATTN", "diagnostics.opt_attn"),
    (
        "SPARKFLASH_RESIDENT_ATTENTION",
        "diagnostics.resident_attention",
    ),
    ("SPARKFLASH_OPT_SELECT", "diagnostics.opt_select"),
    ("SPARKFLASH_SELECT_RANK", "diagnostics.select_rank"),
    ("SPARKFLASH_BOUNDARY_TRACE", "diagnostics.boundary_trace"),
    (
        "SPARKFLASH_TP4_BOUNDARY_DIR",
        "diagnostics.tp4_boundary_dir",
    ),
    ("SPARKFLASH_TRACE_POSITIONS", "diagnostics.trace_positions"),
    ("SPARKFLASH_TAP_DUMP", "diagnostics.tap_dump"),
    ("SPARKFLASH_ROUTE_DUMP", "diagnostics.route_dump"),
    ("SPARKFLASH_REDUCE_DUMP", "diagnostics.reduce_dump"),
    (
        "SPARKFLASH_DEVICE_DIFF_DUMP",
        "diagnostics.device_diff_dump",
    ),
    ("SPARKFLASH_GENERATE", "diagnostics.generate"),
    ("SPARKFLASH_PRINT_ARGMAX", "diagnostics.print_argmax"),
    ("SPARKFLASH_L2_PREFETCH", "diagnostics.l2_prefetch"),
    ("SPARKFLASH_ENGRAM_CACHE", "diagnostics.engram_cache"),
    (
        "SPARKFLASH_ENGRAM_KEEPALIVE",
        "diagnostics.engram_keepalive",
    ),
    ("SPARKFLASH_ENGRAM_SHARD", "diagnostics.engram_shard"),
    ("SPARKFLASH_SHARED_TP", "diagnostics.shared_tp"),
    (
        "SPARKFLASH_REQUIRE_IO_URING",
        "diagnostics.require_io_uring",
    ),
    ("SPARKFLASH_VISION", "diagnostics.vision"),
    ("SPARKFLASH_BATCH_TRACE", "diagnostics.batch_trace"),
    ("SPARKFLASH_SPEC_TRACE", "diagnostics.spec_trace"),
    ("SPARKFLASH_DEVICE_TRACE", "diagnostics.device_trace"),
    (
        "SPARKFLASH_DEVICE_TRACE_EVENTS",
        "diagnostics.device_trace_events",
    ),
    ("SPARKFLASH_SEAM_REF_HOST", "diagnostics.seam_ref_host"),
    ("SPARKFLASH_SEAM_DEBUG", "diagnostics.seam_debug"),
    (
        "SPARKFLASH_TP4_BOUNDARY_POSITION",
        "diagnostics.tp4_boundary_position",
    ),
    ("SPARKFLASH_TOLERANCE_ARM", "diagnostics.tolerance_arm"),
    ("SPARKFLASH_ROUTER_TWIN", "diagnostics.router_twin"),
    (
        "SPARKFLASH_HEAD_HOST_ASSEMBLY",
        "diagnostics.head_host_assembly",
    ),
    (
        "SPARKFLASH_FORWARD_EVIDENCE",
        "diagnostics.forward_evidence",
    ),
    ("SPARKFLASH_DEVICE_GATHER", "diagnostics.device_gather"),
    ("SPARKFLASH_ENGRAM_DIRECT", "diagnostics.engram_direct"),
    ("SPARKFLASH_DEVICE_DIFF", "diagnostics.device_diff"),
    ("SPARKFLASH_SEAM_WARM", "diagnostics.seam_warm"),
    ("SPARKFLASH_LEAN_HEAD", "diagnostics.lean_head"),
    ("SPARKFLASH_QWEN4_PROBE", "diagnostics.probe"),
    ("SPARKFLASH_QWEN4_TRACE", "diagnostics.trace"),
    ("SPARKFLASH_QWEN4_SPEC_TRACE", "diagnostics.spec_trace"),
    ("SPARKFLASH_DRAFT_LOGITS_MMA", "drafter.logits"),
    ("SPARKFLASH_DRAFT_HEAD_TILED", "drafter.head_tiled"),
    ("SPARKFLASH_DRAFT_MARKOV_MMA", "drafter.markov_mma"),
    ("SPARKFLASH_DRAFT_SEQS", "drafter.seqs"),
    ("SPARKFLASH_DRAFT_MOE_DEDUP", "drafter.moe_dedup"),
    ("SPARKFLASH_DSPARK_HEAD_FP8", "drafter.head_fp8"),
    ("SPARKFLASH_DRAFT_LOGITS_MMA_ROWS", "drafter.logits_rows"),
    ("SPARKFLASH_DRAFT_SEED_SPLIT", "drafter.seed_split"),
    ("SPARKFLASH_DRAFT_TABLE", "drafter.row_table"),
];

/// Reads and writes one `Option<bool>` field of a [ModelProfile].
type BoolField = fn(&mut ModelProfile) -> &mut Option<bool>;
/// Reads and writes one `Option<usize>` field of a [ModelProfile].
type CountField = fn(&mut ModelProfile) -> &mut Option<usize>;
/// Reads and writes one `Option<f32>` field of a [ModelProfile].
type F32Field = fn(&mut ModelProfile) -> &mut Option<f32>;
/// Reads and writes one `Option<f64>` field of a [ModelProfile].
type F64Field = fn(&mut ModelProfile) -> &mut Option<f64>;
/// Reads and writes one `Option<String>` field of a [ModelProfile].
type TextField = fn(&mut ModelProfile) -> &mut Option<String>;

/// The boolean fields an [ENV_OVERRIDES] alias writes.
const BOOL_FIELDS: &[(&str, BoolField)] = &[
    ("prefill.enabled", |p| &mut p.prefill.enabled),
    ("prefill.exact", |p| &mut p.prefill.exact),
    ("prefill.moe_merge", |p| &mut p.prefill.moe_merge),
    ("prefill.warmup", |p| &mut p.prefill.warmup),
    ("prefill.tick_ahead", |p| &mut p.prefill.tick_ahead),
    ("prefill.tensor_cores", |p| &mut p.prefill.tensor_cores),
    ("prefill.drafting_prompt", |p| {
        &mut p.prefill.drafting_prompt
    }),
    ("decode.tensor_cores", |p| &mut p.decode.tensor_cores),
    ("speculation.enabled", |p| &mut p.speculation.enabled),
    ("speculation.batch_spec", |p| &mut p.speculation.batch_spec),
    ("sequence.parallel", |p| &mut p.sequence.parallel),
    ("sequence.overlap", |p| &mut p.sequence.overlap),
    ("diagnostics.profile", |p| &mut p.diagnostics.profile),
    ("diagnostics.trace", |p| &mut p.diagnostics.trace),
    ("diagnostics.spec_trace", |p| &mut p.diagnostics.spec_trace),
    ("diagnostics.forward_evidence", |p| {
        &mut p.diagnostics.forward_evidence
    }),
    ("diagnostics.device_gather", |p| {
        &mut p.diagnostics.device_gather
    }),
    ("diagnostics.engram_direct", |p| {
        &mut p.diagnostics.engram_direct
    }),
    ("diagnostics.device_diff", |p| {
        &mut p.diagnostics.device_diff
    }),
    ("diagnostics.seam_warm", |p| &mut p.diagnostics.seam_warm),
    ("diagnostics.lean_head", |p| &mut p.diagnostics.lean_head),
    ("diagnostics.prefill_sp", |p| &mut p.diagnostics.prefill_sp),
    ("diagnostics.engram_uring", |p| {
        &mut p.diagnostics.engram_uring
    }),
    ("diagnostics.seam_split", |p| &mut p.diagnostics.seam_split),
    ("diagnostics.attn_split", |p| &mut p.diagnostics.attn_split),
    ("diagnostics.engram_staged", |p| {
        &mut p.diagnostics.engram_staged
    }),
    ("diagnostics.engram_split", |p| {
        &mut p.diagnostics.engram_split
    }),
    ("diagnostics.rows_ring", |p| &mut p.diagnostics.rows_ring),
    ("diagnostics.rows_groups", |p| {
        &mut p.diagnostics.rows_groups
    }),
    ("diagnostics.head_screen", |p| {
        &mut p.diagnostics.head_screen
    }),
    ("diagnostics.head_screen_check", |p| {
        &mut p.diagnostics.head_screen_check
    }),
    ("diagnostics.embed_replica", |p| {
        &mut p.diagnostics.embed_replica
    }),
    ("diagnostics.prefetch_wqb", |p| {
        &mut p.diagnostics.prefetch_wqb
    }),
    ("diagnostics.rows_select", |p| {
        &mut p.diagnostics.rows_select
    }),
    ("diagnostics.select_together", |p| {
        &mut p.diagnostics.select_together
    }),
    ("diagnostics.rows_indexer", |p| {
        &mut p.diagnostics.rows_indexer
    }),
    ("diagnostics.row_table", |p| &mut p.diagnostics.row_table),
    ("diagnostics.prefix_cache", |p| {
        &mut p.diagnostics.prefix_cache
    }),
    ("diagnostics.engram_ahead", |p| {
        &mut p.diagnostics.engram_ahead
    }),
    ("diagnostics.prefill_decoder_skip", |p| {
        &mut p.diagnostics.prefill_decoder_skip
    }),
    ("diagnostics.opt_moe", |p| &mut p.diagnostics.opt_moe),
    ("diagnostics.opt_gemv", |p| &mut p.diagnostics.opt_gemv),
    ("diagnostics.opt_misc", |p| &mut p.diagnostics.opt_misc),
    ("diagnostics.opt_head", |p| &mut p.diagnostics.opt_head),
    ("diagnostics.opt_router", |p| &mut p.diagnostics.opt_router),
    ("diagnostics.opt_seams", |p| &mut p.diagnostics.opt_seams),
    ("diagnostics.oneshot", |p| &mut p.diagnostics.oneshot),
    ("diagnostics.opt_attn", |p| &mut p.diagnostics.opt_attn),
    ("diagnostics.opt_select", |p| &mut p.diagnostics.opt_select),
    ("diagnostics.resident_attention", |p| {
        &mut p.diagnostics.resident_attention
    }),
    ("runtime.constrained", |p| &mut p.runtime.constrained),
    ("diagnostics.select_rank", |p| {
        &mut p.diagnostics.select_rank
    }),
    ("diagnostics.generate", |p| &mut p.diagnostics.generate),
    ("diagnostics.print_argmax", |p| {
        &mut p.diagnostics.print_argmax
    }),
    ("diagnostics.l2_prefetch", |p| {
        &mut p.diagnostics.l2_prefetch
    }),
    ("diagnostics.engram_cache", |p| {
        &mut p.diagnostics.engram_cache
    }),
    ("diagnostics.engram_keepalive", |p| {
        &mut p.diagnostics.engram_keepalive
    }),
    ("diagnostics.shared_tp", |p| &mut p.diagnostics.shared_tp),
    ("diagnostics.require_io_uring", |p| {
        &mut p.diagnostics.require_io_uring
    }),
    ("diagnostics.vision", |p| &mut p.diagnostics.vision),
    ("diagnostics.batch_trace", |p| {
        &mut p.diagnostics.batch_trace
    }),
    ("diagnostics.device_trace", |p| {
        &mut p.diagnostics.device_trace
    }),
    ("diagnostics.seam_ref_host", |p| {
        &mut p.diagnostics.seam_ref_host
    }),
    ("diagnostics.seam_debug", |p| &mut p.diagnostics.seam_debug),
    ("diagnostics.router_twin", |p| {
        &mut p.diagnostics.router_twin
    }),
    ("diagnostics.head_host_assembly", |p| {
        &mut p.diagnostics.head_host_assembly
    }),
    ("runtime.arena_mmap", |p| &mut p.runtime.arena_mmap),
    ("runtime.arena_hostmapped", |p| {
        &mut p.runtime.arena_hostmapped
    }),
    ("runtime.verify_weights", |p| &mut p.runtime.verify_weights),
    ("drafter.head_tiled", |p| &mut p.drafter.head_tiled),
    ("drafter.markov_mma", |p| &mut p.drafter.markov_mma),
    ("drafter.moe_dedup", |p| &mut p.drafter.moe_dedup),
    ("drafter.head_fp8", |p| &mut p.drafter.head_fp8),
    ("drafter.seed_split", |p| &mut p.drafter.seed_split),
    ("drafter.row_table", |p| &mut p.drafter.row_table),
];

/// The count fields an [ENV_OVERRIDES] alias writes.
const COUNT_FIELDS: &[(&str, CountField)] = &[
    ("runtime.context_tokens", |p| &mut p.runtime.context_tokens),
    ("runtime.concurrency", |p| &mut p.runtime.concurrency),
    ("runtime.lane_memory_mb", |p| &mut p.runtime.lane_memory_mb),
    ("runtime.mem_floor_gib", |p| &mut p.runtime.mem_floor_gib),
    ("runtime.rows", |p| &mut p.runtime.rows),
    ("runtime.rows_from", |p| &mut p.runtime.rows_from),
    ("prefill.chunk", |p| &mut p.prefill.chunk),
    ("prefill.chunk_busy", |p| &mut p.prefill.chunk_busy),
    ("prefill.exact_busy", |p| &mut p.prefill.exact_busy),
    ("prefill.read_threads", |p| &mut p.prefill.read_threads),
    ("prefill.bulk_ctas", |p| &mut p.prefill.bulk_ctas),
    ("prefill.bulk_min", |p| &mut p.prefill.bulk_min),
    ("prefill.ways", |p| &mut p.prefill.ways),
    ("prefill.pair", |p| &mut p.prefill.pair),
    ("prefill.warmup_steps", |p| &mut p.prefill.warmup_steps),
    ("speculation.drafts", |p| &mut p.speculation.drafts),
    ("speculation.depth", |p| &mut p.speculation.depth),
    ("speculation.dspark_k", |p| &mut p.speculation.dspark_k),
    ("speculation.dspark_steps", |p| {
        &mut p.speculation.dspark_steps
    }),
    ("speculation.min_depth", |p| &mut p.speculation.min_depth),
    ("diagnostics.seg_blocks", |p| &mut p.diagnostics.seg_blocks),
    ("diagnostics.tp4_boundary_position", |p| {
        &mut p.diagnostics.tp4_boundary_position
    }),
    ("drafter.seqs", |p| &mut p.drafter.seqs),
    ("drafter.logits_rows", |p| &mut p.drafter.logits_rows),
    ("plan.threads", |p| &mut p.plan.threads),
];

/// The float fields an [ENV_OVERRIDES] alias writes.
const F32_FIELDS: &[(&str, F32Field)] = &[("speculation.mtp_fold_scale", |p| {
    &mut p.speculation.mtp_fold_scale
})];

/// The double fields an [ENV_OVERRIDES] alias writes.
const F64_FIELDS: &[(&str, F64Field)] = &[
    ("speculation.batch_spec_floor", |p| {
        &mut p.speculation.batch_spec_floor
    }),
    ("prefill.decode_share", |p| &mut p.prefill.decode_share),
    ("prefill.deadline_ms", |p| &mut p.prefill.deadline_ms),
];

/// The text fields an [ENV_OVERRIDES] alias writes.
const TEXT_FIELDS: &[(&str, TextField)] = &[
    ("runtime.converted_dir", |p| &mut p.runtime.converted_dir),
    ("speculation.drafter", |p| &mut p.speculation.drafter),
    ("diagnostics.layer_device", |p| {
        &mut p.diagnostics.layer_device
    }),
    ("diagnostics.cuda_graph", |p| &mut p.diagnostics.cuda_graph),
    ("diagnostics.boundary_trace", |p| {
        &mut p.diagnostics.boundary_trace
    }),
    ("diagnostics.tp4_boundary_dir", |p| {
        &mut p.diagnostics.tp4_boundary_dir
    }),
    ("diagnostics.trace_positions", |p| {
        &mut p.diagnostics.trace_positions
    }),
    ("diagnostics.tap_dump", |p| &mut p.diagnostics.tap_dump),
    ("diagnostics.route_dump", |p| &mut p.diagnostics.route_dump),
    ("diagnostics.reduce_dump", |p| {
        &mut p.diagnostics.reduce_dump
    }),
    ("diagnostics.device_diff_dump", |p| {
        &mut p.diagnostics.device_diff_dump
    }),
    ("diagnostics.tolerance_arm", |p| {
        &mut p.diagnostics.tolerance_arm
    }),
    ("diagnostics.engram_shard", |p| {
        &mut p.diagnostics.engram_shard
    }),
    ("diagnostics.device_trace_events", |p| {
        &mut p.diagnostics.device_trace_events
    }),
];

/// The fields whose override value names one of a closed set of variants.
#[cfg(test)]
const ENUM_FIELDS: &[&str] = &[
    "prefill.activation",
    "attention.latent_cache",
    "diagnostics.probe",
    "drafter.logits",
    "speculation.depth_policy",
];

impl ModelProfile {
    /// Parse a profile from TOML, refusing unknown keys.
    ///
    /// # Errors
    /// A syntax error, an unknown key or an unknown enum value.
    pub fn from_toml(text: &str) -> Result<Self, ProfileError> {
        let profile: Self =
            toml::from_str(text).map_err(|error| ProfileError::Parse(error.to_string()))?;
        profile.validate()?;
        Ok(profile)
    }

    /// Read a profile from a file.
    ///
    /// # Errors
    /// A read error, or as [Self::from_toml].
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text)
    }

    /// Refuse an internally inconsistent profile.
    ///
    /// # Errors
    /// A zero width, a chunk that is not a multiple of the 64-token granularity, a draft depth
    /// of zero, or a non-positive fold scale.
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.runtime.context_tokens == Some(0) {
            return Err(ProfileError::Invalid(
                "runtime.context_tokens is zero".to_owned(),
            ));
        }
        if self.runtime.concurrency == Some(0) {
            return Err(ProfileError::Invalid(
                "runtime.concurrency is zero".to_owned(),
            ));
        }
        if let Some(chunk) = self.prefill.chunk
            && (chunk == 0 || !chunk.is_multiple_of(64))
        {
            return Err(ProfileError::Invalid(format!(
                "prefill.chunk {chunk} is not a positive multiple of 64"
            )));
        }
        if self.speculation.depth == Some(0) {
            return Err(ProfileError::Invalid(
                "speculation.depth is zero".to_owned(),
            ));
        }
        if let Some(scale) = self.speculation.mtp_fold_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(ProfileError::Invalid(format!(
                "speculation.mtp_fold_scale {scale} is not a positive finite number"
            )));
        }
        Ok(())
    }

    /// Apply environment overrides, refusing values outside a field's domain.
    ///
    /// # Errors
    /// A malformed number, a switch other than 0, 1, off or on, or an unknown probe or activation
    /// name.
    pub fn apply_env(
        &mut self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<(), ProfileError> {
        for (name, _) in ENV_OVERRIDES {
            if let Some(value) = lookup(name) {
                self.apply_one(name, &value)?;
            }
        }
        self.validate()
    }

    /// Apply the process environment.
    ///
    /// # Errors
    /// As [Self::apply_env].
    pub fn apply_process_env(&mut self) -> Result<(), ProfileError> {
        self.apply_env(|name| std::env::var(name).ok())
    }

    fn apply_one(&mut self, name: &'static str, value: &str) -> Result<(), ProfileError> {
        let field = ENV_OVERRIDES
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map_or("unknown", |(_, field)| *field);
        if let Some((_, target)) = BOOL_FIELDS
            .iter()
            .find(|(candidate, _)| *candidate == field)
        {
            *target(self) = Some(parse_bool(name, value, field)?);
            return Ok(());
        }
        if let Some((_, target)) = COUNT_FIELDS
            .iter()
            .find(|(candidate, _)| *candidate == field)
        {
            *target(self) = Some(parse_usize(name, value, field)?);
            return Ok(());
        }
        // `sum` is the spelling the Qwen arm has always documented for this switch (it means the
        // plain sum of the hyper-connection streams, a scale of one); anything else is the scale
        // itself. Without this the one value the arm accepts could not pass the typed path at all:
        // it parsed as a float and failed with "a positive number".
        if field == "speculation.mtp_fold_scale" && value == "sum" {
            self.speculation.mtp_fold_scale = Some(1.0);
            return Ok(());
        }
        if let Some((_, target)) = F32_FIELDS.iter().find(|(candidate, _)| *candidate == field) {
            *target(self) = Some(parse_float(name, value, field)?);
            return Ok(());
        }
        if let Some((_, target)) = F64_FIELDS.iter().find(|(candidate, _)| *candidate == field) {
            *target(self) = Some(f64::from(parse_float(name, value, field)?));
            return Ok(());
        }
        if let Some((_, target)) = TEXT_FIELDS
            .iter()
            .find(|(candidate, _)| *candidate == field)
        {
            *target(self) = Some(value.to_owned());
            return Ok(());
        }
        match field {
            "prefill.activation" => {
                self.prefill.activation = Some(match value {
                    "fp8" => ActivationMode::Fp8,
                    "split" => ActivationMode::Split,
                    "bf16" => ActivationMode::Bf16,
                    other => return Err(env_error(name, other, field, "fp8, split or bf16")),
                });
            }
            "attention.latent_cache" => {
                self.attention.latent_cache = Some(match value {
                    "auto" => LatentCacheMode::Auto,
                    "f32" => LatentCacheMode::F32,
                    "fp8" => LatentCacheMode::Fp8,
                    other => return Err(env_error(name, other, field, "auto, f32 or fp8")),
                });
            }
            "diagnostics.probe" => {
                self.diagnostics.probe = Some(match value {
                    "0" | "off" => ProbeMode::Off,
                    "1" | "finite" => ProbeMode::Finite,
                    "hash" => ProbeMode::Hash,
                    "dump" => ProbeMode::Dump,
                    other => return Err(env_error(name, other, field, "0, 1, hash or dump")),
                });
            }
            "speculation.depth_policy" => {
                self.speculation.depth_policy = Some(match value {
                    "adaptive" | "auto" => DepthMode::Adaptive,
                    "fixed" => DepthMode::Fixed,
                    other => {
                        return Err(env_error(name, other, field, "adaptive or fixed"));
                    }
                });
            }
            "drafter.logits" => {
                self.drafter.logits = Some(match value {
                    "0" | "simt" => DraftLogitsKernel::Simt,
                    "1" | "on" | "bf16" => DraftLogitsKernel::Bf16,
                    "fp8" => DraftLogitsKernel::Fp8,
                    other => {
                        return Err(env_error(name, other, field, "0, bf16 or fp8"));
                    }
                });
            }
            other => return Err(ProfileError::Invalid(format!("no override for {other}"))),
        }
        Ok(())
    }

    /// Render the resolved profile as TOML for the boot log.
    ///
    /// # Errors
    /// A serialization failure, which cannot happen for this plain struct.
    pub fn to_toml(&self) -> Result<String, ProfileError> {
        toml::to_string_pretty(self).map_err(|error| ProfileError::Parse(error.to_string()))
    }

    /// A one-block boot summary: every set field, with its typed value.
    #[must_use]
    pub fn boot_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "sparkflash profile (model={}, family={}, weights={})",
            self.model.as_deref().unwrap_or("-"),
            self.family.as_deref().unwrap_or("-"),
            self.weights.as_deref().unwrap_or("-")
        );
        let mut rows = BTreeMap::new();
        macro_rules! row {
            ($key:literal, $value:expr) => {
                if let Some(value) = $value {
                    rows.insert($key, format!("{value:?}"));
                }
            };
        }
        row!("runtime.context_tokens", self.runtime.context_tokens);
        row!("runtime.concurrency", self.runtime.concurrency);
        row!("runtime.kv_budget_gib", self.runtime.kv_budget_gib);
        row!("runtime.device", self.runtime.device);
        row!("prefill.enabled", self.prefill.enabled);
        row!("prefill.chunk", self.prefill.chunk);
        row!("prefill.tensor_cores", self.prefill.tensor_cores);
        row!("prefill.activation", self.prefill.activation);
        row!("prefill.exact", self.prefill.exact);
        row!("prefill.drafting_prompt", self.prefill.drafting_prompt);
        row!("decode.tensor_cores", self.decode.tensor_cores);
        row!("attention.latent_cache", self.attention.latent_cache);
        row!("attention.selected_tokens", self.attention.selected_tokens);
        row!("speculation.enabled", self.speculation.enabled);
        row!("speculation.depth", self.speculation.depth);
        row!(
            "speculation.mtp_fold_scale",
            self.speculation.mtp_fold_scale
        );
        row!("speculation.drafter", self.speculation.drafter.as_deref());
        row!("drafter.logits", self.drafter.logits);
        row!("drafter.head_tiled", self.drafter.head_tiled);
        row!("drafter.markov_mma", self.drafter.markov_mma);
        row!("drafter.seqs", self.drafter.seqs);
        row!("drafter.moe_dedup", self.drafter.moe_dedup);
        row!("drafter.head_fp8", self.drafter.head_fp8);
        row!("drafter.logits_rows", self.drafter.logits_rows);
        row!("drafter.seed_split", self.drafter.seed_split);
        row!("drafter.row_table", self.drafter.row_table);
        row!("sequence.parallel", self.sequence.parallel);
        row!("sequence.overlap", self.sequence.overlap);
        row!("diagnostics.profile", self.diagnostics.profile);
        row!("diagnostics.probe", self.diagnostics.probe);
        row!("diagnostics.trace", self.diagnostics.trace);
        row!("diagnostics.spec_trace", self.diagnostics.spec_trace);
        for (key, value) in rows {
            let _ = writeln!(out, "  {key} = {value}");
        }
        out
    }
}

fn parse_usize(
    name: &'static str,
    value: &str,
    field: &'static str,
) -> Result<usize, ProfileError> {
    value
        .parse()
        .map_err(|_| env_error(name, value, field, "a non-negative integer"))
}

fn parse_float(name: &'static str, value: &str, field: &'static str) -> Result<f32, ProfileError> {
    value
        .parse()
        .map_err(|_| env_error(name, value, field, "a positive number"))
}

fn parse_bool(name: &'static str, value: &str, field: &'static str) -> Result<bool, ProfileError> {
    match value {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        other => Err(env_error(name, other, field, "0 or 1 (off or on)")),
    }
}

fn env_error(
    name: &'static str,
    value: &str,
    field: &'static str,
    expected: &'static str,
) -> ProfileError {
    ProfileError::Env {
        name,
        value: value.to_owned(),
        field,
        expected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = r#"
family = "qwen4-exp"
weights = "qwen3.8-flash-next-nvfp4"

[runtime]
context_tokens = 262144
concurrency = 2

[prefill]
enabled = true
chunk = 4096
tensor_cores = true
activation = "split"

[attention]
latent_cache = "auto"

[diagnostics]
probe = "hash"
"#;

    #[test]
    fn a_typed_profile_parses_and_prints_every_set_field() {
        let profile = ModelProfile::from_toml(PROFILE).expect("the profile parses");
        assert_eq!(profile.prefill.activation, Some(ActivationMode::Split));
        assert_eq!(profile.diagnostics.probe, Some(ProbeMode::Hash));
        let summary = profile.boot_summary();
        assert!(summary.contains("prefill.chunk = 4096"), "{summary}");
        assert!(summary.contains("diagnostics.probe = Hash"), "{summary}");
    }

    /// `scripts/numerics.py ci` asks each arm profile's `model` whose references a changed setting
    /// needs recaptured. This holds that answer to the descriptor the profile's weights select, so
    /// a wrong `model` cannot send the recapture to another model's references.
    #[test]
    fn every_arm_profile_names_the_model_its_weights_select() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates folder");
        let mut profiles = Vec::new();
        for arm in std::fs::read_dir(crates).expect("the crates folder lists") {
            let arm = arm.expect("a crate").path();
            let is_arm = arm
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sparkflash-model-"));
            let Ok(config) = std::fs::read_dir(arm.join("config")) else {
                continue;
            };
            if !is_arm {
                continue;
            }
            profiles.extend(
                config
                    .map(|file| file.expect("a profile").path())
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "toml")
                    }),
            );
        }
        assert!(profiles.len() >= 3, "found only {profiles:?}");
        for path in profiles {
            let shown = path.display();
            let profile =
                ModelProfile::load(&path).unwrap_or_else(|error| panic!("{shown}: {error}"));
            let weights = profile
                .weights
                .as_deref()
                .unwrap_or_else(|| panic!("{shown} names no weights"));
            let descriptor = crate::descriptor::find_model_by_weights(weights)
                .unwrap_or_else(|| panic!("{shown}: no descriptor serves the weights {weights}"));
            assert_eq!(
                profile.model.as_deref(),
                Some(descriptor.id.as_str()),
                "{shown} must name the model its weights {weights} select"
            );
        }
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let error = ModelProfile::from_toml("[runtime]\nconcurreny = 2\n").expect_err("typo");
        assert!(matches!(error, ProfileError::Parse(_)));
    }

    #[test]
    fn an_unknown_enum_value_is_refused() {
        let error = ModelProfile::from_toml("[prefill]\nactivation = \"fp4\"\n")
            .expect_err("unknown activation");
        assert!(matches!(error, ProfileError::Parse(_)));
    }

    #[test]
    fn the_environment_is_an_override_with_a_typed_domain() {
        let mut profile = ModelProfile::from_toml(PROFILE).expect("the profile parses");
        profile
            .apply_env(|name| match name {
                "SPARKFLASH_QWEN4_PREFILL_CHUNK" => Some("2048".to_owned()),
                "SPARKFLASH_QWEN4_PROBE" => Some("dump".to_owned()),
                "SPARKFLASH_QWEN4_SPEC" => Some("0".to_owned()),
                _ => None,
            })
            .expect("the overrides are valid");
        assert_eq!(profile.prefill.chunk, Some(2048));
        assert_eq!(profile.diagnostics.probe, Some(ProbeMode::Dump));
        assert_eq!(profile.speculation.enabled, Some(false));
    }

    #[test]
    fn an_environment_value_outside_the_domain_is_refused() {
        let mut profile = ModelProfile::default();
        let error = profile
            .apply_env(|name| (name == "SPARKFLASH_QWEN4_PREFILL_MMA").then(|| "yes".to_owned()))
            .expect_err("yes is not a bool");
        assert!(matches!(error, ProfileError::Env { .. }));
    }

    #[test]
    fn a_switch_takes_the_deepseek_spellings_too() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| (name == "SPARKFLASH_DECODE_MMA").then(|| "off".to_owned()))
            .expect("off is a switch value");
        assert_eq!(profile.decode.tensor_cores, Some(false));
        profile
            .apply_env(|name| (name == "SPARKFLASH_DECODE_MMA").then(|| "on".to_owned()))
            .expect("on is a switch value");
        assert_eq!(profile.decode.tensor_cores, Some(true));
    }

    #[test]
    fn a_zero_or_misaligned_chunk_is_refused() {
        let error = ModelProfile::from_toml("[prefill]\nchunk = 100\n").expect_err("misaligned");
        assert!(matches!(error, ProfileError::Invalid(_)));
    }

    /// A value every field of the given kind accepts.
    fn accepted_value(field: &str) -> &'static str {
        if BOOL_FIELDS.iter().any(|(candidate, _)| *candidate == field) {
            "1"
        } else if COUNT_FIELDS
            .iter()
            .any(|(candidate, _)| *candidate == field)
        {
            "64"
        } else if F32_FIELDS.iter().any(|(candidate, _)| *candidate == field)
            || F64_FIELDS.iter().any(|(candidate, _)| *candidate == field)
        {
            "1"
        } else if TEXT_FIELDS.iter().any(|(candidate, _)| *candidate == field) {
            "set"
        } else {
            match field {
                "prefill.activation" | "attention.latent_cache" | "drafter.logits" => "fp8",
                "diagnostics.probe" => "hash",
                "speculation.depth_policy" => "fixed",
                other => panic!("{other} has no accepted value"),
            }
        }
    }

    /// Every alias in the table writes a typed field, and every typed field has an alias.
    #[test]
    fn every_override_settles_on_one_typed_field() {
        let named = |field: &str| {
            BOOL_FIELDS
                .iter()
                .map(|(field, _)| *field)
                .chain(COUNT_FIELDS.iter().map(|(field, _)| *field))
                .chain(F32_FIELDS.iter().map(|(field, _)| *field))
                .chain(F64_FIELDS.iter().map(|(field, _)| *field))
                .chain(TEXT_FIELDS.iter().map(|(field, _)| *field))
                .chain(ENUM_FIELDS.iter().copied())
                .filter(move |candidate| *candidate == field)
                .count()
        };
        for (name, field) in ENV_OVERRIDES {
            assert!(!field.is_empty(), "{name} has no field");
            assert_eq!(named(field), 1, "{name} sets {field} twice or not at all");
        }
        for field in BOOL_FIELDS.iter().map(|(field, _)| *field).chain(
            COUNT_FIELDS
                .iter()
                .map(|(field, _)| *field)
                .chain(F32_FIELDS.iter().map(|(field, _)| *field))
                .chain(F64_FIELDS.iter().map(|(field, _)| *field))
                .chain(TEXT_FIELDS.iter().map(|(field, _)| *field))
                .chain(ENUM_FIELDS.iter().copied()),
        ) {
            assert!(
                ENV_OVERRIDES.iter().any(|(_, declared)| *declared == field),
                "{field} has a setter but no alias"
            );
        }
    }

    /// Every alias moves its field: a table entry the applier never reaches would leave the
    /// profile at its default and the arm reading the default where an operator asked otherwise.
    #[test]
    fn every_override_moves_its_field() {
        for (name, field) in ENV_OVERRIDES {
            let value = accepted_value(field);
            let mut profile = ModelProfile::default();
            profile
                .apply_env(|candidate| (candidate == *name).then(|| value.to_owned()))
                .unwrap_or_else(|error| panic!("{name}={value}: {error}"));
            assert_ne!(
                profile,
                ModelProfile::default(),
                "{name} left {field} at its default"
            );
        }
    }

    /// The one spelling the Qwen arm accepts for its fold switch reaches the typed field.
    ///
    /// `SPARKFLASH_QWEN4_MTP_FOLD=sum` is what the arm documents, and the field is typed as the
    /// scale itself. Until this was handled, the only value the arm acts on was the one the typed
    /// path refused: it parsed as a float and failed with "a positive number".
    #[test]
    fn the_fold_switch_accepts_the_spelling_the_arm_documents() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| (name == "SPARKFLASH_QWEN4_MTP_FOLD").then(|| "sum".to_owned()))
            .expect("sum is the arm's own spelling");
        assert_eq!(profile.speculation.mtp_fold_scale, Some(1.0));
        let mut numeric = ModelProfile::default();
        numeric
            .apply_env(|name| (name == "SPARKFLASH_QWEN4_MTP_FOLD").then(|| "0.5".to_owned()))
            .expect("a scale is a number");
        assert_eq!(numeric.speculation.mtp_fold_scale, Some(0.5));
    }

    /// A variant name outside the set a head-logits kernel accepts is refused rather than
    /// silently taking the tensor-core form.
    #[test]
    fn a_drafter_variant_names_one_of_its_kernels() {
        let mut profile = ModelProfile::default();
        profile
            .apply_env(|name| (name == "SPARKFLASH_DRAFT_LOGITS_MMA").then(|| "fp8".to_owned()))
            .expect("fp8 is a head-logits kernel");
        assert_eq!(profile.drafter.logits, Some(DraftLogitsKernel::Fp8));
        assert!(
            !profile.drafter.head_tiled(),
            "a named kernel turns tiling off"
        );
        assert!(
            profile
                .apply_env(|name| (name == "SPARKFLASH_DRAFT_LOGITS_MMA").then(|| "yes".to_owned()))
                .is_err(),
            "yes is not a head-logits kernel"
        );
    }
}
