// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small device arm: the schedule of [crate::device_plan] executed on a GB10.
//!
//! The plan decides the launches and their order; this module owns the pointers, the state and
//! the argument words. Nothing here chooses a sequence of ops: it walks
//! [InklingDevicePlan::globals] and [InklingDevicePlan::layers] and dispatches each [Op] with its
//! [Target] to the entry the plan named, so a schedule change is a plan change rather than a
//! second walk kept in step by hand.
//!
//! # State
//!
//! A session holds the two key/value caches a layer, the four short-convolution histories a
//! layer, the block-summary stack and one scratch arena. A sliding layer's cache is its window
//! and the attention reads it as a ring, which is what makes the memory figure in the note a
//! computation: 8 GiB at a million tokens instead of 41.
//!
//! # Prefill
//!
//! The decode walk is the only one wired: [ModelSession::prefill] loops it a token at a time.
//! That is correct and slow, and it is what the plan's [crate::device_plan::Pass::Prefill]
//! entries are for - they name the chunked launches that replace the loop next.
//!
//! # Tensor parallelism
//!
//! One rank. The schedule already describes TP4, and the three [crate::device_plan::Collective]
//! steps are the only ones that are no-ops at world one, so opening a rank of a larger world is
//! refused rather than run without its exchanges: a rank that computed its own expert subset and
//! never reduced it would answer with a quarter of each token's MoE.

use std::path::Path;
use std::sync::{Arc, Mutex};

use sparkflash_kernels::cuda_driver::{
    Cuda, DEFAULT_DIRECT_CHUNK_BYTES, DEFAULT_DIRECT_CONCURRENCY, Deviceptr, Kernel, Module,
};
use sparkflash_kernels::images::{
    glm_prefill_cubin, inkling_cubin, moe_plan_cubin, qwen4_exp_cubin, qwen4_prefill_cubin,
};
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_model::vision::{ImageSpan, VisionImage};
use sparkflash_model::{Capabilities, FusedArm, ModelDescriptor, Speculator, TpPlan, lookup_draft};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_weights::WeightSource;
use sparkflash_weights::quant::bf16_to_f32;

use crate::collective::{Exchange, Solo};
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device_plan::{
    BandOp, Collective, ConvSlot, ExpertBody, InklingDevicePlan, NormRole, Op, Pass, Projection,
    ResidualInput, Step, Target,
};
use crate::forward::trace_on;
use crate::host::choose;
use crate::tensor::InklingTensorPlan;
use crate::vision::SpanCheck;
use sparkflash_ops::sampler::Sampler;

mod chunk;
mod draft;
mod tower;

use chunk::{Injection, Rows, exchange_gather, exchange_sum, walk_rows};
pub use draft::DRAFT_FILE;
use draft::{DraftModel, DraftState, FINAL_TAP, TapSink};
pub use tower::InklingVisionTower;

/// The backend name the diagnostics use.
pub const DEVICE: &str = "inkling_device";

/// The threads a block of the token image's dense GEMV or RMSNorm runs.
const TOKEN_BLOCK: u32 = 256;
/// The threads the Inkling elementwise bodies run.
const ELEMENT_BLOCK: u32 = 256;
/// The threads the serial attention body runs: one warp a head.
const ATTENTION_BLOCK: u32 = 32;
/// The threads a block of the split walk's key-scale and score launches runs
/// (32 * INKLING_SCORE_WARPS), and the keys one block covers (INKLING_SCORE_RUN a warp).
const SCORE_BLOCK: u32 = 128;
const SCORE_KEYS: usize = 16;
/// The threads one (head, row) of the split walk's recurrence runs (INKLING_MAX_HEAD_DIM: a dim a
/// thread).
const MIX_BLOCK: u32 = 128;
/// The query heads one KV group may hold for the split walk's scores (INKLING_MAX_GROUPS); a
/// wider group takes the serial body.
const SCORE_GROUPS: usize = 8;
/// The f32 slots of the split walk's running state a (row, head) (INKLING_STATE_WIDTH).
const MIX_STATE: usize = 2 + 128;
/// The split-key decode walk's keys a run and threads a block (INKLING_PART_KEYS and
/// INKLING_PART_THREADS in kernels/inkling_attn_sm121.cu).
const PART_KEYS: usize = 256;
const PART_THREADS: u32 = 128;
/// The scores one decode split-walk pass may hold across its heads: ~175k keys, so a sliding
/// layer is one pass and a full layer at a million positions a handful.
const SCORE_BUDGET: usize = 1 << 21;
/// The scores one chunk split-walk pass may hold across its rows and heads (256 MiB): a 128-row
/// chunk covers ~43k keys a pass, a 4,096-row one 1,344 and an 8,192-row one 672. The pass
/// count does not move the bits (the recurrence carries its state across passes exactly).
const CHUNK_SCORE_BUDGET: usize = 1 << 26;
/// The threads the router runs: the block shares the walk over the experts and keeps the
/// sequential scan's choice (inkling_router_topk).
const ROUTER_BLOCK: u32 = 128;
/// The threads one band-share launch runs: one a slot, past the released top-8.
const SHARES_BLOCK: u32 = 64;
/// The rows one block of a decode band projection computes (INKLING_GROUP_ROWS: a row a
/// half-warp of the 256-thread block).
const BAND_ROWS: usize = 16;
/// The column run one thread of a band projection reads a step (INKLING_GROUP): a band row's
/// width, and an MXFP8 scale block, are whole runs of it.
const BAND_RUN: usize = 16;
/// The column block a released MXFP8 expert stores one ue8m0 code for.
const MXFP8_BLOCK: usize = 32;

/// The switch between a prompt's walks ([PrefillWalk]): `tensor` (the default), `chunk`, `rows`
/// or `gemm`. An unknown value is refused at open rather than read as the default.
pub const PREFILL_ENV: &str = "SPARKFLASH_INKLING_PREFILL";

/// How a prompt's positions are forwarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillWalk {
    /// Chunks whose dense projections run on the tensor cores: the shared wide tile
    /// (`glm_prefill_bf16_wide`, the body GLM's and Dory's prompt projections take), each f32
    /// activation split into a BF16 head and the BF16 of its remainder, accumulated in f32. A row's
    /// sums are in the tile's order and not the decode GEMV's, so a prompt row is not the bits a
    /// decode step would give it; it is the same bits at every chunk width, because an output's
    /// chain depends on K alone and not on which rows share its tile.
    Tensor,
    /// Chunks through the row kernels' own arithmetic: every row is the bits a decode step gives
    /// it, which is what the bit gate reproduces. Its attention is the exact walk unless
    /// [ATTENTION_ENV] names another.
    Chunk,
    /// One position at a time through the decode step.
    Rows,
    /// Chunks whose projections take the reordered tiled GEMM: measured against the row walk
    /// the two traces part company at layer four and the greedy token flips by the first decode
    /// step. It is kept for the comparison the note's table quotes.
    Gemm,
}

impl PrefillWalk {
    /// The walk [PREFILL_ENV] names, `Tensor` when it is unset.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(PREFILL_ENV).as_deref() {
            Err(_) | Ok("" | "tensor") => Ok(Self::Tensor),
            Ok("chunk") => Ok(Self::Chunk),
            Ok("rows") => Ok(Self::Rows),
            Ok("gemm") => Ok(Self::Gemm),
            Ok(other) => Err(format!(
                "{PREFILL_ENV}={other} is not a prefill walk: use tensor (the default), chunk \
                 (the decode step's bits), rows or gemm"
            )),
        }
    }

    /// Whether a prompt is forwarded in chunks rather than a row at a time.
    #[must_use]
    pub const fn chunked(self) -> bool {
        !matches!(self, Self::Rows)
    }
}

/// The switch between attention walks: `fast`, `split`, `serial` or `check`. Unset or empty
/// follows the prompt's walk: `fast` under [PrefillWalk::Tensor], standing down to `split` with
/// one printed line on a shape the fast walks are not built for, and `split` under the others, so
/// [PrefillWalk::Chunk] keeps the decode step's bits. `fast` or `check` named on a shape they are
/// not built for is refused, as is an unknown value.
pub const ATTENTION_ENV: &str = "SPARKFLASH_INKLING_ATTENTION";

/// How attention is walked.
///
/// `split` and `serial` give the same bits - the split walk is the serial body's arithmetic with
/// the recurrence alone left serial, compared bit for bit against it over ring, full, multi-pass,
/// tied and NaN cases - and the serial body is kept as the reference it is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionWalk {
    /// A prompt chunk's rows on the tensor cores (inkling_attn_mma: the shared prefill attention
    /// body Dory's dory_attn_mma runs, with Inkling's clips, bias, sink and gate as its policy),
    /// and a single row's keys split into runs merged in run order (inkling_attn_part and
    /// inkling_attn_merge after the split walk's scores). Neither is the exact walks' bits: the
    /// tensor-core walk's scores and value sum are MMA products, q and the softmax weights
    /// carried as a BF16 head and the BF16 of its remainder, and the split-key walk adds a row's
    /// terms in another order.
    Fast,
    /// The split walk everywhere.
    Split,
    /// The serial body everywhere.
    Serial,
    /// The split walk's output everywhere, with the fast walks run beside it on every prompt
    /// chunk and decode row and each layer's largest difference printed: the per-layer bound of
    /// the fast walks on a real prompt, each layer from exact inputs.
    Check,
}

impl AttentionWalk {
    /// The walk [ATTENTION_ENV] names for a rank that holds `groups` query heads a kv head and
    /// forwards prompts by `prefill`, and the tensor-core walk's dynamic shared memory when it
    /// runs.
    fn select(
        config: &Inkling2Config,
        groups: usize,
        prefill: PrefillWalk,
    ) -> Result<(Self, Option<u32>), String> {
        let value = std::env::var(ATTENTION_ENV).ok();
        Self::choose(value.as_deref(), config, groups, prefill)
    }

    /// [Self::select] given the variable's value, `None` when it is unset.
    fn choose(
        value: Option<&str>,
        config: &Inkling2Config,
        groups: usize,
        prefill: PrefillWalk,
    ) -> Result<(Self, Option<u32>), String> {
        let named = match value {
            None | Some("") => None,
            Some("fast") => Some(Self::Fast),
            Some("split") => Some(Self::Split),
            Some("serial") => Some(Self::Serial),
            Some("check") => Some(Self::Check),
            Some(other) => {
                return Err(format!(
                    "{ATTENTION_ENV}={other} is not an attention walk: use fast (the default: \
                     prompt rows on the tensor cores, a decode row's keys split into runs), split \
                     or serial (the exact walks), or check (split, with each layer's difference \
                     from fast printed)"
                ));
            }
        };
        match named {
            Some(walk @ (Self::Split | Self::Serial)) => Ok((walk, None)),
            Some(walk) => match fast_attention_fits(config, groups) {
                Ok(shared) => Ok((walk, Some(shared))),
                Err(why) => Err(format!(
                    "{ATTENTION_ENV}={}: {why}; set {ATTENTION_ENV}=split for the exact walk",
                    walk.name()
                )),
            },
            // The exact prompt walks keep the exact attention: chunk is the decode step's bits.
            None if prefill != PrefillWalk::Tensor => Ok((Self::Split, None)),
            None => match fast_attention_fits(config, groups) {
                Ok(shared) => Ok((Self::Fast, Some(shared))),
                Err(why) => {
                    eprintln!(
                        "[inkling] attention takes the split walk: {why} \
                         ({ATTENTION_ENV}=split selects it without this line)"
                    );
                    Ok((Self::Split, None))
                }
            },
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Split => "split",
            Self::Serial => "serial",
            Self::Check => "check",
        }
    }

    /// Whether the fast walks run, alone or beside the split walk.
    const fn fast(self) -> bool {
        matches!(self, Self::Fast | Self::Check)
    }

    /// Whether the exact walk is the split one rather than the serial body.
    const fn split(self) -> bool {
        !matches!(self, Self::Serial)
    }
}

/// The tensor-core walk's geometry: query positions a block (SHARED_ATTN_Q in
/// kernels/shared_attn_mma_sm121.cuh), keys a tile (SHARED_ATTN_KEYS), the head width it is built
/// for (INKLING_MAX_HEAD_DIM), the relative state it takes (INKLING_MMA_D_REL), the f32 a row of
/// a warp's bias tile (INKLING_MMA_REL_STRIDE) and the opt-in ceiling on one block's dynamic
/// shared memory on sm_121.
const MMA_Q: usize = 16;
const MMA_KEYS: usize = 32;
const MMA_HEAD_DIM: usize = 128;
const MMA_D_REL: usize = 16;
const MMA_REL_STRIDE: usize = 52;
const MMA_SHARED_LIMIT: usize = 101_376;

/// inkling_attn_mma's dynamic shared memory for `groups` query heads a kv head: q's BF16 head
/// and remainder for the group's rows, a key and a value tile, the tile's key gains and value
/// scales, and a bias tile a warp.
const fn mma_attention_shared(groups: usize) -> usize {
    (2 * groups * MMA_Q + 2 * MMA_KEYS) * MMA_HEAD_DIM * 2
        + 2 * MMA_KEYS * 4
        + groups * MMA_Q * MMA_REL_STRIDE * 4
}

/// The tensor-core walk's dynamic shared memory for this shape, or why the fast walks are not
/// built for it (the split-key walk's own bounds - a head width of whole 8-dim loads at most
/// INKLING_MAX_HEAD_DIM, at most INKLING_MAX_GROUPS heads a kv head - are inside these).
fn fast_attention_fits(config: &Inkling2Config, groups: usize) -> Result<u32, String> {
    if config.head_dim != MMA_HEAD_DIM {
        return Err(format!(
            "the tensor-core walk is built for head_dim {MMA_HEAD_DIM}, and this checkpoint has {}",
            config.head_dim
        ));
    }
    if config.d_rel > MMA_D_REL {
        return Err(format!(
            "the tensor-core walk takes a relative state of at most {MMA_D_REL}, and this \
             checkpoint has {}",
            config.d_rel
        ));
    }
    let shared = mma_attention_shared(groups);
    if groups == 0 || groups > SCORE_GROUPS || shared > MMA_SHARED_LIMIT {
        return Err(format!(
            "the tensor-core walk serves 1..={SCORE_GROUPS} query heads a kv head within \
             {MMA_SHARED_LIMIT} bytes of shared memory, and this rank holds {groups} ({shared} \
             bytes)"
        ));
    }
    Ok(shared as u32)
}

/// The keys a split-walk pass covers for `rows` rows of `heads` heads: a whole number of the
/// recurrence's 32-key tiles. A `split` arena's rows each take the split-key walk (a decode step's
/// one row, a verify's drafted rows), and its passes are the decode step's whatever its rows:
/// whole split-key runs, so a pass's runs are each row's own.
fn attention_pass_slots(rows: usize, heads: usize, split: bool) -> usize {
    let (budget, unit, rows) = if split {
        (SCORE_BUDGET, PART_KEYS, 1)
    } else {
        (CHUNK_SCORE_BUDGET, 32, rows.max(1))
    };
    ((budget / (rows * heads.max(1))) / unit * unit).max(unit)
}

/// The shared wide tile's geometry (kernels/shared_prefill_mma_sm121.cuh): GLM_WIDE_BN output
/// columns by GLM_BM token rows a block of OM_THREADS threads, through GLM_WIDE_SHARED bytes of
/// dynamic shared memory, over a matrix whose columns and inner width are multiples of
/// WIDE_MULTIPLE (a projection that is not takes the row GEMV).
const WIDE_COLUMNS: usize = 256;
const WIDE_ROWS: usize = 128;
const WIDE_THREADS: u32 = 256;
const WIDE_SHARED: u32 = 98_304;
const WIDE_MULTIPLE: usize = 32;

/// The tokens one group of a batched projection launch carries (the row kernel's
/// `Q4E_MAX_BATCH`).
const BATCH_ROWS: usize = 8;
/// The most blocks a launch's grid may have along y (CUDA's limit), which bounds the token
/// groups one batched projection launch carries.
const MAX_GRID_Y: u32 = 65_535;
/// The rows one chunked-prefill pass forwards when [PREFILL_CHUNK_ENV] is unset, and the width a
/// long prompt's chunks widen to: the two chunk sizes the scheduler's specification allows
/// (`sparkflash_scheduler::config::PREFILL_CHUNK_CHOICES`, spec 5.6.1).
///
/// Every kernel of a chunk computes a row's numbers on their own - the projections a row at a
/// time inside a batch or a tile, the experts a (row, slot) pair at a time, attention a row at a
/// time - so the width changes which launches carry a row and not the row's bits. What it
/// changes is how many launches and weight passes a prompt costs.
const DEFAULT_PREFILL_CHUNK: usize = 4096;
const LONG_PREFILL_CHUNK: usize = 8192;
/// The positions a prompt still has to forward at which an unset width takes
/// [LONG_PREFILL_CHUNK] rather than [DEFAULT_PREFILL_CHUNK]: two long chunks' worth, so a prompt
/// that would fill one wide chunk and leave a short tail keeps the narrow ones.
const LONG_PREFILL_FROM: usize = 2 * LONG_PREFILL_CHUNK;
/// The widest chunk [PREFILL_CHUNK_ENV] may ask for: a chunk's rows are a launch's grid rows.
const MAX_PREFILL_CHUNK: usize = 65_535;

/// The rows one chunked-prefill pass forwards. Unset, a chunk is [DEFAULT_PREFILL_CHUNK] rows and
/// widens to [LONG_PREFILL_CHUNK] while [LONG_PREFILL_FROM] or more positions remain; set, every
/// chunk is that width.
pub const PREFILL_CHUNK_ENV: &str = "SPARKFLASH_INKLING_PREFILL_CHUNK";

/// The chunk widths a process walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkWidths {
    /// A chunk's rows while fewer than `long_from` positions remain.
    base: usize,
    /// A chunk's rows while `long_from` or more positions remain; `base` when the width is fixed.
    long: usize,
    long_from: usize,
}

impl ChunkWidths {
    /// The widths [PREFILL_CHUNK_ENV] names, each at most the context; a value that is not a
    /// width in `1..=MAX_PREFILL_CHUNK` is refused rather than clamped.
    fn from_env(context: usize) -> Result<Self, String> {
        let widths = match std::env::var(PREFILL_CHUNK_ENV) {
            Err(_) => Self {
                base: DEFAULT_PREFILL_CHUNK,
                long: LONG_PREFILL_CHUNK,
                long_from: LONG_PREFILL_FROM,
            },
            Ok(raw) => {
                let chunk = raw
                    .parse::<usize>()
                    .ok()
                    .filter(|chunk| (1..=MAX_PREFILL_CHUNK).contains(chunk))
                    .ok_or_else(|| {
                        format!(
                            "{PREFILL_CHUNK_ENV}={raw} is not a chunk width in \
                             1..={MAX_PREFILL_CHUNK}"
                        )
                    })?;
                Self {
                    base: chunk,
                    long: chunk,
                    long_from: usize::MAX,
                }
            }
        };
        // A chunk is never wider than the prompt the context admits, so its buffers need not be.
        let context = context.max(1);
        Ok(Self {
            base: widths.base.min(context),
            long: widths.long.min(context),
            long_from: widths.long_from,
        })
    }

    /// The widest chunk any prompt walks: what the rings and the exchange slots are sized for.
    const fn widest(self) -> usize {
        if self.long > self.base {
            self.long
        } else {
            self.base
        }
    }

    /// The rows the next chunk forwards when `left` positions remain.
    fn next(self, left: usize) -> usize {
        let width = if left >= self.long_from {
            self.long
        } else {
            self.base
        };
        left.min(width)
    }
}

/// The blocks a flat elementwise launch takes: one a 256 elements, at least one and at most the
/// grid limit, so a launch never asks for a zero-sized grid.
fn flat(elements: usize) -> u32 {
    u32::try_from(elements.div_ceil(ELEMENT_BLOCK as usize).max(1))
        .unwrap_or(u32::MAX)
        .min(65_535)
}

/// The blocks that cover `items` at `per_block` a block, at least one.
fn grid_blocks(items: usize, per_block: usize) -> u32 {
    u32::try_from(items.div_ceil(per_block).max(1)).unwrap_or(u32::MAX)
}

/// The blocks a dense GEMV takes: the kernel strides rows over its warps times the grid's blocks.
/// The matrices `q4e_gemv4_bf16` stacks in one launch.
const GEMV_FUSED: usize = 4;

fn gemv_grid(rows: usize) -> u32 {
    let warps = (TOKEN_BLOCK / 32) as usize;
    u32::try_from(rows.div_ceil(warps).max(1))
        .unwrap_or(u32::MAX)
        .min(4_096)
}

/// One f32 argument as the driver's one-word cell.
fn bits(value: f32) -> u64 {
    u64::from(value.to_bits())
}

/// A host vector of f32 as the little-endian bytes the driver copies.
fn row_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The f32 values of a little-endian byte buffer.
fn from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One loaded symbol.
#[derive(Debug, Clone, Copy)]
struct Entry {
    kernel: Kernel,
}

/// Every kernel the walk launches, loaded by the name the plan records.
struct Kernels {
    /// The modules the symbols came from. A dropped module unloads its symbols, so the backend
    /// keeps them for as long as it launches.
    _modules: Vec<Module>,
    embed: Entry,
    rmsnorm: Entry,
    gemv: Entry,
    /// A decode row's run of projections over one input, up to [GEMV_FUSED] in one launch.
    gemv4: Entry,
    block_residual: Entry,
    short_conv: Entry,
    short_conv_dynamic: Entry,
    kv_store: Entry,
    router: Entry,
    expert_nvfp4: Entry,
    expert_mxfp8: Entry,
    /// A chunk's projections of one expert over every pair that chose it, a weight run decoded
    /// once a group of pairs.
    expert_nvfp4_pairs: Entry,
    expert_mxfp8_pairs: Entry,
    /// The decode band's projections, reading the router's choice on the device.
    expert_nvfp4_band: Entry,
    expert_mxfp8_band: Entry,
    /// The decode band's mix shares: the router's weight for an owned slot, zero otherwise.
    band_shares: Entry,
    /// The device leg's sum: this rank's partial and every peer's slot, ascending rank from zero.
    sum_slots: Entry,
    swiglu: Entry,
    expert_mix: Entry,
    residual_add: Entry,
    logit_softcap: Entry,
    widen: Entry,
    /// The reordered tiled GEMM the `gemm` walk's projections take.
    gemm: Entry,
    /// The tensor-core wide tile the `tensor` walk's projections take (GLM's prefill image).
    wide: Entry,
    /// The `tensor` walk's NVFP4 routed experts, Qwen's grouped W4A16 MMA (its prefill image):
    /// the activation split into bf16 head and remainder planes, the gate_up pass with the
    /// interleaved rows and softcapped SwiGLU, and the down pass as Qwen runs it.
    split_bf16: Entry,
    moe_gate_up: Entry,
    moe_down: Entry,
    /// The shared routed-expert plan (kernels/moe_plan_sm121.cu): the chunk's slots sorted by
    /// this rank's expert into the items the MMA passes walk.
    moe_plan: Entry,
    /// The stride-carrying row RMSNorm both passes use at a chunk's width.
    rmsnorm_rows: Entry,
    /// The per-row attention: a chunk's rows against the prefix the step has stored.
    attention_prefill: Entry,
    /// The split walk: each key's clip gain and value scale, the scores of a pass, and the
    /// recurrence over them.
    attention_scales: Entry,
    attention_scores: Entry,
    attention_mix: Entry,
    /// The split-key decode walk: a run of a row's keys a block after the split walk's scores,
    /// then the runs merged in order.
    attention_part: Entry,
    attention_merge: Entry,
    /// The tensor-core prompt walk: each row's query side as the exact walks compute it, then the
    /// shared MMA body over it. `attention_mma` is loaded only when the walk runs
    /// ([AttentionWalk::fast]), sized to the rank's group.
    attention_prepare: Entry,
    attention_mma: Option<Entry>,
    /// The static short convolution over a chunk, reading its taps across the chunk's rows.
    short_conv_rows: Entry,
    /// The dynamic (static taps plus a per-row projection) short convolution over a chunk.
    short_conv_dynamic_rows: Entry,
    /// Append a chunk's rows to their summary stacks.
    summary_push: Entry,
    /// Convert a convolution history between the decode layout and a chunk's row layout.
    conv_history: Entry,
}

impl Kernels {
    /// `mma_shared` is the tensor-core attention's dynamic shared memory, or `None` when the walk
    /// does not run.
    fn load(cuda: &Cuda, mma_shared: Option<u32>) -> Result<Self, String> {
        let inkling = inkling_cubin();
        let token = qwen4_exp_cubin();
        if inkling.is_empty() {
            return Err(
                "this build has no Inkling cubin: rebuild on a host with nvcc (or set \
                        SPARKFLASH_NVCC) so build.rs compiles kernels/inkling_sm121.cu"
                    .to_owned(),
            );
        }
        if token.is_empty() {
            return Err(
                "this build has no token-image cubin: the arm's embedding, RMSNorm and \
                        dense projections are kernels/qwen4_exp_sm121.cu"
                    .to_owned(),
            );
        }
        let prefill = glm_prefill_cubin();
        if prefill.is_empty() {
            return Err(
                "this build has no GLM prefill cubin: the tensor-core prefill's wide \
                        tile is kernels/glm_prefill_sm121.cu's glm_prefill_bf16_wide"
                    .to_owned(),
            );
        }
        let experts = qwen4_prefill_cubin();
        if experts.is_empty() {
            return Err(
                "this build has no Qwen prefill cubin: the tensor-core prefill's routed \
                        experts are kernels/qwen4_prefill_sm121.cu's grouped MMA"
                    .to_owned(),
            );
        }
        let moe_plan = moe_plan_cubin();
        if moe_plan.is_empty() {
            return Err(
                "this build has no routed-expert plan cubin: the tensor-core prefill \
                        sorts its slots with kernels/moe_plan_sm121.cu"
                    .to_owned(),
            );
        }
        let mut modules = Vec::new();
        let (wide_module, wide) =
            cuda.load_named_shared(prefill, "glm_prefill_bf16_wide", WIDE_SHARED)?;
        modules.push(wide_module);
        let attention_mma = match mma_shared {
            Some(bytes) => {
                let (module, kernel) =
                    cuda.load_named_shared(inkling, "inkling_attn_mma", bytes)?;
                modules.push(module);
                Some(Entry { kernel })
            }
            None => None,
        };
        let mut load = |cubin: &'static [u8], name: &'static str| -> Result<Entry, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(Entry { kernel })
        };
        Ok(Self {
            wide: Entry { kernel: wide },
            split_bf16: load(experts, "q4p_split_bf16")?,
            moe_gate_up: load(experts, "q4p_moe_gate_up_capped")?,
            moe_down: load(experts, "q4p_moe_down")?,
            moe_plan: load(moe_plan, "moe_plan")?,
            embed: load(token, "q4e_embed")?,
            rmsnorm: load(token, "q4e_rmsnorm")?,
            gemv: load(token, "q4e_gemv_bf16")?,
            gemv4: load(token, "q4e_gemv4_bf16")?,
            block_residual: load(inkling, "inkling_block_residual")?,
            short_conv: load(inkling, "inkling_short_conv_decode")?,
            short_conv_dynamic: load(inkling, "inkling_short_conv_dynamic_decode")?,
            kv_store: load(inkling, "inkling_kv_store")?,
            router: load(inkling, "inkling_router_topk")?,
            expert_nvfp4: load(inkling, "inkling_nvfp4_expert_matvec")?,
            expert_mxfp8: load(inkling, "inkling_mxfp8_expert_matvec")?,
            expert_nvfp4_pairs: load(inkling, "inkling_nvfp4_expert_pairs")?,
            expert_mxfp8_pairs: load(inkling, "inkling_mxfp8_expert_pairs")?,
            expert_nvfp4_band: load(inkling, "inkling_nvfp4_expert_band")?,
            expert_mxfp8_band: load(inkling, "inkling_mxfp8_expert_band")?,
            band_shares: load(inkling, "inkling_band_shares")?,
            sum_slots: load(token, "q4e_sum_slots")?,
            swiglu: load(inkling, "inkling_softcapped_swiglu")?,
            expert_mix: load(inkling, "inkling_expert_mix")?,
            residual_add: load(inkling, "inkling_residual_add")?,
            logit_softcap: load(inkling, "inkling_logit_softcap")?,
            widen: load(inkling, "inkling_widen_f32")?,
            gemm: load(token, "q4e_gemm_bf16")?,
            rmsnorm_rows: load(token, "q4e_rmsnorm_rows")?,
            attention_prefill: load(inkling, "inkling_attn_prefill")?,
            attention_scales: load(inkling, "inkling_attn_key_scales")?,
            attention_scores: load(inkling, "inkling_attn_scores")?,
            attention_mix: load(inkling, "inkling_attn_mix")?,
            attention_part: load(inkling, "inkling_attn_part")?,
            attention_merge: load(inkling, "inkling_attn_merge")?,
            attention_prepare: load(inkling, "inkling_attn_prepare")?,
            attention_mma,
            short_conv_rows: load(inkling, "inkling_short_conv")?,
            short_conv_dynamic_rows: load(inkling, "inkling_short_conv_dynamic")?,
            summary_push: load(inkling, "inkling_summary_push")?,
            conv_history: load(inkling, "inkling_conv_history_copy")?,
            _modules: modules,
        })
    }
}

// SAFETY: a Kernel is a driver function handle plus its name. The driver serialises context
// access and every launch in this module happens on the process's default stream, so moving the
// handle between threads cannot race a launch; the same argument the Dory and GLM kernels sets
// make for their own handles.
unsafe impl Send for Kernels {}
// SAFETY: as above.
unsafe impl Sync for Kernels {}

/// One served slot on the device.
#[derive(Debug, Clone, Copy)]
struct Slot {
    pointer: Deviceptr,
    bytes: usize,
}

/// The rank's weights, in the served plan's order.
struct Weights {
    slots: Vec<Slot>,
    /// The device allocations the slots point into, which are what is freed: one a slot from a
    /// checkpoint source ([upload]), one for the whole data section of a rank file
    /// ([upload_rank_file]).
    allocations: Vec<Deviceptr>,
    /// [hidden] zeros, for a launch that names a term this arm does not use.
    zeros: Deviceptr,
    total_bytes: usize,
}

impl Weights {
    fn get(&self, index: usize) -> Slot {
        self.slots[index]
    }
}

/// The two widened L2-clip gain vectors a layer's attention reads.
#[derive(Debug, Clone, Copy)]
struct ClipScales {
    query: Deviceptr,
    relative: Deviceptr,
}

/// The per-expert scale a stored format puts on an expert's own output.
///
/// An NVFP4 expert stores amax over its outer-scale denominator, and that factor is
/// linear, so it rides on the expert's output: the arm multiplies it into the mix's share and no
/// kernel needs a per-expert amax. A band without one records nothing.
struct ExpertScale {
    gate_up: Vec<f32>,
    down: Vec<f32>,
    /// The same two vectors on the device, one f32 a local expert, for the decode band to index
    /// by the expert the router chose without the host reading the choice.
    device_gate_up: Deviceptr,
    device_down: Deviceptr,
    /// The layer's experts as the grouped MMA (q4p_moe_gate_up_capped / q4p_moe_down) reads them:
    /// six pointers a local expert - w13's codes and block scales, the same two one row on (the
    /// interleaved up rows), w2's codes and block scales - and three f32 outer scales, w13's for
    /// the gate and the up and w2's for the down.
    mma_table: Deviceptr,
    mma_scales: Deviceptr,
}

/// The pointers a q4p MMA table holds a local expert, and the outer scales.
const MMA_TABLE_ENTRIES: usize = 6;
const MMA_SCALES: usize = 3;

/// Everything one backend shares with its sessions.
///
/// The driver handle is the shared wrapper rather than the raw context: the context holds a
/// process-wide pointer that is neither Send nor Sync, and the backend has to be both because a
/// serving front end opens sessions from another thread.
struct Shared {
    cuda: CudaHandle,
    plan: InklingDevicePlan,
    /// The same schedule at a prefill chunk's width, or `None` when this build has no prefill
    /// kernel: the chunked path decodes instead of walking a schedule it cannot launch.
    prefill: Option<InklingDevicePlan>,
    config: Inkling2Config,
    kernels: Kernels,
    weights: Weights,
    clip: Vec<ClipScales>,
    clip_arena: Deviceptr,
    experts: Vec<Option<ExpertScale>>,
    /// The allocation every ExpertScale's device vectors are offsets into.
    expert_arena: Deviceptr,
    /// [top_k] ones in that arena: the routing weights the grouped MMA's down pass multiplies by.
    route_ones: Deviceptr,
    context: usize,
    /// The peers' slots a device-leg collective lands in: a world of the widest payload, a
    /// chunk's residual rows. `None` when the exchange has no device leg.
    exchange_slots: Option<Deviceptr>,
    exchange: Box<dyn Exchange>,
    /// How attention is walked ([ATTENTION_ENV]).
    attention: AttentionWalk,
    /// The rows prefill chunks forward ([PREFILL_CHUNK_ENV]).
    chunk_widths: ChunkWidths,
    /// The widest of them: what the rings and the exchange slots are sized for.
    prefill_chunk: usize,
    /// How a prompt is forwarded ([PREFILL_ENV]).
    prefill_walk: PrefillWalk,
    /// The drafter a sequence speculates with and the most drafts a round verifies
    /// ([DRAFT_ENV]); `None` steps a token at a time.
    draft: Option<(Drafter, usize)>,
    /// The DSpark drafter's weights, once [InklingDeviceBackend::load_drafter] loaded them.
    dspark: Option<DraftModel>,
    /// The vision tower, on rank 0 when it was loaded ([InklingDeviceBackend::load_vision]): the
    /// one rank that encodes a prompt's images, whose rows the group's sum then broadcasts.
    vision: Option<Mutex<InklingVisionTower>>,
}

impl Shared {
    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    /// What this rank checks of a prompt's image spans; the rank that encodes needs the canvases.
    fn span_check(&self) -> SpanCheck {
        SpanCheck::new(self.config.vision.clone(), self.vision.is_some())
    }

    /// Whether this rank encodes the group's images: rank 0, always, so every rank knows which
    /// one rank's rows the broadcast carries.
    fn encodes(&self) -> bool {
        self.exchange.rank() == 0
    }

    fn kv_width(&self) -> usize {
        self.plan.rank_kv_heads() * self.config.head_dim
    }

    fn heads(&self) -> usize {
        self.plan.rank_heads()
    }

    fn kernel_width(&self) -> usize {
        self.config.sconv_kernel_size
    }

    fn launch(
        &self,
        entry: Entry,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        self.cuda
            .launch(entry.kernel, grid, block, args)
            .map_err(|error| format!("{}: {error}", entry.kernel.name()))
    }

    /// One dense BF16 projection of one row.
    fn gemv(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.gemv,
            (gemv_grid(rows), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[weight.0, input.0, out.0, rows as u64, columns as u64, 1],
        )
    }

    /// Up to four dense BF16 projections of one row that read the same input, as one launch.
    ///
    /// `q4e_gemv4_bf16` walks the matrices' rows stacked, and each row's sum is the one
    /// [Self::gemv] computes - the same row dot and warp sum over the same input - so a fused
    /// group is the bits of its projections launched one at a time. A slot past `parts` has no
    /// rows. `parts` holds (weight, output, rows).
    fn gemv_fused(
        &self,
        parts: &[(Deviceptr, Deviceptr, usize)],
        input: Deviceptr,
        columns: usize,
    ) -> Result<(), String> {
        if parts.is_empty() || parts.len() > GEMV_FUSED {
            return Err(format!(
                "a fused projection takes 1 to {GEMV_FUSED} matrices, not {}",
                parts.len()
            ));
        }
        // (w0..w3, x, y0..y3, r0..r3, columns, batch): an absent slot is a null matrix of no rows.
        let mut args = [0u64; 3 * GEMV_FUSED + 3];
        for (slot, (weight, out, rows)) in parts.iter().enumerate() {
            args[slot] = weight.0;
            args[GEMV_FUSED + 1 + slot] = out.0;
            args[2 * GEMV_FUSED + 1 + slot] = *rows as u64;
        }
        args[GEMV_FUSED] = input.0;
        args[3 * GEMV_FUSED + 1] = columns as u64;
        args[3 * GEMV_FUSED + 2] = 1;
        let total: usize = parts.iter().map(|(_, _, rows)| rows).sum();
        self.launch(
            self.kernels.gemv4,
            (gemv_grid(total), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &args,
        )
    }

    /// One dense BF16 projection of `batch` rows, through the row kernel's own batched walk.
    ///
    /// The entry groups each (weight row, token) sum exactly as its one-token walk groups it -
    /// four chunks added together, then a pair, then one at a time - so a row of a batch is the
    /// bits its own launch would give it. That is the property a chunked prefill needs: the
    /// schedule may widen a launch, but the token it picks has to stay the token.
    ///
    /// `groups` launches that many groups of `batch` tokens at once along the grid's y, each the
    /// group's own walk (consecutive `batch`-token blocks of `input` and `out`).
    #[allow(clippy::too_many_arguments)]
    fn gemv_batch(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
        batch: usize,
        groups: usize,
    ) -> Result<(), String> {
        let groups = u32::try_from(groups)
            .ok()
            .filter(|groups| (1..=MAX_GRID_Y).contains(groups))
            .ok_or_else(|| format!("{groups} token groups do not fit one launch's grid"))?;
        self.launch(
            self.kernels.gemv,
            (gemv_grid(rows), groups, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                weight.0,
                input.0,
                out.0,
                rows as u64,
                columns as u64,
                batch as u64,
            ],
        )
    }

    /// Whether `walk` takes the wide tile for a `width` x `inner` projection: the tensor walk
    /// does when its columns and its inner width are whole 32-column tiles, as the tile reads
    /// them. The walk is the caller's and not this rank's prompt walk, because a step of fused
    /// lanes walks exactly whatever a prompt walks.
    fn wide_fits(walk: PrefillWalk, width: usize, inner: usize) -> bool {
        walk == PrefillWalk::Tensor
            && width.is_multiple_of(WIDE_MULTIPLE)
            && inner.is_multiple_of(WIDE_MULTIPLE)
    }

    /// One dense BF16 projection of `rows` token rows on the wide tile: out[rows][width] =
    /// input[rows][inner] times the row-major weight's transpose.
    fn wide(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        width: usize,
        inner: usize,
        rows: usize,
    ) -> Result<(), String> {
        if !width.is_multiple_of(WIDE_MULTIPLE) || !inner.is_multiple_of(WIDE_MULTIPLE) {
            return Err(format!(
                "the wide tile reads whole {WIDE_MULTIPLE}-column tiles, not a {width} x {inner} \
                 projection"
            ));
        }
        self.launch(
            self.kernels.wide,
            (
                grid_blocks(width, WIDE_COLUMNS),
                grid_blocks(rows, WIDE_ROWS),
                1,
            ),
            (WIDE_THREADS, 1, 1),
            &[
                weight.0,
                input.0,
                out.0,
                width as u64,
                inner as u64,
                rows as u64,
            ],
        )
    }

    /// One RMSNorm of one row.
    fn rmsnorm(
        &self,
        input: Deviceptr,
        weight: Deviceptr,
        out: Deviceptr,
        rows: usize,
        width: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.rmsnorm,
            (rows as u32, 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                input.0,
                weight.0,
                out.0,
                width as u64,
                width as u64,
                bits(self.config.architecture.norm_epsilon as f32),
                0,
                0,
            ],
        )
    }

    /// The block-attention residual over the summary stack plus the current stream.
    ///
    /// `rows` rows each read their own stack of `blocks` summaries, laid out one stride apart; a
    /// single-row walk passes `rows` 1 and a stride of 0.
    #[allow(clippy::too_many_arguments)]
    fn block_residual(
        &self,
        summaries: Deviceptr,
        partial: Deviceptr,
        queries: Deviceptr,
        out: Deviceptr,
        blocks: usize,
        queries_count: usize,
        rows: usize,
        summary_stride: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.block_residual,
            (queries_count as u32, 1, rows as u32),
            (ELEMENT_BLOCK, 1, 1),
            &[
                summaries.0,
                partial.0,
                queries.0,
                out.0,
                blocks as u64,
                self.hidden() as u64,
                queries_count as u64,
                bits(self.config.architecture.norm_epsilon as f32),
                summary_stride as u64,
            ],
        )
    }

    /// A dense BF16 projection over a chunk: the token image's tensor-core tile.
    ///
    /// The tile is 64 output rows by QG_TOKENS (128) tokens, so a chunk of at most that many
    /// rows is one launch and its rows' sums are the sums a launch of their own would give.
    fn gemm(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
        batch: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.gemm,
            (u32::try_from(rows.div_ceil(64)).unwrap_or(1).max(1), 1, 1),
            (128, 1, 1),
            &[
                weight.0,
                input.0,
                out.0,
                rows as u64,
                columns as u64,
                batch as u64,
                0,
                bits(1.0),
                1,
                0,
            ],
        )
    }

    /// One RMSNorm a row over a chunk, with the source and destination strides the chunk needs.
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_rows(
        &self,
        input: Deviceptr,
        weight: Deviceptr,
        out: Deviceptr,
        rows: usize,
        width: usize,
        src_stride: usize,
        dst_stride: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.rmsnorm_rows,
            (1, rows as u32, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                input.0,
                weight.0,
                out.0,
                width as u64,
                width as u64,
                bits(self.config.architecture.norm_epsilon as f32),
                0,
                0,
                src_stride as u64,
                dst_stride as u64,
            ],
        )
    }

    /// The static short convolution over a chunk, reading `lead` rows of history in front of the
    /// first row.
    fn short_conv_rows(
        &self,
        input: Deviceptr,
        taps: Deviceptr,
        out: Deviceptr,
        rows: usize,
        channels: usize,
        lead: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.short_conv_rows,
            (flat(channels), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                input.0,
                taps.0,
                out.0,
                rows as u64,
                channels as u64,
                self.kernel_width() as u64,
                lead as u64,
            ],
        )
    }

    /// The dynamic short convolution over a chunk: the static taps plus a per-row projection.
    #[allow(clippy::too_many_arguments)]
    fn short_conv_dynamic_rows(
        &self,
        input: Deviceptr,
        static_taps: Deviceptr,
        dynamic_taps: Deviceptr,
        out: Deviceptr,
        rows: usize,
        channels: usize,
        lead: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.short_conv_dynamic_rows,
            (flat(channels), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                input.0,
                static_taps.0,
                dynamic_taps.0,
                out.0,
                rows as u64,
                channels as u64,
                self.kernel_width() as u64,
                self.config.head_dim as u64,
                lead as u64,
            ],
        )
    }

    /// Write the [KeyTables] entries of the `rows` keys just stored at first_position on: the
    /// walks' own inkling_attn_key_scales launch over those slots alone.
    fn store_key_tables(
        &self,
        tables: KeyTables,
        keys: Deviceptr,
        values: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
    ) -> Result<(), String> {
        let config = &self.config;
        if first_position + rows > tables.stride {
            return Err(format!(
                "keys {first_position}..{} fall past the {} slots a full layer's tables hold",
                first_position + rows,
                tables.stride
            ));
        }
        let at = first_position * std::mem::size_of::<f32>();
        self.launch(
            self.kernels.attention_scales,
            (grid_blocks(rows, SCORE_KEYS), kv_heads as u32, 1),
            (SCORE_BLOCK, 1, 1),
            &[
                keys.0,
                values.0,
                0,
                tables.gains.offset(at).0,
                tables.scales.offset(at).0,
                kv_heads as u64,
                config.head_dim as u64,
                // A full layer's cache is addressed absolutely.
                0,
                bits(config.qk_l2_clip_threshold as f32),
                1,
                bits(config.architecture.norm_epsilon as f32),
                first_position as u64,
                rows as u64,
                tables.stride as u64,
            ],
        )
    }

    /// The relative attention of `rows` rows (row r at absolute position first_position + r)
    /// against the cache prefix the step stored; `taus` is the per-row scale table or null for
    /// `tau` on every row. `layer` is the schedule's layer, for [AttentionWalk::Check]'s line;
    /// `tables` are a full layer's [KeyTables], which take the rows' entries here whichever walk
    /// runs.
    #[allow(clippy::too_many_arguments)]
    fn attention_rows(
        &self,
        scratch: &Scratch,
        layer: usize,
        queries: Deviceptr,
        gate: Deviceptr,
        keys: Deviceptr,
        values: Deviceptr,
        tables: Option<KeyTables>,
        relative: Deviceptr,
        bank: Deviceptr,
        sink: Deviceptr,
        clip: ClipScales,
        out: Deviceptr,
        taus: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
        groups: usize,
        window: usize,
        cache_rows: usize,
        rel_extent: usize,
        tau: f32,
    ) -> Result<(), String> {
        if let Some(tables) = tables {
            self.store_key_tables(tables, keys, values, rows, first_position, kv_heads)?;
        }
        self.attention_walk(
            scratch,
            layer,
            queries,
            gate,
            keys,
            values,
            tables,
            relative,
            bank,
            sink,
            clip,
            out,
            taus,
            rows,
            first_position,
            kv_heads,
            groups,
            window,
            cache_rows,
            rel_extent,
            tau,
        )
    }

    /// [Self::attention_rows] once the rows' table entries are written: the split walk, or the
    /// serial body under [AttentionWalk::Serial].
    #[allow(clippy::too_many_arguments)]
    fn attention_walk(
        &self,
        scratch: &Scratch,
        layer: usize,
        queries: Deviceptr,
        gate: Deviceptr,
        keys: Deviceptr,
        values: Deviceptr,
        tables: Option<KeyTables>,
        relative: Deviceptr,
        bank: Deviceptr,
        sink: Deviceptr,
        clip: ClipScales,
        out: Deviceptr,
        taus: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
        groups: usize,
        window: usize,
        cache_rows: usize,
        rel_extent: usize,
        tau: f32,
    ) -> Result<(), String> {
        let config = &self.config;
        let heads = self.heads();
        if self.attention.split() && groups <= SCORE_GROUPS && rows <= scratch.attention_rows {
            return self.attention_split(
                scratch,
                layer,
                queries,
                gate,
                keys,
                values,
                tables,
                relative,
                bank,
                sink,
                clip,
                out,
                taus,
                rows,
                first_position,
                kv_heads,
                groups,
                window,
                cache_rows,
                rel_extent,
                tau,
            );
        }
        self.launch(
            self.kernels.attention_prefill,
            (heads as u32, rows as u32, 1),
            (ATTENTION_BLOCK, 1, 1),
            &[
                queries.0,
                gate.0,
                keys.0,
                values.0,
                relative.0,
                bank.0,
                sink.0,
                clip.query.0,
                0,
                clip.relative.0,
                out.0,
                heads as u64,
                kv_heads as u64,
                config.head_dim as u64,
                groups as u64,
                rows as u64,
                first_position as u64,
                window as u64,
                cache_rows as u64,
                rel_extent as u64,
                config.d_rel as u64,
                bits(config.qk_l2_clip_threshold as f32),
                bits(config.rel_l2_clip_threshold as f32),
                bits(tau),
                1,
                bits(config.architecture.norm_epsilon as f32),
                taus.0,
            ],
        )
    }

    /// A prompt chunk's attention ([Self::attention_rows]'s arguments, `layer` the schedule's
    /// layer for [AttentionWalk::Check]'s line): the tensor-core walk under [AttentionWalk::Fast],
    /// the exact walk otherwise, and under [AttentionWalk::Check] the exact walk's output with the
    /// tensor-core walk run beside it and the layer's largest difference printed.
    #[allow(clippy::too_many_arguments)]
    fn attention_chunk(
        &self,
        scratch: &Scratch,
        layer: usize,
        queries: Deviceptr,
        gate: Deviceptr,
        keys: Deviceptr,
        values: Deviceptr,
        tables: Option<KeyTables>,
        relative: Deviceptr,
        bank: Deviceptr,
        sink: Deviceptr,
        clip: ClipScales,
        out: Deviceptr,
        taus: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
        groups: usize,
        window: usize,
        cache_rows: usize,
        rel_extent: usize,
        tau: f32,
    ) -> Result<(), String> {
        if let Some(tables) = tables {
            self.store_key_tables(tables, keys, values, rows, first_position, kv_heads)?;
        }
        let tensor = |into: Deviceptr| -> Result<(), String> {
            let mma = scratch.mma.as_ref().ok_or_else(|| {
                format!("a {rows}-row chunk arena holds no tensor-core attention buffers")
            })?;
            self.attention_mma(
                mma,
                queries,
                gate,
                keys,
                values,
                tables,
                relative,
                bank,
                sink,
                clip,
                into,
                taus,
                rows,
                first_position,
                kv_heads,
                groups,
                window,
                cache_rows,
                rel_extent,
                tau,
            )
        };
        if self.attention == AttentionWalk::Fast {
            return tensor(out);
        }
        self.attention_walk(
            scratch,
            layer,
            queries,
            gate,
            keys,
            values,
            tables,
            relative,
            bank,
            sink,
            clip,
            out,
            taus,
            rows,
            first_position,
            kv_heads,
            groups,
            window,
            cache_rows,
            rel_extent,
            tau,
        )?;
        if self.attention != AttentionWalk::Check {
            return Ok(());
        }
        let check = scratch.mma.and_then(|mma| mma.check).ok_or_else(|| {
            "the check walk's arena holds no output beside the exact one".to_owned()
        })?;
        tensor(check)?;
        self.attention_compare(layer, out, check, rows, first_position)
    }

    /// The tensor-core walk of a chunk's rows: the query side as the exact walks compute it
    /// (inkling_attn_prepare), every visible key's clip gain and value scale (a full layer's
    /// [KeyTables], or the split walk's inkling_attn_key_scales over a sliding layer's whole
    /// range), then the MMA body (inkling_attn_mma).
    #[allow(clippy::too_many_arguments)]
    fn attention_mma(
        &self,
        mma: &MmaScratch,
        queries: Deviceptr,
        gate: Deviceptr,
        keys: Deviceptr,
        values: Deviceptr,
        tables: Option<KeyTables>,
        relative: Deviceptr,
        bank: Deviceptr,
        sink: Deviceptr,
        clip: ClipScales,
        out: Deviceptr,
        taus: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
        groups: usize,
        window: usize,
        cache_rows: usize,
        rel_extent: usize,
        tau: f32,
    ) -> Result<(), String> {
        let config = &self.config;
        let heads = self.heads();
        let entry = self
            .kernels
            .attention_mma
            .ok_or_else(|| "the tensor-core attention kernel was not loaded".to_owned())?;
        let qk_clip = bits(config.qk_l2_clip_threshold as f32);
        self.launch(
            self.kernels.attention_prepare,
            (heads as u32, rows as u32, 1),
            (32, 1, 1),
            &[
                queries.0,
                relative.0,
                clip.query.0,
                clip.relative.0,
                mma.q.0,
                mma.rel.0,
                heads as u64,
                config.head_dim as u64,
                rows as u64,
                config.d_rel as u64,
                qk_clip,
                bits(config.rel_l2_clip_threshold as f32),
                bits(tau),
                taus.0,
            ],
        )?;
        let last = first_position + rows - 1;
        // A full layer's tables already hold every key's entries at its absolute slot.
        let (gains, scales, scale_first, scale_stride) = if let Some(tables) = tables {
            (tables.gains, tables.scales, 0, tables.stride)
        } else {
            let scale_first = if window > 0 {
                (first_position + 1).saturating_sub(window)
            } else {
                0
            };
            let span = last + 1 - scale_first;
            if span > mma.stride {
                return Err(format!(
                    "a chunk at {first_position}..={last} sees {span} keys, past the {} the \
                     tensor-core walk's key tables hold",
                    mma.stride
                ));
            }
            self.launch(
                self.kernels.attention_scales,
                (grid_blocks(span, SCORE_KEYS), kv_heads as u32, 1),
                (SCORE_BLOCK, 1, 1),
                &[
                    keys.0,
                    values.0,
                    0,
                    mma.gains.0,
                    mma.scales.0,
                    kv_heads as u64,
                    config.head_dim as u64,
                    cache_rows as u64,
                    qk_clip,
                    1,
                    bits(config.architecture.norm_epsilon as f32),
                    scale_first as u64,
                    span as u64,
                    mma.stride as u64,
                ],
            )?;
            (mma.gains, mma.scales, scale_first, mma.stride)
        };
        self.launch(
            entry,
            (rows.div_ceil(MMA_Q) as u32, kv_heads as u32, 1),
            ((groups * 32) as u32, 1, 1),
            &[
                mma.q.0,
                keys.0,
                values.0,
                mma.rel.0,
                bank.0,
                sink.0,
                gains.0,
                scales.0,
                gate.0,
                out.0,
                heads as u64,
                kv_heads as u64,
                rows as u64,
                first_position as u64,
                window as u64,
                cache_rows as u64,
                rel_extent as u64,
                config.d_rel as u64,
                scale_first as u64,
                scale_stride as u64,
            ],
        )
    }

    /// [AttentionWalk::Check]'s line: the largest difference between the exact walk's output and
    /// the tensor-core walk's over a chunk's rows of one layer, where it is, and the mean.
    fn attention_compare(
        &self,
        layer: usize,
        exact: Deviceptr,
        fast: Deviceptr,
        rows: usize,
        first_position: usize,
    ) -> Result<(), String> {
        let heads = self.heads();
        let head_dim = self.config.head_dim;
        let count = rows * heads * head_dim;
        self.cuda.synchronize()?;
        let read = |pointer: Deviceptr| -> Result<Vec<f32>, String> {
            let mut bytes = vec![0u8; count * 2];
            self.cuda
                .copy_dtoh(&mut bytes, &pointer)
                .map_err(|error| error.to_string())?;
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
                .collect())
        };
        let exact = read(exact)?;
        let fast = read(fast)?;
        let (mut worst, mut at, mut sum, mut largest, mut unequal_finite) =
            (0f32, 0usize, 0f64, 0f32, 0);
        for (index, (e, t)) in exact.iter().zip(&fast).enumerate() {
            if e.is_finite() != t.is_finite() {
                unequal_finite += 1;
                continue;
            }
            let difference = (e - t).abs();
            if difference > worst {
                worst = difference;
                at = index;
            }
            sum += f64::from(difference);
            largest = largest.max(e.abs());
        }
        let row = at / (heads * head_dim);
        eprintln!(
            "[inkling] attention check rank {} layer {layer} positions {first_position}..{}: max \
             |fast - exact| {worst:.3e} at row {row} head {} dim {} (exact {:.5}, fast {:.5}), \
             mean {:.3e}, max |exact| {largest:.4}, {unequal_finite} finite on one side only",
            self.plan.tp().rank(),
            first_position + rows - 1,
            at / head_dim % heads,
            at % head_dim,
            exact[at],
            fast[at],
            sum / count.max(1) as f64,
        );
        Ok(())
    }

    /// The split walk of [Self::attention_rows]: per pass, the keys' clip gains and value scales
    /// (read from a full layer's [KeyTables] instead), the scores, and the recurrence
    /// (inkling_attn_sm121.cu).
    #[allow(clippy::too_many_arguments)]
    fn attention_split(
        &self,
        scratch: &Scratch,
        layer: usize,
        queries: Deviceptr,
        gate: Deviceptr,
        keys: Deviceptr,
        values: Deviceptr,
        tables: Option<KeyTables>,
        relative: Deviceptr,
        bank: Deviceptr,
        sink: Deviceptr,
        clip: ClipScales,
        out: Deviceptr,
        taus: Deviceptr,
        rows: usize,
        first_position: usize,
        kv_heads: usize,
        groups: usize,
        window: usize,
        cache_rows: usize,
        rel_extent: usize,
        tau: f32,
    ) -> Result<(), String> {
        let config = &self.config;
        let heads = self.heads();
        let pass_slots = scratch.attention_pass;
        let scale_stride = pass_slots + scratch.attention_rows;
        let last = first_position + rows - 1;
        let first_slot = |position: usize| {
            if window > 0 {
                (position + 1).saturating_sub(window)
            } else {
                0
            }
        };
        // The last row sees the most keys; every row's own range is indexed from its first slot.
        let widest = last - first_slot(last) + 1;
        let passes = widest.div_ceil(pass_slots);
        let qk_clip = bits(config.qk_l2_clip_threshold as f32);
        let eps = bits(config.architecture.norm_epsilon as f32);
        // A split arena's rows take the split-key walk when the fast walks run: under Fast it is
        // the rows' output, under Check it runs beside the exact mix into its own buffer. Every
        // number of a row's walk - its scores, its runs of PART_KEYS from its own first slot, each
        // run's partial and the merge in run order - is the row's alone, so a verify's row is the
        // bits its own decode step computes.
        let parts = scratch.parts;
        let exact = parts.is_none() || self.attention == AttentionWalk::Check;
        if parts.is_some() && !pass_slots.is_multiple_of(PART_KEYS) {
            return Err(format!(
                "a split-key pass of {pass_slots} keys is not whole runs of {PART_KEYS}"
            ));
        }
        for pass in 0..passes {
            let pass_first = pass * pass_slots;
            let span = (widest - pass_first).min(pass_slots);
            // A full layer's tables hold every key's entries at its absolute slot; otherwise the
            // pass computes its own slots' into the arena.
            let (key_gains, value_scales, scale_first, scale_stride) = if let Some(tables) = tables
            {
                (tables.gains, tables.scales, 0, tables.stride)
            } else {
                let scale_first = first_slot(first_position) + pass_first;
                let scale_end = (first_slot(last) + pass_first + pass_slots).min(last + 1);
                let scale_span = scale_end - scale_first;
                self.launch(
                    self.kernels.attention_scales,
                    (grid_blocks(scale_span, SCORE_KEYS), kv_heads as u32, 1),
                    (SCORE_BLOCK, 1, 1),
                    &[
                        keys.0,
                        values.0,
                        0,
                        scratch.key_gains.0,
                        scratch.value_scales.0,
                        kv_heads as u64,
                        config.head_dim as u64,
                        cache_rows as u64,
                        qk_clip,
                        1,
                        eps,
                        scale_first as u64,
                        scale_span as u64,
                        scale_stride as u64,
                    ],
                )?;
                (
                    scratch.key_gains,
                    scratch.value_scales,
                    scale_first,
                    scale_stride,
                )
            };
            self.launch(
                self.kernels.attention_scores,
                (grid_blocks(span, SCORE_KEYS), kv_heads as u32, rows as u32),
                (SCORE_BLOCK, 1, 1),
                &[
                    queries.0,
                    keys.0,
                    relative.0,
                    bank.0,
                    clip.query.0,
                    0,
                    clip.relative.0,
                    key_gains.0,
                    scratch.scores.0,
                    heads as u64,
                    kv_heads as u64,
                    config.head_dim as u64,
                    groups as u64,
                    rows as u64,
                    first_position as u64,
                    window as u64,
                    cache_rows as u64,
                    rel_extent as u64,
                    config.d_rel as u64,
                    qk_clip,
                    bits(config.rel_l2_clip_threshold as f32),
                    bits(tau),
                    taus.0,
                    pass_first as u64,
                    pass_slots as u64,
                    scale_first as u64,
                    scale_stride as u64,
                ],
            )?;
            if exact {
                self.launch(
                    self.kernels.attention_mix,
                    (heads as u32, rows as u32, 1),
                    (MIX_BLOCK, 1, 1),
                    &[
                        gate.0,
                        values.0,
                        sink.0,
                        scratch.scores.0,
                        value_scales.0,
                        scratch.attention_state.0,
                        out.0,
                        heads as u64,
                        kv_heads as u64,
                        config.head_dim as u64,
                        groups as u64,
                        rows as u64,
                        first_position as u64,
                        window as u64,
                        cache_rows as u64,
                        1,
                        pass_first as u64,
                        pass_slots as u64,
                        scale_first as u64,
                        scale_stride as u64,
                        u64::from(pass == 0),
                        u64::from(pass + 1 == passes),
                    ],
                )?;
            }
            if let Some(parts) = parts {
                self.launch(
                    self.kernels.attention_part,
                    (
                        kv_heads as u32,
                        span.div_ceil(PART_KEYS) as u32,
                        rows as u32,
                    ),
                    (PART_THREADS, 1, 1),
                    &[
                        values.0,
                        scratch.scores.0,
                        value_scales.0,
                        parts.parts.0,
                        heads as u64,
                        kv_heads as u64,
                        config.head_dim as u64,
                        groups as u64,
                        rows as u64,
                        first_position as u64,
                        window as u64,
                        cache_rows as u64,
                        1,
                        pass_first as u64,
                        pass_slots as u64,
                        scale_first as u64,
                        scale_stride as u64,
                        parts.stride as u64,
                    ],
                )?;
            }
        }
        let Some(parts) = parts else {
            return Ok(());
        };
        let merged = if self.attention == AttentionWalk::Check {
            parts
                .check
                .ok_or_else(|| "a check arena holds no split-key output buffer".to_owned())?
        } else {
            out
        };
        self.launch(
            self.kernels.attention_merge,
            (heads as u32, rows as u32, 1),
            (config.head_dim as u32, 1, 1),
            &[
                gate.0,
                sink.0,
                parts.parts.0,
                merged.0,
                heads as u64,
                config.head_dim as u64,
                rows as u64,
                first_position as u64,
                window as u64,
                parts.stride as u64,
            ],
        )?;
        if self.attention == AttentionWalk::Check {
            self.attention_compare(layer, out, merged, rows, first_position)?;
        }
        Ok(())
    }

    /// The sigmoid router, one block a row.
    fn router_rows(
        &self,
        logits: Deviceptr,
        bias: Deviceptr,
        indices: Deviceptr,
        shares: Deviceptr,
        rows: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.router,
            (rows as u32, 1, 1),
            (ROUTER_BLOCK, 1, 1),
            &[
                logits.0,
                bias.0,
                indices.0,
                shares.0,
                self.config.n_routed_experts as u64,
                self.config.num_experts_per_tok as u64,
                bits(self.config.route_scale as f32),
            ],
        )
    }

    /// Append a chunk's rows to their own summary stacks.
    fn summary_push(
        &self,
        x: Deviceptr,
        summaries: Deviceptr,
        rows: usize,
        blocks: usize,
        block_stride: usize,
    ) -> Result<(), String> {
        self.launch(
            self.kernels.summary_push,
            (flat(self.hidden()), rows as u32, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                x.0,
                summaries.0,
                self.hidden() as u64,
                blocks as u64,
                block_stride as u64,
            ],
        )
    }

    /// Move one convolution history between the two layouts a chunk and a decode step read.
    ///
    /// `to_pad` reads the channel form and writes the row form; zero does the reverse. The two
    /// are the same numbers, so the conversion cannot round.
    fn conv_history(
        &self,
        source: Deviceptr,
        destination: Deviceptr,
        channels: usize,
        to_pad: bool,
    ) -> Result<(), String> {
        let kernel = self.kernel_width();
        let elements = channels * kernel.saturating_sub(1);
        self.launch(
            self.kernels.conv_history,
            (flat(elements), 1, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                source.0,
                destination.0,
                channels as u64,
                kernel as u64,
                u64::from(to_pad),
            ],
        )
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let mut pointers: Vec<Deviceptr> = self.weights.allocations.clone();
        pointers.push(self.weights.zeros);
        pointers.push(self.clip_arena);
        pointers.push(self.expert_arena);
        pointers.extend(self.exchange_slots);
        let refs: Vec<&Deviceptr> = pointers.iter().collect();
        self.cuda.free_buffers(&refs);
        if let Some(mut drafter) = self.dspark.take() {
            drafter.free(&self.cuda);
        }
        if let Some(tower) = self.vision.take() {
            match tower.into_inner() {
                Ok(mut tower) => tower.free(&self.cuda),
                Err(poisoned) => poisoned.into_inner().free(&self.cuda),
            }
        }
    }
}

/// A device backend over one Inkling-2-Small rank.
pub struct InklingDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for InklingDeviceBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingDeviceBackend")
            .field("descriptor", &self.descriptor.id)
            .field("context", &self.shared.context)
            .field("weight_bytes", &self.shared.weights.total_bytes)
            .finish()
    }
}

impl InklingDeviceBackend {
    /// Whether this build carries both cubins the walk needs.
    #[must_use]
    pub fn available() -> bool {
        !inkling_cubin().is_empty() && !qwen4_exp_cubin().is_empty()
    }

    /// Open the arm over a checkpoint source, at rank zero of a world of one.
    ///
    /// # Errors
    /// A missing cubin, a context outside the model window, a checkpoint whose block residual is
    /// not normalised, or a weight the plan names and the source cannot produce.
    pub fn open(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        source: Arc<dyn WeightSource>,
        context: usize,
    ) -> Result<Self, String> {
        let tp = TpPlan::new(1, 0).map_err(|error| error.detail)?;
        Self::open_rank(descriptor, config, source, context, tp, Box::new(Solo))
    }

    /// Open one rank of a larger world, exchanging through the given collective.
    ///
    /// The three collectives the schedule names are the only steps whose behaviour depends on the
    /// world: the embedding row is gathered from its owner, the routed experts' mix is reduced
    /// across the ranks that computed it, and the logit slices are gathered before the softcap.
    ///
    /// # Errors
    /// As [Self::open], and a plan the world does not divide.
    pub fn open_rank(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        source: Arc<dyn WeightSource>,
        context: usize,
        tp: TpPlan,
        exchange: Box<dyn Exchange>,
    ) -> Result<Self, String> {
        Self::open_with(descriptor, config, context, tp, exchange, |cuda, plan| {
            upload(cuda, source.as_ref(), plan)
        })
    }

    /// [Self::open_rank] over a converted rank file, read the way the GLM and DeepSeek ranks
    /// read theirs: the whole data section into one allocation in parallel direct-I/O chunks,
    /// each tensor a span of it. A buffered read a tensor at a time runs near 1 GB/s and leaves
    /// the file in the page cache beside the weights it was copied into.
    ///
    /// # Errors
    /// As [Self::open_rank], and a rank file whose read fails or whose tensors are not each one
    /// span of its data section.
    pub fn open_rank_file(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        file: &RankFileSource,
        context: usize,
        tp: TpPlan,
        exchange: Box<dyn Exchange>,
    ) -> Result<Self, String> {
        // The file decides how the experts are divided; the released configuration never says.
        let mut config = config;
        config.expert_split = file.expert_split();
        Self::open_with(descriptor, config, context, tp, exchange, |cuda, plan| {
            upload_rank_file(cuda, file, plan)
        })
    }

    /// The rank around the weights `weights` places, once the configuration is one the kernels
    /// take: every refusal here runs before a weight is read.
    fn open_with(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        context: usize,
        tp: TpPlan,
        exchange: Box<dyn Exchange>,
        weights: impl FnOnce(&Cuda, &InklingTensorPlan) -> Result<Weights, String>,
    ) -> Result<Self, String> {
        if !Self::available() {
            return Err(
                "this build carries no Inkling or token cubin: rebuild on a host with nvcc \
                        (or set SPARKFLASH_NVCC)"
                    .to_owned(),
            );
        }
        let window = config.context_tokens();
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for {}",
                descriptor.id
            ));
        }
        if !config.attn_res_normalize_values {
            return Err(
                "the released block residual normalises its summary stack; a checkpoint \
                        that clears attn_res_normalize_values has no kernel in this arm"
                    .to_owned(),
            );
        }
        if exchange.world() != tp.world() || exchange.rank() != tp.rank() {
            return Err(format!(
                "the collective is rank {}/{} and the plan rank {}/{}",
                exchange.rank(),
                exchange.world(),
                tp.rank(),
                tp.world()
            ));
        }
        let prefill_walk = PrefillWalk::from_env()?;
        let chunk_widths = ChunkWidths::from_env(context)?;
        let prefill_chunk = chunk_widths.widest();
        let draft = draft_from_env()?;
        // A verify's rows land in the sliding rings, which hold a window and a chunk's rows.
        if let Some((_, depth)) = draft
            && depth + 1 > prefill_chunk
        {
            return Err(format!(
                "{DRAFT_DEPTH_ENV}={depth} verifies {} rows, wider than the {prefill_chunk}-row \
                 chunks the sliding rings are sized for",
                depth + 1
            ));
        }
        // The DSpark block proposes at most its rows less the anchor's.
        if let Some((Drafter::Dspark, depth)) = draft {
            let block = config
                .mtp
                .as_ref()
                .ok_or_else(|| {
                    format!("{DRAFT_ENV}=dspark and this checkpoint describes no drafter")
                })?
                .block_size;
            if depth + 1 > block {
                return Err(format!(
                    "{DRAFT_DEPTH_ENV}={depth}: the DSpark block of {block} rows drafts at most {}",
                    block.saturating_sub(1)
                ));
            }
        }
        // The decode schedule is built for the chunk too: the two address the same sliding
        // rings, and a ring is sized by the widest write into it.
        let plan = InklingDevicePlan::build_for_chunk(
            &config,
            tp,
            Pass::Decode,
            1,
            context,
            prefill_chunk,
        )?;
        // The chunk schedule is the decode one at a chunk's rows: every projection is the same
        // shared GEMM at a wider M, and the two kernels that cannot take a chunk (the decode
        // attention and the four-tap convolution) have their own row forms. The plan is built
        // here rather than at the first long prompt so a shape it refuses is an open-time error
        // and not a request the server has already admitted.
        let prefill = InklingDevicePlan::build(&config, tp, Pass::Prefill, prefill_chunk, context)?;
        let (attention, mma_shared) = AttentionWalk::select(
            &config,
            plan.rank_heads() / plan.rank_kv_heads().max(1),
            prefill_walk,
        )?;
        let cuda = CudaHandle::new(Cuda::load()?);
        let kernels = Kernels::load(&cuda, mma_shared)?;
        let weights = weights(&cuda, plan.served())?;
        let (clip, clip_arena) = widen_clips(&cuda, &kernels, &plan, &weights)?;
        let (experts, expert_arena, route_ones) = expert_scales(&cuda, &plan, &weights)?;
        let exchange_slots = if exchange.has_device_leg() {
            let widest = (prefill_chunk * config.hidden_size).max(plan.rank_vocab());
            let mut total = 0usize;
            Some(cuda.alloc(
                widest * exchange.world() * std::mem::size_of::<f32>(),
                &mut total,
            )?)
        } else {
            None
        };
        // Every launch, device copy and collective of a step queues on the default stream, and
        // the only reads back are stream-ordered copies, so nothing needs the driver's
        // per-launch drain; SPARKFLASH_SYNC_LAUNCHES=1 brings it back. The Qwen and Dory arms
        // make the same switch at their open.
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        if trace_on() {
            eprintln!("[inkling] device arm: {}", plan.boot_summary());
        }
        // Every rank meets on the host leg once its weights are resident. The loads differ by
        // minutes across the four ranks, and without this the first device collective is where
        // the ranks that finished early wait - inside the direct-RDMA exchange, whose round gives
        // up after SPARK_COMM_TIMEOUT_MS (seconds), rather than on the host transport's frame
        // deadline, which is sized for the load.
        if exchange.world() > 1 {
            let mut word = [0f32];
            exchange
                .all_reduce_f32(&mut word)
                .map_err(|error| format!("the meeting after the weight load: {error}"))?;
        }
        Ok(Self {
            descriptor,
            shared: Arc::new(Shared {
                cuda,
                plan,
                prefill: Some(prefill),
                config,
                kernels,
                weights,
                clip,
                clip_arena,
                experts,
                expert_arena,
                route_ones,
                context,
                exchange_slots,
                exchange,
                attention,
                chunk_widths,
                prefill_chunk,
                prefill_walk,
                draft,
                dspark: None,
                vision: None,
            }),
        })
    }

    /// Loads the vision tower onto this rank from `source` (the checkpoint, which holds the
    /// tower's tensors; the converted rank files hold only the text model's). Rank 0 only, and
    /// before any session or lanes open, since they share this rank's state.
    ///
    /// Returns the device bytes the tower's weights take.
    ///
    /// # Errors
    /// A rank other than 0, a session already open, a checkpoint with no tower or one whose output
    /// width is not the text model's, or the tower's own load ([InklingVisionTower::load]).
    pub fn load_vision(&mut self, source: &dyn WeightSource) -> Result<usize, String> {
        let shared = Arc::get_mut(&mut self.shared)
            .ok_or("the vision tower loads before any session or lanes open on this rank")?;
        if !shared.encodes() {
            return Err(format!(
                "rank {} loads no tower: rank 0 encodes the group's images",
                shared.exchange.rank()
            ));
        }
        let config = shared
            .config
            .vision
            .as_ref()
            .ok_or("this checkpoint's config describes no vision tower")?;
        if config.out_hidden_size != shared.config.hidden_size {
            return Err(format!(
                "the tower answers {}-wide rows and the text model is {} wide",
                config.out_hidden_size, shared.config.hidden_size
            ));
        }
        shared.cuda.make_current()?;
        let tower = InklingVisionTower::load(&shared.cuda, config, source)?;
        let bytes = tower.device_bytes();
        shared.vision = Some(Mutex::new(tower));
        Ok(bytes)
    }

    /// Whether this rank drafts with the checkpoint's DSpark drafter ([DRAFT_ENV]=dspark), which
    /// [Self::load_drafter] must load before a session opens.
    #[must_use]
    pub fn wants_drafter(&self) -> bool {
        matches!(self.shared.draft, Some((Drafter::Dspark, _)))
    }

    /// Loads the DSpark drafter onto this rank from `directory`'s [DRAFT_FILE] (the released
    /// checkpoint's: the rank files hold only the text model). Every rank of a group loads its
    /// share, and before any session or lanes open.
    ///
    /// Returns the device bytes the drafter's weights take.
    ///
    /// # Errors
    /// A rank that was not asked to draft with it, a session already open, or the drafter's own
    /// load ([DraftModel::load]).
    pub fn load_drafter(&mut self, directory: &Path) -> Result<usize, String> {
        if !self.wants_drafter() {
            return Err(format!(
                "the DSpark drafter loads when {DRAFT_ENV}=dspark; this rank drafts with {:?}",
                self.shared.draft.map(|(drafter, _)| drafter)
            ));
        }
        let shared = Arc::get_mut(&mut self.shared)
            .ok_or("the drafter loads before any session or lanes open on this rank")?;
        shared.cuda.make_current()?;
        let model = DraftModel::load(shared, directory)?;
        let bytes = model.device_bytes();
        shared.dspark = Some(model);
        Ok(bytes)
    }

    /// What this rank checks of a prompt's image spans before it prefills them; a group's rank 0
    /// runs it before it broadcasts a request, so a span no rank would take is refused before any
    /// rank starts.
    #[must_use]
    pub fn span_check(&self) -> SpanCheck {
        self.shared.span_check()
    }

    /// The schedule this arm executes.
    #[must_use]
    pub fn plan(&self) -> &InklingDevicePlan {
        &self.shared.plan
    }

    /// The bytes the weights occupy on the device.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.weights.total_bytes
    }
}

impl ModelBackend for InklingDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> Capabilities {
        // Images run when this rank holds the tower ([Self::load_vision]). Speculation is the
        // session's own ([DRAFT_ENV]): a round verifies its drafts on the device and feeds the
        // caller one token at a time, so the engine sees plain steps.
        Capabilities {
            vision: self.shared.vision.is_some(),
            ..self.descriptor.capabilities
        }
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        // A CUDA context is per-thread: [Cuda::load] makes the primary context current on the
        // thread that opened the backend, and the generic host driver opens a sequence on its own
        // worker thread. Without this bind the first allocation answers CUDA_ERROR_INVALID_CONTEXT
        // (201) - which is exactly how the four-rank server failed its first request.
        Ok(Box::new(InklingDeviceSession::open(&self.shared)?))
    }
}

impl InklingDeviceBackend {
    /// This rank's sessions as fused lanes: a slot a request, and each step one walk over every
    /// waiting lane's row ([FusedArm]), at most `max_batch` rows a step.
    ///
    /// # Errors
    /// A batch of no rows.
    pub fn lanes(&self, max_batch: usize) -> Result<InklingLanes, BackendError> {
        if max_batch == 0 {
            return Err(BackendError::new(
                DEVICE,
                "fused lanes need at least one row a step",
            ));
        }
        // A lane's round leaves its state past the drafts until the lane settles, and a fused
        // step walks every lane's state as it stands: drafting serves one request at a time.
        if max_batch > 1
            && let Some((drafter, _)) = self.shared.draft
        {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "{DRAFT_ENV} drafts ({drafter:?}) one request at a time and fused lanes step \
                     {max_batch}: serve one lane, or set {DRAFT_ENV}=off"
                ),
            ));
        }
        Ok(InklingLanes {
            shared: Arc::clone(&self.shared),
            buffer: Mutex::new(None),
            max_batch,
        })
    }
}

/// One rank's sessions stepped together: the [FusedArm] the shared engine drives, and what every
/// rank of a fused group mirrors.
///
/// A lane is a whole session - its cache, its convolution histories, its sampler - and a step of
/// several lanes is one walk of the chunk schedule ([walk_rows] over [Rows::Lanes]): the norms,
/// the projections, the router, the routed band and the collectives read the weights once for
/// every lane's row, and the cache store, the attention and the four convolutions run each row on
/// its own lane's state through the decode step's launches. The walk is the exact one
/// ([PrefillWalk::Chunk]) whatever this rank's prompt walk is, so a lane's token is the token it
/// would decode alone, and a step of one lane is the decode step itself.
///
/// The lanes share one row buffer. A step and a prompt both run on the engine's thread, one at a
/// time, so a prompt borrows the buffer for its chunks rather than each lane holding a prompt
/// buffer of its own: at 8,192 rows that is 6.4 GB a lane.
pub struct InklingLanes {
    shared: Arc<Shared>,
    buffer: Mutex<Option<Chunk>>,
    max_batch: usize,
}

/// One request's state in [InklingLanes]: a session the engine owns.
pub struct InklingLane {
    session: InklingDeviceSession,
}

impl std::fmt::Debug for InklingLanes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingLanes")
            .field("max_batch", &self.max_batch)
            .finish_non_exhaustive()
    }
}

impl Drop for InklingLanes {
    fn drop(&mut self) {
        let held = self.buffer.get_mut().ok().and_then(Option::take);
        if let Some(chunk) = held {
            let pointers = chunk.buffers();
            self.shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
        }
    }
}

impl InklingLanes {
    /// Binds the arm's context on the calling thread: the engine steps on a thread of its own.
    fn bind(&self) -> Result<(), BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))
    }

    fn buffer(&self) -> Result<std::sync::MutexGuard<'_, Option<Chunk>>, BackendError> {
        self.buffer
            .lock()
            .map_err(|_| BackendError::new(DEVICE, "the lanes' row buffer lock is poisoned"))
    }
}

impl FusedArm for InklingLanes {
    type Slot = InklingLane;

    fn open_slot(&self) -> Result<InklingLane, BackendError> {
        Ok(InklingLane {
            session: InklingDeviceSession::open(&self.shared)?,
        })
    }

    fn holds_request(&self, slot: &InklingLane) -> bool {
        !slot.session.history.is_empty()
    }

    fn renew(&self, slot: &mut InklingLane) -> Result<(), BackendError> {
        self.bind()?;
        slot.session.forget()
    }

    fn set_sampling(&self, slot: &mut InklingLane, sampling: Sampling) {
        slot.session.set_sampling(sampling);
    }

    fn prefill(&self, slot: &mut InklingLane, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(slot, tokens, &[])
    }

    fn prefill_images(
        &self,
        slot: &mut InklingLane,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        self.bind()?;
        let mut buffer = self.buffer()?;
        slot.session.chunk = buffer.take();
        let answer = slot.session.prompt(tokens, images);
        *buffer = slot.session.chunk.take();
        answer
    }

    fn step(&self, members: &mut [(InklingLane, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        self.bind()?;
        if let [(lane, token)] = members {
            return Ok(vec![lane.session.decode(*token)?]);
        }
        let shared = &self.shared;
        for (lane, token) in members.iter() {
            if lane.session.state.position >= shared.context {
                return Err(BackendError::new(
                    "context",
                    "a lane's context is exhausted",
                ));
            }
            if *token as usize >= shared.config.vocab_size {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "token {token} is outside the vocabulary {}",
                        shared.config.vocab_size
                    ),
                ));
            }
        }
        let tokens: Vec<u32> = members.iter().map(|(_, token)| *token).collect();
        let rows = {
            let mut buffer = self.buffer()?;
            open_buffer(
                shared,
                &mut buffer,
                tokens.len(),
                tokens.len().next_power_of_two(),
            )?;
            let chunk = buffer.as_mut().expect("the lanes' buffer was just opened");
            let mut states: Vec<&mut State> = members
                .iter_mut()
                .map(|(lane, _)| &mut lane.session.state)
                .collect();
            walk_rows(
                shared,
                Rows::Lanes(&mut states),
                chunk,
                &tokens,
                PrefillWalk::Chunk,
                &[],
                None,
            )?
        };
        Ok(members
            .iter_mut()
            .zip(rows)
            .map(|((lane, token), logits)| lane.session.decoded(*token, logits))
            .collect())
    }

    fn speculation(&self, _slot: &InklingLane) -> Speculation {
        Speculation::default()
    }

    fn max_batch(&self) -> usize {
        self.max_batch
    }
}

/// Upload every served slot, refusing a source whose bytes disagree with the plan.
fn upload(
    cuda: &Cuda,
    source: &dyn WeightSource,
    plan: &InklingTensorPlan,
) -> Result<Weights, String> {
    let specs = plan.plan().specs();
    let mut slots = Vec::with_capacity(specs.len());
    let mut total = 0usize;
    for index in 0..specs.len() {
        let spec = plan.rank_spec_at(index);
        let bytes = usize::try_from(spec.storage_bytes())
            .map_err(|_| format!("{}: {} bytes past usize", spec.name, spec.storage_bytes()))?;
        let data = source
            .read(&spec.name)
            .map_err(|error| format!("{}: {error}", spec.name))?;
        if data.len() != bytes {
            return Err(format!(
                "{}: the source holds {} bytes and the rank plan {bytes}",
                spec.name,
                data.len()
            ));
        }
        let pointer = cuda.alloc(bytes.max(1), &mut total)?;
        cuda.copy_htod(&pointer, &data)?;
        slots.push(Slot { pointer, bytes });
    }
    let allocations = slots.iter().map(|slot| slot.pointer).collect();
    let zeros = zeros(cuda, plan, &mut total)?;
    Ok(Weights {
        slots,
        allocations,
        zeros,
        total_bytes: total,
    })
}

/// Upload a converted rank file's whole data section into one allocation with the shared
/// parallel direct-I/O read ([Cuda::upload_file_parallel], the GLM rank's `Allocations::file`),
/// and point each served tensor at its span of it.
///
/// Every span is checked against the plan before the allocation, so a file that disagrees with
/// the plan costs no device memory. Each tensor starts on a 256-byte slot boundary
/// ([sparkflash_weights::rank_file::SLOT_ALIGN_BYTES]), the alignment the GLM slab serves its
/// kernels at; the data section itself is page-aligned and page-padded, which is what the direct
/// read needs.
fn upload_rank_file(
    cuda: &Cuda,
    file: &RankFileSource,
    plan: &InklingTensorPlan,
) -> Result<Weights, String> {
    let (path, data_offset, data_bytes) = file.data_section();
    let specs = plan.plan().specs();
    let mut spans = Vec::with_capacity(specs.len());
    for index in 0..specs.len() {
        let spec = plan.rank_spec_at(index);
        let bytes = usize::try_from(spec.storage_bytes())
            .map_err(|_| format!("{}: {} bytes past usize", spec.name, spec.storage_bytes()))?;
        let (start, held) = file.tensor_span(&spec.name)?;
        if held != bytes {
            return Err(format!(
                "{}: the rank file holds {held} bytes and the rank plan {bytes}",
                spec.name
            ));
        }
        spans.push((start, bytes));
    }
    let mut total = 0usize;
    let arena = cuda.alloc(data_bytes.max(1), &mut total)?;
    if let Err(error) = cuda.upload_file_parallel(
        &arena,
        path,
        data_offset,
        data_bytes,
        DEFAULT_DIRECT_CHUNK_BYTES,
        DEFAULT_DIRECT_CONCURRENCY,
        true,
    ) {
        cuda.free_buffers(&[&arena]);
        return Err(format!("{}: {error}", path.display()));
    }
    let slots = spans
        .into_iter()
        .map(|(start, bytes)| Slot {
            pointer: arena.offset(start),
            bytes,
        })
        .collect();
    let zeros = zeros(cuda, plan, &mut total)?;
    Ok(Weights {
        slots,
        allocations: vec![arena],
        zeros,
        total_bytes: total,
    })
}

/// [hidden] f32 zeros, for a launch that names a term this arm does not use.
fn zeros(cuda: &Cuda, plan: &InklingTensorPlan, total: &mut usize) -> Result<Deviceptr, String> {
    let hidden = plan
        .index_of("model.llm.norm.weight")
        .map_or(4096, |index| plan.rank_spec_at(index).shape[0]);
    let bytes = hidden.max(1) * std::mem::size_of::<f32>();
    let zeros = cuda.alloc(bytes, total)?;
    cuda.memset_d8_async(zeros, 0, bytes)?;
    cuda.synchronize()?;
    Ok(zeros)
}

/// Widen every layer's two L2-clip gain vectors from the checkpoint's BF16 to the f32 the
/// attention body reads.
///
/// The released q_l2_clip.scale is [heads][head_dim] and rel_l2_clip.scale is [heads][d_rel],
/// both BF16; the kernel's arguments are float pointers because the gains multiply a vector that
/// is already f32. The widening is exact, so this is the checkpoint's own value and not a
/// re-quantisation.
fn widen_clips(
    cuda: &Cuda,
    kernels: &Kernels,
    plan: &InklingDevicePlan,
    weights: &Weights,
) -> Result<(Vec<ClipScales>, Deviceptr), String> {
    let heads = plan.rank_heads();
    let head_dim = plan.config().head_dim;
    let d_rel = plan.config().d_rel;
    let per_layer = heads * head_dim + heads * d_rel;
    let mut total = 0usize;
    let bytes = plan.layers().len().max(1) * per_layer * std::mem::size_of::<f32>();
    let arena = cuda.alloc(bytes, &mut total)?;
    cuda.memset_d8_async(arena, 0, bytes)?;
    cuda.synchronize()?;
    let mut clip = Vec::with_capacity(plan.layers().len());
    for layer in plan.layers() {
        let prefix = format!("model.llm.layers.{}.attn", layer.index);
        let query_name = format!("{prefix}.q_l2_clip.scale");
        let relative_name = format!("{prefix}.rel_l2_clip.scale");
        let query_index = plan
            .served()
            .index_of(&query_name)
            .ok_or_else(|| format!("{query_name} is not a served tensor"))?;
        let relative_index = plan
            .served()
            .index_of(&relative_name)
            .ok_or_else(|| format!("{relative_name} is not a served tensor"))?;
        let query = arena.offset(layer.index * per_layer * 4);
        let relative = arena.offset(layer.index * per_layer * 4 + heads * head_dim * 4);
        for (source, destination, count) in [
            (query_index, query, heads * head_dim),
            (relative_index, relative, heads * d_rel),
        ] {
            cuda.launch(
                kernels.widen.kernel,
                (flat(count), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[weights.get(source).pointer.0, destination.0, count as u64],
            )
            .map_err(|error| format!("widen {source}: {error}"))?;
        }
        clip.push(ClipScales { query, relative });
    }
    cuda.synchronize()?;
    Ok((clip, arena))
}

/// Read the released per-expert scales off the device once, so a token's band does not have to,
/// and lay out each NVFP4 layer's grouped-MMA table. Returns the scales, the arena they live in,
/// and the [top_k] ones the MMA's down pass reads as routing weights.
fn expert_scales(
    cuda: &Cuda,
    plan: &InklingDevicePlan,
    weights: &Weights,
) -> Result<(Vec<Option<ExpertScale>>, Deviceptr, Deviceptr), String> {
    let mut table = Vec::with_capacity(plan.layers().len());
    for layer in plan.layers() {
        if layer.experts != ExpertBody::NvFp4 {
            table.push(None);
            continue;
        }
        let prefix = format!("model.llm.layers.{}.mlp.experts", layer.index);
        let read = |suffix: &str| -> Result<Vec<f32>, String> {
            let name = format!("{prefix}.{suffix}");
            let index = plan
                .served()
                .index_of(&name)
                .ok_or_else(|| format!("{name} is not a served tensor"))?;
            let slot = weights.get(index);
            let mut bytes = vec![0u8; slot.bytes];
            cuda.copy_dtoh(&mut bytes, &slot.pointer)
                .map_err(|error| format!("{name}: {error}"))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
        };
        // ModelOpt's NVFP4 outer scale is amax over the outer-scale denominator, which already
        // carries the six times the E4M3 maximum: the same convention the host forward's
        // nvfp4_outer_scale records.
        let scale = |amax: Vec<f32>, denominator: Vec<f32>| -> Vec<f32> {
            let denominator = denominator
                .first()
                .copied()
                .filter(|value| *value != 0.0)
                .unwrap_or(1.0);
            amax.into_iter().map(|value| value / denominator).collect()
        };
        let gate_up = scale(
            read("w13_weight_amax")?,
            read("w13_weight_outer_scale_denominator")?,
        );
        let down = scale(
            read("w2_weight_amax")?,
            read("w2_weight_outer_scale_denominator")?,
        );
        table.push(Some((gate_up, down)));
    }
    // One arena for every layer's vectors and MMA tables, uploaded once. Each piece starts on an
    // 8-byte boundary, which the tables' pointers need.
    let words = |count: usize| count.next_multiple_of(2);
    let floats: usize = table
        .iter()
        .flatten()
        .map(|(gate_up, down): &(Vec<f32>, Vec<f32>)| {
            let owned = gate_up.len();
            words(gate_up.len())
                + words(down.len())
                + words(owned * MMA_TABLE_ENTRIES * 2)
                + words(owned * MMA_SCALES)
        })
        .sum::<usize>()
        + words(plan.config().num_experts_per_tok);
    let mut total = 0usize;
    let arena = cuda.alloc((floats * 4).max(4), &mut total)?;
    let mut cursor = 0usize;
    let mut place = |bytes: &[u8]| -> Result<Deviceptr, String> {
        let pointer = arena.offset(cursor * 4);
        if !bytes.is_empty() {
            cuda.copy_htod(&pointer, bytes)?;
        }
        cursor += words(bytes.len() / 4);
        Ok(pointer)
    };
    // The down pass multiplies its sum by a slot's routing weight; the mix already applies the
    // shares, so the MMA reads one for every slot.
    let route_ones = place(&row_bytes(&vec![1.0f32; plan.config().num_experts_per_tok]))?;
    let config = plan.config();
    let (hidden, intermediate) = (config.hidden_size, plan.expert_intermediate());
    let mut scales = Vec::with_capacity(table.len());
    for (layer, entry) in plan.layers().iter().zip(table) {
        scales.push(match entry {
            None => None,
            Some((gate_up, down)) => {
                let device_gate_up = place(&row_bytes(&gate_up))?;
                let device_down = place(&row_bytes(&down))?;
                let codes = |op: Op, which: usize| -> Result<Deviceptr, String> {
                    layer
                        .steps
                        .iter()
                        .find(|step| step.op == op)
                        .and_then(|step| step.weights.get(which))
                        .map(|index| weights.get(*index).pointer)
                        .ok_or_else(|| {
                            format!(
                                "layer {}: no {op:?} weight {which} for the MMA table",
                                layer.index
                            )
                        })
                };
                let (w13, w13_scales) = (codes(Op::ExpertGateUp, 0)?, codes(Op::ExpertGateUp, 1)?);
                let (w2, w2_scales) = (codes(Op::ExpertDown, 0)?, codes(Op::ExpertDown, 1)?);
                let w13_bytes = codes_bytes(2 * intermediate, hidden, ExpertBody::NvFp4);
                let w13_scale_bytes = scales_bytes(2 * intermediate, hidden, ExpertBody::NvFp4);
                let w2_bytes = codes_bytes(hidden, intermediate, ExpertBody::NvFp4);
                let w2_scale_bytes = scales_bytes(hidden, intermediate, ExpertBody::NvFp4);
                // One w13 row: hidden / 2 code bytes and hidden / 16 block scales.
                let (code_row, scale_row) = (hidden / 2, hidden / 16);
                let mut pointers = Vec::with_capacity(gate_up.len() * MMA_TABLE_ENTRIES);
                let mut outer = Vec::with_capacity(gate_up.len() * MMA_SCALES);
                for local in 0..gate_up.len() {
                    let codes = w13.offset(local * w13_bytes);
                    let block = w13_scales.offset(local * w13_scale_bytes);
                    pointers.extend([
                        codes.0,
                        block.0,
                        codes.offset(code_row).0,
                        block.offset(scale_row).0,
                        w2.offset(local * w2_bytes).0,
                        w2_scales.offset(local * w2_scale_bytes).0,
                    ]);
                    outer.extend([gate_up[local], gate_up[local], down[local]]);
                }
                // The tile copies 16 bytes of a row at a time and four bytes of its scales.
                if pointers
                    .chunks(MMA_TABLE_ENTRIES)
                    .any(|row| row[0] % 16 != 0 || row[2] % 16 != 0 || row[4] % 16 != 0)
                    || pointers.iter().any(|pointer| pointer % 4 != 0)
                {
                    return Err(format!(
                        "layer {}: an expert's codes are not 16-byte aligned for the MMA tile",
                        layer.index
                    ));
                }
                let pointer_bytes: Vec<u8> = pointers
                    .iter()
                    .flat_map(|pointer| pointer.to_le_bytes())
                    .collect();
                let mma_table = place(&pointer_bytes)?;
                let mma_scales = place(&row_bytes(&outer))?;
                Some(ExpertScale {
                    gate_up,
                    down,
                    device_gate_up,
                    device_down,
                    mma_table,
                    mma_scales,
                })
            }
        });
    }
    Ok((scales, arena, route_ones))
}

/// The bytes one expert's stored codes occupy.
fn codes_bytes(rows: usize, columns: usize, body: ExpertBody) -> usize {
    match body {
        ExpertBody::Bf16 => rows * columns * 2,
        ExpertBody::NvFp4 => rows * columns.div_ceil(2),
        ExpertBody::MxFp8 => rows * columns,
    }
}

/// The bytes one expert's block scales occupy.
fn scales_bytes(rows: usize, columns: usize, body: ExpertBody) -> usize {
    match body {
        ExpertBody::Bf16 => 0,
        ExpertBody::NvFp4 => rows * columns.div_ceil(16),
        ExpertBody::MxFp8 => rows * columns.div_ceil(MXFP8_BLOCK),
    }
}

/// A full layer's per-key attention tables (kernels/inkling_attn_sm121.cu): every stored key's
/// clip gain and its value row's RMS scale, `[kv_heads][stride]` f32 at the key's absolute slot.
///
/// Both depend on the key and value rows alone, so they are computed once, when the rows are
/// stored ([Shared::store_key_tables]), by the same inkling_attn_key_scales launch the walks
/// otherwise run over every visible key on every step: the same bits, without reading a long
/// context's whole cache a second time each step. A sliding layer's ring sees at most its window
/// plus a chunk and keeps the per-step tables.
#[derive(Debug, Clone, Copy)]
struct KeyTables {
    gains: Deviceptr,
    scales: Deviceptr,
    stride: usize,
}

/// One session's device state.
struct State {
    keys: Vec<Deviceptr>,
    values: Vec<Deviceptr>,
    /// Each full layer's [KeyTables] when a split or fast walk reads them; `None` on a sliding
    /// layer and under [AttentionWalk::Serial].
    tables: Vec<Option<KeyTables>>,
    conv: Vec<[Deviceptr; 4]>,
    summaries: Deviceptr,
    summaries_bytes: usize,
    summary_rows: usize,
    blocks: usize,
    scratch: Scratch,
    position: usize,
}

impl State {
    fn allocate(shared: &Shared) -> Result<Self, String> {
        let plan = &shared.plan;
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let mut total = 0usize;
        let kv_bytes = |rows: usize, width: usize| rows.max(1) * width * 2;
        let mut keys = Vec::with_capacity(plan.layers().len());
        let mut values = Vec::with_capacity(plan.layers().len());
        let mut tables = Vec::with_capacity(plan.layers().len());
        for layer in plan.layers() {
            let bytes = kv_bytes(layer.kv_rows, shared.kv_width());
            let key = cuda.alloc(bytes, &mut total)?;
            let value = cuda.alloc(bytes, &mut total)?;
            cuda.memset_d8_async(key, 0, bytes)?;
            cuda.memset_d8_async(value, 0, bytes)?;
            keys.push(key);
            values.push(value);
            // A slot's entries are written when its key is stored and read only after, so the
            // tables need no clearing.
            tables.push(
                (!layer.sliding && shared.attention.split())
                    .then(|| -> Result<KeyTables, String> {
                        let stride = layer.kv_rows.max(1);
                        let bytes = plan.rank_kv_heads().max(1) * stride * 4;
                        Ok(KeyTables {
                            gains: cuda.alloc(bytes, &mut total)?,
                            scales: cuda.alloc(bytes, &mut total)?,
                            stride,
                        })
                    })
                    .transpose()?,
            );
        }
        let history = shared.config.conv_history().max(1);
        let mut conv = Vec::with_capacity(plan.layers().len());
        for _ in 0..plan.layers().len() {
            let mut slots = [Deviceptr(0); 4];
            let widths = [shared.kv_width(), shared.kv_width(), hidden, hidden];
            for (slot, width) in widths.into_iter().enumerate() {
                let bytes = width.max(1) * history * std::mem::size_of::<f32>();
                let pointer = cuda.alloc(bytes, &mut total)?;
                cuda.memset_d8_async(pointer, 0, bytes)?;
                slots[slot] = pointer;
            }
            conv.push(slots);
        }
        // Eight summaries for the released block size, the count the plan's buffer layout uses.
        let summaries_rows = plan
            .layers()
            .iter()
            .filter(|layer| {
                layer
                    .index
                    .is_multiple_of(shared.config.n_layers_per_attn_res_block.max(1))
            })
            .count()
            .max(1);
        let summaries_bytes = summaries_rows * hidden * std::mem::size_of::<f32>();
        let summaries = cuda.alloc(summaries_bytes, &mut total)?;
        cuda.memset_d8_async(summaries, 0, summaries_bytes)?;
        let scratch = Scratch::allocate(shared, &mut total)?;
        cuda.synchronize()?;
        Ok(Self {
            keys,
            values,
            tables,
            conv,
            summaries,
            summaries_bytes,
            summary_rows: summaries_rows,
            blocks: 0,
            scratch,
            position: 0,
        })
    }

    fn reset(&mut self, shared: &Shared) -> Result<(), String> {
        for (index, (key, value)) in self.keys.iter().zip(&self.values).enumerate() {
            let bytes = shared.plan.layers()[index].kv_rows.max(1) * shared.kv_width() * 2;
            shared.cuda.memset_d8_async(*key, 0, bytes)?;
            shared.cuda.memset_d8_async(*value, 0, bytes)?;
        }
        let history = shared.config.conv_history().max(1);
        for slots in &self.conv {
            // The key/value histories are the cache width; the attention and MLP ones the hidden
            // width. Only the wider of the two matters: a memset of the larger size is safe for
            // both because each slot owns its own allocation.
            let widest = shared.hidden().max(shared.kv_width());
            let bytes = widest * history * std::mem::size_of::<f32>();
            for (slot, width) in [
                shared.kv_width(),
                shared.kv_width(),
                shared.hidden(),
                shared.hidden(),
            ]
            .into_iter()
            .enumerate()
            {
                let own = width * history * std::mem::size_of::<f32>();
                debug_assert!(own <= bytes);
                shared.cuda.memset_d8_async(slots[slot], 0, own)?;
            }
        }
        shared
            .cuda
            .memset_d8_async(self.summaries, 0, self.summaries_bytes)?;
        self.blocks = 0;
        self.position = 0;
        Ok(())
    }
}

/// The scratch arena one session allocates: one row for a decode step, the prefill chunk's width
/// ([PREFILL_CHUNK_ENV]) for a prefill chunk.
///
/// Every buffer below is `[rows][width]`, and the four that a short convolution reads carry
/// `history` rows in front of their first row. Those rows are the previous launch's tail, so the
/// multi-row convolution entry needs no branch on a chunk's first rows and no separate state; the
/// pointers stored here already address row zero, and the headroom is arithmetic on the offset.
struct Scratch {
    x: Deviceptr,
    /// The block-attention residual's output: `[rows][nq][hidden]`, three rows a row on a full
    /// layer.
    rows: Deviceptr,
    /// The single-row walk's norms: three consecutive `[hidden]` slabs a row.
    normed: Deviceptr,
    /// The chunk walk's three separated norm outputs: the shared GEMM reads one contiguous
    /// `[rows][hidden]` per input, which three slabs of one row cannot be.
    q_in: Deviceptr,
    k_in: Deviceptr,
    v_in: Deviceptr,
    /// The heads of the four convolution inputs, whose first `history` rows are the tail of the
    /// row before this arena's first row. The four pointers above address row zero of each.
    k_head: Deviceptr,
    v_head: Deviceptr,
    delta_head: Deviceptr,
    mixed_head: Deviceptr,
    q: Deviceptr,
    gate: Deviceptr,
    attn_bf16: Deviceptr,
    attn: Deviceptr,
    k: Deviceptr,
    v: Deviceptr,
    k_next: Deviceptr,
    v_next: Deviceptr,
    taps: Deviceptr,
    rel: Deviceptr,
    sink: Deviceptr,
    delta: Deviceptr,
    delta_conv: Deviceptr,
    router_logits: Deviceptr,
    indices: Deviceptr,
    shares: Deviceptr,
    band: Deviceptr,
    activation: Deviceptr,
    partials: Deviceptr,
    scales: Deviceptr,
    mixed: Deviceptr,
    mixed_conv: Deviceptr,
    logits: Deviceptr,
    logits_full: Deviceptr,
    logits_bf16: Deviceptr,
    final_row: Deviceptr,
    /// The band's (row, slot) pairs, one pair an entry, uploaded once a layer; under the grouped
    /// MMA, the shared plan's slots sorted by expert.
    plan: Deviceptr,
    /// The grouped MMA's plan: each slot's expert as this rank numbers them (-1 for another
    /// rank's), the items moe_plan cuts (four words each) and how many it wrote.
    local: Deviceptr,
    items: Deviceptr,
    item_count: Deviceptr,
    /// The per-row attention scale a chunk's rows take (`TauRule::at`, computed on the host so
    /// the chunk and the row-at-a-time walk agree to the bit).
    taus: Deviceptr,
    /// The split attention walk's buffers: each key's clip gain and value scale
    /// (`[kv_heads][attention_pass + attention_rows]`), a pass's scores
    /// (`[rows][heads][attention_pass]`) and the recurrence's state between passes
    /// (`[rows][heads][MIX_STATE]`).
    key_gains: Deviceptr,
    value_scales: Deviceptr,
    scores: Deviceptr,
    attention_state: Deviceptr,
    /// The keys one split-walk pass covers, and the rows the buffers are sized for.
    attention_pass: usize,
    attention_rows: usize,
    /// The tensor-core walk's buffers, in a multi-row arena when the walk runs.
    mma: Option<MmaScratch>,
    /// The split-key decode walk's buffers, in a split arena (a decode step's, a verify's) when
    /// the walk runs.
    parts: Option<PartScratch>,
    /// The one allocation every pointer above is an offset into, kept so the session can give it
    /// back: a 1M-token context holds 8 GiB of cache beside it, and a leak per request would
    /// spend the budget three requests in.
    arena: Deviceptr,
}

/// The tensor-core attention walk's buffers (kernels/inkling_attn_sm121.cu): each row's query
/// (`[rows][heads][head_dim]`) and relative state (`[rows][heads][16]`) as the exact walks compute
/// them, a sliding layer's visible keys' clip gains and value scales (`[kv_heads][stride]`, its
/// window and a chunk; a full layer reads its session's [KeyTables]), and under
/// [AttentionWalk::Check] the walk's own output beside the exact one.
#[derive(Debug, Clone, Copy)]
struct MmaScratch {
    q: Deviceptr,
    rel: Deviceptr,
    gains: Deviceptr,
    scales: Deviceptr,
    stride: usize,
    check: Option<Deviceptr>,
}

/// The split-key decode walk's buffers (kernels/inkling_attn_sm121.cu): every run's partial
/// (`[rows][heads][stride][2 + head_dim]`, `stride` the runs a row at the context can have) and
/// under [AttentionWalk::Check] the walk's own output beside the exact one.
#[derive(Debug, Clone, Copy)]
struct PartScratch {
    parts: Deviceptr,
    stride: usize,
    check: Option<Deviceptr>,
}

impl Scratch {
    /// The arena a single-row (decode) session walks in.
    fn allocate(shared: &Shared, total: &mut usize) -> Result<Self, String> {
        Self::allocate_rows(shared, 1, 1, true, total)
    }

    /// The arena for `rows` rows of the schedule, `answers` of which run the head: a prompt
    /// chunk answers its last row, fused lanes and a verify every row. `split` is an arena whose
    /// rows each take the decode step's attention (the split-key walk under the fast walks): the
    /// decode step's own, and a verify's; any other multi-row arena holds the tensor-core walk's
    /// buffers instead.
    fn allocate_rows(
        shared: &Shared,
        rows: usize,
        answers: usize,
        split: bool,
        total: &mut usize,
    ) -> Result<Self, String> {
        let plan = &shared.plan;
        let config = &shared.config;
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let heads = shared.heads();
        let head_dim = config.head_dim;
        let kv_width = shared.kv_width();
        let activation = plan.expert_intermediate();
        let top_k = config.num_experts_per_tok;
        let vocab = plan.rank_vocab();
        let taps = plan.rank_kv_heads() * config.sconv_kernel_size;
        let rows = rows.max(1);
        let history = config.conv_history().max(1);
        let mut cursor = 0usize;
        // Every buffer starts on a 256-byte boundary: the tensor-core tiles copy 16 bytes at a
        // time from the band and activation buffers.
        let mut take = |elements: usize, width: usize| {
            let at = cursor.next_multiple_of(256);
            cursor = at + (elements.max(1) * width).max(1);
            at
        };
        let x = take(rows * hidden, 4);
        let block_inputs = take(rows * 3 * hidden, 4);
        let normed = take(3 * hidden, 4);
        let q_in = take(rows * hidden, 4);
        let k_in = take(rows * hidden, 4);
        let v_in = take(rows * hidden, 4);
        let q = take(rows * heads * head_dim, 4);
        let gate = take(rows * heads * head_dim, 4);
        let attn_bf16 = take(rows * heads * head_dim, 2);
        let attn = take(rows * heads * head_dim, 4);
        // The four convolution inputs: their first `history` rows are the tail of the row before
        // this arena's first row, which is why the stored pointer is `history` rows in.
        let k_head = take((rows + history) * kv_width, 4);
        let v_head = take((rows + history) * kv_width, 4);
        let k = k_head + history * kv_width * 4;
        let v = v_head + history * kv_width * 4;
        let k_next = take(rows * kv_width, 4);
        let v_next = take(rows * kv_width, 4);
        let taps_buffer = take(rows * taps, 4);
        let rel = take(rows * heads * config.d_rel, 4);
        let sink = take(rows * heads, 4);
        let delta_head = take((rows + history) * hidden, 4);
        let delta = delta_head + history * hidden * 4;
        let delta_conv = take(rows * hidden, 4);
        let router_logits = take(rows * config.n_routed_experts, 4);
        let indices = take(rows * top_k, 4);
        let shares = take(rows * top_k, 4);
        let band = take(rows * top_k * 2 * activation, 4);
        let activation_buffer = take(rows * top_k * activation, 4);
        let partials = take(rows * top_k * hidden, 4);
        let scales = take(rows * top_k, 4);
        let mixed_head = take((rows + history) * hidden, 4);
        let mixed = mixed_head + history * hidden * 4;
        let mixed_conv = take(rows * hidden, 4);
        let answers = answers.clamp(1, rows);
        let logits = take(answers * vocab.max(1), 4);
        let logits_full = take(answers * vocab.max(1) * shared.exchange.world(), 4);
        // The softcap trims the whole gathered row, so the released bf16 row is as wide as the
        // gathered vocabulary and not as wide as this rank's slice of it.
        let logits_bf16 = take(answers * vocab.max(1) * shared.exchange.world(), 2);
        let final_row = take(answers * hidden, 4);
        let plan = take(rows * top_k * 2, 4);
        let local = take(rows * top_k, 4);
        let items = take(
            chunk::moe_items(rows * top_k, shared.plan.expert_range().1) * 4,
            4,
        );
        let item_count = take(1, 4);
        let taus = take(rows, 4);
        let attention_pass = attention_pass_slots(rows, heads, split);
        let kv_heads = shared.plan.rank_kv_heads();
        let key_gains = take(kv_heads * (attention_pass + rows), 4);
        let value_scales = take(kv_heads * (attention_pass + rows), 4);
        let scores = take(rows * heads * attention_pass, 4);
        let attention_state = take(rows * heads * MIX_STATE, 4);
        // A full layer reads its session's [KeyTables]; a sliding layer's chunk sees at most
        // window + rows - 1 slots, which these tables hold.
        let mma = (!split && rows > 1 && shared.attention.fast()).then(|| {
            let stride = (config.sliding_window_size + rows - 1)
                .min(shared.context)
                .max(1);
            (
                take(rows * heads * head_dim, 4),
                take(rows * heads * MMA_D_REL, 4),
                take(kv_heads * stride, 4),
                take(kv_heads * stride, 4),
                stride,
                (shared.attention == AttentionWalk::Check)
                    .then(|| take(rows * heads * head_dim, 2)),
            )
        });
        let parts = (split && shared.attention.fast()).then(|| {
            let stride = shared.context.max(1).div_ceil(PART_KEYS);
            (
                take(rows * heads * stride * (2 + head_dim), 4),
                stride,
                (shared.attention == AttentionWalk::Check)
                    .then(|| take(rows * heads * head_dim, 2)),
            )
        });
        let arena = cuda.alloc(cursor, total)?;
        cuda.memset_d8_async(arena, 0, cursor)?;
        let at = |offset: usize| arena.offset(offset);
        Ok(Self {
            x: at(x),
            rows: at(block_inputs),
            normed: at(normed),
            q_in: at(q_in),
            k_in: at(k_in),
            v_in: at(v_in),
            k_head: at(k_head),
            v_head: at(v_head),
            delta_head: at(delta_head),
            mixed_head: at(mixed_head),
            q: at(q),
            gate: at(gate),
            attn_bf16: at(attn_bf16),
            attn: at(attn),
            k: at(k),
            v: at(v),
            k_next: at(k_next),
            v_next: at(v_next),
            taps: at(taps_buffer),
            rel: at(rel),
            sink: at(sink),
            delta: at(delta),
            delta_conv: at(delta_conv),
            router_logits: at(router_logits),
            indices: at(indices),
            shares: at(shares),
            band: at(band),
            activation: at(activation_buffer),
            partials: at(partials),
            scales: at(scales),
            mixed: at(mixed),
            mixed_conv: at(mixed_conv),
            logits: at(logits),
            logits_full: at(logits_full),
            logits_bf16: at(logits_bf16),
            final_row: at(final_row),
            plan: at(plan),
            local: at(local),
            items: at(items),
            item_count: at(item_count),
            taus: at(taus),
            key_gains: at(key_gains),
            value_scales: at(value_scales),
            scores: at(scores),
            attention_state: at(attention_state),
            attention_pass,
            attention_rows: rows,
            mma: mma.map(|(q, rel, gains, scales, stride, check)| MmaScratch {
                q: at(q),
                rel: at(rel),
                gains: at(gains),
                scales: at(scales),
                stride,
                check: check.map(at),
            }),
            parts: parts.map(|(parts, stride, check)| PartScratch {
                parts: at(parts),
                stride,
                check: check.map(at),
            }),
            arena,
        })
    }
}

/// One serving session.
struct InklingDeviceSession {
    shared: Arc<Shared>,
    state: State,
    /// The chunk buffer and its device state.
    ///
    /// A chunk's rows are walked together rather than one at a time, and two of the braids they
    /// need are per-pass rather than per-layer: the block summary stack the rows push as they
    /// cross a boundary, and the four-tap histories the last row of each convolution leaves.
    /// [Chunk::open] copies the sequence's own stack and histories in, [Chunk::close] copies them
    /// back, and the chunk's working copies are sized for one pass of the chunk width's rows.
    chunk: Option<Chunk>,
    history: Vec<u32>,
    /// The image spans `history` holds. A history is ids, and an image's placeholders are the
    /// same ids whatever the image, so a prompt reuses a prefix only when it carries these same
    /// images at these same spans.
    held_images: Vec<ImageSpan>,
    last_logits: Vec<f32>,
    cached: usize,
    sampling: Sampling,
    sampler: Sampler,
    /// The verify buffer, opened at the first round that drafts.
    verify: Option<VerifyBuffer>,
    /// The last verify, while its drafts are fed back.
    round: Option<Round>,
    /// What this request's drafts have been worth.
    speculator: Speculator,
    /// The DSpark drafter's context for this sequence, when the rank drafts with it.
    drafter: Option<DraftState>,
}

/// The device state one prefill chunk walks in: `rows` rows of every scratch buffer, and a working
/// copy of the two pieces of state a chunk advances per pass rather than per layer.
struct Chunk {
    /// The widest chunk this buffer holds.
    rows: usize,
    /// The rows whose logits it holds: one for a prompt's chunks, a lane's for fused steps.
    answers: usize,
    scratch: Scratch,
    /// The block-summary stacks the chunk's rows push into: one stack a row, because the stack is
    /// a token's residual chain and not a history.
    summaries: Deviceptr,
    /// The elements one row's stack holds, between rows.
    summary_stride: usize,
    /// The rows of one stack, which the block residual's bound checks.
    summary_rows: usize,
}

impl Chunk {
    /// The fewest rows a chunk buffer is opened with: a short prompt's buffer is this wide, so a
    /// conversation's next few turns fit it rather than reopening it.
    const MIN_ROWS: usize = 128;

    /// The rows a buffer for a chunk of `rows` is opened with: the next power of two, at least
    /// [Self::MIN_ROWS] and at most the widest chunk, so a sequence whose chunks grow reopens its
    /// buffer a handful of times rather than at every turn.
    fn rows_for(shared: &Shared, rows: usize) -> usize {
        rows.next_power_of_two()
            .max(Self::MIN_ROWS)
            .min(shared.prefill_chunk)
            .max(rows)
    }

    /// A chunk buffer of `rows` rows for this rank, allocated when a sequence first needs one and
    /// reopened wider when a chunk outgrows it.
    ///
    /// The arena is the session's own rather than a shared one: a chunk's intermediates are
    /// live from the first layer to the last, and two sessions that shared them would have to
    /// serialise a pass apiece. A buffer is ~0.7 MiB a row a rank, so it is opened at the width
    /// the sequence's prompt needs and not at the widest chunk.
    ///
    /// `answers` is the rows whose logits it holds (at most `rows`): a lane steps one row and
    /// answers it, a prompt answers its last. `split` is a verify's buffer, whose rows take the
    /// decode step's attention ([Scratch::allocate_rows]).
    fn allocate(shared: &Shared, rows: usize, answers: usize, split: bool) -> Result<Self, String> {
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let mut total = 0usize;
        let answers = answers.clamp(1, rows.max(1));
        let scratch = Scratch::allocate_rows(shared, rows, answers, split, &mut total)?;
        // A stack grows one row a block boundary, so its depth is the schedule's, not a guess.
        let block = shared.config.n_layers_per_attn_res_block.max(1);
        let summary_rows = shared.plan.layers().len().div_ceil(block) + 1;
        // The stride is in elements, which is the unit the kernel indexes with; the allocation
        // is in bytes.
        let summary_stride = summary_rows * hidden;
        let summaries_bytes = summary_stride * std::mem::size_of::<f32>() * rows;
        let summaries = cuda.alloc(summaries_bytes, &mut total)?;
        cuda.memset_d8_async(summaries, 0, summaries_bytes)?;
        cuda.synchronize()?;
        let bytes = total;
        if trace_on() {
            eprintln!(
                "[inkling] chunk buffer: {rows} rows, {bytes} bytes, {} of them the summary stacks",
                summaries_bytes
            );
        }
        Ok(Self {
            rows,
            answers,
            scratch,
            summaries,
            summary_stride,
            summary_rows,
        })
    }

    /// The device memory this buffer owns.
    fn buffers(&self) -> [Deviceptr; 2] {
        [self.scratch.arena, self.summaries]
    }

    /// The head of one convolution input for this chunk: the rows before its first row.
    fn conv_head(&self, slot: usize) -> Deviceptr {
        match slot {
            0 => self.scratch.k_head,
            1 => self.scratch.v_head,
            2 => self.scratch.delta_head,
            _ => self.scratch.mixed_head,
        }
    }
}

/// A verify's convolution inputs, a layer at a time: each of the four convolutions' history rows
/// before the verify's first row and the verify's own rows, in the row layout a chunk reads
/// (`[history + rows][channels]`, the key and value slots the cache width, the attention and MLP
/// slots the hidden width). The histories after a leading run of `kept` rows are the `history`
/// rows from row `kept` on, which is what [InklingDeviceSession::settle] leaves when the caller
/// fed fewer of the drafts than the verify ran.
#[derive(Debug, Clone, Copy)]
struct VerifyTails {
    base: Deviceptr,
    /// The rows a verify may have.
    rows: usize,
    history: usize,
    kv_width: usize,
    hidden: usize,
}

impl VerifyTails {
    fn new(shared: &Shared, base: Deviceptr, rows: usize) -> Self {
        Self {
            base,
            rows,
            history: shared.config.conv_history().max(1),
            kv_width: shared.kv_width(),
            hidden: shared.hidden(),
        }
    }

    /// The bytes the tails of `layers` layers take.
    fn bytes(rows: usize, shared: &Shared) -> usize {
        let tails = Self::new(shared, Deviceptr(0), rows);
        shared.plan.layers().len() * tails.layer_floats() * std::mem::size_of::<f32>()
    }

    /// A convolution slot's width, in the state's slot order.
    fn channels(&self, slot: usize) -> usize {
        if slot < 2 { self.kv_width } else { self.hidden }
    }

    fn layer_floats(&self) -> usize {
        (self.history + self.rows) * 2 * (self.kv_width + self.hidden)
    }

    /// Where one layer's slot keeps its rows.
    fn at(&self, layer: usize, slot: usize) -> Deviceptr {
        let before: usize = (0..slot).map(|slot| self.channels(slot)).sum();
        self.base.offset(
            (layer * self.layer_floats() + before * (self.history + self.rows))
                * std::mem::size_of::<f32>(),
        )
    }
}

/// A sequence's verify buffer: a split chunk buffer as deep as its deepest round, every row
/// answered, and the [VerifyTails] that let it keep a leading run of the rows.
struct VerifyBuffer {
    chunk: Chunk,
    tails: VerifyTails,
}

impl VerifyBuffer {
    fn allocate(shared: &Shared, rows: usize) -> Result<Self, String> {
        let chunk = Chunk::allocate(shared, rows, rows, true)?;
        let mut total = 0usize;
        let base = shared
            .cuda
            .alloc(VerifyTails::bytes(rows, shared), &mut total)
            .inspect_err(|_| {
                let pointers = chunk.buffers();
                shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
            })?;
        Ok(Self {
            chunk,
            tails: VerifyTails::new(shared, base, rows),
        })
    }

    fn buffers(&self) -> [Deviceptr; 3] {
        let [arena, summaries] = self.chunk.buffers();
        [arena, summaries, self.tails.base]
    }
}

/// The last verify, while the caller feeds its drafts back.
///
/// A verify consumed the anchor and every draft, and holds each row's logits. The caller then
/// feeds tokens one at a time, as it would to plain steps; each that is the next draft takes that
/// draft's row, and the first that is not ends the round, which keeps the rows fed and gives the
/// rest back ([InklingDeviceSession::settle]). The tokens fed are the caller's own choices, so the
/// output is the plain steps' whatever the sampler: a draft only saves the step that would have
/// computed the same row.
#[derive(Debug, Clone)]
struct Round {
    /// The position of the anchor's row.
    start: usize,
    /// The anchor and the drafts the verify ran.
    chunk: Vec<u32>,
    /// Each row's logits.
    rows: Vec<Vec<f32>>,
    /// The rows the caller has fed: the anchor, then each draft it fed in turn.
    fed: usize,
}

/// The drafter a sequence speculates with ([DRAFT_ENV]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drafter {
    /// Prompt lookup: the tokens that followed the latest earlier occurrence of the history's
    /// last n-gram ([sparkflash_model::spec::lookup_draft]).
    Lookup,
    /// The checkpoint's own DSpark block drafter ([draft]), loaded from its [DRAFT_FILE].
    Dspark,
}

/// The drafter, unset or `off` for none: `lookup` (prompt lookup) or `dspark` (the checkpoint's
/// block drafter, whose block bounds the depth at seven).
pub const DRAFT_ENV: &str = "SPARKFLASH_INKLING_DRAFT";
/// The most drafts a round verifies, `1..=MAX_DRAFT_DEPTH`; [DEFAULT_DRAFT_DEPTH] unset.
pub const DRAFT_DEPTH_ENV: &str = "SPARKFLASH_INKLING_DRAFT_DEPTH";
const DEFAULT_DRAFT_DEPTH: usize = 7;
const MAX_DRAFT_DEPTH: usize = 15;
/// The n-grams prompt lookup matches, longest first.
const LOOKUP_NGRAMS: (usize, usize) = (2, 4);

/// The drafter and its depth from [DRAFT_ENV] and [DRAFT_DEPTH_ENV]; a value that is neither is
/// refused rather than read as off.
fn draft_from_env() -> Result<Option<(Drafter, usize)>, String> {
    let drafter = match std::env::var(DRAFT_ENV).ok().as_deref() {
        None | Some("" | "off") => return Ok(None),
        Some("lookup") => Drafter::Lookup,
        Some("dspark") => Drafter::Dspark,
        Some(other) => {
            return Err(format!(
                "{DRAFT_ENV}={other} is not a drafter: use lookup (prompt lookup), dspark (the \
                 checkpoint's block drafter) or off"
            ));
        }
    };
    let depth = match std::env::var(DRAFT_DEPTH_ENV).ok().as_deref() {
        None | Some("") => DEFAULT_DRAFT_DEPTH,
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|depth| (1..=MAX_DRAFT_DEPTH).contains(depth))
            .ok_or_else(|| {
                format!("{DRAFT_DEPTH_ENV}={value} is not a depth in 1..={MAX_DRAFT_DEPTH}")
            })?,
    };
    Ok(Some((drafter, depth)))
}

/// Make `buffer` at least `rows` wide with logits for at least `answers` rows: opened on first
/// use, reopened wider when a chunk or a batch of lanes outgrows it. A walk copies a sequence's
/// histories in at its start and back at its end, so a reopened buffer carries nothing over; a
/// reopened buffer keeps the answered rows the old one held, so a prompt chunk never narrows a
/// lanes buffer.
fn open_buffer(
    shared: &Shared,
    buffer: &mut Option<Chunk>,
    rows: usize,
    answers: usize,
) -> Result<(), BackendError> {
    let held = buffer.as_ref().map_or(1, |chunk| chunk.answers);
    if buffer
        .as_ref()
        .is_some_and(|chunk| chunk.rows < rows || chunk.answers < answers)
    {
        let old = buffer.take().expect("a chunk buffer was just checked");
        let pointers = old.buffers();
        shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
    }
    if buffer.is_none() {
        let width = Chunk::rows_for(shared, rows);
        *buffer = Some(
            Chunk::allocate(shared, width, answers.max(held), false)
                .map_err(|error| BackendError::new(DEVICE, error))?,
        );
    }
    Ok(())
}

impl Drop for InklingDeviceSession {
    /// Give the sequence's device memory back: the cache, the convolution histories, the summary
    /// stack, and both scratch arenas.
    ///
    /// A served request ends by dropping its session, and the next request allocates its own, so
    /// this is what keeps the arm servable at a long context: at 1M tokens a session's cache is
    /// 8.04 GiB a rank, and a session that leaked it would spend the whole budget in a request or
    /// two.
    fn drop(&mut self) {
        let state = &self.state;
        let mut pointers: Vec<Deviceptr> = Vec::new();
        pointers.extend(state.keys.iter().copied());
        pointers.extend(state.values.iter().copied());
        for tables in state.tables.iter().flatten() {
            pointers.extend([tables.gains, tables.scales]);
        }
        for taps in &state.conv {
            pointers.extend(taps.iter().copied());
        }
        pointers.push(state.summaries);
        pointers.push(state.scratch.arena);
        if let Some(chunk) = &self.chunk {
            pointers.extend(chunk.buffers());
        }
        if let Some(verify) = &self.verify {
            pointers.extend(verify.buffers());
        }
        if let Some(drafter) = &self.drafter {
            pointers.extend(drafter.buffers());
        }
        let refs: Vec<&Deviceptr> = pointers.iter().collect();
        self.shared.cuda.free_buffers(&refs);
    }
}

impl ModelSession for InklingDeviceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
        self.sampler.reset();
        // A request reports its own drafts, not the session's lifetime.
        self.speculator.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prompt(tokens, &[])
    }

    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        self.prompt(tokens, images)
    }

    /// One token: the next draft of the last verify takes that draft's row; any other token
    /// ends the round ([Self::settle]) and steps, drafting first when a drafter is set.
    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        if let Some(round) = self.round.as_mut()
            && round.chunk.get(round.fed) == Some(&token)
        {
            let logits = round.rows[round.fed].clone();
            round.fed += 1;
            return Ok(self.decoded(token, logits));
        }
        self.settle()?;
        if self.state.position >= self.shared.context {
            return Err(BackendError::new("context", "the context is exhausted"));
        }
        let drafts = self.draft(token)?;
        if drafts.is_empty() {
            let position = self.state.position;
            let logits = self.step(token)?;
            self.inject(1, position)?;
            return Ok(self.decoded(token, logits));
        }
        let start = self.state.position;
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(token);
        chunk.extend_from_slice(&drafts);
        let rows = self.verify_rows(&chunk)?;
        let logits = rows
            .first()
            .cloned()
            .ok_or_else(|| Self::fail("a verify answered no row"))?;
        self.round = Some(Round {
            start,
            chunk,
            rows,
            fed: 1,
        });
        Ok(self.decoded(token, logits))
    }

    fn speculation(&self) -> Speculation {
        self.speculator.stats()
    }

    /// The tokens the sequence was fed. A verify's cache may hold drafts past them; the next
    /// prompt gives those back ([Self::settle]) before it extends the history, so the prefix is
    /// one this sequence can honour.
    fn reusable_prefix(&self) -> Option<&[u32]> {
        Some(&self.history)
    }

    fn cached_tokens(&self) -> usize {
        self.cached
    }
}

/// One image's rows in a prompt: `count` positions from `position` on take the rows at `source`.
#[derive(Debug, Clone, Copy)]
struct ImageRun {
    position: usize,
    count: usize,
    source: Deviceptr,
}

impl InklingDeviceSession {
    /// Consumes a prompt whose image spans take `images`' rows ([ModelSession::prefill_images]).
    ///
    /// The held history is reused when the prompt extends it and carries the images it holds at
    /// the spans it holds them; otherwise the sequence starts over. The images after the reused
    /// prefix are encoded on rank 0 into a zeroed row buffer every rank holds, the group's sum
    /// makes every rank's buffer rank 0's, and each chunk writes its image rows over the
    /// placeholders' after the embedding norm ([Injection]).
    fn prompt(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("request", "the prompt is empty"));
        }
        self.shared
            .span_check()
            .check(tokens, images)
            .map_err(|error| BackendError::new("request", error))?;
        if !images.is_empty() && self.shared.encodes() && self.shared.vision.is_none() {
            return Err(BackendError::new(
                "vision",
                format!(
                    "the request carries {} images and this rank holds no vision tower",
                    images.len()
                ),
            ));
        }
        // A verify's unfed drafts are given back first, so the state ends where the history does.
        self.settle()?;
        let held = self.history.len();
        let reuse = tokens.starts_with(&self.history)
            && self
                .held_images
                .iter()
                .all(|span| images.iter().any(|image| image.span == *span))
            && images
                .iter()
                .all(|image| image.span.start >= held || self.held_images.contains(&image.span));
        let start = if reuse {
            held
        } else {
            self.state
                .reset(&self.shared)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            if let Some(drafter) = self.drafter.as_mut() {
                drafter.reset();
            }
            self.history.clear();
            self.held_images.clear();
            self.last_logits.clear();
            0
        };
        self.cached = start;
        let remaining = tokens.len().saturating_sub(start);
        if self.state.position + remaining > self.shared.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {remaining} positions, {} remain",
                    self.shared.context.saturating_sub(self.state.position)
                ),
            ));
        }
        // The penalties count the whole prompt, the reused prefix included: it is one request.
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        if remaining == 0 && !self.last_logits.is_empty() {
            return Ok(self.output(self.last_logits.clone()));
        }
        if remaining == 0 {
            return Err(BackendError::new(
                "request",
                "the prompt adds no position and this sequence holds no logits",
            ));
        }
        let fresh: Vec<&VisionImage> = images
            .iter()
            .filter(|image| image.span.start >= start)
            .collect();
        let rows: usize = fresh.iter().map(|image| image.span.grid.image_rows()).sum();
        let buffer = if rows == 0 {
            None
        } else {
            Some(self.encode_images(&fresh, rows)?)
        };
        let mut runs = Vec::with_capacity(fresh.len());
        let row_bytes = self.shared.hidden() * std::mem::size_of::<f32>();
        let mut at = 0usize;
        for image in &fresh {
            let count = image.span.grid.image_rows();
            if let Some(buffer) = buffer {
                runs.push(ImageRun {
                    position: image.span.start + 1,
                    count,
                    source: buffer.offset(at * row_bytes),
                });
            }
            at += count;
        }
        let answer = self.forward(tokens, start, &runs);
        if let Some(buffer) = buffer {
            // The chunks' copies out of the buffer are queued; they finish before it goes.
            let drained = self.shared.cuda.synchronize();
            self.shared.cuda.free_buffers(&[&buffer]);
            drained.map_err(Self::fail)?;
        }
        let logits = answer?;
        self.held_images = images.iter().map(|image| image.span).collect();
        self.last_logits = logits.clone();
        Ok(self.output(logits))
    }

    /// Forwards `tokens[start..]` a chunk at a time, writing `runs`' image rows over their
    /// positions, and answers the last position's logits.
    fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        runs: &[ImageRun],
    ) -> Result<Vec<f32>, BackendError> {
        let remaining = tokens.len() - start;
        let row_bytes = self.shared.hidden() * std::mem::size_of::<f32>();
        // A prompt's positions are forwarded a chunk at a time. Under the chunk walk a tail of
        // one position takes the decode step, which computes the same numbers at a narrower
        // launch; under the tensor walk it stays a chunk, because a decode step's projections are
        // the GEMV's sums and a prompt's last row would then depend on where the chunks fell. An
        // image row is always a chunk's: the decode step has no rows to write it over.
        let walk = self.shared.prefill_walk;
        let mut logits = Vec::new();
        let mut forwarded = 0usize;
        while forwarded < remaining {
            let take = if walk.chunked() {
                self.shared.chunk_widths.next(remaining - forwarded)
            } else {
                1
            };
            let first = start + forwarded;
            let slice = &tokens[first..first + take];
            let images: Vec<Injection> = runs
                .iter()
                .filter_map(|run| {
                    let from = run.position.max(first);
                    let to = (run.position + run.count).min(first + take);
                    (from < to).then(|| Injection {
                        row: from - first,
                        source: run.source.offset((from - run.position) * row_bytes),
                        count: to - from,
                    })
                })
                .collect();
            logits = if take == 1 && walk != PrefillWalk::Tensor && images.is_empty() {
                self.step(slice[0])?
            } else {
                let taps = self.drafter.as_ref().map(DraftState::sink);
                // The rows walk forwards an image row as a one-row chunk of the exact walk.
                let walk = if walk.chunked() {
                    walk
                } else {
                    PrefillWalk::Chunk
                };
                self.open_chunk(take)?;
                let chunk = self.chunk.as_mut().expect("a chunk buffer was just opened");
                walk_rows(
                    &self.shared,
                    Rows::Chunk(&mut self.state),
                    chunk,
                    slice,
                    walk,
                    &images,
                    taps,
                )?
                .pop()
                .ok_or_else(|| Self::fail("a chunk answered no row"))?
            };
            self.inject(take, first)?;
            self.history.extend_from_slice(slice);
            forwarded += take;
        }
        Ok(logits)
    }

    /// The image rows of `images` (`rows` of them in all), on every rank: a zeroed f32 buffer that
    /// rank 0's tower fills, summed across the group so every rank holds rank 0's rows. The sum
    /// goes a chunk's worth of rows at a time, the widest the exchange carries; zeros plus rows is
    /// the rows exactly.
    fn encode_images(
        &self,
        images: &[&VisionImage],
        rows: usize,
    ) -> Result<Deviceptr, BackendError> {
        let shared = &self.shared;
        let cuda = &shared.cuda;
        let hidden = shared.hidden();
        let row_bytes = hidden * std::mem::size_of::<f32>();
        let mut total = 0usize;
        let buffer = cuda
            .alloc(rows * row_bytes, &mut total)
            .map_err(Self::fail)?;
        let fill = || -> Result<(), BackendError> {
            cuda.memset_d8_async(buffer, 0, rows * row_bytes)
                .map_err(Self::fail)?;
            if shared.encodes() {
                let tower = shared
                    .vision
                    .as_ref()
                    .ok_or_else(|| Self::fail("rank 0 holds no vision tower"))?;
                let mut tower = tower
                    .lock()
                    .map_err(|_| Self::fail("the vision tower's lock is poisoned"))?;
                let mut at = 0usize;
                for image in images {
                    let grid = &image.span.grid;
                    let made = tower
                        .encode(
                            cuda,
                            &image.pixels,
                            grid.best_height,
                            grid.best_width,
                            buffer.offset(at * row_bytes),
                        )
                        .map_err(Self::fail)?;
                    if made != grid.image_rows() {
                        return Err(Self::fail(format!(
                            "the tower made {made} rows of a span that holds {}",
                            grid.image_rows()
                        )));
                    }
                    at += made;
                }
            }
            let piece = shared.prefill_chunk.max(1);
            let mut done = 0usize;
            while done < rows {
                let count = piece.min(rows - done);
                exchange_sum(shared, buffer.offset(done * row_bytes), count * hidden)?;
                done += count;
            }
            Ok(())
        };
        match fill() {
            Ok(()) => Ok(buffer),
            Err(error) => {
                let _ = cuda.synchronize();
                cuda.free_buffers(&[&buffer]);
                Err(error)
            }
        }
    }

    /// A fresh sequence on this rank: its own cache, histories and stack, no chunk buffer yet.
    fn open(shared: &Arc<Shared>) -> Result<Self, BackendError> {
        shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let state = State::allocate(shared).map_err(|error| BackendError::new(DEVICE, error))?;
        let drafter = match (shared.draft, shared.dspark.as_ref()) {
            (Some((Drafter::Dspark, _)), Some(model)) => Some(
                DraftState::allocate(shared, model)
                    .map_err(|error| BackendError::new(DEVICE, error))?,
            ),
            (Some((Drafter::Dspark, _)), None) => {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "{DRAFT_ENV}=dspark and this rank has not loaded the drafter \
                         (InklingDeviceBackend::load_drafter)"
                    ),
                ));
            }
            _ => None,
        };
        Ok(Self {
            drafter,
            shared: Arc::clone(shared),
            state,
            chunk: None,
            history: Vec::new(),
            held_images: Vec::new(),
            last_logits: Vec::new(),
            cached: 0,
            sampling: Sampling::default(),
            sampler: Sampler::new(),
            verify: None,
            round: None,
            speculator: Speculator::new(shared.draft.map_or(0, |(_, depth)| depth), None),
        })
    }

    /// Forget the sequence, keeping its allocations: the next prompt starts at position zero.
    fn forget(&mut self) -> Result<(), BackendError> {
        self.state
            .reset(&self.shared)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        self.round = None;
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.reset();
        }
        self.history.clear();
        self.held_images.clear();
        self.last_logits.clear();
        self.cached = 0;
        self.sampler.reset();
        Ok(())
    }

    /// The bookkeeping of one decoded token whose logits a step computed: the history, the
    /// penalties' counts, and the next choice.
    fn decoded(&mut self, token: u32, logits: Vec<f32>) -> StepOutput {
        self.history.push(token);
        self.sampler.record(token);
        self.last_logits = logits.clone();
        self.cached = 0;
        self.output(logits)
    }

    /// The drafts to verify after `anchor`: none without a drafter, and never more than the
    /// context has room for past the anchor.
    fn draft(&mut self, anchor: u32) -> Result<Vec<u32>, BackendError> {
        let Some((drafter, _)) = self.shared.draft else {
            return Ok(Vec::new());
        };
        let room = self.shared.context.saturating_sub(self.state.position + 1);
        let depth = self.speculator.depth(&[], room);
        match drafter {
            Drafter::Lookup => {
                let mut context = Vec::with_capacity(self.history.len() + 1);
                context.extend_from_slice(&self.history);
                context.push(anchor);
                let (shortest, longest) = LOOKUP_NGRAMS;
                Ok(lookup_draft(&context, shortest, longest, depth))
            }
            Drafter::Dspark => {
                if depth == 0 {
                    return Ok(Vec::new());
                }
                let shared = Arc::clone(&self.shared);
                let (Some(model), Some(drafter)) = (shared.dspark.as_ref(), self.drafter.as_mut())
                else {
                    return Err(Self::fail("a DSpark round with no drafter on this rank"));
                };
                // The anchor sits at the position the sequence steps next; the drafter's context
                // holds every row before it.
                if drafter.committed() != self.state.position {
                    return Err(Self::fail(format!(
                        "the drafter's context ends at {} and the anchor is at {}",
                        drafter.committed(),
                        self.state.position
                    )));
                }
                drafter.draft(&shared, model, anchor, depth.min(model.max_depth()))
            }
        }
    }

    /// The drafter's context takes the walked rows `position..position + rows`, whose taps the
    /// walk just left: a prompt's chunk, a step's row, or the fed rows of a verify.
    fn inject(&mut self, rows: usize, position: usize) -> Result<(), BackendError> {
        let Some(drafter) = self.drafter.as_mut() else {
            return Ok(());
        };
        let shared = Arc::clone(&self.shared);
        let model = shared
            .dspark
            .as_ref()
            .ok_or_else(|| Self::fail("a drafter state with no drafter on this rank"))?;
        drafter.inject(&shared, model, rows, position)
    }

    /// Walks `chunk` (an anchor and its drafts) as one verify and answers every row's logits;
    /// the state is left past every row until [Self::settle] keeps what was fed.
    fn verify_rows(&mut self, chunk: &[u32]) -> Result<Vec<Vec<f32>>, BackendError> {
        if self.verify.is_none() {
            let rows = self.speculator.max_depth() + 1;
            self.verify = Some(VerifyBuffer::allocate(&self.shared, rows).map_err(Self::fail)?);
        }
        let verify = self
            .verify
            .as_mut()
            .expect("the verify buffer was just opened");
        let tails = verify.tails;
        walk_rows(
            &self.shared,
            Rows::Verify(&mut self.state, tails),
            &mut verify.chunk,
            chunk,
            PrefillWalk::Chunk,
            &[],
            self.drafter.as_ref().map(DraftState::sink),
        )
    }

    /// Ends the last round at the rows fed: the cache keeps them, and the convolution histories
    /// and the position go back to the last fed row. The rows past it stay in the cache, past the
    /// position, where the next rows stored overwrite them before any row reads them.
    fn settle(&mut self) -> Result<(), BackendError> {
        let Some(round) = self.round.take() else {
            return Ok(());
        };
        self.speculator
            .observe(&[], round.chunk.len() - 1, round.fed);
        // The verify left every row's taps; the drafter keeps the rows the caller fed.
        self.inject(round.fed, round.start)?;
        if round.fed == round.chunk.len() {
            return Ok(());
        }
        let tails = self
            .verify
            .as_ref()
            .ok_or_else(|| Self::fail("a round settled with no verify buffer"))?
            .tails;
        let byte = std::mem::size_of::<f32>();
        for (layer, slots) in self.state.conv.iter().enumerate() {
            for (slot, history) in slots.iter().enumerate() {
                let channels = tails.channels(slot);
                self.shared
                    .conv_history(
                        tails.at(layer, slot).offset(round.fed * channels * byte),
                        *history,
                        channels,
                        false,
                    )
                    .map_err(Self::fail)?;
            }
        }
        self.state.position = round.start + round.fed;
        Ok(())
    }

    fn output(&mut self, logits: Vec<f32>) -> StepOutput {
        let next_token = choose(
            &self.shared.config,
            &mut self.sampler,
            &logits,
            &self.sampling,
        );
        StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        }
    }

    /// Make this sequence's chunk buffer at least `rows` wide: opened on the first chunk, reopened
    /// wider when a chunk outgrows it. A chunk copies the sequence's stacks and histories in at
    /// its start and back at its end, so a reopened buffer carries nothing over.
    fn open_chunk(&mut self, rows: usize) -> Result<(), BackendError> {
        open_buffer(&self.shared, &mut self.chunk, rows, 1)
    }

    /// A device refusal, with no borrow of the session: the walk mutates the state while it
    /// reports, so the error path must not hold a reference to it.
    fn fail(error: impl std::fmt::Display) -> BackendError {
        BackendError::new(DEVICE, error.to_string())
    }

    fn weight(&self, index: usize) -> Deviceptr {
        self.shared.weights.get(index).pointer
    }

    /// One all-reduce of a residual-width f32 vector, in place.
    ///
    /// Every collective this arm performs over the hidden stream is a sum of partial sums, so the
    /// exchange is the same staged round-trip whatever produced the partial: a download, the
    /// transport's sum, an upload.
    fn all_reduce(
        &self,
        shared: &Shared,
        pointer: Deviceptr,
        elements: usize,
    ) -> Result<(), BackendError> {
        exchange_sum(shared, pointer, elements)
    }

    /// The root-mean-square of one f32 buffer, for the trace. A debug read: it synchronises.
    fn rms(
        &self,
        shared: &Shared,
        label: &str,
        pointer: Deviceptr,
        elements: usize,
    ) -> Result<(), BackendError> {
        if !trace_on() {
            return Ok(());
        }
        let mut bytes = vec![0u8; elements * 4];
        shared
            .cuda
            .copy_dtoh(&mut bytes, &pointer)
            .map_err(Self::fail)?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let rms = (values.iter().map(|value| value * value).sum::<f32>() / elements as f32).sqrt();
        eprintln!("[inkling] {label} rms={rms:.5} first={:.5}", values[0]);
        Ok(())
    }

    /// One token: the whole schedule, in the plan's order.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        let shared = Arc::clone(&self.shared);
        let config = &shared.config;
        let plan = &shared.plan;
        if token as usize >= config.vocab_size {
            return Err(Self::fail(format!(
                "token {token} is outside the vocabulary {}",
                config.vocab_size
            )));
        }
        let position = self.state.position;
        // The block-summary stack is this token's stream at each block boundary, not a history:
        // the previous token's summaries are a different residual chain.
        self.state.blocks = 0;
        let hidden = shared.hidden();
        let kv_width = shared.kv_width();
        let heads = shared.heads();
        let head_dim = config.head_dim;
        let groups = heads / plan.rank_kv_heads().max(1);
        let run = |result: Result<(), String>| result.map_err(Self::fail);

        for step in plan.prologue() {
            match (step.op, step.target) {
                (Op::Embed, _) => {
                    // The embedding is vocabulary-parallel: exactly one rank holds this token's
                    // row, and the gather below sums the row it widens with the zeros the others
                    // contribute.
                    let rank_vocab = plan.rank_vocab();
                    let first = plan.tp().rank() * rank_vocab;
                    let local = (token as usize).checked_sub(first);
                    let weight = self.weight(step.weights[0]);
                    match local {
                        Some(row) if row < rank_vocab => run(shared.launch(
                            shared.kernels.embed,
                            (flat(hidden), 1, 1),
                            (TOKEN_BLOCK, 1, 1),
                            &[
                                weight.0,
                                self.state.scratch.x.0,
                                row as u64,
                                hidden as u64,
                                hidden as u64,
                                0,
                            ],
                        ))?,
                        _ => run(shared
                            .cuda
                            .memset_d8_async(
                                self.state.scratch.x,
                                0,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?,
                    }
                }
                (Op::RmsNorm, Target::Norm(NormRole::Embedding)) => {
                    let weight = self.weight(step.weights[0]);
                    run(shared.rmsnorm(
                        self.state.scratch.x,
                        weight,
                        self.state.scratch.x,
                        1,
                        hidden,
                    ))?;
                    self.rms(&shared, "embed_norm", self.state.scratch.x, hidden)?;
                }
                (Op::Collective(Collective::EmbeddingRow), _) => {
                    // Each rank embeds the ids its vocabulary slice owns and zeros the rest, so
                    // the sum is one nonzero term and any summing order gives the row.
                    exchange_sum(&shared, self.state.scratch.x, hidden)?;
                    self.rms(&shared, "embed_row", self.state.scratch.x, hidden)?;
                }
                (Op::Collective(_), _) | (Op::SummaryPush, _) => {}
                (op, target) => {
                    return Err(Self::fail(format!(
                        "the schedule's prologue {op:?} at {target:?} has no device dispatch"
                    )));
                }
            }
        }

        let taps = self.drafter.as_ref().map(DraftState::sink);
        for (index, schedule) in plan.layers().iter().enumerate() {
            let layer = schedule.index;
            if let Some((sink, slot)) = taps.and_then(|sink| Some((sink, sink.slot(layer)?))) {
                sink.capture(&shared, self.state.scratch.x, 1, slot)?;
            }
            if trace_on() {
                eprintln!(
                    "[inkling] layer {layer} sliding={} blocks={}",
                    schedule.sliding, self.state.blocks
                );
            }
            let sliding = schedule.sliding;
            let cache_rows = schedule.kv_rows;
            let clips = shared.clip[index];
            let mut previous_conv: Option<ConvSlot> = None;
            let mut cursor = 0usize;
            while cursor < schedule.steps.len() {
                let step = &schedule.steps[cursor];
                if trace_on() {
                    eprintln!("[inkling]   {}/{:?}", step.op.name(), step.target);
                }
                match (step.op, step.target) {
                    (Op::BlockResidual, Target::Input(ResidualInput::Attention)) => {
                        let queries = self.weight(step.weights[0]);
                        run(shared.block_residual(
                            self.state.summaries,
                            self.state.scratch.x,
                            queries,
                            self.state.scratch.rows,
                            self.state.blocks,
                            if sliding { 1 } else { 3 },
                            1,
                            0,
                        ))?;
                    }
                    (Op::SummaryPush, _) => {
                        if self.state.blocks >= self.state.summary_rows {
                            return Err(Self::fail(format!(
                                "layer {layer}: the summary stack is full ({} rows) and the \
                                 plan pushes another",
                                self.state.summary_rows
                            )));
                        }
                        let destination = self
                            .state
                            .summaries
                            .offset(self.state.blocks * hidden * std::mem::size_of::<f32>());
                        run(shared
                            .cuda
                            .copy_dtod(
                                &destination,
                                &self.state.scratch.x,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?;
                        self.state.blocks += 1;
                        // At a boundary the sublayer output *is* the stream: there is no additive
                        // residual and the pre-attention stream is now a summary.
                        run(shared
                            .cuda
                            .copy_dtod(
                                &self.state.scratch.x,
                                &self.state.scratch.delta,
                                hidden * std::mem::size_of::<f32>(),
                            )
                            .map_err(|error| error.to_string()))?;
                    }
                    (Op::RmsNorm, Target::Norm(role)) => {
                        let weight = self.weight(step.weights[0]);
                        let (source, destination) = match (role, sliding) {
                            (NormRole::Stream, _) => {
                                (self.state.scratch.x, self.state.scratch.normed)
                            }
                            (NormRole::Attention, _) => {
                                (self.state.scratch.rows, self.state.scratch.normed)
                            }
                            (NormRole::Query, false) => {
                                (self.state.scratch.rows, self.state.scratch.normed)
                            }
                            (NormRole::Key, false) => (
                                self.state.scratch.rows.offset(hidden * 4),
                                self.state.scratch.normed.offset(hidden * 4),
                            ),
                            (NormRole::Value, false) => (
                                self.state.scratch.rows.offset(2 * hidden * 4),
                                self.state.scratch.normed.offset(2 * hidden * 4),
                            ),
                            (NormRole::Mlp, _) => {
                                (self.state.scratch.rows, self.state.scratch.normed)
                            }
                            (other, _) => {
                                return Err(Self::fail(format!(
                                    "layer {layer}: norm role {other:?} is not an input norm"
                                )));
                            }
                        };
                        run(shared.rmsnorm(source, weight, destination, 1, hidden))?;
                        if trace_on() && layer == 0 && matches!(role, NormRole::Mlp) {
                            self.rms(&shared, "row mlp_norm", destination, hidden)?;
                        }
                    }
                    (Op::Gemv, Target::Project(projection)) => {
                        let normed = self.state.scratch.normed;
                        let io = |projection: Projection| match projection {
                            Projection::Query => {
                                Ok((normed, self.state.scratch.q, heads * head_dim, hidden))
                            }
                            Projection::Gate => {
                                Ok((normed, self.state.scratch.gate, heads * head_dim, hidden))
                            }
                            Projection::Key => Ok((
                                if sliding {
                                    normed
                                } else {
                                    normed.offset(hidden * 4)
                                },
                                self.state.scratch.k,
                                kv_width,
                                hidden,
                            )),
                            Projection::Value => Ok((
                                if sliding {
                                    normed
                                } else {
                                    normed.offset(2 * hidden * 4)
                                },
                                self.state.scratch.v,
                                kv_width,
                                hidden,
                            )),
                            Projection::Relative => {
                                Ok((normed, self.state.scratch.rel, heads * config.d_rel, hidden))
                            }
                            Projection::Sink => {
                                Ok((normed, self.state.scratch.sink, heads, hidden))
                            }
                            Projection::AttentionOut => Ok((
                                self.state.scratch.attn,
                                self.state.scratch.delta,
                                hidden,
                                heads * head_dim,
                            )),
                            Projection::Router => Ok((
                                normed,
                                self.state.scratch.router_logits,
                                config.n_routed_experts,
                                hidden,
                            )),
                            Projection::Head => Err(Self::fail("the head is a global step")),
                        };
                        let (input, out, rows, columns) = io(projection)?;
                        // The projections that follow over the same input go out with this one,
                        // up to four a launch: a sliding layer's six attention projections all
                        // read the one normed row, so they are two launches rather than six. A
                        // fused row's sum is its own launch's bits (Shared::gemv_fused).
                        let mut parts = [(Deviceptr(0), Deviceptr(0), 0usize); GEMV_FUSED];
                        parts[0] = (self.weight(step.weights[0]), out, rows);
                        let mut fused = 1;
                        while fused < GEMV_FUSED {
                            let Some(next) = schedule.steps.get(cursor + fused) else {
                                break;
                            };
                            let (Op::Gemv, Target::Project(following)) = (next.op, next.target)
                            else {
                                break;
                            };
                            let (next_input, next_out, next_rows, next_columns) = io(following)?;
                            if next_input.0 != input.0 || next_columns != columns {
                                break;
                            }
                            parts[fused] = (self.weight(next.weights[0]), next_out, next_rows);
                            fused += 1;
                        }
                        if fused > 1 {
                            run(shared.gemv_fused(&parts[..fused], input, columns))?;
                            cursor += fused;
                            continue;
                        }
                        run(shared.gemv(parts[0].0, input, out, rows, columns))?;
                        if layer == 0 && matches!(projection, Projection::AttentionOut) {
                            self.rms(&shared, "attention_out", out, rows)?;
                        }
                    }
                    (Op::ShortConv, Target::Conv(slot)) => {
                        let taps = self.weight(step.weights[0]);
                        let (input, out, channels, state) = match slot {
                            ConvSlot::Key => (
                                self.state.scratch.k,
                                self.state.scratch.k_next,
                                kv_width,
                                self.state.conv[layer][0],
                            ),
                            ConvSlot::Value => (
                                self.state.scratch.v,
                                self.state.scratch.v_next,
                                kv_width,
                                self.state.conv[layer][1],
                            ),
                            ConvSlot::Attention => (
                                self.state.scratch.delta,
                                self.state.scratch.delta,
                                hidden,
                                self.state.conv[layer][2],
                            ),
                            ConvSlot::Mlp => (
                                self.state.scratch.mixed,
                                self.state.scratch.mixed,
                                hidden,
                                self.state.conv[layer][3],
                            ),
                        };
                        run(shared.launch(
                            shared.kernels.short_conv,
                            (flat(channels), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                input.0,
                                taps.0,
                                state.0,
                                out.0,
                                channels as u64,
                                shared.kernel_width() as u64,
                            ],
                        ))?;
                        previous_conv = Some(slot);
                    }
                    (Op::ShortConvDynamic, Target::Conv(slot)) => {
                        let static_taps = self.weight(step.weights[0]);
                        let projection = self.weight(step.weights[1]);
                        let (input, out, state, source) = match slot {
                            ConvSlot::Key => (
                                self.state.scratch.k,
                                self.state.scratch.k_next,
                                self.state.conv[layer][0],
                                self.state.scratch.normed.offset(hidden * 4),
                            ),
                            ConvSlot::Value => (
                                self.state.scratch.v,
                                self.state.scratch.v_next,
                                self.state.conv[layer][1],
                                self.state.scratch.normed.offset(2 * hidden * 4),
                            ),
                            other => {
                                return Err(Self::fail(format!(
                                    "layer {layer}: a dynamic convolution has no {other:?} slot"
                                )));
                            }
                        };
                        // The dynamic taps are a projection of this slot's input, one kernel width
                        // a key/value head.
                        let taps = plan.rank_kv_heads() * shared.kernel_width();
                        run(shared.gemv(
                            projection,
                            source,
                            self.state.scratch.taps,
                            taps,
                            hidden,
                        ))?;
                        run(shared.launch(
                            shared.kernels.short_conv_dynamic,
                            (flat(kv_width), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                input.0,
                                static_taps.0,
                                self.state.scratch.taps.0,
                                state.0,
                                out.0,
                                kv_width as u64,
                                shared.kernel_width() as u64,
                                head_dim as u64,
                            ],
                        ))?;
                        previous_conv = Some(slot);
                    }
                    (Op::Widen, _) => {
                        run(shared.launch(
                            shared.kernels.widen,
                            (flat(heads * head_dim), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.attn_bf16.0,
                                self.state.scratch.attn.0,
                                (heads * head_dim) as u64,
                            ],
                        ))?;
                    }
                    (Op::KvStore, _) => {
                        if trace_on() {
                            let rank = plan.tp().rank();
                            let read = |pointer: Deviceptr, count: usize| -> Vec<f32> {
                                let mut bytes = vec![0u8; count * 4];
                                let _ = shared.cuda.copy_dtoh(&mut bytes, &pointer);
                                from_bytes(&bytes)
                            };
                            let q = read(self.state.scratch.q, 2);
                            let k = read(self.state.scratch.k_next, 2);
                            let v = read(self.state.scratch.v_next, 2);
                            let gate = read(self.state.scratch.gate, 1);
                            let sink = read(self.state.scratch.sink, 1);
                            let rel = read(self.state.scratch.rel, 1);
                            eprintln!(
                                "[inkling] device rank {rank} layer {layer} q={:.6} {:.6} g={:.6}                                  s={:.6} r={:.6} k={:.6} {:.6} v={:.6} {:.6}",
                                q[0], q[1], gate[0], sink[0], rel[0], k[0], k[1], v[0], v[1]
                            );
                        }
                        // A ring cache indexes by the position modulo its length; the attention
                        // body is given the absolute position and maps it itself.
                        let slot_position = position % cache_rows.max(1);
                        run(shared.launch(
                            shared.kernels.kv_store,
                            (flat(kv_width), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.k_next.0,
                                self.state.scratch.v_next.0,
                                self.state.keys[index].0,
                                self.state.values[index].0,
                                1,
                                slot_position as u64,
                                kv_width as u64,
                                // The single-row walk wraps the slot here; a chunk passes the
                                // ring length and lets the entry wrap each of its rows.
                                0,
                            ],
                        ))?;
                    }
                    (Op::Attention, _) => {
                        let bank = self.weight(step.weights[0]);
                        let scratch = &self.state.scratch;
                        run(shared.attention_rows(
                            scratch,
                            index,
                            scratch.q,
                            scratch.gate,
                            self.state.keys[index],
                            self.state.values[index],
                            self.state.tables[index],
                            scratch.rel,
                            bank,
                            scratch.sink,
                            clips,
                            scratch.attn_bf16,
                            Deviceptr(0),
                            1,
                            position,
                            plan.rank_kv_heads(),
                            groups,
                            if sliding { schedule.window } else { 0 },
                            // A sliding layer's cache is its window and is read as a ring; a
                            // full layer's is the context and is addressed absolutely.
                            if sliding { cache_rows } else { 0 },
                            schedule.rel_extent,
                            schedule.tau.at(position),
                        ))?;
                    }
                    (Op::ResidualAdd, _) => {
                        if layer == 0 {
                            self.rms(&shared, "mixed", self.state.scratch.mixed, hidden)?;
                        }
                        let delta = match previous_conv {
                            Some(ConvSlot::Mlp) => self.state.scratch.mixed,
                            _ => self.state.scratch.delta,
                        };
                        run(shared.launch(
                            shared.kernels.residual_add,
                            (flat(hidden), 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                self.state.scratch.x.0,
                                delta.0,
                                self.state.scratch.x.0,
                                1,
                                hidden as u64,
                            ],
                        ))?;
                        previous_conv = None;
                    }
                    (Op::BlockResidual, Target::Input(ResidualInput::Mlp)) => {
                        let queries = self.weight(step.weights[0]);
                        run(shared.block_residual(
                            self.state.summaries,
                            self.state.scratch.x,
                            queries,
                            self.state.scratch.rows,
                            self.state.blocks,
                            1,
                            1,
                            0,
                        ))?;
                    }
                    (Op::Router, _) => {
                        let bias = self.weight(step.weights[0]);
                        if trace_on() && layer == 0 {
                            self.rms(
                                &shared,
                                "row router_logits",
                                self.state.scratch.router_logits,
                                config.n_routed_experts,
                            )?;
                        }
                        run(shared.launch(
                            shared.kernels.router,
                            (1, 1, 1),
                            (ROUTER_BLOCK, 1, 1),
                            &[
                                self.state.scratch.router_logits.0,
                                bias.0,
                                self.state.scratch.indices.0,
                                self.state.scratch.shares.0,
                                config.n_routed_experts as u64,
                                config.num_experts_per_tok as u64,
                                bits(config.route_scale as f32),
                            ],
                        ))?;
                    }
                    (Op::ExpertGateUp, Target::Band(BandOp::GateUp)) => {
                        let activation = &schedule.steps[cursor + 1];
                        let down = &schedule.steps[cursor + 2];
                        self.run_band(&shared, index, step, activation, down)?;
                        // The band's three launches and its local mix; the all-reduce that
                        // follows them is the next step and stays the walk's.
                        cursor += 4;
                        continue;
                    }
                    (Op::Collective(Collective::AttentionDelta), _) => {
                        // The output projection sharded the heads, its contraction axis, so the
                        // dense result is a partial sum of every element of the layer's attention
                        // output. The convolution that follows reads a complete row.
                        self.all_reduce(&shared, self.state.scratch.delta, hidden)?;
                        if trace_on() {
                            let label = format!("attn_proj layer {layer}");
                            self.rms(&shared, &label, self.state.scratch.delta, hidden)?;
                        }
                    }
                    (Op::Collective(Collective::ExpertDelta), _) => {
                        // Each rank mixed its own experts and zeroed the slots it does not own,
                        // so the sum over ranks is the layer's whole expert contribution.
                        self.all_reduce(&shared, self.state.scratch.mixed, hidden)?;
                    }
                    (Op::Collective(_), _) => {}
                    (op, target) => {
                        return Err(Self::fail(format!(
                            "layer {layer}: the schedule's {op:?} at {target:?} has no device \
                             dispatch"
                        )));
                    }
                }
                cursor += 1;
            }
            if trace_on() {
                let mut probe = vec![0u8; hidden * 4];
                shared
                    .cuda
                    .copy_dtoh(&mut probe, &self.state.scratch.x)
                    .map_err(Self::fail)?;
                let values: Vec<f32> = probe
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect();
                let rms =
                    (values.iter().map(|value| value * value).sum::<f32>() / hidden as f32).sqrt();
                eprintln!(
                    "[inkling] row rank {} layer {layer} x0={:.5} x1={:.5} rms={rms:.5}",
                    plan.tp().rank(),
                    values[0],
                    values[1]
                );
            }
        }

        for step in plan.epilogue() {
            match (step.op, step.target) {
                (Op::BlockResidual, Target::Input(ResidualInput::Output)) => {
                    let queries = self.weight(step.weights[0]);
                    run(shared.block_residual(
                        self.state.summaries,
                        self.state.scratch.x,
                        queries,
                        self.state.scratch.final_row,
                        self.state.blocks,
                        1,
                        1,
                        0,
                    ))?;
                }
                (Op::RmsNorm, Target::Norm(NormRole::Final)) => {
                    self.rms(&shared, "final_input", self.state.scratch.final_row, hidden)?;
                    let weight = self.weight(step.weights[0]);
                    run(shared.rmsnorm(
                        self.state.scratch.final_row,
                        weight,
                        self.state.scratch.final_row,
                        1,
                        hidden,
                    ))?;
                    if let Some(sink) = taps {
                        sink.capture(&shared, self.state.scratch.final_row, 1, FINAL_TAP)?;
                    }
                }
                (Op::Gemv, Target::Project(Projection::Head)) => {
                    self.rms(&shared, "final_norm", self.state.scratch.final_row, hidden)?;
                    let weight = self.weight(step.weights[0]);
                    run(shared.gemv(
                        weight,
                        self.state.scratch.final_row,
                        self.state.scratch.logits,
                        plan.rank_vocab(),
                        hidden,
                    ))?;
                }
                (Op::Collective(Collective::LogitSlice), _) => {
                    exchange_gather(
                        &shared,
                        self.state.scratch.logits,
                        self.state.scratch.logits_full,
                        plan.rank_vocab(),
                    )?;
                }
                (Op::Collective(_), _) => {}
                (Op::LogitSoftcap, _) => {
                    self.rms(&shared, "raw_logits", self.state.scratch.logits, 4)?;
                    // The gathered row is the whole padded vocabulary, so the released trim is
                    // against that width and not against one rank's slice of it.
                    let padded = plan.rank_vocab() * shared.exchange.world();
                    let unpadded = config.unpadded_vocab_size.min(padded).max(1);
                    run(shared.launch(
                        shared.kernels.logit_softcap,
                        (1, 1, 1),
                        (ELEMENT_BLOCK, 1, 1),
                        &[
                            self.state.scratch.logits_full.0,
                            self.state.scratch.logits_bf16.0,
                            1,
                            // The gathered row is the whole padded vocabulary, which is what
                            // the softcap trims to the released 200,058.
                            (plan.rank_vocab() * shared.exchange.world()) as u64,
                            unpadded as u64,
                            bits(config.final_logit_softcap.unwrap_or(0.0) as f32),
                        ],
                    ))?;
                }
                (op, target) => {
                    return Err(Self::fail(format!(
                        "the schedule's epilogue {op:?} at {target:?} has no device dispatch"
                    )));
                }
            }
        }

        self.state.position += 1;
        shared.cuda.synchronize().map_err(Self::fail)?;
        // A doorbell round that was given up leaves its fold on whatever was in the slots; the
        // logits are not read back over one.
        shared.exchange.check_device().map_err(Self::fail)?;
        let padded = plan.rank_vocab() * shared.exchange.world();
        let unpadded = config.unpadded_vocab_size.min(padded).max(1);
        let mut bytes = vec![0u8; unpadded * 2];
        shared
            .cuda
            .copy_dtoh(&mut bytes, &self.state.scratch.logits_bf16)
            .map_err(Self::fail)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
            .collect())
    }

    /// The routed band: each selected expert's three launches, then the weighted mix.
    ///
    /// The band is a loop over the experts the router selected, not a step a expert: the plan
    /// records the three launches once and this runs them for every selected expert the rank
    /// owns.
    fn run_band(
        &mut self,
        shared: &Shared,
        layer: usize,
        gate_up: &Step,
        activation: &Step,
        down: &Step,
    ) -> Result<(), BackendError> {
        if shared.plan.layers()[layer].experts != ExpertBody::Bf16 {
            return self.run_device_band(shared, layer, gate_up, down);
        }
        let config = &shared.config;
        let schedule = &shared.plan.layers()[layer];
        let top_k = config.num_experts_per_tok;
        let intermediate = shared.plan.expert_intermediate();
        let hidden = config.hidden_size;
        let mut index_bytes = vec![0u8; top_k * std::mem::size_of::<i32>()];
        let mut share_bytes = vec![0u8; top_k * std::mem::size_of::<f32>()];
        shared
            .cuda
            .copy_dtoh(&mut index_bytes, &self.state.scratch.indices)
            .map_err(Self::fail)?;
        shared
            .cuda
            .copy_dtoh(&mut share_bytes, &self.state.scratch.shares)
            .map_err(Self::fail)?;
        let selected: Vec<usize> = index_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word).max(0) as usize)
            .collect();
        let shares: Vec<f32> = share_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let scales = shared.experts[layer].as_ref();
        let input = self.state.scratch.normed;
        let band = self.state.scratch.band;
        let activated = self.state.scratch.activation;
        let partials = self.state.scratch.partials;
        let mut scale_bytes = vec![0u8; top_k * 4];
        let gate_up_codes = self.weight(gate_up.weights[0]);
        let gate_up_scales = gate_up.weights.get(1).map(|index| self.weight(*index));
        let down_codes = self.weight(down.weights[0]);
        let down_scales = down.weights.get(1).map(|index| self.weight(*index));
        // The router is replicated and selects global experts; each rank holds one contiguous run
        // of them. A slot outside the run is not this rank's to compute - its share stays zero, so
        // the mix contributes nothing and the all-reduce that follows is the whole exchange.
        let (first, owned) = shared.plan.expert_range();
        for (slot, expert) in selected.iter().copied().enumerate() {
            if expert >= config.n_routed_experts || expert < first || expert >= first + owned {
                continue;
            }
            // The rank's own weights and per-expert scales are indexed from zero, not from the
            // released expert number.
            let local = expert - first;
            let gate_up_offset = local * codes_bytes(2 * intermediate, hidden, schedule.experts);
            let destination = band.offset(slot * 2 * intermediate * std::mem::size_of::<f32>());
            match schedule.experts {
                ExpertBody::Bf16 => {
                    shared
                        .gemv(
                            gate_up_codes.offset(gate_up_offset),
                            input,
                            destination,
                            2 * intermediate,
                            hidden,
                        )
                        .map_err(Self::fail)?;
                }
                ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                    let scale_offset =
                        local * scales_bytes(2 * intermediate, hidden, schedule.experts);
                    let scale_pointer =
                        gate_up_scales.map_or(Deviceptr(0), |pointer| pointer.offset(scale_offset));
                    let (entry, last) = match schedule.experts {
                        ExpertBody::NvFp4 => (
                            shared.kernels.expert_nvfp4,
                            bits(scales.map_or(1.0, |scale| scale.gate_up[local])),
                        ),
                        _ => (shared.kernels.expert_mxfp8, MXFP8_BLOCK as u64),
                    };
                    shared
                        .launch(
                            entry,
                            (2 * intermediate as u32, 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                gate_up_codes.offset(gate_up_offset).0,
                                scale_pointer.0,
                                input.0,
                                destination.0,
                                2 * intermediate as u64,
                                hidden as u64,
                                last,
                                // One row, one expert: the launch's own row and band slot.
                                0,
                            ],
                        )
                        .map_err(Self::fail)?;
                }
            }
            if trace_on() && layer == 0 && self.state.position == 0 {
                eprintln!("[inkling] row gate_up slot {slot}");
                self.rms(shared, "row gate_up", destination, 2 * intermediate)?;
            }
            let activation_out = activated.offset(slot * intermediate * 4);
            shared
                .launch(
                    shared.kernels.swiglu,
                    (flat(intermediate), 1, 1),
                    (ELEMENT_BLOCK, 1, 1),
                    &[
                        destination.0,
                        activation_out.0,
                        intermediate as u64,
                        bits(config.mlp_gate_softcap as f32),
                        bits(config.mlp_up_softcap as f32),
                    ],
                )
                .map_err(Self::fail)?;
            let down_offset = local * codes_bytes(hidden, intermediate, schedule.experts);
            let partial = partials.offset(slot * hidden * 4);
            match schedule.experts {
                ExpertBody::Bf16 => {
                    shared
                        .gemv(
                            down_codes.offset(down_offset),
                            activation_out,
                            partial,
                            hidden,
                            intermediate,
                        )
                        .map_err(Self::fail)?;
                }
                ExpertBody::NvFp4 | ExpertBody::MxFp8 => {
                    let scale_offset = local * scales_bytes(hidden, intermediate, schedule.experts);
                    let scale_pointer =
                        down_scales.map_or(Deviceptr(0), |pointer| pointer.offset(scale_offset));
                    let (entry, last) = match schedule.experts {
                        ExpertBody::NvFp4 => (
                            shared.kernels.expert_nvfp4,
                            bits(scales.map_or(1.0, |scale| scale.down[local])),
                        ),
                        _ => (shared.kernels.expert_mxfp8, MXFP8_BLOCK as u64),
                    };
                    shared
                        .launch(
                            entry,
                            (hidden as u32, 1, 1),
                            (ELEMENT_BLOCK, 1, 1),
                            &[
                                down_codes.offset(down_offset).0,
                                scale_pointer.0,
                                activation_out.0,
                                partial.0,
                                hidden as u64,
                                intermediate as u64,
                                last,
                                // One row, one expert: the launch's own row and band slot.
                                0,
                            ],
                        )
                        .map_err(Self::fail)?;
                }
            }
            if trace_on() && layer == 0 && self.state.position == 0 {
                self.rms(shared, "row swiglu", activation_out, intermediate)?;
                self.rms(shared, "row down", partial, hidden)?;
            }
            // The share this expert's output carries into the mix, which is the router's weight
            // times whatever per-expert scale the stored format puts on its own output.
            let share = shares[slot];
            scale_bytes[slot * 4..slot * 4 + 4].copy_from_slice(&share.to_le_bytes());
        }
        let _ = activation;
        shared
            .cuda
            .copy_htod(&self.state.scratch.scales, &scale_bytes)
            .map_err(Self::fail)?;
        shared
            .launch(
                shared.kernels.expert_mix,
                (flat(hidden), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    self.state.scratch.mixed.0,
                    partials.0,
                    self.state.scratch.scales.0,
                    1,
                    top_k as u64,
                    hidden as u64,
                ],
            )
            .map_err(Self::fail)?;
        if trace_on() && layer == 0 {
            eprintln!("[inkling] row router layer 0 slots {selected:?} shares {shares:?}");
            self.rms(shared, "row band", self.state.scratch.mixed, hidden)?;
        }
        Ok(())
    }
    /// The routed band of a quantised layer with the router's choice left on the device.
    ///
    /// Every launch covers the band's top-k slots at once (grid y), and a block whose slot names
    /// an expert another rank holds exits: [ExpertBody::NvFp4] and [ExpertBody::MxFp8] read the
    /// selection through `indices`, the NVFP4 scale_2 through the layer's device vector, and the
    /// mix's shares come from inkling_band_shares. Each row is the same body the host-steered
    /// loop launched one expert at a time, so the band's bits are that loop's; what is gone is the
    /// readback of the selection that made every layer wait for the host.
    fn run_device_band(
        &mut self,
        shared: &Shared,
        layer: usize,
        gate_up: &Step,
        down: &Step,
    ) -> Result<(), BackendError> {
        let config = &shared.config;
        let schedule = &shared.plan.layers()[layer];
        let top_k = config.num_experts_per_tok;
        let intermediate = shared.plan.expert_intermediate();
        let hidden = config.hidden_size;
        let (first, owned) = shared.plan.expert_range();
        let indices = self.state.scratch.indices;
        let band = self.state.scratch.band;
        let activated = self.state.scratch.activation;
        let partials = self.state.scratch.partials;
        let weight = |step: &Step, at: usize| {
            step.weights
                .get(at)
                .map_or(Deviceptr(0), |index| self.weight(*index))
        };
        let scale_2 = shared.experts[layer].as_ref();
        let projection = |step: &Step,
                          input: Deviceptr,
                          out: Deviceptr,
                          rows: usize,
                          columns: usize,
                          per_slot: bool,
                          factor: Option<Deviceptr>|
         -> Result<(), BackendError> {
            let (entry, tail) = match schedule.experts {
                ExpertBody::NvFp4 => (
                    shared.kernels.expert_nvfp4_band,
                    factor
                        .ok_or_else(|| {
                            Self::fail(format!(
                                "layer {layer}: an NVFP4 band has no scale_2 vector"
                            ))
                        })?
                        .0,
                ),
                _ => (shared.kernels.expert_mxfp8_band, MXFP8_BLOCK as u64),
            };
            if !columns.is_multiple_of(BAND_RUN) || !MXFP8_BLOCK.is_multiple_of(BAND_RUN) {
                return Err(Self::fail(format!(
                    "layer {layer}: a band row of {columns} columns (MXFP8 block \
                     {MXFP8_BLOCK}) is not whole runs of the {BAND_RUN} columns a thread reads"
                )));
            }
            shared
                .launch(
                    entry,
                    (rows.div_ceil(BAND_ROWS) as u32, top_k as u32, 1),
                    (ELEMENT_BLOCK, 1, 1),
                    &[
                        weight(step, 0).0,
                        weight(step, 1).0,
                        input.0,
                        out.0,
                        rows as u64,
                        columns as u64,
                        tail,
                        indices.0,
                        first as u64,
                        owned as u64,
                        u64::from(per_slot),
                    ],
                )
                .map_err(Self::fail)
        };
        projection(
            gate_up,
            self.state.scratch.normed,
            band,
            2 * intermediate,
            hidden,
            false,
            scale_2.map(|scale| scale.device_gate_up),
        )?;
        // Every slot's activation, owned or not: a slot another rank holds reads a stale band
        // row, and its share below is zero.
        shared
            .launch(
                shared.kernels.swiglu,
                (flat(intermediate), top_k as u32, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    band.0,
                    activated.0,
                    intermediate as u64,
                    bits(config.mlp_gate_softcap as f32),
                    bits(config.mlp_up_softcap as f32),
                ],
            )
            .map_err(Self::fail)?;
        projection(
            down,
            activated,
            partials,
            hidden,
            intermediate,
            true,
            scale_2.map(|scale| scale.device_down),
        )?;
        shared
            .launch(
                shared.kernels.band_shares,
                (1, 1, 1),
                (SHARES_BLOCK, 1, 1),
                &[
                    indices.0,
                    self.state.scratch.shares.0,
                    self.state.scratch.scales.0,
                    top_k as u64,
                    first as u64,
                    owned as u64,
                    // A decode step's experts take the matvec, which reads no plan.
                    0,
                ],
            )
            .map_err(Self::fail)?;
        shared
            .launch(
                shared.kernels.expert_mix,
                (flat(hidden), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    self.state.scratch.mixed.0,
                    partials.0,
                    self.state.scratch.scales.0,
                    1,
                    top_k as u64,
                    hidden as u64,
                ],
            )
            .map_err(Self::fail)?;
        if trace_on() && layer == 0 {
            self.rms(shared, "row band", self.state.scratch.mixed, hidden)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionWalk, ChunkWidths, DEFAULT_PREFILL_CHUNK, LONG_PREFILL_CHUNK, LONG_PREFILL_FROM,
        PrefillWalk, mma_attention_shared,
    };
    use crate::config::Inkling2Config;

    #[test]
    fn the_attention_walk_follows_the_prompt_walk_unless_one_is_named() {
        // The released attention shape: head_dim 128, a 16-wide relative state, six query heads
        // a kv head on a rank of four.
        let mut released = Inkling2Config::tiny();
        released.head_dim = 128;
        released.d_rel = 16;
        assert_eq!(mma_attention_shared(6), 85_760);
        let shared = Some(85_760);
        let choose = |value, config, prefill| AttentionWalk::choose(value, config, 6, prefill);
        assert_eq!(
            choose(None, &released, PrefillWalk::Tensor),
            Ok((AttentionWalk::Fast, shared))
        );
        assert_eq!(
            choose(Some(""), &released, PrefillWalk::Tensor),
            Ok((AttentionWalk::Fast, shared))
        );
        // The exact prompt walks keep the exact attention, so chunk stays the decode step's bits.
        for prefill in [PrefillWalk::Chunk, PrefillWalk::Rows, PrefillWalk::Gemm] {
            assert_eq!(
                choose(None, &released, prefill),
                Ok((AttentionWalk::Split, None))
            );
        }
        // A named walk is the walk, whatever the prompt's.
        assert_eq!(
            choose(Some("fast"), &released, PrefillWalk::Chunk),
            Ok((AttentionWalk::Fast, shared))
        );
        assert_eq!(
            choose(Some("check"), &released, PrefillWalk::Tensor),
            Ok((AttentionWalk::Check, shared))
        );
        assert_eq!(
            choose(Some("split"), &released, PrefillWalk::Tensor),
            Ok((AttentionWalk::Split, None))
        );
        assert_eq!(
            choose(Some("serial"), &released, PrefillWalk::Tensor),
            Ok((AttentionWalk::Serial, None))
        );
        let error = choose(Some("mma"), &released, PrefillWalk::Tensor).expect_err("unknown");
        assert!(error.contains("is not an attention walk"), "{error}");

        // A shape the tensor-core walk is not built for: unset stands down, named is refused.
        let tiny = Inkling2Config::tiny();
        assert_eq!(
            choose(None, &tiny, PrefillWalk::Tensor),
            Ok((AttentionWalk::Split, None))
        );
        for named in ["fast", "check"] {
            let error = choose(Some(named), &tiny, PrefillWalk::Tensor).expect_err("no fit");
            assert!(error.contains("head_dim 128"), "{error}");
            assert!(error.contains("=split for the exact walk"), "{error}");
        }
        let error = AttentionWalk::choose(Some("fast"), &released, 9, PrefillWalk::Tensor)
            .expect_err("nine heads a kv head");
        assert!(error.contains("this rank holds 9"), "{error}");
    }

    #[test]
    fn a_long_prompt_widens_its_chunks_until_its_tail() {
        let widths = ChunkWidths {
            base: DEFAULT_PREFILL_CHUNK,
            long: LONG_PREFILL_CHUNK,
            long_from: LONG_PREFILL_FROM,
        };
        assert_eq!(widths.widest(), LONG_PREFILL_CHUNK);
        let walk = |mut left: usize| {
            let mut chunks = Vec::new();
            while left > 0 {
                let take = widths.next(left);
                chunks.push(take);
                left -= take;
            }
            chunks
        };
        assert_eq!(walk(100), vec![100]);
        assert_eq!(walk(10_000), vec![4096, 4096, 1808]);
        assert_eq!(walk(16_383), vec![4096, 4096, 4096, 4095]);
        assert_eq!(walk(16_384), vec![8192, 4096, 4096]);
        assert_eq!(walk(40_000), vec![8192, 8192, 8192, 4096, 4096, 4096, 3136]);
    }

    #[test]
    fn a_fixed_width_never_widens() {
        let fixed = ChunkWidths {
            base: 512,
            long: 512,
            long_from: usize::MAX,
        };
        assert_eq!(fixed.widest(), 512);
        assert_eq!(fixed.next(1_000_000), 512);
        assert_eq!(fixed.next(7), 7);
    }
}
