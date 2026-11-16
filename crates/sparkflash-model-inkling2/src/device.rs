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
    SideStream, StreamEvent,
};
use sparkflash_kernels::images::{
    glm_prefill_cubin, inkling_cubin, moe_plan_cubin, qwen4_exp_cubin, qwen4_prefill_cubin,
};
use sparkflash_kernels::lane_memory::{ChunkPool, LaneShape, LaneSpan, Layout};
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_model::vision::{ImageSpan, VisionImage};
use sparkflash_model::{
    Capabilities, DepthPolicy, ModelDescriptor, Speculator, StepCost, TpPlan, lookup_draft,
};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_runtime::{ExpertUnionWall, FusedArm, Opened, RoundArm, StepWall, run_round};
use sparkflash_weights::WeightSource;
use sparkflash_weights::quant::bf16_to_f32;

use crate::collective::{Exchange, Solo};
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device_plan::{
    BandOp, Collective, ConvSlot, ExpertBody, InklingDevicePlan, LayerSchedule, NormRole, Op, Pass,
    Projection, ResidualInput, Step, Target,
};
use crate::forward::trace_on;
use crate::host::choose;
use crate::tensor::{ExpertSplit, InklingTensorPlan};
use crate::vision::SpanCheck;
use sparkflash_ops::sampler::{Sampler, SamplerMark};

mod chunk;
mod draft;
mod packed;
mod tower;

use chunk::{
    BandOut, Injection, Rows, Segment, band_exchange, band_mix, exchange_gather, exchange_sum,
    walk_rows,
};
pub use draft::DRAFT_FILE;
use draft::{DraftModel, DraftState, FINAL_TAP, TapSink};
use packed::{DenseForm, DenseWeights, pack_dense};
pub use tower::InklingVisionTower;

/// The backend name the diagnostics use.
pub const DEVICE: &str = "inkling_device";

/// The threads a block of the token image's dense GEMV or RMSNorm runs.
const TOKEN_BLOCK: u32 = 256;
/// The threads the Inkling elementwise bodies run.
const ELEMENT_BLOCK: u32 = 256;
/// The rows a block-attention residual reads at most: the summary stack (INKLING_MAX_BLOCKS)
/// and the current stream.
const RESIDUAL_ROWS: usize = 17;
/// The (row, query) pairs at or under which the residual takes its two-launch form, every
/// stack row in a block of its own: a decode row's one or three queries fill a block or three
/// of the one-block form, which then walks its rows one after another. A chunk's pairs fill
/// the device already.
const RESIDUAL_SPLIT_PAIRS: usize = 8;
/// The f32s of a sequence row's share of [Scratch::residual_stats]: the two-launch form's
/// (inverse, score) for up to three queries over [RESIDUAL_ROWS] rows. The chained form
/// (`inkling_block_residual_chain_*`, [RESIDUAL_CHAIN_ENV]) carries its scores through the same
/// rows, a walk taking one form or the other.
const RESIDUAL_STATS_ROW: usize = 3 * RESIDUAL_ROWS * 2;
/// The summary rows the chained form holds (`INKLING_CHAIN_SUMMARIES`): a thread keeps the next
/// call's dots a summary row in registers through its whole pass.
const RESIDUAL_CHAIN_SUMMARIES: usize = 8;
/// The chained form's entries, by a call's query count and the next call's:
/// `inkling_block_residual_chain_q<queries>_n<next>`.
const RESIDUAL_CHAIN_PAIRS: [(usize, usize, &str); 6] = [
    (1, 0, "inkling_block_residual_chain_q1_n0"),
    (1, 1, "inkling_block_residual_chain_q1_n1"),
    (1, 3, "inkling_block_residual_chain_q1_n3"),
    (3, 0, "inkling_block_residual_chain_q3_n0"),
    (3, 1, "inkling_block_residual_chain_q3_n1"),
    (3, 3, "inkling_block_residual_chain_q3_n3"),
];
/// The threads the serial attention body runs: one warp a head.
const ATTENTION_BLOCK: u32 = 32;
/// The threads a block of the split walk's key-scale and score launches runs
/// (32 * INKLING_SCORE_WARPS), and the keys one block covers (INKLING_SCORE_RUN a warp).
const SCORE_BLOCK: u32 = 128;
const SCORE_KEYS: usize = 16;
/// The most blocks a row's score launch takes along its keys: inkling_attn_scores strides its
/// warps over the pass, so past this a warp walks several runs with the next run's loads in
/// flight instead of the grid growing. 96 (four blocks an SM with two KV heads) measured best at
/// 1k, 5k, 93k and 262k keys; the grid changes which warp scores a key, never its bits.
const SCORE_GRID: u32 = 96;

/// A score launch's blocks a row along its keys: [SCORE_GRID] shared by the rows, so a verify's
/// rows are resident together and walk the same keys at once (block b of every row starts at the
/// same key and strides the same), and all but the first find each key in L2 instead of DRAM.
fn score_grid(rows: usize) -> u32 {
    SCORE_GRID
        .div_ceil(u32::try_from(rows.max(1)).unwrap_or(u32::MAX))
        .max(1)
}
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
/// The scored partial (inkling_attn_scored_part): its threads (32 * INKLING_SCORED_WARPS), the
/// widest group it holds (INKLING_SCORED_GROUPS), and the runs a row's pass needs before it
/// takes the pass. Below that the pass's runs are too few blocks to beat the scores spread over
/// [SCORE_GRID] blocks and a part launch: it measured 7% slower at 16k keys and 4% faster at 32k,
/// 7-8% at 93k-262k. Either way the bits are the same.
const SCORED_THREADS: u32 = 256;
const SCORED_GROUPS: usize = 6;
const SCORED_RUNS: usize = 128;
/// A sliding layer's walk in one launch (inkling_attn_window): its cluster's blocks
/// (INKLING_WINDOW_RUNS, a run of [PART_KEYS] each, so the widest window it holds), threads
/// (32 * INKLING_WINDOW_WARPS), widest group (INKLING_WINDOW_GROUPS), head width
/// (INKLING_MAX_HEAD_DIM) and dynamic shared bytes (sizeof(InklingWindowShared<6>): the run's
/// staged values, its scores, scales and reductions, and its partial).
const WINDOW_RUNS: usize = 4;
const WINDOW_THREADS: u32 = 512;
const WINDOW_GROUPS: usize = 6;
const WINDOW_HEAD_DIM: usize = 128;
const WINDOW_SHARED: u32 = 75_920;
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

/// The switch for the MXFP8 layers' routed experts under [PrefillWalk::Tensor] ([Mxfp8Walk]):
/// `mma` (the default there) or `pairs`. Under the other walks they take the pair walk, and `mma`
/// named there is refused, as is an unknown value.
pub const MXFP8_ENV: &str = "SPARKFLASH_INKLING_MXFP8";

/// How a prompt chunk's MXFP8 experts (Inkling-2-Small's layers 40-47) are walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp8Walk {
    /// The grouped MMA the NVFP4 layers take under the tensor walk, over MXFP8 weights
    /// (q4p_moe_gate_up_capped_mx8 and q4p_moe_down_mx8): each weight exact in BF16, the
    /// activation the tensor walk's BF16 head and remainder, accumulated in f32 in the tile's
    /// order. Its table is built at open after every MXFP8 scale code is checked to keep the
    /// weights exact ([admit_mxfp8_scales]).
    Mma,
    /// The exact pair walk, one launch an expert over the rows that chose it.
    Pairs,
}

impl Mxfp8Walk {
    /// The walk [MXFP8_ENV] names under `walk`: `Mma` under [PrefillWalk::Tensor] when it is
    /// unset, `Pairs` under the others.
    pub fn from_env(walk: PrefillWalk) -> Result<Self, String> {
        Self::parse(std::env::var(MXFP8_ENV).ok().as_deref(), walk)
    }

    fn parse(value: Option<&str>, walk: PrefillWalk) -> Result<Self, String> {
        match (value, walk) {
            (None | Some("" | "mma"), PrefillWalk::Tensor) => Ok(Self::Mma),
            (None | Some(""), _) | (Some("pairs"), _) => Ok(Self::Pairs),
            (Some("mma"), other) => Err(format!(
                "{MXFP8_ENV}=mma runs under {PREFILL_ENV}=tensor, and the walk is {other:?}, \
                 whose experts keep their own arithmetic; unset {MXFP8_ENV} or set it to pairs"
            )),
            (Some(other), _) => Err(format!(
                "{MXFP8_ENV}={other} is not an MXFP8 expert walk: use mma (the tensor walk's \
                 default) or pairs (the exact pair walk)"
            )),
        }
    }
}

/// The switch for the activation the tensor walk's routed-expert MMA reads ([ActivationPlanes]):
/// unset or `both` (the default) keeps its bf16 head and remainder planes, `head` the head alone.
/// `head` is bit-changing and opt-in; it is refused where no MMA band runs (any other walk), as is
/// an unknown value.
pub const PLANES_ENV: &str = "SPARKFLASH_INKLING_PLANES";

/// The activation operand of the tensor walk's routed-expert MMA: the expert input into gate_up
/// and the SwiGLU rows into down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationPlanes {
    /// The bf16 head and the bf16 remainder of each f32 value, 16 significant bits, both through
    /// the same weight fragments into one f32 accumulator.
    Both,
    /// The bf16 head alone (the q4p `_head` passes): half the MMAs and half the activation bytes
    /// gathered under the row split, and 8 significant bits.
    Head,
}

impl ActivationPlanes {
    /// The operand [PLANES_ENV] names under `walk`.
    pub fn from_env(walk: PrefillWalk) -> Result<Self, String> {
        Self::parse(std::env::var(PLANES_ENV).ok().as_deref(), walk)
    }

    fn parse(value: Option<&str>, walk: PrefillWalk) -> Result<Self, String> {
        match (value, walk) {
            (None | Some("" | "both"), _) => Ok(Self::Both),
            (Some("head"), PrefillWalk::Tensor) => Ok(Self::Head),
            (Some("head"), other) => Err(format!(
                "{PLANES_ENV}=head is the routed-expert MMA's operand under {PREFILL_ENV}=tensor, \
                 and the walk is {other:?}, which runs no MMA band; unset {PLANES_ENV}"
            )),
            (Some(other), _) => Err(format!(
                "{PLANES_ENV}={other} is not an activation operand: use both (the default) or \
                 head (bit-changing, the bf16 head alone)"
            )),
        }
    }

    /// The bf16 planes an activation row carries.
    pub(crate) fn count(self) -> usize {
        match self {
            Self::Both => 2,
            Self::Head => 1,
        }
    }

    /// A pass's kernel: `both`'s name, or `head`'s.
    fn pick(self, both: &'static str, head: &'static str) -> &'static str {
        match self {
            Self::Both => both,
            Self::Head => head,
        }
    }
}

/// The switch for the row split's two-halves band under [PrefillWalk::Tensor] (BandOverlap):
/// unset or `on` runs a chunk's band as two halves of its rows with each half's gathers on a side
/// stream, and its attention's output (under [AttentionWalk::Fast]) as two halves with each half's
/// delta summed there; `off` runs both whole with their collectives on the run's stream. The bits
/// are the same either way. `on` named where the halves cannot run (another split or walk, one rank, no device leg) is
/// refused, as is an unknown value.
pub const OVERLAP_ENV: &str = "SPARKFLASH_INKLING_OVERLAP";

/// What [OVERLAP_ENV] asks for: `None` (unset or empty, the halves wherever they run),
/// `Some(true)` (`on`, refused where they cannot) or `Some(false)` (`off`).
fn overlap_request(value: Option<&str>) -> Result<Option<bool>, String> {
    match value {
        None | Some("") => Ok(None),
        Some("on") => Ok(Some(true)),
        Some("off") => Ok(Some(false)),
        Some(other) => Err(format!(
            "{OVERLAP_ENV}={other} is not a band overlap: use on (the default where the row \
             split's tensor walk runs) or off (the whole band, its gathers on the run's stream)"
        )),
    }
}

/// The switch for the prompt-boundary checkpoint. Unset, empty or `on` keeps a copy, at the end of
/// every prompt, of the state a sequence's cache rows do not hold: each layer's convolution
/// histories, and the drafter's Markov feature. A next prompt that keeps that prompt but parts from
/// the reply after it then rewinds to the prompt's end rather than starting over, as a chat client
/// that renders the reply again with other ids makes it. `off` keeps no copy. Anything else is
/// refused.
///
/// A rewind reads the rows its cache holds before the boundary, so the reply and the drafts after
/// it must not have written over them: a sliding ring holds its window and a widest chunk, and the
/// drafter's local rings are opened as wide under `on` (window + widest chunk - 1 rows of each
/// local layer's keys and values, 32 KiB a row on Inkling-2's drafter).
pub const REWIND_ENV: &str = "SPARKFLASH_INKLING_REWIND";

/// A switch that is on unless it is named off: unset, empty or `on` is `true`, `off` is `false`,
/// and anything else is refused with what each of the two does.
fn default_on(name: &str, value: Option<&str>, on: &str, off: &str) -> Result<bool, String> {
    match value {
        None | Some("" | "on") => Ok(true),
        Some("off") => Ok(false),
        Some(other) => Err(format!(
            "{name}={other} is not a switch: use on (the default: {on}) or off ({off})"
        )),
    }
}

/// What [REWIND_ENV] asks for: `true` keeps the checkpoint.
fn rewind_request(value: Option<&str>) -> Result<bool, String> {
    default_on(
        REWIND_ENV,
        value,
        "a prompt that keeps the last prompt and parts from its reply rewinds to the prompt's end",
        "it starts over",
    )
}

/// The switch for the chunk walk's block residuals (`inkling_block_residual_chain_*`). Unset, empty
/// or `on` runs every residual of a walk over more than [RESIDUAL_SPLIT_PAIRS] rows as one block a
/// row for all of the call's queries, its mix pass also scoring the next call's queries against
/// the summary rows, so a call reads its stack once. `off` runs each call as the one-block form,
/// which reads the stack twice for every query. The bits are the same either way. A model whose
/// stack outgrows the chained form ([RESIDUAL_CHAIN_SUMMARIES] summary rows) stands down to the
/// one-block form when the switch is unset and is refused under `on`. Anything else is refused.
pub const RESIDUAL_CHAIN_ENV: &str = "SPARKFLASH_INKLING_RESIDUAL_CHAIN";

/// What [RESIDUAL_CHAIN_ENV] asks for over a stack of at most `summaries` summary rows: `true`
/// chains the residuals.
fn residual_chain_request(value: Option<&str>, summaries: usize) -> Result<bool, String> {
    let on = default_on(
        RESIDUAL_CHAIN_ENV,
        value,
        "a walk's residuals each read their summary stack once",
        "each call reads its stack twice for every query",
    )?;
    if !on || summaries <= RESIDUAL_CHAIN_SUMMARIES {
        return Ok(on);
    }
    if value == Some("on") {
        return Err(format!(
            "{RESIDUAL_CHAIN_ENV}=on: the chained block residual holds \
             {RESIDUAL_CHAIN_SUMMARIES} summary rows and this model's stack grows to {summaries}; \
             unset it or set it off"
        ));
    }
    Ok(false)
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

/// Where a chunk's tensor-core attention splits its rows (a multiple of [MMA_Q]), and what runs
/// over each half's rows once that half's MMA is queued, called as `(first_row, rows)`
/// (chunk::attention_half).
type AttentionHalves<'h> = (usize, &'h mut dyn FnMut(usize, usize) -> Result<(), String>);

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
    /// The same two walks over packed matrices ([packed]), and the form's packer and unpacker.
    gemv_packed: Entry,
    gemv4_packed: Entry,
    pack: Entry,
    unpack: Entry,
    block_residual: Entry,
    /// The residual's two-launch form: every stack row's statistics in a block of its own,
    /// then the softmax and the mix.
    block_residual_stats: Entry,
    block_residual_mix: Entry,
    /// The chunk walk's chained form: one block a row for every query, the stack read once; one
    /// entry a pair of [RESIDUAL_CHAIN_PAIRS], in its order.
    block_residual_chain: Vec<Entry>,
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
    /// A band of a few rows over moe_plan's items: the pair walk, an expert read once for every
    /// slot that picked it, with the plan made on the device.
    expert_nvfp4_items: Entry,
    expert_mxfp8_items: Entry,
    /// The decode band's mix shares: the router's weight for an owned slot, zero otherwise.
    band_shares: Entry,
    /// The device leg's sum: this rank's partial and every peer's slot, ascending rank from zero.
    sum_slots: Entry,
    swiglu: Entry,
    expert_mix: Entry,
    /// The row expert split's gathered runs side by side: [world][rows][piece] to
    /// [rows][world * piece].
    join_pieces: Entry,
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
    /// The same two passes over the MXFP8 layers' E4M3 bytes and ue8m0 codes ([Mxfp8Walk::Mma]).
    moe_gate_up_mx8: Entry,
    moe_down_mx8: Entry,
    /// The two down passes over the row split's gathered activation runs, unjoined.
    moe_down_runs: Entry,
    moe_down_runs_mx8: Entry,
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
    /// then the runs merged in order. `attention_scored` is a long pass's scores and parts in
    /// one launch.
    attention_part: Entry,
    attention_scored: Entry,
    attention_merge: Entry,
    /// A sliding layer's split-key walk in one launch: a cluster a (KV head, row), a block a run.
    attention_window: Entry,
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
    /// Every one of a verify's histories back to the decode layout from its kept rows, in one
    /// launch ([InklingDeviceSession::settle]).
    conv_restore: Entry,
}

impl Kernels {
    /// `mma_shared` is the tensor-core attention's dynamic shared memory, or `None` when the walk
    /// does not run; `planes` picks the routed-expert MMA's passes.
    fn load(
        cuda: &Cuda,
        mma_shared: Option<u32>,
        planes: ActivationPlanes,
    ) -> Result<Self, String> {
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
        let (window_module, window) =
            cuda.load_named_shared(inkling, "inkling_attn_window", WINDOW_SHARED)?;
        modules.push(window_module);
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
            moe_gate_up: load(
                experts,
                planes.pick("q4p_moe_gate_up_capped", "q4p_moe_gate_up_capped_head"),
            )?,
            moe_down: load(experts, planes.pick("q4p_moe_down", "q4p_moe_down_head"))?,
            moe_gate_up_mx8: load(
                experts,
                planes.pick(
                    "q4p_moe_gate_up_capped_mx8",
                    "q4p_moe_gate_up_capped_mx8_head",
                ),
            )?,
            moe_down_mx8: load(
                experts,
                planes.pick("q4p_moe_down_mx8", "q4p_moe_down_mx8_head"),
            )?,
            moe_down_runs: load(
                experts,
                planes.pick("q4p_moe_down_runs", "q4p_moe_down_runs_head"),
            )?,
            moe_down_runs_mx8: load(
                experts,
                planes.pick("q4p_moe_down_runs_mx8", "q4p_moe_down_runs_mx8_head"),
            )?,
            moe_plan: load(moe_plan, "moe_plan")?,
            embed: load(token, "q4e_embed")?,
            rmsnorm: load(token, "q4e_rmsnorm")?,
            gemv: load(token, "q4e_gemv_bf16")?,
            gemv4: load(token, "q4e_gemv4_bf16")?,
            gemv_packed: load(token, "q4e_gemv_packed")?,
            gemv4_packed: load(token, "q4e_gemv4_packed")?,
            pack: load(token, "q4e_pack_bf16")?,
            unpack: load(token, "q4e_unpack_bf16")?,
            block_residual: load(inkling, "inkling_block_residual")?,
            block_residual_stats: load(inkling, "inkling_block_residual_stats")?,
            block_residual_mix: load(inkling, "inkling_block_residual_mix")?,
            block_residual_chain: RESIDUAL_CHAIN_PAIRS
                .iter()
                .map(|&(_, _, name)| load(inkling, name))
                .collect::<Result<_, _>>()?,
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
            expert_nvfp4_items: load(inkling, "inkling_nvfp4_expert_items")?,
            expert_mxfp8_items: load(inkling, "inkling_mxfp8_expert_items")?,
            band_shares: load(inkling, "inkling_band_shares")?,
            sum_slots: load(token, "q4e_sum_slots")?,
            swiglu: load(inkling, "inkling_softcapped_swiglu")?,
            expert_mix: load(inkling, "inkling_expert_mix")?,
            join_pieces: load(inkling, "inkling_join_pieces")?,
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
            attention_scored: load(inkling, "inkling_attn_scored_part")?,
            attention_merge: load(inkling, "inkling_attn_merge")?,
            attention_window: Entry { kernel: window },
            attention_prepare: load(inkling, "inkling_attn_prepare")?,
            attention_mma,
            short_conv_rows: load(inkling, "inkling_short_conv")?,
            short_conv_dynamic_rows: load(inkling, "inkling_short_conv_dynamic")?,
            summary_push: load(inkling, "inkling_summary_push")?,
            conv_history: load(inkling, "inkling_conv_history_copy")?,
            conv_restore: load(inkling, "inkling_conv_history_restore")?,
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
}

/// A layer's experts as the grouped MMA (q4p_moe_gate_up_capped / q4p_moe_down, and their _mx8
/// forms) reads them: six pointers a local expert - w13's codes and block scales, the same two one
/// row on (the interleaved up rows), w2's codes and block scales - and three f32 outer scales,
/// w13's for the gate and the up and w2's for the down (ones for MXFP8, which stores none).
#[derive(Debug, Clone, Copy)]
struct MmaTable {
    table: Deviceptr,
    scales: Deviceptr,
}

/// The side stream and events of the row split's two-halves passes, made at open: the band's four
/// gathers (chunk::run_band_halves) and the attention delta's two sums (chunk::attention_half) run
/// on `side` while the run's stream computes the other half.
struct BandOverlap {
    /// At the context's greatest priority: a gather that queues behind the other half's grids
    /// holds up the peers too.
    side: SideStream,
    /// Marked on `side` once a half's activation runs are gathered.
    gathered: [StreamEvent; 2],
    /// Marked on `side` once a half's mix rows are gathered.
    mixed: [StreamEvent; 2],
    /// Marked on `side` once both halves of the attention delta are summed.
    summed: StreamEvent,
}

// SAFETY: a stream and events of the backend's context. The band's lock gives them to one chunk
// at a time, on the thread that runs it, as the run's own stream is.
unsafe impl Send for BandOverlap {}

impl BandOverlap {
    fn new(cuda: &Cuda) -> Result<Self, String> {
        let event = || cuda.stream_event();
        Ok(Self {
            side: cuda.side_stream_urgent()?,
            gathered: [event()?, event()?],
            mixed: [event()?, event()?],
            summed: event()?,
        })
    }
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
    /// The dense projections packed in place ([packed]), by their slots' pointers.
    dense: DenseWeights,
    clip: Vec<ClipScales>,
    clip_arena: Deviceptr,
    experts: Vec<Option<ExpertScale>>,
    /// Each layer's grouped-MMA table: the NVFP4 layers', and the MXFP8 layers' under
    /// [Mxfp8Walk::Mma].
    mma: Vec<Option<MmaTable>>,
    /// How a prompt chunk's MXFP8 experts are walked ([MXFP8_ENV]).
    mxfp8_walk: Mxfp8Walk,
    /// The routed-expert MMA's activation operand ([PLANES_ENV]).
    planes: ActivationPlanes,
    /// The allocation every ExpertScale's device vectors and MmaTable's tables are offsets into.
    expert_arena: Deviceptr,
    /// [top_k] ones in that arena: the routing weights the grouped MMA's down pass multiplies by.
    route_ones: Deviceptr,
    /// The row split's two-halves band ([OVERLAP_ENV]), or `None` where the band runs whole.
    overlap: Option<Mutex<BandOverlap>>,
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
    /// Whether a session keeps the prompt-boundary checkpoint ([REWIND_ENV]).
    rewind: bool,
    /// Whether a chunk walk chains its block residuals ([RESIDUAL_CHAIN_ENV]).
    residual_chain: bool,
    /// The drafter a sequence speculates with and the most drafts a round verifies
    /// ([DRAFT_ENV]); `None` steps a token at a time.
    draft: Option<(Drafter, usize)>,
    /// How many of a DSpark round's drafts to verify ([DRAFT_POLICY_ENV]): each session starts
    /// from this fresh policy and calibrates its own.
    depth_policy: Option<DepthPolicy>,
    /// The DSpark drafter's weights, once [InklingDeviceBackend::load_drafter] loaded them.
    dspark: Option<DraftModel>,
    /// The vision tower, on rank 0 when it was loaded ([InklingDeviceBackend::load_vision]): the
    /// one rank that encodes a prompt's images, whose rows the group's sum then broadcasts.
    vision: Option<Mutex<InklingVisionTower>>,
    /// The sessions' lane memory, planned when the first session opens (the drafter loads
    /// before it, and its rings are part of a lane).
    lanes: Mutex<Option<Arc<LaneMemory>>>,
}

impl Shared {
    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    /// The sessions' [LaneMemory], planned on first use.
    fn lane_memory(&self) -> Result<Arc<LaneMemory>, String> {
        let mut lanes = self
            .lanes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(memory) = lanes.as_ref() {
            return Ok(Arc::clone(memory));
        }
        let memory = Arc::new(LaneMemory::plan(self)?);
        if trace_on() || self.exchange.rank() == 0 {
            let chunk = memory.shape.granularity;
            eprintln!(
                "[inkling] rank {}: session caches back as they grow: {} MiB a lane at most, a \
                 {} MiB pool, {} KiB chunks",
                self.exchange.rank(),
                memory.shape.stride >> 20,
                (memory.pool().budget() * chunk) >> 20,
                chunk >> 10
            );
        }
        *lanes = Some(Arc::clone(&memory));
        Ok(memory)
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

    /// The bytes of each of a layer's four convolution histories, in slot order: the key's and
    /// the value's at the cache width, the attention's and the MLP's at the hidden width.
    fn conv_slot_bytes(&self) -> [usize; 4] {
        let history = self.config.conv_history().max(1);
        [
            self.kv_width(),
            self.kv_width(),
            self.hidden(),
            self.hidden(),
        ]
        .map(|width| width.max(1) * history * std::mem::size_of::<f32>())
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

    /// One dense BF16 projection of one row (a packed matrix's through the packed walk, the same
    /// bits).
    fn gemv(
        &self,
        weight: Deviceptr,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
    ) -> Result<(), String> {
        if let Some(packed) = self.dense.get(weight) {
            return self.gemv_packed(packed, input, out, rows, columns, 1, 1);
        }
        self.launch(
            self.kernels.gemv,
            (gemv_grid(rows), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[weight.0, input.0, out.0, rows as u64, columns as u64, 1],
        )
    }

    /// Up to four dense BF16 projections of `batch` rows that read the same input, as one launch.
    ///
    /// `q4e_gemv4_bf16` walks the matrices' rows stacked, and each row's sum is the one
    /// [Self::gemv_batch] computes - the same row dot and warp sum over the same input, a
    /// batch's weight rows two a warp with each row's sums its own - so a fused group is the bits
    /// of its projections launched one at a time. A slot past `parts` has no rows. `parts` holds
    /// (weight, output, rows); input is `[batch][columns]` and each output `[batch][rows]`.
    fn gemv_fused(
        &self,
        parts: &[(Deviceptr, Deviceptr, usize)],
        input: Deviceptr,
        columns: usize,
        batch: usize,
    ) -> Result<(), String> {
        if parts.is_empty() || parts.len() > GEMV_FUSED {
            return Err(format!(
                "a fused projection takes 1 to {GEMV_FUSED} matrices, not {}",
                parts.len()
            ));
        }
        if !(1..=BATCH_ROWS).contains(&batch) {
            return Err(format!(
                "a fused projection carries 1 to {BATCH_ROWS} rows, not {batch}"
            ));
        }
        // (w0..w3, x, y0..y3, r0..r3, columns, batch): an absent slot is a null matrix of no rows.
        // Packed matrices take the packed launch; a group that mixes the two forms goes out a
        // matrix at a time, which is the same bits.
        let packed: Vec<_> = parts
            .iter()
            .filter_map(|(weight, out, rows)| Some((self.dense.get(*weight)?, *out, *rows)))
            .collect();
        if packed.len() == parts.len() {
            return self.gemv4_packed(&packed, input, columns, batch);
        }
        if !packed.is_empty() {
            for (weight, out, rows) in parts {
                self.gemv_batch(*weight, input, *out, *rows, columns, batch, 1)?;
            }
            return Ok(());
        }
        let mut args = [0u64; 3 * GEMV_FUSED + 3];
        for (slot, (weight, out, rows)) in parts.iter().enumerate() {
            args[slot] = weight.0;
            args[GEMV_FUSED + 1 + slot] = out.0;
            args[2 * GEMV_FUSED + 1 + slot] = *rows as u64;
        }
        args[GEMV_FUSED] = input.0;
        args[3 * GEMV_FUSED + 1] = columns as u64;
        args[3 * GEMV_FUSED + 2] = batch as u64;
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
        if let Some(packed) = self.dense.get(weight) {
            return self.gemv_packed(packed, input, out, rows, columns, batch, groups);
        }
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
        let weight = self.bf16_rows(weight, width, inner)?;
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
    /// single-row walk passes `rows` 1 and a stride of 0. `stats` is the arena's
    /// [Scratch::residual_stats], which the two-launch form passes its rows' statistics through;
    /// both forms give the same bits.
    #[allow(clippy::too_many_arguments)]
    fn block_residual(
        &self,
        summaries: Deviceptr,
        partial: Deviceptr,
        queries: Deviceptr,
        out: Deviceptr,
        stats: Deviceptr,
        blocks: usize,
        queries_count: usize,
        rows: usize,
        summary_stride: usize,
    ) -> Result<(), String> {
        let stack = blocks + 1;
        if stack > RESIDUAL_ROWS {
            return Err(format!(
                "the block residual reads {stack} rows, more than the {RESIDUAL_ROWS} its \
                 kernels hold"
            ));
        }
        let hidden = self.hidden();
        let eps = bits(self.config.architecture.norm_epsilon as f32);
        if rows * queries_count > RESIDUAL_SPLIT_PAIRS {
            return self.launch(
                self.kernels.block_residual,
                (queries_count as u32, 1, rows as u32),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    summaries.0,
                    partial.0,
                    queries.0,
                    out.0,
                    blocks as u64,
                    hidden as u64,
                    queries_count as u64,
                    eps,
                    summary_stride as u64,
                ],
            );
        }
        self.launch(
            self.kernels.block_residual_stats,
            (stack as u32, queries_count as u32, rows as u32),
            (ELEMENT_BLOCK, 1, 1),
            &[
                summaries.0,
                partial.0,
                queries.0,
                stats.0,
                blocks as u64,
                hidden as u64,
                queries_count as u64,
                eps,
                summary_stride as u64,
            ],
        )?;
        self.launch(
            self.kernels.block_residual_mix,
            (
                hidden.div_ceil(ELEMENT_BLOCK as usize) as u32,
                queries_count as u32,
                rows as u32,
            ),
            (ELEMENT_BLOCK, 1, 1),
            &[
                summaries.0,
                partial.0,
                stats.0,
                out.0,
                blocks as u64,
                hidden as u64,
                queries_count as u64,
                summary_stride as u64,
            ],
        )
    }

    /// The chained block residual (`inkling_block_residual_chain_*`, [RESIDUAL_CHAIN_ENV]): one
    /// block a row for all `queries_count` queries, the same bits as [Self::block_residual].
    ///
    /// `carry` is the arena's [Scratch::residual_stats]. Its first `carried` stack rows are taken
    /// as scored against these queries by the previous call; the rest are read for their
    /// statistics. With `next` (the next call's queries and their count), the mix pass scores
    /// them against the summary rows into `carry` for that call.
    #[allow(clippy::too_many_arguments)]
    fn block_residual_chain(
        &self,
        summaries: Deviceptr,
        partial: Deviceptr,
        queries: Deviceptr,
        out: Deviceptr,
        carry: Deviceptr,
        blocks: usize,
        queries_count: usize,
        carried: usize,
        next: Option<(Deviceptr, usize)>,
        rows: usize,
        summary_stride: usize,
    ) -> Result<(), String> {
        let next_count = next.map_or(0, |(_, count)| count);
        let entry = RESIDUAL_CHAIN_PAIRS
            .iter()
            .position(|&(queries, next, _)| (queries, next) == (queries_count, next_count))
            .filter(|_| blocks <= RESIDUAL_CHAIN_SUMMARIES && carried <= blocks)
            .ok_or_else(|| {
                format!(
                    "the chained block residual reads {blocks} summary rows ({carried} carried) \
                     for {queries_count} queries and scores {next_count}: its kernels hold \
                     {RESIDUAL_CHAIN_SUMMARIES} summary rows for the query counts {:?}; \
                     {RESIDUAL_CHAIN_ENV}=off runs the one-block form",
                    RESIDUAL_CHAIN_PAIRS.map(|(queries, next, _)| (queries, next))
                )
            })?;
        self.launch(
            self.kernels.block_residual_chain[entry],
            (rows as u32, 1, 1),
            (ELEMENT_BLOCK, 1, 1),
            &[
                summaries.0,
                partial.0,
                queries.0,
                out.0,
                carry.0,
                next.map_or(0, |(queries, _)| queries.0),
                blocks as u64,
                self.hidden() as u64,
                bits(self.config.architecture.norm_epsilon as f32),
                summary_stride as u64,
                carried as u64,
                RESIDUAL_STATS_ROW as u64,
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
        let weight = self.bf16_rows(weight, rows, columns)?;
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
    /// tensor-core walk run beside it and the layer's largest difference printed. `halves` runs
    /// the tensor-core walk's MMA as two launches ([Self::attention_mma]), which the other walks
    /// refuse.
    #[allow(clippy::too_many_arguments)]
    fn attention_chunk<'h>(
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
        halves: Option<AttentionHalves<'h>>,
    ) -> Result<(), String> {
        if halves.is_some() && self.attention != AttentionWalk::Fast {
            return Err(format!(
                "a chunk's attention runs as two halves on the tensor-core walk alone, and this \
                 backend's walk is {:?}",
                self.attention
            ));
        }
        if let Some(tables) = tables {
            self.store_key_tables(tables, keys, values, rows, first_position, kv_heads)?;
        }
        let tensor = |into: Deviceptr, halves: Option<AttentionHalves<'h>>| -> Result<(), String> {
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
                halves,
            )
        };
        if self.attention == AttentionWalk::Fast {
            return tensor(out, halves);
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
        tensor(check, None)?;
        self.attention_compare(layer, out, check, rows, first_position)
    }

    /// The tensor-core walk of a chunk's rows: the query side as the exact walks compute it
    /// (inkling_attn_prepare), every visible key's clip gain and value scale (a full layer's
    /// [KeyTables], or the split walk's inkling_attn_key_scales over a sliding layer's whole
    /// range), then the MMA body (inkling_attn_mma).
    ///
    /// Under `halves` the body is two launches, the rows before the split and the rows from it,
    /// each followed by the hook over its rows. A block of the body reads its own 16 rows' query
    /// side, sink and gate, and takes its keys, tiles and bias distances from its first row's
    /// position alone, so at a split on a block's edge every block is the one launch's block and
    /// writes the same bits.
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
        halves: Option<AttentionHalves<'_>>,
    ) -> Result<(), String> {
        let config = &self.config;
        let heads = self.heads();
        let entry = self
            .kernels
            .attention_mma
            .ok_or_else(|| "the tensor-core attention kernel was not loaded".to_owned())?;
        if let Some((split, _)) = halves
            && (split == 0 || split >= rows || !split.is_multiple_of(MMA_Q))
        {
            return Err(format!(
                "a {rows}-row chunk's attention splits on a {MMA_Q}-row block's edge inside it, \
                 not at row {split}"
            ));
        }
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
        // The rows from `first` on: every per-row operand at its row, the positions from its own.
        let body = |first: usize, count: usize| {
            let float = std::mem::size_of::<f32>();
            let query = heads * MMA_HEAD_DIM;
            self.launch(
                entry,
                (count.div_ceil(MMA_Q) as u32, kv_heads as u32, 1),
                ((groups * 32) as u32, 1, 1),
                &[
                    mma.q.offset(first * query * float).0,
                    keys.0,
                    values.0,
                    mma.rel.offset(first * heads * MMA_D_REL * float).0,
                    bank.0,
                    // No sink is a null pointer, which the body reads as none.
                    if sink.0 == 0 {
                        0
                    } else {
                        sink.offset(first * heads * float).0
                    },
                    gains.0,
                    scales.0,
                    gate.offset(first * query * float).0,
                    out.offset(first * query * std::mem::size_of::<u16>()).0,
                    heads as u64,
                    kv_heads as u64,
                    count as u64,
                    (first_position + first) as u64,
                    window as u64,
                    cache_rows as u64,
                    rel_extent as u64,
                    config.d_rel as u64,
                    scale_first as u64,
                    scale_stride as u64,
                ],
            )
        };
        let Some((split, hook)) = halves else {
            return body(0, rows);
        };
        body(0, split)?;
        hook(0, split)?;
        body(split, rows - split)?;
        hook(split, rows - split)
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
        // A sliding layer's rows each see one cluster's runs: its split-key walk is one launch
        // after the passes (which then run only the exact mix's launches under Check).
        let windowed = parts.is_some()
            && tables.is_none()
            && window > 0
            && window <= WINDOW_RUNS * PART_KEYS
            && groups <= WINDOW_GROUPS
            && config.head_dim == WINDOW_HEAD_DIM;
        // Under Fast a windowed layer runs no pass: the window launch is its whole walk.
        let walked = if windowed && !exact { 0 } else { passes };
        for pass in 0..walked {
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
            // A long pass's runs score and take their partials in one launch; the exact mix
            // still reads the scores buffer, so under Check both run.
            let scored = parts.is_some()
                && groups <= SCORED_GROUPS
                && span.div_ceil(PART_KEYS) >= SCORED_RUNS;
            if exact || !(scored || windowed) {
                self.launch(
                    self.kernels.attention_scores,
                    (
                        grid_blocks(span, SCORE_KEYS).min(score_grid(rows)),
                        kv_heads as u32,
                        rows as u32,
                    ),
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
            }
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
            if windowed {
                continue;
            }
            if let Some(parts) = parts.filter(|_| scored) {
                self.launch(
                    self.kernels.attention_scored,
                    (
                        kv_heads as u32,
                        rows as u32,
                        span.div_ceil(PART_KEYS) as u32,
                    ),
                    (SCORED_THREADS, 1, 1),
                    &[
                        queries.0,
                        keys.0,
                        values.0,
                        relative.0,
                        bank.0,
                        clip.query.0,
                        0,
                        clip.relative.0,
                        key_gains.0,
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
                        rel_extent as u64,
                        config.d_rel as u64,
                        qk_clip,
                        bits(config.rel_l2_clip_threshold as f32),
                        bits(tau),
                        taus.0,
                        1,
                        pass_first as u64,
                        pass_slots as u64,
                        scale_first as u64,
                        scale_stride as u64,
                        parts.stride as u64,
                    ],
                )?;
            } else if let Some(parts) = parts {
                self.launch(
                    self.kernels.attention_part,
                    (
                        kv_heads as u32,
                        rows as u32,
                        span.div_ceil(PART_KEYS) as u32,
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
        if windowed {
            self.launch(
                self.kernels.attention_window,
                (kv_heads as u32, rows as u32, WINDOW_RUNS as u32),
                (WINDOW_THREADS, 1, 1),
                &[
                    queries.0,
                    keys.0,
                    values.0,
                    relative.0,
                    bank.0,
                    clip.query.0,
                    0,
                    clip.relative.0,
                    gate.0,
                    sink.0,
                    merged.0,
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
                    1,
                    eps,
                ],
            )?;
        } else {
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
        }
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
        pointers.extend(self.dense.allocations());
        pointers.push(self.weights.zeros);
        pointers.push(self.clip_arena);
        pointers.push(self.expert_arena);
        pointers.extend(self.exchange_slots);
        let refs: Vec<&Deviceptr> = pointers.iter().collect();
        self.cuda.free_buffers(&refs);
        if let Some(memory) = self
            .lanes
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            memory.pool().drain(&*self.cuda);
        }
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
        let mxfp8_walk = Mxfp8Walk::from_env(prefill_walk)?;
        let planes = ActivationPlanes::from_env(prefill_walk)?;
        let dense_form = DenseForm::from_env()?;
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
        let depth_policy = depth_policy_from_env(draft, |name| std::env::var(name).ok())?;
        let rewind = rewind_request(std::env::var(REWIND_ENV).ok().as_deref())?;
        // A stack grows one row a block boundary (layer % block == 0).
        let summaries = config
            .num_hidden_layers
            .div_ceil(config.n_layers_per_attn_res_block.max(1));
        let chain_switch = std::env::var(RESIDUAL_CHAIN_ENV).ok();
        let residual_chain = residual_chain_request(chain_switch.as_deref(), summaries)?;
        if !residual_chain && chain_switch.as_deref() != Some("off") {
            eprintln!(
                "inkling: the block residuals run unchained: the chained form holds \
                 {RESIDUAL_CHAIN_SUMMARIES} summary rows and the stack grows to {summaries}"
            );
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
        if config.expert_split == ExpertSplit::Rows {
            // The row split gathers the ranks' activation runs and mix rows on the device leg in
            // whole 16-byte words, and only the quantised bands read the gathered layout.
            if tp.world() > 1 && !exchange.has_device_leg() {
                return Err("a row expert split gathers on the device leg, which this \
                            exchange does not have"
                    .to_owned());
            }
            if let Some(layer) = plan
                .layers()
                .iter()
                .find(|layer| layer.experts == ExpertBody::Bf16)
            {
                return Err(format!(
                    "layer {}: a row expert split has no BF16 expert band",
                    layer.index
                ));
            }
            let (width, rows) = (plan.expert_intermediate(), plan.expert_down().rows);
            if !(width * 2).is_multiple_of(16) || !(rows * 4).is_multiple_of(16) {
                return Err(format!(
                    "a row expert split's pieces ({width} columns, {rows} rows a rank) are not \
                     whole 16-byte words"
                ));
            }
        }
        let (attention, mma_shared) = AttentionWalk::select(
            &config,
            plan.rank_heads() / plan.rank_kv_heads().max(1),
            prefill_walk,
        )?;
        let cuda = CudaHandle::new(Cuda::load()?);
        let kernels = Kernels::load(&cuda, mma_shared, planes)?;
        let weights = weights(&cuda, plan.served())?;
        let dense = pack_dense(&cuda, &kernels, &[&plan, &prefill], &weights, dense_form)?;
        eprintln!("[inkling] rank {}: {}", tp.rank(), dense.summary());
        let (clip, clip_arena) = widen_clips(&cuda, &kernels, &plan, &weights)?;
        let ExpertTables {
            scales: experts,
            mma,
            arena: expert_arena,
            route_ones,
        } = expert_scales(&cuda, &plan, &weights, mxfp8_walk)?;
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
        let halves_run = config.expert_split == ExpertSplit::Rows
            && exchange.world() > 1
            && exchange.has_device_leg()
            && prefill_walk == PrefillWalk::Tensor;
        let overlap = match overlap_request(std::env::var(OVERLAP_ENV).ok().as_deref())? {
            Some(true) if !halves_run => {
                return Err(format!(
                    "{OVERLAP_ENV}=on runs the row expert split's band over more than one rank on \
                     the device leg under {PREFILL_ENV}=tensor, and this group is {:?} over {} \
                     ranks under {prefill_walk:?}",
                    config.expert_split,
                    exchange.world()
                ));
            }
            Some(false) => None,
            _ if halves_run => Some(Mutex::new(BandOverlap::new(&cuda)?)),
            _ => None,
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
                dense,
                clip,
                clip_arena,
                experts,
                mma,
                mxfp8_walk,
                planes,
                expert_arena,
                route_ones,
                overlap,
                context,
                exchange_slots,
                exchange,
                attention,
                chunk_widths,
                prefill_chunk,
                prefill_walk,
                rewind,
                residual_chain,
                draft,
                depth_policy,
                dspark: None,
                vision: None,
                lanes: Mutex::new(None),
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
        let mut model = DraftModel::load(shared, directory)?;
        // The drafter's projections take the target's dense form: the same lossless rows its
        // GEMVs read in about three quarters of the bytes, and the same bits.
        let widest = model
            .dense_matrices()
            .iter()
            .map(|(_, rows, columns)| rows * columns * 2)
            .max()
            .unwrap_or(0);
        match shared.dense.pack(
            &shared.cuda,
            &shared.kernels,
            model.dense_matrices(),
            widest,
        ) {
            Ok((bf16_bytes, packed_bytes)) if bf16_bytes > 0 => eprintln!(
                "[inkling] rank {}: drafter projections packed: {:.1} -> {:.1} MB",
                shared.exchange.rank(),
                bf16_bytes as f64 / 1e6,
                packed_bytes as f64 / 1e6
            ),
            Ok(_) => {}
            Err(error) => {
                model.free(&shared.cuda);
                return Err(format!("the drafter's projections: {error}"));
            }
        }
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
        let joint = sparkflash_runtime::joint_rounds()
            .map_err(|error| BackendError::new("config", error))?;
        let config = &self.shared.config;
        Ok(InklingLanes {
            shared: Arc::clone(&self.shared),
            buffer: Mutex::new(None),
            joint: Mutex::new(None),
            joint_rounds: joint && self.shared.draft.is_some(),
            wall: ExpertUnionWall {
                experts: config.n_routed_experts,
                top_k: config.num_experts_per_tok,
                ..JOINT_WALL
            },
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
    /// The shared walk's buffer when drafting lanes verify together: a split buffer whose every
    /// row is answered, as wide as the widest walk so far.
    joint: Mutex<Option<Chunk>>,
    /// Whether drafting lanes verify together ([sparkflash_runtime::JOINT_ROUNDS_ENV]).
    joint_rounds: bool,
    /// What a shared walk costs, for the row share.
    wall: ExpertUnionWall,
    max_batch: usize,
}

/// A shared walk's cost at TP4 on four GB10-class nodes, fit to the 93k-token gate: a plain step
/// 29.6 ms and a four-row verify 61.5 ms, at 1.03 ms a distinct expert the trace measured; each
/// lane past the first adds its 8.9 ms draft. The expert counts are the checkpoint's.
const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 18.6,
    expert_ms: 1.03,
    row_ms: 2.7,
    seq_ms: 8.9,
    experts: 0,
    top_k: 0,
};

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
        let joint = self.joint.get_mut().ok().and_then(Option::take);
        for chunk in held.into_iter().chain(joint) {
            let pointers = chunk.buffers();
            self.shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
        }
    }
}

impl RoundArm for InklingLanes {
    type Member = InklingLane;

    fn open(&self, member: &mut InklingLane, token: u32) -> Result<Opened, BackendError> {
        member.session.open_round(token)
    }

    /// One walk of every lane's segment: its anchor and the drafts it was given, at its own
    /// positions, through the verify's per-sequence stages on its own state and tails
    /// ([Rows::Segments]).
    fn verify(
        &self,
        round: &mut [(&mut InklingLane, u32, usize)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let shared = &self.shared;
        let mut tokens = Vec::new();
        let mut shapes = Vec::with_capacity(round.len());
        for (lane, token, depth) in round.iter_mut() {
            let session = &lane.session;
            let opening = session
                .opening
                .as_ref()
                .ok_or_else(|| BackendError::new(DEVICE, "a lane walks a round it never opened"))?;
            let depth = (*depth).min(opening.drafts.len());
            let tails = session
                .verify
                .as_ref()
                .map(|verify| verify.tails)
                .ok_or_else(|| BackendError::new(DEVICE, "a lane walks with no verify buffer"))?;
            tokens.push(*token);
            tokens.extend_from_slice(&opening.drafts[..depth]);
            shapes.push((depth, tails, session.drafter.as_ref().map(DraftState::sink)));
        }
        let rows = {
            let mut joint = self.joint.lock().map_err(|_| {
                BackendError::new(DEVICE, "the joint walk's buffer lock is poisoned")
            })?;
            if joint
                .as_ref()
                .is_some_and(|chunk| chunk.rows < tokens.len())
            {
                let old = joint.take().expect("the joint buffer was just checked");
                let pointers = old.buffers();
                shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
            }
            if joint.is_none() {
                // As wide as every lane's deepest round, so it opens once.
                let widest = self.max_batch * (shared.draft.map_or(0, |(_, depth)| depth) + 1);
                let width = widest.max(tokens.len());
                *joint = Some(
                    Chunk::allocate(shared, width, width, true)
                        .map_err(|error| BackendError::new(DEVICE, error))?,
                );
            }
            let chunk = joint.as_mut().expect("the joint buffer was just opened");
            let mut segments: Vec<Segment<'_>> = round
                .iter_mut()
                .zip(&shapes)
                .map(|((lane, _, _), &(depth, tails, taps))| Segment {
                    state: &mut lane.session.state,
                    tails,
                    rows: depth + 1,
                    taps,
                })
                .collect();
            walk_rows(
                shared,
                Rows::Segments(&mut segments),
                chunk,
                &tokens,
                PrefillWalk::Chunk,
                &[],
                None,
            )?
        };
        let mut rows = rows.into_iter();
        round
            .iter_mut()
            .zip(shapes)
            .map(|((lane, _, _), (depth, _, _))| {
                let mine: Vec<Vec<f32>> = rows.by_ref().take(depth + 1).collect();
                lane.session.close(depth, mine)
            })
            .collect()
    }

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    fn budget(&self, seqs: usize) -> usize {
        seqs * self.shared.draft.map_or(0, |(_, depth)| depth)
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

    /// With the drafter on, the lanes' rounds verify in one walk ([run_round]): each lane opens
    /// its round, the walk's rows are shared among the chains, and every lane's anchor and drafts
    /// are one segment of the walk. A lone lane, or with [sparkflash_runtime::JOINT_ROUNDS_ENV]
    /// off every lane, decodes on its own with its own depth. A lane's tokens are the ones it
    /// would decode alone either way.
    fn step(&self, members: &mut [(InklingLane, u32)]) -> Result<Vec<StepOutput>, BackendError> {
        self.bind()?;
        if members.len() > 1 && self.joint_rounds {
            return run_round(self, members);
        }
        if members.len() == 1 || self.shared.draft.is_some() {
            return members
                .iter_mut()
                .map(|(lane, token)| lane.session.decode(*token))
                .collect();
        }
        let shared = &self.shared;
        for (lane, token) in members.iter_mut() {
            if lane.session.state.position >= shared.context {
                return Err(BackendError::new(
                    "context",
                    "a lane's context is exhausted",
                ));
            }
            let rows = lane.session.state.position + 1;
            lane.session
                .state
                .grow(shared, rows)
                .map_err(|error| BackendError::new("context", error))?;
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

    fn speculation(&self, slot: &InklingLane) -> Speculation {
        slot.session.speculation()
    }

    fn ready(&self, slot: &InklingLane, token: u32) -> bool {
        slot.session.held(token)
    }

    fn give_back(&self, slot: &mut InklingLane, count: usize) -> Result<(), BackendError> {
        slot.session.give_back(count)
    }

    /// A prompt that neither continues the history nor rewinds into it starts over, and the
    /// chunks the lane holds go back to the pool first.
    fn room(&self, slot: &InklingLane, prompt: Option<&[u32]>) -> bool {
        let session = &slot.session;
        match prompt {
            Some(tokens) => {
                let keeps = tokens.starts_with(&session.history)
                    || session.rewind_prefix().is_some_and(|prefix| {
                        prefix.len() < tokens.len() && tokens.starts_with(prefix)
                    });
                session.state.room(tokens.len(), !keeps)
            }
            None => {
                let rows =
                    session.state.position + 1 + self.shared.draft.map_or(0, |(_, depth)| depth);
                session.state.room(rows, false)
            }
        }
    }

    /// Forgets the lane's sequence, which gives its chunks back to the pool.
    fn release(&self, slot: &mut InklingLane) -> Result<(), BackendError> {
        self.bind()?;
        slot.session.forget()
    }

    /// A lane keeps its sequence between requests as a session alone does: the next prompt that
    /// starts with it forwards only the rest ([InklingDeviceSession::prompt]).
    fn history<'s>(&self, slot: &'s InklingLane) -> Option<&'s [u32]> {
        slot.session.reusable_prefix()
    }

    fn rewind_prefix<'s>(&self, slot: &'s InklingLane) -> Option<&'s [u32]> {
        slot.session.rewind_prefix()
    }

    fn held_images<'s>(&self, slot: &'s InklingLane) -> &'s [ImageSpan] {
        slot.session.held_images()
    }

    fn cached_tokens(&self, slot: &InklingLane) -> usize {
        slot.session.cached_tokens()
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

/// What [expert_scales] lays out: the NVFP4 layers' scales, every grouped-MMA table, the arena
/// they all live in, and the [top_k] ones the MMA's down pass reads as routing weights.
struct ExpertTables {
    scales: Vec<Option<ExpertScale>>,
    mma: Vec<Option<MmaTable>>,
    arena: Deviceptr,
    route_ones: Deviceptr,
}

/// Read the released per-expert scales off the device once, so a token's band does not have to,
/// and lay out each NVFP4 layer's grouped-MMA table, and each MXFP8 layer's under
/// [Mxfp8Walk::Mma] once its scale codes are admitted ([admit_mxfp8_scales]).
fn expert_scales(
    cuda: &Cuda,
    plan: &InklingDevicePlan,
    weights: &Weights,
    mxfp8: Mxfp8Walk,
) -> Result<ExpertTables, String> {
    let mma_mx8 =
        |layer: &LayerSchedule| mxfp8 == Mxfp8Walk::Mma && layer.experts == ExpertBody::MxFp8;
    let (_, owned) = plan.expert_range();
    for layer in plan.layers().iter().filter(|layer| mma_mx8(layer)) {
        admit_mxfp8_scales(cuda, layer, weights)?;
    }
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
    let mma_words = |owned: usize| words(owned * MMA_TABLE_ENTRIES * 2) + words(owned * MMA_SCALES);
    let floats: usize = table
        .iter()
        .flatten()
        .map(|(gate_up, down): &(Vec<f32>, Vec<f32>)| {
            words(gate_up.len()) + words(down.len()) + mma_words(gate_up.len())
        })
        .sum::<usize>()
        + plan.layers().iter().filter(|layer| mma_mx8(layer)).count() * mma_words(owned)
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
    let mut scales = Vec::with_capacity(table.len());
    let mut mma = Vec::with_capacity(table.len());
    for (layer, entry) in plan.layers().iter().zip(table) {
        // The outer scales a local expert's gate, up and down rows take.
        let outer: Vec<f32> = match entry {
            Some((gate_up, down)) => {
                let device_gate_up = place(&row_bytes(&gate_up))?;
                let device_down = place(&row_bytes(&down))?;
                let outer = gate_up
                    .iter()
                    .zip(&down)
                    .flat_map(|(gate_up, down)| [*gate_up, *gate_up, *down])
                    .collect();
                scales.push(Some(ExpertScale {
                    gate_up,
                    down,
                    device_gate_up,
                    device_down,
                }));
                outer
            }
            // An MXFP8 expert stores no outer scale.
            None if mma_mx8(layer) => {
                scales.push(None);
                vec![1.0; owned * MMA_SCALES]
            }
            None => {
                scales.push(None);
                mma.push(None);
                continue;
            }
        };
        let pointers = mma_pointers(plan, layer, weights, outer.len() / MMA_SCALES)?;
        let pointer_bytes: Vec<u8> = pointers
            .iter()
            .flat_map(|pointer| pointer.to_le_bytes())
            .collect();
        mma.push(Some(MmaTable {
            table: place(&pointer_bytes)?,
            scales: place(&row_bytes(&outer))?,
        }));
    }
    Ok(ExpertTables {
        scales,
        mma,
        arena,
        route_ones,
    })
}

/// A layer's grouped-MMA pointers for `owned` local experts ([MmaTable]), in its stored body.
fn mma_pointers(
    plan: &InklingDevicePlan,
    layer: &LayerSchedule,
    weights: &Weights,
    owned: usize,
) -> Result<Vec<u64>, String> {
    let body = layer.experts;
    let config = plan.config();
    let (hidden, intermediate) = (config.hidden_size, plan.expert_intermediate());
    let down_shape = plan.expert_down();
    let w13_bytes = codes_bytes(2 * intermediate, hidden, body);
    let w13_scale_bytes = scales_bytes(2 * intermediate, hidden, body);
    let w2_bytes = codes_bytes(down_shape.rows, down_shape.columns, body);
    let w2_scale_bytes = scales_bytes(down_shape.rows, down_shape.columns, body);
    // A slot the table walks has to hold every local expert it names.
    let slot = |op: Op, which: usize, expert_bytes: usize| -> Result<Deviceptr, String> {
        let index = layer
            .steps
            .iter()
            .find(|step| step.op == op)
            .and_then(|step| step.weights.get(which))
            .ok_or_else(|| {
                format!(
                    "layer {}: no {op:?} weight {which} for the MMA table",
                    layer.index
                )
            })?;
        let slot = weights.get(*index);
        if slot.bytes < owned * expert_bytes {
            return Err(format!(
                "layer {}: {op:?} weight {which} holds {} bytes, not the {owned} experts of {} \
                 bytes the MMA table walks",
                layer.index, slot.bytes, expert_bytes
            ));
        }
        Ok(slot.pointer)
    };
    let w13 = slot(Op::ExpertGateUp, 0, w13_bytes)?;
    let w13_scales = slot(Op::ExpertGateUp, 1, w13_scale_bytes)?;
    let w2 = slot(Op::ExpertDown, 0, w2_bytes)?;
    let w2_scales = slot(Op::ExpertDown, 1, w2_scale_bytes)?;
    // One w13 row: its code bytes and its block scales (NVFP4 hidden / 2 and hidden / 16, MXFP8
    // hidden and hidden / 32).
    let (code_row, scale_row) = (codes_bytes(1, hidden, body), scales_bytes(1, hidden, body));
    if body == ExpertBody::MxFp8 {
        // The MXFP8 tile walks a row in 64-column stages and takes a stage's two ue8m0 codes from
        // the half of the aligned word that holds them, so a row it walks is whole stages: its
        // codes then start on a two-byte boundary and a pair never straddles two words.
        let run = 2 * MXFP8_BLOCK;
        if !hidden.is_multiple_of(run) || !down_shape.columns.is_multiple_of(run) {
            return Err(format!(
                "layer {}: the MXFP8 MMA walks rows of whole {run}-column stages, and the \
                 experts' rows are {hidden} and {} wide; {MXFP8_ENV}=pairs walks this band",
                layer.index, down_shape.columns
            ));
        }
    }
    let mut pointers = Vec::with_capacity(owned * MMA_TABLE_ENTRIES);
    for local in 0..owned {
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
    Ok(pointers)
}

/// Refuse an MXFP8 layer whose weights the grouped MMA would not hold exactly.
///
/// The MMA forms each weight as the pair walk does - the E4M3 value times 2^(code - 127) in f32 -
/// and rounds it to BF16. An E4M3 value has at most four significant bits, so the rounding is
/// exact (an f32 overflow is infinite in both) unless the product falls under BF16's subnormal
/// step, 2^-133: the smallest E4M3 step is 2^-9, which happens for codes 1 and 2 alone. Code 0
/// scales to zero in both walks. Every code of the layer's w13 and w2 is read once, at open.
fn admit_mxfp8_scales(cuda: &Cuda, layer: &LayerSchedule, weights: &Weights) -> Result<(), String> {
    for op in [Op::ExpertGateUp, Op::ExpertDown] {
        let index = layer
            .steps
            .iter()
            .find(|step| step.op == op)
            .and_then(|step| step.weights.get(1))
            .ok_or_else(|| format!("layer {}: no {op:?} block scales", layer.index))?;
        let slot = weights.get(*index);
        let mut codes = vec![0u8; slot.bytes];
        cuda.copy_dtoh(&mut codes, &slot.pointer)
            .map_err(|error| format!("layer {} {op:?} scales: {error}", layer.index))?;
        if let Some(at) = codes.iter().position(|code| matches!(code, 1 | 2)) {
            return Err(format!(
                "layer {}: {op:?} scale code {} (byte {at}) puts the smallest E4M3 steps under \
                 BF16's subnormal step, so the grouped MMA would not hold those weights exactly; \
                 {MXFP8_ENV}=pairs walks the MXFP8 layers exactly",
                layer.index, codes[at]
            ));
        }
    }
    Ok(())
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

/// Rows past the furthest position a walk writes that a session backs too: attention reads whole
/// key tiles past a position (and masks them), and the drafter's block writes its rows past it.
const READ_PAST: usize = 1024;

/// A session's lane grows in steps of this many rows, so a decode step rarely maps anything.
const LANE_QUANTUM: usize = 2048;

/// Where every session keeps the state that grows with its context, and the memory behind it
/// ([sparkflash_kernels::lane_memory], the backing DeepSeek's lanes use too).
///
/// A session reserves a span of virtual memory as long as a whole context's full-layer keys and
/// values, their [KeyTables] and the drafter's full-layer rings, and physical memory backs the
/// span only as far as the sequence has reached ([State::grow]). The chunks come from one pool a
/// rank's sessions share, so several conversations hold together what one whole context would:
/// the budget is one full context's worth unless `runtime.lane_memory_mb`
/// (SPARKFLASH_LANE_MEMORY_MB) says otherwise. It is a number, the same on every rank, so every
/// rank of a group refuses the same growth.
///
/// A sliding layer's ring, the convolution histories and the scratch stay plain allocations:
/// they do not grow with the context.
struct LaneMemory {
    shape: LaneShape,
    /// Each layer's key and value parts, `None` for a sliding layer.
    kv: Vec<Option<(usize, usize)>>,
    /// Each layer's [KeyTables] parts (gains, scales), where the layer keeps tables.
    tables: Vec<Option<(usize, usize)>>,
    /// Each drafter layer's key and value ring, `None` for one that keeps a window.
    draft: Vec<Option<(usize, usize)>>,
    pool: Mutex<ChunkPool>,
}

impl LaneMemory {
    fn plan(shared: &Shared) -> Result<Self, String> {
        let granularity = shared
            .cuda
            .vmm_granularity()
            .ok_or("the device has no virtual memory management for the sessions' caches")?;
        let mut layout = Layout::default();
        let plan = &shared.plan;
        let row = shared.kv_width() * 2;
        let mut kv = Vec::with_capacity(plan.layers().len());
        let mut tables = Vec::with_capacity(plan.layers().len());
        for layer in plan.layers() {
            if layer.sliding {
                kv.push(None);
                tables.push(None);
                continue;
            }
            let rows = layer.kv_rows.max(1);
            kv.push(Some((
                layout.part(rows * row, row, 0),
                layout.part(rows * row, row, 0),
            )));
            // A table is `[kv head][row]`: each head's rows are a part of their own.
            let mut table = || {
                let heads = plan.rank_kv_heads().max(1);
                let first = layout.reserve(heads * rows * 4);
                for head in 0..heads {
                    layout.region(first + head * rows * 4, rows * 4, 4, 0);
                }
                first
            };
            tables.push(shared.attention.split().then(|| (table(), table())));
        }
        let draft = match (shared.draft, shared.dspark.as_ref()) {
            (Some((Drafter::Dspark, _)), Some(model)) => {
                let row = model.ring_row_bytes();
                model
                    .whole_context_rings()
                    .into_iter()
                    .map(|whole| {
                        whole.then(|| {
                            let rows = shared.context.max(1);
                            (
                                layout.part(rows * row, row, 0),
                                layout.part(rows * row, row, 0),
                            )
                        })
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        let shape = layout.finish(granularity, LANE_QUANTUM);
        let budget = sparkflash_model::active_profile()
            .runtime
            .lane_memory_mb
            .map_or_else(
                || shape.chunks_for(shared.context + READ_PAST, 0) * granularity,
                |mb| mb << 20,
            );
        Ok(Self {
            shape,
            kv,
            tables,
            draft,
            pool: Mutex::new(ChunkPool::new(granularity, budget)),
        })
    }

    fn pool(&self) -> std::sync::MutexGuard<'_, ChunkPool> {
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One session's span of the [LaneMemory].
struct Lane {
    memory: Arc<LaneMemory>,
    span: LaneSpan,
}

/// One session's device state.
struct State {
    lane: Lane,
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
        let memory = shared.lane_memory()?;
        let base = cuda.vmm_reserve(memory.shape.stride)?;
        let lane = Lane {
            span: LaneSpan::new(base.0, &memory.shape),
            memory: Arc::clone(&memory),
        };
        let kv_bytes = |rows: usize, width: usize| rows.max(1) * width * 2;
        let mut keys = Vec::with_capacity(plan.layers().len());
        let mut values = Vec::with_capacity(plan.layers().len());
        let mut tables = Vec::with_capacity(plan.layers().len());
        for (index, layer) in plan.layers().iter().enumerate() {
            // A full layer's rows are the lane's, zeroed as they are backed.
            if let Some((key, value)) = memory.kv[index] {
                keys.push(base.offset(key));
                values.push(base.offset(value));
            } else {
                let bytes = kv_bytes(layer.kv_rows, shared.kv_width());
                let key = cuda.alloc(bytes, &mut total)?;
                let value = cuda.alloc(bytes, &mut total)?;
                cuda.memset_d8_async(key, 0, bytes)?;
                cuda.memset_d8_async(value, 0, bytes)?;
                keys.push(key);
                values.push(value);
            }
            // A slot's entries are written when its key is stored and read only after, so the
            // tables need no clearing.
            tables.push(memory.tables[index].map(|(gains, scales)| KeyTables {
                gains: base.offset(gains),
                scales: base.offset(scales),
                stride: layer.kv_rows.max(1),
            }));
        }
        let mut conv = Vec::with_capacity(plan.layers().len());
        for _ in 0..plan.layers().len() {
            let mut slots = [Deviceptr(0); 4];
            for (slot, bytes) in shared.conv_slot_bytes().into_iter().enumerate() {
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
            lane,
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
            if self.lane.memory.kv[index].is_some() {
                continue;
            }
            let bytes = shared.plan.layers()[index].kv_rows.max(1) * shared.kv_width() * 2;
            shared.cuda.memset_d8_async(*key, 0, bytes)?;
            shared.cuda.memset_d8_async(*value, 0, bytes)?;
        }
        // The lane's rows go back to the pool, and come back zeroed as the next prompt grows.
        self.release(shared)?;
        for slots in &self.conv {
            for (slot, bytes) in slots.iter().zip(shared.conv_slot_bytes()) {
                shared.cuda.memset_d8_async(*slot, 0, bytes)?;
            }
        }
        shared
            .cuda
            .memset_d8_async(self.summaries, 0, self.summaries_bytes)?;
        self.blocks = 0;
        self.position = 0;
        Ok(())
    }

    /// Backs the lane for a sequence of `rows` positions and the rows a walk reads past them.
    ///
    /// # Errors
    /// A pool with too few chunks left, or a driver failure.
    fn grow(&mut self, shared: &Shared, rows: usize) -> Result<(), String> {
        let memory = Arc::clone(&self.lane.memory);
        let mut pool = memory.pool();
        let want = rows + READ_PAST;
        if self
            .lane
            .span
            .ensure(&*shared.cuda, &memory.shape, &mut pool, want)?
        {
            return Ok(());
        }
        Err(Self::short(&memory, &self.lane.span, &pool, want))
    }

    /// Whether the pool can back `rows` positions here now, counting the chunks a reset gives
    /// back first when `reset`.
    fn room(&self, rows: usize, reset: bool) -> bool {
        let memory = &self.lane.memory;
        let pool = memory.pool();
        let want = rows + READ_PAST;
        if reset {
            memory.shape.chunks_for(want, 0) <= pool.spare() + self.lane.span.mapped()
        } else {
            self.lane.span.fits(&memory.shape, &pool, want)
        }
    }

    fn short(memory: &LaneMemory, span: &LaneSpan, pool: &ChunkPool, rows: usize) -> String {
        let chunk = memory.shape.granularity;
        format!(
            "lane memory: {rows} positions need {} MiB more and the sessions' pool has {} MiB of \
             {} MiB spare (runtime.lane_memory_mb / SPARKFLASH_LANE_MEMORY_MB sets it)",
            (span.chunks_needed(&memory.shape, rows) * chunk) >> 20,
            (pool.spare() * chunk) >> 20,
            (pool.budget() * chunk) >> 20
        )
    }

    /// Gives every chunk the lane holds back to the pool, once the device is done with them.
    fn release(&mut self, shared: &Shared) -> Result<(), String> {
        if self.lane.span.mapped() == 0 {
            return Ok(());
        }
        shared.cuda.synchronize()?;
        let memory = Arc::clone(&self.lane.memory);
        let mut pool = memory.pool();
        self.lane
            .span
            .release(&*shared.cuda, &memory.shape, &mut pool)
    }

    /// The allocations this state owns outside its lane.
    fn buffers(&self) -> Vec<Deviceptr> {
        let memory = &self.lane.memory;
        let mut pointers: Vec<Deviceptr> = Vec::new();
        for (index, (key, value)) in self.keys.iter().zip(&self.values).enumerate() {
            if memory.kv[index].is_none() {
                pointers.extend([*key, *value]);
            }
        }
        for taps in &self.conv {
            pointers.extend(taps.iter().copied());
        }
        pointers.push(self.summaries);
        pointers.push(self.scratch.arena);
        pointers
    }

    /// Gives the lane back: its chunks to the pool and its span to the device.
    fn free_lane(&mut self, shared: &Shared) {
        let _ = self.release(shared);
        shared.cuda.vmm_free(
            Deviceptr(self.lane.span.base()),
            self.lane.memory.shape.stride,
        );
    }

    /// The drafter's whole-context rings in this lane, by drafter layer.
    fn draft_rings(&self) -> Vec<Option<(Deviceptr, Deviceptr)>> {
        let base = Deviceptr(self.lane.span.base());
        self.lane
            .memory
            .draft
            .iter()
            .map(|ring| ring.map(|(keys, values)| (base.offset(keys), base.offset(values))))
            .collect()
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
    /// The two-launch residual's statistics: `[rows][3][RESIDUAL_ROWS][2]`, each stack row's
    /// RMS inverse and score ([RESIDUAL_STATS_ROW] a row). A chained walk carries its scores
    /// here instead.
    residual_stats: Deviceptr,
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
        let residual_stats = take(rows * RESIDUAL_STATS_ROW, 4);
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
        // The row split's down projection lands in the band once the activation has read it,
        // and its gathered activation runs and mix rows wait in `partials` (see BandOut).
        let down = plan.expert_down();
        let band = take(rows * top_k * (2 * activation).max(down.rows), 4);
        let activation_buffer = take(rows * top_k * activation.max(down.columns), 4);
        // The expert-parallel splits project each slot's residual-wide row into `partials`. The
        // row split never does: it holds every rank's activation runs of every slot there (f32,
        // or the MMA walk's two bf16 planes, the same bytes) and later every rank's mix rows,
        // which at Inkling's shapes is half of a residual-wide row a slot.
        let partials = if config.expert_split == ExpertSplit::Rows {
            let world = shared.exchange.world();
            take(
                (rows * top_k * activation * world).max(rows * down.rows * world),
                4,
            )
        } else {
            take(rows * top_k * hidden.max(down.columns), 4)
        };
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
        // The two-halves band plans each half on its own: two counts, and up to one more short
        // item and one more item an expert than the whole chunk's plan.
        let owned = shared.plan.expert_range().1;
        let items = take((chunk::moe_items(rows * top_k, owned) + owned + 1) * 4, 4);
        let item_count = take(2, 4);
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
            residual_stats: at(residual_stats),
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
    /// A round opened for a shared walk that has not run yet.
    opening: Option<Opening>,
    /// What this request's drafts have been worth.
    speculator: Speculator,
    /// The DSpark drafter's context for this sequence, when the rank drafts with it.
    drafter: Option<DraftState>,
    /// The last prompt's end, kept to rewind to ([REWIND_ENV]); `None` when the switch is off.
    boundary: Option<Boundary>,
}

/// The prompt-boundary checkpoint ([REWIND_ENV]): what a sequence held at the end of its last
/// prompt that its cache rows do not hold.
///
/// A position's keys and values stay in the cache until a later position's write lands on their
/// slot, so a sequence can go back to an earlier position by its cache alone, but the convolution
/// histories and the drafter's Markov feature are the state after the last row and nothing else:
/// these are copied out at the prompt's end and back in at a rewind.
struct Boundary {
    /// One allocation: every layer's four histories in slot order, then the drafter's feature.
    arena: Deviceptr,
    /// Where the feature's copy starts in the arena, when the sequence drafts with DSpark.
    feature: Option<usize>,
    /// The prompt's length while the copies hold its end; `None` before a prompt ends, and from
    /// the moment a prompt starts forwarding until it ends.
    position: Option<usize>,
    /// The image spans the prompt held.
    images: Vec<ImageSpan>,
}

impl Boundary {
    fn allocate(shared: &Shared, drafter: Option<&DraftState>) -> Result<Self, String> {
        let conv = shared.plan.layers().len() * shared.conv_slot_bytes().iter().sum::<usize>();
        let feature = drafter
            .zip(shared.dspark.as_ref())
            .map(|(drafter, model)| drafter.feature(model).1);
        let at = conv.next_multiple_of(256);
        let mut total = 0usize;
        let arena = shared.cuda.alloc(at + feature.unwrap_or(0), &mut total)?;
        Ok(Self {
            arena,
            feature: feature.map(|_| at),
            position: None,
            images: Vec::new(),
        })
    }

    /// Copies `state`'s histories and `drafter`'s feature into the checkpoint (`back` false) or
    /// out of it into them (`back` true), on the run's stream.
    fn copy(
        &self,
        shared: &Shared,
        state: &State,
        drafter: Option<&DraftState>,
        back: bool,
    ) -> Result<(), String> {
        let cuda = &shared.cuda;
        let copy = |live: Deviceptr, kept: Deviceptr, bytes: usize| {
            if back {
                cuda.copy_dtod_async(&live, &kept, bytes)
            } else {
                cuda.copy_dtod_async(&kept, &live, bytes)
            }
        };
        let mut at = 0usize;
        for slots in &state.conv {
            for (slot, bytes) in slots.iter().zip(shared.conv_slot_bytes()) {
                copy(*slot, self.arena.offset(at), bytes)?;
                at += bytes;
            }
        }
        if let (Some(offset), Some(drafter), Some(model)) =
            (self.feature, drafter, shared.dspark.as_ref())
        {
            let (feature, bytes) = drafter.feature(model);
            copy(feature, self.arena.offset(offset), bytes)?;
        }
        Ok(())
    }
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
/// rows from row `kept` on, which is what [InklingDeviceSession::settle] restores every slot's
/// history from, however many of the drafts the caller fed: the verify leaves the state's
/// histories as they were before it.
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
/// answered, the [VerifyTails] that let it keep a leading run of the rows, and the restore's
/// table: for every (layer, slot) of `conv`, the slot's tail, the sequence's history it
/// restores, and its channels.
struct VerifyBuffer {
    chunk: Chunk,
    tails: VerifyTails,
    restore: Deviceptr,
    entries: usize,
}

impl VerifyBuffer {
    fn allocate(shared: &Shared, rows: usize, conv: &[[Deviceptr; 4]]) -> Result<Self, String> {
        let chunk = Chunk::allocate(shared, rows, rows, true)?;
        let free_chunk = || {
            let pointers = chunk.buffers();
            shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
        };
        let mut total = 0usize;
        let base = shared
            .cuda
            .alloc(VerifyTails::bytes(rows, shared), &mut total)
            .inspect_err(|_| free_chunk())?;
        let tails = VerifyTails::new(shared, base, rows);
        let mut table = Vec::with_capacity(conv.len() * 4 * 3 * 8);
        for (layer, slots) in conv.iter().enumerate() {
            for (slot, history) in slots.iter().enumerate() {
                for word in [
                    tails.at(layer, slot).0,
                    history.0,
                    tails.channels(slot) as u64,
                ] {
                    table.extend_from_slice(&word.to_le_bytes());
                }
            }
        }
        let entries = conv.len() * 4;
        let restore = shared
            .cuda
            .alloc(table.len().max(8), &mut total)
            .and_then(|restore| {
                shared
                    .cuda
                    .copy_htod(&restore, &table)
                    .map(|()| restore)
                    .inspect_err(|_| shared.cuda.free_buffers(&[&restore]))
            })
            .inspect_err(|_| {
                free_chunk();
                shared.cuda.free_buffers(&[&base]);
            })?;
        Ok(Self {
            chunk,
            tails,
            restore,
            entries,
        })
    }

    fn buffers(&self) -> [Deviceptr; 4] {
        let [arena, summaries] = self.chunk.buffers();
        [arena, summaries, self.tails.base, self.restore]
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
    /// The probability the drafter gave each draft it made, the verified ones first: what the
    /// depth policy learns from when the round settles.
    confidence: Vec<f32>,
    /// Each row's logits.
    rows: Vec<Vec<f32>>,
    /// The rows the caller has fed: the anchor, then each draft it fed in turn.
    fed: usize,
    /// The sampler as it stood before each draft fed after the anchor, so the last answers can be
    /// given back ([InklingDeviceSession::give_back]).
    marks: Vec<SamplerMark>,
}

/// A round opened for a shared walk ([InklingLanes]' [RoundArm::open]): the sequence settled and
/// drafted, and waits for the walk that verifies its anchor and the drafts it is given.
#[derive(Debug, Clone)]
struct Opening {
    /// The anchor's position, where the walk starts this sequence's rows.
    start: usize,
    anchor: u32,
    /// The whole chain the drafter made, the walk's share of it first.
    drafts: Vec<u32>,
    confidence: Vec<f32>,
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
/// How many of a DSpark round's drafts the verify walks: `adaptive` (unset, as every drafting
/// arm's default) picks the depth from the probabilities the drafter gave its drafts, `fixed`
/// walks all [DRAFT_DEPTH_ENV] of them. Read through the profile's `speculation.depth_policy`.
pub const DRAFT_POLICY_ENV: &str = "SPARKFLASH_INKLING_DRAFT_POLICY";
/// The fewest drafts an adaptive DSpark round verifies, 0 unset: the profile's
/// `speculation.min_depth`.
pub const DRAFT_MIN_DEPTH_ENV: &str = "SPARKFLASH_INKLING_DRAFT_MIN_DEPTH";

/// The modelled wall of a DSpark round at TP4 on four GB10-class nodes that verifies `depth`
/// drafts (ms), for [DepthPolicy]: the draft and the anchor's row at 41.6 ms, and 9.6 ms a verified
/// draft (mostly the experts its row adds). Fit to the 93k-token gate's 70.4 ms rounds at depth 3
/// against its 29.6 ms plain step and 8.9 ms draft; the 5k-token repeat prompt's rounds at 3 and 7
/// drafts fit 39.6 and 7.9. The first round assumes the 21.4 ms a token depth 3 made at 93k.
const DSPARK_STEP_COST: StepCost = StepCost {
    step_ms: 41.6,
    row_ms: 9.6,
    prior_tokens_per_ms: 1.0 / 21.4,
};

/// The verify-depth policy a DSpark sequence starts from ([DRAFT_POLICY_ENV],
/// [DRAFT_MIN_DEPTH_ENV]), over chains of the drafter's `depth`. Only DSpark gives its drafts a
/// probability, so either switch set for another drafter, or none, is refused rather than ignored,
/// as is a value outside the switch's domain or a minimum past the depth.
fn depth_policy_from_env(
    draft: Option<(Drafter, usize)>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<Option<DepthPolicy>, String> {
    const SWITCHES: [&str; 2] = [DRAFT_POLICY_ENV, DRAFT_MIN_DEPTH_ENV];
    let ours = |name: &str| {
        SWITCHES
            .contains(&name)
            .then(|| lookup(name))
            .flatten()
            .filter(|value| !value.is_empty())
    };
    let Some((Drafter::Dspark, depth)) = draft else {
        return match SWITCHES.iter().find(|name| ours(name).is_some()) {
            Some(name) => Err(format!(
                "{name} sets how many DSpark drafts a round verifies, and {DRAFT_ENV} is not \
                 dspark: no other drafter gives its drafts a probability"
            )),
            None => Ok(None),
        };
    };
    let mut profile = sparkflash_model::profile::ModelProfile::default();
    profile.apply_env(ours).map_err(|error| error.to_string())?;
    let speculation = &profile.speculation;
    if let Some(least) = speculation.min_depth
        && least > depth
    {
        return Err(format!(
            "{DRAFT_MIN_DEPTH_ENV}={least} is past the {depth} drafts a round makes \
             ({DRAFT_DEPTH_ENV})"
        ));
    }
    Ok(Some(DepthPolicy::from_speculation(
        speculation,
        depth,
        DSPARK_STEP_COST,
    )))
}
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
        let shared = Arc::clone(&self.shared);
        self.state.free_lane(&shared);
        let mut pointers = self.state.buffers();
        if let Some(chunk) = &self.chunk {
            pointers.extend(chunk.buffers());
        }
        if let Some(verify) = &self.verify {
            pointers.extend(verify.buffers());
        }
        if let Some(drafter) = &self.drafter {
            pointers.extend(drafter.buffers());
        }
        if let Some(boundary) = &self.boundary {
            pointers.push(boundary.arena);
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
            round.marks.push(self.sampler.mark());
            return Ok(self.decoded(token, logits));
        }
        self.settle()?;
        if self.state.position >= self.shared.context {
            return Err(BackendError::new("context", "the context is exhausted"));
        }
        // The round's token and every draft it may verify.
        let rows = self.state.position + 1 + self.shared.draft.map_or(0, |(_, depth)| depth);
        self.state
            .grow(&self.shared, rows)
            .map_err(|error| BackendError::new("context", error))?;
        let (drafts, confidence) = self.draft(token)?;
        if drafts.is_empty() {
            let position = self.state.position;
            let logits = self.step(token)?;
            self.inject(1, position)?;
            // A round that verifies nothing is still a round: the policy weighs the next depth at
            // the rate its rounds have made.
            self.speculator.observe(&confidence, 0, 1);
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
            confidence,
            rows,
            fed: 1,
            marks: Vec::new(),
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

    /// The last prompt, while the checkpoint holds its end and nothing since has written over the
    /// rows a rewound chunk reads before it.
    ///
    /// A sliding ring holds its window and a widest chunk, so those rows survive while the
    /// positions written past the boundary - the reply, and a verify's drafts beyond its last fed
    /// token - number at most a widest chunk; the drafter's local rings are opened as wide under
    /// [REWIND_ENV]. The next prompt gives a verify's unfed drafts back before it rewinds.
    fn rewind_prefix(&self) -> Option<&[u32]> {
        let point = self.boundary.as_ref()?.position?;
        let past = self.history.len().checked_sub(point)?;
        let drafts = self.shared.draft.map_or(0, |(_, depth)| depth);
        (past > 0 && past + drafts <= self.shared.prefill_chunk).then(|| &self.history[..point])
    }

    /// The last prompt's images. The rewind point is that prompt's end, so it holds them all.
    fn held_images(&self) -> &[ImageSpan] {
        &self.held_images
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
        let rewind = if reuse {
            None
        } else {
            self.rewind_point(tokens, images)
        };
        let start = if reuse {
            held
        } else if let Some(point) = rewind {
            self.rewind_to(point)?;
            point
        } else {
            self.state
                .reset(&self.shared)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            if let Some(drafter) = self.drafter.as_mut() {
                drafter.reset();
            }
            if let Some(boundary) = self.boundary.as_mut() {
                boundary.position = None;
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
        self.state
            .grow(&self.shared, tokens.len())
            .map_err(|error| BackendError::new("context", error))?;
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
        // From the first forwarded row the checkpoint no longer holds a prompt's end; it is
        // taken again once this prompt's last row is in.
        if let Some(boundary) = self.boundary.as_mut() {
            boundary.position = None;
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
        if let Some(boundary) = self.boundary.as_mut() {
            boundary
                .copy(&self.shared, &self.state, self.drafter.as_ref(), false)
                .map_err(Self::fail)?;
            boundary.position = Some(tokens.len());
            boundary.images.clone_from(&self.held_images);
        }
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
                DraftState::allocate(shared, model, &state.draft_rings())
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
        let mut session = Self {
            drafter,
            boundary: None,
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
            opening: None,
            speculator: Speculator::new(
                shared.draft.map_or(0, |(_, depth)| depth),
                shared.depth_policy.clone(),
            ),
        };
        // Opened last, so a refusal drops the session and gives back everything before it.
        if shared.rewind {
            session.boundary = Some(
                Boundary::allocate(shared, session.drafter.as_ref())
                    .map_err(|error| BackendError::new(DEVICE, error))?,
            );
        }
        Ok(session)
    }

    /// Where a prompt that does not extend the held history may rewind to: the last prompt's end
    /// ([ModelSession::rewind_prefix]), when `tokens` keeps that prompt and goes past it and
    /// carries its images where it held them.
    fn rewind_point(&self, tokens: &[u32], images: &[VisionImage]) -> Option<usize> {
        let prefix = self.rewind_prefix()?;
        let point = prefix.len();
        let boundary = self.boundary.as_ref()?;
        (point < tokens.len()
            && tokens.starts_with(prefix)
            && boundary
                .images
                .iter()
                .all(|span| images.iter().any(|image| image.span == *span))
            && images
                .iter()
                .all(|image| image.span.start >= point || boundary.images.contains(&image.span)))
        .then_some(point)
    }

    /// Takes the sequence back to the last prompt's end ([Self::rewind_point]): its histories
    /// and the drafter's feature from the checkpoint, its position and history to the boundary.
    fn rewind_to(&mut self, point: usize) -> Result<(), BackendError> {
        let boundary = self
            .boundary
            .as_ref()
            .filter(|boundary| boundary.position == Some(point))
            .ok_or_else(|| Self::fail(format!("no checkpoint at position {point}")))?;
        boundary
            .copy(&self.shared, &self.state, self.drafter.as_ref(), true)
            .map_err(Self::fail)?;
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.rewind(point).map_err(Self::fail)?;
        }
        self.state.position = point;
        self.history.truncate(point);
        self.held_images = boundary.images.clone();
        self.last_logits.clear();
        Ok(())
    }

    /// Forget the sequence, keeping its allocations: the next prompt starts at position zero.
    fn forget(&mut self) -> Result<(), BackendError> {
        self.state
            .reset(&self.shared)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        if let Some(boundary) = self.boundary.as_mut() {
            boundary.position = None;
        }
        self.round = None;
        self.opening = None;
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

    /// Whether the last verify holds the row a step fed `token` takes: its next draft.
    fn held(&self, token: u32) -> bool {
        self.round
            .as_ref()
            .is_some_and(|round| round.chunk.get(round.fed) == Some(&token))
    }

    /// Rolls back the last `count` drafts the caller was answered and never used
    /// ([FusedArm::give_back]): the history, the sampler and the round's fed rows go back to
    /// where they stood, so the round settles at the caller's last token.
    fn give_back(&mut self, count: usize) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        let round = self
            .round
            .as_mut()
            .filter(|round| round.marks.len() >= count)
            .ok_or_else(|| {
                Self::fail(format!(
                    "{count} answers to give back, and the round holds fewer"
                ))
            })?;
        for _ in 0..count {
            let mark = round.marks.pop().expect("the marks were just counted");
            self.sampler.rewind(mark);
            self.history.pop();
            round.fed -= 1;
        }
        self.last_logits = round.rows[round.fed - 1].clone();
        Ok(())
    }

    /// Opens this sequence's round on `token` for a shared walk ([RoundArm::open]): the step the
    /// last verify holds, or the last round settled and the next chain drafted whole, with the
    /// chance each leading run of it is accepted. The walk's share of the chain comes later
    /// ([Self::close]).
    fn open_round(&mut self, token: u32) -> Result<Opened, BackendError> {
        if self.held(token) {
            return self.decode(token).map(Opened::Held);
        }
        self.settle()?;
        self.opening = None;
        if self.state.position >= self.shared.context {
            return Err(BackendError::new("context", "the context is exhausted"));
        }
        if token as usize >= self.shared.config.vocab_size {
            return Err(Self::fail(format!(
                "token {token} is outside the vocabulary {}",
                self.shared.config.vocab_size
            )));
        }
        let rows = self.state.position + 1 + self.shared.draft.map_or(0, |(_, depth)| depth);
        self.state
            .grow(&self.shared, rows)
            .map_err(|error| BackendError::new("context", error))?;
        if self.verify.is_none() {
            let rows = self.speculator.max_depth() + 1;
            self.verify = Some(
                VerifyBuffer::allocate(&self.shared, rows, &self.state.conv).map_err(Self::fail)?,
            );
        }
        let (drafts, confidence) = self.draft_chain(token)?;
        let room = self.shared.context.saturating_sub(self.state.position + 1);
        let mut chain = self.speculator.chain(&confidence, room);
        chain.truncate(drafts.len());
        self.opening = Some(Opening {
            start: self.state.position,
            anchor: token,
            drafts,
            confidence,
        });
        Ok(Opened::Round(chain))
    }

    /// The opened round's walk ran, verifying its anchor and the first `depth` drafts into
    /// `rows`: the round takes them, and the anchor's step is answered.
    fn close(&mut self, depth: usize, rows: Vec<Vec<f32>>) -> Result<StepOutput, BackendError> {
        let opening = self
            .opening
            .take()
            .ok_or_else(|| Self::fail("a shared walk closed a round nobody opened"))?;
        let mut chunk = Vec::with_capacity(depth + 1);
        chunk.push(opening.anchor);
        chunk.extend_from_slice(&opening.drafts[..depth.min(opening.drafts.len())]);
        if rows.len() != chunk.len() {
            return Err(Self::fail(format!(
                "a walk of {} rows answered {}",
                chunk.len(),
                rows.len()
            )));
        }
        let logits = rows[0].clone();
        self.round = Some(Round {
            start: opening.start,
            chunk,
            confidence: opening.confidence,
            rows,
            fed: 1,
            marks: Vec::new(),
        });
        Ok(self.decoded(opening.anchor, logits))
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

    /// The drafts to verify after `anchor`, and the probability the drafter gave each draft it
    /// made: none without a drafter, never more than the context has room for past the anchor,
    /// and of DSpark's chain the leading drafts the depth policy picks from those probabilities.
    fn draft(&mut self, anchor: u32) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
        let (mut drafts, confidence) = self.draft_chain(anchor)?;
        if matches!(self.shared.draft, Some((Drafter::Dspark, _))) {
            let room = self.shared.context.saturating_sub(self.state.position + 1);
            drafts.truncate(self.speculator.depth(&confidence, room));
        }
        Ok((drafts, confidence))
    }

    /// The drafts after `anchor` before any depth is picked: DSpark's whole chain, with the
    /// probability it gave each draft, or prompt lookup's drafts at the policy's depth.
    fn draft_chain(&mut self, anchor: u32) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
        let Some((drafter, _)) = self.shared.draft else {
            return Ok((Vec::new(), Vec::new()));
        };
        let room = self.shared.context.saturating_sub(self.state.position + 1);
        match drafter {
            Drafter::Lookup => {
                let depth = self.speculator.depth(&[], room);
                let mut context = Vec::with_capacity(self.history.len() + 1);
                context.extend_from_slice(&self.history);
                context.push(anchor);
                let (shortest, longest) = LOOKUP_NGRAMS;
                Ok((lookup_draft(&context, shortest, longest, depth), Vec::new()))
            }
            Drafter::Dspark => {
                // The whole chain is drafted before the depth is known, since the depth comes from
                // the drafts' probabilities: a Markov step past the depth costs about 0.2 ms, a
                // verified row about 9.6 ms.
                let chain = self.speculator.max_depth().min(room);
                if chain == 0 {
                    return Ok((Vec::new(), Vec::new()));
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
                // A sampled request's drafts take the noise its sample takes.
                let noise = crate::host::gumbel_sampler(&self.sampling).map(|gumbel| {
                    let (inv_t, seed) = gumbel.draft_noise();
                    (f32::from_bits(inv_t as u32), seed)
                });
                drafter.draft(&shared, model, anchor, chain.min(model.max_depth()), noise)
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
            self.verify = Some(
                VerifyBuffer::allocate(&self.shared, rows, &self.state.conv).map_err(Self::fail)?,
            );
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
    /// and the position go to the last fed row. The rows past it stay in the cache, past the
    /// position, where the next rows stored overwrite them before any row reads them. The verify
    /// left the histories as they were before it, so every one is restored here from the tails,
    /// all of them in one launch, whether the caller fed every row or fewer.
    fn settle(&mut self) -> Result<(), BackendError> {
        let Some(round) = self.round.take() else {
            return Ok(());
        };
        self.speculator
            .observe(&round.confidence, round.chunk.len() - 1, round.fed);
        // The verify left every row's taps; the drafter keeps the rows the caller fed.
        self.inject(round.fed, round.start)?;
        let verify = self
            .verify
            .as_ref()
            .ok_or_else(|| Self::fail("a round settled with no verify buffer"))?;
        let entries = u32::try_from(verify.entries)
            .ok()
            .filter(|entries| (1..=MAX_GRID_Y).contains(entries))
            .ok_or_else(|| Self::fail(format!("{} histories to restore", verify.entries)))?;
        let shared = &self.shared;
        let widest = verify.tails.channels(0).max(verify.tails.channels(2));
        shared
            .launch(
                shared.kernels.conv_restore,
                (flat(widest * verify.tails.history), entries, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    verify.restore.0,
                    u64::from(entries),
                    shared.kernel_width() as u64,
                    round.fed as u64,
                ],
            )
            .map_err(Self::fail)?;
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
                            self.state.scratch.residual_stats,
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
                            run(shared.gemv_fused(&parts[..fused], input, columns, 1))?;
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
                            self.state.scratch.residual_stats,
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
                        // so the sum over ranks is the layer's whole expert contribution; under
                        // the row split each rank mixed its own rows, which the gather joins.
                        band_exchange(&shared, &self.state.scratch, 1)?;
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
                        self.state.scratch.residual_stats,
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
                        (flat(unpadded), 1, 1),
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
        band_mix(shared, &self.state.scratch, 1)?;
        if trace_on() && layer == 0 {
            eprintln!("[inkling] row router layer 0 slots {selected:?} shares {shares:?}");
            self.rms(shared, "row band", self.state.scratch.mixed, hidden)?;
        }
        Ok(())
    }
    /// The routed band of a quantised layer with the router's choice left on the device: the
    /// one-row case of [chunk::run_steered_band], which a verify's rows and small fused lanes
    /// share.
    fn run_device_band(
        &mut self,
        shared: &Shared,
        layer: usize,
        gate_up: &Step,
        down: &Step,
    ) -> Result<(), BackendError> {
        let scratch = &self.state.scratch;
        chunk::run_steered_band(shared, scratch, layer, gate_up, down, scratch.normed, 1)?;
        if trace_on() && layer == 0 && !BandOut::of(shared, scratch, 1).gathers {
            self.rms(shared, "row band", scratch.mixed, shared.config.hidden_size)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationPlanes, AttentionWalk, ChunkWidths, DEFAULT_PREFILL_CHUNK, DRAFT_MIN_DEPTH_ENV,
        DRAFT_POLICY_ENV, DSPARK_STEP_COST, DepthPolicy, Drafter, LONG_PREFILL_CHUNK,
        LONG_PREFILL_FROM, Mxfp8Walk, PrefillWalk, depth_policy_from_env, mma_attention_shared,
        overlap_request, residual_chain_request, rewind_request,
    };
    use crate::config::Inkling2Config;

    #[test]
    fn the_mxfp8_walk_is_the_mma_under_the_tensor_walk_alone() {
        for value in [None, Some(""), Some("mma")] {
            assert_eq!(
                Mxfp8Walk::parse(value, PrefillWalk::Tensor),
                Ok(Mxfp8Walk::Mma)
            );
        }
        assert_eq!(
            Mxfp8Walk::parse(Some("pairs"), PrefillWalk::Tensor),
            Ok(Mxfp8Walk::Pairs)
        );
        // The exact walks keep the pair walk, and naming the MMA there is refused, not ignored.
        for prefill in [PrefillWalk::Chunk, PrefillWalk::Rows, PrefillWalk::Gemm] {
            for value in [None, Some(""), Some("pairs")] {
                assert_eq!(Mxfp8Walk::parse(value, prefill), Ok(Mxfp8Walk::Pairs));
            }
            assert!(Mxfp8Walk::parse(Some("mma"), prefill).is_err());
        }
        assert!(Mxfp8Walk::parse(Some("tensor"), PrefillWalk::Tensor).is_err());
    }

    #[test]
    fn the_head_only_operand_is_opt_in_under_the_tensor_walk_alone() {
        for prefill in [
            PrefillWalk::Tensor,
            PrefillWalk::Chunk,
            PrefillWalk::Rows,
            PrefillWalk::Gemm,
        ] {
            for value in [None, Some(""), Some("both")] {
                assert_eq!(
                    ActivationPlanes::parse(value, prefill),
                    Ok(ActivationPlanes::Both)
                );
            }
        }
        assert_eq!(
            ActivationPlanes::parse(Some("head"), PrefillWalk::Tensor),
            Ok(ActivationPlanes::Head)
        );
        // Where no MMA band runs, naming the head is refused, not ignored.
        for prefill in [PrefillWalk::Chunk, PrefillWalk::Rows, PrefillWalk::Gemm] {
            assert!(ActivationPlanes::parse(Some("head"), prefill).is_err());
        }
        assert!(ActivationPlanes::parse(Some("bf16"), PrefillWalk::Tensor).is_err());
        assert_eq!(ActivationPlanes::Both.count(), 2);
        assert_eq!(ActivationPlanes::Head.count(), 1);
    }

    #[test]
    fn the_band_overlap_is_on_unless_turned_off_and_refuses_other_values() {
        assert_eq!(overlap_request(None), Ok(None));
        assert_eq!(overlap_request(Some("")), Ok(None));
        assert_eq!(overlap_request(Some("on")), Ok(Some(true)));
        assert_eq!(overlap_request(Some("off")), Ok(Some(false)));
        for value in ["1", "0", "halves", "ON"] {
            assert!(overlap_request(Some(value)).is_err(), "{value}");
        }
    }

    #[test]
    fn the_on_by_default_switches_turn_off_only_by_name_and_refuse_other_values() {
        // Over Inkling-2-Small's stack: 48 layers in blocks of 6.
        fn chain(value: Option<&str>) -> Result<bool, String> {
            residual_chain_request(value, 8)
        }
        for request in [rewind_request, chain] {
            for value in [None, Some(""), Some("on")] {
                assert_eq!(request(value), Ok(true), "{value:?}");
            }
            assert_eq!(request(Some("off")), Ok(false));
            for value in ["1", "0", "yes", "OFF"] {
                let refusal = request(Some(value)).unwrap_err();
                assert!(
                    refusal.contains("use on") && refusal.contains("or off"),
                    "{refusal}"
                );
            }
        }
    }

    #[test]
    fn the_chained_residual_stands_down_unset_and_refuses_on_past_its_stack() {
        for value in [None, Some(""), Some("off")] {
            assert_eq!(residual_chain_request(value, 9), Ok(false), "{value:?}");
        }
        let refusal = residual_chain_request(Some("on"), 9).unwrap_err();
        assert!(refusal.contains("unset it or set it off"), "{refusal}");
        assert_eq!(residual_chain_request(Some("on"), 8), Ok(true));
    }

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

    /// The verify-depth policy's switches as `pairs` set them, and nothing else.
    fn policy_with(
        draft: Option<(Drafter, usize)>,
        pairs: &[(&str, &str)],
    ) -> Result<Option<DepthPolicy>, String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        depth_policy_from_env(draft, |name| {
            pairs
                .iter()
                .find(|(set, _)| set == name)
                .map(|(_, value)| value.clone())
        })
    }

    #[test]
    fn a_dspark_round_picks_its_depth_unless_told_to_verify_every_draft() {
        let dspark = Some((Drafter::Dspark, 7));
        let adaptive = policy_with(dspark, &[]).unwrap().unwrap();
        assert!(adaptive.is_adaptive());
        assert_eq!(adaptive.cost(), DSPARK_STEP_COST);
        let fixed = policy_with(dspark, &[(DRAFT_POLICY_ENV, "fixed")])
            .unwrap()
            .unwrap();
        assert!(!fixed.is_adaptive());
        assert_eq!(fixed.depth(&[0.0; 7]), 7);
        // A floor holds however unlikely the drafts.
        let floored = policy_with(dspark, &[(DRAFT_MIN_DEPTH_ENV, "2")])
            .unwrap()
            .unwrap();
        assert!(floored.depth(&[0.0; 7]) >= 2);
        // An empty value is unset.
        assert!(
            policy_with(dspark, &[(DRAFT_POLICY_ENV, "")])
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_depth_switch_the_round_cannot_honour_is_refused() {
        // A value outside the switch's domain.
        assert!(policy_with(Some((Drafter::Dspark, 7)), &[(DRAFT_POLICY_ENV, "always")]).is_err());
        assert!(policy_with(Some((Drafter::Dspark, 7)), &[(DRAFT_MIN_DEPTH_ENV, "two")]).is_err());
        // A floor past the drafts a round makes.
        assert!(policy_with(Some((Drafter::Dspark, 3)), &[(DRAFT_MIN_DEPTH_ENV, "4")]).is_err());
        // No other drafter gives its drafts a probability, and no drafter drafts nothing.
        for draft in [None, Some((Drafter::Lookup, 7))] {
            assert_eq!(policy_with(draft, &[]), Ok(None));
            for name in [DRAFT_POLICY_ENV, DRAFT_MIN_DEPTH_ENV] {
                let refused = policy_with(draft, &[(name, "1")]).unwrap_err();
                assert!(refused.contains(name), "{refused}");
            }
        }
    }
}
