// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Inkling-2-Small's DSpark drafter on the device: the checkpoint's `mtp.safetensors`, a block
//! of eight rows through five bidirectional layers over the target's fused taps, then the
//! `rnn_hidden` Markov head that turns the block's rows into up to seven drafts. The forward is
//! written up in docs/models/inkling2-small-drafter.md; the kernels are
//! kernels/inkling_draft_sm121.cu and DFlash 2's block attention.
//!
//! # Taps and context
//!
//! Every committed target row carries eight taps ([TAPS]): the in-block partial sum before each
//! of seven target layers and the post-final-norm hidden. The walks leave them in the session's
//! tap buffer ([TapSink]); [DraftState::inject] then mixes each row's taps into every draft
//! layer's context keys and values (DFlare's fused features and context projections) and keeps
//! the last row's sixth fusion for the Markov head. A prompt injects its chunks as they are
//! walked, a plain step its row, and a verify only the rows the caller fed, when the round
//! settles.
//!
//! # Sharding
//!
//! The block is tensor-parallel like the target: each rank holds its share of the query, key,
//! value, gate and up rows and of the output and down columns, and the two sums a layer are the
//! group's. The head is the target's own vocabulary slice, the Markov head's fold is computed
//! against it, and each step's greedy pick is the best of the ranks' slices.
//!
//! Attribution: the forward follows SGLang's Inkling 2 drafter (github.com/sgl-project/sglang,
//! branch inkling-2-small-support at 778fb834e745bffd9a303ff388435fb44b644746, Apache-2.0,
//! Copyright 2023-2024 SGLang Team: python/sglang/srt/models/inkling2_mtp.py); no source is
//! copied. The method is DFlash's block drafting (MIT, Copyright (c) 2026 Z Lab), DFlare's layer
//! fusion (arXiv:2606.02091) and DSpark's Markov head (arXiv:2607.05147). See the root NOTICE.

use std::path::Path;

use sparkflash_kernels::images::dflash2_cubin;
use sparkflash_ops::dflash::{DraftUploader, Part, single_file_source};
use sparkflash_weights::quant::f32_to_bf16_bits;

use super::chunk::{device_error, project};
use super::*;

/// The drafter's file in the checkpoint directory; the checkpoint's index does not list it.
pub const DRAFT_FILE: &str = "mtp.safetensors";

/// A window for the drafter's full-attention layers: unset or `full` (the default) reads the
/// whole context, a count of positions reads only that many before the block, as a local layer
/// does. The drafter only proposes tokens, so the window moves acceptance, never an emitted
/// token. Anything else is refused.
pub(crate) const DRAFT_WINDOW_ENV: &str = "SPARKFLASH_INKLING_DRAFT_WINDOW";

/// Drafter weight quantization: unset or `nvfp4` (the default) quantizes the drafter's BF16
/// projections, LM head slice, and Markov fold to per-row NVFP4 at load time; `bf16` or `off`
/// keeps them in BF16. Because the drafter only proposes tokens and the target verifies every
/// token with exact weights, this moves draft speed and acceptance, never emitted tokens.
const DRAFT_QUANT_ENV: &str = "SPARKFLASH_INKLING_DRAFT_QUANT";

fn draft_quant_nvfp4(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("" | "nvfp4" | "on" | "1") => Ok(true),
        Some("bf16" | "off" | "0") => Ok(false),
        Some(text) => Err(format!(
            "{DRAFT_QUANT_ENV}={text} is not a drafter quantization mode: use nvfp4 (default) or bf16"
        )),
    }
}

/// Pre-swizzled MMA-tiled per-row NVFP4 matrix for `inkling_draft_nvfp4_gemv(3)`.
#[derive(Clone, Copy)]
struct Nvfp4Tiled {
    codes: Deviceptr,
    scales: Deviceptr,
    row_scale: Deviceptr,
}

/// Row-major per-row NVFP4 matrix for `inkling_draft_nvfp4_markov_logits`.
#[derive(Clone, Copy)]
struct Nvfp4Row {
    codes: Deviceptr,
    scales: Deviceptr,
    row_scale: Deviceptr,
}

/// The full layers' window [DRAFT_WINDOW_ENV] names; `None` reads the whole context.
fn full_window(value: Option<&str>) -> Result<Option<usize>, String> {
    match value {
        None | Some("" | "full") => Ok(None),
        Some(text) => match text.parse::<usize>() {
            Ok(window) if window > 0 => Ok(Some(window)),
            _ => Err(format!(
                "{DRAFT_WINDOW_ENV}={text} is not a window: use full (the default: the drafter's \
                 full layers read the whole context) or a positive count of positions"
            )),
        },
    }
}

// The taps and fusion rows are the checkpoint's: `mtp_config.target_layer_ids` names one tap a
// pre-layer partial plus the post-final-norm hidden, and `fuse_weight` carries one fusion row a
// draft layer plus the Markov head's. Inkling-2-Small has eight taps and six rows; Inkling-2-Mini
// has four and four, and the kernels take the count at run time.
/// The context rows one injection pass mixes and projects at once.
const INJECT_ROWS: usize = 1024;
/// The drafter's head width, which the attention and the rotary assume.
const HEAD_DIM: usize = 128;
/// Threads of a norm, mix-norm, gate or pick block (a multiple of 32, and hidden <= 32 a thread).
const BLOCK: u32 = 256;
/// The contiguous runs the block attention splits a span's keys into, a block each
/// (q4d_attn_span_chunk). 48 measured as fast as 96 or 192 at a 93k span and fastest at a local
/// layer's, on Inkling's shapes at TP4.
const ATTENTION_CHUNKS: usize = 48;
/// The floats one run leaves for the merge (Q4D_ATTN_PART).
const ATTENTION_PART: usize = HEAD_DIM + 4;
/// The queries a warp of the attention's first stage holds (Q4D_SPAN_QUERIES).
const ATTENTION_QUERIES: usize = 8;
/// The most queries of a key head one block holds: a warp's worth per 32 threads, 1,024 threads.
const ATTENTION_MAX_QUERIES: usize = ATTENTION_QUERIES * 32;
/// The logits one step-logits thread scores (IKD_LOGIT_PER_THREAD).
const LOGIT_PER_THREAD: usize = 8;
/// The floats a rank's pick slot takes (IKD_PICK_SLOT).
const PICK_SLOT: usize = 4;

/// The drafter's kernels.
struct DraftKernels {
    capture: Bound,
    mix_norm: Bound,
    masks: Bound,
    round: Bound,
    norm: Bound,
    add_round: Bound,
    swiglu: Bound,
    rope: Bound,
    context: Bound,
    markov_input: Bound,
    markov_gates: Bound,
    markov_logits: Bound,
    markov_local: Bound,
    markov_last: Bound,
    quantize_nvfp4_tiled: Bound,
    quantize_nvfp4_row: Bound,
    nvfp4_gemv: Bound,
    nvfp4_gemv3: Bound,
    nvfp4_markov_logits: Bound,
    /// DFlash 2's block attention over one span of the rings: every query's walks, and their
    /// merge.
    attention_part: Bound,
    attention_part_tc: Bound,
    attention_merge: Bound,
}

// SAFETY: as for the target's Kernels: driver function handles, launched only on the process's
// default stream, so moving or sharing them between threads cannot race a launch.
unsafe impl Send for DraftKernels {}
// SAFETY: as above.
unsafe impl Sync for DraftKernels {}

impl DraftKernels {
    fn load(cuda: &Cuda) -> Result<Self, String> {
        let inkling = inkling_cubin();
        let dflash = dflash2_cubin();
        if dflash.is_empty() {
            return Err(
                "this build has no DFlash 2 cubin: the drafter's block attention is \
                 kernels/dflash2_sm121.cu's q4d_attn_span_chunk and q4d_attn_span_merge"
                    .to_owned(),
            );
        }
        let load = |image: Image,
                    cubin: &'static [u8],
                    name: &'static str,
                    precision: Precision|
         -> Result<Bound, String> {
            Bound::new(cuda, image, cubin, name, precision)
        };
        Ok(Self {
            capture: load(
                Image::Inkling,
                inkling,
                "inkling_draft_capture",
                Precision::F32,
            )?,
            mix_norm: load(
                Image::Inkling,
                inkling,
                "inkling_draft_mix_norm",
                Precision::Bf16,
            )?,
            masks: load(
                Image::Inkling,
                inkling,
                "inkling_draft_masks",
                Precision::Bf16,
            )?,
            round: load(
                Image::Inkling,
                inkling,
                "inkling_draft_round",
                Precision::F32,
            )?,
            norm: load(
                Image::Inkling,
                inkling,
                "inkling_draft_norm",
                Precision::F32,
            )?,
            add_round: load(
                Image::Inkling,
                inkling,
                "inkling_draft_add_round",
                Precision::F32,
            )?,
            swiglu: load(
                Image::Inkling,
                inkling,
                "inkling_draft_swiglu",
                Precision::F32,
            )?,
            rope: load(
                Image::Inkling,
                inkling,
                "inkling_draft_rope",
                Precision::F32,
            )?,
            context: load(
                Image::Inkling,
                inkling,
                "inkling_draft_context_bf16",
                Precision::F32,
            )?,
            markov_input: load(
                Image::Inkling,
                inkling,
                "inkling_draft_markov_input",
                Precision::F32,
            )?,
            markov_gates: load(
                Image::Inkling,
                inkling,
                "inkling_draft_markov_gates",
                Precision::F32,
            )?,
            markov_logits: load(
                Image::Inkling,
                inkling,
                "inkling_draft_markov_logits",
                Precision::F32,
            )?,
            markov_local: load(
                Image::Inkling,
                inkling,
                "inkling_draft_markov_local",
                Precision::F32,
            )?,
            markov_last: load(
                Image::Inkling,
                inkling,
                "inkling_draft_markov_last",
                Precision::F32,
            )?,
            quantize_nvfp4_tiled: load(
                Image::Inkling,
                inkling,
                "inkling_draft_quantize_nvfp4_tiled",
                Precision::NvFp4,
            )?,
            quantize_nvfp4_row: load(
                Image::Inkling,
                inkling,
                "inkling_draft_quantize_nvfp4_row",
                Precision::NvFp4,
            )?,
            nvfp4_gemv: load(
                Image::Inkling,
                inkling,
                "inkling_draft_nvfp4_gemv",
                Precision::NvFp4,
            )?,
            nvfp4_gemv3: load(
                Image::Inkling,
                inkling,
                "inkling_draft_nvfp4_gemv3",
                Precision::NvFp4,
            )?,
            nvfp4_markov_logits: load(
                Image::Inkling,
                inkling,
                "inkling_draft_nvfp4_markov_logits",
                Precision::NvFp4,
            )?,
            attention_part: load(
                Image::Dflash2,
                dflash,
                "q4d_attn_span_chunk_bf16",
                Precision::Bf16,
            )?,
            attention_part_tc: Bound::shared(
                cuda,
                Image::Dflash2,
                dflash,
                "q4d_attn_span_chunk_bf16_tc",
                Precision::Bf16,
                49_408,
            )?,
            attention_merge: load(
                Image::Dflash2,
                dflash,
                "q4d_attn_span_merge",
                Precision::Bf16,
            )?,
        })
    }
}

/// One draft layer's share of the weights on this rank.
struct DraftLayer {
    input_norm: Deviceptr,
    post_norm: Deviceptr,
    /// The norm over this layer's fused context feature.
    fusion_norm: Deviceptr,
    query: Deviceptr,
    key: Deviceptr,
    value: Deviceptr,
    context_key: Deviceptr,
    context_value: Deviceptr,
    out: Deviceptr,
    query_norm: Deviceptr,
    key_norm: Deviceptr,
    context_key_norm: Deviceptr,
    gate: Deviceptr,
    up: Deviceptr,
    down: Deviceptr,
    query_q: Option<Nvfp4Tiled>,
    key_q: Option<Nvfp4Tiled>,
    value_q: Option<Nvfp4Tiled>,
    out_q: Option<Nvfp4Tiled>,
    gate_q: Option<Nvfp4Tiled>,
    up_q: Option<Nvfp4Tiled>,
    down_q: Option<Nvfp4Tiled>,
    /// A local layer's window, or a full layer's under [DRAFT_WINDOW_ENV]; `None` for a full
    /// layer that reads the whole context.
    window: Option<usize>,
}

/// The drafter's weights and geometry on this rank: loaded once ([DraftModel::load]) and shared
/// by the rank's sessions.
pub(super) struct DraftModel {
    kernels: DraftKernels,
    layers: Vec<DraftLayer>,
    /// `bf16(softmax(fuse_weight))`, [fusion_rows][taps].
    alpha: Deviceptr,
    mask: Deviceptr,
    norm: Deviceptr,
    /// The Markov head's token table `markov_w1`, [vocabulary][rank], whole on every rank.
    markov_embed: Deviceptr,
    rnn_weight: Deviceptr,
    rnn_q: Option<Nvfp4Tiled>,
    rnn_bias: Deviceptr,
    /// The fold against this rank's head slice: `bf16(W_lm W_ho)` [rank_vocab][rank] and
    /// `bf16(W_lm b_ho)` [rank_vocab].
    fold_weight: Deviceptr,
    fold_q: Option<Nvfp4Row>,
    fold_bias: Deviceptr,
    markov_norm: Deviceptr,
    /// The target layers whose pre-layer partials are taps 0..taps-1.
    tap_layers: Vec<usize>,
    /// The taps a committed row carries, the post-final hidden included.
    taps: usize,
    /// The fusion rows: one a draft layer, then the Markov head's.
    fusion_rows: usize,
    /// The target's embedding table, embedding norm and head (this rank's slices).
    embed_table: Deviceptr,
    embed_norm: Deviceptr,
    head: Deviceptr,
    head_q: Option<Nvfp4Tiled>,
    block: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    rank: usize,
    eps: f32,
    theta: f32,
    /// The matrices only the GEMV walks and the prompt tiles read, as (slot, rows, columns), that
    /// the rank packs in place once they are loaded, as it packs the target's: the layers'
    /// projections but the block's key and value, the Markov head's input projection and its
    /// fold ([Self::dense_matrices]).
    dense: Vec<(Deviceptr, usize, usize)>,
    allocations: Vec<Deviceptr>,
    bytes: usize,
}

/// A walk's sink for the drafter's taps: the tap buffer, `[rows][taps][hidden]` BF16, and the
/// target layers whose pre-layer partials it keeps.
#[derive(Debug, Clone)]
pub(super) struct TapSink {
    capture: Bound,
    taps: Deviceptr,
    layers: Vec<usize>,
    /// The taps a row carries, the post-final hidden included.
    per_row: usize,
    /// The rows the buffer holds.
    rows: usize,
}

// SAFETY: the capture entry is a driver function handle launched on the default stream, and the
// rest are device addresses and counts.
unsafe impl Send for TapSink {}
// SAFETY: as above.
unsafe impl Sync for TapSink {}

impl TapSink {
    /// The tap slot of the post-final-norm hidden.
    pub(super) fn final_slot(&self) -> usize {
        self.per_row - 1
    }

    /// The slot of the tap taken before target layer `layer`, if any.
    pub(super) fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|tap| *tap == layer)
    }

    /// Tap `slot` of `rows` rows of `stream` (f32 rows `hidden` apart), rounded to BF16.
    pub(super) fn capture(
        &self,
        shared: &Shared,
        stream: Deviceptr,
        rows: usize,
        slot: usize,
    ) -> Result<(), BackendError> {
        if rows > self.rows {
            return Err(device_error(format!(
                "{rows} rows of taps overflow the drafter's {}-row tap buffer",
                self.rows
            )));
        }
        let hidden = shared.hidden();
        shared
            .launch(
                self.capture,
                (grid_blocks(hidden, BLOCK as usize), rows as u32, 1),
                (BLOCK, 1, 1),
                &[
                    stream.0,
                    self.taps.0,
                    hidden as u64,
                    hidden as u64,
                    self.per_row as u64,
                    slot as u64,
                ],
            )
            .map_err(device_error)
    }
}

/// The weight index the first `op` step names (at `target`, when one is given), for the drafter's
/// reuse of the target's tensors.
fn step_weight(steps: &[Step], op: Op, target: Option<Target>) -> Option<usize> {
    steps
        .iter()
        .find(|step| step.op == op && target.is_none_or(|target| step.target == target))
        .and_then(|step| step.weights.first().copied())
}

impl DraftModel {
    /// Which layers keep a ring as long as the whole context (a full layer without
    /// [DRAFT_WINDOW_ENV]): a session's lane holds those ([DraftState::allocate]).
    pub(super) fn whole_context_rings(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|layer| layer.window.is_none())
            .collect()
    }

    /// The bytes one position adds to a layer's key ring (and as many to its value ring): BF16.
    pub(super) fn ring_row_bytes(&self) -> usize {
        self.kv_heads * HEAD_DIM * 2
    }

    /// The bytes the Markov feature holds: what a prompt-boundary checkpoint copies of the
    /// sequence's drafter state ([DraftState::feature]).
    pub(super) fn feature_bytes(&self) -> usize {
        self.hidden * std::mem::size_of::<f32>()
    }

    /// Loads the drafter from `directory`'s [DRAFT_FILE] onto this rank: the block's shares, the
    /// fusion weights, the Markov head, and its fold against this rank's head slice.
    ///
    /// # Errors
    /// A checkpoint with no drafter configuration or one this arm has no kernel for, a file that
    /// does not open or lacks a tensor, or a device failure.
    pub(super) fn load(shared: &Shared, directory: &Path) -> Result<Self, String> {
        let config = &shared.config;
        let mtp = config
            .mtp
            .as_ref()
            .ok_or("this checkpoint's config describes no DSpark drafter (mtp_config)")?;
        let world = shared.exchange.world();
        let rank = shared.exchange.rank();
        let hidden = config.hidden_size;
        if mtp.hidden_size != hidden {
            return Err(format!(
                "the drafter is {} wide and the target {hidden}: its taps are the target's rows",
                mtp.hidden_size
            ));
        }
        if mtp.head_dim != HEAD_DIM {
            return Err(format!(
                "the drafter's heads are {} wide; the block attention takes {HEAD_DIM}",
                mtp.head_dim
            ));
        }
        if mtp.block_size < 2 || mtp.block_size > BATCH_ROWS {
            return Err(format!(
                "a {}-row draft block: the block GEMV walks 2..={BATCH_ROWS} rows",
                mtp.block_size
            ));
        }
        if hidden > BLOCK as usize * 32 {
            return Err(format!(
                "a {hidden}-wide row is wider than the mix-norm block holds"
            ));
        }
        for (what, count) in [
            ("query heads", mtp.num_attention_heads),
            ("key/value heads", mtp.num_key_value_heads),
            ("MLP rows", mtp.intermediate_size),
        ] {
            if count == 0 || !count.is_multiple_of(world) {
                return Err(format!(
                    "the drafter's {count} {what} do not split over {world} ranks"
                ));
            }
        }
        let heads = mtp.num_attention_heads / world;
        let kv_heads = mtp.num_key_value_heads / world;
        if !heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "{heads} query heads a rank do not group over {kv_heads} key/value heads"
            ));
        }
        if (heads / kv_heads) * mtp.block_size > ATTENTION_MAX_QUERIES {
            return Err(format!(
                "{} queries of a key head ({heads} query heads over {kv_heads} key/value heads, \
                 {} block rows): the block attention holds at most {ATTENTION_MAX_QUERIES}",
                (heads / kv_heads) * mtp.block_size,
                mtp.block_size
            ));
        }
        let intermediate = mtp.intermediate_size / world;
        let markov = mtp.markov_rank;
        if markov == 0 {
            return Err("the drafter has no Markov head (markov_rank 0)".to_owned());
        }
        // The taps count the `applications` a token runs, not the stored layers: a looped
        // release's last tap is the hidden after its last application, so the count is the
        // schedule's length and the pre-layer taps are application indices.
        let applications = config.applications().len();
        let ids = &mtp.target_layer_ids;
        let taps = ids.len();
        let fusion_rows = mtp.num_hidden_layers + 1;
        if taps < 2 || ids.last() != Some(&applications) {
            return Err(format!(
                "the drafter taps {ids:?}: the list names the pre-layer taps and then the \
                 post-final hidden ({applications})"
            ));
        }
        let mut tap_layers = Vec::with_capacity(taps - 1);
        for layer in &ids[..taps - 1] {
            if !shared
                .plan
                .layers()
                .iter()
                .any(|schedule| schedule.index == *layer)
            {
                return Err(format!("tap layer {layer} is not a layer of this schedule"));
            }
            tap_layers.push(*layer);
        }
        if mtp
            .local_layer_ids
            .iter()
            .any(|layer| *layer >= mtp.num_hidden_layers)
        {
            return Err(format!(
                "local layers {:?} outside the drafter's {}",
                mtp.local_layer_ids, mtp.num_hidden_layers
            ));
        }
        if !mtp.local_layer_ids.is_empty() && mtp.sliding_window_size == 0 {
            return Err("the drafter's local layers have no window".to_owned());
        }
        let full_window = full_window(std::env::var(DRAFT_WINDOW_ENV).ok().as_deref())?;
        let use_nvfp4 = draft_quant_nvfp4(std::env::var(DRAFT_QUANT_ENV).ok().as_deref())?;
        let prologue = shared.plan.prologue();
        let embed_table =
            step_weight(prologue, Op::Embed, None).ok_or("the schedule embeds no token")?;
        let embed_norm = step_weight(
            prologue,
            Op::RmsNorm,
            Some(Target::Norm(NormRole::Embedding)),
        )
        .ok_or("the schedule has no embedding norm")?;
        let head = step_weight(
            shared.plan.epilogue(),
            Op::Gemv,
            Some(Target::Project(Projection::Head)),
        )
        .ok_or("the schedule has no head")?;
        let head = shared.weights.get(head).pointer;

        let cuda: &Cuda = &shared.cuda;
        let kernels = DraftKernels::load(cuda)?;
        let source = single_file_source(directory, DRAFT_FILE)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        let up = DraftUploader::new(cuda, &source);
        let mut dense = Vec::new();
        let mut allocations = Vec::new();
        let mut temp_bf16 = Vec::new();
        let mut bytes = 0usize;
        let mut quant_bytes = 0usize;
        let quantize_tiled = |bf16_ptr: Deviceptr,
                              rows: usize,
                              cols: usize,
                              allocations: &mut Vec<Deviceptr>,
                              bytes: &mut usize,
                              quant_bytes: &mut usize|
         -> Result<Nvfp4Tiled, String> {
            let row_tiles = rows.div_ceil(16);
            let k_tiles = cols / 64;
            let codes_bytes = row_tiles * k_tiles * 32 * 16;
            let scales_bytes = row_tiles * k_tiles * 16 * 4;
            let rs_bytes = rows * 4;
            let mut tmp = 0usize;
            let codes = cuda.alloc(codes_bytes, &mut tmp)?;
            let scales = cuda.alloc(scales_bytes, &mut tmp)?;
            let row_scale = cuda.alloc(rs_bytes, &mut tmp)?;
            allocations.extend([codes, scales, row_scale]);
            *bytes += codes_bytes + scales_bytes + rs_bytes;
            *quant_bytes += codes_bytes + scales_bytes + rs_bytes;
            shared.launch(
                kernels.quantize_nvfp4_tiled,
                (row_tiles as u32, 1, 1),
                (BLOCK, 1, 1),
                &[
                    bf16_ptr.0,
                    codes.0,
                    scales.0,
                    row_scale.0,
                    rows as u64,
                    cols as u64,
                ],
            )?;
            Ok(Nvfp4Tiled {
                codes,
                scales,
                row_scale,
            })
        };
        let quantize_row = |bf16_ptr: Deviceptr,
                            rows: usize,
                            cols: usize,
                            allocations: &mut Vec<Deviceptr>,
                            bytes: &mut usize,
                            quant_bytes: &mut usize|
         -> Result<Nvfp4Row, String> {
            let codes_bytes = rows * (cols / 2);
            let scales_bytes = rows * (cols / 16);
            let rs_bytes = rows * 4;
            let mut tmp = 0usize;
            let codes = cuda.alloc(codes_bytes, &mut tmp)?;
            let scales = cuda.alloc(scales_bytes, &mut tmp)?;
            let row_scale = cuda.alloc(rs_bytes, &mut tmp)?;
            allocations.extend([codes, scales, row_scale]);
            *bytes += codes_bytes + scales_bytes + rs_bytes;
            *quant_bytes += codes_bytes + scales_bytes + rs_bytes;
            shared.launch(
                kernels.quantize_nvfp4_row,
                (rows as u32, 1, 1),
                (BLOCK, 1, 1),
                &[
                    bf16_ptr.0,
                    codes.0,
                    scales.0,
                    row_scale.0,
                    rows as u64,
                    cols as u64,
                ],
            )?;
            Ok(Nvfp4Row {
                codes,
                scales,
                row_scale,
            })
        };
        let q_width = heads * HEAD_DIM;
        let kv_width = kv_heads * HEAD_DIM;
        let q_total = mtp.num_attention_heads * HEAD_DIM;
        let kv_total = mtp.num_key_value_heads * HEAD_DIM;
        let mut layers = Vec::with_capacity(mtp.num_hidden_layers);
        for layer in 0..mtp.num_hidden_layers {
            let name = |tail: &str| format!("layers.{layer}.{tail}");
            let mut vector = |tail: &str, elements: usize| -> Result<Deviceptr, String> {
                let ptr = up.vector(&name(tail), elements)?;
                allocations.push(ptr);
                bytes += elements * 2;
                Ok(ptr)
            };
            let input_norm = vector("input_layernorm.weight", hidden)?;
            let post_norm = vector("post_attention_layernorm.weight", hidden)?;
            let fusion_norm = vector("dflare_norm.weight", hidden)?;
            let query_norm = vector("self_attn.q_norm.weight", HEAD_DIM)?;
            let key_norm = vector("self_attn.k_norm.weight", HEAD_DIM)?;
            let context_key_norm = vector("self_attn.k_norm_ctx.weight", HEAD_DIM)?;
            let rows = |count: usize| Part::Rows(rank * count, count);
            let columns = |count: usize| Part::Columns(rank * count, count);
            let mut load_part =
                |tail: &str,
                 shape: (usize, usize),
                 part: Part,
                 allow_quant: bool|
                 -> Result<(Deviceptr, Option<Nvfp4Tiled>), String> {
                    let matrix = up.mat_part(&name(tail), shape, part)?;
                    if allow_quant && use_nvfp4 && matrix.columns.is_multiple_of(64) {
                        let q = quantize_tiled(
                            matrix.pointer,
                            matrix.rows,
                            matrix.columns,
                            &mut allocations,
                            &mut bytes,
                            &mut quant_bytes,
                        )?;
                        temp_bf16.push(matrix.pointer);
                        Ok((Deviceptr(0), Some(q)))
                    } else {
                        allocations.push(matrix.pointer);
                        bytes += matrix.rows * matrix.columns * 2;
                        dense.push((matrix.pointer, matrix.rows, matrix.columns));
                        Ok((matrix.pointer, None))
                    }
                };
            let (query, query_q) = load_part(
                "self_attn.q_proj.weight",
                (q_total, hidden),
                rows(q_width),
                true,
            )?;
            let (key, key_q) = load_part(
                "self_attn.k_proj.weight",
                (kv_total, hidden),
                rows(kv_width),
                true,
            )?;
            let (value, value_q) = load_part(
                "self_attn.v_proj.weight",
                (kv_total, hidden),
                rows(kv_width),
                true,
            )?;
            let (context_key, _) = load_part(
                "self_attn.k_proj_ctx.weight",
                (kv_total, hidden),
                rows(kv_width),
                false,
            )?;
            let (context_value, _) = load_part(
                "self_attn.v_proj_ctx.weight",
                (kv_total, hidden),
                rows(kv_width),
                false,
            )?;
            let (out, out_q) = load_part(
                "self_attn.o_proj.weight",
                (hidden, q_total),
                columns(q_width),
                true,
            )?;
            let (gate, gate_q) = load_part(
                "mlp.gate_proj.weight",
                (mtp.intermediate_size, hidden),
                rows(intermediate),
                true,
            )?;
            let (up_w, up_q) = load_part(
                "mlp.up_proj.weight",
                (mtp.intermediate_size, hidden),
                rows(intermediate),
                true,
            )?;
            let (down, down_q) = load_part(
                "mlp.down_proj.weight",
                (hidden, mtp.intermediate_size),
                columns(intermediate),
                true,
            )?;
            layers.push(DraftLayer {
                input_norm,
                post_norm,
                fusion_norm,
                query,
                key,
                value,
                context_key,
                context_value,
                out,
                query_norm,
                key_norm,
                context_key_norm,
                gate,
                up: up_w,
                down,
                query_q,
                key_q,
                value_q,
                out_q,
                gate_q,
                up_q,
                down_q,
                window: if mtp.local_layer_ids.contains(&layer) {
                    Some(mtp.sliding_window_size)
                } else {
                    full_window
                },
            });
        }
        // The fusion rows' softmax over the taps, in fp32 and rounded to BF16 as the reference
        // keeps them.
        let fuse = up.read("fuse_weight", fusion_rows * taps * 2)?;
        let fuse: Vec<f32> = fuse
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
            .collect();
        let mut alpha_buf = Vec::with_capacity(fusion_rows * taps * 2);
        for row in fuse.chunks(taps) {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (value - top).exp()).collect();
            let total: f32 = exps.iter().sum();
            for value in exps {
                alpha_buf.extend_from_slice(&f32_to_bf16_bits(value / total).to_le_bytes());
            }
        }
        let alpha = up.bytes(&alpha_buf)?;
        allocations.push(alpha);
        bytes += alpha_buf.len();
        let mask = up.vector("mask_embed", hidden)?;
        allocations.push(mask);
        bytes += hidden * 2;
        let norm = up.vector("norm.weight", hidden)?;
        allocations.push(norm);
        bytes += hidden * 2;
        let markov_norm = up.vector("markov_head.dflare_norm.weight", hidden)?;
        allocations.push(markov_norm);
        bytes += hidden * 2;
        let vocabulary = config.vocab_size;
        let markov_embed = up.mat("markov_head.markov_w1.weight", vocabulary, markov)?.pointer;
        allocations.push(markov_embed);
        bytes += vocabulary * markov * 2;
        let z_width = 2 * markov + 2 * hidden;
        let rnn = up.mat("markov_head.rnn_proj.weight", 3 * markov, z_width)?;
        let (rnn_weight, rnn_q) = if use_nvfp4 && z_width.is_multiple_of(64) {
            let q = quantize_tiled(
                rnn.pointer,
                3 * markov,
                z_width,
                &mut allocations,
                &mut bytes,
                &mut quant_bytes,
            )?;
            temp_bf16.push(rnn.pointer);
            (Deviceptr(0), Some(q))
        } else {
            allocations.push(rnn.pointer);
            bytes += 3 * markov * z_width * 2;
            dense.push((rnn.pointer, 3 * markov, z_width));
            (rnn.pointer, None)
        };
        let rnn_bias = up.vector("markov_head.rnn_proj.bias", 3 * markov)?;
        allocations.push(rnn_bias);
        bytes += 6 * markov;
        let (fold_weight_raw, fold_bias) = fold(shared, &up, head, hidden, markov)?;
        let rank_vocab = shared.plan.rank_vocab();
        let (fold_weight, fold_q) = if use_nvfp4 && markov.is_multiple_of(16) && markov <= 256 {
            let q = quantize_row(
                fold_weight_raw,
                rank_vocab,
                markov,
                &mut allocations,
                &mut bytes,
                &mut quant_bytes,
            )?;
            temp_bf16.push(fold_weight_raw);
            (Deviceptr(0), Some(q))
        } else {
            allocations.push(fold_weight_raw);
            bytes += rank_vocab * markov * 2;
            dense.push((fold_weight_raw, rank_vocab, markov));
            (fold_weight_raw, None)
        };
        allocations.push(fold_bias);
        bytes += rank_vocab * 2;
        let head_q = if use_nvfp4 && hidden.is_multiple_of(64) {
            match shared.bf16_rows(head, rank_vocab, hidden) {
                Ok(head_bf16) => Some(quantize_tiled(
                    head_bf16,
                    rank_vocab,
                    hidden,
                    &mut allocations,
                    &mut bytes,
                    &mut quant_bytes,
                )?),
                Err(_) => None,
            }
        } else {
            None
        };
        if quant_bytes > 0 {
            cuda.synchronize()?;
            let temp_refs: Vec<&Deviceptr> = temp_bf16.iter().collect();
            cuda.free_buffers(&temp_refs);
            eprintln!(
                "[inkling] rank {rank}: drafter NVFP4 quantized: {:.1} MB",
                quant_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(Self {
            kernels,
            layers,
            alpha,
            mask,
            norm,
            markov_embed,
            rnn_weight,
            rnn_q,
            rnn_bias,
            fold_weight,
            fold_q,
            fold_bias,
            markov_norm,
            tap_layers,
            taps,
            fusion_rows,
            embed_table: shared.weights.get(embed_table).pointer,
            embed_norm: shared.weights.get(embed_norm).pointer,
            head,
            head_q,
            block: mtp.block_size,
            hidden,
            heads,
            kv_heads,
            intermediate,
            rank: markov,
            eps: mtp.rms_norm_eps as f32,
            theta: mtp.rope_theta as f32,
            dense,
            allocations,
            bytes,
        })
    }

    /// The device bytes the drafter's weights take on this rank.
    pub(super) fn device_bytes(&self) -> usize {
        self.bytes
    }

    /// The matrices the rank may pack in place: see the field.
    pub(super) fn dense_matrices(&self) -> &[(Deviceptr, usize, usize)] {
        &self.dense
    }

    /// The most drafts a block proposes.
    pub(super) fn max_depth(&self) -> usize {
        self.block - 1
    }

    /// Gives the weights back.
    pub(super) fn free(&mut self, cuda: &Cuda) {
        let refs: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&refs);
        self.allocations.clear();
    }
}

/// The Markov head's fold against this rank's head slice `head` ([rank_vocab][hidden] BF16):
/// `bf16(W_lm W_ho)` as [rank_vocab][rank] and `bf16(W_lm b_ho)`, both summed on the device with
/// the decode GEMV's f32 sums (the reference folds in fp64; a BF16 result moves only where an
/// f32 sum straddles a rounding).
fn fold(
    shared: &Shared,
    up: &DraftUploader<'_>,
    head: Deviceptr,
    hidden: usize,
    markov: usize,
) -> Result<(Deviceptr, Deviceptr), String> {
    let cuda: &Cuda = &shared.cuda;
    let rank_vocab = shared.plan.rank_vocab();
    let weight = up.read("markov_head.hidden_out.weight", hidden * markov * 2)?;
    let bias = up.read("markov_head.hidden_out.bias", hidden * 2)?;
    let weight: Vec<f32> = weight
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
        .collect();
    // W_ho is [hidden][rank]; its transpose's rows are the rank inputs the head multiplies.
    let mut transposed = vec![0f32; markov * hidden];
    for row in 0..hidden {
        for column in 0..markov {
            transposed[column * hidden + row] = weight[row * markov + column];
        }
    }
    let bias: Vec<f32> = bias
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
        .collect();
    let mut total = 0usize;
    let inputs = cuda.alloc((markov + 1) * hidden * 4, &mut total)?;
    let outputs = cuda.alloc((markov + 1) * rank_vocab * 4, &mut total)?;
    let run = || -> Result<(Vec<u8>, Vec<u8>), String> {
        cuda.copy_htod(&inputs, &row_bytes(&transposed))?;
        cuda.copy_htod(&inputs.offset(markov * hidden * 4), &row_bytes(&bias))?;
        project(
            shared,
            PrefillWalk::Chunk,
            head,
            inputs,
            outputs,
            rank_vocab,
            hidden,
            markov + 1,
        )
        .map_err(|error| error.message().to_owned())?;
        cuda.synchronize()?;
        let mut product = vec![0u8; markov * rank_vocab * 4];
        cuda.copy_dtoh(&mut product, &outputs)?;
        let mut shift = vec![0u8; rank_vocab * 4];
        cuda.copy_dtoh(&mut shift, &outputs.offset(markov * rank_vocab * 4))?;
        Ok((product, shift))
    };
    let result = run();
    cuda.free_buffers(&[&inputs, &outputs]);
    let (product, shift) = result?;
    let product = from_bytes(&product);
    let mut folded = Vec::with_capacity(rank_vocab * markov * 2);
    for id in 0..rank_vocab {
        for column in 0..markov {
            let value = product[column * rank_vocab + id];
            folded.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
        }
    }
    let shift: Vec<u8> = from_bytes(&shift)
        .into_iter()
        .flat_map(|value| f32_to_bf16_bits(value).to_le_bytes())
        .collect();
    Ok((up.bytes(&folded)?, up.bytes(&shift)?))
}

/// One region of a session's drafter arena.
#[derive(Debug, Clone, Copy)]
struct Region(usize);

/// Hands out 256-byte aligned regions of one allocation.
struct Layout {
    bytes: usize,
}

impl Layout {
    fn take(&mut self, bytes: usize) -> Region {
        let at = self.bytes;
        self.bytes = (at + bytes).next_multiple_of(256);
        Region(at)
    }
}

/// One sequence's drafter state: each layer's context rings, the tap buffer the walks fill, and
/// the block's and the Markov head's scratch.
pub(super) struct DraftState {
    arena: Deviceptr,
    /// Each layer's key and value rings, `[ring][kv_width]` f32, and the ring's rows.
    rings: Vec<(Deviceptr, Deviceptr, usize)>,
    sink: TapSink,
    fused: Deviceptr,
    context_keys: Deviceptr,
    context_values: Deviceptr,
    /// The Markov head's fused feature of the last committed row.
    feature: Deviceptr,
    block: Deviceptr,
    normed: Deviceptr,
    queries: Deviceptr,
    keys: Deviceptr,
    values: Deviceptr,
    attention: Deviceptr,
    /// Each query's runs of the block attention, `[rows][heads][ATTENTION_CHUNKS]` of
    /// ATTENTION_PART floats.
    attention_parts: Deviceptr,
    delta: Deviceptr,
    gate: Deviceptr,
    up: Deviceptr,
    base: Deviceptr,
    z: Deviceptr,
    projected: Deviceptr,
    state: Deviceptr,
    latent: Deviceptr,
    product: Deviceptr,
    part_values: Deviceptr,
    part_ids: Deviceptr,
    /// Each step-logits block's mass: its sum of exp(logit - the block's best).
    part_sums: Deviceptr,
    /// Each step-logits block's best perturbed score under a sampled request's noise.
    part_scores: Deviceptr,
    slots: Deviceptr,
    tokens: Deviceptr,
    /// The probability the step's softmax gave each pick, beside [Self::tokens].
    confidence: Deviceptr,
    parts: usize,
    /// The positions whose context the rings hold: 0..committed.
    committed: usize,
}

impl DraftState {
    /// A sequence's drafter state: a full layer's rings hold the whole context, a local layer's
    /// its window (and a widest chunk past it under [REWIND_ENV], so a rewind still finds the
    /// window before the boundary), and the tap buffer the widest chunk a prompt walks.
    ///
    /// A full layer's whole-context rings are the session's lane's (`lane`, by layer, from
    /// [DraftModel::whole_context_rings]), backed as the sequence grows; the rest is one arena.
    pub(super) fn allocate(
        shared: &Shared,
        model: &DraftModel,
        lane: &[Option<(Deviceptr, Deviceptr)>],
    ) -> Result<Self, String> {
        let hidden = model.hidden;
        let kv_width = model.kv_heads * HEAD_DIM;
        let q_width = model.heads * HEAD_DIM;
        let rows = model.block;
        let rank_vocab = shared.plan.rank_vocab();
        let world = shared.exchange.world();
        let tap_rows = shared.prefill_chunk.max(rows);
        let parts = if model.fold_q.is_some() {
            rank_vocab.div_ceil(BLOCK as usize)
        } else {
            rank_vocab.div_ceil(BLOCK as usize * LOGIT_PER_THREAD)
        };
        let mut layout = Layout { bytes: 0 };
        let f = std::mem::size_of::<f32>();
        let rings: Vec<(Option<(Region, Region)>, usize)> = model
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let rewind_rows = if shared.rewind {
                    shared.prefill_chunk.saturating_sub(1)
                } else {
                    0
                };
                let ring = layer
                    .window
                    .map_or(shared.context, |window| window + rewind_rows)
                    .min(shared.context)
                    .max(1);
                let in_lane = lane.get(index).copied().flatten().is_some();
                (
                    (!in_lane).then(|| {
                        (
                            layout.take(ring * kv_width * 2),
                            layout.take(ring * kv_width * 2),
                        )
                    }),
                    ring,
                )
            })
            .collect();
        let taps = layout.take(tap_rows * model.taps * hidden * 2);
        let fused = layout.take(INJECT_ROWS * hidden * f);
        let context_keys = layout.take(INJECT_ROWS * kv_width * f);
        let context_values = layout.take(INJECT_ROWS * kv_width * f);
        let feature = layout.take(hidden * f);
        let block = layout.take(rows * hidden * f);
        let normed = layout.take(rows * hidden * f);
        let queries = layout.take(rows * q_width * f);
        let keys = layout.take(rows * kv_width * f);
        let values = layout.take(rows * kv_width * f);
        let attention = layout.take(rows * q_width * f);
        let attention_parts =
            layout.take(rows * model.heads * ATTENTION_CHUNKS * ATTENTION_PART * f);
        let delta = layout.take(rows * hidden * f);
        let gate = layout.take(rows * model.intermediate * f);
        let up = layout.take(rows * model.intermediate * f);
        let base = layout.take(rows * rank_vocab * f);
        let z = layout.take((2 * model.rank + 2 * hidden) * f);
        let projected = layout.take(3 * model.rank * f);
        let state = layout.take(model.rank * f);
        let latent = layout.take(model.rank * f);
        let product = layout.take(rank_vocab * f);
        let part_values = layout.take(parts * f);
        let part_ids = layout.take(parts * 4);
        let part_sums = layout.take(parts * f);
        let part_scores = layout.take(parts * f);
        let slots = layout.take(world * PICK_SLOT * f);
        let tokens = layout.take(rows * 4);
        let confidence = layout.take(rows * f);
        let mut total = 0usize;
        let arena = shared.cuda.alloc(layout.bytes, &mut total)?;
        let at = |region: Region| arena.offset(region.0);
        if trace_on() {
            eprintln!("[inkling] drafter state: {} bytes", layout.bytes);
        }
        Ok(Self {
            arena,
            rings: rings
                .into_iter()
                .enumerate()
                .map(|(index, (own, ring))| match own {
                    Some((keys, values)) => (at(keys), at(values), ring),
                    None => {
                        let (keys, values) =
                            lane[index].expect("a ring outside the arena is the lane's");
                        (keys, values, ring)
                    }
                })
                .collect(),
            sink: TapSink {
                capture: model.kernels.capture,
                taps: at(taps),
                layers: model.tap_layers.clone(),
                per_row: model.taps,
                rows: tap_rows,
            },
            fused: at(fused),
            context_keys: at(context_keys),
            context_values: at(context_values),
            feature: at(feature),
            block: at(block),
            normed: at(normed),
            queries: at(queries),
            keys: at(keys),
            values: at(values),
            attention: at(attention),
            attention_parts: at(attention_parts),
            delta: at(delta),
            gate: at(gate),
            up: at(up),
            base: at(base),
            z: at(z),
            projected: at(projected),
            state: at(state),
            latent: at(latent),
            product: at(product),
            part_values: at(part_values),
            part_ids: at(part_ids),
            part_sums: at(part_sums),
            part_scores: at(part_scores),
            slots: at(slots),
            tokens: at(tokens),
            confidence: at(confidence),
            parts,
            committed: 0,
        })
    }

    /// The device memory this state owns.
    pub(super) fn buffers(&self) -> [Deviceptr; 1] {
        [self.arena]
    }

    /// Where the walks leave this sequence's taps.
    pub(super) fn sink(&self) -> TapSink {
        self.sink.clone()
    }

    /// Forget the context: the next prompt injects from position zero. The rings keep their
    /// bytes; nothing reads a position at or past [Self::committed].
    pub(super) fn reset(&mut self) {
        self.committed = 0;
    }

    /// The Markov feature of the last committed row and its bytes: what a prompt-boundary
    /// checkpoint copies, since the rings hold every other piece of the context.
    pub(super) fn feature(&self, model: &DraftModel) -> (Deviceptr, usize) {
        (self.feature, model.hidden * std::mem::size_of::<f32>())
    }

    /// Takes the context back to `position`, once the caller has copied that position's feature
    /// back ([Self::feature]). The rings keep the rows before it: nothing past it is read until a
    /// later inject writes it again.
    ///
    /// # Errors
    /// A position past the context the rings hold.
    pub(super) fn rewind(&mut self, position: usize) -> Result<(), String> {
        if position > self.committed {
            return Err(format!(
                "the drafter's context ends at {} and cannot rewind to {position}",
                self.committed
            ));
        }
        self.committed = position;
        Ok(())
    }

    /// The positions whose context the rings hold.
    pub(super) fn committed(&self) -> usize {
        self.committed
    }

    /// Mixes the tap buffer's first `rows` rows (positions `position..position + rows`) into
    /// every layer's context, and keeps the last row's Markov feature. A local layer takes only
    /// the rows its window keeps.
    ///
    /// # Errors
    /// A position that does not continue the context, or a device failure.
    pub(super) fn inject(
        &mut self,
        shared: &Shared,
        model: &DraftModel,
        rows: usize,
        position: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 {
            return Ok(());
        }
        if position != self.committed {
            return Err(device_error(format!(
                "the drafter's context ends at {} and the rows to inject start at {position}",
                self.committed
            )));
        }
        if rows > self.sink.rows {
            return Err(device_error(format!(
                "{rows} rows to inject; the tap buffer holds {}",
                self.sink.rows
            )));
        }
        let hidden = model.hidden;
        let kv_width = model.kv_heads * HEAD_DIM;
        let tap_row = model.taps * hidden * 2;
        let kernels = &model.kernels;
        let run = |result: Result<(), String>| result.map_err(device_error);
        for (index, layer) in model.layers.iter().enumerate() {
            let (key_ring, value_ring, ring) = self.rings[index];
            let from = layer.window.map_or(0, |window| rows.saturating_sub(window));
            let mut at = from;
            while at < rows {
                let take = (rows - at).min(INJECT_ROWS);
                run(shared.launch(
                    kernels.mix_norm,
                    (take as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.sink.taps.offset(at * tap_row).0,
                        model.alpha.offset(index * model.taps * 2).0,
                        layer.fusion_norm.0,
                        self.fused.0,
                        hidden as u64,
                        model.taps as u64,
                        bits(model.eps),
                    ],
                ))?;
                let walk = if take <= BATCH_ROWS {
                    PrefillWalk::Chunk
                } else {
                    shared.prefill_walk
                };
                if walk == PrefillWalk::Chunk && take <= BATCH_ROWS {
                    run(shared.gemv_fused(
                        &[
                            (layer.context_key, self.context_keys, kv_width),
                            (layer.context_value, self.context_values, kv_width),
                        ],
                        self.fused,
                        hidden,
                        take,
                    ))?;
                } else {
                    project(
                        shared,
                        walk,
                        layer.context_key,
                        self.fused,
                        self.context_keys,
                        kv_width,
                        hidden,
                        take,
                    )?;
                    project(
                        shared,
                        walk,
                        layer.context_value,
                        self.fused,
                        self.context_values,
                        kv_width,
                        hidden,
                        take,
                    )?;
                }
                run(shared.launch(
                    kernels.context,
                    (model.kv_heads as u32, take as u32, 1),
                    (HEAD_DIM as u32, 1, 1),
                    &[
                        self.context_keys.0,
                        self.context_values.0,
                        layer.context_key_norm.0,
                        key_ring.0,
                        value_ring.0,
                        model.kv_heads as u64,
                        (position + at) as u64,
                        ring as u64,
                        bits(model.eps),
                        bits(model.theta),
                    ],
                ))?;
                at += take;
            }
        }
        run(shared.launch(
            kernels.mix_norm,
            (1, 1, 1),
            (BLOCK, 1, 1),
            &[
                self.sink.taps.offset((rows - 1) * tap_row).0,
                model
                    .alpha
                    .offset((model.fusion_rows - 1) * model.taps * 2)
                    .0,
                model.markov_norm.0,
                self.feature.0,
                hidden as u64,
                model.taps as u64,
                bits(model.eps),
            ],
        ))?;
        self.committed = position + rows;
        Ok(())
    }

    /// Up to `depth` drafts after `anchor`, the token at position [Self::committed], and the
    /// probability the drafter gave each: the block through the five layers, its rows' base
    /// logits through the head, and the Markov head's steps.
    ///
    /// A step picks its largest logit, or under `noise` (a sampled request's 1/T and seed,
    /// [sparkflash_model::gumbel::GumbelSampler::draft_noise]) the largest of
    /// `logit / T + G(seed, index, id)` over every id, `index` the sequence index the draft lands
    /// at: the noise the target's sample takes, so a draft the drafter's distribution shares with
    /// the target's tends to be the target's sample. Every rank must be given the same noise.
    ///
    /// A draft's probability is its step's softmax at the step's largest logit (the pick itself
    /// unless noise moved it). It is computed on the device from the values every rank exchanged,
    /// so every rank reads the same floats and a depth chosen from them is the same on each.
    ///
    /// # Errors
    /// A sequence with no committed row (the Markov head's feature is the last one's), a depth
    /// past the block, or a device failure.
    pub(super) fn draft(
        &mut self,
        shared: &Shared,
        model: &DraftModel,
        anchor: u32,
        depth: usize,
        noise: Option<(f32, u64)>,
        early_stop: bool,
        mut keep_going: impl FnMut(&[f32]) -> bool,
    ) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
        if depth == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if depth > model.max_depth() {
            return Err(device_error(format!(
                "{depth} drafts from a {}-row block",
                model.block
            )));
        }
        if self.committed == 0 {
            return Err(device_error(
                "the drafter drafts after a committed row: this sequence has none",
            ));
        }
        let config = &shared.config;
        if anchor as usize >= config.vocab_size {
            return Err(device_error(format!(
                "anchor {anchor} is outside the vocabulary {}",
                config.vocab_size
            )));
        }
        let kernels = &model.kernels;
        let run = |result: Result<(), String>| result.map_err(device_error);
        let rows = model.block;
        let hidden = model.hidden;
        let heads = model.heads;
        let kv_heads = model.kv_heads;
        let q_width = heads * HEAD_DIM;
        let kv_width = kv_heads * HEAD_DIM;
        let f = std::mem::size_of::<f32>();
        let position = self.committed;
        let world = shared.exchange.world();
        let rank_vocab = shared.plan.rank_vocab();
        let first_id = shared.exchange.rank() * rank_vocab;
        let norm = |input: Deviceptr, weight: Deviceptr, out: Deviceptr, groups: usize, group| {
            shared.launch(
                kernels.norm,
                (groups as u32, 1, 1),
                (
                    if group < BLOCK as usize {
                        HEAD_DIM as u32
                    } else {
                        BLOCK
                    },
                    1,
                    1,
                ),
                &[input.0, weight.0, out.0, group as u64, bits(model.eps)],
            )
        };
        let elementwise = |entry: Bound, args: &[u64], count: usize| {
            shared.launch(entry, (flat(count), 1, 1), (ELEMENT_BLOCK, 1, 1), args)
        };
        let nvfp4_gemv = |w: Nvfp4Tiled,
                          input: Deviceptr,
                          out: Deviceptr,
                          rows_out: usize,
                          cols: usize,
                          batch: usize,
                          rt_per_warp: u32| {
            // Microbenchmarked with IKD_TC_K_CHUNK=512 on sm_121a: 64 threads (2 warps/block)
            // wins across out (16.2 us), down (23.6 us), and rnn (16.6 us); 256 threads with
            // rt_per_warp=2 wins on LM head (1140 us, preserving L1 cache).
            let block_threads: u32 = if rt_per_warp >= 2 { BLOCK } else { 64 };
            let warps = (block_threads / 32) as usize;
            let rows_per_block = warps * 16 * if rt_per_warp >= 2 { 2 } else { 1 };
            shared.launch(
                kernels.nvfp4_gemv,
                (rows_out.div_ceil(rows_per_block) as u32, 1, 1),
                (block_threads, 1, 1),
                &[
                    w.codes.0,
                    w.scales.0,
                    w.row_scale.0,
                    input.0,
                    out.0,
                    rows_out as u64,
                    cols as u64,
                    batch as u64,
                    u64::from(rt_per_warp),
                ],
            )
        };
        let nvfp4_gemv3 = |m0: (Nvfp4Tiled, Deviceptr, usize),
                           m1: (Nvfp4Tiled, Deviceptr, usize),
                           m2: Option<(Nvfp4Tiled, Deviceptr, usize)>,
                           input: Deviceptr,
                           cols: usize,
                           batch: usize| {
            let nrt0 = m0.2.div_ceil(16);
            let nrt1 = m1.2.div_ceil(16);
            let (c2, s2, rs2, o2, r2) = match m2 {
                Some((w, o, r)) => (w.codes.0, w.scales.0, w.row_scale.0, o.0, r),
                None => (0, 0, 0, 0, 0),
            };
            let nrt2 = r2.div_ceil(16);
            shared.launch(
                kernels.nvfp4_gemv3,
                ((nrt0 + nrt1 + nrt2).div_ceil(8) as u32, 1, 1),
                (BLOCK, 1, 1),
                &[
                    m0.0.codes.0,
                    m0.0.scales.0,
                    m0.0.row_scale.0,
                    m0.1.0,
                    m0.2 as u64,
                    m1.0.codes.0,
                    m1.0.scales.0,
                    m1.0.row_scale.0,
                    m1.1.0,
                    m1.2 as u64,
                    c2,
                    s2,
                    rs2,
                    o2,
                    r2 as u64,
                    input.0,
                    cols as u64,
                    batch as u64,
                ],
            )
        };

        // Row 0: the anchor's normed embedding, which only the rank that owns the id holds.
        match (anchor as usize).checked_sub(first_id) {
            Some(local) if local < rank_vocab => {
                Embed::new(shared.kernels.embed, model.embed_table, hidden, 1)?.row(
                    shared,
                    self.block,
                    local as u32,
                )?;
                run(shared.rmsnorm(self.block, model.embed_norm, self.block, 1, hidden))?;
            }
            _ => run(shared.cuda.memset_d8_async(self.block, 0, hidden * f))?,
        }
        exchange_sum(shared, self.block, hidden)?;
        run(shared.launch(
            kernels.masks,
            (grid_blocks(hidden, BLOCK as usize), (rows - 1) as u32, 1),
            (BLOCK, 1, 1),
            &[model.mask.0, self.block.0, hidden as u64],
        ))?;
        run(elementwise(
            kernels.round,
            &[self.block.0, hidden as u64],
            hidden,
        ))?;

        for (index, layer) in model.layers.iter().enumerate() {
            let (key_ring, value_ring, ring) = self.rings[index];
            run(norm(
                self.block,
                layer.input_norm,
                self.normed,
                rows,
                hidden,
            ))?;
            if let (Some(qq), Some(kq), Some(vq)) = (layer.query_q, layer.key_q, layer.value_q) {
                run(nvfp4_gemv3(
                    (qq, self.queries, q_width),
                    (kq, self.keys, kv_width),
                    Some((vq, self.values, kv_width)),
                    self.normed,
                    hidden,
                    rows,
                ))?;
            } else {
                run(shared.gemv_fused(
                    &[
                        (layer.query, self.queries, q_width),
                        (layer.key, self.keys, kv_width),
                        (layer.value, self.values, kv_width),
                    ],
                    self.normed,
                    hidden,
                    rows,
                ))?;
            }
            run(norm(
                self.queries,
                layer.query_norm,
                self.queries,
                rows * heads,
                HEAD_DIM,
            ))?;
            run(norm(
                self.keys,
                layer.key_norm,
                self.keys,
                rows * kv_heads,
                HEAD_DIM,
            ))?;
            run(elementwise(
                kernels.round,
                &[self.values.0, (rows * kv_width) as u64],
                rows * kv_width,
            ))?;
            for (pointer, count, stride) in [
                (self.queries, heads, q_width),
                (self.keys, kv_heads, kv_width),
            ] {
                run(shared.launch(
                    kernels.rope,
                    (count as u32, rows as u32, 1),
                    ((HEAD_DIM / 2) as u32, 1, 1),
                    &[
                        pointer.0,
                        HEAD_DIM as u64,
                        position as u64,
                        bits(model.theta),
                        stride as u64,
                    ],
                ))?;
            }
            // Every row sees the same keys: the window's (or the whole context's) committed
            // positions before the block, and the block's own rows. A block of the first stage
            // walks one of a key head's ATTENTION_CHUNKS runs of them for every query reading
            // that head, so the span is read once, not once a query.
            let lowest = layer
                .window
                .map_or(0, |window| position.saturating_sub(window));
            let group_queries = (heads / kv_heads) * rows;
            let span_total = position.saturating_sub(lowest) + rows;
            let (attn_kernel, attn_block, chunks) = if group_queries <= 24 {
                let chunks = if span_total >= 8192 {
                    ATTENTION_CHUNKS
                } else {
                    span_total.div_ceil(128).clamp(1, 12)
                };
                (kernels.attention_part_tc, 256u32, chunks)
            } else {
                (
                    kernels.attention_part,
                    (group_queries.div_ceil(ATTENTION_QUERIES) * 32) as u32,
                    ATTENTION_CHUNKS,
                )
            };
            run(shared.launch(
                attn_kernel,
                (kv_heads as u32, chunks as u32, 1),
                (attn_block, 1, 1),
                &[
                    self.queries.0,
                    key_ring.0,
                    value_ring.0,
                    self.keys.0,
                    self.values.0,
                    self.attention_parts.0,
                    heads as u64,
                    kv_heads as u64,
                    HEAD_DIM as u64,
                    position as u64,
                    lowest as u64,
                    ring as u64,
                    rows as u64,
                    bits(1.0 / (HEAD_DIM as f32).sqrt()),
                ],
            ))?;
            run(shared.launch(
                kernels.attention_merge,
                (heads as u32, rows as u32, 1),
                (32, 1, 1),
                &[
                    self.attention_parts.0,
                    self.attention.0,
                    heads as u64,
                    HEAD_DIM as u64,
                    chunks as u64,
                ],
            ))?;
            run(elementwise(
                kernels.round,
                &[self.attention.0, (rows * q_width) as u64],
                rows * q_width,
            ))?;
            if let Some(oq) = layer.out_q {
                run(nvfp4_gemv(oq, self.attention, self.delta, hidden, q_width, rows, 1))?;
            } else {
                run(shared.gemv_batch(
                    layer.out,
                    self.attention,
                    self.delta,
                    hidden,
                    q_width,
                    rows,
                    1,
                ))?;
            }
            exchange_sum(shared, self.delta, rows * hidden)?;
            run(elementwise(
                kernels.add_round,
                &[self.block.0, self.delta.0, (rows * hidden) as u64],
                rows * hidden,
            ))?;
            run(norm(self.block, layer.post_norm, self.normed, rows, hidden))?;
            let inner = model.intermediate;
            if let (Some(gq), Some(uq)) = (layer.gate_q, layer.up_q) {
                run(nvfp4_gemv3(
                    (gq, self.gate, inner),
                    (uq, self.up, inner),
                    None,
                    self.normed,
                    hidden,
                    rows,
                ))?;
            } else {
                run(shared.gemv_fused(
                    &[
                        (layer.gate, self.gate, inner),
                        (layer.up, self.up, inner),
                    ],
                    self.normed,
                    hidden,
                    rows,
                ))?;
            }
            run(elementwise(
                kernels.swiglu,
                &[self.gate.0, self.up.0, self.gate.0, (rows * inner) as u64],
                rows * inner,
            ))?;
            if let Some(dq) = layer.down_q {
                run(nvfp4_gemv(dq, self.gate, self.delta, hidden, inner, rows, 1))?;
            } else {
                run(shared.gemv_batch(layer.down, self.gate, self.delta, hidden, inner, rows, 1))?;
            }
            exchange_sum(shared, self.delta, rows * hidden)?;
            run(elementwise(
                kernels.add_round,
                &[self.block.0, self.delta.0, (rows * hidden) as u64],
                rows * hidden,
            ))?;
        }
        run(norm(self.block, model.norm, self.normed, rows, hidden))?;
        // The base logits of rows 1..=depth: this rank's slice of the head.
        if let Some(hq) = model.head_q {
            run(nvfp4_gemv(
                hq,
                self.normed.offset(hidden * f),
                self.base,
                rank_vocab,
                hidden,
                depth,
                2,
            ))?;
        } else {
            run(shared.gemv_batch(
                model.head,
                self.normed.offset(hidden * f),
                self.base,
                rank_vocab,
                hidden,
                depth,
                1,
            ))?;
        }

        let markov = model.rank;
        let z_width = 2 * markov + 2 * hidden;
        let padded = rank_vocab * world;
        let valid = config.unpadded_vocab_size.min(padded).max(1);
        let cap = config.final_logit_softcap.unwrap_or(0.0) as f32;
        let (inv_t, seed) = noise.filter(|(inv_t, _)| *inv_t > 0.0).unwrap_or((0.0, 0));
        let noisy = inv_t > 0.0;
        run(shared.cuda.memset_d8_async(self.state, 0, markov * f))?;
        let early_stop = early_stop
            && !matches!(
                std::env::var("SPARKFLASH_INKLING_DRAFT_EARLY_STOP").ok().as_deref(),
                Some("off" | "0")
            );
        let mut drafted = 0usize;
        let mut stopped_early = false;
        let mut probs_so_far = Vec::with_capacity(depth);
        let mut conf_word = [0u8; 4];
        for step in 0..depth {
            run(shared.launch(
                kernels.markov_input,
                (flat(z_width), 1, 1),
                (ELEMENT_BLOCK, 1, 1),
                &[
                    self.z.0,
                    self.state.0,
                    model.markov_embed.0,
                    self.tokens.0,
                    u64::from(anchor),
                    self.slots.0,
                    world as u64,
                    step as u64,
                    self.normed.offset((step + 1) * hidden * f).0,
                    self.feature.0,
                    markov as u64,
                    hidden as u64,
                    self.confidence.0,
                    u64::from(noisy),
                ],
            ))?;
            if let Some(rq) = model.rnn_q {
                run(nvfp4_gemv(
                    rq,
                    self.z,
                    self.projected,
                    3 * markov,
                    z_width,
                    1,
                    1,
                ))?;
            } else {
                run(shared.gemv(
                    model.rnn_weight,
                    self.z,
                    self.projected,
                    3 * markov,
                    z_width,
                ))?;
            }
            run(shared.launch(
                kernels.markov_gates,
                (1, 1, 1),
                (BLOCK, 1, 1),
                &[
                    self.projected.0,
                    model.rnn_bias.0,
                    self.state.0,
                    self.latent.0,
                    markov as u64,
                ],
            ))?;
            if let Some(fq) = model.fold_q {
                run(shared.launch(
                    kernels.nvfp4_markov_logits,
                    (self.parts as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        fq.codes.0,
                        fq.scales.0,
                        fq.row_scale.0,
                        self.latent.0,
                        self.base.offset(step * rank_vocab * f).0,
                        model.fold_bias.0,
                        rank_vocab as u64,
                        markov as u64,
                        first_id as u64,
                        valid as u64,
                        bits(cap),
                        self.part_values.0,
                        self.part_ids.0,
                        self.part_sums.0,
                        bits(inv_t),
                        seed,
                        (position + 1 + step) as u64,
                        self.part_scores.0,
                    ],
                ))?;
            } else {
                run(shared.gemv(
                    model.fold_weight,
                    self.latent,
                    self.product,
                    rank_vocab,
                    markov,
                ))?;
                run(shared.launch(
                    kernels.markov_logits,
                    (self.parts as u32, 1, 1),
                    (BLOCK, 1, 1),
                    &[
                        self.base.offset(step * rank_vocab * f).0,
                        self.product.0,
                        model.fold_bias.0,
                        rank_vocab as u64,
                        first_id as u64,
                        valid as u64,
                        bits(cap),
                        self.part_values.0,
                        self.part_ids.0,
                        self.part_sums.0,
                        bits(inv_t),
                        seed,
                        // The draft of step `step` lands after the anchor and the drafts before it.
                        (position + 1 + step) as u64,
                        self.part_scores.0,
                    ],
                ))?;
            }
            run(shared.launch(
                kernels.markov_local,
                (1, 1, 1),
                (BLOCK, 1, 1),
                &[
                    self.part_values.0,
                    self.part_ids.0,
                    self.parts as u64,
                    self.slots.0,
                    shared.exchange.rank() as u64,
                    world as u64,
                    self.part_sums.0,
                    if noisy { self.part_scores.0 } else { 0 },
                ],
            ))?;
            exchange_sum(shared, self.slots, world * PICK_SLOT)?;
            drafted = step + 1;
            if early_stop && drafted < depth {
                run(shared.launch(
                    kernels.markov_last,
                    (1, 1, 1),
                    (32, 1, 1),
                    &[
                        self.slots.0,
                        world as u64,
                        self.tokens.0,
                        drafted as u64,
                        self.confidence.0,
                        u64::from(noisy),
                    ],
                ))?;
                run(shared.cuda.synchronize())?;
                run(shared.exchange.check_device())?;
                run(shared
                    .cuda
                    .copy_dtoh(&mut conf_word, &self.confidence.offset(drafted * 4)))?;
                probs_so_far.push(f32::from_le_bytes(conf_word));
                if !keep_going(&probs_so_far) {
                    stopped_early = true;
                    break;
                }
            }
        }
        if !stopped_early {
            run(shared.launch(
                kernels.markov_last,
                (1, 1, 1),
                (32, 1, 1),
                &[
                    self.slots.0,
                    world as u64,
                    self.tokens.0,
                    drafted as u64,
                    self.confidence.0,
                    u64::from(noisy),
                ],
            ))?;
            run(shared.cuda.synchronize())?;
            run(shared.exchange.check_device())?;
        }
        // Word 0 of each is the anchor's, which no step picks.
        let mut bytes = vec![0u8; (drafted + 1) * 4];
        run(shared.cuda.copy_dtoh(&mut bytes, &self.tokens))?;
        let drafts: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .skip(1)
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        if let Some(bad) = drafts.iter().find(|id| **id as usize >= valid) {
            return Err(device_error(format!(
                "the drafter picked {bad}, past the {valid} ids it may draft"
            )));
        }
        let confidence: Vec<f32> = if stopped_early {
            probs_so_far
        } else {
            run(shared.cuda.copy_dtoh(&mut bytes, &self.confidence))?;
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .skip(1)
                .map(|word| f32::from_le_bytes(*word))
                .collect()
        };
        if let Some(bad) = confidence.iter().find(|p| !(0.0..=1.0).contains(*p)) {
            return Err(device_error(format!(
                "the drafter gave a pick the probability {bad}"
            )));
        }
        Ok((drafts, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::full_window;

    #[test]
    fn full_window_reads_full_or_a_count_and_refuses_the_rest() {
        assert_eq!(full_window(None), Ok(None));
        assert_eq!(full_window(Some("")), Ok(None));
        assert_eq!(full_window(Some("full")), Ok(None));
        assert_eq!(full_window(Some("8192")), Ok(Some(8192)));
        for bad in ["0", "-1", "8k", "window"] {
            assert!(full_window(Some(bad)).is_err(), "{bad} was taken");
        }
    }
}
