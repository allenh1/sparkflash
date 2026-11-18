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

/// The taps a committed row carries: seven pre-layer partials, then the post-final-norm hidden.
pub(super) const TAPS: usize = 8;
/// The tap slot of the post-final-norm hidden.
pub(super) const FINAL_TAP: usize = TAPS - 1;
/// The fusion rows: one a draft layer, then the Markov head's.
const FUSION_ROWS: usize = 6;
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
    _modules: Vec<Module>,
    capture: Entry,
    mix_norm: Entry,
    masks: Entry,
    round: Entry,
    norm: Entry,
    add_round: Entry,
    swiglu: Entry,
    rope: Entry,
    context: Entry,
    markov_input: Entry,
    markov_gates: Entry,
    markov_logits: Entry,
    markov_local: Entry,
    markov_last: Entry,
    /// DFlash 2's block attention over one span of the rings: every query's walks, and their
    /// merge.
    attention_part: Entry,
    attention_merge: Entry,
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
        let mut modules = Vec::new();
        let mut load = |cubin: &'static [u8], name: &'static str| -> Result<Entry, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(Entry { kernel })
        };
        Ok(Self {
            capture: load(inkling, "inkling_draft_capture")?,
            mix_norm: load(inkling, "inkling_draft_mix_norm")?,
            masks: load(inkling, "inkling_draft_masks")?,
            round: load(inkling, "inkling_draft_round")?,
            norm: load(inkling, "inkling_draft_norm")?,
            add_round: load(inkling, "inkling_draft_add_round")?,
            swiglu: load(inkling, "inkling_draft_swiglu")?,
            rope: load(inkling, "inkling_draft_rope")?,
            context: load(inkling, "inkling_draft_context")?,
            markov_input: load(inkling, "inkling_draft_markov_input")?,
            markov_gates: load(inkling, "inkling_draft_markov_gates")?,
            markov_logits: load(inkling, "inkling_draft_markov_logits")?,
            markov_local: load(inkling, "inkling_draft_markov_local")?,
            markov_last: load(inkling, "inkling_draft_markov_last")?,
            attention_part: load(dflash, "q4d_attn_span_chunk")?,
            attention_merge: load(dflash, "q4d_attn_span_merge")?,
            _modules: modules,
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
    /// A local layer's window, or a full layer's under [DRAFT_WINDOW_ENV]; `None` for a full
    /// layer that reads the whole context.
    window: Option<usize>,
}

/// The drafter's weights and geometry on this rank: loaded once ([DraftModel::load]) and shared
/// by the rank's sessions.
pub(super) struct DraftModel {
    kernels: DraftKernels,
    layers: Vec<DraftLayer>,
    /// `bf16(softmax(fuse_weight))`, [FUSION_ROWS][TAPS].
    alpha: Deviceptr,
    mask: Deviceptr,
    norm: Deviceptr,
    /// The Markov head's token table `markov_w1`, [vocabulary][rank], whole on every rank.
    markov_embed: Deviceptr,
    rnn_weight: Deviceptr,
    rnn_bias: Deviceptr,
    /// The fold against this rank's head slice: `bf16(W_lm W_ho)` [rank_vocab][rank] and
    /// `bf16(W_lm b_ho)` [rank_vocab].
    fold_weight: Deviceptr,
    fold_bias: Deviceptr,
    markov_norm: Deviceptr,
    /// The target layers whose pre-layer partials are taps 0..FINAL_TAP.
    tap_layers: [usize; FINAL_TAP],
    /// The target's embedding table, embedding norm and head (this rank's slices).
    embed_table: Deviceptr,
    embed_norm: Deviceptr,
    head: Deviceptr,
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

/// A walk's sink for the drafter's taps: the tap buffer, `[rows][TAPS][hidden]` BF16, and the
/// target layers whose pre-layer partials it keeps.
#[derive(Debug, Clone, Copy)]
pub(super) struct TapSink {
    capture: Entry,
    taps: Deviceptr,
    layers: [usize; FINAL_TAP],
    /// The rows the buffer holds.
    rows: usize,
}

// SAFETY: the capture entry is a driver function handle launched on the default stream, and the
// rest are device addresses and counts.
unsafe impl Send for TapSink {}
// SAFETY: as above.
unsafe impl Sync for TapSink {}

impl TapSink {
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
                    TAPS as u64,
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

    /// The bytes one position adds to a layer's key ring (and as many to its value ring).
    pub(super) fn ring_row_bytes(&self) -> usize {
        self.kv_heads * HEAD_DIM * std::mem::size_of::<f32>()
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
        let layers_total = config.num_hidden_layers;
        let ids = &mtp.target_layer_ids;
        if ids.len() != TAPS || ids.last() != Some(&layers_total) {
            return Err(format!(
                "the drafter taps {ids:?}: this arm takes {FINAL_TAP} pre-layer partials and \
                 the post-final-norm hidden ({layers_total})"
            ));
        }
        let mut tap_layers = [0usize; FINAL_TAP];
        for (slot, layer) in ids[..FINAL_TAP].iter().enumerate() {
            if !shared
                .plan
                .layers()
                .iter()
                .any(|schedule| schedule.index == *layer)
            {
                return Err(format!("tap layer {layer} is not a layer of this schedule"));
            }
            tap_layers[slot] = *layer;
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
        let mut bytes = 0usize;
        let mut keep = |pointer: Deviceptr, size: usize| -> Deviceptr {
            allocations.push(pointer);
            bytes += size;
            pointer
        };
        let q_width = heads * HEAD_DIM;
        let kv_width = kv_heads * HEAD_DIM;
        let q_total = mtp.num_attention_heads * HEAD_DIM;
        let kv_total = mtp.num_key_value_heads * HEAD_DIM;
        let mut layers = Vec::with_capacity(mtp.num_hidden_layers);
        for layer in 0..mtp.num_hidden_layers {
            let name = |tail: &str| format!("layers.{layer}.{tail}");
            let mut vector = |tail: &str, elements: usize| -> Result<Deviceptr, String> {
                Ok(keep(up.vector(&name(tail), elements)?, elements * 2))
            };
            let input_norm = vector("input_layernorm.weight", hidden)?;
            let post_norm = vector("post_attention_layernorm.weight", hidden)?;
            let fusion_norm = vector("dflare_norm.weight", hidden)?;
            let query_norm = vector("self_attn.q_norm.weight", HEAD_DIM)?;
            let key_norm = vector("self_attn.k_norm.weight", HEAD_DIM)?;
            let context_key_norm = vector("self_attn.k_norm_ctx.weight", HEAD_DIM)?;
            let mut part = |tail: &str, shape: (usize, usize), part: Part| {
                up.mat_part(&name(tail), shape, part)
                    .map(|matrix| keep(matrix.pointer, matrix.rows * matrix.columns * 2))
            };
            let rows = |count: usize| Part::Rows(rank * count, count);
            let columns = |count: usize| Part::Columns(rank * count, count);
            layers.push(DraftLayer {
                input_norm,
                post_norm,
                fusion_norm,
                query: part("self_attn.q_proj.weight", (q_total, hidden), rows(q_width))?,
                key: part(
                    "self_attn.k_proj.weight",
                    (kv_total, hidden),
                    rows(kv_width),
                )?,
                value: part(
                    "self_attn.v_proj.weight",
                    (kv_total, hidden),
                    rows(kv_width),
                )?,
                context_key: part(
                    "self_attn.k_proj_ctx.weight",
                    (kv_total, hidden),
                    rows(kv_width),
                )?,
                context_value: part(
                    "self_attn.v_proj_ctx.weight",
                    (kv_total, hidden),
                    rows(kv_width),
                )?,
                out: part(
                    "self_attn.o_proj.weight",
                    (hidden, q_total),
                    columns(q_width),
                )?,
                query_norm,
                key_norm,
                context_key_norm,
                gate: part(
                    "mlp.gate_proj.weight",
                    (mtp.intermediate_size, hidden),
                    rows(intermediate),
                )?,
                up: part(
                    "mlp.up_proj.weight",
                    (mtp.intermediate_size, hidden),
                    rows(intermediate),
                )?,
                down: part(
                    "mlp.down_proj.weight",
                    (hidden, mtp.intermediate_size),
                    columns(intermediate),
                )?,
                window: if mtp.local_layer_ids.contains(&layer) {
                    Some(mtp.sliding_window_size)
                } else {
                    full_window
                },
            });
            // The block's key and value stay BF16: at the block's 8 rows their 256-row packed walk
            // measured slower (26.5 us against 19.5), where every other projection, the context
            // pair at a round's few fed rows included, measured as fast or faster packed.
            let layer = layers.last().expect("the layer was just pushed");
            dense.extend([
                (layer.query, q_width, hidden),
                (layer.context_key, kv_width, hidden),
                (layer.context_value, kv_width, hidden),
                (layer.out, hidden, q_width),
                (layer.gate, intermediate, hidden),
                (layer.up, intermediate, hidden),
                (layer.down, hidden, intermediate),
            ]);
        }
        // The fusion rows' softmax over the taps, in fp32 and rounded to BF16 as the reference
        // keeps them.
        let fuse = up.read("fuse_weight", FUSION_ROWS * TAPS * 2)?;
        let fuse: Vec<f32> = fuse
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
            .collect();
        let mut alpha = Vec::with_capacity(FUSION_ROWS * TAPS * 2);
        for row in fuse.chunks(TAPS) {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (value - top).exp()).collect();
            let total: f32 = exps.iter().sum();
            for value in exps {
                alpha.extend_from_slice(&f32_to_bf16_bits(value / total).to_le_bytes());
            }
        }
        let alpha = keep(up.bytes(&alpha)?, alpha.len());
        let mask = keep(up.vector("mask_embed", hidden)?, hidden * 2);
        let norm = keep(up.vector("norm.weight", hidden)?, hidden * 2);
        let markov_norm = keep(
            up.vector("markov_head.dflare_norm.weight", hidden)?,
            hidden * 2,
        );
        let vocabulary = config.vocab_size;
        let markov_embed = up.mat("markov_head.markov_w1.weight", vocabulary, markov)?;
        let markov_embed = keep(markov_embed.pointer, vocabulary * markov * 2);
        let z_width = 2 * markov + 2 * hidden;
        let rnn = up.mat("markov_head.rnn_proj.weight", 3 * markov, z_width)?;
        let rnn_weight = keep(rnn.pointer, 3 * markov * z_width * 2);
        dense.push((rnn_weight, 3 * markov, z_width));
        let rnn_bias = keep(
            up.vector("markov_head.rnn_proj.bias", 3 * markov)?,
            6 * markov,
        );
        let (fold_weight, fold_bias) = fold(shared, &up, head, hidden, markov)?;
        let rank_vocab = shared.plan.rank_vocab();
        let fold_weight = keep(fold_weight, rank_vocab * markov * 2);
        dense.push((fold_weight, rank_vocab, markov));
        let fold_bias = keep(fold_bias, rank_vocab * 2);
        Ok(Self {
            kernels,
            layers,
            alpha,
            mask,
            norm,
            markov_embed,
            rnn_weight,
            rnn_bias,
            fold_weight,
            fold_bias,
            markov_norm,
            tap_layers,
            embed_table: shared.weights.get(embed_table).pointer,
            embed_norm: shared.weights.get(embed_norm).pointer,
            head,
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
        let parts = rank_vocab.div_ceil(BLOCK as usize * LOGIT_PER_THREAD);
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
                            layout.take(ring * kv_width * f),
                            layout.take(ring * kv_width * f),
                        )
                    }),
                    ring,
                )
            })
            .collect();
        let taps = layout.take(tap_rows * TAPS * hidden * 2);
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
                layers: model.tap_layers,
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
        self.sink
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
        let tap_row = TAPS * hidden * 2;
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
                        model.alpha.offset(index * TAPS * 2).0,
                        layer.fusion_norm.0,
                        self.fused.0,
                        hidden as u64,
                        TAPS as u64,
                        bits(model.eps),
                    ],
                ))?;
                let walk = if take <= BATCH_ROWS {
                    PrefillWalk::Chunk
                } else {
                    shared.prefill_walk
                };
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
                model.alpha.offset((FUSION_ROWS - 1) * TAPS * 2).0,
                model.markov_norm.0,
                self.feature.0,
                hidden as u64,
                TAPS as u64,
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
        let elementwise = |entry: Entry, args: &[u64], count: usize| {
            shared.launch(entry, (flat(count), 1, 1), (ELEMENT_BLOCK, 1, 1), args)
        };

        // Row 0: the anchor's normed embedding, which only the rank that owns the id holds.
        match (anchor as usize).checked_sub(first_id) {
            Some(local) if local < rank_vocab => {
                run(shared.call(
                    shared.kernels.embed,
                    sparkflash_ops::calls::embed(
                        model.embed_table.0,
                        self.block.0,
                        local as u32,
                        (hidden, hidden),
                    ),
                ))?;
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
            for (weight, out, width) in [
                (layer.query, self.queries, q_width),
                (layer.key, self.keys, kv_width),
                (layer.value, self.values, kv_width),
            ] {
                run(shared.gemv_batch(weight, self.normed, out, width, hidden, rows, 1))?;
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
            run(shared.launch(
                kernels.attention_part,
                (kv_heads as u32, ATTENTION_CHUNKS as u32, 1),
                (
                    (group_queries.div_ceil(ATTENTION_QUERIES) * 32) as u32,
                    1,
                    1,
                ),
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
                    ATTENTION_CHUNKS as u64,
                ],
            ))?;
            run(elementwise(
                kernels.round,
                &[self.attention.0, (rows * q_width) as u64],
                rows * q_width,
            ))?;
            run(shared.gemv_batch(
                layer.out,
                self.attention,
                self.delta,
                hidden,
                q_width,
                rows,
                1,
            ))?;
            exchange_sum(shared, self.delta, rows * hidden)?;
            run(elementwise(
                kernels.add_round,
                &[self.block.0, self.delta.0, (rows * hidden) as u64],
                rows * hidden,
            ))?;
            run(norm(self.block, layer.post_norm, self.normed, rows, hidden))?;
            let inner = model.intermediate;
            run(shared.gemv_batch(layer.gate, self.normed, self.gate, inner, hidden, rows, 1))?;
            run(shared.gemv_batch(layer.up, self.normed, self.up, inner, hidden, rows, 1))?;
            run(elementwise(
                kernels.swiglu,
                &[self.gate.0, self.up.0, self.gate.0, (rows * inner) as u64],
                rows * inner,
            ))?;
            run(shared.gemv_batch(layer.down, self.gate, self.delta, hidden, inner, rows, 1))?;
            exchange_sum(shared, self.delta, rows * hidden)?;
            run(elementwise(
                kernels.add_round,
                &[self.block.0, self.delta.0, (rows * hidden) as u64],
                rows * hidden,
            ))?;
        }
        run(norm(self.block, model.norm, self.normed, rows, hidden))?;
        // The base logits of rows 1..=depth: this rank's slice of the head.
        run(shared.gemv_batch(
            model.head,
            self.normed.offset(hidden * f),
            self.base,
            rank_vocab,
            hidden,
            depth,
            1,
        ))?;

        let markov = model.rank;
        let z_width = 2 * markov + 2 * hidden;
        let padded = rank_vocab * world;
        let valid = config.unpadded_vocab_size.min(padded).max(1);
        let cap = config.final_logit_softcap.unwrap_or(0.0) as f32;
        let (inv_t, seed) = noise.filter(|(inv_t, _)| *inv_t > 0.0).unwrap_or((0.0, 0));
        let noisy = inv_t > 0.0;
        run(shared.cuda.memset_d8_async(self.state, 0, markov * f))?;
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
            run(shared.gemv(
                model.rnn_weight,
                self.z,
                self.projected,
                3 * markov,
                z_width,
            ))?;
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
        }
        run(shared.launch(
            kernels.markov_last,
            (1, 1, 1),
            (32, 1, 1),
            &[
                self.slots.0,
                world as u64,
                self.tokens.0,
                depth as u64,
                self.confidence.0,
                u64::from(noisy),
            ],
        ))?;
        run(shared.cuda.synchronize())?;
        run(shared.exchange.check_device())?;
        // Word 0 of each is the anchor's, which no step picks.
        let mut bytes = vec![0u8; (depth + 1) * 4];
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
        run(shared.cuda.copy_dtoh(&mut bytes, &self.confidence))?;
        let confidence: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .skip(1)
            .map(|word| f32::from_le_bytes(*word))
            .collect();
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
