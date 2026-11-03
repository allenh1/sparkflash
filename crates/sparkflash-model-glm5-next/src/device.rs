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

//! The device arm: one tensor-parallel rank of GLM-5.3-Flash on one GPU.
//!
//! [DeviceRank::load] uploads rank `layout.rank`'s slice of every tensor the served plan names, in
//! the format the checkpoint stores it, or [DeviceRank::load_converted] reads the same bytes from
//! the rank's converted file ([crate::convert]). [DeviceRank::forward_tp] and
//! [DeviceRank::forward_batch] run tokens through it with the Qwen4-Exp pack's kernels
//! ([sparkflash_model_qwen4_exp::device]): the tiled decode GEMVs, the Gated DeltaNet convolution
//! and recurrence in their per-channel KDA form, the selected-key attention, and the GLM kernels at
//! the end of that pack's source (the mHC mixers, the clamped SwiGLU, the noaux router, the k-pool
//! indexer and the routed experts' FP8 GEMV).
//!
//! A rank is the host twin ([crate::forward]) step for step: every sublayer's partial sum crosses
//! the host for the same [TpRuntime] all-reduce, and the LM head's rows for the same
//! [BitGatherHandle], so a device world and a host world see the same seams. The kernels sum in
//! their own association, so the two agree within a tolerance, not bit for bit; the ranks of one
//! device world agree with each other bit for bit.
//!
//! # Device leg
//!
//! A collective with a device leg ([TpRuntime::device_wire] of [Wire::F32]) keeps the step on the
//! device: each sublayer's partials are exchanged between the ranks' devices and summed in rank
//! order from 0.0 inside q4e_mhc_post, Wire::F32's sum on the host, and the LM head writes this
//! rank's slot of the gathered logits, which one exchange fills; the step's one host copy is the
//! gathered logits. The bits are the host round trip's, and a device leg on another wire is
//! refused.
//!
//! # Formats
//!
//! A BF16 linear runs q4e_gemv_tiled_bf16_parts on its f32 input, and a block-scaled FP8 one
//! q4e_gemv_tiled_e4m3_block, which quantises its input per 128 columns as the host's
//! [sparkflash_weights::quant::quantize_activation_fp8_tiles] does, or multiplies it unquantised
//! when the configuration clears [Glm5NextConfig::dynamic_activations], as the host twin then
//! does. An f32 vector uploads as
//! stored or widened exactly from BF16; a norm, convolution or embedding the kernels read as BF16
//! uploads as stored. Nothing is converted in a way that rounds: a tensor in another format is
//! refused.
//!
//! # Fused projections
//!
//! Projections that read the same input upload as one matrix, their rows concatenated, so a
//! sublayer launches one GEMV where the host multiplies several: KDA's q, k, v, f_a, g_a and b;
//! the MLA's q_a and kv_a; the indexer's wk, compress gate and weights_proj; and every MLP's gate
//! and up. A row's sum reads only its own weights; the fused shape sets only the split count.
//! The GEMV writes each part as one block of every row's outputs (Q4eTgPartStore), the
//! contiguous input the next kernel reads; the MLP's gate and up stay one part, each row's
//! `[gate | up]` together, as the SwiGLU reads them.
//!
//! # Rows
//!
//! A step carries up to [MAX_ROWS] rows, the next tokens of one sequence or of several, through
//! each weight once: every GEMV, norm, mixer and router runs once over all of them, and only what
//! touches a sequence's own state (the convolution and recurrence, the cache appends, the indexer
//! and the attention) runs once per sequence on its rows, in order. Each kernel computes a row as
//! it computes that row alone, so a sequence's logits are the same bits whatever it shares a step
//! with and however its tokens are cut into steps. The partials of all the rows cross the host in
//! one all-reduce a sublayer, and the LM head's rows in one gather a step.
//!
//! # Drafting
//!
//! A sequence drafts with the MTP layer ([DeviceRank::new_drafting_sequence]) or with a DFlash 2
//! block drafter ([DeviceRank::load_block_drafter]): Qwen4-Exp's ([GpuDflash2]), which a rank runs
//! through [DraftDevice] on its own stream. Every rank feeds the block drafter the same bits (a
//! target layer's streams after the all-reduce) and holds its share of the drafter's heads and MLP
//! width; each sublayer's partials are all-reduced as the model's are, so every rank's residual
//! stream is the same. Each scores the block's mask rows with an E4M3 copy of its own slice of the
//! LM head. The ranks exchange each mask row's top candidates in their slices, and merging them
//! gives the whole vocabulary's on every rank, so every rank walks the same chain.
//!
//! The MTP layer's drafts score with the same E4M3 copy ([DeviceRank::draft_head]): a draft reads
//! half the BF16 head's bytes, and a draft is only a proposal the verify checks against the BF16
//! head, so the copy changes how many drafts a verify keeps, never what it keeps.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use sparkflash_kernels::cuda_driver::{
    Cuda, DEFAULT_DIRECT_CHUNK_BYTES, DEFAULT_DIRECT_CONCURRENCY, Deviceptr, HostMapped, Kernel,
    RunStream, SideStream,
};
use sparkflash_model::{BackendError, StepCost};
use sparkflash_model_qwen4_exp::device::{
    CONV_HISTORY, DraftDevice, DraftScratch, GDN_KEY, GDN_QK_ROW, GDN_VALUE, GEMV_MAX_BATCH,
    GpuDflash2, GpuMat, GpuMatE4m3, MAX_BUDGET_BLOCKS, MAX_SELECTED_TOKENS, TILED_GEMV_BLOCKS,
    TILED_GEMV_PARTIALS, TILED_GEMV_ROUND, cubin_available, embedded_cubin,
    embedded_glm_prefill_cubin, embedded_prefill_cubin, gdn_scan_shape, gemv_e4m3, reduce_two_shot,
    splits_two_shot, tiled_gemv_e4m3_splits, tiled_gemv_splits,
};
use sparkflash_weights::linalg::{ScaleGrid, argmax};
use sparkflash_weights::mhc::HcGeometry;
use sparkflash_weights::tp::{BitGatherHandle, TpLayout, TpRuntime, Wire};
use sparkflash_weights::{DType, TensorSpec, WeightSource};

use crate::config::{Glm5NextConfig, IndexerKind, LayerKind, MlpKind};
use crate::convert::{ConvertedSlot, ConvertedSpecs, expect_slot};
use crate::forward::{INDEX_KEY_NORM_EPS, inverse_sqrt};
use crate::tensor::{Glm5NextTensorPlan, LM_PREFIX, Split};

/// The representation a sequence caches the MLA latent in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatentCache {
    /// The representation the expansion itself used: E4M3 when the projections quantise their
    /// activations (an FP8 release), f32 otherwise. The parity tests run this.
    #[default]
    Auto,
    /// The normed latent as f32: the expansion's own operand, exact, 2 KiB a position a layer.
    F32,
    /// The release's E4M3 activation: 528 B a position a layer, which is what fits the model's
    /// full window. It quantises the latent, so it is not the expansion's arithmetic; vLLM's
    /// \`fp8_ds_mla\` makes the same trade.
    Fp8,
}

/// How the bulk prefill's GEMMs take their f32 activations (SPARKFLASH_GLM_PREFILL_ACTIVATION).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillActivation {
    /// `fp8`, the default for a release whose projections quantise their activations: an FP8
    /// weight's input as the release's dynamic E4M3 codes (per 128 columns, as the decode GEMV
    /// quantises it) into an E4M3 x E4M3 MMA, and a BF16 weight's as [Self::Split].
    Fp8,
    /// `split`, the default otherwise: each activation as a bf16 head and remainder, two MMA
    /// chains a stage.
    Split,
    /// `bf16`: the bf16 head alone, one chain.
    Bf16,
}

impl PrefillActivation {
    fn from_env(config: &Glm5NextConfig) -> Result<Self, BackendError> {
        match std::env::var("SPARKFLASH_GLM_PREFILL_ACTIVATION") {
            Err(std::env::VarError::NotPresent) => Ok(if config.dynamic_activations {
                Self::Fp8
            } else {
                Self::Split
            }),
            Ok(value) => match value.as_str() {
                "fp8" if config.dynamic_activations => Ok(Self::Fp8),
                "fp8" => Err(refusal(
                    "SPARKFLASH_GLM_PREFILL_ACTIVATION=fp8 quantises activations this release \
                     does not; take split or bf16",
                )),
                "split" => Ok(Self::Split),
                "bf16" => Ok(Self::Bf16),
                other => Err(refusal(format!(
                    "SPARKFLASH_GLM_PREFILL_ACTIVATION={other} is none of fp8, split and bf16"
                ))),
            },
            Err(error) => Err(refusal(format!(
                "SPARKFLASH_GLM_PREFILL_ACTIVATION is not readable: {error}"
            ))),
        }
    }
}

/// The most rows one step carries: the tiled GEMVs' batch.
pub const MAX_ROWS: usize = GEMV_MAX_BATCH;

/// The default bulk prefill chunk. A chunk of more than eight rows reads every weight once for
/// the chunk instead of once for eight rows, and 4,096 is the widest measured: about 315 tok/s at a
/// 6.1k-token prompt against 205 at 512 rows, at about 8 GiB of chunk scratch (the per-layer
/// buffers shared, no verify windows) beside the 78 GB of weights and the 1M window. 8,192 rows
/// does not load, so 4,096 is also the cap.
pub const DEFAULT_PREFILL_ROWS: usize = 4096;
/// The narrowest bulk chunk: below this the token GEMVs remain the prefill path.
pub const MIN_PREFILL_ROWS: usize = 16;
/// The widest chunk the scratch is sized for at the default context.
pub const MAX_PREFILL_ROWS: usize = 4096;

/// The modelled wall of a block-drafted round at TP=4 on four GB10-class nodes that verifies
/// `depth` drafts (ms), for [DepthPolicy](sparkflash_model::DepthPolicy): the drafter's block and
/// the one-token step at 41.7 ms, and 5.9 ms a verified draft (mostly the experts its row adds).
/// Fit to one stream's 59.4, 71.6 and 82.9 ms rounds at 3, 5 and 7 drafts over the NCCL leg; the
/// first round assumes the 46 tok/s the drafter made there at 3.
pub const DFLASH2_STEP_COST: StepCost = StepCost {
    step_ms: 41.7,
    row_ms: 5.9,
    prior_tokens_per_ms: 0.046,
};

/// The side of a block-scaled FP8 tile the kernels read (Q4E_FP8_BLOCK).
const FP8_BLOCK: usize = 128;
/// The dynamic shared memory of the bulk prefill's E4M3 x E4M3 GEMMs (GLM_Q8_SHARED): their
/// cp.async stages and a column tile's weight scales.
const Q8_SHARED: u32 = 68_608;
/// The widest inner those GEMMs take (GLM_Q8_BLOCKS 128-column blocks): the kernel holds a column
/// tile's weight scales in shared memory and does nothing for a wider one.
const Q8_MAX_INNER: usize = 128 * FP8_BLOCK;
/// Threads in a block of every kernel the arm launches (Q4E_BLOCK and Q4E_TG_THREADS), but the
/// recurrence's and the router's one warp and the argmax's [ARGMAX_BLOCK]. The routed GEMV holds
/// a step's picks a thread each.
const BLOCK: u32 = 256;
/// Threads in a block of q4e_argmax_slices, which takes a row's whole vocabulary: the 32 warps its
/// block reduction holds.
const ARGMAX_BLOCK: u32 = 1024;
/// Threads in a block of q4e_gdn_recur, and the rows of a tiled GEMV's tile.
const RECUR_THREADS: u32 = 128;
/// The most blocks q4e_gdn_recur_columns splits a KDA head's value columns over: each column is its
/// own chain, so the head's state streams through that many SMs with the whole head's bits.
const RECUR_PARTS: usize = 4;
const TILE_ROWS: usize = 32;
/// The most rows one grouped tiled GEMV launch takes ([DeviceRank::gemv_small]): that many over
/// [MAX_ROWS] groups, each with its own split partials and tile counters in the scratch.
const GEMV_GROUP_ROWS: usize = 512;
const GEMV_GROUPS: usize = GEMV_GROUP_ROWS / MAX_ROWS;
/// The most (copy, row) pairs of an `fn` projection q4e_mhc_post works out: Q4E_MHC_PAIRS a warp.
const MHC_FUSED_PAIRS: usize = 8 * 12;
/// Pools a block of q4e_glm_index_scores scores (Q4E_QSA_TILE).
const SCORE_TILE: usize = 64;
/// glm_prefill_mla_attend's geometry (GMA_HEADS, GMA_LATENT): a rank's heads as one MMA's M, over
/// the latent's four 128-column scale blocks.
const GMA_HEADS: usize = 16;
const GMA_LATENT: usize = 512;
/// The decode attention's bounds (qwen4_exp_sm121.cu's Q4E_GLM_MLA_HEADS and
/// Q4E_GLM_MAX_LATENT): a rank's heads q4e_glm_mla_scores stages, over the widest latent.
const MLA_HEADS: usize = 16;
const MLA_MAX_LATENT: usize = 512;
/// q4e_glm_mla_scores' positions a block, q4e_glm_mla_mix's heads a block and
/// q4e_glm_mla_project's value columns a block.
const MLA_SCORE_SLOTS: usize = 64;
const MLA_MIX_HEADS: usize = 8;
const MLA_PROJECT_THREADS: u32 = 64;
/// A routed-expert item's rows (glm_prefill_sm121.cu's GLM_MOE_ROWS): one expert's slots, one
/// 128-row MMA tile.
const MOE_ITEM_ROWS: usize = 128;

/// The most items `slots` routed slots over `experts` experts lay out: every expert's last item
/// may be partial.
fn moe_items(slots: usize, experts: usize) -> usize {
    slots.div_ceil(MOE_ITEM_ROWS) + experts.min(slots) + 1
}
/// The bounds of the kernels' shared buffers: Q4E_MAX_EXPERTS, Q4E_MAX_TOPK, Q4E_MHC_MAX_HC,
/// Q4E_GLM_MAX_POOL, Q4E_GLM_MAX_INDEX_HEADS and Q4E_GLM_MAX_INDEX_DIM, and q4e_attn's head width.
const MAX_EXPERTS: usize = 512;
const MAX_TOPK: usize = 64;
const MAX_HC: usize = 5;
/// Q4E_MHC_COLUMNS: the columns of every stream copy a thread of q4e_mhc_in holds, so the hidden
/// width is a multiple of [BLOCK] of at most this many.
const MHC_COLUMNS: usize = 16;
const MAX_POOL: usize = 16;
const MAX_INDEX_HEADS: usize = 32;
const MAX_INDEX_DIM: usize = 128;
const MAX_ATTENTION_HEAD_DIM: usize = 256;

fn cuda_error(error: String) -> BackendError {
    BackendError::new("glm5_device", format!("CUDA: {error}"))
}

/// Refuses an inner wider than the bulk E4M3 x E4M3 GEMMs take ([Q8_MAX_INNER]).
fn q8_inner(columns: usize) -> Result<(), BackendError> {
    if columns > Q8_MAX_INNER {
        return Err(BackendError::new(
            "glm5_device",
            format!(
                "an FP8-activation prefill GEMM over {columns} columns is wider than its \
                 {Q8_MAX_INNER}; SPARKFLASH_GLM_PREFILL_ACTIVATION=split takes it"
            ),
        ));
    }
    Ok(())
}

/// The CUDA context a rank and its sequences share, owned so a rank outlives the call that opened
/// it and a sequence can hold the handle for as long as its allocations live.
///
/// The driver's context is process-wide and retained by the driver; this wrapper is what makes that
/// handle `Send`/`Sync` for the serving worker threads, exactly as the Qwen4-Exp arm's
/// `CudaHandle` does. Every device call goes through it, and its `Deref` keeps the call sites on
/// the driver API.
pub struct CudaHandle(Cuda);

impl CudaHandle {
    /// Takes ownership of an opened context.
    #[must_use]
    pub fn new(cuda: Cuda) -> Self {
        Self(cuda)
    }
}

impl std::ops::Deref for CudaHandle {
    type Target = Cuda;

    fn deref(&self) -> &Cuda {
        &self.0
    }
}

// SAFETY: the handle owns one retained driver context, and the driver's context is process-wide:
// every method this crate calls on it is a query or a launch on a stream the caller names, and the
// CUDA driver serialises context access. A session owns its own stream and buffers, and a rank's
// forward takes `&mut self`, so a `Send` handle cannot be used concurrently for the same rank's
// state.
unsafe impl Send for CudaHandle {}
// SAFETY: as above: the handle itself carries no mutable per-thread state, and the run-scope
// caches behind it lock internally.
unsafe impl Sync for CudaHandle {}

pub(crate) fn refusal(detail: impl Into<String>) -> BackendError {
    BackendError::new("glm5_config", detail)
}

fn other_kind() -> BackendError {
    refusal("a layer state of the other attention kind")
}

fn weights_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new("glm5_weights", error.to_string())
}

/// A float kernel argument: its bits in the low word of the slot.
fn float(value: f32) -> u64 {
    u64::from(value.to_bits())
}

/// Blocks of [BLOCK] threads covering `count` elements.
fn blocks(count: usize) -> u32 {
    count.div_ceil(BLOCK as usize).max(1) as u32
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// This rank's slice of a stored tensor's bytes, `width` bytes an element: the split is checked
/// in elements and cut in bytes, an element's bytes being the last dimension's innermost run. A
/// tensor the split leaves whole comes back as it was read, so a replicated table is never copied.
fn slice_stored(
    split: Split,
    shape: &[usize],
    bytes: Vec<u8>,
    width: usize,
    layout: TpLayout,
) -> Result<(Vec<usize>, Vec<u8>), String> {
    if bytes.len() != elements(shape) * width {
        return Err(format!(
            "{} bytes are not a {shape:?} tensor of {width}-byte elements",
            bytes.len()
        ));
    }
    let local = split.rank_shape(shape, layout.world_size)?;
    if local == shape {
        return Ok((local, bytes));
    }
    let mut in_bytes = shape.to_vec();
    if let Some(last) = in_bytes.last_mut() {
        *last *= width;
    }
    let (_, kept) = split.slice(&in_bytes, bytes, layout.rank, layout.world_size)?;
    Ok((local, kept))
}

/// Appends `bytes` to `into`, taking them as they are when nothing precedes them.
fn append(into: &mut Vec<u8>, bytes: Vec<u8>) {
    if into.is_empty() {
        *into = bytes;
    } else {
        into.extend(bytes);
    }
}

/// The kernels a rank launches, by name from the pack's cubin.
#[derive(Clone, Copy)]
struct Kernels {
    embed: Kernel,
    argmax: Kernel,
    gemv_bf16: Kernel,
    gemv_fp8: Kernel,
    /// The E4M3 copy of the LM head drafts score with ([DeviceRank::draft_head]): its quantizer
    /// and its GEMV.
    quantize_e4m3: Kernel,
    gemv_e4m3: Kernel,
    expert_fp8: Kernel,
    mhc_in: Kernel,
    mhc_post: Kernel,
    sum_slots: Kernel,
    rmsnorm: Kernel,
    rmsnorm_rows: Kernel,
    conv: Kernel,
    kda_decay: Kernel,
    recur: Kernel,
    /// A bulk chunk's convolution and recurrence (Qwen's prefill image,
    /// kernels/qwen4_prefill_sm121.cu): the convolution a thread an output, then the history
    /// advanced, the token kernel's products in its order; the rows' normed q and k and beta,
    /// the scan with the state in registers, and the gated norm. The token kernels ([Self::conv],
    /// [Self::recur]) stay the decode and verify path, whose bits a replay is defined by.
    conv_rows: Kernel,
    conv_history: Kernel,
    kda_prep: Kernel,
    kda_scan: Kernel,
    kda_gate: Kernel,
    copy_flat: Kernel,
    layer_norm: Kernel,
    pool: Kernel,
    index_scores: Kernel,
    pick: Kernel,
    /// The absorbed MLA ([DeviceRank::dsa_attend]): the latent cache writer, the query's
    /// projection into the latent space, then the attention's logits, softmax, weighted latent
    /// and value projection.
    mla_cache: Kernel,
    mla_cache_fp8: Kernel,
    mla_absorb: Kernel,
    mla_scores: Kernel,
    mla_softmax: Kernel,
    mla_mix: Kernel,
    mla_project: Kernel,
    swiglu: Kernel,
    router: Kernel,
    combine: Kernel,
    fold: Kernel,
    prefetch_l2: Kernel,
    /// The bulk prefill image (kernels/glm_prefill_sm121.cu): dense block-scaled FP8 and BF16
    /// GEMMs, the padded routed-expert plan, and its gather/SwiGLU/scatter.
    prefill_dense_bf16: Kernel,
    prefill_dense_fp8: Kernel,
    /// The same two GEMMs over the bf16 head of each activation alone, and the routed one.
    prefill_dense_bf16_bf16a: Kernel,
    prefill_dense_fp8_bf16a: Kernel,
    prefill_moe_mma_bf16a: Kernel,
    prefill_moe_plan: Kernel,
    prefill_moe_mma: Kernel,
    prefill_moe_swiglu: Kernel,
    /// The checkpoint's dynamic activation scheme ([PrefillActivation::Fp8]): rows quantised per
    /// 128 columns, the SwiGLU into the down projection's codes, and the E4M3 x E4M3 dense and
    /// routed GEMMs over them.
    prefill_quantise: Kernel,
    prefill_moe_swiglu_q8: Kernel,
    prefill_dense_q8: Kernel,
    prefill_moe_q8: Kernel,
    /// A BF16 part of at most 8 (i + 1) rows ([DeviceRank::bulk_gemm]'s mixers and gates).
    prefill_skinny: [Kernel; 4],
    /// The chunk's MLA ([DeviceRank::bulk_attend]): the tensor-core sparse attention over the
    /// E4M3 latent (and over its bf16 heads alone), and the per-head GEMM of the absorbed query
    /// and the value projection.
    prefill_mla_attend: Kernel,
    prefill_mla_attend_bf16a: Kernel,
    prefill_heads_bf16: Kernel,
    prefill_heads_fp8: Kernel,
}

// SAFETY: a kernel is an immutable function handle loaded from the cubin. Every launch goes
// through a rank's `&mut self` on that rank's own stream, so a handle that moves to another
// thread is never used concurrently.
unsafe impl Send for Kernels {}

impl Kernels {
    fn load(cuda: &Cuda, head_dim: usize) -> Result<Self, String> {
        let named = |name| {
            cuda.load_named(embedded_cubin(), name)
                .map(|(_, kernel)| kernel)
        };
        if embedded_glm_prefill_cubin().is_empty() {
            return Err("this build embeds no GLM bulk prefill cubin".to_owned());
        }
        let prefill = |name| {
            cuda.load_named(embedded_glm_prefill_cubin(), name)
                .map(|(_, kernel)| kernel)
        };
        let prefill_q8 = |name| {
            cuda.load_named_shared(embedded_glm_prefill_cubin(), name, Q8_SHARED)
                .map(|(_, kernel)| kernel)
        };
        if embedded_prefill_cubin().is_empty() {
            return Err("this build embeds no prefill cubin for the bulk KDA scan".to_owned());
        }
        let scan = |name| {
            cuda.load_named(embedded_prefill_cubin(), name)
                .map(|(_, kernel)| kernel)
        };
        // The recurrence stages a block's slice of a head's state in shared memory.
        let (_, recur) = cuda.load_named_shared(
            embedded_cubin(),
            "q4e_gdn_recur_columns",
            (head_dim * head_dim / recur_parts(head_dim) * 4) as u32,
        )?;
        Ok(Self {
            embed: named("q4e_embed")?,
            argmax: named("q4e_argmax_slices")?,
            gemv_bf16: named("q4e_gemv_tiled_bf16_parts")?,
            gemv_fp8: named("q4e_gemv_tiled_e4m3_block")?,
            quantize_e4m3: named("q4e_quantize_e4m3_rows")?,
            gemv_e4m3: named("q4e_gemv_tiled_e4m3")?,
            expert_fp8: named("q4e_expert_gemv_e4m3_block")?,
            mhc_in: named("q4e_mhc_in")?,
            mhc_post: named("q4e_mhc_post")?,
            sum_slots: named("q4e_sum_slots")?,
            rmsnorm: named("q4e_rmsnorm")?,
            rmsnorm_rows: named("q4e_rmsnorm_rows")?,
            conv: named("q4e_gdn_conv")?,
            kda_decay: named("q4e_kda_decay")?,
            recur,
            conv_rows: scan("q4p_gdn_conv")?,
            conv_history: scan("q4p_gdn_conv_state")?,
            kda_prep: scan("q4p_kda_prep")?,
            kda_scan: scan("q4p_kda_scan")?,
            kda_gate: scan("q4p_gdn_gate")?,
            copy_flat: named("q4e_copy_flat")?,
            layer_norm: named("q4e_layer_norm")?,
            pool: named("q4e_glm_pool")?,
            index_scores: named("q4e_glm_index_scores")?,
            pick: named("q4e_glm_pick")?,
            mla_cache: named("q4e_glm_mla_cache")?,
            mla_cache_fp8: named("q4e_glm_mla_cache_fp8")?,
            mla_absorb: named("q4e_glm_mla_absorb")?,
            mla_scores: named("q4e_glm_mla_scores")?,
            mla_softmax: named("q4e_glm_mla_softmax")?,
            mla_mix: named("q4e_glm_mla_mix")?,
            mla_project: named("q4e_glm_mla_project")?,
            swiglu: named("q4e_swiglu_limited")?,
            router: named("q4e_noaux_router")?,
            combine: named("q4e_moe_combine")?,
            prefetch_l2: named("q4e_prefetch_l2")?,
            fold: named("q4e_fold_streams")?,
            prefill_dense_bf16: prefill("glm_prefill_bf16_mma")?,
            prefill_dense_fp8: prefill("glm_prefill_fp8b128_mma")?,
            prefill_dense_bf16_bf16a: prefill("glm_prefill_bf16_mma_bf16a")?,
            prefill_dense_fp8_bf16a: prefill("glm_prefill_fp8b128_mma_bf16a")?,
            prefill_moe_mma_bf16a: prefill("glm_moe_fp8b128_mma_bf16a")?,
            prefill_moe_plan: prefill("glm_moe_plan")?,
            prefill_moe_mma: prefill("glm_moe_fp8b128_mma")?,
            prefill_moe_swiglu: prefill("glm_moe_swiglu")?,
            prefill_quantise: prefill("glm_quantise_rows")?,
            prefill_moe_swiglu_q8: prefill("glm_moe_swiglu_q8")?,
            prefill_dense_q8: prefill_q8("glm_prefill_q8_mma")?,
            prefill_moe_q8: prefill_q8("glm_moe_q8_mma")?,
            prefill_skinny: [
                prefill("glm_prefill_skinny_bf16_1")?,
                prefill("glm_prefill_skinny_bf16_2")?,
                prefill("glm_prefill_skinny_bf16_3")?,
                prefill("glm_prefill_skinny_bf16_4")?,
            ],
            prefill_mla_attend: prefill("glm_prefill_mla_attend")?,
            prefill_mla_attend_bf16a: prefill("glm_prefill_mla_attend_bf16a")?,
            prefill_heads_bf16: prefill("glm_prefill_heads_bf16_mma")?,
            prefill_heads_fp8: prefill("glm_prefill_heads_fp8b128_mma")?,
        })
    }
}

/// The blocks q4e_gdn_recur_columns splits a KDA head's `head` value columns over: up to
/// [RECUR_PARTS], each a multiple of four columns, its copies' width.
fn recur_parts(head: usize) -> usize {
    (1..=RECUR_PARTS)
        .rev()
        .find(|parts| head.is_multiple_of(4 * parts))
        .unwrap_or(1)
}

/// This rank's kv_b rows as the absorbed MLA reads them, f32: the checkpoint's E4M3 codes times
/// their 128x128 block scales, or its BF16 values widened. A head keeps its `nope` key rows as
/// they are, `[nope][kv_lora]`, then its `value` rows transposed, `[kv_lora][value]`: each kernel
/// walks the latent in order for one output, and either way a warp's loads are consecutive words.
///
/// A rank's copy is 16.8 MB at the release's geometry, against the 82 GB rank. Dequantising once
/// at load keeps every attention step a plain f32 dot product, and the cached latent carries the
/// release's activation quantisation, so the products are the expansion's own.
///
/// A BF16 kv_b also uploads its key rows transposed as BF16, `[head][kv_lora][nope]`, the B operand
/// the bulk prefill's per-head absorption GEMM reads (a widened BF16 weight narrows back exactly).
fn absorb_kv_b(
    cuda: &Cuda,
    stream: &RunStream,
    memory: &mut Allocations,
    linear: &DeviceLinear,
    nope: usize,
    value: usize,
) -> Result<(Deviceptr, Option<Deviceptr>), BackendError> {
    let (rows, columns) = (linear.rows, linear.columns);
    let per_head = nope + value;
    if per_head == 0 || !rows.is_multiple_of(per_head) {
        return Err(refusal(format!(
            "kv_b keeps {rows} rows here, not whole heads of {nope} key and {value} value rows"
        )));
    }
    let widened = dequantise_kv_b(cuda, linear)?;
    let mut laid = Vec::with_capacity(widened.len() * 4);
    let mut key_t = Vec::with_capacity(if linear.scales.is_none() {
        rows / per_head * nope * columns * 2
    } else {
        0
    });
    for head in widened.chunks_exact(per_head * columns) {
        let (keys, values) = head.split_at(nope * columns);
        for weight in keys {
            laid.extend_from_slice(&weight.to_le_bytes());
        }
        for column in 0..columns {
            for row in values.chunks_exact(columns) {
                laid.extend_from_slice(&row[column].to_le_bytes());
            }
        }
        if linear.scales.is_none() {
            for column in 0..columns {
                for row in keys.chunks_exact(columns) {
                    key_t.extend_from_slice(&((row[column].to_bits() >> 16) as u16).to_le_bytes());
                }
            }
        }
    }
    let key_t = if key_t.is_empty() {
        None
    } else {
        Some(memory.upload(stream, &key_t)?)
    };
    Ok((memory.upload(stream, &laid)?, key_t))
}

/// kv_b's rows, `[rows][columns]` row-major, as the f32 the absorbed MLA multiplies.
fn dequantise_kv_b(cuda: &Cuda, linear: &DeviceLinear) -> Result<Vec<f32>, BackendError> {
    let (rows, columns) = (linear.rows, linear.columns);
    let Some(scales) = linear.scales else {
        // BF16, widened exactly: its bits are an f32's top half.
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| refusal("kv_b's shape overflows a usize"))?;
        let mut bytes = vec![0u8; elements * 2];
        cuda.copy_dtoh(&mut bytes, &linear.weight)
            .map_err(cuda_error)?;
        return Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect());
    };
    if !rows.is_multiple_of(FP8_BLOCK) || !columns.is_multiple_of(FP8_BLOCK) {
        return Err(refusal(format!(
            "kv_b keeps {rows} x {columns} here, not whole {FP8_BLOCK}-wide FP8 tiles"
        )));
    }
    let codes = rows
        .checked_mul(columns)
        .ok_or_else(|| refusal("kv_b's shape overflows a usize"))?;
    let mut weight = vec![0u8; codes];
    cuda.copy_dtoh(&mut weight, &linear.weight)
        .map_err(cuda_error)?;
    let grid = ScaleGrid::Tile(FP8_BLOCK);
    let scale_count = grid.len(rows, columns);
    let mut scale_bytes = vec![0u8; scale_count * 4];
    cuda.copy_dtoh(&mut scale_bytes, &scales)
        .map_err(cuda_error)?;
    let scales_host: Vec<f32> = scale_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let bands = columns / FP8_BLOCK;
    let mut widened = Vec::with_capacity(codes);
    for row in 0..rows {
        let band = (row / FP8_BLOCK) * bands;
        for column in 0..columns {
            let scale = scales_host[band + column / FP8_BLOCK];
            widened.push(
                sparkflash_weights::quant::f8e4m3_to_f32(weight[row * columns + column]) * scale,
            );
        }
    }
    Ok(widened)
}

/// Device allocations, and page-locked host slabs the device maps, freed together when they drop.
struct Allocations {
    cuda: Arc<CudaHandle>,
    buffers: Vec<Deviceptr>,
    mapped: Vec<HostMapped>,
    bytes: usize,
}

impl Allocations {
    fn new(cuda: Arc<CudaHandle>) -> Self {
        Self {
            cuda,
            buffers: Vec::new(),
            mapped: Vec::new(),
            bytes: 0,
        }
    }

    /// `bytes` of page-locked host memory the device maps, which the host reads after a copy or
    /// a kernel writes without one.
    fn mapped(&mut self, bytes: usize) -> Result<HostMapped, BackendError> {
        let mut host_bytes = 0;
        let mapped = self
            .cuda
            .alloc_host_mapped(bytes.max(4), &mut host_bytes)
            .map_err(cuda_error)?;
        self.mapped.push(mapped);
        Ok(mapped)
    }

    fn upload(&mut self, stream: &RunStream, bytes: &[u8]) -> Result<Deviceptr, BackendError> {
        let pointer = self
            .cuda
            .alloc(bytes.len().max(4), &mut self.bytes)
            .map_err(cuda_error)?;
        self.buffers.push(pointer);
        if !bytes.is_empty() {
            self.cuda
                .copy_htod_on(stream, &pointer, bytes)
                .map_err(cuda_error)?;
        }
        Ok(pointer)
    }

    fn zeroed(&mut self, stream: &RunStream, bytes: usize) -> Result<Deviceptr, BackendError> {
        self.upload(stream, &vec![0; bytes])
    }

    /// `bytes` of device memory, left as the allocator returns it.
    fn bytes(&mut self, bytes: usize) -> Result<Deviceptr, BackendError> {
        let pointer = self
            .cuda
            .alloc(bytes.max(4), &mut self.bytes)
            .map_err(cuda_error)?;
        self.buffers.push(pointer);
        Ok(pointer)
    }

    /// `floats` f32 of scratch, left as the allocator returns it.
    fn floats(&mut self, floats: usize) -> Result<Deviceptr, BackendError> {
        let pointer = self
            .cuda
            .alloc(floats.max(1) * 4, &mut self.bytes)
            .map_err(cuda_error)?;
        self.buffers.push(pointer);
        Ok(pointer)
    }

    /// `len` bytes of `path` from `offset`, read in parallel direct-I/O chunks into one allocation.
    fn file(&mut self, path: &Path, offset: u64, len: usize) -> Result<Deviceptr, BackendError> {
        let pointer = self
            .cuda
            .alloc(len.max(4), &mut self.bytes)
            .map_err(cuda_error)?;
        self.buffers.push(pointer);
        self.cuda
            .upload_file_parallel(
                &pointer,
                path,
                offset,
                len,
                DEFAULT_DIRECT_CHUNK_BYTES,
                DEFAULT_DIRECT_CONCURRENCY,
                true,
            )
            .map_err(cuda_error)?;
        Ok(pointer)
    }
}

/// Where the [Uploader] puts a rank's weights, one upload at a time in a fixed order: device
/// allocations straight from the checkpoint ([Direct]), a converted rank file
/// ([crate::convert]), or the slab a converted rank file was read into ([Slab]).
pub(crate) trait Sink {
    /// Whether [Sink::weight] needs the bytes; a sink that already holds them, or only plans
    /// where they go, takes the shapes alone and is handed no bytes.
    fn wants_bytes(&self) -> bool;

    /// The device address of the next weight, `key`, `len` bytes, which are `bytes` when the sink
    /// wants them and empty when it does not.
    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, BackendError>;

    /// Uploads bytes built from addresses (the routed experts' pointer tables), which no rank file
    /// holds.
    fn derived(&mut self, bytes: &[u8]) -> Result<Deviceptr, BackendError>;

    /// Notes the spec of a checkpoint tensor read on the way to a weight.
    fn spec(&mut self, spec: &TensorSpec) {
        let _ = spec;
    }
}

/// A weight's bytes against the length its shapes give, which every sink handed bytes checks.
pub(crate) fn check_length(key: &str, len: usize, bytes: &[u8]) -> Result<(), BackendError> {
    if bytes.len() == len {
        Ok(())
    } else {
        Err(refusal(format!(
            "{key} came to {} bytes where its shapes give {len}",
            bytes.len()
        )))
    }
}

/// Each weight in its own device allocation, uploaded as it is read.
struct Direct<'a> {
    stream: &'a RunStream,
    memory: &'a mut Allocations,
}

impl Sink for Direct<'_> {
    fn wants_bytes(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, BackendError> {
        check_length(key, len, &bytes)?;
        self.memory.upload(self.stream, &bytes)
    }

    fn derived(&mut self, bytes: &[u8]) -> Result<Deviceptr, BackendError> {
        self.memory.upload(self.stream, bytes)
    }
}

/// The weights of a converted rank file, already on the device in one allocation at `base`: each
/// upload is the next slot of the file's index, which must be the same weight at the same length.
struct Slab<'a> {
    base: Deviceptr,
    slots: &'a [ConvertedSlot],
    next: usize,
    stream: &'a RunStream,
    memory: &'a mut Allocations,
}

impl Sink for Slab<'_> {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, BackendError> {
        let at = expect_slot(self.slots, self.next, key, len)?.offset;
        self.next += 1;
        Ok(self.base.offset(at))
    }

    fn derived(&mut self, bytes: &[u8]) -> Result<Deviceptr, BackendError> {
        self.memory.upload(self.stream, bytes)
    }
}

impl Drop for Allocations {
    fn drop(&mut self) {
        // No queued launch may outlive the buffers it reads.
        let _ = self.cuda.synchronize();
        let buffers: Vec<&Deviceptr> = self.buffers.iter().collect();
        self.cuda.free_buffers(&buffers);
        for mapped in &self.mapped {
            self.cuda.free_host_mapped(mapped);
        }
    }
}

/// One uploaded linear, `rows x columns`: BF16, or E4M3 with its f32 tile scales.
#[derive(Debug, Clone, Copy)]
struct DeviceLinear {
    weight: Deviceptr,
    /// `[ceil(rows / 128)][columns / 128]` for an E4M3 weight; `None` for a BF16 one.
    scales: Option<Deviceptr>,
    rows: usize,
    columns: usize,
    /// Where parts two to four of a fused projection's rows start, every one at `rows` for a
    /// single part: the GEMV writes each part's outputs for all its rows together.
    parts: [usize; 3],
}

impl DeviceLinear {
    /// This linear with its outputs in parts that start at `starts`.
    fn parted(self, starts: [usize; 3]) -> Self {
        Self {
            parts: starts,
            ..self
        }
    }

    /// Part `part`'s outputs in `output` after a GEMV over `rows` rows.
    fn part(&self, output: Deviceptr, part: usize, rows: usize) -> Deviceptr {
        let start = if part == 0 { 0 } else { self.parts[part - 1] };
        output.offset(rows * start * 4)
    }

    /// The weight's bytes, one a value for E4M3 and two for BF16, as a [DeviceRank::prefetch]
    /// span.
    fn span(&self) -> (Deviceptr, usize) {
        let size = if self.scales.is_some() { 1 } else { 2 };
        (self.weight, self.rows * self.columns * size)
    }
}

/// One mHC mixer: the BF16 `fn` projection and the f32 base and scale.
struct DeviceHyper {
    projection: DeviceLinear,
    base: Deviceptr,
    scale: Deviceptr,
}

/// A KDA layer's slice: this rank's heads.
struct DeviceKda {
    /// `[q; k; v; f_a; g_a; b]`, `3 width + 2 head_dim + heads` rows, in four parts: the
    /// convolution's `[q | k | v]`, f_a, g_a and b.
    fused: DeviceLinear,
    /// The q, k and v convolutions, `[3 width][taps]` BF16.
    conv: Deviceptr,
    f_b: DeviceLinear,
    g_b: DeviceLinear,
    a_log: Deviceptr,
    dt_bias: Deviceptr,
    o_norm: Deviceptr,
    o: DeviceLinear,
    heads: usize,
    width: usize,
}

/// An indexer, replicated on every rank.
struct DeviceIndexer {
    /// `[wk; compress gate; weights_proj]`, `2 index_head_dim + index_n_heads` rows, a part each.
    fused: DeviceLinear,
    wq_b: DeviceLinear,
    k_norm_weight: Deviceptr,
    k_norm_bias: Deviceptr,
    ape: Deviceptr,
}

/// A DSA layer's slice: this rank's heads.
struct DeviceDsa {
    /// `[q_a; kv_a]`, replicated, a part each.
    fused: DeviceLinear,
    q_a_norm: Deviceptr,
    kv_a_norm: Deviceptr,
    q_b: DeviceLinear,
    kv_b: DeviceLinear,
    o: DeviceLinear,
    indexer: Option<DeviceIndexer>,
    /// This rank's `kv_b` rows dequantised to f32, a head's key rows then its value rows
    /// transposed ([absorb_kv_b]): the two absorbed operands. See [DeviceRank::dsa_attend].
    kv_dequant: Deviceptr,
    /// Whether the latent cache holds the release's quantised activation.
    quantise: bool,
    /// This rank's BF16 kv_b key rows transposed, `[head][kv_lora][nope]`: the bulk prefill's
    /// absorption as a per-head GEMM ([DeviceRank::bulk_attend]). Absent for an FP8 kv_b, whose
    /// chunks absorb on the decode kernel.
    kv_key_t: Option<Deviceptr>,
    heads: usize,
}

/// A clamped SwiGLU MLP's slice: `[gate; up]` and down.
struct DeviceMlp {
    gate_up: DeviceLinear,
    down: DeviceLinear,
}

/// A MoE layer's slice: the router, the shared expert, and each routed expert's E4M3 weights and
/// scales as the `[experts][2]` pointer tables q4e_expert_gemv_e4m3_block reads.
struct DeviceMoe {
    router: DeviceLinear,
    bias: Deviceptr,
    shared: DeviceMlp,
    gate_up: Deviceptr,
    down: Deviceptr,
    /// One expert's slice of the intermediate width.
    intermediate: usize,
}

enum DeviceBranch {
    Linear(DeviceKda),
    Sparse(DeviceDsa),
}

enum DeviceFfn {
    Dense(DeviceMlp),
    Moe(DeviceMoe),
}

struct DeviceLayer {
    input_norm: Deviceptr,
    post_norm: Deviceptr,
    attn_hc: DeviceHyper,
    ffn_hc: DeviceHyper,
    branch: DeviceBranch,
    ffn: DeviceFfn,
}

impl DeviceMoe {
    /// The weights a MoE reads before its routed experts, in the order it reads them.
    fn reads(&self) -> [(Deviceptr, usize); 3] {
        [
            self.shared.gate_up.span(),
            self.shared.down.span(),
            self.router.span(),
        ]
    }
}

impl DeviceLayer {
    /// The weights the attention half reads first, in the order it reads them.
    fn attention_reads(&self) -> Vec<(Deviceptr, usize)> {
        let mut reads = vec![self.attn_hc.projection.span()];
        match &self.branch {
            DeviceBranch::Linear(kda) => reads.push(kda.fused.span()),
            DeviceBranch::Sparse(dsa) => {
                reads.extend([dsa.fused.span(), dsa.q_b.span(), dsa.kv_b.span()]);
            }
        }
        reads
    }

    /// The weights the FFN half reads first, in the order it reads them.
    fn ffn_reads(&self) -> Vec<(Deviceptr, usize)> {
        let mut reads = vec![self.ffn_hc.projection.span()];
        match &self.ffn {
            DeviceFfn::Dense(mlp) => reads.extend([mlp.gate_up.span(), mlp.down.span()]),
            DeviceFfn::Moe(moe) => reads.extend(moe.reads()),
        }
        reads
    }
}

/// The MTP layer's slice, the drafter. `eh_proj` fuses the normed embedding of a token with the
/// normed hidden the model left at the position before it; one DSA + MoE layer without
/// hyper-connections follows, and its output after its own norm is what the LM head scores and
/// the next draft's hidden.
struct DeviceMtp {
    enorm: Deviceptr,
    hnorm: Deviceptr,
    head_norm: Deviceptr,
    /// `[hidden][embedding | hidden]`, replicated.
    eh_proj: DeviceLinear,
    input_norm: Deviceptr,
    post_norm: Deviceptr,
    dsa: DeviceDsa,
    moe: DeviceMoe,
}

/// A rank's weights as the [Uploader] places them.
struct RankWeights {
    embed: Deviceptr,
    norm: Deviceptr,
    lm_head: DeviceLinear,
    layers: Vec<DeviceLayer>,
    /// The MTP layer, when the configuration names one.
    mtp: Option<DeviceMtp>,
}

/// Puts one rank's slice of the checkpoint into a [Sink], by the served plan's splits, in the
/// same order whatever the sink.
pub(crate) struct Uploader<'a, S: WeightSource + ?Sized> {
    source: &'a S,
    config: &'a Glm5NextConfig,
    plan: Glm5NextTensorPlan,
    layout: TpLayout,
    sink: &'a mut dyn Sink,
}

/// The elements of a shape.
fn elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl<'a, S: WeightSource + ?Sized> Uploader<'a, S> {
    pub(crate) fn new(
        source: &'a S,
        config: &'a Glm5NextConfig,
        layout: TpLayout,
        sink: &'a mut dyn Sink,
    ) -> Self {
        Self {
            source,
            config,
            plan: Glm5NextTensorPlan::served_and_mtp(config),
            layout,
            sink,
        }
    }

    /// Every weight of the rank: the embedding, the final norm, the LM head, each layer, then the
    /// MTP layer.
    fn weights(&mut self) -> Result<RankWeights, BackendError> {
        let (vocab, hidden) = (self.config.vocab_size, self.config.hidden_size);
        Ok(RankWeights {
            embed: self.bf16(
                &[format!("{LM_PREFIX}.embed_tokens.weight")],
                &[vocab, hidden],
            )?,
            norm: self.bf16(&[format!("{LM_PREFIX}.norm.weight")], &[hidden])?,
            lm_head: self.linear(&[("lm_head.weight".to_owned(), vocab, hidden)])?,
            layers: (0..self.config.num_hidden_layers)
                .map(|layer| self.layer(layer))
                .collect::<Result<Vec<_>, _>>()?,
            mtp: self.mtp()?,
        })
    }

    /// The MTP layer, if the configuration has one: the plan's MTP tensors by their splits, its
    /// attention and experts as a decoder layer's.
    fn mtp(&mut self) -> Result<Option<DeviceMtp>, BackendError> {
        let config = self.config;
        if config.num_nextn_predict_layers == 0 {
            return Ok(None);
        }
        let hidden = config.hidden_size;
        let prefix = format!("{LM_PREFIX}.layers.{}", config.mtp_layer());
        let mut norm = |name: &str| self.bf16(&[format!("{prefix}.{name}.weight")], &[hidden]);
        let (enorm, hnorm, head_norm) = (norm("enorm")?, norm("hnorm")?, norm("shared_head.norm")?);
        let (input_norm, post_norm) = (norm("input_layernorm")?, norm("post_attention_layernorm")?);
        Ok(Some(DeviceMtp {
            enorm,
            hnorm,
            head_norm,
            eh_proj: self.linear(&[(format!("{prefix}.eh_proj.weight"), hidden, 2 * hidden)])?,
            input_norm,
            post_norm,
            dsa: self.dsa(&format!("{prefix}.self_attn"), true)?,
            moe: self.moe(&format!("{prefix}.mlp"))?,
        }))
    }

    /// Places every weight, for a sink that keeps no addresses (a rank file's writer).
    pub(crate) fn place(&mut self) -> Result<(), BackendError> {
        self.weights().map(|_| ())
    }

    /// This rank's slice of `name` as the checkpoint stores it, once its shape is `shape`: its
    /// format, its shape on this rank, and its bytes when the sink wants them.
    fn stored(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<(DType, Vec<usize>, Vec<u8>), BackendError> {
        let spec = self
            .source
            .spec(name)
            .map_err(|_| refusal(format!("{name} is missing from the checkpoint")))?;
        if spec.shape != shape {
            return Err(refusal(format!(
                "{name} is stored as {:?}, expected {shape:?}",
                spec.shape
            )));
        }
        let split = self
            .plan
            .split_of(name)
            .ok_or_else(|| refusal(format!("{name} is a tensor the served plan does not name")))?;
        self.sink.spec(&spec);
        let rank_shape = split
            .rank_shape(shape, self.layout.world_size)
            .map_err(|detail| refusal(format!("{name}: {detail}")))?;
        if !self.sink.wants_bytes() {
            return Ok((spec.dtype, rank_shape, Vec::new()));
        }
        let bytes = self.source.read(name).map_err(weights_error)?;
        let (shape, bytes) = slice_stored(
            split,
            shape,
            bytes,
            spec.dtype.bytes_per_element(),
            self.layout,
        )
        .map_err(|detail| refusal(format!("{name}: {detail}")))?;
        if shape != rank_shape {
            return Err(refusal(format!(
                "{name} sliced to {shape:?}, where its split gives {rank_shape:?}"
            )));
        }
        Ok((spec.dtype, shape, bytes))
    }

    /// BF16 tensors, each `shape`, uploaded as stored and end to end.
    fn bf16(&mut self, names: &[String], shape: &[usize]) -> Result<Deviceptr, BackendError> {
        let mut bytes = Vec::new();
        let mut len = 0;
        for name in names {
            let (dtype, rank_shape, stored) = self.stored(name, shape)?;
            if dtype != DType::BF16 {
                return Err(refusal(format!(
                    "{name} is stored as {}, and its kernel reads BF16",
                    dtype.name()
                )));
            }
            len += dtype.storage_bytes(elements(&rank_shape));
            append(&mut bytes, stored);
        }
        self.sink.weight(&names.join("+"), len, bytes)
    }

    /// An f32 vector: stored F32, or BF16 widened exactly.
    fn f32(&mut self, name: &str, shape: &[usize]) -> Result<Deviceptr, BackendError> {
        let (dtype, rank_shape, bytes) = self.stored(name, shape)?;
        let len = 4 * elements(&rank_shape);
        let bytes = match dtype {
            DType::F32 => bytes,
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|pair| (u32::from(u16::from_le_bytes(*pair)) << 16).to_le_bytes())
                .collect(),
            other => {
                return Err(refusal(format!(
                    "{name} is stored as {}, and its kernel reads f32",
                    other.name()
                )));
            }
        };
        self.sink.weight(name, len, bytes)
    }

    /// The weights `parts` name (`name, rows, columns`), this rank's slices with their rows
    /// concatenated into one matrix. An E4M3 part brings its `{name}_scale_inv` tiles; a fused
    /// E4M3 matrix needs every part's slice whole tiles high, so the scale rows line up.
    fn linear(&mut self, parts: &[(String, usize, usize)]) -> Result<DeviceLinear, BackendError> {
        let mut weight = Vec::new();
        let mut scales = Vec::new();
        let (mut weight_len, mut scale_len) = (0, 0);
        let mut format: Option<DType> = None;
        let mut columns: Option<usize> = None;
        let mut rows = 0;
        for (name, part_rows, part_columns) in parts {
            let (dtype, shape, bytes) = self.stored(name, &[*part_rows, *part_columns])?;
            if format.replace(dtype).is_some_and(|first| first != dtype) {
                return Err(refusal(format!(
                    "{name} is {} where the projection it fuses with is not",
                    dtype.name()
                )));
            }
            if columns
                .replace(shape[1])
                .is_some_and(|first| first != shape[1])
            {
                return Err(refusal(format!(
                    "{name} is {} wide where the projection it fuses with is not",
                    shape[1]
                )));
            }
            match dtype {
                DType::BF16 => {}
                DType::F8E4M3 => {
                    if self.config.fp8_block != Some(FP8_BLOCK) {
                        return Err(refusal(format!(
                            "{name} is FP8 with {:?} tiles; the kernels read {FP8_BLOCK}",
                            self.config.fp8_block
                        )));
                    }
                    if parts.len() > 1 && !shape[0].is_multiple_of(FP8_BLOCK) {
                        return Err(refusal(format!(
                            "{name} keeps {} rows here, not whole {FP8_BLOCK}-row tiles, so it \
                             cannot fuse",
                            shape[0]
                        )));
                    }
                    let grid = ScaleGrid::Tile(FP8_BLOCK).shape(*part_rows, *part_columns);
                    let scale_name = format!("{name}_scale_inv");
                    let (scale_type, tile_shape, tiles) = self.stored(&scale_name, &grid)?;
                    if scale_type != DType::F32 {
                        return Err(refusal(format!(
                            "{scale_name} is stored as {}, not F32",
                            scale_type.name()
                        )));
                    }
                    scale_len += scale_type.storage_bytes(elements(&tile_shape));
                    append(&mut scales, tiles);
                }
                other => {
                    return Err(refusal(format!(
                        "{name} is stored as {}, which no GEMV here reads",
                        other.name()
                    )));
                }
            }
            rows += shape[0];
            weight_len += dtype.storage_bytes(elements(&shape));
            append(&mut weight, bytes);
        }
        let (Some(format), Some(columns)) = (format, columns) else {
            return Err(refusal("a projection of no weights"));
        };
        let fp8 = format == DType::F8E4M3;
        let chunk = if fp8 { FP8_BLOCK } else { 8 };
        if !columns.is_multiple_of(chunk) {
            return Err(refusal(format!(
                "{} is {columns} wide here, not whole {chunk}-column chunks",
                parts[0].0
            )));
        }
        let key = parts
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect::<Vec<_>>()
            .join("+");
        let scale_key = format!("{key}_scale_inv");
        let weight = self.sink.weight(&key, weight_len, weight)?;
        let scales = if fp8 {
            Some(self.sink.weight(&scale_key, scale_len, scales)?)
        } else {
            None
        };
        Ok(DeviceLinear {
            weight,
            scales,
            rows,
            columns,
            parts: [rows; 3],
        })
    }

    fn hyper(&mut self, prefix: &str) -> Result<DeviceHyper, BackendError> {
        let geometry = HcGeometry::from(self.config);
        let mixes = geometry.hc_mix_count();
        Ok(DeviceHyper {
            projection: self.linear(&[(format!("{prefix}_fn"), mixes, geometry.hc_width())])?,
            base: self.f32(&format!("{prefix}_base"), &[mixes])?,
            scale: self.f32(&format!("{prefix}_scale"), &[3])?,
        })
    }

    fn kda(&mut self, prefix: &str) -> Result<DeviceKda, BackendError> {
        let config = self.config;
        let (hidden, head, heads) = (
            config.hidden_size,
            config.linear_head_dim,
            config.linear_num_heads,
        );
        let width = config.linear_width();
        let world = self.layout.world_size;
        let mixes = 3 * width / world;
        let fused = self
            .linear(&[
                (format!("{prefix}.q_proj.weight"), width, hidden),
                (format!("{prefix}.k_proj.weight"), width, hidden),
                (format!("{prefix}.v_proj.weight"), width, hidden),
                (format!("{prefix}.f_a_proj.weight"), head, hidden),
                (format!("{prefix}.g_a_proj.weight"), head, hidden),
                (format!("{prefix}.b_proj.weight"), heads, hidden),
            ])?
            .parted([mixes, mixes + head, mixes + 2 * head]);
        let conv: Vec<String> = ["q", "k", "v"]
            .iter()
            .map(|name| format!("{prefix}.{name}_conv1d.weight"))
            .collect();
        Ok(DeviceKda {
            fused,
            conv: self.bf16(&conv, &[width, 1, config.linear_conv_kernel_dim])?,
            f_b: self.linear(&[(format!("{prefix}.f_b_proj.weight"), width, head)])?,
            g_b: self.linear(&[(format!("{prefix}.g_b_proj.weight"), width, head)])?,
            a_log: self.f32(&format!("{prefix}.A_log"), &[heads])?,
            dt_bias: self.f32(&format!("{prefix}.dt_bias"), &[width])?,
            o_norm: self.bf16(&[format!("{prefix}.o_norm.weight")], &[head])?,
            o: self.linear(&[(format!("{prefix}.o_proj.weight"), hidden, width)])?,
            heads: heads / world,
            width: width / world,
        })
    }

    fn indexer(&mut self, prefix: &str) -> Result<DeviceIndexer, BackendError> {
        let config = self.config;
        let (hidden, dim, heads) = (
            config.hidden_size,
            config.index_head_dim,
            config.index_n_heads,
        );
        Ok(DeviceIndexer {
            fused: self
                .linear(&[
                    (format!("{prefix}.wk.weight"), dim, hidden),
                    (format!("{prefix}.index_kpool_compress_gate"), dim, hidden),
                    (format!("{prefix}.weights_proj.weight"), heads, hidden),
                ])?
                .parted([dim, 2 * dim, 2 * dim + heads]),
            wq_b: self.linear(&[(
                format!("{prefix}.wq_b.weight"),
                heads * dim,
                config.q_lora_rank,
            )])?,
            k_norm_weight: self.bf16(&[format!("{prefix}.k_norm.weight")], &[dim])?,
            k_norm_bias: self.bf16(&[format!("{prefix}.k_norm.bias")], &[dim])?,
            ape: self.bf16(
                &[format!("{prefix}.index_kpool_compress_ape")],
                &[config.index_kpool, dim],
            )?,
        })
    }

    fn dsa(&mut self, prefix: &str, own_indexer: bool) -> Result<DeviceDsa, BackendError> {
        let config = self.config;
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let (nope, value) = (config.qk_nope_head_dim, config.v_head_dim);
        let latent_end = config.q_lora_rank + config.kv_lora_rank;
        Ok(DeviceDsa {
            fused: self
                .linear(&[
                    (
                        format!("{prefix}.q_a_proj.weight"),
                        config.q_lora_rank,
                        hidden,
                    ),
                    (
                        format!("{prefix}.kv_a_proj_with_mqa.weight"),
                        config.kv_lora_rank,
                        hidden,
                    ),
                ])?
                .parted([config.q_lora_rank, latent_end, latent_end]),
            q_a_norm: self.bf16(
                &[format!("{prefix}.q_a_layernorm.weight")],
                &[config.q_lora_rank],
            )?,
            kv_a_norm: self.bf16(
                &[format!("{prefix}.kv_a_layernorm.weight")],
                &[config.kv_lora_rank],
            )?,
            q_b: self.linear(&[(
                format!("{prefix}.q_b_proj.weight"),
                heads * nope,
                config.q_lora_rank,
            )])?,
            kv_b: self.linear(&[(
                format!("{prefix}.kv_b_proj.weight"),
                heads * (nope + value),
                config.kv_lora_rank,
            )])?,
            o: self.linear(&[(format!("{prefix}.o_proj.weight"), hidden, heads * value)])?,
            indexer: if own_indexer {
                Some(self.indexer(&format!("{prefix}.indexer"))?)
            } else {
                None
            },
            // Derived from `kv_b` once the rank is built ([DeviceRank::absorb_mla]).
            kv_dequant: Deviceptr(0),
            quantise: false,
            kv_key_t: None,
            heads: heads / self.layout.world_size,
        })
    }

    fn mlp(&mut self, prefix: &str, intermediate: usize) -> Result<DeviceMlp, BackendError> {
        let hidden = self.config.hidden_size;
        Ok(DeviceMlp {
            gate_up: self.linear(&[
                (format!("{prefix}.gate_proj.weight"), intermediate, hidden),
                (format!("{prefix}.up_proj.weight"), intermediate, hidden),
            ])?,
            down: self.linear(&[(format!("{prefix}.down_proj.weight"), hidden, intermediate)])?,
        })
    }

    fn moe(&mut self, prefix: &str) -> Result<DeviceMoe, BackendError> {
        let config = self.config;
        let experts = config.n_routed_experts;
        let intermediate = config.moe_intermediate_size;
        let mut gate_up = Vec::with_capacity(experts * 16);
        let mut down = Vec::with_capacity(experts * 16);
        for expert in 0..experts {
            let mlp = self.mlp(&format!("{prefix}.experts.{expert}"), intermediate)?;
            for (table, linear) in [(&mut gate_up, mlp.gate_up), (&mut down, mlp.down)] {
                let Some(scales) = linear.scales else {
                    return Err(refusal(format!(
                        "{prefix}.experts.{expert} is BF16; the routed expert GEMV reads FP8"
                    )));
                };
                table.extend(linear.weight.0.to_le_bytes());
                table.extend(scales.0.to_le_bytes());
            }
        }
        Ok(DeviceMoe {
            router: self.linear(&[(
                format!("{prefix}.gate.weight"),
                experts,
                config.hidden_size,
            )])?,
            bias: self.f32(
                &format!("{prefix}.gate.e_score_correction_bias"),
                &[experts],
            )?,
            shared: self.mlp(
                &format!("{prefix}.shared_experts"),
                intermediate * config.n_shared_experts,
            )?,
            gate_up: self.sink.derived(&gate_up)?,
            down: self.sink.derived(&down)?,
            intermediate: intermediate / self.layout.world_size,
        })
    }

    fn layer(&mut self, layer: usize) -> Result<DeviceLayer, BackendError> {
        let config = self.config;
        let prefix = format!("{LM_PREFIX}.layers.{layer}");
        let hidden = config.hidden_size;
        Ok(DeviceLayer {
            input_norm: self.bf16(&[format!("{prefix}.input_layernorm.weight")], &[hidden])?,
            post_norm: self.bf16(
                &[format!("{prefix}.post_attention_layernorm.weight")],
                &[hidden],
            )?,
            attn_hc: self.hyper(&format!("{prefix}.hc_attn"))?,
            ffn_hc: self.hyper(&format!("{prefix}.hc_ffn"))?,
            branch: match config.layer_types[layer] {
                LayerKind::Linear => {
                    DeviceBranch::Linear(self.kda(&format!("{prefix}.self_attn"))?)
                }
                LayerKind::Sparse => DeviceBranch::Sparse(self.dsa(
                    &format!("{prefix}.self_attn"),
                    config.indexer_types[layer] == IndexerKind::Full,
                )?),
            },
            ffn: match config.mlp_layer_types[layer] {
                MlpKind::Dense => {
                    DeviceFfn::Dense(self.mlp(&format!("{prefix}.mlp"), config.intermediate_size)?)
                }
                MlpKind::Moe => DeviceFfn::Moe(self.moe(&format!("{prefix}.mlp"))?),
            },
        })
    }
}

/// One rank's activations, sized for [MAX_ROWS] rows, row-major unless a fused GEMV's parts say
/// otherwise.
struct Scratch {
    /// `[hc][hidden]` streams, the `fn` projection and the coefficients it gives.
    streams: Deviceptr,
    raw: Deviceptr,
    coeffs: Deviceptr,
    /// `[rows][splits][mixes]`: the next mixer's `fn` projection as the tiled GEMV's split
    /// partials, when `q4e_mhc_post` works them out ([DeviceRank::fused_projection]).
    mix_parts: Deviceptr,
    /// The normed sublayer input, and the sublayer's partial (then reduced) output.
    collapsed: Deviceptr,
    partial: Deviceptr,
    /// `[world][rows][hidden]`: every peer's partial, when the collective has a device leg.
    slots: Deviceptr,
    /// Each KDA layer's own, in layer order.
    linear: Vec<LinearScratch>,
    forget: Deviceptr,
    gate: Deviceptr,
    kda_out: Deviceptr,
    /// A bulk chunk's scan inputs ([DeviceRank::scan]): `[rows][heads][GDN_QK_ROW]` normed q
    /// and k, padded as the scan stages them, and `[rows][heads]` (unused factor, beta) pairs.
    /// One float each in a scratch no bulk chunk runs on.
    kda_qk: Deviceptr,
    kda_beta: Deviceptr,
    dsa_fused: Deviceptr,
    query: Deviceptr,
    /// `[rows][heads][kv_lora]`: the query projected into the latent space ([DeviceRank::dsa_attend]).
    q_abs: Deviceptr,
    /// A decode step's attention ([DeviceRank::dsa_attend]): `[rows][heads][MAX_SELECTED_TOKENS]`
    /// logits, then softmax weights, `[rows][heads]` softmax sums and `[rows][heads][kv_lora]`
    /// weighted latents, for at most [MAX_ROWS] rows.
    mla_logits: Deviceptr,
    mla_total: Deviceptr,
    mla_mixed: Deviceptr,
    attn_out: Deviceptr,
    index_fused: Deviceptr,
    index_query: Deviceptr,
    scores: Deviceptr,
    selected: Deviceptr,
    count: Deviceptr,
    gate_up: Deviceptr,
    activated: Deviceptr,
    shared_out: Deviceptr,
    router_logits: Deviceptr,
    expert_indices: Deviceptr,
    expert_weights: Deviceptr,
    expert_gate_up: Deviceptr,
    expert_activated: Deviceptr,
    expert_out: Deviceptr,
    head: Deviceptr,
    head_normed: Deviceptr,
    /// The MTP layer's rows: the tokens' embeddings, `[embedding | hidden]` after their norms,
    /// the residual (one stream, so `mtp_coeffs` mixes it as `x + reduced`), and its output after
    /// the head norm.
    mtp_embed: Deviceptr,
    mtp_in: Deviceptr,
    mtp_x: Deviceptr,
    mtp_out: Deviceptr,
    mtp_coeffs: Deviceptr,
    /// `[world][rows][vocab / world]`: each rank's slice of the rows' logits, this rank's alone
    /// unless the collective has a device leg.
    logits: Deviceptr,
    /// Where the logits land on the host: one copy into page-locked memory, each row's slices
    /// read from it.
    logits_host: HostMapped,
    /// `[MAX_ROWS][MAX_ROWS]` tokens of a draft, `[iteration][row]`: iteration 0 the tokens the
    /// drafts follow, iteration `i + 1` the argmax of iteration `i`'s logits, which the device
    /// writes and the next iteration's embedding reads with no host round trip.
    drafted: HostMapped,
    /// `[MAX_ROWS]` tokens: a greedy step's picks, which the device writes ([DeviceRank::picks]).
    picks: HostMapped,
    /// The tiled GEMVs' split sums and per-tile counters, which each launch leaves zeroed.
    partials: Deviceptr,
    counters: Deviceptr,
    /// The KDA recurrence's per-head counters, which each launch leaves zeroed.
    recur_counters: Deviceptr,
    /// The bulk prefill's routed-expert plan and its padded operands: `slot_of` a slot (or -1) a
    /// padded row, `items` the (expert, .., count, ..) list, and the gate/up and activated rows,
    /// [MOE_ITEM_ROWS] an item.
    moe_slot_of: Deviceptr,
    moe_items: Deviceptr,
    moe_item_count: Deviceptr,
    moe_gate_up: Deviceptr,
    moe_activated: Deviceptr,
    /// [PrefillActivation::Fp8]'s quantised operands: a projection's input as E4M3 codes and one
    /// f32 scale a row's 128 columns (up to the streams' width a row), and the routed experts'
    /// activated rows the same way.
    q8_codes: Deviceptr,
    q8_scales: Deviceptr,
    moe_act_codes: Deviceptr,
    moe_act_scales: Deviceptr,
}

/// A KDA layer's activations, its own so they last until the next step: what a speculative
/// verify's commit replays ([DeviceRank::commit]).
struct LinearScratch {
    /// The fused projection's outputs: the convolution's input, f_a, g_a and b.
    fused: Deviceptr,
    /// The convolved `[q | k | v]` rows and the per-channel decay, the recurrence's inputs.
    mixed: Deviceptr,
    decay: Deviceptr,
    /// The convolution windows after each of a step's rows but its last, row `r`'s in region `r`.
    windows: Deviceptr,
}

/// One sequence's state on one rank: each KDA layer's convolution history and recurrent memory,
/// each DSA layer's latent and indexer caches, `capacity` positions long.
pub struct DeviceSequence {
    position: usize,
    capacity: usize,
    /// The last verify's rows, until [DeviceRank::commit] keeps some of them.
    pending: Option<Pending>,
    /// Where its last kept row's logits sit after a greedy step ([DeviceRank::last_logits]).
    logits_at: Option<LogitsAt>,
    layers: Vec<SequenceLayer>,
    /// The MTP layer's state, for a sequence made to draft.
    drafter: Option<SequenceDrafter>,
    /// The block drafter's, for a sequence made to block-draft: the target rows its steps capture
    /// and the context its commits fill.
    block: Option<DraftScratch>,
    /// The block drafter's confidence in each of the sequence's last drafts.
    confidence: Vec<f32>,
    memory: Allocations,
}

/// What a drafting sequence keeps for the MTP layer: its caches, whose position `q` pairs the
/// model's hidden at `q` with the token at `q + 1`, and the model's normed hidden at each row of
/// the sequence's last step, which the next step's first token pairs with.
struct SequenceDrafter {
    layer: SequenceLayer,
    /// `[MAX_ROWS][hidden]`: the last step's last [MAX_ROWS] rows ([DeviceRank::fill]), all a
    /// later step reads (the row at the last kept position, and a verify keeps at most its rows).
    hidden: Deviceptr,
    /// The row of `hidden` at the last position the sequence keeps; `None` before its first step.
    last: Option<usize>,
}

/// A speculative verify waiting for its commit: the rank and the step that ran it, and the rows
/// of that step that were this sequence's.
#[derive(Debug, Clone, Copy)]
struct Pending {
    rank: u64,
    step: u64,
    first: usize,
    rows: usize,
    step_rows: usize,
}

/// Which rows of a step go through the LM head: all of them, or only the last (a prefill chunk
/// needs one row's logits, not one a token).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadRows {
    All,
    Last,
}

/// What a step hands back for each of its rows.
trait StepRow: Sized {
    /// Whether the rows' logits stay on the device for [DeviceRank::last_logits].
    const ON_DEVICE: bool;
    /// The rows of `input` through the LM head.
    fn head(
        rank: &DeviceRank,
        collectives: (&TpRuntime, &BitGatherHandle),
        input: Deviceptr,
        rows: usize,
        device_leg: bool,
    ) -> Result<Vec<Self>, BackendError>;
}

/// A row's full-vocabulary logits, on the host.
impl StepRow for Vec<f32> {
    const ON_DEVICE: bool = false;
    fn head(
        rank: &DeviceRank,
        (tp, gather): (&TpRuntime, &BitGatherHandle),
        input: Deviceptr,
        rows: usize,
        device_leg: bool,
    ) -> Result<Vec<Self>, BackendError> {
        rank.logits(tp, gather, input, rows, (device_leg, false))
    }
}

/// A row's greedy token ([DeviceRank::picks]).
impl StepRow for u32 {
    const ON_DEVICE: bool = true;
    fn head(
        rank: &DeviceRank,
        collectives: (&TpRuntime, &BitGatherHandle),
        input: Deviceptr,
        rows: usize,
        device_leg: bool,
    ) -> Result<Vec<Self>, BackendError> {
        rank.picks(collectives, input, rows, device_leg)
    }
}

/// A row of a greedy step's logits, which stay in `logits` until the rank's next LM head: the
/// rank, the head that wrote them ([DeviceRank::heads]), the row and the step's rows.
#[derive(Debug, Clone, Copy)]
struct LogitsAt {
    rank: u64,
    head: u64,
    row: usize,
    rows: usize,
}

enum SequenceLayer {
    Linear {
        conv: Deviceptr,
        recurrent: Deviceptr,
        /// Where a verify leaves its last row's recurrent memory, the memory it read staying in
        /// `recurrent` until the commit; the two swap when it does.
        spare: Deviceptr,
    },
    Sparse {
        /// The normed latent: `[capacity][kv_lora]` E4M3 codes when `latent_scales` is set, f32
        /// otherwise. Attention reads it in latent space ([DeviceRank::dsa_attend]), so the cache
        /// is one row a position rather than `heads` expanded key and value slices.
        latent: Deviceptr,
        /// `[capacity][kv_lora / 128]` f32 scales of the E4M3 cache; zero for the f32 one.
        latent_scales: Deviceptr,
        /// The indexer's per-position keys and gates and its pooled keys; unused by a layer
        /// that shares the previous layer's selection.
        index_keys: Deviceptr,
        index_gates: Deviceptr,
        pooled: Deviceptr,
    },
}

/// One sequence's rows of a step: its next `len` tokens from `position`, rows
/// `first..first + len` of the step. A verify's run leaves the state its rows read for the commit;
/// a drafting sequence's run also fills the MTP layer's caches.
struct Run<'s> {
    layers: &'s [SequenceLayer],
    position: usize,
    first: usize,
    len: usize,
    verify: bool,
    drafter: Option<&'s SequenceDrafter>,
    block: Option<&'s DraftScratch>,
}

impl DeviceSequence {
    /// Positions this sequence has consumed, a verify's rows too until its commit gives back the
    /// ones it does not keep.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// The positions this sequence was made for.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The block drafter's confidence in each of the sequence's last drafts, in `(0, 1]`: the
    /// selector's probability for its pick among the row's candidates. Empty after an MTP draft,
    /// which keeps none.
    #[must_use]
    pub fn draft_confidence(&self) -> &[f32] {
        &self.confidence
    }
}

/// A rank's block drafter and where the ranks exchange their candidates for a draft's rows:
/// `[world][rows][values, ids][block - 1][top_k]`, this rank's slot from its slice of the head.
struct BlockDrafter {
    draft: GpuDflash2,
    exchange: Deviceptr,
    /// The exchange on the TCP leg: this rank's slot copied out, then every rank's, which the
    /// merge reads in place.
    exchange_host: HostMapped,
    /// `[MAX_ROWS][2 * block]`: each row's chain and its confidences, which the device writes.
    chains: HostMapped,
}

/// One tensor-parallel rank of GLM-5.3-Flash on one GPU, for decode up to [MAX_ROWS] rows a step.
pub struct DeviceRank {
    cuda: Arc<CudaHandle>,
    stream: RunStream,
    kernels: Kernels,
    config: Glm5NextConfig,
    layout: TpLayout,
    geometry: HcGeometry,
    embed: Deviceptr,
    norm: Deviceptr,
    lm_head: DeviceLinear,
    /// An E4M3 copy of this rank's slice of a BF16 LM head, which drafts score with: the MTP
    /// layer's and the block drafter's ([Self::draft_head]). Half the bytes a draft reads, and only
    /// drafts read it, which a verify checks against the BF16 head.
    draft_head: Option<GpuMatE4m3>,
    layers: Vec<DeviceLayer>,
    mtp: Option<DeviceMtp>,
    /// The DFlash 2 block drafter, once [DeviceRank::load_block_drafter] loads one.
    block: Option<BlockDrafter>,
    scratch: Scratch,
    /// The bulk prefill's scratch, sized for [DeviceRank::prefill_rows] rows. Swapped in around a
    /// prefill step so the decode scratch's size does not grow with the chunk.
    prefill: Option<Scratch>,
    prefill_rows: usize,
    /// Whether the router and the indexer's projections keep the token GEMV's bits in bulk
    /// prefill, so expert choice and selected tokens are the decode path's.
    exact_prefill: bool,
    /// How the bulk prefill's GEMMs take their f32 activations.
    prefill_activation: PrefillActivation,
    /// The fewest rows a device-leg all-reduce takes in two shots ([Self::reduce_into]).
    two_shot_rows: usize,
    /// Steps this rank has run: a verify's commit has to come before the next one, which
    /// overwrites the inputs it replays.
    steps: u64,
    /// LM heads this rank has run, a step's or a draft's: each overwrites the last one's logits.
    heads: AtomicU64,
    /// The side stream [Self::prefetch] warms the L2 from, unless SPARKFLASH_L2_PREFETCH=0.
    prefetch: Option<Prefetch>,
    memory: Allocations,
}

/// A side stream that pulls the weights the next kernels read into the L2 while the rank's stream
/// waits in an exchange or runs latency-bound kernels, and the bytes it asks for a wait.
struct Prefetch {
    side: SideStream,
    budget: usize,
}

// SAFETY: a side stream is a queue on the rank's own context, and only the rank's launches
// record waits on it. A `Send` rank moves the stream with it; nothing shares it.
unsafe impl Send for Prefetch {}

impl Prefetch {
    /// The bytes a wait asks for unless SPARKFLASH_L2_PREFETCH_MB says otherwise: about what DRAM
    /// moves while an exchange, its mix-back and the next mixer's projection run.
    const BUDGET_MB: usize = 8;

    /// The prefetch SPARKFLASH_L2_PREFETCH and SPARKFLASH_L2_PREFETCH_MB select: none under
    /// SPARKFLASH_L2_PREFETCH=0 or a budget of zero.
    ///
    /// # Errors
    /// A budget that is not a whole number of MiB, or the driver's failure to make the stream.
    fn new(cuda: &Cuda) -> Result<Option<Self>, BackendError> {
        if std::env::var("SPARKFLASH_L2_PREFETCH").as_deref() == Ok("0") {
            return Ok(None);
        }
        let mb = match std::env::var("SPARKFLASH_L2_PREFETCH_MB") {
            Ok(value) => value.trim().parse::<usize>().map_err(|_| {
                refusal(format!(
                    "SPARKFLASH_L2_PREFETCH_MB={value} is not a whole number of MiB"
                ))
            })?,
            Err(_) => Self::BUDGET_MB,
        };
        if mb == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            side: cuda.side_stream().map_err(cuda_error)?,
            budget: mb << 20,
        }))
    }
}

impl DeviceRank {
    /// Upload rank `layout.rank`'s slice of the text model.
    ///
    /// # Errors
    /// When this build embeds no cubin, the configuration does not split over the world or
    /// outgrows a kernel's bounds, a tensor is missing, shaped or stored other than the plan
    /// says, or the device refuses an allocation or a copy.
    pub fn load<S: WeightSource + ?Sized>(
        cuda: Arc<CudaHandle>,
        config: &Glm5NextConfig,
        source: &S,
        layout: TpLayout,
    ) -> Result<Self, BackendError> {
        Self::load_with_cache(cuda, config, source, layout, LatentCache::Auto)
    }

    /// [Self::load] with an explicit latent-cache representation.
    ///
    /// # Errors
    /// As [Self::load].
    pub fn load_with_cache<S: WeightSource + ?Sized>(
        cuda: Arc<CudaHandle>,
        config: &Glm5NextConfig,
        source: &S,
        layout: TpLayout,
        cache: LatentCache,
    ) -> Result<Self, BackendError> {
        Self::build(cuda, config, layout, cache, |stream, memory| {
            let mut sink = Direct { stream, memory };
            Uploader::new(source, config, layout, &mut sink).weights()
        })
    }

    /// Reads rank `layout.rank`'s converted file at `path` ([crate::convert::convert_rank]) into
    /// one device allocation in parallel direct-I/O chunks, and places the weights in it in the
    /// order [DeviceRank::load] uploads them, so the rank is the same bits without the checkpoint.
    ///
    /// # Errors
    /// What [DeviceRank::load] refuses, and a file that is not a GLM rank file of this rank and
    /// world, fails its index check, or holds other weights than this configuration loads.
    pub fn load_converted(
        cuda: Arc<CudaHandle>,
        config: &Glm5NextConfig,
        path: &Path,
        layout: TpLayout,
    ) -> Result<Self, BackendError> {
        Self::load_converted_with_cache(cuda, config, path, layout, LatentCache::Auto)
    }

    /// [Self::load_converted] with an explicit latent-cache representation.
    ///
    /// # Errors
    /// As [Self::load_converted].
    pub fn load_converted_with_cache(
        cuda: Arc<CudaHandle>,
        config: &Glm5NextConfig,
        path: &Path,
        layout: TpLayout,
        cache: LatentCache,
    ) -> Result<Self, BackendError> {
        let (preamble, index) = crate::convert::read_index(path, layout).map_err(refusal)?;
        let specs = ConvertedSpecs::new(&index).map_err(refusal)?;
        let data = usize::try_from(preamble.padded_data_bytes)
            .map_err(|_| refusal("the rank file's data does not fit this machine's range"))?;
        Self::build(cuda, config, layout, cache, |stream, memory| {
            let base = memory.file(path, preamble.data_offset, data)?;
            let mut sink = Slab {
                base,
                slots: &index.slots,
                next: 0,
                stream,
                memory,
            };
            let weights = Uploader::new(&specs, config, layout, &mut sink).weights()?;
            if sink.next != index.slots.len() {
                return Err(refusal(format!(
                    "the rank file holds {} weights, and this configuration loads {}",
                    index.slots.len(),
                    sink.next
                )));
            }
            Ok(weights)
        })
    }

    /// The rank around the weights `upload` places, once the configuration is one the kernels
    /// take.
    fn build(
        cuda: Arc<CudaHandle>,
        config: &Glm5NextConfig,
        layout: TpLayout,
        cache: LatentCache,
        upload: impl FnOnce(&RunStream, &mut Allocations) -> Result<RankWeights, BackendError>,
    ) -> Result<Self, BackendError> {
        if !cubin_available() {
            return Err(refusal(
                "this build embeds no Qwen4-Exp cubin, so GLM has no device kernels",
            ));
        }
        Self::check_rank(config, layout)?;
        let stream = cuda.run_stream().map_err(cuda_error)?;
        let kernels = Kernels::load(&cuda, config.linear_head_dim).map_err(cuda_error)?;
        let mut memory = Allocations::new(Arc::clone(&cuda));
        let RankWeights {
            embed,
            norm,
            lm_head,
            mut layers,
            mut mtp,
        } = upload(&stream, &mut memory)?;
        // The absorbed MLA operands: each sparse layer's kv_b, dequantised once. The latent
        // cache's representation is the caller's ([LatentCache]).
        let (nope, value) = (config.qk_nope_head_dim, config.v_head_dim);
        let quantise = |dsa: &DeviceDsa| {
            (match cache {
                LatentCache::Auto => config.dynamic_activations && dsa.kv_b.scales.is_some(),
                LatentCache::F32 => false,
                LatentCache::Fp8 => true,
            }) && config.kv_lora_rank.is_multiple_of(FP8_BLOCK)
        };
        for layer in &mut layers {
            if let DeviceBranch::Sparse(dsa) = &mut layer.branch {
                dsa.quantise = quantise(dsa);
                (dsa.kv_dequant, dsa.kv_key_t) =
                    absorb_kv_b(&cuda, &stream, &mut memory, &dsa.kv_b, nope, value)?;
            }
        }
        if let Some(mtp) = &mut mtp {
            mtp.dsa.quantise = quantise(&mtp.dsa);
            (mtp.dsa.kv_dequant, mtp.dsa.kv_key_t) =
                absorb_kv_b(&cuda, &stream, &mut memory, &mtp.dsa.kv_b, nope, value)?;
        }
        // A chunk's scratch is about 9.6 GiB at 4,096 rows, and there is no point paying it for a
        // configuration the chunk cannot take (the f32 latent cache, or a shape the bulk MLA does
        // not serve): SPARKFLASH_GLM_KV_CACHE=f32 at the 1M window loads as it did before the bulk
        // prefill, and its prompts run the token path.
        let chunk_shapes = config.kv_lora_rank == GMA_LATENT
            && config.v_head_dim.is_multiple_of(128)
            && layers.iter().all(|layer| match &layer.branch {
                DeviceBranch::Sparse(dsa) => dsa.quantise && dsa.heads == GMA_HEADS,
                DeviceBranch::Linear(_) => true,
            });
        let prefill_rows = Self::prefill_rows_from_env()?.filter(|_| chunk_shapes);
        let before = memory.bytes;
        let prefill = prefill_rows
            .map(|rows| Self::scratch(config, layout, rows, &stream, &mut memory, true))
            .transpose()?;
        if let Some(rows) = prefill_rows.filter(|_| prefill.is_some()) {
            // The chunk scratch beside the weights and the latent window, which is what decides
            // the widest chunk a node takes: printed so a load that just fits says how close it is.
            eprintln!(
                "GLM bulk prefill: {rows} rows a chunk, {:.1} GiB of chunk scratch",
                (memory.bytes - before) as f64 / (1u64 << 30) as f64
            );
        }
        let scratch = Self::scratch(config, layout, MAX_ROWS, &stream, &mut memory, false)?;
        let prefetch = Prefetch::new(&cuda)?;
        let mut rank = Self {
            cuda,
            stream,
            kernels,
            config: config.clone(),
            layout,
            geometry: HcGeometry::from(config),
            embed,
            norm,
            lm_head,
            draft_head: None,
            layers,
            mtp,
            block: None,
            scratch,
            prefill,
            prefill_rows: prefill_rows.unwrap_or(MAX_ROWS),
            exact_prefill: std::env::var("SPARKFLASH_GLM_PREFILL_EXACT").as_deref() != Ok("0"),
            prefill_activation: PrefillActivation::from_env(config)?,
            two_shot_rows: sparkflash_weights::tp::two_shot_rows().map_err(weights_error)?,
            steps: 0,
            heads: AtomicU64::new(0),
            prefetch,
            memory,
        };
        if rank.mtp.is_some() && rank.lm_head.scales.is_none() {
            rank.draft_head()?;
        }
        Ok(rank)
    }

    /// The E4M3 copy of this rank's slice of the LM head that drafts score with, quantized on
    /// first use and kept with the rank's weights: a rank with an MTP layer makes it as it loads.
    ///
    /// # Errors
    /// When the head is not BF16, or the device refuses an allocation or the quantization.
    fn draft_head(&mut self) -> Result<GpuMatE4m3, BackendError> {
        if let Some(head) = self.draft_head {
            return Ok(head);
        }
        if self.lm_head.scales.is_some() {
            return Err(refusal("drafts score with an E4M3 copy of a BF16 LM head"));
        }
        // The quantization runs on the default stream, which this rank's uploads are not ordered
        // against.
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let matrix = GpuMat {
            pointer: self.lm_head.weight,
            rows: self.lm_head.rows,
            columns: self.lm_head.columns,
        };
        let memory = &mut self.memory;
        let head = GpuMatE4m3::quantize(
            &self.cuda,
            self.kernels.quantize_e4m3,
            &matrix,
            &mut memory.bytes,
        )
        .map_err(cuda_error)?;
        memory.buffers.extend([head.weight, head.scales]);
        self.draft_head = Some(head);
        Ok(head)
    }

    /// Refuses a layout or a configuration this arm cannot run as rank `layout.rank`.
    pub(crate) fn check_rank(
        config: &Glm5NextConfig,
        layout: TpLayout,
    ) -> Result<(), BackendError> {
        layout.validate().map_err(weights_error)?;
        config
            .check_world(layout.world_size)
            .map_err(|error| refusal(error.to_string()))?;
        Self::check_bounds(config)
    }

    /// Bytes this rank holds on the device, weights and scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.memory.bytes
    }

    /// This rank's place in the tensor-parallel world.
    #[must_use]
    pub fn layout(&self) -> TpLayout {
        self.layout
    }

    /// The configuration this rank was built for.
    #[must_use]
    pub fn config(&self) -> &Glm5NextConfig {
        &self.config
    }

    /// Refuses a configuration a kernel's fixed bounds would truncate.
    fn check_bounds(config: &Glm5NextConfig) -> Result<(), BackendError> {
        let selected = config.index_pools() * config.index_kpool + config.index_kpool - 1;
        for (what, value, bound) in [
            ("hc_mult", config.hc_mult, MAX_HC),
            ("n_routed_experts", config.n_routed_experts, MAX_EXPERTS),
            ("num_experts_per_tok", config.num_experts_per_tok, MAX_TOPK),
            (
                "a step's expert picks",
                MAX_ROWS * config.num_experts_per_tok,
                BLOCK as usize,
            ),
            ("index_kpool", config.index_kpool, MAX_POOL),
            ("index_n_heads", config.index_n_heads, MAX_INDEX_HEADS),
            ("index_head_dim", config.index_head_dim, MAX_INDEX_DIM),
            (
                "the indexer's pool budget",
                config.index_pools(),
                MAX_BUDGET_BLOCKS,
            ),
            ("the tokens a query selects", selected, MAX_SELECTED_TOKENS),
            ("kv_lora_rank", config.kv_lora_rank, MLA_MAX_LATENT),
            (
                "qk_nope_head_dim",
                config.qk_nope_head_dim,
                MAX_ATTENTION_HEAD_DIM,
            ),
        ] {
            if value > bound {
                return Err(refusal(format!(
                    "{what} = {value} is past the device kernels' bound of {bound}"
                )));
            }
        }
        let block = BLOCK as usize;
        if config.hidden_size == 0
            || !config.hidden_size.is_multiple_of(block)
            || config.hidden_size / block > MHC_COLUMNS
        {
            return Err(refusal(format!(
                "hidden_size = {} is not a multiple of {block} of at most {MHC_COLUMNS}, the \
                 columns of every stream a thread of the mixer holds",
                config.hidden_size
            )));
        }
        if config.qk_nope_head_dim != config.v_head_dim {
            return Err(refusal(format!(
                "the MLA's key heads are {} wide and its value heads {}; the cache split reads \
                 one width",
                config.qk_nope_head_dim, config.v_head_dim
            )));
        }
        if config.num_experts_per_tok > config.n_routed_experts {
            return Err(refusal("more experts a token than the model has"));
        }
        // A layer that shares an indexer reads the selection the sparse layer before it wrote.
        for layer in 0..config.num_hidden_layers {
            let shared = config.layer_types[layer] == LayerKind::Sparse
                && config.indexer_types[layer] != IndexerKind::Full;
            if shared && (layer == 0 || config.layer_types[layer - 1] != LayerKind::Sparse) {
                return Err(refusal(format!(
                    "layer {layer} shares an indexer, and no sparse layer comes before it"
                )));
            }
        }
        Ok(())
    }

    /// The bulk prefill chunk `SPARKFLASH_GLM_PREFILL_CHUNK` asks for, or the default; `None` when
    /// `SPARKFLASH_GLM_PREFILL=0` keeps the token path.
    fn prefill_rows_from_env() -> Result<Option<usize>, BackendError> {
        if std::env::var("SPARKFLASH_GLM_PREFILL").as_deref() == Ok("0") {
            return Ok(None);
        }
        let rows = match std::env::var("SPARKFLASH_GLM_PREFILL_CHUNK") {
            Ok(value) => value.trim().parse::<usize>().map_err(|_| {
                refusal(format!(
                    "SPARKFLASH_GLM_PREFILL_CHUNK={value} is not a whole number of rows"
                ))
            })?,
            Err(_) => DEFAULT_PREFILL_ROWS,
        };
        if rows < MIN_PREFILL_ROWS {
            return Ok(None);
        }
        Ok(Some(rows.min(MAX_PREFILL_ROWS)))
    }

    /// Floats in one KDA layer's convolution windows on a rank of `width` channels a mix.
    fn window_floats(config: &Glm5NextConfig, width: usize) -> usize {
        config.linear_conv_kernel_dim.saturating_sub(1) * 3 * width
    }

    /// One rank's scratch for `rows` rows. `chunk` sizes it for a bulk prefill chunk: the
    /// per-layer buffers of a kind are one allocation shared by every layer (a layer's launches
    /// are ordered on the rank's stream, so no two layers hold one buffer at once), and the KDA
    /// convolution windows are a placeholder, because a chunk never verifies and only a verify
    /// keeps windows for a commit to restore. At 4096 rows that is 8.4 GiB instead of 55 GiB of
    /// per-layer buffers, which is what lets a chunk wider than 2048 rows fit beside the weights.
    fn scratch(
        config: &Glm5NextConfig,
        layout: TpLayout,
        rows: usize,
        stream: &RunStream,
        memory: &mut Allocations,
        chunk: bool,
    ) -> Result<Scratch, BackendError> {
        let world = layout.world_size;
        let hidden = config.hidden_size;
        let head = config.linear_head_dim;
        let width = config.linear_width() / world;
        let linear_heads = config.linear_num_heads / world;
        let heads = config.num_attention_heads / world;
        let (nope, value) = (config.qk_nope_head_dim, config.v_head_dim);
        let (index_dim, index_heads) = (config.index_head_dim, config.index_n_heads);
        let topk = config.num_experts_per_tok;
        let expert = config.moe_intermediate_size / world;
        let intermediate = (config.intermediate_size / world)
            .max(config.moe_intermediate_size * config.n_shared_experts / world);
        let mixes = HcGeometry::from(config).hc_mix_count();
        let moe_items_max = moe_items(rows * topk, config.n_routed_experts);
        let moe_rows = moe_items_max * MOE_ITEM_ROWS;
        // A chunk's grouped GEMVs hold a launch's groups' partials at once.
        let groups = rows.div_ceil(MAX_ROWS).min(GEMV_GROUPS);
        Ok(Scratch {
            streams: memory.floats(rows * config.hc_mult * hidden)?,
            raw: memory.floats(rows * mixes)?,
            coeffs: memory.floats(rows * mixes)?,
            mix_parts: memory
                .floats(rows * (config.hc_mult * hidden).div_ceil(TILED_GEMV_ROUND) * mixes)?,
            collapsed: memory.floats(rows * hidden)?,
            partial: memory.floats(rows * hidden)?,
            slots: memory.floats(world * rows * hidden)?,
            linear: {
                let mut one = || {
                    Ok(LinearScratch {
                        fused: memory.floats(rows * (3 * width + 2 * head + linear_heads))?,
                        mixed: memory.floats(rows * 3 * width)?,
                        decay: memory.floats(rows * width)?,
                        windows: if chunk {
                            memory.floats(1)?
                        } else {
                            memory.floats((rows - 1) * Self::window_floats(config, width))?
                        },
                    })
                };
                if chunk {
                    let shared = one()?;
                    (0..config.linear_layers())
                        .map(|_| Ok(LinearScratch { ..shared }))
                        .collect::<Result<Vec<_>, BackendError>>()?
                } else {
                    (0..config.linear_layers())
                        .map(|_| one())
                        .collect::<Result<Vec<_>, BackendError>>()?
                }
            },
            forget: memory.floats(rows * width)?,
            gate: memory.floats(rows * width)?,
            kda_out: memory.floats(rows * width)?,
            kda_qk: memory.floats(if rows > MAX_ROWS {
                rows * linear_heads * GDN_QK_ROW
            } else {
                1
            })?,
            kda_beta: memory.floats(if rows > MAX_ROWS {
                rows * linear_heads * 2
            } else {
                1
            })?,
            dsa_fused: memory.floats(rows * (config.q_lora_rank + config.kv_lora_rank))?,
            query: memory.floats(rows * heads * nope)?,
            q_abs: memory.floats(rows * heads * config.kv_lora_rank)?,
            mla_logits: memory.floats(rows.min(MAX_ROWS) * heads * MAX_SELECTED_TOKENS)?,
            mla_total: memory.floats(rows.min(MAX_ROWS) * heads)?,
            mla_mixed: memory.floats(rows.min(MAX_ROWS) * heads * config.kv_lora_rank)?,
            attn_out: memory.floats(rows * heads * value)?,
            index_fused: memory.floats(rows * (2 * index_dim + index_heads))?,
            index_query: memory.floats(rows * index_heads * index_dim)?,
            scores: memory.floats(rows * (config.max_position_embeddings / config.index_kpool))?,
            selected: memory.floats(rows * MAX_SELECTED_TOKENS)?,
            count: memory.floats(rows)?,
            gate_up: memory.floats(rows * 2 * intermediate)?,
            activated: memory.floats(rows * intermediate)?,
            shared_out: memory.floats(rows * hidden)?,
            router_logits: memory.floats(rows * config.n_routed_experts)?,
            expert_indices: memory.floats(rows * topk)?,
            expert_weights: memory.floats(rows * topk)?,
            expert_gate_up: memory.floats(rows * topk * 2 * expert)?,
            expert_activated: memory.floats(rows * topk * expert)?,
            expert_out: memory.floats(rows * topk * hidden)?,
            head: memory.floats(rows * hidden)?,
            head_normed: memory.floats(rows * hidden)?,
            mtp_embed: memory.floats(rows * hidden)?,
            mtp_in: memory.floats(rows * 2 * hidden)?,
            mtp_x: memory.floats(rows * hidden)?,
            mtp_out: memory.floats(rows * hidden)?,
            // q4e_mhc_post's [pre | post | comb] for one stream: 1.0 * reduced + 1.0 * x.
            mtp_coeffs: memory.upload(stream, &to_bytes(&[0.0, 1.0, 1.0].repeat(rows)))?,
            logits: memory.floats(world * rows * (config.vocab_size / world))?,
            logits_host: memory.mapped(world * rows * (config.vocab_size / world) * 4)?,
            drafted: memory.mapped(rows * rows * 4)?,
            picks: memory.mapped(rows * 4)?,
            partials: memory.floats(TILED_GEMV_PARTIALS * groups)?,
            counters: memory.zeroed(stream, TILED_GEMV_BLOCKS * groups * 4)?,
            recur_counters: memory.zeroed(stream, linear_heads * 4)?,
            moe_slot_of: memory.zeroed(stream, moe_rows * 4)?,
            moe_items: memory.zeroed(stream, moe_items_max * 16)?,
            moe_item_count: memory.zeroed(stream, 4)?,
            moe_gate_up: memory.floats(moe_rows * 2 * expert)?,
            moe_activated: memory.floats(moe_rows * expert)?,
            q8_codes: memory.bytes(rows * config.hc_mult * hidden)?,
            q8_scales: memory.floats(rows * (config.hc_mult * hidden).div_ceil(FP8_BLOCK))?,
            moe_act_codes: memory.bytes(moe_rows * expert)?,
            moe_act_scales: memory.floats(moe_rows * expert.div_ceil(FP8_BLOCK))?,
        })
    }

    /// A fresh sequence of up to `capacity` positions.
    ///
    /// # Errors
    /// When `capacity` is past the context window or the device refuses an allocation.
    pub fn new_sequence(&self, capacity: usize) -> Result<DeviceSequence, BackendError> {
        self.sequence(capacity, false)
    }

    /// Puts `sequence` back at position 0 with no history, the state [Self::new_sequence] opens
    /// one in, keeping its window: a server that holds one window between requests skips the
    /// allocation, 18.2 GiB a rank at the 1M default. As in a fresh window, only the convolution
    /// history and the recurrent memory are cleared; the caches are read below the position only.
    ///
    /// # Errors
    /// A block-drafting sequence, whose drafter context this does not clear, or the device refuses
    /// a copy.
    pub fn reset(&self, sequence: &mut DeviceSequence) -> Result<(), BackendError> {
        if sequence.block.is_some() {
            return Err(refusal(
                "a block-drafting sequence is not reset in place; open a new one",
            ));
        }
        self.clear(sequence)
    }

    /// Zeroes `sequence`'s convolution history and recurrent memory and rewinds it to position 0.
    fn clear(&self, sequence: &mut DeviceSequence) -> Result<(), BackendError> {
        let head = self.config.linear_head_dim;
        let history = self.config.linear_conv_kernel_dim.saturating_sub(1);
        let mut zeros = Vec::new();
        for (layer, state) in self.layers.iter().zip(&sequence.layers) {
            let (
                DeviceBranch::Linear(kda),
                SequenceLayer::Linear {
                    conv, recurrent, ..
                },
            ) = (&layer.branch, state)
            else {
                continue;
            };
            let cleared = [
                (conv, history * 3 * kda.width * 4),
                (recurrent, kda.heads * head * head * 4),
            ];
            for (pointer, bytes) in cleared.into_iter().filter(|&(_, bytes)| bytes > 0) {
                zeros.resize(zeros.len().max(bytes), 0u8);
                self.cuda
                    .copy_htod_on(&self.stream, pointer, &zeros[..bytes])
                    .map_err(cuda_error)?;
            }
        }
        sequence.position = 0;
        sequence.pending = None;
        sequence.logits_at = None;
        sequence.confidence.clear();
        if let Some(drafter) = sequence.drafter.as_mut() {
            drafter.last = None;
        }
        Ok(())
    }

    /// A fresh sequence of up to `capacity` positions that keeps the MTP layer's state, for
    /// [Self::draft]: each of its steps also fills the MTP layer's caches.
    ///
    /// # Errors
    /// As [Self::new_sequence], and when this rank has no MTP layer.
    pub fn new_drafting_sequence(&self, capacity: usize) -> Result<DeviceSequence, BackendError> {
        if self.mtp.is_none() {
            return Err(refusal("this rank has no MTP layer to draft with"));
        }
        self.sequence(capacity, true)
    }

    /// Load the DFlash 2 block drafter in `directory`, for [Self::new_block_drafting_sequence]:
    /// this rank's share of its heads and MLP width, sharded over the world as the target is
    /// ([GpuDflash2::load]), scoring its blocks with the E4M3 copy of this rank's slice of the LM
    /// head the MTP layer's drafts score with too ([Self::draft_head]).
    ///
    /// # Errors
    /// When the rank has a block drafter already or a head other than BF16, the drafter cannot be
    /// read, is not a DFlash 2 checkpoint this arm runs or was trained for another target, or the
    /// device refuses an allocation or a copy.
    pub fn load_block_drafter(&mut self, directory: &Path) -> Result<(), BackendError> {
        if self.block.is_some() {
            return Err(refusal("this rank has a block drafter already"));
        }
        let head = self.draft_head()?;
        let shard = (self.layout.rank, self.layout.world_size);
        let draft = GpuDflash2::load(&self.cuda, directory, &self.config, &head, shard).map_err(
            |error| {
                BackendError::new(
                    "glm5_dflash2",
                    format!("block drafter {}: {error}", directory.display()),
                )
            },
        )?;
        let config = draft.config();
        let exchange = self.layout.world_size * MAX_ROWS * Self::candidates(&draft);
        let chains = MAX_ROWS * 2 * config.block_size * 4;
        let mut buffers = || -> Result<_, BackendError> {
            Ok((
                self.memory.floats(exchange)?,
                self.memory.mapped(exchange * 4)?,
                self.memory.mapped(chains)?,
            ))
        };
        match buffers() {
            Ok((exchange, exchange_host, chains)) => {
                self.block = Some(BlockDrafter {
                    draft,
                    exchange,
                    exchange_host,
                    chains,
                });
                Ok(())
            }
            Err(error) => {
                draft.free(&self.cuda);
                Err(error)
            }
        }
    }

    /// The floats of a draft row's candidates: values and ids for each mask row's top k.
    fn candidates(draft: &GpuDflash2) -> usize {
        let config = draft.config();
        2 * (config.block_size - 1) * config.selector_top_k
    }

    /// A fresh sequence of up to `capacity` positions that keeps the block drafter's state, for
    /// [Self::draft]: each of its steps also captures the drafter's target rows and commits the
    /// positions it keeps to the drafter's context.
    ///
    /// # Errors
    /// As [Self::new_sequence], and when this rank has no block drafter.
    pub fn new_block_drafting_sequence(
        &self,
        capacity: usize,
    ) -> Result<DeviceSequence, BackendError> {
        let Some(block) = &self.block else {
            return Err(refusal(
                "this rank has no block drafter; load_block_drafter loads one",
            ));
        };
        let mut sequence = self.sequence(capacity, false)?;
        let memory = &mut sequence.memory;
        let scratch = DraftScratch::allocate(&block.draft, MAX_ROWS, |floats| {
            memory.floats(floats).map_err(|error| error.to_string())
        })
        .map_err(cuda_error)?;
        sequence.block = Some(scratch);
        Ok(sequence)
    }

    fn sequence(&self, capacity: usize, drafting: bool) -> Result<DeviceSequence, BackendError> {
        let config = &self.config;
        if capacity > config.max_position_embeddings {
            return Err(refusal(format!(
                "a sequence of {capacity} positions is past the context window {}",
                config.max_position_embeddings
            )));
        }
        let mut memory = Allocations::new(Arc::clone(&self.cuda));
        let head = config.linear_head_dim;
        let history = config.linear_conv_kernel_dim.saturating_sub(1);
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(match &layer.branch {
                    DeviceBranch::Linear(kda) => SequenceLayer::Linear {
                        conv: memory.bytes(history * 3 * kda.width * 4)?,
                        recurrent: memory.bytes(kda.heads * head * head * 4)?,
                        spare: memory.floats(kda.heads * head * head)?,
                    },
                    DeviceBranch::Sparse(dsa) => self.sparse_state(&mut memory, dsa, capacity)?,
                })
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let drafter = match &self.mtp {
            Some(mtp) if drafting => Some(SequenceDrafter {
                layer: self.sparse_state(&mut memory, &mtp.dsa, capacity)?,
                hidden: memory.floats(MAX_ROWS * config.hidden_size)?,
                last: None,
            }),
            _ => None,
        };
        let mut sequence = DeviceSequence {
            position: 0,
            capacity,
            pending: None,
            logits_at: None,
            layers,
            drafter,
            block: None,
            confidence: Vec::new(),
            memory,
        };
        self.clear(&mut sequence)?;
        Ok(sequence)
    }

    /// A DSA layer's caches for `capacity` positions, the indexer's only for a layer with one.
    fn sparse_state(
        &self,
        memory: &mut Allocations,
        dsa: &DeviceDsa,
        capacity: usize,
    ) -> Result<SequenceLayer, BackendError> {
        let config = &self.config;
        let index_dim = config.index_head_dim;
        // The latent is the release's E4M3 activation when the projections quantise theirs, and
        // f32 for a BF16 release.
        let width = config.kv_lora_rank;
        let (latent, latent_scales) = if dsa.quantise {
            (
                memory.bytes(capacity * width)?,
                memory.floats(capacity * width.div_ceil(FP8_BLOCK))?,
            )
        } else {
            (memory.floats(capacity * width)?, Deviceptr(0))
        };
        let mut index = |floats: usize| {
            if dsa.indexer.is_some() {
                memory.floats(floats)
            } else {
                Ok(Deviceptr(0))
            }
        };
        Ok(SequenceLayer::Sparse {
            index_keys: index(capacity * index_dim)?,
            index_gates: index(capacity * index_dim)?,
            pooled: index(capacity / config.index_kpool * index_dim)?,
            latent,
            latent_scales,
        })
    }

    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        threads: u32,
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch_on(&self.stream, kernel, grid, (threads, 1, 1), args)
            .map_err(|error| cuda_error(format!("{}: {error}", kernel.name())))
    }

    /// `output = W input` for `rows` rows, BF16 or block-scaled FP8 by the weight, each of a fused
    /// projection's parts as one block of every row's outputs. A prefill-sized batch takes the
    /// bulk tensor-core GEMM; a decode step keeps the tiled GEMV, whose bits a verified row is
    /// defined by.
    fn gemv(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows > MAX_ROWS {
            return self.bulk_gemm(linear, input, output, rows);
        }
        self.gemv_small(linear, input, output, rows)
    }

    /// The bulk prefill's projection: one tensor-core GEMM over all `rows` of each part, an FP8
    /// part's over the input's E4M3 codes ([PrefillActivation::Fp8], quantised once for all
    /// parts). A BF16 part of at most 32 rows the 32-column tile cannot cover (a 24-row mixer, a
    /// 16-row gate) takes the skinny kernel, and any other such part the tiled GEMV.
    fn bulk_gemm(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if !input.0.is_multiple_of(16) {
            return Err(refusal(
                "a bulk GEMM's input is not 16-byte aligned; its reads would be torn",
            ));
        }
        let columns = linear.columns;
        let element = if linear.scales.is_some() { 1 } else { 2 };
        let starts = [
            0,
            linear.parts[0],
            linear.parts[1],
            linear.parts[2],
            linear.rows,
        ];
        let mut quantised = false;
        for part in 0..4 {
            let (first, end) = (starts[part], starts[part + 1]);
            if end <= first {
                continue;
            }
            let part_rows = end - first;
            let weight = linear.weight.offset(first * columns * element);
            let scales = linear.scales.map(|scales| {
                scales.offset(
                    (first / FP8_BLOCK) * (columns / FP8_BLOCK) * std::mem::size_of::<f32>(),
                )
            });
            let out = output.offset(rows * first * 4);
            // The block-scaled kernel indexes its scales a 128-column K block at a time, so a
            // scaled part also needs the K run to be whole blocks, not just whole 32-column
            // stages; the skinny kernel reads eight floats a lane a 256-column step.
            let tiled = part_rows % 32 == 0
                && columns.is_multiple_of(32)
                && scales.is_none_or(|_| columns.is_multiple_of(FP8_BLOCK))
                && scales.is_none_or(|_| first.is_multiple_of(FP8_BLOCK));
            if !tiled && scales.is_none() && part_rows <= 32 && columns.is_multiple_of(256) {
                self.launch(
                    self.kernels.prefill_skinny[part_rows.div_ceil(8) - 1],
                    (rows.div_ceil(16) as u32, 1, 1),
                    BLOCK,
                    &[
                        weight.0,
                        input.0,
                        out.0,
                        part_rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?;
                continue;
            }
            if !tiled {
                let row = DeviceLinear {
                    weight,
                    scales,
                    rows: part_rows,
                    columns,
                    parts: [part_rows; 3],
                };
                self.gemv_grouped(&row, input, out, rows)?;
                continue;
            }
            let grid = (part_rows.div_ceil(128) as u32, rows.div_ceil(128) as u32, 1);
            let (fp8, bf16) = if self.prefill_activation == PrefillActivation::Bf16 {
                (
                    self.kernels.prefill_dense_fp8_bf16a,
                    self.kernels.prefill_dense_bf16_bf16a,
                )
            } else {
                (
                    self.kernels.prefill_dense_fp8,
                    self.kernels.prefill_dense_bf16,
                )
            };
            match scales {
                Some(scales)
                    if self.prefill_activation == PrefillActivation::Fp8
                        && columns.is_multiple_of(FP8_BLOCK) =>
                {
                    q8_inner(columns)?;
                    if !quantised {
                        self.quantise(input, rows, columns)?;
                        quantised = true;
                    }
                    let s = &self.scratch;
                    self.launch(
                        self.kernels.prefill_dense_q8,
                        grid,
                        BLOCK,
                        &[
                            weight.0,
                            scales.0,
                            s.q8_codes.0,
                            s.q8_scales.0,
                            out.0,
                            part_rows as u64,
                            columns as u64,
                            rows as u64,
                        ],
                    )?
                }
                Some(scales) => self.launch(
                    fp8,
                    grid,
                    BLOCK,
                    &[
                        weight.0,
                        scales.0,
                        input.0,
                        out.0,
                        part_rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?,
                None => self.launch(
                    bf16,
                    grid,
                    BLOCK,
                    &[
                        weight.0,
                        input.0,
                        out.0,
                        part_rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?,
            }
        }
        Ok(())
    }

    /// `rows` rows of `columns` f32 at `input` as the release's dynamic activation, into the
    /// scratch's E4M3 codes and per-128-column scales.
    fn quantise(&self, input: Deviceptr, rows: usize, columns: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let widest = self.config.hc_mult * self.config.hidden_size;
        if !columns.is_multiple_of(FP8_BLOCK) || columns > widest {
            return Err(refusal(format!(
                "the prefill quantises whole 128-column blocks of rows of at most {widest} \
                 columns, not {columns}"
            )));
        }
        // One warp a 128-column block.
        let warps = rows * columns / FP8_BLOCK;
        self.launch(
            self.kernels.prefill_quantise,
            ((warps * 32).div_ceil(BLOCK as usize) as u32, 1, 1),
            BLOCK,
            &[
                input.0,
                s.q8_codes.0,
                s.q8_scales.0,
                rows as u64,
                columns as u64,
            ],
        )
    }

    /// A projection for a bulk chunk, with the token GEMV when `exact` keeps a routing or
    /// selection projection's bits. The token GEMV runs eight rows a group; its per-row chain
    /// does not depend on the batch or the group, so this is the decode path's arithmetic.
    fn projection(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
        exact: bool,
    ) -> Result<(), BackendError> {
        if exact && rows > MAX_ROWS {
            return self.gemv_chunked(linear, input, output, rows);
        }
        self.gemv(linear, input, output, rows)
    }

    /// The token GEMV over a batch larger than its eight-row tile: up to [GEMV_GROUP_ROWS] rows a
    /// launch, each part on its own past that.
    fn gemv_chunked(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows <= GEMV_GROUP_ROWS {
            return self.gemv_small(linear, input, output, rows);
        }
        let columns = linear.columns;
        let element = if linear.scales.is_some() { 1 } else { 2 };
        let starts = [
            0,
            linear.parts[0],
            linear.parts[1],
            linear.parts[2],
            linear.rows,
        ];
        for part in 0..4 {
            let (first, end) = (starts[part], starts[part + 1]);
            if end <= first {
                continue;
            }
            let part_rows = end - first;
            let row = DeviceLinear {
                weight: linear.weight.offset(first * columns * element),
                scales: linear.scales.map(|scales| {
                    scales.offset(
                        (first / FP8_BLOCK) * (columns / FP8_BLOCK) * std::mem::size_of::<f32>(),
                    )
                }),
                rows: part_rows,
                columns,
                parts: [part_rows; 3],
            };
            self.gemv_grouped(&row, input, output.offset(rows * first * 4), rows)?;
        }
        Ok(())
    }

    /// A one-part `linear` over `rows` rows, [GEMV_GROUP_ROWS] a grouped launch.
    fn gemv_grouped(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let mut done = 0;
        while done < rows {
            let count = GEMV_GROUP_ROWS.min(rows - done);
            self.gemv_small(
                linear,
                input.offset(done * linear.columns * 4),
                output.offset(done * linear.rows * 4),
                count,
            )?;
            done += count;
        }
        Ok(())
    }

    /// The tiled GEMV: the decode path and the fallback for a part the bulk tile cannot cover. Past
    /// [MAX_ROWS] rows (up to [GEMV_GROUP_ROWS]) it launches a group of eight rows a z-block, each
    /// the eight-row launch's arithmetic.
    fn gemv_small(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if !input.0.is_multiple_of(16) {
            return Err(refusal(
                "a tiled GEMV's input is not 16-byte aligned; its reads would be torn",
            ));
        }
        if rows > GEMV_GROUP_ROWS {
            return Err(refusal(format!(
                "a tiled GEMV takes at most {GEMV_GROUP_ROWS} rows a launch, not {rows}"
            )));
        }
        let tiles = linear.rows.div_ceil(TILE_ROWS);
        let scratch = &self.scratch;
        let (kernel, splits, mut args) = match linear.scales {
            None => (
                self.kernels.gemv_bf16,
                tiled_gemv_splits(tiles, linear.columns),
                vec![linear.weight.0],
            ),
            Some(scales) => (
                self.kernels.gemv_fp8,
                tiled_gemv_e4m3_splits(tiles, linear.columns),
                vec![linear.weight.0, scales.0],
            ),
        };
        args.extend([
            input.0,
            output.0,
            linear.rows as u64,
            linear.columns as u64,
            rows as u64,
            scratch.partials.0,
            scratch.counters.0,
        ]);
        if linear.scales.is_some() {
            args.push(u64::from(self.config.dynamic_activations));
        }
        args.extend(linear.parts.map(|start| start as u64));
        let groups = rows.div_ceil(MAX_ROWS) as u32;
        self.launch(kernel, (tiles as u32, splits as u32, groups), BLOCK, &args)
    }

    /// The routed experts' FP8 GEMV over the router's picks for `rows` rows: `outputs` a pick, a
    /// pick reading its own input row (`per_slot`) or its row's. The picks of one expert run as
    /// one batch, so a step reads each picked expert once.
    #[allow(clippy::too_many_arguments)]
    fn expert_gemv(
        &self,
        table: Deviceptr,
        weights: Deviceptr,
        input: Deviceptr,
        output: Deviceptr,
        outputs: usize,
        columns: usize,
        per_slot: bool,
        rows: usize,
    ) -> Result<(), BackendError> {
        let topk = self.config.num_experts_per_tok;
        self.launch(
            self.kernels.expert_fp8,
            (outputs.div_ceil(TILE_ROWS) as u32, 1, (rows * topk) as u32),
            BLOCK,
            &[
                table.0,
                self.scratch.expert_indices.0,
                weights.0,
                input.0,
                output.0,
                outputs as u64,
                columns as u64,
                topk as u64,
                u64::from(per_slot),
                u64::from(self.config.dynamic_activations),
            ],
        )
    }

    /// The first [Prefetch::budget] bytes of `reads`, in order, toward the L2 from the side stream
    /// once the work queued on the rank's stream so far is done: queued before an exchange or a
    /// run of latency-bound kernels, the weights the next GEMVs read come in while DRAM would
    /// idle. A hint, so it changes no value; nothing without a [Prefetch].
    fn prefetch(&self, reads: &[(Deviceptr, usize)]) -> Result<(), BackendError> {
        let Some(prefetch) = &self.prefetch else {
            return Ok(());
        };
        let mut left = prefetch.budget;
        let spans: Vec<(u64, u64)> = reads
            .iter()
            .map(|&(at, bytes)| {
                let bytes = bytes.min(left);
                left -= bytes;
                (at.0, bytes as u64)
            })
            .filter(|&(_, bytes)| bytes > 0)
            .collect();
        for four in spans.chunks(4) {
            let mut args = [0u64; 8];
            for (slot, &(address, bytes)) in four.iter().enumerate() {
                args[2 * slot] = address;
                args[2 * slot + 1] = bytes;
            }
            self.cuda
                .side_launch_on(
                    &self.stream,
                    &prefetch.side,
                    self.kernels.prefetch_l2,
                    (1, 1, 1),
                    (32, 1, 1),
                    &args,
                )
                .map_err(cuda_error)?;
        }
        Ok(())
    }

    /// The mixer's coefficients from each row's streams, and its normed sublayer input into
    /// `collapsed`: two blocks a row, the input's beside the coefficients'. With `fused` the
    /// `q4e_mhc_post` before left `hyper`'s projection of the streams in `mix_parts`, and its GEMV
    /// does not run.
    fn mix_in(
        &self,
        hyper: &DeviceHyper,
        norm: Deviceptr,
        rows: usize,
        fused: bool,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let geometry = self.geometry;
        let (hc, hidden) = (geometry.hc_mult, geometry.dim);
        if !fused {
            self.gemv(&hyper.projection, s.streams, s.raw, rows)?;
        }
        self.launch(
            self.kernels.mhc_in,
            (rows as u32, 2, 1),
            BLOCK,
            &[
                s.raw.0,
                s.streams.0,
                hyper.scale.0,
                hyper.base.0,
                s.coeffs.0,
                norm.0,
                s.collapsed.0,
                hidden as u64,
                hc as u64,
                geometry.hc_sinkhorn_iters as u64,
                float(geometry.hc_eps),
                float(geometry.norm_eps),
                float(self.config.rms_norm_eps as f32),
                if fused { s.mix_parts.0 } else { 0 },
            ],
        )
    }

    /// `hyper`'s `fn` projection when `q4e_mhc_post` can work it out of the streams it writes: a
    /// one-part BF16 matrix of at most [MHC_FUSED_PAIRS] (copy, row) pairs over a hidden of whole
    /// blocks, which the tiled GEMV splits into one 256-column round a split. Its partials are
    /// then that GEMV's to the bit. Only with a [Prefetch], which has the matrix in the L2 by the
    /// exchange's end: an exchange does not launch its dependents early, so cold, the fused
    /// kernel's few blocks read it on the critical path, slower than the GEMV it replaces.
    fn fused_projection<'a>(&self, hyper: &'a DeviceHyper) -> Option<&'a DeviceLinear> {
        let linear = &hyper.projection;
        let (hc, hidden) = (self.geometry.hc_mult, self.geometry.dim);
        let tiles = linear.rows.div_ceil(TILE_ROWS);
        (self.prefetch.is_some()
            && linear.scales.is_none()
            && linear.parts == [linear.rows; 3]
            && linear.columns == hc * hidden
            && BLOCK as usize == TILED_GEMV_ROUND
            && hidden.is_multiple_of(TILED_GEMV_ROUND)
            && tiled_gemv_splits(tiles, linear.columns) == linear.columns / TILED_GEMV_ROUND
            && hc * linear.rows <= MHC_FUSED_PAIRS)
            .then_some(linear)
    }

    /// All-reduces the rows' partials in one collective and mixes them back into `hc` streams a
    /// row by each row's `coeffs`: the model's hyper-connections, or the MTP layer's one stream
    /// and `mtp_coeffs`. On the device leg the peers' partials land in `slots` and `q4e_mhc_post`
    /// sums them with this rank's in rank order, as [Wire::F32] does on the host; otherwise they
    /// cross the host. With `next`, the mixer that reads the streams next, `q4e_mhc_post` also
    /// works out its `fn` projection where it can ([Self::fused_projection]); returns whether it
    /// did.
    fn mix_out(
        &self,
        tp: &TpRuntime,
        residual: (Deviceptr, Deviceptr, usize),
        next: Option<&DeviceHyper>,
        rows: usize,
        device_leg: bool,
    ) -> Result<bool, BackendError> {
        // The fused mixer writes one tiled GEMV's split partials; a prefill chunk takes the bulk
        // GEMM instead, whose 128-row tiles the mixer's 24-row projection does not feed.
        let next = if rows > MAX_ROWS {
            None
        } else {
            next.and_then(|hyper| self.fused_projection(hyper))
        };
        self.reduce_into(tp, self.scratch.partial, residual, next, rows, device_leg)?;
        Ok(next.is_some())
    }

    /// Whether a device-leg all-reduce of `rows` rows takes two shots: at least
    /// [Self::two_shot_rows] rows (0 for never) of whole float4 chunks a rank.
    fn two_shot(&self, rows: usize) -> bool {
        let world = self.layout.world_size;
        self.two_shot_rows > 0
            && rows >= self.two_shot_rows
            && splits_two_shot(rows * self.geometry.dim, world)
    }

    /// [Self::mix_out] of the partials at `partial`, which the two-shot and host legs overwrite
    /// with their sum.
    fn reduce_into(
        &self,
        tp: &TpRuntime,
        partial: Deviceptr,
        (streams, coeffs, hc): (Deviceptr, Deviceptr, usize),
        next: Option<&DeviceLinear>,
        rows: usize,
        device_leg: bool,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let hidden = self.geometry.dim;
        let world = self.layout.world_size;
        let world = if device_leg && self.two_shot(rows) {
            // Two shots, half the one-shot's bytes. Each element is q4e_mhc_post's fold, so the
            // bits are the one-shot's.
            reduce_two_shot(
                tp,
                self.stream.raw(),
                partial,
                s.slots,
                rows * hidden,
                |mine, count| {
                    self.launch(
                        self.kernels.sum_slots,
                        (blocks(count / 4), 1, 1),
                        BLOCK,
                        &[
                            mine.0,
                            s.slots.0,
                            count as u64,
                            self.layout.rank as u64,
                            world as u64,
                        ],
                    )
                },
            )?;
            1
        } else if device_leg {
            // `partial` holds the rows' partials and `slots` a world of them, and every rank runs
            // the same steps.
            tp.exchange_on_device(self.stream.raw(), partial.0, s.slots.0, rows * hidden * 4)
                .map_err(weights_error)?;
            world
        } else {
            let mut bytes = vec![0u8; rows * hidden * 4];
            self.cuda
                .copy_dtoh_on(&self.stream, &mut bytes, &partial)
                .map_err(cuda_error)?;
            let mut values = to_f32(&bytes);
            tp.all_reduce(&mut values).map_err(weights_error)?;
            self.cuda
                .copy_htod_on(&self.stream, &partial, &to_bytes(&values))
                .map_err(cuda_error)?;
            1
        };
        self.launch(
            self.kernels.mhc_post,
            (blocks(hidden), rows as u32, 1),
            BLOCK,
            &[
                streams.0,
                partial.0,
                coeffs.0,
                hidden as u64,
                hc as u64,
                s.slots.0,
                self.layout.rank as u64,
                world as u64,
                next.map_or(0, |linear| linear.weight.0),
                if next.is_some() { s.mix_parts.0 } else { 0 },
                next.map_or(0, |linear| linear.rows as u64),
            ],
        )
    }

    /// The rows through KDA layer `layer`'s slice, their partial outputs into `partial`: the
    /// projections over every row, the convolution and recurrence over each sequence's rows in
    /// order on its own state.
    fn kda(
        &self,
        kda: &DeviceKda,
        layer: usize,
        own: &LinearScratch,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (head, width, heads) = (config.linear_head_dim, kda.width, kda.heads);
        let mixes = 3 * width;
        let window = Self::window_floats(config, width);
        let floats = |count: usize| count * 4;
        self.gemv(&kda.fused, s.collapsed, own.fused, rows)?;
        // The convolution, the decay's and gate's small GEMVs and the recurrence leave DRAM
        // idle; the output projection comes in meanwhile.
        self.prefetch(&[kda.o.span()])?;
        let f_a = kda.fused.part(own.fused, 1, rows);
        let g_a = kda.fused.part(own.fused, 2, rows);
        let beta = kda.fused.part(own.fused, 3, rows);
        for run in runs {
            let SequenceLayer::Linear { conv, .. } = &run.layers[layer] else {
                return Err(other_kind());
            };
            let (raw, mixed) = (
                own.fused.offset(floats(run.first * mixes)),
                own.mixed.offset(floats(run.first * mixes)),
            );
            if self.bulk(run) {
                self.launch(
                    self.kernels.conv_rows,
                    (blocks(mixes), run.len as u32, 1),
                    BLOCK,
                    &[
                        raw.0,
                        conv.0,
                        kda.conv.0,
                        mixed.0,
                        mixes as u64,
                        config.linear_conv_kernel_dim as u64,
                    ],
                )?;
                self.launch(
                    self.kernels.conv_history,
                    (blocks(mixes), 1, 1),
                    BLOCK,
                    &[
                        raw.0,
                        conv.0,
                        mixes as u64,
                        config.linear_conv_kernel_dim as u64,
                        run.len as u64,
                        // A chunk never verifies, so it keeps no windows.
                        0,
                        0,
                        0,
                    ],
                )?;
                continue;
            }
            // A verify keeps the windows after each of its rows but the last, for a commit that
            // keeps fewer rows to restore.
            let (windows, kept) = if run.verify {
                (
                    own.windows.offset(floats(run.first * window)).0,
                    run.len - 1,
                )
            } else {
                (0, 0)
            };
            self.launch(
                self.kernels.conv,
                (blocks(mixes), 1, 1),
                BLOCK,
                &[
                    raw.0,
                    conv.0,
                    kda.conv.0,
                    mixed.0,
                    mixes as u64,
                    config.linear_conv_kernel_dim as u64,
                    run.len as u64,
                    windows,
                    kept as u64,
                    window as u64,
                ],
            )?;
        }
        self.gemv(&kda.f_b, f_a, s.forget, rows)?;
        self.gemv(&kda.g_b, g_a, s.gate, rows)?;
        self.launch(
            self.kernels.kda_decay,
            (blocks(rows * width), 1, 1),
            BLOCK,
            &[
                s.forget.0,
                kda.a_log.0,
                kda.dt_bias.0,
                own.decay.0,
                heads as u64,
                head as u64,
                float(config.linear_lower_bound as f32),
                rows as u64,
            ],
        )?;
        for run in runs {
            let SequenceLayer::Linear {
                recurrent, spare, ..
            } = &run.layers[layer]
            else {
                return Err(other_kind());
            };
            let inputs = [
                own.mixed.offset(floats(run.first * mixes)),
                s.gate.offset(floats(run.first * width)),
                beta.offset(floats(run.first * heads)),
                own.decay.offset(floats(run.first * width)),
            ];
            let out = s.kda_out.offset(floats(run.first * width));
            if self.bulk(run) {
                self.scan(kda, inputs, *recurrent, out, run.len)?;
                continue;
            }
            // A verify leaves the memory it read in place and its last row's in the spare.
            let into = if run.verify { spare } else { recurrent };
            self.recur(kda, inputs, (*recurrent, *into), Some(out), run.len)?;
        }
        self.gemv(&kda.o, s.kda_out, s.partial, rows)
    }

    /// Whether a run's KDA takes the bulk chunk's convolution and scan ([Self::scan]): more rows
    /// than a step's tile, never a verify (whose windows and memory a commit restores), at the
    /// head and history widths the prefill image is compiled for.
    fn bulk(&self, run: &Run<'_>) -> bool {
        let config = &self.config;
        !run.verify
            && run.len > MAX_ROWS
            && config.linear_head_dim == GDN_KEY
            && config.linear_head_dim == GDN_VALUE
            && (2..=CONV_HISTORY + 1).contains(&config.linear_conv_kernel_dim)
    }

    /// A bulk chunk's KDA recurrence over `len` rows of one sequence, Qwen's prefill scan with
    /// the state in registers ([Self::recur]'s inputs, the memory stepped in place): a 4096-row
    /// chunk's is ~16 ms a layer through the token kernel, whose steps are 128-term chains on a
    /// quarter of its threads. The arithmetic is the token kernel's in another association, so
    /// a chunk matches it to f32 rounding, as the bulk GEMMs match the GEMVs.
    fn scan(
        &self,
        kda: &DeviceKda,
        [mixed, gate, beta, decay]: [Deviceptr; 4],
        state: Deviceptr,
        out: Deviceptr,
        len: usize,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let s = &self.scratch;
        let heads = kda.heads;
        let (key_width, value) = (kda.width as u64, GDN_VALUE as u64);
        // One value head a key head.
        let groups = 1;
        self.launch(
            self.kernels.kda_prep,
            (heads as u32, len as u32, 1),
            GDN_KEY as u32,
            &[
                mixed.0,
                beta.0,
                s.kda_qk.0,
                s.kda_beta.0,
                key_width,
                heads as u64,
                heads as u64,
                value,
                groups,
            ],
        )?;
        let (scan_grid, scan_threads) = gdn_scan_shape(heads);
        self.launch(
            self.kernels.kda_scan,
            scan_grid,
            scan_threads,
            &[
                s.kda_qk.0,
                s.kda_beta.0,
                decay.0,
                mixed.0,
                state.0,
                out.0,
                key_width,
                heads as u64,
                heads as u64,
                value,
                groups,
                len as u64,
                // No per-row snapshots: a chunk never verifies.
                0,
                0,
                0,
            ],
        )?;
        self.launch(
            self.kernels.kda_gate,
            (heads as u32, len as u32, 1),
            GDN_VALUE as u32,
            &[
                out.0,
                gate.0,
                kda.o_norm.0,
                out.0,
                heads as u64,
                value,
                float(config.rms_norm_eps as f32),
                // The sigmoid output gate.
                1,
            ],
        )
    }

    /// KDA's recurrence over `len` rows of one sequence: `inputs` are the convolved rows, the
    /// output gate, b and the per-channel decay, `state` the memory it reads and where it leaves
    /// the memory after the last row. Without an output it only steps the memory, as a commit's
    /// replay does; the memory is the same bits either way.
    fn recur(
        &self,
        kda: &DeviceKda,
        [mixed, gate, beta, decay]: [Deviceptr; 4],
        (from, into): (Deviceptr, Deviceptr),
        out: Option<Deviceptr>,
        len: usize,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let head = config.linear_head_dim;
        self.launch(
            self.kernels.recur,
            (kda.heads as u32, recur_parts(head) as u32, 1),
            RECUR_THREADS,
            &[
                mixed.0,
                gate.0,
                beta.0,
                // The per-head decay's inputs, which the channel form does not read.
                0,
                0,
                0,
                kda.o_norm.0,
                from.0,
                into.0,
                out.map_or(0, |out| out.0),
                kda.width as u64,
                kda.heads as u64,
                head as u64,
                head as u64,
                // One value head a key head.
                1,
                float(config.rms_norm_eps as f32),
                // The sigmoid output gate.
                1,
                len as u64,
                u64::from(self.kernels.recur.shared_bytes()),
                decay.0,
                self.scratch.recur_counters.0,
            ],
        )
    }

    /// The rows through DSA layer `layer`'s slice, their partial outputs into `partial`. A layer
    /// with its own indexer writes each row's selection; one without reads the previous one's.
    fn dsa(
        &self,
        dsa: &DeviceDsa,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        self.dsa_cache(dsa, layer, runs, rows)?;
        self.dsa_attend(dsa, layer, runs, rows)
    }

    /// The half of a DSA sublayer that writes each row's position of the caches: the normed
    /// latent, and with an indexer its index key and gate and any pool the row completes. It
    /// leaves the normed query latent in `dsa_fused` for [Self::dsa_attend].
    fn dsa_cache(
        &self,
        dsa: &DeviceDsa,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let eps = float(config.rms_norm_eps as f32);
        let (q_lora, kv_lora) = (config.q_lora_rank, config.kv_lora_rank);
        let floats = |count: usize| count * 4;
        self.gemv(&dsa.fused, s.collapsed, s.dsa_fused, rows)?;
        let latent = dsa.fused.part(s.dsa_fused, 1, rows);
        for (part, norm, width) in [
            (s.dsa_fused, dsa.q_a_norm, q_lora),
            (latent, dsa.kv_a_norm, kv_lora),
        ] {
            self.launch(
                self.kernels.rmsnorm,
                (rows as u32, 1, 1),
                BLOCK,
                &[
                    part.0,
                    norm.0,
                    part.0,
                    width as u64,
                    width as u64,
                    eps,
                    0,
                    0,
                ],
            )?;
        }
        // The latent, not its expansion: attention reads it in latent space, and kv_b's rows are
        // the absorbed operands it reads instead ([Self::dsa_attend]).
        for run in runs {
            let SequenceLayer::Sparse {
                latent: cache,
                latent_scales: scales,
                ..
            } = &run.layers[layer]
            else {
                return Err(other_kind());
            };
            let source = latent.offset(floats(run.first * kv_lora)).0;
            if dsa.quantise {
                self.launch(
                    self.kernels.mla_cache_fp8,
                    (run.len as u32, 1, 1),
                    BLOCK,
                    &[
                        source,
                        cache.0,
                        scales.0,
                        run.position as u64,
                        run.len as u64,
                        kv_lora as u64,
                    ],
                )?;
            } else {
                self.launch(
                    self.kernels.mla_cache,
                    (run.len as u32, 1, 1),
                    BLOCK,
                    &[
                        source,
                        cache.0,
                        run.position as u64,
                        run.len as u64,
                        kv_lora as u64,
                    ],
                )?;
            }
        }
        match &dsa.indexer {
            Some(indexer) => self.index_cache(indexer, layer, runs, rows),
            None => Ok(()),
        }
    }

    /// The half of a DSA sublayer that attends: each row's query, its selection, and the attention
    /// over the positions it selects, the partial outputs into `partial`. Every position a row
    /// reads is in the caches by then ([Self::dsa_cache]).
    fn dsa_attend(
        &self,
        dsa: &DeviceDsa,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (heads, nope, value) = (dsa.heads, config.qk_nope_head_dim, config.v_head_dim);
        let latent = config.kv_lora_rank;
        self.gemv(&dsa.q_b, s.dsa_fused, s.query, rows)?;
        // The indexer's pick and the attention leave DRAM idle; the output projection comes in
        // meanwhile.
        self.prefetch(&[dsa.o.span()])?;
        if let Some(indexer) = &dsa.indexer {
            self.index_pick(indexer, layer, runs, rows)?;
        }
        if rows > MAX_ROWS && dsa.quantise {
            self.bulk_attend(dsa, layer, runs, rows)?;
            return self.gemv(&dsa.o, s.attn_out, s.partial, rows);
        }
        if heads > MLA_HEADS {
            return Err(refusal(format!(
                "the decode MLA stages at most {MLA_HEADS} heads a rank, not {heads}"
            )));
        }
        // The absorbed form: the query projected into the latent space, then the attention over
        // the latent cache in four launches (qwen4_exp_sm121.cu's q4e_glm_mla_scores) and kv_b's
        // value rows.
        for run in runs {
            let SequenceLayer::Sparse {
                latent: cache,
                latent_scales: scales,
                ..
            } = &run.layers[layer]
            else {
                return Err(other_kind());
            };
            let (first, len) = (run.first as u64, run.len as u32);
            self.launch(
                self.kernels.mla_absorb,
                (heads as u32, len, latent.div_ceil(BLOCK as usize) as u32),
                BLOCK,
                &[
                    s.query.0,
                    dsa.kv_dequant.0,
                    s.q_abs.0,
                    first,
                    nope as u64,
                    value as u64,
                    latent as u64,
                ],
            )?;
            let cache = [cache.0, cache.0, scales.0];
            let selection = [s.selected.0, s.count.0];
            let quantise = u64::from(dsa.quantise);
            self.launch(
                self.kernels.mla_scores,
                ((MAX_SELECTED_TOKENS / MLA_SCORE_SLOTS) as u32, len, 1),
                BLOCK,
                &[
                    &[s.q_abs.0][..],
                    &cache,
                    &selection,
                    &[
                        s.mla_logits.0,
                        first,
                        heads as u64,
                        nope as u64,
                        latent as u64,
                        MAX_SELECTED_TOKENS as u64,
                        quantise,
                    ],
                ]
                .concat(),
            )?;
            self.launch(
                self.kernels.mla_softmax,
                (heads as u32, len, 1),
                BLOCK,
                &[s.mla_logits.0, s.count.0, s.mla_total.0, first],
            )?;
            self.launch(
                self.kernels.mla_mix,
                (
                    latent.div_ceil(32) as u32,
                    heads.div_ceil(MLA_MIX_HEADS) as u32,
                    len,
                ),
                (32 * MLA_MIX_HEADS) as u32,
                &[
                    &cache[..],
                    &selection,
                    &[
                        s.mla_logits.0,
                        s.mla_total.0,
                        s.mla_mixed.0,
                        first,
                        heads as u64,
                        latent as u64,
                        MAX_SELECTED_TOKENS as u64,
                        quantise,
                    ],
                ]
                .concat(),
            )?;
            self.launch(
                self.kernels.mla_project,
                (
                    heads as u32,
                    (value as u32).div_ceil(MLA_PROJECT_THREADS),
                    1,
                ),
                MLA_PROJECT_THREADS,
                &[
                    s.mla_mixed.0,
                    dsa.kv_dequant.0,
                    s.count.0,
                    s.attn_out.0,
                    first,
                    u64::from(len),
                    nope as u64,
                    value as u64,
                    latent as u64,
                ],
            )?;
        }
        self.gemv(&dsa.o, s.attn_out, s.partial, rows)
    }

    /// A bulk chunk's attention ([Self::dsa_attend]) on the tensor cores: the query absorbed
    /// into the latent space by a per-head GEMM over kv_b's transposed key rows (the decode
    /// kernel for an FP8 kv_b), each row's 16 heads over its selected E4M3 latents at once
    /// (glm_prefill_mla_attend, the weighted latent over the query), and the value projection
    /// by a per-head GEMM over kv_b's value rows, into `attn_out`.
    fn bulk_attend(
        &self,
        dsa: &DeviceDsa,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (heads, nope, value) = (dsa.heads, config.qk_nope_head_dim, config.v_head_dim);
        let latent = config.kv_lora_rank;
        if latent != GMA_LATENT || heads != GMA_HEADS || !value.is_multiple_of(128) {
            return Err(refusal(format!(
                "the bulk MLA takes {GMA_HEADS} heads of a {GMA_LATENT}-wide latent and whole \
                 128-column value tiles, not {heads} of {latent} and {value}"
            )));
        }
        let row_tiles = rows.div_ceil(128) as u32;
        match dsa.kv_key_t {
            Some(key_t) => self.launch(
                self.kernels.prefill_heads_bf16,
                ((latent / 128) as u32, row_tiles, heads as u32),
                BLOCK,
                &[
                    key_t.0,
                    s.query.0,
                    s.q_abs.0,
                    latent as u64,
                    nope as u64,
                    rows as u64,
                    (heads * nope) as u64,
                    (heads * latent) as u64,
                    nope as u64,
                    latent as u64,
                    (latent * nope * 2) as u64,
                ],
            )?,
            None => self.launch(
                self.kernels.mla_absorb,
                (
                    heads as u32,
                    rows as u32,
                    latent.div_ceil(BLOCK as usize) as u32,
                ),
                BLOCK,
                &[
                    s.query.0,
                    dsa.kv_dequant.0,
                    s.q_abs.0,
                    0,
                    nope as u64,
                    value as u64,
                    latent as u64,
                ],
            )?,
        }
        let attend = if self.prefill_activation == PrefillActivation::Bf16 {
            self.kernels.prefill_mla_attend_bf16a
        } else {
            self.kernels.prefill_mla_attend
        };
        for run in runs {
            let SequenceLayer::Sparse {
                latent: cache,
                latent_scales: scales,
                ..
            } = &run.layers[layer]
            else {
                return Err(other_kind());
            };
            self.launch(
                attend,
                (run.len as u32, 1, 1),
                BLOCK,
                &[
                    s.q_abs.0,
                    cache.0,
                    scales.0,
                    s.selected.0,
                    s.count.0,
                    run.first as u64,
                    MAX_SELECTED_TOKENS as u64,
                    float((nope as f32).sqrt()),
                ],
            )?;
        }
        // kv_b's value rows of head h: rows h (nope + value) + nope .. + value of the matrix.
        let per_head = nope + value;
        let element = if dsa.kv_b.scales.is_some() { 1 } else { 2 };
        let weight = dsa.kv_b.weight.offset(nope * latent * element);
        let geometry = [
            value as u64,
            latent as u64,
            rows as u64,
            (heads * latent) as u64,
            (heads * value) as u64,
            latent as u64,
            value as u64,
            (per_head * latent * element) as u64,
        ];
        let grid = ((value / 128) as u32, row_tiles, heads as u32);
        match dsa.kv_b.scales {
            Some(scales) => {
                let blocks = latent / FP8_BLOCK;
                let mut args = vec![
                    weight.0,
                    scales.offset((nope / FP8_BLOCK) * blocks * 4).0,
                    s.q_abs.0,
                    s.attn_out.0,
                ];
                args.extend(geometry);
                args.push(((per_head / FP8_BLOCK) * blocks) as u64);
                self.launch(self.kernels.prefill_heads_fp8, grid, BLOCK, &args)
            }
            None => {
                let mut args = vec![weight.0, s.q_abs.0, s.attn_out.0];
                args.extend(geometry);
                self.launch(self.kernels.prefill_heads_bf16, grid, BLOCK, &args)
            }
        }
    }

    /// The indexer's caching half: each row's key and gate into the caches, then any block each
    /// row completes pooled, a sequence's rows in one launch (a pool reads only cached keys and
    /// gates). A pool is only ever scored by a row at or past the one that completes it, so every
    /// pool a row scores is in by [Self::index_pick]. The scores' head weights stay in
    /// `index_fused`.
    fn index_cache(
        &self,
        indexer: &DeviceIndexer,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (dim, pool) = (config.index_head_dim, config.index_kpool);
        let floats = |count: usize| count * 4;
        self.projection(
            &indexer.fused,
            s.collapsed,
            s.index_fused,
            rows,
            self.exact_prefill,
        )?;
        let gates = indexer.fused.part(s.index_fused, 1, rows);
        for run in runs {
            let SequenceLayer::Sparse {
                index_keys,
                index_gates,
                pooled,
                ..
            } = &run.layers[layer]
            else {
                return Err(other_kind());
            };
            let cached = floats(run.position * dim);
            self.launch(
                self.kernels.layer_norm,
                (run.len as u32, 1, 1),
                BLOCK,
                &[
                    s.index_fused.offset(floats(run.first * dim)).0,
                    indexer.k_norm_weight.0,
                    indexer.k_norm_bias.0,
                    index_keys.offset(cached).0,
                    dim as u64,
                    float(INDEX_KEY_NORM_EPS),
                ],
            )?;
            self.launch(
                self.kernels.copy_flat,
                (blocks(run.len * dim), 1, 1),
                BLOCK,
                &[
                    index_gates.offset(cached).0,
                    gates.offset(floats(run.first * dim)).0,
                    (run.len * dim) as u64,
                ],
            )?;
            self.launch(
                self.kernels.pool,
                (run.len as u32, 1, 1),
                BLOCK,
                &[
                    index_keys.0,
                    index_gates.0,
                    indexer.ape.0,
                    pooled.0,
                    dim as u64,
                    pool as u64,
                    run.position as u64,
                ],
            )?;
        }
        Ok(())
    }

    /// The indexer's picking half: each row's query scores every whole pool, and the scores pick
    /// the tokens the row attends over, a sequence's rows in one launch of each.
    fn index_pick(
        &self,
        indexer: &DeviceIndexer,
        layer: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (dim, heads, pool) = (
            config.index_head_dim,
            config.index_n_heads,
            config.index_kpool,
        );
        let score_stride = config.max_position_embeddings / pool;
        let floats = |count: usize| count * 4;
        self.projection(
            &indexer.wq_b,
            s.dsa_fused,
            s.index_query,
            rows,
            self.exact_prefill,
        )?;
        let head_weights = indexer.fused.part(s.index_fused, 2, rows);
        for run in runs {
            let SequenceLayer::Sparse { pooled, .. } = &run.layers[layer] else {
                return Err(other_kind());
            };
            let (row, position) = (run.first, run.position);
            // The last row scores the most pools; a block past a row's own pools is a no-op.
            let last_pools = (position + run.len) / pool;
            self.launch(
                self.kernels.index_scores,
                (
                    last_pools.div_ceil(SCORE_TILE).max(1) as u32,
                    run.len as u32,
                    1,
                ),
                BLOCK,
                &[
                    s.index_query.offset(floats(row * heads * dim)).0,
                    head_weights.offset(floats(row * heads)).0,
                    pooled.0,
                    s.scores.offset(floats(row * score_stride)).0,
                    position as u64,
                    heads as u64,
                    dim as u64,
                    pool as u64,
                    float(inverse_sqrt(dim)),
                    float(inverse_sqrt(heads)),
                    score_stride as u64,
                ],
            )?;
            self.launch(
                self.kernels.pick,
                (run.len as u32, 1, 1),
                BLOCK,
                &[
                    s.scores.offset(floats(row * score_stride)).0,
                    s.selected.offset(floats(row * MAX_SELECTED_TOKENS)).0,
                    s.count.offset(floats(row)).0,
                    position as u64,
                    pool as u64,
                    config.index_pools() as u64,
                    score_stride as u64,
                    MAX_SELECTED_TOKENS as u64,
                ],
            )?;
        }
        Ok(())
    }

    /// `output = down(swiglu(gate_up input))` for `rows` rows.
    fn mlp(&self, mlp: &DeviceMlp, output: Deviceptr, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let width = mlp.gate_up.rows / 2;
        self.gemv(&mlp.gate_up, s.collapsed, s.gate_up, rows)?;
        self.launch(
            self.kernels.swiglu,
            (blocks(width), rows as u32, 1),
            BLOCK,
            &[
                s.gate_up.0,
                s.activated.0,
                width as u64,
                float(self.config.swiglu_limit as f32),
            ],
        )?;
        self.gemv(&mlp.down, s.activated, output, rows)
    }

    /// Each row's routed experts in ascending expert order, each times its weight, then the shared
    /// expert, into `partial`.
    fn moe(&self, moe: &DeviceMoe, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hidden, width, topk) = (
            config.hidden_size,
            moe.intermediate,
            config.num_experts_per_tok,
        );
        self.mlp(&moe.shared, s.shared_out, rows)?;
        self.projection(
            &moe.router,
            s.collapsed,
            s.router_logits,
            rows,
            self.exact_prefill,
        )?;
        self.launch(
            self.kernels.router,
            (rows as u32, 1, 1),
            // One warp a row picks.
            32,
            &[
                s.router_logits.0,
                moe.bias.0,
                s.expert_indices.0,
                s.expert_weights.0,
                config.n_routed_experts as u64,
                topk as u64,
                u64::from(config.norm_topk_prob),
                float(config.routed_scaling_factor as f32),
            ],
        )?;
        if rows > MAX_ROWS {
            return self.bulk_moe(moe, rows);
        }
        self.expert_gemv(
            moe.gate_up,
            Deviceptr(0),
            s.collapsed,
            s.expert_gate_up,
            2 * width,
            hidden,
            false,
            rows,
        )?;
        self.launch(
            self.kernels.swiglu,
            (blocks(width), (rows * topk) as u32, 1),
            BLOCK,
            &[
                s.expert_gate_up.0,
                s.expert_activated.0,
                width as u64,
                float(config.swiglu_limit as f32),
            ],
        )?;
        self.expert_gemv(
            moe.down,
            s.expert_weights,
            s.expert_activated,
            s.expert_out,
            hidden,
            width,
            true,
            rows,
        )?;
        self.combine_experts(rows)
    }

    /// The routed experts over a prefill chunk, grouped by expert: one item (up to
    /// [MOE_ITEM_ROWS] slots) a block set, the same tensor-core body as a dense projection. The
    /// gate/up GEMM reads each item row from its token's row, and the down GEMM writes each kept
    /// row, times its routing weight, to its slot's row.
    fn bulk_moe(&self, moe: &DeviceMoe, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hidden, width, topk) = (
            config.hidden_size,
            moe.intermediate,
            config.num_experts_per_tok,
        );
        let slots = rows * topk;
        let items = moe_items(slots, config.n_routed_experts);
        let moe_rows = items * MOE_ITEM_ROWS;
        self.launch(
            self.kernels.prefill_moe_plan,
            (1, 1, 1),
            1024,
            &[
                s.expert_indices.0,
                slots as u64,
                config.n_routed_experts as u64,
                s.moe_slot_of.0,
                s.moe_items.0,
                s.moe_item_count.0,
                items as u64,
            ],
        )?;
        if self.prefill_activation == PrefillActivation::Fp8
            && width.is_multiple_of(FP8_BLOCK)
            && hidden.is_multiple_of(FP8_BLOCK)
        {
            self.bulk_moe_q8(moe, rows, items)?;
            return self.combine_experts(rows);
        }
        let mma = if self.prefill_activation == PrefillActivation::Bf16 {
            self.kernels.prefill_moe_mma_bf16a
        } else {
            self.kernels.prefill_moe_mma
        };
        self.launch(
            mma,
            ((2 * width).div_ceil(128) as u32, 1, items as u32),
            BLOCK,
            &[
                moe.gate_up.0,
                s.moe_items.0,
                s.collapsed.0,
                s.moe_gate_up.0,
                (2 * width) as u64,
                hidden as u64,
                s.moe_slot_of.0,
                topk as u64,
                0,
            ],
        )?;
        self.launch(
            self.kernels.prefill_moe_swiglu,
            (blocks(moe_rows * width), 1, 1),
            BLOCK,
            &[
                s.moe_gate_up.0,
                width as u64,
                moe_rows as u64,
                float(config.swiglu_limit as f32),
                s.moe_activated.0,
            ],
        )?;
        self.launch(
            mma,
            (hidden.div_ceil(128) as u32, 1, items as u32),
            BLOCK,
            &[
                moe.down.0,
                s.moe_items.0,
                s.moe_activated.0,
                s.expert_out.0,
                hidden as u64,
                width as u64,
                s.moe_slot_of.0,
                topk as u64,
                s.expert_weights.0,
            ],
        )?;
        self.combine_experts(rows)
    }

    /// [Self::bulk_moe]'s GEMMs over the release's dynamic activation: the chunk's rows quantised
    /// once for every expert's gate/up, and the SwiGLU into the down projection's codes, each an
    /// E4M3 x E4M3 MMA as the decode GEMV quantises the same rows.
    fn bulk_moe_q8(&self, moe: &DeviceMoe, rows: usize, items: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hidden, width, topk) = (
            config.hidden_size,
            moe.intermediate,
            config.num_experts_per_tok,
        );
        q8_inner(hidden)?;
        q8_inner(width)?;
        self.quantise(s.collapsed, rows, hidden)?;
        self.launch(
            self.kernels.prefill_moe_q8,
            ((2 * width).div_ceil(128) as u32, 1, items as u32),
            BLOCK,
            &[
                moe.gate_up.0,
                s.moe_items.0,
                s.q8_codes.0,
                s.q8_scales.0,
                s.moe_gate_up.0,
                (2 * width) as u64,
                hidden as u64,
                s.moe_slot_of.0,
                topk as u64,
                0,
            ],
        )?;
        let moe_rows = items * MOE_ITEM_ROWS;
        // One warp a 128-column block of an item row.
        let warps = moe_rows * width / FP8_BLOCK;
        self.launch(
            self.kernels.prefill_moe_swiglu_q8,
            ((warps * 32).div_ceil(BLOCK as usize) as u32, 1, 1),
            BLOCK,
            &[
                s.moe_gate_up.0,
                width as u64,
                moe_rows as u64,
                float(config.swiglu_limit as f32),
                s.moe_act_codes.0,
                s.moe_act_scales.0,
            ],
        )?;
        self.launch(
            self.kernels.prefill_moe_q8,
            (hidden.div_ceil(128) as u32, 1, items as u32),
            BLOCK,
            &[
                moe.down.0,
                s.moe_items.0,
                s.moe_act_codes.0,
                s.moe_act_scales.0,
                s.expert_out.0,
                hidden as u64,
                width as u64,
                s.moe_slot_of.0,
                topk as u64,
                s.expert_weights.0,
            ],
        )?;
        Ok(())
    }

    /// The routed experts' rows and the shared expert's into each row's partial.
    fn combine_experts(&self, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        self.launch(
            self.kernels.combine,
            (blocks(config.hidden_size), rows as u32, 1),
            BLOCK,
            &[
                s.partial.0,
                s.expert_out.0,
                s.shared_out.0,
                0,
                0,
                0,
                config.num_experts_per_tok as u64,
                config.hidden_size as u64,
                config.hc_mult as u64,
            ],
        )
    }

    /// Refuses a sequence this rank cannot run `tokens` through.
    fn check_sequence(
        &self,
        sequence: &DeviceSequence,
        tokens: &[u32],
    ) -> Result<(), BackendError> {
        let config = &self.config;
        if sequence.position + tokens.len() > sequence.capacity {
            return Err(refusal(format!(
                "the chunk would end at {} past the sequence's {} positions",
                sequence.position + tokens.len(),
                sequence.capacity
            )));
        }
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= config.vocab_size)
        {
            return Err(refusal(format!(
                "token {token} is outside the vocabulary {}",
                config.vocab_size
            )));
        }
        let matches = sequence.layers.len() == self.layers.len()
            && sequence
                .layers
                .iter()
                .zip(&self.layers)
                .all(|(state, layer)| {
                    matches!(
                        (state, &layer.branch),
                        (SequenceLayer::Linear { .. }, DeviceBranch::Linear(_))
                            | (SequenceLayer::Sparse { .. }, DeviceBranch::Sparse(_))
                    )
                })
            && (sequence.drafter.is_none() || self.mtp.is_some())
            && (sequence.block.is_none() || self.block.is_some());
        if !matches {
            return Err(refusal("a sequence made for another model"));
        }
        Ok(())
    }

    /// Run `tokens` through this rank's slice and return one row of full-vocabulary logits per
    /// token, the same on every rank. Every rank of the world runs the same tokens in step, as
    /// [crate::Glm5NextWeights::forward_tp] does. The tokens go [MAX_ROWS] a step, each the same
    /// bits as it is alone ([Self::forward_batch]).
    ///
    /// # Errors
    /// When `tp` or `gather` is not this rank of this world, the sequence is another model's, a
    /// token is outside the vocabulary or past the sequence's capacity, a collective fails, or
    /// the device refuses a launch or a copy.
    pub fn forward_tp(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        sequence: &mut DeviceSequence,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let mut logits = self.forward_batch(tp, gather, &mut [(sequence, tokens)])?;
        Ok(logits.pop().unwrap_or_default())
    }

    /// Whether `sequence`'s prompt may run the bulk prefill. The rank's chunk scratch exists only
    /// for a configuration the chunk can take (see `load`), and the block drafter keeps a step's
    /// rows of features, so a chunk's rows would run past those. Otherwise the prompt runs the
    /// token path, which is what these configurations ran before there was a bulk prefill.
    fn chunk_ready(&self, sequence: &DeviceSequence) -> bool {
        self.prefill.is_some() && sequence.block.is_none()
    }
    /// Prefill `tokens` in bulk chunks and return the last row's full-vocabulary logits. A chunk
    /// carries up to [DeviceRank::prefill_rows] rows through the tensor-core GEMMs and the padded
    /// routed-expert path; a prompt shorter than [MIN_PREFILL_ROWS], a rank opened with
    /// SPARKFLASH_GLM_PREFILL=0, or a sequence [Self::chunk_ready] refuses runs the token path
    /// exactly as [Self::forward_batch] does.
    ///
    /// # Errors
    /// As [Self::forward_batch], and when the prefill scratch is missing from a rank that was
    /// built with one.
    pub fn forward_prefill(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        sequence: &mut DeviceSequence,
        tokens: &[u32],
    ) -> Result<Vec<f32>, BackendError> {
        if tokens.is_empty() {
            return Err(refusal("an empty prompt has no logits"));
        }
        if self.prefill.is_none() || tokens.len() < MIN_PREFILL_ROWS || !self.chunk_ready(sequence)
        {
            let mut logits = self.forward_batch(tp, gather, &mut [(sequence, tokens)])?;
            return Ok(logits
                .pop()
                .and_then(|mut rows| rows.pop())
                .unwrap_or_default());
        }
        self.check_sequence(sequence, tokens)?;
        if sequence.pending.is_some() {
            return Err(refusal(
                "the sequence's verify is not committed; commit it before its next step",
            ));
        }
        let device_leg = self.collective(tp, gather)?;
        let chunk = self.prefill_rows;
        let mut last = Vec::new();
        for part in tokens.chunks(chunk) {
            let run = Run {
                layers: &sequence.layers,
                position: sequence.position,
                first: 0,
                len: part.len(),
                verify: false,
                drafter: sequence.drafter.as_ref(),
                block: sequence.block.as_ref(),
            };
            self.steps = self.steps.wrapping_add(1);
            let prefill = self
                .prefill
                .take()
                .ok_or_else(|| refusal("the prefill scratch is missing"))?;
            let decode = std::mem::replace(&mut self.scratch, prefill);
            let result =
                self.step::<Vec<f32>>(tp, gather, &[run], part, device_leg, HeadRows::Last);
            self.prefill = Some(std::mem::replace(&mut self.scratch, decode));
            let rows = result?;
            last = rows.into_iter().next().unwrap_or_default();
            sequence.position += part.len();
            sequence.logits_at = None;
            if let Some(drafter) = &mut sequence.drafter {
                // The chunk's last row is its stash's last ([Self::fill]).
                drafter.last = Some(part.len().min(MAX_ROWS) - 1);
            }
        }
        Ok(last)
    }

    /// Run each sequence's tokens through this rank's slice together and return each sequence's
    /// logits, one row a token. A step carries up to [MAX_ROWS] rows, the next tokens of each
    /// sequence in turn, so sequences at any positions share each pass over the weights, and each
    /// row is the same bits as when its sequence runs one token a step. Every rank of the world
    /// runs the same batch in step.
    ///
    /// # Errors
    /// As [Self::forward_tp], for any sequence of the batch; when one is refused, none has run.
    pub fn forward_batch(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, &[u32])],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        self.run_batch(tp, gather, batch, false)
    }

    /// A speculative verify: [Self::forward_batch] in one step, each sequence's rows the same
    /// bits, after which a sequence of more than one row keeps its state as its rows found it
    /// until [Self::commit] says how many of them to keep. Until then every call but the commit
    /// refuses the sequence. A sequence of one row needs no commit.
    ///
    /// # Errors
    /// As [Self::forward_batch], and when the rows are more than one step's [MAX_ROWS].
    pub fn verify_batch(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, &[u32])],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        Self::one_step(batch)?;
        self.run_batch(tp, gather, batch, true)
    }

    /// [Self::forward_batch] for greedy decoding: each row's token, its logits' argmax as
    /// [argmax] takes it, picked on the device from the gathered logits, so a step hands the host
    /// a token a row rather than its logits. Each sequence's last row's logits stay on the device
    /// until this rank's next LM head ([Self::last_logits]).
    ///
    /// # Errors
    /// As [Self::forward_batch].
    pub fn forward_batch_greedy(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, &[u32])],
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        self.run_batch(tp, gather, batch, false)
    }

    /// [Self::verify_batch] with [Self::forward_batch_greedy]'s picks; a commit leaves the last
    /// kept row's logits for [Self::last_logits].
    ///
    /// # Errors
    /// As [Self::verify_batch].
    pub fn verify_batch_greedy(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, &[u32])],
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        Self::one_step(batch)?;
        self.run_batch(tp, gather, batch, true)
    }

    /// The full-vocabulary logits of `sequence`'s last row after a greedy step: its last kept
    /// row's once a verify is committed, the bits [Self::forward_batch] would have returned.
    ///
    /// # Errors
    /// When the sequence's last step was not a greedy one of this rank, its verify is not
    /// committed, an LM head (a step's or a draft's) has run since, or the device refuses a copy.
    pub fn last_logits(&self, sequence: &DeviceSequence) -> Result<Vec<f32>, BackendError> {
        let Some(at) = sequence.logits_at else {
            return Err(refusal(
                "the sequence's last step was not a greedy step; its logits went to the host then",
            ));
        };
        if sequence.pending.is_some() {
            return Err(refusal(
                "the sequence's verify is not committed, so its last row is not known yet",
            ));
        }
        if (at.rank, at.head) != (self.identity(), self.heads.load(Ordering::Relaxed)) {
            return Err(refusal(
                "the sequence's last logits are gone: another rank ran its step, or an LM head \
                 has written over them since",
            ));
        }
        let (world, local) = (self.layout.world_size, self.lm_head.rows);
        let mut bytes = vec![0u8; local * 4];
        let mut logits = Vec::with_capacity(world * local);
        for rank in 0..world {
            let slice = self
                .scratch
                .logits
                .offset((rank * at.rows + at.row) * local * 4);
            self.cuda
                .copy_dtoh_on(&self.stream, &mut bytes, &slice)
                .map_err(cuda_error)?;
            logits.extend(to_f32(&bytes));
        }
        Ok(logits)
    }

    /// Refuses a verify of more rows than one step carries.
    fn one_step(batch: &[(&mut DeviceSequence, &[u32])]) -> Result<(), BackendError> {
        let rows: usize = batch.iter().map(|(_, tokens)| tokens.len()).sum();
        if rows > MAX_ROWS {
            return Err(refusal(format!(
                "a verify of {rows} rows; a verify is one step of at most {MAX_ROWS}"
            )));
        }
        Ok(())
    }

    /// Keep the first `kept` rows of `sequence`'s verify ([Self::verify_batch]) and give back the
    /// rest. Each KDA layer replays the kept rows from the memory the verify read, with the
    /// verify's own kernel on its own inputs, and restores the windows after the last kept row;
    /// keeping every row only swaps in the memory the verify left. The DSA caches need nothing:
    /// the positions given back are written again before any row reads them, and a pool is
    /// scored only once the positions it pools are all in. The sequence is then the same bits as
    /// if only the kept rows had run.
    ///
    /// # Errors
    /// When the sequence has no verify to commit, this rank did not run it or has run a step
    /// since, `kept` is 0 or past the verify's rows, or the device refuses a launch.
    pub fn commit(&self, sequence: &mut DeviceSequence, kept: usize) -> Result<(), BackendError> {
        let Some(pending) = sequence.pending else {
            return Err(refusal("the sequence has no verify to commit"));
        };
        if (pending.rank, pending.step) != (self.identity(), self.steps) {
            return Err(refusal(
                "a verify commits on the rank that ran it, before that rank's next step",
            ));
        }
        if !(1..=pending.rows).contains(&kept) {
            return Err(refusal(format!(
                "keeping {kept} rows of a verify of {}; a commit keeps 1 to all of them",
                pending.rows
            )));
        }
        let config = &self.config;
        let floats = |count: usize| count * 4;
        let mut linear = self.scratch.linear.iter();
        for (layer, state) in self.layers.iter().zip(&mut sequence.layers) {
            let (
                DeviceBranch::Linear(kda),
                SequenceLayer::Linear {
                    conv,
                    recurrent,
                    spare,
                },
            ) = (&layer.branch, state)
            else {
                continue;
            };
            let own = linear.next().ok_or_else(other_kind)?;
            if kept < pending.rows {
                let (first, width) = (pending.first, kda.width);
                let beta = kda.fused.part(own.fused, 3, pending.step_rows);
                self.recur(
                    kda,
                    [
                        own.mixed.offset(floats(first * 3 * width)),
                        // The output gate, which a recurrence without an output does not read.
                        Deviceptr(0),
                        beta.offset(floats(first * kda.heads)),
                        own.decay.offset(floats(first * width)),
                    ],
                    (*recurrent, *spare),
                    None,
                    kept,
                )?;
                let window = Self::window_floats(config, width);
                self.launch(
                    self.kernels.copy_flat,
                    (blocks(window), 1, 1),
                    BLOCK,
                    &[
                        conv.0,
                        own.windows.offset(floats((first + kept - 1) * window)).0,
                        window as u64,
                    ],
                )?;
            }
            std::mem::swap(recurrent, spare);
        }
        sequence.position -= pending.rows - kept;
        sequence.pending = None;
        if let Some(at) = &mut sequence.logits_at {
            at.row = pending.first + kept - 1;
        }
        // The MTP layer's caches past the kept rows are written again, as the DSA caches are.
        if let Some(drafter) = &mut sequence.drafter {
            drafter.last = Some(kept - 1);
        }
        // The block drafter's context takes the kept rows, which the verify captured.
        if let (Some(block), Some(scratch)) = (&self.block, &sequence.block) {
            block
                .draft
                .commit(self, scratch, 0, kept, sequence.position - kept)?;
        }
        Ok(())
    }

    /// Refuses a collective or gather of another rank or world, and says whether the collectives
    /// take the device leg.
    fn collective(&self, tp: &TpRuntime, gather: &BitGatherHandle) -> Result<bool, BackendError> {
        let layout = self.layout;
        for (what, rank, world) in [
            ("TP collective", tp.rank(), tp.world()),
            ("bit gather", gather.rank(), gather.world()),
        ] {
            if (rank, world) != (layout.rank, layout.world_size) {
                return Err(refusal(format!(
                    "the {what} is rank {rank} of {world}, the weights rank {} of {}",
                    layout.rank, layout.world_size
                )));
            }
        }
        // The device leg moves raw bits and sums them on the device in Wire::F32's order; a
        // collective whose device leg rounds another way is refused, never summed as if it were.
        match tp.device_wire() {
            None => Ok(false),
            Some(Wire::F32) => Ok(layout.world_size > 1),
            Some(wire) => Err(refusal(format!(
                "the TP collective's device leg sums on the {wire:?} wire; a rank sums its device \
                 leg only as Wire::F32"
            ))),
        }
    }

    /// What tells this rank from any other with live scratch: its scratch's device address.
    fn identity(&self) -> u64 {
        self.scratch.partial.0
    }

    fn run_batch<R: StepRow>(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, &[u32])],
        verify: bool,
    ) -> Result<Vec<Vec<R>>, BackendError> {
        let device_leg = self.collective(tp, gather)?;
        for (sequence, tokens) in batch.iter() {
            self.check_sequence(sequence, tokens)?;
            if sequence.pending.is_some() {
                return Err(refusal(
                    "the sequence's verify is not committed; commit it before its next step",
                ));
            }
        }
        let mut logits: Vec<Vec<R>> = batch
            .iter()
            .map(|(_, tokens)| Vec::with_capacity(tokens.len()))
            .collect();
        // A block drafter's next block sits just past a sequence's tokens and reaches its window
        // back, so the rows further back than that are never read and their commits are skipped.
        let visible: Vec<usize> = batch
            .iter()
            .map(|(sequence, tokens)| {
                self.block.as_ref().map_or(0, |block| {
                    (sequence.position + tokens.len())
                        .saturating_sub(block.draft.config().window_left)
                })
            })
            .collect();
        loop {
            let mut runs = Vec::new();
            let mut tokens = Vec::with_capacity(MAX_ROWS);
            let mut members = Vec::new();
            for (member, (sequence, chunk)) in batch.iter().enumerate() {
                let done = logits[member].len();
                let take = (chunk.len() - done).min(MAX_ROWS - tokens.len());
                if take > 0 {
                    runs.push(Run {
                        layers: &sequence.layers,
                        position: sequence.position,
                        first: tokens.len(),
                        len: take,
                        verify: verify && take > 1,
                        drafter: sequence.drafter.as_ref(),
                        block: sequence.block.as_ref(),
                    });
                    tokens.extend_from_slice(&chunk[done..done + take]);
                    members.push((member, take));
                }
            }
            if tokens.is_empty() {
                return Ok(logits);
            }
            let firsts: Vec<usize> = runs.iter().map(|run| run.first).collect();
            self.steps = self.steps.wrapping_add(1);
            let mut rows = self
                .step::<R>(tp, gather, &runs, &tokens, device_leg, HeadRows::All)?
                .into_iter();
            let head = self.heads.load(Ordering::Relaxed);
            for ((member, take), first) in members.into_iter().zip(firsts) {
                logits[member].extend(rows.by_ref().take(take));
                let sequence = &mut *batch[member].0;
                let position = sequence.position;
                sequence.position += take;
                sequence.logits_at = R::ON_DEVICE.then_some(LogitsAt {
                    rank: self.identity(),
                    head,
                    row: first + take - 1,
                    rows: tokens.len(),
                });
                if let Some(drafter) = &mut sequence.drafter {
                    drafter.last = Some(take - 1);
                }
                if verify && take > 1 {
                    sequence.pending = Some(Pending {
                        rank: self.identity(),
                        step: self.steps,
                        first,
                        rows: take,
                        step_rows: tokens.len(),
                    });
                } else if let (Some(block), Some(scratch)) = (&self.block, &sequence.block) {
                    // A verify's rows wait for its commit, which knows how many the model kept.
                    let unseen = visible[member].saturating_sub(position).min(take);
                    block.draft.commit(
                        &*self,
                        scratch,
                        unseen,
                        take - unseen,
                        position + unseen,
                    )?;
                }
            }
        }
    }

    /// One step: row `row` runs `tokens[row]`, and `runs` say which sequence's position each row
    /// is. Returns what `R` takes from the LM head a row: its full-vocabulary logits, or its pick
    /// with the logits left on the device. On the device leg every collective, the logits' gather
    /// too, is exchanged on the device, and the step's one host copy is the gathered logits or
    /// the picks.
    fn step<R: StepRow>(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        runs: &[Run<'_>],
        tokens: &[u32],
        device_leg: bool,
        head_rows: HeadRows,
    ) -> Result<Vec<R>, BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hc, hidden) = (config.hc_mult, config.hidden_size);
        let rows = tokens.len();
        for (row, &token) in tokens.iter().enumerate() {
            self.embed_row(token, s.streams.offset(row * hc * hidden * 4), hc)?;
        }
        let streams = (s.streams, s.coeffs, hc);
        let mut linear = s.linear.iter();
        let mut fused = false;
        for (index, layer) in self.layers.iter().enumerate() {
            self.mix_in(&layer.attn_hc, layer.input_norm, rows, fused)?;
            match &layer.branch {
                DeviceBranch::Linear(kda) => {
                    let own = linear.next().ok_or_else(other_kind)?;
                    self.kda(kda, index, own, runs, rows)?;
                }
                DeviceBranch::Sparse(dsa) => self.dsa(dsa, index, runs, rows)?,
            }
            self.prefetch(&layer.ffn_reads())?;
            fused = self.mix_out(tp, streams, Some(&layer.ffn_hc), rows, device_leg)?;
            self.mix_in(&layer.ffn_hc, layer.post_norm, rows, fused)?;
            match &layer.ffn {
                DeviceFfn::Dense(mlp) => self.mlp(mlp, s.partial, rows)?,
                DeviceFfn::Moe(moe) => self.moe(moe, rows)?,
            }
            let next = self.layers.get(index + 1);
            match next {
                Some(next) => self.prefetch(&next.attention_reads())?,
                None => self.prefetch(&[self.lm_head.span()])?,
            }
            fused = self.mix_out(tp, streams, next.map(|n| &n.attn_hc), rows, device_leg)?;
            self.capture(index, runs)?;
        }
        self.launch(
            self.kernels.fold,
            (blocks(rows * hidden), 1, 1),
            BLOCK,
            &[
                s.streams.0,
                s.head.0,
                hidden as u64,
                hc as u64,
                rows as u64,
                float(1.0 / hc as f32),
            ],
        )?;
        self.launch(
            self.kernels.rmsnorm,
            (rows as u32, 1, 1),
            BLOCK,
            &[
                s.head.0,
                self.norm.0,
                s.head_normed.0,
                hidden as u64,
                hidden as u64,
                float(config.rms_norm_eps as f32),
                0,
                0,
            ],
        )?;
        let logits = match head_rows {
            HeadRows::All => R::head(self, (tp, gather), s.head_normed, rows, device_leg)?,
            HeadRows::Last => R::head(
                self,
                (tp, gather),
                s.head_normed.offset((rows - 1) * hidden * 4),
                1,
                device_leg,
            )?,
        };
        self.fill(runs, tokens)?;
        Ok(logits)
    }

    /// Layer `index`'s finished rows into the block drafter's target rows of each run that keeps
    /// them, when the drafter reads that layer: the mean of each row's streams.
    fn capture(&self, index: usize, runs: &[Run<'_>]) -> Result<(), BackendError> {
        let Some(block) = &self.block else {
            return Ok(());
        };
        let (hc, hidden) = (self.config.hc_mult, self.config.hidden_size);
        for run in runs {
            if let Some(scratch) = run.block {
                let streams = self.scratch.streams.offset(run.first * hc * hidden * 4);
                block
                    .draft
                    .capture(self, scratch, index, (streams, hc), 0, run.len)?;
            }
        }
        Ok(())
    }

    /// Each of `rows` rows of `input` through the LM head into this rank's slot of the logits,
    /// and on the device leg every rank's slot gathered there: with `draft`, through the E4M3 copy
    /// drafts score with when the rank holds one ([Self::draft_head]).
    fn lm_head(
        &self,
        tp: &TpRuntime,
        input: Deviceptr,
        rows: usize,
        (device_leg, draft): (bool, bool),
    ) -> Result<(), BackendError> {
        self.heads.fetch_add(1, Ordering::Relaxed);
        let s = &self.scratch;
        // On the device leg this rank's slice goes straight into its slot of the gathered logits.
        let slot = rows * self.lm_head.rows * 4;
        let own = if device_leg {
            s.logits.offset(self.layout.rank * slot)
        } else {
            s.logits
        };
        match self.draft_head.filter(|_| draft) {
            Some(head) => gemv_e4m3(self, self.kernels.gemv_e4m3, &head, input, own, rows)?,
            None => self.gemv(&self.lm_head, input, own, rows)?,
        }
        if !device_leg {
            return Ok(());
        }
        tp.exchange_on_device(self.stream.raw(), own.0, s.logits.0, slot)
            .map_err(weights_error)
    }

    /// Each of `rows` rows of `input` through the LM head, every rank's slice gathered into the
    /// row's full-vocabulary logits; with `draft`, the head drafts score with ([Self::lm_head]).
    fn logits(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        input: Deviceptr,
        rows: usize,
        (device_leg, draft): (bool, bool),
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        self.lm_head(tp, input, rows, (device_leg, draft))?;
        let s = &self.scratch;
        let (world, local) = (self.layout.world_size, self.lm_head.rows);
        let slot = rows * local * 4;
        let copied = if device_leg { world * slot } else { slot };
        self.cuda
            .copy_dtoh_mapped_on(&self.stream, &s.logits_host, 0, &s.logits, copied)
            .map_err(cuda_error)?;
        let mut logits: Vec<Vec<f32>> = (0..rows).map(|_| vec![0.0; world * local]).collect();
        // Both lay the ranks' payloads end to end, each this step's rows of its slice.
        if device_leg {
            for (row, values) in logits.iter_mut().enumerate() {
                for (rank, chunk) in values.chunks_mut(local).enumerate() {
                    let at = (rank * rows + row) * local * 4;
                    s.logits_host
                        .read_at(at, bytemuck::cast_slice_mut(chunk))
                        .map_err(cuda_error)?;
                }
            }
        } else {
            let mut own = vec![0.0f32; rows * local];
            s.logits_host
                .read_at(0, bytemuck::cast_slice_mut(&mut own))
                .map_err(cuda_error)?;
            let mut gathered = vec![0.0f32; rows * self.config.vocab_size];
            gather
                .gather_bits(&own, &mut gathered)
                .map_err(weights_error)?;
            for (row, values) in logits.iter_mut().enumerate() {
                for (chunk, rank) in values.chunks_mut(local).zip(gathered.chunks(rows * local)) {
                    chunk.copy_from_slice(&rank[row * local..(row + 1) * local]);
                }
            }
        }
        Ok(logits)
    }

    /// Each of `rows` rows of `input` through the LM head and its argmax as [argmax] takes it,
    /// picked on the device from every rank's slice in `logits`, so the host reads a token a row
    /// and the logits stay on the device. Without a device leg the slices meet on the host as
    /// [Self::logits] gathers them and go back to `logits` laid out as the device leg lays them.
    fn picks(
        &self,
        (tp, gather): (&TpRuntime, &BitGatherHandle),
        input: Deviceptr,
        rows: usize,
        device_leg: bool,
    ) -> Result<Vec<u32>, BackendError> {
        let s = &self.scratch;
        if device_leg {
            self.lm_head(tp, input, rows, (true, false))?;
        } else {
            let logits = self.logits(tp, gather, input, rows, (false, false))?;
            let local = self.lm_head.rows;
            let mut laid = vec![0.0f32; self.layout.world_size * rows * local];
            for (row, values) in logits.iter().enumerate() {
                for (rank, chunk) in values.chunks(local).enumerate() {
                    laid[(rank * rows + row) * local..][..local].copy_from_slice(chunk);
                }
            }
            self.cuda
                .copy_htod_on(&self.stream, &s.logits, &to_bytes(&laid))
                .map_err(cuda_error)?;
        }
        self.argmax_rows(rows, s.picks.device())?;
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let mut picked = vec![0u8; rows * 4];
        s.picks.read_at(0, &mut picked).map_err(cuda_error)?;
        Ok(picked
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect())
    }

    /// The argmax of each of `rows` rows of the gathered logits into `tokens`, a u32 a row.
    fn argmax_rows(&self, rows: usize, tokens: Deviceptr) -> Result<(), BackendError> {
        self.launch(
            self.kernels.argmax,
            (rows as u32, 1, 1),
            ARGMAX_BLOCK,
            &[
                self.scratch.logits.0,
                tokens.0,
                self.layout.world_size as u64,
                rows as u64,
                self.lm_head.rows as u64,
            ],
        )
    }

    /// Token `token`'s embedding into `copies` copies at `out`.
    fn embed_row(&self, token: u32, out: Deviceptr, copies: usize) -> Result<(), BackendError> {
        let hidden = self.config.hidden_size;
        self.launch(
            self.kernels.embed,
            (blocks(copies * hidden), 1, 1),
            BLOCK,
            &[
                self.embed.0,
                out.0,
                u64::from(token),
                hidden as u64,
                (copies * hidden) as u64,
                0,
            ],
        )
    }

    /// The MTP layer's input embeddings of a draft's iteration `iteration`, one a row of `rows`,
    /// from the tokens the device holds for it in `drafted`.
    fn embed_drafted(&self, iteration: usize, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let hidden = self.config.hidden_size;
        let tokens = s.drafted.device().offset(iteration * MAX_ROWS * 4);
        self.launch(
            self.kernels.embed,
            (blocks(hidden), rows as u32, 1),
            BLOCK,
            &[
                self.embed.0,
                s.mtp_embed.0,
                0,
                hidden as u64,
                hidden as u64,
                tokens.0,
            ],
        )
    }

    /// `rows` rows of `x`, `src_stride` floats apart, RMS-normed by `weight` into `out`,
    /// `dst_stride` floats apart.
    fn norm_rows(
        &self,
        (x, src_stride): (Deviceptr, usize),
        weight: Deviceptr,
        (out, dst_stride): (Deviceptr, usize),
        rows: usize,
    ) -> Result<(), BackendError> {
        let hidden = self.config.hidden_size;
        self.launch(
            self.kernels.rmsnorm_rows,
            (1, rows as u32, 1),
            BLOCK,
            &[
                x.0,
                weight.0,
                out.0,
                hidden as u64,
                hidden as u64,
                float(self.config.rms_norm_eps as f32),
                0,
                0,
                src_stride as u64,
                dst_stride as u64,
            ],
        )
    }

    /// `rows` hiddens, one after another from `from`, normed into the hidden halves of the MTP
    /// rows from `row`.
    fn mtp_hidden(
        &self,
        mtp: &DeviceMtp,
        from: Deviceptr,
        row: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        let hidden = self.config.hidden_size;
        let into = self.scratch.mtp_in.offset((2 * row + 1) * hidden * 4);
        self.norm_rows((from, hidden), mtp.hnorm, (into, 2 * hidden), rows)
    }

    /// The MTP layer's input for `rows` rows whose tokens' embeddings are in `mtp_embed` and
    /// normed hiddens in `mtp_in` ([Self::mtp_hidden]): the normed embeddings beside the hiddens,
    /// `eh_proj` into the residual `mtp_x`, and its norm into `collapsed`.
    fn mtp_input(&self, mtp: &DeviceMtp, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let hidden = self.config.hidden_size;
        self.norm_rows(
            (s.mtp_embed, hidden),
            mtp.enorm,
            (s.mtp_in, 2 * hidden),
            rows,
        )?;
        self.gemv(&mtp.eh_proj, s.mtp_in, s.mtp_x, rows)?;
        self.norm_rows(
            (s.mtp_x, hidden),
            mtp.input_norm,
            (s.collapsed, hidden),
            rows,
        )
    }

    /// The MTP layer's caches for each drafting run of the step just run. MTP position `q` pairs
    /// the model's normed hidden at `q` with the token at `q + 1`, so a run from `position` fills
    /// MTP positions from `position - 1`, its first row's hidden the sequence's stash (a run that
    /// starts the sequence has none, and fills a row fewer from 0). Then the run's hiddens
    /// replace the stash. The rows only write caches, so no collective runs.
    fn fill(&self, runs: &[Run<'_>], tokens: &[u32]) -> Result<(), BackendError> {
        let Some(mtp) = &self.mtp else {
            return Ok(());
        };
        let s = &self.scratch;
        let hidden = self.config.hidden_size;
        let floats = |count: usize| count * 4;
        let mut filled = Vec::new();
        let mut rows = 0;
        for run in runs {
            let Some(drafter) = run.drafter else {
                continue;
            };
            let skip = usize::from(drafter.last.is_none());
            for (row, &token) in tokens[run.first + skip..run.first + run.len]
                .iter()
                .enumerate()
            {
                self.embed_row(token, s.mtp_embed.offset(floats((rows + row) * hidden)), 1)?;
            }
            if let Some(last) = drafter.last {
                self.mtp_hidden(mtp, drafter.hidden.offset(floats(last * hidden)), rows, 1)?;
            }
            // Row `r > 0` pairs with the hidden the run's row `r - 1` left.
            let own = run.len - 1;
            if own > 0 {
                let from = s.head_normed.offset(floats(run.first * hidden));
                self.mtp_hidden(mtp, from, rows + 1 - skip, own)?;
            }
            if run.len > skip {
                filled.push(Run {
                    layers: std::slice::from_ref(&drafter.layer),
                    position: run.position + skip - 1,
                    first: rows,
                    len: run.len - skip,
                    verify: false,
                    drafter: None,
                    block: None,
                });
                rows += run.len - skip;
            }
        }
        if rows > 0 {
            self.mtp_input(mtp, rows)?;
            self.dsa_cache(&mtp.dsa, 0, &filled, rows)?;
        }
        // Each run's last [MAX_ROWS] rows into its stash: a prefill chunk's earlier rows are never
        // read again, and the stash holds no more.
        for run in runs {
            if let Some(drafter) = run.drafter {
                let kept = run.len.min(MAX_ROWS);
                self.launch(
                    self.kernels.copy_flat,
                    (blocks(kept * hidden), 1, 1),
                    BLOCK,
                    &[
                        drafter.hidden.0,
                        s.head_normed
                            .offset(floats((run.first + run.len - kept) * hidden))
                            .0,
                        (kept * hidden) as u64,
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Draft `count` tokens after each sequence's `next`, the token at its position: the MTP
    /// layer runs one row a sequence an iteration, on the model's hidden before `next` and then on
    /// its own output, and each draft is the first greatest of its logits from the head drafts
    /// score with ([Self::draft_head]). Every rank of the world
    /// drafts the same batch in step, and every rank's drafts are the same, as are a sequence's
    /// alone or batched. The drafts write the MTP layer's caches past what the sequence keeps,
    /// which its next step writes again before any row reads them. On the device leg the
    /// device picks each draft and embeds it for the next iteration, so the host waits once, for
    /// all the drafts together.
    ///
    /// # Errors
    /// When `tp` or `gather` is not this rank of this world, there are more sequences than a
    /// step's [MAX_ROWS] or more drafts than a verify can check after the token they follow, a
    /// sequence was not made to draft, has run no step, has a verify to commit or would draft
    /// past its capacity, `next` is outside the vocabulary, a collective fails, or the device
    /// refuses a launch or a copy.
    pub fn draft(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, u32)],
        count: usize,
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        let device_leg = self.collective(tp, gather)?;
        if batch.iter().any(|(sequence, _)| sequence.block.is_some()) {
            return self.draft_blocks((tp, gather, device_leg), batch, count);
        }
        let Some(mtp) = &self.mtp else {
            return Err(refusal("this rank has no MTP layer to draft with"));
        };
        let rows = batch.len();
        if rows > MAX_ROWS {
            return Err(refusal(format!(
                "a draft of {rows} sequences; a draft runs a row a sequence, at most {MAX_ROWS}"
            )));
        }
        if count >= MAX_ROWS {
            return Err(refusal(format!(
                "{count} drafts; a verify checks them after the token they follow, one step of \
                 at most {MAX_ROWS} rows"
            )));
        }
        let s = &self.scratch;
        let hidden = self.config.hidden_size;
        let mut stashes = Vec::with_capacity(rows);
        for (sequence, next) in batch.iter() {
            self.check_sequence(sequence, &[*next])?;
            if sequence.pending.is_some() {
                return Err(refusal(
                    "the sequence's verify is not committed; commit it before its next step",
                ));
            }
            let Some(drafter) = &sequence.drafter else {
                return Err(refusal(
                    "the sequence keeps no MTP state; new_drafting_sequence makes one that does",
                ));
            };
            let Some(last) = drafter.last else {
                return Err(refusal(
                    "the sequence has run no step; its first draft needs the hidden one leaves",
                ));
            };
            // The drafts fill MTP positions position - 1 .. position + count - 2.
            if sequence.position + count > sequence.capacity + 1 {
                return Err(refusal(format!(
                    "{count} drafts from {} are past the sequence's {} positions",
                    sequence.position, sequence.capacity
                )));
            }
            stashes.push((drafter, drafter.hidden.offset(last * hidden * 4)));
        }
        // Iteration 0 embeds the tokens the drafts follow. No launch still reads `drafted`: a
        // draft waits for all of its launches before it returns.
        let at = |iteration: usize, row: usize| (iteration * MAX_ROWS + row) * 4;
        for (row, (_, next)) in batch.iter().enumerate() {
            s.drafted
                .write_at(at(0, row), &next.to_le_bytes())
                .map_err(cuda_error)?;
        }
        for iteration in 0..count {
            self.embed_drafted(iteration, rows)?;
            if iteration == 0 {
                for (row, (_, stash)) in stashes.iter().enumerate() {
                    self.mtp_hidden(mtp, *stash, row, 1)?;
                }
            } else {
                self.mtp_hidden(mtp, s.mtp_out, 0, rows)?;
            }
            self.mtp_input(mtp, rows)?;
            let runs: Vec<Run<'_>> = batch
                .iter()
                .zip(&stashes)
                .enumerate()
                .map(|(row, ((sequence, _), (drafter, _)))| Run {
                    layers: std::slice::from_ref(&drafter.layer),
                    position: sequence.position - 1 + iteration,
                    first: row,
                    len: 1,
                    verify: false,
                    drafter: None,
                    block: None,
                })
                .collect();
            let residual = (s.mtp_x, s.mtp_coeffs, 1);
            self.dsa(&mtp.dsa, 0, &runs, rows)?;
            self.prefetch(&mtp.moe.reads())?;
            self.mix_out(tp, residual, None, rows, device_leg)?;
            self.norm_rows(
                (s.mtp_x, hidden),
                mtp.post_norm,
                (s.collapsed, hidden),
                rows,
            )?;
            self.moe(&mtp.moe, rows)?;
            match self.draft_head {
                Some(head) => self.prefetch(&[(head.weight, head.rows * head.columns)])?,
                None => self.prefetch(&[self.lm_head.span()])?,
            }
            self.mix_out(tp, residual, None, rows, device_leg)?;
            self.norm_rows((s.mtp_x, hidden), mtp.head_norm, (s.mtp_out, hidden), rows)?;
            if device_leg {
                self.lm_head(tp, s.mtp_out, rows, (true, true))?;
                self.argmax_rows(rows, s.drafted.device().offset(at(iteration + 1, 0)))?;
            } else {
                let logits = self.logits(tp, gather, s.mtp_out, rows, (false, true))?;
                for (row, values) in logits.iter().enumerate() {
                    let token =
                        argmax(values).ok_or_else(|| refusal("a draft's logits are empty"))?;
                    s.drafted
                        .write_at(at(iteration + 1, row), &(token as u32).to_le_bytes())
                        .map_err(cuda_error)?;
                }
            }
        }
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let mut drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(count); rows];
        let mut picked = [0u8; 4];
        for iteration in 1..=count {
            for (row, drafted) in drafts.iter_mut().enumerate() {
                s.drafted
                    .read_at(at(iteration, row), &mut picked)
                    .map_err(cuda_error)?;
                drafted.push(u32::from_le_bytes(picked));
            }
        }
        for (sequence, _) in batch.iter_mut() {
            sequence.confidence.clear();
        }
        Ok(drafts)
    }

    /// [Self::draft] for block-drafting sequences: each sequence's block after its `next` runs
    /// through the drafter, the ranks merge their slices' candidates for its mask rows, and the
    /// selector walks them into a chain whose first `count` tokens are the drafts. One exchange
    /// carries every sequence's candidates, and the host waits once, for all the chains.
    fn draft_blocks(
        &self,
        (tp, gather, device_leg): (&TpRuntime, &BitGatherHandle, bool),
        batch: &mut [(&mut DeviceSequence, u32)],
        count: usize,
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        let Some(block) = &self.block else {
            return Err(refusal("this rank has no block drafter to draft with"));
        };
        let (draft, world) = (&block.draft, self.layout.world_size);
        let block_size = draft.config().block_size;
        let rows = batch.len();
        if rows > MAX_ROWS {
            return Err(refusal(format!(
                "a draft of {rows} sequences; a draft runs a block a sequence, at most {MAX_ROWS}"
            )));
        }
        if count >= block_size {
            return Err(refusal(format!(
                "{count} drafts; the block drafter proposes at most {} after the token they follow",
                block_size - 1
            )));
        }
        let mut scratches = Vec::with_capacity(rows);
        for (sequence, next) in batch.iter() {
            self.check_sequence(sequence, &[*next])?;
            if sequence.pending.is_some() {
                return Err(refusal(
                    "the sequence's verify is not committed; commit it before its next step",
                ));
            }
            let Some(scratch) = sequence.block else {
                return Err(refusal(
                    "a draft mixes block-drafting sequences with others; one draft drafts alike",
                ));
            };
            scratches.push(scratch);
        }
        let floats = |count: usize| count * 4;
        let candidates = Self::candidates(draft);
        // Row `row`'s candidates sit `row * candidates` into each rank's slot, ids after values.
        let lists = |base: Deviceptr, row: usize| {
            let values = base.offset(floats(row * candidates));
            (values, values.offset(floats(candidates / 2)))
        };
        let slot = rows * candidates;
        let own = block.exchange.offset(floats(self.layout.rank * slot));
        let first_id = self.layout.rank * self.lm_head.rows;
        let scoring = Scoring {
            rank: self,
            tp,
            device_leg,
        };
        for (row, ((sequence, next), scratch)) in batch.iter().zip(&scratches).enumerate() {
            let at = sequence.position;
            draft.score(
                &scoring,
                scratch,
                self.embed,
                *next,
                at,
                first_id,
                lists(own, row),
            )?;
        }
        let gathered = if device_leg {
            tp.exchange_on_device(self.stream.raw(), own.0, block.exchange.0, floats(slot))
                .map_err(weights_error)?;
            block.exchange
        } else {
            let host = &block.exchange_host;
            self.cuda
                .copy_dtoh_mapped_on(&self.stream, host, 0, &own, floats(slot))
                .map_err(cuda_error)?;
            let mut mine = vec![0.0f32; slot];
            host.read_at(0, bytemuck::cast_slice_mut(&mut mine))
                .map_err(cuda_error)?;
            let mut every = vec![0.0f32; world * slot];
            gather
                .gather_bits(&mine, &mut every)
                .map_err(weights_error)?;
            host.write_at(0, bytemuck::cast_slice(&every))
                .map_err(cuda_error)?;
            host.device()
        };
        let chain = floats(2 * block_size);
        for (row, ((_, next), scratch)) in batch.iter().zip(&scratches).enumerate() {
            draft.merge(self, scratch, lists(gathered, row), world, slot)?;
            draft.select(self, scratch, *next)?;
            self.launch(
                self.kernels.copy_flat,
                (blocks(2 * block_size), 1, 1),
                BLOCK,
                &[
                    block.chains.device().offset(row * chain).0,
                    scratch.drafts().0,
                    (2 * block_size) as u64,
                ],
            )?;
        }
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        batch
            .iter_mut()
            .enumerate()
            .map(|(row, (sequence, _))| {
                let mut drafts = vec![0u32; count];
                block
                    .chains
                    .read_at(row * chain, bytemuck::cast_slice_mut(&mut drafts))
                    .map_err(cuda_error)?;
                sequence.confidence.resize(count, 0.0);
                block
                    .chains
                    .read_at(
                        row * chain + floats(block_size),
                        bytemuck::cast_slice_mut(&mut sequence.confidence),
                    )
                    .map_err(cuda_error)?;
                Ok(drafts)
            })
            .collect()
    }
}

/// The block drafter's launches go on the rank's stream, and its BF16 GEMVs through the rank's.
impl DraftDevice for DeviceRank {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch_on(&self.stream, kernel, grid, block, args)
            .map_err(cuda_error)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let linear = DeviceLinear {
            weight: matrix.pointer,
            scales: None,
            rows: matrix.rows,
            columns: matrix.columns,
            parts: [matrix.rows; 3],
        };
        DeviceRank::gemv(self, &linear, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.scratch.partials, self.scratch.counters)
    }
}

/// The rank as the block drafter's device while it scores a block: the rank's launches, GEMVs and
/// tiled scratch, and the TP collective that sums the sharded drafter's partials as
/// [DeviceRank::mix_out] sums the model's, into one stream by `mtp_coeffs` (`x + reduced`).
struct Scoring<'r> {
    rank: &'r DeviceRank,
    tp: &'r TpRuntime,
    device_leg: bool,
}

impl DraftDevice for Scoring<'_> {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        DraftDevice::launch(self.rank, kernel, grid, block, args)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        DraftDevice::gemv(self.rank, matrix, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        DraftDevice::tiled_scratch(self.rank)
    }

    fn reduce_add(
        &self,
        partial: Deviceptr,
        residual: Deviceptr,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        let rank = self.rank;
        if rows > MAX_ROWS || width != rank.geometry.dim {
            return Err(refusal(format!(
                "{rows} rows of {width} partials; the rank sums at most {MAX_ROWS} rows of {}",
                rank.geometry.dim
            )));
        }
        let one_stream = (residual, rank.scratch.mtp_coeffs, 1);
        rank.reduce_into(self.tp, partial, one_stream, None, rows, self.device_leg)
    }
}

impl Drop for DeviceRank {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize_stream(&self.stream);
        if let Some(prefetch) = &self.prefetch {
            self.cuda.destroy_side_stream(&prefetch.side);
        }
        if let Some(block) = &self.block {
            block.draft.free(&self.cuda);
        }
        self.cuda.destroy_run_stream(&self.stream);
    }
}
