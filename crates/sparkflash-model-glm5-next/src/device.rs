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
//! and recurrence in their per-channel KDA form, and the selected-key attention. The GLM decode
//! kernels come from this arm's own image ([Kernels::load]'s `decode`): the mHC mixers, the clamped
//! SwiGLU, the noaux router, the k-pool indexer and the absorbed MLA, the tiled GEMV wrappers the
//! GLM projections use, and the routed experts' FP8 GEMV.
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

use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sparkflash_kernels::checkpoint::Checkpoint;
use sparkflash_kernels::cuda_driver::{
    Cuda, DEFAULT_DIRECT_CHUNK_BYTES, DEFAULT_DIRECT_CONCURRENCY, Deviceptr, HostMapped, Kernel,
    RunStream, SideStream, StreamEvent,
};
use sparkflash_kernels::images::{self, Image};
use sparkflash_kernels::lane_memory::{self, BackError, LanePlans, Layout, SequenceLanes};
use sparkflash_model::{BackendError, Part, StepCost};
use sparkflash_ops::OpPlan;
use sparkflash_ops::calls::RmsNorm;
use sparkflash_ops::dflash::{DraftDevice, DraftScratch, GpuDflash2};
use sparkflash_ops::formats::{
    Bf16, CodePlanes, F32 as F32Format, FormatKind, Fp8E4m3, Mat, Scales, Tensor,
    upload as upload_format,
};
use sparkflash_ops::gdn::{
    CONV_HISTORY, ChunkConv, ConvRows, GDN_KEY, GDN_QK_ROW, GDN_VALUE, GatedNorm, TokenConv,
};
use sparkflash_ops::gemv::{
    GEMV_MAX_BATCH, GemvDevice, GpuMat, GpuMatE4m3, TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS,
    TILED_GEMV_ROUND, gemv_e4m3, tiled_gemv_e4m3_splits, tiled_gemv_splits,
};
use sparkflash_ops::indexer::{IndexerPass, MAX_BUDGET_BLOCKS, MAX_SELECTED_TOKENS};
use sparkflash_ops::moe::{Launch, MoePlanCall, combine_grid};
use sparkflash_ops::packed::{self, Form, PackedSet, Placement, Tiled};
use sparkflash_ops::plan::Precision;
use sparkflash_ops::projection::{
    Fp8Tile, Route, WIDE_SHARED, WidePolicy, WideTile, fp8_tile_fits,
};
use sparkflash_ops::roles;
use sparkflash_ops::roles::{Bound, Embed, Launcher, LayerNorm, Norm};
use sparkflash_ops::scan::{ScanKind, ScanPlan};
use sparkflash_ops::tp::{host_sum, reduce_scatter, reduce_two_shot, takes_two_shots};
use sparkflash_ops::widen::{FormatLedger, LedgerRow, WidenLog, widen};

mod numerics;

fn cubin_available() -> bool {
    !images::qwen4_exp_cubin().is_empty() && !images::glm_decode_cubin().is_empty()
}

fn embedded_cubin() -> &'static [u8] {
    images::qwen4_exp_cubin()
}

fn embedded_glm_decode_cubin() -> &'static [u8] {
    images::glm_decode_cubin()
}

fn embedded_moe_cubin() -> &'static [u8] {
    images::qwen4_moe_cubin()
}

fn embedded_glm_prefill_cubin() -> &'static [u8] {
    images::glm_prefill_cubin()
}

fn embedded_glm_moe_cubin() -> &'static [u8] {
    images::glm_moe_cubin()
}

fn embedded_glm_attention_cubin() -> &'static [u8] {
    images::glm_attention_cubin()
}

fn embedded_scan_cubin() -> &'static [u8] {
    images::qwen4_scan_cubin()
}
use sparkflash_weights::linalg::{ScaleGrid, argmax};
use sparkflash_weights::mhc::HcGeometry;
use sparkflash_weights::rank_file::SlotCursor;
use sparkflash_weights::tp::{BitGatherHandle, TpLayout, TpRuntime, Wire};
use sparkflash_weights::{DType, TensorSpec, WeightSource, check_length};

use crate::config::{Glm5NextConfig, IndexerKind, LayerKind, MlpKind};
use crate::convert::{ConvertedIndex, ConvertedSpecs};
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

impl LatentCache {
    /// Whether a rank with this cache stores its latent as E4M3 codes.
    ///
    /// [LatentCache::Auto] is the per-tensor choice: the latent is quantised only when the
    /// rank's own `kv_b` is block-scaled and the projections quantise their activations. The
    /// checkpoint-wide `fp8_block` flag names the block width of the other projections and
    /// says nothing about `kv_b_proj`; the GLM tensor plan stores that weight BF16
    /// ([crate::tensor]), so Auto is the F32 latent there even for an FP8 checkpoint. A caller
    /// that must admit the cache before it has read a tensor passes an explicit representation
    /// instead ([crate::SINGLE_RANK_LATENT_FP8]).
    ///
    /// The E4M3 latent needs a latent width that is whole 128-wide scale tiles; a narrower
    /// width falls back to f32 whatever the caller asked for, which is the same condition
    /// [crate::kv_plan_with_latent] callers pass.
    #[must_use]
    pub fn quantises(self, config: &Glm5NextConfig, kv_b_block_scaled: bool) -> bool {
        (match self {
            Self::Auto => config.dynamic_activations && kv_b_block_scaled,
            Self::F32 => false,
            Self::Fp8 => true,
        }) && config.kv_lora_rank.is_multiple_of(FP8_BLOCK)
    }
}

/// The latent-cache row one sparse layer allocates a position.
///
/// Returns `(latent elements, f32 scale floats)`: the E4M3 latent is one code a element and one
/// f32 scale a whole [FP8_BLOCK]-wide tile, while the f32 latent is four bytes a element and no
/// scales. The representation is resolved exactly as [LatentCache::quantises] does it; this is
/// the caches [SparseParts::plan] lays out from the same pair.
fn latent_row(config: &Glm5NextConfig, quantise: bool) -> (usize, usize) {
    let width = config.kv_lora_rank;
    (
        width,
        if quantise {
            width.div_ceil(FP8_BLOCK)
        } else {
            0
        },
    )
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
    fn from_profile(
        config: &Glm5NextConfig,
        profile: &sparkflash_model::ModelProfile,
    ) -> Result<Self, BackendError> {
        use sparkflash_model::profile::ActivationMode;
        match profile.prefill.activation {
            None => Ok(if config.dynamic_activations {
                Self::Fp8
            } else {
                Self::Split
            }),
            Some(ActivationMode::Fp8) if config.dynamic_activations => Ok(Self::Fp8),
            Some(ActivationMode::Fp8) => Err(refusal(
                "prefill.activation=fp8 quantises activations this release does not; take split \
                 or bf16",
            )),
            Some(ActivationMode::Split) => Ok(Self::Split),
            Some(ActivationMode::Bf16) => Ok(Self::Bf16),
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

/// How a prompt runs ([DeviceRank::begin_prefill]).
#[derive(Debug)]
pub enum Prefill {
    /// The token path ran it whole: its last row's logits.
    Whole(Vec<f32>),
    /// It takes bulk chunks of `width` tokens, the last one shorter, ending at position `end`
    /// ([DeviceRank::prefill_chunk]).
    Chunks {
        /// The position past the prompt's last token.
        end: usize,
        /// The tokens a chunk takes.
        width: usize,
    },
}

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

/// The modelled wall of an MTP round that drafts and verifies `depth` drafts (ms), for
/// [DepthPolicy](sparkflash_model::DepthPolicy): the one-token step at 34.0 ms and 9.0 ms a draft,
/// its MTP pass and its verify row together, since the MTP layer drafts one token a pass. Fit to
/// one stream's 34.0, 43.5, 52 and 61 ms rounds at 0 to 3 drafts over the NCCL leg
/// (docs/models/glm53-flash.md); the first round assumes the 45 tok/s it made at 2.
pub const MTP_STEP_COST: StepCost = StepCost {
    step_ms: 34.0,
    row_ms: 9.0,
    prior_tokens_per_ms: 0.045,
};

/// The side of a block-scaled FP8 tile the kernels read (Q4E_FP8_BLOCK).
const FP8_BLOCK: usize = 128;
/// The dynamic shared memory of the bulk E4M3 x E4M3 GEMMs in the prefill and MoE images
/// (GLM_Q8_SHARED): their cp.async stages and a column tile's weight scales.
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
/// glm_prefill_exact_bf16's geometry (GPE_OUTS, GPE_ROWS, GPE_THREADS): the outputs and rows a
/// block computes.
const EXACT_OUTS: usize = 32;
const EXACT_ROWS: usize = 32;
const EXACT_THREADS: u32 = 128;
/// glm_prefill_mla_attend's geometry (GMA_HEADS, GMA_LATENT): a rank's heads as one MMA's M, over
/// the latent's four 128-column scale blocks.
const GMA_HEADS: usize = 16;
const GMA_LATENT: usize = 512;
/// The decode attention's bounds (glm_decode_sm121.cu's Q4E_GLM_MLA_HEADS and
/// Q4E_GLM_MAX_LATENT): a rank's heads q4e_glm_mla_scores stages, over the widest latent.
const MLA_HEADS: usize = 16;
const MLA_MAX_LATENT: usize = 512;
/// q4e_glm_mla_scores' positions a block, q4e_glm_mla_mix's heads a block and
/// q4e_glm_mla_project's value columns a block.
const MLA_SCORE_SLOTS: usize = 64;
const MLA_MIX_HEADS: usize = 8;
const MLA_PROJECT_THREADS: u32 = 64;
/// A routed-expert item's rows (glm_moe_sm121.cu's GLM_MOE_ROWS): one expert's slots, one
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
const MAX_HC: usize = roles::MAX_STREAMS;
/// Q4E_MHC_COLUMNS: the columns of every stream copy a thread of q4e_mhc_in holds, so the hidden
/// width is a multiple of [BLOCK] of at most this many. The kernel's array is 16 columns: a lift
/// to 32 would add some 224 live floats a thread and a larger cap no released config uses
/// (D98's correction).
const MHC_COLUMNS: usize = roles::MIX_COLUMNS;
const MAX_POOL: usize = roles::MAX_POOL;
const MAX_INDEX_HEADS: usize = roles::MAX_INDEX_HEADS;
const MAX_INDEX_DIM: usize = roles::MAX_INDEX_DIM;
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

// The shared context handle: one definition in `sparkflash_ops`, so both arms carry the same
// `Send`/`Sync` wrapper around the driver context.
pub use sparkflash_ops::gemv::CudaHandle;

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

/// The smallest BF16 projection GLM packs is one of more than 2^24 weights. The tiled packed GEMV
/// reads three quarters of the bytes but rebuilds each chunk and runs two blocks an SM, so it
/// gains on a matrix that streams long enough: streaming from DRAM on one Spark, 38,720 x 4,096
/// (the head's slice) took 1,061 against 1,325 us at one row and 1,144 against 1,573 at eight;
/// 6,160 x 4,096 191 against 216 us at one row, even at eight; 4,096 x 2,048 79 against 78 us at
/// one row and 102 against 81 at eight; the 288 x 4,096 router lost at every batch.
const PACK_MIN_WEIGHTS: usize = 1 << 24;

/// The kernels a rank launches, by name from the pack's cubin.
#[derive(Clone, Copy)]
struct Kernels {
    embed: Bound,
    argmax: Bound,
    /// [Self::argmax] that also reports the pick's softmax probability: an MTP draft's confidence.
    argmax_prob: Bound,
    gemv_bf16: Bound,
    /// [Self::gemv_bf16] over a packed copy ([sparkflash_ops::packed]): the same sums.
    gemv_packed: Bound,
    gemv_fp8: Bound,
    /// The E4M3 copy of the LM head drafts score with ([DeviceRank::draft_head]): its quantizer
    /// and its GEMV.
    quantize_e4m3: Bound,
    gemv_e4m3: Bound,
    expert_fp8: Bound,
    mhc_in: Bound,
    mhc_post: Bound,
    sum_slots: Bound,
    norm: Norm,
    conv: TokenConv,
    kda_decay: Bound,
    recur: Bound,
    /// A bulk chunk's convolution and recurrence (Qwen's prefill image,
    /// kernels/qwen4_prefill_sm121.cu): the convolution a thread an output, then the history
    /// advanced, the token kernel's products in its order; the rows' normed q and k and beta,
    /// the scan with the state in registers, and the gated norm. The token kernels ([Self::conv],
    /// [Self::recur]) stay the decode and verify path, whose bits a replay is defined by.
    chunk_conv: ChunkConv,
    kda_prep: Bound,
    kda_scan: Bound,
    kda_gate: GatedNorm,
    copy_flat: Bound,
    layer_norm: LayerNorm,
    pool: Bound,
    index_scores: Bound,
    pick: Bound,
    /// The absorbed MLA ([DeviceRank::dsa_attend]): the latent cache writer, the query's
    /// projection into the latent space, then the attention's logits, softmax, weighted latent
    /// and value projection.
    mla_cache: Bound,
    mla_cache_fp8: Bound,
    mla_absorb: Bound,
    mla_scores: Bound,
    mla_softmax: Bound,
    mla_mix: Bound,
    mla_project: Bound,
    swiglu: Bound,
    router: Bound,
    combine: Bound,
    fold: Bound,
    prefetch_l2: Bound,
    /// The bulk prefill image (kernels/glm_prefill_sm121.cu): dense block-scaled FP8 and BF16
    /// GEMMs (the BF16 one over the 128 x 256 tile, [WIDE_SHARED]), and the skinny projections.
    /// The padded routed-expert plan and its gather/SwiGLU/scatter are in the MoE image
    /// (kernels/glm_moe_sm121.cu).
    prefill_dense_bf16: Bound,
    prefill_dense_fp8: Bound,
    /// The same two GEMMs over the bf16 head of each activation alone, and the routed one.
    prefill_dense_bf16_bf16a: Bound,
    prefill_dense_fp8_bf16a: Bound,
    /// The FP8 tile's few-row form (kernels/glm_decode_sm121.cu), the decode path's body: one
    /// to eight rows, a row the prompt tile's bits over GLM's f32 128 x 128 block scales.
    prefill_dense_fp8_rows: Bound,
    /// The bulk routed-expert image (kernels/glm_moe_sm121.cu): the padded plan, the
    /// gather/SwiGLU/scatter, and the gate/up and down MMAs over f32 and E4M3 activations.
    prefill_moe_mma_bf16a: Bound,
    prefill_moe_plan: Bound,
    prefill_moe_mma: Bound,
    prefill_moe_swiglu: Bound,
    /// The checkpoint's dynamic activation scheme ([PrefillActivation::Fp8]): rows quantised per
    /// 128 columns, the SwiGLU into the down projection's codes, and the E4M3 x E4M3 dense and
    /// routed GEMMs over them.
    prefill_quantise: Bound,
    prefill_moe_swiglu_q8: Bound,
    prefill_dense_q8: Bound,
    prefill_moe_q8: Bound,
    /// A BF16 part of at most 8 (i + 1) rows ([DeviceRank::bulk_gemm]'s mixers and gates).
    prefill_skinny: [Bound; 4],
    /// The tiled GEMV's floats for a BF16 part it splits one round a block (the exact prefill's
    /// router and indexer parts, [DeviceRank::gemv_chunked]).
    prefill_exact: Bound,
    /// The chunk's MLA ([DeviceRank::bulk_attend]) from the bulk attention image
    /// (kernels/glm_attention_sm121.cu): the tensor-core sparse attention over the E4M3 latent
    /// (and over its bf16 heads alone), and the per-head GEMM of the absorbed query and the
    /// value projection.
    prefill_mla_attend: Bound,
    prefill_index_scores: Bound,
    prefill_mla_attend_bf16a: Bound,
    prefill_heads_bf16: Bound,
    prefill_heads_fp8: Bound,
}

// SAFETY: a kernel is an immutable function handle loaded from the cubin. Every launch goes
// through a rank's `&mut self` on that rank's own stream, so a handle that moves to another
// thread is never used concurrently.
unsafe impl Send for Kernels {}

impl Kernels {
    fn load(cuda: &Cuda, head_dim: usize) -> Result<Self, String> {
        let token = |symbol: &'static str, precision: Precision| {
            Bound::new(cuda, Image::Qwen4Exp, embedded_cubin(), symbol, precision)
        };
        let decode = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::GlmDecode,
                embedded_glm_decode_cubin(),
                symbol,
                precision,
            )
        };
        if embedded_glm_prefill_cubin().is_empty() {
            return Err("this build embeds no GLM bulk prefill cubin".to_owned());
        }
        let prefill = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::GlmPrefill,
                embedded_glm_prefill_cubin(),
                symbol,
                precision,
            )
        };
        let prefill_q8 = |symbol: &'static str, precision: Precision| {
            Bound::shared(
                cuda,
                Image::GlmPrefill,
                embedded_glm_prefill_cubin(),
                symbol,
                precision,
                Q8_SHARED,
            )
        };
        let prefill_wide = |symbol: &'static str, precision: Precision| {
            Bound::shared(
                cuda,
                Image::GlmPrefill,
                embedded_glm_prefill_cubin(),
                symbol,
                precision,
                WIDE_SHARED,
            )
        };
        if embedded_glm_moe_cubin().is_empty() {
            return Err("this build embeds no GLM routed-expert cubin".to_owned());
        }
        let glm_moe = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::GlmMoe,
                embedded_glm_moe_cubin(),
                symbol,
                precision,
            )
        };
        // The routed-expert plan is the shared image (kernels/moe_plan_sm121.cu), not GLM's
        // own: Qwen's token path, this arm's bulk prefill and DeepSeek's bulk prefill run one
        // implementation of the counting pass.
        if images::moe_plan_cubin().is_empty() {
            return Err("this build embeds no routed-expert plan cubin".to_owned());
        }
        let plan = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::MoePlan,
                images::moe_plan_cubin(),
                symbol,
                precision,
            )
        };
        let glm_moe_q8 = |symbol: &'static str, precision: Precision| {
            Bound::shared(
                cuda,
                Image::GlmMoe,
                embedded_glm_moe_cubin(),
                symbol,
                precision,
                Q8_SHARED,
            )
        };
        if embedded_glm_attention_cubin().is_empty() {
            return Err("this build embeds no GLM MLA attention cubin".to_owned());
        }
        let attention = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::GlmAttention,
                embedded_glm_attention_cubin(),
                symbol,
                precision,
            )
        };
        if embedded_scan_cubin().is_empty() {
            return Err("this build embeds no scan cubin for the bulk KDA scan".to_owned());
        }
        let scan = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::Qwen4Scan,
                embedded_scan_cubin(),
                symbol,
                precision,
            )
        };
        if embedded_moe_cubin().is_empty() {
            return Err("this build embeds no Qwen4 MoE cubin for the routed combine".to_owned());
        }
        let moe = |symbol: &'static str, precision: Precision| {
            Bound::new(
                cuda,
                Image::Qwen4Moe,
                embedded_moe_cubin(),
                symbol,
                precision,
            )
        };
        if embedded_glm_decode_cubin().is_empty() {
            return Err("this build embeds no GLM decode/token cubin".to_owned());
        }
        // The recurrence stages a block's slice of a head's state in shared memory.
        let recur = Bound::shared(
            cuda,
            Image::Qwen4Exp,
            embedded_cubin(),
            "q4e_gdn_recur_columns",
            Precision::Bf16,
            (head_dim * head_dim / recur_parts(head_dim) * 4) as u32,
        )?;
        Ok(Self {
            embed: token("q4e_embed", Precision::Bf16)?,
            argmax: token("q4e_argmax_slices", Precision::Bf16)?,
            argmax_prob: token("q4e_argmax_prob_slices", Precision::Bf16)?,
            gemv_bf16: decode("q4e_gemv_tiled_bf16_parts", Precision::Bf16)?,
            gemv_packed: decode("q4e_gemv_tiled_packed_parts", Precision::Bf16)?,
            gemv_fp8: decode("q4e_gemv_tiled_e4m3_block", Precision::Fp8E4M3)?,
            quantize_e4m3: token("q4e_quantize_e4m3_rows", Precision::Bf16)?,
            gemv_e4m3: token("q4e_gemv_tiled_e4m3", Precision::Fp8E4M3)?,
            expert_fp8: decode("q4e_expert_gemv_e4m3_block", Precision::Fp8E4M3)?,
            mhc_in: decode("q4e_mhc_in", Precision::Bf16)?,
            mhc_post: decode("q4e_mhc_post", Precision::Bf16)?,
            sum_slots: token("q4e_sum_slots", Precision::Bf16)?,
            norm: Norm::new(
                token("q4e_rmsnorm", Precision::Bf16)?,
                Some(token("q4e_rmsnorm_rows", Precision::Bf16)?),
            )?,
            conv: TokenConv::new(token("q4e_gdn_conv", Precision::Bf16)?)?,
            kda_decay: token("q4e_kda_decay", Precision::Bf16)?,
            recur,
            chunk_conv: ChunkConv::new(
                scan("q4p_gdn_conv", Precision::Bf16)?,
                scan("q4p_gdn_conv_state", Precision::Bf16)?,
            )?,
            kda_prep: scan("q4p_kda_prep", Precision::Bf16)?,
            kda_scan: scan("q4p_kda_scan", Precision::Bf16)?,
            kda_gate: GatedNorm::new(scan("q4p_gdn_gate", Precision::Bf16)?)?,
            copy_flat: token("q4e_copy_flat", Precision::F32)?,
            layer_norm: LayerNorm::new(decode("q4e_layer_norm", Precision::Bf16)?)?,
            pool: decode("q4e_glm_pool", Precision::Bf16)?,
            index_scores: decode("q4e_glm_index_scores", Precision::Bf16)?,
            pick: decode("q4e_glm_pick", Precision::Bf16)?,
            mla_cache: decode("q4e_glm_mla_cache", Precision::Bf16)?,
            mla_cache_fp8: decode("q4e_glm_mla_cache_fp8", Precision::Fp8E4M3)?,
            mla_absorb: decode("q4e_glm_mla_absorb", Precision::Bf16)?,
            mla_scores: decode("q4e_glm_mla_scores", Precision::Bf16)?,
            mla_softmax: decode("q4e_glm_mla_softmax", Precision::Bf16)?,
            mla_mix: decode("q4e_glm_mla_mix", Precision::Bf16)?,
            mla_project: decode("q4e_glm_mla_project", Precision::Bf16)?,
            swiglu: decode("q4e_swiglu_limited", Precision::Bf16)?,
            router: decode("q4e_noaux_router", Precision::Bf16)?,
            combine: moe("q4e_moe_combine", Precision::Bf16)?,
            prefetch_l2: decode("q4e_prefetch_l2", Precision::F32)?,
            fold: token("q4e_fold_streams", Precision::F32)?,
            prefill_dense_bf16: prefill_wide("glm_prefill_bf16_wide", Precision::Bf16)?,
            prefill_dense_fp8: prefill("glm_prefill_fp8b128_mma", Precision::Fp8E4M3)?,
            prefill_dense_bf16_bf16a: prefill_wide("glm_prefill_bf16_wide_bf16a", Precision::Bf16)?,
            prefill_dense_fp8_bf16a: prefill("glm_prefill_fp8b128_mma_bf16a", Precision::Fp8E4M3)?,
            prefill_dense_fp8_rows: decode("glm_decode_fp8b128_rows", Precision::Fp8E4M3)?,
            prefill_moe_mma_bf16a: glm_moe("glm_moe_fp8b128_mma_bf16a", Precision::Fp8E4M3)?,
            prefill_moe_plan: plan("moe_plan", Precision::Bf16)?,
            prefill_moe_mma: glm_moe("glm_moe_fp8b128_mma", Precision::Fp8E4M3)?,
            prefill_moe_swiglu: glm_moe("glm_moe_swiglu", Precision::Bf16)?,
            // The dynamic activation quantiser stays in the prefill image: the dense q8 GEMM
            // reads it too (kernels/glm_prefill_sm121.cu).
            prefill_quantise: prefill("glm_quantise_rows", Precision::F32)?,
            prefill_moe_swiglu_q8: glm_moe("glm_moe_swiglu_q8", Precision::Bf16)?,
            prefill_dense_q8: prefill_q8("glm_prefill_q8_mma", Precision::Fp8E4M3)?,
            prefill_moe_q8: glm_moe_q8("glm_moe_q8_mma", Precision::Fp8E4M3)?,
            prefill_skinny: [
                prefill("glm_prefill_skinny_bf16_1", Precision::Bf16)?,
                prefill("glm_prefill_skinny_bf16_2", Precision::Bf16)?,
                prefill("glm_prefill_skinny_bf16_3", Precision::Bf16)?,
                prefill("glm_prefill_skinny_bf16_4", Precision::Bf16)?,
            ],
            prefill_exact: prefill("glm_prefill_exact_bf16", Precision::Bf16)?,
            prefill_mla_attend: attention("glm_prefill_mla_attend", Precision::Bf16)?,
            prefill_index_scores: attention("glm_prefill_index_scores", Precision::F32)?,
            prefill_mla_attend_bf16a: attention("glm_prefill_mla_attend_bf16a", Precision::Bf16)?,
            prefill_heads_bf16: attention("glm_prefill_heads_bf16_mma", Precision::Bf16)?,
            prefill_heads_fp8: attention("glm_prefill_heads_fp8b128_mma", Precision::Fp8E4M3)?,
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
    let mut key_t = Vec::with_capacity(if !linear.weight.is_fp8() {
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
        if !linear.weight.is_fp8() {
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
    let Some(scales) = linear.weight.scales() else {
        // BF16, widened exactly: its bits are an f32's top half.
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| refusal("kv_b's shape overflows a usize"))?;
        let mut bytes = vec![0u8; elements * 2];
        cuda.copy_dtoh(&mut bytes, &linear.weight.data())
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
    cuda.copy_dtoh(&mut weight, &linear.weight.data())
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
        check_length(key, len, &bytes).map_err(refusal)?;
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
    slots: SlotCursor<'a>,
    stream: &'a RunStream,
    memory: &'a mut Allocations,
}

impl Sink for Slab<'_> {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, BackendError> {
        let at = self.slots.take(key, len).map_err(refusal)?;
        Ok(self.base.offset(at))
    }

    fn derived(&mut self, bytes: &[u8]) -> Result<Deviceptr, BackendError> {
        self.memory.upload(self.stream, bytes)
    }
}

impl Drop for Allocations {
    fn drop(&mut self) {
        // A sequence can be returned to its pool from a worker thread that has not touched the
        // device yet, so bind the context before the synchronise and the frees.
        let _ = self.cuda.make_current();
        // No queued launch may outlive the buffers it reads.
        let _ = self.cuda.synchronize();
        let buffers: Vec<&Deviceptr> = self.buffers.iter().collect();
        self.cuda.free_buffers(&buffers);
        for mapped in &self.mapped {
            self.cuda.free_host_mapped(mapped);
        }
    }
}

/// One uploaded linear's values: the typed matrix its kernel reads, BF16 or E4M3 with the
/// inverse-scale plane the format requires.
#[derive(Debug, Clone)]
enum LinearWeight {
    Bf16(Mat<Bf16>),
    Fp8(Mat<Fp8E4m3>),
}

impl LinearWeight {
    /// The values' device address.
    fn data(&self) -> Deviceptr {
        match self {
            Self::Bf16(mat) => mat.data(),
            Self::Fp8(mat) => mat.data(),
        }
    }

    /// The inverse-scale plane, absent for BF16.
    fn scales(&self) -> Option<Deviceptr> {
        match self {
            Self::Bf16(_) => None,
            Self::Fp8(mat) => mat.scales().blocks(),
        }
    }

    /// Whether the weights are E4M3 codes over a block-scale plane.
    fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8(_))
    }

    /// The block-scaled FP8 matrix, when the weight is one: the typed codes and the f32 128 x 128
    /// scale plane the tile's few-row body reads. A BF16 part has none. The served route is the
    /// tiled GEMV; the body stays catalogued and this accessor with it.
    #[allow(dead_code)]
    fn fp8(&self) -> Option<&Mat<Fp8E4m3>> {
        match self {
            Self::Bf16(_) => None,
            Self::Fp8(matrix) => Some(matrix),
        }
    }

    /// One value's stored bytes: one for E4M3, two for BF16.
    fn element(&self) -> usize {
        if self.is_fp8() { 1 } else { 2 }
    }
}

/// One uploaded linear, `rows x columns`: BF16, or E4M3 with its f32 tile scales.
#[derive(Debug, Clone)]
struct DeviceLinear {
    weight: LinearWeight,
    rows: usize,
    columns: usize,
    /// Where parts two to four of a fused projection's rows start, every one at `rows` for a
    /// single part: the GEMV writes each part's outputs for all its rows together.
    parts: [usize; 3],
}

impl DeviceLinear {
    /// A sub-view of an uploaded projection: a part of its rows, typed from the same format.
    fn view(
        data: Deviceptr,
        rows: usize,
        columns: usize,
        fp8: bool,
        scales: Option<Deviceptr>,
    ) -> Self {
        let spec = TensorSpec {
            name: "view".to_owned(),
            dtype: if fp8 { DType::F8E4M3 } else { DType::BF16 },
            shape: vec![rows, columns],
        };
        let weight = if fp8 {
            let entries = rows.div_ceil(FP8_BLOCK) * columns.div_ceil(FP8_BLOCK);
            let scales = scales.expect("an FP8 view carries its scale plane");
            LinearWeight::Fp8(
                upload_format::<Fp8E4m3>(Tensor {
                    stored: &spec,
                    shape: &[rows, columns],
                    data,
                    stride: columns,
                    scales: Scales::fp8_e4m3(scales, entries, DType::F32),
                })
                .expect("an FP8 view of an uploaded projection"),
            )
        } else {
            LinearWeight::Bf16(
                upload_format::<Bf16>(Tensor {
                    stored: &spec,
                    shape: &[rows, columns],
                    data,
                    stride: columns * 2,
                    scales: Scales::none(),
                })
                .expect("a BF16 view of an uploaded projection"),
            )
        };
        Self {
            weight,
            rows,
            columns,
            parts: [rows; 3],
        }
    }

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

    /// The code planes the tile's few-row FP8 body reads, when the weight is a block-scaled FP8
    /// matrix: the E4M3 codes and their f32 128 x 128 scale plane, typed by the upload. A BF16
    /// part has none, so no route can choose the FP8 body over it. The body is catalogued; the
    /// served route is the tiled GEMV.
    #[allow(dead_code)]
    fn fp8_planes(&self) -> Option<CodePlanes<'_, Fp8E4m3>> {
        CodePlanes::of(self.weight.fp8()?)
    }

    /// The weight's bytes, one a value for E4M3 and two for BF16, as a [DeviceRank::prefetch]
    /// span.
    fn span(&self) -> (Deviceptr, usize) {
        let size = if self.weight.is_fp8() { 1 } else { 2 };
        (self.weight.data(), self.rows * self.columns * size)
    }
}

/// The FP8 tile body's row rule: one to [MAX_ROWS] token rows take it, and a prefill-sized
/// batch keeps the bulk GEMM. The body stages the same eight rows the tiled GEMV's largest
/// launch does, and its own `SPM_ROWS_MAX` is eight.
#[cfg(test)]
fn fp8_token_body_rows(rows: usize) -> bool {
    (1..=MAX_ROWS).contains(&rows)
}

/// Whether the FP8 tile body takes this projection at `rows` rows: the row rule, a K run of
/// whole 128-column blocks, and parts of whole eight-column output writes whose scales start on a
/// 128-row block.
///
/// The weight is the operand, not a shape: only [CodePlanes]<[Fp8E4m3]> reaches this rule, which
/// only an E4M3 weight with its block-scale plane produces ([DeviceLinear::fp8_planes]). A shape
/// it does not take, and a BF16 part, keep the tiled GEMV.
#[cfg(test)]
fn fp8_token_body_fits(
    planes: CodePlanes<'_, Fp8E4m3>,
    linear: &DeviceLinear,
    rows: usize,
) -> bool {
    if !fp8_token_body_rows(rows)
        || !linear.columns.is_multiple_of(FP8_BLOCK)
        || planes.matrix().columns() != linear.columns
    {
        return false;
    }
    let starts = [
        0,
        linear.parts[0],
        linear.parts[1],
        linear.parts[2],
        linear.rows,
    ];
    (0..4).all(|part| {
        let (first, end) = (starts[part], starts[part + 1]);
        end <= first || ((end - first).is_multiple_of(8) && first.is_multiple_of(FP8_BLOCK))
    })
}

/// The FP8 tile's few-row body over each part of `linear`: one launch a part, over the part's
/// weight rows, its 128-column scale block and its outputs for all `rows` rows. The body's
/// arguments are the prompt tile's (`w, ws, a, out, cols, inner, tokens`) and its activations
/// are the f32 rows the GEMV read, so nothing is converted between the two.
#[allow(clippy::too_many_arguments)]
fn launch_fp8_token_body(
    cuda: &Cuda,
    stream: &RunStream,
    kernel: Kernel,
    planes: CodePlanes<'_, Fp8E4m3>,
    linear: &DeviceLinear,
    input: Deviceptr,
    output: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    let columns = linear.columns;
    let scales = planes.blocks();
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
        let weight = planes.codes().offset(first * columns);
        let part_scales =
            scales.offset((first / FP8_BLOCK) * (columns / FP8_BLOCK) * std::mem::size_of::<f32>());
        let out = output.offset(rows * first * 4);
        cuda.launch_on(
            stream,
            kernel,
            (part_rows.div_ceil(32) as u32, 1, 1),
            (128, 1, 1),
            &[
                weight.0,
                part_scales.0,
                input.0,
                out.0,
                part_rows as u64,
                columns as u64,
                rows as u64,
            ],
        )
        .map_err(|error| cuda_error(format!("{}: {error}", kernel.name())))?;
    }
    Ok(())
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

// A sparse layer carries the indexer and the absorbed operands; boxing it would only move
// the allocation.
#[allow(clippy::large_enum_variant)]
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
    /// Every uploaded part's format ledger row, in upload order.
    ledger: Vec<LedgerRow>,
    /// The exact widenings this walk recorded, for the boot ledger.
    widen: WidenLog,
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
            ledger: Vec::new(),
            widen: WidenLog::new(),
        }
    }

    /// One part's ledger row: it was read at the format it is stored as.
    fn record(&mut self, name: &str, stored: FormatKind, arithmetic: FormatKind, bytes: u128) {
        self.ledger
            .push(LedgerRow::new(name, stored, arithmetic, bytes));
    }

    /// The boot ledger: every uploaded part, and the exact widenings the walk took.
    fn format_ledger(&self) -> Result<FormatLedger, BackendError> {
        FormatLedger::boot(self.ledger.clone(), &self.widen)
            .map_err(|error| refusal(error.to_string()))
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
        let key = names.join("+");
        self.record(&key, FormatKind::Bf16, FormatKind::Bf16, len as u128);
        self.sink.weight(&key, len, bytes)
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
        if dtype == DType::BF16 {
            // A released checkpoint stores these vectors BF16 and the kernel reads f32.
            widen::<Bf16, F32Format>(
                name,
                "the kernel's argument is f32 and the checkpoint stores the vector BF16; the                  widening is exact",
                &mut self.widen,
            )
            .map_err(|error| refusal(error.to_string()))?;
            self.record(name, FormatKind::Bf16, FormatKind::F32, len as u128);
        } else {
            self.record(name, FormatKind::F32, FormatKind::F32, len as u128);
        }
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
        let spec = TensorSpec {
            name: key.clone(),
            dtype: format,
            shape: vec![rows, columns],
        };
        // The typed upload checks the container's dtype and the scale plane's entry count; the
        // ledger row takes the matrix's own bytes a token.
        let typed = if fp8 {
            let entries = rows.div_ceil(FP8_BLOCK) * columns.div_ceil(FP8_BLOCK);
            let scales = scales.ok_or_else(|| refusal("an FP8 projection without its scales"))?;
            let mat = upload_format::<Fp8E4m3>(Tensor {
                stored: &spec,
                shape: &[rows, columns],
                data: weight,
                stride: columns,
                scales: Scales::fp8_e4m3(scales, entries, DType::F32),
            })
            .map_err(|error| refusal(error.to_string()))?;
            self.record(
                &key,
                FormatKind::Fp8E4m3,
                FormatKind::Fp8E4m3,
                mat.bytes_per_token(),
            );
            LinearWeight::Fp8(mat)
        } else {
            let mat = upload_format::<Bf16>(Tensor {
                stored: &spec,
                shape: &[rows, columns],
                data: weight,
                stride: columns * 2,
                scales: Scales::none(),
            })
            .map_err(|error| refusal(error.to_string()))?;
            self.record(
                &key,
                FormatKind::Bf16,
                FormatKind::Bf16,
                mat.bytes_per_token(),
            );
            LinearWeight::Bf16(mat)
        };
        Ok(DeviceLinear {
            weight: typed,
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
                let Some(scales) = linear.weight.scales() else {
                    return Err(refusal(format!(
                        "{prefix}.experts.{expert} is BF16; the routed expert GEMV reads FP8"
                    )));
                };
                table.extend(linear.weight.data().0.to_le_bytes());
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
        // The shared walk decides the branch and the FFN; the typed lists are not read here.
        let steps = config.steps();
        let step = steps.get(layer).ok_or_else(|| {
            refusal(format!(
                "layer {layer} is past the {} the shared architecture walks",
                steps.len()
            ))
        })?;
        Ok(DeviceLayer {
            input_norm: self.bf16(&[format!("{prefix}.input_layernorm.weight")], &[hidden])?,
            post_norm: self.bf16(
                &[format!("{prefix}.post_attention_layernorm.weight")],
                &[hidden],
            )?,
            attn_hc: self.hyper(&format!("{prefix}.hc_attn"))?,
            ffn_hc: self.hyper(&format!("{prefix}.hc_ffn"))?,
            branch: match step.attention {
                Part::KdaScan => DeviceBranch::Linear(self.kda(&format!("{prefix}.self_attn"))?),
                Part::MlaAttention => DeviceBranch::Sparse(self.dsa(
                    &format!("{prefix}.self_attn"),
                    step.indexer == Some(Part::SparseIndexer),
                )?),
                other => {
                    return Err(refusal(format!(
                        "layer {layer}'s attention is {other}, and this arm uploads KDA or MLA"
                    )));
                }
            },
            ffn: match step.ffn {
                Part::DenseMlp => {
                    DeviceFfn::Dense(self.mlp(&format!("{prefix}.mlp"), config.intermediate_size)?)
                }
                Part::Moe => DeviceFfn::Moe(self.moe(&format!("{prefix}.mlp"))?),
                other => {
                    return Err(refusal(format!(
                        "layer {layer}'s FFN is {other}, and this arm uploads a dense MLP or MoE"
                    )));
                }
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
    /// `[MAX_ROWS][MAX_ROWS]` f32 beside [Self::drafted]: entry `[i + 1][row]` the probability
    /// the MTP layer gave its draft `i + 1`, which the depth policy reads as its confidence.
    draft_probs: HostMapped,
    /// A step's token ids, a u32 a row, which its one embedding launch reads in place
    /// ([DeviceRank::embed_rows]).
    step_tokens: HostMapped,
    /// The MTP layer's token ids of a step's drafting runs ([DeviceRank::fill]), as
    /// [Self::step_tokens].
    mtp_tokens: HostMapped,
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
    /// The shared plan's rank-ordered slot list ([moe_plan]): each item reads its rows from
    /// here, at the item's first rank.
    moe_sorted: Deviceptr,
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
    /// The sparse layers' caches, backed as the sequence grows.
    lane: SequenceLane,
    /// The rewind point: each KDA layer's convolution history and recurrent memory, and the MTP
    /// layer's hidden rows, as they were at the end of the last prompt ([DeviceRank::keep_point]),
    /// when the rank keeps one ([sparkflash_runtime::REWIND_ENV]). The sparse caches need no copy:
    /// a position's rows stay until a later write at that position, and an index pool is
    /// rewritten from the cached keys when a row completes it.
    point: Option<Checkpoint>,
    /// The MTP layer's [SequenceDrafter::last] at the point.
    point_last: Option<usize>,
}

/// Rows past a step's last position a sequence's lane backs too: the MTP layer's drafts write
/// up to a step's rows past it, and a kernel that reads a whole tile past the position never
/// reaches memory nothing backs.
const LANE_READ_PAST: usize = 1024;

/// Where each of a sequence's caches that grow with its context lies in its lane
/// ([SequenceLanes], the lane memory Inkling's and Qwen's sequences use too): every DSA layer's
/// latent, scales and indexer rows, and the MTP layer's in a drafting sequence. A sequence
/// reserves a span as long as its whole capacity's caches and physical memory backs it only as
/// far as the sequence has reached ([DeviceRank::back]), from one pool the rank's sequences share.
///
/// The KDA layers' state and the drafters' scratch stay plain allocations: they do not grow.
struct LaneParts {
    /// Each layer's caches, `None` for a KDA layer.
    layers: Vec<Option<SparseParts>>,
    /// The MTP layer's, in a drafting sequence.
    mtp: Option<SparseParts>,
}

/// Which sequences share one [SequenceLanes]: those that draft with the MTP layer or not, of one
/// capacity.
type LaneKey = (bool, usize);

/// One DSA layer's caches in a lane, as offsets from the span's base.
#[derive(Debug, Clone, Copy)]
struct SparseParts {
    latent: usize,
    latent_scales: Option<usize>,
    /// The indexer's keys, gates and pooled keys, for a layer with an indexer.
    index: Option<[usize; 3]>,
}

impl SparseParts {
    /// A DSA layer's caches for `capacity` positions, the indexer's only for a layer with one.
    fn plan(
        layout: &mut Layout,
        config: &Glm5NextConfig,
        dsa: &DeviceDsa,
        capacity: usize,
    ) -> Self {
        // The latent is the release's E4M3 activation when the projections quantise theirs, and
        // f32 for a BF16 release.
        let (elements, scale_floats) = latent_row(config, dsa.quantise);
        let latent_row = if dsa.quantise { elements } else { elements * 4 };
        let latent = layout.part(capacity * latent_row, latent_row, 0);
        let latent_scales = dsa
            .quantise
            .then(|| layout.part(capacity * scale_floats * 4, scale_floats * 4, 0));
        let index = dsa.indexer.is_some().then(|| {
            let row = config.index_head_dim * 4;
            let kpool = config.index_kpool.max(1);
            [
                layout.part(capacity * row, row, 0),
                layout.part(capacity * row, row, 0),
                // A pooled row covers `kpool` positions; a position adds its share, rounded up.
                layout.part(capacity / kpool * row, row.div_ceil(kpool), 0),
            ]
        });
        Self {
            latent,
            latent_scales,
            index,
        }
    }

    /// The layer's state in the lane at `base`.
    fn layer(self, base: Deviceptr) -> SequenceLayer {
        let index = |part: usize| {
            self.index
                .map_or(Deviceptr(0), |index| base.offset(index[part]))
        };
        SequenceLayer::Sparse {
            latent: base.offset(self.latent),
            latent_scales: self
                .latent_scales
                .map_or(Deviceptr(0), |offset| base.offset(offset)),
            index_keys: index(0),
            index_gates: index(1),
            pooled: index(2),
        }
    }
}

/// One sequence's span of its [SequenceLanes], freed with the sequence.
struct SequenceLane {
    cuda: Arc<CudaHandle>,
    lane: lane_memory::SequenceLane<LaneParts>,
}

impl SequenceLane {
    /// Gives every chunk back to the pool, once the device is done with them.
    fn release(&mut self) -> Result<(), BackendError> {
        self.lane.release(&self.cuda).map_err(cuda_error)
    }
}

impl Drop for SequenceLane {
    fn drop(&mut self) {
        // A sequence can be returned to its pool from a worker thread that has not touched the
        // device yet, as [Allocations] does: free makes the context current.
        self.lane.free(&self.cuda);
    }
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
pub(crate) enum HeadRows {
    All,
    Last,
}

/// What a step hands back for each of its rows.
pub(crate) trait StepRow: Sized {
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
pub(crate) struct LogitsAt {
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
pub(crate) struct Run<'s> {
    layers: &'s [SequenceLayer],
    position: usize,
    first: usize,
    len: usize,
    verify: bool,
    drafter: Option<&'s SequenceDrafter>,
    /// A block-drafting sequence's drafter scratch and the first of the run's rows it captures,
    /// into its first feature row: 0 for a step's rows, and for a prompt's chunk the first row
    /// the drafter's next block reads ([DeviceRank::forward_prefill]).
    block: Option<(DraftScratch, usize)>,
}

impl<'s> Run<'s> {
    /// `sequence`'s `len` rows from its position as rows `first..first + len` of a token step:
    /// a verify's run when `verify` and it has more than one row, its block drafter capturing
    /// every row.
    pub(crate) fn token(
        sequence: &'s DeviceSequence,
        first: usize,
        len: usize,
        verify: bool,
    ) -> Self {
        Self {
            layers: &sequence.layers,
            position: sequence.position,
            first,
            len,
            verify: verify && len > 1,
            drafter: sequence.drafter.as_ref(),
            block: sequence.block.map(|scratch| (scratch, 0)),
        }
    }

    /// `sequence`'s `len` rows from its position as a bulk chunk's one run, capturing from the
    /// row `plan` says.
    pub(crate) fn chunk(sequence: &'s DeviceSequence, len: usize, plan: &ChunkPlan) -> Self {
        Self {
            layers: &sequence.layers,
            position: sequence.position,
            first: 0,
            len,
            verify: false,
            drafter: sequence.drafter.as_ref(),
            block: plan.scratch.zip(plan.captured),
        }
    }
}

/// What a bulk chunk's block drafter reads: its prompt-row scratch, the window rows the chunk
/// commits to the drafter's context and the first row it captures ([DeviceRank::chunk_plan]).
pub(crate) struct ChunkPlan {
    scratch: Option<DraftScratch>,
    commits: Vec<(usize, usize, usize)>,
    captured: Option<usize>,
}

/// What every stage of one step reads ([DeviceRank::step_set]): whether the collectives take
/// the device leg, the step's rows, this rank's rows of a sequence-parallel step, and the block
/// drafter's captures (its scratch, the first row and the rows).
#[derive(Clone)]
pub(crate) struct StepSet {
    device_leg: bool,
    rows: usize,
    share: Option<(usize, usize, usize)>,
    captures: Vec<(DraftScratch, usize, usize)>,
}

impl StepSet {
    /// This rank's rows of the step, over `tp`, when the step is sequence-parallel.
    fn share<'t>(&self, tp: &'t TpRuntime) -> Option<Share<'t>> {
        self.share.map(|(first, count, live)| Share {
            tp,
            first,
            count,
            live,
        })
    }
}

/// This rank's rows of a sequence-parallel step ([DeviceRank::sequence_parallel]).
#[derive(Clone, Copy)]
struct Share<'t> {
    tp: &'t TpRuntime,
    /// The first of this rank's rows: its rank times `count`.
    first: usize,
    /// Rows a rank: the step's over the world, rounded up. The collectives move this many; the
    /// last rank's past the step's are padding no kernel reads.
    count: usize,
    /// This rank's rows of the step's: `count`, but fewer on the last rank when padded.
    live: usize,
}

impl Share<'_> {
    /// Every rank's rows of `width` floats gathered into `base`, from `mine`, this rank's rows of
    /// it, in place.
    fn gather(
        self,
        rank: &DeviceRank,
        mine: Deviceptr,
        base: Deviceptr,
        width: usize,
    ) -> Result<(), BackendError> {
        self.tp
            .exchange_on_device(rank.stream.raw(), mine.0, base.0, self.count * width * 4)
            .map_err(weights_error)
    }
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

    /// Whether the sequence holds a verify [DeviceRank::commit] has not settled.
    #[must_use]
    pub(crate) fn verifying(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether the rank's pool can back this sequence for `rows` positions now, counting the
    /// chunks a reset gives back first when `reset`: the answer [DeviceRank::back] would give.
    #[must_use]
    pub fn room(&self, rows: usize, reset: bool) -> bool {
        self.lane.lane.room(rows, reset)
    }

    /// The position the sequence can go back to ([DeviceRank::rewind]): the end of its last
    /// prompt, while the copies hold it.
    #[must_use]
    pub fn rewind_point(&self) -> Option<usize> {
        self.point.as_ref().and_then(Checkpoint::point)
    }

    /// The copies no longer hold a point: a prompt began to forward past it and will keep its own
    /// end.
    pub fn forget_point(&mut self) {
        if let Some(point) = self.point.as_mut() {
            point.forget();
        }
    }

    /// The live regions the rewind point copies, in its order.
    fn point_regions(&self) -> Vec<Deviceptr> {
        let mut live = Vec::new();
        for layer in &self.layers {
            if let SequenceLayer::Linear {
                conv, recurrent, ..
            } = layer
            {
                live.extend([*conv, *recurrent]);
            }
        }
        live.extend(self.drafter.as_ref().map(|drafter| drafter.hidden));
        live.extend(self.block.as_ref().map(DraftScratch::rings));
        live
    }

    /// The drafter's confidence in each of the sequence's last drafts: the block drafter's
    /// selector's probability for its pick among the row's candidates, in `(0, 1]`, or the MTP
    /// layer's softmax probability of its argmax (0 for a row whose logits have no finite best).
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
    /// `[rows][feature_width]` A bulk prefill chunk's captured target rows, on a rank with the
    /// chunk: at most the drafter's `window_left` of a prompt's last rows, the only ones its first
    /// block reads, or a chunk's width if that is less ([DeviceRank::forward_prefill]).
    prompt: Option<Deviceptr>,
}

/// One tensor-parallel rank of GLM-5.3-Flash on one GPU, for decode up to [MAX_ROWS] rows a step.
pub struct DeviceRank {
    cuda: Arc<CudaHandle>,
    stream: RunStream,
    /// Per-stage timings, on only under `SPARKFLASH_GLM_PROF` (`diagnostics.profile`).
    prof: Mutex<sparkflash_model::StageProfiler>,
    kernels: Kernels,
    config: Glm5NextConfig,
    layout: TpLayout,
    /// The bulk KDA scan the rank's plan resolves ([Self::kda_scan_plan]); `None` where runs
    /// take the token kernels.
    kda_scan: Option<ScanPlan>,
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
    /// The sequences' [SequenceLanes], one a kind, each planned on first use.
    lanes: LanePlans<LaneKey, LaneParts>,
    scratch: Scratch,
    /// Whether a block-drafting sequence's prompt runs through the bulk chunk
    /// (`prefill.drafting_prompt`, on by default) rather than the token path.
    drafting_prefill: bool,
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
    /// Whether a bulk step's hyper-connections run on each rank's own rows
    /// (`sequence.parallel`, [Self::sequence_parallel]).
    sequence_parallel: bool,
    /// What the hyper-connection fold multiplies its summed streams by
    /// (`speculation.mtp_fold_scale`): the mean over them unless the profile names a scale.
    fold_scale: f32,
    /// Steps this rank has run: a verify's commit has to come before the next one, which
    /// overwrites the inputs it replays.
    steps: u64,
    /// LM heads this rank has run, a step's or a draft's: each overwrites the last one's logits.
    heads: AtomicU64,
    /// The numerics switches, read once when the rank loads (docs/NUMERICS.md section 5).
    numerics: numerics::Switches,
    /// This step's recorders; `None` between steps and with both switches off.
    record: RefCell<Option<numerics::Forward>>,
    /// Whether `SPARKFLASH_NUMERICS_COVERAGE` names a path, checked a layer without a borrow.
    covering: bool,
    /// Whether `SPARKFLASH_NUMERICS_MONITORS` names a path, checked a layer without a borrow.
    monitoring: bool,
    /// The side stream [Self::prefetch] warms the L2 from, unless SPARKFLASH_L2_PREFETCH=0.
    prefetch: Option<Prefetch>,
    /// Recorded after each embedding launch that reads a token slab ([Self::embed_rows]).
    tokens_staged: TokenFence,
    /// Whether a sequence keeps a rewind point at the end of each prompt
    /// ([sparkflash_runtime::REWIND_ENV], on by default).
    rewind: bool,
    /// Packed copies of the BF16 projections the tiled GEMV reads ([Self::pack_linears]), beside
    /// the BF16 rows the bulk prefill and the drafts' E4M3 head still read.
    packed: PackedSet<Tiled>,
    memory: Allocations,
}

/// The event a rank records after each launch that reads one of its token slabs. Launches are
/// queued, so the host reaches the next step's embedding (or a prefill's next chunk) while the
/// last one may still be queued; a slab is rewritten only once every launch that read it has run.
struct TokenFence(StreamEvent);

// SAFETY: an event handle of the rank's own context, recorded on the rank's stream and waited on
// by the thread that holds the rank. A `Send` rank moves the event with it; nothing shares it.
unsafe impl Send for TokenFence {}

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

/// Asked after each MTP pass with the probability of every row's newest draft: true drafts
/// another pass. [DeviceRank::draft_while] takes it.
pub type KeepDrafting<'a> = &'a mut dyn FnMut(&[f32]) -> bool;

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
            let mut uploader = Uploader::new(source, config, layout, &mut sink);
            let weights = uploader.weights()?;
            eprintln!("[glm] rank {}: {}", layout.rank, uploader.format_ledger()?);
            Ok(weights)
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
        let (preamble, index) = ConvertedIndex::read(path, layout).map_err(refusal)?;
        let specs = ConvertedSpecs::new(&index).map_err(refusal)?;
        let data = usize::try_from(preamble.padded_data_bytes)
            .map_err(|_| refusal("the rank file's data does not fit this machine's range"))?;
        Self::build(cuda, config, layout, cache, |stream, memory| {
            let base = memory.file(path, preamble.data_offset, data)?;
            let mut sink = Slab {
                base,
                slots: SlotCursor::new(&index.slots),
                stream,
                memory,
            };
            let mut uploader = Uploader::new(&specs, config, layout, &mut sink);
            let weights = uploader.weights()?;
            eprintln!("[glm] rank {}: {}", layout.rank, uploader.format_ledger()?);
            sink.slots.finish().map_err(refusal)?;
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
        let quantise = |dsa: &DeviceDsa| cache.quantises(config, dsa.kv_b.weight.is_fp8());
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
        let prefill_rows = Self::prefill_rows_from_profile().filter(|_| chunk_shapes);
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
        let tokens_staged = TokenFence(cuda.stream_event().map_err(cuda_error)?);
        let kda_scan = Self::kda_scan_plan(config, layout)?;
        // The numerics switches are read once, when the rank loads: a forward then pays one
        // `Option` check and a layer one, and off does no formatting or allocation. The coverage
        // recorder is seeded with the model's per-layer routed-expert counts: a dense layer is
        // listed with none, as the committed layout is.
        let experts = config
            .mlp_layer_types
            .iter()
            .map(|kind| match kind {
                MlpKind::Moe => config.n_routed_experts,
                MlpKind::Dense => 0,
            })
            .collect();
        let numerics = numerics::Switches::from_env(crate::MODEL_ID, experts, config.hidden_size);
        let covering = numerics.coverage_enabled();
        let monitoring = numerics.monitors_enabled();
        let mut rank = Self {
            cuda,
            stream,
            kernels,
            config: config.clone(),
            layout,
            kda_scan,
            geometry: HcGeometry::from(config),
            embed,
            norm,
            lm_head,
            draft_head: None,
            layers,
            mtp,
            block: None,
            lanes: LanePlans::default(),
            scratch,
            prefill,
            drafting_prefill: true,
            prefill_rows: prefill_rows.unwrap_or(MAX_ROWS),
            exact_prefill: sparkflash_model::active_profile()
                .prefill
                .exact
                .unwrap_or(true),
            prefill_activation: PrefillActivation::from_profile(
                config,
                sparkflash_model::active_profile(),
            )?,
            two_shot_rows: sparkflash_weights::tp::two_shot_rows().map_err(weights_error)?,
            sequence_parallel: sparkflash_model::active_profile()
                .sequence
                .parallel
                .unwrap_or(true),
            fold_scale: sparkflash_model::active_profile()
                .speculation
                .mtp_fold_scale
                .unwrap_or(1.0 / config.hc_mult as f32),
            steps: 0,
            heads: AtomicU64::new(0),
            numerics,
            record: RefCell::new(None),
            covering,
            monitoring,
            prefetch,
            prof: Mutex::new(sparkflash_model::StageProfiler::new(
                Self::profile_enabled()?
            )),
            tokens_staged,
            rewind: sparkflash_runtime::rewind().map_err(refusal)?,
            packed: PackedSet::new(Form::from_env().map_err(refusal)?, Placement::Beside).tiled(),
            memory,
        };
        if rank.mtp.is_some() && !rank.lm_head.weight.is_fp8() {
            rank.draft_head()?;
        }
        rank.pack_linears()?;
        Ok(rank)
    }

    /// Packs the large BF16 projections every step reads through the tiled GEMV: of each KDA
    /// layer's fused projection and output, the MLA and indexer projections stored as BF16, the
    /// router and the LM head, those of more than [PACK_MIN_WEIGHTS] weights (on the released
    /// model, the KDA fused projections and the head). A matrix only another kernel reads (the
    /// mixers' projections, which q4e_mhc_post folds in; kv_b, which the attention absorbs) stays
    /// BF16 alone. Every packed matrix is unpacked and compared with its BF16 rows here, and one
    /// differing word refuses the load.
    fn pack_linears(&mut self) -> Result<(), BackendError> {
        let mut matrices = Vec::new();
        let mut add = |linear: &DeviceLinear| {
            if !linear.weight.is_fp8() && linear.rows * linear.columns > PACK_MIN_WEIGHTS {
                matrices.push((linear.weight.data(), linear.rows, linear.columns));
            }
        };
        for layer in &self.layers {
            match &layer.branch {
                DeviceBranch::Linear(kda) => [&kda.fused, &kda.o].into_iter().for_each(&mut add),
                DeviceBranch::Sparse(dsa) => {
                    [&dsa.fused, &dsa.q_b, &dsa.o]
                        .into_iter()
                        .for_each(&mut add);
                    if let Some(indexer) = &dsa.indexer {
                        [&indexer.fused, &indexer.wq_b]
                            .into_iter()
                            .for_each(&mut add);
                    }
                }
            }
            if let DeviceFfn::Moe(moe) = &layer.ffn {
                add(&moe.router);
            }
        }
        add(&self.lm_head);
        if matrices.is_empty() || Form::from_env().map_err(refusal)? == Form::Checkpoint {
            eprintln!("[glm] {}", self.packed.summary());
            return Ok(());
        }
        let kernels = packed::Kernels::load(&self.cuda).map_err(refusal)?;
        // The packing runs on the default stream, which this rank's uploads are not ordered
        // against.
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let (_, packed_bytes) = self
            .packed
            .pack(&self.cuda, &kernels, &matrices, 0)
            .map_err(refusal)?;
        self.memory.bytes += packed_bytes;
        self.memory.buffers.extend(self.packed.allocations());
        eprintln!("[glm] rank {}: {}", self.layout.rank, self.packed.summary());
        Ok(())
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
        if self.lm_head.weight.is_fp8() {
            return Err(refusal("drafts score with an E4M3 copy of a BF16 LM head"));
        }
        // The quantization runs on the default stream, which this rank's uploads are not ordered
        // against.
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let matrix = GpuMat {
            pointer: self.lm_head.weight.data(),
            rows: self.lm_head.rows,
            columns: self.lm_head.columns,
        };
        let memory = &mut self.memory;
        let head = GpuMatE4m3::quantize(
            &self.cuda,
            self.kernels.quantize_e4m3.kernel(),
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

    /// Whether this rank profiles its stages, from the typed `diagnostics.profile` field.
    ///
    /// `SPARKFLASH_GLM_PROF` is the declared alias ([sparkflash_model::profile::ENV_OVERRIDES]),
    /// so the switch is a profile field rather than a variable read at the point of use.
    fn profile_enabled() -> Result<bool, BackendError> {
        let mut profile = sparkflash_model::ModelProfile::default();
        profile
            .apply_process_env()
            .map_err(|error| refusal(format!("the profile: {error}")))?;
        Ok(profile.diagnostics.profile == Some(true))
    }

    /// Starts one forward's stage accounting ([sparkflash_model::StageProfiler]).
    pub(crate) fn prof_begin(&self) {
        if let Ok(mut prof) = self.prof.lock() {
            prof.start();
        }
    }

    /// Charges the time since the last mark to `name`, if profiling is on.
    ///
    /// The stream is drained first: that drain is what makes the charge real, and it is also why a
    /// profiled step is slower than an unprofiled one.
    fn prof_mark(&self, name: &'static str) {
        let Ok(mut prof) = self.prof.lock() else {
            return;
        };
        if !prof.is_enabled() {
            return;
        }
        let cuda = &self.cuda;
        if cuda.synchronize_stream(&self.stream).is_err() {
            return;
        }
        prof.mark(name);
    }

    /// Prints the forward's report, when profiling is on.
    pub(crate) fn prof_end(&self, rows: usize, total_ms: f64) {
        if let Ok(mut prof) = self.prof.lock()
            && let Some(text) = prof.report(rows, total_ms)
        {
            eprint!("{text}");
        }
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

    /// Each layer's token mixer and feed-forward, as the parts the rank built them as: a KDA
    /// recurrence or a DSA attention, a dense MLP or routed experts.
    pub(crate) fn layer_parts(&self) -> Vec<(Part, Part)> {
        self.layers
            .iter()
            .map(|layer| {
                let mixer = match layer.branch {
                    DeviceBranch::Linear(_) => Part::KdaScan,
                    DeviceBranch::Sparse(_) => Part::MlaAttention,
                };
                let ffn = match layer.ffn {
                    DeviceFfn::Dense(_) => Part::DenseMlp,
                    DeviceFfn::Moe(_) => Part::Moe,
                };
                (mixer, ffn)
            })
            .collect()
    }

    /// The rows one bulk prefill chunk may hold: the active profile's `prefill.chunk`, or
    /// [MAX_ROWS] when this rank takes the token path for every prompt.
    ///
    /// This is the rank's own scratch width ([Self::prefill_rows]), so a serving walk that sizes
    /// its chunks from it cannot hand a chunk to scratch that was not allocated for one.
    #[must_use]
    pub fn prefill_rows(&self) -> usize {
        self.prefill_rows
    }

    /// The cache plan this rank's sequences allocate: one positional row a position a sparse
    /// layer (the release's E4M3 latent and its scales, or the f32 latent, plus the indexer's
    /// own rows) and the linear-attention state a session.
    ///
    /// This is [Glm5NextConfig::architecture]'s plan at this rank's world size and the latent
    /// representation the layers were built with, so a serving walk's [crate::walk::GlmWalk]
    /// layout describes the caches the rank actually holds rather than a compiled-in guess.
    ///
    /// # Errors
    /// A configuration the shared plan cannot model (its own error says which), which a serving
    /// walk must not turn into an empty layout.
    pub fn kv_plan(&self) -> Result<sparkflash_model::KvPlan, sparkflash_model::ConfigError> {
        self.config
            .architecture
            .kv_plan(self.layout.world_size, self.latent_fp8())
    }

    /// Whether this rank's sparse layers store their latent cache as E4M3 codes.
    #[must_use]
    pub fn latent_fp8(&self) -> bool {
        self.layers
            .iter()
            .find_map(|layer| match &layer.branch {
                DeviceBranch::Sparse(dsa) => Some(dsa.quantise),
                DeviceBranch::Linear(_) => None,
            })
            .unwrap_or(false)
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

    /// The bulk prefill chunk the active profile asks for, or the default; `None` when
    /// `prefill.enabled = false` keeps the token path.
    fn prefill_rows_from_profile() -> Option<usize> {
        let profile = sparkflash_model::active_profile();
        if profile.prefill.enabled == Some(false) {
            return None;
        }
        let rows = profile.prefill.chunk.unwrap_or(DEFAULT_PREFILL_ROWS);
        if rows < MIN_PREFILL_ROWS {
            return None;
        }
        Some(rows.min(MAX_PREFILL_ROWS))
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
            draft_probs: memory.mapped(rows * rows * 4)?,
            step_tokens: memory.mapped(rows * 4)?,
            mtp_tokens: memory.mapped(rows * 4)?,
            picks: memory.mapped(rows * 4)?,
            partials: memory.floats(TILED_GEMV_PARTIALS * groups)?,
            counters: memory.zeroed(stream, TILED_GEMV_BLOCKS * groups * 4)?,
            recur_counters: memory.zeroed(stream, linear_heads * 4)?,
            moe_sorted: memory.zeroed(stream, moe_rows * 4)?,
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

    /// Binds this rank's retained context on the calling thread.
    ///
    /// The serving driver opens a session on one worker thread and may run its steps on another;
    /// the driver's current context is thread-local, so every device entry point makes it current
    /// here instead of assuming the thread that loaded the weights is the thread that uses them.
    pub(crate) fn make_current(&self) -> Result<(), BackendError> {
        self.cuda.make_current().map_err(cuda_error)
    }

    /// A fresh sequence of up to `capacity` positions.
    ///
    /// # Errors
    /// When `capacity` is past the context window or the device refuses an allocation.
    pub fn new_sequence(&self, capacity: usize) -> Result<DeviceSequence, BackendError> {
        self.sequence(capacity, false, false)
    }

    /// Whether this rank loaded a DFlash 2 block drafter ([Self::load_block_drafter]).
    ///
    /// A serving arm asks this to choose [Self::new_block_drafting_sequence] over
    /// [Self::new_drafting_sequence], and a backend that has neither cannot draft at all.
    #[must_use]
    pub fn has_block_drafter(&self) -> bool {
        self.block.is_some()
    }

    /// Whether the checkpoint carries an MTP layer this rank can draft with.
    #[must_use]
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Whether a block-drafting sequence's prompt runs through the bulk chunk
    /// (`prefill.drafting_prompt`, on by default) or the token path, which keeps the token path's
    /// bits. Either way the drafter's context takes the prompt rows its first block reads.
    pub fn set_drafting_prefill(&mut self, drafting: bool) {
        self.drafting_prefill = drafting;
    }

    /// The most drafts a step of this rank verifies after the token they follow.
    ///
    /// A block drafter proposes its whole block, one row for every draft; the MTP layer proposes
    /// up to a step's rows. A serving arm sizes its draft depth from this, so a block drafter is
    /// not capped at the MTP layer's depth.
    #[must_use]
    pub fn max_drafts(&self) -> usize {
        match &self.block {
            Some(block) => block.draft.config().block_size.saturating_sub(1),
            None => MAX_ROWS - 1,
        }
    }

    /// Puts `sequence` back at position 0 with no history, the state [Self::new_sequence] opens
    /// one in, keeping its span and its plain allocations. The lane's chunks go back to the pool
    /// (up to 18.2 GiB a rank at the 1M default) for any sequence to grow into. As in a fresh
    /// window, the convolution history and the recurrent memory are cleared, and the caches come
    /// back zeroed as the next prompt grows; a block drafter's rings are zeroed.
    ///
    /// # Errors
    /// When the device refuses a copy.
    pub fn reset(&self, sequence: &mut DeviceSequence) -> Result<(), BackendError> {
        self.clear(sequence)
    }

    /// Zeroes `sequence`'s convolution history and recurrent memory and rewinds it to position 0.
    fn clear(&self, sequence: &mut DeviceSequence) -> Result<(), BackendError> {
        self.make_current()?;
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
        // The lane's rows go back to the pool, and come back zeroed as the next prompt grows.
        sequence.lane.release()?;
        sequence.position = 0;
        sequence.pending = None;
        sequence.logits_at = None;
        sequence.confidence.clear();
        // A reused window's drafter rings still hold the last request's rows, and a ring slot is
        // read by position: without this a request's first drafts attend to another's positions.
        if let (Some(block), Some(scratch)) = (&self.block, &sequence.block) {
            let (rings, floats) = scratch.rings_region(block.draft.config());
            let zeros = vec![0u8; floats * 4];
            self.cuda
                .copy_htod_on(&self.stream, &rings, &zeros)
                .map_err(cuda_error)?;
        }
        if let Some(drafter) = sequence.drafter.as_mut() {
            drafter.last = None;
        }
        sequence.forget_point();
        Ok(())
    }

    /// Whether this rank's sequences keep a rewind point ([Self::keep_point]): unless
    /// [sparkflash_runtime::REWIND_ENV] is off.
    #[must_use]
    pub fn rewinds(&self) -> bool {
        self.rewind
    }

    /// Keeps `sequence`'s state at its position as its rewind point ([DeviceSequence::rewind_point]):
    /// copies, on the rank's stream, of each KDA layer's convolution history and recurrent memory
    /// and of the MTP layer's hidden rows or the block drafter's rings. A sequence with no point (the rank keeps none) is left
    /// as it is.
    ///
    /// # Errors
    /// A sequence with a verify waiting for its commit, or a copy the device refuses (the sequence
    /// then holds no point).
    pub fn keep_point(&self, sequence: &mut DeviceSequence) -> Result<(), BackendError> {
        if sequence.point.is_none() {
            return Ok(());
        }
        if sequence.pending.is_some() {
            return Err(refusal(
                "a sequence keeps its rewind point only once its verify is committed",
            ));
        }
        self.make_current()?;
        let live = sequence.point_regions();
        let position = sequence.position;
        let point = sequence
            .point
            .as_mut()
            .ok_or_else(|| refusal("no rewind point"))?;
        point
            .save(&self.cuda, Some(&self.stream), &live, position)
            .map_err(cuda_error)?;
        sequence.point_last = sequence.drafter.as_ref().and_then(|drafter| drafter.last);
        Ok(())
    }

    /// Takes `sequence` back to its rewind point `point` ([Self::keep_point]): the copies back
    /// into its KDA layers and the MTP layer's hidden rows, on the rank's stream, and its position
    /// to the point. A verify still waiting for its commit is dropped with the rows past the point.
    ///
    /// # Errors
    /// A point the sequence does not hold, or a copy the device refuses.
    pub fn rewind(&self, sequence: &mut DeviceSequence, point: usize) -> Result<(), BackendError> {
        if sequence.rewind_point() != Some(point) {
            return Err(refusal(format!(
                "the sequence holds no rewind point at position {point}"
            )));
        }
        self.make_current()?;
        let live = sequence.point_regions();
        let kept = sequence
            .point
            .as_ref()
            .ok_or_else(|| refusal("no rewind point"))?;
        kept.restore(&self.cuda, Some(&self.stream), &live, point)
            .map_err(cuda_error)?;
        sequence.position = point;
        sequence.pending = None;
        sequence.logits_at = None;
        sequence.confidence.clear();
        if let Some(drafter) = sequence.drafter.as_mut() {
            drafter.last = sequence.point_last;
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
        self.sequence(capacity, true, false)
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
        // A rank with the bulk chunk captures a drafting prompt's last rows during it.
        let prompt = self
            .prefill
            .is_some()
            .then(|| config.window_left.min(self.prefill_rows) * config.feature_width());
        let mut buffers = || -> Result<_, BackendError> {
            Ok((
                self.memory.floats(exchange)?,
                self.memory.mapped(exchange * 4)?,
                self.memory.mapped(chains)?,
                prompt
                    .map(|floats| self.memory.floats(floats))
                    .transpose()?,
            ))
        };
        match buffers() {
            Ok((exchange, exchange_host, chains, prompt)) => {
                self.block = Some(BlockDrafter {
                    draft,
                    exchange,
                    exchange_host,
                    chains,
                    prompt,
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
        if self.block.is_none() {
            return Err(refusal(
                "this rank has no block drafter; load_block_drafter loads one",
            ));
        }
        self.sequence(capacity, false, true)
    }

    /// A fresh sequence of up to `capacity` positions; `drafting` gives it the MTP layer's state
    /// (when the rank has the layer), `block` the block drafter's (when the rank has one).
    fn sequence(
        &self,
        capacity: usize,
        drafting: bool,
        block: bool,
    ) -> Result<DeviceSequence, BackendError> {
        self.make_current()?;
        let config = &self.config;
        if capacity > config.max_position_embeddings {
            return Err(refusal(format!(
                "a sequence of {capacity} positions is past the context window {}",
                config.max_position_embeddings
            )));
        }
        let drafting = drafting && self.mtp.is_some();
        let lanes = self.sequence_lanes(capacity, drafting)?;
        let lane = SequenceLane {
            lane: lane_memory::SequenceLane::reserve(&self.cuda, &lanes).map_err(cuda_error)?,
            cuda: Arc::clone(&self.cuda),
        };
        let base = lane.lane.base();
        let parts = lanes.parts();
        let mut memory = Allocations::new(Arc::clone(&self.cuda));
        let head = config.linear_head_dim;
        let history = config.linear_conv_kernel_dim.saturating_sub(1);
        let layers = self
            .layers
            .iter()
            .zip(&parts.layers)
            .map(|(layer, parts)| {
                Ok(match (&layer.branch, parts) {
                    (DeviceBranch::Linear(kda), _) => SequenceLayer::Linear {
                        conv: memory.bytes(history * 3 * kda.width * 4)?,
                        recurrent: memory.bytes(kda.heads * head * head * 4)?,
                        spare: memory.floats(kda.heads * head * head)?,
                    },
                    (DeviceBranch::Sparse(_), Some(parts)) => parts.layer(base),
                    (DeviceBranch::Sparse(_), None) => {
                        return Err(refusal("a sparse layer the lane has no caches for"));
                    }
                })
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let drafter = match parts.mtp {
            Some(parts) => Some(SequenceDrafter {
                layer: parts.layer(base),
                hidden: memory.floats(MAX_ROWS * config.hidden_size)?,
                last: None,
            }),
            None => None,
        };
        let block = match (&self.block, block) {
            (Some(drafter), true) => Some(
                DraftScratch::allocate(&drafter.draft, MAX_ROWS, |floats| {
                    memory.floats(floats).map_err(|error| error.to_string())
                })
                .map_err(cuda_error)?,
            ),
            _ => None,
        };
        let mut sequence = DeviceSequence {
            position: 0,
            capacity,
            pending: None,
            logits_at: None,
            layers,
            drafter,
            block,
            confidence: Vec::new(),
            memory,
            lane,
            point: None,
            point_last: None,
        };
        if self.rewind {
            let mut sizes = Vec::new();
            for layer in &self.layers {
                if let DeviceBranch::Linear(kda) = &layer.branch {
                    sizes.extend([history * 3 * kda.width * 4, kda.heads * head * head * 4]);
                }
            }
            if sequence.drafter.is_some() {
                sizes.push(MAX_ROWS * config.hidden_size * 4);
            }
            // The block drafter's rings hold its context by position modulo the ring, so a reply
            // longer than a block overwrites the window before the point: the point keeps them
            // whole.
            if let (Some(drafter), Some(scratch)) = (&self.block, &sequence.block) {
                sizes.push(scratch.rings_region(drafter.draft.config()).1 * 4);
            }
            let arena = sequence.memory.bytes(Checkpoint::arena_bytes(&sizes))?;
            sequence.point = Some(Checkpoint::new(arena, sizes));
        }
        self.clear(&mut sequence)?;
        Ok(sequence)
    }

    /// The [SequenceLanes] of sequences of `capacity` positions that draft with the MTP layer or
    /// not, planned on first use.
    fn sequence_lanes(
        &self,
        capacity: usize,
        drafting: bool,
    ) -> Result<Arc<SequenceLanes<LaneParts>>, BackendError> {
        let plan = || {
            SequenceLanes::plan(
                &self.cuda,
                capacity,
                LANE_READ_PAST,
                sparkflash_model::active_profile().runtime.lane_memory_mb,
                |layout| {
                    let layers = self
                        .layers
                        .iter()
                        .map(|layer| match &layer.branch {
                            DeviceBranch::Linear(_) => None,
                            DeviceBranch::Sparse(dsa) => {
                                Some(SparseParts::plan(layout, &self.config, dsa, capacity))
                            }
                        })
                        .collect();
                    let mtp = self
                        .mtp
                        .as_ref()
                        .filter(|_| drafting)
                        .map(|mtp| SparseParts::plan(layout, &self.config, &mtp.dsa, capacity));
                    LaneParts { layers, mtp }
                },
            )
            .map_err(refusal)
        };
        self.lanes.get((drafting, capacity), plan, |lanes| {
            if self.layout.rank == 0 {
                eprintln!(
                    "[glm5] rank 0: {}sequences of {capacity} positions back their caches as they \
                     grow: {}",
                    if drafting { "drafting " } else { "" },
                    lanes.describe()
                );
            }
        })
    }

    /// Backs `sequence`'s lane for `rows` positions and the rows past them a step reads or a
    /// draft writes ([LANE_READ_PAST]).
    ///
    /// # Errors
    /// The pool's refusal when it has too few chunks left, or a driver
    /// failure.
    pub(crate) fn back(
        &self,
        sequence: &mut DeviceSequence,
        rows: usize,
    ) -> Result<(), BackendError> {
        sequence
            .lane
            .lane
            .back(&self.cuda, rows)
            .map_err(|error| match error {
                BackError::Refused(error) => refusal(error),
                BackError::Device(error) => cuda_error(error),
            })
    }

    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        threads: u32,
        args: &[u64],
    ) -> Result<(), BackendError> {
        Launcher::launch(self, kernel, (grid, (threads, 1, 1)), args)
    }

    /// `output = W input` for `rows` rows, BF16 or block-scaled FP8 by the weight, each of a fused
    /// projection's parts as one block of every row's outputs. A prefill-sized batch takes the
    /// bulk tensor-core GEMM; a decode step takes the tiled GEMV, whose bits a verified row is
    /// defined by.
    fn gemv(
        &self,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows > MAX_ROWS {
            self.bulk_gemm(linear, input, output, rows)
        } else {
            self.gemv_small(linear, input, output, rows)
        }
    }

    /// The FP8 tile's few-row body over each part of `linear`: one launch a part, over the
    /// part's weight rows, its 128-column scale block and its outputs for all `rows` rows. The
    /// body's arguments are the prompt tile's (`w, ws, a, out, cols, inner, tokens`) and its
    /// activations are the f32 rows the GEMV read, so nothing is converted between the two.
    /// Version 3 serves the tiled GEMV; the body stays in the catalogue, tested directly.
    #[allow(dead_code)]
    fn fp8_token_body(
        &self,
        planes: CodePlanes<'_, Fp8E4m3>,
        linear: &DeviceLinear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        launch_fp8_token_body(
            &self.cuda,
            &self.stream,
            self.kernels.prefill_dense_fp8_rows.kernel(),
            planes,
            linear,
            input,
            output,
            rows,
        )
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
        let element = linear.weight.element();
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
            let weight = linear.weight.data().offset(first * columns * element);
            let scales = linear.weight.scales().map(|scales| {
                scales.offset(
                    (first / FP8_BLOCK) * (columns / FP8_BLOCK) * std::mem::size_of::<f32>(),
                )
            });
            let out = output.offset(rows * first * 4);
            // A BF16 part takes what the dispatcher gives GLM's bulk walk: the wide tile where it
            // covers the part, else the skinny kernel (at most 32 outputs over whole 256-column
            // steps), else the tiled GEMV.
            let Some(scales) = scales else {
                match WidePolicy::GLM.bulk(part_rows, columns, rows) {
                    Route::Wide => {
                        let bf16 = if self.prefill_activation == PrefillActivation::Bf16 {
                            self.kernels.prefill_dense_bf16_bf16a.kernel()
                        } else {
                            self.kernels.prefill_dense_bf16.kernel()
                        };
                        let tile =
                            WideTile::call(weight.0, input.0, out.0, part_rows, columns, rows)
                                .map_err(refusal)?;
                        self.launch(bf16, tile.grid, tile.block.0, &tile.args)?;
                    }
                    Route::Skinny => self.launch(
                        self.kernels.prefill_skinny[part_rows.div_ceil(8) - 1].kernel(),
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
                    )?,
                    _ => {
                        let row = DeviceLinear::view(weight, part_rows, columns, false, None);
                        self.gemv_grouped(&row, input, out, rows)?;
                    }
                }
                continue;
            };
            // The block-scaled kernel indexes its scales a 128-column K block at a time, so a
            // scaled part needs the K run to be whole blocks, not just whole 32-column stages.
            let tiled = fp8_tile_fits(part_rows, columns) && first.is_multiple_of(FP8_BLOCK);
            if !tiled {
                let row = DeviceLinear::view(weight, part_rows, columns, true, Some(scales));
                self.gemv_grouped(&row, input, out, rows)?;
                continue;
            }
            let grid = Fp8Tile::grid(part_rows, rows);
            let fp8 = if self.prefill_activation == PrefillActivation::Bf16 {
                self.kernels.prefill_dense_fp8_bf16a.kernel()
            } else {
                self.kernels.prefill_dense_fp8.kernel()
            };
            if self.prefill_activation == PrefillActivation::Fp8
                && columns.is_multiple_of(FP8_BLOCK)
            {
                q8_inner(columns)?;
                if !quantised {
                    self.quantise(input, rows, columns)?;
                    quantised = true;
                }
                let s = &self.scratch;
                self.launch(
                    self.kernels.prefill_dense_q8.kernel(),
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
                )?;
            } else {
                self.launch(
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
                )?;
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
            self.kernels.prefill_quantise.kernel(),
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
    /// launch, each part on its own past that. A BF16 part the tiled GEMV splits one 256-column
    /// round a block takes glm_prefill_exact_bf16 over every row instead, which computes the same
    /// floats, up to 2.5 times as fast.
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
        let element = linear.weight.element();
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
            let row = DeviceLinear::view(
                linear.weight.data().offset(first * columns * element),
                part_rows,
                columns,
                linear.weight.is_fp8(),
                linear.weight.scales().map(|scales| {
                    scales.offset(
                        (first / FP8_BLOCK) * (columns / FP8_BLOCK) * std::mem::size_of::<f32>(),
                    )
                }),
            );
            let out = output.offset(rows * first * 4);
            let rounds = columns / TILED_GEMV_ROUND;
            if !row.weight.is_fp8()
                && columns.is_multiple_of(TILED_GEMV_ROUND)
                && input.0.is_multiple_of(16)
                && row.weight.data().0.is_multiple_of(16)
                && tiled_gemv_splits(part_rows.div_ceil(TILE_ROWS), columns) == rounds
            {
                self.launch(
                    self.kernels.prefill_exact.kernel(),
                    (
                        part_rows.div_ceil(EXACT_OUTS) as u32,
                        rows.div_ceil(EXACT_ROWS) as u32,
                        1,
                    ),
                    EXACT_THREADS,
                    &[
                        row.weight.data().0,
                        input.0,
                        out.0,
                        part_rows as u64,
                        columns as u64,
                        rows as u64,
                    ],
                )?;
                continue;
            }
            self.gemv_grouped(&row, input, out, rows)?;
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
        // A packed copy of these rows, when there is one: the same launch over the packed planes.
        let packed = self
            .packed
            .get(linear.weight.data())
            .filter(|packed| packed.admits(linear.rows, linear.columns, input).is_ok());
        let (kernel, splits, mut args) = match (linear.weight.scales(), packed) {
            (None, Some(packed)) => (
                self.kernels.gemv_packed.kernel(),
                tiled_gemv_splits(tiles, linear.columns),
                packed.tiled_planes()[..3].to_vec(),
            ),
            (None, None) => (
                self.kernels.gemv_bf16.kernel(),
                tiled_gemv_splits(tiles, linear.columns),
                vec![linear.weight.data().0],
            ),
            (Some(scales), _) => (
                self.kernels.gemv_fp8.kernel(),
                tiled_gemv_e4m3_splits(tiles, linear.columns),
                vec![linear.weight.data().0, scales.0],
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
        if linear.weight.is_fp8() {
            args.push(u64::from(self.config.dynamic_activations));
        }
        args.extend(linear.parts.map(|start| start as u64));
        if let (None, Some(packed)) = (linear.weight.scales(), packed) {
            args.push(packed.tiled_planes()[3]);
        }
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
            self.kernels.expert_fp8.kernel(),
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
                // A packed matrix's GEMV reads its packed rows, not the BF16 ones.
                let (at, bytes) = self.packed.get(at).map_or((at, bytes), |p| p.span());
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
                    self.kernels.prefetch_l2.kernel(),
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
    ///
    /// With `share`, a sequence-parallel step's ([Self::sequence_parallel]), the mixer runs over
    /// this rank's own rows only and the ranks then gather every row's normed input.
    fn mix_in(
        &self,
        hyper: &DeviceHyper,
        norm: Deviceptr,
        rows: usize,
        fused: bool,
        share: Option<Share<'_>>,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let geometry = self.geometry;
        let (hc, hidden) = (geometry.hc_mult, geometry.dim);
        let mixes = geometry.hc_mix_count();
        let (first, count) = share.map_or((0, rows), |share| (share.first, share.live));
        let row = |base: Deviceptr, width: usize| base.offset(first * width * 4);
        let (streams, raw) = (row(s.streams, hc * hidden), row(s.raw, mixes));
        let collapsed = row(s.collapsed, hidden);
        if share.is_some() {
            // The step's rows take the bulk projection ([Self::gemv]), so a share of them takes
            // it too: its rows are the step's own floats.
            self.bulk_gemm(&hyper.projection, streams, raw, count)?;
        } else if !fused {
            self.gemv(&hyper.projection, streams, raw, rows)?;
        }
        self.launch(
            self.kernels.mhc_in.kernel(),
            (2, count as u32, 1),
            BLOCK,
            &[
                raw.0,
                streams.0,
                hyper.scale.0,
                hyper.base.0,
                row(s.coeffs, mixes).0,
                norm.0,
                collapsed.0,
                hidden as u64,
                hc as u64,
                geometry.hc_sinkhorn_iters as u64,
                float(geometry.hc_eps),
                float(geometry.norm_eps),
                float(self.config.rms_norm_eps as f32),
                if fused && share.is_none() {
                    s.mix_parts.0
                } else {
                    0
                },
            ],
        )?;
        match share {
            Some(share) => share.gather(self, collapsed, s.collapsed, hidden),
            None => Ok(()),
        }
    }

    /// This rank's share of a `rows`-row step whose hyper-connections run sequence-parallel, else
    /// None.
    ///
    /// The mixers, the stream updates and the final fold are replicated work: every rank mixes
    /// every row, about a sixth of a bulk prefill chunk. Sequence-parallel, each rank runs them
    /// over its own `rows / world` rows: the branch partials are reduce-scattered (the two-shot
    /// all-reduce's first shot), so a rank sums and folds only its own rows, and the ranks gather
    /// the normed rows each branch reads, where the two-shot reduce gathered the sums. The bytes
    /// moved are the two-shot reduce's. Every kernel involved computes a row from that row alone,
    /// so each row keeps its bits. A step of rows the world does not divide is padded to the next
    /// multiple for the collectives; the padding rows are never computed or read.
    ///
    /// Only a bulk step the two-shot reduce takes, on the device leg, whose padded rows fit the
    /// prefill scratch, and none of whose runs captures rows for the block drafter, which reads
    /// every row's streams.
    fn sequence_parallel<'t>(
        &self,
        tp: &'t TpRuntime,
        runs: &[Run<'_>],
        rows: usize,
        device_leg: bool,
    ) -> Option<Share<'t>> {
        let world = self.layout.world_size;
        let captures = runs
            .iter()
            .any(|run| run.block.is_some_and(|(_, from)| from < run.len));
        let count = rows.div_ceil(world);
        let first = self.layout.rank * count;
        (self.sequence_parallel
            && device_leg
            && rows > MAX_ROWS
            && count * world <= self.prefill_rows
            && first < rows
            && self.two_shot(count * world)
            && !captures)
            .then(|| Share {
                tp,
                first,
                count,
                live: count.min(rows - first),
            })
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
            && !linear.weight.is_fp8()
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
    ///
    /// With `share` ([Self::sequence_parallel]) a rank sums and mixes its own rows only.
    fn mix_out(
        &self,
        tp: &TpRuntime,
        residual: (Deviceptr, Deviceptr, usize),
        next: Option<&DeviceHyper>,
        rows: usize,
        device_leg: bool,
        share: Option<Share<'_>>,
    ) -> Result<bool, BackendError> {
        // The fused mixer writes one tiled GEMV's split partials; a prefill chunk takes the bulk
        // GEMM instead, whose 128-row tiles the mixer's 24-row projection does not feed.
        let next = if rows > MAX_ROWS {
            None
        } else {
            next.and_then(|hyper| self.fused_projection(hyper))
        };
        let partial = self.scratch.partial;
        match share {
            Some(share) => self.reduce_share(partial, residual, rows, share)?,
            None => self.reduce_into(tp, partial, residual, next, rows, device_leg)?,
        }
        Ok(next.is_some())
    }

    /// `q4e_sum_slots` of this rank's `count` floats at `mine` and the peers' in `slots`: the
    /// two-shot reduce's sum, in rank order from 0.0 as `q4e_mhc_post` folds the one-shot's.
    fn sum_slots(&self, mine: Deviceptr, count: usize) -> Result<(), BackendError> {
        self.launch(
            self.kernels.sum_slots.kernel(),
            (blocks(count / 4), 1, 1),
            BLOCK,
            &[
                mine.0,
                self.scratch.slots.0,
                count as u64,
                self.layout.rank as u64,
                self.layout.world_size as u64,
            ],
        )
    }

    /// [Self::mix_out] of a sequence-parallel step: the partials reduce-scattered, so this rank
    /// holds the sum of its own rows, and those rows' streams mixed.
    fn reduce_share(
        &self,
        partial: Deviceptr,
        (streams, coeffs, hc): (Deviceptr, Deviceptr, usize),
        rows: usize,
        share: Share<'_>,
    ) -> Result<(), BackendError> {
        let hidden = self.geometry.dim;
        let padded = share.count * share.tp.world();
        debug_assert!(padded >= rows);
        let mine = reduce_scatter(
            share.tp,
            self.stream.raw(),
            partial,
            self.scratch.slots,
            padded * hidden,
            |mine, count| self.sum_slots(mine, count),
        )?;
        let row = |base: Deviceptr, width: usize| base.offset(share.first * width * 4);
        if mine != row(partial, hidden) {
            return Err(refusal(
                "the reduce-scatter's chunk is not this rank's share of the rows",
            ));
        }
        self.post(
            (row(streams, hc * hidden), mine, row(coeffs, (2 + hc) * hc)),
            hc,
            share.live,
            1,
            None,
        )
    }

    /// `q4e_mhc_post` over `rows` rows: each row's summed branch (with `world > 1` this rank's
    /// partial, the peers' in `slots`) mixed into its `hc` streams by its `coeffs`.
    fn post(
        &self,
        (streams, branch, coeffs): (Deviceptr, Deviceptr, Deviceptr),
        hc: usize,
        rows: usize,
        world: usize,
        next: Option<&DeviceLinear>,
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let hidden = self.geometry.dim;
        self.launch(
            self.kernels.mhc_post.kernel(),
            (blocks(hidden), rows as u32, 1),
            BLOCK,
            &[
                streams.0,
                branch.0,
                coeffs.0,
                hidden as u64,
                hc as u64,
                s.slots.0,
                self.layout.rank as u64,
                world as u64,
                next.map_or(0, |linear| linear.weight.data().0),
                if next.is_some() { s.mix_parts.0 } else { 0 },
                next.map_or(0, |linear| linear.rows as u64),
            ],
        )
    }

    /// Whether a device-leg all-reduce of `rows` rows takes two shots: at least
    /// [Self::two_shot_rows] rows (0 for never) of whole float4 chunks a rank.
    fn two_shot(&self, rows: usize) -> bool {
        let world = self.layout.world_size;
        takes_two_shots(self.two_shot_rows, rows, rows * self.geometry.dim, world)
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
                |mine, count| self.sum_slots(mine, count),
            )?;
            1
        } else if device_leg {
            // `partial` holds the rows' partials and `slots` a world of them, and every rank runs
            // the same steps.
            tp.exchange_on_device(self.stream.raw(), partial.0, s.slots.0, rows * hidden * 4)
                .map_err(weights_error)?;
            world
        } else {
            host_sum(&self.cuda, Some(&self.stream), tp, partial, rows * hidden)?;
            1
        };
        self.post((streams, partial, coeffs), hc, rows, world, next)
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
            let rows_in = ConvRows {
                raw: own.fused.offset(floats(run.first * mixes)),
                history: *conv,
                weight: kda.conv,
                out: own.mixed.offset(floats(run.first * mixes)),
            };
            let width = (mixes, config.linear_conv_kernel_dim);
            if self.bulk(run) {
                // A chunk never verifies, so it keeps no windows.
                self.kernels
                    .chunk_conv
                    .run(self, &rows_in, width, run.len, (0, 0, 0))?;
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
            self.kernels.conv.run(
                self,
                &rows_in,
                width,
                run.len,
                (windows, kept, window as u64),
            )?;
        }
        self.gemv(&kda.f_b, f_a, s.forget, rows)?;
        self.gemv(&kda.g_b, g_a, s.gate, rows)?;
        self.launch(
            self.kernels.kda_decay.kernel(),
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
        !run.verify && run.len > MAX_ROWS && self.kda_scan.is_some()
    }

    /// The bulk KDA scan this rank's plan resolves ([ScanPlan::for_architecture]), at the head
    /// and history widths the prefill image is compiled for; `None` elsewhere, where every run
    /// takes the token kernels.
    ///
    /// # Errors
    /// A plan the rank cannot build, or one whose scan is not this rank's share.
    fn kda_scan_plan(
        config: &Glm5NextConfig,
        layout: TpLayout,
    ) -> Result<Option<ScanPlan>, BackendError> {
        if !(config.linear_head_dim == GDN_KEY
            && config.linear_head_dim == GDN_VALUE
            && (2..=CONV_HISTORY + 1).contains(&config.linear_conv_kernel_dim))
        {
            return Ok(None);
        }
        let plan = ScanPlan::for_architecture(
            &config.architecture,
            layout.world_size,
            layout.rank,
            ScanKind::Kda,
            MAX_ROWS + 1,
        )
        .map_err(|error| BackendError::new("glm5_device", error.to_string()))?;
        let heads = config.linear_num_heads / layout.world_size.max(1);
        if let Some(scan) = &plan
            && (scan.heads, scan.key_dim, scan.value_dim)
                != (heads, config.linear_head_dim, GDN_VALUE)
        {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the plan's KDA scan is {} heads of {} x {}, this rank's {heads} of {} x {}",
                    scan.heads, scan.key_dim, scan.value_dim, config.linear_head_dim, GDN_VALUE
                ),
            ));
        }
        Ok(plan)
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
            self.kernels.kda_prep.kernel(),
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
        // The plan's scan: its kernel and its grid ([ScanPlan]).
        let launch = self
            .kda_scan
            .as_ref()
            .ok_or_else(|| BackendError::new("glm5_device", "the rank's plan has no KDA scan"))?
            .launch();
        if launch.kernel.symbol != "q4p_kda_scan" {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the prefill image loads no KDA scan named {}",
                    launch.kernel.symbol
                ),
            ));
        }
        self.launch(
            self.kernels.kda_scan.kernel(),
            launch.grid,
            launch.block.0,
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
        // The sigmoid output gate.
        self.kernels.kda_gate.run(
            self,
            (out, gate, kda.o_norm, out),
            (heads, GDN_VALUE),
            (config.rms_norm_eps as f32, true),
            len,
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
            self.kernels.recur.kernel(),
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
                u64::from(self.kernels.recur.kernel().shared_bytes()),
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
        let eps = config.rms_norm_eps as f32;
        let (q_lora, kv_lora) = (config.q_lora_rank, config.kv_lora_rank);
        let floats = |count: usize| count * 4;
        self.gemv(&dsa.fused, s.collapsed, s.dsa_fused, rows)?;
        let latent = dsa.fused.part(s.dsa_fused, 1, rows);
        for (part, norm, width) in [
            (s.dsa_fused, dsa.q_a_norm, q_lora),
            (latent, dsa.kv_a_norm, kv_lora),
        ] {
            self.kernels
                .norm
                .run(self, RmsNorm::new(part.0, norm.0, part.0, width, eps), rows)?;
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
                    self.kernels.mla_cache_fp8.kernel(),
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
                    self.kernels.mla_cache.kernel(),
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
        // The absorbed form: the query projected into the latent space, then the attention over
        // the latent cache in four launches (glm_decode_sm121.cu's q4e_glm_mla_scores) and kv_b's
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
                self.kernels.mla_absorb.kernel(),
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
                self.kernels.mla_scores.kernel(),
                (
                    (MAX_SELECTED_TOKENS / MLA_SCORE_SLOTS) as u32,
                    len,
                    heads.div_ceil(MLA_HEADS) as u32,
                ),
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
                self.kernels.mla_softmax.kernel(),
                (heads as u32, len, 1),
                BLOCK,
                &[s.mla_logits.0, s.count.0, s.mla_total.0, first],
            )?;
            self.launch(
                self.kernels.mla_mix.kernel(),
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
                self.kernels.mla_project.kernel(),
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
        if latent != GMA_LATENT || heads == 0 || !value.is_multiple_of(128) {
            return Err(refusal(format!(
                "the bulk MLA takes a {GMA_LATENT}-wide latent (in {GMA_HEADS}-head tiles, so \
                 any head count) and whole 128-column value tiles, not latent {latent} and value \
                 {value} with {heads} heads"
            )));
        }
        let row_tiles = rows.div_ceil(128) as u32;
        match dsa.kv_key_t {
            Some(key_t) => self.launch(
                self.kernels.prefill_heads_bf16.kernel(),
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
                self.kernels.mla_absorb.kernel(),
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
            self.kernels.prefill_mla_attend_bf16a.kernel()
        } else {
            self.kernels.prefill_mla_attend.kernel()
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
                (run.len as u32, 1, heads.div_ceil(GMA_HEADS) as u32),
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
                    heads as u64,
                ],
            )?;
        }
        // kv_b's value rows of head h: rows h (nope + value) + nope .. + value of the matrix.
        let per_head = nope + value;
        let element = dsa.kv_b.weight.element();
        let weight = dsa.kv_b.weight.data().offset(nope * latent * element);
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
        match dsa.kv_b.weight.scales() {
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
                self.launch(self.kernels.prefill_heads_fp8.kernel(), grid, BLOCK, &args)
            }
            None => {
                let mut args = vec![weight.0, s.q_abs.0, s.attn_out.0];
                args.extend(geometry);
                self.launch(self.kernels.prefill_heads_bf16.kernel(), grid, BLOCK, &args)
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
            self.kernels.layer_norm.run(
                self,
                s.index_fused.offset(floats(run.first * dim)),
                (indexer.k_norm_weight, indexer.k_norm_bias),
                index_keys.offset(cached),
                dim,
                INDEX_KEY_NORM_EPS,
                run.len,
            )?;
            self.launch(
                self.kernels.copy_flat.kernel(),
                (blocks(run.len * dim), 1, 1),
                BLOCK,
                &[
                    index_gates.offset(cached).0,
                    gates.offset(floats(run.first * dim)).0,
                    (run.len * dim) as u64,
                ],
            )?;
            let (grid, block) = IndexerPass::new(run.position, run.len, pool).per_row();
            self.launch(
                self.kernels.pool.kernel(),
                grid,
                block.0,
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
            // The grids are the shared indexer's (IndexerPass), the rule the rank's boot plan
            // names: the last row scores the most pools, and a block past a row's own pools is a
            // no-op.
            let pass = IndexerPass::new(position, run.len, pool);
            let query = s.index_query.offset(floats(row * heads * dim)).0;
            let weights = head_weights.offset(floats(row * heads)).0;
            let scores = s.scores.offset(floats(row * score_stride)).0;
            if let Some((grid, block)) = pass.run_scores(heads, dim, MAX_ROWS) {
                // A prefill run's rows share their pooled keys: glm_prefill_index_scores keeps a
                // tile of them while it walks the rows, and computes the decode kernel's floats.
                self.launch(
                    self.kernels.prefill_index_scores.kernel(),
                    grid,
                    block.0,
                    &[
                        query,
                        weights,
                        pooled.0,
                        scores,
                        position as u64,
                        pool as u64,
                        float(inverse_sqrt(dim)),
                        float(inverse_sqrt(heads)),
                        score_stride as u64,
                        run.len as u64,
                    ],
                )?;
            } else {
                let (grid, block) = pass.scores();
                self.launch(
                    self.kernels.index_scores.kernel(),
                    grid,
                    block.0,
                    &[
                        query,
                        weights,
                        pooled.0,
                        scores,
                        position as u64,
                        heads as u64,
                        dim as u64,
                        pool as u64,
                        float(inverse_sqrt(dim)),
                        float(inverse_sqrt(heads)),
                        score_stride as u64,
                    ],
                )?;
            }
            let (grid, block) = pass.per_row();
            self.launch(
                self.kernels.pick.kernel(),
                grid,
                block.0,
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
            self.kernels.swiglu.kernel(),
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
    fn moe(&self, moe: &DeviceMoe, layer: Option<usize>, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hidden, width, topk) = (
            config.hidden_size,
            moe.intermediate,
            config.num_experts_per_tok,
        );
        self.prof_mark("moe_enter");
        self.mlp(&moe.shared, s.shared_out, rows)?;
        self.prof_mark("moe_shared");
        self.projection(
            &moe.router,
            s.collapsed,
            s.router_logits,
            rows,
            self.exact_prefill,
        )?;
        self.launch(
            self.kernels.router.kernel(),
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
        self.prof_mark("moe_router");
        if rows > MAX_ROWS {
            return self.bulk_moe(moe, layer, rows);
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
        self.prof_mark("moe_up");
        self.launch(
            self.kernels.swiglu.kernel(),
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
        self.prof_mark("moe_down");
        self.combine_experts(layer, rows)
    }

    /// The routed experts over a prefill chunk, grouped by expert: one item (up to
    /// [MOE_ITEM_ROWS] slots) a block set, the same tensor-core body as a dense projection. The
    /// gate/up GEMM reads each item row from its token's row, and the down GEMM writes each kept
    /// row, times its routing weight, to its slot's row.
    fn bulk_moe(
        &self,
        moe: &DeviceMoe,
        layer: Option<usize>,
        rows: usize,
    ) -> Result<(), BackendError> {
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
        // A slot no expert owns is left out of the plan rather than counted as expert 0. This
        // arm's combine sums every slot's row, so a dropped slot's row is zeroed here: the down
        // projection writes only the item rows, and would otherwise leave whatever the previous
        // chunk left in that slot.
        let plan = MoePlanCall::items(
            s.expert_indices.0,
            s.moe_sorted.0,
            (s.moe_items.0, s.moe_item_count.0),
            (slots, config.n_routed_experts, topk),
            MOE_ITEM_ROWS,
            items,
        )
        .zeroing(s.expert_out.0, hidden);
        let (grid, block) = MoePlanCall::launch();
        self.launch(
            self.kernels.prefill_moe_plan.kernel(),
            grid,
            block.0,
            &plan.args(),
        )?;
        if self.prefill_activation == PrefillActivation::Fp8
            && width.is_multiple_of(FP8_BLOCK)
            && hidden.is_multiple_of(FP8_BLOCK)
        {
            self.bulk_moe_q8(moe, rows, items)?;
            return self.combine_experts(layer, rows);
        }
        let mma = if self.prefill_activation == PrefillActivation::Bf16 {
            self.kernels.prefill_moe_mma_bf16a.kernel()
        } else {
            self.kernels.prefill_moe_mma.kernel()
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
                s.moe_sorted.0,
                topk as u64,
                0,
            ],
        )?;
        self.launch(
            self.kernels.prefill_moe_swiglu.kernel(),
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
                s.moe_sorted.0,
                topk as u64,
                s.expert_weights.0,
            ],
        )?;
        self.combine_experts(layer, rows)
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
            self.kernels.prefill_moe_q8.kernel(),
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
                s.moe_sorted.0,
                topk as u64,
                0,
            ],
        )?;
        let moe_rows = items * MOE_ITEM_ROWS;
        // One warp a 128-column block of an item row.
        let warps = moe_rows * width / FP8_BLOCK;
        self.launch(
            self.kernels.prefill_moe_swiglu_q8.kernel(),
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
            self.kernels.prefill_moe_q8.kernel(),
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
                s.moe_sorted.0,
                topk as u64,
                s.expert_weights.0,
            ],
        )?;
        Ok(())
    }

    /// The routed experts' rows and the shared expert's into each row's partial.
    ///
    /// `layer` is the decoder layer, for the monitor probe (docs/NUMERICS.md section 5); `None` is
    /// the MTP layer's draft pass, which is not recorded. The layer index changes no arithmetic.
    fn combine_experts(&self, layer: Option<usize>, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (grid, block) = combine_grid(config.hidden_size, rows);
        self.launch(
            self.kernels.combine.kernel(),
            grid,
            block.0,
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
        )?;
        // Coverage counts each layer's routed ids after its router launch; the monitor samples
        // the layer's row after the combine kernel. Both are diagnostics, and both read-backs
        // leave the layer's arithmetic untouched.
        self.record_routing(layer, rows);
        if let Some(layer) = layer {
            self.record_layer(layer);
        }
        Ok(())
    }

    /// Open this forward's recorders, when a switch names a path.
    ///
    /// Coverage opens an input on every forward, rank 0 or not: the router's chosen ids are the
    /// same view of the layer on every rank, and the coverage format has no rank field. The
    /// monitor dump is rank 0's slot of the tensor-parallel expert sum: the shared expert and the
    /// routed experts' columns are split, so `partial` and `expert_out` hold this rank's partial
    /// products. Every rank computes its slot the same way, and a reference build and a candidate
    /// run the same split, so the judge compares like with like. `rows` is this step's row count:
    /// only a decode step can be the probe, so a prompt's prefill leaves it for the first decode
    /// step after it. With a switch off, this is one `Option` check.
    fn begin_numerics(&self, tokens: &[u32], position: usize, rows: usize) {
        self.record.replace(
            self.numerics
                .begin(tokens, position, rows, self.layout.rank == 0),
        );
    }

    /// Write the dumps this step filled, if it had any. A write that fails is logged once and
    /// serving continues; the dumps are diagnostics, not part of the answer.
    fn finish_numerics(&self) {
        if let Some(forward) = self.record.replace(None) {
            forward.finish();
        }
    }

    /// Count this layer's chosen experts in the coverage dump: copy the router's
    /// `[rows][top-k]` ids back and hand them to [numerics::record_routing].
    ///
    /// With the switch off this is one `bool` check a layer. `layer` is the decoder layer; the
    /// MTP layer's draft pass passes `None` and records nothing, not even the ids' read-back. A
    /// read-back failure is a diagnostic's failure: it is logged once, the recorder is dropped and
    /// the layer's arithmetic is untouched.
    fn record_routing(&self, layer: Option<usize>, rows: usize) {
        let Some(layer) = layer else {
            return;
        };
        if !self.covering {
            return;
        }
        {
            let record = self.record.borrow();
            let Some(forward) = record.as_ref() else {
                return;
            };
            if !forward.coverage_enabled() {
                return;
            }
        }
        let count = rows * self.config.num_experts_per_tok;
        let mut bytes = vec![0u8; count * 4];
        match self
            .cuda
            .copy_dtoh_on(&self.stream, &mut bytes, &self.scratch.expert_indices)
        {
            Ok(()) => {
                numerics::record_routing(
                    &mut self.record.borrow_mut(),
                    Some(layer),
                    &numerics::experts(&bytes),
                );
            }
            Err(error) => {
                numerics::warn_coverage_read_back(&format!(
                    "layer {layer}: the router ids: {error}"
                ));
                self.record.replace(None);
            }
        }
    }

    /// Record this layer's probe row: the combined output and each routed expert's body.
    ///
    /// With the switch off this is one `bool` check a layer. On, it copies back the probe's row of
    /// the router's ids and weights, the layer's combined row and each slot's weighted expert row,
    /// and hands them to [numerics::record_probe]. A read-back failure is a diagnostic's failure:
    /// it is logged once, the recorder is dropped and the layer's arithmetic is untouched.
    fn record_layer(&self, layer: usize) {
        if !self.monitoring || self.record.borrow().is_none() {
            return;
        }
        match self.read_probe_row(layer) {
            Ok(probe) => {
                numerics::record_probe(
                    &mut self.record.borrow_mut(),
                    layer,
                    &probe.routed,
                    &probe.values,
                    &probe.experts,
                );
            }
            Err(error) => {
                numerics::warn_monitor_read_back(&error);
                self.record.replace(None);
            }
        }
    }

    /// Read back the probe row: `(expert id, combine weight)` and each slot's row as the down
    /// projection wrote it, `weight * body`.
    fn read_probe_row(&self, layer: usize) -> Result<numerics::ProbeRow, String> {
        let s = &self.scratch;
        let config = &self.config;
        let (hidden, topk) = (config.hidden_size, config.num_experts_per_tok);
        let mut ids = vec![0u8; topk * 4];
        self.cuda
            .copy_dtoh_on(&self.stream, &mut ids, &s.expert_indices)
            .map_err(|error| format!("layer {layer}: the router ids: {error}"))?;
        let mut weights = vec![0u8; topk * 4];
        self.cuda
            .copy_dtoh_on(&self.stream, &mut weights, &s.expert_weights)
            .map_err(|error| format!("layer {layer}: the router weights: {error}"))?;
        let mut row = vec![0u8; hidden * 4];
        self.cuda
            .copy_dtoh_on(&self.stream, &mut row, &s.partial)
            .map_err(|error| format!("layer {layer}: the combined row: {error}"))?;
        let ids: Vec<i32> = ids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| i32::from_le_bytes(*chunk))
            .collect();
        let weights = numerics::floats(&weights);
        let mut routed = Vec::with_capacity(topk);
        let mut experts = Vec::with_capacity(topk);
        for (slot, id) in ids.iter().enumerate() {
            let Ok(expert) = usize::try_from(*id) else {
                continue;
            };
            let weight = weights.get(slot).copied().unwrap_or(0.0);
            let mut bytes = vec![0u8; hidden * 4];
            self.cuda
                .copy_dtoh_on(
                    &self.stream,
                    &mut bytes,
                    &s.expert_out.offset(slot * hidden * 4),
                )
                .map_err(|error| format!("layer {layer} expert {expert}: {error}"))?;
            experts.push(numerics::floats(&bytes));
            routed.push((expert, weight));
        }
        Ok(numerics::ProbeRow {
            routed,
            values: numerics::floats(&row),
            experts,
        })
    }

    /// Refuses a sequence this rank cannot run `tokens` through.
    pub(crate) fn check_sequence(
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

    /// Whether `sequence`'s prompt may run the bulk prefill: the rank's chunk scratch exists only
    /// for a configuration the chunk can take (see `load`). A block-drafting sequence takes it too
    /// unless the profile says otherwise (`prefill.drafting_prompt`): the chunk captures the rows
    /// the drafter's first block reads, as the token path's steps would.
    fn chunk_ready(&self, sequence: &DeviceSequence) -> bool {
        self.prefill.is_some() && (sequence.block.is_none() || self.drafting_prefill)
    }

    /// Whether a prompt of `rows` rows on `sequence` takes the bulk chunks, not the token path
    /// ([Self::begin_prefill]).
    pub(crate) fn takes_chunks(&self, sequence: &DeviceSequence, rows: usize) -> bool {
        self.prefill.is_some() && rows >= MIN_PREFILL_ROWS && self.chunk_ready(sequence)
    }
    /// Prefill `tokens` in bulk chunks and return the last row's full-vocabulary logits
    /// ([Self::begin_prefill], then each [Self::prefill_chunk] in turn). A chunk
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
        let (end, width) = match self.begin_prefill(tp, gather, sequence, tokens)? {
            Prefill::Whole(logits) => return Ok(logits),
            Prefill::Chunks { end, width } => (end, width),
        };
        let mut last = Vec::new();
        for part in tokens.chunks(width) {
            last = self.prefill_chunk(tp, gather, sequence, part, end)?;
        }
        Ok(last)
    }

    /// Starts [Self::forward_prefill]'s prompt: a prompt the token path takes runs whole here;
    /// one that takes the bulk chunks has its rows backed, and [Self::prefill_chunk] then runs its
    /// chunks of the width answered, in order. Between two chunks the sequence holds everything
    /// the next reads: the chunk scratch is the rank's, but each chunk fills it from the start
    /// and commits the drafter's window rows before it ends, so a step or chunk of another
    /// sequence between two of them carries nothing into this one.
    ///
    /// # Errors
    /// As [Self::forward_prefill].
    pub fn begin_prefill(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        sequence: &mut DeviceSequence,
        tokens: &[u32],
    ) -> Result<Prefill, BackendError> {
        if tokens.is_empty() {
            return Err(refusal("an empty prompt has no logits"));
        }
        if !self.takes_chunks(sequence, tokens.len()) {
            let mut logits = self.forward_batch(tp, gather, &mut [(sequence, tokens)])?;
            return Ok(Prefill::Whole(
                logits
                    .pop()
                    .and_then(|mut rows| rows.pop())
                    .unwrap_or_default(),
            ));
        }
        self.admit(sequence, tokens)?;
        self.back(sequence, sequence.position + tokens.len())?;
        self.collective(tp, gather)?;
        Ok(Prefill::Chunks {
            end: sequence.position + tokens.len(),
            width: self.prefill_rows,
        })
    }

    /// One bulk chunk of the prompt [Self::begin_prefill] started, which ends at position `end`:
    /// `part` is its next [Prefill::Chunks] width of tokens or the rest. Answers the chunk's last
    /// row's full-vocabulary logits.
    ///
    /// # Errors
    /// As [Self::forward_prefill].
    pub fn prefill_chunk(
        &mut self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        sequence: &mut DeviceSequence,
        part: &[u32],
        end: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let device_leg = self.collective(tp, gather)?;
        let plan = self.chunk_plan(sequence, part.len(), end)?;
        self.next_step();
        self.swap_prefill()?;
        let run = Run::chunk(sequence, part.len(), &plan);
        let result = self.step::<Vec<f32>>(tp, gather, &[run], part, device_leg, HeadRows::Last);
        self.swap_prefill()?;
        let rows = result?;
        let last = rows.into_iter().next().unwrap_or_default();
        self.after_chunk(sequence, part.len(), &plan)?;
        Ok(last)
    }

    /// What a bulk chunk of `len` rows from `sequence`'s position, of a prompt that ends at
    /// `end`, hands its block drafter.
    ///
    /// A block drafter's first block reads the prompt's last `window_left` positions: a chunk
    /// captures its rows from the first of those on into the drafter's prompt rows, a scratch
    /// that is otherwise the sequence's, and commits them to its context a step's rows at a
    /// time, as the token path's steps would ([Dflash2Config::window_commits]).
    ///
    /// # Errors
    /// A block drafter with no prompt rows.
    pub(crate) fn chunk_plan(
        &self,
        sequence: &DeviceSequence,
        len: usize,
        end: usize,
    ) -> Result<ChunkPlan, BackendError> {
        let block = match (&self.block, &sequence.block) {
            (Some(block), Some(scratch)) => {
                let prompt = block.prompt.ok_or_else(|| {
                    refusal("the block drafter has no prompt rows for the bulk chunk")
                })?;
                Some((scratch.with_features(prompt), block.draft.config().clone()))
            }
            _ => None,
        };
        let start = sequence.position;
        let commits: Vec<(usize, usize, usize)> =
            block.as_ref().map_or_else(Vec::new, |(_, config)| {
                config.window_commits((start, len), end, MAX_ROWS).collect()
            });
        let captured = commits.first().map(|&(row, _, _)| row);
        Ok(ChunkPlan {
            scratch: block.map(|(scratch, _)| scratch),
            commits,
            captured,
        })
    }

    /// Swaps the bulk chunk's scratch in for the decode scratch, or back: a chunk runs between
    /// two swaps.
    ///
    /// # Errors
    /// A rank with no chunk scratch.
    pub(crate) fn swap_prefill(&mut self) -> Result<(), BackendError> {
        let other = self
            .prefill
            .as_mut()
            .ok_or_else(|| refusal("the prefill scratch is missing"))?;
        std::mem::swap(&mut self.scratch, other);
        Ok(())
    }

    /// Counts a step: a verify commits before the rank's next.
    pub(crate) fn next_step(&mut self) {
        self.steps = self.steps.wrapping_add(1);
    }

    /// `sequence` after a bulk chunk of `len` rows ran with `plan`: its position, its MTP stash's
    /// last row, and the block drafter's window commits.
    ///
    /// # Errors
    /// The device refuses a commit.
    pub(crate) fn after_chunk(
        &self,
        sequence: &mut DeviceSequence,
        len: usize,
        plan: &ChunkPlan,
    ) -> Result<(), BackendError> {
        sequence.position += len;
        sequence.logits_at = None;
        if let Some(drafter) = &mut sequence.drafter {
            // The chunk's last row is its stash's last ([Self::fill]).
            drafter.last = Some(len.min(MAX_ROWS) - 1);
        }
        if let (Some(drafter), Some(scratch), Some(captured)) =
            (&self.block, &plan.scratch, plan.captured)
        {
            for &(row, rows, position) in &plan.commits {
                drafter
                    .draft
                    .commit(self, scratch, row - captured, rows, position)?;
            }
        }
        Ok(())
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
        self.logits_row(at)
    }

    /// The full-vocabulary logits of the greedy step's row `at`, which stay on the device until
    /// this rank's next LM head ([Self::last_logits]).
    ///
    /// # Errors
    /// When another rank ran the step, an LM head has run since, or the device refuses a copy.
    pub(crate) fn logits_row(&self, at: LogitsAt) -> Result<Vec<f32>, BackendError> {
        self.make_current()?;
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
        self.make_current()?;
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
                    self.kernels.copy_flat.kernel(),
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
    pub(crate) fn collective(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
    ) -> Result<bool, BackendError> {
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
        self.make_current()?;
        let prof_started = Instant::now();
        self.prof_begin();
        let device_leg = self.collective(tp, gather)?;
        for (sequence, tokens) in batch.iter() {
            self.admit(sequence, tokens)?;
        }
        for (sequence, tokens) in batch.iter_mut() {
            let end = sequence.position + tokens.len();
            self.back(sequence, end)?;
        }
        let mut logits: Vec<Vec<R>> = batch
            .iter()
            .map(|(_, tokens)| Vec::with_capacity(tokens.len()))
            .collect();
        // A block drafter's next block sits just past a sequence's tokens and reaches its window
        // back, so the rows further back than that are never read and their commits are skipped
        // ([Dflash2Config::window_commits]).
        let ends: Vec<usize> = batch
            .iter()
            .map(|(sequence, tokens)| sequence.position + tokens.len())
            .collect();
        loop {
            let mut runs = Vec::new();
            let mut tokens = Vec::with_capacity(MAX_ROWS);
            let mut members = Vec::new();
            for (member, (sequence, chunk)) in batch.iter().enumerate() {
                let done = logits[member].len();
                let take = (chunk.len() - done).min(MAX_ROWS - tokens.len());
                if take > 0 {
                    runs.push(Run::token(sequence, tokens.len(), take, verify));
                    tokens.extend_from_slice(&chunk[done..done + take]);
                    members.push((member, take));
                }
            }
            if tokens.is_empty() {
                return Ok(logits);
            }
            // The marks a report drains were placed by the step above; a forward that placed none
            // (the switch is off) prints nothing.
            let firsts: Vec<usize> = runs.iter().map(|run| run.first).collect();
            self.next_step();
            let prof_rows = tokens.len();
            let mut rows = self
                .step::<R>(tp, gather, &runs, &tokens, device_leg, HeadRows::All)?
                .into_iter();
            self.prof_end(prof_rows, prof_started.elapsed().as_secs_f64() * 1000.0);
            let head = self.heads();
            for ((member, take), first) in members.into_iter().zip(firsts) {
                logits[member].extend(rows.by_ref().take(take));
                let sequence = &mut *batch[member].0;
                let at = R::ON_DEVICE.then_some(head);
                self.after_run(
                    sequence,
                    (first, take, tokens.len()),
                    verify,
                    ends[member],
                    at,
                )?;
            }
        }
    }

    /// Refuses `tokens` on `sequence` when the sequence is another model's, past its capacity or
    /// an uncommitted verify's.
    ///
    /// # Errors
    /// As [Self::check_sequence], and a verify not committed.
    pub(crate) fn admit(
        &self,
        sequence: &DeviceSequence,
        tokens: &[u32],
    ) -> Result<(), BackendError> {
        self.check_sequence(sequence, tokens)?;
        if sequence.pending.is_some() {
            return Err(refusal(
                "the sequence's verify is not committed; commit it before its next step",
            ));
        }
        Ok(())
    }

    /// The LM heads this rank has run: what a greedy step's logits are stamped with.
    pub(crate) fn heads(&self) -> u64 {
        self.heads.load(Ordering::Relaxed)
    }

    /// `sequence` after its run of a token step, rows `first..first + take` of the step's
    /// `step_rows`, of a pass whose rows of it end at `end`: its position, where its last row's
    /// logits stay (`head`, the step's LM head, when they stay on the device), its MTP stash's
    /// last row, and a verify's pending commit (`verify`, more than one row) or the block
    /// drafter's window commits. Answers where the last row's logits stay.
    ///
    /// # Errors
    /// The device refuses a commit.
    pub(crate) fn after_run(
        &self,
        sequence: &mut DeviceSequence,
        (first, take, step_rows): (usize, usize, usize),
        verify: bool,
        end: usize,
        head: Option<u64>,
    ) -> Result<Option<LogitsAt>, BackendError> {
        let position = sequence.position;
        sequence.position += take;
        sequence.logits_at = head.map(|head| LogitsAt {
            rank: self.identity(),
            head,
            row: first + take - 1,
            rows: step_rows,
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
                step_rows,
            });
        } else if let (Some(block), Some(scratch)) = (&self.block, &sequence.block) {
            // A verify's rows wait for its commit, which knows how many the model kept.
            let draft = &block.draft;
            for (row, rows, at) in draft
                .config()
                .window_commits((position, take), end, MAX_ROWS)
            {
                draft.commit(self, scratch, row, rows, at)?;
            }
        }
        Ok(sequence.logits_at)
    }

    /// One step: row `row` runs `tokens[row]`, and `runs` say which sequence's position each row
    /// is. Returns what `R` takes from the LM head a row: its full-vocabulary logits, or its pick
    /// with the logits left on the device. On the device leg every collective, the logits' gather
    /// too, is exchanged on the device, and the step's one host copy is the gathered logits or
    /// the picks.
    ///
    /// The step is its stages in order, the ones a planned walk runs a frame at a time
    /// ([crate::walk::GlmWalk]): the embed, then each layer's entry, token mixer and exit, its
    /// entry, feed-forward and exit, then the tail.
    fn step<R: StepRow>(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        runs: &[Run<'_>],
        tokens: &[u32],
        device_leg: bool,
        head_rows: HeadRows,
    ) -> Result<Vec<R>, BackendError> {
        let set = self.step_set(tp, runs, tokens.len(), device_leg);
        self.step_embed(&set, runs.first().map(|run| run.position), tokens)?;
        let mut fused = false;
        for index in 0..self.layers.len() {
            for ffn in [false, true] {
                self.step_enter(tp, &set, index, ffn, fused)?;
                if ffn {
                    self.step_ffn(index, set.rows)?;
                } else {
                    self.step_mixer(index, runs, set.rows)?;
                }
                fused = self.step_leave(tp, &set, index, ffn)?;
            }
        }
        self.step_tail((tp, gather), &set, runs, tokens, head_rows)
    }

    /// What every stage of a step of `rows` rows over `runs` reads: the device leg, this rank's
    /// rows of a sequence-parallel step ([Self::sequence_parallel]) and the block drafter's
    /// captures.
    pub(crate) fn step_set(
        &self,
        tp: &TpRuntime,
        runs: &[Run<'_>],
        rows: usize,
        device_leg: bool,
    ) -> StepSet {
        let share = self
            .sequence_parallel(tp, runs, rows, device_leg)
            .map(|share| (share.first, share.count, share.live));
        let captures = runs
            .iter()
            .filter_map(|run| {
                run.block
                    .filter(|&(_, from)| from < run.len)
                    .map(|(scratch, from)| (scratch, run.first + from, run.len - from))
            })
            .collect();
        StepSet {
            device_leg,
            rows,
            share,
            captures,
        }
    }

    /// A step's embed: the residual streams of this rank's rows of `tokens`, the first run at
    /// `position`.
    ///
    /// This forward is one named coverage input, and the first decode step through the rank is
    /// the monitor probe (docs/NUMERICS.md section 5): a prompt's prefill is wider than the
    /// few-row band and does not consume it, so the step after it is what a pair measures. With
    /// both switches off, this is one `Option` check a forward.
    pub(crate) fn step_embed(
        &self,
        set: &StepSet,
        position: Option<usize>,
        tokens: &[u32],
    ) -> Result<(), BackendError> {
        let s = &self.scratch;
        let (hc, hidden) = (self.config.hc_mult, self.config.hidden_size);
        let rows = set.rows;
        if let Some(position) = position {
            self.begin_numerics(tokens, position, rows);
        }
        // A sequence-parallel step widens, mixes and folds this rank's rows alone.
        let (first, count) = set
            .share
            .map_or((0, rows), |(first, _, live)| (first, live));
        self.embed_rows(
            &s.step_tokens,
            &tokens[first..first + count],
            s.streams.offset(first * hc * hidden * 4),
            hc,
        )?;
        self.prof_mark("embed");
        Ok(())
    }

    /// Layer `index`'s entry into its token mixer, or with `ffn` into its feed-forward: the
    /// streams' mix in and the norm. `fused` is what the exit before it answered.
    pub(crate) fn step_enter(
        &self,
        tp: &TpRuntime,
        set: &StepSet,
        index: usize,
        ffn: bool,
        fused: bool,
    ) -> Result<(), BackendError> {
        let layer = &self.layers[index];
        let share = set.share(tp);
        if ffn {
            self.mix_in(&layer.ffn_hc, layer.post_norm, set.rows, fused, share)
        } else {
            self.mix_in(&layer.attn_hc, layer.input_norm, set.rows, fused, share)?;
            self.prof_mark("mhc_in");
            Ok(())
        }
    }

    /// Layer `index`'s token mixer over `runs`: its KDA recurrence, or its DSA attention with the
    /// indexer's selection.
    pub(crate) fn step_mixer(
        &self,
        index: usize,
        runs: &[Run<'_>],
        rows: usize,
    ) -> Result<(), BackendError> {
        match &self.layers[index].branch {
            DeviceBranch::Linear(kda) => {
                // The linear layers' scratch, one a KDA layer in layer order.
                let before = self.layers[..index]
                    .iter()
                    .filter(|layer| matches!(layer.branch, DeviceBranch::Linear(_)))
                    .count();
                let own = self.scratch.linear.get(before).ok_or_else(other_kind)?;
                self.kda(kda, index, own, runs, rows)?;
                self.prof_mark("attn_kda");
            }
            DeviceBranch::Sparse(dsa) => {
                self.dsa(dsa, index, runs, rows)?;
                self.prof_mark("attn_dsa");
            }
        }
        Ok(())
    }

    /// Layer `index`'s feed-forward: its dense MLP or its routed experts.
    pub(crate) fn step_ffn(&self, index: usize, rows: usize) -> Result<(), BackendError> {
        match &self.layers[index].ffn {
            DeviceFfn::Dense(mlp) => {
                self.mlp(mlp, self.scratch.partial, rows)?;
                self.prof_mark("mlp");
            }
            DeviceFfn::Moe(moe) => {
                self.moe(moe, Some(index), rows)?;
                self.prof_mark("moe");
            }
        }
        Ok(())
    }

    /// Layer `index`'s exit from its token mixer, or with `ffn` from its feed-forward: the next
    /// reads prefetched, the partials summed and mixed out into the streams, and after the
    /// feed-forward the block drafter's captures of the layer. Answers whether the mix out also
    /// worked out the next entry's projection ([Self::mix_out]).
    pub(crate) fn step_leave(
        &self,
        tp: &TpRuntime,
        set: &StepSet,
        index: usize,
        ffn: bool,
    ) -> Result<bool, BackendError> {
        let s = &self.scratch;
        let streams = (s.streams, s.coeffs, self.config.hc_mult);
        let share = set.share(tp);
        let layer = &self.layers[index];
        if !ffn {
            self.prefetch(&layer.ffn_reads())?;
            return self.mix_out(
                tp,
                streams,
                Some(&layer.ffn_hc),
                set.rows,
                set.device_leg,
                share,
            );
        }
        let next = self.layers.get(index + 1);
        match next {
            Some(next) => self.prefetch(&next.attention_reads())?,
            None => self.prefetch(&[self.lm_head.span()])?,
        }
        let fused = self.mix_out(
            tp,
            streams,
            next.map(|n| &n.attn_hc),
            set.rows,
            set.device_leg,
            share,
        )?;
        self.capture(index, &set.captures)?;
        Ok(fused)
    }

    /// A step's tail: the streams folded and normed, a sequence-parallel step's rows gathered,
    /// the LM head over `head_rows` and the MTP layer's caches of `runs`.
    pub(crate) fn step_tail<R: StepRow>(
        &self,
        (tp, gather): (&TpRuntime, &BitGatherHandle),
        set: &StepSet,
        runs: &[Run<'_>],
        tokens: &[u32],
        head_rows: HeadRows,
    ) -> Result<Vec<R>, BackendError> {
        let s = &self.scratch;
        let config = &self.config;
        let (hc, hidden) = (config.hc_mult, config.hidden_size);
        let rows = set.rows;
        let share = set.share(tp);
        let (first, count) = share.map_or((0, rows), |share| (share.first, share.live));
        let row = |base: Deviceptr, width: usize| base.offset(first * width * 4);
        self.finish_numerics();
        self.prof_mark("tail");
        self.launch(
            self.kernels.fold.kernel(),
            (blocks(count * hidden), 1, 1),
            BLOCK,
            &[
                row(s.streams, hc * hidden).0,
                row(s.head, hidden).0,
                hidden as u64,
                hc as u64,
                count as u64,
                float(self.fold_scale),
            ],
        )?;
        let normed = row(s.head_normed, hidden);
        self.kernels.norm.run(
            self,
            RmsNorm::new(
                row(s.head, hidden).0,
                self.norm.0,
                normed.0,
                hidden,
                config.rms_norm_eps as f32,
            ),
            count,
        )?;
        // The LM head and the MTP layer's caches read every row's normed hidden.
        if let Some(share) = share {
            share.gather(self, normed, s.head_normed, hidden)?;
        }
        let logits = match head_rows {
            HeadRows::All => R::head(self, (tp, gather), s.head_normed, rows, set.device_leg)?,
            HeadRows::Last => R::head(
                self,
                (tp, gather),
                s.head_normed.offset((rows - 1) * hidden * 4),
                1,
                set.device_leg,
            )?,
        };
        self.fill(runs, tokens)?;
        Ok(logits)
    }

    /// Layer `index`'s finished rows into the block drafter's target rows of each capture
    /// ([StepSet]), when the drafter reads that layer: the mean of each row's streams.
    fn capture(
        &self,
        index: usize,
        captures: &[(DraftScratch, usize, usize)],
    ) -> Result<(), BackendError> {
        let Some(block) = &self.block else {
            return Ok(());
        };
        let (hc, hidden) = (self.config.hc_mult, self.config.hidden_size);
        for (scratch, first, rows) in captures {
            let streams = self.scratch.streams.offset(first * hc * hidden * 4);
            block
                .draft
                .capture(self, scratch, index, (streams, hc), 0, *rows)?;
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
            Some(head) => gemv_e4m3(
                self,
                self.kernels.gemv_e4m3.kernel(),
                &head,
                input,
                own,
                rows,
            )?,
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

    /// [Self::argmax_rows], and each pick's softmax probability into `probs`, an f32 a row.
    fn argmax_prob_rows(
        &self,
        rows: usize,
        tokens: Deviceptr,
        probs: Deviceptr,
    ) -> Result<(), BackendError> {
        self.launch(
            self.kernels.argmax_prob.kernel(),
            (rows as u32, 1, 1),
            ARGMAX_BLOCK,
            &[
                self.scratch.logits.0,
                tokens.0,
                probs.0,
                self.layout.world_size as u64,
                rows as u64,
                self.lm_head.rows as u64,
            ],
        )
    }

    /// The argmax of each of `rows` rows of the gathered logits into `tokens`, a u32 a row.
    fn argmax_rows(&self, rows: usize, tokens: Deviceptr) -> Result<(), BackendError> {
        self.launch(
            self.kernels.argmax.kernel(),
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

    /// The embeddings of `tokens`, one a row from `out`, each row `copies` copies of it: one
    /// launch, grid row `t` widening `tokens[t]`, over the ids staged in `slab`, which it reads in
    /// place. A launch a token cost a prefill chunk 4,096 launches of a few microseconds each.
    fn embed_rows(
        &self,
        slab: &HostMapped,
        tokens: &[u32],
        out: Deviceptr,
        copies: usize,
    ) -> Result<(), BackendError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let hidden = self.config.hidden_size;
        self.cuda
            .host_wait(&self.tokens_staged.0)
            .map_err(cuda_error)?;
        slab.write_at(0, bytemuck::cast_slice(tokens))
            .map_err(cuda_error)?;
        Embed::new(self.kernels.embed, self.embed, hidden, copies)?.rows(
            self,
            out,
            slab.device(),
            tokens.len(),
        )?;
        self.cuda
            .record_event_on(&self.stream, &self.tokens_staged.0)
            .map_err(cuda_error)
    }

    /// The MTP layer's input embeddings of a draft's iteration `iteration`, one a row of `rows`,
    /// from the tokens the device holds for it in `drafted`.
    fn embed_drafted(&self, iteration: usize, rows: usize) -> Result<(), BackendError> {
        let s = &self.scratch;
        let hidden = self.config.hidden_size;
        let tokens = s.drafted.device().offset(iteration * MAX_ROWS * 4);
        Embed::new(self.kernels.embed, self.embed, hidden, 1)?.rows(self, s.mtp_embed, tokens, rows)
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
        let norm = RmsNorm::new(
            x.0,
            weight.0,
            out.0,
            hidden,
            self.config.rms_norm_eps as f32,
        );
        self.kernels
            .norm
            .strided(self, norm, rows, (src_stride, dst_stride))
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
        // The runs' MTP rows one after another, each run's tokens past the first it has no hidden
        // for: their embeddings are one launch once the runs are laid out.
        let mut ids = Vec::new();
        for run in runs {
            let Some(drafter) = run.drafter else {
                continue;
            };
            let skip = usize::from(drafter.last.is_none());
            ids.extend_from_slice(&tokens[run.first + skip..run.first + run.len]);
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
            self.embed_rows(&s.mtp_tokens, &ids, s.mtp_embed, 1)?;
            self.mtp_input(mtp, rows)?;
            self.dsa_cache(&mtp.dsa, 0, &filled, rows)?;
        }
        // Each run's last [MAX_ROWS] rows into its stash: a prefill chunk's earlier rows are never
        // read again, and the stash holds no more.
        for run in runs {
            if let Some(drafter) = run.drafter {
                let kept = run.len.min(MAX_ROWS);
                self.launch(
                    self.kernels.copy_flat.kernel(),
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
        self.draft_while(tp, gather, batch, count, None)
    }

    /// [Self::draft] of at most `count` drafts, where the MTP layer stops early: after each draft
    /// short of `count`, `more` sees each row's probabilities so far and the layer drafts another
    /// only while it asks for one for some row. Each pass is the same whatever `more` answers, so
    /// a draft it lets run is the token [Self::draft] would have drafted; every rank sees the same
    /// probabilities and stops at the same pass. Reading them waits for the stream once a pass. A
    /// block drafter proposes its whole block in one pass and ignores `more`.
    ///
    /// # Errors
    /// As [Self::draft].
    pub fn draft_while(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        batch: &mut [(&mut DeviceSequence, u32)],
        count: usize,
        mut more: Option<KeepDrafting<'_>>,
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
        let probs_at = |iteration: usize, row: usize| (iteration * MAX_ROWS + row) * 4;
        let mut drafted_count = count;
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
            self.mix_out(tp, residual, None, rows, device_leg, None)?;
            self.norm_rows(
                (s.mtp_x, hidden),
                mtp.post_norm,
                (s.collapsed, hidden),
                rows,
            )?;
            // The MTP layer is a draft pass, not a decoder layer: it records no probe.
            self.moe(&mtp.moe, None, rows)?;
            match self.draft_head {
                Some(head) => self.prefetch(&[(head.weight, head.rows * head.columns)])?,
                None => self.prefetch(&[self.lm_head.span()])?,
            }
            self.mix_out(tp, residual, None, rows, device_leg, None)?;
            self.norm_rows((s.mtp_x, hidden), mtp.head_norm, (s.mtp_out, hidden), rows)?;
            if device_leg {
                self.lm_head(tp, s.mtp_out, rows, (true, true))?;
                self.argmax_prob_rows(
                    rows,
                    s.drafted.device().offset(at(iteration + 1, 0)),
                    s.draft_probs.device().offset(probs_at(iteration + 1, 0)),
                )?;
            } else {
                let logits = self.logits(tp, gather, s.mtp_out, rows, (false, true))?;
                for (row, values) in logits.iter().enumerate() {
                    let token =
                        argmax(values).ok_or_else(|| refusal("a draft's logits are empty"))?;
                    s.drafted
                        .write_at(at(iteration + 1, row), &(token as u32).to_le_bytes())
                        .map_err(cuda_error)?;
                    s.draft_probs
                        .write_at(
                            probs_at(iteration + 1, row),
                            &pick_probability(values, token).to_le_bytes(),
                        )
                        .map_err(cuda_error)?;
                }
            }
            let Some(more) = more.as_deref_mut() else {
                continue;
            };
            if iteration + 1 == count {
                break;
            }
            self.cuda
                .synchronize_stream(&self.stream)
                .map_err(cuda_error)?;
            let mut wanted = false;
            for row in 0..rows {
                let probs = Self::read_probs(&s.draft_probs, row, iteration + 1, probs_at)?;
                wanted |= more(&probs);
            }
            if !wanted {
                drafted_count = iteration + 1;
                break;
            }
        }
        self.cuda
            .synchronize_stream(&self.stream)
            .map_err(cuda_error)?;
        let count = drafted_count;
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
        for (row, (sequence, _)) in batch.iter_mut().enumerate() {
            sequence.confidence = Self::read_probs(&s.draft_probs, row, count, probs_at)?;
        }
        Ok(drafts)
    }

    /// Row `row`'s probabilities of its first `count` MTP drafts, `[1..=count][row]` of `probs`.
    fn read_probs(
        probs: &HostMapped,
        row: usize,
        count: usize,
        at: impl Fn(usize, usize) -> usize,
    ) -> Result<Vec<f32>, BackendError> {
        let mut word = [0u8; 4];
        (1..=count)
            .map(|iteration| {
                probs
                    .read_at(at(iteration, row), &mut word)
                    .map_err(cuda_error)?;
                Ok(f32::from_le_bytes(word))
            })
            .collect()
    }

    /// [Self::draft_while] under `spec`'s depth rule, after `anchor`, for one sequence: the drafts
    /// it would verify and the drafter's confidence in each draft it proposed (as many as
    /// [DeviceSequence::draft_confidence] holds, which may be more than the drafts kept).
    ///
    /// The drafter proposes up to the speculator's depth within `room`; an adaptive policy stops
    /// the MTP layer once the drafts so far are all it would verify, and cuts the chain to the
    /// depth it picks from the proposed drafts' own confidences. Every serving shape of the arm
    /// drafts here, so they all verify the same chains.
    ///
    /// # Errors
    /// As [Self::draft].
    pub fn draft_chain(
        &self,
        tp: &TpRuntime,
        gather: &BitGatherHandle,
        sequence: &mut DeviceSequence,
        anchor: u32,
        spec: &sparkflash_model::Speculator,
        room: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
        let count = spec.max_depth().min(room);
        if count == 0 {
            sequence.confidence.clear();
            return Ok((Vec::new(), Vec::new()));
        }
        let mut more = |probs: &[f32]| spec.depth(probs, count) >= probs.len();
        let more: Option<KeepDrafting<'_>> = if spec.is_adaptive() {
            Some(&mut more)
        } else {
            None
        };
        let mut drafted = {
            let mut batch = [(&mut *sequence, anchor)];
            self.draft_while(tp, gather, &mut batch, count, more)?
                .pop()
                .unwrap_or_default()
        };
        let confidence = sequence.draft_confidence().to_vec();
        drafted.truncate(spec.depth(&confidence, count));
        Ok((drafted, confidence))
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
                self.kernels.copy_flat.kernel(),
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

/// Every launch of the rank is on its stream.
impl Launcher for DeviceRank {
    fn launch(
        &self,
        kernel: impl Into<Kernel>,
        (grid, block): Launch,
        args: &[u64],
    ) -> Result<(), BackendError> {
        let kernel = kernel.into();
        self.cuda
            .launch_on(&self.stream, kernel, grid, block, args)
            .map_err(|error| cuda_error(format!("{}: {error}", kernel.name())))
    }
}

/// The block drafter's launches go on the rank's stream, and its BF16 GEMVs through the rank's.
impl GemvDevice for DeviceRank {
    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let linear = DeviceLinear::view(matrix.pointer, matrix.rows, matrix.columns, false, None);
        DeviceRank::gemv(self, &linear, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.scratch.partials, self.scratch.counters)
    }
}

impl DraftDevice for DeviceRank {}

/// The rank as the block drafter's device while it scores a block: the rank's launches, GEMVs and
/// tiled scratch, and the TP collective that sums the sharded drafter's partials as
/// [DeviceRank::mix_out] sums the model's, into one stream by `mtp_coeffs` (`x + reduced`).
struct Scoring<'r> {
    rank: &'r DeviceRank,
    tp: &'r TpRuntime,
    device_leg: bool,
}

impl Launcher for Scoring<'_> {
    fn launch(
        &self,
        kernel: impl Into<Kernel>,
        launch: Launch,
        args: &[u64],
    ) -> Result<(), BackendError> {
        Launcher::launch(self.rank, kernel, launch, args)
    }
}

impl GemvDevice for Scoring<'_> {
    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        GemvDevice::gemv(self.rank, matrix, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        GemvDevice::tiled_scratch(self.rank)
    }
}

impl DraftDevice for Scoring<'_> {
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
        let _ = self.cuda.make_current();
        let _ = self.cuda.synchronize_stream(&self.stream);
        self.cuda.destroy_event(&self.tokens_staged.0);
        if let Some(prefetch) = &self.prefetch {
            self.cuda.destroy_side_stream(&prefetch.side);
        }
        if let Some(block) = &self.block {
            block.draft.free(&self.cuda);
        }
        // A sequence still alive gives its chunks back when it goes.
        self.lanes.drain(&self.cuda);
        self.cuda.destroy_run_stream(&self.stream);
    }
}

/// The softmax probability of `values[pick]`, as q4e_argmax_prob_slices takes it: 1 over the sum
/// of e^(value - best), NaN values left out, and 0 when the best is not finite.
fn pick_probability(values: &[f32], pick: usize) -> f32 {
    let best = values.get(pick).copied().unwrap_or(f32::NAN);
    if !best.is_finite() {
        return 0.0;
    }
    let total: f32 = values
        .iter()
        .filter(|value| !value.is_nan())
        .map(|value| (value - best).exp())
        .sum();
    if total > 0.0 { 1.0 / total } else { 0.0 }
}
#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::architecture::AttentionKind;

    /// The per-position sparse caches [SparseParts::plan] lays out for `config`: every
    /// backbone MLA layer, each layer that owns an indexer, and the MTP stage (which the device
    /// always builds with its own indexer).
    fn allocated_sparse_per_token(config: &Glm5NextConfig, quantise: bool) -> u64 {
        let (latent_elements, latent_scales) = latent_row(config, quantise);
        let latent =
            latent_elements as u64 * if quantise { 1 } else { 4 } + latent_scales as u64 * 4;
        let index_dim = config.index_head_dim as u64;
        let indexer = index_dim * 8 + (index_dim * 4).div_ceil(config.index_kpool as u64);
        let mla_layers: Vec<_> = config
            .architecture
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Mla { .. }))
            .collect();
        let owns_indexer = mla_layers
            .iter()
            .filter(|layer| {
                layer
                    .attention
                    .indexer
                    .as_ref()
                    .is_some_and(|indexer| indexer.owns_keys)
            })
            .count() as u64;
        let stages = config
            .architecture
            .mtp
            .as_ref()
            .map_or(0, |mtp| mtp.stages as u64);
        mla_layers.len() as u64 * latent + owns_indexer * indexer + stages * (latent + indexer)
    }

    /// A copy of the tiny fixture whose latent is one whole FP8 scale tile wide.
    fn tile_wide_tiny() -> Glm5NextConfig {
        let mut config = Glm5NextConfig::tiny();
        config.kv_lora_rank = FP8_BLOCK;
        for layer in &mut config.architecture.layers {
            if let AttentionKind::Mla { kv_lora_rank, .. } = &mut layer.attention.kind {
                *kv_lora_rank = FP8_BLOCK;
            }
        }
        config
    }

    /// A decode step's FP8 projection, through the launch the arm makes, is the prompt tile's
    /// row: the same E4M3 weights, GLM's f32 128 x 128 block scales and f32 activations, the
    /// body over every part of a fused projection, and `glm_prefill_fp8b128_mma` over the same
    /// operands. Each row of each launch is compared bit for bit.
    #[test]
    #[ignore = "needs a CUDA device and the GLM decode and prefill images"]
    fn a_decode_step_on_the_token_body_is_the_prefill_tiles_row() {
        use sparkflash_kernels::m4_quant::fp8_e4m3_encode;
        use sparkflash_kernels::test_io;

        let Some(cuda) = test_io::cuda_or_skip(
            (!cubin_available() || images::glm_prefill_cubin().is_empty())
                .then_some("the GLM decode and prefill images"),
        ) else {
            return;
        };
        let cuda = CudaHandle::new(cuda);
        let run = cuda.run_stream().expect("a run stream");
        let tile = Bound::new(
            &cuda,
            Image::GlmPrefill,
            images::glm_prefill_cubin(),
            "glm_prefill_fp8b128_mma",
            Precision::Fp8E4M3,
        )
        .expect("glm_prefill_fp8b128_mma")
        .kernel();
        let body = Bound::new(
            &cuda,
            Image::GlmDecode,
            images::glm_decode_cubin(),
            "glm_decode_fp8b128_rows",
            Precision::Fp8E4M3,
        )
        .expect("glm_decode_fp8b128_rows")
        .kernel();

        const COLS: usize = 256;
        const INNER: usize = 512;
        const TOKENS: usize = MAX_ROWS;
        let weights: Vec<u8> = (0..COLS * INNER)
            .map(|at| {
                let value = ((at as u64 * 37 + 11) % 251) as f32 * 0.013 - 1.6;
                fp8_e4m3_encode(value)
            })
            .collect();
        let scales: Vec<f32> = (0..COLS.div_ceil(128) * (INNER / 128))
            .map(|at| 0.5 + ((at as u64 * 13 + 5) % 17) as f32 * 0.0625)
            .collect();
        let activations: Vec<f32> = (0..TOKENS * INNER)
            .map(|at| {
                let mixed = (at as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((mixed >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();

        let weight = test_io::upload::<u8>(&cuda, &weights);
        let scale = test_io::upload::<f32>(&cuda, &scales);
        let input = test_io::upload::<f32>(&cuda, &activations);
        let tile_out = test_io::upload::<f32>(&cuda, &vec![0.0; TOKENS * COLS]);
        let body_out = test_io::upload::<f32>(&cuda, &vec![0.0; TOKENS * COLS]);
        // Two whole-tile parts, as the MLA's fused projections are: the body's part offsets,
        // weight slice and block-scale slice all have to line up with the tile's full matrix.
        let linear = DeviceLinear::view(weight, COLS, INNER, true, Some(scale)).parted([
            COLS / 2,
            COLS,
            COLS,
        ]);

        for count in 1..=TOKENS {
            cuda.launch(
                tile,
                (COLS.div_ceil(128) as u32, count.div_ceil(128) as u32, 1),
                (256, 1, 1),
                &[
                    weight.0,
                    scale.0,
                    input.0,
                    tile_out.0,
                    COLS as u64,
                    INNER as u64,
                    count as u64,
                ],
            )
            .expect("the prompt tile");
            let planes = linear.fp8_planes().expect("an FP8 projection's planes");
            launch_fp8_token_body(&cuda, &run, body, planes, &linear, input, body_out, count)
                .expect("the token body");
            let expected: Vec<u32> = test_io::download::<u32>(&cuda, &tile_out, count * COLS);
            let got: Vec<u32> = test_io::download::<u32>(&cuda, &body_out, count * COLS);
            // The tile writes [row][column]; the body writes each part's rows together, the
            // layout the GEMV's callers read ([DeviceLinear::part]).
            let starts = [0, linear.parts[0], linear.parts[1], linear.parts[2], COLS];
            for (part, bounds) in starts.windows(2).enumerate() {
                let (first, end) = (bounds[0], bounds[1]);
                for token in 0..count {
                    for column in first..end {
                        let tile_at = token * COLS + column;
                        let body_at = count * first + token * (end - first) + (column - first);
                        assert_eq!(
                            got[body_at],
                            expected[tile_at],
                            "{count} rows, part {part}, token {token}, column {column}: the body \
                             {} and the tile {} must agree bit for bit",
                            f32::from_bits(got[body_at]),
                            f32::from_bits(expected[tile_at])
                        );
                    }
                }
            }
        }
    }

    /// The decode path's routing boundary: one to eight rows take the tile's FP8 body, and the
    /// ninth keeps the bulk GEMM. A projection whose shape the body does not take is not routed
    /// to it even inside the band.
    #[test]
    fn the_fp8_token_body_takes_one_to_eight_rows() {
        assert_eq!(MAX_ROWS, 8, "the body's band is the arm's eight-row step");
        for rows in 1..=MAX_ROWS {
            assert!(fp8_token_body_rows(rows), "{rows} rows take the body");
        }
        for rows in [0, MAX_ROWS + 1, 512] {
            assert!(!fp8_token_body_rows(rows), "{rows} rows do not");
        }

        // The operand is the weight's code planes: a BF16 weight has none, so the rule is never
        // reached with it (the type says so; this pins the refusal at the accessor).
        let fits = |weight: u8, rows: usize, parts: [usize; 3]| {
            let scales = (weight != 0).then_some(Deviceptr(0));
            let linear =
                DeviceLinear::view(Deviceptr(0), rows, 512, weight != 0, scales).parted(parts);
            linear
                .fp8_planes()
                .is_some_and(|planes| fp8_token_body_fits(planes, &linear, 8))
        };
        assert!(fits(1, 8, [8, 8, 8]), "one eight-row part");
        assert!(fits(1, 256, [128, 256, 256]), "whole-tile parts");
        assert!(!fits(0, 8, [8, 8, 8]), "a BF16 weight has no FP8 operand");
        assert!(
            !fits(1, 196, [128, 196, 196]),
            "a part that is not whole eight-column writes"
        );
        assert!(
            !fits(1, 144, [128, 136, 144]),
            "a part whose scale block does not start on 128 rows"
        );
    }

    /// The shipped `glm53_flash.toml` leaves `decode.tensor_cores` unset, so version 3's
    /// one-to-eight-row FP8 parts take the tiled row GEMV: version 1's bits. The switch is
    /// still read as a typed field; the serve check refuses `on` on this version.
    #[test]
    fn the_shipped_profile_decodes_on_the_row_gemv() {
        let mut profile =
            sparkflash_model::ModelProfile::from_toml(include_str!("../config/glm53_flash.toml"))
                .expect("the shipped GLM profile");
        assert_eq!(profile.decode.tensor_cores, None, "the file names no value");
        profile
            .apply_env(|name| (name == "SPARKFLASH_DECODE_MMA").then(|| "on".to_owned()))
            .expect("on is a switch value");
        assert_eq!(profile.decode.tensor_cores, Some(true));
    }

    #[test]
    fn the_single_rank_plan_matches_the_latent_the_allocator_sizes() {
        // The tiny fixture's tensor plan stores `kv_b_proj.weight` BF16, so Auto (the allocator's
        // default) is the f32 latent even though `fp8_block` tiles the other projections.
        let config = Glm5NextConfig::tiny();
        let auto = LatentCache::Auto.quantises(&config, false);
        assert!(!auto, "a BF16 kv_b does not select the E4M3 latent");
        assert_eq!(
            latent_row(&config, auto),
            (config.kv_lora_rank, 0),
            "the allocator sizes the f32 row, not a 128-wide E4M3 one"
        );
        assert_eq!(
            crate::single_rank_kv_plan(&config).per_token(),
            allocated_sparse_per_token(&config, auto),
            "the admitted plan and `SparseParts::plan` agree on the tiny fixture"
        );

        // With a whole-tile latent the per-tensor choice changes the allocation, and the plan
        // follows the same condition rather than the checkpoint-wide flag.
        let wide = tile_wide_tiny();
        let bf16_kv_b = LatentCache::Auto.quantises(&wide, false);
        let block_kv_b = LatentCache::Auto.quantises(&wide, true);
        assert!(!bf16_kv_b, "BF16 kv_b keeps the f32 latent");
        assert!(block_kv_b, "a block-scaled kv_b selects the E4M3 latent");
        assert_eq!(latent_row(&wide, bf16_kv_b), (FP8_BLOCK, 0));
        assert_eq!(latent_row(&wide, block_kv_b), (FP8_BLOCK, 1));
        assert_eq!(
            crate::single_rank_kv_plan(&wide).per_token(),
            allocated_sparse_per_token(&wide, bf16_kv_b)
        );
        assert_eq!(
            crate::kv_plan_with_latent(&wide, block_kv_b).per_token(),
            allocated_sparse_per_token(&wide, block_kv_b)
        );
        assert_eq!(
            allocated_sparse_per_token(&wide, bf16_kv_b)
                - allocated_sparse_per_token(&wide, block_kv_b),
            3 * (FP8_BLOCK as u64 * 4 - (FP8_BLOCK as u64 + 4)),
            "two backbone MLA layers and the MTP stage move to the f32 latent"
        );
    }
}
