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

//! DFlash 2 block drafting on the device (kernels/dflash2_sm121.cu), for the drafters
//! [crate::dflash] describes: Qwen3.8-Flash-Next's on this arm, and GLM-5.3-Flash's on each rank
//! of its tensor-parallel world. An arm runs the drafter through [DraftDevice], which puts the
//! drafter's launches on the arm's stream and its BF16 GEMVs through the arm's own.
//!
//! A sequence asks two things of the drafter. After every forward it commits the positions the
//! target kept: their captured rows go through the drafter's context path and land in each
//! layer's key and value ring. Then it drafts the block after the token the target chose: the
//! anchor and `block_size - 1` mask rows run through the five layers at once, the target's own
//! head scores the mask rows, and the selector walks their top candidates into one chain.
//!
//! The drafter's projections, norms, rotary and SwiGLU are the token image's kernels on the
//! drafter's weights; its embedding table is the target's, and its head an E4M3 copy of the
//! target's. The block's projections are E4M3 as well, at half the bytes a draft reads: the
//! drafts are only proposals, and the context path, which also runs a prefill's rows, keeps its
//! BF16 matrices for the GEMM. A verify then runs the
//! chain as one forward of the target, whose rows compute as the decode steps they stand in for,
//! so a draft only ever changes how many positions a forward covers, never what they hold.
//!
//! # Attribution
//!
//! * Block-diffusion speculative drafting is the DFlash method of Jian Chen, Yesheng Liang and
//!   Zhijian Liu ("DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026), with its
//!   reference implementation in [z-lab/dflash](https://github.com/z-lab/dflash) (MIT,
//!   Copyright (c) 2026 Z Lab).
//! * The DFlash 2 drafter for Qwen3.8-Flash-Next is Inco AI's
//!   [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
//!   (Apache-2.0). Its weights are loaded from the user's copy, not redistributed.
//! * The forward and the verify protocol follow SGLang's DFlash support
//!   ([sgl-project/sglang](https://github.com/sgl-project/sglang), Apache-2.0). No source is
//!   copied. See the root NOTICE file.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model::{BackendError, StepCost};
use sparkflash_weights::safetensors::{
    CheckpointIndex, MAX_HEADER_BYTES, SafeTensorsHeader, SafeTensorsSource,
};

use super::{
    DeviceSequence, GEMV_MAX_BATCH, GpuMat, GpuMatE4m3, Part, Uploader, tiled_gemv_e4m3_splits,
    tiled_operands,
};
use crate::config::Qwen4ExpConfig;
use crate::dflash::{Dflash2Config, DraftTarget};

/// The drafter's own errors, whichever arm runs it.
const DOMAIN: &str = "dflash2";

/// The modelled wall of a DFlash 2 step on an RTX Spark (GB10-class, 48 SMs) that verifies
/// `depth` drafts (ms): the one-token step with the drafter's block at 59 ms, and 8.4 ms a verified
/// draft (mostly the experts its row adds). Measured over the greedy benchmark prompts at block 8:
/// 48.3 ms a token undrafted and 109-131 ms a seven-draft verify. The first step assumes 34 tok/s,
/// the drafter's mean over those prompts.
pub(super) const DFLASH2_STEP_COST: StepCost = StepCost {
    step_ms: 59.0,
    row_ms: 8.4,
    prior_tokens_per_ms: 0.034,
};

/// The drafter cubin build.rs embeds. Empty on a CPU-only build.
#[cfg(sparkflash_qwen4_cubin)]
const DFLASH2_CUBIN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_DFLASH2_CUBIN"));
#[cfg(not(sparkflash_qwen4_cubin))]
const DFLASH2_CUBIN: &[u8] = &[];

/// The drafter cubin this build embeds, for kernel-level tests.
#[must_use]
pub fn embedded_dflash2_cubin() -> &'static [u8] {
    DFLASH2_CUBIN
}

/// The released drafter is one safetensors file with no index.
const DRAFT_SHARD: &str = "model.safetensors";

/// Threads per top-k block and the values each holds (Q4D_TOPK_PER_THREAD), so a first-stage
/// block covers this many ids.
const TOPK_THREADS: usize = 256;
const TOPK_PER_THREAD: usize = 8;
const TOPK_CHUNK: usize = TOPK_THREADS * TOPK_PER_THREAD;

/// Threads per q4d_attn block: eight warps (Q4D_ATTN_WARPS) splitting the keys.
const ATTN_THREADS: u32 = 256;

/// What the drafter asks of the arm that runs it: launches in order on the arm's stream, the
/// arm's BF16 GEMV, the split sums and tile counters its tiled GEMVs share, and a profiling mark.
/// Qwen's session and GLM's rank are the two.
pub trait DraftDevice {
    /// Launch `kernel` on the arm's stream.
    ///
    /// # Errors
    /// When the device refuses the launch.
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError>;

    /// `output = matrix input` for `rows` rows of `matrix.columns` floats.
    ///
    /// # Errors
    /// When the arm has no GEMV for the shape or the device refuses a launch.
    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError>;

    /// The tiled GEMVs' split sums ([super::TILED_GEMV_PARTIALS] floats) and per-tile counters
    /// ([super::TILED_GEMV_BLOCKS], zero between launches), which the drafter's E4M3 GEMVs use
    /// as the arm's own do.
    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr);

    /// Charge the time since the last mark to `name`, when the arm profiles.
    ///
    /// # Errors
    /// When the stream cannot be drained.
    fn mark(&self, _name: &'static str) -> Result<(), BackendError> {
        Ok(())
    }

    /// `residual += partial` over `rows` rows of `width` floats, where `partial` is this rank's
    /// share of a drafter sharded over a tensor-parallel world ([GpuDflash2::load]): every rank's
    /// partials summed in rank order, so every rank adds the same bits.
    ///
    /// # Errors
    /// An arm that runs the drafter whole has no world to sum over and refuses; otherwise when
    /// the collective fails or the device refuses a launch.
    fn reduce_add(
        &self,
        _partial: Deviceptr,
        _residual: Deviceptr,
        _rows: usize,
        _width: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::new(
            DOMAIN,
            "this arm runs the drafter whole; a sharded drafter's partials need a world to sum over",
        ))
    }
}

/// The drafter's kernels: its own (kernels/dflash2_sm121.cu) and the token image's that it runs
/// on its own weights, loaded for the drafter so every arm runs the same ones.
#[derive(Debug, Clone, Copy)]
struct Dflash2Kernels {
    capture: Kernel,
    block_embed: Kernel,
    conv: Kernel,
    kv_ring: Kernel,
    attn: Kernel,
    topk_partial: Kernel,
    topk_final: Kernel,
    select: Kernel,
    quantize: Kernel,
    gemv_e4m3: Kernel,
    rmsnorm: Kernel,
    rope: Kernel,
    swiglu: Kernel,
    add: Kernel,
    copy_flat: Kernel,
}

// SAFETY: a Kernel is a function handle loaded from the cubin; the handle is immutable and each
// arm launches it on its own stream.
unsafe impl Send for Dflash2Kernels {}
// SAFETY: as above.
unsafe impl Sync for Dflash2Kernels {}

impl Dflash2Kernels {
    fn load(cuda: &Cuda) -> Result<Self, String> {
        let token = super::embedded_cubin();
        if DFLASH2_CUBIN.is_empty() || token.is_empty() {
            return Err(
                "this build has no Qwen4-Exp DFlash 2 cubin: rebuild on a host with nvcc so \
                 build.rs compiles crates/sparkflash-model-qwen4-exp/kernels/dflash2_sm121.cu"
                    .to_owned(),
            );
        }
        let named = |cubin: &[u8], name: &'static str| -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(cubin, name)?;
            Ok(kernel)
        };
        Ok(Self {
            capture: named(DFLASH2_CUBIN, "q4d_capture")?,
            block_embed: named(DFLASH2_CUBIN, "q4d_block_embed")?,
            conv: named(DFLASH2_CUBIN, "q4d_conv")?,
            kv_ring: named(DFLASH2_CUBIN, "q4d_kv_ring")?,
            attn: named(DFLASH2_CUBIN, "q4d_attn")?,
            topk_partial: named(DFLASH2_CUBIN, "q4d_topk_partial")?,
            topk_final: named(DFLASH2_CUBIN, "q4d_topk_final")?,
            select: named(DFLASH2_CUBIN, "q4d_select")?,
            quantize: named(token, "q4e_quantize_e4m3_rows")?,
            gemv_e4m3: named(token, "q4e_gemv_tiled_e4m3")?,
            rmsnorm: named(token, "q4e_rmsnorm")?,
            rope: named(token, "q4e_rope")?,
            swiglu: named(token, "q4e_swiglu")?,
            add: named(token, "q4e_add")?,
            copy_flat: named(token, "q4e_copy_flat")?,
        })
    }
}

/// One sublayer's grouped convolution: the checkpoint's [2][taps][hidden] base kernel and the
/// projection that turns the sublayer's input into per-row, per-group corrections to it.
#[derive(Debug)]
struct GpuConv {
    base: Deviceptr,
    projection: GpuMatE4m3,
}

#[derive(Debug)]
struct GpuDraftLayer {
    input_norm: Deviceptr,
    post_norm: Deviceptr,
    attention_conv: GpuConv,
    mlp_conv: GpuConv,
    q_proj: GpuMatE4m3,
    /// The context path projects a forward's rows into the rings with these too.
    k_proj: GpuMat,
    v_proj: GpuMat,
    o_proj: GpuMatE4m3,
    q_norm: Deviceptr,
    k_norm: Deviceptr,
    gate_proj: GpuMatE4m3,
    up_proj: GpuMatE4m3,
    down_proj: GpuMatE4m3,
}

/// The drafter's resident weights and kernels: about 0.8 GB (the selector's codebooks are a third
/// of it). It scores with the E4M3 copy of the head the arm gives it (0.64 GB of Qwen3.8's whole
/// head), which the arm keeps.
#[derive(Debug)]
pub struct GpuDflash2 {
    /// The drafter's configuration as this rank runs it: its share of the attention heads and of
    /// the MLP's width ([Self::load]).
    pub(super) config: Dflash2Config,
    /// The ranks the drafter's heads and MLP are sharded over, 1 when it runs whole.
    world: usize,
    kernels: Dflash2Kernels,
    /// The target's head, or a rank's slice of it, as E4M3, which scores the mask rows: the arm's
    /// copy, which it frees.
    head: GpuMatE4m3,
    fc: GpuMat,
    hidden_norm: Deviceptr,
    norm: Deviceptr,
    layers: Vec<GpuDraftLayer>,
    hidden_projection: GpuMat,
    predecessor: Deviceptr,
    successor: Deviceptr,
}

impl GpuDflash2 {
    /// The drafter `SPARKFLASH_QWEN4_DFLASH2` names, or the canonical
    /// `incoai/Qwen3.8-Flash-Next-DFlash2` snapshot in the Hugging Face cache, or none when
    /// neither is configured. `head` is the target's, of which it scores with an E4M3 copy that
    /// lasts as the drafter does, the arm's whole life.
    ///
    /// # Errors
    /// As [Self::load], and when the head cannot be quantized.
    pub(super) fn from_env(
        cuda: &Cuda,
        target: &Qwen4ExpConfig,
        head: &GpuMat,
    ) -> Result<Option<Self>, BackendError> {
        let Some(directory) = crate::dflash::drafter_directory() else {
            return Ok(None);
        };
        let loaded = cuda
            .load_named(super::embedded_cubin(), "q4e_quantize_e4m3_rows")
            .and_then(|(_, quantize)| GpuMatE4m3::quantize(cuda, quantize, head, &mut 0))
            .and_then(|head| {
                Self::load(cuda, &directory, target, &head, (0, 1)).inspect_err(|_| head.free(cuda))
            });
        loaded.map(Some).map_err(|error| {
            BackendError::new(
                "qwen4_dflash2",
                format!("drafter {}: {error}", directory.display()),
            )
        })
    }

    /// The drafter in `directory` for `target`, scoring its mask rows with `head`, an E4M3 copy
    /// ([GpuMatE4m3::quantize]) of the target's BF16 head or of one rank's slice of its rows, whose
    /// candidates then carry the ids the slice starts at ([Self::score]). The caller keeps the
    /// copy and frees it after the drafter.
    ///
    /// `(rank, world)` shards the block's layers over a tensor-parallel world as the target's
    /// are: the rank holds its `1 / world` of the query and key-value heads (its share of every
    /// layer's ring) and of the MLP's width, with `o_proj` and `down_proj` cut along their input
    /// columns, so each sublayer leaves a partial that [DraftDevice::reduce_add] sums over the
    /// ranks. `(0, 1)` runs it whole. The E4M3 parts hold the whole matrices' codes and row
    /// scales, so only the sums' order sets a sharded drafter apart from a whole one.
    ///
    /// # Errors
    /// A drafter that cannot be read, that is not a DFlash 2 checkpoint this arm runs, or that
    /// was trained for another target; a head of another width or more rows than the
    /// vocabulary; heads or an MLP width the world does not divide; a missing kernel or a device
    /// allocation failure.
    pub fn load(
        cuda: &Cuda,
        directory: &Path,
        target: impl Into<DraftTarget>,
        head: &GpuMatE4m3,
        (rank, world): (usize, usize),
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(directory.join("config.json"))
            .map_err(|error| format!("cannot read config.json: {error}"))?;
        let whole = Dflash2Config::from_json(&text).map_err(|error| error.to_string())?;
        whole
            .check_target(target)
            .map_err(|error| error.to_string())?;
        if rank >= world
            || !whole.num_key_value_heads.is_multiple_of(world)
            || !whole.intermediate_size.is_multiple_of(world)
        {
            return Err(format!(
                "rank {rank} of {world} cannot hold a share of {} key-value heads and an MLP {} wide",
                whole.num_key_value_heads, whole.intermediate_size
            ));
        }
        // The heads divide by the key-value heads ([Dflash2Config::from_json]), so by the world.
        let config = Dflash2Config {
            num_attention_heads: whole.num_attention_heads / world,
            num_key_value_heads: whole.num_key_value_heads / world,
            intermediate_size: whole.intermediate_size / world,
            ..whole.clone()
        };
        let rows = |width: usize| Part::Rows(rank * width, width);
        let columns = |width: usize| Part::Columns(rank * width, width);
        let (q_width, kv_width) = (config.q_width(), config.kv_width());
        let intermediate = config.intermediate_size;
        if head.columns != config.hidden_size || !(1..=config.vocab_size).contains(&head.rows) {
            return Err(format!(
                "a [{}, {}] head does not score a {}-wide drafter over {} tokens",
                head.rows, head.columns, config.hidden_size, config.vocab_size
            ));
        }
        let kernels = Dflash2Kernels::load(cuda)?;
        let quantize = kernels.quantize;
        let source = single_file_source(directory)?;
        let uploader = Uploader { cuda };
        let hidden = config.hidden_size;
        let conv = |prefix: String| -> Result<GpuConv, String> {
            Ok(GpuConv {
                base: uploader.vector(
                    &source,
                    &format!("{prefix}.base_kernel"),
                    2 * config.conv_kernel_size * hidden,
                )?,
                projection: uploader.mat_e4m3(
                    quantize,
                    &source,
                    &format!("{prefix}.kernel_projection.weight"),
                    config.conv_taps_width(),
                    hidden,
                )?,
            })
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let at = |name: &str| format!("layers.{index}.{name}");
            layers.push(GpuDraftLayer {
                input_norm: uploader.vector(&source, &at("input_layernorm.weight"), hidden)?,
                post_norm: uploader.vector(
                    &source,
                    &at("post_attention_layernorm.weight"),
                    hidden,
                )?,
                attention_conv: conv(at("attention_conv"))?,
                mlp_conv: conv(at("mlp_conv"))?,
                q_proj: uploader.mat_e4m3_part(
                    quantize,
                    &source,
                    &at("self_attn.q_proj.weight"),
                    (whole.q_width(), hidden),
                    rows(q_width),
                )?,
                k_proj: uploader.mat_part(
                    &source,
                    &at("self_attn.k_proj.weight"),
                    (whole.kv_width(), hidden),
                    rows(kv_width),
                )?,
                v_proj: uploader.mat_part(
                    &source,
                    &at("self_attn.v_proj.weight"),
                    (whole.kv_width(), hidden),
                    rows(kv_width),
                )?,
                o_proj: uploader.mat_e4m3_part(
                    quantize,
                    &source,
                    &at("self_attn.o_proj.weight"),
                    (hidden, whole.q_width()),
                    columns(q_width),
                )?,
                q_norm: uploader.vector(
                    &source,
                    &at("self_attn.q_norm.weight"),
                    config.head_dim,
                )?,
                k_norm: uploader.vector(
                    &source,
                    &at("self_attn.k_norm.weight"),
                    config.head_dim,
                )?,
                gate_proj: uploader.mat_e4m3_part(
                    quantize,
                    &source,
                    &at("mlp.gate_proj.weight"),
                    (whole.intermediate_size, hidden),
                    rows(intermediate),
                )?,
                up_proj: uploader.mat_e4m3_part(
                    quantize,
                    &source,
                    &at("mlp.up_proj.weight"),
                    (whole.intermediate_size, hidden),
                    rows(intermediate),
                )?,
                down_proj: uploader.mat_e4m3_part(
                    quantize,
                    &source,
                    &at("mlp.down_proj.weight"),
                    (hidden, whole.intermediate_size),
                    columns(intermediate),
                )?,
            });
        }
        let codebook = config.vocab_size * config.selector_rank;
        Ok(Self {
            kernels,
            head: *head,
            fc: uploader.mat(&source, "fc.weight", hidden, config.feature_width())?,
            hidden_norm: uploader.vector(&source, "hidden_norm.weight", hidden)?,
            norm: uploader.vector(&source, "norm.weight", hidden)?,
            layers,
            hidden_projection: uploader.mat(
                &source,
                "candidate_selector.hidden_projection.weight",
                config.selector_rank,
                hidden,
            )?,
            predecessor: uploader.vector(
                &source,
                "candidate_selector.predecessor_codebook",
                codebook,
            )?,
            successor: uploader.vector(
                &source,
                "candidate_selector.successor_codebook",
                codebook,
            )?,
            config,
            world,
        })
    }

    /// The drafter's shape, as the kernels run it.
    #[must_use]
    pub fn config(&self) -> &Dflash2Config {
        &self.config
    }

    /// Frees the drafter's device buffers; nothing may run on them after.
    pub fn free(&self, cuda: &Cuda) {
        let mut buffers = vec![
            &self.fc.pointer,
            &self.hidden_norm,
            &self.norm,
            &self.hidden_projection.pointer,
            &self.predecessor,
            &self.successor,
        ];
        for layer in &self.layers {
            for conv in [&layer.attention_conv, &layer.mlp_conv] {
                buffers.extend([&conv.base, &conv.projection.weight, &conv.projection.scales]);
            }
            for matrix in [
                &layer.q_proj,
                &layer.o_proj,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            ] {
                buffers.extend([&matrix.weight, &matrix.scales]);
            }
            buffers.extend([
                &layer.input_norm,
                &layer.post_norm,
                &layer.k_proj.pointer,
                &layer.v_proj.pointer,
                &layer.q_norm,
                &layer.k_norm,
            ]);
        }
        cuda.free_buffers(&buffers);
    }

    /// The first top-k stage's blocks per row.
    fn topk_parts(&self) -> usize {
        self.head.rows.div_ceil(TOPK_CHUNK)
    }
}

/// A source over the drafter's one file, its index made from the file's own header. Every
/// tensor must be BF16: the uploader checks byte lengths, and only the dtype tells a BF16 matrix
/// from an F16 one of the same shape.
fn single_file_source(directory: &Path) -> Result<SafeTensorsSource, String> {
    let path = directory.join(DRAFT_SHARD);
    let mut file = std::fs::File::open(&path)
        .map_err(|error| format!("cannot open {DRAFT_SHARD}: {error}"))?;
    let file_len = file
        .metadata()
        .map_err(|error| format!("cannot stat {DRAFT_SHARD}: {error}"))?
        .len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .map_err(|error| format!("cannot read {DRAFT_SHARD}'s header length: {error}"))?;
    let header_len = u64::from_le_bytes(length);
    // Bounded before the buffer is sized, so a corrupt length costs an error, not an allocation.
    if header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(format!(
            "{DRAFT_SHARD} declares a {header_len}-byte header in a {file_len}-byte file"
        ));
    }
    let mut prefix = vec![0u8; 8 + header_len as usize];
    prefix[..8].copy_from_slice(&length);
    file.read_exact(&mut prefix[8..])
        .map_err(|error| format!("cannot read {DRAFT_SHARD}'s header: {error}"))?;
    let header = SafeTensorsHeader::from_container(&prefix, file_len, DRAFT_SHARD)
        .map_err(|error| error.to_string())?;
    if let Some((name, tensor)) = header.tensors.iter().find(|(_, t)| t.dtype != "BF16") {
        return Err(format!("{name} is {}, not BF16", tensor.dtype));
    }
    let weight_map = header
        .tensors
        .keys()
        .map(|name| (name.clone(), DRAFT_SHARD.to_owned()))
        .collect();
    Ok(SafeTensorsSource::new(
        directory,
        CheckpointIndex {
            weight_map,
            metadata: BTreeMap::new(),
        },
    ))
}

/// A sequence's drafter buffers: Qwen pools them with the rest of a session's scratch, and GLM
/// keeps them with a sequence's state.
#[derive(Debug, Clone, Copy)]
pub struct DraftScratch {
    /// [rows][feature_width] The captured target rows of the last forward, one slice per layer.
    features: Deviceptr,
    /// [rows][hidden] The context path's fc output, normed in place.
    context: Deviceptr,
    /// [rows][kv_width] One layer's context keys and values on their way into its ring.
    context_keys: Deviceptr,
    context_values: Deviceptr,
    /// [layers][keys, values][ring][kv_width] Every layer's committed positions, position p at
    /// slot p % ring.
    rings: Deviceptr,
    /// [block][hidden] A sublayer's output, the residual stream, a sublayer's normed input, its
    /// convolved input, and a projection's output before the closing convolution.
    hidden: Deviceptr,
    residual: Deviceptr,
    normed: Deviceptr,
    convolved: Deviceptr,
    projected: Deviceptr,
    /// [block][conv_taps_width] The current sublayer's projected convolution taps.
    taps: Deviceptr,
    /// [block][q_width] Queries, then the attention's output.
    query: Deviceptr,
    attention: Deviceptr,
    /// [block][kv_width] The block's own keys and values.
    keys: Deviceptr,
    values: Deviceptr,
    /// [block][intermediate] The MLP's gate, up and activated rows.
    gate: Deviceptr,
    up: Deviceptr,
    activated: Deviceptr,
    /// [block][selector_rank] The mask rows' hidden projections.
    selector: Deviceptr,
    /// [block][head rows] The head's logits for the mask rows.
    logits: Deviceptr,
    /// [block][parts][top_k] The first top-k stage's values and ids.
    part_values: Deviceptr,
    part_ids: Deviceptr,
    /// [block][top_k] Each mask row's candidates, best first.
    candidate_values: Deviceptr,
    candidate_ids: Deviceptr,
    /// [block] The selected chain, then [block] each draft's confidence (f32).
    drafts: Deviceptr,
}

impl DraftScratch {
    /// Buffers for a forward of up to `rows` rows and the drafter's block, each of the floats
    /// `floats` allocates.
    ///
    /// # Errors
    /// What `floats` reports.
    pub fn allocate(
        draft: &GpuDflash2,
        rows: usize,
        mut floats: impl FnMut(usize) -> Result<Deviceptr, String>,
    ) -> Result<Self, String> {
        let config = &draft.config;
        let block = config.block_size;
        let hidden = config.hidden_size;
        let mut floats = |elements: usize| floats(elements.max(1));
        let ring = config.ring_positions() * config.kv_width();
        let parts = draft.topk_parts() * config.selector_top_k;
        Ok(Self {
            features: floats(rows * config.feature_width())?,
            context: floats(rows * hidden)?,
            context_keys: floats(rows * config.kv_width())?,
            context_values: floats(rows * config.kv_width())?,
            rings: floats(config.num_hidden_layers * 2 * ring)?,
            hidden: floats(block * hidden)?,
            residual: floats(block * hidden)?,
            normed: floats(block * hidden)?,
            convolved: floats(block * hidden)?,
            projected: floats(block * hidden)?,
            taps: floats(block * config.conv_taps_width())?,
            query: floats(block * config.q_width())?,
            attention: floats(block * config.q_width())?,
            keys: floats(block * config.kv_width())?,
            values: floats(block * config.kv_width())?,
            gate: floats(block * config.intermediate_size)?,
            up: floats(block * config.intermediate_size)?,
            activated: floats(block * config.intermediate_size)?,
            selector: floats(block * config.selector_rank)?,
            logits: floats(block * draft.head.rows)?,
            part_values: floats(block * parts)?,
            part_ids: floats(block * parts)?,
            candidate_values: floats(block * config.selector_top_k)?,
            candidate_ids: floats(block * config.selector_top_k)?,
            drafts: floats(2 * block)?,
        })
    }

    pub(super) fn pointers(&self) -> [&Deviceptr; 25] {
        [
            &self.features,
            &self.context,
            &self.context_keys,
            &self.context_values,
            &self.rings,
            &self.hidden,
            &self.residual,
            &self.normed,
            &self.convolved,
            &self.projected,
            &self.taps,
            &self.query,
            &self.attention,
            &self.keys,
            &self.values,
            &self.gate,
            &self.up,
            &self.activated,
            &self.selector,
            &self.logits,
            &self.part_values,
            &self.part_ids,
            &self.candidate_values,
            &self.candidate_ids,
            &self.drafts,
        ]
    }

    /// Where [GpuDflash2::select] leaves the chain: [block] tokens, then [block] confidences.
    #[must_use]
    pub fn drafts(&self) -> Deviceptr {
        self.drafts
    }

    /// Layer `layer`'s key ring and value ring.
    fn ring(&self, config: &Dflash2Config, layer: usize) -> (Deviceptr, Deviceptr) {
        let floats = config.ring_positions() * config.kv_width();
        let keys = self.rings.offset(2 * layer * floats * 4);
        (keys, keys.offset(floats * 4))
    }
}

/// Threads for a top-k block that holds `candidates` values at once, eight a thread.
fn topk_threads(candidates: usize) -> Result<u32, BackendError> {
    let threads = candidates
        .div_ceil(TOPK_PER_THREAD)
        .next_multiple_of(32)
        .max(TOPK_THREADS);
    if threads > 1024 {
        return Err(BackendError::new(
            DOMAIN,
            format!("{candidates} candidates overflow one top-k block"),
        ));
    }
    Ok(threads as u32)
}

/// `output = matrix input` over an E4M3 matrix ([GpuMatE4m3]) for up to GEMV_MAX_BATCH rows, by
/// `kernel` (`q4e_gemv_tiled_e4m3`) on `device`'s stream: the tiled kernel at every width. The
/// drafter's projections and the head copy drafts score with, the block drafter's and an MTP
/// layer's alike.
///
/// # Errors
/// When there are more rows than the kernel takes, the operands are not 16-byte rows, or the
/// device refuses the launch.
pub fn gemv_e4m3(
    device: &impl DraftDevice,
    kernel: Kernel,
    matrix: &GpuMatE4m3,
    input: Deviceptr,
    output: Deviceptr,
    rows: usize,
) -> Result<(), BackendError> {
    if rows > GEMV_MAX_BATCH || !tiled_operands(&[matrix.weight, input], matrix.columns) {
        return Err(BackendError::new(
            DOMAIN,
            format!(
                "no E4M3 GEMV of {rows} rows over a [{}, {}] matrix",
                matrix.rows, matrix.columns
            ),
        ));
    }
    let tiles = matrix.rows.div_ceil(32);
    let (partials, counters) = device.tiled_scratch();
    device.launch(
        kernel,
        (
            tiles as u32,
            tiled_gemv_e4m3_splits(tiles, matrix.columns) as u32,
            1,
        ),
        (256, 1, 1),
        &[
            matrix.weight.0,
            matrix.scales.0,
            input.0,
            output.0,
            matrix.rows as u64,
            matrix.columns as u64,
            rows as u64,
            partials.0,
            counters.0,
        ],
    )
}

impl GpuDflash2 {
    /// Copy target layer `index`'s completed rows into the feature rows, when the drafter reads
    /// that layer: the mean of the `hc` hyper-connection streams of each of `rows` rows, laid
    /// `[rows][hc][hidden]` at `streams`.
    ///
    /// # Errors
    /// When the device refuses the launch.
    pub fn capture(
        &self,
        device: &impl DraftDevice,
        scratch: &DraftScratch,
        index: usize,
        (streams, hc): (Deviceptr, usize),
        rows: usize,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let Some(slot) = config.target_layer_ids.iter().position(|&id| id == index) else {
            return Ok(());
        };
        let hidden = config.hidden_size;
        device.launch(
            self.kernels.capture,
            (hidden.div_ceil(256) as u32, rows as u32, 1),
            (256, 1, 1),
            &[
                streams.0,
                scratch.features.0,
                hidden as u64,
                hc as u64,
                config.feature_width() as u64,
                slot as u64,
            ],
        )
    }

    /// Commit `rows` captured rows of the last forward, from row `first_row` on, which sit at
    /// `position` onward: the drafter's context path, then every layer's ring.
    ///
    /// # Errors
    /// When a GEMV has no kernel for the rows or the device refuses a launch.
    pub fn commit(
        &self,
        device: &impl DraftDevice,
        scratch: &DraftScratch,
        first_row: usize,
        rows: usize,
        position: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 {
            return Ok(());
        }
        let config = &self.config;
        let hidden = config.hidden_size;
        let kv_width = config.kv_width();
        let features = scratch
            .features
            .offset(first_row * config.feature_width() * 4);
        device.gemv(&self.fc, features, scratch.context, rows)?;
        self.norm(device, scratch.context, self.hidden_norm, rows, hidden)?;
        for (index, layer) in self.layers.iter().enumerate() {
            device.gemv(&layer.k_proj, scratch.context, scratch.context_keys, rows)?;
            device.gemv(&layer.v_proj, scratch.context, scratch.context_values, rows)?;
            self.norm(
                device,
                scratch.context_keys,
                layer.k_norm,
                rows * config.num_key_value_heads,
                config.head_dim,
            )?;
            self.rope(
                device,
                scratch.context_keys,
                (config.num_key_value_heads, kv_width),
                position,
                rows,
            )?;
            let (key_ring, value_ring) = scratch.ring(config, index);
            device.launch(
                self.kernels.kv_ring,
                (kv_width.div_ceil(256) as u32, rows as u32, 1),
                (256, 1, 1),
                &[
                    key_ring.0,
                    value_ring.0,
                    scratch.context_keys.0,
                    scratch.context_values.0,
                    kv_width as u64,
                    position as u64,
                    config.ring_positions() as u64,
                ],
            )?;
        }
        device.mark("d_commit")
    }

    /// Score the block after `anchor`, the token at `base`: the anchor and `block_size - 1` mask
    /// rows through the layers from `embed` (the target's BF16 embedding table), the head scoring
    /// the mask rows, and each mask row's `selector_top_k` best head rows, as ids from `first_id`,
    /// into `candidates`: `[block - 1][top_k]` values at the first pointer, their ids at the
    /// second. Each row's candidates are greatest first, the lower id first on a tie.
    ///
    /// # Errors
    /// When a GEMV has no kernel for the block or the device refuses a launch.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        device: &impl DraftDevice,
        scratch: &DraftScratch,
        embed: Deviceptr,
        anchor: u32,
        base: usize,
        first_id: usize,
        (values, ids): (Deviceptr, Deviceptr),
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let kernels = &self.kernels;
        let block = config.block_size;
        let hidden = config.hidden_size;
        device.launch(
            kernels.block_embed,
            (hidden.div_ceil(256) as u32, block as u32, 1),
            (256, 1, 1),
            &[
                embed.0,
                scratch.hidden.0,
                u64::from(anchor),
                u64::from(config.mask_token_id),
                hidden as u64,
            ],
        )?;
        let rows = block * hidden;
        let into_residual = |kernel: Kernel| {
            device.launch(
                kernel,
                (rows.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                &[scratch.residual.0, scratch.hidden.0, rows as u64],
            )
        };
        // A sharded drafter's sublayers leave this rank's partials, which every rank sums.
        let add_output = || {
            if self.world > 1 {
                device.reduce_add(scratch.hidden, scratch.residual, block, hidden)
            } else {
                into_residual(kernels.add)
            }
        };
        for (index, layer) in self.layers.iter().enumerate() {
            // The residual stream starts as the embeddings and takes each sublayer's output.
            if index == 0 {
                into_residual(kernels.copy_flat)?;
            } else {
                add_output()?;
            }
            self.attention(device, layer, index, scratch, base)?;
            add_output()?;
            self.mlp(device, layer, scratch)?;
        }
        device.mark("d_layers")?;
        add_output()?;
        self.norm_into(
            device,
            scratch.residual,
            self.norm,
            scratch.normed,
            block,
            hidden,
        )?;
        // The anchor row only carries context; the mask rows are the ones the head scores.
        let edges = block - 1;
        let mask_rows = scratch.normed.offset(hidden * 4);
        self.gemv_e4m3(device, &self.head, mask_rows, scratch.logits, edges)?;
        device.mark("d_lm_head")?;
        device.gemv(&self.hidden_projection, mask_rows, scratch.selector, edges)?;
        let top_k = config.selector_top_k;
        let parts = self.topk_parts();
        device.launch(
            kernels.topk_partial,
            (parts as u32, edges as u32, 1),
            (TOPK_THREADS as u32, 1, 1),
            &[
                scratch.logits.0,
                scratch.part_values.0,
                scratch.part_ids.0,
                self.head.rows as u64,
                TOPK_CHUNK as u64,
                top_k as u64,
                first_id as u64,
            ],
        )?;
        self.merge_into(
            device,
            (scratch.part_values, scratch.part_ids),
            (parts, parts * top_k, top_k),
            (values, ids),
        )
    }

    /// Each mask row's best `selector_top_k` of `parts` candidate lists ([Self::score]'s), list
    /// `p` of row `r` at float `p * part_stride + r * top_k` of `candidates`' values and of its
    /// ids, into the candidates [Self::select] walks. One list a rank of a tensor-parallel world
    /// makes this the whole vocabulary's top candidates, the same on every rank.
    ///
    /// # Errors
    /// When the lists overflow one top-k block or the device refuses the launch.
    pub fn merge(
        &self,
        device: &impl DraftDevice,
        scratch: &DraftScratch,
        candidates: (Deviceptr, Deviceptr),
        parts: usize,
        part_stride: usize,
    ) -> Result<(), BackendError> {
        let top_k = self.config.selector_top_k;
        self.merge_into(
            device,
            candidates,
            (parts, top_k, part_stride),
            (scratch.candidate_values, scratch.candidate_ids),
        )
    }

    /// q4d_topk_final over `parts` lists of each mask row, row `r`'s from float `r * row_stride`
    /// and list `p` `p * part_stride` floats on.
    fn merge_into(
        &self,
        device: &impl DraftDevice,
        (part_values, part_ids): (Deviceptr, Deviceptr),
        (parts, row_stride, part_stride): (usize, usize, usize),
        (values, ids): (Deviceptr, Deviceptr),
    ) -> Result<(), BackendError> {
        let top_k = self.config.selector_top_k;
        device.launch(
            self.kernels.topk_final,
            ((self.config.block_size - 1) as u32, 1, 1),
            (topk_threads(parts * top_k)?, 1, 1),
            &[
                part_values.0,
                part_ids.0,
                values.0,
                ids.0,
                parts as u64,
                top_k as u64,
                row_stride as u64,
                part_stride as u64,
            ],
        )
    }

    /// Walk the mask rows' candidates into one chain after `anchor`, into
    /// [DraftScratch::drafts].
    ///
    /// # Errors
    /// When the device refuses the launch.
    pub fn select(
        &self,
        device: &impl DraftDevice,
        scratch: &DraftScratch,
        anchor: u32,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let block = config.block_size;
        device.launch(
            self.kernels.select,
            (1, 1, 1),
            (256, 1, 1),
            &[
                scratch.candidate_ids.0,
                scratch.candidate_values.0,
                scratch.selector.0,
                self.predecessor.0,
                self.successor.0,
                scratch.drafts.0,
                scratch.drafts.offset(block * 4).0,
                u64::from(anchor),
                (block - 1) as u64,
                config.selector_top_k as u64,
                config.selector_rank as u64,
            ],
        )
    }

    /// The mask rows' candidates for [Self::score] to leave where [Self::select] walks them, when
    /// no merge runs between.
    #[must_use]
    pub fn candidates(scratch: &DraftScratch) -> (Deviceptr, Deviceptr) {
        (scratch.candidate_values, scratch.candidate_ids)
    }

    /// One layer's attention sublayer over the block in `scratch.residual`, its output (after the
    /// closing convolution) left in `scratch.hidden`.
    fn attention(
        &self,
        device: &impl DraftDevice,
        layer: &GpuDraftLayer,
        index: usize,
        scratch: &DraftScratch,
        base: usize,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let block = config.block_size;
        let hidden = config.hidden_size;
        self.norm_into(
            device,
            scratch.residual,
            layer.input_norm,
            scratch.normed,
            block,
            hidden,
        )?;
        self.gemv_e4m3(
            device,
            &layer.attention_conv.projection,
            scratch.normed,
            scratch.taps,
            block,
        )?;
        self.conv(
            device,
            (scratch.normed, scratch.taps),
            layer.attention_conv.base,
            scratch.convolved,
            0,
        )?;
        self.gemv_e4m3(
            device,
            &layer.q_proj,
            scratch.convolved,
            scratch.query,
            block,
        )?;
        device.gemv(&layer.k_proj, scratch.convolved, scratch.keys, block)?;
        device.gemv(&layer.v_proj, scratch.convolved, scratch.values, block)?;
        self.norm(
            device,
            scratch.query,
            layer.q_norm,
            block * config.num_attention_heads,
            config.head_dim,
        )?;
        self.norm(
            device,
            scratch.keys,
            layer.k_norm,
            block * config.num_key_value_heads,
            config.head_dim,
        )?;
        self.rope(
            device,
            scratch.query,
            (config.num_attention_heads, config.q_width()),
            base,
            block,
        )?;
        self.rope(
            device,
            scratch.keys,
            (config.num_key_value_heads, config.kv_width()),
            base,
            block,
        )?;
        let (key_ring, value_ring) = scratch.ring(config, index);
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        device.launch(
            self.kernels.attn,
            (config.num_attention_heads as u32, block as u32, 1),
            (ATTN_THREADS, 1, 1),
            &[
                scratch.query.0,
                key_ring.0,
                value_ring.0,
                scratch.keys.0,
                scratch.values.0,
                scratch.attention.0,
                config.num_attention_heads as u64,
                config.num_key_value_heads as u64,
                config.head_dim as u64,
                base as u64,
                config.window_left as u64,
                config.ring_positions() as u64,
                block as u64,
                u64::from(scale.to_bits()),
            ],
        )?;
        self.gemv_e4m3(
            device,
            &layer.o_proj,
            scratch.attention,
            scratch.projected,
            block,
        )?;
        self.conv(
            device,
            (scratch.projected, scratch.taps),
            layer.attention_conv.base,
            scratch.hidden,
            1,
        )
    }

    /// One layer's MLP sublayer over the block in `scratch.residual`, its output (after the
    /// closing convolution) left in `scratch.hidden`.
    fn mlp(
        &self,
        device: &impl DraftDevice,
        layer: &GpuDraftLayer,
        scratch: &DraftScratch,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let block = config.block_size;
        let hidden = config.hidden_size;
        self.norm_into(
            device,
            scratch.residual,
            layer.post_norm,
            scratch.normed,
            block,
            hidden,
        )?;
        self.gemv_e4m3(
            device,
            &layer.mlp_conv.projection,
            scratch.normed,
            scratch.taps,
            block,
        )?;
        self.conv(
            device,
            (scratch.normed, scratch.taps),
            layer.mlp_conv.base,
            scratch.convolved,
            0,
        )?;
        self.gemv_e4m3(
            device,
            &layer.gate_proj,
            scratch.convolved,
            scratch.gate,
            block,
        )?;
        self.gemv_e4m3(device, &layer.up_proj, scratch.convolved, scratch.up, block)?;
        let width = block * config.intermediate_size;
        device.launch(
            self.kernels.swiglu,
            (width.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[
                scratch.gate.0,
                scratch.up.0,
                scratch.activated.0,
                width as u64,
            ],
        )?;
        self.gemv_e4m3(
            device,
            &layer.down_proj,
            scratch.activated,
            scratch.projected,
            block,
        )?;
        self.conv(
            device,
            (scratch.projected, scratch.taps),
            layer.mlp_conv.base,
            scratch.hidden,
            1,
        )
    }

    /// [gemv_e4m3] over one of the drafter's E4M3 matrices.
    fn gemv_e4m3(
        &self,
        device: &impl DraftDevice,
        matrix: &GpuMatE4m3,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        gemv_e4m3(device, self.kernels.gemv_e4m3, matrix, input, output, rows)
    }

    /// The block convolution on `side` of a sublayer, from `input` with its projected `taps`
    /// into `output`.
    fn conv(
        &self,
        device: &impl DraftDevice,
        (input, taps): (Deviceptr, Deviceptr),
        base: Deviceptr,
        output: Deviceptr,
        side: u64,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        let hidden = config.hidden_size;
        device.launch(
            self.kernels.conv,
            (hidden.div_ceil(256) as u32, config.block_size as u32, 1),
            (256, 1, 1),
            &[
                input.0,
                taps.0,
                base.0,
                output.0,
                hidden as u64,
                config.conv_group_size as u64,
                config.conv_kernel_size as u64,
                side,
                config.conv_taps_width() as u64,
                config.block_size as u64,
            ],
        )
    }

    /// The drafter's RMSNorm over `groups` consecutive groups of `group` floats, in place. Its
    /// gains multiply directly, unlike the target's zero-centred ones.
    fn norm(
        &self,
        device: &impl DraftDevice,
        values: Deviceptr,
        weight: Deviceptr,
        groups: usize,
        group: usize,
    ) -> Result<(), BackendError> {
        self.norm_into(device, values, weight, values, groups, group)
    }

    fn norm_into(
        &self,
        device: &impl DraftDevice,
        input: Deviceptr,
        weight: Deviceptr,
        output: Deviceptr,
        groups: usize,
        group: usize,
    ) -> Result<(), BackendError> {
        let eps = self.config.rms_norm_eps as f32;
        device.launch(
            self.kernels.rmsnorm,
            (groups as u32, 1, 1),
            (256, 1, 1),
            &[
                input.0,
                weight.0,
                output.0,
                group as u64,
                group as u64,
                u64::from(eps.to_bits()),
                0,
                1,
            ],
        )
    }

    /// The drafter's full-width NeoX rotary over `tokens` rows of `heads` heads, `row_stride`
    /// floats apart, row t at `position + t`.
    fn rope(
        &self,
        device: &impl DraftDevice,
        rows: Deviceptr,
        (heads, row_stride): (usize, usize),
        position: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let config = &self.config;
        device.launch(
            self.kernels.rope,
            (heads as u32, tokens as u32, 1),
            (128, 1, 1),
            &[
                rows.0,
                config.head_dim as u64,
                config.head_dim as u64,
                position as u64,
                u64::from((config.rope_theta as f32).to_bits()),
                row_stride as u64,
                tokens as u64,
            ],
        )
    }
}

impl DraftDevice for DeviceSequence {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        DeviceSequence::launch(self, kernel, grid, block, args)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        DeviceSequence::gemv(self, matrix, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.gemm_partials, self.tiled_counters)
    }

    fn mark(&self, name: &'static str) -> Result<(), BackendError> {
        self.prof_mark(name)
    }
}

impl DeviceSequence {
    /// Whether this sequence drafts with DFlash 2 rather than the MTP head.
    pub(super) fn drafting(&self) -> bool {
        self.spec_enabled && self.draft.is_some() && self.backend.dflash.is_some()
    }

    /// Copy target layer `index`'s completed rows of this forward into the drafter's feature
    /// rows, when the drafter reads that layer: the mean of the hyper-connection streams.
    pub(super) fn capture_features(&self, index: usize, batch: usize) -> Result<(), BackendError> {
        let (Some(draft), Some(scratch)) = (self.backend.dflash.as_ref(), self.draft.as_ref())
        else {
            return Ok(());
        };
        if !self.spec_enabled {
            return Ok(());
        }
        let streams = (self.hyper, self.backend.config.hc_count);
        draft.capture(self, scratch, index, streams, batch)
    }

    /// Commit `rows` captured rows of the last forward, from row `first_row` on, which sit at
    /// `position` onward: the drafter's context path, then every layer's ring.
    pub(super) fn dflash_commit(
        &self,
        first_row: usize,
        rows: usize,
        position: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 || !self.drafting() {
            return Ok(());
        }
        let (Some(draft), Some(scratch)) = (self.backend.dflash.as_ref(), self.draft.as_ref())
        else {
            return Ok(());
        };
        // What the host did with the last forward's logits is charged before the drafter's own.
        self.prof_mark("host_answer")?;
        draft.commit(self, scratch, first_row, rows, position)
    }

    /// Draft the block after `anchor`, the token at `base`, into `spec_drafts`: nothing when the
    /// verify that would follow could not fit before the context limit.
    pub(super) fn dflash_propose(&mut self, anchor: u32, base: usize) -> Result<(), BackendError> {
        self.spec_drafts.clear();
        self.spec_confidence.clear();
        if !self.drafting() {
            return Ok(());
        }
        let backend = std::sync::Arc::clone(&self.backend);
        let (Some(draft), Some(scratch)) = (backend.dflash.as_ref(), self.draft) else {
            return Ok(());
        };
        let block = draft.config.block_size;
        if base + block > self.context_limit {
            return Ok(());
        }
        let embed = backend.weights.embed.pointer;
        let candidates = GpuDflash2::candidates(&scratch);
        draft.score(self, &scratch, embed, anchor, base, 0, candidates)?;
        draft.select(self, &scratch, anchor)?;
        let mut bytes = vec![0u8; 2 * block * 4];
        // The readback has to be ordered behind the launches above *on this session's stream*: a
        // plain synchronous copy runs on the default stream, which a session's work no longer
        // shares, and would return the previous step's block (or zeros) while the selector was
        // still queued - drafting from stale tokens, which the target then rejects.
        backend
            .cuda
            .copy_dtoh_on(self.clock(), &mut bytes, &scratch.drafts)
            .map_err(|error| BackendError::new("qwen4_dflash2", error))?;
        self.prof_mark("d_select")?;
        let edges = block - 1;
        let words = bytes.as_chunks::<4>().0;
        // A stop token ends the chain: the model's own answer there is the end of the sequence,
        // which no draft can match, so nothing past it could be accepted.
        for (chunk, confidence) in words[..edges].iter().zip(&words[block..block + edges]) {
            let token = u32::from_le_bytes(*chunk);
            if self.eos.contains(&token) {
                break;
            }
            self.spec_drafts.push(token);
            self.spec_confidence.push(f32::from_le_bytes(*confidence));
        }
        if let Some(policy) = self.depth_policy.as_ref() {
            let depth = policy
                .depth(&self.spec_confidence)
                .min(self.spec_drafts.len());
            self.spec_drafts.truncate(depth);
            self.spec_confidence.truncate(depth);
        }
        Ok(())
    }
}
