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

//! The chunk-wide prefill path on the tensor cores (kernels/qwen4_prefill_sm121.cu), with its
//! Gated DeltaNet / Kimi Delta Attention scan in its own image (kernels/qwen4_scan_sm121.cu).
//!
//! A prompt chunk runs its dense projections, its routed experts and its Gated DeltaNet recurrence
//! here, and so do a decode step and a speculative verify ([PrefillPlan::decode], on by default).
//! Every form computes a row from that row alone, in an order the matrix fixes: a dense
//! projection splits K as a DENSE_ROWS piece does and a token GEMM as a GEMM_TILE_TOKENS tile
//! does, whatever the launch's row count. So a row gets the same bits in a 4,096-row chunk, a
//! 22-row suffix, a verify and a one-token step, and a conversation that reuses its transcript's
//! state lands exactly where a cold prefill of the transcript does. `SPARKFLASH_DECODE_MMA=off`
//! puts the steps and the verifies back on the token kernels, and chunks under GEMM_MIN_BATCH
//! rows with them.
//!
//! The forms split every f32 activation - the dense projections' and the routed experts' - into a
//! bf16 head and the bf16 of its remainder and accumulate in f32, so a chunk here matches the
//! token kernels to a tolerance rather than bit for bit - as the token GEMM's own summation order
//! already did.
//!
//! # Attribution
//!
//! * Model architecture (Gated DeltaNet linear attention, the 512-expert top-10 MoE) originates
//!   from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Alibaba Cloud
//!   Qwen team) and the vLLM reference implementation
//!   ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0). The recurrence is
//!   that of Yang, Kautz and Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta
//!   Rule" (ICLR 2025).
//! * The NVFP4 weight layout originates from
//!   [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
//!   (NVIDIA TensorRT ModelOpt 0.46).
//! * The SM120/SM121 mma.sync / cp.async tiling patterns draw on prior art from
//!   [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0; Luke Alonso @lukealonso,
//!   @original-el8, and contributors) and
//!   [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0), through the
//!   shared kernels/opt_mma_*_sm121.cuh headers. See the root NOTICE file.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model::BackendError;

use super::{
    DeviceSequence, GEMM_MIN_BATCH, GEMM_PARTIAL_FLOATS, GEMM_SPLIT_TICKETS, GpuMat, split_tickets,
};
use crate::config::Qwen4ExpConfig;

/// The tensor-core prefill cubin this build embeds, for kernel-level tests.
#[must_use]
pub fn embedded_prefill_cubin() -> &'static [u8] {
    sparkflash_kernels::images::qwen4_prefill_cubin()
}

/// The Qwen3.8 scan cubin this build embeds: the Gated DeltaNet and Kimi Delta Attention kernels
/// that used to sit at the end of the prefill image.
#[must_use]
pub fn embedded_scan_cubin() -> &'static [u8] {
    sparkflash_kernels::images::qwen4_scan_cubin()
}

/// The GLM bulk prefill cubin this build embeds.
#[must_use]
pub fn embedded_glm_prefill_cubin() -> &'static [u8] {
    sparkflash_kernels::images::glm_prefill_cubin()
}

/// The GLM routed-expert (MoE) cubin this build embeds: the plan, MMA, SwiGLU and q8 kernels
/// that used to sit in the prefill image.
#[must_use]
pub fn embedded_glm_moe_cubin() -> &'static [u8] {
    sparkflash_kernels::images::glm_moe_cubin()
}

/// Tokens per layer-major prefill chunk, unless `SPARKFLASH_QWEN4_PREFILL_CHUNK` says otherwise.
///
/// The routed experts are 1.26 GB of NVFP4 a layer and a chunk reads each expert it routes to
/// once per 32 of its rows, so the chunk width is what amortises them: at 128 tokens the weight
/// traffic alone caps the prefill near 580 tokens a second, at 512 near 2,300. At 4,096 an expert
/// has about eighty rows, three items that the grid runs side by side.
pub const DEFAULT_PREFILL_CHUNK: usize = 4096;

/// The widest chunk the scratch is sized for: about 0.85 MiB a token past the attention rows,
/// 3.4 GiB at this width.
pub const MAX_PREFILL_CHUNK: usize = 4096;

/// Rows a dense projection's K split is chosen for. A chunk wider than this is a whole number of
/// them and runs its dense projections this many rows at a time, each piece split as a chunk this
/// wide was, so a row's sums run in the same order, to the same bits, at every chunk width.
pub const DENSE_ROWS: usize = 512;

/// The chunk when the tensor-core path is switched off: the token kernels' measured width.
pub const TOKEN_PREFILL_CHUNK: usize = 128;

/// The narrowest chunk: a speculative verify runs its rows through the same scratch as one
/// forward, so it must hold the widest verify.
pub const MIN_PREFILL_CHUNK: usize = 16;

/// Rows the indexer's score, selection and attention launches take at a time. A row's block
/// scores are a context's worth of floats, so their scratch is sized for this many rows only.
pub const ATTENTION_CHUNK: usize = 128;

/// Routed (token, slot) rows per expert item (Q4P_MOE_ROWS).
const MOE_ROWS: usize = 32;
/// The plan's shared histogram (Q4P_MAX_EXPERTS).
const MOE_MAX_EXPERTS: usize = 512;
/// Intermediate columns a gate_up CTA covers: 64 gate rows and the matching 64 up rows.
const MOE_GATE_UP_COLUMNS: usize = 64;
/// Output columns a down CTA covers (Q4P_BN).
const MOE_DOWN_COLUMNS: usize = 128;
/// K per pipeline stage (Q4P_BK): four 16-column E4M3 scale groups.
const MOE_K_STAGE: usize = 64;
/// q4p_moe_rows' intermediate (gate_up) and output (down) columns a CTA: four warps of 8 and 16.
const MOE_ROWS_GATE_UP_COLUMNS: usize = 32;
const MOE_ROWS_DOWN_COLUMNS: usize = 64;
/// The K a q4p_moe_rows lane reads at once: two stages, 16 bytes of a packed row.
const MOE_ROWS_K: usize = 2 * MOE_K_STAGE;
// The scan geometry lives in the shared op library so the Qwen and GLM arms cannot drift.
pub use sparkflash_ops::gdn::{
    CONV_HISTORY, GDN_KEY, GDN_QK_ROW, GDN_VALUE, gdn_scan_shape, gdn_scan_shape_for,
};

/// The floats q4p_gdn_prep stages for one (token, key head): q then k, each key_dim wide with
/// four padding floats after every 32 (Q4P_GDN_ROW).
pub fn gdn_qk_row(key_dim: usize) -> usize {
    2 * (key_dim + key_dim / 8)
}
/// The dense MMA's token and column tile (OM_BM, OM_BN).
const DENSE_TILE: usize = 128;
/// q4p_dense_rows' rows (one m16 tile) and columns (four warps of 8) a CTA, for launches of fewer
/// than GEMM_MIN_BATCH rows.
const DENSE_ROWS_TILE: usize = 16;
const DENSE_ROWS_COLUMNS: usize = 32;
/// CTAs a split dense MMA aims for: two per SM on GB10's forty-eight.
const DENSE_TARGET_CTAS: usize = 96;
/// The thinnest K slice a split takes, in 32-column blocks.
const DENSE_MIN_SLICE_BLOCKS: usize = 8;

/// The partials and tickets a q4p_dense_rows launch of `group` over `batch` rows split
/// `segments` ways takes: none unsplit, where every CTA stores its own finished outputs.
fn dense_rows_need(
    group: &[(&GpuMat, Deviceptr)],
    batch: usize,
    segments: usize,
) -> (usize, usize) {
    if segments == 1 {
        return (0, 0);
    }
    group
        .iter()
        .fold((0, 0), |(partials, tickets), (matrix, _)| {
            (
                partials + segments * batch * matrix.rows,
                tickets
                    + matrix.rows.div_ceil(DENSE_ROWS_COLUMNS) * batch.div_ceil(DENSE_ROWS_TILE),
            )
        })
}

/// One projection of a [DeviceSequence::gemm_group]: `matrix` into `output`, finished by the
/// token GEMM's `epilogue` (1 = silu(v / scale), 2 = sigmoid(v / scale) scaled to two, otherwise
/// the sum).
#[derive(Clone, Copy)]
pub(super) struct Projection<'a> {
    pub(super) matrix: &'a GpuMat,
    pub(super) output: Deviceptr,
    pub(super) epilogue: u32,
    pub(super) scale: f32,
}

impl<'a> Projection<'a> {
    /// `matrix` into `output`, the sum itself.
    pub(super) fn plain(matrix: &'a GpuMat, output: Deviceptr) -> Self {
        Self {
            matrix,
            output,
            epilogue: 0,
            scale: 1.0,
        }
    }
}

/// How a prompt is cut and whether its chunks take the tensor cores, fixed when the backend
/// loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    /// Tokens per layer-major chunk.
    pub chunk: usize,
    /// Whether a chunk of GEMM_MIN_BATCH tokens or more takes this image.
    /// `SPARKFLASH_QWEN4_PREFILL_MMA=0` keeps every chunk on the token kernels, for A/B runs.
    pub tensor_cores: bool,
    /// Whether every other forward takes it too - a short chunk, a decode step, a speculative
    /// verify - so a row computes to the same bits at every row count (`SPARKFLASH_DECODE_MMA`,
    /// on wherever `tensor_cores` is). Off, those stay on the token kernels, as before.
    pub decode: bool,
}

impl PrefillPlan {
    /// A plan for `chunk`-token chunks, clamped to the widths the scratch supports and, past
    /// DENSE_ROWS, rounded down to a whole number of them. The tensor cores are only taken when
    /// this build carries the prefill image; the decode steps take them with the chunks.
    #[must_use]
    pub fn new(chunk: usize, tensor_cores: bool) -> Self {
        let chunk = chunk.clamp(MIN_PREFILL_CHUNK, MAX_PREFILL_CHUNK);
        let tensor_cores = tensor_cores && !embedded_prefill_cubin().is_empty();
        Self {
            chunk: if chunk > DENSE_ROWS {
                chunk - chunk % DENSE_ROWS
            } else {
                chunk
            },
            tensor_cores,
            decode: tensor_cores,
        }
    }

    /// This plan with the decode steps, verifies and short chunks on the token kernels instead
    /// (`decode` false), or on the tensor cores with the chunks (true, where they are).
    #[must_use]
    pub fn with_decode(self, decode: bool) -> Self {
        Self {
            decode: decode && self.tensor_cores,
            ..self
        }
    }

    /// The forwards a prompt of `tokens` runs as: `chunk` tokens each. With the decode steps on
    /// the token kernels, a tail past the last whole DENSE_ROWS too short for the tensor cores
    /// runs on its own, so every row computes as it did when chunks were DENSE_ROWS wide: dense
    /// pieces start where those chunks did, and such a tail was a chunk of its own on the token
    /// kernels then too. With them on the tensor cores a row's bits do not depend on its forward,
    /// and the tail stays in the chunk.
    #[must_use]
    pub fn forwards(&self, tokens: usize) -> Vec<std::ops::Range<usize>> {
        let mut forwards = Vec::with_capacity(tokens.div_ceil(self.chunk.max(1)) + 1);
        let mut start = 0;
        while start < tokens {
            let mut end = (start + self.chunk).min(tokens);
            let tail = (end - start) % DENSE_ROWS;
            if !self.decode && end - start > DENSE_ROWS && tail < GEMM_MIN_BATCH {
                end -= tail;
            }
            forwards.push(start..end);
            start = end;
        }
        forwards
    }

    /// The serving default, as `SPARKFLASH_QWEN4_PREFILL_MMA`, `SPARKFLASH_QWEN4_PREFILL_CHUNK`
    /// and `SPARKFLASH_DECODE_MMA` adjust it.
    ///
    /// # Errors
    /// A value outside its switch's domain, or `SPARKFLASH_DECODE_MMA=on` where the chunks are
    /// not on the tensor cores (`SPARKFLASH_QWEN4_PREFILL_MMA=0`, or a build without the image).
    pub fn from_env() -> Result<Self, BackendError> {
        Self::from_active()
    }

    /// The serving default from the process's installed profile.
    ///
    /// # Errors
    /// As [Self::from_profile].
    pub fn from_active() -> Result<Self, BackendError> {
        Self::from_profile(sparkflash_model::active_profile())
    }

    /// The plan a typed profile describes.
    ///
    /// # Errors
    /// `decode.tensor_cores = true` with the chunks off the tensor-core image, or a build without
    /// the prefill image.
    pub fn from_profile(profile: &sparkflash_model::ModelProfile) -> Result<Self, BackendError> {
        let tensor_cores =
            !embedded_prefill_cubin().is_empty() && profile.prefill.tensor_cores.unwrap_or(true);
        if profile.decode.tensor_cores == Some(true)
            && profile.prefill.tensor_cores == Some(false)
            && !tensor_cores
        {
            return Err(BackendError::new(
                "qwen4_config",
                "decode.tensor_cores=true runs the decode steps on the prefill's tensor-core \
                 image, which is off here (prefill.tensor_cores=false, or a build without the \
                 image); unset it or set it to false",
            ));
        }
        let chunk = if tensor_cores {
            profile.prefill.chunk.unwrap_or(DEFAULT_PREFILL_CHUNK)
        } else {
            TOKEN_PREFILL_CHUNK
        };
        Ok(Self::new(chunk, tensor_cores).with_decode(profile.decode.tensor_cores.unwrap_or(true)))
    }

    /// [Self::from_env] over `lookup` instead of the installed profile, for a test that pins one
    /// switch at a time.
    ///
    /// # Errors
    /// As [Self::from_profile].
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, BackendError> {
        const SWITCHES: [&str; 3] = [
            "SPARKFLASH_QWEN4_PREFILL_CHUNK",
            "SPARKFLASH_QWEN4_PREFILL_MMA",
            "SPARKFLASH_DECODE_MMA",
        ];
        let refuse = |error: sparkflash_model::ProfileError| {
            BackendError::new("qwen4_config", error.to_string())
        };
        let ours = |name: &str| SWITCHES.contains(&name).then(|| lookup(name)).flatten();
        let mut profile = sparkflash_model::profile::ModelProfile::from_toml(include_str!(
            "../../config/qwen38_flash_next_nvfp4.toml"
        ))
        .map_err(refuse)?;
        profile.apply_env(ours).map_err(refuse)?;
        Self::from_profile(&profile)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PrefillKernels {
    dense_mma: Kernel,
    dense_rows: Kernel,
    pub(super) gemm_rows: Kernel,
    epilogue: Kernel,
    split_bf16: Kernel,
    moe_plan: Kernel,
    moe_gate_up: Kernel,
    moe_down: Kernel,
    moe_rows_gate_up: Kernel,
    moe_rows_down: Kernel,
    /// The block-scaled FP8 grouped prefill MMA (D95), on the shared 128 x 128 tile.
    moe_gate_up_fp8: Kernel,
    moe_swiglu_fp8: Kernel,
    moe_down_fp8: Kernel,
    gdn_conv: Kernel,
    gdn_conv_state: Kernel,
    gdn_prep: Kernel,
    gdn_scan: Kernel,
    gdn_scan_64_64: Kernel,
    gdn_scan_128_256: Kernel,
    gdn_gate: Kernel,
}

impl PrefillKernels {
    pub(super) fn load(cuda: &Cuda) -> Result<Self, String> {
        if embedded_prefill_cubin().is_empty() {
            return Err(
                "this build has no Qwen4-Exp prefill cubin: rebuild on a host with nvcc so \
                 the shared sparkflash-kernels build script compiles kernels/qwen4_prefill_sm121.cu"
                    .to_owned(),
            );
        }
        if embedded_scan_cubin().is_empty() {
            return Err(
                "this build has no Qwen4-Exp scan cubin: rebuild on a host with nvcc so \
                 the shared sparkflash-kernels build script compiles kernels/qwen4_scan_sm121.cu"
                    .to_owned(),
            );
        }
        let named = |name: &'static str| -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(embedded_prefill_cubin(), name)?;
            Ok(kernel)
        };
        // The Gated DeltaNet and Kimi Delta Attention kernels live in the scan image, not here.
        let scan = |name: &'static str| -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(embedded_scan_cubin(), name)?;
            Ok(kernel)
        };
        Ok(Self {
            dense_mma: named("q4p_dense_mma")?,
            dense_rows: named("q4p_dense_rows")?,
            gemm_rows: named("q4p_gemm_rows")?,
            epilogue: named("q4p_epilogue")?,
            split_bf16: named("q4p_split_bf16")?,
            // The routed-expert plan is the shared image (kernels/moe_plan_sm121.cu), as the
            // token path's is: one counting pass, one item layout, four callers.
            moe_plan: {
                if sparkflash_kernels::images::moe_plan_cubin().is_empty() {
                    return Err("this build has no routed-expert plan cubin".to_owned());
                }
                cuda.load_named(sparkflash_kernels::images::moe_plan_cubin(), "moe_plan")?
                    .1
            },
            moe_gate_up: named("q4p_moe_gate_up")?,
            moe_down: named("q4p_moe_down")?,
            moe_rows_gate_up: named("q4p_moe_rows_gate_up")?,
            moe_rows_down: named("q4p_moe_rows_down")?,
            moe_gate_up_fp8: named("q4p_moe_gate_up_fp8")?,
            moe_swiglu_fp8: named("q4p_moe_swiglu_fp8")?,
            moe_down_fp8: named("q4p_moe_down_fp8")?,
            gdn_conv: scan("q4p_gdn_conv")?,
            gdn_conv_state: scan("q4p_gdn_conv_state")?,
            gdn_prep: scan("q4p_gdn_prep")?,
            gdn_scan: scan("q4p_gdn_scan")?,
            gdn_scan_64_64: scan("q4p_gdn_scan_64_64")?,
            gdn_scan_128_256: scan("q4p_gdn_scan_128_256")?,
            gdn_gate: scan("q4p_gdn_gate")?,
        })
    }
}

/// Device buffers only this path reads, sized for one chunk.
#[derive(Debug, Clone, Copy)]
pub(super) struct PrefillScratch {
    /// The routed experts' input as two [tokens][hidden] BF16 planes, head then remainder.
    expert_input: Deviceptr,
    /// The routing plan's (expert, first sorted slot, rows, 0) items: at most one per 32 slots
    /// plus one per expert.
    items: Deviceptr,
    /// How many of `items` the plan wrote; the expert grids read it and idle past it.
    item_count: Deviceptr,
    /// [tokens][key heads][GDN_QK_ROW] normalised queries and keys, padded as the scan stages them.
    gdn_qk: Deviceptr,
    /// [tokens][value heads] (decay factor, beta).
    gdn_decay_beta: Deviceptr,
    /// The FP8 routed experts' grouped prefill output: items of 32 rows x 2 * intermediate f32
    /// columns (gate then up), which q4p_moe_swiglu_fp8 reduces to [slots][intermediate].
    moe_gate_up: Deviceptr,
}

impl PrefillScratch {
    pub(super) fn allocate(
        cuda: &Cuda,
        config: &Qwen4ExpConfig,
        chunk: usize,
        total: &mut usize,
    ) -> Result<Self, String> {
        let slots = chunk * config.num_experts_per_tok;
        let items = slots.div_ceil(MOE_ROWS) + config.num_experts.min(slots);
        Ok(Self {
            expert_input: cuda.alloc((chunk * config.hidden_size * 4).max(4), total)?,
            items: cuda.alloc(items.max(1) * 16, total)?,
            item_count: cuda.alloc(4, total)?,
            gdn_qk: cuda.alloc(
                (chunk * config.linear_num_key_heads * gdn_qk_row(config.linear_key_head_dim) * 4)
                    .max(4),
                total,
            )?,
            gdn_decay_beta: cuda
                .alloc((chunk * config.linear_num_value_heads * 8).max(4), total)?,
            // The most items the plan can write is ceil(slots / 32) + experts; each is padded
            // to the 32-row item and holds gate and up.
            moe_gate_up: cuda.alloc(
                ((slots.div_ceil(MOE_ROWS) + config.num_experts.min(slots))
                    * MOE_ROWS
                    * 2
                    * config.moe_intermediate_size
                    * 4)
                .max(4),
                total,
            )?,
        })
    }

    pub(super) fn pointers(&self) -> [&Deviceptr; 6] {
        [
            &self.expert_input,
            &self.items,
            &self.item_count,
            &self.gdn_qk,
            &self.gdn_decay_beta,
            &self.moe_gate_up,
        ]
    }
}

/// Which of the image's three paths a chunk may take: `SPARKFLASH_QWEN4_PREFILL_PARTS`, a comma
/// list of `dense`, `experts` and `gdn` (all three when unset), for bisecting a difference
/// against the token kernels one path at a time.
fn part(name: &str) -> bool {
    static PARTS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PARTS
        .get_or_init(|| std::env::var("SPARKFLASH_QWEN4_PREFILL_PARTS").ok())
        .as_deref()
        .is_none_or(|parts| parts.split(',').any(|part| part.trim() == name))
}

fn aligned(pointer: Deviceptr) -> bool {
    pointer.0.is_multiple_of(16)
}

/// The K split of a dense MMA over `tokens` rows of a `rows` x `k_blocks` 32-column matrix: enough
/// slices to bring the grid near DENSE_TARGET_CTAS, as many as the partials hold, none thinner
/// than DENSE_MIN_SLICE_BLOCKS, and all the same width, so a divisor of the K blocks.
fn dense_segments(rows: usize, k_blocks: usize, tokens: usize) -> usize {
    let tiles = rows.div_ceil(DENSE_TILE) * tokens.div_ceil(DENSE_TILE);
    let ceiling = DENSE_TARGET_CTAS
        .div_ceil(tiles)
        .min(GEMM_PARTIAL_FLOATS / (tokens * rows).max(1))
        .min(k_blocks / DENSE_MIN_SLICE_BLOCKS)
        .max(1);
    (1..=ceiling)
        .rev()
        .find(|segments| k_blocks.is_multiple_of(*segments))
        .unwrap_or(1)
}

impl DeviceSequence {
    /// Whether a forward of `batch` rows takes this path: every forward with the decode steps on
    /// it ([PrefillPlan::decode]), otherwise a chunk of GEMM_MIN_BATCH rows or more.
    pub(super) fn chunk_path(&self, batch: usize) -> bool {
        let plan = &self.backend.prefill;
        plan.tensor_cores && (plan.decode || batch >= GEMM_MIN_BATCH)
    }

    /// A projection over the chunk on the tensor cores, when its shape fits the MMA tile: 32-row
    /// multiples on both sides and 16-byte aligned operands. Returns false, launching nothing,
    /// when it does not (the hyper-connection's four-row injection, the 48-row GDN gates, the
    /// one-row shared-expert gate); the caller then runs the token GEMM.
    ///
    /// `epilogue` and `scale` are the token GEMM's: 1 = silu(v / scale), 2 = sigmoid(v / scale)
    /// scaled to two. A matrix too narrow to fill the GPU splits its inner dimension over the
    /// grid's third axis and q4e_gemm_reduce adds the slices in order, applying the epilogue.
    ///
    /// The split is chosen per DENSE_ROWS piece of the chunk. A row's sums depend on the split and
    /// nothing else in the launch, so whole pieces go in one launch, as many as the partials hold.
    /// With the decode steps here every piece is split as a whole one is, so a row's split is the
    /// matrix's alone; otherwise a short piece (a tail, or a chunk under DENSE_ROWS) takes the
    /// split for its own width, as it did before the decode steps came here.
    pub(super) fn dense_mma(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
        epilogue: u32,
        scale: f32,
    ) -> Result<bool, BackendError> {
        if !self.dense_fits(matrix, input, output, batch) {
            return Ok(false);
        }
        let mut done = 0usize;
        while done < batch {
            let left = batch - done;
            let piece = left.min(DENSE_ROWS);
            let segments = self.dense_split(matrix, piece);
            let rows = if piece < DENSE_ROWS {
                piece
            } else if segments > 1 {
                let fit = GEMM_PARTIAL_FLOATS / (segments * DENSE_ROWS * matrix.rows);
                DENSE_ROWS * (left / DENSE_ROWS).min(fit).max(1)
            } else {
                DENSE_ROWS * (left / DENSE_ROWS)
            };
            self.dense_launch(
                matrix,
                input.offset(done * matrix.columns * 4),
                output.offset(done * matrix.rows * 4),
                rows,
                segments,
                epilogue,
                scale,
            )?;
            done += rows;
        }
        Ok(true)
    }

    /// Whether `batch` rows of `matrix` from `input` into `output` take the dense MMA.
    fn dense_fits(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
    ) -> bool {
        self.chunk_path(batch)
            && part("dense")
            && matrix.rows.is_multiple_of(32)
            && matrix.columns.is_multiple_of(32)
            && [matrix.pointer, input, output].into_iter().all(aligned)
    }

    /// The K split of `matrix` for a piece of `piece` rows ([Self::dense_mma]).
    fn dense_split(&self, matrix: &GpuMat, piece: usize) -> usize {
        let reference = if self.backend.prefill.decode {
            DENSE_ROWS
        } else {
            piece
        };
        dense_segments(matrix.rows, matrix.columns / 32, reference)
    }

    /// Projections of one `input` over `batch` rows, each into its own output.
    ///
    /// A few rows on the tensor cores put the matrices whose splits and epilogues agree in one
    /// q4p_dense_rows launch, up to four, as many as the partials and tickets hold, and up to two
    /// matrices off the tile ride along on the first as q4p_gemm_rows CTAs: a layer's projections
    /// of one input then cost one launch where they cost one each, and every output has the bits
    /// its own launch gives it. Everything else - a wide chunk, the token kernels, a rider past
    /// what the partials hold - is [Self::gemm] a matrix at a time.
    pub(super) fn gemm_group(
        &self,
        projections: &[Projection],
        input: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        // (split, epilogue, scale bits) and the matrices launched together under them.
        type Group<'a> = ((usize, u32, u32), Vec<(&'a GpuMat, Deviceptr)>);
        let mut grouped: Vec<Group> = Vec::new();
        let mut off_tile = Vec::new();
        for projection in projections {
            let (matrix, output) = (projection.matrix, projection.output);
            if batch >= GEMM_MIN_BATCH || matrix.columns != projections[0].matrix.columns {
                self.gemm(
                    matrix,
                    input,
                    output,
                    batch,
                    projection.epilogue,
                    projection.scale,
                )?;
                continue;
            }
            if !self.dense_fits(matrix, input, output, batch) {
                off_tile.push(projection);
                continue;
            }
            let key = (
                self.dense_split(matrix, batch),
                projection.epilogue,
                projection.scale.to_bits(),
            );
            let segments = key.0;
            let joins = grouped.iter_mut().find(|(at, group)| {
                let (partials, tickets) = dense_rows_need(group, batch, segments);
                *at == key
                    && group.len() < 4
                    && (segments == 1
                        || (partials + segments * batch * matrix.rows <= GEMM_PARTIAL_FLOATS
                            && tickets
                                + matrix.rows.div_ceil(DENSE_ROWS_COLUMNS)
                                    * batch.div_ceil(DENSE_ROWS_TILE)
                                <= GEMM_SPLIT_TICKETS))
            });
            match joins {
                Some((_, group)) => group.push((matrix, output)),
                None => grouped.push((key, vec![(matrix, output)])),
            }
        }
        let mut riders = Vec::new();
        if let Some(((segments, _, _), group)) = grouped.first() {
            let (mut partials, mut tickets) = dense_rows_need(group, batch, *segments);
            for projection in &off_tile {
                let (more, counters) = self.rider_need(projection.matrix, batch);
                if riders.len() == 2
                    || partials + more > GEMM_PARTIAL_FLOATS
                    || tickets + counters > GEMM_SPLIT_TICKETS
                {
                    break;
                }
                riders.push(**projection);
                partials += more;
                tickets += counters;
            }
        }
        for projection in &off_tile[riders.len()..] {
            self.gemm(
                projection.matrix,
                input,
                projection.output,
                batch,
                projection.epilogue,
                projection.scale,
            )?;
        }
        for (index, ((segments, epilogue, scale), group)) in grouped.into_iter().enumerate() {
            let riding: &[Projection] = if index == 0 { &riders } else { &[] };
            self.dense_rows_launch(
                &group,
                input,
                batch,
                segments,
                epilogue,
                f32::from_bits(scale),
                riding,
            )?;
        }
        Ok(())
    }

    /// The partials and tickets a q4p_gemm_rows launch of `matrix` over `batch` rows takes.
    fn rider_need(&self, matrix: &GpuMat, batch: usize) -> (usize, usize) {
        let split = self.token_split(matrix, batch);
        let count = batch * matrix.rows;
        if split > 1 {
            (split * count, count.div_ceil(128))
        } else {
            (0, 0)
        }
    }

    /// One q4p_dense_rows launch of `group` over `batch` rows split `segments` ways, and the
    /// `riders` (at most two) that run beside it, each as its own q4p_gemm_rows launch would: its
    /// split and epilogue, its slices and counters in the partials region past the group's.
    #[allow(clippy::too_many_arguments)]
    fn dense_rows_launch(
        &self,
        group: &[(&GpuMat, Deviceptr)],
        input: Deviceptr,
        batch: usize,
        segments: usize,
        epilogue: u32,
        scale: f32,
        riders: &[Projection],
    ) -> Result<(), BackendError> {
        let mut columns = 0usize;
        let mut matrices = [0u64; 16];
        let (mut partials, mut tiles) = (0usize, 0usize);
        for (slot, &(matrix, output)) in group.iter().enumerate() {
            let target = if segments > 1 {
                self.gemm_partials.offset(partials * 4)
            } else {
                output
            };
            matrices[slot * 4..slot * 4 + 4].copy_from_slice(&[
                matrix.pointer.0,
                target.0,
                output.0,
                matrix.rows as u64,
            ]);
            columns += matrix.rows.div_ceil(DENSE_ROWS_COLUMNS);
            let (more, counters) = dense_rows_need(&[(matrix, output)], batch, segments);
            partials += more;
            tiles += counters;
        }
        let mut riding = [0u64; 16];
        let mut ctas = 0usize;
        for (slot, rider) in riders.iter().enumerate().take(2) {
            let split = self.token_split(rider.matrix, batch);
            let count = batch * rider.matrix.rows;
            riding[slot * 8..slot * 8 + 8].copy_from_slice(&[
                rider.matrix.pointer.0,
                rider.output.0,
                rider.matrix.rows as u64,
                u64::from(rider.epilogue),
                u64::from(rider.scale.to_bits()),
                split as u64,
                self.gemm_partials.offset(partials * 4).0,
                split_tickets(self.gemm_partials).offset(tiles * 4).0,
            ]);
            let (more, counters) = self.rider_need(rider.matrix, batch);
            partials += more;
            tiles += counters;
            ctas += count.div_ceil(128) * split;
        }
        if group.is_empty()
            || group.len() > 4
            || riders.len() > 2
            || partials > GEMM_PARTIAL_FLOATS
            || tiles > GEMM_SPLIT_TICKETS
        {
            return Err(BackendError::new(
                "qwen4_prefill",
                format!(
                    "a q4p_dense_rows launch of {} matrices and {} riders over {batch} rows split \
                     {segments} ways needs {partials} partials and {tiles} tickets, past the \
                     {GEMM_PARTIAL_FLOATS} and {GEMM_SPLIT_TICKETS} there are",
                    group.len(),
                    riders.len()
                ),
            ));
        }
        // The riders' CTAs, numbered over every (y, z) of the grid past the dense tiles.
        let planes = batch.div_ceil(DENSE_ROWS_TILE) * segments;
        let mut args = vec![
            input.0,
            group[0].0.columns as u64,
            batch as u64,
            segments as u64,
            split_tickets(self.gemm_partials).0,
            u64::from(epilogue),
            u64::from(scale.to_bits()),
            group.len() as u64,
        ];
        args.extend_from_slice(&matrices);
        args.extend_from_slice(&[columns as u64, riders.len() as u64]);
        args.extend_from_slice(&riding);
        self.launch(
            self.backend.kernels.prefill.dense_rows,
            (
                (columns + ctas.div_ceil(planes)) as u32,
                batch.div_ceil(DENSE_ROWS_TILE) as u32,
                segments as u32,
            ),
            (128, 1, 1),
            &args,
        )
    }

    /// One dense MMA launch over `batch` rows split `segments` ways, and its reduction or
    /// epilogue.
    #[allow(clippy::too_many_arguments)]
    fn dense_launch(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
        segments: usize,
        epilogue: u32,
        scale: f32,
    ) -> Result<(), BackendError> {
        let kernels = &self.backend.kernels;
        let count = batch * matrix.rows;
        let target = if segments > 1 {
            self.gemm_partials
        } else {
            output
        };
        // A few rows take q4p_dense_rows, the same sums on a tile cut for them, which adds a split
        // tile's slices and applies the epilogue itself.
        if batch < GEMM_MIN_BATCH {
            return self.dense_rows_launch(
                &[(matrix, output)],
                input,
                batch,
                segments,
                epilogue,
                scale,
                &[],
            );
        }
        self.launch(
            kernels.prefill.dense_mma,
            (
                matrix.rows.div_ceil(DENSE_TILE) as u32,
                batch.div_ceil(DENSE_TILE) as u32,
                segments as u32,
            ),
            (256, 1, 1),
            &[
                matrix.pointer.0,
                input.0,
                target.0,
                matrix.rows as u64,
                matrix.columns as u64,
                batch as u64,
                segments as u64,
            ],
        )?;
        if segments > 1 {
            self.launch(
                kernels.gemm_reduce,
                (count.div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[
                    self.gemm_partials.0,
                    output.0,
                    count as u64,
                    segments as u64,
                    u64::from(epilogue),
                    u64::from(scale.to_bits()),
                ],
            )?;
        } else if epilogue != 0 {
            self.launch(
                kernels.prefill.epilogue,
                (count.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
                (256, 1, 1),
                &[
                    output.0,
                    count as u64,
                    u64::from(epilogue),
                    u64::from(scale.to_bits()),
                ],
            )?;
        }
        Ok(())
    }

    /// The chunk's routed experts as grouped W4A16 MMA, into the same [slot][hidden] partials the
    /// token kernels write. Returns false, launching nothing, for a configuration outside the
    /// kernels' tiles.
    ///
    /// The plan sorts the chunk's slots by expert into items of up to 32 rows; gate_up writes
    /// each item's SwiGLU rows as two bf16 planes, head and remainder, over `expert_activated`
    /// (its f32 rows hold exactly both), and down scales each row by its routing weight into
    /// `expert_accumulator`. An expert's weights are read once per item: at a 512-token chunk
    /// (ten rows an expert on average) about once per chunk, at 4,096 about three times, by
    /// items next to each other in the grid. A few rows - a decode step, a verify - take the
    /// q4p_moe_rows forms, the same sums on a tile cut for them, when both K dimensions are whole
    /// stage pairs.
    pub(super) fn routed_experts_mma(
        &self,
        table: Deviceptr,
        scales: Deviceptr,
        mixed: Deviceptr,
        batch: usize,
    ) -> Result<bool, BackendError> {
        let config = &self.backend.config;
        let hidden = config.hidden_size;
        let intermediate = config.moe_intermediate_size;
        if !part("experts")
            || !hidden.is_multiple_of(MOE_DOWN_COLUMNS)
            || !hidden.is_multiple_of(MOE_K_STAGE)
            || !intermediate.is_multiple_of(MOE_GATE_UP_COLUMNS)
            || !intermediate.is_multiple_of(MOE_K_STAGE)
            || config.num_experts > MOE_MAX_EXPERTS
        {
            return Ok(false);
        }
        let kernels = &self.backend.kernels.prefill;
        let scratch = &self.prefill;
        let topk = config.num_experts_per_tok;
        let slots = batch * topk;
        let elements = batch * hidden;
        self.launch(
            kernels.split_bf16,
            ((elements / 2).div_ceil(256).clamp(1, 4096) as u32, 1, 1),
            (256, 1, 1),
            &[mixed.0, scratch.expert_input.0, elements as u64],
        )?;
        // The most items the plan can write, and the grid's bound: past the count it wrote, a
        // CTA returns before touching anything.
        let items = (slots.div_ceil(MOE_ROWS) + config.num_experts.min(slots)) as u32;
        self.launch(
            kernels.moe_plan,
            (1, 1, 1),
            (
                sparkflash_model::active_profile().plan.block_threads(),
                1,
                1,
            ),
            &[
                self.expert_indices.0,
                self.expert_perm.0,
                scratch.items.0,
                scratch.item_count.0,
                0,
                slots as u64,
                config.num_experts as u64,
                topk as u64,
                MOE_ROWS as u64,
                items as u64,
                0,
                // A slot routed to no expert is left out of the items, and its partial row
                // is zeroed so the combine still sums one row per slot.
                2,
                0,
                0,
                0,
                0,
                self.expert_accumulator.0,
                hidden as u64,
            ],
        )?;
        let rows = batch < GEMM_MIN_BATCH
            && hidden.is_multiple_of(MOE_ROWS_K)
            && intermediate.is_multiple_of(MOE_ROWS_K);
        let (gate_up, gate_up_columns, down, down_columns, block) = if rows {
            (
                kernels.moe_rows_gate_up,
                MOE_ROWS_GATE_UP_COLUMNS,
                kernels.moe_rows_down,
                MOE_ROWS_DOWN_COLUMNS,
                (128, 1, 1),
            )
        } else {
            (
                kernels.moe_gate_up,
                MOE_GATE_UP_COLUMNS,
                kernels.moe_down,
                MOE_DOWN_COLUMNS,
                (256, 1, 1),
            )
        };
        self.launch(
            gate_up,
            ((intermediate / gate_up_columns) as u32, items, 1),
            block,
            &[
                table.0,
                scales.0,
                self.expert_perm.0,
                scratch.items.0,
                scratch.item_count.0,
                scratch.expert_input.0,
                elements as u64,
                self.expert_activated.0,
                (slots * intermediate) as u64,
                topk as u64,
                hidden as u64,
                intermediate as u64,
            ],
        )?;
        self.launch(
            down,
            ((hidden / down_columns) as u32, items, 1),
            block,
            &[
                table.0,
                scales.0,
                self.expert_perm.0,
                scratch.items.0,
                scratch.item_count.0,
                self.expert_activated.0,
                (slots * intermediate) as u64,
                self.probabilities.0,
                self.expert_accumulator.0,
                topk as u64,
                hidden as u64,
                intermediate as u64,
                config.num_experts as u64,
            ],
        )?;
        Ok(true)
    }

    /// The chunk's block-scaled FP8 routed experts on the shared 128 x 128 tensor-core tile
    /// (D95). Returns false, launching nothing, when the plan or the shapes do not fit the tile,
    /// or when the batch is below [GEMM_MIN_BATCH]: a decode step keeps the token FP8 kernels
    /// (their f32 activation is the exact path), as the NVFP4 arm's few-rows form serves small
    /// batches.
    ///
    /// The token path's FP8 experts are one CTA an expert run on CUDA cores; a prefill chunk
    /// sorts the same slots into items and runs them through q4p_moe_gate_up_fp8 /
    /// q4p_moe_down_fp8, which instantiate the shared body with Qwen's BF16 128 x 128 tile
    /// scales. Gate and up read the item's token row and write its slot row (the plan's
    /// permutation included) of one [slots][2 * intermediate] f32 buffer, SwiGLU maps those rows
    /// into [slots][intermediate], and down applies the slot's routing weight from the
    /// [batch][experts] probability row into the same [slot][hidden] partials the token kernels
    /// write, so the combine is unchanged.
    pub(super) fn routed_experts_mma_fp8(
        &self,
        table: Deviceptr,
        mixed: Deviceptr,
        batch: usize,
    ) -> Result<bool, BackendError> {
        let config = &self.backend.config;
        let hidden = config.hidden_size;
        let intermediate = config.moe_intermediate_size;
        // A chunk's grouped tile reads one expert's 128 x 128 weight blocks once for many rows;
        // a decode step or a short verify has no rows to share, and the token FP8 kernels' f32
        // activation is the existing exact path (the NVFP4 arm keeps its few-rows form the same
        // way). Only a real chunk takes the grouped tile.
        if batch < GEMM_MIN_BATCH
            || !part("experts")
            || !hidden.is_multiple_of(128)
            || !intermediate.is_multiple_of(128)
            || config.num_experts > MOE_MAX_EXPERTS
        {
            return Ok(false);
        }
        let kernels = &self.backend.kernels.prefill;
        let scratch = &self.prefill;
        let topk = config.num_experts_per_tok;
        let slots = batch * topk;
        let items = (slots.div_ceil(MOE_ROWS) + config.num_experts.min(slots)) as u32;
        self.launch(
            kernels.moe_plan,
            (1, 1, 1),
            (
                sparkflash_model::active_profile().plan.block_threads(),
                1,
                1,
            ),
            &[
                self.expert_indices.0,
                self.expert_perm.0,
                scratch.items.0,
                scratch.item_count.0,
                0,
                slots as u64,
                config.num_experts as u64,
                topk as u64,
                MOE_ROWS as u64,
                items as u64,
                0,
                2,
                0,
                0,
                0,
                0,
                self.expert_accumulator.0,
                hidden as u64,
            ],
        )?;
        // Gate and up share the one gate_up buffer: gate in the first `intermediate` columns,
        // up in the second.
        for (matrix, at) in [(0u64, 0usize), (2u64, intermediate)] {
            self.launch(
                kernels.moe_gate_up_fp8,
                ((intermediate / 128) as u32, 1, items),
                (256, 1, 1),
                &[
                    table.0,
                    scratch.items.0,
                    mixed.0,
                    scratch.moe_gate_up.offset(at * 4).0,
                    intermediate as u64,
                    hidden as u64,
                    self.expert_perm.0,
                    topk as u64,
                    matrix,
                ],
            )?;
        }
        self.launch(
            kernels.moe_swiglu_fp8,
            (intermediate.div_ceil(256).max(1) as u32, slots as u32, 1),
            (256, 1, 1),
            &[
                scratch.moe_gate_up.0,
                self.expert_activated.0,
                slots as u64,
                intermediate as u64,
            ],
        )?;
        self.launch(
            kernels.moe_down_fp8,
            ((hidden / 128) as u32, 1, items),
            (256, 1, 1),
            &[
                table.0,
                scratch.items.0,
                self.expert_activated.0,
                self.probabilities.0,
                self.expert_accumulator.0,
                hidden as u64,
                intermediate as u64,
                self.expert_perm.0,
                topk as u64,
                config.num_experts as u64,
            ],
        )?;
        Ok(true)
    }

    /// Whether this configuration's Gated DeltaNet is the one the scan is compiled for.
    fn gdn_fits(&self) -> bool {
        let config = &self.backend.config;
        let groups = config.value_groups();
        let key_dim = config.linear_key_head_dim;
        let value_dim = config.linear_value_head_dim;
        part("gdn")
            && matches!((key_dim, value_dim), (64, 64) | (128, 128) | (128, 256))
            && (2..=CONV_HISTORY + 1).contains(&config.linear_conv_kernel_dim)
            && groups != 0
            && groups <= key_dim
            && config.linear_num_key_heads * groups == config.linear_num_value_heads
    }

    /// The forward's Gated DeltaNet after its projections: the conv over every row at once, the
    /// normalised queries and keys and each row's decay and beta, the scan with the state in
    /// registers, and the gated norm in place. Reads `linear_qkv` and `linear_z` and the rows'
    /// `b` and `a` from `inputs`, writes the convolved rows to `inputs`' first, and leaves the
    /// gated rows in `linear_out`. The conv history advances in place; the recurrent state is
    /// read from `recurrent_state` and written to `recurrent_out`, which is the same buffer
    /// except in a speculative verify. Returns false, launching nothing, for a configuration
    /// other than the one the scan is compiled for.
    ///
    /// `snapshot` is the verify's (conv windows, rows, region stride): region r receives the
    /// windows after row r, for the first `rows` rows, as q4e_gdn_conv writes them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_chunk(
        &self,
        conv_weight: Deviceptr,
        conv_state: Deviceptr,
        (recurrent_state, recurrent_out): (Deviceptr, Deviceptr),
        a_log: Deviceptr,
        dt_bias: Deviceptr,
        norm: Deviceptr,
        inputs: [Deviceptr; 3],
        batch: usize,
        (snapshot, snapshot_rows, region_stride): (u64, usize, u64),
    ) -> Result<bool, BackendError> {
        if !self.gdn_fits() {
            return Ok(false);
        }
        let config = &self.backend.config;
        let taps = config.linear_conv_kernel_dim;
        let heads = config.linear_num_value_heads;
        let width = config.linear_conv_width();
        let kernels = &self.backend.kernels.prefill;
        let [conv_out, ..] = inputs;
        let columns = width.div_ceil(256).max(1) as u32;
        self.launch(
            kernels.gdn_conv,
            (columns, batch as u32, 1),
            (256, 1, 1),
            &[
                self.linear_qkv.0,
                conv_state.0,
                conv_weight.0,
                conv_out.0,
                width as u64,
                taps as u64,
            ],
        )?;
        self.launch(
            kernels.gdn_conv_state,
            (columns, 1, 1),
            (256, 1, 1),
            &[
                self.linear_qkv.0,
                conv_state.0,
                width as u64,
                taps as u64,
                batch as u64,
                snapshot,
                snapshot_rows as u64,
                region_stride,
            ],
        )?;
        self.prof_mark("gdn_conv")?;
        self.gdn_recurrence(
            a_log,
            dt_bias,
            inputs,
            (recurrent_state, recurrent_out),
            batch,
        )?;
        self.launch(
            kernels.gdn_gate,
            (heads as u32, batch as u32, 1),
            (config.linear_value_head_dim as u32, 1, 1),
            &[
                self.linear_out.0,
                self.linear_z.0,
                norm.0,
                self.linear_out.0,
                heads as u64,
                config.linear_value_head_dim as u64,
                u64::from((config.rms_norm_eps as f32).to_bits()),
                u64::from(config.output_gate_is_sigmoid()),
            ],
        )?;
        self.prof_mark("gdn_recur")?;
        Ok(true)
    }

    /// The recurrence over `rows` convolved rows and their `b` and `a` ([conv, b, a]), from the
    /// state in `from` to the state in `to`, the raw head outputs in `linear_out`. A verify's
    /// commit replays its kept rows through this from the inputs the verify left, so the state is
    /// bit for bit the one those rows left behind. Returns false, launching nothing, where
    /// [Self::gdn_chunk] would.
    pub(super) fn gdn_recurrence(
        &self,
        a_log: Deviceptr,
        dt_bias: Deviceptr,
        [conv, b, a]: [Deviceptr; 3],
        (from, to): (Deviceptr, Deviceptr),
        rows: usize,
    ) -> Result<bool, BackendError> {
        if !self.gdn_fits() {
            return Ok(false);
        }
        let config = &self.backend.config;
        let key_heads = config.linear_num_key_heads;
        let heads = config.linear_num_value_heads;
        let groups = config.value_groups();
        let key_width = config.linear_key_width();
        let kernels = &self.backend.kernels.prefill;
        let scratch = &self.prefill;
        self.launch(
            kernels.gdn_prep,
            (key_heads as u32, rows as u32, 1),
            (config.linear_key_head_dim as u32, 1, 1),
            &[
                conv.0,
                a.0,
                b.0,
                a_log.0,
                dt_bias.0,
                scratch.gdn_qk.0,
                scratch.gdn_decay_beta.0,
                key_width as u64,
                key_heads as u64,
                heads as u64,
                config.linear_value_head_dim as u64,
                groups as u64,
            ],
        )?;
        let value_dim = config.linear_value_head_dim;
        let scan = match (config.linear_key_head_dim, value_dim) {
            (64, 64) => kernels.gdn_scan_64_64,
            (128, 256) => kernels.gdn_scan_128_256,
            _ => kernels.gdn_scan,
        };
        let (scan_grid, scan_threads) = gdn_scan_shape_for(heads, value_dim);
        // The verify's windows are the conv's snapshot; the recurrent state is replayed rather
        // than saved a row at a time, so the scan writes no snapshot of its own.
        self.launch(
            scan,
            scan_grid,
            (scan_threads, 1, 1),
            &[
                scratch.gdn_qk.0,
                scratch.gdn_decay_beta.0,
                conv.0,
                from.0,
                to.0,
                self.linear_out.0,
                key_width as u64,
                key_heads as u64,
                heads as u64,
                value_dim as u64,
                groups as u64,
                rows as u64,
                0,
                0,
                0,
            ],
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(pairs: &[(&str, &str)]) -> Result<PrefillPlan, BackendError> {
        PrefillPlan::from_lookup(|name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        })
    }

    #[test]
    fn the_decode_steps_follow_the_chunks_by_default() {
        let plan = plan(&[]).expect("the default plan");
        assert_eq!(plan.decode, plan.tensor_cores);
    }

    #[test]
    fn decode_mma_off_keeps_the_decode_steps_on_the_token_kernels() {
        for value in ["off", "0", "false"] {
            let plan = plan(&[("SPARKFLASH_DECODE_MMA", value)]).expect("off is a value");
            assert!(!plan.decode, "SPARKFLASH_DECODE_MMA={value}");
        }
    }

    #[test]
    fn a_decode_mma_value_outside_the_switch_is_refused() {
        let error = plan(&[("SPARKFLASH_DECODE_MMA", "fast")]).expect_err("not a switch value");
        assert_eq!(error.code(), "qwen4_config");
        let error = plan(&[("SPARKFLASH_QWEN4_PREFILL_MMA", "2")]).expect_err("not a switch value");
        assert_eq!(error.code(), "qwen4_config");
    }

    #[test]
    fn decode_mma_on_without_the_tensor_core_chunks_is_refused() {
        let error = plan(&[
            ("SPARKFLASH_QWEN4_PREFILL_MMA", "0"),
            ("SPARKFLASH_DECODE_MMA", "on"),
        ])
        .expect_err("the decode steps cannot take an image the chunks do not");
        assert_eq!(error.code(), "qwen4_config");
        assert!(error.message().contains("decode.tensor_cores=true"));
    }
}
